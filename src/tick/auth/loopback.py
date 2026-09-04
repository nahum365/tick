"""The loopback half of the OAuth ceremony: the URL out, the redirect back.

Robinhood's authorization server advertises `token_endpoint_auth_methods:
["none"]` and open dynamic registration, which is the public-client shape: the
authorization code comes back to a redirect URI, and PKCE — not a client
secret — is what proves the code was redeemed by whoever asked for it. The
redirect URI here is a **loopback** address, `http://127.0.0.1:<port>/…`,
because that is the one destination that keeps invariant 1 true: the code
never travels to a host anybody else operates, Tick included.

Three properties are enforced rather than assumed, and each has a test.

- **The listener binds 127.0.0.1, never 0.0.0.0.** A callback server on all
  interfaces accepts an authorization code from the local network.
- **`state` is compared, and a mismatch refuses.** The value Tick put in the
  authorization URL is the only one it will accept back. A redirect carrying
  someone else's `state` is a cross-site request forgery attempt or a stale
  browser tab, and either way the code in it is not ours to redeem. The
  refusal returns nothing — the mismatched code is not stored, not logged and
  not handed back to the caller.
- **The listener is one-shot and bounded.** It serves until it has an answer
  or the timeout expires, then stops. A callback server left running is an
  open door on the user's machine long after the ceremony ended.

`state` is read out of the authorization URL rather than generated here, so
the value compared on the way back is exactly the value the SDK sent — a
second source of truth for it is a second thing that can drift.
"""

from __future__ import annotations

import asyncio
import re
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import TracebackType
from urllib.parse import parse_qs, urlparse

from mcp.shared.auth import AuthorizationCodeResult

from .errors import CallbackError

__all__ = [
    "LOOPBACK_HOST",
    "CALLBACK_PATH",
    "LoopbackAuthorization",
    "state_of",
]

#: The only interface the callback listener binds. Not `0.0.0.0`, ever.
LOOPBACK_HOST = "127.0.0.1"
#: A native-app custom scheme redirect (RFC 8252 §7.1): scheme, host-ish label, path.
_SAFE_OVERRIDE = re.compile(r"^[a-z][a-z0-9+.-]{1,31}://[a-z0-9.-]{1,64}/[A-Za-z0-9/_-]{1,128}$")

#: The path the authorization server is told to redirect to.
CALLBACK_PATH = "/tick/callback"

_PAGE_OK = (
    "<html><body><h3>Tick is connected.</h3>"
    "<p>You can close this tab and return to the terminal.</p></body></html>"
)
_PAGE_BAD = (
    "<html><body><h3>Tick did not accept this redirect.</h3>"
    "<p>Start the connection again from the terminal.</p></body></html>"
)


def state_of(authorization_url: str) -> str | None:
    """The `state` parameter in an authorization URL, or `None` if it carries none."""
    values = parse_qs(urlparse(authorization_url).query).get("state")
    return values[0] if values else None


class _CallbackServer(HTTPServer):
    """An `HTTPServer` that keeps the one redirect it was waiting for."""

    captured: dict[str, str] | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Answers exactly one useful GET and says nothing to the console."""

    server: _CallbackServer

    def do_GET(self) -> None:  # noqa: N802 - the name http.server requires
        query = parse_qs(urlparse(self.path).query)
        captured = {key: values[0] for key, values in query.items() if values}
        if "code" not in captured and "error" not in captured:
            # A favicon request, or a stray probe. Not the redirect; keep waiting.
            self._respond(404, _PAGE_BAD)
            return
        self.server.captured = captured
        self._respond(200, _PAGE_OK)

    def _respond(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Silence. The console belongs to the ceremony, not to the listener."""


class LoopbackAuthorization:
    """The redirect and callback handlers `OAuthClientProvider` needs.

    Every argument is required. `port` may be 0 for an ephemeral port — the
    bound port is known once the context is entered, which is why
    `redirect_uri` is only readable from inside it. `announce` receives the
    lines this ceremony prints (`typer.echo` in the CLI, a list's `append` in a
    test), and `open_browser` says whether the URL is also handed to the
    machine's browser: printing is always done, opening is a choice, and a
    ceremony that only opened a browser would be unusable over SSH.
    """

    def __init__(
        self,
        *,
        port: int,
        timeout_seconds: float,
        open_browser: bool,
        announce: Callable[[str], None],
        redirect_uri_override: str | None,
        on_callback: Callable[[], None] | None,
    ) -> None:
        """`redirect_uri_override` registers a non-loopback redirect (a phone's own URL
        scheme, e.g. `tick://broker/callback`) while the state check and code handling
        stay here: the phone cannot intercept a loopback redirect, so the
        app posts the captured URL back through `complete_redirect_url`. None keeps
        the loopback redirect the CLI ceremony uses."""
        if redirect_uri_override is not None and not _SAFE_OVERRIDE.fullmatch(
            redirect_uri_override
        ):
            raise ValueError(
                "redirect_uri_override must be a custom-scheme URL like tick://broker/callback"
            )
        self._override = redirect_uri_override
        if port < 0 or port > 65535:
            raise ValueError(f"port ({port}) must be between 0 and 65535; 0 picks a free one")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds ({timeout_seconds}) must be > 0")
        self._port = port
        self._timeout = timeout_seconds
        self._open_browser = open_browser
        self._announce = announce
        self._on_callback = on_callback
        self._callback_announced = False
        self._server: _CallbackServer | None = None
        self._expected_state: str | None = None
        #: The authorization URL that was announced, for the CLI to report.
        self.authorization_url: str | None = None

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------

    def __enter__(self) -> LoopbackAuthorization:
        self._server = _CallbackServer((LOOPBACK_HOST, self._port), _CallbackHandler)
        self._server.captured = None
        self._server.timeout = self._timeout
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        """The port actually bound. Only meaningful inside the context."""
        return self._require_server().server_address[1]

    @property
    def redirect_uri(self) -> str:
        """Where the authorization server is told to send the code back."""
        if self._override is not None:
            self._require_server()
            return self._override
        return f"http://{LOOPBACK_HOST}:{self.port}{CALLBACK_PATH}"

    def _require_server(self) -> _CallbackServer:
        if self._server is None:
            raise CallbackError(
                "the loopback listener is not open. Use LoopbackAuthorization as a "
                "context manager: the redirect URI is not known until it has bound a port."
            )
        return self._server

    # ------------------------------------------------------------------
    # The two handlers OAuthClientProvider calls
    # ------------------------------------------------------------------

    async def redirect_handler(self, authorization_url: str) -> None:
        """Show the user where to authorise, and remember the `state` we sent."""
        self.authorization_url = authorization_url
        self._expected_state = state_of(authorization_url)
        self._announce("Open this URL to authorise Tick with Robinhood:")
        self._announce(authorization_url)
        self._announce(f"Waiting for the redirect on {self.redirect_uri} …")
        if self._open_browser:
            webbrowser.open(authorization_url)

    async def callback_handler(self) -> AuthorizationCodeResult:
        """Wait for the redirect and return the code, or refuse with a reason."""
        return await asyncio.to_thread(self.wait_for_redirect)

    # ------------------------------------------------------------------
    # The blocking wait, usable on its own from a test
    # ------------------------------------------------------------------

    def wait_for_redirect(self) -> AuthorizationCodeResult:
        """Serve until the redirect arrives, then check it. Blocking."""
        server = self._require_server()
        remaining = self._timeout
        while server.captured is None and remaining > 0:
            before = time.monotonic()
            # Re-armed each pass: a browser asking for /favicon.ico is a request
            # this listener answers and keeps waiting through, and the deadline
            # is the ceremony's, not each individual request's.
            server.timeout = remaining
            server.handle_request()
            remaining -= time.monotonic() - before
        captured = server.captured
        if captured is None:
            raise CallbackError(
                f"no redirect arrived on {self.redirect_uri} within {self._timeout:.0f}s. "
                f"Nothing was authorised and nothing was written; run the connect "
                f"command again when you are ready to finish it in the browser."
            )
        result = self._accept(captured)
        self._notify_callback()
        return result

    def complete_redirect_url(self, redirect_url: str) -> None:
        """Hand the phone-captured loopback redirect to the same state-checking path."""
        parsed = urlparse(redirect_url)
        server = self._require_server()
        if self._override is not None:
            expected = urlparse(self._override)
            matches = (
                parsed.scheme == expected.scheme
                and parsed.netloc == expected.netloc
                and parsed.path == expected.path
            )
        else:
            matches = (
                parsed.hostname == LOOPBACK_HOST
                and parsed.port == self.port
                and parsed.path == CALLBACK_PATH
            )
        if not matches:
            raise CallbackError(
                "the posted redirect URL does not target this box's active callback. "
                "Return to the current authorization session and post its complete redirect URL."
            )
        query = parse_qs(parsed.query)
        captured = {key: values[0] for key, values in query.items() if values}
        if "code" not in captured and "error" not in captured:
            raise CallbackError(
                "the posted redirect carries neither a code nor an authorization error. "
                "Post the complete URL intercepted after authorization."
            )
        # Validate before waking the OAuth callback. A mismatched code is never retained.
        self._accept(captured)
        server.captured = captured
        self._notify_callback()

    def _notify_callback(self) -> None:
        """Close a related pixel stream once this exact callback has passed state checks."""
        if not self._callback_announced and self._on_callback is not None:
            self._callback_announced = True
            self._on_callback()

    def _accept(self, captured: dict[str, str]) -> AuthorizationCodeResult:
        """Turn a captured redirect into a result, or refuse it."""
        if "error" in captured:
            description = captured.get("error_description", "no description was given")
            raise CallbackError(
                f"Robinhood refused the authorization: {captured['error']} "
                f"({description}). Nothing was written to this machine."
            )
        returned = captured.get("state")
        if self._expected_state is not None and returned != self._expected_state:
            # Deliberately says nothing about the code it is discarding.
            raise CallbackError(
                "the redirect carried a state value Tick did not send, so its "
                "authorization code was discarded unread. This happens when an old "
                "authorization tab is reloaded, and it is what a forged redirect looks "
                "like. Run the connect command again and use the fresh URL."
            )
        return AuthorizationCodeResult(code=captured["code"], state=returned)
