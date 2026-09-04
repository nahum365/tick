"""The loopback listener: it catches one redirect, and checks it.

These tests bind `127.0.0.1` on an ephemeral port and drive it from the same
process. That is a socket on the test machine, not a request to anybody: no
name is resolved, no route leaves the host, and no authorization server exists
anywhere in this file.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request

import pytest

from tick.auth import CALLBACK_PATH, LOOPBACK_HOST, CallbackError, LoopbackAuthorization, state_of

#: The `state` the fake authorization URL carries, and the only one accepted back.
STATE = "state-placeholder"

AUTH_URL = (
    "https://agent.robinhood.com/oauth/trading/authorize"
    f"?response_type=code&client_id=client-placeholder&state={STATE}"
    "&code_challenge=challenge-placeholder&code_challenge_method=S256"
)


def loopback(announce=None, *, open_browser: bool = False) -> LoopbackAuthorization:
    return LoopbackAuthorization(
        port=0,
        timeout_seconds=5.0,
        open_browser=open_browser,
        announce=announce if announce is not None else (lambda line: None),
        redirect_uri_override=None,
    )


def hit(url: str) -> int:
    """GET `url` on the loopback listener and return its status."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def redirect_after(auth: LoopbackAuthorization, query: str) -> threading.Thread:
    """Fire the redirect from another thread once the listener is waiting."""

    def send() -> None:
        hit(f"{auth.redirect_uri}?{query}")

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    return thread


def test_the_listener_binds_only_the_loopback_interface():
    """Never 0.0.0.0: a callback server on every interface accepts a code from the LAN."""
    with loopback() as auth:
        assert auth.redirect_uri.startswith(f"http://{LOOPBACK_HOST}:")
        assert auth.redirect_uri.endswith(CALLBACK_PATH)
        assert auth._require_server().server_address[0] == LOOPBACK_HOST


def test_an_ephemeral_port_is_known_once_the_listener_is_open():
    """The redirect URI has to be known before the client metadata is registered."""
    with loopback() as auth:
        assert auth.port > 0


def test_the_redirect_uri_is_unavailable_before_a_port_is_bound():
    """A URI quoted before anything is listening would send the code nowhere."""
    with pytest.raises(CallbackError):
        _ = loopback().redirect_uri


async def test_the_callback_captures_the_code_from_the_redirect():
    lines: list[str] = []
    with loopback(lines.append) as auth:
        await auth.redirect_handler(AUTH_URL)
        thread = redirect_after(auth, f"code=code-placeholder&state={STATE}")
        result = await auth.callback_handler()
        thread.join(timeout=5)

    assert result.code == "code-placeholder"
    assert result.state == STATE


async def test_the_phone_can_post_the_same_loopback_redirect_url():
    with loopback() as auth:
        await auth.redirect_handler(AUTH_URL)
        auth.complete_redirect_url(f"{auth.redirect_uri}?code=code-placeholder&state={STATE}")
        result = await auth.callback_handler()

    assert result.code == "code-placeholder"
    assert result.state == STATE


async def test_a_phone_posted_mismatched_state_is_refused_before_it_is_retained():
    with loopback() as auth:
        await auth.redirect_handler(AUTH_URL)
        with pytest.raises(CallbackError, match="state value Tick did not send"):
            auth.complete_redirect_url(
                f"{auth.redirect_uri}?code=discarded-placeholder&state=wrong-state"
            )
        assert auth._require_server().captured is None


async def test_a_mismatched_state_is_refused_and_its_code_never_returned():
    """The value Tick sent is the only one it accepts back."""
    with loopback() as auth:
        await auth.redirect_handler(AUTH_URL)
        thread = redirect_after(auth, "code=forged-placeholder&state=someone-elses-state")
        with pytest.raises(CallbackError) as caught:
            await auth.callback_handler()
        thread.join(timeout=5)

    message = str(caught.value)
    assert "state value Tick did not send" in message
    assert "forged-placeholder" not in message


async def test_an_authorization_error_is_reported_rather_than_swallowed():
    with loopback() as auth:
        await auth.redirect_handler(AUTH_URL)
        thread = redirect_after(
            auth, f"error=access_denied&error_description=user+declined&state={STATE}"
        )
        with pytest.raises(CallbackError) as caught:
            await auth.callback_handler()
        thread.join(timeout=5)

    assert "access_denied" in str(caught.value)
    assert "Nothing was written" in str(caught.value)


async def test_a_stray_request_does_not_end_the_wait():
    """A browser asks for /favicon.ico; that is not the redirect."""
    with loopback() as auth:
        await auth.redirect_handler(AUTH_URL)

        def send() -> None:
            hit(f"http://{LOOPBACK_HOST}:{auth.port}/favicon.ico")
            hit(f"{auth.redirect_uri}?code=code-placeholder&state={STATE}")

        thread = threading.Thread(target=send, daemon=True)
        thread.start()
        result = await auth.callback_handler()
        thread.join(timeout=5)

    assert result.code == "code-placeholder"


def test_a_wait_that_times_out_says_nothing_was_authorised():
    """Fail safe: the ceremony ends, the listener closes, and nothing is half-done."""
    auth = LoopbackAuthorization(
        port=0,
        timeout_seconds=0.2,
        open_browser=False,
        announce=lambda line: None,
        redirect_uri_override=None,
    )
    with auth:
        with pytest.raises(CallbackError) as caught:
            auth.wait_for_redirect()

    assert "Nothing was authorised and nothing was written" in str(caught.value)


async def test_the_url_is_announced_and_the_browser_is_not_opened_unless_asked(monkeypatch):
    """Printing always; opening is a choice — a ceremony over SSH has no browser."""
    opened: list[str] = []
    monkeypatch.setattr("tick.auth.loopback.webbrowser.open", lambda url: opened.append(url))
    lines: list[str] = []

    with loopback(lines.append) as auth:
        await auth.redirect_handler(AUTH_URL)

    assert AUTH_URL in lines
    assert opened == []


async def test_the_browser_is_opened_when_asked(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("tick.auth.loopback.webbrowser.open", lambda url: opened.append(url))

    with loopback(open_browser=True) as auth:
        await auth.redirect_handler(AUTH_URL)

    assert opened == [AUTH_URL]


def test_the_state_compared_is_the_one_the_sdk_sent():
    """Read out of the URL, not generated here: one source of truth for the value."""
    assert state_of(AUTH_URL) == STATE
    assert (
        state_of("https://agent.robinhood.com/oauth/trading/authorize?response_type=code") is None
    )


@pytest.mark.parametrize("port", [-1, 70000])
def test_an_impossible_port_is_refused(port: int):
    with pytest.raises(ValueError):
        LoopbackAuthorization(
            port=port,
            timeout_seconds=1.0,
            open_browser=False,
            announce=lambda line: None,
            redirect_uri_override=None,
        )


def test_a_non_positive_timeout_is_refused():
    with pytest.raises(ValueError):
        LoopbackAuthorization(
            port=0,
            timeout_seconds=0.0,
            open_browser=False,
            announce=lambda line: None,
            redirect_uri_override=None,
        )


def test_a_custom_scheme_override_is_registered_and_accepted_on_completion():
    """The phone app cannot intercept http://127.0.0.1; it registers tick://broker/callback."""
    with LoopbackAuthorization(
        port=0,
        timeout_seconds=5.0,
        open_browser=False,
        announce=lambda _line: None,
        redirect_uri_override="tick://broker/callback",
    ) as auth:
        assert auth.redirect_uri == "tick://broker/callback"
        auth._expected_state = "st4te"
        auth.complete_redirect_url("tick://broker/callback?code=c0de&state=st4te")
        assert auth._server.captured == {"code": "c0de", "state": "st4te"}
        with pytest.raises(CallbackError):
            auth.complete_redirect_url("http://127.0.0.1:1/tick/callback?code=x&state=st4te")


def test_an_override_that_is_not_a_custom_scheme_url_is_refused():
    with pytest.raises(ValueError):
        LoopbackAuthorization(
            port=0,
            timeout_seconds=5.0,
            open_browser=False,
            announce=lambda _line: None,
            redirect_uri_override="https://evil.example.invalid/callback?x=",
        )
