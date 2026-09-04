from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tick.serve.handlers import (
    APIError,
    browser_close,
    browser_frames,
    browser_input,
    browser_session_start,
    provider_browser_login_start,
)


class FakeBridge:
    def __init__(self) -> None:
        self.closed: dict[str, str] = {}
        self.inputs = []

    def open(self, url, viewport, purpose):
        self.opened = (url, viewport, purpose)
        return {"session_id": "browser-1", "origin": "https://login.example.invalid"}

    def knows(self, session_id):
        return session_id == "browser-1"

    def frames(self, session_id):
        assert session_id == "browser-1"
        yield 1234, b"fixture-jpeg", "https://login.example.invalid"

    def close_reason(self, session_id):
        return self.closed.get(session_id, "callback_received")

    def input(self, session_id, events):
        self.inputs.append((session_id, events))

    def close(self, session_id, reason):
        self.closed[session_id] = reason


class Context:
    def __init__(self, tmp_path):
        self.home = tmp_path / "tick-home"
        self.now = lambda: datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
        self.browser_bridge = FakeBridge()
        self.browser_ceremony_url = lambda _purpose: (
            "https://login.example.invalid/authorize?state=test"
        )
        self.provider_browser_login_start = lambda viewport: {
            "login_id": "login-browser-1",
            "session_id": "browser-1",
            "origin": "https://login.example.invalid",
        }


def test_session_route_refuses_a_host_the_box_did_not_produce(tmp_path):
    context = Context(tmp_path)
    with pytest.raises(APIError) as refused:
        browser_session_start(
            context,
            {
                "url": "https://unrelated.example.invalid/",
                "viewport": {"width": 390, "height": 760},
                "purpose": "broker_connect",
            },
        )
    assert refused.value.code == "BROWSER_URL_NOT_A_CEREMONY"
    assert refused.value.reason.endswith("open the URL it returns.")


def test_session_frames_input_and_close_have_the_closed_wire_shapes(tmp_path):
    context = Context(tmp_path)
    status, opened = browser_session_start(
        context,
        {
            "url": "https://login.example.invalid/authorize?private=value",
            "viewport": {"width": 390, "height": 760},
            "purpose": "broker_connect",
        },
    )
    assert status == 201
    assert opened == {
        "session_id": "browser-1",
        "origin": "https://login.example.invalid",
    }
    assert list(browser_frames(context, "browser-1")) == [
        {
            "t": 1234,
            "jpeg": "Zml4dHVyZS1qcGVn",
            "origin": "https://login.example.invalid",
            "done": False,
        },
        {"done": True, "reason": "callback_received"},
    ]
    assert browser_input(
        context,
        "browser-1",
        {"events": [{"kind": "tap", "x": 10, "y": 20}]},
    ) == (202, {"accepted": 1})
    assert browser_close(context, "browser-1") == (
        200,
        {"session_id": "browser-1", "reason": "user_closed"},
    )


def test_codex_browser_login_requires_and_forwards_the_viewport(tmp_path):
    context = Context(tmp_path)
    status, payload = provider_browser_login_start(
        context, {"viewport": {"width": 390, "height": 760}}
    )
    assert status == 202
    assert payload["session_id"] == "browser-1"

    with pytest.raises(APIError) as refused:
        provider_browser_login_start(context, {})
    assert refused.value.code == "BROWSER_VIEWPORT_INVALID"
    assert "visible frame size" in refused.value.reason


@pytest.mark.parametrize(
    ("body", "code", "sentence"),
    (
        ({}, "BROWSER_SESSION_INVALID", "Start that ceremony again"),
        (
            {
                "url": "https://login.example.invalid",
                "viewport": {"width": 390, "height": 760},
                "purpose": "other",
            },
            "BROWSER_PURPOSE_INVALID",
            "Start that ceremony and retry",
        ),
        (
            {
                "url": "https://login.example.invalid",
                "viewport": {"width": 0, "height": 760},
                "purpose": "broker_connect",
            },
            "BROWSER_VIEWPORT_INVALID",
            "visible frame size and retry",
        ),
    ),
)
def test_session_body_refusals_are_actionable(tmp_path, body, code, sentence):
    with pytest.raises(APIError) as refused:
        browser_session_start(Context(tmp_path), body)
    assert refused.value.code == code
    assert sentence in refused.value.reason


def test_frames_input_and_close_refuse_unknown_or_malformed_sessions(tmp_path):
    context = Context(tmp_path)
    with pytest.raises(APIError) as frames:
        browser_frames(context, "missing")
    assert frames.value.code == "BROWSER_SESSION_NOT_FOUND"
    assert frames.value.reason.endswith("if you still need its browser.")

    with pytest.raises(APIError) as events:
        browser_input(context, "browser-1", {})
    assert events.value.code == "BROWSER_EVENTS_REQUIRED"
    assert events.value.reason.endswith("Send the gesture again.")

    class MissingBridge(FakeBridge):
        def close(self, session_id, reason):
            from tick.serve.browser_bridge import BrowserBridgeError

            raise BrowserBridgeError(
                "BROWSER_SESSION_NOT_FOUND",
                "that browser session is not active. Start the ceremony again.",
            )

    context.browser_bridge = MissingBridge()
    with pytest.raises(APIError) as close:
        browser_close(context, "missing")
    assert close.value.code == "BROWSER_SESSION_NOT_FOUND"
    assert close.value.reason.endswith("Start the ceremony again.")
