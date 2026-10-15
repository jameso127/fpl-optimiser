from typing import Any

import pytest
import requests

from common.config import Settings
from common.fpl_client import FplClient
from tests.fakes import Handler, http_response, http_session


def _client(handler: Handler, **kwargs: Any) -> FplClient:
    settings = Settings(fpl_min_interval_seconds=0)
    return FplClient(settings, session=http_session(handler), backoff_base=0.0, **kwargs)


def test_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: requests.PreparedRequest) -> requests.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return http_response(429)
        return http_response(200, {"ok": True})

    assert _client(handler).get("bootstrap-static/") == {"ok": True}
    assert calls["n"] == 3


def test_gives_up_after_max_retries() -> None:
    def handler(request: requests.PreparedRequest) -> requests.Response:
        return http_response(503)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _client(handler, max_retries=2).get("fixtures/")


def test_connection_errors_are_retried() -> None:
    calls = {"n": 0}

    def handler(request: requests.PreparedRequest) -> requests.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("connection reset")
        return http_response(200, [1])

    assert _client(handler).get("fixtures/") == [1]
    assert calls["n"] == 2


def test_responses_are_cached() -> None:
    calls = {"n": 0}

    def handler(request: requests.PreparedRequest) -> requests.Response:
        calls["n"] += 1
        return http_response(200, [1])

    client = _client(handler)
    client.get("fixtures/")
    client.get("fixtures/")
    assert calls["n"] == 1


def test_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: requests.PreparedRequest) -> requests.Response:
        calls["n"] += 1
        return http_response(404)

    with pytest.raises(requests.HTTPError):
        _client(handler).get("event/99/live/")
    assert calls["n"] == 1


def test_sends_the_configured_user_agent() -> None:
    seen: list[str] = []

    def handler(request: requests.PreparedRequest) -> requests.Response:
        seen.append(str(request.headers["User-Agent"]))
        return http_response(200, {})

    settings = Settings(fpl_min_interval_seconds=0, fpl_user_agent="test-agent/1")
    FplClient(settings, session=http_session(handler)).get("bootstrap-static/")
    assert seen == ["test-agent/1"]
