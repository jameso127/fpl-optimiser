from collections.abc import Callable
from typing import Any

import httpx
import pytest

from common.config import Settings
from common.fpl_client import FplClient


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> FplClient:
    settings = Settings(fpl_min_interval_seconds=0)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return FplClient(settings, client=http, backoff_base=0.0, **kwargs)


def test_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    assert _client(handler).get("bootstrap-static/") == {"ok": True}
    assert calls["n"] == 3


def test_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _client(handler, max_retries=2).get("fixtures/")


def test_responses_are_cached() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[1])

    client = _client(handler)
    client.get("fixtures/")
    client.get("fixtures/")
    assert calls["n"] == 1


def test_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).get("event/99/live/")
    assert calls["n"] == 1
