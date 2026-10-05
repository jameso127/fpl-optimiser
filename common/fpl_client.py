"""Polite FPL API client: throttled, cached, exponential backoff, descriptive User-Agent."""

import logging
import time
from typing import Any, Protocol

import httpx

from common.config import Settings

log = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class FplSource(Protocol):
    """What ingest needs; the dry-run source implements this with fixture data."""

    def bootstrap(self) -> dict[str, Any]: ...
    def fixtures(self) -> list[dict[str, Any]]: ...
    def live(self, gameweek: int) -> dict[str, Any]: ...


class FplClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        base_url: str | None = None,
    ) -> None:
        """`base_url` lets other polite, public sources (e.g. CSV on GitHub) reuse the client."""
        self._base = (base_url or settings.fpl_base_url).rstrip("/")
        self._min_interval = settings.fpl_min_interval_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._client = client or httpx.Client(
            headers={"User-Agent": settings.fpl_user_agent}, timeout=30.0
        )
        self._cache: dict[str, Any] = {}
        self._last_request = 0.0

    def get(self, path: str, raw: bool = False) -> Any:
        """GET `path` (relative to the base URL), parsed as JSON or, with `raw`, as bytes.

        Responses are cached for the client's lifetime.
        """
        if path in self._cache:
            return self._cache[path]

        url = f"{self._base}/{path.lstrip('/')}"
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                error: str = repr(exc)
            else:
                self._last_request = time.monotonic()
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    data = response.content if raw else response.json()
                    self._cache[path] = data
                    return data
                error = f"HTTP {response.status_code}"

            if attempt == self._max_retries:
                raise RuntimeError(
                    f"FPL request failed after {attempt + 1} attempts: {url} ({error})"
                )
            delay = self._backoff_base * 2**attempt
            log.warning("retrying FPL request", extra={"url": url, "error": error, "delay": delay})
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def bootstrap(self) -> dict[str, Any]:
        result: dict[str, Any] = self.get("bootstrap-static/")
        return result

    def fixtures(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self.get("fixtures/")
        return result

    def live(self, gameweek: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"event/{gameweek}/live/")
        return result

    # Per-manager endpoints (used by the optimiser to rebuild a squad).
    def entry_history(self, team_id: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"entry/{team_id}/history/")
        return result

    def entry_picks(self, team_id: int, gameweek: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"entry/{team_id}/event/{gameweek}/picks/")
        return result

    def entry_transfers(self, team_id: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self.get(f"entry/{team_id}/transfers/")
        return result

    def element_summary(self, player_id: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"element-summary/{player_id}/")
        return result
