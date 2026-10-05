"""A minimal Telegram Bot API client: send a text message to a chat."""

import time
from typing import Protocol

import httpx

API = "https://api.telegram.org"
MAX_LENGTH = 4096  # Telegram rejects longer messages


class TelegramError(RuntimeError):
    pass


class Sender(Protocol):
    def send(self, chat_id: int, text: str) -> None: ...


class TelegramSender:
    def __init__(self, token: str, client: httpx.Client | None = None) -> None:
        self._url = f"{API}/bot{token}/sendMessage"
        self._client = client or httpx.Client(timeout=15)

    def send(self, chat_id: int, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        for attempt in range(2):
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.HTTPError as exc:
                # Not `from exc`: httpx error messages contain the URL, which contains the token.
                raise TelegramError(f"could not reach Telegram ({type(exc).__name__})") from None
            if response.status_code == 429 and attempt == 0:
                time.sleep(float(response.json().get("parameters", {}).get("retry_after", 1)))
                continue
            if response.is_success:
                return
            reason = response.json().get("description", response.text) if response.content else ""
            raise TelegramError(f"Telegram refused the message ({response.status_code}): {reason}")
