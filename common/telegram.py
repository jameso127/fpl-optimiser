"""A minimal Telegram Bot API client: send a text message to a chat, optionally with link
buttons under it and an animated effect (private chats only)."""

import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_LENGTH = 4096  # Telegram rejects longer messages

# Telegram's built-in message effects. The ids are not in the Bot API docs (they come from the
# clients), so an unknown one is dropped and the message sent without it, never lost.
EFFECTS = {
    "fire": "5104841245755180586",
    "thumbs_up": "5107584321108051014",
    "party": "5046509860389126442",
}

Button = tuple[str, str]  # (label, url)


def ipv4_client() -> httpx.Client:
    """An HTTP client that only uses IPv4, and gives up on a connection after 5 seconds.

    api.telegram.org has an IPv6 address, but Cloud Run cannot send traffic over IPv6: a
    connection tried that way hung for the whole 30-second timeout before falling back to IPv4,
    so some bot replies took over 30 seconds. Binding to 0.0.0.0 rules IPv6 out.
    """
    return httpx.Client(
        timeout=httpx.Timeout(30, connect=5),
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    )


class TelegramError(RuntimeError):
    pass


class Sender(Protocol):
    def send(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None: ...

    def send_photo(
        self,
        chat_id: int,
        png: bytes,
        caption: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None: ...


def _keyboard(buttons: Sequence[Sequence[Button]]) -> dict[str, Any]:
    return {"inline_keyboard": [[{"text": t, "url": u} for t, u in row] for row in buttons]}


class TelegramSender:
    def __init__(self, token: str, client: httpx.Client | None = None) -> None:
        self._api = f"{API}/bot{token}"
        self._client = client or ipv4_client()

    def send(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if buttons:
            payload["reply_markup"] = _keyboard(buttons)
        self._with_effect("sendMessage", payload, effect_id)

    def typing(self, chat_id: int) -> None:
        """Show "typing..." in the chat until the next message arrives (or about 5 seconds)."""
        self._post("sendChatAction", {"chat_id": chat_id, "action": "typing"}, None)

    def send_photo(
        self,
        chat_id: int,
        png: bytes,
        caption: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        # Multipart: the picture is uploaded with the message, nothing is hosted anywhere.
        data = {"chat_id": str(chat_id), "caption": caption, "parse_mode": "HTML"}
        if buttons:
            data["reply_markup"] = json.dumps(_keyboard(buttons))  # form fields are strings
        self._with_effect("sendPhoto", data, effect_id, {"photo": ("team.png", png, "image/png")})

    def _with_effect(
        self,
        method: str,
        payload: dict[str, Any],
        effect_id: str | None,
        files: dict[str, Any] | None = None,
    ) -> None:
        if effect_id:
            payload["message_effect_id"] = effect_id
        try:
            self._post(method, payload, files)
        except TelegramError as exc:
            if not effect_id or "effect" not in str(exc).lower():
                raise
            log.warning("message effect refused, sending without it", extra={"reason": str(exc)})
            del payload["message_effect_id"]
            self._post(method, payload, files)

    def _post(self, method: str, payload: dict[str, Any], files: dict[str, Any] | None) -> None:
        url = f"{self._api}/{method}"
        for attempt in range(2):
            try:
                if files:
                    response = self._client.post(url, data=payload, files=files)
                else:
                    response = self._client.post(url, json=payload)
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
