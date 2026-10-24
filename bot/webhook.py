"""The webhook's shared secret, and registering the webhook with Telegram (done once, by hand).

Telegram sends the secret in the X-Telegram-Bot-Api-Secret-Token header of every update, and
the bot drops requests without it: the service is public, so this is what keeps strangers from
posting fake updates. It is derived from the bot token, so there is no second secret to store;
a new token means a new secret, and the webhook must be registered again.

Register (or re-register) after the first deploy, or whenever the token changes, with the
service URL from the infra repo's `terraform output bot_url`:

    TELEGRAM_BOT_TOKEN=... uv run python -m bot.webhook https://fpl-bot-....run.app

This changes Telegram's settings for the bot, not anything in Google Cloud.
"""

import argparse
import hashlib

import httpx

from common.config import get_settings
from common.telegram import API

SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"

COMMANDS = [
    ("start", "Register, or see your status"),
    ("team", "Link your FPL team: /team 1234567"),
    ("pause", "Stop the deadline-day messages"),
    ("resume", "Start the messages again"),
    ("delete", "Forget everything about you"),
    ("help", "What I do"),
]


def webhook_secret(token: str) -> str:
    """64 hex characters (Telegram allows A-Z, a-z, 0-9, _ and -, up to 256)."""
    return hashlib.sha256(f"fpl-bot-webhook:{token}".encode()).hexdigest()


def register(token: str, url: str, client: httpx.Client) -> None:
    calls = {
        "setWebhook": {
            "url": url,
            "secret_token": webhook_secret(token),
            "allowed_updates": ["message"],
            "max_connections": 1,  # one at a time, matching the single-threaded server
        },
        "setMyCommands": {"commands": [{"command": c, "description": d} for c, d in COMMANDS]},
    }
    for method, payload in calls.items():
        try:
            response = client.post(f"{API}/bot{token}/{method}", json=payload)
        except httpx.HTTPError as exc:
            # Not `from exc`: httpx error messages contain the URL, which contains the token.
            raise SystemExit(f"could not reach Telegram ({type(exc).__name__})") from None
        body = response.json() if response.content else {}
        if not body.get("ok"):
            raise SystemExit(f"{method} failed: {body.get('description', response.status_code)}")
        print(f"{method}: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="the fpl-bot service URL (https://...)")
    args = parser.parse_args()
    if not args.url.startswith("https://"):
        raise SystemExit("Telegram only calls https:// webhooks")

    token = get_settings().telegram_bot_token
    if token is None:
        raise SystemExit("set TELEGRAM_BOT_TOKEN")
    with httpx.Client(timeout=30) as client:
        register(token.get_secret_value(), args.url, client)


if __name__ == "__main__":
    main()
