"""Bot service: Telegram's webhook. Each message sent to the bot arrives as a POST, and the bot
replies (see bot/handlers.py) and registers users in Firestore.

Runs on Cloud Run, scaled to zero: it starts when a message arrives. The server is the standard
library's, single-threaded: a few users need nothing more, and one request at a time means two
quick messages from the same chat cannot register it twice.

Telegram re-sends an update until it gets a 2xx, so every update that passed the secret check
gets 200, even when handling it failed (the error is logged): a retry would fail the same way
and hold up that chat's later messages.

    USERS_BACKEND=memory TELEGRAM_BOT_TOKEN=... uv run python -m bot.main
"""

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

from bot.handlers import reply
from bot.webhook import SECRET_HEADER, webhook_secret
from common.config import Settings, get_settings
from common.fpl_client import FplClient
from common.logging import configure_logging
from common.telegram import Sender, TelegramError, TelegramSender
from common.users import UserRepository, get_user_repository

log = logging.getLogger(__name__)

MAX_BODY = 1_000_000  # Telegram updates are a few KB


class Bot:
    def __init__(
        self, settings: Settings, repo: UserRepository, sender: Sender, secret: str
    ) -> None:
        self.settings, self.repo, self.sender, self.secret = settings, repo, sender, secret

    def handle(self, update: dict[str, Any]) -> None:
        try:
            # A new FPL client per update: its cache lives as long as it does, and this
            # process can live for many messages.
            text = reply(update, self.repo, FplClient(self.settings), self.settings.max_users)
            if text is not None:
                self.sender.send(int(update["message"]["chat"]["id"]), text)
        except TelegramError as exc:
            log.warning("could not reply", extra={"error": str(exc)})
        except Exception:
            log.exception("could not handle an update")


class Server(HTTPServer):
    bot: Bot


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (the name http.server looks for)
        bot = cast(Server, self.server).bot
        if not hmac.compare_digest(self.headers.get(SECRET_HEADER, ""), bot.secret):
            self._respond(403)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            self._respond(400)
            return
        try:
            update = json.loads(self.rfile.read(length))
        except ValueError:
            self._respond(400)
            return
        if isinstance(update, dict):
            bot.handle(update)
        self._respond(200)

    def _respond(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the default writes every request to stderr; errors are logged above instead


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.telegram_bot_token is None:
        raise SystemExit("set TELEGRAM_BOT_TOKEN")
    token = settings.telegram_bot_token.get_secret_value()

    server = Server(("", settings.port), Handler)
    repo = get_user_repository(settings)
    server.bot = Bot(settings, repo, TelegramSender(token), webhook_secret(token))
    log.info("bot listening", extra={"port": settings.port})
    server.serve_forever()


if __name__ == "__main__":
    main()
