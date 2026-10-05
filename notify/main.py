"""Notify job: send each active user their recommendation for the latest gameweek on Telegram.

A user is only ever sent a gameweek once (the claim is atomic), so re-running the job, or
running two copies, does not send duplicates. If sending fails, the claim is released so a
retry can send it.

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... FPL_TEAM_ID=... uv run python -m notify.main
"""

import logging

from common.config import Settings, get_settings
from common.logging import configure_logging
from common.storage import latest_gameweek, read_json, recommendation_path
from common.users import UserRepository, get_user_repository
from notify.format import format_recommendation
from notify.telegram import Sender, TelegramSender

log = logging.getLogger(__name__)


def run(settings: Settings, repo: UserRepository, sender: Sender) -> int:
    """Send to every active user. Returns how many were sent; raises if any failed."""
    season, gameweek = latest_gameweek(settings, "predictions", "run predict and optimise first")
    sent = failed = 0
    for user in repo.list_active():
        assert user.fpl_team_id is not None  # list_active only returns users with a team
        path = recommendation_path(season, gameweek, user.fpl_team_id)
        try:
            rec = read_json(settings, path)
        except FileNotFoundError:
            failed += 1
            log.error("no recommendation to send", extra={"team": user.fpl_team_id, "path": path})
            continue
        if not repo.claim_notification(user.chat_id, season, gameweek):
            log.info("already notified", extra={"gameweek": gameweek})
            continue
        try:
            for message in format_recommendation(rec):
                sender.send(user.chat_id, message)
        except Exception:
            repo.release_notification(user.chat_id, season, gameweek)
            failed += 1
            log.exception("send failed", extra={"team": user.fpl_team_id})
            continue
        sent += 1
        log.info("notified", extra={"team": user.fpl_team_id, "gameweek": gameweek})
    if failed:
        raise RuntimeError(f"{failed} notification(s) failed")
    return sent


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.telegram_bot_token is None:
        raise RuntimeError("set TELEGRAM_BOT_TOKEN")
    sender = TelegramSender(settings.telegram_bot_token.get_secret_value())
    run(settings, get_user_repository(settings), sender)


if __name__ == "__main__":
    main()
