"""Notify job: send each active user their recommendation for the latest gameweek on Telegram.

A user is only ever sent a gameweek once (the claim is atomic), so re-running the job, or
running two copies, does not send duplicates. If sending fails, the claim is released so a
retry can send it.

The pipeline runs daily, but this only sends when the next deadline is today (UK time);
FORCE_NOTIFY=true sends anyway.

Each user gets pictures, not paragraphs: the team on a pitch (captioned with the verdict, with an
animated effect), then the transfers with each player's recent form, with link buttons to FPL.
If the pictures cannot be drawn, everything is sent as text instead: a missing picture must never
cost the advice itself.

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... FPL_TEAM_ID=... uv run python -m notify.main
"""

import datetime as dt
import logging
from typing import Any

from common import schedule
from common.config import Settings, get_settings
from common.fpl_client import FplClient, FplSource
from common.logging import configure_logging
from common.storage import latest_gameweek, read_json, recommendation_path
from common.users import UserRepository, get_user_repository
from notify.format import format_recommendation, notes, summary
from notify.pitch import draw_pitch
from notify.telegram import EFFECTS, Button, Sender, TelegramSender
from notify.transfers import draw_transfers

log = logging.getLogger(__name__)


def buttons(settings: Settings) -> list[list[Button]]:
    site = settings.fpl_site_url.rstrip("/")
    return [[("🔁 Make transfers", f"{site}/transfers"), ("👕 Pick team", f"{site}/my-team")]]


def effect_for(rec: dict[str, Any]) -> str:
    """Fire when there are transfers to make, a thumbs up when holding is the advice."""
    return EFFECTS["fire"] if rec["recommended_transfers"] else EFFECTS["thumbs_up"]


def send_recommendation(
    settings: Settings,
    sender: Sender,
    chat_id: int,
    rec: dict[str, Any],
    deadline: dt.datetime | None,
) -> None:
    """The pitch picture captioned with the verdict, then (when there are transfers) the
    transfers picture captioned with the assumptions; buttons on the last, effect on the first."""
    try:
        pitch, moves = draw_pitch(rec, deadline), draw_transfers(rec)
    except Exception:
        log.exception("could not draw the pictures, sending text only")
        messages = format_recommendation(rec, deadline)
        for i, message in enumerate(messages):
            sender.send(
                chat_id,
                message,
                buttons=buttons(settings) if i == len(messages) - 1 else (),
                effect_id=effect_for(rec) if i == 0 else None,
            )
        return
    if moves is None:  # holding: one picture says it all
        photos = [(pitch, f"{summary(rec, deadline)}\n\n{notes(rec)}")]
    else:
        photos = [(pitch, summary(rec, deadline)), (moves, notes(rec))]
    for i, (png, caption) in enumerate(photos):
        sender.send_photo(
            chat_id,
            png,
            caption,
            buttons=buttons(settings) if i == len(photos) - 1 else (),
            effect_id=effect_for(rec) if i == 0 else None,
        )


def next_deadline(source: FplSource, now: dt.datetime) -> schedule.Schedule | None:
    """The next gameweek's deadline and whether it is today (UK time); None if there is none."""
    plan = schedule.plan(source.bootstrap()["events"], now)
    log.info(
        "schedule",
        extra={
            "next_gameweek": plan and plan.gameweek,
            "deadline": plan and plan.deadline.isoformat(),
            "deadline_day": bool(plan and plan.deadline_day),
        },
    )
    return plan


def run(
    settings: Settings, repo: UserRepository, sender: Sender, deadline: dt.datetime | None = None
) -> int:
    """Send to every active user. Returns how many were sent; raises if any failed.
    The deadline, when known, is shown at the top of the message."""
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
            send_recommendation(settings, sender, user.chat_id, rec, deadline)
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
    plan = next_deadline(FplClient(settings), dt.datetime.now(dt.UTC))
    if not (plan and plan.deadline_day) and not settings.force_notify:
        log.info("the next deadline is not today, so nothing was sent")
        return
    sender = TelegramSender(settings.telegram_bot_token.get_secret_value())
    deadline = plan.deadline if plan else None  # shown in the message, also on forced sends
    run(settings, get_user_repository(settings), sender, deadline)


if __name__ == "__main__":
    main()
