"""What the bot says back. Pure logic: an update in, a reply out; no HTTP and no sending.

Anyone who messages the bot can register, up to MAX_USERS. A user is a chat plus a public FPL
team id (see common/users/models.py), so registering asks for nothing else.

    /start         register, and ask for a team id
    <number>       link that FPL team (also `/team <number>`)
    /pause         stop the deadline-day messages; /resume starts them again
    /delete        forget everything about this chat
    /help          what the bot does and the commands
"""

import html
import logging
import time
from typing import Any, Protocol

import requests

from common.users import User, UserRepository

log = logging.getLogger(__name__)

MAX_TEAM_ID = 1_000_000_000  # far above any real FPL id; stops silly numbers reaching the API

HOW_TO_FIND_ID = (
    "To find your team id, open <b>Points</b> on the FPL website: it's the number in the "
    "address, e.g. fantasy.premierleague.com/entry/<b>1234567</b>/event/8"
)

HELP = (
    "On the morning of each FPL deadline I send you the transfers I'd make, with your team on a "
    "pitch.\n\n"
    "/team &lt;id&gt; – link your FPL team (or just send the number)\n"
    "/pause – stop the messages · /resume – start them again\n"
    "/delete – forget everything about you\n"
    "/help – this message"
)


FULL = "Sorry, the bot is full right now, so I can't take new users."


class EntryLookup(Protocol):
    def entry(self, team_id: int) -> dict[str, Any]: ...


def reply(
    update: dict[str, Any], repo: UserRepository, fpl: EntryLookup, max_users: int
) -> str | None:
    """The reply to one Telegram update, or None when there is nothing to answer (not a text
    message, or not a private chat: the bot is not meant for groups)."""
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    text = (message.get("text") or "").strip()
    if chat.get("type") != "private" or not text or "id" not in chat:
        return None
    chat_id = int(chat["id"])

    if text.isdigit():
        return _link(chat_id, text, repo, fpl, max_users)
    command, _, arg = text.partition(" ")
    command = command.split("@")[0].lower()  # "/start@SomeBot" -> "/start"
    if command == "/start":
        return _start(chat_id, repo, max_users)
    if command == "/team":
        return _link(chat_id, arg.strip(), repo, fpl, max_users)
    if command in ("/pause", "/resume"):
        return _set_active(chat_id, repo, active=command == "/resume")
    if command == "/delete":
        repo.delete(chat_id)
        return "Done: I've forgotten everything about you. Send /start if you want to come back."
    return HELP


def _register(chat_id: int, repo: UserRepository, max_users: int) -> User | None:
    """The chat's user, registering it if there is room; None when the bot is full."""
    user = repo.get(chat_id)
    if user is not None:
        return user
    if repo.count() >= max_users:
        log.warning("registration refused: the bot is full", extra={"max_users": max_users})
        return None
    user = User(chat_id=chat_id)
    repo.save(user)
    log.info("user registered")
    return user


def _start(chat_id: int, repo: UserRepository, max_users: int) -> str:
    existing = repo.get(chat_id)
    user = _register(chat_id, repo, max_users)
    if user is None:
        return FULL
    if existing is None:
        return f"👋 Welcome! Send me your FPL team id to get started.\n\n{HOW_TO_FIND_ID}"
    if user.fpl_team_id is None:
        return f"You're registered but haven't linked a team yet.\n\n{HOW_TO_FIND_ID}"
    paused = " Messages are paused: /resume to start them again." if not user.active else ""
    return f"You're all set: I'm following team {user.fpl_team_id}.{paused}\n\n{HELP}"


def _link(chat_id: int, arg: str, repo: UserRepository, fpl: EntryLookup, max_users: int) -> str:
    if not arg.isdigit() or not 0 < int(arg) < MAX_TEAM_ID:
        return f"That doesn't look like a team id.\n\n{HOW_TO_FIND_ID}"
    team_id = int(arg)
    start = time.monotonic()
    try:
        entry = fpl.entry(team_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"I couldn't find FPL team {team_id}.\n\n{HOW_TO_FIND_ID}"
        log.warning("FPL team lookup failed", extra={"error": repr(exc)})
        return "I couldn't reach FPL just now. Please try again in a few minutes."
    except RuntimeError as exc:  # the client gave up after retrying
        log.warning("FPL team lookup failed", extra={"error": repr(exc)})
        return "I couldn't reach FPL just now. Please try again in a few minutes."
    finally:
        log.info("FPL team lookup", extra={"fpl_ms": round((time.monotonic() - start) * 1000)})

    user = _register(chat_id, repo, max_users)
    if user is None:
        return FULL
    repo.save(user.model_copy(update={"fpl_team_id": team_id}))
    name = html.escape(str(entry.get("name") or f"team {team_id}"))
    return (
        f"✅ Linked <b>{name}</b>. I'll message you on the morning of each deadline with the "
        "transfers I'd make.\n\n/help for the other commands."
    )


def _set_active(chat_id: int, repo: UserRepository, active: bool) -> str:
    user = repo.get(chat_id)
    if user is None:
        return "You're not registered yet: send /start."
    repo.save(user.model_copy(update={"active": active}))
    if active:
        return "▶️ Resumed: you'll get a message on the morning of each deadline."
    return "⏸ Paused: no more messages until you send /resume."
