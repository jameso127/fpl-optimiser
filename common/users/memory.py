"""In-memory user repository: for tests and local runs. Thread-safe, nothing persisted."""

import datetime as dt
import threading

from common.users.models import (
    DeclaredTransfer,
    Invite,
    User,
    gameweek_key,
    is_valid_invite_code,
)


class InMemoryUserRepository:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._users: dict[int, User] = {}
        self._declared: dict[int, list[DeclaredTransfer]] = {}
        self._invites: dict[str, Invite] = {}

    def get(self, chat_id: int) -> User | None:
        with self._lock:
            user = self._users.get(chat_id)
            return user.model_copy() if user else None

    def save(self, user: User) -> None:
        with self._lock:
            self._users[user.chat_id] = user.model_copy()

    def delete(self, chat_id: int) -> None:
        with self._lock:
            self._users.pop(chat_id, None)
            self._declared.pop(chat_id, None)

    def list_active(self) -> list[User]:
        with self._lock:
            return [
                u.model_copy()
                for u in sorted(self._users.values(), key=lambda u: u.chat_id)
                if u.active and u.fpl_team_id is not None
            ]

    def add_declared(self, chat_id: int, transfer: DeclaredTransfer) -> None:
        with self._lock:
            self._declared.setdefault(chat_id, []).append(transfer.model_copy())

    def declared(self, chat_id: int, season: str, gameweek: int) -> list[DeclaredTransfer]:
        with self._lock:
            found = [
                t.model_copy()
                for t in self._declared.get(chat_id, [])
                if t.season == season and t.gameweek == gameweek
            ]
        return sorted(found, key=lambda t: t.declared_at)

    def undo_declared(self, chat_id: int, season: str, gameweek: int) -> DeclaredTransfer | None:
        with self._lock:
            mine = [
                t
                for t in self._declared.get(chat_id, [])
                if t.season == season and t.gameweek == gameweek
            ]
            if not mine:
                return None
            latest = max(mine, key=lambda t: t.declared_at)
            self._declared[chat_id].remove(latest)
            return latest.model_copy()

    def clear_declared(self, chat_id: int, season: str, gameweek: int) -> int:
        with self._lock:
            before = self._declared.get(chat_id, [])
            keep = [t for t in before if not (t.season == season and t.gameweek == gameweek)]
            self._declared[chat_id] = keep
            return len(before) - len(keep)

    def create_invite(self, invite: Invite) -> None:
        with self._lock:
            self._invites[invite.code] = invite.model_copy()

    def redeem_invite(self, code: str, chat_id: int, now: dt.datetime) -> bool:
        if not is_valid_invite_code(code):
            return False
        with self._lock:
            if chat_id in self._users:
                return True  # already registered: nothing to spend
            invite = self._invites.get(code)
            if invite is None or invite.uses_left <= 0 or invite.expires_at <= now:
                return False
            invite.uses_left -= 1
            self._users[chat_id] = User(chat_id=chat_id)
            return True

    def claim_notification(self, chat_id: int, season: str, gameweek: int) -> bool:
        key = gameweek_key(season, gameweek)
        with self._lock:
            user = self._users.get(chat_id)
            if user is None or user.last_notified == key:
                return False
            user.last_notified = key
            return True

    def release_notification(self, chat_id: int, season: str, gameweek: int) -> None:
        with self._lock:
            user = self._users.get(chat_id)
            if user is not None and user.last_notified == gameweek_key(season, gameweek):
                user.last_notified = None
