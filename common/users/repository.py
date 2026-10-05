"""The storage port for users. Code depends on this interface, not on a database.

Two implementations: `InMemoryUserRepository` (tests, local runs) and
`FirestoreUserRepository` (production). A shared contract test suite runs against both, so
they cannot drift apart.

Operations that must not race (redeeming an invite, claiming a notification) are single
methods that the implementation makes atomic, so callers never do read-modify-write.
"""

import datetime as dt
from typing import Protocol

from common.users.models import DeclaredTransfer, Invite, User


class UserRepository(Protocol):
    def get(self, chat_id: int) -> User | None: ...

    def save(self, user: User) -> None: ...

    def delete(self, chat_id: int) -> None:
        """Remove the user and everything stored about them (the `/delete` command)."""
        ...

    def list_active(self) -> list[User]:
        """Active users who have linked an FPL team: who the batch jobs work for."""
        ...

    # --- transfers the user says they have already made --------------------------------
    def add_declared(self, chat_id: int, transfer: DeclaredTransfer) -> None: ...

    def declared(self, chat_id: int, season: str, gameweek: int) -> list[DeclaredTransfer]:
        """In the order they were declared."""
        ...

    def undo_declared(self, chat_id: int, season: str, gameweek: int) -> DeclaredTransfer | None:
        """Remove and return the most recently declared transfer, if any."""
        ...

    def clear_declared(self, chat_id: int, season: str, gameweek: int) -> int: ...

    # --- invitations --------------------------------------------------------------------
    def create_invite(self, invite: Invite) -> None: ...

    def redeem_invite(self, code: str, chat_id: int, now: dt.datetime) -> bool:
        """Atomically spend one use of a valid, unexpired invite and register the chat.

        True if the chat is now registered (including when it already was, in which case no
        use is spent). False for an unknown, exhausted or expired code.
        """
        ...

    # --- notifications ------------------------------------------------------------------
    def claim_notification(self, chat_id: int, season: str, gameweek: int) -> bool:
        """Atomically mark a gameweek as notified. True only for the caller that claimed it,
        so a re-run or a second worker never sends the same message twice."""
        ...

    def release_notification(self, chat_id: int, season: str, gameweek: int) -> None:
        """Undo a claim when sending failed, so a retry is allowed."""
        ...
