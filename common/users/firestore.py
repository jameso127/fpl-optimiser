"""Firestore implementation of the user repository.

Layout (Native mode, scale-to-zero, free tier at this size):

    users/{any id}                     a User, found by its chat_id field
    users/{any id}/declared/{auto}     a DeclaredTransfer (TTL: expires_at)
    invites/{code}                     an Invite           (TTL: expires_at)

A user document's id means nothing: the chat_id field is the key, so documents added by hand
in the console can keep an auto id. Lookups are equality queries on chat_id (single-field
indexes are automatic). If two documents ever share a chat_id, the first one is used.

Operations that must be atomic (redeeming an invite, claiming a notification, undoing a
declared transfer) run in Firestore transactions. Queries use only equality filters, which
need no composite index; ordering is done in Python.

Tested by the shared contract suite against the Firestore emulator (see
tests/test_users_contract.py); set FIRESTORE_EMULATOR_HOST to run it.
"""

import datetime as dt
import logging
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from common.users.models import (
    DeclaredTransfer,
    Invite,
    User,
    gameweek_key,
    is_valid_invite_code,
)

_BATCH = 400  # Firestore allows 500 writes per batch

log = logging.getLogger(__name__)


def _data(snapshot: Any) -> dict[str, Any]:
    """A snapshot's fields (empty for a missing document)."""
    data: dict[str, Any] | None = snapshot.to_dict()
    return data or {}


class FirestoreUserRepository:
    def __init__(self, client: firestore.Client) -> None:
        self._db = client

    # --- references -----------------------------------------------------------------------
    def _find(self, chat_id: int, txn: firestore.Transaction | None = None) -> Any | None:
        """The snapshot of the user document with this chat_id, or None."""
        query = self._db.collection("users").where(filter=FieldFilter("chat_id", "==", chat_id))
        docs = list(query.limit(2).stream(transaction=txn))
        if len(docs) > 1:
            log.warning("several user documents share a chat id; using the first")
        return docs[0] if docs else None

    def _declared_query(
        self, user: firestore.DocumentReference, season: str, gameweek: int
    ) -> firestore.Query:
        return (
            user.collection("declared")
            .where(filter=FieldFilter("season", "==", season))
            .where(filter=FieldFilter("gameweek", "==", gameweek))
        )

    # --- users ----------------------------------------------------------------------------
    def get(self, chat_id: int) -> User | None:
        snap = self._find(chat_id)
        return User.model_validate(_data(snap)) if snap else None

    def save(self, user: User) -> None:
        snap = self._find(user.chat_id)
        ref = snap.reference if snap else self._db.collection("users").document()
        ref.set(user.model_dump())

    def delete(self, chat_id: int) -> None:
        snap = self._find(chat_id)
        if snap is None:
            return
        self._delete_all(snap.reference.collection("declared").stream())
        snap.reference.delete()

    def list_active(self) -> list[User]:
        query = self._db.collection("users").where(filter=FieldFilter("active", "==", True))
        users = [User.model_validate(_data(d)) for d in query.stream()]
        return sorted((u for u in users if u.fpl_team_id is not None), key=lambda u: u.chat_id)

    # --- declared transfers ---------------------------------------------------------------
    def add_declared(self, chat_id: int, transfer: DeclaredTransfer) -> None:
        snap = self._find(chat_id)
        if snap is None:
            raise LookupError("no user with this chat id")
        snap.reference.collection("declared").add(transfer.model_dump())

    def declared(self, chat_id: int, season: str, gameweek: int) -> list[DeclaredTransfer]:
        snap = self._find(chat_id)
        if snap is None:
            return []
        docs = self._declared_query(snap.reference, season, gameweek).stream()
        found = [DeclaredTransfer.model_validate(_data(d)) for d in docs]
        return sorted(found, key=lambda t: t.declared_at)

    def undo_declared(self, chat_id: int, season: str, gameweek: int) -> DeclaredTransfer | None:
        snap = self._find(chat_id)
        if snap is None:
            return None
        query = self._declared_query(snap.reference, season, gameweek)

        @firestore.transactional
        def undo(txn: firestore.Transaction) -> DeclaredTransfer | None:
            docs = list(query.stream(transaction=txn))
            if not docs:
                return None
            latest = max(docs, key=lambda d: _data(d)["declared_at"])
            txn.delete(latest.reference)
            return DeclaredTransfer.model_validate(_data(latest))

        result: DeclaredTransfer | None = undo(self._db.transaction())
        return result

    def clear_declared(self, chat_id: int, season: str, gameweek: int) -> int:
        snap = self._find(chat_id)
        if snap is None:
            return 0
        return self._delete_all(self._declared_query(snap.reference, season, gameweek).stream())

    def _delete_all(self, docs: Any) -> int:
        batch, pending, total = self._db.batch(), 0, 0
        for doc in docs:
            batch.delete(doc.reference)
            pending += 1
            total += 1
            if pending == _BATCH:
                batch.commit()
                batch, pending = self._db.batch(), 0
        if pending:
            batch.commit()
        return total

    # --- invitations ----------------------------------------------------------------------
    def create_invite(self, invite: Invite) -> None:
        self._db.collection("invites").document(invite.code).set(invite.model_dump())

    def redeem_invite(self, code: str, chat_id: int, now: dt.datetime) -> bool:
        if not is_valid_invite_code(code):
            return False
        invite_ref = self._db.collection("invites").document(code)

        @firestore.transactional
        def redeem(txn: firestore.Transaction) -> bool:
            if self._find(chat_id, txn):
                return True  # already registered: nothing to spend
            snap = invite_ref.get(transaction=txn)
            if not snap.exists:
                return False
            invite = Invite.model_validate(_data(snap))
            if invite.uses_left <= 0 or invite.expires_at <= now:
                return False
            txn.update(invite_ref, {"uses_left": invite.uses_left - 1})
            txn.set(self._db.collection("users").document(), User(chat_id=chat_id).model_dump())
            return True

        result: bool = redeem(self._db.transaction())
        return result

    # --- notifications --------------------------------------------------------------------
    def claim_notification(self, chat_id: int, season: str, gameweek: int) -> bool:
        key = gameweek_key(season, gameweek)

        @firestore.transactional
        def claim(txn: firestore.Transaction) -> bool:
            snap = self._find(chat_id, txn)
            if snap is None or _data(snap).get("last_notified") == key:
                return False
            txn.update(snap.reference, {"last_notified": key})
            return True

        result: bool = claim(self._db.transaction())
        return result

    def release_notification(self, chat_id: int, season: str, gameweek: int) -> None:
        key = gameweek_key(season, gameweek)

        @firestore.transactional
        def release(txn: firestore.Transaction) -> None:
            snap = self._find(chat_id, txn)
            if snap and _data(snap).get("last_notified") == key:
                txn.update(snap.reference, {"last_notified": None})

        release(self._db.transaction())
