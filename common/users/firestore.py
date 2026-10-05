"""Firestore implementation of the user repository.

Layout (Native mode, scale-to-zero, free tier at this size):

    users/{chat_id}                    a User
    users/{chat_id}/declared/{auto}    a DeclaredTransfer (TTL: expires_at)
    invites/{code}                     an Invite           (TTL: expires_at)

Operations that must be atomic (redeeming an invite, claiming a notification, undoing a
declared transfer) run in Firestore transactions. Queries use only equality filters, which
need no composite index; ordering is done in Python.

Tested by the shared contract suite against the Firestore emulator (see
tests/test_users_contract.py); set FIRESTORE_EMULATOR_HOST to run it.
"""

import datetime as dt
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


def _data(snapshot: Any) -> dict[str, Any]:
    """A snapshot's fields (empty for a missing document)."""
    data: dict[str, Any] | None = snapshot.to_dict()
    return data or {}


class FirestoreUserRepository:
    def __init__(self, client: firestore.Client) -> None:
        self._db = client

    # --- references -----------------------------------------------------------------------
    def _user(self, chat_id: int) -> firestore.DocumentReference:
        return self._db.collection("users").document(str(chat_id))

    def _declared_collection(self, chat_id: int) -> firestore.CollectionReference:
        return self._user(chat_id).collection("declared")

    def _declared_query(self, chat_id: int, season: str, gameweek: int) -> firestore.Query:
        return (
            self._declared_collection(chat_id)
            .where(filter=FieldFilter("season", "==", season))
            .where(filter=FieldFilter("gameweek", "==", gameweek))
        )

    # --- users ----------------------------------------------------------------------------
    def get(self, chat_id: int) -> User | None:
        snap = self._user(chat_id).get()
        return User.model_validate(_data(snap)) if snap.exists else None

    def save(self, user: User) -> None:
        self._user(user.chat_id).set(user.model_dump())

    def delete(self, chat_id: int) -> None:
        self._delete_all(self._declared_collection(chat_id).stream())
        self._user(chat_id).delete()

    def list_active(self) -> list[User]:
        query = self._db.collection("users").where(filter=FieldFilter("active", "==", True))
        users = [User.model_validate(_data(d)) for d in query.stream()]
        return sorted((u for u in users if u.fpl_team_id is not None), key=lambda u: u.chat_id)

    # --- declared transfers ---------------------------------------------------------------
    def add_declared(self, chat_id: int, transfer: DeclaredTransfer) -> None:
        self._declared_collection(chat_id).add(transfer.model_dump())

    def declared(self, chat_id: int, season: str, gameweek: int) -> list[DeclaredTransfer]:
        docs = self._declared_query(chat_id, season, gameweek).stream()
        found = [DeclaredTransfer.model_validate(_data(d)) for d in docs]
        return sorted(found, key=lambda t: t.declared_at)

    def undo_declared(self, chat_id: int, season: str, gameweek: int) -> DeclaredTransfer | None:
        query = self._declared_query(chat_id, season, gameweek)

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
        return self._delete_all(self._declared_query(chat_id, season, gameweek).stream())

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
        user_ref = self._user(chat_id)
        invite_ref = self._db.collection("invites").document(code)

        @firestore.transactional
        def redeem(txn: firestore.Transaction) -> bool:
            if user_ref.get(transaction=txn).exists:
                return True  # already registered: nothing to spend
            snap = invite_ref.get(transaction=txn)
            if not snap.exists:
                return False
            invite = Invite.model_validate(_data(snap))
            if invite.uses_left <= 0 or invite.expires_at <= now:
                return False
            txn.update(invite_ref, {"uses_left": invite.uses_left - 1})
            txn.set(user_ref, User(chat_id=chat_id).model_dump())
            return True

        result: bool = redeem(self._db.transaction())
        return result

    # --- notifications --------------------------------------------------------------------
    def claim_notification(self, chat_id: int, season: str, gameweek: int) -> bool:
        key = gameweek_key(season, gameweek)
        ref = self._user(chat_id)

        @firestore.transactional
        def claim(txn: firestore.Transaction) -> bool:
            snap = ref.get(transaction=txn)
            if not snap.exists or _data(snap).get("last_notified") == key:
                return False
            txn.update(ref, {"last_notified": key})
            return True

        result: bool = claim(self._db.transaction())
        return result

    def release_notification(self, chat_id: int, season: str, gameweek: int) -> None:
        key = gameweek_key(season, gameweek)
        ref = self._user(chat_id)

        @firestore.transactional
        def release(txn: firestore.Transaction) -> None:
            snap = ref.get(transaction=txn)
            if snap.exists and _data(snap).get("last_notified") == key:
                txn.update(ref, {"last_notified": None})

        release(self._db.transaction())
