"""Contract tests for the user repository.

The same suite runs against every implementation, so the in-memory version used by the other
tests cannot drift from the Firestore one used in production. The Firestore leg needs the
emulator: set FIRESTORE_EMULATOR_HOST (CI starts one); it is skipped otherwise.
"""

import datetime as dt
import os
from collections.abc import Iterator

import httpx
import pytest

from common.config import Settings
from common.users import (
    DeclaredTransfer,
    InMemoryUserRepository,
    Invite,
    User,
    UserRepository,
    get_user_repository,
)
from common.users.models import DECLARED_TRANSFER_RETENTION, is_valid_invite_code

NOW = dt.datetime(2026, 10, 10, 9, 0, tzinfo=dt.UTC)
EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST")
PROJECT = "demo-fpl-test"


@pytest.fixture(params=["memory", "firestore"])
def repo(request: pytest.FixtureRequest) -> Iterator[UserRepository]:
    if request.param == "memory":
        yield InMemoryUserRepository()
        return
    if not EMULATOR:
        pytest.skip("set FIRESTORE_EMULATOR_HOST to run the Firestore leg")
    from google.cloud import firestore

    from common.users.firestore import FirestoreUserRepository

    httpx.delete(f"http://{EMULATOR}/emulator/v1/projects/{PROJECT}/databases/(default)/documents")
    yield FirestoreUserRepository(firestore.Client(project=PROJECT))


def _transfer(out_id: int = 1, in_id: int = 2, minutes: int = 0, gw: int = 6) -> DeclaredTransfer:
    return DeclaredTransfer(
        season="2026-27", gameweek=gw, out_id=out_id, in_id=in_id, out_price=55, in_price=60,
        declared_at=NOW + dt.timedelta(minutes=minutes),
    )  # fmt: skip


def test_a_user_round_trips_and_a_missing_one_is_none(repo: UserRepository) -> None:
    assert repo.get(1) is None
    repo.save(User(chat_id=1, fpl_team_id=99, max_transfers=2, free_transfers_override=3))
    got = repo.get(1)

    assert got is not None
    assert (got.fpl_team_id, got.max_transfers, got.free_transfers_override) == (99, 2, 3)
    assert got.active and got.last_notified is None


def test_only_active_users_with_a_team_are_listed_in_order(repo: UserRepository) -> None:
    repo.save(User(chat_id=3, fpl_team_id=30))
    repo.save(User(chat_id=1, fpl_team_id=10))
    repo.save(User(chat_id=2, fpl_team_id=20, active=False))  # paused
    repo.save(User(chat_id=4))  # registered but no team linked yet

    assert [u.chat_id for u in repo.list_active()] == [1, 3]


def test_declared_transfers_come_back_in_order_for_the_right_gameweek(
    repo: UserRepository,
) -> None:
    repo.save(User(chat_id=1))
    repo.add_declared(1, _transfer(out_id=3, in_id=4, minutes=5))
    repo.add_declared(1, _transfer(out_id=1, in_id=2, minutes=1))
    repo.add_declared(1, _transfer(out_id=7, in_id=8, gw=7))  # a different gameweek

    got = repo.declared(1, "2026-27", 6)
    assert [(t.out_id, t.in_id) for t in got] == [(1, 2), (3, 4)]
    assert repo.declared(1, "2026-27", 7)[0].in_id == 8
    assert repo.declared(1, "2025-26", 6) == []
    assert repo.declared(2, "2026-27", 6) == []  # another user's transfers are private


def test_undo_removes_only_the_latest_declared_transfer(repo: UserRepository) -> None:
    repo.save(User(chat_id=1))
    assert repo.undo_declared(1, "2026-27", 6) is None
    repo.add_declared(1, _transfer(out_id=1, in_id=2, minutes=1))
    repo.add_declared(1, _transfer(out_id=3, in_id=4, minutes=2))

    undone = repo.undo_declared(1, "2026-27", 6)
    assert undone is not None and (undone.out_id, undone.in_id) == (3, 4)
    assert [(t.out_id, t.in_id) for t in repo.declared(1, "2026-27", 6)] == [(1, 2)]


def test_clearing_reports_how_many_were_removed(repo: UserRepository) -> None:
    repo.save(User(chat_id=1))
    repo.add_declared(1, _transfer())
    repo.add_declared(1, _transfer(out_id=5, in_id=6))
    repo.add_declared(1, _transfer(gw=7))

    assert repo.clear_declared(1, "2026-27", 6) == 2
    assert repo.declared(1, "2026-27", 6) == []
    assert len(repo.declared(1, "2026-27", 7)) == 1


def test_deleting_a_user_removes_everything_about_them(repo: UserRepository) -> None:
    repo.save(User(chat_id=1, fpl_team_id=5))
    repo.save(User(chat_id=2, fpl_team_id=6))
    repo.add_declared(1, _transfer())
    repo.add_declared(2, _transfer())
    repo.delete(1)

    assert repo.get(1) is None and repo.declared(1, "2026-27", 6) == []
    assert repo.get(2) is not None and len(repo.declared(2, "2026-27", 6)) == 1
    repo.delete(1)  # deleting again is harmless


def test_a_valid_invite_registers_the_chat_and_is_spent(repo: UserRepository) -> None:
    repo.create_invite(Invite(code="welcome-1", uses_left=1, expires_at=NOW + dt.timedelta(days=1)))

    assert repo.redeem_invite("welcome-1", 10, NOW) is True
    registered = repo.get(10)
    assert registered is not None and registered.fpl_team_id is None
    assert repo.redeem_invite("welcome-1", 11, NOW) is False  # the one use is gone
    assert repo.get(11) is None


def test_bad_invites_are_refused(repo: UserRepository) -> None:
    repo.create_invite(Invite(code="expired-1", expires_at=NOW - dt.timedelta(seconds=1)))
    repo.create_invite(Invite(code="spent-123", uses_left=0, expires_at=NOW + dt.timedelta(days=1)))

    for code in ("expired-1", "spent-123", "no-such-code", "bad code!", "x", "a/b/c/d/e"):
        assert repo.redeem_invite(code, 10, NOW) is False
    assert repo.get(10) is None


def test_an_already_registered_chat_does_not_spend_another_use(repo: UserRepository) -> None:
    repo.create_invite(Invite(code="welcome-1", uses_left=2, expires_at=NOW + dt.timedelta(days=1)))
    assert repo.redeem_invite("welcome-1", 10, NOW)
    assert repo.redeem_invite("welcome-1", 10, NOW)  # same chat again
    assert repo.redeem_invite("welcome-1", 11, NOW)  # the second (and last) use
    assert not repo.redeem_invite("welcome-1", 12, NOW)


def test_a_gameweek_can_be_claimed_for_notification_only_once(repo: UserRepository) -> None:
    repo.save(User(chat_id=1, fpl_team_id=5))

    assert repo.claim_notification(1, "2026-27", 6) is True
    assert repo.claim_notification(1, "2026-27", 6) is False  # a re-run must not resend
    assert repo.claim_notification(1, "2026-27", 7) is True  # the next gameweek is fine
    assert repo.claim_notification(999, "2026-27", 6) is False  # unknown chat

    repo.release_notification(1, "2026-27", 7)  # sending failed: allow a retry
    assert repo.claim_notification(1, "2026-27", 7) is True
    repo.release_notification(1, "2026-27", 6)  # not the latest claim: no effect
    assert repo.claim_notification(1, "2026-27", 7) is False


def test_invite_codes_use_a_safe_alphabet() -> None:
    assert is_valid_invite_code("abc-DEF_123")
    for bad in ("short", "has space", "slash/inside", "x" * 65, "", "émoji-✓-code"):
        assert not is_valid_invite_code(bad)
    with pytest.raises(ValueError):
        Invite(code="bad code")


def test_declared_transfers_expire_after_the_retention_period() -> None:
    t = DeclaredTransfer(season="2026-27", gameweek=6, out_id=1, in_id=2, out_price=1, in_price=1)

    assert t.expires_at - t.declared_at == DECLARED_TRANSFER_RETENTION


def test_the_factory_gives_memory_when_asked_and_never_touches_firestore() -> None:
    assert isinstance(get_user_repository(Settings(users_backend="memory")), InMemoryUserRepository)


def test_single_user_mode_has_just_the_owner() -> None:
    settings = Settings(users_backend="memory", telegram_chat_id=7, fpl_team_id=99)
    users = get_user_repository(settings).list_active()

    assert [(u.chat_id, u.fpl_team_id) for u in users] == [(7, 99)]
    assert get_user_repository(Settings(users_backend="memory")).list_active() == []
