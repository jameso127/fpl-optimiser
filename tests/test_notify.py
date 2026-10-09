import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

import notify.telegram as telegram
from common.config import Settings
from common.users import InMemoryUserRepository, User
from notify.format import format_recommendation
from notify.main import next_deadline
from notify.main import run as notify_run
from notify.telegram import MAX_LENGTH, TelegramError, TelegramSender
from optimise.main import run as optimise_run
from tests import fakes

TOKEN = "123456:secret-token"


class RecordingSender:
    def __init__(self, fail_for: set[int] | None = None) -> None:
        self.sent: list[tuple[int, str]] = []
        self.fail_for = fail_for or set()

    def send(self, chat_id: int, text: str) -> None:
        if chat_id in self.fail_for:
            raise TelegramError("boom")
        self.sent.append((chat_id, text))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=str(tmp_path), season=fakes.SEASON)


@pytest.fixture
def world(settings: Settings) -> fakes.World:
    return fakes.seed_storage(settings)


def _recommend(settings: Settings, world: fakes.World, team_id: int) -> dict[str, Any]:
    return optimise_run(settings, fakes.FakeEntrySource(world), team_id)


# --- message formatting -----------------------------------------------------------------------


def test_a_recommendation_becomes_readable_messages(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    messages = format_recommendation(rec)
    text = "\n".join(messages)

    assert f"Gameweek {fakes.GAMEWEEK}" in text
    assert "Starting XI" in text and "Captain" in text and "Bench:" in text
    assert "Every option" in text and "Free transfers left" in text
    assert all(len(m) <= MAX_LENGTH for m in messages)


def test_the_deadline_is_shown_in_uk_time(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    deadline = dt.datetime(2025, 8, 29, 17, 30, tzinfo=dt.UTC)  # 18:30 BST

    assert "Deadline Fri 29 Aug, 18:30 UK" in format_recommendation(rec, deadline)[0]
    assert "Deadline" not in format_recommendation(rec)[0]


def test_team_table_rows_fit_a_phone_screen(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec["options"][0]["starting_xi"][0]["name"] = "Alexander-Arnold-Longname"
    text = "\n".join(format_recommendation({**rec, "recommended_transfers": 0}))
    tables = [t.split("</pre>")[0] for t in text.split("<pre>")[1:]]

    assert "Alexander-Ar…" in text
    assert all(len(row) <= 30 for table in tables for row in table.split("\n"))


def test_the_verdict_says_hold_or_how_many_transfers(
    settings: Settings, world: fakes.World
) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    hold = {**rec, "recommended_transfers": 0}
    moved = next(o for o in rec["options"] if o["transfers"] > 0)
    move = {**rec, "recommended_transfers": moved["transfers"]}

    hold_text = "".join(format_recommendation(hold))
    move_text = "".join(format_recommendation(move))
    first = moved["moves"][0]

    assert "Hold this week" in hold_text and "Transfers" not in hold_text
    assert f"Make {moved['transfers']} transfer" in move_text and "Transfers" in move_text
    assert f"{first['out']['name']} ➜ <b>{first['in']['name']}</b>" in move_text


def test_text_from_outside_is_escaped_for_telegram_html(
    settings: Settings, world: fakes.World
) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec["assumptions"] = ["<script>alert(1)</script> & more"]
    rec["options"][0]["bench"][0]["name"] = "<b>Mallory</b>"

    text = "\n".join(format_recommendation(rec))

    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "<b>Mallory</b>" not in text


def test_long_recommendations_are_split_without_breaking_a_block(
    settings: Settings, world: fakes.World
) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec["assumptions"] = [f"assumption number {i} " + "x" * 200 for i in range(40)]
    messages = format_recommendation(rec)

    assert len(messages) > 1
    assert all(len(m) <= MAX_LENGTH for m in messages)
    for m in messages:
        assert m.count("<pre>") == m.count("</pre>")


# --- the Telegram client ----------------------------------------------------------------------


def _sender(handler: Any) -> TelegramSender:
    return TelegramSender(TOKEN, httpx.Client(transport=httpx.MockTransport(handler)))


def test_a_message_is_posted_as_html_to_the_chat() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    _sender(handler).send(42, "<b>hi</b>")
    body = json.loads(seen[0].content)

    assert seen[0].url.path == f"/bot{TOKEN}/sendMessage"
    assert body["chat_id"] == 42 and body["text"] == "<b>hi</b>" and body["parse_mode"] == "HTML"


def test_a_rate_limit_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    waited: list[float] = []
    monkeypatch.setattr(telegram.time, "sleep", waited.append)
    replies = iter(
        [httpx.Response(429, json={"parameters": {"retry_after": 3}}), httpx.Response(200, json={})]
    )

    _sender(lambda request: next(replies)).send(1, "x")

    assert waited == [3.0]


def test_a_refusal_raises_with_telegrams_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"description": "Forbidden: bot was blocked by the user"})

    with pytest.raises(TelegramError, match="blocked by the user"):
        _sender(handler).send(1, "x")


def test_errors_never_contain_the_bot_token() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(TelegramError) as raised:
        _sender(unreachable).send(1, "x")

    assert TOKEN not in str(raised.value) and raised.value.__cause__ is None
    assert TOKEN not in repr(Settings(telegram_bot_token=SecretStr(TOKEN)))


# --- the job ----------------------------------------------------------------------------------


def _repo(*users: User) -> InMemoryUserRepository:
    repo = InMemoryUserRepository()
    for user in users:
        repo.save(user)
    return repo


def test_each_active_user_gets_their_own_recommendation(
    settings: Settings, world: fakes.World
) -> None:
    _recommend(settings, world, 11)
    _recommend(settings, world, 22)
    repo = _repo(
        User(chat_id=1, fpl_team_id=11),
        User(chat_id=2, fpl_team_id=22),
        User(chat_id=3, fpl_team_id=33, active=False),  # paused: skipped, so no file needed
    )
    sender = RecordingSender()

    assert notify_run(settings, repo, sender) == 2
    assert {chat for chat, _ in sender.sent} == {1, 2}


def test_rerunning_does_not_send_a_gameweek_twice(settings: Settings, world: fakes.World) -> None:
    _recommend(settings, world, 11)
    repo = _repo(User(chat_id=1, fpl_team_id=11))
    sender = RecordingSender()

    assert notify_run(settings, repo, sender) == 1
    sent_once = len(sender.sent)
    assert notify_run(settings, repo, sender) == 0
    assert len(sender.sent) == sent_once


def test_a_failed_send_can_be_retried_and_does_not_block_other_users(
    settings: Settings, world: fakes.World
) -> None:
    _recommend(settings, world, 11)
    _recommend(settings, world, 22)
    repo = _repo(User(chat_id=1, fpl_team_id=11), User(chat_id=2, fpl_team_id=22))
    flaky = RecordingSender(fail_for={1})

    with pytest.raises(RuntimeError, match="1 notification"):
        notify_run(settings, repo, flaky)
    assert {chat for chat, _ in flaky.sent} == {2}  # the other user still got theirs

    retry = RecordingSender()
    assert notify_run(settings, repo, retry) == 1  # only the one that failed
    assert {chat for chat, _ in retry.sent} == {1}


def test_a_user_with_no_recommendation_is_reported_not_skipped_silently(
    settings: Settings, world: fakes.World
) -> None:
    _recommend(settings, world, 22)
    repo = _repo(User(chat_id=1, fpl_team_id=11), User(chat_id=2, fpl_team_id=22))
    sender = RecordingSender()

    with pytest.raises(RuntimeError, match="1 notification"):
        notify_run(settings, repo, sender)
    assert {chat for chat, _ in sender.sent} == {2}


def test_it_only_sends_on_the_day_of_the_next_deadline() -> None:
    # The fake API's next deadline (gameweek 3) is 2025-08-29 17:30 UTC.
    source = fakes.FakeFplSource()

    on_the_day = next_deadline(source, dt.datetime(2025, 8, 29, 9, 0, tzinfo=dt.UTC))
    day_before = next_deadline(source, dt.datetime(2025, 8, 28, 9, 0, tzinfo=dt.UTC))

    assert on_the_day is not None and on_the_day.deadline_day
    assert day_before is not None and not day_before.deadline_day
    assert day_before.deadline == dt.datetime(2025, 8, 29, 17, 30, tzinfo=dt.UTC)


def test_no_upcoming_deadline_means_nothing_is_sent() -> None:
    class SeasonOver(fakes.FakeFplSource):
        def bootstrap(self) -> dict[str, Any]:
            boot = super().bootstrap()
            return {**boot, "events": [{**e, "is_next": False} for e in boot["events"]]}

    assert not next_deadline(SeasonOver(), dt.datetime(2025, 8, 29, 9, 0, tzinfo=dt.UTC))


def test_apostrophes_in_names_are_left_alone(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec["options"][0]["bench"][0]["name"] = "O'Reilly"

    assert "O'Reilly" in "\n".join(format_recommendation({**rec, "recommended_transfers": 0}))
