import datetime as dt
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

import notify.main
import notify.telegram as telegram
from common.config import Settings
from common.users import InMemoryUserRepository, User
from notify.format import (
    CAPTION_LENGTH,
    details,
    fixtures_text,
    format_recommendation,
    notes,
    pick,
    summary,
)
from notify.main import next_deadline, send_recommendation
from notify.main import run as notify_run
from notify.pitch import draw_pitch
from notify.telegram import EFFECTS, MAX_LENGTH, Button, TelegramError, TelegramSender
from notify.transfers import draw_transfers
from optimise.main import run as optimise_run
from tests import fakes

TOKEN = "123456:secret-token"


class RecordingSender:
    """Records every message (photo captions included) as (chat id, text), plus the extras."""

    def __init__(self, fail_for: set[int] | None = None) -> None:
        self.sent: list[tuple[int, str]] = []
        self.photos: list[bytes] = []
        self.buttons: list[Sequence[Sequence[Button]]] = []
        self.effects: list[str | None] = []
        self.fail_for = fail_for or set()

    def send(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        if chat_id in self.fail_for:
            raise TelegramError("boom")
        self.sent.append((chat_id, text))
        self.buttons.append(buttons)
        self.effects.append(effect_id)

    def send_photo(
        self,
        chat_id: int,
        png: bytes,
        caption: str,
        *,
        buttons: Sequence[Sequence[Button]] = (),
        effect_id: str | None = None,
    ) -> None:
        if chat_id in self.fail_for:
            raise TelegramError("boom")
        self.photos.append(png)
        self.sent.append((chat_id, caption))
        self.buttons.append(buttons)
        self.effects.append(effect_id)


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


def test_the_summary_fits_a_photo_caption(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    caption = summary(rec, dt.datetime(2025, 8, 29, 17, 30, tzinfo=dt.UTC))

    assert len(caption) <= CAPTION_LENGTH
    assert "Gameweek" in caption and "Captain" in caption


def test_the_notes_say_what_the_advice_assumes(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec["assumptions"] = ["Free transfers left: 1 (estimated).", "Bought <cheap>."]

    assert notes(rec) == "ℹ️ <i>Free transfers left: 1 (estimated). Bought &lt;cheap&gt;.</i>"


def test_the_deadline_is_shown_in_uk_time(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    deadline = dt.datetime(2025, 8, 29, 17, 30, tzinfo=dt.UTC)  # 18:30 BST

    assert "Deadline Fri 29 Aug, 18:30 UK" in format_recommendation(rec, deadline)[0]
    assert "Deadline" not in format_recommendation(rec)[0]


def _with_stats(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The recommendation, set to make its first transfer option, with stats on that move."""
    option = next(o for o in rec["options"] if o["transfers"] > 0)
    move = option["moves"][0]
    move["in"].update(
        fixtures=[{"opponent": "ARS", "home": False, "difficulty": 4}], form=6.2, ownership=23.4
    )
    move["out"].update(chance_of_playing=25.0, news="Hamstring injury")
    return {**rec, "recommended_transfers": option["transfers"]}, move


def test_transfers_show_the_fixture_form_and_injury_news(
    settings: Settings, world: fakes.World
) -> None:
    rec, move = _with_stats(_recommend(settings, world, fakes.TEAM_ID))
    text = "\n".join(details(rec))

    assert "🟠 ARS (A) · form 6.2 · 23% owned" in text
    assert f"🚑 {move['out']['name']} 25%: Hamstring injury" in text


def test_doubtful_players_in_the_team_are_flagged(settings: Settings, world: fakes.World) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    best = pick(rec)
    best["bench"][0].update(chance_of_playing=75.0, news="Knock")
    text = "\n".join(details(rec))

    assert "Fitness doubts" in text and f"{best['bench'][0]['name']} 75%: Knock" in text
    assert "no game" not in text  # fixtures unknown here: nothing is said about them


def test_a_blank_gameweek_says_no_game() -> None:
    assert fixtures_text({"fixtures": []}) == "no game"
    assert fixtures_text({}) == ""
    double = {"fixtures": [{"opponent": "LIV", "home": True, "difficulty": 5},
                           {"opponent": "BOU", "home": False, "difficulty": 2}]}  # fmt: skip
    assert fixtures_text(double) == "🔴 LIV (H), 🟢 BOU (A)"


# --- the picture ------------------------------------------------------------------------------


def test_the_team_is_drawn_as_a_png(settings: Settings, world: fakes.World) -> None:
    rec, _ = _with_stats(_recommend(settings, world, fakes.TEAM_ID))
    pick(rec)["starting_xi"][0]["name"] = "João Pedro-Guéhi Extraordinaire"  # accents, too long

    png = draw_pitch(rec, dt.datetime(2025, 8, 29, 17, 30, tzinfo=dt.UTC))
    image = Image.open(io.BytesIO(png))

    assert image.format == "PNG" and image.size == (1080, 1350)
    assert draw_pitch({**rec, "recommended_transfers": 0})[:4] == b"\x89PNG"  # hold, no deadline


def test_the_transfers_are_drawn_with_form_charts(settings: Settings, world: fakes.World) -> None:
    rec, move = _with_stats(_recommend(settings, world, fakes.TEAM_ID))
    move["out"]["recent"] = [{"gameweek": gw, "points": p, "minutes": 90}
                             for gw, p in ((1, 2), (2, -1), (3, 12))]  # fmt: skip
    move["in"]["recent"] = [{"gameweek": 3, "points": 0, "minutes": 0}]  # did not play

    png = draw_transfers(rec)  # other moves have no history or fixtures: they still draw
    assert png is not None
    image = Image.open(io.BytesIO(png))

    assert image.format == "PNG" and image.size[0] == 1080
    assert draw_transfers({**rec, "recommended_transfers": 0}) is None  # nothing to show on hold


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
        assert m.count("<b>") == m.count("</b>")


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


def test_buttons_and_effect_are_sent_with_the_message() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    _sender(handler).send(1, "x", buttons=[[("FPL", "https://example.org/t")]], effect_id="42")

    assert seen[0]["reply_markup"] == {
        "inline_keyboard": [[{"text": "FPL", "url": "https://example.org/t"}]]
    }
    assert seen[0]["message_effect_id"] == "42"


def test_a_photo_is_uploaded_with_its_caption_and_buttons() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    _sender(handler).send_photo(
        7, b"\x89PNG-bytes", "<b>GW 3</b>", buttons=[[("FPL", "https://example.org/t")]]
    )
    body = seen[0].read()

    assert seen[0].url.path == f"/bot{TOKEN}/sendPhoto"
    assert b"\x89PNG-bytes" in body and b"<b>GW 3</b>" in body and b'name="chat_id"' in body
    assert b'"inline_keyboard": [[{"text": "FPL", "url": "https://example.org/t"}]]' in body


def test_an_unknown_effect_is_dropped_not_the_message() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "message_effect_id" in body:
            return httpx.Response(400, json={"description": "Bad Request: EFFECT_ID_INVALID"})
        return httpx.Response(200, json={"ok": True})

    _sender(handler).send(1, "x", effect_id="nope")

    assert len(bodies) == 2 and "message_effect_id" not in bodies[1]


# --- the job ----------------------------------------------------------------------------------


def test_transfers_are_sent_as_two_pictures_and_no_text(
    settings: Settings, world: fakes.World
) -> None:
    rec = _recommend(settings, world, fakes.TEAM_ID)
    rec, _ = _with_stats(rec)  # make sure there are transfers to show
    sender = RecordingSender()

    send_recommendation(settings, sender, 1, rec, None)

    assert len(sender.photos) == len(sender.sent) == 2  # pitch, then transfers; no text
    assert [caption for _, caption in sender.sent] == [summary(rec), notes(rec)]
    assert sender.effects == [EFFECTS["fire"], None]
    assert sender.buttons[0] == () and sender.buttons[1][0][0][1].endswith("/transfers")


def test_holding_is_one_picture_with_everything(settings: Settings, world: fakes.World) -> None:
    rec = {**_recommend(settings, world, fakes.TEAM_ID), "recommended_transfers": 0}
    sender = RecordingSender()

    send_recommendation(settings, sender, 1, rec, None)

    assert len(sender.photos) == len(sender.sent) == 1
    caption = sender.sent[0][1]
    assert "Hold this week" in caption and notes(rec) in caption
    assert len(caption) <= CAPTION_LENGTH
    assert sender.effects == [EFFECTS["thumbs_up"]] and sender.buttons[0]


def test_without_a_picture_everything_is_sent_as_text(
    settings: Settings, world: fakes.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any) -> bytes:
        raise OSError("no font")

    monkeypatch.setattr(notify.main, "draw_pitch", broken)
    rec = _recommend(settings, world, fakes.TEAM_ID)
    sender = RecordingSender()

    send_recommendation(settings, sender, 1, rec, None)

    assert sender.photos == []
    assert [text for _, text in sender.sent] == format_recommendation(rec)
    assert sender.effects[0] is not None and sender.buttons[-1]


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
