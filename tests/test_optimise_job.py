import json
from collections import Counter

import httpx
import pytest

from common.config import Settings
from common.fpl_client import FplClient
from common.storage import read_json, recommendation_path
from optimise.main import build_pool, run
from tests import fakes


def _settings(tmp_path, **kwargs) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=str(tmp_path), season=fakes.SEASON, **kwargs)


def _run(tmp_path, **kwargs):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path, **kwargs)
    world = fakes.seed_storage(settings)
    return settings, world, run(settings, fakes.FakeEntrySource(world), fakes.TEAM_ID)


def test_writes_a_recommendation_for_every_number_of_transfers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, _, rec = _run(tmp_path, max_transfers=3)
    stored = read_json(settings, recommendation_path(fakes.SEASON, fakes.GAMEWEEK, fakes.TEAM_ID))

    assert stored == json.loads(json.dumps(rec))  # what was returned is what was stored
    assert [o["transfers"] for o in stored["options"]] == [0, 1, 2, 3]
    assert stored["recommended_transfers"] in {0, 1, 2, 3}
    assert stored["team_id"] == fakes.TEAM_ID and stored["gameweek"] == fakes.GAMEWEEK
    assert stored["model_version"] == "test-model"


def test_every_option_is_a_legal_team(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, _, rec = _run(tmp_path, max_transfers=4)

    for option in rec["options"]:
        xi, bench = option["starting_xi"], option["bench"]
        squad = xi + bench
        assert len(xi) == 11 and len(bench) == 4
        assert len({p["id"] for p in squad}) == 15
        positions = Counter(p["position"] for p in squad)
        assert positions == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        assert max(Counter(p["team"] for p in squad).values()) <= 3
        formation = Counter(p["position"] for p in xi)
        assert formation["GK"] == 1 and 3 <= formation["DEF"] <= 5 and 1 <= formation["FWD"] <= 3
        assert sum(p["captain"] for p in xi) == 1 and sum(p["vice_captain"] for p in xi) == 1
        assert len(option["moves"]) == option["transfers"]
        assert option["bank_after"] >= 0


def test_gain_is_measured_against_holding_and_hits_are_charged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, _, rec = _run(tmp_path, max_transfers=3)
    hold = rec["options"][0]

    assert rec["free_transfers"] == 1  # the fixture manager used the free transfer in gameweek 2
    assert hold["gain_vs_hold"] == 0 and hold["hit_cost"] == 0 and hold["moves"] == []
    for option in rec["options"]:
        assert option["gain_vs_hold"] == pytest.approx(
            option["net_xpts"] - hold["net_xpts"], abs=0.011
        )
        assert option["hit_cost"] == 4.0 * max(0, option["transfers"] - rec["free_transfers"])


def test_selling_prices_come_from_the_purchase_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, world, rec = _run(tmp_path)
    market = sum(world.purchase.values())
    # One player was bought 0.6m cheaper than he is now, so only half of that gain is ours.
    assert rec["squad_value"] * 10 == market - 3
    assert rec["bank"] == world.bank / 10


def test_rerunning_overwrites_with_the_same_answer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, world, first = _run(tmp_path)
    second = run(settings, fakes.FakeEntrySource(world), fakes.TEAM_ID)

    assert first["options"] == second["options"]
    assert first["recommended_transfers"] == second["recommended_transfers"]


def test_a_missing_team_id_or_predictions_is_a_clear_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    world = fakes.synthetic_world()
    with pytest.raises(RuntimeError, match="FPL_TEAM_ID"):
        run(settings, fakes.FakeEntrySource(world))

    empty = _settings(tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="no data found"):
        run(empty, fakes.FakeEntrySource(world), team_id=1)


def test_the_pool_has_positive_scorers_and_everyone_already_owned() -> None:
    world = fakes.synthetic_world()
    xpts = world.players.set_index("id")["xpts"]
    owned = set(world.squad)
    pool_ids = {p.id for p in build_pool(world.players, xpts, owned)}

    assert owned <= pool_ids
    assert all(i in owned or xpts[i] > 0 for i in pool_ids)
    assert not any(i not in owned and xpts[i] == 0 for i in pool_ids)


def test_the_client_exposes_the_per_manager_endpoints() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    client = FplClient(
        Settings(fpl_min_interval_seconds=0),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.entry_history(9)
    client.entry_picks(9, 5)
    client.entry_transfers(9)
    client.element_summary(77)

    assert seen == [
        "/api/entry/9/history/", "/api/entry/9/event/5/picks/", "/api/entry/9/transfers/",
        "/api/element-summary/77/",
    ]  # fmt: skip


def test_declared_transfers_change_the_squad_the_advice_starts_from(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import datetime as dt

    from common.users.models import DeclaredTransfer

    settings = _settings(tmp_path, max_transfers=2)
    world = fakes.seed_storage(settings)
    rows = {int(r["id"]): r for r in world.players.to_dict("records")}
    owned = set(world.squad)
    club_count = Counter(int(rows[i]["team"]) for i in owned)
    sold = next(i for i in world.squad if rows[i]["element_type"] == 3)
    allowance = world.purchase[sold] + world.bank  # what the manager can spend on his replacement
    candidates = [
        i for i, r in rows.items()
        if r["element_type"] == 3 and i not in owned and int(r["now_cost"]) <= allowance
        and (club_count[int(r["team"])] < 3 or r["team"] == rows[sold]["team"])
    ]  # fmt: skip
    bought = candidates[0]
    declared = DeclaredTransfer(
        season=fakes.SEASON, gameweek=fakes.GAMEWEEK, out_id=sold, in_id=bought,
        out_price=world.purchase[sold], in_price=int(rows[bought]["now_cost"]),
        declared_at=dt.datetime(2026, 10, 9, 8, 0, tzinfo=dt.UTC),
    )  # fmt: skip
    rec = run(settings, fakes.FakeEntrySource(world), fakes.TEAM_ID, declared=[declared])

    hold = rec["options"][0]
    held = {p["id"] for p in hold["starting_xi"] + hold["bench"]}
    assert bought in held and sold not in held  # the hold option is the squad *after* his move
    assert rec["planned_transfers"] == 1 and rec["free_transfers"] == 0  # it used the free one
    text = " ".join(rec["assumptions"])
    assert "Plus the transfers you told me about" in text and "Free transfers left: 0" in text
    assert rec["options"][1]["hit_cost"] == 4.0  # any further transfer is now a hit


def test_the_assumptions_always_say_what_the_squad_is_based_on(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, _, rec = _run(tmp_path)

    assert rec["assumptions"][0].startswith(
        "Your team as FPL shows it after the gameweek 2 deadline"
    )
    assert any(line.startswith("Free transfers left: 1 (estimated)") for line in rec["assumptions"])


def test_the_job_optimises_every_user_and_one_failure_does_not_stop_the_rest(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch
) -> None:
    import optimise.main as job
    from common.storage import exists
    from common.users import InMemoryUserRepository, User

    settings = _settings(tmp_path)
    world = fakes.seed_storage(settings)

    class Source(fakes.FakeEntrySource):
        def entry_history(self, team_id: int):  # type: ignore[no-untyped-def]
            if team_id == 2:
                raise RuntimeError("FPL is down for this team")
            return super().entry_history(team_id)

    repo = InMemoryUserRepository()
    for chat, team in ((10, 1), (20, 2), (30, 3)):
        repo.save(User(chat_id=chat, fpl_team_id=team))
    monkeypatch.setattr(job, "get_settings", lambda: settings)
    monkeypatch.setattr(job, "get_user_repository", lambda s: repo)
    monkeypatch.setattr(job, "FplClient", lambda s: Source(world))

    with pytest.raises(SystemExit, match="1 user"):
        job.main()

    for team, expected in ((1, True), (2, False), (3, True)):
        assert exists(settings, recommendation_path(fakes.SEASON, fakes.GAMEWEEK, team)) is expected
