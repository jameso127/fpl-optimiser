import json
from collections import Counter

import httpx
import pytest

from common.config import Settings, get_settings
from common.fpl_client import FplClient
from common.storage import read_json
from optimise import dry_run
from optimise.main import build_pool, recommendation_path, run
from optimise.main import main as optimise_main


def _settings(tmp_path, **kwargs) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=str(tmp_path), dry_run=True, season=dry_run.SEASON, **kwargs)


def _run(tmp_path, **kwargs):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path, **kwargs)
    world = dry_run.seed_storage(settings)
    return settings, world, run(settings, dry_run.DryRunEntrySource(world), dry_run.TEAM_ID)


def test_dry_run_writes_a_recommendation_for_every_number_of_transfers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings, _, rec = _run(tmp_path, max_transfers=3)
    stored = read_json(
        settings, recommendation_path(dry_run.SEASON, dry_run.GAMEWEEK, dry_run.TEAM_ID)
    )

    assert stored == json.loads(json.dumps(rec))  # what was returned is what was stored
    assert [o["transfers"] for o in stored["options"]] == [0, 1, 2, 3]
    assert stored["recommended_transfers"] in {0, 1, 2, 3}
    assert stored["team_id"] == dry_run.TEAM_ID and stored["gameweek"] == dry_run.GAMEWEEK
    assert stored["model_version"] == "dry-run"


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
    second = run(settings, dry_run.DryRunEntrySource(world), dry_run.TEAM_ID)

    assert first["options"] == second["options"]
    assert first["recommended_transfers"] == second["recommended_transfers"]


def test_a_missing_team_id_or_predictions_is_a_clear_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    world = dry_run.synthetic_world()
    with pytest.raises(RuntimeError, match="FPL_TEAM_ID"):
        run(settings, dry_run.DryRunEntrySource(world))

    empty = _settings(tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="no data found"):
        run(empty, dry_run.DryRunEntrySource(world), team_id=1)


def test_the_pool_has_positive_scorers_and_everyone_already_owned() -> None:
    world = dry_run.synthetic_world()
    xpts = world.players.set_index("id")["xpts"]
    owned = set(world.squad)
    pool_ids = {p.id for p in build_pool(world.players, xpts, owned)}

    assert owned <= pool_ids
    assert all(i in owned or xpts[i] > 0 for i in pool_ids)
    assert not any(i not in owned and xpts[i] == 0 for i in pool_ids)


def test_the_job_entrypoint_runs_in_dry_run_mode(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_BUCKET", raising=False)
    get_settings.cache_clear()
    try:
        optimise_main()
    finally:
        get_settings.cache_clear()

    path = (
        tmp_path
        / "dry_run"
        / recommendation_path(dry_run.SEASON, dry_run.GAMEWEEK, dry_run.TEAM_ID)
    )
    assert path.is_file()


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
