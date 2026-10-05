from pathlib import Path
from typing import Any

import pandas as pd

from common.config import Settings
from common.storage import available_gameweeks, available_seasons, exists, gw_path, read_parquet
from ingest import transform
from ingest.main import run
from tests.fakes import FakeFplSource

SEASON = "2025-26"  # derived from the fake API's gameweek 1 deadline (August 2025)


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    return Settings(data_dir=str(tmp_path), data_bucket=None, **kwargs)


def test_writes_the_snapshot_and_live_stats(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    gameweek = run(settings, FakeFplSource())

    assert gameweek == 3
    for name in ("players", "teams", "events", "fixtures"):
        assert exists(settings, gw_path(SEASON, 3, name))
    assert exists(settings, gw_path(SEASON, 1, "live"))
    assert exists(settings, gw_path(SEASON, 2, "live"))
    assert not exists(settings, gw_path(SEASON, 3, "live"))
    assert available_seasons(settings) == [SEASON]
    assert available_gameweeks(settings, SEASON, "live") == [1, 2]


def test_season_label() -> None:
    boot = {"events": [{"deadline_time": "2026-08-22T17:30:00Z"}, {"deadline_time": "2026-09-01Z"}]}
    assert transform.season_label(boot) == "2026-27"
    assert transform.season_label({"events": [{"deadline_time": "2099-12-31T00:00:00Z"}]}) == (
        "2099-00"
    )
    assert transform.season_label({"events": [{"deadline_time": "2027-02-01T00:00:00Z"}]}) == (
        "2026-27"
    )


def test_season_setting_overrides_derived_label(tmp_path: Path) -> None:
    settings = _settings(tmp_path, season="2030-31")
    run(settings, FakeFplSource())
    assert available_seasons(settings) == ["2030-31"]


def test_players_keep_null_chance_of_playing_and_numeric_types(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run(settings, FakeFplSource())
    players = read_parquet(settings, gw_path(SEASON, 3, "players"))

    assert players["chance_of_playing_next_round"].isna().sum() == 7
    assert pd.api.types.is_float_dtype(players["ep_next"])
    assert len(players) == 8


def test_live_stats_are_numeric(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run(settings, FakeFplSource())
    live = read_parquet(settings, gw_path(SEASON, 1, "live"))

    assert pd.api.types.is_float_dtype(live["ict_index"])
    assert pd.api.types.is_float_dtype(live["expected_goal_involvements"])


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run(settings, FakeFplSource())
    first = read_parquet(settings, gw_path(SEASON, 3, "players"))
    run(settings, FakeFplSource())
    second = read_parquet(settings, gw_path(SEASON, 3, "players"))

    pd.testing.assert_frame_equal(first, second)


def test_gameweek_override(tmp_path: Path) -> None:
    settings = _settings(tmp_path, gameweek=2)
    assert run(settings, FakeFplSource()) == 2
    assert exists(settings, gw_path(SEASON, 2, "players"))
    assert exists(settings, gw_path(SEASON, 1, "live"))
    assert not exists(settings, gw_path(SEASON, 2, "live"))


def test_target_gameweek_falls_back_when_season_over() -> None:
    bootstrap = {"events": [{"id": 37, "finished": True}, {"id": 38, "finished": True}]}
    assert transform.target_gameweek(bootstrap) == 38
