import pandas as pd

from common.config import Settings
from common.storage import exists, gw_path, read_parquet
from ingest import transform
from ingest.dry_run import DryRunSource
from ingest.main import run


def _settings(tmp_path, **kwargs) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=str(tmp_path), data_bucket=None, dry_run=True, **kwargs)


def test_dry_run_writes_snapshot_and_live(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    gameweek = run(settings, DryRunSource())

    assert gameweek == 3
    for name in ("players", "teams", "events", "fixtures"):
        assert exists(settings, gw_path(3, name))
    assert exists(settings, gw_path(1, "live"))
    assert exists(settings, gw_path(2, "live"))
    assert not exists(settings, gw_path(3, "live"))


def test_players_keep_null_chance_of_playing_and_numeric_types(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    run(settings, DryRunSource())
    players = read_parquet(settings, gw_path(3, "players"))

    assert players["chance_of_playing_next_round"].isna().sum() == 7
    assert pd.api.types.is_float_dtype(players["ep_next"])
    assert len(players) == 8


def test_rerun_is_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    run(settings, DryRunSource())
    first = read_parquet(settings, gw_path(3, "players"))
    run(settings, DryRunSource())
    second = read_parquet(settings, gw_path(3, "players"))

    pd.testing.assert_frame_equal(first, second)


def test_gameweek_override(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path, gameweek=2)
    assert run(settings, DryRunSource()) == 2
    assert exists(settings, gw_path(2, "players"))
    assert exists(settings, gw_path(1, "live"))
    assert not exists(settings, gw_path(2, "live"))


def test_target_gameweek_falls_back_when_season_over() -> None:
    bootstrap = {"events": [{"id": 37, "finished": True}, {"id": 38, "finished": True}]}
    assert transform.target_gameweek(bootstrap) == 38
