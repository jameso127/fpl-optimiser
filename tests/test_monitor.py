import logging

import pandas as pd
import pytest

from common.config import Settings
from common.storage import gw_path, read_parquet, write_parquet
from ml import monitor

SEASON = "2025-26"


def _store(settings: Settings, gw: int, xpts_good: bool, ep_good: bool = False) -> None:
    ids = list(range(1, 61))
    points = [float(i % 9) for i in ids]
    # A "good" forecast ranks by the outcome; a bad one is the reverse of it.
    xpts = [p + 0.1 for p in points] if xpts_good else [9 - p for p in points]
    ep_next = points if ep_good else [9 - p for p in points]
    write_parquet(
        settings,
        pd.DataFrame(
            {
                "id": ids, "gameweek": gw, "xpts": xpts, "ep_next": ep_next,
                "fix_n": 1, "model_version": "v7",
            }
        ),
        gw_path(SEASON, gw, "predictions"),
    )  # fmt: skip
    write_parquet(
        settings,
        pd.DataFrame({"id": ids, "gameweek": gw, "total_points": points}),
        gw_path(SEASON, gw, "live"),
    )


def test_live_performance_compares_served_predictions_with_results_and_ep_next(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    settings = Settings(data_dir=str(tmp_path))
    _store(settings, 3, xpts_good=True)
    # Gameweek 4 has predictions but no results yet: it must be ignored.
    write_parquet(
        settings,
        read_parquet(settings, gw_path(SEASON, 3, "predictions")).assign(gameweek=4),
        gw_path(SEASON, 4, "predictions"),
    )
    perf = monitor.live_performance(settings, SEASON)

    assert list(perf["gameweek"]) == [3]
    row = perf.iloc[0]
    assert row["model_version"] == "v7"
    assert row["top30_xpts"] > row["top30_ep_next"]
    assert row["spearman_xpts"] > 0.9 > 0 > row["spearman_ep_next"]


def test_a_blank_gameweek_player_is_not_scored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    _store(settings, 3, xpts_good=True)
    preds = read_parquet(settings, gw_path(SEASON, 3, "predictions"))
    preds.loc[preds["id"] <= 10, "fix_n"] = 0
    write_parquet(settings, preds, gw_path(SEASON, 3, "predictions"))

    assert monitor.live_performance(settings, SEASON).iloc[0]["players"] == 50


def test_warns_when_the_served_model_lately_trails_fpls_expected_points(  # type: ignore[no-untyped-def]
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(data_dir=str(tmp_path))
    for gw in range(1, 6):
        _store(settings, gw, xpts_good=False, ep_good=True)  # served model ranks worse
    with caplog.at_level(logging.WARNING):
        perf = monitor.write_live_performance(settings, SEASON)

    assert len(perf) == 5
    assert any("trails FPL ep_next" in r.getMessage() for r in caplog.records)
    stored = read_parquet(settings, f"monitoring/season={SEASON}/live_performance.parquet")
    assert len(stored) == 5


def test_no_warning_for_a_model_that_keeps_up_and_too_little_history() -> None:
    good = pd.DataFrame({"gameweek": range(1, 6), "top30_xpts": 5.0, "top30_ep_next": 4.0})
    assert monitor.check_recent(good) is None
    assert monitor.check_recent(good.head(2).assign(top30_xpts=0.0)) is None  # < 4 gameweeks
