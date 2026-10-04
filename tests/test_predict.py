import pandas as pd
import pytest

from common.config import Settings
from common.storage import gw_path, read_parquet, write_parquet
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from predict import backtest, model
from predict.main import run as run_predict


def test_availability_rules() -> None:
    players = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "status": ["a", "d", "i", "u", "s"],
            "chance_of_playing_next_round": [None, 75.0, 0.0, None, 100.0],
        }
    )
    avail = model.availability(players)

    assert avail.loc[1] == 1.0  # no news
    assert avail.loc[2] == 0.75
    assert avail.loc[3] == 0.0
    assert avail.loc[4] == 0.0  # unavailable even with null chance
    assert avail.loc[5] == 0.0  # suspended overrides 100


def test_dry_run_job_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path), dry_run=True)
    run_ingest(settings, DryRunSource())
    gameweek = run_predict(settings)
    out = read_parquet(settings, gw_path(gameweek, "predictions"))

    assert gameweek == 3
    assert len(out) == 8
    assert out["xpts"].notna().all()
    # Player 6 has 50% chance of playing; the model output is scaled accordingly.
    p6 = out[out["id"] == 6].iloc[0]
    assert p6["availability"] == 0.5
    assert p6["xpts"] == pytest.approx(p6["xpts_raw"] * 0.5)
    # Dry runs stay out of the real data root.
    assert settings.data_root.endswith("dry_run")


def test_dry_run_never_resolves_to_the_bucket() -> None:
    settings = Settings(data_bucket="real-bucket", dry_run=True)
    assert "real-bucket" not in settings.data_root


def test_job_falls_back_to_ep_next_with_no_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path), dry_run=True)
    run_ingest(settings, DryRunSource())
    # Pretend it is gameweek 1: no earlier live data exists.
    for name in ("players", "teams", "fixtures"):
        write_parquet(settings, read_parquet(settings, gw_path(3, name)), gw_path(1, name))
    out = read_parquet(
        settings,
        gw_path(
            run_predict(Settings(data_dir=str(tmp_path), dry_run=True, gameweek=1)), "predictions"
        ),
    )

    # No fixtures in gameweek 1 of the dry-run fixture list, so blanks score zero.
    assert (out["xpts"] == 0).all()


def test_walk_forward_trains_only_on_earlier_gameweeks(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    from predict.features import build_features

    feats = build_features(
        synthetic["live"], synthetic["players"], synthetic["teams"], synthetic["fixtures"],
        list(range(1, 9)),
    )  # fmt: skip
    preds = backtest.walk_forward(feats, last_gw=8)

    assert sorted(set(preds["gameweek"])) == [3, 4, 5, 6, 7, 8]
    assert {"model", "baseline_l5", "baseline_season"} <= set(preds.columns)

    report = backtest.render_report(
        backtest.attach_ep_next(preds, Settings(data_dir="/nonexistent"))
    )
    assert "not yet available" in report  # honest about the missing ep_next benchmark
    assert "Verdict" in report


def test_report_when_no_history() -> None:
    assert "Not enough" in backtest.render_report(pd.DataFrame())
