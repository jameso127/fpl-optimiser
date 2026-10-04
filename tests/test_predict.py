import pandas as pd
import pytest

from common.config import Settings
from common.storage import gw_path, read_parquet, write_parquet
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from predict import backtest, model
from predict.features import build_features
from predict.main import run as run_predict

SEASON = "2025-26"


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
    out = read_parquet(settings, gw_path(SEASON, gameweek, "predictions"))

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
        write_parquet(
            settings,
            read_parquet(settings, gw_path(SEASON, 3, name)),
            gw_path(SEASON, 1, name),
        )
    gameweek = run_predict(Settings(data_dir=str(tmp_path), dry_run=True, gameweek=1))
    out = read_parquet(settings, gw_path(SEASON, gameweek, "predictions"))

    # No fixtures in gameweek 1 of the dry-run fixture list, so blanks score zero.
    assert (out["xpts"] == 0).all()


def test_predict_trains_on_earlier_seasons(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path), dry_run=True)
    run_ingest(settings, DryRunSource())
    # Copy the season's data as an earlier season: it must be picked up as training history.
    for gw in (1, 2, 3):
        for name in ("live", "players", "teams", "fixtures"):
            try:
                df = read_parquet(settings, gw_path(SEASON, gw, name))
            except FileNotFoundError:
                continue
            write_parquet(settings, df, gw_path("2024-25", gw, name))

    run_predict(settings)
    out = read_parquet(settings, gw_path(SEASON, 3, "predictions"))
    assert len(out) == 8


def _two_season_features(synthetic: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for season in ("2024-25", "2025-26"):
        f = build_features(
            synthetic["live"], synthetic["players"], synthetic["teams"], synthetic["fixtures"],
            list(range(1, 9)),
        )  # fmt: skip
        f.insert(0, "season", season)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_walk_forward_never_trains_on_the_test_gameweek_or_later(
    synthetic: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    feats = _two_season_features(synthetic)
    seen: list[pd.DataFrame] = []
    real_train = backtest.model.train

    def spy(x: pd.DataFrame, y: pd.Series):  # type: ignore[no-untyped-def]
        seen.append(x)
        return real_train(x, y)

    monkeypatch.setattr(backtest.model, "train", spy)
    labelled = feats.assign(row=range(len(feats)))
    preds = backtest.walk_forward(feats, ["2025-26"])

    assert sorted(set(preds["gameweek"])) == [3, 4, 5, 6, 7, 8]
    assert set(preds["season"]) == {"2025-26"}
    assert len(seen) == 6
    # The earlier season is always fully available; the test season only before gameweek g.
    for g, train_x in zip(range(3, 9), seen, strict=True):
        rows = labelled.loc[train_x.index]
        assert (rows["season"] == "2024-25").sum() == (
            (labelled["season"] == "2024-25") & (labelled["fix_n"] >= 1)
        ).sum()
        in_season = rows[rows["season"] == "2025-26"]
        assert (in_season["gameweek"] < g).all()


def test_report_is_honest_about_missing_ep_next(synthetic: dict[str, pd.DataFrame]) -> None:
    preds = backtest.walk_forward(_two_season_features(synthetic), ["2025-26"])
    report = backtest.render_report(
        backtest.attach_ep_next(preds, Settings(data_dir="/nonexistent"))
    )

    assert "not available" in report
    assert "Verdict" in report
    assert "2025-26: all players" in report


def test_ep_next_is_only_scored_on_rows_that_have_it() -> None:
    preds = pd.DataFrame(
        {
            "season": "2025-26",
            "gameweek": [3, 3, 4, 4],
            "target": [2.0, 4.0, 1.0, 5.0],
            "min_l5": 90.0,
            "model": [2.0, 4.0, 1.0, 5.0],
            "baseline_l5": [1.0, 1.0, 1.0, 1.0],
            "baseline_season": [1.0, 1.0, 1.0, 1.0],
            "ep_next": [None, None, 3.0, 3.0],  # not captured for gameweek 3
        }
    )
    tables = backtest.summarise(preds)

    all_players = tables["2025-26: all players"]
    assert "ep_next" not in set(all_players["method"])
    assert set(all_players["rows"]) == {4}
    h2h = tables["2025-26: head-to-head with ep_next, all players"]
    assert set(h2h["rows"]) == {2}  # every method scored on the same two rows
    assert "1 of 2 tested gameweeks" in backtest.render_report(preds)


def test_report_when_no_history() -> None:
    assert "Not enough" in backtest.render_report(pd.DataFrame())
