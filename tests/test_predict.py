import numpy as np
import pandas as pd
import pytest

from common.config import Settings
from common.storage import gw_path, read_parquet, write_parquet
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from predict import backtest, features, model
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


def _ingested(tmp_path, **kwargs) -> Settings:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path), dry_run=True, **kwargs)
    run_ingest(settings, DryRunSource())
    return settings


def _predictions(settings: Settings, gameweek: int = 3) -> pd.DataFrame:
    return read_parquet(settings, gw_path(SEASON, gameweek, "predictions"))


def test_default_source_is_fpls_ep_next_used_as_is(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert Settings().xpts_source == "ep_next"
    settings = _ingested(tmp_path)
    assert run_predict(settings) == 3
    out = _predictions(settings).set_index("id")

    assert len(out) == 8
    # FPL's figure already includes chance of playing: player 6 (50%) is NOT scaled again.
    assert out.loc[6, "availability"] == 0.5
    assert out.loc[6, "xpts"] == pytest.approx(out.loc[6, "ep_next"])
    assert (out["xpts"] == out["ep_next"]).all()
    assert out["xpts_model"].isna().all()
    # Dry runs stay out of the real data root.
    assert settings.data_root.endswith("dry_run")


def test_ep_next_source_needs_no_history_or_live_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path)
    for gw in (1, 2):
        (tmp_path / "dry_run" / f"season={SEASON}" / f"gw={gw}" / "live.parquet").unlink()

    assert run_predict(settings) == 3  # nothing to train on, and it does not matter


def test_ep_next_source_zeroes_a_blank_gameweek_and_keeps_doubles_as_given(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    settings = _ingested(tmp_path)
    fixtures = read_parquet(settings, gw_path(SEASON, 3, "fixtures"))
    # Team 1 doubles up; team 4 has no fixture.
    extra = fixtures.iloc[[0]].assign(id=99, team_h=1, team_a=2)
    fixtures = pd.concat([fixtures, extra]).query("team_h != 3")
    write_parquet(settings, fixtures, gw_path(SEASON, 3, "fixtures"))
    run_predict(settings)
    out = _predictions(settings).set_index("id")
    players = read_parquet(settings, gw_path(SEASON, 3, "players")).set_index("id")

    team1 = players.index[players["team"] == 1]
    assert (out.loc[team1, "fix_n"] == 2).all()
    assert (out.loc[team1, "xpts"] == out.loc[team1, "ep_next"]).all()  # not doubled again
    blank = players.index[players["team"].isin([3, 4])]
    assert (out.loc[blank, "fix_n"] == 0).all()
    assert (out.loc[blank, "xpts"] == 0).all()


def test_model_source_scales_by_availability(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path, xpts_source="model")
    run_predict(settings)
    out = _predictions(settings).set_index("id")

    assert out["xpts"].notna().all()
    assert (out["xpts"] == out["xpts_model"]).all()
    # The model knows nothing about injury news, so availability is applied on top.
    assert out.loc[6, "availability"] == 0.5


def test_ep_next_feature_is_off_by_default_and_switchable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert Settings().use_ep_next is False
    settings = _ingested(tmp_path, xpts_source="model", use_ep_next=True)

    run_predict(settings)
    out = _predictions(settings)
    assert len(out) == 8 and out["xpts"].notna().all()


def test_dry_run_never_resolves_to_the_bucket() -> None:
    settings = Settings(data_bucket="real-bucket", dry_run=True)
    assert "real-bucket" not in settings.data_root


def test_model_source_falls_back_to_ep_next_with_no_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path, xpts_source="model")
    # Pretend it is gameweek 1: no earlier live data exists.
    for name in ("players", "teams", "fixtures"):
        write_parquet(
            settings,
            read_parquet(settings, gw_path(SEASON, 3, name)),
            gw_path(SEASON, 1, name),
        )
    gameweek = run_predict(
        Settings(data_dir=str(tmp_path), dry_run=True, gameweek=1, xpts_source="model")
    )
    out = _predictions(settings, gameweek)

    # No fixtures in gameweek 1 of the dry-run fixture list, so blanks score zero.
    assert (out["xpts"] == 0).all()


def test_model_source_trains_on_earlier_seasons(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path, xpts_source="model")
    # Copy the season's data as an earlier season: it must be picked up as training history.
    for gw in (1, 2, 3):
        for name in ("live", "players", "teams", "fixtures"):
            try:
                df = read_parquet(settings, gw_path(SEASON, gw, name))
            except FileNotFoundError:
                continue
            write_parquet(settings, df, gw_path("2024-25", gw, name))

    run_predict(settings)
    assert len(_predictions(settings)) == 8


def _two_season_features(synthetic: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for season in ("2024-25", "2025-26"):
        f = build_features(
            synthetic["live"], synthetic["players"], synthetic["teams"], synthetic["fixtures"],
            list(range(1, 9)), synthetic["ep"],
        )  # fmt: skip
        f.insert(0, "season", season)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_walk_forward_never_trains_on_the_test_gameweek_or_later(
    synthetic: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    feats = _two_season_features(synthetic)
    seen: list[tuple[pd.DataFrame, list[str] | None]] = []
    real_train = backtest.model.train

    def spy(x: pd.DataFrame, y: pd.Series, columns: list[str] | None = None):  # type: ignore[no-untyped-def]
        seen.append((x, columns))
        return real_train(x, y, columns)

    monkeypatch.setattr(backtest.model, "train", spy)
    preds = backtest.walk_forward(feats, ["2025-26"])

    assert sorted(set(preds["gameweek"])) == [3, 4, 5, 6, 7, 8]
    assert set(preds["season"]) == {"2025-26"}
    assert len(seen) == 12  # a plain model and an ep_next model for each of 6 gameweeks
    earlier_rows = ((feats["season"] == "2024-25") & (feats["fix_n"] >= 1)).sum()
    for g, (plain, xp) in zip(range(3, 9), zip(seen[0::2], seen[1::2], strict=True), strict=True):
        assert plain[1] == features.FEATURES and xp[1] == features.XP_FEATURES
        for train_x, _ in (plain, xp):
            rows = feats.loc[train_x.index]
            # The earlier season is always fully available; the test season only before g.
            assert (rows["season"] == "2024-25").sum() == earlier_rows
            assert (rows[rows["season"] == "2025-26"]["gameweek"] < g).all()


def test_model_using_a_good_ep_next_beats_the_model_without_it(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    preds = backtest.walk_forward(_two_season_features(synthetic), ["2025-26"])
    mae = lambda col: (preds[col] - preds["target"]).abs().mean()  # noqa: E731

    assert mae("model_xp") < mae("model")


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
            "model_xp": [2.0, 4.0, 1.0, 5.0],
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


def _audit_frame(leaky: bool) -> pd.DataFrame:
    """ep_next that either contains its own gameweek's points (leaky) or only the previous's."""
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(1, 301):
        pts = rng.poisson(3.0, 12).astype(float)
        for gw in range(1, 13):
            known = pts[gw - 1] if leaky else (pts[gw - 2] if gw > 1 else np.nan)
            rows.append(
                {"season": "2024-25", "id": pid, "gameweek": gw, "target": pts[gw - 1],
                 "min_l5": 90.0, "ep_next": 2.0 + 0.5 * known}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_leak_audit_flags_ep_next_that_contains_its_own_gameweek() -> None:
    clean = backtest.leak_audit(_audit_frame(leaky=False))
    leaky = backtest.leak_audit(_audit_frame(leaky=True))

    assert list(clean["status"]) == ["OK"]
    assert abs(clean["corr_with_points_in_g"].iloc[0]) < 0.1
    assert list(leaky["status"]) == ["LEAK WARNING"]
    assert leaky["corr_with_points_in_g"].iloc[0] > 0.3


def test_report_shows_the_leak_audit(synthetic: dict[str, pd.DataFrame]) -> None:
    preds = backtest.walk_forward(_two_season_features(synthetic), ["2025-26"])
    report = backtest.render_report(preds, backtest.leak_audit(_audit_frame(leaky=True)))

    assert "Leak audit" in report
    assert "LEAK WARNING" in report


def test_report_when_no_history() -> None:
    assert "Not enough" in backtest.render_report(pd.DataFrame())
