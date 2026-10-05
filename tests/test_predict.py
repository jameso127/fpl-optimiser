import numpy as np
import pandas as pd
import pytest

from common.config import Settings
from common.storage import gw_path, read_parquet, write_parquet
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from ml import backtest, features, model, registry
from ml.features import build_features
from predict.main import run as run_predict
from train.main import run as train_run

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


def test_default_source_is_the_model() -> None:
    assert Settings().xpts_source == "model"


def test_ep_next_source_uses_fpls_figure_as_is(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path, xpts_source="ep_next")
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
    settings = _ingested(tmp_path, xpts_source="ep_next")
    for gw in (1, 2):
        (tmp_path / "dry_run" / f"season={SEASON}" / f"gw={gw}" / "live.parquet").unlink()

    assert run_predict(settings) == 3  # nothing to train on, and it does not matter


def test_ep_next_source_zeroes_a_blank_gameweek_and_keeps_doubles_as_given(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    settings = _ingested(tmp_path, xpts_source="ep_next")
    fixtures = read_parquet(settings, gw_path(SEASON, 3, "fixtures"))
    # Team 1 doubles up; team 4 has no fixture.
    extra = fixtures[fixtures["event"] == 3].iloc[[0]].assign(id=99, team_h=1, team_a=2)
    fixtures = pd.concat([fixtures, extra])
    fixtures = fixtures[~((fixtures["event"] == 3) & (fixtures["team_h"] == 3))]
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


def test_dry_run_never_resolves_to_the_bucket() -> None:
    settings = Settings(data_bucket="real-bucket", dry_run=True)
    assert "real-bucket" not in settings.data_root


def test_model_source_serves_the_promoted_model_and_never_trains(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch
) -> None:
    settings = _ingested(tmp_path, xpts_source="model")
    card = train_run(settings)
    assert card.gate is not None and card.gate.promoted

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the predict job must not train")

    monkeypatch.setattr(model, "train_hurdle", refuse)
    monkeypatch.setattr(model, "train_v2", refuse)
    run_predict(settings)
    out = _predictions(settings).set_index("id")

    assert (out["model_version"] == card.version).all()
    assert out["xpts_model"].notna().all() and (out["xpts"] == out["xpts_model"]).all()


def test_model_source_scales_the_chance_of_playing_by_injury_news(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    settings = _ingested(tmp_path, xpts_source="model")
    train_run(settings)
    snap = gw_path(SEASON, 3, "players")
    players = read_parquet(settings, snap)

    def xpts_for_player_6(chance: float) -> float:
        write_parquet(settings, players.assign(chance_of_playing_next_round=chance), snap)
        run_predict(settings)
        return float(_predictions(settings).set_index("id")["xpts"].loc[6])

    full, half = xpts_for_player_6(100.0), xpts_for_player_6(50.0)
    assert full > 0
    assert half == pytest.approx(full * 0.5)


def test_model_source_falls_back_to_ep_next_when_no_model_is_promoted(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    settings = _ingested(tmp_path, xpts_source="model")
    run_predict(settings)
    out = _predictions(settings).set_index("id")

    assert (out["model_version"] == "ep_next-fallback").all()
    assert (out["xpts"] == out["ep_next"] * (out["fix_n"] >= 1)).all()


def test_serving_stops_if_the_features_no_longer_match_the_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _ingested(tmp_path, xpts_source="model")
    card = train_run(settings)
    card.features = [*card.features, "a_feature_the_code_no_longer_builds"]
    registry.update_card(settings, card)

    with pytest.raises(ValueError, match="Retrain"):
        run_predict(settings)


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
    seen: list[pd.DataFrame] = []
    real_v1, real_v2 = backtest.model.train, backtest.model.train_v2

    def spy_v1(x: pd.DataFrame, y: pd.Series, columns: list[str] | None = None):  # type: ignore[no-untyped-def]
        seen.append(x)
        return real_v1(x, y, columns)

    def spy_v2(x: pd.DataFrame, columns: list[str] | None = None):  # type: ignore[no-untyped-def]
        seen.append(x)
        return real_v2(x, columns)

    monkeypatch.setattr(backtest.model, "train", spy_v1)
    monkeypatch.setattr(backtest.model, "train_v2", spy_v2)
    preds = backtest.walk_forward(feats, ["2025-26"])

    assert sorted(set(preds["gameweek"])) == [3, 4, 5, 6, 7, 8]
    assert set(preds["season"]) == {"2025-26"}
    assert len(seen) == 12  # the first model and the current model, for each of 6 gameweeks
    earlier_rows = ((feats["season"] == "2024-25") & (feats["fix_n"] >= 1)).sum()
    for g, pair in zip(range(3, 9), zip(seen[0::2], seen[1::2], strict=True), strict=True):
        for train_x in pair:
            rows = feats.loc[train_x.index]
            # The earlier season is always fully available; the test season only before g.
            assert (rows["season"] == "2024-25").sum() == earlier_rows
            assert (rows[rows["season"] == "2025-26"]["gameweek"] < g).all()


def test_backtest_scores_the_current_model_the_first_model_and_baselines(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    preds = backtest.walk_forward(_two_season_features(synthetic), ["2025-26"])

    assert {"model", "model_v1", "baseline_l5", "baseline_season"} <= set(preds.columns)
    assert preds[["model", "model_v1"]].notna().all().all()


def test_ablation_removes_one_feature_group_at_a_time(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    abl = backtest.ablation(_two_season_features(synthetic), "2025-26", stride=3)

    assert list(abl["variant"]) == [
        "full model",
        *[f"without {g}" for g in features.GROUPS],
        "first model's features only",
    ]
    assert abl.drop(columns="variant").notna().all().all()
    report = backtest.render_report(
        backtest.walk_forward(_two_season_features(synthetic), ["2025-26"]),
        ablated=abl,
        ablated_season="2025-26",
    )
    assert "What each feature group is worth" in report and "without team_form" in report


def test_hurdle_expected_points_is_chance_of_playing_times_points_if_playing(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    # Players 1-10 never play; everyone else always does.
    never = synthetic["live"]["id"] <= 10
    synthetic["live"].loc[never, ["minutes", "total_points"]] = 0
    feats = _two_season_features(synthetic)
    train = features.usable(feats).dropna(subset=["target"])
    hurdle = model.train_v2(train)
    p_play, pts = model.predict_hurdle(hurdle, train)

    assert ((p_play >= 0) & (p_play <= 1)).all()
    assert np.allclose(model.expected_points(hurdle, train), p_play * pts)
    benched = train["id"].to_numpy() <= 10
    assert p_play[benched].mean() < 0.2 < 0.8 < p_play[~benched].mean()


def test_recency_weights_decay_by_season_age() -> None:
    seasons = pd.Series(["2022-23", "2023-24", "2024-25", "2024-25"])
    w = model.recency_weights(seasons, decay=0.5)

    assert list(w) == [0.25, 0.5, 1.0, 1.0]


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
            "model_v1": [2.0, 4.0, 1.0, 5.0],
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
