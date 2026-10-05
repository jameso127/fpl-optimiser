"""LightGBM expected-points model plus the availability adjustment applied after prediction."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.features import FEATURES, FEATURES_V2

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_data_in_leaf": 30,
    "lambda_l2": 5.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
ROUNDS = 150


def train(x: pd.DataFrame, y: pd.Series, columns: list[str] | None = None) -> lgb.Booster:
    """Fit on `columns` (default: form and fixture features, without FPL's ep_next)."""
    cols = columns or FEATURES
    return lgb.train(PARAMS, lgb.Dataset(x[cols], label=y), num_boost_round=ROUNDS)


def predict(booster: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(booster.predict(x[booster.feature_name()]), dtype=float)


HURDLE_PARAMS = {
    "learning_rate": 0.04,
    "num_leaves": 15,
    "min_data_in_leaf": 60,
    "lambda_l2": 10.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
HURDLE_ROUNDS = 200
SEASON_DECAY = 0.7  # each season back counts 0.7x as much as the newest in training


@dataclass
class Hurdle:
    """Chance a player plays at all, and the points they score if they do."""

    plays: lgb.Booster
    points_if_plays: lgb.Booster
    columns: list[str]


def train_hurdle(
    x: pd.DataFrame,
    columns: list[str],
    weight: pd.Series | None = None,
    params: dict[str, float | int | str | bool] | None = None,
    rounds: int = HURDLE_ROUNDS,
) -> Hurdle:
    """Fit on rows with `target` (points) and `target_minutes`.

    The two parts are fitted independently on real outcomes (no stacking), so there is no
    in-sample leakage from one stage into the other. `weight` (aligned to `x`) down-weights
    rows, e.g. older seasons.
    """
    base = {**HURDLE_PARAMS, **(params or {})}
    played = x["target_minutes"] > 0
    w_all = weight.loc[x.index] if weight is not None else None
    w_played = weight.loc[x.index[played]] if weight is not None else None
    plays = lgb.train(
        {**base, "objective": "binary"},
        lgb.Dataset(x[columns], label=played.astype(int), weight=w_all),
        num_boost_round=rounds,
    )
    points = lgb.train(
        {**base, "objective": "regression"},
        lgb.Dataset(x.loc[played, columns], label=x.loc[played, "target"], weight=w_played),
        num_boost_round=rounds,
    )
    return Hurdle(plays, points, columns)


def recency_weights(seasons: pd.Series, decay: float = SEASON_DECAY) -> pd.Series:
    """Weight 1 for the newest season in `seasons`, `decay` per season further back."""
    order = {s: i for i, s in enumerate(sorted(seasons.unique()))}
    newest = max(order.values())
    return seasons.map(lambda s: decay ** (newest - order[s])).astype(float)


def train_v2(train: pd.DataFrame, columns: list[str] | None = None) -> Hurdle:
    """The current model: hurdle on all features, recent seasons weighted more."""
    return train_hurdle(train, columns or FEATURES_V2, weight=recency_weights(train["season"]))


def expected_points(h: Hurdle, x: pd.DataFrame) -> np.ndarray:
    """P(plays) x E[points | plays]. Injury news is applied separately by the caller."""
    p_play, pts = predict_hurdle(h, x)
    return p_play * pts


def predict_hurdle(h: Hurdle, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(probability of playing, expected points given they play)."""
    p_play = np.asarray(h.plays.predict(x[h.columns]), dtype=float)
    pts = np.asarray(h.points_if_plays.predict(x[h.columns]), dtype=float)
    return p_play, pts


def availability(players: pd.DataFrame) -> pd.Series:
    """Probability (0-1) a player features next gameweek.

    Status `u` (unavailable) or `s` (suspended) means 0. Otherwise FPL's own
    chance_of_playing_next_round applies, and null means no news (fully available). Historical
    chance-of-playing is not stored, so this is applied after the model rather than learned.
    """
    chance = players["chance_of_playing_next_round"] / 100
    avail = chance.fillna(1.0)
    avail[players["status"].isin(["u", "s"])] = 0.0
    avail.index = players["id"]
    return avail.clip(0.0, 1.0)
