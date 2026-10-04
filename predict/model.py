"""LightGBM expected-points model plus the availability adjustment applied after prediction."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from predict.features import FEATURES

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
