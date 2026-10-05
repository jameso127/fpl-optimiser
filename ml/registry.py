"""Model registry: versioned artifacts, a model card for each, and a `latest` pointer.

Layout under the data root (a GCS bucket or a local directory):

    models/<name>/<version>/plays.txt                 LightGBM model text, chance of playing
    models/<name>/<version>/points_if_plays.txt       LightGBM model text, points if playing
    models/<name>/<version>/model_card.json           everything needed to audit or reproduce it
    models/<name>/<version>/holdout_predictions.parquet   the predictions the gate judged
    models/<name>/latest.json                         which version serves

A version is written first and only becomes `latest` if it passes the promotion gate, so the
serving job always loads a model that was evaluated. Versions are never overwritten.
"""

import datetime as dt
import hashlib
import importlib.metadata
import platform
from typing import Any

import lightgbm as lgb
import pandas as pd
from pydantic import BaseModel, Field

from common.config import Settings
from common.storage import (
    list_subdirs,
    read_bytes,
    read_json,
    write_bytes,
    write_json,
    write_parquet,
)
from ml import model
from ml.model import Hurdle


class GateOutcome(BaseModel):
    promoted: bool
    reasons: list[str]


class ModelCard(BaseModel):
    """The audit trail for one trained model."""

    name: str
    version: str
    created_at: str
    git_sha: str
    # Data the model was fitted on.
    trained_through_season: str
    trained_through_gameweek: int
    train_seasons: list[str]
    train_rows: int
    played_rows: int
    data_hash: str = Field(description="sha256 of the training rows, in a stable order")
    # The model itself.
    features: list[str]
    feature_dtypes: dict[str, str]
    hyperparameters: dict[str, Any]
    rounds: int
    season_decay: float
    libraries: dict[str, str]
    # Evidence.
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate: GateOutcome | None = None


def make_version(git_sha: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.UTC)
    return f"{now:%Y%m%dT%H%M%S}{now.microsecond // 1000:03d}Z-{git_sha[:7]}"


def library_versions() -> dict[str, str]:
    names = ["lightgbm", "numpy", "pandas", "pyarrow"]
    return {
        "python": platform.python_version(),
        **{n: importlib.metadata.version(n) for n in names},
    }


def hash_training_data(train: pd.DataFrame, columns: list[str]) -> str:
    """Fingerprint of exactly the rows and columns the model saw, independent of row order."""
    cols = ["season", "id", "gameweek", "target", "target_minutes", *columns]
    ordered = train[cols].sort_values(["season", "gameweek", "id"]).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def build_card(
    settings: Settings,
    version: str,
    train: pd.DataFrame,
    hurdle: Hurdle,
    metrics: dict[str, Any],
) -> ModelCard:
    last = train.sort_values(["season", "gameweek"]).iloc[-1]
    params = {**model.HURDLE_PARAMS}
    return ModelCard(
        name=settings.model_name,
        version=version,
        created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        git_sha=settings.git_sha or "unknown",
        trained_through_season=str(last["season"]),
        trained_through_gameweek=int(last["gameweek"]),
        train_seasons=sorted(map(str, train["season"].unique())),
        train_rows=len(train),
        played_rows=int((train["target_minutes"] > 0).sum()),
        data_hash=hash_training_data(train, hurdle.columns),
        features=list(hurdle.columns),
        feature_dtypes={c: str(train[c].dtype) for c in hurdle.columns},
        hyperparameters=params,
        rounds=model.HURDLE_ROUNDS,
        season_decay=model.SEASON_DECAY,
        libraries=library_versions(),
        metrics=metrics,
    )


def _dir(name: str, version: str) -> str:
    return f"models/{name}/{version}"


def save(
    settings: Settings,
    hurdle: Hurdle,
    card: ModelCard,
    holdout_predictions: pd.DataFrame | None = None,
) -> str:
    """Write a new version. Refuses to overwrite one that exists (versions are immutable)."""
    d = _dir(card.name, card.version)
    if card.version in list_subdirs(settings, f"models/{card.name}"):
        raise FileExistsError(f"model version already exists: {d}")
    write_bytes(settings, hurdle.plays.model_to_string().encode(), f"{d}/plays.txt")
    write_bytes(
        settings, hurdle.points_if_plays.model_to_string().encode(), f"{d}/points_if_plays.txt"
    )
    if holdout_predictions is not None:
        write_parquet(settings, holdout_predictions, f"{d}/holdout_predictions.parquet")
    return write_json(settings, card.model_dump(), f"{d}/model_card.json")


def update_card(settings: Settings, card: ModelCard) -> None:
    """Rewrite a version's card (used once, to record the gate's decision)."""
    write_json(settings, card.model_dump(), f"{_dir(card.name, card.version)}/model_card.json")


def promote(settings: Settings, name: str, version: str) -> None:
    write_json(
        settings,
        {"version": version, "promoted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")},
        f"models/{name}/latest.json",
    )


def latest_version(settings: Settings, name: str) -> str | None:
    try:
        return str(read_json(settings, f"models/{name}/latest.json")["version"])
    except FileNotFoundError:
        return None


def load_card(settings: Settings, name: str, version: str) -> ModelCard:
    return ModelCard.model_validate(read_json(settings, f"{_dir(name, version)}/model_card.json"))


def load(settings: Settings, name: str, version: str | None = None) -> tuple[Hurdle, ModelCard]:
    """Load a version, or the promoted one. Raises FileNotFoundError if there is none."""
    version = version or latest_version(settings, name)
    if version is None:
        raise FileNotFoundError(f"no promoted model named {name!r}")
    card = load_card(settings, name, version)
    d = _dir(name, version)
    plays = lgb.Booster(model_str=read_bytes(settings, f"{d}/plays.txt").decode())
    points = lgb.Booster(model_str=read_bytes(settings, f"{d}/points_if_plays.txt").decode())
    return Hurdle(plays, points, list(card.features)), card


def check_schema(card: ModelCard, feature_frame: pd.DataFrame) -> None:
    """Fail loudly if the features the code now produces differ from the ones trained on.

    Silent drift between training and serving is how models quietly go wrong, so a mismatch
    stops the job instead of scoring with misaligned columns.
    """
    missing = [c for c in card.features if c not in feature_frame.columns]
    if missing:
        raise ValueError(
            f"model {card.version} expects features the code no longer produces: {missing}. "
            "Retrain (job `train`) before serving."
        )
    # Numeric kinds must agree (a float column silently becoming object would corrupt scores).
    bad = [
        c
        for c in card.features
        if pd.api.types.is_numeric_dtype(feature_frame[c]) != _is_numeric(card.feature_dtypes[c])
    ]
    if bad:
        raise ValueError(f"model {card.version}: feature types changed for {bad}. Retrain.")


def _is_numeric(dtype_name: str) -> bool:
    return pd.api.types.is_numeric_dtype(pd.Series(dtype=dtype_name))
