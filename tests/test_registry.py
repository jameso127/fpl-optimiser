import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from common.config import Settings
from common.storage import read_json
from ml import features, model, registry


def _settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=str(tmp_path), git_sha="abc1234def")


def _trained(labelled: pd.DataFrame, settings: Settings, version: str = "v1"):  # type: ignore[no-untyped-def]
    hurdle = model.train_v2(labelled)
    card = registry.build_card(settings, version, labelled, hurdle, metrics={"holdout": {"x": 1}})
    return hurdle, card


def test_version_names_sort_chronologically_and_carry_the_git_sha() -> None:
    early = registry.make_version("abcdef123456", dt.datetime(2026, 10, 5, 9, 0, 0, tzinfo=dt.UTC))
    late = registry.make_version("abcdef123456", dt.datetime(2026, 10, 12, 9, 0, 0, tzinfo=dt.UTC))

    assert early == "20261005T090000000Z-abcdef1"
    assert early < late


def test_saved_model_scores_identically_after_loading(labelled, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    hurdle, card = _trained(labelled, settings)
    registry.save(settings, hurdle, card)
    loaded, loaded_card = registry.load(settings, card.name, card.version)

    np.testing.assert_allclose(
        model.expected_points(hurdle, labelled), model.expected_points(loaded, labelled)
    )
    assert loaded.columns == hurdle.columns
    assert loaded_card == card


def test_versions_are_immutable(labelled, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    hurdle, card = _trained(labelled, settings)
    registry.save(settings, hurdle, card)

    with pytest.raises(FileExistsError):
        registry.save(settings, hurdle, card)


def test_only_a_promoted_version_is_served(labelled, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    hurdle, card = _trained(labelled, settings, "v1")
    registry.save(settings, hurdle, card)

    assert registry.latest_version(settings, card.name) is None
    with pytest.raises(FileNotFoundError):
        registry.load(settings, card.name)  # saved but not promoted: nothing to serve

    registry.promote(settings, card.name, "v1")
    assert registry.latest_version(settings, card.name) == "v1"
    assert registry.load(settings, card.name)[1].version == "v1"
    # An older version can still be loaded explicitly (rollback, audit).
    other, other_card = _trained(labelled, settings, "v2")
    registry.save(settings, other, other_card)
    assert registry.latest_version(settings, card.name) == "v1"
    assert registry.load(settings, card.name, "v2")[1].version == "v2"


def test_model_card_records_what_is_needed_to_reproduce_and_audit(  # type: ignore[no-untyped-def]
    labelled, tmp_path
) -> None:
    settings = _settings(tmp_path)
    hurdle, card = _trained(labelled, settings)
    registry.save(settings, hurdle, card)
    raw = read_json(settings, f"models/{card.name}/{card.version}/model_card.json")

    assert raw["git_sha"] == "abc1234def"
    assert raw["features"] == features.FEATURES_V2
    assert set(raw["feature_dtypes"]) == set(features.FEATURES_V2)
    assert raw["train_rows"] == len(labelled) and raw["played_rows"] > 0
    assert raw["train_seasons"] == ["2024-25", "2025-26"]
    assert raw["trained_through_season"] == "2025-26" and raw["trained_through_gameweek"] == 8
    assert raw["hyperparameters"]["seed"] == 0 and raw["hyperparameters"]["deterministic"] is True
    assert {"python", "lightgbm", "numpy", "pandas"} <= set(raw["libraries"])
    assert raw["metrics"] == {"holdout": {"x": 1}}
    json.dumps(raw)  # serialisable


def test_data_hash_ignores_row_order_but_notices_any_change(labelled) -> None:  # type: ignore[no-untyped-def]
    cols = features.FEATURES_V2
    base = registry.hash_training_data(labelled, cols)

    assert registry.hash_training_data(labelled.sample(frac=1, random_state=1), cols) == base
    edited = labelled.copy()
    edited.iloc[0, edited.columns.get_loc("target")] += 1
    assert registry.hash_training_data(edited, cols) != base


def test_training_is_reproducible(labelled) -> None:  # type: ignore[no-untyped-def]
    a = model.expected_points(model.train_v2(labelled), labelled)
    b = model.expected_points(model.train_v2(labelled), labelled)

    np.testing.assert_array_equal(a, b)


def test_schema_check_accepts_matching_and_rejects_changed_features(  # type: ignore[no-untyped-def]
    labelled, tmp_path
) -> None:
    settings = _settings(tmp_path)
    _, card = _trained(labelled, settings)
    registry.check_schema(card, labelled)  # unchanged: fine

    with pytest.raises(ValueError, match="no longer produces"):
        registry.check_schema(card, labelled.drop(columns=["pts_l3"]))
    broken = labelled.assign(pts_l3=labelled["pts_l3"].astype(str))
    with pytest.raises(ValueError, match="types changed"):
        registry.check_schema(card, broken)
