from pathlib import Path

import pytest

from common.config import Settings
from common.storage import list_subdirs
from ingest.main import run as run_ingest
from ml import gate, registry
from ml.registry import GateOutcome
from tests.fakes import FakeFplSource
from train.main import run as train_run


def _ingested(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=str(tmp_path), git_sha="deadbeefcafe")
    run_ingest(settings, FakeFplSource())
    return settings


def test_a_model_that_passes_the_gate_is_registered_and_promoted(
    tmp_path: Path, gate_passes: None
) -> None:
    settings = _ingested(tmp_path)
    card = train_run(settings)

    assert registry.latest_version(settings, card.name) == card.version
    assert card.version.endswith("-deadbee")
    assert card.gate is not None and card.gate.promoted
    assert card.train_rows > 0 and card.git_sha == "deadbeefcafe"
    # The artifact on disk reloads to a usable model.
    hurdle, loaded = registry.load(settings, card.name)
    assert loaded.version == card.version and hurdle.columns == card.features


def test_a_model_without_enough_evidence_is_saved_but_not_served(tmp_path: Path) -> None:
    settings = _ingested(tmp_path)  # two finished gameweeks: nothing to hold out
    card = train_run(settings)

    assert card.gate is not None and not card.gate.promoted
    assert "holdout gameweeks" in card.gate.reasons[0]
    assert registry.latest_version(settings, card.name) is None
    assert registry.load_card(settings, card.name, card.version).version == card.version


def test_each_training_run_creates_a_new_immutable_version(tmp_path: Path) -> None:
    settings = _ingested(tmp_path)
    first = train_run(settings)
    second = train_run(settings)

    assert first.version != second.version
    assert first.data_hash == second.data_hash  # same data in, same fingerprint out
    assert list_subdirs(settings, f"models/{first.name}") == sorted([first.version, second.version])


def test_a_model_that_fails_the_gate_is_saved_but_not_served(
    tmp_path: Path, gate_passes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _ingested(tmp_path)
    champion = train_run(settings)

    failing = GateOutcome(promoted=False, reasons=["FAIL: forced for the test"])
    monkeypatch.setattr(gate, "evaluate_gate", lambda *a, **k: failing)
    challenger = train_run(settings)

    assert challenger.gate is not None and not challenger.gate.promoted
    assert registry.latest_version(settings, champion.name) == champion.version  # unchanged
    # ...but the rejected version is kept, with the reasons, for inspection.
    kept = registry.load_card(settings, champion.name, challenger.version)
    assert kept.gate is not None and kept.gate.reasons == ["FAIL: forced for the test"]


@pytest.mark.parametrize("n", [1, 2])
def test_holdout_never_includes_gameweeks_before_the_first_testable_one(n: int) -> None:
    import pandas as pd

    from train.main import _holdout_gameweeks

    labelled = pd.DataFrame({"season": "2025-26", "gameweek": [1, 2, 3, 4, 5, 6]})
    season, hold = _holdout_gameweeks(labelled, n)

    assert season == "2025-26" and hold == set(range(7 - n, 7))
    assert min(_holdout_gameweeks(labelled, 99)[1]) == 3
