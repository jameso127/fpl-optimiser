from pathlib import Path

import pytest

from scripts.changed_components import IMAGE_CONTENTS, components_for, deployable, matrix

ALL = ["ingest", "notify", "optimise", "predict", "train"]


def test_a_change_in_one_component_rebuilds_only_what_contains_it() -> None:
    assert components_for(["optimise/model.py"], ALL) == ["optimise"]
    assert components_for(["notify/telegram.py"], ALL) == ["notify"]
    assert components_for(["train/main.py"], ALL) == ["train"]
    assert components_for(["predict/main.py"], ALL) == ["predict"]


def test_shared_ml_code_rebuilds_the_jobs_that_ship_it() -> None:
    assert components_for(["ml/features.py"], ALL) == ["predict", "train"]


def test_ingest_changes_also_rebuild_the_jobs_that_copy_it_for_dry_runs() -> None:
    assert components_for(["ingest/main.py"], ALL) == ["ingest", "predict", "train"]


def test_common_and_dependency_changes_rebuild_everything() -> None:
    assert components_for(["common/storage.py"], ALL) == ALL
    for dep in ("pyproject.toml", "uv.lock", ".python-version"):
        assert components_for([dep, "README.md"], ALL) == ALL


def test_changes_that_ship_in_no_image_rebuild_nothing() -> None:
    assert (
        components_for(["README.md", "docs/backtest.md", "tests/test_x.py", ".github/x.yml"], ALL)
        == []
    )
    assert components_for([], ALL) == []


def test_only_components_with_a_dockerfile_are_considered() -> None:
    assert components_for(["optimise/model.py", "ingest/main.py"], ["ingest"]) == ["ingest"]
    assert components_for(["common/config.py"], []) == []


def test_windows_style_paths_and_blank_lines_are_handled() -> None:
    assert components_for(["", "ml\\model.py", "  "], ALL) == ["predict", "train"]


def test_matrix_has_the_shape_github_actions_expects() -> None:
    assert matrix(["ingest", "train"]) == {
        "include": [{"component": "ingest"}, {"component": "train"}]
    }
    assert matrix([]) == {"include": []}


def test_deployable_finds_components_with_a_dockerfile(tmp_path: Path) -> None:
    for component in ("ingest", "train"):
        (tmp_path / component).mkdir()
        (tmp_path / component / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "predict").mkdir()  # no Dockerfile: not deployable yet

    assert deployable(tmp_path) == ["ingest", "train"]


def test_every_dockerfile_in_the_repo_matches_the_declared_image_contents() -> None:
    """If a Dockerfile copies a directory, a change to it must trigger that component."""
    root = Path(__file__).resolve().parents[1]
    for component in deployable(root):
        copied = {
            line.split()[1].rstrip("/") + "/"
            for line in (root / component / "Dockerfile").read_text().splitlines()
            if line.startswith("COPY ") and not line.startswith("COPY --from")
            and line.split()[1].rstrip("/") in {"common", "ingest", "ml", "train", "predict",
                                               "optimise", "notify"}
        }  # fmt: skip
        assert copied == set(IMAGE_CONTENTS[component]), component


@pytest.mark.parametrize("component", sorted(IMAGE_CONTENTS))
def test_each_component_always_includes_common(component: str) -> None:
    assert "common/" in IMAGE_CONTENTS[component]
