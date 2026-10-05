"""Decide which deployable components a change touches, for CI/CD path filtering.

Each component is a Cloud Run job (or service) built from `<component>/Dockerfile`. A
component is rebuilt when a file it copies into its image changes. Changes to `common/`
(and to the dependency files) rebuild everything.

    git diff --name-only <before> <after> | python scripts/changed_components.py
    python scripts/changed_components.py --all

Prints a GitHub Actions matrix as JSON: {"include": [{"component": "ingest"}, ...]}.
"""

import argparse
import json
import sys
from pathlib import Path

# What each component's image contains (see the COPY lines in its Dockerfile).
IMAGE_CONTENTS: dict[str, tuple[str, ...]] = {
    "ingest": ("common/", "ingest/"),
    "train": ("common/", "ingest/", "ml/", "train/"),
    "predict": ("common/", "ingest/", "ml/", "predict/"),
    "optimise": ("common/", "optimise/"),
    "notify": ("common/", "notify/"),
}

# Files that change every image.
EVERYTHING = ("pyproject.toml", "uv.lock", ".python-version")


def deployable(root: Path) -> list[str]:
    """Components that have a Dockerfile right now (others are not built yet)."""
    return sorted(c for c in IMAGE_CONTENTS if (root / c / "Dockerfile").is_file())


def components_for(changed: list[str], available: list[str]) -> list[str]:
    """The available components whose image contents include any of the changed files."""
    paths = [p.strip().replace("\\", "/") for p in changed if p.strip()]
    if any(p in EVERYTHING for p in paths):
        return sorted(available)
    return sorted(
        c
        for c in available
        if any(p.startswith(prefix) for p in paths for prefix in IMAGE_CONTENTS[c])
    )


def matrix(components: list[str]) -> dict[str, list[dict[str, str]]]:
    return {"include": [{"component": c} for c in components]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="every deployable component")
    parser.add_argument("--root", default=".", help="repo root (to find Dockerfiles)")
    args = parser.parse_args()

    available = deployable(Path(args.root))
    chosen = available if args.all else components_for(sys.stdin.read().splitlines(), available)
    print(json.dumps(matrix(chosen)))


if __name__ == "__main__":
    main()
