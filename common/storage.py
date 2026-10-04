"""Parquet I/O against GCS (gs://bucket/...) or a local directory, chosen by Settings.

Layout: season=<yyyy-yy>/gw=<n>/<name>.parquet
"""

import re

import fsspec
import pandas as pd

from common.config import Settings


def gw_path(season: str, gameweek: int, name: str) -> str:
    """Relative path of a gameweek artefact, e.g. season=2026-27/gw=12/players.parquet."""
    return f"season={season}/gw={gameweek}/{name}.parquet"


def _uri(settings: Settings, rel_path: str) -> str:
    return f"{settings.data_root.rstrip('/')}/{rel_path}"


def write_parquet(settings: Settings, df: pd.DataFrame, rel_path: str) -> str:
    """Write (overwrite) a Parquet file. Overwriting keeps jobs idempotent."""
    uri = _uri(settings, rel_path)
    fs, path = fsspec.core.url_to_fs(uri)
    fs.makedirs(fs._parent(path), exist_ok=True)
    with fs.open(path, "wb") as f:
        df.to_parquet(f, index=False)
    return uri


def read_parquet(settings: Settings, rel_path: str) -> pd.DataFrame:
    fs, path = fsspec.core.url_to_fs(_uri(settings, rel_path))
    with fs.open(path, "rb") as f:
        return pd.read_parquet(f)


def available_seasons(settings: Settings) -> list[str]:
    """Sorted seasons that have any data (labels sort chronologically)."""
    fs, root = fsspec.core.url_to_fs(settings.data_root.rstrip("/"))
    found = fs.glob(f"{root}/season=*/gw=*/*.parquet")
    return sorted({m.group(1) for p in found if (m := re.search(r"season=([\d-]+)", p))})


def available_gameweeks(settings: Settings, season: str, name: str) -> list[int]:
    """Sorted gameweeks of a season that have a `<name>.parquet` artefact."""
    fs, root = fsspec.core.url_to_fs(settings.data_root.rstrip("/"))
    found = fs.glob(f"{root}/season={season}/gw=*/{name}.parquet")
    return sorted(int(m.group(1)) for p in found if (m := re.search(r"gw=(\d+)", p)))


def exists(settings: Settings, rel_path: str) -> bool:
    fs, path = fsspec.core.url_to_fs(_uri(settings, rel_path))
    return bool(fs.exists(path))
