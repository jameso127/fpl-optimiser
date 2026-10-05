"""Parquet I/O against GCS (gs://bucket/...) or a local directory, chosen by Settings.

Layout: season=<yyyy-yy>/gw=<n>/<name>.parquet
"""

import json
import re
from typing import Any

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


def write_bytes(settings: Settings, data: bytes, rel_path: str) -> str:
    uri = _uri(settings, rel_path)
    fs, path = fsspec.core.url_to_fs(uri)
    fs.makedirs(fs._parent(path), exist_ok=True)
    with fs.open(path, "wb") as f:
        f.write(data)
    return uri


def read_bytes(settings: Settings, rel_path: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(_uri(settings, rel_path))
    with fs.open(path, "rb") as f:
        data: bytes = f.read()
    return data


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(settings: Settings, obj: Any, rel_path: str) -> str:
    body = json.dumps(obj, indent=2, sort_keys=True, default=_json_default)
    return write_bytes(settings, body.encode(), rel_path)


def read_json(settings: Settings, rel_path: str) -> Any:
    return json.loads(read_bytes(settings, rel_path))


def list_subdirs(settings: Settings, rel_dir: str) -> list[str]:
    """Names of the immediate subdirectories of `rel_dir` (empty if it does not exist)."""
    fs, path = fsspec.core.url_to_fs(_uri(settings, rel_dir.rstrip("/")))
    if not fs.exists(path):
        return []
    return sorted(
        p.rstrip("/").rsplit("/", 1)[-1] for p in fs.ls(path, detail=False) if fs.isdir(p)
    )


def exists(settings: Settings, rel_path: str) -> bool:
    fs, path = fsspec.core.url_to_fs(_uri(settings, rel_path))
    return bool(fs.exists(path))
