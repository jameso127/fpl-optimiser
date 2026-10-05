"""Predict job (serving): stored data + the promoted model -> predictions.parquet.

Scores gameweek n with the model currently promoted in the registry. It never trains: the
`train` job does that on its own schedule. Needs only the current season's live data (plus
last season's rates), so it is quick and light.

The expected points always come from the model (chance of playing x points if playing, scaled
by FPL's injury news). If no model has been promoted there is nothing to serve, so the job
fails with a clear message; run the `train` job first.

Columns: id, gameweek, xpts (what the optimiser uses), ep_next (FPL's own figure, kept only
as a benchmark for monitoring, never used for xpts), availability, fix_n, model_version.

Re-running overwrites.
"""

import logging

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.logging import configure_logging
from common.storage import (
    available_gameweeks,
    available_seasons,
    gw_path,
    read_parquet,
    write_parquet,
)
from ml import features, model, registry
from ml.data import season_features

log = logging.getLogger(__name__)


class NoModelError(RuntimeError):
    """Raised when there is no promoted model to serve."""


def _score(
    settings: Settings, season: str, gameweek: int, players: pd.DataFrame
) -> tuple[pd.Series, str]:
    """(expected points per player id, model version) from the promoted model."""
    try:
        hurdle, card = registry.load(settings, settings.model_name)
    except FileNotFoundError as exc:
        raise NoModelError(
            f"no promoted model named {settings.model_name!r}: run the train job first "
            "(it registers a model and promotes it if it passes the gate)"
        ) from exc
    feats = season_features(
        settings, season, [gameweek], snapshot_gw=gameweek, live_before=gameweek
    )
    registry.check_schema(card, feats)
    raw = model.expected_points(hurdle, feats)
    # Injury news has no history to learn from, so it scales the chance of playing here.
    scale = feats["id"].map(model.availability(players))
    # A blank gameweek (no fixture) means no points, whatever the model says.
    xpts = raw * scale * (feats["fix_n"] >= 1)
    log.info("scored", extra={"model_version": card.version, "rows": len(feats)})
    return pd.Series(np.asarray(xpts, dtype=float), index=feats["id"].to_numpy()), card.version


def run(settings: Settings) -> int:
    seasons = available_seasons(settings)
    if not seasons:
        raise RuntimeError("no data found; run ingest first")
    season = settings.season or seasons[-1]
    snapshots = available_gameweeks(settings, season, "players")
    if not snapshots:
        raise RuntimeError(f"no players snapshot for season {season}; run ingest first")
    gameweek = settings.gameweek or snapshots[-1]
    log.info("predict start", extra={"season": season, "gameweek": gameweek})

    players = read_parquet(settings, gw_path(season, gameweek, "players"))
    teams = read_parquet(settings, gw_path(season, gameweek, "teams"))
    fixtures = read_parquet(settings, gw_path(season, gameweek, "fixtures"))
    per_team = features.fixture_features(fixtures, teams)
    fix_n = per_team[per_team["gameweek"] == gameweek].set_index("team")["fix_n"]

    xpts, version = _score(settings, season, gameweek, players)
    out = pd.DataFrame({"id": players["id"], "gameweek": gameweek})
    out["xpts"] = out["id"].map(xpts)
    out["ep_next"] = players["ep_next"].to_numpy()
    out["availability"] = out["id"].map(model.availability(players))
    out["fix_n"] = players["team"].map(fix_n).fillna(0).astype(int).to_numpy()
    out["model_version"] = version

    uri = write_parquet(settings, out, gw_path(season, gameweek, "predictions"))
    log.info("predict done", extra={"uri": uri, "rows": len(out), "model_version": version})
    return gameweek


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    run(settings)


if __name__ == "__main__":
    main()
