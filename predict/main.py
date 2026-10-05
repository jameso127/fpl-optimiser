"""Predict job (serving): stored data + the promoted model -> predictions.parquet.

Scores gameweek n with the model currently promoted in the registry. It never trains: the
`train` job does that on its own schedule. Needs only the current season's live data (plus
last season's rates), so it is quick and light.

Columns: id, gameweek, xpts (what the optimiser uses), ep_next, xpts_model, availability,
fix_n, model_version.

`XPTS_SOURCE` picks `xpts`:
- `model` (default): the promoted hurdle model (chance of playing x points if playing), scaled
  by FPL's injury news. If no model has been promoted yet, falls back to `ep_next` and says
  so in `model_version` and the logs instead of failing the pipeline.
- `ep_next`: FPL's own expected points from the players snapshot. FPL already folds
  chance-of-playing and double gameweeks into it (ep_next = form x chance/100, doubled in a
  double gameweek), so it is used as is; only a blank gameweek is forced to zero.

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
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from ml import features, model, registry
from ml.data import season_features

log = logging.getLogger(__name__)

EP_NEXT_VERSION = "ep_next"
FALLBACK_VERSION = "ep_next-fallback"


def _model_xpts(
    settings: Settings, season: str, gameweek: int, players: pd.DataFrame
) -> tuple[pd.Series, str] | None:
    """(expected points per player id, model version) from the promoted model, or None."""
    try:
        hurdle, card = registry.load(settings, settings.model_name)
    except FileNotFoundError:
        return None
    feats = season_features(
        settings, season, [gameweek], snapshot_gw=gameweek, live_before=gameweek
    )
    registry.check_schema(card, feats)
    raw = model.expected_points(hurdle, feats)
    # Injury news has no history to learn from, so it scales the chance of playing here.
    scale = feats["id"].map(model.availability(players))
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
    log.info(
        "predict start",
        extra={
            "season": season,
            "gameweek": gameweek,
            "xpts_source": settings.xpts_source,
            "dry_run": settings.dry_run,
        },
    )

    players = read_parquet(settings, gw_path(season, gameweek, "players"))
    teams = read_parquet(settings, gw_path(season, gameweek, "teams"))
    fixtures = read_parquet(settings, gw_path(season, gameweek, "fixtures"))
    per_team = features.fixture_features(fixtures, teams)
    fix_n = per_team[per_team["gameweek"] == gameweek].set_index("team")["fix_n"]

    out = pd.DataFrame({"id": players["id"], "gameweek": gameweek})
    out["ep_next"] = players["ep_next"].to_numpy()
    out["fix_n"] = players["team"].map(fix_n).fillna(0).astype(int).to_numpy()
    out["availability"] = out["id"].map(model.availability(players))
    out["xpts_model"] = np.nan
    # ep_next already reflects chance of playing and double gameweeks. A blank gameweek has
    # no fixture, so nothing to score.
    out["xpts"] = out["ep_next"].fillna(0.0) * (out["fix_n"] >= 1)

    version = EP_NEXT_VERSION
    if settings.xpts_source == "model":
        scored = _model_xpts(settings, season, gameweek, players)
        if scored is None:
            version = FALLBACK_VERSION
            log.warning(
                "no promoted model; using ep_next. Run the train job.",
                extra={"model_name": settings.model_name},
            )
        else:
            xpts_model, version = scored
            out["xpts_model"] = out["id"].map(xpts_model)
            out["xpts"] = out["xpts_model"]
    out["model_version"] = version

    uri = write_parquet(settings, out, gw_path(season, gameweek, "predictions"))
    log.info("predict done", extra={"uri": uri, "rows": len(out), "model_version": version})
    return gameweek


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.dry_run:
        run_ingest(settings, DryRunSource())
    run(settings)


if __name__ == "__main__":
    main()
