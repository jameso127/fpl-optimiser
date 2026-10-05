"""Predict job: stored data -> season=<s>/gw=<n>/predictions.parquet.

Columns: id, gameweek, xpts (what the optimiser uses), ep_next, xpts_model, availability, fix_n.

`XPTS_SOURCE` picks `xpts`:
- `model` (default): a hurdle model (chance of playing x points if playing, LightGBM) trained
  on every earlier season plus this season's finished gameweeks, then scaled by FPL's injury
  news. See the README and docs/backtest.md for how it compares with `ep_next`.
- `ep_next`: FPL's own expected points from the players snapshot. Simple, needs no history or
  model. FPL already folds chance-of-playing and double gameweeks into it (ep_next = form x
  chance/100, doubled in a double gameweek), so it is used as is; only a blank gameweek is
  forced to zero.

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
from predict import features, model
from predict.data import season_features

log = logging.getLogger(__name__)


def _model_xpts(settings: Settings, season: str, seasons: list[str], gameweek: int) -> pd.Series:
    """Model expected points per player id, availability-adjusted, 0 for a blank gameweek."""
    players = read_parquet(settings, gw_path(season, gameweek, "players"))
    current = season_features(
        settings, season, list(range(1, gameweek + 1)), snapshot_gw=gameweek, live_before=gameweek
    )
    past = [
        season_features(settings, s)
        for s in seasons
        if s < season and available_gameweeks(settings, s, "live")
    ]
    history = pd.concat([*past, current[current["gameweek"] < gameweek]], ignore_index=True)
    train_rows = features.usable(history).dropna(subset=["target"])
    to_score = current[current["gameweek"] == gameweek].reset_index(drop=True)

    if train_rows.empty:
        log.warning("no training data; falling back to ep_next")
        ep = players.set_index("id")["ep_next"]
        raw = to_score["id"].map(ep).fillna(0.0).to_numpy()
        scale: float | pd.Series = 1.0  # ep_next already includes availability
    else:
        raw = model.expected_points(model.train_v2(train_rows), to_score)
        # Injury news has no history to learn from, so it scales the chance of playing here.
        scale = to_score["id"].map(model.availability(players))
        log.info(
            "trained",
            extra={"train_rows": len(train_rows), "train_seasons": len(past), "season": season},
        )
    xpts = raw * scale * (to_score["fix_n"] >= 1)
    return pd.Series(np.asarray(xpts, dtype=float), index=to_score["id"].to_numpy())


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
    if settings.xpts_source == "model":
        out["xpts_model"] = out["id"].map(_model_xpts(settings, season, seasons, gameweek))
        out["xpts"] = out["xpts_model"]
    else:
        # ep_next already reflects chance of playing and double gameweeks. A blank gameweek
        # has no fixture, so nothing to score.
        out["xpts"] = out["ep_next"].fillna(0.0) * (out["fix_n"] >= 1)

    uri = write_parquet(settings, out, gw_path(season, gameweek, "predictions"))
    log.info("predict done", extra={"uri": uri, "rows": len(out)})
    return gameweek


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.dry_run:
        run_ingest(settings, DryRunSource())
    run(settings)


if __name__ == "__main__":
    main()
