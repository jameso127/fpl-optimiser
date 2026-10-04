"""Predict job: stored data -> season=<s>/gw=<n>/predictions.parquet.

Trains LightGBM on every earlier season in storage plus the current season's finished
gameweeks, and scores gameweek n. If there is no training data at all, falls back to FPL's
`ep_next`. Re-running overwrites.
"""

import logging

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
        extra={"season": season, "gameweek": gameweek, "dry_run": settings.dry_run},
    )

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
    else:
        booster = model.train(train_rows, train_rows["target"])
        raw = model.predict(booster, to_score)
        log.info(
            "trained",
            extra={"train_rows": len(train_rows), "train_seasons": len(past), "season": season},
        )

    out = pd.DataFrame({"id": to_score["id"], "gameweek": gameweek, "xpts_raw": raw})
    out["availability"] = out["id"].map(model.availability(players))
    # No fixture this gameweek (blank) means no points, whatever the model says.
    out["xpts"] = out["xpts_raw"] * out["availability"] * (to_score["fix_n"] >= 1)
    out["ep_next"] = out["id"].map(players.set_index("id")["ep_next"])
    out["fix_n"] = to_score["fix_n"].astype(int)

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
