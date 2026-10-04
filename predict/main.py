"""Predict job: gw=<n>/ snapshot + live history -> gw=<n>/predictions.parquet.

Trains LightGBM on all finished gameweeks before n and scores gameweek n. If there is no
training data (gameweek 1), falls back to FPL's `ep_next`. Re-running overwrites.
"""

import logging

import pandas as pd

from common.config import Settings, get_settings
from common.logging import configure_logging
from common.storage import available_gameweeks, gw_path, read_parquet, write_parquet
from ingest.dry_run import DryRunSource
from ingest.main import run as run_ingest
from predict import features, model

log = logging.getLogger(__name__)


def load_live(settings: Settings, before: int) -> pd.DataFrame:
    frames = [
        read_parquet(settings, gw_path(gw, "live"))
        for gw in available_gameweeks(settings, "live")
        if gw < before
    ]
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "gameweek"])
    )


def run(settings: Settings) -> int:
    snapshots = available_gameweeks(settings, "players")
    if not snapshots:
        raise RuntimeError("no players snapshot found; run ingest first")
    gameweek = settings.gameweek or snapshots[-1]
    log.info("predict start", extra={"gameweek": gameweek, "dry_run": settings.dry_run})

    players = read_parquet(settings, gw_path(gameweek, "players"))
    teams = read_parquet(settings, gw_path(gameweek, "teams"))
    fixtures = read_parquet(settings, gw_path(gameweek, "fixtures"))
    live = load_live(settings, before=gameweek)

    feats = features.build_features(live, players, teams, fixtures, list(range(1, gameweek + 1)))
    train_rows = features.usable(feats[feats["gameweek"] < gameweek]).dropna(subset=["target"])
    to_score = feats[feats["gameweek"] == gameweek].reset_index(drop=True)

    if train_rows.empty:
        log.warning("no training data; falling back to ep_next")
        ep = players.set_index("id")["ep_next"]
        raw = to_score["id"].map(ep).fillna(0.0).to_numpy()
    else:
        booster = model.train(train_rows, train_rows["target"])
        raw = model.predict(booster, to_score)
        log.info("trained", extra={"train_rows": len(train_rows)})

    out = pd.DataFrame({"id": to_score["id"], "gameweek": gameweek, "xpts_raw": raw})
    out["availability"] = out["id"].map(model.availability(players))
    # No fixture this gameweek (blank) means no points, whatever the model says.
    out["xpts"] = out["xpts_raw"] * out["availability"] * (to_score["fix_n"] >= 1)
    out["ep_next"] = out["id"].map(players.set_index("id")["ep_next"])
    out["fix_n"] = to_score["fix_n"].astype(int)

    uri = write_parquet(settings, out, gw_path(gameweek, "predictions"))
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
