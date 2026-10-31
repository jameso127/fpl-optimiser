"""Predict job (serving): stored data + the promoted model -> predictions.parquet.

Scores the next gameweek n, and each of the HORIZON gameweeks from it, with the model currently
promoted in the registry. Later gameweeks use form as it stands before n's deadline with each
gameweek's own fixture (see features.features_ahead), and injury news fades by a simple rule
(model.availability_ahead); the optimiser weighs them less because they are less certain.

It never trains: the `train` job does that on its own schedule. Needs only the current season's
live data (plus last season's rates), so it is quick and light.

The expected points always come from the model (chance of playing x points if playing, scaled
by FPL's injury news). If no model has been promoted there is nothing to serve, so the job
fails with a clear message; run the `train` job first.

One row per player per gameweek. Columns: id, gameweek, horizon (1 = the next gameweek),
xpts (what the optimiser uses), ep_next (FPL's own figure for the next gameweek only, kept as a
benchmark for monitoring, never used for xpts), availability, fix_n, model_version. Anything
judging the predictions against results (monitoring) uses only the `horizon == 1` rows.

Re-running overwrites.
"""

import logging

import pandas as pd

from common.config import Settings, get_settings
from common.logging import configure_logging
from common.storage import gw_path, latest_gameweek, read_parquet, write_parquet
from ml import features, model, registry
from ml.data import season_features

log = logging.getLogger(__name__)


class NoModelError(RuntimeError):
    """Raised when there is no promoted model to serve."""


def horizon_gameweeks(fixtures: pd.DataFrame, gameweek: int, horizon: int) -> list[int]:
    """The gameweeks to predict: `horizon` of them from `gameweek`, stopping at the season's end."""
    last = int(fixtures["event"].dropna().max()) if fixtures["event"].notna().any() else gameweek
    return list(range(gameweek, min(gameweek + horizon - 1, last) + 1)) or [gameweek]


def _score(
    settings: Settings,
    season: str,
    gameweek: int,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """(id, gameweek, horizon, xpts, availability, fix_n per player and gameweek, model version)."""
    try:
        hurdle, card = registry.load(settings, settings.model_name)
    except FileNotFoundError as exc:
        raise NoModelError(
            f"no promoted model named {settings.model_name!r}: run the train job first "
            "(it registers a model and promotes it if it passes the gate)"
        ) from exc
    now = season_features(settings, season, [gameweek], snapshot_gw=gameweek, live_before=gameweek)
    gameweeks = horizon_gameweeks(fixtures, gameweek, settings.horizon)
    feats = features.features_ahead(now, players, teams, fixtures, gameweeks)
    registry.check_schema(card, feats)
    raw = model.expected_points(hurdle, feats)

    out = feats[["id", "gameweek", "fix_n"]].copy()
    out["horizon"] = out["gameweek"] - gameweek + 1
    # Injury news has no history to learn from, so it scales the chance of playing here.
    avail = model.availability_ahead(players, len(gameweeks))
    out = out.merge(avail, on=["id", "horizon"], how="left")
    out["availability"] = out["availability"].fillna(1.0)
    # A blank gameweek (no fixture) means no points, whatever the model says.
    out["xpts"] = raw * out["availability"].to_numpy() * (out["fix_n"] >= 1).to_numpy()
    out["fix_n"] = out["fix_n"].astype(int)
    log.info(
        "scored",
        extra={"model_version": card.version, "rows": len(out), "gameweeks": gameweeks},
    )
    return out, card.version


def run(settings: Settings) -> int:
    season, gameweek = latest_gameweek(settings, "players", "run ingest first")
    log.info("predict start", extra={"season": season, "gameweek": gameweek})

    players = read_parquet(settings, gw_path(season, gameweek, "players"))
    teams = read_parquet(settings, gw_path(season, gameweek, "teams"))
    fixtures = read_parquet(settings, gw_path(season, gameweek, "fixtures"))

    scored, version = _score(settings, season, gameweek, players, teams, fixtures)
    ep_next = players.set_index("id")["ep_next"]
    out = scored[["id", "gameweek", "horizon", "xpts"]].copy()
    out["ep_next"] = out["id"].map(ep_next).where(out["horizon"] == 1)
    out["availability"] = scored["availability"]
    out["fix_n"] = scored["fix_n"]
    out["model_version"] = version
    out = out.sort_values(["horizon", "id"]).reset_index(drop=True)

    uri = write_parquet(settings, out, gw_path(season, gameweek, "predictions"))
    log.info("predict done", extra={"uri": uri, "rows": len(out), "model_version": version})
    return gameweek


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    run(settings)


if __name__ == "__main__":
    main()
