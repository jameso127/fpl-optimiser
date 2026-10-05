"""Load stored Parquet and turn each season into feature rows (tagged with a `season` column).

Player ids are only meaningful within a season, so form features never cross a season
boundary. The model sees seasons as separate pools of (player, gameweek) rows.
"""

import pandas as pd

from common.config import Settings
from common.storage import available_gameweeks, available_seasons, gw_path, read_parquet
from predict import features


def load_live(settings: Settings, season: str, before: int | None = None) -> pd.DataFrame:
    """Live stats for a season's finished gameweeks, optionally only those < `before`."""
    frames = [
        read_parquet(settings, gw_path(season, gw, "live"))
        for gw in available_gameweeks(settings, season, "live")
        if before is None or gw < before
    ]
    if not frames:
        return pd.DataFrame(columns=["id", "gameweek"])
    return pd.concat(frames, ignore_index=True)


def load_ep_next(settings: Settings, season: str, upto: int | None = None) -> pd.DataFrame:
    """FPL's expected points per (player, gameweek) from each stored players snapshot <= `upto`.

    Snapshot g holds the value published for gameweek g, so rows line up one to one.
    """
    frames = []
    for gw in available_gameweeks(settings, season, "players"):
        if upto is not None and gw > upto:
            continue
        snap = read_parquet(settings, gw_path(season, gw, "players"))
        frames.append(snap[["id", "ep_next"]].assign(gameweek=gw))
    if not frames:
        return pd.DataFrame(columns=["id", "gameweek", "ep_next"])
    return pd.concat(frames, ignore_index=True)


def load_prior(settings: Settings, season: str) -> pd.DataFrame | None:
    """Last season's per-player rates (keyed by `code`), or None if it isn't stored."""
    prev = features.previous_season(season)
    if prev not in available_seasons(settings) or not available_gameweeks(settings, prev, "live"):
        return None
    snap = max(available_gameweeks(settings, prev, "players"))
    players = read_parquet(settings, gw_path(prev, snap, "players"))
    if "code" not in players.columns:
        return None
    return features.prior_season_rates(load_live(settings, prev), players)


def season_features(
    settings: Settings,
    season: str,
    gameweeks: list[int] | None = None,
    snapshot_gw: int | None = None,
    live_before: int | None = None,
) -> pd.DataFrame:
    """Feature rows for one season.

    Defaults suit a finished (or backtested) season: every gameweek that has live data, with
    static tables from the season's latest snapshot. For prediction, pass the target
    gameweek's snapshot, `live_before` and the gameweeks up to and including the target.
    """
    snap = snapshot_gw or max(available_gameweeks(settings, season, "players"))
    players = read_parquet(settings, gw_path(season, snap, "players"))
    teams = read_parquet(settings, gw_path(season, snap, "teams"))
    fixtures = read_parquet(settings, gw_path(season, snap, "fixtures"))
    live = load_live(settings, season, before=live_before)
    if gameweeks is None:
        gameweeks = list(range(1, int(live["gameweek"].max()) + 1))
    ep_next = load_ep_next(settings, season, upto=snapshot_gw)
    prior = load_prior(settings, season)
    feats = features.build_features(live, players, teams, fixtures, gameweeks, ep_next, prior)
    feats.insert(0, "season", season)
    feats["dc_era"] = float(season >= features.DC_FIRST_SEASON)
    return feats
