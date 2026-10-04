"""Load stored Parquet and turn each season into feature rows (tagged with a `season` column).

Player ids are only meaningful within a season, so form features never cross a season
boundary. The model sees seasons as separate pools of (player, gameweek) rows.
"""

import pandas as pd

from common.config import Settings
from common.storage import available_gameweeks, gw_path, read_parquet
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
    feats = features.build_features(live, players, teams, fixtures, gameweeks)
    feats.insert(0, "season", season)
    return feats
