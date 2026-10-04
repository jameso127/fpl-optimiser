"""One-off import of past seasons into the same Parquet layout as live ingest.

Source: the community archive github.com/vaastav/Fantasy-Premier-League (a third-party
dataset, not FPL itself). The official API keeps no per-gameweek history for finished
seasons. For each season it writes season=<yyyy-yy>/gw=<n>/{live,players,teams,fixtures}.parquet:

- live:    per-player stats for the gameweek. Double gameweeks (one source row per fixture)
           are summed, matching the live API.
- players: static info from players_raw (id, name, team, position) plus, for that gameweek,
           `ep_next` = the archive's `xP` summed over the gameweek's fixtures, and `now_cost`.
           This stands in for FPL's pre-deadline expected points as the backtest benchmark.
           NOTE: how early in the gameweek the archive captured `xP` is not documented, so
           treat it as a benchmark of unknown timing, not a guaranteed pre-deadline number.
- teams, fixtures: the season's tables, repeated per gameweek like live snapshots.

Players' status and chance-of-playing are not available historically and are omitted.
Re-running overwrites. Run locally; upload the result to the data bucket afterwards.

    uv run python -m ingest.history                  # default seasons
    uv run python -m ingest.history 2024-25 2025-26
"""

import argparse
import io
import logging
from collections.abc import Callable

import pandas as pd

from common.config import Settings, get_settings
from common.fpl_client import FplClient
from common.logging import configure_logging
from common.storage import gw_path, write_parquet
from ingest.transform import FIXTURE_COLUMNS, TEAM_COLUMNS, select_columns

log = logging.getLogger(__name__)

DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

LIVE_STATS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus",
    "bps", "influence", "creativity", "threat", "ict_index", "starts", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution",
    "total_points",
]  # fmt: skip

PLAYER_STATIC = ["id", "web_name", "first_name", "second_name", "team", "element_type"]

Fetch = Callable[[str], bytes]


def read_csv(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    except UnicodeDecodeError:  # some older seasons are not UTF-8
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", low_memory=False)


def live_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, gameweek), stats summed over that gameweek's fixtures."""
    df = merged.rename(columns={"element": "id", "GW": "gameweek"})
    stats = [c for c in LIVE_STATS if c in df.columns]
    numeric = df[stats].apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([df[["id", "gameweek"]], numeric], axis=1)
    return combined.groupby(["id", "gameweek"], as_index=False).sum(min_count=1)


def expected_points_by_gameweek(merged: pd.DataFrame) -> pd.DataFrame:
    """id, gameweek, ep_next (summed `xP`) and now_cost (price after the last fixture)."""
    df = merged.rename(columns={"element": "id", "GW": "gameweek"})
    grouped = df.groupby(["id", "gameweek"])
    out = grouped["xP"].sum(min_count=1).rename("ep_next").to_frame()
    out["now_cost"] = grouped["value"].last()
    out = out.reset_index()
    # The archive records a gameweek it failed to capture as xP = 0 for every player (seen in
    # 2025-26). A real gameweek always has some positive xP, so all-zero means missing.
    captured = out.groupby("gameweek")["ep_next"].transform(lambda s: (s.fillna(0) > 0).any())
    out.loc[~captured.astype(bool), "ep_next"] = float("nan")
    return out


def players_snapshot(players_raw: pd.DataFrame, ep_for_gw: pd.DataFrame) -> pd.DataFrame:
    static = select_columns(players_raw, PLAYER_STATIC)
    return static.merge(ep_for_gw[["id", "ep_next", "now_cost"]], on="id", how="left")


def import_season(settings: Settings, season: str, fetch: Fetch) -> int:
    merged = read_csv(fetch(f"{season}/gws/merged_gw.csv"))
    players_raw = read_csv(fetch(f"{season}/players_raw.csv"))
    teams = select_columns(read_csv(fetch(f"{season}/teams.csv")), TEAM_COLUMNS)
    fixtures = select_columns(read_csv(fetch(f"{season}/fixtures.csv")), FIXTURE_COLUMNS)

    live = live_from_merged(merged)
    ep = expected_points_by_gameweek(merged)
    gameweeks = sorted(int(g) for g in live["gameweek"].unique())
    for gw in gameweeks:
        frames = {
            "live": live[live["gameweek"] == gw].reset_index(drop=True),
            "players": players_snapshot(players_raw, ep[ep["gameweek"] == gw]),
            "teams": teams,
            "fixtures": fixtures,
        }
        for name, df in frames.items():
            write_parquet(settings, df, gw_path(season, gw, name))
    log.info(
        "imported season",
        extra={"season": season, "gameweeks": len(gameweeks), "live_rows": len(live)},
    )
    return len(gameweeks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("seasons", nargs="*", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    client = FplClient(settings, base_url=settings.history_base_url)
    for season in args.seasons:
        import_season(settings, season, lambda path: client.get(path, raw=True))


if __name__ == "__main__":
    main()
