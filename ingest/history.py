"""One-off import of past seasons into the same Parquet layout as live ingest.

Source: the community archive github.com/vaastav/Fantasy-Premier-League (a third-party
dataset, not FPL itself). The official API keeps no per-gameweek history for finished
seasons. For each season it writes season=<yyyy-yy>/gw=<n>/{live,players,teams,fixtures}.parquet:

- live:    per-player stats for the gameweek. Double gameweeks (one source row per fixture)
           are summed, matching the live API.
- players: static info from players_raw (id, name, team, position) plus `ep_next`, FPL's
           expected points as known BEFORE that gameweek's deadline. The archive's `xP` on row g
           was captured after gameweek g (verified: its change tracks points scored in g), so
           it is stored against gameweek g+1. Using the same-gameweek value would leak the
           outcome. Gameweek 1 therefore has no `ep_next`. Gameweeks the archive recorded as
           all-zero are null, and values above 20 are treated as data errors.
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

MAX_PLAUSIBLE_XP = 20.0

PLAYER_STATIC = ["id", "code", "web_name", "first_name", "second_name", "team", "element_type"]

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
    """id, gameweek, ep_next: FPL's expected points as published BEFORE `gameweek`'s deadline.

    The archive's `xP` on row g was captured after gameweek g's matches (its week-on-week
    change tracks points scored in g, correlation +0.4 to +0.6), so using it for g would leak
    the outcome. It is the expected-points figure FPL showed for g+1, so it is stored against
    gameweek g+1, which matches the meaning of a live pre-deadline snapshot. Gameweek 1 has none.
    """
    df = merged.rename(columns={"element": "id", "GW": "gameweek"})
    # In a double gameweek FPL repeats the same gameweek-level figure (already about double a
    # single-fixture value) on each fixture row, so take it once rather than summing.
    out = df.groupby(["id", "gameweek"])["xP"].mean().rename("ep_next").reset_index()
    # The archive records a gameweek it failed to capture as xP = 0 for every player (seen in
    # 2025-26). A real gameweek always has some positive xP, so all-zero means missing.
    captured = out.groupby("gameweek")["ep_next"].transform(lambda s: (s.fillna(0) > 0).any())
    out.loc[~captured.astype(bool), "ep_next"] = float("nan")
    # Implausible values are data errors (a single player's xP above 20 in one gameweek; the
    # archive has ~120 such rows, up to 52.8), not predictions.
    out.loc[out["ep_next"] > MAX_PLAUSIBLE_XP, "ep_next"] = float("nan")
    out["gameweek"] += 1  # captured after gameweek g, so it is what was known before g+1
    return out


def selectable_players(players_raw: pd.DataFrame) -> pd.DataFrame:
    """Drop non-players (managers, assistant managers) that some seasons' files include."""
    return players_raw[players_raw["element_type"].isin([1, 2, 3, 4])]


def players_snapshot(players_raw: pd.DataFrame, ep_for_gw: pd.DataFrame) -> pd.DataFrame:
    static = select_columns(selectable_players(players_raw), PLAYER_STATIC)
    return static.merge(ep_for_gw[["id", "ep_next"]], on="id", how="left")


def import_season(settings: Settings, season: str, fetch: Fetch) -> int:
    merged = read_csv(fetch(f"{season}/gws/merged_gw.csv"))
    players_raw = read_csv(fetch(f"{season}/players_raw.csv"))
    teams = select_columns(read_csv(fetch(f"{season}/teams.csv")), TEAM_COLUMNS)
    fixtures = select_columns(read_csv(fetch(f"{season}/fixtures.csv")), FIXTURE_COLUMNS)

    live = live_from_merged(merged)
    live = live[live["id"].isin(selectable_players(players_raw)["id"])]
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
