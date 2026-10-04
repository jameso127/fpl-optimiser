"""Leakage-safe features.

A row is (player, gameweek g). Form features use only live stats from gameweeks < g (every
rolling window is lagged by one gameweek). Fixture features for g (opponent, home/away, FPL
difficulty and strength ratings) are known before g's deadline, so they are allowed.

`ep_next` (FPL's expected points) for gameweek g is allowed only because it is published
before g's deadline. It is taken from the players snapshot of gameweek g itself, never from a
later one. For past seasons the snapshot comes from a community archive whose capture timing
is undocumented; see docs/backtest.md for the audit and caveats.
"""

import numpy as np
import pandas as pd

LIVE_SOURCES = [
    "total_points", "minutes", "starts", "expected_goal_involvements",
    "expected_goals_conceded", "bps", "ict_index",
]  # fmt: skip

FEATURES = [
    "element_type", "pts_l3", "pts_l5", "pts_season", "min_l3", "min_l5", "starts_l5",
    "xgi_l5", "xgc_l5", "bps_l5", "ict_l5", "games_played",
    "fix_n", "fix_difficulty", "fix_home", "own_att", "own_def", "opp_att", "opp_def",
]  # fmt: skip

XP_FEATURES = [*FEATURES, "ep_next"]


def _lagged_mean(grid: pd.DataFrame, col: str, window: int | None) -> pd.Series:
    """Per-player mean of `col` over the `window` gameweeks before each row (None = all prior)."""

    def per_player(s: pd.Series) -> pd.Series:
        lag = s.shift(1)
        if window is None:
            return lag.expanding(min_periods=1).mean()
        return lag.rolling(window, min_periods=1).mean()

    return grid.groupby("id")[col].transform(per_player)


def fixture_features(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, gameweek); double gameweeks are averaged and counted in `fix_n`."""
    f = fixtures.dropna(subset=["event"]).copy()
    f["event"] = f["event"].astype(int)
    home = pd.DataFrame(
        {"team": f["team_h"], "opp": f["team_a"], "gameweek": f["event"],
         "difficulty": f["team_h_difficulty"], "is_home": 1}
    )  # fmt: skip
    away = pd.DataFrame(
        {"team": f["team_a"], "opp": f["team_h"], "gameweek": f["event"],
         "difficulty": f["team_a_difficulty"], "is_home": 0}
    )  # fmt: skip
    long = pd.concat([home, away], ignore_index=True)

    strength = teams.set_index("id")
    h = long["is_home"] == 1
    for side, tcol in (("own", "team"), ("opp", "opp")):
        for kind in ("att", "def"):
            name = "attack" if kind == "att" else "defence"
            at_home = long[tcol].map(strength[f"strength_{name}_home"])
            at_away = long[tcol].map(strength[f"strength_{name}_away"])
            # Own side plays at its fixture venue; the opponent plays at the opposite venue.
            use_home = h if side == "own" else ~h
            long[f"{side}_{kind}"] = np.where(use_home, at_home, at_away)

    return (
        long.groupby(["team", "gameweek"])
        .agg(
            fix_n=("opp", "size"), fix_difficulty=("difficulty", "mean"),
            fix_home=("is_home", "mean"), own_att=("own_att", "mean"),
            own_def=("own_def", "mean"), opp_att=("opp_att", "mean"),
            opp_def=("opp_def", "mean"),
        )
        .reset_index()
    )  # fmt: skip


def build_features(
    live: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    gameweeks: list[int],
    ep_next: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for every player in `players` for each of `gameweeks`.

    `target` is that gameweek's actual points where `live` has them (NaN otherwise, e.g. the
    gameweek being predicted). Position and team come from the `players` snapshot. `ep_next`
    (columns id, gameweek, ep_next) is FPL's expected points as published for each gameweek.
    """
    ids = players["id"].unique()
    grid = pd.MultiIndex.from_product(
        [ids, range(1, max(gameweeks) + 1)], names=["id", "gameweek"]
    ).to_frame(index=False)

    cols = [c for c in LIVE_SOURCES if c in live.columns]
    live_num = live[["id", "gameweek", *cols]].copy()
    live_num[cols] = live_num[cols].apply(pd.to_numeric, errors="coerce")
    grid = grid.merge(live_num, on=["id", "gameweek"], how="left")
    for c in LIVE_SOURCES:
        if c not in grid.columns:
            grid[c] = np.nan
    grid["played"] = (grid["minutes"] > 0).astype(float).where(grid["minutes"].notna())

    out = grid[["id", "gameweek"]].copy()
    out["target"] = grid["total_points"]
    out["pts_l3"] = _lagged_mean(grid, "total_points", 3)
    out["pts_l5"] = _lagged_mean(grid, "total_points", 5)
    out["pts_season"] = _lagged_mean(grid, "total_points", None)
    out["min_l3"] = _lagged_mean(grid, "minutes", 3)
    out["min_l5"] = _lagged_mean(grid, "minutes", 5)
    out["starts_l5"] = _lagged_mean(grid, "starts", 5)
    out["xgi_l5"] = _lagged_mean(grid, "expected_goal_involvements", 5)
    out["xgc_l5"] = _lagged_mean(grid, "expected_goals_conceded", 5)
    out["bps_l5"] = _lagged_mean(grid, "bps", 5)
    out["ict_l5"] = _lagged_mean(grid, "ict_index", 5)
    out["games_played"] = grid.groupby("id")["played"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).sum()
    ).fillna(0)  # fmt: skip

    static = players[["id", "element_type", "team"]]
    out = out.merge(static, on="id", how="left")
    out = out.merge(fixture_features(fixtures, teams), on=["team", "gameweek"], how="left")
    out["fix_n"] = out["fix_n"].fillna(0)

    # Merged on (id, gameweek) so a row only ever sees the expected points published for its
    # own gameweek, never a later one. Gameweeks with no stored value stay NaN.
    if ep_next is None:
        out["ep_next"] = np.nan
    else:
        out = out.merge(ep_next[["id", "gameweek", "ep_next"]], on=["id", "gameweek"], how="left")

    out = out[out["gameweek"].isin(gameweeks)].reset_index(drop=True)
    return out[["id", "gameweek", "target", *XP_FEATURES]]


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that can be trained or scored on: the player had at least one fixture."""
    return df[df["fix_n"] >= 1]
