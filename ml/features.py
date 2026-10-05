"""Leakage-safe features.

A row is (player, gameweek g). Every form feature (player and team) uses only data from
gameweeks < g: each rolling window is lagged by one gameweek. Fixture features for g
(opponent, home/away, FPL difficulty and strength ratings, rest days) are known before g's
deadline, so they are allowed. Previous-season features come from a finished season.

`ep_next` (FPL's expected points) for gameweek g is allowed only because it is published
before g's deadline. It is taken from the players snapshot of gameweek g itself, never from a
later one. See docs/backtest.md for the audit and caveats.
"""

import numpy as np
import pandas as pd

LIVE_SOURCES = [
    "total_points", "minutes", "starts", "expected_goal_involvements",
    "expected_goals_conceded", "bps", "ict_index", "goals_scored", "assists", "clean_sheets",
    "saves", "bonus", "threat", "creativity", "influence", "expected_goals",
    "expected_assists", "yellow_cards", "red_cards", "defensive_contribution",
]  # fmt: skip

# The first model's features (kept for the v1 comparison in the backtest).
FEATURES = [
    "element_type", "pts_l3", "pts_l5", "pts_season", "min_l3", "min_l5", "starts_l5",
    "xgi_l5", "xgc_l5", "bps_l5", "ict_l5", "games_played",
    "fix_n", "fix_difficulty", "fix_home", "own_att", "own_def", "opp_att", "opp_def",
]  # fmt: skip

XP_FEATURES = [*FEATURES, "ep_next"]

# Feature groups added for the second model; each can be switched off for an ablation.
PLAYER_DETAIL = [
    "pts_l1", "min_l1", "starts_l3", "played_l5", "full_l5", "sub_l5", "pts_trend",
    "pts90_l5", "pts90_season", "xg90_l5", "xa90_l5", "bps90_l5", "threat90_l5",
    "creat90_l5", "inf90_l5", "saves90_l5", "cs_rate_l5", "bonus_l5", "dc_l5", "yc_l5",
]  # fmt: skip
TEAM_FORM = [
    "t_gf_l5", "t_ga_l5", "t_xgf_l5", "t_xga_l5", "t_cs_l5",
    "o_gf_l5", "o_ga_l5", "o_xgf_l5", "o_xga_l5", "o_cs_l5",
]  # fmt: skip
REST = ["rest_days", "opp_rest_days"]
# 2025-26 introduced defensive-contribution points, shifting defender and midfielder scoring.
RULES = ["dc_era"]
DC_FIRST_SEASON = "2025-26"
PRIOR = ["prev_pts90", "prev_min", "prev_games", "prev_xgi90", "prev_starts"]

GROUPS = {
    "player_detail": PLAYER_DETAIL,
    "team_form": TEAM_FORM,
    "rest": REST,
    "prior_season": PRIOR,
    "scoring_regime": RULES,
}
FEATURES_V2 = [*FEATURES, *PLAYER_DETAIL, *TEAM_FORM, *REST, *PRIOR, *RULES]


def _lagged(grid: pd.DataFrame, col: str, window: int | None, how: str, key: str) -> pd.Series:
    """Per-`key` mean/sum of `col` over the `window` gameweeks before each row (None = all)."""

    def per_group(s: pd.Series) -> pd.Series:
        lag = s.shift(1)
        roll = lag.expanding(min_periods=1) if window is None else lag.rolling(window, 1)
        return roll.mean() if how == "mean" else roll.sum()

    return grid.groupby(key)[col].transform(per_group)


def _lagged_mean(grid: pd.DataFrame, col: str, window: int | None) -> pd.Series:
    return _lagged(grid, col, window, "mean", "id")


def _per90(grid: pd.DataFrame, col: str, window: int | None, min_minutes: float = 90) -> pd.Series:
    """Rate per 90 minutes over prior gameweeks; NaN until the player has `min_minutes`."""
    total = _lagged(grid, col, window, "sum", "id")
    minutes = _lagged(grid, "minutes", window, "sum", "id")
    return (total / minutes * 90).where(minutes >= min_minutes)


def _fixture_long(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture): opponent, venue, goals for/against, kickoff."""
    f = fixtures.dropna(subset=["event"]).copy()
    f["event"] = f["event"].astype(int)
    fid = f["id"] if "id" in f.columns else pd.Series(range(len(f)), index=f.index)
    kick = (
        pd.to_datetime(f["kickoff_time"], utc=True, errors="coerce")
        if "kickoff_time" in f.columns
        else pd.Series(pd.NaT, index=f.index)
    )
    hs = f["team_h_score"] if "team_h_score" in f.columns else pd.Series(np.nan, index=f.index)
    as_ = f["team_a_score"] if "team_a_score" in f.columns else pd.Series(np.nan, index=f.index)
    home = pd.DataFrame(
        {"fid": fid, "team": f["team_h"], "opp": f["team_a"], "gameweek": f["event"],
         "difficulty": f["team_h_difficulty"], "is_home": 1, "gf": hs, "ga": as_, "kickoff": kick}
    )  # fmt: skip
    away = pd.DataFrame(
        {"fid": fid, "team": f["team_a"], "opp": f["team_h"], "gameweek": f["event"],
         "difficulty": f["team_a_difficulty"], "is_home": 0, "gf": as_, "ga": hs, "kickoff": kick}
    )  # fmt: skip
    return pd.concat([home, away], ignore_index=True)


def fixture_features(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, gameweek); double gameweeks are averaged and counted in `fix_n`."""
    long = _fixture_long(fixtures)
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


def rest_days(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous league match, for the team and its opponent.

    Premier League fixtures only (cup and European games are not in the data), so this is a
    partial congestion signal. Kickoff times are those in the stored fixtures table.
    """
    long = _fixture_long(fixtures)
    if long["kickoff"].isna().all():
        return pd.DataFrame(columns=["team", "gameweek", "rest_days", "opp_rest_days"])
    long = long.sort_values(["team", "kickoff"])
    long["rest"] = long.groupby("team")["kickoff"].diff().dt.total_seconds() / 86400
    long["rest"] = long["rest"].clip(upper=14)
    opp_rest = long[["fid", "team", "rest"]].rename(columns={"team": "opp", "rest": "opp_rest"})
    long = long.merge(opp_rest, on=["fid", "opp"], how="left")
    return (
        long.groupby(["team", "gameweek"])
        .agg(rest_days=("rest", "mean"), opp_rest_days=("opp_rest", "mean"))
        .reset_index()
    )


def team_form(
    live: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame, max_gw: int
) -> pd.DataFrame:
    """Rolling attack/defence form per team (and its opponents') from gameweeks before each row.

    Goals come from fixture results; xG for is the sum of the team's players' expected goals,
    xG against the largest `expected_goals_conceded` among its players who played 45+ minutes.
    Per-game values, so double gameweeks are not inflated.
    """
    long = _fixture_long(fixtures)
    results = long.groupby(["team", "gameweek"]).agg(gf=("gf", "mean"), ga=("ga", "mean"))
    games = long.groupby(["team", "gameweek"]).size().rename("games")

    stats = live.merge(players[["id", "team"]], on="id", how="inner")
    for col in ("expected_goals", "expected_goals_conceded", "minutes"):
        stats[col] = pd.to_numeric(stats[col], errors="coerce") if col in stats else np.nan
    xgf = stats.groupby(["team", "gameweek"])["expected_goals"].sum(min_count=1)
    played = stats[stats["minutes"] >= 45]
    xga = played.groupby(["team", "gameweek"])["expected_goals_conceded"].max()

    team_ids = sorted(set(long["team"]))
    grid = pd.MultiIndex.from_product(
        [team_ids, range(1, max_gw + 1)], names=["team", "gameweek"]
    ).to_frame(index=False)
    grid = (
        grid.merge(results.reset_index(), on=["team", "gameweek"], how="left")
        .merge(games.reset_index(), on=["team", "gameweek"], how="left")
        .merge(xgf.rename("xgf").reset_index(), on=["team", "gameweek"], how="left")
        .merge(xga.rename("xga").reset_index(), on=["team", "gameweek"], how="left")
    )
    grid["xgf"] = grid["xgf"] / grid["games"]
    grid["xga"] = grid["xga"] / grid["games"]
    grid["cs"] = (grid["ga"] == 0).astype(float).where(grid["ga"].notna())
    form = grid[["team", "gameweek"]].copy()
    for src, name in (("gf", "gf"), ("ga", "ga"), ("xgf", "xgf"), ("xga", "xga"), ("cs", "cs")):
        form[f"t_{name}_l5"] = _lagged(grid, src, 5, "mean", "team")

    # Opponents' form for the same fixtures, averaged over a double gameweek.
    opp = form.rename(columns={"team": "opp", **{c: c.replace("t_", "o_") for c in TEAM_FORM}})
    opp_cols = [c for c in opp.columns if c.startswith("o_")]
    pairs = long[["team", "opp", "gameweek"]].merge(opp, on=["opp", "gameweek"], how="left")
    opp_form = pairs.groupby(["team", "gameweek"])[opp_cols].mean().reset_index()
    return form.merge(opp_form, on=["team", "gameweek"], how="left")


def previous_season(season: str) -> str:
    """'2025-26' -> '2024-25'."""
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def prior_season_rates(live: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """A finished season's per-player rates keyed by the stable cross-season `code`."""
    df = live.copy()
    for col in ("minutes", "total_points", "expected_goal_involvements", "starts"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    n_gws = max(int(df["gameweek"].nunique()), 1)
    agg = df.groupby("id").agg(
        minutes=("minutes", "sum"), points=("total_points", "sum"),
        xgi=("expected_goal_involvements", "sum"), starts=("starts", "sum"),
        games=("minutes", lambda s: int((s > 0).sum())),
    )  # fmt: skip
    enough = agg["minutes"] >= 450
    out = pd.DataFrame(
        {
            "prev_pts90": (agg["points"] / agg["minutes"] * 90).where(enough),
            "prev_min": agg["minutes"] / n_gws,
            "prev_games": agg["games"],
            "prev_xgi90": (agg["xgi"] / agg["minutes"] * 90).where(enough),
            "prev_starts": agg["starts"],
        }
    )
    out = out.reset_index().merge(players[["id", "code"]], on="id", how="inner")
    return out.drop(columns="id")


def build_features(
    live: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    gameweeks: list[int],
    ep_next: pd.DataFrame | None = None,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for every player in `players` for each of `gameweeks`.

    `target` / `target_minutes` are that gameweek's actual points / minutes where `live` has
    them (NaN otherwise, e.g. the gameweek being predicted). Position and team come from the
    `players` snapshot. `ep_next` (columns id, gameweek, ep_next) is FPL's expected points as
    published for each gameweek. `prior` (columns code, prev_*) holds last season's rates.
    """
    max_gw = max(gameweeks)
    ids = players["id"].unique()
    index = pd.MultiIndex.from_product([ids, range(1, max_gw + 1)], names=["id", "gameweek"])
    grid = index.to_frame(index=False)

    cols = [c for c in LIVE_SOURCES if c in live.columns]
    live_num = live[["id", "gameweek", *cols]].copy()
    live_num[cols] = live_num[cols].apply(pd.to_numeric, errors="coerce")
    grid = grid.merge(live_num, on=["id", "gameweek"], how="left")
    for c in LIVE_SOURCES:
        if c not in grid.columns:
            grid[c] = np.nan
    has_minutes = grid["minutes"].notna()
    grid["played"] = (grid["minutes"] > 0).astype(float).where(has_minutes)
    grid["full90"] = (grid["minutes"] >= 85).astype(float).where(has_minutes)
    grid["sub"] = ((grid["minutes"] > 0) & (grid["minutes"] < 60)).astype(float).where(has_minutes)

    out = grid[["id", "gameweek"]].copy()
    out["target"] = grid["total_points"]
    out["target_minutes"] = grid["minutes"]
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
    out["games_played"] = _lagged(grid, "played", None, "sum", "id").fillna(0)

    out["pts_l1"] = _lagged_mean(grid, "total_points", 1)
    out["min_l1"] = _lagged_mean(grid, "minutes", 1)
    out["starts_l3"] = _lagged_mean(grid, "starts", 3)
    out["played_l5"] = _lagged_mean(grid, "played", 5)
    out["full_l5"] = _lagged_mean(grid, "full90", 5)
    out["sub_l5"] = _lagged_mean(grid, "sub", 5)
    out["pts_trend"] = out["pts_l3"] - out["pts_season"]
    out["pts90_l5"] = _per90(grid, "total_points", 5)
    out["pts90_season"] = _per90(grid, "total_points", None)
    out["xg90_l5"] = _per90(grid, "expected_goals", 5)
    out["xa90_l5"] = _per90(grid, "expected_assists", 5)
    out["bps90_l5"] = _per90(grid, "bps", 5)
    out["threat90_l5"] = _per90(grid, "threat", 5)
    out["creat90_l5"] = _per90(grid, "creativity", 5)
    out["inf90_l5"] = _per90(grid, "influence", 5)
    out["saves90_l5"] = _per90(grid, "saves", 5)
    games5 = _lagged(grid, "played", 5, "sum", "id")
    out["cs_rate_l5"] = (_lagged(grid, "clean_sheets", 5, "sum", "id") / games5).where(games5 > 0)
    out["bonus_l5"] = _lagged_mean(grid, "bonus", 5)
    out["dc_l5"] = _lagged_mean(grid, "defensive_contribution", 5)
    out["yc_l5"] = _lagged(grid, "yellow_cards", 5, "sum", "id")

    static = players[["id", "element_type", "team"]]
    out = out.merge(static, on="id", how="left")
    out = out.merge(fixture_features(fixtures, teams), on=["team", "gameweek"], how="left")
    out["fix_n"] = out["fix_n"].fillna(0)

    out = out.merge(team_form(live, players, fixtures, max_gw), on=["team", "gameweek"], how="left")
    rest = rest_days(fixtures)
    if rest.empty:
        out["rest_days"] = np.nan
        out["opp_rest_days"] = np.nan
    else:
        out = out.merge(rest, on=["team", "gameweek"], how="left")

    if prior is not None and not prior.empty and "code" in players.columns:
        codes = players[["id", "code"]].merge(prior, on="code", how="left").drop(columns="code")
        out = out.merge(codes, on="id", how="left")
    for col in PRIOR:
        if col not in out.columns:
            out[col] = np.nan
    out["dc_era"] = 0.0  # set by the caller, which knows the season (see data.season_features)

    # Merged on (id, gameweek) so a row only ever sees the expected points published for its
    # own gameweek, never a later one. Gameweeks with no stored value stay NaN.
    if ep_next is None:
        out["ep_next"] = np.nan
    else:
        out = out.merge(ep_next[["id", "gameweek", "ep_next"]], on=["id", "gameweek"], how="left")

    out = out[out["gameweek"].isin(gameweeks)].reset_index(drop=True)
    return out[["id", "gameweek", "target", "target_minutes", *FEATURES_V2, "ep_next"]]


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that can be trained or scored on: the player had at least one fixture."""
    return df[df["fix_n"] >= 1]
