"""Flatten raw FPL API payloads into tidy DataFrames (pure functions, no I/O)."""

from typing import Any

import pandas as pd

PLAYER_COLUMNS = [
    "id", "web_name", "first_name", "second_name", "team", "element_type", "status",
    "now_cost", "chance_of_playing_this_round", "chance_of_playing_next_round",
    "total_points", "minutes", "form", "points_per_game", "ep_this", "ep_next",
    "selected_by_percent", "goals_scored", "assists", "clean_sheets", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded",
]  # fmt: skip

# The API returns these as strings.
_NUMERIC_STRINGS = [
    "form", "points_per_game", "ep_this", "ep_next", "selected_by_percent",
    "influence", "creativity", "threat", "ict_index", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
]  # fmt: skip

TEAM_COLUMNS = [
    "id", "name", "short_name", "strength", "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away", "strength_defence_home",
    "strength_defence_away",
]  # fmt: skip

FIXTURE_COLUMNS = [
    "id", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty",
    "kickoff_time", "finished", "team_h_score", "team_a_score",
]  # fmt: skip


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The listed columns that exist in `df` (APIs and CSVs differ between seasons)."""
    return df[[c for c in columns if c in df.columns]].copy()


def season_label(bootstrap: dict[str, Any]) -> str:
    """e.g. '2026-27', from the first gameweek's deadline (seasons start in August)."""
    first = min(e["deadline_time"] for e in bootstrap["events"])
    year, month = int(first[:4]), int(first[5:7])
    start = year if month >= 6 else year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def players_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    df = select_columns(pd.DataFrame(bootstrap["elements"]), PLAYER_COLUMNS)
    for col in _NUMERIC_STRINGS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep nulls as nulls: null chance_of_playing means "no news", not "0%".
    for col in ("chance_of_playing_this_round", "chance_of_playing_next_round"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def teams_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    return select_columns(pd.DataFrame(bootstrap["teams"]), TEAM_COLUMNS)


def events_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    return select_columns(
        pd.DataFrame(bootstrap["events"]),
        ["id", "name", "deadline_time", "finished", "is_current", "is_next"],
    )


def fixtures_frame(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
    return select_columns(pd.DataFrame(fixtures), FIXTURE_COLUMNS)


def live_frame(live: dict[str, Any], gameweek: int) -> pd.DataFrame:
    """One row per player: id, gameweek and the per-gameweek stats (minutes, goals, bps, ...)."""
    rows = [{"id": e["id"], "gameweek": gameweek, **e["stats"]} for e in live["elements"]]
    df = pd.DataFrame(rows)
    for col in _NUMERIC_STRINGS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def target_gameweek(bootstrap: dict[str, Any]) -> int:
    """The next upcoming gameweek, else the current one, else the last (end of season)."""
    events = bootstrap["events"]
    for flag in ("is_next", "is_current"):
        for event in events:
            if event.get(flag):
                return int(event["id"])
    return int(events[-1]["id"])


def finished_gameweeks(bootstrap: dict[str, Any]) -> list[int]:
    return [int(e["id"]) for e in bootstrap["events"] if e.get("finished")]
