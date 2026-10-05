"""Test doubles: a fake FPL API and a small synthetic league.

`FakeFplSource` stands in for the FPL API when testing ingest. The league is for the optimiser:
10 clubs of 9 players (1 GK, 3 DEF, 3 MID, 2 FWD), so a legal 15-man squad (2/5/5/3, at most 3
per club) can be built and improved. Expected points rise with price, with noise, and a few
players are unavailable (0).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings
from common.storage import gw_path, write_parquet

SEASON = "2025-26"
GAMEWEEK = 3
TEAM_ID = 4242

_PLAYERS = [
    # id, first, second, team, position (1 GK, 2 DEF, 3 MID, 4 FWD), cost
    (1, "Alex", "Keeper", 1, 1, 50),
    (2, "Ben", "Stopper", 2, 1, 45),
    (3, "Chris", "Back", 1, 2, 55),
    (4, "Dan", "Wing", 2, 2, 60),
    (5, "Ed", "Mid", 3, 3, 80),
    (6, "Finn", "Playmaker", 4, 3, 95),
    (7, "Gus", "Striker", 3, 4, 90),
    (8, "Hal", "Poacher", 4, 4, 70),
]


def _bootstrap() -> dict[str, Any]:
    elements = [
        {
            "id": pid, "code": 1000 + pid, "web_name": second, "first_name": first,
            "second_name": second, "team": team, "element_type": pos, "status": "a",
            "now_cost": cost, "news": "", "penalties_order": None,
            "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": None,
            "chance_of_playing_this_round": None,
            "chance_of_playing_next_round": 50 if pid == 6 else None,
            "total_points": 10 + pid, "minutes": 180, "form": f"{pid / 2:.1f}",
            "points_per_game": "4.0", "ep_this": "3.0", "ep_next": f"{2 + pid / 4:.1f}",
            "selected_by_percent": "10.0", "goals_scored": 1, "assists": 1,
            "clean_sheets": 1, "bonus": 1, "bps": 30, "influence": "20.0",
            "creativity": "15.0", "threat": "10.0", "ict_index": "4.5",
            "expected_goals": "0.5", "expected_assists": "0.3",
            "expected_goal_involvements": "0.8", "expected_goals_conceded": "1.2",
        }
        for pid, first, second, team, pos, cost in _PLAYERS
    ]  # fmt: skip
    teams = [
        {
            "id": tid, "name": f"Team {tid}", "short_name": f"T{tid}", "strength": 3,
            "strength_overall_home": 1100, "strength_overall_away": 1080,
            "strength_attack_home": 1100, "strength_attack_away": 1080,
            "strength_defence_home": 1100, "strength_defence_away": 1080,
        }
        for tid in range(1, 5)
    ]  # fmt: skip
    events = [
        {"id": 1, "name": "Gameweek 1", "deadline_time": "2025-08-15T17:30:00Z",
         "finished": True, "is_current": False, "is_next": False},
        {"id": 2, "name": "Gameweek 2", "deadline_time": "2025-08-22T17:30:00Z",
         "finished": True, "is_current": True, "is_next": False},
        {"id": 3, "name": "Gameweek 3", "deadline_time": "2025-08-29T17:30:00Z",
         "finished": False, "is_current": False, "is_next": True},
    ]  # fmt: skip
    return {"elements": elements, "teams": teams, "events": events}


class FakeFplSource:
    """Stands in for the FPL API: 8 players, 4 teams, gameweeks 1-3 (1 and 2 finished)."""

    def bootstrap(self) -> dict[str, Any]:
        return _bootstrap()

    def fixtures(self) -> list[dict[str, Any]]:
        """Two matches a gameweek; gameweeks 1 and 2 are finished with scores."""
        rows = []
        for event, day in ((1, "15"), (2, "22"), (3, "30")):
            done = event < 3
            for k, (home, away, score) in enumerate(((1, 2, (2, 0)), (3, 4, (1, 1)))):
                rows.append(
                    {
                        "id": event * 10 + k, "event": event, "team_h": home, "team_a": away,
                        "team_h_difficulty": 2 + k, "team_a_difficulty": 4 - k,
                        "kickoff_time": f"2025-08-{day}T{14 + 2 * k}:00:00Z", "finished": done,
                        "team_h_score": score[0] if done else None,
                        "team_a_score": score[1] if done else None,
                    }
                )  # fmt: skip
        return rows

    def live(self, gameweek: int) -> dict[str, Any]:
        return {
            "elements": [
                {
                    "id": pid,
                    "stats": {
                        "minutes": 90, "goals_scored": int(pid % 3 == 0), "assists": 0,
                        "clean_sheets": 0, "bonus": 0, "bps": 20 + pid,
                        "total_points": 2 + (pid + gameweek) % 6, "starts": 1,
                        "expected_goal_involvements": "0.4", "expected_goals_conceded": "0.9",
                        "ict_index": "3.2",
                    },
                }
                for pid, *_ in _PLAYERS
            ]
        }  # fmt: skip


PER_CLUB = ((1, 1), (2, 3), (3, 3), (4, 2))  # (position, how many) per club
BASE_PRICE = {1: 40, 2: 40, 3: 45, 4: 45}
SPREAD = {1: 15, 2: 30, 3: 80, 4: 70}


@dataclass(frozen=True)
class World:
    players: pd.DataFrame  # id, web_name, team, element_type, now_cost, xpts, ...
    teams: pd.DataFrame
    squad: tuple[int, ...]
    purchase: dict[int, int]
    bank: int


def synthetic_world(seed: int = 7, clubs: int = 10) -> World:
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for club in range(1, clubs + 1):
        for position, count in PER_CLUB:
            for _ in range(count):
                pid += 1
                price = BASE_PRICE[position] + int(rng.integers(0, SPREAD[position] // 5)) * 5
                xpts = max(0.0, (price - 35) / 12 + rng.normal(0, 0.9))
                xpts = 0.0 if rng.random() < 0.05 else xpts  # injured or suspended
                rows.append(
                    {
                        "id": pid, "web_name": f"P{pid:02d}", "team": club,
                        "element_type": position, "now_cost": price, "xpts": round(xpts, 3),
                    }
                )  # fmt: skip
    players = pd.DataFrame(rows)
    teams = pd.DataFrame(
        {
            "id": range(1, clubs + 1),
            "name": [f"Club {i}" for i in range(1, clubs + 1)],
            "short_name": [f"C{i:02d}" for i in range(1, clubs + 1)],
        }
    )
    squad = _legal_squad(players, rng)
    purchase = {i: int(players.loc[players["id"] == i, "now_cost"].iloc[0]) for i in squad}
    return World(players, teams, tuple(squad), purchase, bank=5)


def _legal_squad(players: pd.DataFrame, rng: np.random.Generator) -> list[int]:
    """A random legal squad costing at most 100.0m."""
    want = {1: 2, 2: 5, 3: 5, 4: 3}
    while True:
        chosen: list[int] = []
        per_club: dict[int, int] = {}
        for position, count in want.items():
            pool = players[players["element_type"] == position].sample(
                frac=1, random_state=int(rng.integers(1 << 30))
            )
            for _, p in pool.iterrows():
                if count == 0:
                    break
                if per_club.get(int(p["team"]), 0) < 3:
                    chosen.append(int(p["id"]))
                    per_club[int(p["team"])] = per_club.get(int(p["team"]), 0) + 1
                    count -= 1
        cost = int(players[players["id"].isin(chosen)]["now_cost"].sum())
        if len(chosen) == 15 and cost <= 995:
            return chosen


def seed_storage(settings: Settings, world: World | None = None) -> World:
    """Write the fixture league and its predictions where the optimise job reads them."""
    world = world or synthetic_world()
    p = world.players
    write_parquet(
        settings,
        p[["id", "web_name", "team", "element_type", "now_cost"]].assign(status="a"),
        gw_path(SEASON, GAMEWEEK, "players"),
    )
    write_parquet(settings, world.teams, gw_path(SEASON, GAMEWEEK, "teams"))
    events = pd.DataFrame(
        {"id": [1, 2, 3], "finished": [True, True, False], "is_next": [False, False, True]}
    )
    write_parquet(settings, events, gw_path(SEASON, GAMEWEEK, "events"))
    write_parquet(
        settings,
        pd.DataFrame(
            {
                "id": p["id"],
                "gameweek": GAMEWEEK,
                "xpts": p["xpts"],
                "ep_next": p["xpts"],
                "availability": 1.0,
                "fix_n": 1,
                "model_version": "test-model",
            }
        ),  # fmt: skip
        gw_path(SEASON, GAMEWEEK, "predictions"),
    )
    return world


class FakeEntrySource:
    """Answers the per-manager endpoints for the fixture squad, with one past transfer so that
    one player's selling price differs from his market price."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.bought = world.squad[0]  # transferred in during gameweek 2, 0.6m cheaper than now
        self.cost_when_bought = world.purchase[self.bought] - 6

    def entry_history(self, team_id: int) -> dict[str, Any]:
        return {
            "current": [
                {"event": 1, "bank": 5, "value": 1000, "event_transfers": 0},
                {"event": 2, "bank": 5, "value": 1000, "event_transfers": 1},
            ],
            "chips": [],
        }

    def entry_picks(self, team_id: int, gameweek: int) -> dict[str, Any]:
        return {
            "entry_history": {"bank": self.world.bank},
            "picks": [{"element": i} for i in self.world.squad],
        }

    def entry_transfers(self, team_id: int) -> list[dict[str, Any]]:
        return [
            {
                "event": 2, "time": "2025-08-20T10:00:00Z", "element_in": self.bought,
                "element_in_cost": self.cost_when_bought, "element_out": 0, "element_out_cost": 0,
            }
        ]  # fmt: skip

    def element_summary(self, player_id: int) -> dict[str, Any]:
        price = self.world.purchase[player_id]
        return {"history": [{"round": 1, "value": price}]}
