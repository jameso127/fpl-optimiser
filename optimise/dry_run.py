"""Fixture data for DRY_RUN and tests: a small league, expected points, and a manager's squad.

The league has 10 clubs of 9 players (1 GK, 3 DEF, 3 MID, 2 FWD), so a legal 15-man squad
(2/5/5/3, at most 3 per club) can be built and improved. Expected points rise with price, with
noise, and a few players are unavailable (0).
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
                "model_version": "dry-run",
            }
        ),  # fmt: skip
        gw_path(SEASON, GAMEWEEK, "predictions"),
    )
    return world


class DryRunEntrySource:
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
