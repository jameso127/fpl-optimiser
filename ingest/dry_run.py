"""Fixture data standing in for the FPL API when DRY_RUN=true (works out of season and in CI)."""

from typing import Any

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
            "id": pid, "web_name": second, "first_name": first, "second_name": second,
            "team": team, "element_type": pos, "status": "a", "now_cost": cost,
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


class DryRunSource:
    def bootstrap(self) -> dict[str, Any]:
        return _bootstrap()

    def fixtures(self) -> list[dict[str, Any]]:
        return [
            {"id": 1, "event": 3, "team_h": 1, "team_a": 2, "team_h_difficulty": 2,
             "team_a_difficulty": 4, "kickoff_time": "2025-08-30T14:00:00Z",
             "finished": False, "team_h_score": None, "team_a_score": None},
            {"id": 2, "event": 3, "team_h": 3, "team_a": 4, "team_h_difficulty": 3,
             "team_a_difficulty": 3, "kickoff_time": "2025-08-30T16:30:00Z",
             "finished": False, "team_h_score": None, "team_a_score": None},
        ]  # fmt: skip

    def live(self, gameweek: int) -> dict[str, Any]:
        return {
            "elements": [
                {
                    "id": pid,
                    "stats": {
                        "minutes": 90, "goals_scored": int(pid % 3 == 0), "assists": 0,
                        "clean_sheets": 0, "bonus": 0, "bps": 20 + pid,
                        "total_points": 2 + (pid + gameweek) % 6,
                    },
                }
                for pid, *_ in _PLAYERS
            ]
        }  # fmt: skip
