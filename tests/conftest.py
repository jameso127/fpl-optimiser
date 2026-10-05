import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic() -> dict[str, pd.DataFrame]:
    """40 players over 4 teams, 8 finished gameweeks plus a 9th upcoming one."""
    rng = np.random.default_rng(0)
    n_players, n_gws = 40, 8
    players = pd.DataFrame(
        {
            "id": range(1, n_players + 1),
            "code": [1000 + i for i in range(1, n_players + 1)],
            "element_type": [1 + i % 4 for i in range(n_players)],
            "team": [1 + i % 4 for i in range(n_players)],
            "status": "a",
            "chance_of_playing_next_round": np.nan,
            "ep_next": rng.uniform(1, 6, n_players),
        }
    )
    skill = rng.uniform(0.5, 6, n_players)
    live = pd.DataFrame(
        [
            {
                "id": pid, "gameweek": gw,
                "total_points": float(rng.poisson(skill[pid - 1])),
                "minutes": 90 if skill[pid - 1] > 2 else 20,
                "starts": 1, "bps": 20, "ict_index": 5.0, "expected_goals": 0.2,
                "expected_goal_involvements": 0.3, "expected_goals_conceded": 1.0,
            }
            for pid in players["id"]
            for gw in range(1, n_gws + 1)
        ]
    )  # fmt: skip
    teams = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            **{
                f"strength_{k}_{v}": [1100, 1150, 1200, 1250]
                for k in ("attack", "defence")
                for v in ("home", "away")
            },
        }
    )
    base = pd.Timestamp("2024-08-17T14:00:00Z")
    fixtures = pd.DataFrame(
        [
            {
                "id": gw * 10 + k, "event": gw, "team_h": h, "team_a": a,
                "team_h_difficulty": 2 + h % 3, "team_a_difficulty": 2 + a % 3,
                "kickoff_time": (base + pd.Timedelta(days=7 * (gw - 1) + k)).isoformat(),
                "team_h_score": float(rng.integers(0, 4)) if gw <= n_gws else np.nan,
                "team_a_score": float(rng.integers(0, 4)) if gw <= n_gws else np.nan,
            }
            for gw in range(1, n_gws + 2)
            for k, (h, a) in enumerate(((1, 2), (3, 4)))
        ]
    )  # fmt: skip

    # FPL-style expected points: the player's true underlying rate for every gameweek (a good
    # but imperfect forecast, since realised points are Poisson draws around it). Never derived
    # from the realised points themselves.
    ep = pd.DataFrame(
        [
            {"id": pid, "gameweek": gw, "ep_next": float(skill[pid - 1])}
            for pid in players["id"]
            for gw in range(1, n_gws + 2)
        ]
    )
    return {"players": players, "live": live, "teams": teams, "fixtures": fixtures, "ep": ep}
