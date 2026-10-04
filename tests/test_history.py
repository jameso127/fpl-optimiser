from pathlib import Path

import pandas as pd
import pytest

from common.config import Settings
from common.storage import available_gameweeks, available_seasons, gw_path, read_parquet
from ingest import history

MERGED = """element,name,position,team,xP,minutes,total_points,GW,value,starts,expected_goals,bps
1,Alice,GK,Alpha,2.0,90,6,1,50,1,0.0,30
1,Alice,GK,Alpha,3.5,90,2,2,50,1,0.0,10
1,Alice,GK,Alpha,1.5,45,3,2,51,0,0.0,12
2,Bob,FWD,Beta,5.0,90,9,1,80,1,0.7,40
"""
PLAYERS_RAW = """id,web_name,first_name,second_name,team,element_type,code,status
1,Alice,Alice,A,1,1,101,a
2,Bob,Bob,B,2,4,102,a
3,Carol,Carol,C,1,2,103,i
"""
TEAMS = (
    "id,name,short_name,strength,strength_overall_home,strength_overall_away,"
    "strength_attack_home,strength_attack_away,strength_defence_home,strength_defence_away,extra\n"
    "1,Alpha,ALP,3,1100,1080,1100,1080,1100,1080,x\n"
    "2,Beta,BET,4,1200,1180,1200,1180,1200,1180,x\n"
)
FIXTURES = (
    "id,event,team_h,team_a,team_h_difficulty,team_a_difficulty,kickoff_time,finished,"
    "team_h_score,team_a_score\n"
    "1,1,1,2,2,4,2024-08-17T14:00:00Z,True,1,0\n"
)


def _fetch(path: str) -> bytes:
    files = {
        "2024-25/gws/merged_gw.csv": MERGED,
        "2024-25/players_raw.csv": PLAYERS_RAW,
        "2024-25/teams.csv": TEAMS,
        "2024-25/fixtures.csv": FIXTURES,
    }
    return files[path].encode()


def test_import_season_writes_season_layout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    assert history.import_season(settings, "2024-25", _fetch) == 2

    assert available_seasons(settings) == ["2024-25"]
    for name in ("live", "players", "teams", "fixtures"):
        assert available_gameweeks(settings, "2024-25", name) == [1, 2]


def test_double_gameweek_rows_are_summed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)
    live = read_parquet(settings, gw_path("2024-25", 2, "live")).set_index("id")

    assert live.loc[1, "total_points"] == 5
    assert live.loc[1, "minutes"] == 135
    assert live.loc[1, "starts"] == 1


def test_players_snapshot_has_all_players_and_no_leaky_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)
    gw2 = read_parquet(settings, gw_path("2024-25", 2, "players")).set_index("id")

    assert gw2.loc[3].isna()["ep_next"]  # in players_raw but no archive row
    assert gw2.loc[2, "element_type"] == 4
    assert "status" not in gw2.columns  # end-of-season status must not leak into history


def test_xp_is_stored_against_the_next_gameweek_to_avoid_leaking_the_outcome(
    tmp_path: Path,
) -> None:
    """The archive's xP on row g was captured after gameweek g, so it is the forecast for g+1."""
    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)
    gw1 = read_parquet(settings, gw_path("2024-25", 1, "players")).set_index("id")
    gw2 = read_parquet(settings, gw_path("2024-25", 2, "players")).set_index("id")

    assert gw1["ep_next"].isna().all()  # nothing was known before gameweek 1
    assert gw2.loc[1, "ep_next"] == 2.0  # archive row GW1 (xP 2.0) -> known before gameweek 2
    assert gw2.loc[2, "ep_next"] == 5.0  # the player's gameweek-1 xP, not gameweek 2's


def test_double_gameweek_xp_is_summed_and_shifted() -> None:
    merged = pd.DataFrame(
        {
            "element": [1, 1, 1],
            "GW": [1, 2, 2],
            "xP": [2.0, 3.5, 1.5],  # gameweek 2 is a double gameweek
        }
    )
    ep = history.expected_points_by_gameweek(merged).set_index("gameweek")["ep_next"]

    assert ep.loc[2] == 2.0
    assert ep.loc[3] == pytest.approx(5.0)
    assert 1 not in ep.index


def test_uncaptured_gameweek_xp_is_null_not_zero() -> None:
    merged = pd.DataFrame(
        {
            "element": [1, 2, 1, 2],
            "GW": [1, 1, 2, 2],
            "xP": [2.0, 0.0, 0.0, 0.0],  # gameweek 2: archive recorded zeros for everyone
        }
    )
    ep = history.expected_points_by_gameweek(merged)

    known_for_gw2 = ep[ep["gameweek"] == 2].set_index("id")["ep_next"]
    assert known_for_gw2.loc[1] == 2.0
    assert known_for_gw2.loc[2] == 0.0  # a genuine zero in a captured gameweek stays zero
    assert ep[ep["gameweek"] == 3]["ep_next"].isna().all()


def test_implausible_xp_is_treated_as_missing() -> None:
    merged = pd.DataFrame({"element": [1, 2], "GW": [1, 1], "xP": [4.0, 52.8]})
    ep = history.expected_points_by_gameweek(merged).set_index("id")["ep_next"]

    assert ep.loc[1] == 4.0
    assert pd.isna(ep.loc[2])


def test_managers_are_dropped_from_players_and_live(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def fetch(path: str) -> bytes:
        if path.endswith("players_raw.csv"):
            return (PLAYERS_RAW + "9,Coach,Coach,C,1,5,109,a\n").encode()
        if path.endswith("merged_gw.csv"):
            return (MERGED + "9,Coach,AM,Alpha,50.0,0,0,1,5,0,0.0,0\n").encode()
        return _fetch(path)

    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", fetch)

    assert 9 not in set(read_parquet(settings, gw_path("2024-25", 1, "players"))["id"])
    assert 9 not in set(read_parquet(settings, gw_path("2024-25", 1, "live"))["id"])


def test_load_ep_next_reads_each_snapshot_for_its_own_gameweek(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from predict.data import load_ep_next

    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)

    ep = load_ep_next(settings, "2024-25")
    assert ep[(ep["id"] == 1) & (ep["gameweek"] == 2)]["ep_next"].iloc[0] == 2.0
    assert ep[(ep["id"] == 1) & (ep["gameweek"] == 1)]["ep_next"].isna().all()
    # `upto` stops a prediction for gameweek 1 from ever seeing gameweek 2's snapshot.
    assert set(load_ep_next(settings, "2024-25", upto=1)["gameweek"]) == {1}


def test_teams_and_fixtures_keep_only_known_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)

    assert "extra" not in read_parquet(settings, gw_path("2024-25", 1, "teams")).columns
    fixtures = read_parquet(settings, gw_path("2024-25", 1, "fixtures"))
    assert list(fixtures["team_h"]) == [1]


def test_reimport_is_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=str(tmp_path))
    history.import_season(settings, "2024-25", _fetch)
    first = read_parquet(settings, gw_path("2024-25", 2, "live"))
    history.import_season(settings, "2024-25", _fetch)

    pd.testing.assert_frame_equal(first, read_parquet(settings, gw_path("2024-25", 2, "live")))


def test_read_csv_falls_back_to_latin1() -> None:
    raw = "name,x\nJos\xe9,1\n".encode("latin-1")
    assert history.read_csv(raw)["name"].iloc[0] == "José"
