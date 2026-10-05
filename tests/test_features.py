import pandas as pd
import pytest

from ml import features


def _build(
    d: dict[str, pd.DataFrame],
    live: pd.DataFrame | None = None,
    ep: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return features.build_features(
        d["live"] if live is None else live,
        d["players"], d["teams"], d["fixtures"], list(range(1, 10)),
        d["ep"] if ep is None else ep,
    )  # fmt: skip


def test_ep_next_comes_only_from_the_same_gameweek(synthetic: dict[str, pd.DataFrame]) -> None:
    ep = synthetic["ep"].assign(ep_next=lambda df: df["gameweek"].astype(float))
    feats = _build(synthetic, ep=ep)

    # Row for gameweek g carries exactly the value published for g: not g-1, not g+1.
    assert (feats["ep_next"] == feats["gameweek"]).all()


def test_changing_later_ep_next_does_not_change_earlier_rows(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    base = _build(synthetic)
    ep = synthetic["ep"].copy()
    ep.loc[ep["gameweek"] >= 6, "ep_next"] = 99.0
    altered = _build(synthetic, ep=ep)

    before = base["gameweek"] <= 5
    pd.testing.assert_frame_equal(base[before], altered[altered["gameweek"] <= 5])
    assert (altered.loc[altered["gameweek"] >= 6, "ep_next"] == 99.0).all()


def test_ep_next_never_leaks_into_form_features(synthetic: dict[str, pd.DataFrame]) -> None:
    base = _build(synthetic)
    altered = _build(synthetic, ep=synthetic["ep"].assign(ep_next=-7.0))

    pd.testing.assert_frame_equal(base[features.FEATURES], altered[features.FEATURES])


def test_gameweeks_without_a_stored_snapshot_have_null_ep_next(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    ep = synthetic["ep"][synthetic["ep"]["gameweek"] == 9]
    feats = _build(synthetic, ep=ep)

    assert feats.loc[feats["gameweek"] < 9, "ep_next"].isna().all()
    assert feats.loc[feats["gameweek"] == 9, "ep_next"].notna().all()
    assert (
        _build(synthetic, ep=pd.DataFrame(columns=["id", "gameweek", "ep_next"]))["ep_next"]
        .isna()
        .all()
    )


def test_rolling_features_use_only_prior_gameweeks(synthetic: dict[str, pd.DataFrame]) -> None:
    feats = _build(synthetic)
    live = synthetic["live"]
    row = feats[(feats["id"] == 1) & (feats["gameweek"] == 6)].iloc[0]
    pts = live[live["id"] == 1].set_index("gameweek")["total_points"]

    assert row["pts_l3"] == pytest.approx(pts.loc[3:5].mean())
    assert row["pts_l5"] == pytest.approx(pts.loc[1:5].mean())
    assert row["pts_season"] == pytest.approx(pts.loc[1:5].mean())
    assert row["games_played"] == 5


def _tamper_from_gameweek_5(d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Wreck every outcome from gameweek 5 on: player stats, goals scored, xG."""
    live = d["live"].copy()
    late = live["gameweek"] >= 5
    stat_cols = [c for c in live.columns if c not in ("id", "gameweek")]
    live.loc[late, stat_cols] = 99.0
    fixtures = d["fixtures"].copy()
    scored = fixtures["event"] >= 5
    fixtures.loc[scored, ["team_h_score", "team_a_score"]] = 9.0
    return {**d, "live": live, "fixtures": fixtures}


def test_changing_gameweek_g_and_later_does_not_change_features_for_g(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """The leakage guard: features for gameweek 5 must ignore gameweeks 5, 6, 7, 8.

    Covers every feature the current model uses: player form, per-90 rates, team and
    opponent form (which come from fixture results and team xG), rest days and prior season.
    """
    base = _build(synthetic)
    tampered = _tamper_from_gameweek_5(synthetic)
    altered = _build(tampered)

    cols = features.FEATURES_V2
    pd.testing.assert_frame_equal(
        base[base["gameweek"] == 5][cols].reset_index(drop=True),
        altered[altered["gameweek"] == 5][cols].reset_index(drop=True),
    )
    # ...while gameweek 6 features legitimately do see the changed gameweek 5, in both the
    # player and the team features.
    for col in ("pts_l3", "t_gf_l5", "t_xgf_l5", "o_gf_l5"):
        assert not base[base["gameweek"] == 6][col].equals(altered[altered["gameweek"] == 6][col])


def test_per90_and_detail_features_are_computed_from_prior_gameweeks_only(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    feats = _build(synthetic)
    live = synthetic["live"]
    row = feats[(feats["id"] == 1) & (feats["gameweek"] == 6)].iloc[0]
    prior = live[(live["id"] == 1) & live["gameweek"].between(1, 5)]

    assert row["pts90_l5"] == pytest.approx(
        prior["total_points"].sum() / prior["minutes"].sum() * 90
    )
    assert (
        row["pts_l1"] == live[(live["id"] == 1) & (live["gameweek"] == 5)]["total_points"].iloc[0]
    )
    assert (
        row["played_l5"] == 1.0
        and row["full_l5"]
        == feats.loc[(feats["id"] == 1) & (feats["gameweek"] == 6), "full_l5"].iloc[0]
    )


def test_rest_days_use_the_previous_match_for_the_team_and_its_opponent() -> None:
    fixtures = pd.DataFrame(
        [
            {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 3,
             "team_a_difficulty": 3, "kickoff_time": "2024-08-10T14:00:00Z"},
            {"id": 2, "event": 2, "team_h": 3, "team_a": 1, "team_h_difficulty": 3,
             "team_a_difficulty": 3, "kickoff_time": "2024-08-13T14:00:00Z"},
            {"id": 3, "event": 3, "team_h": 2, "team_a": 1, "team_h_difficulty": 3,
             "team_a_difficulty": 3, "kickoff_time": "2024-08-30T14:00:00Z"},
        ]
    )  # fmt: skip
    rest = features.rest_days(fixtures).set_index(["team", "gameweek"])

    assert pd.isna(rest.loc[(1, 1), "rest_days"])  # season opener
    assert rest.loc[(1, 2), "rest_days"] == 3.0
    assert rest.loc[(1, 3), "rest_days"] == 14.0  # capped (17 days)
    # Team 3's opponent in gameweek 2 is team 1, who had 3 days; team 2 had 20 in gameweek 3.
    assert rest.loc[(3, 2), "opp_rest_days"] == 3.0
    assert rest.loc[(1, 3), "opp_rest_days"] == 14.0  # team 2 last played on 10 Aug (20 days)


def test_prior_season_rates_are_keyed_by_code_and_need_enough_minutes() -> None:
    live = pd.DataFrame(
        {
            "id": [1] * 10 + [2] * 10,
            "gameweek": list(range(1, 11)) * 2,
            "minutes": [90] * 10 + [10] * 10,
            "total_points": [4] * 10 + [1] * 10,
            "expected_goal_involvements": [0.5] * 10 + [0.0] * 10,
            "starts": [1] * 10 + [0] * 10,
        }
    )
    players = pd.DataFrame({"id": [1, 2], "code": [501, 502]})
    prior = features.prior_season_rates(live, players).set_index("code")

    assert prior.loc[501, "prev_pts90"] == pytest.approx(4.0)  # 40 points over 900 minutes
    assert prior.loc[501, "prev_min"] == pytest.approx(90.0)
    assert prior.loc[501, "prev_starts"] == 10
    assert pd.isna(prior.loc[502, "prev_pts90"])  # 100 minutes is too few for a rate


def test_prior_season_features_attach_by_code_and_stay_constant(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    prior = pd.DataFrame(
        {
            "code": [1001, 1002], "prev_pts90": [5.0, 3.0], "prev_min": [80.0, 60.0],
            "prev_games": [30, 20], "prev_xgi90": [0.4, 0.2], "prev_starts": [28, 15],
        }
    )  # fmt: skip
    feats = features.build_features(
        synthetic["live"], synthetic["players"], synthetic["teams"], synthetic["fixtures"],
        list(range(1, 10)), None, prior,
    )  # fmt: skip

    p1 = feats[feats["id"] == 1]
    assert (p1["prev_pts90"] == 5.0).all()  # the same in every gameweek of the season
    assert feats[feats["id"] == 3]["prev_pts90"].isna().all()  # no code match: no prior season


def test_previous_season_label() -> None:
    assert features.previous_season("2025-26") == "2024-25"
    assert features.previous_season("2000-01") == "1999-00"


def test_upcoming_gameweek_has_no_target_but_has_fixture_features(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    feats = _build(synthetic)
    upcoming = feats[feats["gameweek"] == 9]

    assert upcoming["target"].isna().all()
    assert (upcoming["fix_n"] == 1).all()
    assert upcoming["pts_l5"].notna().all()


def test_double_gameweek_counts_fixtures_and_blank_is_zero(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    extra = pd.DataFrame(
        [{"event": 9, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}]
    )
    synthetic["fixtures"] = pd.concat([synthetic["fixtures"], extra], ignore_index=True)
    synthetic["fixtures"] = synthetic["fixtures"][
        ~((synthetic["fixtures"]["event"] == 9) & (synthetic["fixtures"]["team_h"] == 3))
    ]
    feats = _build(synthetic)
    gw9 = feats[feats["gameweek"] == 9].set_index("id")
    team = synthetic["players"].set_index("id")["team"]

    assert (gw9[team == 1]["fix_n"] == 2).all()  # double gameweek
    assert (gw9[team == 3]["fix_n"] == 0).all()  # blank
    assert (
        features.usable(feats[feats["gameweek"] == 9])["id"].isin(team[team == 3].index).sum() == 0
    )


def test_venue_specific_strength_is_used() -> None:
    teams = pd.DataFrame(
        {
            "id": [1, 2],
            "strength_attack_home": [1, 2], "strength_attack_away": [10, 20],
            "strength_defence_home": [100, 200], "strength_defence_away": [1000, 2000],
        }
    )  # fmt: skip
    fixtures = pd.DataFrame(
        [{"event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4}]
    )
    ff = features.fixture_features(fixtures, teams).set_index("team")

    # Home side 1: own at home, opponent (team 2) away.
    assert (ff.loc[1, "own_att"], ff.loc[1, "opp_att"]) == (1, 20)
    assert (ff.loc[1, "own_def"], ff.loc[1, "opp_def"]) == (100, 2000)
    # Away side 2: own away, opponent (team 1) at home.
    assert (ff.loc[2, "own_att"], ff.loc[2, "opp_att"]) == (20, 1)
    assert ff.loc[2, "fix_difficulty"] == 4
