import pandas as pd
import pytest

from predict import features


def _build(d: dict[str, pd.DataFrame], live: pd.DataFrame | None = None) -> pd.DataFrame:
    return features.build_features(
        d["live"] if live is None else live,
        d["players"], d["teams"], d["fixtures"], list(range(1, 10)),
    )  # fmt: skip


def test_rolling_features_use_only_prior_gameweeks(synthetic: dict[str, pd.DataFrame]) -> None:
    feats = _build(synthetic)
    live = synthetic["live"]
    row = feats[(feats["id"] == 1) & (feats["gameweek"] == 6)].iloc[0]
    pts = live[live["id"] == 1].set_index("gameweek")["total_points"]

    assert row["pts_l3"] == pytest.approx(pts.loc[3:5].mean())
    assert row["pts_l5"] == pytest.approx(pts.loc[1:5].mean())
    assert row["pts_season"] == pytest.approx(pts.loc[1:5].mean())
    assert row["games_played"] == 5


def test_changing_gameweek_g_and_later_does_not_change_features_for_g(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """The leakage guard: features for gameweek 5 must ignore gameweeks 5, 6, 7, 8."""
    base = _build(synthetic)
    altered_live = synthetic["live"].copy()
    late = altered_live["gameweek"] >= 5
    altered_live.loc[late, ["total_points", "minutes", "bps", "ict_index"]] = 99.0
    altered = _build(synthetic, altered_live)

    cols = features.FEATURES
    pd.testing.assert_frame_equal(
        base[base["gameweek"] == 5][cols].reset_index(drop=True),
        altered[altered["gameweek"] == 5][cols].reset_index(drop=True),
    )
    # ...while gameweek 6 features legitimately do see the changed gameweek 5.
    assert not base[base["gameweek"] == 6]["pts_l3"].equals(
        altered[altered["gameweek"] == 6]["pts_l3"]
    )


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
