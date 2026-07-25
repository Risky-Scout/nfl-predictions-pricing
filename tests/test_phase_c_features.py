import numpy as np
import pandas as pd

from nfl_hybrid.features.opponent_adjustment import fit_opponent_adjusted_ratings
from nfl_hybrid.features.roster_continuity import build_roster_continuity
from nfl_hybrid.features.team_game import aggregate_team_game_efficiency
from nfl_hybrid.priors.coach import HierarchicalCoachPrior


def _tiny_pbp():
    return pd.DataFrame(
        {
            "game_id": ["G"] * 8,
            "season": [2025] * 8,
            "week": [1] * 8,
            "posteam": ["A"] * 4 + ["B"] * 4,
            "defteam": ["B"] * 4 + ["A"] * 4,
            "drive": [1, 1, 1, 1, 2, 2, 2, 2],
            "down": [1, 2, 3, 1, 1, 2, 3, 4],
            "ydstogo": [10, 5, 2, 10, 10, 4, 1, 1],
            "yardline_100": [75, 50, 18, 5, 80, 60, 25, 10],
            "goal_to_go": [0, 0, 0, 1, 0, 0, 0, 1],
            "game_seconds_remaining": [3500, 3450, 3400, 3350, 3300, 3250, 3200, 3150],
            "score_differential": [0] * 8,
            "qb_dropback": [1, 0, 1, 1, 0, 1, 0, 1],
            "pass_attempt": [1, 0, 1, 1, 0, 1, 0, 1],
            "rush_attempt": [0, 1, 0, 0, 1, 0, 1, 0],
            "sack": [0, 0, 0, 0, 0, 1, 0, 0],
            "interception": [0, 0, 0, 0, 0, 0, 0, 1],
            "fumble_lost": [0] * 8,
            "touchdown": [0, 0, 0, 1, 0, 0, 0, 0],
            "third_down_converted": [0, 0, 1, 0, 0, 0, 0, 0],
            "fourth_down_converted": [0, 0, 0, 0, 0, 0, 0, 1],
            "passing_yards": [12, 0, 22, 5, 0, -7, 0, 0],
            "rushing_yards": [0, 6, 0, 0, 12, 0, 2, 0],
            "epa": [0.2, 0.1, 0.6, 1.5, 0.3, -1.0, -0.1, -2.0],
            "success": [1, 1, 1, 1, 1, 0, 0, 0],
            "penalty": [0] * 8,
            "no_huddle": [0, 1, 0, 0, 0, 0, 0, 0],
            "shotgun": [1, 0, 1, 1, 0, 1, 0, 1],
            "posteam_score": [0, 0, 0, 0, 0, 0, 0, 0],
            "posteam_score_post": [0, 0, 0, 7, 0, 0, 0, 0],
        }
    )


def test_team_game_aggregation_has_two_sides_and_defense():
    result = aggregate_team_game_efficiency(_tiny_pbp())
    assert len(result) == 2
    assert {
        "offense_epa_per_play",
        "defense_allowed_epa_per_play",
        "offense_points_per_drive",
    }.issubset(result.columns)
    row_a = result[result["team_id"] == "A"].iloc[0]
    assert row_a["offense_points_per_drive"] == 7.0
    assert row_a["defense_allowed_turnovers"] == 1


def test_opponent_adjustment_returns_team_effects():
    rows = []
    teams = ["A", "B", "C", "D"]
    for repeat in range(4):
        for i, team in enumerate(teams):
            opponent = teams[(i + 1 + repeat) % len(teams)]
            rows.append(
                {
                    "season": 2025,
                    "team_id": team,
                    "opponent_id": opponent,
                    "offense_epa_per_play": 0.10 * i - 0.02 * repeat,
                }
            )
    ratings = fit_opponent_adjusted_ratings(
        pd.DataFrame(rows),
        metric_columns=["offense_epa_per_play"],
    )
    assert len(ratings) == 4
    assert ratings["team_effect"].notna().all()


def test_roster_continuity_same_team_share():
    snaps = pd.DataFrame(
        {
            "team_id": ["A", "A", "B"],
            "player_id": ["P1", "P2", "P3"],
            "position_group": ["OL", "WR", "DL"],
            "offensive_snaps": [700, 300, 0],
            "defensive_snaps": [0, 0, 800],
            "special_teams_snaps": [20, 10, 30],
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": ["A", "B", "A"],
            "player_id": ["P1", "P2", "P3"],
        }
    )
    result = build_roster_continuity(snaps, roster).set_index("team_id")
    assert np.isclose(result.loc["A", "returning_offensive_snap_percentage"], 0.7)
    assert np.isclose(result.loc["B", "returning_defensive_snap_percentage"], 0.0)


def test_new_coach_gets_league_prior_and_more_uncertainty():
    history = pd.DataFrame(
        {
            "coach_id": ["C1", "C2", "C1", "C2"],
            "unit": ["offense"] * 4,
            "residual_value": [0.10, -0.10, 0.08, -0.12],
            "games": [16, 16, 16, 16],
        }
    )
    targets = pd.DataFrame(
        {"coach_id": ["C1", "NEW"], "unit": ["offense", "offense"]}
    )
    priors = HierarchicalCoachPrior().build(history, target_coaches=targets)
    indexed = priors.set_index("coach_id")
    assert indexed.loc["NEW", "regression_weight"] == 0
    assert (
        indexed.loc["NEW", "prior_standard_deviation"]
        >= indexed.loc["C1", "prior_standard_deviation"]
    )
