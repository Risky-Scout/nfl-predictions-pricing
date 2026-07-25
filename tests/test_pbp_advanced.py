import numpy as np
import pandas as pd

from nfl_hybrid.features.pbp_advanced import (
    aggregate_advanced_team_game,
    aggregate_qb_game_efficiency,
)


def _sample_pbp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G1"] * 10,
            "play_id": list(range(1, 11)),
            "season": [2025] * 10,
            "season_type": ["REG"] * 10,
            "week": [1] * 10,
            "home_team": ["A"] * 10,
            "away_team": ["B"] * 10,
            "posteam": ["A"] * 5 + ["B"] * 5,
            "defteam": ["B"] * 5 + ["A"] * 5,
            "passer_player_id": [
                "QB_A",
                None,
                "QB_A",
                None,
                None,
                "QB_B",
                None,
                "QB_B",
                None,
                None,
            ],
            "drive": [1] * 5 + [2] * 5,
            "down": [1, 2, 3, np.nan, 1, 1, 2, 3, 4, np.nan],
            "game_seconds_remaining": [
                3500,
                3450,
                3400,
                3350,
                3300,
                3250,
                3200,
                3150,
                3100,
                3050,
            ],
            "half_seconds_remaining": [
                1700,
                1650,
                1600,
                1550,
                1500,
                1450,
                1400,
                1350,
                1300,
                1250,
            ],
            "score_differential": [0] * 10,
            "yardline_100": [75, 50, 35, 30, 25, 80, 60, 45, 30, 20],
            "goal_to_go": [0] * 10,
            "play_type": [
                "pass",
                "run",
                "pass",
                "field_goal",
                "no_play",
                "pass",
                "run",
                "pass",
                "punt",
                "no_play",
            ],
            "no_play": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "qb_kneel": [0] * 10,
            "qb_dropback": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "pass_attempt": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            "rush_attempt": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "sack": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "scramble": [0] * 10,
            "complete_pass": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "interception": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "fumble_lost": [0] * 10,
            "touchdown": [0] * 10,
            "third_down_converted": [0] * 10,
            "fourth_down_converted": [0] * 10,
            "passing_yards": [25, 0, 0, 0, 0, 12, 0, 0, 0, 0],
            "rushing_yards": [0, 6, 0, 0, 0, 0, 11, 0, 0, 0],
            "air_yards": [18, 0, 0, 0, 0, 7, 0, 9, 0, 0],
            "cpoe": [5, np.nan, np.nan, np.nan, np.nan, 2, np.nan, -15, np.nan, np.nan],
            "epa": [0.8, 0.1, -1.0, 0.4, 0, 0.3, 0.5, -2.0, 0.2, 0],
            "success": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "special_teams_play": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "penalty": [0] * 10,
            "no_huddle": [0] * 10,
            "shotgun": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "posteam_score": [0] * 10,
            "posteam_score_post": [0] * 10,
        }
    )


def test_advanced_team_game_has_two_rows():
    result = aggregate_advanced_team_game(_sample_pbp())

    assert len(result) == 2
    assert result["team_id"].nunique() == 2
    assert result.groupby("game_id")["team_id"].nunique().eq(2).all()


def test_scrimmage_and_special_teams_are_separate():
    result = aggregate_advanced_team_game(_sample_pbp())
    team_a = result[result["team_id"] == "A"].iloc[0]

    assert team_a["offense_scrimmage_plays"] == 3
    assert team_a["special_teams_plays"] == 1
    assert team_a["offense_pass_attempts"] == 1
    assert team_a["offense_dropbacks"] == 2
    assert team_a["offense_explosive_pass_rate"] == 1.0


def test_defensive_mirror_uses_opponent_offense():
    result = aggregate_advanced_team_game(_sample_pbp())
    team_a = result[result["team_id"] == "A"].iloc[0]

    assert team_a["defense_allowed_turnovers"] == 1
    assert team_a["defense_allowed_scrimmage_plays"] == 3


def test_qb_game_aggregation():
    result = aggregate_qb_game_efficiency(_sample_pbp())

    assert len(result) == 2

    qb_a = result[result["player_id"] == "QB_A"].iloc[0]
    qb_b = result[result["player_id"] == "QB_B"].iloc[0]

    assert qb_a["dropbacks"] == 2
    assert qb_a["attempts"] == 1
    assert qb_b["interceptions"] == 1
