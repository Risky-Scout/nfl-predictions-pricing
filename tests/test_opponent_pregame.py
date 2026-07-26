import numpy as np
import pandas as pd

from nfl_hybrid.features.opponent_pregame import (
    OpponentAdjustedConfig,
    StrengthMetricSpec,
    build_game_opponent_adjusted_matrix,
    build_opponent_adjusted_team_features,
)


SPEC = (
    StrengthMetricSpec(
        metric="offense_epa_per_play",
        exposure="offense_scrimmage_plays",
        prior_exposure=100.0,
        prior_mean=0.0,
        prior_sd=0.20,
    ),
)

CONFIG = OpponentAdjustedConfig(
    ridge_alpha=1.0,
    season_decay=1.0,
    max_history_seasons=2,
    minimum_history_rows=1,
)


def _games():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4"],
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "gameday": [
                "2025-09-01",
                "2025-09-08",
                "2025-09-15",
                "2025-09-22",
            ],
            "home_team_id": [
                "KC",
                "BUF",
                "KC",
                "BUF",
            ],
            "away_team_id": [
                "BUF",
                "KC",
                "BUF",
                "KC",
            ],
        }
    )


def _team_games():
    values = {
        "G1": {"KC": 0.20, "BUF": -0.20},
        "G2": {"KC": 0.30, "BUF": -0.30},
        "G3": {"KC": 0.40, "BUF": -0.40},
        "G4": {"KC": 0.50, "BUF": -0.50},
    }

    rows = []

    for game_id, teams in values.items():
        for team, value in teams.items():
            opponent = "BUF" if team == "KC" else "KC"

            rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": int(game_id[-1]),
                    "team_id": team,
                    "opponent_id": opponent,
                    "offense_epa_per_play": value,
                    "offense_scrimmage_plays": 60,
                }
            )

    return pd.DataFrame(rows)


def test_first_game_uses_only_prior_distribution():
    result = build_opponent_adjusted_team_features(
        _team_games(),
        _games(),
        metric_specs=SPEC,
        config=CONFIG,
    )

    first = result[
        (result["game_id"] == "G1")
        & (result["team_id"] == "KC")
    ].iloc[0]

    assert first["oa_prior_offense_games"] == 0
    assert first[
        "oa_epa_per_play_offense_reliability"
    ] == 0
    assert np.isclose(
        first["oa_epa_per_play_offense_mean"],
        0.0,
    )


def test_current_game_cannot_change_its_own_rating():
    original = _team_games()
    modified = original.copy()

    modified.loc[
        (modified["game_id"] == "G2")
        & (modified["team_id"] == "KC"),
        "offense_epa_per_play",
    ] = 50.0

    original_features = (
        build_opponent_adjusted_team_features(
            original,
            _games(),
            metric_specs=SPEC,
            config=CONFIG,
        )
    )

    modified_features = (
        build_opponent_adjusted_team_features(
            modified,
            _games(),
            metric_specs=SPEC,
            config=CONFIG,
        )
    )

    column = "oa_epa_per_play_offense_mean"

    original_g2 = original_features.loc[
        (original_features["game_id"] == "G2")
        & (original_features["team_id"] == "KC"),
        column,
    ].iloc[0]

    modified_g2 = modified_features.loc[
        (modified_features["game_id"] == "G2")
        & (modified_features["team_id"] == "KC"),
        column,
    ].iloc[0]

    original_g3 = original_features.loc[
        (original_features["game_id"] == "G3")
        & (original_features["team_id"] == "KC"),
        column,
    ].iloc[0]

    modified_g3 = modified_features.loc[
        (modified_features["game_id"] == "G3")
        & (modified_features["team_id"] == "KC"),
        column,
    ].iloc[0]

    assert np.isclose(
        original_g2,
        modified_g2,
    )

    assert not np.isclose(
        original_g3,
        modified_g3,
    )


def test_stronger_offense_receives_higher_effect():
    result = build_opponent_adjusted_team_features(
        _team_games(),
        _games(),
        metric_specs=SPEC,
        config=CONFIG,
    )

    game_four = result[
        result["game_id"] == "G4"
    ].set_index("team_id")

    assert (
        game_four.loc[
            "KC",
            "oa_epa_per_play_offense_effect",
        ]
        >
        game_four.loc[
            "BUF",
            "oa_epa_per_play_offense_effect",
        ]
    )


def test_game_strength_matrix_has_one_row_per_game():
    team_strength = (
        build_opponent_adjusted_team_features(
            _team_games(),
            _games(),
            metric_specs=SPEC,
            config=CONFIG,
        )
    )

    matrix = build_game_opponent_adjusted_matrix(
        _games(),
        team_strength,
    )

    assert len(matrix) == 4
    assert matrix["game_id"].nunique() == 4
    assert {
        "home_oa_epa_per_play_offense_mean",
        "away_oa_epa_per_play_offense_mean",
    }.issubset(matrix.columns)
