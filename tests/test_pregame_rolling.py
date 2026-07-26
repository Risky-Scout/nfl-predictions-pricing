import numpy as np
import pandas as pd

from nfl_hybrid.features.pregame_rolling import (
    build_game_pregame_matrix,
    build_team_pregame_features,
)


def _games():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "gameday": [
                "2025-09-01",
                "2025-09-08",
                "2025-09-15",
            ],
            "home_team_id": ["KC", "BUF", "KC"],
            "away_team_id": ["BUF", "KC", "BUF"],
        }
    )


def _team_games():
    return pd.DataFrame(
        {
            "game_id": [
                "G1", "G1",
                "G2", "G2",
                "G3", "G3",
            ],
            "season": [2025] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team_id": ["KC", "BUF", "BUF", "KC", "KC", "BUF"],
            "opponent_id": ["BUF", "KC", "KC", "BUF", "BUF", "KC"],
            "offense_epa_per_play": [
                0.10, -0.10,
                0.20, -0.20,
                0.30, -0.30,
            ],
            "defense_allowed_epa_per_play": [
                -0.10, 0.10,
                -0.20, 0.20,
                -0.30, 0.30,
            ],
        }
    )


def test_first_game_has_no_current_game_information():
    result = build_team_pregame_features(
        _team_games(),
        _games(),
    )

    first_a = result[
        (result["game_id"] == "G1")
        & (result["team_id"] == "KC")
    ].iloc[0]

    assert first_a["team_prior_games"] == 0
    assert np.isnan(
        first_a[
            "offense_epa_per_play__all_mean"
        ]
    )


def test_second_game_uses_only_first_game():
    result = build_team_pregame_features(
        _team_games(),
        _games(),
    )

    second_a = result[
        (result["game_id"] == "G2")
        & (result["team_id"] == "KC")
    ].iloc[0]

    assert second_a["team_prior_games"] == 1
    assert np.isclose(
        second_a[
            "offense_epa_per_play__all_mean"
        ],
        0.10,
    )
    assert np.isclose(
        second_a[
            "offense_epa_per_play__last4_mean"
        ],
        0.10,
    )


def test_current_game_perturbation_does_not_change_current_features():
    original = _team_games()
    modified = original.copy()

    modified.loc[
        (modified["game_id"] == "G2")
        & (modified["team_id"] == "KC"),
        "offense_epa_per_play",
    ] = 100.0

    original_features = build_team_pregame_features(
        original,
        _games(),
    )

    modified_features = build_team_pregame_features(
        modified,
        _games(),
    )

    column = "offense_epa_per_play__all_mean"

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

    assert np.isclose(original_g2, modified_g2)
    assert not np.isclose(original_g3, modified_g3)


def test_game_matrix_has_one_row_per_game():
    team_features = build_team_pregame_features(
        _team_games(),
        _games(),
    )

    matrix = build_game_pregame_matrix(
        _games(),
        team_features,
    )

    assert len(matrix) == 3
    assert matrix["game_id"].nunique() == 3
    assert {
        "home_offense_epa_per_play__all_mean",
        "away_offense_epa_per_play__all_mean",
    }.issubset(matrix.columns)


def test_game_matrix_normalizes_rams_provider_alias():
    games = pd.DataFrame(
        {
            "game_id": ["G_RAMS"],
            "home_team_id": ["LAR"],
            "away_team_id": ["DAL"],
        }
    )

    team_features = pd.DataFrame(
        {
            "game_id": ["G_RAMS", "G_RAMS"],
            "team_id": ["LA", "DAL"],
            "team_prior_games": [5, 5],
            "offense_epa_per_play__all_mean": [
                0.15,
                0.05,
            ],
        }
    )

    matrix = build_game_pregame_matrix(
        games,
        team_features,
    )

    assert len(matrix) == 1

    assert np.isclose(
        matrix.loc[
            0,
            "home_offense_epa_per_play__all_mean",
        ],
        0.15,
    )

    assert np.isclose(
        matrix.loc[
            0,
            "away_offense_epa_per_play__all_mean",
        ],
        0.05,
    )
