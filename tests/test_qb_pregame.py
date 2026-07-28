
import numpy as np
import pandas as pd

from nfl_hybrid.features.qb_pregame import (
    QBPregameConfig,
    QBMetricSpec,
    actual_starter_candidates,
    build_game_qb_matrix,
    build_qb_pregame_team_features,
)


SPECS = (
    QBMetricSpec(
        "epa_per_dropback",
        "dropbacks",
        0.0,
        0.25,
        prior_dropbacks=50.0,
    ),
    QBMetricSpec(
        "success_rate",
        "dropbacks",
        0.45,
        0.10,
        prior_dropbacks=50.0,
        lower_bound=0.0,
        upper_bound=1.0,
    ),
)

CONFIG = QBPregameConfig(
    half_life_dropbacks=400.0,
    max_dropbacks=1000.0,
    max_history_seasons=2,
    season_decay=1.0,
)


def _games():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2025, 2025, 2025],
            "gameday": [
                "2025-09-01",
                "2025-09-08",
                "2025-09-15",
            ],
            "home_team_id": ["KC", "BUF", "KC"],
            "away_team_id": ["BUF", "KC", "BUF"],
            "home_qb_id": ["Q1", "Q2", "Q1"],
            "away_qb_id": ["Q2", "Q1", "Q2"],
        }
    )


def _qb_games():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "team_id": ["KC", "BUF", "BUF", "KC", "KC", "BUF"],
            "player_id": ["Q1", "Q2", "Q2", "Q1", "Q1", "Q2"],
            "season": [2025] * 6,
            "dropbacks": [30, 30, 32, 32, 35, 35],
            "attempts": [28, 28, 30, 30, 33, 33],
            "epa_per_dropback": [0.20, -0.20, -0.10, 0.30, 0.40, -0.30],
            "success_rate": [0.55, 0.35, 0.40, 0.60, 0.62, 0.33],
        }
    )


def test_actual_candidates_has_two_rows_per_game():
    candidates = actual_starter_candidates(_games())
    assert len(candidates) == 6
    assert (
        candidates.groupby("game_id")["team_id"]
        .nunique()
        .eq(2)
        .all()
    )
    assert candidates["starter_probability"].eq(1.0).all()


def test_first_game_uses_prior_only():
    result = build_qb_pregame_team_features(
        _qb_games(),
        _games(),
        metric_specs=SPECS,
        config=CONFIG,
    )
    first = result[
        (result["game_id"] == "G1")
        & (result["team_id"] == "KC")
    ].iloc[0]
    assert first["qb_epa_per_dropback_reliability"] == 0.0
    assert np.isclose(
        first["qb_epa_per_dropback_mean"],
        0.0,
    )


def test_current_game_perturbation_does_not_change_current_prior():
    original = _qb_games()
    modified = original.copy()
    modified.loc[
        (modified["game_id"] == "G2")
        & (modified["team_id"] == "KC"),
        "epa_per_dropback",
    ] = 50.0

    original_features = build_qb_pregame_team_features(
        original,
        _games(),
        metric_specs=SPECS,
        config=CONFIG,
    )
    modified_features = build_qb_pregame_team_features(
        modified,
        _games(),
        metric_specs=SPECS,
        config=CONFIG,
    )

    column = "qb_epa_per_dropback_mean"
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


def test_starter_mixture_adds_uncertainty():
    candidates = actual_starter_candidates(_games())
    extra = pd.DataFrame(
        {
            "game_id": ["G3", "G3"],
            "season": [2025, 2025],
            "team_id": ["KC", "KC"],
            "player_id": ["Q1", "Q3"],
            "starter_probability": [0.5, 0.5],
            "starter_source": ["test", "test"],
        }
    )
    candidates = candidates.loc[
        ~(
            candidates["game_id"].eq("G3")
            & candidates["team_id"].eq("KC")
        )
    ]
    candidates = pd.concat([candidates, extra], ignore_index=True)

    mixture = build_qb_pregame_team_features(
        _qb_games(),
        _games(),
        starter_candidates=candidates,
        metric_specs=SPECS,
        config=CONFIG,
    )
    single = build_qb_pregame_team_features(
        _qb_games(),
        _games(),
        metric_specs=SPECS,
        config=CONFIG,
    )

    mix_row = mixture[
        (mixture["game_id"] == "G3")
        & (mixture["team_id"] == "KC")
    ].iloc[0]
    single_row = single[
        (single["game_id"] == "G3")
        & (single["team_id"] == "KC")
    ].iloc[0]

    assert mix_row["qb_starter_candidates"] == 2
    assert mix_row["qb_starter_entropy"] > 0
    assert (
        mix_row["qb_epa_per_dropback_sd"]
        >= single_row["qb_epa_per_dropback_sd"]
    )


def test_game_qb_matrix_has_one_row_per_game():
    team_features = build_qb_pregame_team_features(
        _qb_games(),
        _games(),
        metric_specs=SPECS,
        config=CONFIG,
    )
    matrix = build_game_qb_matrix(_games(), team_features)
    assert len(matrix) == 3
    assert matrix["game_id"].nunique() == 3
    assert {
        "home_qb_epa_per_dropback_mean",
        "away_qb_epa_per_dropback_mean",
    }.issubset(matrix.columns)
