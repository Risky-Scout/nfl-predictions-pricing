from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.unified_development_tournament import (
    UnifiedTournamentConfig,
    _feature_sets,
    _inner_split,
    _load_warehouse_metadata,
    _market_and_targets,
    _three_way_scores,
    fit_offset_logistic,
)


def test_three_way_scores_are_finite_and_proper():
    log_loss, brier = _three_way_scores(
        np.array([0.60, 0.20]),
        np.array([0.38, 0.75]),
        np.array([0.02, 0.05]),
        np.array([0, 1]),
        1e-9,
    )
    assert np.isfinite(log_loss).all()
    assert np.isfinite(brier).all()
    assert log_loss[0] == pytest.approx(-np.log(0.60))


def test_offset_logistic_can_learn_residual_signal():
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(size=n)
    market = np.full(n, 0.50)
    probability = 1.0 / (1.0 + np.exp(-1.5 * x))
    y = rng.binomial(1, probability)

    frame = pd.DataFrame({"signal": x})
    model = fit_offset_logistic(
        frame,
        ["signal"],
        y,
        market,
        regularization=2.0,
        clip=1e-9,
    )
    prediction = model.predict(frame, market, 1e-9)

    assert model.coefficients[0] > 0
    assert np.corrcoef(prediction, probability)[0, 1] > 0.9


def test_constant_features_are_removed():
    frame = pd.DataFrame(
        {
            "constant": [1.0] * 20,
            "signal": np.linspace(-1, 1, 20),
        }
    )
    y = np.array([0] * 10 + [1] * 10)
    market = np.full(20, 0.5)

    model = fit_offset_logistic(
        frame,
        ["constant", "signal"],
        y,
        market,
        regularization=2.0,
        clip=1e-9,
    )

    assert model.active_columns == ["signal"]


def test_inner_split_is_chronological_by_season():
    frame = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "week": [1, 2, 1, 2],
            "game_id": ["a", "b", "c", "d"],
        }
    )
    train, validation = _inner_split(frame)
    assert set(train["season"]) == {2020}
    assert set(validation["season"]) == {2021}


def test_feature_sets_exclude_canonical_anchor_from_residual_x():
    manifest = {
        "features": [
            "football_feature",
            "market_t10_novig_probability",
            "market_t10_consensus_line",
            "market_probability_movement",
            "market_t10_probability_sd",
        ]
    }
    sets = _feature_sets(manifest)
    assert sets["football_only"] == ["football_feature"]
    assert "market_t10_novig_probability" not in sets["all_canonical"]
    assert "market_probability_movement" in sets["movement_only"]


def test_warehouse_metadata_loader_does_not_read_outcomes(tmp_path):
    path = tmp_path / "warehouse.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1"],
            "scheduled_kickoff_utc": ["2023-09-10T17:00:00Z"],
            "home_score": [99],
            "away_score": [0],
            "target_home_win": [1],
        }
    ).to_parquet(path, index=False)

    metadata = _load_warehouse_metadata(path)

    assert list(metadata.columns) == ["game_id", "kickoff_utc"]
    assert "home_score" not in metadata.columns
    assert "target_home_win" not in metadata.columns


def test_config_defaults_do_not_include_holdout_seasons():
    config = UnifiedTournamentConfig()
    assert config.development_seasons == (2021, 2022, 2023)
    assert 2024 not in config.development_seasons
    assert 2025 not in config.development_seasons



def test_moneyline_tie_allows_nullable_binary_home_win_target():
    frame = pd.DataFrame(
        {
            "market_t10_novig_probability": [0.60, 0.52, 0.40],
            "target_home_margin": [7.0, 0.0, -3.0],
            "target_home_win": [1.0, np.nan, 0.0],
            "target_tie": [0, 1, 0],
        }
    )

    market_probability, first, action = _market_and_targets(
        "pregame_moneyline",
        frame,
    )

    assert market_probability.tolist() == pytest.approx(
        [0.60, 0.52, 0.40]
    )
    assert first.tolist() == [1, 0, 0]
    assert action.tolist() == [True, False, True]
