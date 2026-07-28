from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.confirmation_2024 import (
    ConfirmationConfig,
    _deterministic_regularization,
    _paired_week_bootstrap,
    run_2024_confirmation,
)


def test_confirmation_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="allow-2024-confirmation"):
        run_2024_confirmation(
            combined_canonical_root=tmp_path,
            warehouse_path=tmp_path / "warehouse.parquet",
            final_development_root=tmp_path,
            freeze_spec_path=tmp_path / "spec.json",
            repo_root=tmp_path,
            confirmation_config_path=tmp_path / "config.json",
            output_root=tmp_path / "output",
            allow_confirmation=False,
        )


def test_regularization_aggregation_uses_mode(tmp_path):
    path = tmp_path / "tuning.csv"
    pd.DataFrame(
        {
            "market": ["pregame_total"] * 3,
            "feature_set": ["movement_only"] * 3,
            "regularization": [2.0, 2.0, 50.0],
        }
    ).to_csv(path, index=False)

    assert _deterministic_regularization(
        path,
        "pregame_total",
        "movement_only",
    ) == pytest.approx(2.0)


def test_regularization_tie_uses_strongest_value(tmp_path):
    path = tmp_path / "tuning.csv"
    pd.DataFrame(
        {
            "market": ["pregame_total"] * 3,
            "feature_set": ["movement_only"] * 3,
            "regularization": [0.5, 2.0, 50.0],
        }
    ).to_csv(path, index=False)

    assert _deterministic_regularization(
        path,
        "pregame_total",
        "movement_only",
    ) == pytest.approx(50.0)


def test_week_cluster_bootstrap_reports_gain():
    rows = []
    for week in (1, 2, 3):
        for game_number in range(5):
            game_id = f"{week}_{game_number}"
            rows.extend(
                [
                    {
                        "game_id": game_id,
                        "season": 2024,
                        "week": week,
                        "market": "pregame_moneyline",
                        "model_family": "market_baseline",
                        "model_name": "market_t10_canonical",
                        "log_loss": 0.70,
                        "brier": 0.50,
                    },
                    {
                        "game_id": game_id,
                        "season": 2024,
                        "week": week,
                        "market": "pregame_moneyline",
                        "model_family": "candidate",
                        "model_name": "candidate_model",
                        "log_loss": 0.60,
                        "brier": 0.40,
                    },
                ]
            )

    # Add empty-equivalent rows for the other markets so the helper can loop.
    for market in ("pregame_ats", "pregame_total"):
        for week in (1, 2, 3):
            for game_number in range(5):
                game_id = f"{market}_{week}_{game_number}"
                rows.extend(
                    [
                        {
                            "game_id": game_id,
                            "season": 2024,
                            "week": week,
                            "market": market,
                            "model_family": "market_baseline",
                            "model_name": "market_t10_canonical",
                            "log_loss": 0.70,
                            "brier": 0.50,
                        },
                        {
                            "game_id": game_id,
                            "season": 2024,
                            "week": week,
                            "market": market,
                            "model_family": "candidate",
                            "model_name": "candidate_model",
                            "log_loss": 0.60,
                            "brier": 0.40,
                        },
                    ]
                )

    result = _paired_week_bootstrap(
        pd.DataFrame(rows),
        ConfirmationConfig(
            bootstrap_repetitions=100,
            expected_games=15,
        ),
    )

    assert len(result) == 3
    assert (result["mean_log_loss_gain"] > 0).all()
    assert (result["mean_brier_gain"] > 0).all()


def test_confirmation_config_excludes_2025():
    config = ConfirmationConfig()
    assert config.confirmation_season == 2024
    assert 2025 not in config.training_seasons
