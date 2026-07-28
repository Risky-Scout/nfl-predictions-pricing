import pandas as pd
import pytest

from nfl_hybrid.selection.final_test_2025 import (
    FinalTestConfig,
    _final_decision,
    run_final_2025_evaluation,
)


def test_final_test_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="allow-final-test"):
        run_final_2025_evaluation(
            combined_canonical_root=tmp_path,
            warehouse_path=tmp_path / "warehouse.parquet",
            frozen_spec_path=tmp_path / "spec.json",
            rare_event_config_path=tmp_path / "rare.json",
            final_test_config_path=tmp_path / "final.json",
            output_root=tmp_path / "output",
            access_log_path=tmp_path / "access.jsonl",
            allow_final_test=False,
        )


def test_market_baseline_remains_market_baseline():
    decision = _final_decision(
        market="pregame_moneyline",
        frozen_family="market_baseline",
        frozen_model_name="market_t10_canonical",
        final_row=pd.Series(
            {
                "games": 285,
                "mean_log_loss": 0.60,
                "mean_brier": 0.42,
                "rare_calibration_error": 0.001,
            }
        ),
        baseline_row=pd.Series(
            {
                "mean_log_loss": 0.60,
                "mean_brier": 0.42,
            }
        ),
        bootstrap_row=pd.Series(
            {
                "mean_log_loss_gain": 0.0,
                "log_loss_ci_lower": 0.0,
                "log_loss_ci_upper": 0.0,
                "mean_brier_gain": 0.0,
                "brier_ci_lower": 0.0,
                "brier_ci_upper": 0.0,
            }
        ),
        config=FinalTestConfig(),
    )

    assert (
        decision["production_decision"]
        == "PRODUCTION_MARKET_BASELINE"
    )
    assert decision["production_model_name"] == "market_t10_canonical"


def test_challenger_falls_back_when_proper_scores_worsen():
    decision = _final_decision(
        market="pregame_ats",
        frozen_family="rare_event_market_baseline",
        frozen_model_name="ats_push_exact_margin_strength_32",
        final_row=pd.Series(
            {
                "games": 285,
                "mean_log_loss": 0.79,
                "mean_brier": 0.53,
                "rare_calibration_error": 0.002,
            }
        ),
        baseline_row=pd.Series(
            {
                "mean_log_loss": 0.78,
                "mean_brier": 0.52,
            }
        ),
        bootstrap_row=pd.Series(
            {
                "mean_log_loss_gain": -0.01,
                "log_loss_ci_lower": -0.02,
                "log_loss_ci_upper": 0.00,
                "mean_brier_gain": -0.01,
                "brier_ci_lower": -0.02,
                "brier_ci_upper": 0.00,
            }
        ),
        config=FinalTestConfig(),
    )

    assert (
        decision["production_decision"]
        == "PRODUCTION_MARKET_FALLBACK"
    )
    assert decision["production_model_name"] == "market_t10_canonical"


def test_challenger_is_kept_when_all_predeclared_gates_pass():
    decision = _final_decision(
        market="pregame_ats",
        frozen_family="rare_event_market_baseline",
        frozen_model_name="ats_push_exact_margin_strength_32",
        final_row=pd.Series(
            {
                "games": 285,
                "mean_log_loss": 0.77,
                "mean_brier": 0.51,
                "rare_calibration_error": 0.002,
            }
        ),
        baseline_row=pd.Series(
            {
                "mean_log_loss": 0.78,
                "mean_brier": 0.52,
            }
        ),
        bootstrap_row=pd.Series(
            {
                "mean_log_loss_gain": 0.01,
                "log_loss_ci_lower": -0.01,
                "log_loss_ci_upper": 0.03,
                "mean_brier_gain": 0.01,
                "brier_ci_lower": -0.01,
                "brier_ci_upper": 0.03,
            }
        ),
        config=FinalTestConfig(),
    )

    assert (
        decision["production_decision"]
        == "PRODUCTION_CONFIRMED_CHALLENGER"
    )
    assert (
        decision["production_model_name"]
        == "ats_push_exact_margin_strength_32"
    )


def test_final_test_config_is_locked_to_2025():
    config = FinalTestConfig()
    assert config.final_test_season == 2025
    assert config.training_and_selection_seasons == (
        2020,
        2021,
        2022,
        2023,
        2024,
    )
    assert config.retuning_prohibited
    assert config.feature_changes_prohibited
    assert config.calibration_changes_prohibited
