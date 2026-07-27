from pathlib import Path

import pandas as pd
import yaml

from nfl_hybrid.governance import (
    InputState,
    classify_distributional_readiness,
    evaluate_abstention,
    validate_governance_contracts,
)


def test_repository_governance_contracts_pass():
    result = validate_governance_contracts(Path("config"))
    assert result["status"] == "PASS"


def test_abstains_on_stale_odds():
    status, reasons = evaluate_abstention(
        InputState(
            odds_age_minutes=20,
            eligible_books=4,
        )
    )
    assert status == "NO_PREDICTION"
    assert "STALE_ODDS" in reasons


def test_abstains_when_model_did_not_qualify():
    status, reasons = evaluate_abstention(
        InputState(
            odds_age_minutes=5,
            eligible_books=4,
            expected_value=0.05,
            edge=0.06,
            uncertainty=0.01,
            readiness_status="RETAIN_BASELINE",
        )
    )
    assert status == "NO_BET"
    assert "MODEL_DID_NOT_QUALIFY" in reasons


def test_positive_ci_is_statistically_supported():
    frame = pd.DataFrame(
        [
            {
                "market": "pregame_moneyline",
                "variant": "football_only",
                "role": "selected",
                "mean_binary_log_loss_gain": 0.04,
                "binary_log_loss_gain_ci_lower": 0.01,
                "binary_log_loss_gain_ci_upper": 0.07,
                "mean_binary_brier_gain": 0.02,
                "binary_brier_gain_ci_lower": 0.005,
                "binary_brier_gain_ci_upper": 0.03,
            }
        ]
    )
    result = classify_distributional_readiness(frame)
    assert result.iloc[0]["readiness_status"] == "STATISTICALLY_SUPPORTED"


def test_ci_crossing_zero_is_provisional():
    frame = pd.DataFrame(
        [
            {
                "market": "pregame_total",
                "variant": "market_augmented",
                "role": "selected",
                "mean_binary_log_loss_gain": 0.005,
                "binary_log_loss_gain_ci_lower": -0.004,
                "binary_log_loss_gain_ci_upper": 0.013,
                "mean_binary_brier_gain": 0.002,
                "binary_brier_gain_ci_lower": -0.002,
                "binary_brier_gain_ci_upper": 0.006,
            }
        ]
    )
    result = classify_distributional_readiness(frame)
    assert result.iloc[0]["readiness_status"] == "PROVISIONAL_ONLY"


def test_negative_challenger_retains_baseline():
    frame = pd.DataFrame(
        [
            {
                "market": "pregame_ats",
                "variant": "market_augmented",
                "role": "best_challenger",
                "mean_binary_log_loss_gain": -0.001,
                "binary_log_loss_gain_ci_lower": -0.01,
                "binary_log_loss_gain_ci_upper": 0.005,
                "mean_binary_brier_gain": -0.001,
            }
        ]
    )
    result = classify_distributional_readiness(frame)
    assert result.iloc[0]["readiness_status"] == "RETAIN_BASELINE"
