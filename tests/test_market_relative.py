import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation.market_relative import (
    MarketRelativeConfig,
    classwise_ece,
    evaluate_market_relative,
    market_implied_probabilities_from_lines,
)
from nfl_hybrid.governance.readiness import classify_market_relative_readiness


def test_classwise_ece_perfect_is_zero():
    # perfectly calibrated deterministic predictions
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = y.astype(float)
    assert classwise_ece(y, p, n_bins=10) == pytest.approx(0.0)


def test_classwise_ece_detects_miscalibration():
    y = np.zeros(100)
    p = np.full(100, 0.9)  # predicts 0.9 but truth is always 0
    assert classwise_ece(y, p, n_bins=10) == pytest.approx(0.9, abs=1e-9)


def test_reference_probs_labeled_proxy():
    frame = pd.DataFrame({"home_spread": [-7.0, 3.0, 0.0]})
    ref = market_implied_probabilities_from_lines(frame)
    assert (ref["market_source"] == "REFERENCE-LINE PROXY").all()
    # home favored by 7 -> home win prob > 0.5
    assert ref["market_ml_home_probability"].iloc[0] > 0.5
    # cover/over are 0.5 at the reference line
    assert ref["market_cover_home_probability"].iloc[0] == 0.5
    assert ref["market_over_probability"].iloc[0] == 0.5


def _prediction_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    season = rng.choice([2023, 2024], size=n)
    home_spread = rng.normal(0, 6, size=n).round()
    home_margin = -home_spread + rng.normal(0, 13, size=n)
    total_line = rng.normal(45, 5, size=n).round()
    total_points = total_line + rng.normal(0, 13, size=n)
    return pd.DataFrame(
        {
            "season": season,
            "home_spread": home_spread,
            "total_line": total_line,
            "home_margin": home_margin,
            "total_points": total_points,
            "home_win": (home_margin > 0).astype(int),
            "home_cover": (home_margin + home_spread > 0).astype(int),
            "over": (total_points > total_line).astype(int),
            "predicted_margin": -home_spread,  # model == market baseline
            "home_win_probability_no_tie": np.clip(0.5 - home_spread * 0.03, 0.02, 0.98),
            "home_cover_probability_no_push": np.full(n, 0.5),
            "over_probability_no_push": np.full(n, 0.5),
        }
    )


def test_evaluate_market_relative_shapes_and_labels():
    frame = _prediction_frame()
    result = evaluate_market_relative(
        frame, config=MarketRelativeConfig(bootstrap_repetitions=200)
    )
    prob = result["probability_scorecard"]
    margin = result["margin_scorecard"]
    assert {"pooled"}.issubset(set(prob["segment"]))
    assert (prob["market_source"] == "REFERENCE-LINE PROXY").all()
    # required metric columns present
    for col in [
        "brier_gain_vs_market",
        "log_loss_gain_vs_market",
        "pick_accuracy",
        "pick_accuracy_ci_lower",
        "pick_accuracy_ci_upper",
        "model_ece",
        "beats_breakeven_ci",
    ]:
        assert col in prob.columns
    assert "fraction_model_beats_market" in margin.columns


def test_market_baseline_yields_retain_baseline():
    # model == market -> no edge -> RETAIN_BASELINE
    frame = _prediction_frame()
    result = evaluate_market_relative(
        frame, config=MarketRelativeConfig(bootstrap_repetitions=200)
    )
    readiness = classify_market_relative_readiness(result["probability_scorecard"])
    assert (readiness["readiness_status"] == "RETAIN_BASELINE").all()
