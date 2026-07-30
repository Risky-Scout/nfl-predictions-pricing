import numpy as np
import pandas as pd

from nfl_hybrid.monitoring.calibration_drift import (
    DriftConfig,
    cusum_drift,
    rolling_ece,
    monitor_calibration_drift,
    build_recalibration_request,
)


def test_rolling_ece_nan_before_window():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.full(6, 0.5)
    out = rolling_ece(y, p, window=4, n_bins=5)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()


def test_cusum_flags_systematic_drift():
    # model always predicts 0.9 but outcome is always 0 -> big positive residual
    residual = np.full(50, 0.9)
    out = cusum_drift(residual, slack_k=0.02, threshold_h=1.5)
    assert out["alarm"].any()


def test_cusum_quiet_when_calibrated():
    rng = np.random.default_rng(0)
    # residuals centered at zero
    residual = rng.normal(0.0, 0.05, size=200)
    out = cusum_drift(residual, slack_k=0.05, threshold_h=2.0)
    assert not out["alarm"].any()


def test_monitor_and_request_end_to_end():
    n = 120
    rng = np.random.default_rng(7)
    # chronological order: unique increasing week index so the stable sort is a no-op
    weeks = np.arange(n)
    # a well-calibrated market: random balanced coin flips priced at 0.5
    good = pd.DataFrame(
        {
            "market": "moneyline",
            "season": 2025,
            "week": weeks,
            "home_win": rng.integers(0, 2, size=n),
            "prob": 0.5,
        }
    )
    # a drifting market: model insists 0.9 but the event never happens
    bad = pd.DataFrame(
        {
            "market": "ats",
            "season": 2025,
            "week": weeks,
            "home_win": np.zeros(n, dtype=int),
            "prob": 0.9,
        }
    )
    frame = pd.concat([good, bad], ignore_index=True)
    report = monitor_calibration_drift(
        frame,
        outcome_col="home_win",
        probability_col="prob",
        market_col="market",
        config=DriftConfig(minimum_resolved=16),
    )
    req = build_recalibration_request(report)
    assert req["recalibration_required"] is True
    assert "ats" in req["markets"]
    assert "moneyline" not in req["markets"]
    assert req["action"] == "RUN_CALIBRATION_ADOPTION_GATE"


def test_no_drift_no_action():
    n = 80
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "market": "total",
            "season": 2025,
            "week": np.arange(n),
            "over": rng.integers(0, 2, size=n),
            "prob": 0.5,
        }
    )
    report = monitor_calibration_drift(
        frame, outcome_col="over", probability_col="prob",
        config=DriftConfig(minimum_resolved=16),
    )
    req = build_recalibration_request(report)
    assert req["recalibration_required"] is False
    assert req["action"] == "NO_ACTION"
