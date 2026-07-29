import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.staking.kelly import (
    StakingPolicy,
    binary_kl_divergence,
    penalized_edge,
    full_kelly_fraction,
    stake_bets,
)


def test_kl_zero_when_equal():
    assert binary_kl_divergence(0.6, 0.6) == pytest.approx(0.0, abs=1e-12)
    assert binary_kl_divergence(0.6, 0.6) >= 0.0


def test_kl_nonnegative():
    for p, q in [(0.7, 0.5), (0.2, 0.5), (0.99, 0.5)]:
        assert binary_kl_divergence(p, q) > 0.0


def test_penalized_edge_penalizes_divergence():
    # equal probs -> edge exactly zero
    assert penalized_edge(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
    # a real edge is always reduced by the KL penalty
    raw = 0.60 - 0.50
    pen = float(penalized_edge(0.60, 0.50))
    assert pen < raw
    assert pen > 0.0


def test_full_kelly_matches_closed_form():
    # p=0.6, decimal 2.0 (b=1), no push -> f* = (0.6*1 - 0.4)/1 = 0.2
    assert float(full_kelly_fraction(0.6, 2.0, 0.0)) == pytest.approx(0.2)


def test_full_kelly_clips_no_edge():
    # fair coin at even money -> zero
    assert float(full_kelly_fraction(0.5, 2.0, 0.0)) == pytest.approx(0.0)
    # negative edge clips to 0
    assert float(full_kelly_fraction(0.4, 2.0, 0.0)) == 0.0


def _frame(model_p, market_p, readiness="STATISTICALLY_SUPPORTED", odds=2.0, n=1, week=1):
    return pd.DataFrame(
        {
            "model_probability": [model_p] * n,
            "market_fair_probability": [market_p] * n,
            "offered_decimal": [odds] * n,
            "season": [2025] * n,
            "week": [week] * n,
            "readiness_status": [readiness] * n,
        }
    )


def test_retain_baseline_stakes_zero():
    out = stake_bets(_frame(0.70, 0.50, readiness="RETAIN_BASELINE"))
    assert out["recommended_stake"].iloc[0] == 0.0
    assert out["should_bet"].iloc[0] is np.False_ or not out["should_bet"].iloc[0]
    assert "no-bet: edge not established" in out["no_bet_reason"].iloc[0]


def test_minimum_edge_gate():
    # small edge below 2% penalized threshold -> no bet
    out = stake_bets(_frame(0.51, 0.50))
    assert out["penalized_edge"].iloc[0] < 0.02
    assert out["recommended_stake"].iloc[0] == 0.0


def test_qualified_bet_is_capped_at_5pct():
    # huge edge -> Kelly large, but per-bet cap is 5%
    out = stake_bets(_frame(0.90, 0.50, odds=2.0))
    assert out["should_bet"].iloc[0]
    assert out["recommended_stake"].iloc[0] == pytest.approx(0.05)


def test_weekly_exposure_cap_scales_proportionally():
    # 10 strong same-week bets each want 5% -> 50% > 15% cap -> scaled to 15% total
    frame = _frame(0.90, 0.50, n=10, week=3)
    out = stake_bets(frame)
    assert out["recommended_stake"].sum() == pytest.approx(0.15, abs=1e-9)
    # cap keyed by (season, week), not team: all equal shares
    assert out["recommended_stake"].nunique() == 1


def test_weekly_cap_independent_across_weeks():
    f1 = _frame(0.90, 0.50, n=5, week=1)
    f2 = _frame(0.90, 0.50, n=5, week=2)
    out = stake_bets(pd.concat([f1, f2], ignore_index=True))
    wk1 = out[out["week"] == 1]["recommended_stake"].sum()
    wk2 = out[out["week"] == 2]["recommended_stake"].sum()
    assert wk1 == pytest.approx(0.15)
    assert wk2 == pytest.approx(0.15)


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        stake_bets(pd.DataFrame({"model_probability": [0.5]}))


def test_policy_validation():
    with pytest.raises(ValueError):
        StakingPolicy(per_bet_cap=1.5)
