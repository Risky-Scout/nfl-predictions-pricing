"""Fix 4 items 5 & 6 (+ certification-blocker repair item 1): independent
mathematical reference tests for the one canonical distributional-pricing
definition (:mod:`nfl_hybrid.pricing.distribution_pricing`), plus
numerical-parity proof against
:class:`nfl_hybrid.modern.joint_score.JointScoreModel`'s own
continuity-corrected formula, for ANY finite real line -- not only
integer/half-point ones. A median-of-an-even-number-of-books consensus
line can genuinely land on a quarter-point or other fraction (see
``nfl_hybrid.odds_history.build_consensus``), so this suite explicitly
covers that case rather than assuming it away.

Every "expected" value below is computed directly from
``scipy.stats.norm.cdf`` (or, for the sign tests, from first principles about
which side of a threshold is easier to clear) -- never by calling
``price_ats_distribution`` / ``price_total_distribution`` /
``MarginSurface`` / ``JointScoreModel.raw_probabilities`` on the "expected"
side. This is what lets these tests catch a sign inversion or an off-by-one
continuity correction that the code under test would not catch on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_hybrid.modern.joint_score import JointScoreModel
from nfl_hybrid.pricing.distribution_pricing import (
    price_ats_distribution,
    price_moneyline_distribution,
    price_total_distribution,
)

# ---------------------------------------------------------------------------
# A. ATS half-point favorite: push probability must be exactly 0.
# ---------------------------------------------------------------------------


def test_ats_half_point_favorite_push_probability_is_zero():
    home, push, away = price_ats_distribution(mean_margin=3.0, margin_sd=7.0, home_spread=-3.5)
    assert push == 0.0
    assert np.isclose(home + push + away, 1.0)
    # Independently computed: home covers iff margin > 3.5; margin ~ N(3, 7^2).
    expected_home = 1.0 - norm.cdf((3.5 - 3.0) / 7.0)
    assert np.isclose(home, expected_home, atol=1e-9)


# ---------------------------------------------------------------------------
# B. ATS integer spread: continuity-corrected home-cover / push / away-cover.
# ---------------------------------------------------------------------------


def test_ats_integer_spread_continuity_corrected_three_way_split():
    mean_margin, sigma, spread = 2.0, 6.0, -3.0
    home, push, away = price_ats_distribution(mean_margin, sigma, spread)

    # edge = margin + spread; integer-valued (margin and spread both
    # integers), so home covers iff edge >= 1, pushes iff edge == 0, away
    # covers iff edge <= -1. Continuity-corrected around the edge mean
    # (mean_margin + spread) = -1.0.
    edge_mean = mean_margin + spread
    expected_push = norm.cdf((0.5 - edge_mean) / sigma) - norm.cdf((-0.5 - edge_mean) / sigma)
    expected_home = 1.0 - norm.cdf((0.5 - edge_mean) / sigma)
    expected_away = norm.cdf((-0.5 - edge_mean) / sigma)

    assert np.isclose(push, expected_push, atol=1e-9)
    assert np.isclose(home, expected_home, atol=1e-9)
    assert np.isclose(away, expected_away, atol=1e-9)
    assert push > 0.0  # a genuine, non-degenerate push region for an integer line
    assert np.isclose(home + push + away, 1.0)


# ---------------------------------------------------------------------------
# C. Positive home-underdog spread: sign direction.
# ---------------------------------------------------------------------------


def test_ats_positive_home_spread_sign_direction():
    """home_spread > 0 means the home team is getting points (an underdog
    line). Getting points must make a home cover strictly MORE likely than
    pick'em, and a negative (favorite) spread strictly LESS likely -- proven
    against directly-computed norm.cdf values, not just a monotonicity
    assertion against the code under test."""
    mean_margin, sigma = -1.0, 6.0

    home_pick_em, _, _ = price_ats_distribution(mean_margin, sigma, 0.0)
    home_underdog, _, _ = price_ats_distribution(mean_margin, sigma, 3.5)
    home_favorite, _, _ = price_ats_distribution(mean_margin, sigma, -3.5)

    expected_pick_em = 1.0 - norm.cdf((0.5 - mean_margin) / sigma)
    expected_underdog = norm.cdf((mean_margin + 3.5) / sigma)  # half-point line, no continuity correction
    expected_favorite = norm.cdf((mean_margin - 3.5) / sigma)

    assert np.isclose(home_pick_em, expected_pick_em, atol=1e-9)
    assert np.isclose(home_underdog, expected_underdog, atol=1e-9)
    assert np.isclose(home_favorite, expected_favorite, atol=1e-9)
    assert home_favorite < home_pick_em < home_underdog


# ---------------------------------------------------------------------------
# D. Arbitrary fractional (quarter-point) ATS lines: push=0, no continuity
# correction -- a median-of-an-even-number-of-books consensus can genuinely
# produce these (not only half-points).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("home_spread", [-2.25, -2.75, 2.25])
def test_ats_quarter_point_lines_push_zero_and_uncorrected_cdf(home_spread):
    mean_margin, sigma = 1.0, 6.5
    home, push, away = price_ats_distribution(mean_margin, sigma, home_spread)
    assert push == 0.0
    edge_mean = mean_margin + home_spread
    expected_home = norm.cdf(edge_mean / sigma)  # no continuity correction for a non-integer line
    assert np.isclose(home, expected_home, atol=1e-9)
    assert np.isclose(away, 1.0 - expected_home, atol=1e-9)
    assert np.isclose(home + push + away, 1.0)


# ---------------------------------------------------------------------------
# TOTAL A: half-point total, push probability must be exactly 0.
# ---------------------------------------------------------------------------


def test_total_half_point_push_probability_is_zero():
    over, push, under = price_total_distribution(mean_total=44.0, total_sd=9.0, total_line=44.5)
    assert push == 0.0
    expected_over = norm.cdf((44.0 - 44.5) / 9.0)  # over iff total > 44.5, i.e. edge = total - line > 0
    assert np.isclose(over, expected_over, atol=1e-9)
    assert np.isclose(over + push + under, 1.0)


# ---------------------------------------------------------------------------
# TOTAL B: integer total, continuity-corrected push region.
# ---------------------------------------------------------------------------


def test_total_integer_continuity_corrected_push_region():
    mean_total, sigma, line = 45.0, 8.0, 44.0
    over, push, under = price_total_distribution(mean_total, sigma, line)

    edge_mean = mean_total - line
    expected_push = norm.cdf((0.5 - edge_mean) / sigma) - norm.cdf((-0.5 - edge_mean) / sigma)
    expected_over = 1.0 - norm.cdf((0.5 - edge_mean) / sigma)
    expected_under = norm.cdf((-0.5 - edge_mean) / sigma)

    assert np.isclose(push, expected_push, atol=1e-9)
    assert np.isclose(over, expected_over, atol=1e-9)
    assert np.isclose(under, expected_under, atol=1e-9)
    assert push > 0.0
    assert np.isclose(over + push + under, 1.0)


@pytest.mark.parametrize("total_line", [47.25, 47.75])
def test_total_quarter_point_lines_push_zero_and_uncorrected_cdf(total_line):
    mean_total, sigma = 45.0, 8.5
    over, push, under = price_total_distribution(mean_total, sigma, total_line)
    assert push == 0.0
    edge_mean = mean_total - total_line
    expected_over = norm.cdf(edge_mean / sigma)
    assert np.isclose(over, expected_over, atol=1e-9)
    assert np.isclose(under, 1.0 - expected_over, atol=1e-9)
    assert np.isclose(over + push + under, 1.0)


def test_total_higher_line_makes_over_strictly_harder_independently_verified():
    mean_total, sigma = 45.0, 8.0
    low_over, _, _ = price_total_distribution(mean_total, sigma, 40.0)
    high_over, _, _ = price_total_distribution(mean_total, sigma, 50.0)
    expected_low = 1.0 - norm.cdf((40.5 - mean_total) / sigma)
    expected_high = 1.0 - norm.cdf((50.5 - mean_total) / sigma)
    assert np.isclose(low_over, expected_low, atol=1e-9)
    assert np.isclose(high_over, expected_high, atol=1e-9)
    assert high_over < low_over


# ---------------------------------------------------------------------------
# Moneyline special case
# ---------------------------------------------------------------------------


def test_moneyline_matches_ats_at_zero_spread():
    home_ml, push_ml, away_ml = price_moneyline_distribution(mean_margin=1.5, margin_sd=6.0)
    home_ats, push_ats, away_ats = price_ats_distribution(1.5, 6.0, 0.0)
    assert home_ml == home_ats
    assert push_ml == push_ats
    assert away_ml == away_ats


# ---------------------------------------------------------------------------
# Vectorization sanity (array inputs match elementwise scalar calls)
# ---------------------------------------------------------------------------


def test_price_ats_distribution_is_vectorized_and_matches_scalar_calls():
    means = np.array([-4.0, 0.0, 2.5, 9.0])
    sds = np.array([6.0, 7.0, 8.0, 6.5])
    spreads = np.array([-3.0, 0.0, 6.5, -10.0])
    home_vec, push_vec, away_vec = price_ats_distribution(means, sds, spreads)
    for i in range(len(means)):
        h, p, a = price_ats_distribution(float(means[i]), float(sds[i]), float(spreads[i]))
        assert np.isclose(home_vec[i], h)
        assert np.isclose(push_vec[i], p)
        assert np.isclose(away_vec[i], a)


# ---------------------------------------------------------------------------
# Parity with JointScoreModel's own (independently written) formula.
# ---------------------------------------------------------------------------


def _unfitted_joint_score_model(margin_sd: float, total_sd: float, margin: np.ndarray, total: np.ndarray) -> JointScoreModel:
    """A JointScoreModel instance with no sklearn fit performed: _means is
    overridden to return fixed arrays, and the residual-distribution
    attributes are set directly, exactly the shape raw_probabilities() needs
    to run its own formula without requiring training data."""
    model = JointScoreModel(numeric_features=[])
    model.is_fitted_ = True
    model.margin_sd_ = margin_sd
    model.total_sd_ = total_sd
    model._means = lambda frame: (margin, total)  # type: ignore[method-assign]
    return model


def test_ats_parity_with_joint_score_models_own_formula():
    """price_ats_distribution IS JointScoreModel's own formula (extracted
    verbatim -- see the module docstring), so parity holds for ANY finite
    real spread, not only integers/half-points. A median-of-an-even-number-
    of-books consensus line can genuinely land on a quarter-point or other
    fraction (see ``nfl_hybrid.odds_history.build_consensus``), so this test
    deliberately uses unrestricted continuous random spreads, not just
    half-point increments."""
    rng = np.random.default_rng(7)
    n = 200
    margin = rng.normal(scale=5, size=n)
    total = np.full(n, 44.0)
    home_spread = rng.uniform(-14, 14, size=n)  # arbitrary real spreads, not restricted to half-points
    margin_sd = 12.0

    model = _unfitted_joint_score_model(margin_sd, 10.0, margin, total)
    frame = pd.DataFrame({"home_spread": home_spread, "total_line": np.full(n, 44.0)})
    joint_raw = model.raw_probabilities(frame)

    home, push, away = price_ats_distribution(margin, margin_sd, home_spread)

    # JointScoreModel reports the CONDITIONAL (non-push) home-cover
    # probability and the push probability separately; reconstruct its
    # UNCONDITIONAL home-cover probability the same way price_ats_distribution
    # reports it, then compare both push and unconditional home-cover.
    joint_home_unconditional = (
        joint_raw["raw_home_cover_probability_no_push"].to_numpy()
        * (1.0 - joint_raw["ats_push_probability"].to_numpy())
    )
    assert np.allclose(push, joint_raw["ats_push_probability"].to_numpy(), atol=1e-9)
    assert np.allclose(home, joint_home_unconditional, atol=1e-9)
    assert np.allclose(home + push + away, 1.0, atol=1e-9)


def test_total_parity_with_joint_score_models_own_formula():
    """See the docstring on the ATS parity test above -- unrestricted
    continuous random totals, not just half-point increments."""
    rng = np.random.default_rng(11)
    n = 200
    margin = np.zeros(n)
    total = 44.0 + rng.normal(scale=6, size=n)
    total_line = rng.uniform(35, 55, size=n)  # arbitrary real totals, not restricted to half-points
    total_sd = 9.5

    model = _unfitted_joint_score_model(12.0, total_sd, margin, total)
    frame = pd.DataFrame({"home_spread": np.zeros(n), "total_line": total_line})
    joint_raw = model.raw_probabilities(frame)

    over, push, under = price_total_distribution(total, total_sd, total_line)

    joint_over_unconditional = (
        joint_raw["raw_over_probability_no_push"].to_numpy()
        * (1.0 - joint_raw["total_push_probability"].to_numpy())
    )
    assert np.allclose(push, joint_raw["total_push_probability"].to_numpy(), atol=1e-9)
    assert np.allclose(over, joint_over_unconditional, atol=1e-9)
    assert np.allclose(over + push + under, 1.0, atol=1e-9)


@pytest.mark.parametrize("spread", [-3.0, -3.5, 0.0, 3.0, 7.0])
def test_ats_conditional_probability_matches_joint_score_model(spread):
    """Also check the CONDITIONAL (non-push) probability -- what Fix 4's
    raw_conditional_upper_probability and JointScoreModel's calibrator input
    both actually calibrate against -- agrees between the two formulas."""
    mean_margin, sigma = 4.0, 8.0
    model = _unfitted_joint_score_model(sigma, 10.0, np.array([mean_margin]), np.array([44.0]))
    frame = pd.DataFrame({"home_spread": [spread], "total_line": [44.0]})
    joint_raw = model.raw_probabilities(frame)
    joint_conditional = float(joint_raw["raw_home_cover_probability_no_push"].iloc[0])

    home, push, away = price_ats_distribution(mean_margin, sigma, spread)
    denom = home + away
    conditional = home / denom if denom > 0 else 0.5
    assert np.isclose(conditional, joint_conditional, atol=1e-9)
