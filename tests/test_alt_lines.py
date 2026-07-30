import numpy as np
import pytest

from nfl_hybrid.pricing.alt_lines import (
    AltLineConfig,
    price_moneyline,
    price_spread,
    price_spread_ladder,
    price_total,
    price_total_ladder,
    standard_ladder,
)


def test_spread_probabilities_sum_to_one():
    for mean_margin in (-6.0, 0.0, 3.5, 10.0):
        for spread in (-7, -3, 0, 2.5, 6):
            q = price_spread(mean_margin, spread)
            total = q["home_cover_probability"] + q["push_probability"] + q["away_cover_probability"]
            assert total == pytest.approx(1.0, abs=1e-9)


def test_total_probabilities_sum_to_one():
    for mean_total in (40.0, 47.5, 52.0):
        for line in (44, 47.5, 50):
            q = price_total(mean_total, line)
            total = q["over_probability"] + q["push_probability"] + q["under_probability"]
            assert total == pytest.approx(1.0, abs=1e-9)


def test_moneyline_sums_to_one():
    q = price_moneyline(3.0)
    assert q["home_win_probability"] + q["tie_probability"] + q["away_win_probability"] == pytest.approx(1.0, abs=1e-9)


def test_spread_ladder_monotonic():
    # more points for home (higher spread) -> higher home cover probability
    ladder = price_spread_ladder(0.0, spreads=range(-14, 15))
    hc = ladder["home_cover_probability"].to_numpy()
    assert np.all(np.diff(hc) >= -1e-12)  # nondecreasing


def test_total_ladder_monotonic():
    # higher total line -> lower over probability
    ladder = price_total_ladder(47.0, totals=range(35, 60))
    ov = ladder["over_probability"].to_numpy()
    assert np.all(np.diff(ov) <= 1e-12)  # nonincreasing


def test_fair_decimal_is_zero_ev():
    q = price_spread(0.0, -3)
    p, u, d = q["home_cover_probability"], q["push_probability"], q["home_fair_decimal"]
    ev = p * (d - 1.0) - (1.0 - p - u)  # push returns stake
    assert ev == pytest.approx(0.0, abs=1e-9)


def test_offered_decimal_has_positive_hold():
    # book's two offered implied probabilities exceed 1 by ~hold
    cfg = AltLineConfig(hold=0.05)
    q = price_spread(0.0, -3, config=cfg)
    implied = 1.0 / q["home_offered_decimal"] + 1.0 / q["away_offered_decimal"]
    assert implied > 1.0
    assert implied == pytest.approx(1.05, abs=1e-6)


def test_standard_ladder_shape():
    spreads, totals = standard_ladder(-3.0, 47.5, step_range=10)
    assert len(spreads) == 21 and len(totals) == 21
    assert -3.0 in spreads and 47.5 in totals
