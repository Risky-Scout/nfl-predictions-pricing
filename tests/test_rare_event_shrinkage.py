from collections import Counter

import numpy as np
import pytest

from nfl_hybrid.distributional.rare_event_shrinkage import (
    _ats_exact_margin_probability,
    _ats_integer_rate_probability,
    _beta_posterior_mean,
    _discrete_normal_mass,
    _is_integer_line,
    _normalized,
    _three_way,
    _total_exact_points_probability,
)


def test_beta_posterior_mean_is_shrunk():
    value = _beta_posterior_mean(
        successes=0,
        trials=10,
        prior_mean=0.005,
        prior_strength=400,
    )
    assert 0.0 < value < 0.005


def test_noninteger_ats_line_has_zero_push_probability():
    assert _ats_exact_margin_probability(
        -3.5,
        Counter({3: 10}),
        games_seen=100,
        prior_strength=64,
        prior_sd=13.5,
    ) == 0.0
    assert _ats_integer_rate_probability(
        -3.5,
        pushes=5,
        games=100,
        prior_strength=64,
        prior_mean=0.04,
    ) == 0.0


def test_integer_ats_line_has_explicit_push_probability():
    value = _ats_exact_margin_probability(
        -3.0,
        Counter({3: 10}),
        games_seen=100,
        prior_strength=64,
        prior_sd=13.5,
    )
    assert value > 0.0


def test_noninteger_total_line_has_zero_push_probability():
    value = _total_exact_points_probability(
        47.5,
        Counter({47: 10}),
        games_seen=100,
        prior_strength=64,
        prior_mean=45.0,
        prior_sd=13.0,
    )
    assert value == 0.0


def test_integer_line_detection():
    assert _is_integer_line(-3.0)
    assert not _is_integer_line(-3.5)
    assert _is_integer_line(47.0)
    assert not _is_integer_line(47.5)


def test_three_way_probability_construction():
    probabilities = _three_way(0.60, 0.04)
    assert probabilities == pytest.approx((0.576, 0.384, 0.04))
    assert sum(probabilities) == pytest.approx(1.0)


def test_probability_normalization():
    values = _normalized((0.576, 0.384, 0.04), 1e-9)
    assert values.sum() == pytest.approx(1.0)
    assert (values >= 0).all()


def test_discrete_normal_mass_positive():
    assert _discrete_normal_mass(3, 0.0, 13.5) > 0.0


def test_invalid_beta_counts_raise():
    with pytest.raises(ValueError):
        _beta_posterior_mean(
            successes=2,
            trials=1,
            prior_mean=0.005,
            prior_strength=400,
        )
