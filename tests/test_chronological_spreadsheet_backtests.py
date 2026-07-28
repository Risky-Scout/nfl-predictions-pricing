from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.chronological_spreadsheet_backtests import (
    BacktestConfig,
    _conditional_market_probabilities,
    _discrete_integer_distribution,
    _margin_probabilities,
    _shrunk_sd,
    _three_way_scores,
    _tie_probability,
    _total_probabilities,
)


def test_discrete_distribution_normalizes():
    support, probabilities = _discrete_integer_distribution(
        3.2,
        13.0,
        lower=-70,
        upper=70,
    )
    assert len(support) == len(probabilities)
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities >= 0).all()


def test_half_point_spread_has_zero_push_probability():
    home, away, push = _margin_probabilities(
        mean_margin=3.0,
        home_spread=-3.5,
        sd=13.0,
    )
    assert home + away + push == pytest.approx(1.0)
    assert push == pytest.approx(0.0, abs=1e-15)


def test_integer_spread_has_explicit_push_probability():
    home, away, push = _margin_probabilities(
        mean_margin=3.0,
        home_spread=-3.0,
        sd=13.0,
    )
    assert home + away + push == pytest.approx(1.0)
    assert push > 0.0


def test_integer_total_has_push_probability():
    over, under, push = _total_probabilities(
        mean_total=47.0,
        total_line=47.0,
        sd=13.0,
    )
    assert over + under + push == pytest.approx(1.0)
    assert push > 0.0


def test_half_point_total_has_zero_push_probability():
    over, under, push = _total_probabilities(
        mean_total=47.0,
        total_line=47.5,
        sd=13.0,
    )
    assert over + under + push == pytest.approx(1.0)
    assert push == pytest.approx(0.0, abs=1e-15)


def test_tie_prior_is_strongly_regularized():
    config = BacktestConfig(
        tie_prior_alpha=2.0,
        tie_prior_beta=398.0,
    )
    assert _tie_probability(0, 0, config) == pytest.approx(0.005)
    assert _tie_probability(1, 100, config) == pytest.approx(0.006)


def test_conditional_market_probability_rescales_for_rare_event():
    home, away, rare = _conditional_market_probabilities(
        0.60,
        0.02,
        clip=1e-6,
    )
    assert home == pytest.approx(0.588)
    assert away == pytest.approx(0.392)
    assert rare == pytest.approx(0.02)
    assert home + away + rare == pytest.approx(1.0)


def test_three_way_scores_are_proper():
    log_loss, brier = _three_way_scores(
        (0.60, 0.38, 0.02),
        0,
        clip=1e-6,
    )
    assert log_loss == pytest.approx(-np.log(0.60))
    assert brier == pytest.approx(
        (0.60 - 1.0) ** 2 + 0.38**2 + 0.02**2
    )


def test_shrunk_sd_uses_prior_when_empty():
    assert _shrunk_sd(
        [],
        prior_sd=13.5,
        prior_games=64,
    ) == pytest.approx(13.5)


def test_shrunk_sd_is_positive():
    value = _shrunk_sd(
        [1.0, -2.0, 4.0, 0.0],
        prior_sd=13.5,
        prior_games=64,
    )
    assert value > 0.0
    assert value < 13.5
