import numpy as np

from nfl_hybrid.distributional.market_models import (
    EmpiricalPushPrior,
    FixedOffsetLogistic,
    ThreeWayProbabilities,
    combine_conditional_with_push,
    conditional_upper_probability,
    empirical_discrete_probabilities,
)


def test_three_way_probabilities_sum_to_one():
    probabilities = ThreeWayProbabilities(
        lower=np.array([0.4, 0.2]),
        push=np.array([0.1, 0.0]),
        upper=np.array([0.5, 0.8]),
    )
    matrix = probabilities.as_array()
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_half_point_threshold_has_zero_push_probability():
    probabilities = empirical_discrete_probabilities(
        predicted_mean=[0.0, 1.0],
        residual_samples=np.arange(-20, 21, dtype=float),
        threshold=[0.5, -2.5],
        grid_offset=0.0,
    )
    assert np.allclose(probabilities.push, 0.0)
    assert np.allclose(probabilities.as_array().sum(axis=1), 1.0)


def test_integer_threshold_has_positive_push_probability():
    probabilities = empirical_discrete_probabilities(
        predicted_mean=[0.0],
        residual_samples=np.arange(-20, 21, dtype=float),
        threshold=0.0,
        grid_offset=0.0,
    )
    assert probabilities.push[0] > 0.0
    assert 0.0 < conditional_upper_probability(probabilities)[0] < 1.0


def test_push_prior_for_half_point_lines_is_zero():
    prior = EmpiricalPushPrior(prior_strength=10.0).fit(
        lines=[-3.0, -3.0, -7.0, -7.0],
        pushes=[1, 0, 1, 0],
    )
    predicted = prior.predict([-3.5, -7.5, -3.0])
    assert predicted[0] == 0.0
    assert predicted[1] == 0.0
    assert predicted[2] > 0.0


def test_fixed_offset_logistic_returns_finite_probabilities():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 2))
    baseline = np.full(300, 0.5)
    logits = 0.8 * x[:, 0] - 0.4 * x[:, 1]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model = FixedOffsetLogistic(alpha=0.1).fit(x, y, baseline)
    probability = model.predict_upper_probability(x, baseline)
    assert np.isfinite(probability).all()
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.corrcoef(probability, logits)[0, 1] > 0.8
