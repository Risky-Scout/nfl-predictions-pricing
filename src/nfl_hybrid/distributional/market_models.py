from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


_EPSILON = 1e-12


@dataclass(frozen=True)
class ThreeWayProbabilities:
    """Probabilities for lower/under/away, exact push/tie, and upper/over/home."""

    lower: np.ndarray
    push: np.ndarray
    upper: np.ndarray

    def as_array(self) -> np.ndarray:
        matrix = np.column_stack(
            [
                np.asarray(self.lower, dtype=float),
                np.asarray(self.push, dtype=float),
                np.asarray(self.upper, dtype=float),
            ]
        )
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(matrix)) or np.any(row_sums <= 0):
            raise ValueError("Invalid three-way probabilities.")
        return matrix / row_sums


def conditional_upper_probability(
    probabilities: ThreeWayProbabilities,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    lower = np.asarray(probabilities.lower, dtype=float)
    upper = np.asarray(probabilities.upper, dtype=float)
    denominator = np.clip(lower + upper, epsilon, None)
    return np.clip(upper / denominator, epsilon, 1.0 - epsilon)


def empirical_discrete_probabilities(
    predicted_mean: Iterable[float],
    residual_samples: Iterable[float],
    threshold: Iterable[float] | float,
    *,
    grid_offset: Iterable[float] | float = 0.0,
    smoothing: float = 0.5,
) -> ThreeWayProbabilities:
    """Convert a mean prediction and empirical residuals into discrete outcomes.

    The simulated outcome is quantized to the valid score grid. NFL margins and
    totals use an integer grid. Market-residual targets use a row-specific grid
    offset so half-point lines cannot receive artificial push probability.
    """

    means = np.asarray(predicted_mean, dtype=float).reshape(-1)
    residuals = np.asarray(residual_samples, dtype=float).reshape(-1)
    if len(residuals) < 20:
        raise ValueError("At least 20 residual samples are required.")
    if not np.isfinite(means).all() or not np.isfinite(residuals).all():
        raise ValueError("Predicted means and residual samples must be finite.")

    thresholds = np.broadcast_to(
        np.asarray(threshold, dtype=float),
        means.shape,
    )
    offsets = np.broadcast_to(
        np.asarray(grid_offset, dtype=float),
        means.shape,
    )
    offsets = np.mod(offsets, 1.0)

    lower = np.empty(len(means), dtype=float)
    push = np.empty(len(means), dtype=float)
    upper = np.empty(len(means), dtype=float)

    for index, (mean, line, offset) in enumerate(
        zip(means, thresholds, offsets)
    ):
        simulated = mean + residuals
        quantized = np.rint(simulated - offset) + offset

        push_possible = bool(
            np.isclose(np.mod(line - offset, 1.0), 0.0, atol=1e-9)
        )
        lower_count = float(np.sum(quantized < line - 1e-9))
        upper_count = float(np.sum(quantized > line + 1e-9))

        if push_possible:
            push_count = float(np.sum(np.isclose(quantized, line, atol=1e-9)))
            denominator = len(quantized) + 3.0 * smoothing
            lower[index] = (lower_count + smoothing) / denominator
            push[index] = (push_count + smoothing) / denominator
            upper[index] = (upper_count + smoothing) / denominator
        else:
            denominator = len(quantized) + 2.0 * smoothing
            lower[index] = (lower_count + smoothing) / denominator
            push[index] = 0.0
            upper[index] = (upper_count + smoothing) / denominator

    result = ThreeWayProbabilities(lower=lower, push=push, upper=upper)
    normalized = result.as_array()
    return ThreeWayProbabilities(
        lower=normalized[:, 0],
        push=normalized[:, 1],
        upper=normalized[:, 2],
    )


class EmpiricalPushPrior:
    """Leakage-safe, line-specific push/tie prior with partial pooling."""

    def __init__(
        self,
        *,
        prior_strength: float = 50.0,
        smoothing: float = 0.5,
        always_push_possible: bool = False,
    ) -> None:
        if prior_strength < 0 or smoothing <= 0:
            raise ValueError("Invalid push-prior hyperparameters.")
        self.prior_strength = float(prior_strength)
        self.smoothing = float(smoothing)
        self.always_push_possible = bool(always_push_possible)
        self._global_rate: float | None = None
        self._line_stats: dict[float, tuple[int, int]] = {}

    @staticmethod
    def _key(values: np.ndarray) -> np.ndarray:
        return np.round(values * 2.0) / 2.0

    def _possible(self, lines: np.ndarray) -> np.ndarray:
        if self.always_push_possible:
            return np.ones(len(lines), dtype=bool)
        return np.isclose(lines, np.rint(lines), atol=1e-9)

    def fit(
        self,
        lines: Iterable[float],
        pushes: Iterable[float],
    ) -> "EmpiricalPushPrior":
        line_values = np.asarray(lines, dtype=float).reshape(-1)
        push_values = np.asarray(pushes, dtype=float).reshape(-1)
        if len(line_values) != len(push_values):
            raise ValueError("Line and push arrays must have equal length.")
        possible = self._possible(line_values)
        valid = (
            np.isfinite(line_values)
            & np.isfinite(push_values)
            & possible
        )
        if not valid.any():
            self._global_rate = self.smoothing / (2.0 * self.smoothing)
            self._line_stats = {}
            return self

        selected_pushes = push_values[valid].astype(int)
        total = int(len(selected_pushes))
        successes = int(selected_pushes.sum())
        self._global_rate = (
            successes + self.smoothing
        ) / (total + 2.0 * self.smoothing)

        self._line_stats = {}
        keys = self._key(line_values[valid])
        for key in np.unique(keys):
            mask = keys == key
            self._line_stats[float(key)] = (
                int(mask.sum()),
                int(selected_pushes[mask].sum()),
            )
        return self

    def predict(self, lines: Iterable[float]) -> np.ndarray:
        if self._global_rate is None:
            raise RuntimeError("EmpiricalPushPrior must be fitted first.")
        line_values = np.asarray(lines, dtype=float).reshape(-1)
        possible = self._possible(line_values)
        keys = self._key(line_values)
        output = np.zeros(len(line_values), dtype=float)

        for index, (line, key, allowed) in enumerate(
            zip(line_values, keys, possible)
        ):
            if not np.isfinite(line) or not allowed:
                output[index] = 0.0
                continue
            count, successes = self._line_stats.get(float(key), (0, 0))
            output[index] = (
                successes + self.prior_strength * self._global_rate
            ) / (count + self.prior_strength)
        return np.clip(output, 0.0, 0.25)


def combine_conditional_with_push(
    conditional_upper: Iterable[float],
    push_probability: Iterable[float],
    *,
    epsilon: float = 1e-9,
) -> ThreeWayProbabilities:
    conditional = np.clip(
        np.asarray(conditional_upper, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    push = np.clip(np.asarray(push_probability, dtype=float), 0.0, 0.999)
    nonpush = 1.0 - push
    return ThreeWayProbabilities(
        lower=nonpush * (1.0 - conditional),
        push=push,
        upper=nonpush * conditional,
    )


class FixedOffsetLogistic:
    """Penalized logistic regression with coefficient 1 fixed on market logit."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        max_iter: int = 1000,
        epsilon: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y: Iterable[int],
        baseline_probability: Iterable[float],
    ) -> "FixedOffsetLogistic":
        target = np.asarray(y, dtype=float).reshape(-1)
        baseline = np.clip(
            np.asarray(baseline_probability, dtype=float).reshape(-1),
            self.epsilon,
            1.0 - self.epsilon,
        )
        if len(target) != len(baseline):
            raise ValueError("Target and baseline lengths differ.")
        transformed = self.imputer.fit_transform(x)
        transformed = self.scaler.fit_transform(transformed)
        offset = np.log(baseline / (1.0 - baseline))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            coefficients = parameters[1:]
            linear = offset + intercept + transformed @ coefficients
            probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -35, 35)))
            loss = np.mean(
                np.logaddexp(0.0, linear) - target * linear
            )
            loss += 0.5 * self.alpha * float(coefficients @ coefficients)
            residual = probability - target
            gradient = np.empty_like(parameters)
            gradient[0] = float(residual.mean())
            gradient[1:] = (
                transformed.T @ residual / len(target)
                + self.alpha * coefficients
            )
            return float(loss), gradient

        initial = np.zeros(transformed.shape[1] + 1, dtype=float)
        result = minimize(
            fun=lambda parameters: objective(parameters)[0],
            x0=initial,
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(
                f"Fixed-offset logistic optimization failed: {result.message}"
            )
        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=float)
        return self

    def predict_upper_probability(
        self,
        x: np.ndarray,
        baseline_probability: Iterable[float],
    ) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("FixedOffsetLogistic must be fitted first.")
        baseline = np.clip(
            np.asarray(baseline_probability, dtype=float).reshape(-1),
            self.epsilon,
            1.0 - self.epsilon,
        )
        transformed = self.imputer.transform(x)
        transformed = self.scaler.transform(transformed)
        offset = np.log(baseline / (1.0 - baseline))
        linear = offset + self.intercept_ + transformed @ self.coef_
        return np.clip(
            1.0 / (1.0 + np.exp(-np.clip(linear, -35, 35))),
            self.epsilon,
            1.0 - self.epsilon,
        )
