from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Iterable


def brier_error(probability: float, outcome: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    if outcome not in (0.0, 1.0):
        raise ValueError("outcome must be 0 or 1.")
    return (float(probability) - float(outcome)) ** 2


def _excel_round(value: float, digits: int) -> float:
    quantum = Decimal("1").scaleb(-digits)
    return float(
        Decimal(str(value)).quantize(
            quantum,
            rounding=ROUND_HALF_UP,
        )
    )


def contest_points(
    probability: float,
    outcome: float,
    *,
    week: int | None = None,
    double_after_week_17: bool = True,
) -> float:
    """
    Spreadsheet contest transformation:
    round(25 - 100 * squared_error, 1), doubled after week 17.
    """
    points = _excel_round(
        25.0 - 100.0 * brier_error(probability, outcome),
        1,
    )
    if (
        double_after_week_17
        and week is not None
        and int(week) > 17
    ):
        points *= 2.0
    return points


def probability_average(*probabilities: float) -> float:
    if not probabilities:
        raise ValueError("At least one probability is required.")
    values = [float(value) for value in probabilities]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Probabilities must be in [0, 1].")
    return sum(values) / len(values)


def exponential_decay_weight(
    *,
    initial_weight: float,
    age: float,
    half_life: float = 360.0,
) -> float:
    if initial_weight < 0:
        raise ValueError("initial_weight cannot be negative.")
    if age < 0:
        raise ValueError("age cannot be negative.")
    if half_life <= 0:
        raise ValueError("half_life must be positive.")
    decay_rate = math.log(0.5) / float(half_life)
    return float(initial_weight) * math.exp(decay_rate * float(age))


def implied_team_points(
    *,
    total_line: float,
    home_spread: float,
) -> tuple[float, float]:
    """
    Negative home_spread means the home team is favored.

    implied margin = -home_spread
    home points = (total + implied margin) / 2
    away points = (total - implied margin) / 2
    """
    implied_margin = -float(home_spread)
    home_points = (float(total_line) + implied_margin) / 2.0
    away_points = (float(total_line) - implied_margin) / 2.0
    return home_points, away_points
