from __future__ import annotations

from dataclasses import dataclass
import math


ELO_SCALE = 400.0
POINTS_PER_ELO = 25.0
HOME_FIELD_ELO = 55.0
TRAVEL_ELO_PER_1000 = -4.0
REST_ELO = 25.0
QB_ELO_MULTIPLIER = 3.3
PLAYOFF_MULTIPLIER = 1.2
OFFSEASON_REGRESSION_FRACTION = 1.0 / 3.0


def elo_win_probability(
    elo_difference: float,
    *,
    scale: float = ELO_SCALE,
) -> float:
    """Spreadsheet formula: 1 / (1 + 10 ** (-elo_difference / 400))."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    return 1.0 / (1.0 + 10.0 ** (-float(elo_difference) / scale))


def elo_expected_margin(
    elo_difference: float,
    *,
    points_per_elo: float = POINTS_PER_ELO,
) -> float:
    """Spreadsheet formula: elo_difference / 25."""
    if points_per_elo <= 0:
        raise ValueError("points_per_elo must be positive.")
    return float(elo_difference) / points_per_elo


def travel_adjustment(
    travel_distance: float,
    *,
    elo_per_1000: float = TRAVEL_ELO_PER_1000,
) -> float:
    if travel_distance < 0:
        raise ValueError("travel_distance cannot be negative.")
    return (float(travel_distance) / 1000.0) * elo_per_1000


def home_field_adjustment(
    *,
    neutral_site: bool,
    home_field_elo: float = HOME_FIELD_ELO,
) -> float:
    return 0.0 if neutral_site else float(home_field_elo)


def rest_adjustment(
    had_rest_week: bool,
    *,
    rest_elo: float = REST_ELO,
) -> float:
    return float(rest_elo) if had_rest_week else 0.0


def adjusted_elo_difference(
    *,
    home_rating: float,
    away_rating: float,
    home_travel_distance: float = 0.0,
    away_travel_distance: float = 0.0,
    neutral_site: bool = False,
    home_had_rest_week: bool = False,
    away_had_rest_week: bool = False,
    home_qb_adjustment: float = 0.0,
    away_qb_adjustment: float = 0.0,
    is_playoff: bool = False,
    playoff_multiplier: float = PLAYOFF_MULTIPLIER,
) -> float:
    """
    Python baseline corresponding to the spreadsheet's adjusted Elo surface.

    Base difference + home field + travel difference + rest difference
    + quarterback-adjustment difference. The spreadsheet's playoff constant
    is applied multiplicatively to the completed difference.
    """
    difference = (
        float(home_rating)
        - float(away_rating)
        + home_field_adjustment(neutral_site=neutral_site)
        + travel_adjustment(home_travel_distance)
        - travel_adjustment(away_travel_distance)
        + rest_adjustment(home_had_rest_week)
        - rest_adjustment(away_had_rest_week)
        + float(home_qb_adjustment)
        - float(away_qb_adjustment)
    )
    if is_playoff:
        difference *= float(playoff_multiplier)
    return difference


def margin_of_victory_multiplier(
    scoring_margin: float,
    winner_elo_difference: float,
) -> float:
    """
    Spreadsheet/FiveThirtyEight Elo margin multiplier:

    ln(margin + 1) * 2.2 / (winner_elo_difference * 0.001 + 2.2)
    """
    if scoring_margin < 0:
        raise ValueError("scoring_margin must be nonnegative.")
    denominator = float(winner_elo_difference) * 0.001 + 2.2
    if denominator <= 0:
        raise ValueError("winner Elo difference produces invalid denominator.")
    return math.log(float(scoring_margin) + 1.0) * 2.2 / denominator


def elo_point_change(
    *,
    k_factor: float,
    actual_result: float,
    expected_probability: float,
    margin_multiplier: float = 1.0,
) -> float:
    if not 0.0 <= actual_result <= 1.0:
        raise ValueError("actual_result must be in [0, 1].")
    if not 0.0 <= expected_probability <= 1.0:
        raise ValueError("expected_probability must be in [0, 1].")
    return (
        float(k_factor)
        * (float(actual_result) - float(expected_probability))
        * float(margin_multiplier)
    )


def update_elo_pair(
    *,
    home_rating: float,
    away_rating: float,
    home_probability: float,
    home_result: float,
    k_factor: float,
    scoring_margin: float,
    winner_elo_difference: float,
) -> tuple[float, float]:
    multiplier = margin_of_victory_multiplier(
        scoring_margin,
        winner_elo_difference,
    )
    change = elo_point_change(
        k_factor=k_factor,
        actual_result=home_result,
        expected_probability=home_probability,
        margin_multiplier=multiplier,
    )
    return float(home_rating) + change, float(away_rating) - change


def offseason_regression(
    rating: float,
    league_mean: float,
    *,
    regression_fraction: float = OFFSEASON_REGRESSION_FRACTION,
) -> float:
    if not 0.0 <= regression_fraction <= 1.0:
        raise ValueError("regression_fraction must be in [0, 1].")
    return (
        float(rating) * (1.0 - regression_fraction)
        + float(league_mean) * regression_fraction
    )
