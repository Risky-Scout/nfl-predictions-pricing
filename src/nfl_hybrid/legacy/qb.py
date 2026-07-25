from __future__ import annotations

from nfl_hybrid.constants import QB_TO_ELO


def quarterback_value(
    attempts: float,
    completions: float,
    passing_yards: float,
    passing_tds: float,
    interceptions: float,
    sacks: float,
    rush_attempts: float,
    rush_yards: float,
    rush_tds: float,
) -> float:
    """Integrated-workbook QB value formula with the interception sign corrected."""
    return (
        -2.2 * attempts
        + 3.7 * completions
        + passing_yards / 5.0
        + 11.3 * passing_tds
        - 14.1 * interceptions
        - 8.0 * sacks
        - 1.1 * rush_attempts
        + 0.6 * rush_yards
        + 15.9 * rush_tds
    )


def opponent_adjusted_qb_value(
    raw_qb_value: float,
    opponent_qb_value_allowed: float,
    league_average_qb_value_allowed: float,
) -> float:
    return raw_qb_value + (
        league_average_qb_value_allowed - opponent_qb_value_allowed
    )


def blend_qb_value(prior_value: float, recent_value: float, recent_weight: float = 0.75) -> float:
    if not 0.0 <= recent_weight <= 1.0:
        raise ValueError("recent_weight must lie in [0, 1].")
    return (1.0 - recent_weight) * prior_value + recent_weight * recent_value


def qb_elo_adjustment(starter_qb_value: float, team_qb_value: float, multiplier: float = QB_TO_ELO) -> float:
    return multiplier * (starter_qb_value - team_qb_value)
