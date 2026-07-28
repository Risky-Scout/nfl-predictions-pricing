from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamScoringProfile:
    attack_strength: float
    defense_weakness: float


def team_attack_strength(
    *,
    home_points_for: float,
    away_points_for: float,
    league_home_points_for: float,
    league_away_points_for: float,
) -> float:
    if league_home_points_for <= 0 or league_away_points_for <= 0:
        raise ValueError("League scoring averages must be positive.")
    return 0.5 * (
        float(home_points_for) / float(league_home_points_for)
        + float(away_points_for) / float(league_away_points_for)
    )


def team_defense_weakness(
    *,
    home_points_against: float,
    away_points_against: float,
    league_home_points_against: float,
    league_away_points_against: float,
) -> float:
    if league_home_points_against <= 0 or league_away_points_against <= 0:
        raise ValueError("League scoring averages must be positive.")
    return 0.5 * (
        float(home_points_against)
        / float(league_home_points_against)
        + float(away_points_against)
        / float(league_away_points_against)
    )


def predict_points(
    *,
    league_home_average: float,
    league_away_average: float,
    home_profile: TeamScoringProfile,
    away_profile: TeamScoringProfile,
) -> tuple[float, float, float, float]:
    home_points = (
        float(league_home_average)
        * home_profile.attack_strength
        * away_profile.defense_weakness
    )
    away_points = (
        float(league_away_average)
        * away_profile.attack_strength
        * home_profile.defense_weakness
    )
    return (
        home_points,
        away_points,
        home_points - away_points,
        home_points + away_points,
    )


def blend_base_and_head_to_head(
    base_total: float,
    head_to_head_total: float,
    *,
    base_weight: float = 0.70,
) -> float:
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base_weight must be in [0, 1].")
    return (
        float(base_total) * base_weight
        + float(head_to_head_total) * (1.0 - base_weight)
    )


def wind_multiplier(
    wind_mph: float,
    *,
    intercept: float = 45.973,
    slope_per_mph: float = -0.3787,
) -> float:
    if wind_mph < 0:
        raise ValueError("wind_mph cannot be negative.")
    if intercept <= 0:
        raise ValueError("intercept must be positive.")
    expected_points = intercept + slope_per_mph * float(wind_mph)
    return max(0.0, expected_points / intercept)
