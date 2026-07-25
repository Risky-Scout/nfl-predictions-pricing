from __future__ import annotations

from dataclasses import dataclass


def american_to_implied_probability(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot equal zero.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


@dataclass(frozen=True)
class DevigResult:
    home_probability: float
    away_probability: float
    overround: float


def devig_two_way(home_odds: float, away_odds: float) -> DevigResult:
    p_home = american_to_implied_probability(home_odds)
    p_away = american_to_implied_probability(away_odds)
    overround = p_home + p_away
    if overround <= 0:
        raise ValueError("Invalid two-way market overround.")
    return DevigResult(
        home_probability=p_home / overround,
        away_probability=p_away / overround,
        overround=overround,
    )


def implied_team_scores(home_spread: float, total_line: float) -> tuple[float, float]:
    """Home spread uses sportsbook sign convention: favorites are negative."""
    home_points = (float(total_line) - float(home_spread)) / 2.0
    away_points = (float(total_line) + float(home_spread)) / 2.0
    return home_points, away_points
