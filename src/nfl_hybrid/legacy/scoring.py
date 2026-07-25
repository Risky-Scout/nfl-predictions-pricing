from __future__ import annotations

from dataclasses import dataclass

from nfl_hybrid.constants import (
    LEGACY_BASE_WEIGHT,
    LEGACY_H2H_WEIGHT,
    WIND_INTERCEPT,
    WIND_SLOPE,
)


@dataclass(frozen=True)
class LegacyScorePrediction:
    expected_home_points: float
    expected_away_points: float
    base_total: float
    blended_total: float
    wind_factor: float
    adjusted_total: float


def legacy_wind_factor(wind_mph: float, *, floor: float = 0.65, ceiling: float = 1.05) -> float:
    """Continuous repair of the workbook's unsafe approximate VLOOKUP."""
    raw = (WIND_INTERCEPT + WIND_SLOPE * float(wind_mph)) / WIND_INTERCEPT
    return min(max(raw, floor), ceiling)


def predict_legacy_score(
    league_home_points: float,
    league_away_points: float,
    home_attack: float,
    away_attack: float,
    home_defense_weakness: float,
    away_defense_weakness: float,
    *,
    h2h_total: float | None = None,
    wind_mph: float = 0.0,
    h2h_weight: float = LEGACY_H2H_WEIGHT,
) -> LegacyScorePrediction:
    expected_home = (
        float(league_home_points)
        * float(home_attack)
        * float(away_defense_weakness)
    )
    expected_away = (
        float(league_away_points)
        * float(away_attack)
        * float(home_defense_weakness)
    )
    base_total = expected_home + expected_away

    if h2h_total is None:
        blended = base_total
    else:
        blended = (1.0 - h2h_weight) * base_total + h2h_weight * float(h2h_total)

    wind_factor = legacy_wind_factor(wind_mph)
    adjusted_total = blended * wind_factor

    return LegacyScorePrediction(
        expected_home_points=expected_home,
        expected_away_points=expected_away,
        base_total=base_total,
        blended_total=blended,
        wind_factor=wind_factor,
        adjusted_total=adjusted_total,
    )
