from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import MutableMapping

from nfl_hybrid.constants import (
    BYE_ELO,
    ELO_INITIAL,
    ELO_K,
    ELO_REVERSION_KEEP,
    ELO_REVERSION_MEAN,
    ELO_SCALE,
    ELO_TO_MARGIN,
    HOME_FIELD_ELO,
    PLAYOFF_ELO_MULTIPLIER,
    TRAVEL_ELO_PER_1000_MILES,
)


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = ELO_INITIAL
    reversion_mean: float = ELO_REVERSION_MEAN
    reversion_keep: float = ELO_REVERSION_KEEP
    k_factor: float = ELO_K
    scale: float = ELO_SCALE
    elo_to_margin: float = ELO_TO_MARGIN
    home_field_elo: float = HOME_FIELD_ELO
    travel_elo_per_1000_miles: float = TRAVEL_ELO_PER_1000_MILES
    bye_elo: float = BYE_ELO
    playoff_multiplier: float = PLAYOFF_ELO_MULTIPLIER


@dataclass(frozen=True)
class EloContext:
    neutral_site: bool = False
    home_travel_miles: float = 0.0
    away_travel_miles: float = 0.0
    home_bye: bool = False
    away_bye: bool = False
    home_qb_adjustment: float = 0.0
    away_qb_adjustment: float = 0.0
    playoff: bool = False


@dataclass(frozen=True)
class EloPrediction:
    home_rating: float
    away_rating: float
    adjusted_difference: float
    home_win_probability: float
    expected_home_margin: float
    home_field_adjustment: float
    travel_adjustment: float
    rest_adjustment: float
    qb_adjustment: float


class LegacyElo:
    """Corrected replica of the adjusted-Elo logic in NFLMODEL.

    Corrections deliberately made:
    * unplayed games cannot update ratings;
    * travel is computed with the intended home-minus-away sign;
    * travel lookup is supplied explicitly rather than using the broken VBA lookup.
    """

    def __init__(
        self,
        ratings: MutableMapping[str, float] | None = None,
        config: EloConfig | None = None,
    ) -> None:
        self.config = config or EloConfig()
        self.ratings: MutableMapping[str, float] = ratings if ratings is not None else {}

    def rating(self, team_id: str) -> float:
        return float(self.ratings.get(team_id, self.config.initial_rating))

    def predict(self, home_team: str, away_team: str, context: EloContext | None = None) -> EloPrediction:
        ctx = context or EloContext()
        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)

        home_field = 0.0 if ctx.neutral_site else self.config.home_field_elo

        # A longer away trip should increase the home-minus-away rating difference.
        travel = self.config.travel_elo_per_1000_miles * (
            float(ctx.away_travel_miles) - float(ctx.home_travel_miles)
        ) / 1000.0

        rest = self.config.bye_elo * (int(ctx.home_bye) - int(ctx.away_bye))
        qb = float(ctx.home_qb_adjustment) - float(ctx.away_qb_adjustment)

        adjusted_difference = home_rating - away_rating + home_field + travel + rest + qb
        if ctx.playoff:
            adjusted_difference *= self.config.playoff_multiplier

        p_home = 1.0 / (1.0 + 10.0 ** (-adjusted_difference / self.config.scale))
        margin = adjusted_difference / self.config.elo_to_margin

        return EloPrediction(
            home_rating=home_rating,
            away_rating=away_rating,
            adjusted_difference=adjusted_difference,
            home_win_probability=p_home,
            expected_home_margin=margin,
            home_field_adjustment=home_field,
            travel_adjustment=travel,
            rest_adjustment=rest,
            qb_adjustment=qb,
        )

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: float | None,
        away_score: float | None,
        context: EloContext | None = None,
        *,
        completed: bool = True,
    ) -> float:
        if not completed or home_score is None or away_score is None:
            raise ValueError("Elo ratings may only be updated from a completed game with both scores.")

        prediction = self.predict(home_team, away_team, context)
        observed_margin = float(home_score) - float(away_score)

        if observed_margin > 0:
            actual = 1.0
            winner_adjusted_difference = prediction.adjusted_difference
        elif observed_margin < 0:
            actual = 0.0
            winner_adjusted_difference = -prediction.adjusted_difference
        else:
            actual = 0.5
            winner_adjusted_difference = 0.0

        denominator = 0.001 * winner_adjusted_difference + 2.2
        if denominator <= 0:
            denominator = 1e-9

        mov_multiplier = log(abs(observed_margin) + 1.0) * 2.2 / denominator
        delta = self.config.k_factor * (actual - prediction.home_win_probability) * mov_multiplier

        self.ratings[home_team] = prediction.home_rating + delta
        self.ratings[away_team] = prediction.away_rating - delta
        return delta

    def regress_to_mean(self) -> None:
        keep = self.config.reversion_keep
        mean = self.config.reversion_mean
        for team, rating in list(self.ratings.items()):
            self.ratings[team] = keep * float(rating) + (1.0 - keep) * mean
