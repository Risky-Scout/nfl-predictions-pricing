from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .elo import (
    elo_expected_margin,
    elo_point_change,
    elo_win_probability,
    margin_of_victory_multiplier,
)
from .market import exponential_decay_weight
from .qb import (
    adjusted_qb_game_value,
    qb_game_value,
    starter_qb_adjustment,
    update_qb_rating,
)
from .totals import (
    TeamScoringProfile,
    blend_base_and_head_to_head,
    predict_points,
    wind_multiplier,
)


SOURCE_WORKBOOKS = {
    "ELOprimervideoaccompanyingfile-200904-092938.xlsx":
        "fb7fa5c40c82d44c59920ec9b7253909c6aad34b0d0d46055f149f086da53703",
    "NFLMODELSTUDENTS-200904-092938.xlsm":
        "78356787d73e02caeabb947f20624a3982fae56e8c44490f98218b956c034824",
    "QuarterbackadjustmentSTUDENTS-200904-092938.xlsx":
        "a983fbf8b9c85e2d015bd0a651de0d0673959879f59ed46a02097ed4f60deb71",
    "TotalPointsSTUDENTS-200904-092938.xlsx":
        "af6bf3cbf6867cb85a4728857e27b652d03da69834a1946251dd8a73eea1488c",
    "NFLscoresmarketcalibrationSTUDENTS-200904-092938.xlsx":
        "b155458c689063c702ab76c061981d89489ffabf8b285a99b07e3115ce06a54b",
    "FiveThirtyEight NFL Predictions Game Modeling by David Glidden.xlsx":
        "94f2cd83a134a158f4c9fcefc5a9e0efe24fd21e8ba4d300b47506e6e777a509",
}


@dataclass(frozen=True)
class ReferenceCase:
    baseline: str
    workbook: str
    sheet: str
    cell: str
    expected: float
    tolerance: float
    calculate: Callable[[], float]


def reference_cases() -> list[ReferenceCase]:
    kc_profile = TeamScoringProfile(
        attack_strength=1.239977423814897,
        defense_weakness=0.920292838580524,
    )
    hou_profile = TeamScoringProfile(
        attack_strength=0.941349399244411,
        defense_weakness=0.9939020774618309,
    )

    def total_home() -> float:
        return predict_points(
            league_home_average=23.647940074906366,
            league_away_average=21.768539325842696,
            home_profile=kc_profile,
            away_profile=hou_profile,
        )[0]

    def total_away() -> float:
        return predict_points(
            league_home_average=23.647940074906366,
            league_away_average=21.768539325842696,
            home_profile=kc_profile,
            away_profile=hou_profile,
        )[1]

    return [
        ReferenceCase(
            "elo_win_probability",
            "ELOprimervideoaccompanyingfile-200904-092938.xlsx",
            "1 event simple demo",
            "F11",
            0.6131368201531431,
            1e-12,
            lambda: elo_win_probability(80.0),
        ),
        ReferenceCase(
            "elo_win_probability",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "Elo-WinProb-Suprem",
            "C2",
            0.003152309183260209,
            1e-15,
            lambda: elo_win_probability(-1000.0),
        ),
        ReferenceCase(
            "elo_expected_margin",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "Calculations",
            "N13",
            13.02,
            1e-12,
            lambda: elo_expected_margin(325.5),
        ),
        ReferenceCase(
            "elo_mov_multiplier",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "Calculations",
            "Q14",
            2.4405288832732133,
            1e-12,
            lambda: margin_of_victory_multiplier(7.0, -325.5),
        ),
        ReferenceCase(
            "elo_point_change",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "Calculations",
            "T14",
            42.31335874842736,
            1e-10,
            lambda: elo_point_change(
                k_factor=20.0,
                actual_result=1.0,
                expected_probability=0.133110879399282,
                margin_multiplier=2.4405288832732133,
            ),
        ),
        ReferenceCase(
            "qb_game_value",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "QB_Rating",
            "O10",
            63.50000000000001,
            1e-12,
            lambda: qb_game_value(
                pass_attempts=24,
                completions=14,
                passing_yards=248,
                passing_touchdowns=3,
                interceptions=1,
                sacks=2,
                rush_attempts=3,
                rushing_yards=24,
                rushing_touchdowns=0,
            ),
        ),
        ReferenceCase(
            "qb_opponent_adjustment",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "QB_Rating",
            "O12",
            57.42000000000001,
            1e-12,
            lambda: adjusted_qb_game_value(
                63.50000000000001,
                league_average_qb_value_allowed=48.92,
                opponent_qb_value_allowed=55.0,
            ),
        ),
        ReferenceCase(
            "qb_rating_update",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "QB_Rating",
            "O14",
            78.561,
            1e-12,
            lambda: update_qb_rating(
                80.91,
                57.42000000000001,
            ),
        ),
        ReferenceCase(
            "qb_starter_adjustment",
            "NFLMODELSTUDENTS-200904-092938.xlsm",
            "QB_Rating",
            "L3",
            8.811000000000005,
            1e-12,
            lambda: starter_qb_adjustment(46.67, 44.0),
        ),
        ReferenceCase(
            "total_home_points",
            "TotalPointsSTUDENTS-200904-092938.xlsx",
            "Points",
            "L10",
            29.14410296778459,
            1e-12,
            total_home,
        ),
        ReferenceCase(
            "total_away_points",
            "TotalPointsSTUDENTS-200904-092938.xlsx",
            "Points",
            "L11",
            18.858458093504805,
            1e-12,
            total_away,
        ),
        ReferenceCase(
            "total_base",
            "TotalPointsSTUDENTS-200904-092938.xlsx",
            "Points",
            "L13",
            48.0025610612894,
            1e-12,
            lambda: total_home() + total_away(),
        ),
        ReferenceCase(
            "total_h2h_blend",
            "TotalPointsSTUDENTS-200904-092938.xlsx",
            "Points",
            "L26",
            49.65179274290257,
            1e-12,
            lambda: blend_base_and_head_to_head(
                total_home() + total_away(),
                53.5,
            ),
        ),
        ReferenceCase(
            "total_wind_adjustment",
            "TotalPointsSTUDENTS-200904-092938.xlsx",
            "Points",
            "L30",
            49.65179274290257,
            1e-12,
            lambda: blend_base_and_head_to_head(
                total_home() + total_away(),
                53.5,
            ) * wind_multiplier(0.0),
        ),
        ReferenceCase(
            "exponential_decay",
            "NFLscoresmarketcalibrationSTUDENTS-200904-092938.xlsx",
            "Sheet3",
            "D7",
            9.980764435756287,
            1e-12,
            lambda: exponential_decay_weight(
                initial_weight=10.0,
                age=1.0,
                half_life=360.0,
            ),
        ),
    ]
