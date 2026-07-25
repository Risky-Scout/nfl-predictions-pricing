from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from nfl_hybrid.evaluation.metrics import probability_metrics, regression_metrics
from nfl_hybrid.modern.joint_score import JointScoreModel


def expanding_season_backtest(
    features: pd.DataFrame,
    *,
    model_factory: Callable[[], JointScoreModel],
    test_seasons: list[int],
    calibration_seasons: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train only on seasons earlier than calibration and test seasons."""
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    seasons = sorted(int(x) for x in features["season"].unique())
    for test_season in test_seasons:
        prior = [s for s in seasons if s < test_season]
        if len(prior) <= calibration_seasons:
            continue

        calibration_set = set(prior[-calibration_seasons:])
        train_seasons = [s for s in prior if s not in calibration_set]
        train = features[features["season"].isin(train_seasons)].copy()
        calibration = features[features["season"].isin(calibration_set)].copy()
        test = features[features["season"] == test_season].copy()

        model = model_factory()
        model.fit(train, calibration=calibration)
        prediction = model.predict_markets(test)
        prediction = pd.concat(
            [
                test[
                    [
                        "game_index", "season", "week", "home_team_id", "away_team_id",
                        "home_score", "away_score", "home_margin", "total_points",
                        "home_win", "home_cover", "over", "home_spread", "total_line",
                        "legacy_home_win_probability", "legacy_expected_margin",
                        "legacy_expected_total",
                    ]
                ].reset_index(drop=True),
                prediction.reset_index(drop=True),
            ],
            axis=1,
        )
        prediction_frames.append(prediction)

        metric_sets = {
            "moneyline": probability_metrics(
                prediction["home_win"].to_numpy(),
                prediction["home_win_probability_no_tie"].to_numpy(),
            ),
            "ats": probability_metrics(
                prediction["home_cover"].to_numpy(),
                prediction["home_cover_probability_no_push"].to_numpy(),
            ),
            "total": probability_metrics(
                prediction["over"].to_numpy(),
                prediction["over_probability_no_push"].to_numpy(),
            ),
            "moneyline_legacy_elo": probability_metrics(
                prediction["home_win"].to_numpy(),
                prediction["legacy_home_win_probability"].to_numpy(),
            ),
            "ats_coin_flip": probability_metrics(
                prediction["home_cover"].to_numpy(),
                np.full(len(prediction), 0.5),
            ),
            "total_coin_flip": probability_metrics(
                prediction["over"].to_numpy(),
                np.full(len(prediction), 0.5),
            ),
            "margin": regression_metrics(
                prediction["home_margin"].to_numpy(),
                prediction["predicted_margin"].to_numpy(),
            ),
            "margin_legacy_elo": regression_metrics(
                prediction["home_margin"].to_numpy(),
                prediction["legacy_expected_margin"].to_numpy(),
            ),
            "points_total": regression_metrics(
                prediction["total_points"].to_numpy(),
                prediction["predicted_total"].to_numpy(),
            ),
            "points_total_legacy": regression_metrics(
                prediction["total_points"].to_numpy(),
                prediction["legacy_expected_total"].to_numpy(),
            ),
        }
        for market, metrics in metric_sets.items():
            metric_rows.append(
                {
                    "test_season": test_season,
                    "market": market,
                    **metrics,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics
