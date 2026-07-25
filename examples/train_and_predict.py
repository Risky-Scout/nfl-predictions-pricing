from pathlib import Path

import pandas as pd

from nfl_hybrid.evaluation.walkforward import expanding_season_backtest
from nfl_hybrid.features.pregame import PregameFeatureBuilder
from nfl_hybrid.modern.joint_score import JointScoreModel


ROOT = Path(__file__).resolve().parents[1]
games = pd.read_csv(ROOT / "data" / "workbook_games_1989_2019.csv")
features = PregameFeatureBuilder().build(games)

numeric_features = [
    "playoff", "neutral_site", "home_spread", "total_line",
    "temperature", "wind_mph", "humidity", "dome",
    "home_elo", "away_elo", "elo_difference_adjusted",
    "legacy_home_win_probability", "legacy_expected_margin",
    "legacy_expected_home_points", "legacy_expected_away_points",
    "legacy_expected_total", "h2h_prior_games", "h2h_prior_total",
    "home_games_prior", "away_games_prior",
    "home_ewma_points_for", "home_ewma_points_against", "home_ewma_margin",
    "away_ewma_points_for", "away_ewma_points_against", "away_ewma_margin",
]
categorical_features = ["home_team_id", "away_team_id"]


def model_factory() -> JointScoreModel:
    return JointScoreModel(numeric_features, categorical_features)


predictions, metrics = expanding_season_backtest(
    features,
    model_factory=model_factory,
    test_seasons=[2018, 2019],
    calibration_seasons=1,
)
print(metrics)
predictions.to_csv(ROOT / "outputs" / "walkforward_predictions.csv", index=False)
metrics.to_csv(ROOT / "outputs" / "walkforward_metrics.csv", index=False)
