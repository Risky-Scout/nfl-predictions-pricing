from pathlib import Path

import pandas as pd

from nfl_hybrid.evaluation.metrics import regression_metrics
from nfl_hybrid.evaluation.walkforward import expanding_season_backtest
from nfl_hybrid.features.pregame import PregameFeatureBuilder
from nfl_hybrid.modern.joint_score import JointScoreModel
from nfl_hybrid.modern.market_residual import MarketResidualJointScoreModel


ROOT = Path(__file__).resolve().parents[1]
games = pd.read_csv(ROOT / "data" / "workbook_games_1989_2019.csv")
features = PregameFeatureBuilder().build(games)

football_features = [
    "playoff",
    "neutral_site",
    "temperature",
    "wind_mph",
    "humidity",
    "dome",
    "home_elo",
    "away_elo",
    "elo_difference_adjusted",
    "legacy_home_win_probability",
    "legacy_expected_margin",
    "legacy_expected_home_points",
    "legacy_expected_away_points",
    "legacy_expected_total",
    "h2h_prior_games",
    "h2h_prior_total",
    "home_games_prior",
    "away_games_prior",
    "home_ewma_points_for",
    "home_ewma_points_against",
    "home_ewma_margin",
    "away_ewma_points_for",
    "away_ewma_points_against",
    "away_ewma_margin",
]
market_features = ["home_spread", "total_line"]
categorical_features = ["home_team_id", "away_team_id"]

model_specs = {
    "joint_score": lambda: JointScoreModel(
        football_features + market_features,
        categorical_features,
    ),
    "market_residual": lambda: MarketResidualJointScoreModel(
        football_features,
        categorical_features,
    ),
}

all_predictions = []
all_metrics = []
for model_name, factory in model_specs.items():
    predictions, metrics = expanding_season_backtest(
        features,
        model_factory=factory,
        test_seasons=[2018, 2019],
        calibration_seasons=1,
    )
    predictions.insert(0, "model", model_name)
    metrics.insert(0, "model", model_name)
    all_predictions.append(predictions)
    all_metrics.append(metrics)

predictions = pd.concat(all_predictions, ignore_index=True)
metrics = pd.concat(all_metrics, ignore_index=True)

market_rows = []
for season, frame in features[features["season"].isin([2018, 2019])].groupby("season"):
    margin_metrics = regression_metrics(
        frame["home_margin"].to_numpy(),
        (-frame["home_spread"]).to_numpy(),
    )
    total_metrics = regression_metrics(
        frame["total_points"].to_numpy(),
        frame["total_line"].to_numpy(),
    )
    market_rows.extend(
        [
            {
                "model": "market_anchor",
                "test_season": int(season),
                "market": "margin",
                **margin_metrics,
            },
            {
                "model": "market_anchor",
                "test_season": int(season),
                "market": "points_total",
                **total_metrics,
            },
        ]
    )
metrics = pd.concat([metrics, pd.DataFrame(market_rows)], ignore_index=True)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
predictions.to_csv(output_dir / "phase_c_model_predictions.csv", index=False)
metrics.to_csv(output_dir / "phase_c_model_metrics.csv", index=False)

selected = metrics[
    metrics["market"].isin(["moneyline", "ats", "total", "margin", "points_total"])
].sort_values(["test_season", "market", "model"])
print(selected.to_string(index=False))
