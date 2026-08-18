"""Walk-forward through 2025 + market-relative evaluation on the real backfill.

Builds a leakage-safe, market-anchored feature matrix from the 2020-2025
canonical backfill (reference closing lines from Spreadspoke, final scores from
nflverse), runs an expanding-season walk-forward with test seasons through 2025
(training strictly on prior seasons), and scores every market against the
de-vigged reference market -- not a coin flip.

The features here are the reference closing spread and total only. This is
deliberately a *market-anchored* demonstration: it measures whether wrapping the
market line in the joint-score model beats simply using the market. It is NOT
the full frozen tournament feature pipeline, and every market row is labelled
REFERENCE-LINE PROXY.

Run:  PYTHONPATH=src python examples/walkforward_market_relative_2025.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_hybrid.data.external_data import resolve
from nfl_hybrid.evaluation.market_relative import (
    MarketRelativeConfig,
    evaluate_market_relative,
)
from nfl_hybrid.evaluation.walkforward import expanding_season_backtest
from nfl_hybrid.governance.readiness import classify_market_relative_readiness
from nfl_hybrid.modern.joint_score import JointScoreModel

OUT = Path("outputs")
MARGIN_SD = 13.5


def build_feature_matrix() -> pd.DataFrame:
    g = pd.read_parquet(resolve("backfill.games")).copy()
    g["home_spread"] = pd.to_numeric(g["home_spread_reference"], errors="coerce")
    g["total_line"] = pd.to_numeric(g["total_line_reference"], errors="coerce")
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")

    # keep only resolved games with both a reference spread and total
    g = g[
        g["home_score"].notna()
        & g["away_score"].notna()
        & g["home_spread"].notna()
        & g["total_line"].notna()
    ].copy()

    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    g["home_win"] = (g["home_margin"] > 0).astype(int)
    g["home_cover"] = (g["home_margin"] + g["home_spread"] > 0).astype(int)
    g["over"] = (g["total_points"] > g["total_line"]).astype(int)

    # market-baseline "legacy" reference columns for the built-in comparisons
    g["legacy_expected_margin"] = -g["home_spread"]
    g["legacy_expected_total"] = g["total_line"]
    g["legacy_home_win_probability"] = 1.0 - norm.cdf(
        (0.0 - (-g["home_spread"])) / MARGIN_SD
    )

    # week -> integer ordinal (explicit ordering; never string-sort)
    week_order = {str(w): w for w in range(1, 19)}
    week_order.update({"Wildcard": 19, "Division": 20, "Conference": 21, "Superbowl": 22})
    g["week"] = g["week"].astype(str).map(week_order).fillna(23).astype(int)

    g = g.sort_values(["season", "week"], kind="stable").reset_index(drop=True)
    g["game_index"] = np.arange(len(g))
    return g


def main() -> None:
    features = build_feature_matrix()
    seasons = sorted(features["season"].unique().tolist())
    print(f"feature matrix: {len(features)} games, seasons {seasons}")

    def factory() -> JointScoreModel:
        return JointScoreModel(numeric_features=["home_spread", "total_line"])

    test_seasons = [s for s in seasons if s >= 2022]  # through 2025
    predictions, metrics = expanding_season_backtest(
        features,
        model_factory=factory,
        test_seasons=test_seasons,
        calibration_seasons=1,
    )
    print(f"walk-forward predictions: {len(predictions)} rows, test seasons {test_seasons}")

    result = evaluate_market_relative(
        predictions, config=MarketRelativeConfig(bootstrap_repetitions=2000)
    )
    prob = result["probability_scorecard"]
    margin = result["margin_scorecard"]
    readiness = classify_market_relative_readiness(prob)

    OUT.mkdir(exist_ok=True)
    predictions.to_csv(OUT / "walkforward_2025_predictions.csv", index=False)
    metrics.to_csv(OUT / "walkforward_2025_metrics.csv", index=False)
    prob.to_csv(OUT / "market_relative_probability_scorecard.csv", index=False)
    margin.to_csv(OUT / "market_relative_margin_scorecard.csv", index=False)
    readiness.to_csv(OUT / "market_relative_readiness.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== POOLED market-relative probability scorecard ===")
    cols = [
        "market", "n", "model_brier", "market_brier", "brier_gain_vs_market",
        "model_log_loss", "market_log_loss", "log_loss_gain_vs_market",
        "pick_accuracy", "pick_accuracy_ci_lower", "pick_accuracy_ci_upper",
        "model_ece", "market_ece", "beats_breakeven_ci",
    ]
    print(prob[prob["segment"] == "pooled"][cols].to_string(index=False))
    print("\n=== POOLED margin / CLV proxy ===")
    print(
        margin[margin["segment"] == "pooled"][
            ["market", "n", "model_mae", "market_mae", "mae_gain_vs_market", "fraction_model_beats_market"]
        ].to_string(index=False)
    )
    print("\n=== readiness vs MARKET ===")
    print(readiness.to_string(index=False))


if __name__ == "__main__":
    main()
