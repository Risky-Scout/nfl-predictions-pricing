"""Run the pre-registered 2026 EPA tournament and score vs REAL-CLOSING.

Every candidate x market number is reported (winners and losers). Applies the
fixed promotion gate from docs/model-selection/candidates-2026/registry.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_hybrid.data.external_data import resolve
from nfl_hybrid.evaluation.market_relative import evaluate_market_relative, MarketRelativeConfig
from nfl_hybrid.features.augmented_matrix import build_augmented_feature_matrix
from nfl_hybrid.selection.epa_tournament import CANDIDATES, run_walk_forward

OUT = Path("outputs")
TEST_SEASONS = [2022, 2023, 2024]
MARKET_MAP = {"home_win": "moneyline", "home_cover": "ats", "over": "total"}


def main() -> None:
    games = pd.read_parquet(resolve("backfill.games"))
    pbp = pd.read_parquet(resolve("backfill.pbp"))
    matrix, manifest = build_augmented_feature_matrix(games, pbp)

    # reference-implied market probabilities (available all seasons) for C5 meta-feature
    matrix["ref_ml_home_prob"] = 1.0 - norm.cdf((0.0 - (-matrix["home_spread"])) / 13.5)
    matrix["ref_cover_prob"] = 0.5
    matrix["ref_over_prob"] = 0.5
    market_prob_cols = ["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"]

    features = manifest["all_features"]
    print(f"features ({len(features)}): {features}")

    bench = pd.read_parquet("outputs/real_closing_benchmark_2022_2024.parquet")

    results = run_walk_forward(
        matrix, features, test_seasons=TEST_SEASONS, market_prob_cols=market_prob_cols
    )

    all_rows = []
    for name in CANDIDATES:
        if name not in results:
            print(f"{name}: no predictions (skipped)")
            continue
        pred = results[name]
        merged = pred.merge(
            bench[[
                "game_id", "market_ml_home_probability", "market_cover_home_probability",
                "market_over_probability", "closing_home_spread", "closing_total_line",
            ]],
            on="game_id", how="inner",
        )
        # score vs REAL-CLOSING: use closing spread for margin MAE, closing probs as benchmark
        merged["home_spread"] = merged["closing_home_spread"]
        merged["total_line"] = merged["closing_total_line"]
        market_probs = merged[[
            "market_ml_home_probability", "market_cover_home_probability", "market_over_probability",
        ]].copy()
        market_probs["market_source"] = "REAL-CLOSING"
        merged = merged.drop(columns=[
            "market_ml_home_probability", "market_cover_home_probability",
            "market_over_probability", "closing_home_spread", "closing_total_line",
        ]).reset_index(drop=True)
        market_probs = market_probs.reset_index(drop=True)

        res = evaluate_market_relative(
            merged, config=MarketRelativeConfig(bootstrap_repetitions=2000),
            market_probabilities=market_probs,
        )
        prob = res["probability_scorecard"]
        pooled = prob[prob["segment"] == "pooled"].copy()
        pooled.insert(0, "candidate", name)
        pooled.insert(1, "benchmark", "REAL-CLOSING")
        all_rows.append(pooled)

    scorecard = pd.concat(all_rows, ignore_index=True)
    scorecard["market_name"] = scorecard["market"].map(MARKET_MAP)

    # pre-registered promotion gate
    def decide(row):
        ll = row["log_loss_gain_vs_market"] > 0
        br = row["brier_gain_vs_market"] >= 0
        if row["market_name"] in ("ats", "total"):
            acc = bool(row["beats_breakeven_ci"])
            return "PROVISIONAL_ONLY" if (ll and br and acc) else "RETAIN_BASELINE"
        return "PROVISIONAL_ONLY" if (ll and br) else "RETAIN_BASELINE"

    scorecard["promotion"] = scorecard.apply(decide, axis=1)

    cols = [
        "candidate", "market_name", "n", "model_brier", "market_brier", "brier_gain_vs_market",
        "model_log_loss", "market_log_loss", "log_loss_gain_vs_market",
        "pick_accuracy", "pick_accuracy_ci_lower", "pick_accuracy_ci_upper",
        "beats_breakeven_ci", "promotion",
    ]
    OUT.mkdir(exist_ok=True)
    scorecard[cols].to_csv(OUT / "epa_tournament_scorecard.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n=== FULL SCORECARD (every candidate x market, vs REAL-CLOSING) ===")
    print(scorecard[cols].to_string(index=False))
    promoted = scorecard[scorecard["promotion"] == "PROVISIONAL_ONLY"]
    print(f"\nPROMOTED (PROVISIONAL_ONLY): {len(promoted)} of {len(scorecard)} candidate-markets")
    if len(promoted):
        print(promoted[["candidate", "market_name", "log_loss_gain_vs_market", "brier_gain_vs_market", "pick_accuracy_ci_lower"]].to_string(index=False))

    summary = {
        "test_seasons": TEST_SEASONS,
        "benchmark": "REAL-CLOSING (de-vigged closing, 2022-2024)",
        "n_candidates": len(CANDIDATES),
        "n_candidate_markets": int(len(scorecard)),
        "n_promoted": int(len(promoted)),
        "promoted": promoted[["candidate", "market_name"]].to_dict("records"),
    }
    (OUT / "epa_tournament_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
