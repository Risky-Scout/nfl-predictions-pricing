from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.selection.compact_tournament import (
    TournamentConfig,
    run_compact_tournament,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the chronological compact NFL feature-family ablation, "
            "integrity audit, baseline veto, OOF diagnostics, and bootstrap "
            "comparison using 2021-2023 only."
        )
    )
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("COMPACT MARKET-SPECIFIC INTEGRITY + ABLATION TOURNAMENT")
    print("=" * 80)
    print("Selection seasons: 2021-2023")
    print("Untouched: 2024 architecture selection, 2025 final test")
    print("Integrity gates: target audit, market anchor, baseline veto, stability")
    print()

    _, aggregate, selected = run_compact_tournament(
        args.compact_root,
        args.output_root,
        config=TournamentConfig(
            bootstrap_repetitions=args.bootstrap_repetitions,
        ),
    )

    display_columns = [
        "market",
        "variant",
        "selection_status",
        "feature_set",
        "model_family",
        "model_name",
        "market_anchor_mode",
        "feature_count",
        "mean_log_loss",
        "mean_brier",
        "mean_calibration_intercept",
        "mean_calibration_slope",
        "mean_ece",
        "log_loss_improvement_vs_baseline",
        "brier_improvement_vs_baseline",
        "gate_reasons",
    ]

    print("GATED PROVISIONAL SELECTIONS")
    print(selected[display_columns].to_string(index=False))
    print()

    top = (
        aggregate.groupby(["market", "variant"], sort=True)
        .head(5)
        .reset_index(drop=True)
    )
    print("TOP RAW AGGREGATE ROWS")
    print(
        top[
            [
                "market",
                "variant",
                "feature_set",
                "model_family",
                "model_name",
                "market_anchor_mode",
                "feature_count",
                "mean_log_loss",
                "mean_brier",
            ]
        ].to_string(index=False)
    )
    print()

    bootstrap_path = args.output_root / "bootstrap_metrics.csv"
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        print("PAIRED OOF BOOTSTRAP SUMMARY")
        print(
            bootstrap[
                [
                    "market",
                    "variant",
                    "role",
                    "mean_log_loss_gain",
                    "log_loss_gain_ci_lower",
                    "log_loss_gain_ci_upper",
                    "mean_brier_gain",
                    "brier_gain_ci_lower",
                    "brier_gain_ci_upper",
                ]
            ].to_string(index=False)
        )
        print()

    print("INTEGRITY-GATED COMPACT TOURNAMENT PASSED")


if __name__ == "__main__":
    main()
