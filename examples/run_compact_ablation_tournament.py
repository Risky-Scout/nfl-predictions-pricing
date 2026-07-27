from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.compact_tournament import (
    TournamentConfig,
    run_compact_tournament,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chronological compact NFL feature-family ablation and "
            "model tournament using 2021-2023 only."
        )
    )
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("COMPACT MARKET-SPECIFIC ABLATION TOURNAMENT")
    print("=" * 80)
    print("Selection seasons: 2021-2023")
    print("Untouched: 2024 architecture selection, 2025 final test")
    print()

    _, aggregate, selected = run_compact_tournament(
        args.compact_root,
        args.output_root,
        config=TournamentConfig(),
    )

    display_columns = [
        "market",
        "variant",
        "feature_set",
        "model_family",
        "model_name",
        "feature_count",
        "mean_log_loss",
        "mean_brier",
        "mean_calibration_intercept",
        "mean_calibration_slope",
        "mean_ece",
        "log_loss_improvement_vs_baseline",
        "brier_improvement_vs_baseline",
    ]

    print("PROVISIONAL SELECTED CONFIGURATIONS")
    print(selected[display_columns].to_string(index=False))
    print()
    print("Top aggregate rows by market and variant:")
    top = (
        aggregate.groupby(["market", "variant"], sort=True)
        .head(5)
        .reset_index(drop=True)
    )
    print(
        top[
            [
                "market",
                "variant",
                "feature_set",
                "model_family",
                "model_name",
                "feature_count",
                "mean_log_loss",
                "mean_brier",
            ]
        ].to_string(index=False)
    )
    print()
    print("COMPACT ABLATION TOURNAMENT PASSED")


if __name__ == "__main__":
    main()
