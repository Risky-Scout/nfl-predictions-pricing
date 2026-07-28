from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.distributional_tournament import (
    DistributionalTournamentConfig,
    run_distributional_tournament,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run corrected fixed-market-offset and discrete score/margin/total "
            "architecture selection on 2021-2023 only."
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
    print("=" * 88)
    print("CORRECTED MARKET-OFFSET + DISCRETE DISTRIBUTIONAL TOURNAMENT")
    print("=" * 88)
    print("Selection seasons: 2021-2023")
    print("Untouched: 2024 architecture confirmation, 2025 final test")
    print("Pushes and ties: included in three-way probability scoring")
    print()

    _, aggregate, selected = run_distributional_tournament(
        args.compact_root,
        args.output_root,
        config=DistributionalTournamentConfig(
            bootstrap_repetitions=args.bootstrap_repetitions,
        ),
    )

    selected_columns = [
        "market",
        "variant",
        "selection_status",
        "architecture",
        "feature_set",
        "model_name",
        "feature_count",
        "pooled_binary_log_loss",
        "pooled_binary_brier",
        "pooled_three_way_log_loss",
        "pooled_three_way_brier",
        "mean_calibration_slope",
        "mean_predicted_push",
        "actual_push_rate",
        "gate_reasons",
    ]
    print("GATED DISTRIBUTIONAL SELECTIONS")
    print(selected[selected_columns].to_string(index=False))
    print()

    top = (
        aggregate.groupby(["market", "variant"], sort=True)
        .head(6)
        .reset_index(drop=True)
    )
    print("TOP RAW DISTRIBUTIONAL CANDIDATES")
    print(
        top[
            [
                "market",
                "variant",
                "architecture",
                "feature_set",
                "model_name",
                "feature_count",
                "pooled_binary_log_loss",
                "pooled_binary_brier",
                "pooled_three_way_log_loss",
                "pooled_three_way_brier",
            ]
        ].to_string(index=False)
    )
    print()
    print("CORRECTED DISTRIBUTIONAL TOURNAMENT PASSED")


if __name__ == "__main__":
    main()
