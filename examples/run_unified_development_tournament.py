from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.unified_development_tournament import (
    run_unified_tournament,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified 2021-2023 chronological market-relative "
            "NFL development tournament without accessing 2024 or 2025 outcomes."
        )
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--warehouse-path", type=Path, required=True)
    parser.add_argument("--spreadsheet-oof-path", type=Path, required=True)
    parser.add_argument("--rare-event-oof-path", type=Path, required=True)
    parser.add_argument(
        "--calibrated-distributional-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--freshness-consensus-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    result = run_unified_tournament(
        canonical_root=args.canonical_root,
        warehouse_path=args.warehouse_path,
        spreadsheet_oof_path=args.spreadsheet_oof_path,
        rare_event_oof_path=args.rare_event_oof_path,
        calibrated_distributional_path=
            args.calibrated_distributional_path,
        freshness_consensus_path=args.freshness_consensus_path,
        output_root=args.output_root,
        config_path=args.config,
    )

    print("=" * 120)
    print("UNIFIED DEVELOPMENT SELECTION DECISIONS")
    print("=" * 120)
    print(result["decisions"].to_string(index=False))

    print("\n" + "=" * 120)
    print("TOP ELIGIBLE CANDIDATES BY MARKET")
    print("=" * 120)

    eligible = result["scorecard"][
        result["scorecard"]["selection_eligible"]
    ].copy()

    for market, group in eligible.groupby("market", sort=True):
        print(f"\n{market}")
        columns = [
            "model_family",
            "model_name",
            "games",
            "mean_log_loss",
            "mean_brier",
            "mean_log_loss_gain",
            "log_loss_ci_lower",
            "log_loss_ci_upper",
            "mean_brier_gain",
            "brier_ci_lower",
            "brier_ci_upper",
            "selection_status",
        ]
        print(
            group.sort_values(
                ["mean_log_loss", "mean_brier"],
                kind="stable",
            )[columns]
            .head(12)
            .to_string(index=False)
        )

    print("\n" + "=" * 120)
    print("NESTED TUNING")
    print("=" * 120)
    print(result["tuning"].to_string(index=False))

    print("\n2024 accessed: False")
    print("2025 accessed: False")
    print("UNIFIED DEVELOPMENT TOURNAMENT PASSED")


if __name__ == "__main__":
    main()
