from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.final_test_2025 import (
    run_final_2025_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-time frozen 2025 NFL final evaluation. "
            "No tuning, feature changes, or recalibration are allowed."
        )
    )
    parser.add_argument(
        "--combined-canonical-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--warehouse-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--frozen-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rare-event-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-test-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--access-log",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
    )
    args = parser.parse_args()

    result = run_final_2025_evaluation(
        combined_canonical_root=args.combined_canonical_root,
        warehouse_path=args.warehouse_path,
        frozen_spec_path=args.frozen_spec,
        rare_event_config_path=args.rare_event_config,
        final_test_config_path=args.final_test_config,
        output_root=args.output_root,
        access_log_path=args.access_log,
        allow_final_test=args.allow_final_test,
    )

    print("=" * 125)
    print("2025 FINAL-TEST PRODUCTION DECISIONS")
    print("=" * 125)
    print(result["decisions"].to_string(index=False))

    print("\n" + "=" * 125)
    print("2025 FINAL-TEST SCORECARD")
    print("=" * 125)
    print(result["scorecard"].to_string(index=False))

    print("\n" + "=" * 125)
    print("2025 FINAL-TEST BOOTSTRAP")
    print("=" * 125)
    print(result["bootstrap"].to_string(index=False))

    print("\nRetuning permitted: False")
    print("2025 final evaluation completed: True")
    print("FROZEN 2025 FINAL EVALUATION PASSED")


if __name__ == "__main__":
    main()
