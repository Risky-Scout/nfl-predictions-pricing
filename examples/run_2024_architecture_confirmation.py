from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.confirmation_2024 import (
    run_2024_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 2024 NFL architecture confirmation. "
            "This command never reads 2025 outcomes."
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
        "--final-development-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--freeze-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allow-2024-confirmation",
        action="store_true",
    )
    args = parser.parse_args()

    result = run_2024_confirmation(
        combined_canonical_root=args.combined_canonical_root,
        warehouse_path=args.warehouse_path,
        final_development_root=args.final_development_root,
        freeze_spec_path=args.freeze_spec,
        repo_root=args.repo_root,
        confirmation_config_path=args.confirmation_config,
        output_root=args.output_root,
        allow_confirmation=args.allow_2024_confirmation,
    )

    print("=" * 120)
    print("2024 CONFIRMATION DECISIONS")
    print("=" * 120)
    print(result["decisions"].to_string(index=False))

    print("\n" + "=" * 120)
    print("2024 CONFIRMATION SCORECARD")
    print("=" * 120)
    print(result["scorecard"].to_string(index=False))

    print("\n2024 accessed: True")
    print("2025 accessed: False")
    print("FROZEN 2024 ARCHITECTURE CONFIRMATION PASSED")


if __name__ == "__main__":
    main()
