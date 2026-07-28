from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.distributional.rare_event_shrinkage import (
    RareEventConfig,
    run_rare_event_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate chronological ATS push, moneyline tie, and total push "
            "shrinkage candidates using 2020 warm-up and 2021-2023 development."
        )
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--warehouse-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    result = run_rare_event_backtest(
        canonical_root=args.canonical_root,
        warehouse_path=args.warehouse_path,
        output_root=args.output_root,
        config=RareEventConfig.from_json(args.config),
    )

    print("=" * 110)
    print("RARE-EVENT SHRINKAGE SCORECARD")
    print("=" * 110)
    print(result["scorecard"].to_string(index=False))
    print()
    print("=" * 110)
    print("RARE-EVENT INTEGRITY AUDIT")
    print("=" * 110)
    print(result["integrity"].to_string(index=False))
    print()
    print("RARE-EVENT SHRINKAGE BACKTEST PASSED")


if __name__ == "__main__":
    main()
