from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.chronological_spreadsheet_backtests import (
    BacktestConfig,
    run_chronological_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete chronological spreadsheet-equivalent NFL "
            "backtests with 2020 warm-up and 2021-2023 development."
        )
    )
    parser.add_argument("--warehouse-path", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qb-games-path", type=Path)
    args = parser.parse_args()

    result = run_chronological_backtest(
        warehouse_path=args.warehouse_path,
        canonical_root=args.canonical_root,
        backfill_root=args.backfill_root,
        output_root=args.output_root,
        config=BacktestConfig.from_json(args.config),
        qb_games_path=args.qb_games_path,
    )

    print("=" * 100)
    print("CHRONOLOGICAL SPREADSHEET BASELINE SCORECARD")
    print("=" * 100)
    print(result["scorecard"].to_string(index=False))
    print()
    print("=" * 100)
    print("LEAKAGE AUDIT")
    print("=" * 100)
    print(result["leakage_audit"].to_string(index=False))
    print()
    print("CHRONOLOGICAL SPREADSHEET BACKTEST PASSED")


if __name__ == "__main__":
    main()
