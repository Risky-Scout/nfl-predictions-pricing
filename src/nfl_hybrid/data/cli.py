from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.backfill import BackfillConfig, HistoricalBackfill
from nfl_hybrid.data.io import read_tabular
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter


def _seasons(value: str) -> tuple[int, ...]:
    if "-" in value:
        first, last = value.split("-", 1)
        return tuple(range(int(first), int(last) + 1))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill NFL hybrid-model data with provider provenance."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seasons", type=_seasons, default=tuple(range(2020, 2026)))
    parser.add_argument("--spreadspoke-csv")
    parser.add_argument(
        "--skip-large-tables",
        action="store_true",
        help="Skip play-by-play and annual rosters.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("open-sources")

    odds = sub.add_parser("plan-odds")
    odds.add_argument("--games", required=True)

    run_odds = sub.add_parser("run-odds")
    run_odds.add_argument("--games", required=True)
    run_odds.add_argument("--max-credits", type=int, required=True)
    run_odds.add_argument("--confirm-cost", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BackfillConfig(
        output_dir=args.output_dir,
        seasons=args.seasons,
        spreadspoke_locator=args.spreadspoke_csv,
        include_large_nflverse_tables=not args.skip_large_tables,
        max_odds_credits=getattr(args, "max_credits", 0),
        confirm_odds_cost=getattr(args, "confirm_cost", False),
    )
    runner = HistoricalBackfill(config=config)

    if args.command == "open-sources":
        outputs = runner.run_open_sources()
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    games = read_tabular(args.games)
    if args.command == "plan-odds":
        plan = runner.plan_historical_odds(games)
        print(
            f"rows={len(plan)} unique_requests={plan.attrs['unique_request_count']} "
            f"estimated_credits={plan.attrs['estimated_credits']}"
        )
        return

    runner.odds = TheOddsAPIAdapter(OddsAPIConfig())
    path = runner.run_historical_odds(games)
    print(path)


if __name__ == "__main__":
    main()
