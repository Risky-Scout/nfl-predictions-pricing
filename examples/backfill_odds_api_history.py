from __future__ import annotations
import argparse
from pathlib import Path
from nfl_hybrid.odds_history import BackfillConfig, run_backfill

def parse_seasons(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seasons", type=parse_seasons, default=(2020,2021,2022,2023))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--maximum-requests", type=int)
    parser.add_argument("--estimated-cost-per-request", type=int, default=30)
    args = parser.parse_args()

    summary = run_backfill(
        args.games_path,
        args.output_root,
        BackfillConfig(
            seasons=args.seasons,
            plan_only=args.plan_only,
            maximum_requests=args.maximum_requests,
            estimated_cost_per_request=args.estimated_cost_per_request,
        ),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(
        "HISTORICAL ODDS BACKFILL PLAN PASSED"
        if args.plan_only
        else "HISTORICAL ODDS BACKFILL PASSED"
    )

if __name__ == "__main__":
    main()
