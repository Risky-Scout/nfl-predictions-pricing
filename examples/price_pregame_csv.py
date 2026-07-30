from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.pricing.production import (
    PricingPolicy,
    price_csv_frame,
    verify_production_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Price NFL pregame selections using the frozen "
            "2025 production specification."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    production_spec = verify_production_spec(
        args.production_spec
    )
    policy = PricingPolicy.from_json(args.policy)

    input_frame = pd.read_csv(args.input_csv)
    output_frame = price_csv_frame(
        input_frame,
        production_spec=production_spec,
        policy=policy,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_frame.to_csv(
        args.output_csv,
        index=False,
    )

    print("=" * 100)
    print("PRODUCTION PRICING SUMMARY")
    print("=" * 100)

    summary = (
        output_frame.groupby(
            [
                "market_normalized",
                "decision",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "rows"})
    )

    print(summary.to_string(index=False))
    print(f"\nRows priced: {len(output_frame)}")
    print(f"Output: {args.output_csv}")
    print("PRODUCTION PRICING COMPLETED")


if __name__ == "__main__":
    main()
