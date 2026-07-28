from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.markets.quote_freshness import (
    QuoteFreshnessConfig,
    run_quote_freshness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit bookmaker-level quote freshness and build a separate "
            "freshness-qualified historical consensus."
        )
    )
    parser.add_argument("--quotes-path", type=Path, required=True)
    parser.add_argument(
        "--current-consensus-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    result = run_quote_freshness_audit(
        quotes_path=args.quotes_path,
        current_consensus_path=args.current_consensus_path,
        output_root=args.output_root,
        config=QuoteFreshnessConfig.from_json(args.config),
    )

    print("=" * 110)
    print("CLOSING QUOTE-FRESHNESS COVERAGE")
    print("=" * 110)
    print(result["closing_coverage"].to_string(index=False))

    print("\n" + "=" * 110)
    print("CONSENSUS COMPARISON SUMMARY")
    print("=" * 110)

    comparison = result["comparison"]
    both = comparison[comparison["_merge"].eq("both")]

    if both.empty:
        print("No paired consensus rows.")
    else:
        print(
            both.groupby("market", as_index=False)
            .agg(
                paired_rows=("game_id", "size"),
                mean_book_count_delta=("book_count_delta", "mean"),
                mean_absolute_line_delta=(
                    "line_delta",
                    lambda values: values.abs().mean(),
                ),
                mean_absolute_probability_delta=(
                    "probability_delta",
                    lambda values: values.abs().mean(),
                ),
                maximum_absolute_probability_delta=(
                    "probability_delta",
                    lambda values: values.abs().max(),
                ),
            )
            .to_string(index=False)
        )

    print()
    print("QUOTE FRESHNESS AUDIT PASSED")


if __name__ == "__main__":
    main()
