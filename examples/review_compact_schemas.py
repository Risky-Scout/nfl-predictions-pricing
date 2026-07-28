from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.selection.integrity import (
    audit_compact_targets,
    review_compact_schemas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review the exact compact feature/target schemas and audit "
            "Moneyline, ATS, and total target construction."
        )
    )
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warehouse-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = review_compact_schemas(
        args.compact_root,
        args.output_root,
    )
    audit, failures = audit_compact_targets(
        args.compact_root,
        args.output_root,
        warehouse_path=args.warehouse_path,
    )
    print("=" * 80)
    print("EXACT COMPACT SCHEMA REVIEW")
    print("=" * 80)
    print(review.to_string(index=False))
    print()
    print("=" * 80)
    print("TARGET AND MARKET-SOURCE AUDIT")
    print("=" * 80)
    print(audit.to_string(index=False))
    print()
    print(f"Failure rows: {len(failures)}")
    print("COMPACT SCHEMA AND TARGET AUDIT PASSED")


if __name__ == "__main__":
    main()
