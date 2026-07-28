from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.features.canonical_market_matrices import build_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role-config", type=Path, required=True)
    args = parser.parse_args()

    audit = build_all(
        enriched_root=args.enriched_root,
        output_root=args.output_root,
        role_config=args.role_config,
    )
    print(audit.to_string(index=False))
    print()
    print("CANONICAL MARKET HARDENING PASSED")


if __name__ == "__main__":
    main()
