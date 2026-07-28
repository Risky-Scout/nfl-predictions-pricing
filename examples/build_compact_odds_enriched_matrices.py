from __future__ import annotations
import argparse
from pathlib import Path
from nfl_hybrid.features.odds_attachment import build_all

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--odds-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    audit = build_all(
        args.compact_root,
        args.odds_root,
        args.output_root,
    )
    print(audit.to_string(index=False))
    print()
    print("COMPACT ODDS ATTACHMENT PASSED")

if __name__ == "__main__":
    main()
