from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.publishing.static_site import (
    build_static_site,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained static NFL pricing site "
            "from the frozen production outputs."
        )
    )
    parser.add_argument(
        "--pricing-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-decisions-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-scorecard-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-bootstrap-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--site-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    manifest = build_static_site(
        pricing_csv_path=args.pricing_csv,
        final_decisions_csv_path=
            args.final_decisions_csv,
        final_scorecard_csv_path=
            args.final_scorecard_csv,
        final_bootstrap_csv_path=
            args.final_bootstrap_csv,
        production_spec_path=args.production_spec,
        site_config_path=args.site_config,
        output_root=args.output_root,
    )

    print("=" * 100)
    print("STATIC SITE BUILD SUMMARY")
    print("=" * 100)
    print(f"Games: {manifest['games']}")
    print(f"Pricing rows: {manifest['pricing_rows']}")
    print(f"Game pages: {len(manifest['game_pages'])}")
    print(
        "Production model: "
        f"{manifest['production_model']}"
    )
    print(
        "Deployed to Wizard of Odds: "
        f"{manifest['deployed_to_wizard_of_odds']}"
    )
    print(f"Output: {args.output_root}")
    print("STATIC SITE PACKAGE BUILT")


if __name__ == "__main__":
    main()
