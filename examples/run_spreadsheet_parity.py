from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from nfl_hybrid.spreadsheet_baselines import (
    SOURCE_WORKBOOKS,
    reference_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact Python parity checks against spreadsheet reference "
            "cells used as NFL model baselines."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in reference_cases():
        actual = float(case.calculate())
        error = abs(actual - case.expected)
        rows.append(
            {
                "baseline": case.baseline,
                "workbook": case.workbook,
                "sheet": case.sheet,
                "cell": case.cell,
                "expected": case.expected,
                "actual": actual,
                "absolute_error": error,
                "tolerance": case.tolerance,
                "status": "PASS" if error <= case.tolerance else "FAIL",
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(
        args.output_root / "spreadsheet_parity_report.csv",
        index=False,
    )

    manifest = {
        "status": (
            "PASS"
            if report["status"].eq("PASS").all()
            else "FAIL"
        ),
        "reference_case_count": len(report),
        "source_workbook_sha256": SOURCE_WORKBOOKS,
        "scope": [
            "elo probability and expected margin",
            "home/travel/rest/QB adjustment constants",
            "margin-of-victory Elo updating",
            "quarterback composite and updating",
            "multiplicative total-points baseline",
            "head-to-head and wind total adjustments",
            "Brier-derived market scoring",
            "360-observation half-life decay",
        ],
        "excluded_from_production_baseline": [
            "long-term season simulation",
            "manual contest heuristics",
            "macro user-interface code",
        ],
    }
    (
        args.output_root / "spreadsheet_parity_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SPREADSHEET-TO-PYTHON BASELINE PARITY")
    print("=" * 100)
    print(report.to_string(index=False))
    print()

    if not report["status"].eq("PASS").all():
        raise RuntimeError("Spreadsheet parity failed.")

    print("SPREADSHEET BASELINE PARITY PASSED")


if __name__ == "__main__":
    main()
