from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from nfl_hybrid.governance import (
    build_lab_record,
    classify_distributional_readiness,
    validate_governance_contracts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--distributional-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    validation = validate_governance_contracts(args.config_root)
    print("GOVERNANCE CONTRACT VALIDATION:", validation["status"])

    bootstrap_path = args.distributional_root / "distributional_bootstrap.csv"
    selected_path = args.distributional_root / "distributional_selected.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    readiness = classify_distributional_readiness(bootstrap)
    readiness_path = args.output_root / "model_readiness.csv"
    readiness.to_csv(readiness_path, index=False)

    build_lab_record(
        args.repo_root,
        args.output_root / "lab_record.json",
        input_files=[bootstrap_path, selected_path, readiness_path],
        model_spec_path=args.config_root / "model_spec.yaml",
        validation_protocol_path=args.config_root / "validation_protocol.yaml",
        notes=(
            "Pre-2024 development evidence freeze. "
            "2024 and 2025 remain untouched."
        ),
    )

    print()
    print(readiness.to_string(index=False))
    print()
    print("GOVERNANCE + READINESS FREEZE PASSED")


if __name__ == "__main__":
    main()
