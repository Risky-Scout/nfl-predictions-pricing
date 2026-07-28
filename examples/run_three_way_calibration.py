from __future__ import annotations

import argparse
from pathlib import Path

from nfl_hybrid.calibration import (
    CalibrationConfig,
    calibrate_selected_distributional_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate selected NFL three-way probabilities."
    )
    parser.add_argument("--distributional-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metrics, bootstrap = calibrate_selected_distributional_models(
        args.distributional_root,
        args.output_root,
        config=CalibrationConfig(
            bootstrap_repetitions=args.bootstrap_repetitions,
        ),
    )

    print("=" * 80)
    print("EXPANDING THREE-WAY CALIBRATION")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print()
    print("CLUSTER BOOTSTRAP")
    print(bootstrap.to_string(index=False))
    print()
    print("THREE-WAY CALIBRATION PASSED")


if __name__ == "__main__":
    main()
