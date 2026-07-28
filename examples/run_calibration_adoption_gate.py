from __future__ import annotations
import argparse
from pathlib import Path
from nfl_hybrid.calibration import run_calibration_adoption_gate

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report, _ = run_calibration_adoption_gate(
        args.calibration_root, args.output_root
    )
    print(report[[
        "market","variant","decision",
        "three_way_log_loss_gain","three_way_log_loss_gain_ci_lower",
        "three_way_log_loss_gain_ci_upper",
        "three_way_brier_gain","three_way_brier_gain_ci_lower",
        "three_way_brier_gain_ci_upper",
    ]].to_string(index=False))
    print()
    print("SELECTIVE CALIBRATION GATE PASSED")

if __name__ == "__main__":
    main()
