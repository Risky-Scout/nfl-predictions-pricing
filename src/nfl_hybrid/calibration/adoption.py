from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def build_calibration_adoption_report(bootstrap: pd.DataFrame) -> pd.DataFrame:
    required = {
        "market","variant",
        "three_way_log_loss_gain","three_way_log_loss_gain_ci_lower","three_way_log_loss_gain_ci_upper",
        "three_way_brier_gain","three_way_brier_gain_ci_lower","three_way_brier_gain_ci_upper",
    }
    missing = sorted(required - set(bootstrap.columns))
    if missing:
        raise ValueError(f"Calibration bootstrap missing columns: {missing}")
    report = bootstrap.copy()
    report["adopt_calibration"] = (
        report["three_way_log_loss_gain_ci_lower"].gt(0.0)
        & report["three_way_brier_gain_ci_lower"].ge(0.0)
    )
    report["decision"] = np.where(
        report["adopt_calibration"],
        "ADOPT_CALIBRATION",
        "RETAIN_UNCALIBRATED_PROBABILITIES",
    )
    report["decision_reason"] = np.where(
        report["adopt_calibration"],
        "POSITIVE_95_CI_FOR_THREE_WAY_LOG_LOSS_AND_NONNEGATIVE_BRIER_CI",
        "CALIBRATION_NOT_STATISTICALLY_SUPPORTED",
    )
    return report

def apply_calibration_adoption_gate(
    calibrated_oof: pd.DataFrame,
    adoption_report: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "market","variant",
        "model_lower_probability","model_push_probability","model_upper_probability",
        "model_conditional_upper_probability",
        "calibrated_lower_probability","calibrated_push_probability",
        "calibrated_upper_probability","calibrated_conditional_upper_probability",
    }
    missing = sorted(required - set(calibrated_oof.columns))
    if missing:
        raise ValueError(f"Calibrated OOF missing columns: {missing}")
    decisions = adoption_report[
        ["market","variant","adopt_calibration","decision","decision_reason"]
    ].drop_duplicates(["market","variant"])
    output = calibrated_oof.merge(
        decisions, on=["market","variant"], how="left", validate="many_to_one"
    )
    if output["adopt_calibration"].isna().any():
        raise ValueError("Missing calibration adoption decision.")
    adopted = output["adopt_calibration"].astype(bool)
    mapping = [
        ("lower_probability","model_lower_probability","calibrated_lower_probability"),
        ("push_probability","model_push_probability","calibrated_push_probability"),
        ("upper_probability","model_upper_probability","calibrated_upper_probability"),
        ("conditional_upper_probability","model_conditional_upper_probability",
         "calibrated_conditional_upper_probability"),
    ]
    for suffix, original, calibrated in mapping:
        output[f"production_{suffix}"] = np.where(
            adopted, output[calibrated], output[original]
        )
    total = (
        output["production_lower_probability"]
        + output["production_push_probability"]
        + output["production_upper_probability"]
    )
    if not np.allclose(total.to_numpy(float), 1.0, atol=1e-9):
        raise ValueError("Production probabilities do not sum to one.")
    return output

def run_calibration_adoption_gate(
    calibration_root: str | Path,
    output_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(calibration_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bootstrap = pd.read_csv(source / "three_way_calibration_bootstrap.csv")
    calibrated = pd.read_parquet(
        source / "calibrated_distributional_oof_predictions.parquet"
    )
    report = build_calibration_adoption_report(bootstrap)
    production = apply_calibration_adoption_gate(calibrated, report)
    report.to_csv(output / "calibration_adoption_report.csv", index=False)
    production.to_parquet(
        output / "production_calibrated_oof_predictions.parquet", index=False
    )
    return report, production
