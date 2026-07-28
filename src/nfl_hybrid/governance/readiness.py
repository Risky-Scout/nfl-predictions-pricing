from __future__ import annotations

import numpy as np
import pandas as pd


def _value(row: pd.Series, name: str, default: float = np.nan) -> float:
    value = row.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_distributional_readiness(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    required = {"market", "variant", "role", "mean_binary_log_loss_gain"}
    missing = sorted(required - set(bootstrap.columns))
    if missing:
        raise ValueError(f"Bootstrap report missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for _, row in bootstrap.iterrows():
        mean_ll = _value(row, "mean_binary_log_loss_gain")
        ll_lower = _value(row, "binary_log_loss_gain_ci_lower")
        mean_brier = _value(row, "mean_binary_brier_gain", 0.0)
        brier_lower = _value(row, "binary_brier_gain_ci_lower")
        role = str(row["role"])

        supported = (
            role == "selected"
            and np.isfinite(ll_lower)
            and ll_lower > 0
            and (
                not np.isfinite(brier_lower)
                or brier_lower >= 0
            )
        )
        provisional = (
            role == "selected"
            and mean_ll > 0
            and mean_brier >= 0
        )

        if supported:
            status = "STATISTICALLY_SUPPORTED"
        elif provisional:
            status = "PROVISIONAL_ONLY"
        else:
            status = "RETAIN_BASELINE"

        rows.append(
            {
                "market": row["market"],
                "variant": row["variant"],
                "role": role,
                "readiness_status": status,
                "mean_binary_log_loss_gain": mean_ll,
                "binary_log_loss_gain_ci_lower": ll_lower,
                "binary_log_loss_gain_ci_upper": _value(
                    row, "binary_log_loss_gain_ci_upper"
                ),
                "mean_binary_brier_gain": mean_brier,
                "binary_brier_gain_ci_lower": brier_lower,
                "binary_brier_gain_ci_upper": _value(
                    row, "binary_brier_gain_ci_upper"
                ),
            }
        )
    return pd.DataFrame(rows)
