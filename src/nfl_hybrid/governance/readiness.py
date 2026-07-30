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


def classify_market_relative_readiness(
    probability_scorecard: pd.DataFrame,
    *,
    segment: str = "pooled",
) -> pd.DataFrame:
    """Classify each market against the MARKET from a market-relative scorecard.

    Consumes the ``probability_scorecard`` produced by
    :func:`nfl_hybrid.evaluation.market_relative.evaluate_market_relative`.

    A market is:

    - ``STATISTICALLY_SUPPORTED`` only when it beats the market on log-loss,
      does not lose on Brier, *and* its pick accuracy 95% bootstrap CI clears the
      52.4% break-even line (``beats_breakeven_ci``);
    - ``PROVISIONAL_ONLY`` when the point estimates favour the model
      (positive log-loss gain, non-negative Brier gain) but the CI is not
      conclusive;
    - ``RETAIN_BASELINE`` otherwise. This is the honest default: matching the
      market is *not* beating it.
    """

    required = {
        "segment",
        "market",
        "log_loss_gain_vs_market",
        "brier_gain_vs_market",
    }
    missing = sorted(required - set(probability_scorecard.columns))
    if missing:
        raise ValueError(f"Market-relative scorecard missing columns: {missing}")

    frame = probability_scorecard[
        probability_scorecard["segment"] == segment
    ].copy()

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        ll_gain = _value(row, "log_loss_gain_vs_market")
        brier_gain = _value(row, "brier_gain_vs_market")
        beats_breakeven = bool(row.get("beats_breakeven_ci", False))

        supported = (
            np.isfinite(ll_gain)
            and ll_gain > 0
            and (not np.isfinite(brier_gain) or brier_gain >= 0)
            and beats_breakeven
        )
        provisional = (
            np.isfinite(ll_gain)
            and ll_gain > 0
            and (not np.isfinite(brier_gain) or brier_gain >= 0)
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
                "segment": segment,
                "readiness_status": status,
                "log_loss_gain_vs_market": ll_gain,
                "brier_gain_vs_market": brier_gain,
                "pick_accuracy": _value(row, "pick_accuracy"),
                "pick_accuracy_ci_lower": _value(row, "pick_accuracy_ci_lower"),
                "beats_breakeven_ci": beats_breakeven,
                "market_source": row.get("market_source", "UNKNOWN"),
            }
        )
    return pd.DataFrame(rows)
