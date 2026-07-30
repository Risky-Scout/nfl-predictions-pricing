"""Calibration-drift detection on resolved games.

Two complementary signals are computed per market, in strict chronological
order over resolved (settled) games:

- **Rolling ECE** over a trailing window flags a calibration level that has
  degraded beyond a tolerance.
- A two-sided **CUSUM** on the calibration residual ``(probability - outcome)``
  detects a slow, systematic drift that a windowed average would miss.

When either alarm fires for a market, :func:`build_recalibration_request`
produces the trigger consumed by the existing calibration-adoption machinery
(:func:`nfl_hybrid.calibration.adoption.run_calibration_adoption_gate`), which
re-derives and re-gates calibrated probabilities. Detection never silently
adopts a new calibrator; it only requests that the audited gate be re-run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_hybrid.evaluation.market_relative import classwise_ece


@dataclass(frozen=True)
class DriftConfig:
    """Drift-detection thresholds.

    Rolling ECE is the primary *level* signal. The CUSUM operates on per-game
    calibration residuals ``(probability - outcome)``, whose magnitude is inherently
    ~0.5 for a binary event, so ``cusum_slack_k`` must be large enough to tolerate
    that in-control binary noise (a small slack would alarm on a balanced random
    walk). With these defaults a persistent systematic bias -- the failure mode
    that matters -- accumulates past ``cusum_threshold_h`` while calibrated
    coin-flip data stays quiet.
    """

    window: int = 64  # trailing resolved games for rolling ECE
    ece_bins: int = 10
    ece_alarm: float = 0.10  # rolling ECE above this triggers recalibration
    cusum_slack_k: float = 0.25  # slack absorbing in-control binary noise
    cusum_threshold_h: float = 4.0  # accumulated bias that raises an alarm
    minimum_resolved: int = 32  # do not alarm before enough games settle


def rolling_ece(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    window: int,
    n_bins: int = 10,
) -> np.ndarray:
    """Trailing-window ECE. Positions before ``window`` games return NaN."""

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    for i in range(n):
        if i + 1 < window:
            continue
        lo = i + 1 - window
        out[i] = classwise_ece(y[lo : i + 1], p[lo : i + 1], n_bins=n_bins)
    return out


def cusum_drift(
    residual: np.ndarray,
    *,
    slack_k: float,
    threshold_h: float,
) -> dict[str, np.ndarray]:
    """Two-sided CUSUM on a residual stream.

    ``residual`` is ``probability - outcome``; a persistent positive residual
    means the model is systematically over-predicting the event.
    """

    r = np.asarray(residual, dtype=float)
    s_pos = np.zeros(r.size)
    s_neg = np.zeros(r.size)
    alarm = np.zeros(r.size, dtype=bool)
    up = 0.0
    down = 0.0
    for i, value in enumerate(r):
        if not np.isfinite(value):
            value = 0.0
        up = max(0.0, up + value - slack_k)
        down = max(0.0, down - value - slack_k)
        s_pos[i] = up
        s_neg[i] = down
        if up > threshold_h or down > threshold_h:
            alarm[i] = True
            up = 0.0
            down = 0.0
    return {"cusum_pos": s_pos, "cusum_neg": s_neg, "alarm": alarm}


def monitor_calibration_drift(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    probability_col: str,
    market_col: str = "market",
    order_cols: tuple[str, ...] = ("season", "week"),
    config: DriftConfig | None = None,
) -> pd.DataFrame:
    """Per-game drift diagnostics with a per-market recalibration flag.

    Returns the input rows (sorted chronologically within market) augmented with
    ``rolling_ece``, ``cusum_pos``, ``cusum_neg``, ``drift_alarm`` and a
    per-market ``recalibration_triggered`` flag.
    """

    config = config or DriftConfig()
    required = {outcome_col, probability_col, market_col, *order_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"monitor_calibration_drift missing columns: {missing}")

    parts: list[pd.DataFrame] = []
    for market, sub in frame.groupby(market_col, sort=False):
        sub = sub.sort_values(list(order_cols), kind="stable").reset_index(drop=True)
        y = pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy(float)
        p = pd.to_numeric(sub[probability_col], errors="coerce").to_numpy(float)
        residual = p - y

        sub["rolling_ece"] = rolling_ece(
            y, p, window=config.window, n_bins=config.ece_bins
        )
        cusum = cusum_drift(
            residual,
            slack_k=config.cusum_slack_k,
            threshold_h=config.cusum_threshold_h,
        )
        sub["cusum_pos"] = cusum["cusum_pos"]
        sub["cusum_neg"] = cusum["cusum_neg"]

        enough = np.arange(len(sub)) + 1 >= config.minimum_resolved
        ece_alarm = np.nan_to_num(sub["rolling_ece"].to_numpy(), nan=0.0) > config.ece_alarm
        sub["drift_alarm"] = enough & (cusum["alarm"] | ece_alarm)
        sub["recalibration_triggered"] = bool(sub["drift_alarm"].any())
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


def build_recalibration_request(drift_report: pd.DataFrame) -> dict[str, object]:
    """Summarise which markets should be re-run through the adoption gate.

    The returned payload is intentionally small and serialisable so a scheduler
    can hand it to
    :func:`nfl_hybrid.calibration.adoption.run_calibration_adoption_gate`.
    """

    triggered = (
        drift_report.loc[
            drift_report["recalibration_triggered"].astype(bool), "market"
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    return {
        "recalibration_required": bool(triggered),
        "markets": sorted(triggered),
        "action": (
            "RUN_CALIBRATION_ADOPTION_GATE"
            if triggered
            else "NO_ACTION"
        ),
    }
