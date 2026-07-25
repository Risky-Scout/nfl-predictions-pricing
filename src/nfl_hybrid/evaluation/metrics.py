from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    root_mean_squared_error,
)


def probability_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    mask = np.isfinite(y) & np.isfinite(probability) & np.isin(y, [0.0, 1.0])
    if mask.sum() == 0:
        return {"n": 0, "brier": np.nan, "log_loss": np.nan}
    return {
        "n": int(mask.sum()),
        "brier": float(brier_score_loss(y[mask], probability[mask])),
        "log_loss": float(log_loss(y[mask], probability[mask], labels=[0, 1])),
    }


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan}
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(y[mask], pred[mask])),
        "rmse": float(root_mean_squared_error(y[mask], pred[mask])),
    }
