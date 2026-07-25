from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RosterAdjustmentModel:
    """Learn offseason continuity adjustments from historical season replays."""

    feature_columns: Iterable[str]
    target_column: str = "early_season_performance_residual"
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

    def __post_init__(self) -> None:
        self.feature_columns = list(self.feature_columns)
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        )
        self.residual_sd_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, offseason_history: pd.DataFrame) -> "RosterAdjustmentModel":
        required = set(self.feature_columns) | {self.target_column}
        missing = required - set(offseason_history.columns)
        if missing:
            raise ValueError(f"Roster adjustment history missing: {sorted(missing)}")
        x = offseason_history[self.feature_columns]
        y = offseason_history[self.target_column].astype(float)
        self.pipeline.fit(x, y)
        residual = y.to_numpy() - self.pipeline.predict(x)
        self.residual_sd_ = max(float(np.std(residual, ddof=1)), 0.05)
        self.is_fitted_ = True
        return self

    def predict(self, offseason_snapshot: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Fit the roster adjustment model first.")
        mean = self.pipeline.predict(offseason_snapshot[self.feature_columns])
        return pd.DataFrame(
            {
                "roster_adjustment_mean": mean,
                "roster_adjustment_standard_deviation": self.residual_sd_,
            },
            index=offseason_snapshot.index,
        )
