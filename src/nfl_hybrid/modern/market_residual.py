from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_hybrid.modern.joint_score import JointScoreConfig, JointScoreModel


@dataclass(frozen=True)
class MarketResidualConfig(JointScoreConfig):
    max_residual_multiplier: float = 1.25
    min_residual_multiplier: float = 0.0


class MarketResidualJointScoreModel(JointScoreModel):
    """Predict residuals around the timestamped market-implied margin and total.

    Canonical home spread uses sportsbook sign convention, so the market's
    expected home margin is `-home_spread`. The continuous targets are:

        margin_residual = actual_home_margin + home_spread
        total_residual = actual_total - total_line

    This model remains coherent across Moneyline, ATS, and totals because final
    mean margin and mean total are reconstructed before probabilities are
    derived.
    """

    def __init__(
        self,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str] = (),
        config: MarketResidualConfig | None = None,
    ) -> None:
        self.market_config = config or MarketResidualConfig()
        super().__init__(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=self.market_config,
        )
        self.margin_residual_multiplier_: float = 1.0
        self.total_residual_multiplier_: float = 1.0

    def _validate_market_frame(self, frame: pd.DataFrame) -> None:
        self._validate_training_frame(frame)
        for column in ("home_spread", "total_line"):
            if column not in frame:
                raise ValueError(f"Missing market anchor column: {column}")
            if frame[column].isna().any():
                raise ValueError(
                    f"{column} contains nulls. Market-residual training requires "
                    "a line at the prediction horizon."
                )

    def fit(
        self,
        train: pd.DataFrame,
        *,
        calibration: pd.DataFrame | None = None,
    ) -> "MarketResidualJointScoreModel":
        self._validate_market_frame(train)
        x_train = self.preprocessor.fit_transform(train[self.feature_columns])

        market_margin = -train["home_spread"].to_numpy(float)
        market_total = train["total_line"].to_numpy(float)
        margin_target = train["home_margin"].to_numpy(float) - market_margin
        total_target = train["total_points"].to_numpy(float) - market_total

        self.margin_model.fit(x_train, margin_target)
        self.total_model.fit(x_train, total_target)

        raw_margin_residual = self.margin_model.predict(x_train)
        raw_total_residual = self.total_model.predict(x_train)
        predicted_margin = market_margin + raw_margin_residual
        predicted_total = market_total + raw_total_residual
        self._set_residual_distribution(
            train["home_margin"].to_numpy(float) - predicted_margin,
            train["total_points"].to_numpy(float) - predicted_total,
        )
        self.is_fitted_ = True

        if calibration is not None and len(calibration):
            self.fit_residual_multipliers(calibration)
            # Re-estimate scale after calibration multipliers on training data.
            predicted_margin = market_margin + (
                self.margin_residual_multiplier_ * raw_margin_residual
            )
            predicted_total = market_total + (
                self.total_residual_multiplier_ * raw_total_residual
            )
            self._set_residual_distribution(
                train["home_margin"].to_numpy(float) - predicted_margin,
                train["total_points"].to_numpy(float) - predicted_total,
            )
            self.fit_calibrators(calibration)
        return self

    def _raw_residual_means(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Fit the model before prediction.")
        x = self.preprocessor.transform(frame[self.feature_columns])
        return self.margin_model.predict(x), self.total_model.predict(x)

    def fit_residual_multipliers(
        self,
        calibration: pd.DataFrame,
    ) -> "MarketResidualJointScoreModel":
        self._validate_market_frame(calibration)
        pred_margin_resid, pred_total_resid = self._raw_residual_means(calibration)
        actual_margin_resid = (
            calibration["home_margin"].to_numpy(float)
            + calibration["home_spread"].to_numpy(float)
        )
        actual_total_resid = (
            calibration["total_points"].to_numpy(float)
            - calibration["total_line"].to_numpy(float)
        )

        self.margin_residual_multiplier_ = self._slope_through_origin(
            pred_margin_resid,
            actual_margin_resid,
        )
        self.total_residual_multiplier_ = self._slope_through_origin(
            pred_total_resid,
            actual_total_resid,
        )
        return self

    def _slope_through_origin(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        valid = np.isfinite(predicted) & np.isfinite(actual)
        if valid.sum() < 20:
            return 1.0
        denominator = float(np.dot(predicted[valid], predicted[valid]))
        if denominator <= 1e-12:
            return 0.0
        slope = float(np.dot(predicted[valid], actual[valid]) / denominator)
        return float(
            np.clip(
                slope,
                self.market_config.min_residual_multiplier,
                self.market_config.max_residual_multiplier,
            )
        )

    def _means(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        margin_residual, total_residual = self._raw_residual_means(frame)
        market_margin = -frame["home_spread"].to_numpy(float)
        market_total = frame["total_line"].to_numpy(float)
        return (
            market_margin + self.margin_residual_multiplier_ * margin_residual,
            market_total + self.total_residual_multiplier_ * total_residual,
        )
