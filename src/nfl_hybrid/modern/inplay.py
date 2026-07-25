from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class InPlayRemainingScoreModel:
    """Provider-agnostic in-play remaining-points model.

    Required training labels:
      remaining_home_points = final_home_score - current_home_score
      remaining_away_points = final_away_score - current_away_score

    The supplied workbooks contain no play-level state, so this class is
    executable but intentionally not pretrained.
    """

    numeric_features: Iterable[str]
    categorical_features: Iterable[str] = ()
    random_state: int = 42

    def __post_init__(self) -> None:
        self.numeric_features = list(self.numeric_features)
        self.categorical_features = list(self.categorical_features)
        self.feature_columns = self.numeric_features + self.categorical_features

        preprocess = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), self.numeric_features),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                    encoded_missing_value=-1,
                                ),
                            ),
                        ]
                    ),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )
        regressor_args = dict(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=23,
            min_samples_leaf=30,
            l2_regularization=1.5,
            random_state=self.random_state,
        )
        self.home_pipeline = Pipeline(
            [("preprocess", preprocess), ("model", HistGradientBoostingRegressor(**regressor_args))]
        )
        self.away_pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", HistGradientBoostingRegressor(**regressor_args)),
            ]
        )
        self.home_residual_sd_ = 7.0
        self.away_residual_sd_ = 7.0
        self.residual_correlation_ = 0.0
        self.is_fitted_ = False

    def fit(self, snapshots: pd.DataFrame) -> "InPlayRemainingScoreModel":
        required = set(self.feature_columns) | {
            "remaining_home_points",
            "remaining_away_points",
        }
        missing = required - set(snapshots.columns)
        if missing:
            raise ValueError(f"Missing in-play columns: {sorted(missing)}")

        x = snapshots[self.feature_columns]
        y_home = snapshots["remaining_home_points"].astype(float)
        y_away = snapshots["remaining_away_points"].astype(float)
        self.home_pipeline.fit(x, y_home)
        self.away_pipeline.fit(x, y_away)

        home_resid = y_home.to_numpy() - self.home_pipeline.predict(x)
        away_resid = y_away.to_numpy() - self.away_pipeline.predict(x)
        self.home_residual_sd_ = max(float(np.std(home_resid, ddof=1)), 1.0)
        self.away_residual_sd_ = max(float(np.std(away_resid, ddof=1)), 1.0)
        self.residual_correlation_ = float(
            np.clip(np.corrcoef(home_resid, away_resid)[0, 1], -0.95, 0.95)
        )
        self.is_fitted_ = True
        return self

    def predict_remaining_points(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Fit the in-play model before prediction.")
        x = snapshots[self.feature_columns]
        return pd.DataFrame(
            {
                "remaining_home_points_mean": np.maximum(self.home_pipeline.predict(x), 0.0),
                "remaining_away_points_mean": np.maximum(self.away_pipeline.predict(x), 0.0),
            },
            index=snapshots.index,
        )
