from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class JointScoreConfig:
    max_iter: int = 180
    learning_rate: float = 0.045
    max_leaf_nodes: int = 23
    min_samples_leaf: int = 24
    l2_regularization: float = 1.0
    random_state: int = 42
    probability_epsilon: float = 1e-6


class _ProbabilityCalibrator:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.model: LogisticRegression | None = None

    def fit(self, raw_probability: np.ndarray, y: np.ndarray) -> "_ProbabilityCalibrator":
        p = np.clip(np.asarray(raw_probability, dtype=float), self.epsilon, 1.0 - self.epsilon)
        target = np.asarray(y, dtype=float)
        mask = np.isfinite(p) & np.isfinite(target)
        if mask.sum() < 30 or np.unique(target[mask]).size < 2:
            self.model = None
            return self
        x = logit(p[mask]).reshape(-1, 1)
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.model.fit(x, target[mask])
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probability, dtype=float), self.epsilon, 1.0 - self.epsilon)
        if self.model is None:
            return p
        return self.model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class JointScoreModel:
    """Market-consistent pregame model.

    The model predicts mean margin and mean total. Residual standard deviations
    and correlation define a joint distribution. Moneyline, ATS, totals, and
    team-score probabilities are derived from that one distribution.
    """

    def __init__(
        self,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str] = (),
        config: JointScoreConfig | None = None,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.config = config or JointScoreConfig()

        categorical_pipeline = Pipeline(
            steps=[
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
        )
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        model_kwargs = dict(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )
        self.margin_model = HistGradientBoostingRegressor(**model_kwargs)
        self.total_model = HistGradientBoostingRegressor(**model_kwargs)

        self.margin_sd_: float = 13.5
        self.total_sd_: float = 13.5
        self.residual_correlation_: float = 0.0

        self.ml_calibrator = _ProbabilityCalibrator(self.config.probability_epsilon)
        self.ats_calibrator = _ProbabilityCalibrator(self.config.probability_epsilon)
        self.ou_calibrator = _ProbabilityCalibrator(self.config.probability_epsilon)
        self.is_fitted_: bool = False

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def fit(
        self,
        train: pd.DataFrame,
        *,
        calibration: pd.DataFrame | None = None,
    ) -> "JointScoreModel":
        self._validate_training_frame(train)
        x_train = self.preprocessor.fit_transform(train[self.feature_columns])

        self.margin_model.fit(x_train, train["home_margin"].astype(float))
        self.total_model.fit(x_train, train["total_points"].astype(float))

        margin_resid = train["home_margin"].to_numpy(float) - self.margin_model.predict(x_train)
        total_resid = train["total_points"].to_numpy(float) - self.total_model.predict(x_train)
        self._set_residual_distribution(margin_resid, total_resid)
        self.is_fitted_ = True

        if calibration is not None and len(calibration):
            self.fit_calibrators(calibration)
        return self

    def _set_residual_distribution(self, margin_resid: np.ndarray, total_resid: np.ndarray) -> None:
        self.margin_sd_ = max(float(np.nanstd(margin_resid, ddof=1)), 1.0)
        self.total_sd_ = max(float(np.nanstd(total_resid, ddof=1)), 1.0)
        valid = np.isfinite(margin_resid) & np.isfinite(total_resid)
        if valid.sum() > 3:
            rho = float(np.corrcoef(margin_resid[valid], total_resid[valid])[0, 1])
            self.residual_correlation_ = float(np.clip(rho, -0.95, 0.95))
        else:
            self.residual_correlation_ = 0.0

    def _validate_training_frame(self, frame: pd.DataFrame) -> None:
        required = set(self.feature_columns) | {"home_margin", "total_points"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model columns: {sorted(missing)}")

    def _means(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Fit the model before prediction.")
        x = self.preprocessor.transform(frame[self.feature_columns])
        return self.margin_model.predict(x), self.total_model.predict(x)

    def raw_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        margin_mean, total_mean = self._means(frame)
        home_spread = frame["home_spread"].to_numpy(float)
        total_line = frame["total_line"].to_numpy(float)

        # Final NFL margins and totals are integer-valued. A continuity
        # correction assigns probability mass to ties and integer-line pushes.
        tie_probability = (
            norm.cdf((0.5 - margin_mean) / self.margin_sd_)
            - norm.cdf((-0.5 - margin_mean) / self.margin_sd_)
        )
        raw_home_win_unconditional = 1.0 - norm.cdf(
            (0.5 - margin_mean) / self.margin_sd_
        )
        non_tie_probability = np.clip(1.0 - tie_probability, 1e-9, 1.0)
        raw_home_win_conditional = np.clip(
            raw_home_win_unconditional / non_tie_probability, 0.0, 1.0
        )

        ats_edge_mean = margin_mean + home_spread
        ats_integer = np.isfinite(home_spread) & np.isclose(
            home_spread, np.round(home_spread)
        )
        ats_push_probability = np.where(
            ats_integer,
            norm.cdf((0.5 - ats_edge_mean) / self.margin_sd_)
            - norm.cdf((-0.5 - ats_edge_mean) / self.margin_sd_),
            0.0,
        )
        raw_home_cover_unconditional = np.where(
            ats_integer,
            1.0 - norm.cdf((0.5 - ats_edge_mean) / self.margin_sd_),
            norm.cdf(ats_edge_mean / self.margin_sd_),
        )
        ats_non_push = np.clip(1.0 - ats_push_probability, 1e-9, 1.0)
        raw_home_cover_conditional = np.clip(
            raw_home_cover_unconditional / ats_non_push, 0.0, 1.0
        )

        total_edge_mean = total_mean - total_line
        total_integer = np.isfinite(total_line) & np.isclose(
            total_line, np.round(total_line)
        )
        total_push_probability = np.where(
            total_integer,
            norm.cdf((0.5 - total_edge_mean) / self.total_sd_)
            - norm.cdf((-0.5 - total_edge_mean) / self.total_sd_),
            0.0,
        )
        raw_over_unconditional = np.where(
            total_integer,
            1.0 - norm.cdf((0.5 - total_edge_mean) / self.total_sd_),
            norm.cdf(total_edge_mean / self.total_sd_),
        )
        total_non_push = np.clip(1.0 - total_push_probability, 1e-9, 1.0)
        raw_over_conditional = np.clip(
            raw_over_unconditional / total_non_push, 0.0, 1.0
        )

        return pd.DataFrame(
            {
                "predicted_margin": margin_mean,
                "predicted_total": total_mean,
                "raw_home_win_probability_no_tie": raw_home_win_conditional,
                "raw_home_cover_probability_no_push": raw_home_cover_conditional,
                "raw_over_probability_no_push": raw_over_conditional,
                "tie_probability": np.clip(tie_probability, 0.0, 1.0),
                "ats_push_probability": np.clip(ats_push_probability, 0.0, 1.0),
                "total_push_probability": np.clip(total_push_probability, 0.0, 1.0),
            },
            index=frame.index,
        )

    def fit_calibrators(self, calibration: pd.DataFrame) -> "JointScoreModel":
        raw = self.raw_probabilities(calibration)
        # Nullable labels: ties/pushes are pd.NA and must never enter a calibrator.
        # ``isin([0, 1])`` excludes them for both Int8 and float label columns.
        ml_mask = calibration["home_win"].isin([0, 1])
        ats_mask = calibration["home_cover"].isin([0, 1])
        ou_mask = calibration["over"].isin([0, 1])

        self.ml_calibrator.fit(
            raw.loc[ml_mask, "raw_home_win_probability_no_tie"].to_numpy(),
            calibration.loc[ml_mask, "home_win"].to_numpy(),
        )
        self.ats_calibrator.fit(
            raw.loc[ats_mask, "raw_home_cover_probability_no_push"].to_numpy(),
            calibration.loc[ats_mask, "home_cover"].to_numpy(),
        )
        self.ou_calibrator.fit(
            raw.loc[ou_mask, "raw_over_probability_no_push"].to_numpy(),
            calibration.loc[ou_mask, "over"].to_numpy(),
        )
        return self

    def predict_markets(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = self.raw_probabilities(frame)
        out = raw.copy()

        home_win_no_tie = self.ml_calibrator.predict(
            raw["raw_home_win_probability_no_tie"].to_numpy()
        )
        home_cover_no_push = self.ats_calibrator.predict(
            raw["raw_home_cover_probability_no_push"].to_numpy()
        )
        over_no_push = self.ou_calibrator.predict(
            raw["raw_over_probability_no_push"].to_numpy()
        )

        out["home_win_probability_no_tie"] = home_win_no_tie
        out["home_cover_probability_no_push"] = home_cover_no_push
        out["over_probability_no_push"] = over_no_push

        non_tie = 1.0 - out["tie_probability"]
        ats_non_push = 1.0 - out["ats_push_probability"]
        total_non_push = 1.0 - out["total_push_probability"]

        out["home_win_probability"] = non_tie * home_win_no_tie
        out["away_win_probability"] = non_tie * (1.0 - home_win_no_tie)
        out["home_cover_probability"] = ats_non_push * home_cover_no_push
        out["away_cover_probability"] = ats_non_push * (1.0 - home_cover_no_push)
        out["over_probability"] = total_non_push * over_no_push
        out["under_probability"] = total_non_push * (1.0 - over_no_push)

        out["predicted_home_points"] = (
            out["predicted_total"] + out["predicted_margin"]
        ) / 2.0
        out["predicted_away_points"] = (
            out["predicted_total"] - out["predicted_margin"]
        ) / 2.0
        return out

    def simulate_scores(
        self,
        frame: pd.DataFrame,
        *,
        n_simulations: int = 20000,
        random_state: int | None = None,
    ) -> list[pd.DataFrame]:
        margin_mean, total_mean = self._means(frame)
        seed = self.config.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        covariance = np.array(
            [
                [self.margin_sd_ ** 2, self.residual_correlation_ * self.margin_sd_ * self.total_sd_],
                [self.residual_correlation_ * self.margin_sd_ * self.total_sd_, self.total_sd_ ** 2],
            ]
        )

        simulations: list[pd.DataFrame] = []
        for mu_m, mu_t in zip(margin_mean, total_mean):
            residual = rng.multivariate_normal([0.0, 0.0], covariance, size=n_simulations)
            margin = mu_m + residual[:, 0]
            total = np.maximum(mu_t + residual[:, 1], 0.0)
            home = np.maximum(np.rint((total + margin) / 2.0), 0.0)
            away = np.maximum(np.rint((total - margin) / 2.0), 0.0)
            simulations.append(
                pd.DataFrame(
                    {
                        "home_score": home.astype(int),
                        "away_score": away.astype(int),
                        "margin": (home - away).astype(int),
                        "total": (home + away).astype(int),
                    }
                )
            )
        return simulations
