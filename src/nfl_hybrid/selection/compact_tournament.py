from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MarketTournamentSpec:
    market: str
    binary_target: str
    push_target: str | None
    continuous_target: str
    baseline_probability_feature: str | None
    continuous_anchor_feature: str | None = None


@dataclass(frozen=True)
class TournamentConfig:
    test_seasons: tuple[int, ...] = (2021, 2022, 2023)
    random_seed: int = 42
    minimum_train_rows: int = 200
    minimum_test_rows: int = 100
    probability_epsilon: float = 1e-6
    calibration_bins: int = 10


DEFAULT_MARKET_SPECS: dict[str, MarketTournamentSpec] = {
    "pregame_moneyline": MarketTournamentSpec(
        market="pregame_moneyline",
        binary_target="target_home_win",
        push_target="target_tie",
        continuous_target="target_home_margin",
        baseline_probability_feature="market_home_ml_novig_prob",
        continuous_anchor_feature="market_implied_margin",
    ),
    "pregame_ats": MarketTournamentSpec(
        market="pregame_ats",
        binary_target="target_home_cover",
        push_target="target_ats_push",
        continuous_target="target_margin_residual",
        baseline_probability_feature="market_home_cover_novig_prob",
    ),
    "pregame_total": MarketTournamentSpec(
        market="pregame_total",
        binary_target="target_over",
        push_target="target_total_push",
        continuous_target="target_total_residual",
        baseline_probability_feature="market_over_novig_prob",
    ),
}


_CONTEXT_FEATURES = frozenset(
    {
        "home_field_indicator",
        "playoff_flag",
        "division_game_flag",
        "rest_days_diff",
        "bye_diff",
        "indoor_flag",
        "prior_games_min",
        "early_season_uncertainty",
    }
)


def _feature_group(feature: str) -> str:
    if (
        feature.startswith("market_")
        or feature.startswith("spread_")
        or feature == "home_favorite_flag"
    ):
        return "market"
    if feature.startswith("qb_"):
        return "qb"
    if feature.startswith("matchup_") or feature.startswith("lagged_"):
        return "strength"
    if feature in _CONTEXT_FEATURES:
        return "context"
    return "context"


def _candidate_feature_sets(
    features: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    selected = tuple(dict.fromkeys(str(x) for x in features))
    groups: dict[str, list[str]] = {
        "context": [],
        "strength": [],
        "qb": [],
        "market": [],
    }
    for feature in selected:
        groups[_feature_group(feature)].append(feature)

    def union(*names: str) -> tuple[str, ...]:
        return tuple(
            feature
            for feature in selected
            if any(feature in groups[name] for name in names)
        )

    candidates: dict[str, tuple[str, ...]] = {
        "context_only": union("context"),
        "strength_only": union("strength"),
        "context_strength": union("context", "strength"),
        "context_strength_qb": union("context", "strength", "qb"),
        "all_football": union("context", "strength", "qb"),
    }
    if groups["market"]:
        candidates.update(
            {
                "market_only": union("market"),
                "market_context": union("market", "context"),
                "market_strength": union("market", "strength"),
                "market_strength_qb": union("market", "strength", "qb"),
                "all_augmented": selected,
            }
        )

    deduplicated: dict[tuple[str, ...], str] = {}
    output: dict[str, tuple[str, ...]] = {}
    for name, values in candidates.items():
        if not values:
            continue
        key = tuple(values)
        if key in deduplicated:
            continue
        deduplicated[key] = name
        output[name] = key
    return output


def _classifier_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    for c_value in (0.05, 0.20, 1.0):
        models[f"logistic_l2_c{c_value:g}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        l1_ratio=0.0,
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    models["hgb_classifier"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=180,
                    max_leaf_nodes=9,
                    min_samples_leaf=35,
                    l2_regularization=3.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    return models


def _regression_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    for alpha in (0.5, 2.0, 10.0):
        models[f"ridge_alpha{alpha:g}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
    models["hgb_residual"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=180,
                    max_leaf_nodes=9,
                    min_samples_leaf=35,
                    l2_regularization=3.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    return models


def _calibration_intercept_slope(
    y: np.ndarray,
    p: np.ndarray,
    epsilon: float,
) -> tuple[float, float]:
    probability = np.clip(np.asarray(p, dtype=float), epsilon, 1.0 - epsilon)
    target = np.asarray(y, dtype=int)
    if len(np.unique(target)) < 2:
        return math.nan, math.nan

    x = logit(probability).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=3000,
    )
    model.fit(x, target)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> float:
    target = np.asarray(y, dtype=float)
    probability = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not mask.any():
            continue
        value += (
            mask.mean()
            * abs(float(target[mask].mean()) - float(probability[mask].mean()))
        )
    return float(value)


def _probability_metrics(
    y: np.ndarray,
    p: np.ndarray,
    config: TournamentConfig,
) -> dict[str, float]:
    target = np.asarray(y, dtype=int)
    probability = np.clip(
        np.asarray(p, dtype=float),
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    )
    intercept, slope = _calibration_intercept_slope(
        target,
        probability,
        config.probability_epsilon,
    )
    return {
        "n": int(len(target)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": _expected_calibration_error(
            target,
            probability,
            config.calibration_bins,
        ),
        "mean_probability": float(probability.mean()),
        "event_rate": float(target.mean()),
    }


def _binary_mask(
    frame: pd.DataFrame,
    spec: MarketTournamentSpec,
) -> pd.Series:
    mask = frame[spec.binary_target].isin([0, 1])
    if spec.push_target and spec.push_target in frame:
        mask &= pd.to_numeric(
            frame[spec.push_target], errors="coerce"
        ).fillna(0).eq(0)
    return mask


def _market_baseline_probability(
    frame: pd.DataFrame,
    spec: MarketTournamentSpec,
) -> np.ndarray:
    if (
        spec.baseline_probability_feature
        and spec.baseline_probability_feature in frame.columns
    ):
        values = pd.to_numeric(
            frame[spec.baseline_probability_feature],
            errors="coerce",
        ).to_numpy(float)
        if np.isfinite(values).all():
            return values
    return np.full(len(frame), 0.5, dtype=float)


def _residual_target_and_anchor(
    frame: pd.DataFrame,
    spec: MarketTournamentSpec,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = pd.to_numeric(
        frame[spec.continuous_target],
        errors="coerce",
    ).to_numpy(float)
    anchor = np.zeros(len(frame), dtype=float)

    if (
        spec.market == "pregame_moneyline"
        and variant == "market_augmented"
        and spec.continuous_anchor_feature
        and spec.continuous_anchor_feature in frame.columns
    ):
        anchor = pd.to_numeric(
            frame[spec.continuous_anchor_feature],
            errors="coerce",
        ).to_numpy(float)
        target = target - anchor
    return target, anchor


def _residual_probability(
    market: str,
    prediction: np.ndarray,
    anchor: np.ndarray,
    residual_sd: float,
) -> np.ndarray:
    scale = max(float(residual_sd), 1e-6)
    if market == "pregame_moneyline":
        score = anchor + prediction
    else:
        score = prediction
    return norm.cdf(score / scale)


def _read_manifest_features(
    compact_root: Path,
    market: str,
    variant: str,
) -> tuple[str, ...]:
    path = compact_root / f"{market}_{variant}.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(payload["features"])


def _evaluate_matrix(
    matrix: pd.DataFrame,
    features: tuple[str, ...],
    *,
    spec: MarketTournamentSpec,
    variant: str,
    config: TournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_sets = _candidate_feature_sets(features)

    for test_season in config.test_seasons:
        train = matrix[matrix["season"] < test_season].copy()
        test = matrix[matrix["season"] == test_season].copy()
        train_binary = train.loc[_binary_mask(train, spec)].copy()
        test_binary = test.loc[_binary_mask(test, spec)].copy()

        if len(train_binary) < config.minimum_train_rows:
            raise ValueError(
                f"{spec.market} {variant} {test_season}: "
                f"only {len(train_binary)} binary training rows."
            )
        if len(test_binary) < config.minimum_test_rows:
            raise ValueError(
                f"{spec.market} {variant} {test_season}: "
                f"only {len(test_binary)} binary test rows."
            )

        baseline_probability = _market_baseline_probability(
            test_binary,
            spec,
        )
        baseline_metrics = _probability_metrics(
            test_binary[spec.binary_target].to_numpy(int),
            baseline_probability,
            config,
        )
        rows.append(
            {
                "market": spec.market,
                "variant": variant,
                "test_season": test_season,
                "feature_set": "market_baseline"
                if variant == "market_augmented"
                else "coin_flip_baseline",
                "model_family": "baseline",
                "model_name": "baseline",
                "feature_count": 0,
                "features": "[]",
                **baseline_metrics,
            }
        )

        classifier_models = _classifier_models(config.random_seed)
        regression_models = _regression_models(config.random_seed)

        for feature_set_name, selected_features in feature_sets.items():
            x_train = train_binary.loc[:, selected_features]
            x_test = test_binary.loc[:, selected_features]
            y_train = train_binary[spec.binary_target].to_numpy(int)
            y_test = test_binary[spec.binary_target].to_numpy(int)

            for model_name, model in classifier_models.items():
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
                metrics = _probability_metrics(y_test, probability, config)
                rows.append(
                    {
                        "market": spec.market,
                        "variant": variant,
                        "test_season": test_season,
                        "feature_set": feature_set_name,
                        "model_family": "classifier",
                        "model_name": model_name,
                        "feature_count": len(selected_features),
                        "features": json.dumps(selected_features),
                        **metrics,
                    }
                )

            train_continuous = train.loc[
                pd.to_numeric(
                    train[spec.continuous_target], errors="coerce"
                ).notna()
            ].copy()
            test_continuous = test_binary.copy()
            y_train_continuous, train_anchor = _residual_target_and_anchor(
                train_continuous,
                spec,
                variant,
            )
            _, test_anchor = _residual_target_and_anchor(
                test_continuous,
                spec,
                variant,
            )

            x_train_continuous = train_continuous.loc[:, selected_features]
            x_test_continuous = test_continuous.loc[:, selected_features]

            for model_name, model in regression_models.items():
                model.fit(x_train_continuous, y_train_continuous)
                train_prediction = model.predict(x_train_continuous)
                residual_sd = float(
                    np.nanstd(y_train_continuous - train_prediction, ddof=1)
                )
                test_prediction = model.predict(x_test_continuous)
                probability = _residual_probability(
                    spec.market,
                    test_prediction,
                    test_anchor,
                    residual_sd,
                )
                metrics = _probability_metrics(
                    test_continuous[spec.binary_target].to_numpy(int),
                    probability,
                    config,
                )
                rows.append(
                    {
                        "market": spec.market,
                        "variant": variant,
                        "test_season": test_season,
                        "feature_set": feature_set_name,
                        "model_family": "residual_regression",
                        "model_name": model_name,
                        "feature_count": len(selected_features),
                        "features": json.dumps(selected_features),
                        "residual_sd": residual_sd,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "market",
        "variant",
        "feature_set",
        "model_family",
        "model_name",
        "feature_count",
        "features",
    ]
    aggregate = (
        results.groupby(group_columns, as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            n=("n", "sum"),
            mean_log_loss=("log_loss", "mean"),
            sd_log_loss=("log_loss", "std"),
            mean_brier=("brier", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_ece=("ece", "mean"),
            worst_season_log_loss=("log_loss", "max"),
        )
    )
    aggregate["se_log_loss"] = (
        aggregate["sd_log_loss"]
        / np.sqrt(aggregate["seasons"].clip(lower=1))
    )
    return aggregate.sort_values(
        [
            "market",
            "variant",
            "mean_log_loss",
            "mean_brier",
            "feature_count",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _select_models(aggregate: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for (market, variant), frame in aggregate.groupby(
        ["market", "variant"],
        sort=True,
    ):
        candidates = frame[frame["model_family"] != "baseline"].copy()
        best = candidates.iloc[candidates["mean_log_loss"].argmin()]
        threshold = float(best["mean_log_loss"]) + float(
            best["se_log_loss"]
            if np.isfinite(best["se_log_loss"])
            else 0.0
        )
        eligible = candidates[
            candidates["mean_log_loss"] <= threshold
        ].copy()
        eligible["calibration_penalty"] = (
            eligible["mean_calibration_intercept"].abs()
            + (eligible["mean_calibration_slope"] - 1.0).abs()
        )
        choice = eligible.sort_values(
            [
                "feature_count",
                "mean_log_loss",
                "mean_brier",
                "calibration_penalty",
            ],
            kind="stable",
        ).iloc[0]
        baseline = frame[frame["model_family"] == "baseline"].iloc[0]
        choice = choice.copy()
        choice["baseline_log_loss"] = float(baseline["mean_log_loss"])
        choice["baseline_brier"] = float(baseline["mean_brier"])
        choice["log_loss_improvement_vs_baseline"] = (
            float(baseline["mean_log_loss"]) - float(choice["mean_log_loss"])
        )
        choice["brier_improvement_vs_baseline"] = (
            float(baseline["mean_brier"]) - float(choice["mean_brier"])
        )
        selected_rows.append(choice)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def run_compact_tournament(
    compact_root: str | Path,
    output_root: str | Path,
    *,
    config: TournamentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or TournamentConfig()
    compact_path = Path(compact_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    for market, spec in DEFAULT_MARKET_SPECS.items():
        for variant in ("football_only", "market_augmented"):
            matrix_path = compact_path / f"{market}_{variant}.parquet"
            if not matrix_path.exists():
                raise FileNotFoundError(matrix_path)
            matrix = pd.read_parquet(matrix_path)
            if any(season >= 2024 for season in cfg.test_seasons):
                raise ValueError("Tournament selection seasons must remain before 2024.")
            features = _read_manifest_features(
                compact_path,
                market,
                variant,
            )
            result = _evaluate_matrix(
                matrix,
                features,
                spec=spec,
                variant=variant,
                config=cfg,
            )
            all_results.append(result)

    fold_results = pd.concat(all_results, ignore_index=True)
    aggregate = _aggregate_results(fold_results)
    selected = _select_models(aggregate)

    fold_results.to_csv(output_path / "compact_tournament_folds.csv", index=False)
    aggregate.to_csv(output_path / "compact_tournament_aggregate.csv", index=False)
    selected.to_csv(output_path / "compact_tournament_selected.csv", index=False)

    payload = {
        "status": "provisional_before_roster_injury_and_2024_selection",
        "selection_seasons": list(cfg.test_seasons),
        "untouched_seasons": [2024, 2025],
        "random_seed": cfg.random_seed,
        "selected": [],
    }
    for row in selected.to_dict(orient="records"):
        payload["selected"].append(
            {
                "market": row["market"],
                "variant": row["variant"],
                "feature_set": row["feature_set"],
                "model_family": row["model_family"],
                "model_name": row["model_name"],
                "feature_count": int(row["feature_count"]),
                "features": json.loads(row["features"]),
                "mean_log_loss": float(row["mean_log_loss"]),
                "mean_brier": float(row["mean_brier"]),
                "mean_calibration_intercept": float(
                    row["mean_calibration_intercept"]
                ),
                "mean_calibration_slope": float(row["mean_calibration_slope"]),
                "mean_ece": float(row["mean_ece"]),
                "log_loss_improvement_vs_baseline": float(
                    row["log_loss_improvement_vs_baseline"]
                ),
                "brier_improvement_vs_baseline": float(
                    row["brier_improvement_vs_baseline"]
                ),
            }
        )
    (output_path / "provisional_selected_models.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (output_path / "tournament_config.json").write_text(
        json.dumps(asdict(cfg), indent=2) + "\n",
        encoding="utf-8",
    )
    return fold_results, aggregate, selected
