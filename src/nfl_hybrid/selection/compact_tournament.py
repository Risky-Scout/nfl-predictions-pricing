from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
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

from .integrity import audit_compact_targets, review_compact_schemas


@dataclass(frozen=True)
class MarketTournamentSpec:
    market: str
    binary_target: str
    push_target: str | None
    continuous_target: str
    baseline_probability_feature: str | None
    continuous_anchor_feature: str | None = None
    residual_target_is_market_anchored: bool = False
    market_anchor_features: tuple[str, ...] = ()
    market_line_feature: str | None = None


@dataclass(frozen=True)
class TournamentConfig:
    test_seasons: tuple[int, ...] = (2021, 2022, 2023)
    random_seed: int = 42
    minimum_train_rows: int = 200
    minimum_test_rows: int = 100
    probability_epsilon: float = 1e-6
    calibration_bins: int = 10
    minimum_log_loss_winning_seasons: int = 2
    maximum_single_season_log_loss_regression: float = 0.005
    minimum_mean_log_loss_improvement: float = 0.0
    minimum_mean_brier_improvement: float = 0.0
    bootstrap_repetitions: int = 2000
    run_integrity_checks: bool = True


DEFAULT_MARKET_SPECS: dict[str, MarketTournamentSpec] = {
    "pregame_moneyline": MarketTournamentSpec(
        market="pregame_moneyline",
        binary_target="target_home_win",
        push_target="target_tie",
        continuous_target="target_home_margin",
        baseline_probability_feature="market_home_ml_novig_prob",
        continuous_anchor_feature="market_implied_margin",
        market_anchor_features=(
            "market_home_ml_novig_prob",
            "market_implied_margin",
        ),
        market_line_feature="market_implied_margin",
    ),
    "pregame_ats": MarketTournamentSpec(
        market="pregame_ats",
        binary_target="target_home_cover",
        push_target="target_ats_push",
        continuous_target="target_margin_residual",
        baseline_probability_feature="market_home_cover_novig_prob",
        residual_target_is_market_anchored=True,
        market_anchor_features=(
            "market_home_spread",
            "market_home_cover_novig_prob",
        ),
        market_line_feature="market_home_spread",
    ),
    "pregame_total": MarketTournamentSpec(
        market="pregame_total",
        binary_target="target_over",
        push_target="target_total_push",
        continuous_target="target_total_residual",
        baseline_probability_feature="market_over_novig_prob",
        residual_target_is_market_anchored=True,
        market_anchor_features=(
            "market_total_line",
            "market_over_novig_prob",
        ),
        market_line_feature="market_total_line",
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


_CONFIG_KEYS = [
    "market",
    "variant",
    "feature_set",
    "model_family",
    "model_name",
    "feature_count",
    "features",
    "market_anchor_mode",
]


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


def _model_by_name(
    model_family: str,
    model_name: str,
    seed: int,
) -> object:
    if model_family == "classifier":
        models = _classifier_models(seed)
    elif model_family == "residual_regression":
        models = _regression_models(seed)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    try:
        return models[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {model_family} model name: {model_name}"
        ) from exc


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


def _market_anchor_mode(
    *,
    spec: MarketTournamentSpec,
    variant: str,
    model_family: str,
    selected_features: tuple[str, ...],
) -> str:
    if variant != "market_augmented":
        return "not_applicable"
    if model_family == "baseline":
        return "market_baseline"
    if set(spec.market_anchor_features).intersection(selected_features):
        return "explicit_market_feature"
    if (
        model_family == "residual_regression"
        and spec.market == "pregame_moneyline"
        and spec.continuous_anchor_feature
    ):
        return "external_market_margin_anchor"
    if (
        model_family == "residual_regression"
        and spec.residual_target_is_market_anchored
    ):
        return "market_residual_target_anchor"
    return "unanchored"


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
                "market_anchor_mode": _market_anchor_mode(
                    spec=spec,
                    variant=variant,
                    model_family="baseline",
                    selected_features=(),
                ),
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
                        "market_anchor_mode": _market_anchor_mode(
                            spec=spec,
                            variant=variant,
                            model_family="classifier",
                            selected_features=selected_features,
                        ),
                        **metrics,
                    }
                )

            train_continuous = train.loc[
                pd.to_numeric(
                    train[spec.continuous_target], errors="coerce"
                ).notna()
            ].copy()
            test_continuous = test_binary.copy()
            y_train_continuous, _ = _residual_target_and_anchor(
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
                        "market_anchor_mode": _market_anchor_mode(
                            spec=spec,
                            variant=variant,
                            model_family="residual_regression",
                            selected_features=selected_features,
                        ),
                        "residual_sd": residual_sd,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        results.groupby(_CONFIG_KEYS, as_index=False)
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


def _build_stability_report(
    fold_results: pd.DataFrame,
    config: TournamentConfig,
) -> pd.DataFrame:
    baselines = fold_results[
        fold_results["model_family"].eq("baseline")
    ][
        [
            "market",
            "variant",
            "test_season",
            "log_loss",
            "brier",
        ]
    ].rename(
        columns={
            "log_loss": "baseline_log_loss",
            "brier": "baseline_brier",
        }
    )
    candidates = fold_results[
        ~fold_results["model_family"].eq("baseline")
    ].merge(
        baselines,
        on=["market", "variant", "test_season"],
        how="left",
        validate="many_to_one",
    )
    candidates["log_loss_gain"] = (
        candidates["baseline_log_loss"] - candidates["log_loss"]
    )
    candidates["brier_gain"] = (
        candidates["baseline_brier"] - candidates["brier"]
    )

    stability = (
        candidates.groupby(_CONFIG_KEYS, as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            mean_log_loss_gain=("log_loss_gain", "mean"),
            mean_brier_gain=("brier_gain", "mean"),
            log_loss_winning_seasons=(
                "log_loss_gain",
                lambda values: int((values > 0).sum()),
            ),
            brier_winning_seasons=(
                "brier_gain",
                lambda values: int((values > 0).sum()),
            ),
            worst_log_loss_gain=("log_loss_gain", "min"),
            worst_brier_gain=("brier_gain", "min"),
        )
    )
    stability["anchor_valid"] = ~stability["market_anchor_mode"].eq(
        "unanchored"
    )
    stability["passes_mean_log_loss"] = (
        stability["mean_log_loss_gain"]
        > config.minimum_mean_log_loss_improvement
    )
    stability["passes_mean_brier"] = (
        stability["mean_brier_gain"]
        > config.minimum_mean_brier_improvement
    )
    stability["passes_season_stability"] = (
        stability["log_loss_winning_seasons"]
        >= config.minimum_log_loss_winning_seasons
    )
    stability["passes_catastrophic_regression_guard"] = (
        stability["worst_log_loss_gain"]
        >= -config.maximum_single_season_log_loss_regression
    )
    stability["qualifies"] = (
        stability["anchor_valid"]
        & stability["passes_mean_log_loss"]
        & stability["passes_mean_brier"]
        & stability["passes_season_stability"]
        & stability["passes_catastrophic_regression_guard"]
    )

    def reasons(row: pd.Series) -> str:
        failed: list[str] = []
        if not row["anchor_valid"]:
            failed.append("MISSING_MARKET_ANCHOR")
        if not row["passes_mean_log_loss"]:
            failed.append("MEAN_LOG_LOSS_NOT_BETTER")
        if not row["passes_mean_brier"]:
            failed.append("MEAN_BRIER_NOT_BETTER")
        if not row["passes_season_stability"]:
            failed.append("INSUFFICIENT_LOG_LOSS_WINNING_SEASONS")
        if not row["passes_catastrophic_regression_guard"]:
            failed.append("SINGLE_SEASON_REGRESSION_TOO_LARGE")
        return "PASS" if not failed else "|".join(failed)

    stability["gate_reasons"] = stability.apply(reasons, axis=1)
    return stability.sort_values(
        ["market", "variant", "qualifies", "mean_log_loss_gain"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _select_models(
    aggregate: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = aggregate[
        ~aggregate["model_family"].eq("baseline")
    ].merge(
        stability,
        on=_CONFIG_KEYS,
        how="left",
        validate="one_to_one",
    )

    selected_rows: list[pd.Series] = []
    challenger_rows: list[pd.Series] = []

    for (market, variant), frame in aggregate.groupby(
        ["market", "variant"],
        sort=True,
    ):
        baseline = frame[frame["model_family"].eq("baseline")].iloc[0].copy()
        market_candidates = candidates[
            candidates["market"].eq(market)
            & candidates["variant"].eq(variant)
        ].copy()
        challenger = market_candidates.sort_values(
            ["mean_log_loss", "mean_brier", "feature_count"],
            kind="stable",
        ).iloc[0].copy()
        challenger["baseline_log_loss"] = float(baseline["mean_log_loss"])
        challenger["baseline_brier"] = float(baseline["mean_brier"])
        challenger["log_loss_improvement_vs_baseline"] = (
            float(baseline["mean_log_loss"])
            - float(challenger["mean_log_loss"])
        )
        challenger["brier_improvement_vs_baseline"] = (
            float(baseline["mean_brier"])
            - float(challenger["mean_brier"])
        )
        challenger_rows.append(challenger)

        qualifying = market_candidates[
            market_candidates["qualifies"].fillna(False)
        ].copy()
        if qualifying.empty:
            choice = baseline.copy()
            choice["selection_status"] = (
                "RETAIN_BASELINE_NO_QUALIFYING_MODEL"
            )
            choice["baseline_log_loss"] = float(baseline["mean_log_loss"])
            choice["baseline_brier"] = float(baseline["mean_brier"])
            choice["log_loss_improvement_vs_baseline"] = 0.0
            choice["brier_improvement_vs_baseline"] = 0.0
            choice["qualifies"] = False
            choice["gate_reasons"] = str(challenger["gate_reasons"])
            choice["challenger_feature_set"] = challenger["feature_set"]
            choice["challenger_model_family"] = challenger["model_family"]
            choice["challenger_model_name"] = challenger["model_name"]
            choice["challenger_mean_log_loss"] = challenger["mean_log_loss"]
            choice["challenger_mean_brier"] = challenger["mean_brier"]
            selected_rows.append(choice)
            continue

        best = qualifying.sort_values(
            ["mean_log_loss", "mean_brier"],
            kind="stable",
        ).iloc[0]
        threshold = float(best["mean_log_loss"]) + float(
            best["se_log_loss"]
            if np.isfinite(best["se_log_loss"])
            else 0.0
        )
        eligible = qualifying[
            qualifying["mean_log_loss"] <= threshold
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
        ).iloc[0].copy()
        choice["selection_status"] = "QUALIFIED_PROVISIONAL_CANDIDATE"
        choice["baseline_log_loss"] = float(baseline["mean_log_loss"])
        choice["baseline_brier"] = float(baseline["mean_brier"])
        choice["log_loss_improvement_vs_baseline"] = (
            float(baseline["mean_log_loss"]) - float(choice["mean_log_loss"])
        )
        choice["brier_improvement_vs_baseline"] = (
            float(baseline["mean_brier"]) - float(choice["mean_brier"])
        )
        choice["challenger_feature_set"] = challenger["feature_set"]
        choice["challenger_model_family"] = challenger["model_family"]
        choice["challenger_model_name"] = challenger["model_name"]
        choice["challenger_mean_log_loss"] = challenger["mean_log_loss"]
        choice["challenger_mean_brier"] = challenger["mean_brier"]
        selected_rows.append(choice)

    return (
        pd.DataFrame(selected_rows).reset_index(drop=True),
        pd.DataFrame(challenger_rows).reset_index(drop=True),
    )


def _prediction_frame(
    test: pd.DataFrame,
    *,
    spec: MarketTournamentSpec,
    variant: str,
    role: str,
    feature_set: str,
    model_family: str,
    model_name: str,
    market_anchor_mode: str,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    predicted_mean: np.ndarray | None,
    residual_sd: float | None,
    config: TournamentConfig,
) -> pd.DataFrame:
    output = pd.DataFrame(index=test.index)
    for column in (
        "game_id",
        "season",
        "home_team_id",
        "away_team_id",
    ):
        if column in test.columns:
            output[column] = test[column].to_numpy()
    output["market"] = spec.market
    output["variant"] = variant
    output["role"] = role
    output["feature_set"] = feature_set
    output["model_family"] = model_family
    output["model_name"] = model_name
    output["market_anchor_mode"] = market_anchor_mode
    output["target"] = pd.to_numeric(
        test[spec.binary_target], errors="coerce"
    ).to_numpy()
    output["push_or_tie"] = (
        pd.to_numeric(test[spec.push_target], errors="coerce")
        .fillna(0)
        .to_numpy()
        if spec.push_target and spec.push_target in test
        else np.zeros(len(test), dtype=float)
    )
    output["continuous_target"] = pd.to_numeric(
        test[spec.continuous_target], errors="coerce"
    ).to_numpy()
    output["market_line"] = (
        pd.to_numeric(test[spec.market_line_feature], errors="coerce").to_numpy()
        if spec.market_line_feature
        and spec.market_line_feature in test.columns
        else np.full(len(test), np.nan)
    )
    output["probability"] = np.clip(
        np.asarray(probability, dtype=float),
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    )
    output["baseline_probability"] = np.clip(
        np.asarray(baseline_probability, dtype=float),
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    )
    output["predicted_mean"] = (
        np.asarray(predicted_mean, dtype=float)
        if predicted_mean is not None
        else np.full(len(test), np.nan)
    )
    output["residual_sd"] = (
        float(residual_sd)
        if residual_sd is not None
        else np.nan
    )
    scored = _binary_mask(test, spec).to_numpy(bool)
    output["scored_binary"] = scored
    target = output["target"].to_numpy(float)
    probability_clipped = output["probability"].to_numpy(float)
    baseline_clipped = output["baseline_probability"].to_numpy(float)
    output["model_log_loss_row"] = np.where(
        scored,
        -(
            target * np.log(probability_clipped)
            + (1.0 - target) * np.log(1.0 - probability_clipped)
        ),
        np.nan,
    )
    output["baseline_log_loss_row"] = np.where(
        scored,
        -(
            target * np.log(baseline_clipped)
            + (1.0 - target) * np.log(1.0 - baseline_clipped)
        ),
        np.nan,
    )
    output["model_brier_row"] = np.where(
        scored,
        (probability_clipped - target) ** 2,
        np.nan,
    )
    output["baseline_brier_row"] = np.where(
        scored,
        (baseline_clipped - target) ** 2,
        np.nan,
    )
    return output.reset_index(drop=True)


def _generate_oof_predictions(
    compact_path: Path,
    selected: pd.DataFrame,
    challengers: pd.DataFrame,
    config: TournamentConfig,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    for market, spec in DEFAULT_MARKET_SPECS.items():
        for variant in ("football_only", "market_augmented"):
            matrix = pd.read_parquet(
                compact_path / f"{market}_{variant}.parquet"
            )
            selected_row = selected[
                selected["market"].eq(market)
                & selected["variant"].eq(variant)
            ].iloc[0]
            challenger_row = challengers[
                challengers["market"].eq(market)
                & challengers["variant"].eq(variant)
            ].iloc[0]

            candidate_configs: list[tuple[str, pd.Series]] = [
                ("best_challenger", challenger_row)
            ]
            if (
                selected_row["model_family"] != "baseline"
                and (
                    selected_row["model_family"] != challenger_row["model_family"]
                    or selected_row["model_name"] != challenger_row["model_name"]
                    or selected_row["feature_set"] != challenger_row["feature_set"]
                )
            ):
                candidate_configs.append(("selected", selected_row))
            elif selected_row["model_family"] != "baseline":
                candidate_configs[0] = ("selected_challenger", challenger_row)

            for test_season in config.test_seasons:
                train = matrix[matrix["season"] < test_season].copy()
                test = matrix[matrix["season"] == test_season].copy()
                baseline_probability = _market_baseline_probability(test, spec)
                outputs.append(
                    _prediction_frame(
                        test,
                        spec=spec,
                        variant=variant,
                        role="baseline",
                        feature_set="market_baseline"
                        if variant == "market_augmented"
                        else "coin_flip_baseline",
                        model_family="baseline",
                        model_name="baseline",
                        market_anchor_mode="market_baseline"
                        if variant == "market_augmented"
                        else "not_applicable",
                        probability=baseline_probability,
                        baseline_probability=baseline_probability,
                        predicted_mean=None,
                        residual_sd=None,
                        config=config,
                    )
                )

                for role, row in candidate_configs:
                    selected_features = tuple(json.loads(row["features"]))
                    model_family = str(row["model_family"])
                    model_name = str(row["model_name"])
                    model = _model_by_name(
                        model_family,
                        model_name,
                        config.random_seed,
                    )
                    predicted_mean: np.ndarray | None = None
                    residual_sd: float | None = None

                    if model_family == "classifier":
                        train_binary = train.loc[_binary_mask(train, spec)]
                        model.fit(
                            train_binary.loc[:, selected_features],
                            train_binary[spec.binary_target].to_numpy(int),
                        )
                        probability = model.predict_proba(
                            test.loc[:, selected_features]
                        )[:, 1]
                    else:
                        train_continuous = train.loc[
                            pd.to_numeric(
                                train[spec.continuous_target],
                                errors="coerce",
                            ).notna()
                        ].copy()
                        target_train, _ = _residual_target_and_anchor(
                            train_continuous,
                            spec,
                            variant,
                        )
                        _, test_anchor = _residual_target_and_anchor(
                            test,
                            spec,
                            variant,
                        )
                        model.fit(
                            train_continuous.loc[:, selected_features],
                            target_train,
                        )
                        fitted = model.predict(
                            train_continuous.loc[:, selected_features]
                        )
                        residual_sd = float(
                            np.nanstd(target_train - fitted, ddof=1)
                        )
                        predicted_mean = model.predict(
                            test.loc[:, selected_features]
                        )
                        probability = _residual_probability(
                            spec.market,
                            predicted_mean,
                            test_anchor,
                            residual_sd,
                        )

                    outputs.append(
                        _prediction_frame(
                            test,
                            spec=spec,
                            variant=variant,
                            role=role,
                            feature_set=str(row["feature_set"]),
                            model_family=model_family,
                            model_name=model_name,
                            market_anchor_mode=str(
                                row["market_anchor_mode"]
                            ),
                            probability=probability,
                            baseline_probability=baseline_probability,
                            predicted_mean=predicted_mean,
                            residual_sd=residual_sd,
                            config=config,
                        )
                    )
    return pd.concat(outputs, ignore_index=True)


def _bootstrap_metric_differences(
    oof: pd.DataFrame,
    config: TournamentConfig,
) -> pd.DataFrame:
    scored = oof[
        oof["scored_binary"]
        & ~oof["role"].eq("baseline")
    ].copy()
    scored["log_loss_gain"] = (
        scored["baseline_log_loss_row"] - scored["model_log_loss_row"]
    )
    scored["brier_gain"] = (
        scored["baseline_brier_row"] - scored["model_brier_row"]
    )
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, object]] = []

    group_columns = [
        "market",
        "variant",
        "role",
        "feature_set",
        "model_family",
        "model_name",
    ]
    for keys, frame in scored.groupby(group_columns, sort=True):
        season_arrays = {
            int(season): season_frame[["log_loss_gain", "brier_gain"]]
            .to_numpy(float)
            for season, season_frame in frame.groupby("season")
        }
        bootstrap_log_loss: list[float] = []
        bootstrap_brier: list[float] = []
        for _ in range(config.bootstrap_repetitions):
            samples: list[np.ndarray] = []
            for values in season_arrays.values():
                index = rng.integers(0, len(values), size=len(values))
                samples.append(values[index])
            sample = np.concatenate(samples, axis=0)
            bootstrap_log_loss.append(float(sample[:, 0].mean()))
            bootstrap_brier.append(float(sample[:, 1].mean()))

        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n": len(frame),
                "mean_log_loss_gain": float(frame["log_loss_gain"].mean()),
                "log_loss_gain_ci_lower": float(
                    np.quantile(bootstrap_log_loss, 0.025)
                ),
                "log_loss_gain_ci_upper": float(
                    np.quantile(bootstrap_log_loss, 0.975)
                ),
                "bootstrap_probability_log_loss_gain_positive": float(
                    np.mean(np.asarray(bootstrap_log_loss) > 0)
                ),
                "mean_brier_gain": float(frame["brier_gain"].mean()),
                "brier_gain_ci_lower": float(
                    np.quantile(bootstrap_brier, 0.025)
                ),
                "brier_gain_ci_upper": float(
                    np.quantile(bootstrap_brier, 0.975)
                ),
                "bootstrap_probability_brier_gain_positive": float(
                    np.mean(np.asarray(bootstrap_brier) > 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _calibration_bands(
    oof: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    scored = oof[oof["scored_binary"]].copy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    scored["probability_band"] = pd.cut(
        scored["probability"],
        bins=edges,
        include_lowest=True,
        right=True,
    ).astype("string")
    group_columns = [
        "market",
        "variant",
        "role",
        "probability_band",
    ]
    return (
        scored.groupby(group_columns, observed=True, as_index=False)
        .agg(
            n=("target", "size"),
            mean_probability=("probability", "mean"),
            event_rate=("target", "mean"),
            mean_log_loss=("model_log_loss_row", "mean"),
            mean_brier=("model_brier_row", "mean"),
        )
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )


def _bucket_report(
    oof: pd.DataFrame,
    *,
    market: str,
) -> pd.DataFrame:
    frame = oof[
        oof["market"].eq(market)
        & oof["scored_binary"]
        & ~oof["role"].eq("baseline")
    ].copy()
    if market == "pregame_ats":
        frame["market_bucket"] = pd.cut(
            frame["market_line"].abs(),
            bins=[-np.inf, 1.5, 3.5, 7.5, 14.5, np.inf],
            labels=[
                "pickem_to_1",
                "1.5_to_3",
                "3.5_to_7",
                "7.5_to_14",
                "14.5_plus",
            ],
            right=False,
        ).astype("string")
    elif market == "pregame_total":
        frame["market_bucket"] = pd.cut(
            frame["market_line"],
            bins=[-np.inf, 40.0, 44.0, 48.0, 52.0, np.inf],
            labels=[
                "under_40",
                "40_to_43.5",
                "44_to_47.5",
                "48_to_51.5",
                "52_plus",
            ],
            right=False,
        ).astype("string")
    else:
        raise ValueError(market)

    frame["log_loss_gain"] = (
        frame["baseline_log_loss_row"] - frame["model_log_loss_row"]
    )
    frame["brier_gain"] = (
        frame["baseline_brier_row"] - frame["model_brier_row"]
    )
    return (
        frame.groupby(
            [
                "market",
                "variant",
                "role",
                "market_bucket",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            n=("target", "size"),
            event_rate=("target", "mean"),
            mean_probability=("probability", "mean"),
            mean_log_loss_gain=("log_loss_gain", "mean"),
            mean_brier_gain=("brier_gain", "mean"),
        )
        .sort_values(
            ["market", "variant", "role", "market_bucket"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


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

    if any(season >= 2024 for season in cfg.test_seasons):
        raise ValueError(
            "Tournament selection seasons must remain before 2024."
        )

    if cfg.run_integrity_checks:
        review_compact_schemas(compact_path, output_path)
        audit_compact_targets(
            compact_path,
            output_path,
            warehouse_path=compact_path.parent
            / "modeling_matrix_stage2_qb.parquet",
        )

    all_results: list[pd.DataFrame] = []
    for market, spec in DEFAULT_MARKET_SPECS.items():
        for variant in ("football_only", "market_augmented"):
            matrix_path = compact_path / f"{market}_{variant}.parquet"
            if not matrix_path.exists():
                raise FileNotFoundError(matrix_path)
            matrix = pd.read_parquet(matrix_path)
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
    stability = _build_stability_report(fold_results, cfg)
    selected, challengers = _select_models(aggregate, stability)

    oof = _generate_oof_predictions(
        compact_path,
        selected,
        challengers,
        cfg,
    )
    bootstrap = _bootstrap_metric_differences(oof, cfg)
    calibration = _calibration_bands(oof, cfg.calibration_bins)
    ats_buckets = _bucket_report(oof, market="pregame_ats")
    total_buckets = _bucket_report(oof, market="pregame_total")

    fold_results.to_csv(
        output_path / "compact_tournament_folds.csv",
        index=False,
    )
    aggregate.to_csv(
        output_path / "compact_tournament_aggregate.csv",
        index=False,
    )
    selected.to_csv(
        output_path / "compact_tournament_selected.csv",
        index=False,
    )
    challengers.to_csv(
        output_path / "compact_tournament_challengers.csv",
        index=False,
    )
    stability.to_csv(
        output_path / "season_stability.csv",
        index=False,
    )
    selected.to_csv(
        output_path / "baseline_veto_report.csv",
        index=False,
    )
    oof.to_parquet(
        output_path / "oof_predictions.parquet",
        index=False,
    )
    bootstrap.to_csv(
        output_path / "bootstrap_metrics.csv",
        index=False,
    )
    calibration.to_csv(
        output_path / "calibration_bands.csv",
        index=False,
    )
    ats_buckets.to_csv(
        output_path / "ats_spread_buckets.csv",
        index=False,
    )
    total_buckets.to_csv(
        output_path / "total_line_buckets.csv",
        index=False,
    )

    payload = {
        "status": "provisional_before_roster_injury_and_2024_selection",
        "selection_seasons": list(cfg.test_seasons),
        "untouched_seasons": [2024, 2025],
        "random_seed": cfg.random_seed,
        "gates": {
            "mandatory_market_anchor": True,
            "minimum_log_loss_winning_seasons": (
                cfg.minimum_log_loss_winning_seasons
            ),
            "maximum_single_season_log_loss_regression": (
                cfg.maximum_single_season_log_loss_regression
            ),
            "must_improve_mean_log_loss": True,
            "must_improve_mean_brier": True,
        },
        "selected": [],
    }
    for row in selected.to_dict(orient="records"):
        payload["selected"].append(
            {
                "market": row["market"],
                "variant": row["variant"],
                "selection_status": row["selection_status"],
                "feature_set": row["feature_set"],
                "model_family": row["model_family"],
                "model_name": row["model_name"],
                "feature_count": int(row["feature_count"]),
                "features": json.loads(row["features"]),
                "market_anchor_mode": row["market_anchor_mode"],
                "mean_log_loss": float(row["mean_log_loss"]),
                "mean_brier": float(row["mean_brier"]),
                "mean_calibration_intercept": float(
                    row["mean_calibration_intercept"]
                ),
                "mean_calibration_slope": float(
                    row["mean_calibration_slope"]
                ),
                "mean_ece": float(row["mean_ece"]),
                "log_loss_improvement_vs_baseline": float(
                    row["log_loss_improvement_vs_baseline"]
                ),
                "brier_improvement_vs_baseline": float(
                    row["brier_improvement_vs_baseline"]
                ),
                "gate_reasons": row.get("gate_reasons", ""),
            }
        )
    (output_path / "provisional_selected_models.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    config_payload = asdict(cfg)
    config_payload["test_seasons"] = list(cfg.test_seasons)
    (output_path / "tournament_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return fold_results, aggregate, selected
