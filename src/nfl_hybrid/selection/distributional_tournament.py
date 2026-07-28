from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable
import json
import math

import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from nfl_hybrid.distributional.market_models import (
    EmpiricalPushPrior,
    FixedOffsetLogistic,
    ThreeWayProbabilities,
    combine_conditional_with_push,
    conditional_upper_probability,
    empirical_discrete_probabilities,
)


@dataclass(frozen=True)
class DistributionalMarketSpec:
    market: str
    binary_target: str
    push_target: str
    absolute_target: str
    residual_target: str | None
    market_line_feature: str | None
    baseline_probability_feature: str
    anchor_feature: str | None


@dataclass(frozen=True)
class DistributionalTournamentConfig:
    test_seasons: tuple[int, ...] = (2021, 2022, 2023)
    ridge_alphas: tuple[float, ...] = (10.0, 50.0, 200.0)
    offset_alphas: tuple[float, ...] = (0.01, 0.10, 1.0)
    inner_time_splits: int = 4
    minimum_inner_train_rows: int = 50
    minimum_residual_samples: int = 50
    minimum_outer_train_rows: int = 200
    minimum_outer_test_rows: int = 100
    residual_smoothing: float = 0.5
    push_prior_strength: float = 50.0
    calibration_bins: int = 10
    probability_epsilon: float = 1e-6
    minimum_winning_seasons: int = 2
    maximum_single_season_log_loss_regression: float = 0.005
    bootstrap_repetitions: int = 2000
    random_seed: int = 42


MARKET_SPECS: dict[str, DistributionalMarketSpec] = {
    "pregame_moneyline": DistributionalMarketSpec(
        market="pregame_moneyline",
        binary_target="target_home_win",
        push_target="target_tie",
        absolute_target="target_home_margin",
        residual_target=None,
        market_line_feature=None,
        baseline_probability_feature="market_home_ml_novig_prob",
        anchor_feature="market_implied_margin",
    ),
    "pregame_ats": DistributionalMarketSpec(
        market="pregame_ats",
        binary_target="target_home_cover",
        push_target="target_ats_push",
        absolute_target="target_home_margin",
        residual_target="target_margin_residual",
        market_line_feature="market_home_spread",
        baseline_probability_feature="market_home_cover_novig_prob",
        anchor_feature=None,
    ),
    "pregame_total": DistributionalMarketSpec(
        market="pregame_total",
        binary_target="target_over",
        push_target="target_total_push",
        absolute_target="target_total_points",
        residual_target="target_total_residual",
        market_line_feature="market_total_line",
        baseline_probability_feature="market_over_novig_prob",
        anchor_feature=None,
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

_CONFIG_COLUMNS = [
    "market",
    "variant",
    "architecture",
    "feature_set",
    "model_name",
    "feature_count",
    "features",
    "market_anchor_mode",
]


def _read_manifest_features(
    compact_root: Path,
    market: str,
    variant: str,
) -> tuple[str, ...]:
    path = compact_root / f"{market}_{variant}.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(value) for value in payload["features"])


def _load_market_surface(
    compact_root: Path,
    market: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    football = pd.read_parquet(
        compact_root / f"{market}_football_only.parquet"
    )
    augmented = pd.read_parquet(
        compact_root / f"{market}_market_augmented.parquet"
    )
    keys = ["game_id", "season", "home_team_id", "away_team_id"]
    football = football.sort_values(
        ["season", "game_id"], kind="stable"
    ).reset_index(drop=True)
    augmented = augmented.sort_values(
        ["season", "game_id"], kind="stable"
    ).reset_index(drop=True)

    if not football[keys].equals(augmented[keys]):
        raise ValueError(f"{market}: compact matrices are not row-aligned.")

    surface = football.copy()
    for column in augmented.columns:
        if column not in surface.columns:
            surface[column] = augmented[column].to_numpy()

    football_features = _read_manifest_features(
        compact_root,
        market,
        "football_only",
    )
    missing = sorted(set(football_features) - set(surface.columns))
    if missing:
        raise ValueError(f"{market}: missing football features: {missing}")
    return surface, football_features


def _feature_group(feature: str) -> str:
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
    ordered = tuple(dict.fromkeys(str(value) for value in features))
    groups = {"context": [], "strength": [], "qb": []}
    for feature in ordered:
        groups[_feature_group(feature)].append(feature)

    def union(*names: str) -> tuple[str, ...]:
        return tuple(
            feature
            for feature in ordered
            if any(feature in groups[name] for name in names)
        )

    raw = {
        "context_only": union("context"),
        "strength_only": union("strength"),
        "context_strength": union("context", "strength"),
        "all_football": union("context", "strength", "qb"),
    }
    output: dict[str, tuple[str, ...]] = {}
    seen: set[tuple[str, ...]] = set()
    for name, values in raw.items():
        if not values or values in seen:
            continue
        seen.add(values)
        output[name] = values
    return output


def _ridge_factory(alpha: float) -> Callable[[], Pipeline]:
    def build() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(alpha))),
            ]
        )
    return build


def _binary_mask(
    frame: pd.DataFrame,
    spec: DistributionalMarketSpec,
) -> np.ndarray:
    target = pd.to_numeric(
        frame[spec.binary_target], errors="coerce"
    )
    push = pd.to_numeric(
        frame[spec.push_target], errors="coerce"
    ).fillna(0)
    return (target.isin([0, 1]) & push.eq(0)).to_numpy(bool)


def _actual_category(
    frame: pd.DataFrame,
    spec: DistributionalMarketSpec,
) -> np.ndarray:
    push = pd.to_numeric(
        frame[spec.push_target], errors="coerce"
    ).fillna(0).to_numpy(int)
    upper = pd.to_numeric(
        frame[spec.binary_target], errors="coerce"
    ).fillna(0).to_numpy(int)
    category = np.zeros(len(frame), dtype=int)
    category[upper == 1] = 2
    category[push == 1] = 1
    return category


def _market_line(
    frame: pd.DataFrame,
    spec: DistributionalMarketSpec,
) -> np.ndarray:
    if spec.market == "pregame_moneyline":
        return np.zeros(len(frame), dtype=float)
    if not spec.market_line_feature:
        raise ValueError(f"{spec.market}: market line is undefined.")
    values = pd.to_numeric(
        frame[spec.market_line_feature], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{spec.market}: market line contains missing values.")
    return values


def _baseline_conditional_probability(
    frame: pd.DataFrame,
    spec: DistributionalMarketSpec,
    variant: str,
) -> np.ndarray:
    if variant == "football_only":
        return np.full(len(frame), 0.5, dtype=float)
    values = pd.to_numeric(
        frame[spec.baseline_probability_feature],
        errors="coerce",
    ).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{spec.market}: market baseline probability is incomplete."
        )
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def _fit_push_prior(
    train: pd.DataFrame,
    spec: DistributionalMarketSpec,
    config: DistributionalTournamentConfig,
) -> EmpiricalPushPrior:
    lines = _market_line(train, spec)
    pushes = pd.to_numeric(
        train[spec.push_target], errors="coerce"
    ).fillna(0).to_numpy(float)
    return EmpiricalPushPrior(
        prior_strength=config.push_prior_strength,
        always_push_possible=spec.market == "pregame_moneyline",
    ).fit(lines, pushes)


def _absolute_threshold_and_grid(
    test: pd.DataFrame,
    spec: DistributionalMarketSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.market == "pregame_moneyline":
        return (
            np.zeros(len(test), dtype=float),
            np.zeros(len(test), dtype=float),
        )
    lines = _market_line(test, spec)
    if spec.market == "pregame_ats":
        return -lines, np.zeros(len(test), dtype=float)
    return lines, np.zeros(len(test), dtype=float)


def _residual_target(
    frame: pd.DataFrame,
    spec: DistributionalMarketSpec,
) -> np.ndarray:
    if spec.market == "pregame_moneyline":
        margin = pd.to_numeric(
            frame[spec.absolute_target], errors="coerce"
        ).to_numpy(float)
        anchor = pd.to_numeric(
            frame[spec.anchor_feature], errors="coerce"
        ).to_numpy(float)
        return margin - anchor
    if not spec.residual_target:
        raise ValueError(f"{spec.market}: residual target is undefined.")
    return pd.to_numeric(
        frame[spec.residual_target], errors="coerce"
    ).to_numpy(float)


def _residual_prediction_surface(
    test: pd.DataFrame,
    prediction: np.ndarray,
    spec: DistributionalMarketSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.market == "pregame_moneyline":
        anchor = pd.to_numeric(
            test[spec.anchor_feature], errors="coerce"
        ).to_numpy(float)
        return (
            anchor + prediction,
            np.zeros(len(test), dtype=float),
            np.zeros(len(test), dtype=float),
        )
    line = _market_line(test, spec)
    if spec.market == "pregame_ats":
        grid = np.mod(line, 1.0)
    else:
        grid = np.mod(-line, 1.0)
    return (
        prediction,
        np.zeros(len(test), dtype=float),
        grid,
    )


def _chronological_residual_samples(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: np.ndarray,
    model_factory: Callable[[], Pipeline],
    config: DistributionalTournamentConfig,
) -> tuple[np.ndarray, str]:
    order = np.lexsort(
        (
            frame["game_id"].astype(str).to_numpy(),
            pd.to_numeric(frame["season"], errors="raise").to_numpy(),
        )
    )
    x = frame.loc[:, features].iloc[order].reset_index(drop=True)
    y = np.asarray(target, dtype=float)[order]
    if not np.isfinite(y).all():
        raise ValueError("Continuous target contains missing values.")

    split_count = min(
        config.inner_time_splits,
        max(2, len(frame) // max(config.minimum_inner_train_rows, 1) - 1),
    )
    splitter = TimeSeriesSplit(n_splits=split_count)
    predictions = np.full(len(y), np.nan, dtype=float)

    for train_index, validation_index in splitter.split(x):
        if len(train_index) < config.minimum_inner_train_rows:
            continue
        model = model_factory()
        model.fit(x.iloc[train_index], y[train_index])
        predictions[validation_index] = model.predict(
            x.iloc[validation_index]
        )

    valid = np.isfinite(predictions)
    residuals = y[valid] - predictions[valid]
    source = "inner_time_series_oof"

    if len(residuals) < config.minimum_residual_samples:
        model = model_factory()
        model.fit(x, y)
        residuals = y - model.predict(x)
        source = "in_sample_fallback"

    if len(residuals) < 20:
        raise ValueError("Insufficient residual samples.")
    residuals = residuals[np.isfinite(residuals)]
    residuals = residuals - float(np.mean(residuals))
    return residuals, source


def _calibration_intercept_slope(
    target: np.ndarray,
    probability: np.ndarray,
    epsilon: float,
) -> tuple[float, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    x = logit(p).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=3000,
    )
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int,
) -> float:
    y = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            output += (
                mask.mean()
                * abs(float(y[mask].mean()) - float(p[mask].mean()))
            )
    return float(output)


def _cluster_id(game_id: object, season: object) -> str:
    text = str(game_id)
    parts = text.split("_")
    week = parts[1] if len(parts) > 1 else "unknown"
    return f"{int(season)}_{week}"


def _prediction_rows(
    test: pd.DataFrame,
    spec: DistributionalMarketSpec,
    *,
    variant: str,
    architecture: str,
    feature_set: str,
    model_name: str,
    features: tuple[str, ...],
    market_anchor_mode: str,
    probabilities: ThreeWayProbabilities,
    baseline_probabilities: ThreeWayProbabilities,
    residual_source: str,
    predicted_mean: np.ndarray | None,
    test_season: int,
    config: DistributionalTournamentConfig,
) -> pd.DataFrame:
    model_matrix = probabilities.as_array()
    baseline_matrix = baseline_probabilities.as_array()
    category = _actual_category(test, spec)
    binary = pd.to_numeric(
        test[spec.binary_target], errors="coerce"
    ).to_numpy(float)
    scored_binary = _binary_mask(test, spec)

    conditional = conditional_upper_probability(
        probabilities,
        epsilon=config.probability_epsilon,
    )
    baseline_conditional = conditional_upper_probability(
        baseline_probabilities,
        epsilon=config.probability_epsilon,
    )
    conditional = np.clip(
        conditional,
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    )
    baseline_conditional = np.clip(
        baseline_conditional,
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    )

    output = pd.DataFrame(
        {
            "game_id": test["game_id"].astype(str).to_numpy(),
            "season": pd.to_numeric(
                test["season"], errors="raise"
            ).to_numpy(int),
            "home_team_id": test["home_team_id"].astype(str).to_numpy(),
            "away_team_id": test["away_team_id"].astype(str).to_numpy(),
            "test_season": int(test_season),
            "market": spec.market,
            "variant": variant,
            "architecture": architecture,
            "feature_set": feature_set,
            "model_name": model_name,
            "feature_count": len(features),
            "features": json.dumps(features),
            "market_anchor_mode": market_anchor_mode,
            "residual_source": residual_source,
            "actual_category": category,
            "binary_target": binary,
            "scored_binary": scored_binary,
            "model_lower_probability": model_matrix[:, 0],
            "model_push_probability": model_matrix[:, 1],
            "model_upper_probability": model_matrix[:, 2],
            "model_conditional_upper_probability": conditional,
            "baseline_lower_probability": baseline_matrix[:, 0],
            "baseline_push_probability": baseline_matrix[:, 1],
            "baseline_upper_probability": baseline_matrix[:, 2],
            "baseline_conditional_upper_probability": baseline_conditional,
            "market_line": _market_line(test, spec),
            "predicted_mean": (
                np.asarray(predicted_mean, dtype=float)
                if predicted_mean is not None
                else np.full(len(test), np.nan)
            ),
        }
    )
    output["cluster_id"] = [
        _cluster_id(game_id, season)
        for game_id, season in zip(output["game_id"], output["season"])
    ]

    epsilon = config.probability_epsilon
    model_true = np.clip(
        model_matrix[np.arange(len(test)), category],
        epsilon,
        1.0,
    )
    baseline_true = np.clip(
        baseline_matrix[np.arange(len(test)), category],
        epsilon,
        1.0,
    )
    output["model_three_way_log_loss_row"] = -np.log(model_true)
    output["baseline_three_way_log_loss_row"] = -np.log(baseline_true)

    one_hot = np.eye(3)[category]
    output["model_three_way_brier_row"] = np.sum(
        (model_matrix - one_hot) ** 2,
        axis=1,
    )
    output["baseline_three_way_brier_row"] = np.sum(
        (baseline_matrix - one_hot) ** 2,
        axis=1,
    )

    model_binary_loss = np.full(len(test), np.nan)
    baseline_binary_loss = np.full(len(test), np.nan)
    model_binary_brier = np.full(len(test), np.nan)
    baseline_binary_brier = np.full(len(test), np.nan)
    y = binary[scored_binary]
    p = conditional[scored_binary]
    p0 = baseline_conditional[scored_binary]
    model_binary_loss[scored_binary] = -(
        y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    )
    baseline_binary_loss[scored_binary] = -(
        y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0)
    )
    model_binary_brier[scored_binary] = (p - y) ** 2
    baseline_binary_brier[scored_binary] = (p0 - y) ** 2

    output["model_binary_log_loss_row"] = model_binary_loss
    output["baseline_binary_log_loss_row"] = baseline_binary_loss
    output["model_binary_brier_row"] = model_binary_brier
    output["baseline_binary_brier_row"] = baseline_binary_brier
    output["actual_push"] = (
        category == 1
    ).astype(int)
    return output


def _baseline_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: DistributionalMarketSpec,
    variant: str,
    config: DistributionalTournamentConfig,
) -> ThreeWayProbabilities:
    push_prior = _fit_push_prior(train, spec, config)
    push_probability = push_prior.predict(_market_line(test, spec))
    conditional = _baseline_conditional_probability(test, spec, variant)
    return combine_conditional_with_push(conditional, push_probability)


def _evaluate_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: DistributionalMarketSpec,
    football_features: tuple[str, ...],
    test_season: int,
    config: DistributionalTournamentConfig,
) -> list[pd.DataFrame]:
    if len(train) < config.minimum_outer_train_rows:
        raise ValueError(
            f"{spec.market} {test_season}: only {len(train)} training rows."
        )
    if len(test) < config.minimum_outer_test_rows:
        raise ValueError(
            f"{spec.market} {test_season}: only {len(test)} test rows."
        )

    feature_sets = _candidate_feature_sets(football_features)
    outputs: list[pd.DataFrame] = []

    baselines = {
        variant: _baseline_probabilities(
            train,
            test,
            spec,
            variant,
            config,
        )
        for variant in ("football_only", "market_augmented")
    }

    for variant in ("football_only", "market_augmented"):
        outputs.append(
            _prediction_rows(
                test,
                spec,
                variant=variant,
                architecture="baseline",
                feature_set=(
                    "coin_flip_baseline"
                    if variant == "football_only"
                    else "market_baseline"
                ),
                model_name="baseline",
                features=(),
                market_anchor_mode=(
                    "not_applicable"
                    if variant == "football_only"
                    else "market_baseline"
                ),
                probabilities=baselines[variant],
                baseline_probabilities=baselines[variant],
                residual_source="not_applicable",
                predicted_mean=None,
                test_season=test_season,
                config=config,
            )
        )

    absolute_target = pd.to_numeric(
        train[spec.absolute_target], errors="coerce"
    ).to_numpy(float)
    absolute_threshold, absolute_grid = _absolute_threshold_and_grid(test, spec)

    for feature_set_name, selected_features in feature_sets.items():
        for alpha in config.ridge_alphas:
            model_name = f"ridge_alpha{alpha:g}"
            factory = _ridge_factory(alpha)
            residuals, residual_source = _chronological_residual_samples(
                train,
                selected_features,
                absolute_target,
                factory,
                config,
            )
            model = factory()
            model.fit(
                train.loc[:, selected_features],
                absolute_target,
            )
            predicted_mean = model.predict(
                test.loc[:, selected_features]
            )
            probabilities = empirical_discrete_probabilities(
                predicted_mean,
                residuals,
                absolute_threshold,
                grid_offset=absolute_grid,
                smoothing=config.residual_smoothing,
            )
            outputs.append(
                _prediction_rows(
                    test,
                    spec,
                    variant="football_only",
                    architecture="absolute_score_discrete",
                    feature_set=feature_set_name,
                    model_name=model_name,
                    features=selected_features,
                    market_anchor_mode=(
                        "not_applicable"
                        if spec.market == "pregame_moneyline"
                        else "line_used_only_for_pricing"
                    ),
                    probabilities=probabilities,
                    baseline_probabilities=baselines["football_only"],
                    residual_source=residual_source,
                    predicted_mean=predicted_mean,
                    test_season=test_season,
                    config=config,
                )
            )

    residual_target = _residual_target(train, spec)
    for feature_set_name, selected_features in feature_sets.items():
        for alpha in config.ridge_alphas:
            model_name = f"ridge_alpha{alpha:g}"
            factory = _ridge_factory(alpha)
            residual_errors, residual_source = _chronological_residual_samples(
                train,
                selected_features,
                residual_target,
                factory,
                config,
            )
            model = factory()
            model.fit(
                train.loc[:, selected_features],
                residual_target,
            )
            correction = model.predict(
                test.loc[:, selected_features]
            )
            predicted_surface, threshold, grid = _residual_prediction_surface(
                test,
                correction,
                spec,
            )
            probabilities = empirical_discrete_probabilities(
                predicted_surface,
                residual_errors,
                threshold,
                grid_offset=grid,
                smoothing=config.residual_smoothing,
            )
            outputs.append(
                _prediction_rows(
                    test,
                    spec,
                    variant="market_augmented",
                    architecture="market_residual_discrete",
                    feature_set=feature_set_name,
                    model_name=model_name,
                    features=selected_features,
                    market_anchor_mode=(
                        "external_market_margin_anchor"
                        if spec.market == "pregame_moneyline"
                        else "market_residual_target_anchor"
                    ),
                    probabilities=probabilities,
                    baseline_probabilities=baselines["market_augmented"],
                    residual_source=residual_source,
                    predicted_mean=predicted_surface,
                    test_season=test_season,
                    config=config,
                )
            )

        train_binary_mask = _binary_mask(train, spec)
        train_conditional = _baseline_conditional_probability(
            train,
            spec,
            "market_augmented",
        )
        test_conditional = _baseline_conditional_probability(
            test,
            spec,
            "market_augmented",
        )
        push_prior = _fit_push_prior(train, spec, config)
        test_push = push_prior.predict(_market_line(test, spec))

        for alpha in config.offset_alphas:
            model_name = f"fixed_offset_alpha{alpha:g}"
            model = FixedOffsetLogistic(alpha=alpha)
            model.fit(
                train.loc[train_binary_mask, selected_features],
                pd.to_numeric(
                    train.loc[train_binary_mask, spec.binary_target],
                    errors="raise",
                ).to_numpy(int),
                train_conditional[train_binary_mask],
            )
            conditional = model.predict_upper_probability(
                test.loc[:, selected_features],
                test_conditional,
            )
            probabilities = combine_conditional_with_push(
                conditional,
                test_push,
            )
            outputs.append(
                _prediction_rows(
                    test,
                    spec,
                    variant="market_augmented",
                    architecture="fixed_market_offset_logistic",
                    feature_set=feature_set_name,
                    model_name=model_name,
                    features=selected_features,
                    market_anchor_mode="fixed_market_logit_offset",
                    probabilities=probabilities,
                    baseline_probabilities=baselines["market_augmented"],
                    residual_source="not_applicable",
                    predicted_mean=None,
                    test_season=test_season,
                    config=config,
                )
            )
    return outputs


def _fold_metrics(
    oof: pd.DataFrame,
    config: DistributionalTournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = _CONFIG_COLUMNS + ["test_season"]
    for keys, frame in oof.groupby(group_columns, sort=True):
        binary = frame[frame["scored_binary"]].copy()
        intercept, slope = _calibration_intercept_slope(
            binary["binary_target"].to_numpy(int),
            binary["model_conditional_upper_probability"].to_numpy(float),
            config.probability_epsilon,
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_all": len(frame),
                "n_binary": len(binary),
                "binary_log_loss": float(
                    binary["model_binary_log_loss_row"].mean()
                ),
                "binary_brier": float(
                    binary["model_binary_brier_row"].mean()
                ),
                "three_way_log_loss": float(
                    frame["model_three_way_log_loss_row"].mean()
                ),
                "three_way_brier": float(
                    frame["model_three_way_brier_row"].mean()
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece": _expected_calibration_error(
                    binary["binary_target"].to_numpy(int),
                    binary[
                        "model_conditional_upper_probability"
                    ].to_numpy(float),
                    config.calibration_bins,
                ),
                "mean_predicted_push": float(
                    frame["model_push_probability"].mean()
                ),
                "actual_push_rate": float(frame["actual_push"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(
    folds: pd.DataFrame,
    oof: pd.DataFrame,
) -> pd.DataFrame:
    fold_aggregate = (
        folds.groupby(_CONFIG_COLUMNS, as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            mean_season_binary_log_loss=("binary_log_loss", "mean"),
            sd_season_binary_log_loss=("binary_log_loss", "std"),
            mean_season_binary_brier=("binary_brier", "mean"),
            mean_season_three_way_log_loss=("three_way_log_loss", "mean"),
            mean_season_three_way_brier=("three_way_brier", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_ece=("ece", "mean"),
            mean_predicted_push=("mean_predicted_push", "mean"),
            actual_push_rate=("actual_push_rate", "mean"),
        )
    )
    fold_aggregate["se_binary_log_loss"] = (
        fold_aggregate["sd_season_binary_log_loss"]
        / np.sqrt(fold_aggregate["seasons"].clip(lower=1))
    )

    pooled_rows: list[dict[str, object]] = []
    for keys, frame in oof.groupby(_CONFIG_COLUMNS, sort=True):
        binary = frame[frame["scored_binary"]]
        pooled_rows.append(
            {
                **dict(zip(_CONFIG_COLUMNS, keys)),
                "n_all": len(frame),
                "n_binary": len(binary),
                "pooled_binary_log_loss": float(
                    binary["model_binary_log_loss_row"].mean()
                ),
                "pooled_binary_brier": float(
                    binary["model_binary_brier_row"].mean()
                ),
                "pooled_three_way_log_loss": float(
                    frame["model_three_way_log_loss_row"].mean()
                ),
                "pooled_three_way_brier": float(
                    frame["model_three_way_brier_row"].mean()
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    return fold_aggregate.merge(
        pooled,
        on=_CONFIG_COLUMNS,
        validate="one_to_one",
    ).sort_values(
        [
            "market",
            "variant",
            "pooled_binary_log_loss",
            "pooled_three_way_log_loss",
            "feature_count",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _stability_and_selection(
    folds: pd.DataFrame,
    aggregate: pd.DataFrame,
    config: DistributionalTournamentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_folds = folds[folds["architecture"].eq("baseline")][
        [
            "market",
            "variant",
            "test_season",
            "binary_log_loss",
            "binary_brier",
            "three_way_log_loss",
            "three_way_brier",
        ]
    ].rename(
        columns={
            "binary_log_loss": "baseline_binary_log_loss",
            "binary_brier": "baseline_binary_brier",
            "three_way_log_loss": "baseline_three_way_log_loss",
            "three_way_brier": "baseline_three_way_brier",
        }
    )
    candidate_folds = folds[
        ~folds["architecture"].eq("baseline")
    ].merge(
        baseline_folds,
        on=["market", "variant", "test_season"],
        how="left",
        validate="many_to_one",
    )
    for metric in (
        "binary_log_loss",
        "binary_brier",
        "three_way_log_loss",
        "three_way_brier",
    ):
        candidate_folds[f"{metric}_gain"] = (
            candidate_folds[f"baseline_{metric}"]
            - candidate_folds[metric]
        )

    stability = (
        candidate_folds.groupby(_CONFIG_COLUMNS, as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            mean_binary_log_loss_gain=("binary_log_loss_gain", "mean"),
            mean_binary_brier_gain=("binary_brier_gain", "mean"),
            mean_three_way_log_loss_gain=("three_way_log_loss_gain", "mean"),
            mean_three_way_brier_gain=("three_way_brier_gain", "mean"),
            binary_log_loss_winning_seasons=(
                "binary_log_loss_gain",
                lambda values: int((values > 0).sum()),
            ),
            worst_binary_log_loss_gain=("binary_log_loss_gain", "min"),
        )
    )
    stability["passes_binary_log_loss"] = (
        stability["mean_binary_log_loss_gain"] > 0
    )
    stability["passes_binary_brier"] = (
        stability["mean_binary_brier_gain"] > 0
    )
    stability["passes_three_way_log_loss"] = (
        stability["mean_three_way_log_loss_gain"] > 0
    )
    stability["passes_three_way_brier"] = (
        stability["mean_three_way_brier_gain"] >= 0
    )
    stability["passes_season_stability"] = (
        stability["binary_log_loss_winning_seasons"]
        >= config.minimum_winning_seasons
    )
    stability["passes_regression_guard"] = (
        stability["worst_binary_log_loss_gain"]
        >= -config.maximum_single_season_log_loss_regression
    )
    stability["qualifies"] = (
        stability["passes_binary_log_loss"]
        & stability["passes_binary_brier"]
        & stability["passes_three_way_log_loss"]
        & stability["passes_three_way_brier"]
        & stability["passes_season_stability"]
        & stability["passes_regression_guard"]
    )

    def reasons(row: pd.Series) -> str:
        failures: list[str] = []
        mapping = [
            ("passes_binary_log_loss", "BINARY_LOG_LOSS_NOT_BETTER"),
            ("passes_binary_brier", "BINARY_BRIER_NOT_BETTER"),
            ("passes_three_way_log_loss", "THREE_WAY_LOG_LOSS_NOT_BETTER"),
            ("passes_three_way_brier", "THREE_WAY_BRIER_NOT_BETTER"),
            ("passes_season_stability", "INSUFFICIENT_WINNING_SEASONS"),
            ("passes_regression_guard", "SINGLE_SEASON_REGRESSION_TOO_LARGE"),
        ]
        for field, label in mapping:
            if not bool(row[field]):
                failures.append(label)
        return "PASS" if not failures else "|".join(failures)

    stability["gate_reasons"] = stability.apply(reasons, axis=1)
    candidates = aggregate[
        ~aggregate["architecture"].eq("baseline")
    ].merge(
        stability,
        on=_CONFIG_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    selected_rows: list[pd.Series] = []
    challenger_rows: list[pd.Series] = []
    for (market, variant), frame in aggregate.groupby(
        ["market", "variant"],
        sort=True,
    ):
        baseline = frame[frame["architecture"].eq("baseline")].iloc[0].copy()
        market_candidates = candidates[
            candidates["market"].eq(market)
            & candidates["variant"].eq(variant)
        ].copy()
        challenger = market_candidates.sort_values(
            [
                "pooled_binary_log_loss",
                "pooled_three_way_log_loss",
                "pooled_binary_brier",
                "feature_count",
            ],
            kind="stable",
        ).iloc[0].copy()
        challenger_rows.append(challenger)

        qualifying = market_candidates[
            market_candidates["qualifies"].fillna(False)
        ].copy()
        if qualifying.empty:
            choice = baseline.copy()
            choice["selection_status"] = (
                "RETAIN_BASELINE_NO_QUALIFYING_DISTRIBUTIONAL_MODEL"
            )
            choice["qualifies"] = False
            choice["gate_reasons"] = str(challenger["gate_reasons"])
            choice["challenger_architecture"] = challenger["architecture"]
            choice["challenger_feature_set"] = challenger["feature_set"]
            choice["challenger_model_name"] = challenger["model_name"]
            choice["challenger_binary_log_loss"] = challenger[
                "pooled_binary_log_loss"
            ]
            choice["challenger_three_way_log_loss"] = challenger[
                "pooled_three_way_log_loss"
            ]
            selected_rows.append(choice)
            continue

        best = qualifying.sort_values(
            ["pooled_binary_log_loss", "pooled_three_way_log_loss"],
            kind="stable",
        ).iloc[0]
        threshold = (
            float(best["mean_season_binary_log_loss"])
            + (
                float(best["se_binary_log_loss"])
                if np.isfinite(best["se_binary_log_loss"])
                else 0.0
            )
        )
        eligible = qualifying[
            qualifying["mean_season_binary_log_loss"] <= threshold
        ].copy()
        choice = eligible.sort_values(
            [
                "feature_count",
                "pooled_binary_log_loss",
                "pooled_three_way_log_loss",
            ],
            kind="stable",
        ).iloc[0].copy()
        choice["selection_status"] = (
            "QUALIFIED_PROVISIONAL_DISTRIBUTIONAL_CANDIDATE"
        )
        choice["challenger_architecture"] = challenger["architecture"]
        choice["challenger_feature_set"] = challenger["feature_set"]
        choice["challenger_model_name"] = challenger["model_name"]
        choice["challenger_binary_log_loss"] = challenger[
            "pooled_binary_log_loss"
        ]
        choice["challenger_three_way_log_loss"] = challenger[
            "pooled_three_way_log_loss"
        ]
        selected_rows.append(choice)

    return (
        stability.sort_values(
            ["market", "variant", "qualifies", "mean_binary_log_loss_gain"],
            ascending=[True, True, False, False],
            kind="stable",
        ).reset_index(drop=True),
        pd.DataFrame(selected_rows).reset_index(drop=True),
        pd.DataFrame(challenger_rows).reset_index(drop=True),
    )


def _bootstrap_selected(
    oof: pd.DataFrame,
    selected: pd.DataFrame,
    challengers: pd.DataFrame,
    config: DistributionalTournamentConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, object]] = []

    for market in MARKET_SPECS:
        for variant in ("football_only", "market_augmented"):
            selected_row = selected[
                selected["market"].eq(market)
                & selected["variant"].eq(variant)
            ].iloc[0]
            if selected_row["architecture"] == "baseline":
                row = challengers[
                    challengers["market"].eq(market)
                    & challengers["variant"].eq(variant)
                ].iloc[0]
                role = "best_challenger"
            else:
                row = selected_row
                role = "selected"

            mask = (
                oof["market"].eq(market)
                & oof["variant"].eq(variant)
                & oof["architecture"].eq(row["architecture"])
                & oof["feature_set"].eq(row["feature_set"])
                & oof["model_name"].eq(row["model_name"])
            )
            frame = oof[mask].copy()
            frame["binary_log_loss_gain"] = (
                frame["baseline_binary_log_loss_row"]
                - frame["model_binary_log_loss_row"]
            )
            frame["binary_brier_gain"] = (
                frame["baseline_binary_brier_row"]
                - frame["model_binary_brier_row"]
            )
            frame["three_way_log_loss_gain"] = (
                frame["baseline_three_way_log_loss_row"]
                - frame["model_three_way_log_loss_row"]
            )
            frame["three_way_brier_gain"] = (
                frame["baseline_three_way_brier_row"]
                - frame["model_three_way_brier_row"]
            )

            cluster_summaries = {}
            for season, season_frame in frame.groupby("season"):
                summaries = []
                for _, cluster in season_frame.groupby("cluster_id"):
                    binary = cluster[cluster["scored_binary"]]
                    summaries.append(
                        {
                            "binary_log_loss_sum": float(
                                binary["binary_log_loss_gain"].sum()
                            ),
                            "binary_brier_sum": float(
                                binary["binary_brier_gain"].sum()
                            ),
                            "binary_n": int(len(binary)),
                            "three_way_log_loss_sum": float(
                                cluster["three_way_log_loss_gain"].sum()
                            ),
                            "three_way_brier_sum": float(
                                cluster["three_way_brier_gain"].sum()
                            ),
                            "all_n": int(len(cluster)),
                        }
                    )
                cluster_summaries[int(season)] = summaries

            bootstrap_values = {
                "binary_log_loss": [],
                "binary_brier": [],
                "three_way_log_loss": [],
                "three_way_brier": [],
            }
            for _ in range(config.bootstrap_repetitions):
                totals = {
                    "binary_log_loss_sum": 0.0,
                    "binary_brier_sum": 0.0,
                    "binary_n": 0,
                    "three_way_log_loss_sum": 0.0,
                    "three_way_brier_sum": 0.0,
                    "all_n": 0,
                }
                for summaries in cluster_summaries.values():
                    indices = rng.integers(
                        0,
                        len(summaries),
                        size=len(summaries),
                    )
                    for index in indices:
                        summary = summaries[int(index)]
                        for key in totals:
                            totals[key] += summary[key]
                bootstrap_values["binary_log_loss"].append(
                    totals["binary_log_loss_sum"] / totals["binary_n"]
                )
                bootstrap_values["binary_brier"].append(
                    totals["binary_brier_sum"] / totals["binary_n"]
                )
                bootstrap_values["three_way_log_loss"].append(
                    totals["three_way_log_loss_sum"] / totals["all_n"]
                )
                bootstrap_values["three_way_brier"].append(
                    totals["three_way_brier_sum"] / totals["all_n"]
                )

            result: dict[str, object] = {
                "market": market,
                "variant": variant,
                "role": role,
                "architecture": row["architecture"],
                "feature_set": row["feature_set"],
                "model_name": row["model_name"],
                "bootstrap_method": "season_stratified_week_cluster",
                "bootstrap_repetitions": config.bootstrap_repetitions,
            }
            point_estimates = {
                "binary_log_loss": float(
                    frame.loc[
                        frame["scored_binary"], "binary_log_loss_gain"
                    ].mean()
                ),
                "binary_brier": float(
                    frame.loc[
                        frame["scored_binary"], "binary_brier_gain"
                    ].mean()
                ),
                "three_way_log_loss": float(
                    frame["three_way_log_loss_gain"].mean()
                ),
                "three_way_brier": float(
                    frame["three_way_brier_gain"].mean()
                ),
            }
            for metric, values in bootstrap_values.items():
                array = np.asarray(values, dtype=float)
                result[f"mean_{metric}_gain"] = point_estimates[metric]
                result[f"{metric}_gain_ci_lower"] = float(
                    np.quantile(array, 0.025)
                )
                result[f"{metric}_gain_ci_upper"] = float(
                    np.quantile(array, 0.975)
                )
                result[f"probability_{metric}_gain_positive"] = float(
                    np.mean(array > 0)
                )
            rows.append(result)
    return pd.DataFrame(rows)


def run_distributional_tournament(
    compact_root: str | Path,
    output_root: str | Path,
    *,
    config: DistributionalTournamentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or DistributionalTournamentConfig()
    if any(season >= 2024 for season in cfg.test_seasons):
        raise ValueError(
            "Distributional selection seasons must remain before 2024."
        )
    compact_path = Path(compact_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    for market, spec in MARKET_SPECS.items():
        surface, football_features = _load_market_surface(
            compact_path,
            market,
        )
        for test_season in cfg.test_seasons:
            train = surface[surface["season"] < test_season].copy()
            test = surface[surface["season"] == test_season].copy()
            predictions.extend(
                _evaluate_outer_fold(
                    train,
                    test,
                    spec,
                    football_features,
                    test_season,
                    cfg,
                )
            )

    oof = pd.concat(predictions, ignore_index=True)
    folds = _fold_metrics(oof, cfg)
    aggregate = _aggregate_metrics(folds, oof)
    stability, selected, challengers = _stability_and_selection(
        folds,
        aggregate,
        cfg,
    )
    bootstrap = _bootstrap_selected(
        oof,
        selected,
        challengers,
        cfg,
    )

    oof.to_parquet(
        output_path / "distributional_oof_predictions.parquet",
        index=False,
    )
    folds.to_csv(
        output_path / "distributional_folds.csv",
        index=False,
    )
    aggregate.to_csv(
        output_path / "distributional_aggregate.csv",
        index=False,
    )
    stability.to_csv(
        output_path / "distributional_stability.csv",
        index=False,
    )
    selected.to_csv(
        output_path / "distributional_selected.csv",
        index=False,
    )
    challengers.to_csv(
        output_path / "distributional_challengers.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_path / "distributional_bootstrap.csv",
        index=False,
    )

    payload = {
        "status": "provisional_before_2024_architecture_confirmation",
        "selection_seasons": list(cfg.test_seasons),
        "untouched_seasons": [2024, 2025],
        "architectures": [
            "absolute_score_discrete",
            "market_residual_discrete",
            "fixed_market_offset_logistic",
        ],
        "selection": [],
    }
    for row in selected.to_dict(orient="records"):
        payload["selection"].append(
            {
                "market": row["market"],
                "variant": row["variant"],
                "selection_status": row["selection_status"],
                "architecture": row["architecture"],
                "feature_set": row["feature_set"],
                "model_name": row["model_name"],
                "feature_count": int(row["feature_count"]),
                "features": json.loads(row["features"]),
                "market_anchor_mode": row["market_anchor_mode"],
                "pooled_binary_log_loss": float(
                    row["pooled_binary_log_loss"]
                ),
                "pooled_binary_brier": float(row["pooled_binary_brier"]),
                "pooled_three_way_log_loss": float(
                    row["pooled_three_way_log_loss"]
                ),
                "pooled_three_way_brier": float(
                    row["pooled_three_way_brier"]
                ),
                "gate_reasons": row.get("gate_reasons", ""),
            }
        )
    (output_path / "distributional_selection.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    config_payload = asdict(cfg)
    for key, value in list(config_payload.items()):
        if isinstance(value, tuple):
            config_payload[key] = list(value)
    (output_path / "distributional_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return folds, aggregate, selected
