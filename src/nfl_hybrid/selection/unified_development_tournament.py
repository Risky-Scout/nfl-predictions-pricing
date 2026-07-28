from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MARKETS = (
    "pregame_moneyline",
    "pregame_ats",
    "pregame_total",
)

MARKET_SOURCE = {
    "pregame_moneyline": "h2h",
    "pregame_ats": "spreads",
    "pregame_total": "totals",
}

MARKET_PROBABILITY_COLUMN = "market_t10_novig_probability"

RARE_MODEL_DEFAULTS = {
    "pregame_moneyline": "tie_beta_strength_400",
    "pregame_ats": "ats_push_blend_strength_64",
    "pregame_total": "total_push_blend_strength_64",
}

CHALLENGER_COLUMNS = {
    "market_line_movement",
    "market_probability_movement",
    "market_t10_line_sd",
    "market_t10_probability_sd",
    "market_opening_horizon_minutes",
    "ats_t10_spread_magnitude",
    "ats_t10_spread_distance_to_3",
    "ats_t10_spread_distance_to_7",
    "ats_t10_spread_integer_flag",
    "ats_t10_spread_half_point_flag",
    "ats_t10_home_favorite_flag",
}


@dataclass(frozen=True)
class UnifiedTournamentConfig:
    warmup_seasons: tuple[int, ...] = (2020,)
    development_seasons: tuple[int, ...] = (2021, 2022, 2023)
    outer_test_seasons: tuple[int, ...] = (2021, 2022, 2023)
    regularization_grid: tuple[float, ...] = (0.5, 2.0, 10.0, 50.0)
    blend_weight_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    feature_sets: tuple[str, ...] = (
        "football_only",
        "movement_only",
        "all_canonical",
    )
    rare_event_defaults: dict[str, str] | None = None
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260327
    probability_clip: float = 1e-9
    ece_bins: int = 10
    minimum_full_coverage_games: int = 854
    statistical_ci_level: float = 0.95
    minimum_winning_seasons: int = 2
    maximum_rare_calibration_error: float = 0.02
    candidate_registry_frozen_before_scoring: bool = True
    legacy_ats_total_distributional_selection_eligible: bool = False
    fresh_consensus_selection_eligible: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "UnifiedTournamentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in (
            "warmup_seasons",
            "development_seasons",
            "outer_test_seasons",
            "regularization_grid",
            "blend_weight_grid",
            "feature_sets",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)

    def rare_defaults(self) -> dict[str, str]:
        return dict(self.rare_event_defaults or RARE_MODEL_DEFAULTS)


@dataclass
class OffsetLogisticModel:
    intercept: float
    coefficients: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    active_columns: list[str]
    regularization: float

    def predict(
        self,
        frame: pd.DataFrame,
        market_probability: np.ndarray,
        clip: float,
    ) -> np.ndarray:
        x = _transform_features(
            frame,
            self.active_columns,
            self.medians,
            self.means,
            self.scales,
        )
        offset = _logit(np.asarray(market_probability, dtype=float), clip)
        linear = offset + self.intercept + x @ self.coefficients
        return _sigmoid(linear)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _logit(probabilities: np.ndarray, clip: float) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)


def _fit_preprocessor(
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if not columns:
        return [], np.array([]), np.array([]), np.array([])

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0, skipna=True).fillna(0.0)
    filled = numeric.fillna(medians)
    means = filled.mean(axis=0)
    raw_scales = filled.std(axis=0, ddof=0).fillna(0.0)

    active = [
        column
        for column in columns
        if float(raw_scales[column]) > 1e-12
    ]
    if not active:
        return [], np.array([]), np.array([]), np.array([])

    return (
        active,
        medians[active].to_numpy(dtype=float),
        means[active].to_numpy(dtype=float),
        raw_scales[active].to_numpy(dtype=float),
    )


def _transform_features(
    frame: pd.DataFrame,
    columns: list[str],
    medians: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=float)

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    missing = ~np.isfinite(numeric)
    if missing.any():
        numeric[missing] = np.take(medians, np.where(missing)[1])
    return (numeric - means) / scales


def fit_offset_logistic(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: np.ndarray,
    market_probability: np.ndarray,
    *,
    regularization: float,
    clip: float,
) -> OffsetLogisticModel:
    active, medians, means, scales = _fit_preprocessor(
        frame,
        feature_columns,
    )
    x = _transform_features(frame, active, medians, means, scales)
    y = np.asarray(target, dtype=float)
    offset = _logit(np.asarray(market_probability, dtype=float), clip)
    n = max(len(y), 1)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        beta = parameters[1:]
        linear = offset + intercept + x @ beta
        probability = _sigmoid(linear)
        loss = float(
            np.mean(np.logaddexp(0.0, linear) - y * linear)
            + 0.5 * regularization * float(beta @ beta) / n
        )
        error = probability - y
        gradient = np.empty_like(parameters)
        gradient[0] = float(np.mean(error))
        if len(beta):
            gradient[1:] = (
                x.T @ error / n
                + regularization * beta / n
            )
        return loss, gradient

    initial = np.zeros(1 + x.shape[1], dtype=float)
    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Offset logistic fit failed: {result.message}")

    return OffsetLogisticModel(
        intercept=float(result.x[0]),
        coefficients=np.asarray(result.x[1:], dtype=float),
        medians=medians,
        means=means,
        scales=scales,
        active_columns=active,
        regularization=float(regularization),
    )


def _binary_log_loss(
    actual: np.ndarray,
    probability: np.ndarray,
    clip: float,
) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _three_way_scores(
    probability_1: np.ndarray,
    probability_2: np.ndarray,
    probability_rare: np.ndarray,
    outcome_index: np.ndarray,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.column_stack(
        [probability_1, probability_2, probability_rare]
    ).astype(float)
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = np.clip(probabilities, clip, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    target = np.zeros_like(probabilities)
    target[np.arange(len(target)), outcome_index.astype(int)] = 1.0

    log_loss = -np.log(
        probabilities[np.arange(len(probabilities)), outcome_index.astype(int)]
    )
    brier = np.sum((probabilities - target) ** 2, axis=1)
    return log_loss, brier


def _market_and_targets(
    market: str,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the conditional market probability, first-side outcome, and
    action mask.

    Binary target columns may legitimately be missing on tie/push rows.
    Therefore.

    Binary target columns may legitimately three-way outcomes are derived from the canonical signed
    settlement variable rather than casting nullable binary targets.
    """
    market_probability = pd.to_numeric(
        frame[MARKET_PROBABILITY_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)

    if market == "pregame_moneyline":
        settlement = pd.to_numeric(
            frame["target_home_margin"],
            errors="raise",
        ).to_numpy(dtype=float)

        declared_rare = pd.to_numeric(
            frame["target_tie"],
            errors="raise",
        ).to_numpy(dtype=int)

    elif market == "pregame_ats":
        settlement = pd.to_numeric(
            frame["target_t10_margin_residual"],
            errors="raise",
        ).to_numpy(dtype=float)

        declared_rare = pd.to_numeric(
            frame["target_t10_ats_push"],
            errors="raise",
        ).to_numpy(dtype=int)

    elif market == "pregame_total":
        settlement = pd.to_numeric(
            frame["target_t10_total_residual"],
            errors="raise",
        ).to_numpy(dtype=float)

        declared_rare = pd.to_numeric(
            frame["target_t10_total_push"],
            errors="raise",
        ).to_numpy(dtype=int)

    else:
        raise ValueError(market)

    rare = np.isclose(
        settlement,
        0.0,
        atol=1e-9,
        rtol=0.0,
    ).astype(int)

    if not np.array_equal(rare, declared_rare):
        mismatch_count = int(np.sum(rare != declared_rare))
        raise ValueError(
            f"{market} canonical rare-event target mismatch: "
            f"{mismatch_count} rows"
        )

    first = (settlement > 0.0).astype(int)
    action = rare == 0

    return market_probability, first, action


def _outcome_index(market: str, frame: pd.DataFrame) -> np.ndarray:
    _, first, action = _market_and_targets(market, frame)
    outcome = np.full(len(frame), 2, dtype=int)
    outcome[action & (first == 1)] = 0
    outcome[action & (first == 0)] = 1
    return outcome


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_sets(
    manifest: dict[str, Any],
) -> dict[str, list[str]]:
    all_features = list(manifest["features"])
    football = [
        column
        for column in all_features
        if column not in CHALLENGER_COLUMNS
        and column != MARKET_PROBABILITY_COLUMN
        and column != "market_t10_consensus_line"
    ]
    movement = [
        column
        for column in all_features
        if column in CHALLENGER_COLUMNS
    ]
    return {
        "football_only": football,
        "movement_only": movement,
        "all_canonical": list(dict.fromkeys(football + movement)),
    }


def _inner_split(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(train["season"].unique().tolist())
    if len(seasons) >= 2:
        validation_season = seasons[-1]
        inner_train = train[train["season"] < validation_season].copy()
        inner_validation = train[
            train["season"].eq(validation_season)
        ].copy()
        if not inner_train.empty and not inner_validation.empty:
            return inner_train, inner_validation

    ordered = train.sort_values(
        ["season", "week", "game_id"],
        kind="stable",
    )
    cutoff = max(1, int(math.floor(0.65 * len(ordered))))
    cutoff = min(cutoff, len(ordered) - 1)
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def _tune_residual_and_blend(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: UnifiedTournamentConfig,
) -> tuple[float, float]:
    inner_train, inner_validation = _inner_split(train)
    market_train, first_train, action_train = _market_and_targets(
        str(train["market_name"].iloc[0]),
        inner_train,
    )
    market_validation, first_validation, action_validation = (
        _market_and_targets(
            str(train["market_name"].iloc[0]),
            inner_validation,
        )
    )

    train_action = inner_train.loc[action_train].copy()
    validation_action = inner_validation.loc[action_validation].copy()
    y_train = first_train[action_train]
    y_validation = first_validation[action_validation]
    p_train = market_train[action_train]
    p_validation = market_validation[action_validation]

    best_lambda = float(config.regularization_grid[0])
    best_loss = math.inf
    fitted_by_lambda: dict[float, OffsetLogisticModel] = {}

    for regularization in config.regularization_grid:
        model = fit_offset_logistic(
            train_action,
            feature_columns,
            y_train,
            p_train,
            regularization=float(regularization),
            clip=config.probability_clip,
        )
        fitted_by_lambda[float(regularization)] = model
        prediction = model.predict(
            validation_action,
            p_validation,
            config.probability_clip,
        )
        loss = _binary_log_loss(
            y_validation,
            prediction,
            config.probability_clip,
        )
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_lambda = float(regularization)

    selected_model = fitted_by_lambda[best_lambda]
    model_prediction = selected_model.predict(
        validation_action,
        p_validation,
        config.probability_clip,
    )

    best_weight = float(config.blend_weight_grid[0])
    best_blend_loss = math.inf
    for weight in config.blend_weight_grid:
        blended = (
            (1.0 - float(weight)) * p_validation
            + float(weight) * model_prediction
        )
        loss = _binary_log_loss(
            y_validation,
            blended,
            config.probability_clip,
        )
        if loss < best_blend_loss - 1e-12:
            best_blend_loss = loss
            best_weight = float(weight)

    return best_lambda, best_weight


def _attach_rare_defaults(
    market: str,
    games: pd.DataFrame,
    rare_predictions: pd.DataFrame,
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    model_name = config.rare_defaults()[market]
    rare = rare_predictions[
        rare_predictions["market"].eq(market)
        & rare_predictions["model_name"].eq(model_name)
    ][["game_id", "probability_rare"]].copy()
    if rare["game_id"].duplicated().any():
        raise ValueError(f"Duplicate rare-event rows for {market}/{model_name}")

    output = games.merge(
        rare.rename(columns={"probability_rare": "default_rare_probability"}),
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    missing = output.loc[
        output["season"].isin(config.development_seasons),
        "default_rare_probability",
    ].isna()
    if missing.any():
        raise ValueError(
            f"Missing default rare-event probabilities for {market}: "
            f"{int(missing.sum())}"
        )
    return output


def _candidate_rows(
    frame: pd.DataFrame,
    *,
    market: str,
    model_family: str,
    model_name: str,
    probability_1: np.ndarray,
    probability_2: np.ndarray,
    probability_rare: np.ndarray,
    selection_eligible: bool,
    training_cutoff: Iterable[Any],
    feature_set: str,
    calibration_method: str = "none",
    metadata: dict[str, Any] | None = None,
    clip: float,
) -> pd.DataFrame:
    outcome = _outcome_index(market, frame)
    log_loss, brier = _three_way_scores(
        probability_1,
        probability_2,
        probability_rare,
        outcome,
        clip,
    )
    rows = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str).to_numpy(),
            "season": frame["season"].astype(int).to_numpy(),
            "week": frame["week"].astype(int).to_numpy(),
            "kickoff_utc": frame["kickoff_utc"].to_numpy(),
            "market": market,
            "model_family": model_family,
            "model_name": model_name,
            "feature_set": feature_set,
            "calibration_method": calibration_method,
            "training_cutoff_utc": list(training_cutoff),
            "probability_1": probability_1,
            "probability_2": probability_2,
            "probability_rare": probability_rare,
            "outcome_index": outcome,
            "log_loss": log_loss,
            "brier": brier,
            "selection_eligible": bool(selection_eligible),
        }
    )
    for key, value in (metadata or {}).items():
        rows[key] = value
    return rows


def build_residual_oof(
    *,
    market: str,
    games: pd.DataFrame,
    manifest: dict[str, Any],
    config: UnifiedTournamentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_map = _feature_sets(manifest)
    ledger_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []

    for feature_set_name in config.feature_sets:
        feature_columns = feature_map[feature_set_name]

        for test_season in config.outer_test_seasons:
            train = games[games["season"] < test_season].copy()
            test = games[games["season"].eq(test_season)].copy()
            if train.empty or test.empty:
                raise ValueError(
                    f"Empty chronological fold for {market}/{test_season}"
                )

            train["market_name"] = market
            test["market_name"] = market

            regularization, blend_weight = _tune_residual_and_blend(
                train,
                feature_columns,
                config,
            )

            market_train, first_train, action_train = _market_and_targets(
                market,
                train,
            )
            train_action = train.loc[action_train].copy()

            model = fit_offset_logistic(
                train_action,
                feature_columns,
                first_train[action_train],
                market_train[action_train],
                regularization=regularization,
                clip=config.probability_clip,
            )

            market_test, _, _ = _market_and_targets(market, test)
            conditional_model = model.predict(
                test,
                market_test,
                config.probability_clip,
            )
            conditional_blend = (
                (1.0 - blend_weight) * market_test
                + blend_weight * conditional_model
            )
            rare = test["default_rare_probability"].to_numpy(dtype=float)
            action_mass = 1.0 - rare

            cutoff = pd.Series(
                pd.to_datetime(test["kickoff_utc"], utc=True)
                - pd.Timedelta(minutes=10),
                index=test.index,
            )

            common_metadata = {
                "outer_test_season": test_season,
                "regularization": regularization,
                "blend_weight": blend_weight,
                "feature_count": len(feature_columns),
            }

            ledger_parts.append(
                _candidate_rows(
                    test,
                    market=market,
                    model_family="canonical_market_residual",
                    model_name=(
                        f"canonical_residual_{feature_set_name}"
                    ),
                    probability_1=action_mass * conditional_model,
                    probability_2=action_mass * (1.0 - conditional_model),
                    probability_rare=rare,
                    selection_eligible=True,
                    training_cutoff=cutoff,
                    feature_set=feature_set_name,
                    metadata=common_metadata,
                    clip=config.probability_clip,
                )
            )

            ledger_parts.append(
                _candidate_rows(
                    test,
                    market=market,
                    model_family="constrained_market_blend",
                    model_name=(
                        f"canonical_blend_{feature_set_name}"
                    ),
                    probability_1=action_mass * conditional_blend,
                    probability_2=action_mass * (1.0 - conditional_blend),
                    probability_rare=rare,
                    selection_eligible=True,
                    training_cutoff=cutoff,
                    feature_set=feature_set_name,
                    metadata=common_metadata,
                    clip=config.probability_clip,
                )
            )

            tuning_rows.append(
                {
                    "market": market,
                    "feature_set": feature_set_name,
                    "test_season": test_season,
                    "regularization": regularization,
                    "blend_weight": blend_weight,
                    "feature_count": len(feature_columns),
                    "active_feature_count": len(model.active_columns),
                    "training_games": int(train["game_id"].nunique()),
                    "test_games": int(test["game_id"].nunique()),
                }
            )

    return pd.concat(ledger_parts, ignore_index=True), pd.DataFrame(tuning_rows)


def _load_warehouse_metadata(path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for holdout-safe metadata loading."
        ) from exc

    columns = set(pq.ParquetFile(path).schema_arrow.names)
    kickoff_column = next(
        (
            column
            for column in (
                "scheduled_kickoff_utc",
                "kickoff_utc",
                "start_time_utc",
                "game_datetime_utc",
            )
            if column in columns
        ),
        None,
    )
    if kickoff_column is None:
        raise ValueError("Warehouse has no kickoff timestamp column.")

    metadata = pd.read_parquet(
        path,
        columns=["game_id", kickoff_column],
    ).rename(columns={kickoff_column: "kickoff_utc"})
    metadata["game_id"] = metadata["game_id"].astype(str)
    metadata["kickoff_utc"] = pd.to_datetime(
        metadata["kickoff_utc"],
        utc=True,
        errors="raise",
    )
    if metadata["game_id"].duplicated().any():
        raise ValueError("Duplicate warehouse game IDs.")
    return metadata


def _load_canonical_market(
    root: Path,
    market: str,
    warehouse_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stem = f"{market}_market_augmented_canonical_t10"
    frame = pd.read_parquet(root / f"{stem}.parquet")
    manifest = _load_manifest(root / f"{stem}.manifest.json")

    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.merge(
        warehouse_metadata,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    if frame["kickoff_utc"].isna().any():
        raise ValueError(f"Missing kickoff metadata for {market}.")
    return frame.sort_values(
        ["season", "week", "game_id"],
        kind="stable",
    ).reset_index(drop=True), manifest


def build_market_baseline(
    market: str,
    games: pd.DataFrame,
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    development = games[
        games["season"].isin(config.development_seasons)
    ].copy()
    market_probability = development[
        MARKET_PROBABILITY_COLUMN
    ].to_numpy(dtype=float)
    rare = development["default_rare_probability"].to_numpy(dtype=float)
    action = 1.0 - rare
    cutoff = (
        pd.to_datetime(development["kickoff_utc"], utc=True)
        - pd.Timedelta(minutes=10)
    )
    return _candidate_rows(
        development,
        market=market,
        model_family="market_baseline",
        model_name="market_t10_canonical",
        probability_1=action * market_probability,
        probability_2=action * (1.0 - market_probability),
        probability_rare=rare,
        selection_eligible=True,
        training_cutoff=cutoff,
        feature_set="canonical_anchor_only",
        clip=config.probability_clip,
    )


def ingest_standard_oof(
    path: Path,
    canonical_games: dict[str, pd.DataFrame],
    *,
    model_family: str,
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    source = pd.read_parquet(path)
    parts: list[pd.DataFrame] = []

    for market in MARKETS:
        group = source[source["market"].eq(market)].copy()
        if group.empty:
            continue

        canonical = canonical_games[market][
            ["game_id", "season", "week", "kickoff_utc"]
        ].copy()

        group["game_id"] = group["game_id"].astype(str)
        group = group.merge(
            canonical,
            on="game_id",
            how="inner",
            suffixes=("", "_canonical"),
            validate="many_to_one",
        )
        group = group[
            group["season_canonical"].isin(config.development_seasons)
        ].copy()
        group["season"] = group["season_canonical"].astype(int)
        group["week"] = group["week_canonical"].astype(int)
        group["kickoff_utc"] = group["kickoff_utc_canonical"]

        for model_name, candidate in group.groupby("model_name", sort=False):
            metadata_columns = [
                column
                for column in (
                    "rare_family",
                    "rare_prior_strength",
                    "training_games_seen",
                )
                if column in candidate.columns
            ]
            metadata = {
                column: candidate[column].iloc[0]
                for column in metadata_columns
            }
            parts.append(
                _candidate_rows(
                    canonical_games[market].set_index("game_id").loc[
                        candidate["game_id"]
                    ].reset_index(),
                    market=market,
                    model_family=model_family,
                    model_name=str(model_name),
                    probability_1=candidate["probability_1"].to_numpy(
                        dtype=float
                    ),
                    probability_2=candidate["probability_2"].to_numpy(
                        dtype=float
                    ),
                    probability_rare=candidate[
                        "probability_rare"
                    ].to_numpy(dtype=float),
                    selection_eligible=bool(
                        candidate.get(
                            "selection_eligible",
                            pd.Series(True, index=candidate.index),
                        ).all()
                    ),
                    training_cutoff=candidate.get(
                        "training_cutoff_utc",
                        candidate["kickoff_utc"]
                        - pd.Timedelta(minutes=10),
                    ),
                    feature_set="external_oof",
                    metadata=metadata,
                    clip=config.probability_clip,
                )
            )

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def ingest_calibrated_distributional(
    path: Path,
    canonical_games: dict[str, pd.DataFrame],
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    source = pd.read_parquet(path)
    parts: list[pd.DataFrame] = []

    for market in MARKETS:
        group = source[source["market"].eq(market)].copy()
        if group.empty:
            continue

        canonical = canonical_games[market].set_index("game_id")
        group["game_id"] = group["game_id"].astype(str)
        group = group[group["game_id"].isin(canonical.index)].copy()

        identity = [
            "variant",
            "architecture",
            "feature_set",
            "model_name",
        ]

        for keys, candidate in group.groupby(identity, sort=False):
            candidate = candidate.sort_values("game_id", kind="stable")
            game_frame = canonical.loc[
                candidate["game_id"]
            ].reset_index()

            base_name = "__".join(str(value) for value in keys)
            selection_eligible = (
                market == "pregame_moneyline"
                or config.legacy_ats_total_distributional_selection_eligible
            )

            for suffix, columns, calibration_method in (
                (
                    "uncalibrated",
                    (
                        "model_upper_probability",
                        "model_lower_probability",
                        "model_push_probability",
                    ),
                    "none",
                ),
                (
                    "calibrated",
                    (
                        "calibrated_upper_probability",
                        "calibrated_lower_probability",
                        "calibrated_push_probability",
                    ),
                    str(candidate["calibration_method"].iloc[0]),
                ),
            ):
                parts.append(
                    _candidate_rows(
                        game_frame,
                        market=market,
                        model_family="existing_distributional",
                        model_name=f"{base_name}__{suffix}",
                        probability_1=candidate[columns[0]].to_numpy(
                            dtype=float
                        ),
                        probability_2=candidate[columns[1]].to_numpy(
                            dtype=float
                        ),
                        probability_rare=candidate[columns[2]].to_numpy(
                            dtype=float
                        ),
                        selection_eligible=selection_eligible,
                        training_cutoff=(
                            pd.to_datetime(
                                game_frame["kickoff_utc"],
                                utc=True,
                            )
                            - pd.Timedelta(minutes=10)
                        ),
                        feature_set=str(keys[2]),
                        calibration_method=calibration_method,
                        metadata={
                            "legacy_target_alignment": (
                                "compatible_moneyline"
                                if market == "pregame_moneyline"
                                else "legacy_line_not_selection_eligible"
                            )
                        },
                        clip=config.probability_clip,
                    )
                )

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ece_classwise(
    probabilities: np.ndarray,
    outcome: np.ndarray,
    bins: int,
) -> float:
    errors = []
    for class_index in range(3):
        p = probabilities[:, class_index]
        y = (outcome == class_index).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        class_error = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            if upper == 1.0:
                mask = (p >= lower) & (p <= upper)
            else:
                mask = (p >= lower) & (p < upper)
            if not mask.any():
                continue
            class_error += (
                mask.mean()
                * abs(float(p[mask].mean()) - float(y[mask].mean()))
            )
        errors.append(class_error)
    return float(np.mean(errors))


def summarize_candidates(
    ledger: pd.DataFrame,
    config: UnifiedTournamentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    season_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for (
        market,
        model_family,
        model_name,
    ), group in ledger.groupby(
        ["market", "model_family", "model_name"],
        sort=False,
    ):
        summary_rows.append(
            {
                "market": market,
                "model_family": model_family,
                "model_name": model_name,
                "games": int(group["game_id"].nunique()),
                "rows": int(len(group)),
                "mean_log_loss": float(group["log_loss"].mean()),
                "mean_brier": float(group["brier"].mean()),
                "mean_predicted_rare": float(
                    group["probability_rare"].mean()
                ),
                "actual_rare_rate": float(
                    group["outcome_index"].eq(2).mean()
                ),
                "rare_calibration_error": float(
                    group["probability_rare"].mean()
                    - group["outcome_index"].eq(2).mean()
                ),
                "selection_eligible": bool(
                    group["selection_eligible"].all()
                ),
            }
        )

        for season, season_group in group.groupby("season"):
            season_rows.append(
                {
                    "market": market,
                    "model_family": model_family,
                    "model_name": model_name,
                    "season": int(season),
                    "games": int(
                        season_group["game_id"].nunique()
                    ),
                    "mean_log_loss": float(
                        season_group["log_loss"].mean()
                    ),
                    "mean_brier": float(
                        season_group["brier"].mean()
                    ),
                }
            )

        probabilities = group[
            ["probability_1", "probability_2", "probability_rare"]
        ].to_numpy(dtype=float)
        outcome = group["outcome_index"].to_numpy(dtype=int)
        calibration_rows.append(
            {
                "market": market,
                "model_family": model_family,
                "model_name": model_name,
                "games": int(group["game_id"].nunique()),
                "classwise_ece": _ece_classwise(
                    probabilities,
                    outcome,
                    config.ece_bins,
                ),
                "rare_mean_probability": float(
                    probabilities[:, 2].mean()
                ),
                "rare_actual_rate": float((outcome == 2).mean()),
                "rare_calibration_error": float(
                    probabilities[:, 2].mean()
                    - (outcome == 2).mean()
                ),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(season_rows),
        pd.DataFrame(calibration_rows),
    )


def paired_bootstrap(
    ledger: pd.DataFrame,
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.bootstrap_seed)
    alpha = 1.0 - config.statistical_ci_level
    rows: list[dict[str, Any]] = []

    for market in MARKETS:
        market_rows = ledger[ledger["market"].eq(market)].copy()
        baseline = market_rows[
            market_rows["model_name"].eq("market_t10_canonical")
        ][["game_id", "season", "week", "log_loss", "brier"]].rename(
            columns={
                "log_loss": "baseline_log_loss",
                "brier": "baseline_brier",
            }
        )
        if baseline["game_id"].duplicated().any():
            raise ValueError(f"Duplicate baseline rows for {market}")

        for (
            family,
            model_name,
        ), candidate in market_rows.groupby(
            ["model_family", "model_name"],
            sort=False,
        ):
            if model_name == "market_t10_canonical":
                continue
            if not bool(candidate["selection_eligible"].all()):
                continue

            paired = candidate.merge(
                baseline,
                on=["game_id", "season", "week"],
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                continue

            paired["log_loss_gain"] = (
                paired["baseline_log_loss"] - paired["log_loss"]
            )
            paired["brier_gain"] = (
                paired["baseline_brier"] - paired["brier"]
            )
            paired["cluster"] = (
                paired["season"].astype(str)
                + "_"
                + paired["week"].astype(str)
            )

            cluster_groups = {
                cluster: group
                for cluster, group in paired.groupby("cluster")
            }
            clusters_by_season = {
                season: sorted(
                    paired.loc[
                        paired["season"].eq(season),
                        "cluster",
                    ].unique()
                )
                for season in sorted(paired["season"].unique())
            }

            log_samples = np.empty(
                config.bootstrap_repetitions,
                dtype=float,
            )
            brier_samples = np.empty(
                config.bootstrap_repetitions,
                dtype=float,
            )

            for repetition in range(config.bootstrap_repetitions):
                selected_frames = []
                for season, clusters in clusters_by_season.items():
                    chosen = rng.choice(
                        clusters,
                        size=len(clusters),
                        replace=True,
                    )
                    selected_frames.extend(
                        cluster_groups[cluster] for cluster in chosen
                    )
                sample = pd.concat(selected_frames, ignore_index=True)
                log_samples[repetition] = sample[
                    "log_loss_gain"
                ].mean()
                brier_samples[repetition] = sample["brier_gain"].mean()

            rows.append(
                {
                    "market": market,
                    "model_family": family,
                    "model_name": model_name,
                    "paired_games": int(paired["game_id"].nunique()),
                    "mean_log_loss_gain": float(
                        paired["log_loss_gain"].mean()
                    ),
                    "log_loss_ci_lower": float(
                        np.quantile(log_samples, alpha / 2.0)
                    ),
                    "log_loss_ci_upper": float(
                        np.quantile(log_samples, 1.0 - alpha / 2.0)
                    ),
                    "probability_log_loss_gain_positive": float(
                        np.mean(log_samples > 0.0)
                    ),
                    "mean_brier_gain": float(
                        paired["brier_gain"].mean()
                    ),
                    "brier_ci_lower": float(
                        np.quantile(brier_samples, alpha / 2.0)
                    ),
                    "brier_ci_upper": float(
                        np.quantile(brier_samples, 1.0 - alpha / 2.0)
                    ),
                    "probability_brier_gain_positive": float(
                        np.mean(brier_samples > 0.0)
                    ),
                }
            )

    return pd.DataFrame(rows)


def select_models(
    summary: pd.DataFrame,
    season: pd.DataFrame,
    bootstrap: pd.DataFrame,
    calibration: pd.DataFrame,
    config: UnifiedTournamentConfig,
) -> pd.DataFrame:
    candidates = (
        summary.merge(
            bootstrap,
            on=["market", "model_family", "model_name"],
            how="left",
        )
        .merge(
            calibration[
                [
                    "market",
                    "model_family",
                    "model_name",
                    "classwise_ece",
                ]
            ],
            on=["market", "model_family", "model_name"],
            how="left",
        )
    )

    winning = []
    for (
        market,
        family,
        model_name,
    ), group in season.groupby(
        ["market", "model_family", "model_name"]
    ):
        baseline = season[
            season["market"].eq(market)
            & season["model_name"].eq("market_t10_canonical")
        ][["season", "mean_log_loss", "mean_brier"]].rename(
            columns={
                "mean_log_loss": "baseline_log_loss",
                "mean_brier": "baseline_brier",
            }
        )
        paired = group.merge(baseline, on="season", how="inner")
        winning.append(
            {
                "market": market,
                "model_family": family,
                "model_name": model_name,
                "log_loss_winning_seasons": int(
                    (
                        paired["mean_log_loss"]
                        < paired["baseline_log_loss"]
                    ).sum()
                ),
                "brier_winning_seasons": int(
                    (
                        paired["mean_brier"]
                        < paired["baseline_brier"]
                    ).sum()
                ),
                "worst_season_log_loss_gain": float(
                    (
                        paired["baseline_log_loss"]
                        - paired["mean_log_loss"]
                    ).min()
                )
                if not paired.empty
                else math.nan,
            }
        )
    candidates = candidates.merge(
        pd.DataFrame(winning),
        on=["market", "model_family", "model_name"],
        how="left",
    )

    status = []
    for row in candidates.itertuples(index=False):
        if row.model_name == "market_t10_canonical":
            status.append("BASELINE")
            continue
        if not bool(row.selection_eligible):
            status.append("NOT_SELECTION_ELIGIBLE")
            continue
        full_coverage = row.games >= config.minimum_full_coverage_games
        rare_ok = (
            abs(row.rare_calibration_error)
            <= config.maximum_rare_calibration_error
        )
        statistically_supported = (
            full_coverage
            and rare_ok
            and row.mean_log_loss_gain > 0
            and row.mean_brier_gain > 0
            and row.log_loss_ci_lower > 0
            and row.brier_ci_lower >= 0
            and row.log_loss_winning_seasons
            >= config.minimum_winning_seasons
            and row.brier_winning_seasons
            >= config.minimum_winning_seasons
        )
        provisional = (
            full_coverage
            and rare_ok
            and row.mean_log_loss_gain > 0
            and row.mean_brier_gain > 0
            and row.log_loss_winning_seasons
            >= config.minimum_winning_seasons
        )
        if statistically_supported:
            status.append("STATISTICALLY_SUPPORTED")
        elif provisional:
            status.append("PROVISIONAL_ONLY")
        else:
            status.append("REJECTED")

    candidates["selection_status"] = status

    decisions = []
    for market in MARKETS:
        group = candidates[candidates["market"].eq(market)].copy()
        if group.empty:
            continue
        supported = group[
            group["selection_status"].eq("STATISTICALLY_SUPPORTED")
        ].sort_values(["mean_log_loss", "mean_brier"])
        provisional = group[
            group["selection_status"].eq("PROVISIONAL_ONLY")
        ].sort_values(["mean_log_loss", "mean_brier"])
        baseline_rows = group[
            group["model_name"].eq("market_t10_canonical")
        ]
        if baseline_rows.empty:
            raise ValueError(f"Missing canonical market baseline for {market}")
        baseline = baseline_rows.iloc[0]

        if not supported.empty:
            selected = supported.iloc[0]
            decision = "SELECT_FOR_2024_CONFIRMATION"
        elif not provisional.empty:
            selected = provisional.iloc[0]
            decision = "PROVISIONAL_FOR_2024_CONFIRMATION"
        else:
            selected = baseline
            decision = "RETAIN_MARKET_BASELINE"

        decisions.append(
            {
                "market": market,
                "decision": decision,
                "selected_model_family": selected["model_family"],
                "selected_model_name": selected["model_name"],
                "selection_status": selected["selection_status"],
                "games": int(selected["games"]),
                "mean_log_loss": float(selected["mean_log_loss"]),
                "mean_brier": float(selected["mean_brier"]),
                "mean_log_loss_gain": float(
                    selected.get("mean_log_loss_gain", 0.0)
                    if pd.notna(selected.get("mean_log_loss_gain", np.nan))
                    else 0.0
                ),
                "mean_brier_gain": float(
                    selected.get("mean_brier_gain", 0.0)
                    if pd.notna(selected.get("mean_brier_gain", np.nan))
                    else 0.0
                ),
                "log_loss_ci_lower": (
                    None
                    if pd.isna(selected.get("log_loss_ci_lower", np.nan))
                    else float(selected["log_loss_ci_lower"])
                ),
                "log_loss_ci_upper": (
                    None
                    if pd.isna(selected.get("log_loss_ci_upper", np.nan))
                    else float(selected["log_loss_ci_upper"])
                ),
                "brier_ci_lower": (
                    None
                    if pd.isna(selected.get("brier_ci_lower", np.nan))
                    else float(selected["brier_ci_lower"])
                ),
                "brier_ci_upper": (
                    None
                    if pd.isna(selected.get("brier_ci_upper", np.nan))
                    else float(selected["brier_ci_upper"])
                ),
            }
        )

    return candidates, pd.DataFrame(decisions)


def run_unified_tournament(
    *,
    canonical_root: str | Path,
    warehouse_path: str | Path,
    spreadsheet_oof_path: str | Path,
    rare_event_oof_path: str | Path,
    calibrated_distributional_path: str | Path,
    freshness_consensus_path: str | Path,
    output_root: str | Path,
    config_path: str | Path,
) -> dict[str, pd.DataFrame]:
    canonical_root = Path(canonical_root).expanduser().resolve()
    warehouse_path = Path(warehouse_path).expanduser().resolve()
    spreadsheet_oof_path = Path(spreadsheet_oof_path).expanduser().resolve()
    rare_event_oof_path = Path(rare_event_oof_path).expanduser().resolve()
    calibrated_distributional_path = (
        Path(calibrated_distributional_path).expanduser().resolve()
    )
    freshness_consensus_path = (
        Path(freshness_consensus_path).expanduser().resolve()
    )
    output_root = Path(output_root).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = UnifiedTournamentConfig.from_json(config_path)
    warehouse_metadata = _load_warehouse_metadata(warehouse_path)
    rare_predictions = pd.read_parquet(rare_event_oof_path)

    canonical_games: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    ledger_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []

    for market in MARKETS:
        games, manifest = _load_canonical_market(
            canonical_root,
            market,
            warehouse_metadata,
        )
        games = _attach_rare_defaults(
            market,
            games,
            rare_predictions,
            config,
        )
        canonical_games[market] = games
        manifests[market] = manifest

        ledger_parts.append(
            build_market_baseline(market, games, config)
        )

        residual, tuning = build_residual_oof(
            market=market,
            games=games,
            manifest=manifest,
            config=config,
        )
        ledger_parts.append(residual)
        tuning_parts.append(tuning)

    spreadsheet = ingest_standard_oof(
        spreadsheet_oof_path,
        canonical_games,
        model_family="spreadsheet_baseline",
        config=config,
    )
    if not spreadsheet.empty:
        ledger_parts.append(spreadsheet)

    rare = ingest_standard_oof(
        rare_event_oof_path,
        canonical_games,
        model_family="rare_event_market_baseline",
        config=config,
    )
    if not rare.empty:
        ledger_parts.append(rare)

    distributional = ingest_calibrated_distributional(
        calibrated_distributional_path,
        canonical_games,
        config,
    )
    if not distributional.empty:
        ledger_parts.append(distributional)

    ledger = pd.concat(ledger_parts, ignore_index=True)
    ledger["model_version"] = _sha256(config_path)
    ledger["feature_set_hash"] = ledger["feature_set"].map(
        lambda value: hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()
    )
    ledger["market_anchor_source"] = "verified_original_t10"
    ledger["fold"] = ledger["season"].map(
        lambda season: f"outer_{int(season)}"
    )

    duplicate_key = [
        "game_id",
        "market",
        "model_family",
        "model_name",
    ]
    duplicates = ledger.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        example = ledger.loc[duplicates, duplicate_key].head(20)
        raise ValueError(
            "Unified ledger contains duplicate candidate rows:\n"
            + example.to_string(index=False)
        )

    summary, season, calibration = summarize_candidates(
        ledger,
        config,
    )
    bootstrap = paired_bootstrap(ledger, config)
    candidate_results, decisions = select_models(
        summary,
        season,
        bootstrap,
        calibration,
        config,
    )

    tuning = pd.concat(tuning_parts, ignore_index=True)

    freshness = pd.read_parquet(freshness_consensus_path)
    closing_freshness = freshness[
        freshness["horizon"].eq("closing_t10")
    ].copy()
    freshness_audit = (
        closing_freshness.groupby(["season", "market"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            minimum_books=("eligible_books", "min"),
            median_books=("eligible_books", "median"),
            maximum_quote_age_minutes=(
                "maximum_market_quote_age_minutes",
                "max",
            ),
        )
    )

    ledger.to_parquet(
        output_root / "development_oof_predictions.parquet",
        index=False,
    )
    candidate_results.to_csv(
        output_root / "market_relative_scorecard.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_root / "paired_cluster_bootstrap.csv",
        index=False,
    )
    season.to_csv(
        output_root / "season_stability.csv",
        index=False,
    )
    calibration.to_csv(
        output_root / "calibration_report.csv",
        index=False,
    )
    tuning.to_csv(
        output_root / "nested_tuning_report.csv",
        index=False,
    )
    decisions.to_csv(
        output_root / "selection_decisions.csv",
        index=False,
    )
    freshness_audit.to_csv(
        output_root / "freshness_confirmation.csv",
        index=False,
    )

    registry = {
        "status": "PASS",
        "candidate_registry_frozen_before_scoring":
            config.candidate_registry_frozen_before_scoring,
        "config": asdict(config),
        "candidate_count": int(
            candidate_results[
                ["market", "model_family", "model_name"]
            ].drop_duplicates().shape[0]
        ),
        "source_hashes": {
            "config": _sha256(config_path),
            "spreadsheet_oof": _sha256(spreadsheet_oof_path),
            "rare_event_oof": _sha256(rare_event_oof_path),
            "calibrated_distributional":
                _sha256(calibrated_distributional_path),
            "freshness_consensus": _sha256(freshness_consensus_path),
        },
    }
    (
        output_root / "candidate_registry.json"
    ).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    proposed_spec = {
        "status": "PROPOSED_NOT_FROZEN",
        "development_seasons": list(config.development_seasons),
        "architecture_confirmation_season": 2024,
        "untouched_final_test_season": 2025,
        "market_decisions": decisions.to_dict(orient="records"),
        "config_sha256": _sha256(config_path),
        "holdout_access": {
            "2024_accessed": False,
            "2025_accessed": False,
        },
    }
    (
        output_root / "proposed_frozen_model_spec.json"
    ).write_text(
        json.dumps(proposed_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lab_record = {
        "status": "PASS",
        "rows": int(len(ledger)),
        "games": int(ledger["game_id"].nunique()),
        "markets": sorted(ledger["market"].unique().tolist()),
        "development_seasons": sorted(
            ledger["season"].unique().tolist()
        ),
        "2024_accessed": False,
        "2025_accessed": False,
        "notes": [
            "Freshness-qualified consensus confirmed quote quality.",
            "Original verified T-10 consensus retained as canonical anchor.",
            "Legacy ATS and total distributional candidates are audit-only.",
        ],
    }
    (
        output_root / "development_lab_record.json"
    ).write_text(
        json.dumps(lab_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "ledger": ledger,
        "scorecard": candidate_results,
        "bootstrap": bootstrap,
        "season": season,
        "calibration": calibration,
        "decisions": decisions,
        "tuning": tuning,
        "freshness": freshness_audit,
    }
