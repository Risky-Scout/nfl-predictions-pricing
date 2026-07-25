from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TeamPriorConfig:
    season_weights: tuple[float, ...] = (0.60, 0.28, 0.12)
    prior_strength: float = 500.0
    minimum_between_entity_sd: float = 0.02
    target_season: int = 2026

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray(self.season_weights, dtype=float)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("season_weights must be nonnegative and sum above zero.")
        return weights / weights.sum()


class EmpiricalBayesTeamPrior:
    """Recency-weighted empirical-Bayes priors for team/unit metrics.

    Input is long format with one row per entity-season-metric and columns:
    entity_id, season, metric, value, sample_size.

    The estimate is first weighted by season recency and sample size, then
    shrunk toward the league mean using `n_eff / (n_eff + prior_strength)`.
    Between-entity variance provides posterior uncertainty.
    """

    REQUIRED = {"entity_id", "season", "metric", "value", "sample_size"}

    def __init__(self, config: TeamPriorConfig | None = None) -> None:
        self.config = config or TeamPriorConfig()

    def build(
        self,
        history: pd.DataFrame,
        *,
        target_season: int | None = None,
        as_of_utc: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        missing = self.REQUIRED - set(history.columns)
        if missing:
            raise ValueError(f"Team prior history missing: {sorted(missing)}")
        target = int(target_season or self.config.target_season)
        weights = self.config.normalized_weights()
        max_lag = len(weights)

        work = history.copy()
        work["season"] = pd.to_numeric(work["season"], errors="coerce")
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        work["sample_size"] = pd.to_numeric(
            work["sample_size"], errors="coerce"
        ).clip(lower=0)
        work["season_lag"] = target - work["season"]
        work = work[work["season_lag"].between(1, max_lag)].copy()
        if work.empty:
            raise ValueError(
                f"No history in the {max_lag} seasons preceding {target}."
            )
        work["season_weight"] = work["season_lag"].map(
            {lag: weights[lag - 1] for lag in range(1, max_lag + 1)}
        )
        work["weighted_sample"] = work["season_weight"] * work["sample_size"]
        work["weighted_value"] = work["weighted_sample"] * work["value"]

        entity = (
            work.groupby(["metric", "entity_id"], as_index=False)
            .agg(
                weighted_value_sum=("weighted_value", "sum"),
                effective_sample_size=("weighted_sample", "sum"),
                seasons_used=("season", "nunique"),
            )
        )
        entity["raw_mean"] = (
            entity["weighted_value_sum"]
            / entity["effective_sample_size"].replace(0, np.nan)
        )

        league = (
            work.groupby("metric", as_index=False)
            .agg(
                weighted_value_sum=("weighted_value", "sum"),
                effective_sample_size=("weighted_sample", "sum"),
            )
        )
        league["league_mean"] = (
            league["weighted_value_sum"]
            / league["effective_sample_size"].replace(0, np.nan)
        )
        entity = entity.merge(
            league[["metric", "league_mean"]],
            on="metric",
            how="left",
        )

        between = (
            entity.groupby("metric")["raw_mean"]
            .var(ddof=1)
            .fillna(self.config.minimum_between_entity_sd**2)
            .clip(lower=self.config.minimum_between_entity_sd**2)
            .rename("between_entity_variance")
            .reset_index()
        )
        entity = entity.merge(between, on="metric", how="left")
        k = float(self.config.prior_strength)
        entity["regression_weight"] = (
            entity["effective_sample_size"]
            / (entity["effective_sample_size"] + k)
        ).clip(0.0, 1.0)
        entity["prior_mean"] = (
            entity["league_mean"]
            + entity["regression_weight"]
            * (entity["raw_mean"] - entity["league_mean"])
        )

        # If K = within-variance / between-variance, conjugate posterior
        # variance equals between_variance * (1 - reliability).
        entity["prior_standard_deviation"] = np.sqrt(
            entity["between_entity_variance"]
            * (1.0 - entity["regression_weight"])
        )
        entity["prior_season"] = target
        entity["as_of_utc"] = pd.to_datetime(as_of_utc, utc=True) if as_of_utc else pd.NaT
        entity["entity_type"] = "team"
        entity["prior_strength"] = k
        entity["season_weights"] = ",".join(f"{w:.8f}" for w in weights)
        return entity[
            [
                "prior_season",
                "as_of_utc",
                "entity_type",
                "entity_id",
                "metric",
                "prior_mean",
                "prior_standard_deviation",
                "raw_mean",
                "league_mean",
                "effective_sample_size",
                "regression_weight",
                "seasons_used",
                "prior_strength",
                "season_weights",
            ]
        ].rename(columns={"metric": "component"})


def _evaluate_config(
    history: pd.DataFrame,
    target_seasons: Iterable[int],
    config: TeamPriorConfig,
) -> float:
    errors: list[float] = []
    builder = EmpiricalBayesTeamPrior(config)
    for season in target_seasons:
        prior = builder.build(history, target_season=int(season))
        actual = history[history["season"] == int(season)][
            ["entity_id", "metric", "value"]
        ].rename(columns={"metric": "component", "value": "actual"})
        merged = prior.merge(actual, on=["entity_id", "component"], how="inner")
        if len(merged):
            errors.extend(
                (merged["actual"].to_numpy(float) - merged["prior_mean"].to_numpy(float)) ** 2
            )
    return float(np.sqrt(np.mean(errors))) if errors else np.inf


def tune_team_prior_hyperparameters(
    history: pd.DataFrame,
    *,
    target_seasons: Iterable[int],
    season_weight_candidates: Iterable[tuple[float, ...]] = (
        (0.70, 0.30, 0.00),
        (0.60, 0.28, 0.12),
        (0.50, 0.30, 0.20),
        (0.45, 0.35, 0.20),
    ),
    prior_strength_candidates: Iterable[float] = (150, 250, 400, 600, 900, 1200),
) -> tuple[TeamPriorConfig, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for weights, strength in product(
        season_weight_candidates,
        prior_strength_candidates,
    ):
        config = TeamPriorConfig(
            season_weights=tuple(weights),
            prior_strength=float(strength),
        )
        rmse = _evaluate_config(history, target_seasons, config)
        records.append(
            {
                "season_weights": tuple(weights),
                "prior_strength": float(strength),
                "walkforward_rmse": rmse,
            }
        )
    results = pd.DataFrame(records).sort_values("walkforward_rmse").reset_index(drop=True)
    best = results.iloc[0]
    config = TeamPriorConfig(
        season_weights=tuple(best["season_weights"]),
        prior_strength=float(best["prior_strength"]),
    )
    return config, results
