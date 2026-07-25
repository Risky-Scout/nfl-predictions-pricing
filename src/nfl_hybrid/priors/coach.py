from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoachPriorConfig:
    prior_games: float = 32.0
    minimum_sd: float = 0.03


class HierarchicalCoachPrior:
    """Shrink historical coach residuals toward unit/role league means."""

    REQUIRED = {
        "coach_id",
        "unit",
        "residual_value",
        "games",
    }

    def __init__(self, config: CoachPriorConfig | None = None) -> None:
        self.config = config or CoachPriorConfig()

    def build(
        self,
        history: pd.DataFrame,
        *,
        target_coaches: pd.DataFrame | None = None,
        as_of_utc: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        missing = self.REQUIRED - set(history.columns)
        if missing:
            raise ValueError(f"Coach history missing: {sorted(missing)}")

        work = history.copy()
        work["residual_value"] = pd.to_numeric(
            work["residual_value"], errors="coerce"
        )
        work["games"] = pd.to_numeric(work["games"], errors="coerce").clip(lower=0)
        work["weighted_value"] = work["residual_value"] * work["games"]

        unit = (
            work.groupby("unit", as_index=False)
            .agg(
                weighted_value=("weighted_value", "sum"),
                games=("games", "sum"),
            )
        )
        unit["league_mean"] = (
            unit["weighted_value"] / unit["games"].replace(0, np.nan)
        )

        coach = (
            work.groupby(["coach_id", "unit"], as_index=False)
            .agg(
                weighted_value=("weighted_value", "sum"),
                effective_games=("games", "sum"),
            )
        )
        coach["raw_mean"] = (
            coach["weighted_value"]
            / coach["effective_games"].replace(0, np.nan)
        )
        coach = coach.merge(
            unit[["unit", "league_mean"]],
            on="unit",
            how="left",
        )

        between = (
            coach.groupby("unit")["raw_mean"]
            .var(ddof=1)
            .fillna(self.config.minimum_sd**2)
            .clip(lower=self.config.minimum_sd**2)
            .rename("between_variance")
            .reset_index()
        )
        coach = coach.merge(between, on="unit", how="left")

        k = float(self.config.prior_games)
        coach["regression_weight"] = (
            coach["effective_games"]
            / (coach["effective_games"] + k)
        ).clip(0.0, 1.0)
        coach["prior_mean"] = (
            coach["league_mean"]
            + coach["regression_weight"]
            * (coach["raw_mean"] - coach["league_mean"])
        )
        coach["prior_standard_deviation"] = np.sqrt(
            coach["between_variance"]
            * (1.0 - coach["regression_weight"])
        ).clip(lower=self.config.minimum_sd)

        if target_coaches is not None:
            required_targets = {"coach_id", "unit"}
            if required_targets - set(target_coaches.columns):
                raise ValueError("target_coaches must contain coach_id and unit.")
            targets = target_coaches.copy()
            coach = targets.merge(
                coach,
                on=["coach_id", "unit"],
                how="left",
            )
            unit_fallback = unit[["unit", "league_mean"]].merge(
                between,
                on="unit",
                how="left",
            ).rename(
                columns={
                    "league_mean": "unit_fallback_mean",
                    "between_variance": "unit_fallback_variance",
                }
            )
            coach = coach.merge(unit_fallback, on="unit", how="left")
            coach["prior_mean"] = coach["prior_mean"].fillna(
                coach["unit_fallback_mean"]
            )
            coach["raw_mean"] = coach["raw_mean"].fillna(
                coach["unit_fallback_mean"]
            )
            coach["effective_games"] = coach["effective_games"].fillna(0.0)
            coach["regression_weight"] = coach["regression_weight"].fillna(0.0)
            coach["between_variance"] = coach["between_variance"].fillna(
                coach["unit_fallback_variance"]
            ).fillna(self.config.minimum_sd**2)
            coach["prior_standard_deviation"] = coach[
                "prior_standard_deviation"
            ].fillna(
                np.sqrt(coach["between_variance"]).clip(
                    lower=self.config.minimum_sd
                )
            )
            coach["league_mean"] = coach["league_mean"].fillna(
                coach["unit_fallback_mean"]
            )
            coach = coach.drop(
                columns=["unit_fallback_mean", "unit_fallback_variance"]
            )

        coach["as_of_utc"] = (
            pd.to_datetime(as_of_utc, utc=True) if as_of_utc else pd.NaT
        )
        coach["entity_type"] = "coach"
        coach["component"] = coach["unit"].astype(str) + "_coach_effect"
        return coach[
            [
                "as_of_utc",
                "entity_type",
                "coach_id",
                "component",
                "prior_mean",
                "prior_standard_deviation",
                "raw_mean",
                "league_mean",
                "effective_games",
                "regression_weight",
            ]
        ]
