from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuarterbackPriorConfig:
    prior_dropbacks: float = 250.0
    half_life_dropbacks: float = 400.0
    max_dropbacks: float = 1000.0
    minimum_sd: float = 0.03


def experience_bucket(years_experience: float | int | None) -> str:
    if years_experience is None or pd.isna(years_experience):
        return "unknown"
    years = float(years_experience)
    if years <= 0:
        return "rookie"
    if years <= 2:
        return "years_1_2"
    if years <= 5:
        return "years_3_5"
    if years <= 9:
        return "years_6_9"
    return "years_10_plus"


class QuarterbackPriorBuilder:
    """Dropback-weighted quarterback priors with experience-group shrinkage."""

    REQUIRED = {
        "player_id",
        "value",
        "dropbacks",
        "recency_dropbacks",
        "years_experience",
    }

    def __init__(self, config: QuarterbackPriorConfig | None = None) -> None:
        self.config = config or QuarterbackPriorConfig()

    def build(
        self,
        history: pd.DataFrame,
        *,
        as_of_utc: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        missing = self.REQUIRED - set(history.columns)
        if missing:
            raise ValueError(f"QB history missing: {sorted(missing)}")
        work = history.copy()
        for column in ("value", "dropbacks", "recency_dropbacks"):
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work["experience_bucket"] = work["years_experience"].map(experience_bucket)
        work["recency_weight"] = np.power(
            0.5,
            work["recency_dropbacks"].clip(lower=0)
            / float(self.config.half_life_dropbacks),
        )
        work["capped_dropbacks"] = work["dropbacks"].clip(
            lower=0,
            upper=float(self.config.max_dropbacks),
        )
        work["weighted_dropbacks"] = (
            work["recency_weight"] * work["capped_dropbacks"]
        )
        work["weighted_value"] = work["weighted_dropbacks"] * work["value"]

        group_baseline = (
            work.groupby("experience_bucket", as_index=False)
            .apply(
                lambda frame: pd.Series(
                    {
                        "group_mean": np.average(
                            frame["value"],
                            weights=np.maximum(frame["weighted_dropbacks"], 1e-9),
                        ),
                        "group_variance": max(
                            float(
                                np.average(
                                    (
                                        frame["value"]
                                        - np.average(
                                            frame["value"],
                                            weights=np.maximum(
                                                frame["weighted_dropbacks"], 1e-9
                                            ),
                                        )
                                    )
                                    ** 2,
                                    weights=np.maximum(
                                        frame["weighted_dropbacks"], 1e-9
                                    ),
                                )
                            ),
                            self.config.minimum_sd**2,
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

        player = (
            work.groupby(["player_id", "experience_bucket"], as_index=False)
            .agg(
                weighted_value_sum=("weighted_value", "sum"),
                effective_dropbacks=("weighted_dropbacks", "sum"),
                latest_years_experience=("years_experience", "max"),
            )
        )
        player["raw_mean"] = (
            player["weighted_value_sum"]
            / player["effective_dropbacks"].replace(0, np.nan)
        )
        player = player.merge(group_baseline, on="experience_bucket", how="left")
        k = float(self.config.prior_dropbacks)
        player["regression_weight"] = (
            player["effective_dropbacks"]
            / (player["effective_dropbacks"] + k)
        ).clip(0, 1)
        player["prior_mean"] = (
            player["group_mean"]
            + player["regression_weight"]
            * (player["raw_mean"] - player["group_mean"])
        )
        player["prior_standard_deviation"] = np.sqrt(
            player["group_variance"] * (1.0 - player["regression_weight"])
        ).clip(lower=self.config.minimum_sd)
        player["as_of_utc"] = pd.to_datetime(as_of_utc, utc=True) if as_of_utc else pd.NaT
        return player[
            [
                "as_of_utc",
                "player_id",
                "experience_bucket",
                "prior_mean",
                "prior_standard_deviation",
                "raw_mean",
                "group_mean",
                "effective_dropbacks",
                "regression_weight",
                "latest_years_experience",
            ]
        ]


def starter_mixture(
    quarterback_priors: pd.DataFrame,
    starter_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Combine starter uncertainty with each quarterback's prior uncertainty."""
    required_priors = {"player_id", "prior_mean", "prior_standard_deviation"}
    required_prob = {"team_id", "player_id", "starter_probability"}
    if required_priors - set(quarterback_priors.columns):
        raise ValueError("quarterback_priors missing required columns.")
    if required_prob - set(starter_probabilities.columns):
        raise ValueError("starter_probabilities missing required columns.")

    merged = starter_probabilities.merge(
        quarterback_priors,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    if merged[["prior_mean", "prior_standard_deviation"]].isna().any().any():
        raise ValueError("A starter candidate is missing a quarterback prior.")

    rows: list[dict[str, object]] = []
    for team_id, group in merged.groupby("team_id"):
        probabilities = group["starter_probability"].to_numpy(float)
        total = probabilities.sum()
        if total <= 0:
            raise ValueError(f"Starter probabilities for {team_id} sum to zero.")
        probabilities = probabilities / total
        means = group["prior_mean"].to_numpy(float)
        variances = group["prior_standard_deviation"].to_numpy(float) ** 2
        mixture_mean = float(np.sum(probabilities * means))
        second_moment = float(
            np.sum(probabilities * (variances + means**2))
        )
        mixture_variance = max(second_moment - mixture_mean**2, 0.0)
        rows.append(
            {
                "team_id": team_id,
                "qb_prior_mean": mixture_mean,
                "qb_prior_standard_deviation": float(np.sqrt(mixture_variance)),
                "starter_entropy": float(
                    -np.sum(
                        probabilities
                        * np.log(np.clip(probabilities, 1e-12, 1.0))
                    )
                ),
                "starter_candidates": len(group),
            }
        )
    return pd.DataFrame(rows)
