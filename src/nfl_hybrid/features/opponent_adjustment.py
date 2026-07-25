from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class OpponentAdjustmentConfig:
    alpha: float = 20.0
    minimum_standard_error: float = 0.01


def fit_opponent_adjusted_ratings(
    team_games: pd.DataFrame,
    *,
    metric_columns: Iterable[str],
    season_column: str = "season",
    team_column: str = "team_id",
    opponent_column: str = "opponent_id",
    config: OpponentAdjustmentConfig | None = None,
) -> pd.DataFrame:
    """Fit season-specific team and opponent effects with ridge shrinkage.

    For each metric:
      value = intercept + team effect + opponent-allowed effect + error

    The returned team effect is an opponent-adjusted rating. Positive
    opponent effects mean that opponent allowed more of the metric.
    """
    cfg = config or OpponentAdjustmentConfig()
    metrics = list(metric_columns)
    required = {season_column, team_column, opponent_column, *metrics}
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"Opponent-adjustment input missing: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for season, season_frame in team_games.groupby(season_column, sort=True):
        teams = sorted(
            set(season_frame[team_column].dropna().astype(str))
            | set(season_frame[opponent_column].dropna().astype(str))
        )
        team_index = {team: i for i, team in enumerate(teams)}
        n = len(season_frame)
        p = len(teams)

        x = np.zeros((n, 2 * p), dtype=float)
        for row_number, (team, opponent) in enumerate(
            zip(
                season_frame[team_column].astype(str),
                season_frame[opponent_column].astype(str),
            )
        ):
            x[row_number, team_index[team]] = 1.0
            x[row_number, p + team_index[opponent]] = 1.0

        for metric in metrics:
            y = pd.to_numeric(season_frame[metric], errors="coerce").to_numpy()
            valid = np.isfinite(y)
            if valid.sum() < max(10, p):
                continue

            model = Ridge(alpha=cfg.alpha, fit_intercept=True)
            model.fit(x[valid], y[valid])
            predicted = model.predict(x[valid])
            residual_sd = max(
                float(np.std(y[valid] - predicted, ddof=1)),
                cfg.minimum_standard_error,
            )
            counts_team = season_frame.loc[valid, team_column].astype(str).value_counts()
            counts_opp = season_frame.loc[valid, opponent_column].astype(str).value_counts()

            team_effects = model.coef_[:p]
            opponent_effects = model.coef_[p:]
            team_effects = team_effects - np.mean(team_effects)
            opponent_effects = opponent_effects - np.mean(opponent_effects)

            for team in teams:
                i = team_index[team]
                team_n = int(counts_team.get(team, 0))
                opp_n = int(counts_opp.get(team, 0))
                records.append(
                    {
                        "season": int(season),
                        "metric": metric,
                        "team_id": team,
                        "league_intercept": float(model.intercept_),
                        "team_effect": float(team_effects[i]),
                        "opponent_allowed_effect": float(opponent_effects[i]),
                        "opponent_adjusted_mean": float(
                            model.intercept_ + team_effects[i]
                        ),
                        "team_games": team_n,
                        "opponent_games": opp_n,
                        "team_standard_error": float(
                            residual_sd / np.sqrt(max(team_n, 1))
                        ),
                        "opponent_standard_error": float(
                            residual_sd / np.sqrt(max(opp_n, 1))
                        ),
                        "ridge_alpha": cfg.alpha,
                    }
                )
    return pd.DataFrame(records)
