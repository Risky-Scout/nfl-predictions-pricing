from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.features.pregame_rolling import (
    build_game_pregame_matrix,
)


@dataclass(frozen=True)
class StrengthMetricSpec:
    metric: str
    exposure: str
    prior_exposure: float
    prior_mean: float
    prior_sd: float
    lower_bound: float | None = None
    upper_bound: float | None = None


DEFAULT_STRENGTH_METRICS: tuple[StrengthMetricSpec, ...] = (
    StrengthMetricSpec(
        "offense_epa_per_play",
        "offense_scrimmage_plays",
        500.0,
        0.0,
        0.18,
    ),
    StrengthMetricSpec(
        "offense_success_rate",
        "offense_scrimmage_plays",
        500.0,
        0.45,
        0.08,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_dropback_epa",
        "offense_dropbacks",
        350.0,
        0.04,
        0.24,
    ),
    StrengthMetricSpec(
        "offense_dropback_success_rate",
        "offense_dropbacks",
        350.0,
        0.46,
        0.09,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_designed_rush_epa",
        "offense_designed_rushes",
        300.0,
        -0.04,
        0.16,
    ),
    StrengthMetricSpec(
        "offense_designed_rush_success_rate",
        "offense_designed_rushes",
        300.0,
        0.40,
        0.09,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_explosive_pass_rate",
        "offense_pass_attempts",
        400.0,
        0.085,
        0.045,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_explosive_rush_rate",
        "offense_designed_rushes",
        350.0,
        0.11,
        0.055,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_sack_rate",
        "offense_dropbacks",
        450.0,
        0.07,
        0.035,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_turnover_rate",
        "offense_scrimmage_plays",
        600.0,
        0.025,
        0.020,
        0.0,
        1.0,
    ),
    StrengthMetricSpec(
        "offense_points_per_drive",
        "offense_drives",
        100.0,
        2.10,
        0.70,
        0.0,
        None,
    ),
    StrengthMetricSpec(
        "special_teams_epa_per_play",
        "special_teams_plays",
        250.0,
        0.0,
        0.25,
    ),
)


@dataclass(frozen=True)
class OpponentAdjustedConfig:
    ridge_alpha: float = 30.0
    season_decay: float = 0.55
    max_history_seasons: int = 3
    minimum_history_rows: int = 8
    league_prior_multiplier: float = 8.0
    minimum_sd: float = 0.005
    snapshot_floor: str = "D"


def _event_schedule(games: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        "kickoff_utc",
        "scheduled_kickoff_utc",
        "gameday",
        "schedule_date",
    )

    time_column = next(
        (
            column
            for column in candidates
            if column in games.columns
        ),
        None,
    )

    if time_column is None:
        raise ValueError(
            "Games table requires kickoff_utc, scheduled_kickoff_utc, "
            "gameday, or schedule_date."
        )

    schedule = games[["game_id", time_column]].copy()

    schedule["event_time"] = pd.to_datetime(
        schedule[time_column],
        errors="coerce",
        utc=True,
    )

    if schedule["event_time"].isna().any():
        bad = schedule.loc[
            schedule["event_time"].isna(),
            "game_id",
        ].head(20)

        raise ValueError(
            "Could not parse event time for games: "
            f"{bad.tolist()}"
        )

    return schedule[["game_id", "event_time"]]


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = float(weights.sum())

    if total <= 0:
        return np.nan

    return float(
        np.sum(values * weights) / total
    )


def _weighted_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    mean = _weighted_mean(values, weights)

    if not np.isfinite(mean):
        return np.nan

    total = float(weights.sum())

    if total <= 0:
        return np.nan

    return float(
        np.sum(
            weights * np.square(values - mean)
        ) / total
    )


def _bound(
    value: float,
    spec: StrengthMetricSpec,
) -> float:
    result = float(value)

    if spec.lower_bound is not None:
        result = max(result, spec.lower_bound)

    if spec.upper_bound is not None:
        result = min(result, spec.upper_bound)

    return result


def _metric_base(metric: str) -> str:
    if metric.startswith("offense_"):
        return metric.removeprefix("offense_")

    return metric


def _fit_metric_snapshot(
    history: pd.DataFrame,
    *,
    current_season: int,
    all_teams: list[str],
    spec: StrengthMetricSpec,
    config: OpponentAdjustedConfig,
) -> dict[str, object]:
    team_effects = {
        team: 0.0
        for team in all_teams
    }

    allowed_effects = {
        team: 0.0
        for team in all_teams
    }

    offense_exposure = {
        team: 0.0
        for team in all_teams
    }

    defense_exposure = {
        team: 0.0
        for team in all_teams
    }

    offense_games = {
        team: 0
        for team in all_teams
    }

    defense_games = {
        team: 0
        for team in all_teams
    }

    if history.empty:
        return {
            "league_mean": spec.prior_mean,
            "residual_sd": spec.prior_sd,
            "team_effects": team_effects,
            "allowed_effects": allowed_effects,
            "offense_exposure": offense_exposure,
            "defense_exposure": defense_exposure,
            "offense_games": offense_games,
            "defense_games": defense_games,
        }

    age = (
        current_season
        - pd.to_numeric(
            history["season"],
            errors="coerce",
        )
    )

    eligible = history.loc[
        age.ge(0)
        & age.lt(config.max_history_seasons)
    ].copy()

    if eligible.empty:
        return {
            "league_mean": spec.prior_mean,
            "residual_sd": spec.prior_sd,
            "team_effects": team_effects,
            "allowed_effects": allowed_effects,
            "offense_exposure": offense_exposure,
            "defense_exposure": defense_exposure,
            "offense_games": offense_games,
            "defense_games": defense_games,
        }

    eligible["_metric"] = pd.to_numeric(
        eligible[spec.metric],
        errors="coerce",
    )

    eligible["_exposure"] = pd.to_numeric(
        eligible[spec.exposure],
        errors="coerce",
    )

    age = (
        current_season
        - pd.to_numeric(
            eligible["season"],
            errors="coerce",
        )
    )

    eligible["_season_weight"] = np.power(
        config.season_decay,
        age,
    )

    eligible["_weight"] = (
        eligible["_exposure"]
        * eligible["_season_weight"]
    )

    valid = eligible.loc[
        eligible["_metric"].notna()
        & eligible["_exposure"].gt(0)
        & eligible["_weight"].gt(0)
    ].copy()

    if valid.empty:
        return {
            "league_mean": spec.prior_mean,
            "residual_sd": spec.prior_sd,
            "team_effects": team_effects,
            "allowed_effects": allowed_effects,
            "offense_exposure": offense_exposure,
            "defense_exposure": defense_exposure,
            "offense_games": offense_games,
            "defense_games": defense_games,
        }

    offense_exposure.update(
        valid.groupby("team_id")["_weight"]
        .sum()
        .to_dict()
    )

    defense_exposure.update(
        valid.groupby("opponent_id")["_weight"]
        .sum()
        .to_dict()
    )

    offense_games.update(
        valid.groupby("team_id")
        .size()
        .astype(int)
        .to_dict()
    )

    defense_games.update(
        valid.groupby("opponent_id")
        .size()
        .astype(int)
        .to_dict()
    )

    y = valid["_metric"].to_numpy(float)
    weights = valid["_weight"].to_numpy(float)

    observed_league_mean = _weighted_mean(
        y,
        weights,
    )

    total_exposure = float(weights.sum())

    league_prior_exposure = (
        spec.prior_exposure
        * config.league_prior_multiplier
    )

    global_reliability = (
        total_exposure
        / (
            total_exposure
            + league_prior_exposure
        )
    )

    league_mean = (
        spec.prior_mean
        + global_reliability
        * (
            observed_league_mean
            - spec.prior_mean
        )
    )

    if len(valid) >= config.minimum_history_rows:
        team_index = {
            team: index
            for index, team in enumerate(all_teams)
        }

        team_count = len(all_teams)

        design = np.zeros(
            (len(valid), 2 * team_count),
            dtype=float,
        )

        for row_number, row in enumerate(
            valid.itertuples(index=False)
        ):
            design[
                row_number,
                team_index[row.team_id],
            ] = 1.0

            design[
                row_number,
                team_count
                + team_index[row.opponent_id],
            ] = 1.0

        model = Ridge(
            alpha=config.ridge_alpha,
            fit_intercept=False,
        )

        model.fit(
            design,
            y - league_mean,
            sample_weight=weights,
        )

        offense_coefficients = (
            model.coef_[:team_count].copy()
        )

        allowed_coefficients = (
            model.coef_[team_count:].copy()
        )

        offense_center = float(
            offense_coefficients.mean()
        )

        allowed_center = float(
            allowed_coefficients.mean()
        )

        league_mean += (
            offense_center
            + allowed_center
        )

        offense_coefficients -= offense_center
        allowed_coefficients -= allowed_center

        team_effects.update(
            {
                team: float(
                    offense_coefficients[
                        team_index[team]
                    ]
                )
                for team in all_teams
            }
        )

        allowed_effects.update(
            {
                team: float(
                    allowed_coefficients[
                        team_index[team]
                    ]
                )
                for team in all_teams
            }
        )

        fitted = (
            league_mean
            + design[:, :team_count]
            @ offense_coefficients
            + design[:, team_count:]
            @ allowed_coefficients
        )

        residuals = y - fitted

        observed_variance = _weighted_variance(
            residuals,
            weights,
        )

        if np.isfinite(observed_variance):
            residual_sd = float(
                np.sqrt(
                    global_reliability
                    * observed_variance
                    + (
                        1.0
                        - global_reliability
                    )
                    * spec.prior_sd**2
                )
            )
        else:
            residual_sd = spec.prior_sd
    else:
        residual_sd = spec.prior_sd

    residual_sd = max(
        residual_sd,
        config.minimum_sd,
    )

    return {
        "league_mean": float(league_mean),
        "residual_sd": float(residual_sd),
        "team_effects": team_effects,
        "allowed_effects": allowed_effects,
        "offense_exposure": offense_exposure,
        "defense_exposure": defense_exposure,
        "offense_games": offense_games,
        "defense_games": defense_games,
    }


def build_opponent_adjusted_team_features(
    team_game: pd.DataFrame,
    games: pd.DataFrame,
    *,
    metric_specs: Sequence[
        StrengthMetricSpec
    ] = DEFAULT_STRENGTH_METRICS,
    config: OpponentAdjustedConfig | None = None,
) -> pd.DataFrame:
    """Build leakage-safe opponent-adjusted pregame team ratings."""

    cfg = config or OpponentAdjustedConfig()

    required = {
        "game_id",
        "team_id",
        "opponent_id",
        "season",
    }

    missing = required - set(team_game.columns)

    if missing:
        raise ValueError(
            f"Team-game table missing: {sorted(missing)}"
        )

    requested_columns = {
        spec.metric
        for spec in metric_specs
    } | {
        spec.exposure
        for spec in metric_specs
    }

    missing_metrics = sorted(
        requested_columns
        - set(team_game.columns)
    )

    if missing_metrics:
        raise ValueError(
            "Missing requested metric or exposure columns: "
            f"{missing_metrics}"
        )

    work = team_game.merge(
        _event_schedule(games),
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    for column in (
        "team_id",
        "opponent_id",
        "home_team",
        "away_team",
    ):
        if column in work.columns:
            work[column] = work[column].map(
                canonical_team_id
            )

    work["season"] = pd.to_numeric(
        work["season"],
        errors="raise",
    ).astype(int)

    work["snapshot_date"] = (
        work["event_time"]
        .dt.floor(cfg.snapshot_floor)
    )

    work = work.sort_values(
        [
            "snapshot_date",
            "game_id",
            "team_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    all_teams = sorted(
        set(work["team_id"])
        | set(work["opponent_id"])
    )

    rows: list[dict[str, object]] = []

    for snapshot_date, current in work.groupby(
        "snapshot_date",
        sort=True,
    ):
        history = work.loc[
            work["snapshot_date"] < snapshot_date
        ]

        current_season = int(
            current["season"].max()
        )

        snapshot_models = {
            spec.metric: _fit_metric_snapshot(
                history,
                current_season=current_season,
                all_teams=all_teams,
                spec=spec,
                config=cfg,
            )
            for spec in metric_specs
        }

        prior_offense_games = (
            history.groupby("team_id")
            .size()
            .astype(int)
            .to_dict()
        )

        prior_defense_games = (
            history.groupby("opponent_id")
            .size()
            .astype(int)
            .to_dict()
        )

        for game_row in current.itertuples(
            index=False
        ):
            team = game_row.team_id

            row: dict[str, object] = {
                "game_id": game_row.game_id,
                "team_id": team,
                "opponent_id": game_row.opponent_id,
                "season": int(game_row.season),
                "event_time": game_row.event_time,
                "oa_prior_offense_games": int(
                    prior_offense_games.get(team, 0)
                ),
                "oa_prior_defense_games": int(
                    prior_defense_games.get(team, 0)
                ),
            }

            if hasattr(game_row, "week"):
                row["week"] = game_row.week

            for spec in metric_specs:
                model = snapshot_models[
                    spec.metric
                ]

                base = _metric_base(spec.metric)

                offense_exposure = float(
                    model["offense_exposure"].get(
                        team,
                        0.0,
                    )
                )

                defense_exposure = float(
                    model["defense_exposure"].get(
                        team,
                        0.0,
                    )
                )

                offense_reliability = (
                    offense_exposure
                    / (
                        offense_exposure
                        + spec.prior_exposure
                    )
                )

                defense_reliability = (
                    defense_exposure
                    / (
                        defense_exposure
                        + spec.prior_exposure
                    )
                )

                raw_offense_effect = float(
                    model["team_effects"].get(
                        team,
                        0.0,
                    )
                )

                raw_allowed_effect = float(
                    model["allowed_effects"].get(
                        team,
                        0.0,
                    )
                )

                offense_effect = (
                    offense_reliability
                    * raw_offense_effect
                )

                allowed_effect = (
                    defense_reliability
                    * raw_allowed_effect
                )

                league_mean = float(
                    model["league_mean"]
                )

                residual_sd = float(
                    model["residual_sd"]
                )

                offense_mean = _bound(
                    league_mean
                    + offense_effect,
                    spec,
                )

                defense_allowed_mean = _bound(
                    league_mean
                    + allowed_effect,
                    spec,
                )

                offense_sd = max(
                    config.minimum_sd
                    if config
                    else cfg.minimum_sd,
                    residual_sd
                    / np.sqrt(
                        1.0
                        + offense_exposure
                        / spec.prior_exposure
                    ),
                )

                defense_sd = max(
                    config.minimum_sd
                    if config
                    else cfg.minimum_sd,
                    residual_sd
                    / np.sqrt(
                        1.0
                        + defense_exposure
                        / spec.prior_exposure
                    ),
                )

                prefix = f"oa_{base}"

                row[
                    f"{prefix}_league_mean"
                ] = league_mean

                row[
                    f"{prefix}_offense_mean"
                ] = offense_mean

                row[
                    f"{prefix}_offense_effect"
                ] = offense_effect

                row[
                    f"{prefix}_offense_reliability"
                ] = offense_reliability

                row[
                    f"{prefix}_offense_exposure"
                ] = offense_exposure

                row[
                    f"{prefix}_offense_sd"
                ] = offense_sd

                row[
                    f"{prefix}_defense_allowed_mean"
                ] = defense_allowed_mean

                # Positive means stronger defense.
                row[
                    f"{prefix}_defense_effect"
                ] = -allowed_effect

                row[
                    f"{prefix}_defense_reliability"
                ] = defense_reliability

                row[
                    f"{prefix}_defense_exposure"
                ] = defense_exposure

                row[
                    f"{prefix}_defense_sd"
                ] = defense_sd

            rows.append(row)

    output = pd.DataFrame(rows)

    duplicate_keys = int(
        output.duplicated(
            ["game_id", "team_id"],
            keep=False,
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(
            "Duplicate opponent-adjusted team-game keys: "
            f"{duplicate_keys}"
        )

    return output


def build_game_opponent_adjusted_matrix(
    games: pd.DataFrame,
    team_strength: pd.DataFrame,
) -> pd.DataFrame:
    return build_game_pregame_matrix(
        games,
        team_strength,
        carrier_columns=(),
    )
