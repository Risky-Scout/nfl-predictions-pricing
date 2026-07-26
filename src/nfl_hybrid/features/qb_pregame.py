
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.priors.quarterback import (
    QuarterbackPriorBuilder,
    QuarterbackPriorConfig,
)


@dataclass(frozen=True)
class QBMetricSpec:
    metric: str
    exposure: str
    prior_mean: float
    prior_sd: float
    prior_dropbacks: float = 250.0
    minimum_sd: float = 0.02
    lower_bound: float | None = None
    upper_bound: float | None = None


DEFAULT_QB_METRICS: tuple[QBMetricSpec, ...] = (
    QBMetricSpec("epa_per_dropback", "dropbacks", 0.04, 0.25),
    QBMetricSpec("success_rate", "dropbacks", 0.46, 0.09, lower_bound=0.0, upper_bound=1.0),
    QBMetricSpec("competitive_epa_per_dropback", "dropbacks", 0.04, 0.27),
    QBMetricSpec("neutral_epa_per_dropback", "dropbacks", 0.04, 0.27),
    QBMetricSpec("completion_rate", "attempts", 0.64, 0.08, lower_bound=0.0, upper_bound=1.0),
    QBMetricSpec("cpoe", "attempts", 0.0, 8.0, minimum_sd=0.50),
    QBMetricSpec("air_yards_per_attempt", "attempts", 8.0, 2.0, minimum_sd=0.10, lower_bound=0.0),
    QBMetricSpec("explosive_pass_rate", "attempts", 0.085, 0.05, lower_bound=0.0, upper_bound=1.0),
    QBMetricSpec("sack_rate", "dropbacks", 0.07, 0.04, lower_bound=0.0, upper_bound=1.0),
    QBMetricSpec("interception_rate", "attempts", 0.023, 0.025, lower_bound=0.0, upper_bound=1.0),
)


@dataclass(frozen=True)
class QBPregameConfig:
    half_life_dropbacks: float = 400.0
    max_dropbacks: float = 1000.0
    max_history_seasons: int = 3
    season_decay: float = 0.60
    snapshot_floor: str = "D"
    probability_tolerance: float = 1e-6


def _event_schedule(games: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        "scheduled_kickoff_utc",
        "kickoff_utc",
        "gameday",
        "schedule_date",
    )
    time_column = next((c for c in candidates if c in games.columns), None)
    if time_column is None:
        raise ValueError(
            "Games table must contain scheduled_kickoff_utc, kickoff_utc, "
            "gameday, or schedule_date."
        )

    schedule = games[["game_id", "season", time_column]].copy()
    schedule["event_time"] = pd.to_datetime(
        schedule[time_column], errors="coerce", utc=True
    )
    if schedule["event_time"].isna().any():
        bad = schedule.loc[schedule["event_time"].isna(), "game_id"].head(20).tolist()
        raise ValueError(f"Could not parse event time for games: {bad}")
    return schedule[["game_id", "season", "event_time"]]


def actual_starter_candidates(games: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "home_team_id",
        "away_team_id",
        "home_qb_id",
        "away_qb_id",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing actual-starter fields: {sorted(missing)}")

    home = games[
        ["game_id", "season", "home_team_id", "home_qb_id"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_qb_id": "player_id",
        }
    )
    away = games[
        ["game_id", "season", "away_team_id", "away_qb_id"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_qb_id": "player_id",
        }
    )
    output = pd.concat([home, away], ignore_index=True)
    output["team_id"] = output["team_id"].map(canonical_team_id)
    output["player_id"] = output["player_id"].astype("string")
    output["starter_probability"] = 1.0
    output["starter_source"] = "actual_at_kickoff"
    if output[["team_id", "player_id"]].isna().any().any():
        raise ValueError("Actual starter candidates contain missing team or player IDs.")
    if output.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError("Duplicate actual-starter candidate rows.")
    return output


def _bound(value: float, spec: QBMetricSpec) -> float:
    result = float(value)
    if spec.lower_bound is not None:
        result = max(result, spec.lower_bound)
    if spec.upper_bound is not None:
        result = min(result, spec.upper_bound)
    return result


def _weighted_league_prior(
    history: pd.DataFrame,
    spec: QBMetricSpec,
    current_season: int,
    config: QBPregameConfig,
) -> tuple[float, float]:
    work = history.copy()
    work["_value"] = pd.to_numeric(work[spec.metric], errors="coerce")
    work["_exposure"] = pd.to_numeric(work[spec.exposure], errors="coerce")
    age = current_season - pd.to_numeric(work["season"], errors="coerce")
    work = work.loc[
        age.ge(0)
        & age.lt(config.max_history_seasons)
        & work["_value"].notna()
        & work["_exposure"].gt(0)
    ].copy()
    if work.empty:
        return spec.prior_mean, spec.prior_sd

    age = current_season - pd.to_numeric(work["season"], errors="coerce")
    weights = (
        work["_exposure"].to_numpy(float)
        * np.power(config.season_decay, age.to_numpy(float))
    )
    values = work["_value"].to_numpy(float)
    total = float(weights.sum())
    if total <= 0:
        return spec.prior_mean, spec.prior_sd

    observed_mean = float(np.sum(weights * values) / total)
    prior_weight = spec.prior_dropbacks * 8.0
    reliability = total / (total + prior_weight)
    mean = spec.prior_mean + reliability * (observed_mean - spec.prior_mean)

    observed_var = float(np.sum(weights * (values - observed_mean) ** 2) / total)
    sd = float(
        np.sqrt(
            reliability * max(observed_var, spec.minimum_sd**2)
            + (1.0 - reliability) * spec.prior_sd**2
        )
    )
    return _bound(mean, spec), max(sd, spec.minimum_sd)


def _history_for_builder(
    history: pd.DataFrame,
    *,
    entity_column: str,
    spec: QBMetricSpec,
    current_season: int,
    config: QBPregameConfig,
) -> pd.DataFrame:
    work = history.copy()
    work["player_id"] = work[entity_column].astype("string")
    work["value"] = pd.to_numeric(work[spec.metric], errors="coerce")
    work["_exposure"] = pd.to_numeric(work[spec.exposure], errors="coerce")
    age = current_season - pd.to_numeric(work["season"], errors="coerce")

    work = work.loc[
        age.ge(0)
        & age.lt(config.max_history_seasons)
        & work["value"].notna()
        & work["_exposure"].gt(0)
        & work["player_id"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "value",
                "dropbacks",
                "recency_dropbacks",
                "years_experience",
            ]
        )

    work = work.sort_values(
        ["player_id", "event_time", "game_id"],
        ascending=[True, False, False],
        kind="stable",
    )
    work["recency_dropbacks"] = (
        work.groupby("player_id", sort=False)["_exposure"].cumsum()
        - work["_exposure"]
    )
    work["_remaining"] = (
        config.max_dropbacks - work["recency_dropbacks"]
    ).clip(lower=0.0)
    work["dropbacks"] = np.minimum(
        work["_exposure"].to_numpy(float),
        work["_remaining"].to_numpy(float),
    )
    age = current_season - pd.to_numeric(work["season"], errors="coerce")
    work["dropbacks"] *= np.power(
        config.season_decay, age.to_numpy(float)
    )
    work = work.loc[work["dropbacks"].gt(0)].copy()
    work["years_experience"] = np.nan

    return work[
        [
            "player_id",
            "value",
            "dropbacks",
            "recency_dropbacks",
            "years_experience",
        ]
    ]


def _build_entity_priors(
    history: pd.DataFrame,
    *,
    entity_column: str,
    spec: QBMetricSpec,
    current_season: int,
    config: QBPregameConfig,
    as_of_utc: pd.Timestamp,
) -> pd.DataFrame:
    prepared = _history_for_builder(
        history,
        entity_column=entity_column,
        spec=spec,
        current_season=current_season,
        config=config,
    )
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "prior_mean",
                "prior_standard_deviation",
                "effective_dropbacks",
                "regression_weight",
            ]
        )

    builder = QuarterbackPriorBuilder(
        QuarterbackPriorConfig(
            prior_dropbacks=spec.prior_dropbacks,
            half_life_dropbacks=config.half_life_dropbacks,
            max_dropbacks=config.max_dropbacks,
            minimum_sd=spec.minimum_sd,
        )
    )
    return builder.build(prepared, as_of_utc=as_of_utc)


def _candidate_metric_rows(
    current: pd.DataFrame,
    history: pd.DataFrame,
    *,
    spec: QBMetricSpec,
    current_season: int,
    config: QBPregameConfig,
    as_of_utc: pd.Timestamp,
) -> pd.DataFrame:
    league_mean, league_sd = _weighted_league_prior(
        history, spec, current_season, config
    )
    player_priors = _build_entity_priors(
        history,
        entity_column="player_id",
        spec=spec,
        current_season=current_season,
        config=config,
        as_of_utc=as_of_utc,
    )
    team_priors = _build_entity_priors(
        history,
        entity_column="team_id",
        spec=spec,
        current_season=current_season,
        config=config,
        as_of_utc=as_of_utc,
    ).rename(
        columns={
            "player_id": "team_id",
            "prior_mean": "team_prior_mean",
            "prior_standard_deviation": "team_prior_sd",
            "effective_dropbacks": "team_effective_dropbacks",
            "regression_weight": "team_regression_weight",
        }
    )

    candidate = current.copy()
    candidate = candidate.merge(
        player_priors[
            [
                "player_id",
                "prior_mean",
                "prior_standard_deviation",
                "effective_dropbacks",
                "regression_weight",
            ]
        ],
        on="player_id",
        how="left",
        validate="m:1",
    )
    candidate = candidate.merge(
        team_priors[
            [
                "team_id",
                "team_prior_mean",
                "team_prior_sd",
                "team_effective_dropbacks",
                "team_regression_weight",
            ]
        ],
        on="team_id",
        how="left",
        validate="m:1",
    )

    numeric_defaults = {
        "prior_mean": league_mean,
        "prior_standard_deviation": league_sd,
        "effective_dropbacks": 0.0,
        "regression_weight": 0.0,
        "team_prior_mean": league_mean,
        "team_prior_sd": league_sd,
        "team_effective_dropbacks": 0.0,
        "team_regression_weight": 0.0,
    }
    for column, default in numeric_defaults.items():
        candidate[column] = pd.to_numeric(
            candidate[column],
            errors="coerce",
        ).fillna(float(default))

    candidate["prior_mean"] = candidate["prior_mean"].map(lambda x: _bound(x, spec))
    candidate["team_prior_mean"] = candidate["team_prior_mean"].map(
        lambda x: _bound(x, spec)
    )
    candidate["league_mean"] = league_mean
    candidate["league_sd"] = league_sd
    return candidate


def _mixture_for_metric(
    candidate: pd.DataFrame,
    spec: QBMetricSpec,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (game_id, team_id), group in candidate.groupby(
        ["game_id", "team_id"], sort=False
    ):
        probabilities = pd.to_numeric(
            group["starter_probability"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError(f"Starter probabilities sum to zero for {game_id}/{team_id}.")
        probabilities = probabilities / total

        means = group["prior_mean"].to_numpy(float)
        sds = group["prior_standard_deviation"].to_numpy(float)
        mixture_mean = float(np.sum(probabilities * means))
        second_moment = float(np.sum(probabilities * (sds**2 + means**2)))
        mixture_sd = float(np.sqrt(max(second_moment - mixture_mean**2, 0.0)))

        team_mean = float(group["team_prior_mean"].iloc[0])
        team_sd = float(group["team_prior_sd"].iloc[0])
        prefix = f"qb_{spec.metric}"
        rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                f"{prefix}_mean": _bound(mixture_mean, spec),
                f"{prefix}_sd": max(mixture_sd, spec.minimum_sd),
                f"{prefix}_effective_dropbacks": float(
                    np.sum(probabilities * group["effective_dropbacks"].to_numpy(float))
                ),
                f"{prefix}_reliability": float(
                    np.sum(probabilities * group["regression_weight"].to_numpy(float))
                ),
                f"{prefix}_team_mean": _bound(team_mean, spec),
                f"{prefix}_team_sd": max(team_sd, spec.minimum_sd),
                f"{prefix}_delta_vs_team": float(mixture_mean - team_mean),
                f"{prefix}_league_mean": float(group["league_mean"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_qb_pregame_team_features(
    qb_game: pd.DataFrame,
    games: pd.DataFrame,
    *,
    starter_candidates: pd.DataFrame | None = None,
    metric_specs: Sequence[QBMetricSpec] = DEFAULT_QB_METRICS,
    config: QBPregameConfig | None = None,
) -> pd.DataFrame:
    """Build leakage-safe QB priors and starter-mixture uncertainty."""

    cfg = config or QBPregameConfig()
    required_qb = {
        "game_id",
        "team_id",
        "player_id",
        "season",
        "dropbacks",
        "attempts",
    } | {spec.metric for spec in metric_specs}
    missing = required_qb - set(qb_game.columns)
    if missing:
        raise ValueError(f"QB-game table missing fields: {sorted(missing)}")

    schedule = _event_schedule(games)
    history = qb_game.merge(
        schedule[["game_id", "event_time"]],
        on="game_id",
        how="left",
        validate="m:1",
    )
    history["team_id"] = history["team_id"].map(canonical_team_id)
    history["player_id"] = history["player_id"].astype("string")
    history["season"] = pd.to_numeric(history["season"], errors="raise").astype(int)
    history["snapshot_date"] = history["event_time"].dt.floor(cfg.snapshot_floor)

    candidates = (
        actual_starter_candidates(games)
        if starter_candidates is None
        else starter_candidates.copy()
    )
    required_candidates = {
        "game_id",
        "team_id",
        "player_id",
        "starter_probability",
    }
    missing_candidates = required_candidates - set(candidates.columns)
    if missing_candidates:
        raise ValueError(
            f"Starter candidates missing fields: {sorted(missing_candidates)}"
        )

    candidates["team_id"] = candidates["team_id"].map(canonical_team_id)
    candidates["player_id"] = candidates["player_id"].astype("string")
    candidates = candidates.merge(
        schedule,
        on=["game_id", "season"],
        how="left",
        validate="m:1",
    )
    candidates["snapshot_date"] = candidates["event_time"].dt.floor(cfg.snapshot_floor)

    probability_sums = candidates.groupby(
        ["game_id", "team_id"]
    )["starter_probability"].sum()
    if (probability_sums <= 0).any():
        raise ValueError("At least one game/team has zero starter probability.")

    output_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []

    for snapshot_date, current in candidates.groupby("snapshot_date", sort=True):
        prior_history = history.loc[history["snapshot_date"] < snapshot_date].copy()
        current_season = int(current["season"].max())

        metric_outputs: list[pd.DataFrame] = []
        for spec in metric_specs:
            candidate_metric = _candidate_metric_rows(
                current,
                prior_history,
                spec=spec,
                current_season=current_season,
                config=cfg,
                as_of_utc=snapshot_date,
            )
            metric_outputs.append(_mixture_for_metric(candidate_metric, spec))

        snapshot_output = metric_outputs[0]
        for metric_output in metric_outputs[1:]:
            snapshot_output = snapshot_output.merge(
                metric_output,
                on=["game_id", "team_id"],
                how="inner",
                validate="1:1",
            )
        output_parts.append(snapshot_output)

        meta_rows: list[dict[str, object]] = []
        for (game_id, team_id), group in current.groupby(
            ["game_id", "team_id"], sort=False
        ):
            probabilities = pd.to_numeric(
                group["starter_probability"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            probabilities = probabilities / probabilities.sum()
            max_index = int(np.argmax(probabilities))
            meta_rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "season": int(group["season"].iloc[0]),
                    "event_time": group["event_time"].iloc[0],
                    "qb_expected_starter_id": str(group.iloc[max_index]["player_id"]),
                    "qb_max_starter_probability": float(probabilities[max_index]),
                    "qb_starter_candidates": int(len(group)),
                    "qb_starter_entropy": float(
                        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
                    ),
                    "qb_starter_source": str(
                        group.get(
                            "starter_source",
                            pd.Series(["provided"] * len(group), index=group.index),
                        ).iloc[max_index]
                    ),
                }
            )
        metadata_parts.append(pd.DataFrame(meta_rows))

    output = pd.concat(output_parts, ignore_index=True)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    output = metadata.merge(
        output,
        on=["game_id", "team_id"],
        how="left",
        validate="1:1",
    )

    if output.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate QB pregame team rows.")
    return output


def build_game_qb_matrix(
    games: pd.DataFrame,
    qb_team_features: pd.DataFrame,
) -> pd.DataFrame:
    required = {"game_id", "home_team_id", "away_team_id"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing fields: {sorted(missing)}")

    game_work = games.copy()
    game_work["_home_team_join"] = game_work["home_team_id"].map(canonical_team_id)
    game_work["_away_team_join"] = game_work["away_team_id"].map(canonical_team_id)

    team_work = qb_team_features.copy()
    team_work["_team_join"] = team_work["team_id"].map(canonical_team_id)

    feature_columns = [
        c
        for c in team_work.columns
        if c
        not in {
            "game_id",
            "team_id",
            "season",
            "event_time",
            "_team_join",
        }
    ]

    home = team_work[["game_id", "_team_join"] + feature_columns].rename(
        columns={
            "_team_join": "_home_feature_team",
            **{c: f"home_{c}" for c in feature_columns},
        }
    )
    away = team_work[["game_id", "_team_join"] + feature_columns].rename(
        columns={
            "_team_join": "_away_feature_team",
            **{c: f"away_{c}" for c in feature_columns},
        }
    )

    matrix = game_work.merge(
        home,
        left_on=["game_id", "_home_team_join"],
        right_on=["game_id", "_home_feature_team"],
        how="left",
        validate="1:1",
    )
    matrix = matrix.merge(
        away,
        left_on=["game_id", "_away_team_join"],
        right_on=["game_id", "_away_feature_team"],
        how="left",
        validate="1:1",
    )

    missing_rows = (
        matrix["_home_feature_team"].isna()
        | matrix["_away_feature_team"].isna()
    )
    if missing_rows.any():
        bad = matrix.loc[
            missing_rows,
            ["game_id", "home_team_id", "away_team_id"],
        ].head(20)
        raise ValueError(
            "Could not attach both QB team rows:\n"
            f"{bad.to_string(index=False)}"
        )

    return matrix.drop(
        columns=[
            "_home_team_join",
            "_away_team_join",
            "_home_feature_team",
            "_away_feature_team",
        ]
    )
