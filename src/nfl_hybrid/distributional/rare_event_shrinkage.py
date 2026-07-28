from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RareEventConfig:
    warmup_seasons: tuple[int, ...] = (2020,)
    development_seasons: tuple[int, ...] = (2021, 2022, 2023)
    tie_prior_mean: float = 0.005
    tie_prior_strengths: tuple[float, ...] = (200.0, 400.0, 800.0)
    ats_prior_strengths: tuple[float, ...] = (32.0, 64.0, 128.0)
    ats_margin_prior_sd: float = 13.5
    ats_integer_push_rate_prior_mean: float = 0.04
    total_prior_strengths: tuple[float, ...] = (32.0, 64.0, 128.0)
    total_points_prior_mean: float = 45.0
    total_points_prior_sd: float = 13.0
    total_integer_push_rate_prior_mean: float = 0.012
    probability_clip: float = 1e-9

    @classmethod
    def from_json(cls, path: str | Path) -> "RareEventConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in (
            "warmup_seasons",
            "development_seasons",
            "tie_prior_strengths",
            "ats_prior_strengths",
            "total_prior_strengths",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"None of these columns exist: {list(candidates)}")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _discrete_normal_mass(value: int, mean: float, sd: float) -> float:
    if sd <= 0:
        raise ValueError("sd must be positive.")
    lower = (float(value) - 0.5 - mean) / sd
    upper = (float(value) + 0.5 - mean) / sd
    return max(0.0, _normal_cdf(upper) - _normal_cdf(lower))


def _beta_posterior_mean(
    successes: int,
    trials: int,
    prior_mean: float,
    prior_strength: float,
) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials.")
    if not 0.0 <= prior_mean <= 1.0:
        raise ValueError("prior_mean must be in [0, 1].")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive.")
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    return (successes + alpha) / (trials + alpha + beta)


def _is_integer_line(line: float) -> bool:
    return bool(np.isclose(abs(float(line)) % 1.0, 0.0, atol=1e-9, rtol=0.0))


def _three_way(
    conditional_first: float,
    rare_probability: float,
) -> tuple[float, float, float]:
    first = float(np.clip(conditional_first, 0.0, 1.0))
    rare = float(np.clip(rare_probability, 0.0, 1.0))
    action = 1.0 - rare
    return action * first, action * (1.0 - first), rare


def _normalized(
    probabilities: tuple[float, float, float],
    clip: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite probability detected.")
    if (values < -1e-12).any():
        raise ValueError("Negative probability detected.")
    values = np.maximum(values, 0.0)
    values /= values.sum()
    values = np.clip(values, clip, 1.0)
    values /= values.sum()
    return values


def _scores(
    probabilities: tuple[float, float, float],
    outcome_index: int,
    clip: float,
) -> tuple[float, float, np.ndarray]:
    values = _normalized(probabilities, clip)
    target = np.zeros(3, dtype=float)
    target[outcome_index] = 1.0
    return (
        -math.log(float(values[outcome_index])),
        float(np.sum((values - target) ** 2)),
        values,
    )


def _ats_exact_margin_probability(
    line: float,
    counts: Counter[int],
    games_seen: int,
    prior_strength: float,
    prior_sd: float,
) -> float:
    if not _is_integer_line(line):
        return 0.0
    target_margin = int(round(-float(line)))
    prior_mass = _discrete_normal_mass(target_margin, 0.0, prior_sd)
    return (
        counts[target_margin] + prior_strength * prior_mass
    ) / (games_seen + prior_strength)


def _ats_integer_rate_probability(
    line: float,
    pushes: int,
    games: int,
    prior_strength: float,
    prior_mean: float,
) -> float:
    if not _is_integer_line(line):
        return 0.0
    return _beta_posterior_mean(pushes, games, prior_mean, prior_strength)


def _total_exact_points_probability(
    line: float,
    counts: Counter[int],
    games_seen: int,
    prior_strength: float,
    prior_mean: float,
    prior_sd: float,
) -> float:
    if not _is_integer_line(line):
        return 0.0
    target_total = int(round(float(line)))
    prior_mass = _discrete_normal_mass(
        target_total,
        prior_mean,
        prior_sd,
    )
    return (
        counts[target_total] + prior_strength * prior_mass
    ) / (games_seen + prior_strength)


def _total_integer_rate_probability(
    line: float,
    pushes: int,
    games: int,
    prior_strength: float,
    prior_mean: float,
) -> float:
    if not _is_integer_line(line):
        return 0.0
    return _beta_posterior_mean(pushes, games, prior_mean, prior_strength)


def _outcome_index(market: str, row: pd.Series) -> int:
    if market == "pregame_moneyline":
        if int(row["target_tie"]) == 1:
            return 2
        return 0 if int(row["target_home_win"]) == 1 else 1
    if market == "pregame_ats":
        if int(row["target_t10_ats_push"]) == 1:
            return 2
        return 0 if int(row["target_t10_home_cover"]) == 1 else 1
    if market == "pregame_total":
        if int(row["target_t10_total_push"]) == 1:
            return 2
        return 0 if int(row["target_t10_over"]) == 1 else 1
    raise ValueError(f"Unsupported market: {market}")


def _load_games(
    canonical_root: Path,
    warehouse_path: Path,
) -> pd.DataFrame:
    ml = pd.read_parquet(
        canonical_root
        / "pregame_moneyline_market_augmented_canonical_t10.parquet"
    )
    ats = pd.read_parquet(
        canonical_root
        / "pregame_ats_market_augmented_canonical_t10.parquet"
    )
    total = pd.read_parquet(
        canonical_root
        / "pregame_total_market_augmented_canonical_t10.parquet"
    )
    warehouse = pd.read_parquet(warehouse_path)
    kickoff_column = _first_existing(
        warehouse.columns,
        (
            "scheduled_kickoff_utc",
            "kickoff_utc",
            "start_time_utc",
            "game_datetime_utc",
        ),
    )
    metadata_columns = ["game_id", kickoff_column]
    if "season_type" in warehouse.columns:
        metadata_columns.append("season_type")
    metadata = warehouse[metadata_columns].copy().rename(
        columns={kickoff_column: "kickoff_utc"}
    )
    metadata["kickoff_utc"] = pd.to_datetime(
        metadata["kickoff_utc"],
        utc=True,
        errors="raise",
    )
    if "season_type" not in metadata.columns:
        metadata["season_type"] = "REG"

    ml = ml[
        [
            "game_id",
            "season",
            "week",
            "target_home_win",
            "target_tie",
            "target_home_margin",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "market_t10_novig_probability":
                "moneyline_conditional_home_probability"
        }
    )
    ats = ats[
        [
            "game_id",
            "target_t10_home_cover",
            "target_t10_ats_push",
            "market_t10_consensus_line",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "market_t10_consensus_line": "ats_line",
            "market_t10_novig_probability":
                "ats_conditional_home_probability",
        }
    )
    total = total[
        [
            "game_id",
            "target_t10_over",
            "target_t10_total_push",
            "target_total_points",
            "market_t10_consensus_line",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "market_t10_consensus_line": "total_line",
            "market_t10_novig_probability":
                "total_conditional_over_probability",
        }
    )

    games = (
        ml.merge(ats, on="game_id", validate="one_to_one")
        .merge(total, on="game_id", validate="one_to_one")
        .merge(metadata, on="game_id", validate="one_to_one")
    )
    return games.sort_values(
        ["kickoff_utc", "game_id"],
        kind="stable",
    ).reset_index(drop=True)


def run_rare_event_backtest(
    *,
    canonical_root: str | Path,
    warehouse_path: str | Path,
    output_root: str | Path,
    config: RareEventConfig,
) -> dict[str, pd.DataFrame]:
    canonical_root = Path(canonical_root).expanduser().resolve()
    warehouse_path = Path(warehouse_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    games = _load_games(canonical_root, warehouse_path)
    allowed_seasons = set(config.warmup_seasons) | set(config.development_seasons)
    games = games[games["season"].isin(allowed_seasons)].copy()

    margin_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    all_games_seen = 0
    integer_ats_games = 0
    integer_ats_pushes = 0
    integer_total_games = 0
    integer_total_pushes = 0
    regular_games = 0
    regular_ties = 0
    prediction_rows: list[dict[str, object]] = []

    for record in games.itertuples(index=False):
        row = pd.Series(record._asdict())
        season = int(row["season"])
        postseason = str(row["season_type"]).upper() in {
            "POST",
            "POSTSEASON",
            "PLAYOFF",
        }
        candidates: list[
            tuple[str, str, tuple[float, float, float], float, str]
        ] = []

        for strength in config.tie_prior_strengths:
            tie_probability = (
                0.0
                if postseason
                else _beta_posterior_mean(
                    regular_ties,
                    regular_games,
                    config.tie_prior_mean,
                    strength,
                )
            )
            candidates.append(
                (
                    "pregame_moneyline",
                    f"tie_beta_strength_{int(strength)}",
                    _three_way(
                        float(row["moneyline_conditional_home_probability"]),
                        tie_probability,
                    ),
                    strength,
                    "beta_binomial_regular_season",
                )
            )

        ats_line = float(row["ats_line"])
        for strength in config.ats_prior_strengths:
            exact = _ats_exact_margin_probability(
                ats_line,
                margin_counts,
                all_games_seen,
                strength,
                config.ats_margin_prior_sd,
            )
            rate = _ats_integer_rate_probability(
                ats_line,
                integer_ats_pushes,
                integer_ats_games,
                strength,
                config.ats_integer_push_rate_prior_mean,
            )
            for family, probability in (
                ("exact_margin", exact),
                ("integer_rate", rate),
                ("blend", 0.5 * (exact + rate)),
            ):
                candidates.append(
                    (
                        "pregame_ats",
                        f"ats_push_{family}_strength_{int(strength)}",
                        _three_way(
                            float(row["ats_conditional_home_probability"]),
                            probability,
                        ),
                        strength,
                        family,
                    )
                )

        total_line = float(row["total_line"])
        for strength in config.total_prior_strengths:
            exact = _total_exact_points_probability(
                total_line,
                total_counts,
                all_games_seen,
                strength,
                config.total_points_prior_mean,
                config.total_points_prior_sd,
            )
            rate = _total_integer_rate_probability(
                total_line,
                integer_total_pushes,
                integer_total_games,
                strength,
                config.total_integer_push_rate_prior_mean,
            )
            for family, probability in (
                ("exact_points", exact),
                ("integer_rate", rate),
                ("blend", 0.5 * (exact + rate)),
            ):
                candidates.append(
                    (
                        "pregame_total",
                        f"total_push_{family}_strength_{int(strength)}",
                        _three_way(
                            float(row["total_conditional_over_probability"]),
                            probability,
                        ),
                        strength,
                        family,
                    )
                )

        if season in config.development_seasons:
            for (
                market,
                model_name,
                probabilities,
                strength,
                family,
            ) in candidates:
                outcome_index = _outcome_index(market, row)
                log_loss, brier, normalized = _scores(
                    probabilities,
                    outcome_index,
                    config.probability_clip,
                )
                prediction_rows.append(
                    {
                        "game_id": row["game_id"],
                        "season": season,
                        "week": int(row["week"]),
                        "kickoff_utc": row["kickoff_utc"],
                        "market": market,
                        "model_name": model_name,
                        "rare_family": family,
                        "rare_prior_strength": strength,
                        "probability_1": normalized[0],
                        "probability_2": normalized[1],
                        "probability_rare": normalized[2],
                        "outcome_index": outcome_index,
                        "rare_actual": int(outcome_index == 2),
                        "log_loss": log_loss,
                        "brier": brier,
                        "training_games_seen": all_games_seen,
                        "training_cutoff_utc":
                            row["kickoff_utc"] - pd.Timedelta(minutes=10),
                        "selection_eligible": True,
                    }
                )

        actual_margin = int(round(float(row["target_home_margin"])))
        actual_total = int(round(float(row["target_total_points"])))
        margin_counts[actual_margin] += 1
        total_counts[actual_total] += 1
        all_games_seen += 1

        if _is_integer_line(ats_line):
            integer_ats_games += 1
            integer_ats_pushes += int(row["target_t10_ats_push"])

        if _is_integer_line(total_line):
            integer_total_games += 1
            integer_total_pushes += int(row["target_t10_total_push"])

        if not postseason:
            regular_games += 1
            regular_ties += int(row["target_tie"])

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("No development rare-event predictions were generated.")

    scorecard = (
        predictions.groupby(["market", "model_name"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            rare_events=("rare_actual", "sum"),
            actual_rare_rate=("rare_actual", "mean"),
            mean_predicted_rare_rate=("probability_rare", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
        )
    )
    scorecard["rare_calibration_error"] = (
        scorecard["mean_predicted_rare_rate"]
        - scorecard["actual_rare_rate"]
    )
    scorecard = scorecard.sort_values(
        ["market", "mean_log_loss", "mean_brier"],
        kind="stable",
    ).reset_index(drop=True)

    by_season = (
        predictions.groupby(
            ["market", "model_name", "season"],
            as_index=False,
        )
        .agg(
            games=("game_id", "nunique"),
            rare_events=("rare_actual", "sum"),
            actual_rare_rate=("rare_actual", "mean"),
            mean_predicted_rare_rate=("probability_rare", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values(["market", "model_name", "season"])
        .reset_index(drop=True)
    )

    sums = predictions[
        ["probability_1", "probability_2", "probability_rare"]
    ].sum(axis=1)
    integrity = pd.DataFrame(
        [
            {
                "check": "development_seasons_only",
                "value": str(sorted(predictions["season"].unique().tolist())),
                "passed": set(predictions["season"].unique())
                == set(config.development_seasons),
            },
            {
                "check": "probabilities_sum_to_one",
                "value": float((sums - 1.0).abs().max()),
                "passed": bool(np.allclose(sums, 1.0, atol=1e-10)),
            },
            {
                "check": "no_negative_probabilities",
                "value": float(
                    predictions[
                        ["probability_1", "probability_2", "probability_rare"]
                    ].min().min()
                ),
                "passed": bool(
                    (
                        predictions[
                            ["probability_1", "probability_2", "probability_rare"]
                        ] >= 0
                    ).all().all()
                ),
            },
        ]
    )

    predictions.to_parquet(
        output_root / "rare_event_oof_predictions.parquet",
        index=False,
    )
    scorecard.to_csv(
        output_root / "rare_event_scorecard.csv",
        index=False,
    )
    by_season.to_csv(
        output_root / "rare_event_by_season.csv",
        index=False,
    )
    scorecard.to_csv(
        output_root / "rare_event_prior_sensitivity.csv",
        index=False,
    )
    integrity.to_csv(
        output_root / "rare_event_integrity_audit.csv",
        index=False,
    )

    registry = {
        "status": "PASS",
        "warmup_seasons": list(config.warmup_seasons),
        "development_seasons": list(config.development_seasons),
        "candidate_count": int(predictions["model_name"].nunique()),
        "selection": "DEFERRED_TO_UNIFIED_NESTED_TOURNAMENT",
        "postseason_tie_probability": 0.0,
        "noninteger_line_push_probability": 0.0,
    }
    (
        output_root / "rare_event_candidate_registry.json"
    ).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not integrity["passed"].all():
        raise ValueError("Rare-event integrity audit failed.")

    return {
        "predictions": predictions,
        "scorecard": scorecard,
        "by_season": by_season,
        "integrity": integrity,
    }
