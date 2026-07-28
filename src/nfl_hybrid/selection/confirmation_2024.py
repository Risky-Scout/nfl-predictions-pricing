from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np
import pandas as pd

from nfl_hybrid.distributional.rare_event_shrinkage import (
    RareEventConfig,
    _ats_exact_margin_probability,
    _ats_integer_rate_probability,
    _beta_posterior_mean,
    _is_integer_line,
    _total_exact_points_probability,
    _total_integer_rate_probability,
)
from nfl_hybrid.selection.unified_development_tournament import (
    CHALLENGER_COLUMNS,
    UnifiedTournamentConfig,
    _market_and_targets,
    _three_way_scores,
    fit_offset_logistic,
)


MARKETS = (
    "pregame_moneyline",
    "pregame_ats",
    "pregame_total",
)


@dataclass(frozen=True)
class ConfirmationConfig:
    confirmation_season: int = 2024
    training_seasons: tuple[int, ...] = (2020, 2021, 2022, 2023)
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260328
    probability_clip: float = 1e-9
    maximum_absolute_rare_calibration_error: float = 0.015
    require_full_game_coverage: bool = True
    expected_games: int = 285
    no_retuning: bool = True
    tuning_aggregation_rule: str = "mode_then_strongest_regularization"

    @classmethod
    def from_json(cls, path: str | Path) -> "ConfirmationConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["training_seasons"] = tuple(payload["training_seasons"])
        return cls(**payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_spec(
    spec_path: Path,
    final_development_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))

    if payload.get("status") != "CONFIRMATION_CANDIDATES_FROZEN":
        raise ValueError("Confirmation candidate specification is not frozen.")

    if payload.get("holdout_access", {}).get("2025_accessed"):
        raise ValueError("Frozen specification indicates 2025 was accessed.")

    expected_hash = payload.get("freeze_sha256")
    if not expected_hash:
        raise ValueError("Frozen specification has no freeze_sha256.")

    canonical_payload = dict(payload)
    canonical_payload.pop("freeze_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen specification hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    source_map = {
        "selection_decisions":
            final_development_root / "selection_decisions.csv",
        "candidate_registry":
            final_development_root / "candidate_registry.json",
        "proposed_model_spec":
            final_development_root / "proposed_frozen_model_spec.json",
        "tournament_config":
            repo_root / "config/unified_development_tournament.json",
        "rare_event_config":
            repo_root / "config/rare_event_shrinkage.json",
        "market_feature_roles":
            repo_root / "config/market_feature_roles.yaml",
    }

    for key, path in source_map.items():
        expected = payload["source_hashes"].get(key)
        if expected is None:
            raise ValueError(f"Frozen specification is missing source hash: {key}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen source hash mismatch for {key}: "
                f"expected={expected}, actual={actual}"
            )

    expected_candidates = {
        "pregame_moneyline": (
            "rare_event_market_baseline",
            "tie_beta_strength_800",
        ),
        "pregame_ats": (
            "rare_event_market_baseline",
            "ats_push_exact_margin_strength_32",
        ),
        "pregame_total": (
            "canonical_market_residual",
            "canonical_residual_movement_only",
        ),
    }

    found = {
        row["market"]: (
            row["selected_model_family"],
            row["selected_model_name"],
        )
        for row in payload["candidates"]
    }

    if found != expected_candidates:
        raise ValueError(
            "Frozen candidate set does not match the reviewed development decision."
        )

    return payload


def _load_metadata(path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required.") from exc

    columns = set(pq.ParquetFile(path).schema_arrow.names)
    kickoff_column = next(
        (
            name
            for name in (
                "scheduled_kickoff_utc",
                "kickoff_utc",
                "start_time_utc",
                "game_datetime_utc",
            )
            if name in columns
        ),
        None,
    )
    if kickoff_column is None:
        raise ValueError("Warehouse has no kickoff timestamp column.")

    selected = ["game_id", kickoff_column]
    if "season_type" in columns:
        selected.append("season_type")

    metadata = pd.read_parquet(path, columns=selected).rename(
        columns={kickoff_column: "kickoff_utc"}
    )
    metadata["game_id"] = metadata["game_id"].astype(str)
    metadata["kickoff_utc"] = pd.to_datetime(
        metadata["kickoff_utc"],
        utc=True,
        errors="raise",
    )
    if "season_type" not in metadata.columns:
        metadata["season_type"] = "REG"
    if metadata["game_id"].duplicated().any():
        raise ValueError("Warehouse metadata has duplicate game IDs.")
    return metadata


def _load_combined_market(
    root: Path,
    market: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stem = f"{market}_market_augmented_canonical_t10"
    frame = pd.read_parquet(root / f"{stem}.parquet")
    manifest = json.loads(
        (root / f"{stem}.manifest.json").read_text(encoding="utf-8")
    )
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.merge(
        metadata,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    if frame["kickoff_utc"].isna().any():
        raise ValueError(f"{market} has missing kickoff metadata.")
    expected = {2020, 2021, 2022, 2023, 2024}
    if set(frame["season"].unique()) != expected:
        raise ValueError(
            f"{market} combined matrix seasons are not exactly 2020-2024."
        )
    return (
        frame.sort_values(
            ["kickoff_utc", "game_id"],
            kind="stable",
        ).reset_index(drop=True),
        manifest,
    )


def _deterministic_regularization(
    tuning_path: Path,
    market: str,
    feature_set: str,
) -> float:
    tuning = pd.read_csv(tuning_path)
    subset = tuning[
        tuning["market"].eq(market)
        & tuning["feature_set"].eq(feature_set)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No development tuning rows for {market}/{feature_set}."
        )

    counts = subset["regularization"].value_counts()
    maximum_count = counts.max()
    tied = sorted(
        float(value)
        for value, count in counts.items()
        if count == maximum_count
    )
    return max(tied)


def _rare_probability_table(
    ml: pd.DataFrame,
    ats: pd.DataFrame,
    total: pd.DataFrame,
    rare_config: RareEventConfig,
    confirmation_config: ConfirmationConfig,
) -> pd.DataFrame:
    common = ml[
        [
            "game_id",
            "season",
            "week",
            "kickoff_utc",
            "season_type",
            "target_home_margin",
            "target_tie",
        ]
    ].copy()

    common = common.merge(
        ats[
            [
                "game_id",
                "market_t10_consensus_line",
                "target_t10_ats_push",
            ]
        ].rename(
            columns={
                "market_t10_consensus_line": "ats_line",
            }
        ),
        on="game_id",
        validate="one_to_one",
    ).merge(
        total[
            [
                "game_id",
                "market_t10_consensus_line",
                "target_t10_total_push",
                "target_total_points",
            ]
        ].rename(
            columns={
                "market_t10_consensus_line": "total_line",
            }
        ),
        on="game_id",
        validate="one_to_one",
    )

    common = common.sort_values(
        ["kickoff_utc", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    margin_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    games_seen = 0
    integer_ats_games = 0
    integer_ats_pushes = 0
    integer_total_games = 0
    integer_total_pushes = 0
    regular_games = 0
    regular_ties = 0

    rows: list[dict[str, Any]] = []

    for row in common.itertuples(index=False):
        postseason = str(row.season_type).upper() in {
            "POST",
            "POSTSEASON",
            "PLAYOFF",
        }

        baseline_tie = (
            0.0
            if postseason
            else _beta_posterior_mean(
                regular_ties,
                regular_games,
                rare_config.tie_prior_mean,
                400.0,
            )
        )
        selected_tie = (
            0.0
            if postseason
            else _beta_posterior_mean(
                regular_ties,
                regular_games,
                rare_config.tie_prior_mean,
                800.0,
            )
        )

        ats_exact_64 = _ats_exact_margin_probability(
            float(row.ats_line),
            margin_counts,
            games_seen,
            64.0,
            rare_config.ats_margin_prior_sd,
        )
        ats_rate_64 = _ats_integer_rate_probability(
            float(row.ats_line),
            integer_ats_pushes,
            integer_ats_games,
            64.0,
            rare_config.ats_integer_push_rate_prior_mean,
        )
        baseline_ats = 0.5 * (ats_exact_64 + ats_rate_64)
        selected_ats = _ats_exact_margin_probability(
            float(row.ats_line),
            margin_counts,
            games_seen,
            32.0,
            rare_config.ats_margin_prior_sd,
        )

        total_exact_64 = _total_exact_points_probability(
            float(row.total_line),
            total_counts,
            games_seen,
            64.0,
            rare_config.total_points_prior_mean,
            rare_config.total_points_prior_sd,
        )
        total_rate_64 = _total_integer_rate_probability(
            float(row.total_line),
            integer_total_pushes,
            integer_total_games,
            64.0,
            rare_config.total_integer_push_rate_prior_mean,
        )
        baseline_total = 0.5 * (total_exact_64 + total_rate_64)

        if int(row.season) == confirmation_config.confirmation_season:
            rows.append(
                {
                    "game_id": row.game_id,
                    "baseline_moneyline_rare": baseline_tie,
                    "selected_moneyline_rare": selected_tie,
                    "baseline_ats_rare": baseline_ats,
                    "selected_ats_rare": selected_ats,
                    "baseline_total_rare": baseline_total,
                    "selected_total_rare": baseline_total,
                }
            )

        actual_margin = int(round(float(row.target_home_margin)))
        actual_total = int(round(float(row.target_total_points)))
        margin_counts[actual_margin] += 1
        total_counts[actual_total] += 1
        games_seen += 1

        if _is_integer_line(float(row.ats_line)):
            integer_ats_games += 1
            integer_ats_pushes += int(row.target_t10_ats_push)

        if _is_integer_line(float(row.total_line)):
            integer_total_games += 1
            integer_total_pushes += int(row.target_t10_total_push)

        if not postseason:
            regular_games += 1
            regular_ties += int(row.target_tie)

    output = pd.DataFrame(rows)
    if len(output) != confirmation_config.expected_games:
        raise ValueError(
            "Rare-event confirmation table has incorrect game count: "
            f"{len(output)}"
        )
    return output


def _build_probabilities(
    market: str,
    test: pd.DataFrame,
    rare_table: pd.DataFrame,
    *,
    selected_conditional: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    merged = test[
        ["game_id", "market_t10_novig_probability"]
    ].merge(
        rare_table,
        on="game_id",
        validate="one_to_one",
    )

    market_conditional = pd.to_numeric(
        merged["market_t10_novig_probability"],
        errors="raise",
    ).to_numpy(dtype=float)

    if market == "pregame_moneyline":
        baseline_rare = merged[
            "baseline_moneyline_rare"
        ].to_numpy(dtype=float)
        selected_rare = merged[
            "selected_moneyline_rare"
        ].to_numpy(dtype=float)
        selected_conditional = market_conditional

    elif market == "pregame_ats":
        baseline_rare = merged[
            "baseline_ats_rare"
        ].to_numpy(dtype=float)
        selected_rare = merged[
            "selected_ats_rare"
        ].to_numpy(dtype=float)
        selected_conditional = market_conditional

    elif market == "pregame_total":
        baseline_rare = merged[
            "baseline_total_rare"
        ].to_numpy(dtype=float)
        selected_rare = merged[
            "selected_total_rare"
        ].to_numpy(dtype=float)
        if selected_conditional is None:
            raise ValueError("Total confirmation requires selected conditional.")

    else:
        raise ValueError(market)

    baseline = np.column_stack(
        [
            (1.0 - baseline_rare) * market_conditional,
            (1.0 - baseline_rare) * (1.0 - market_conditional),
            baseline_rare,
        ]
    )
    selected = np.column_stack(
        [
            (1.0 - selected_rare) * selected_conditional,
            (1.0 - selected_rare) * (1.0 - selected_conditional),
            selected_rare,
        ]
    )
    return baseline, selected


def _candidate_frame(
    market: str,
    test: pd.DataFrame,
    baseline: np.ndarray,
    selected: np.ndarray,
    selected_family: str,
    selected_name: str,
    config: ConfirmationConfig,
) -> pd.DataFrame:
    outcome = np.full(len(test), 2, dtype=int)
    _, first, action = _market_and_targets(market, test)
    outcome[action & (first == 1)] = 0
    outcome[action & (first == 0)] = 1

    baseline_log, baseline_brier = _three_way_scores(
        baseline[:, 0],
        baseline[:, 1],
        baseline[:, 2],
        outcome,
        config.probability_clip,
    )
    selected_log, selected_brier = _three_way_scores(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        outcome,
        config.probability_clip,
    )

    common = {
        "game_id": test["game_id"].to_numpy(),
        "season": test["season"].to_numpy(),
        "week": test["week"].to_numpy(),
        "kickoff_utc": test["kickoff_utc"].to_numpy(),
        "market": market,
        "outcome_index": outcome,
    }

    baseline_frame = pd.DataFrame(
        {
            **common,
            "model_family": "market_baseline",
            "model_name": "market_t10_canonical",
            "probability_1": baseline[:, 0],
            "probability_2": baseline[:, 1],
            "probability_rare": baseline[:, 2],
            "log_loss": baseline_log,
            "brier": baseline_brier,
        }
    )
    selected_frame = pd.DataFrame(
        {
            **common,
            "model_family": selected_family,
            "model_name": selected_name,
            "probability_1": selected[:, 0],
            "probability_2": selected[:, 1],
            "probability_rare": selected[:, 2],
            "log_loss": selected_log,
            "brier": selected_brier,
        }
    )
    return pd.concat(
        [baseline_frame, selected_frame],
        ignore_index=True,
    )


def _paired_week_bootstrap(
    ledger: pd.DataFrame,
    config: ConfirmationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.bootstrap_seed)
    rows: list[dict[str, Any]] = []

    for market in MARKETS:
        baseline = ledger[
            ledger["market"].eq(market)
            & ledger["model_name"].eq("market_t10_canonical")
        ][["game_id", "week", "log_loss", "brier"]].rename(
            columns={
                "log_loss": "baseline_log_loss",
                "brier": "baseline_brier",
            }
        )
        selected = ledger[
            ledger["market"].eq(market)
            & ~ledger["model_name"].eq("market_t10_canonical")
        ].copy()

        paired = selected.merge(
            baseline,
            on=["game_id", "week"],
            validate="one_to_one",
        )
        paired["log_loss_gain"] = (
            paired["baseline_log_loss"] - paired["log_loss"]
        )
        paired["brier_gain"] = (
            paired["baseline_brier"] - paired["brier"]
        )

        weeks = sorted(paired["week"].unique().tolist())
        week_groups = {
            week: group
            for week, group in paired.groupby("week")
        }

        log_samples = np.empty(config.bootstrap_repetitions)
        brier_samples = np.empty(config.bootstrap_repetitions)

        for index in range(config.bootstrap_repetitions):
            chosen = rng.choice(
                weeks,
                size=len(weeks),
                replace=True,
            )
            sample = pd.concat(
                [week_groups[week] for week in chosen],
                ignore_index=True,
            )
            log_samples[index] = sample["log_loss_gain"].mean()
            brier_samples[index] = sample["brier_gain"].mean()

        rows.append(
            {
                "market": market,
                "selected_model_family":
                    selected["model_family"].iloc[0],
                "selected_model_name":
                    selected["model_name"].iloc[0],
                "games": int(paired["game_id"].nunique()),
                "mean_log_loss_gain":
                    float(paired["log_loss_gain"].mean()),
                "log_loss_ci_lower":
                    float(np.quantile(log_samples, 0.025)),
                "log_loss_ci_upper":
                    float(np.quantile(log_samples, 0.975)),
                "mean_brier_gain":
                    float(paired["brier_gain"].mean()),
                "brier_ci_lower":
                    float(np.quantile(brier_samples, 0.025)),
                "brier_ci_upper":
                    float(np.quantile(brier_samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def run_2024_confirmation(
    *,
    combined_canonical_root: str | Path,
    warehouse_path: str | Path,
    final_development_root: str | Path,
    freeze_spec_path: str | Path,
    repo_root: str | Path,
    confirmation_config_path: str | Path,
    output_root: str | Path,
    allow_confirmation: bool,
) -> dict[str, pd.DataFrame]:
    if not allow_confirmation:
        raise PermissionError(
            "2024 confirmation requires --allow-2024-confirmation."
        )

    combined_root = Path(combined_canonical_root).expanduser().resolve()
    warehouse_path = Path(warehouse_path).expanduser().resolve()
    final_root = Path(final_development_root).expanduser().resolve()
    spec_path = Path(freeze_spec_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    confirmation_config_path = (
        Path(confirmation_config_path).expanduser().resolve()
    )
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    confirmation_config = ConfirmationConfig.from_json(
        confirmation_config_path
    )
    frozen_spec = _verify_frozen_spec(
        spec_path,
        final_root,
        repo_root,
    )
    unified_config = UnifiedTournamentConfig.from_json(
        repo_root / "config/unified_development_tournament.json"
    )
    rare_config = RareEventConfig.from_json(
        repo_root / "config/rare_event_shrinkage.json"
    )

    metadata = _load_metadata(warehouse_path)

    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}

    for market in MARKETS:
        frame, manifest = _load_combined_market(
            combined_root,
            market,
            metadata,
        )
        frames[market] = frame
        manifests[market] = manifest

    rare_table = _rare_probability_table(
        frames["pregame_moneyline"],
        frames["pregame_ats"],
        frames["pregame_total"],
        rare_config,
        confirmation_config,
    )

    total_frame = frames["pregame_total"]
    training = total_frame[
        total_frame["season"].isin(
            confirmation_config.training_seasons
        )
    ].copy()
    test_total = total_frame[
        total_frame["season"].eq(
            confirmation_config.confirmation_season
        )
    ].copy()

    total_feature_columns = [
        column
        for column in manifests["pregame_total"]["features"]
        if column in CHALLENGER_COLUMNS
    ]

    regularization = _deterministic_regularization(
        final_root / "nested_tuning_report.csv",
        "pregame_total",
        "movement_only",
    )

    market_train, first_train, action_train = _market_and_targets(
        "pregame_total",
        training,
    )
    total_model = fit_offset_logistic(
        training.loc[action_train],
        total_feature_columns,
        first_train[action_train],
        market_train[action_train],
        regularization=regularization,
        clip=confirmation_config.probability_clip,
    )

    market_test_total, _, _ = _market_and_targets(
        "pregame_total",
        test_total,
    )
    selected_total_conditional = total_model.predict(
        test_total,
        market_test_total,
        confirmation_config.probability_clip,
    )

    decision_map = {
        row["market"]: row
        for row in frozen_spec["candidates"]
    }

    ledger_parts: list[pd.DataFrame] = []

    for market in MARKETS:
        test = frames[market][
            frames[market]["season"].eq(
                confirmation_config.confirmation_season
            )
        ].copy()

        selected_conditional = (
            selected_total_conditional
            if market == "pregame_total"
            else None
        )
        baseline, selected = _build_probabilities(
            market,
            test,
            rare_table,
            selected_conditional=selected_conditional,
        )
        selected_decision = decision_map[market]

        ledger_parts.append(
            _candidate_frame(
                market,
                test,
                baseline,
                selected,
                selected_decision["selected_model_family"],
                selected_decision["selected_model_name"],
                confirmation_config,
            )
        )

    ledger = pd.concat(ledger_parts, ignore_index=True)

    if confirmation_config.require_full_game_coverage:
        for market in MARKETS:
            games = ledger[
                ledger["market"].eq(market)
                & ~ledger["model_name"].eq("market_t10_canonical")
            ]["game_id"].nunique()
            if games != confirmation_config.expected_games:
                raise ValueError(
                    f"{market} confirmation coverage is {games}, "
                    f"expected {confirmation_config.expected_games}."
                )

    scorecard = (
        ledger.groupby(
            ["market", "model_family", "model_name"],
            as_index=False,
        )
        .agg(
            games=("game_id", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            mean_predicted_rare=("probability_rare", "mean"),
            actual_rare_rate=(
                "outcome_index",
                lambda values: float((values == 2).mean()),
            ),
        )
    )
    scorecard["rare_calibration_error"] = (
        scorecard["mean_predicted_rare"]
        - scorecard["actual_rare_rate"]
    )

    bootstrap = _paired_week_bootstrap(
        ledger,
        confirmation_config,
    )

    decisions: list[dict[str, Any]] = []

    for market in MARKETS:
        baseline = scorecard[
            scorecard["market"].eq(market)
            & scorecard["model_name"].eq("market_t10_canonical")
        ].iloc[0]
        selected = scorecard[
            scorecard["market"].eq(market)
            & ~scorecard["model_name"].eq("market_t10_canonical")
        ].iloc[0]
        paired = bootstrap[
            bootstrap["market"].eq(market)
        ].iloc[0]

        full_coverage = (
            int(selected["games"])
            == confirmation_config.expected_games
        )
        log_loss_pass = (
            float(selected["mean_log_loss"])
            <= float(baseline["mean_log_loss"])
        )
        brier_pass = (
            float(selected["mean_brier"])
            <= float(baseline["mean_brier"])
        )
        rare_pass = (
            abs(float(selected["rare_calibration_error"]))
            <= confirmation_config.maximum_absolute_rare_calibration_error
        )
        confirmed = (
            full_coverage
            and log_loss_pass
            and brier_pass
            and rare_pass
        )

        decisions.append(
            {
                "market": market,
                "confirmation_decision": (
                    "CONFIRMED_FOR_2025_FINAL_TEST"
                    if confirmed
                    else "RETAIN_MARKET_BASELINE"
                ),
                "selected_model_family":
                    selected["model_family"],
                "selected_model_name":
                    selected["model_name"],
                "games": int(selected["games"]),
                "candidate_mean_log_loss":
                    float(selected["mean_log_loss"]),
                "market_mean_log_loss":
                    float(baseline["mean_log_loss"]),
                "mean_log_loss_gain":
                    float(paired["mean_log_loss_gain"]),
                "log_loss_ci_lower":
                    float(paired["log_loss_ci_lower"]),
                "log_loss_ci_upper":
                    float(paired["log_loss_ci_upper"]),
                "candidate_mean_brier":
                    float(selected["mean_brier"]),
                "market_mean_brier":
                    float(baseline["mean_brier"]),
                "mean_brier_gain":
                    float(paired["mean_brier_gain"]),
                "brier_ci_lower":
                    float(paired["brier_ci_lower"]),
                "brier_ci_upper":
                    float(paired["brier_ci_upper"]),
                "rare_calibration_error":
                    float(selected["rare_calibration_error"]),
                "full_coverage_pass": full_coverage,
                "log_loss_pass": log_loss_pass,
                "brier_pass": brier_pass,
                "rare_calibration_pass": rare_pass,
            }
        )

    decisions_frame = pd.DataFrame(decisions)

    ledger.to_parquet(
        output_root / "confirmation_2024_predictions.parquet",
        index=False,
    )
    scorecard.to_csv(
        output_root / "confirmation_2024_scorecard.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_root / "confirmation_2024_bootstrap.csv",
        index=False,
    )
    decisions_frame.to_csv(
        output_root / "confirmation_2024_decisions.csv",
        index=False,
    )

    access_record = {
        "event": "2024_ARCHITECTURE_CONFIRMATION",
        "freeze_sha256": frozen_spec["freeze_sha256"],
        "confirmation_config_sha256":
            _sha256(confirmation_config_path),
        "2024_accessed": True,
        "2025_accessed": False,
    }
    (
        output_root / "holdout_access_log.jsonl"
    ).write_text(
        json.dumps(access_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = {
        "status": "PASS",
        "confirmation_season":
            confirmation_config.confirmation_season,
        "training_seasons":
            list(confirmation_config.training_seasons),
        "total_residual_regularization":
            regularization,
        "total_residual_feature_count":
            len(total_feature_columns),
        "total_residual_active_feature_count":
            len(total_model.active_columns),
        "decisions":
            decisions_frame.to_dict(orient="records"),
        "2024_accessed": True,
        "2025_accessed": False,
    }
    (
        output_root / "confirmation_2024_result.json"
    ).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "ledger": ledger,
        "scorecard": scorecard,
        "bootstrap": bootstrap,
        "decisions": decisions_frame,
    }
