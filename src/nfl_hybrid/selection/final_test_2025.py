from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np
import pandas as pd

from nfl_hybrid.distributional.rare_event_shrinkage import RareEventConfig
from nfl_hybrid.selection.confirmation_2024 import (
    ConfirmationConfig,
    _build_probabilities,
    _load_metadata,
    _paired_week_bootstrap,
    _rare_probability_table,
)
from nfl_hybrid.selection.unified_development_tournament import (
    _market_and_targets,
    _three_way_scores,
)


MARKETS = (
    "pregame_moneyline",
    "pregame_ats",
    "pregame_total",
)

EXPECTED_FREEZE_SHA = (
    "7811079a5fc8371e30f7941117dbf875"
    "f1bb6359214e76c01bd4dc16d73847e0"
)

EXPECTED_FINAL_MODELS = {
    "pregame_moneyline": (
        "market_baseline",
        "market_t10_canonical",
    ),
    "pregame_ats": (
        "rare_event_market_baseline",
        "ats_push_exact_margin_strength_32",
    ),
    "pregame_total": (
        "market_baseline",
        "market_t10_canonical",
    ),
}


@dataclass(frozen=True)
class FinalTestConfig:
    final_test_season: int = 2025
    training_and_selection_seasons: tuple[int, ...] = (
        2020,
        2021,
        2022,
        2023,
        2024,
    )
    expected_games: int = 285
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260329
    probability_clip: float = 1e-9
    maximum_absolute_rare_calibration_error: float = 0.015
    require_full_game_coverage: bool = True
    require_nonworse_log_loss_for_challenger: bool = True
    require_nonworse_brier_for_challenger: bool = True
    retuning_prohibited: bool = True
    feature_changes_prohibited: bool = True
    calibration_changes_prohibited: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "FinalTestConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["training_and_selection_seasons"] = tuple(
            payload["training_and_selection_seasons"]
        )
        return cls(**payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_final_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("status") != "FROZEN_FOR_2025_FINAL_TEST":
        raise ValueError("Final model specification is not frozen.")

    if payload.get("freeze_sha256") != EXPECTED_FREEZE_SHA:
        raise ValueError("Final model specification SHA-256 mismatch.")

    canonical_payload = dict(payload)
    freeze_hash = canonical_payload.pop("freeze_sha256", None)

    actual_hash = hashlib.sha256(
        json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    if actual_hash != freeze_hash:
        raise ValueError(
            "Final model specification content hash does not match "
            "its recorded freeze hash."
        )

    found = {
        row["market"]: (
            row["final_model_family"],
            row["final_model_name"],
        )
        for row in payload["final_models"]
    }

    if found != EXPECTED_FINAL_MODELS:
        raise ValueError(
            "Frozen production candidates differ from the reviewed "
            "2024-confirmed specification."
        )

    if payload["holdout_access"]["2025_accessed"]:
        raise ValueError(
            "Frozen specification already records completed 2025 access."
        )

    return payload


def _load_final_combined_market(
    root: Path,
    market: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stem = f"{market}_market_augmented_canonical_t10"
    frame_path = root / f"{stem}.parquet"
    manifest_path = root / f"{stem}.manifest.json"

    frame = pd.read_parquet(frame_path)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.merge(
        metadata,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    if frame["kickoff_utc"].isna().any():
        raise ValueError(
            f"{market} has missing kickoff metadata."
        )

    expected_seasons = {
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    }

    if set(frame["season"].unique()) != expected_seasons:
        raise ValueError(
            f"{market} combined matrix seasons are not exactly "
            "2020-2025."
        )

    if len(frame) != 1693:
        raise ValueError(
            f"{market} combined matrix has {len(frame)} rows; "
            "expected 1693."
        )

    if frame["game_id"].nunique() != 1693:
        raise ValueError(
            f"{market} combined matrix game IDs are not unique."
        )

    return (
        frame.sort_values(
            ["kickoff_utc", "game_id"],
            kind="stable",
        ).reset_index(drop=True),
        manifest,
    )


def _read_access_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_access_event(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_candidate_ledger(
    market: str,
    test: pd.DataFrame,
    baseline: np.ndarray,
    final_probabilities: np.ndarray,
    final_family: str,
    final_model_name: str,
    config: FinalTestConfig,
) -> pd.DataFrame:
    _, first, action = _market_and_targets(market, test)

    outcome_index = np.full(len(test), 2, dtype=int)
    outcome_index[action & (first == 1)] = 0
    outcome_index[action & (first == 0)] = 1

    baseline_log, baseline_brier = _three_way_scores(
        baseline[:, 0],
        baseline[:, 1],
        baseline[:, 2],
        outcome_index,
        config.probability_clip,
    )

    final_log, final_brier = _three_way_scores(
        final_probabilities[:, 0],
        final_probabilities[:, 1],
        final_probabilities[:, 2],
        outcome_index,
        config.probability_clip,
    )

    common = {
        "game_id": test["game_id"].astype(str).to_numpy(),
        "season": test["season"].astype(int).to_numpy(),
        "week": test["week"].astype(int).to_numpy(),
        "kickoff_utc": test["kickoff_utc"].to_numpy(),
        "market": market,
        "outcome_index": outcome_index,
    }

    baseline_frame = pd.DataFrame(
        {
            **common,
            "comparison_role": "market_baseline",
            "model_family": "market_baseline",
            "model_name": "market_t10_canonical",
            "probability_1": baseline[:, 0],
            "probability_2": baseline[:, 1],
            "probability_rare": baseline[:, 2],
            "log_loss": baseline_log,
            "brier": baseline_brier,
        }
    )

    final_display_name = (
        "production_market_t10_canonical"
        if final_family == "market_baseline"
        else final_model_name
    )

    final_frame = pd.DataFrame(
        {
            **common,
            "comparison_role": "frozen_final_model",
            "model_family": final_family,
            "model_name": final_display_name,
            "frozen_source_model_name": final_model_name,
            "probability_1": final_probabilities[:, 0],
            "probability_2": final_probabilities[:, 1],
            "probability_rare": final_probabilities[:, 2],
            "log_loss": final_log,
            "brier": final_brier,
        }
    )

    return pd.concat(
        [baseline_frame, final_frame],
        ignore_index=True,
    )


def _scorecard(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = (
        ledger.groupby(
            [
                "market",
                "comparison_role",
                "model_family",
                "model_name",
            ],
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

    frame["rare_calibration_error"] = (
        frame["mean_predicted_rare"]
        - frame["actual_rare_rate"]
    )
    return frame


def _final_decision(
    *,
    market: str,
    frozen_family: str,
    frozen_model_name: str,
    final_row: pd.Series,
    baseline_row: pd.Series,
    bootstrap_row: pd.Series,
    config: FinalTestConfig,
) -> dict[str, Any]:
    full_coverage = (
        int(final_row["games"]) == config.expected_games
    )
    log_loss_pass = (
        float(final_row["mean_log_loss"])
        <= float(baseline_row["mean_log_loss"])
    )
    brier_pass = (
        float(final_row["mean_brier"])
        <= float(baseline_row["mean_brier"])
    )
    rare_pass = (
        abs(float(final_row["rare_calibration_error"]))
        <= config.maximum_absolute_rare_calibration_error
    )

    if frozen_family == "market_baseline":
        production_decision = "PRODUCTION_MARKET_BASELINE"
        production_family = "market_baseline"
        production_model = "market_t10_canonical"
    else:
        challenger_pass = (
            full_coverage
            and (
                log_loss_pass
                if config.require_nonworse_log_loss_for_challenger
                else True
            )
            and (
                brier_pass
                if config.require_nonworse_brier_for_challenger
                else True
            )
            and rare_pass
        )

        if challenger_pass:
            production_decision = "PRODUCTION_CONFIRMED_CHALLENGER"
            production_family = frozen_family
            production_model = frozen_model_name
        else:
            production_decision = "PRODUCTION_MARKET_FALLBACK"
            production_family = "market_baseline"
            production_model = "market_t10_canonical"

    return {
        "market": market,
        "production_decision": production_decision,
        "production_model_family": production_family,
        "production_model_name": production_model,
        "frozen_evaluated_model_family": frozen_family,
        "frozen_evaluated_model_name": frozen_model_name,
        "games": int(final_row["games"]),
        "candidate_mean_log_loss": float(
            final_row["mean_log_loss"]
        ),
        "market_mean_log_loss": float(
            baseline_row["mean_log_loss"]
        ),
        "mean_log_loss_gain": float(
            bootstrap_row["mean_log_loss_gain"]
        ),
        "log_loss_ci_lower": float(
            bootstrap_row["log_loss_ci_lower"]
        ),
        "log_loss_ci_upper": float(
            bootstrap_row["log_loss_ci_upper"]
        ),
        "candidate_mean_brier": float(
            final_row["mean_brier"]
        ),
        "market_mean_brier": float(
            baseline_row["mean_brier"]
        ),
        "mean_brier_gain": float(
            bootstrap_row["mean_brier_gain"]
        ),
        "brier_ci_lower": float(
            bootstrap_row["brier_ci_lower"]
        ),
        "brier_ci_upper": float(
            bootstrap_row["brier_ci_upper"]
        ),
        "rare_calibration_error": float(
            final_row["rare_calibration_error"]
        ),
        "full_coverage_pass": bool(full_coverage),
        "log_loss_pass": bool(log_loss_pass),
        "brier_pass": bool(brier_pass),
        "rare_calibration_pass": bool(rare_pass),
    }


def run_final_2025_evaluation(
    *,
    combined_canonical_root: str | Path,
    warehouse_path: str | Path,
    frozen_spec_path: str | Path,
    rare_event_config_path: str | Path,
    final_test_config_path: str | Path,
    output_root: str | Path,
    access_log_path: str | Path,
    allow_final_test: bool,
) -> dict[str, pd.DataFrame]:
    if not allow_final_test:
        raise PermissionError(
            "2025 final evaluation requires --allow-final-test."
        )

    combined_root = Path(
        combined_canonical_root
    ).expanduser().resolve()
    warehouse_path = Path(warehouse_path).expanduser().resolve()
    frozen_spec_path = Path(frozen_spec_path).expanduser().resolve()
    rare_event_config_path = Path(
        rare_event_config_path
    ).expanduser().resolve()
    final_test_config_path = Path(
        final_test_config_path
    ).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    access_log_path = Path(access_log_path).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    if (output_root / "final_2025_result.json").exists():
        raise RuntimeError(
            "Final 2025 result already exists. Re-evaluation is prohibited."
        )

    prior_events = _read_access_log(access_log_path)
    if any(
        row.get("event") == "2025_FINAL_TEST_EVALUATION_COMPLETED"
        for row in prior_events
    ):
        raise RuntimeError(
            "2025 final evaluation is already recorded as completed."
        )

    frozen_spec = _verify_final_spec(frozen_spec_path)

    expected_rare_hash = frozen_spec.get(
        "source_hashes",
        {},
    ).get("rare_event_config")

    if (
        expected_rare_hash is not None
        and _sha256(rare_event_config_path) != expected_rare_hash
    ):
        raise ValueError(
            "Rare-event configuration does not match the frozen "
            "2025 specification."
        )

    config = FinalTestConfig.from_json(final_test_config_path)
    rare_config = RareEventConfig.from_json(
        rare_event_config_path
    )

    _append_access_event(
        access_log_path,
        {
            "event": "2025_FINAL_TEST_EVALUATION_STARTED",
            "frozen_model_spec_sha256":
                frozen_spec["freeze_sha256"],
            "retuning_permitted": False,
            "feature_changes_permitted": False,
            "calibration_changes_permitted": False,
        },
    )

    metadata = _load_metadata(warehouse_path)

    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}

    for market in MARKETS:
        frame, manifest = _load_final_combined_market(
            combined_root,
            market,
            metadata,
        )
        frames[market] = frame
        manifests[market] = manifest

    for market, manifest in manifests.items():
        if (
            manifest.get("frozen_model_spec_sha256")
            != frozen_spec["freeze_sha256"]
        ):
            raise ValueError(
                f"{market} combined manifest does not match the "
                "frozen 2025 model specification."
            )

    rare_table = _rare_probability_table(
        frames["pregame_moneyline"],
        frames["pregame_ats"],
        frames["pregame_total"],
        rare_config,
        ConfirmationConfig(
            confirmation_season=2025,
            training_seasons=config.training_and_selection_seasons,
            expected_games=config.expected_games,
            bootstrap_repetitions=config.bootstrap_repetitions,
            bootstrap_seed=config.bootstrap_seed,
            probability_clip=config.probability_clip,
            maximum_absolute_rare_calibration_error=
                config.maximum_absolute_rare_calibration_error,
        ),
    )

    frozen_map = {
        row["market"]: row
        for row in frozen_spec["final_models"]
    }

    ledger_parts: list[pd.DataFrame] = []

    for market in MARKETS:
        test = frames[market][
            frames[market]["season"].eq(
                config.final_test_season
            )
        ].copy()

        if len(test) != config.expected_games:
            raise ValueError(
                f"{market} final-test row count is {len(test)}, "
                f"expected {config.expected_games}."
            )

        if market == "pregame_total":
            selected_conditional = test[
                "market_t10_novig_probability"
            ].to_numpy(dtype=float)
        else:
            selected_conditional = None

        baseline, selected = _build_probabilities(
            market,
            test,
            rare_table,
            selected_conditional=selected_conditional,
        )

        frozen = frozen_map[market]
        frozen_family = frozen["final_model_family"]
        frozen_model_name = frozen["final_model_name"]

        if market == "pregame_ats":
            final_probabilities = selected
        else:
            final_probabilities = baseline

        ledger_parts.append(
            _build_candidate_ledger(
                market,
                test,
                baseline,
                final_probabilities,
                frozen_family,
                frozen_model_name,
                config,
            )
        )

    ledger = pd.concat(ledger_parts, ignore_index=True)

    probability_columns = [
        "probability_1",
        "probability_2",
        "probability_rare",
    ]

    if not np.allclose(
        ledger[probability_columns].sum(axis=1).to_numpy(),
        1.0,
        atol=1e-9,
    ):
        raise ValueError("Final-test probabilities do not sum to one.")

    scorecard = _scorecard(ledger)

    bootstrap_input = ledger.copy()
    bootstrap_input.loc[
        bootstrap_input["comparison_role"].eq("frozen_final_model"),
        "model_name",
    ] = bootstrap_input.loc[
        bootstrap_input["comparison_role"].eq("frozen_final_model"),
        "model_name",
    ].astype(str)

    bootstrap = _paired_week_bootstrap(
        bootstrap_input,
        ConfirmationConfig(
            confirmation_season=2025,
            training_seasons=config.training_and_selection_seasons,
            expected_games=config.expected_games,
            bootstrap_repetitions=config.bootstrap_repetitions,
            bootstrap_seed=config.bootstrap_seed,
            probability_clip=config.probability_clip,
            maximum_absolute_rare_calibration_error=
                config.maximum_absolute_rare_calibration_error,
        ),
    )

    decisions: list[dict[str, Any]] = []

    for market in MARKETS:
        baseline_row = scorecard[
            scorecard["market"].eq(market)
            & scorecard["comparison_role"].eq("market_baseline")
        ].iloc[0]

        final_row = scorecard[
            scorecard["market"].eq(market)
            & scorecard["comparison_role"].eq("frozen_final_model")
        ].iloc[0]

        bootstrap_row = bootstrap[
            bootstrap["market"].eq(market)
        ].iloc[0]

        frozen = frozen_map[market]

        decisions.append(
            _final_decision(
                market=market,
                frozen_family=frozen["final_model_family"],
                frozen_model_name=frozen["final_model_name"],
                final_row=final_row,
                baseline_row=baseline_row,
                bootstrap_row=bootstrap_row,
                config=config,
            )
        )

    decisions_frame = pd.DataFrame(decisions)

    ledger.to_parquet(
        output_root / "final_2025_predictions.parquet",
        index=False,
    )
    scorecard.to_csv(
        output_root / "final_2025_scorecard.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_root / "final_2025_bootstrap.csv",
        index=False,
    )
    decisions_frame.to_csv(
        output_root / "final_2025_decisions.csv",
        index=False,
    )

    production_spec = {
        "status": "FINAL_PRODUCTION_SPEC",
        "final_test_season": config.final_test_season,
        "training_and_selection_seasons": list(
            config.training_and_selection_seasons
        ),
        "frozen_input_spec_sha256":
            frozen_spec["freeze_sha256"],
        "production_models":
            decisions_frame.to_dict(orient="records"),
        "retuning_after_final_test_prohibited": True,
        "feature_changes_after_final_test_prohibited": True,
        "calibration_changes_after_final_test_prohibited": True,
        "source_hashes": {
            "frozen_spec": _sha256(frozen_spec_path),
            "rare_event_config":
                _sha256(rare_event_config_path),
            "final_test_config":
                _sha256(final_test_config_path),
        },
    }

    canonical_payload = json.dumps(
        production_spec,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    production_spec["production_spec_sha256"] = hashlib.sha256(
        canonical_payload
    ).hexdigest()

    (
        output_root / "final_production_model_spec.json"
    ).write_text(
        json.dumps(
            production_spec,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "status": "PASS",
        "final_test_season": config.final_test_season,
        "games": config.expected_games,
        "frozen_model_spec_sha256":
            frozen_spec["freeze_sha256"],
        "production_spec_sha256":
            production_spec["production_spec_sha256"],
        "decisions":
            decisions_frame.to_dict(orient="records"),
        "retuning_permitted": False,
        "2025_final_evaluation_completed": True,
    }

    (
        output_root / "final_2025_result.json"
    ).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _append_access_event(
        access_log_path,
        {
            "event": "2025_FINAL_TEST_EVALUATION_COMPLETED",
            "frozen_model_spec_sha256":
                frozen_spec["freeze_sha256"],
            "production_spec_sha256":
                production_spec["production_spec_sha256"],
            "retuning_permitted": False,
            "feature_changes_permitted": False,
            "calibration_changes_permitted": False,
        },
    )

    return {
        "ledger": ledger,
        "scorecard": scorecard,
        "bootstrap": bootstrap,
        "decisions": decisions_frame,
    }
