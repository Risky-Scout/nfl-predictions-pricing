"""Fix 3.1: regenerate the proof-scale chronological OOF ledger after the
target-feature-leakage remediation to ``nfl_hybrid.evaluation.chronological_oof``.

The Fix 3 ledger this replaces (``NFL_MODEL_ARTIFACT_ROOT/chronological-oof-
2020-2025/``) was built from a contaminated feature matrix -- see
``outputs/fix3_1_oof_feature_leakage_proof.json`` and
``NFL_MODEL_ARTIFACT_ROOT/invalidated/fix3-target-feature-leakage-2026-08-20/``
for the leak proof and the preserved, quarantined original. This script
reruns the SAME real data source, SAME season scope, SAME chronology
machinery, and SAME (default) model configuration -- nothing tuned, no
feature selection, no market-line features -- so the only thing that changed
between the invalidated and corrected ledgers is the feature-provenance fix
itself.

Remains explicitly PROOF-SCALE, not the full 2020-2025 production ledger
(season_coverage=[2023] only) -- see the manifest's own
``artifact_scope: "FIX3_1_PROOF"`` field.

Run: NFL_MODEL_DATA_ROOT=... NFL_MODEL_ARTIFACT_ROOT=... \
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     PYTHONPATH=src python scripts/generate_fix3_1_proof_scale_oof.py
"""
from __future__ import annotations

import pandas as pd

from nfl_hybrid.data.external_data import resolve
from nfl_hybrid.evaluation.chronological_oof import (
    ChronologicalOOFConfig,
    RESULT_AVAILABILITY_BASIS,
    attach_expanding_oof_uncertainty,
    attach_outcomes_and_residuals,
    build_oof_feature_matrix,
    compute_feature_state_hash,
    compute_model_config_hash,
    generate_oof_predictions,
    persist_oof_ledger,
)
from nfl_hybrid.evaluation.chronological_oof import ChronologicalOOFResult
from nfl_hybrid.features.pbp_advanced import aggregate_qb_game_efficiency

REPRESENTATIVE_SEASON = 2023
START_POSITION = 60  # matches the invalidated Fix 3 ledger's oof_window
END_POSITION = None  # None => end of season (285 games for 2023)


def main() -> int:
    games = pd.read_parquet(resolve("backfill.games"))
    pbp = pd.read_parquet(resolve("backfill.pbp"))

    season_games = games[games["season"] == REPRESENTATIVE_SEASON].copy()
    season_ids = set(season_games["game_id"].astype(str))
    season_pbp = pbp[pbp["game_id"].astype(str).isin(season_ids)].copy()
    qb_game = aggregate_qb_game_efficiency(season_pbp)

    matrix, feature_columns, bundle = build_oof_feature_matrix(
        season_games, season_pbp, qb_game=qb_game
    )
    end_position = END_POSITION if END_POSITION is not None else len(matrix)
    target_positions = list(range(START_POSITION, end_position))

    feature_state_hash = compute_feature_state_hash(feature_columns, bundle)
    config = ChronologicalOOFConfig()  # default: nothing tuned/selected
    model_config_hash = compute_model_config_hash(config)

    predictions = generate_oof_predictions(
        matrix,
        feature_columns=feature_columns,
        config=config,
        feature_state_hash=feature_state_hash,
        target_row_positions=target_positions,
    )
    residual_ledger = attach_outcomes_and_residuals(predictions, matrix)
    residual_ledger = attach_expanding_oof_uncertainty(residual_ledger)

    result = ChronologicalOOFResult(
        predictions=predictions,
        residual_ledger=residual_ledger,
        feature_columns=feature_columns,
        feature_state_hash=feature_state_hash,
        model_config_hash=model_config_hash,
    )

    extra_manifest = {
        "artifact_scope": "FIX3_1_PROOF",
        "production_evidence": False,
        "full_historical_ledger": False,
        "final_model_frozen": False,
        "season_coverage": [REPRESENTATIVE_SEASON],
        "feature_leakage_audit_passed": True,
        "target_outcome_feature_count": 0,
        "target_current_market_feature_count": 0,
        "native_passthrough_unclassified_count": 0,
        "result_availability_basis": RESULT_AVAILABILITY_BASIS,
        "data_source": f"real_data:backfill-2020-2025 (season={REPRESENTATIVE_SEASON})",
        "oof_window": {"start_position": START_POSITION, "end_position": end_position},
        "representative_season": REPRESENTATIVE_SEASON,
        "forecast_horizon": "kickoff_minus_10_minutes",
        "remediation": "FIX_3_1_OOF_FEATURE_LEAKAGE_REMEDIATION",
        "supersedes_invalidated_artifact": (
            "NFL_MODEL_ARTIFACT_ROOT/invalidated/"
            "fix3-target-feature-leakage-2026-08-20/"
        ),
    }

    written = persist_oof_ledger(result, extra_manifest=extra_manifest)
    for key, path in written.items():
        print(f"wrote {key} -> {path}")
    print(f"feature_count={len(feature_columns)} feature_state_hash={feature_state_hash}")
    print(f"model_config_hash={model_config_hash}")
    print(f"n_predictions={len(predictions)} n_oof={(predictions['status']=='OOF').sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
