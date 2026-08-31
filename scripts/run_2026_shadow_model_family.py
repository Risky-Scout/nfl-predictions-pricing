"""Prospective 2026 shadow model-family runner (Sections 10-11, 20).

A SEPARATE shadow-evidence path. The canonical production forecast stays
``RIDGE_ALPHA_100`` and is NEVER changed, selected, blocked, or overwritten
by anything here.

For one horizon (TUE/FRI) and a games population, this:

  1. builds the certified card-scoped horizon-as-of matrix
     (:func:`nfl_hybrid.evaluation.official_horizon_oof.build_official_horizon_matrix`,
     the frozen six Elo features, unchanged),
  2. replays the certified chronological OOF batching (one batch per
     ``target_cutoff_utc``; training rows are exactly those with
     ``result_available_at_utc < target_cutoff_utc`` -- only information
     legally available to that cutoff),
  3. for EACH frozen Fix-7 candidate
     (:data:`nfl_hybrid.selection.model_family_selection_2026.CANDIDATE_REGISTRY`),
     fits margin+total on that training mask through the candidate's own
     frozen Fix-7 fitting path and predicts the batch,
  4. writes one immutable shadow-ledger record per
     ``(game_id, horizon, candidate, target_cutoff_utc)`` under
     ``$NFL_MODEL_ARTIFACT_ROOT/production-2026/shadow-model-family-ledger/``
     (or ``--shadow-root``). NO outcome column is ever written.

It NEVER touches the forecast ledger, the evaluation ledger, the
production calibration seed, or any historical scientific evidence. A
shadow fit that fails is recorded as ``status="SHADOW_FIT_UNAVAILABLE"``
and cannot affect anything else.

Section 20: run this against a historical / already-elapsed games
population only. It does not fabricate live 2026 evidence -- with the
committed 2020-2025 ``backfill.games`` estate every ``target_cutoff_utc``
it replays is in the past.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve  # noqa: E402
from nfl_hybrid.evaluation import official_horizon_oof as ohf  # noqa: E402
from nfl_hybrid.evaluation import prospective_strength_2026 as ps  # noqa: E402
from nfl_hybrid.features import horizon_elo as he  # noqa: E402
from nfl_hybrid.selection import feature_deduction_2026 as fd  # noqa: E402
from nfl_hybrid.selection import model_family_selection_2026 as mfs  # noqa: E402

_MIN_TRAIN = ohf.OfficialHorizonOOFConfig().min_training_games
_FEATURES = list(ohf.ELO_FEATURE_COLUMNS)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=True
        ).stdout.strip()
    except Exception:
        return None


def _predict_candidate(candidate, x_train, y_margin, y_total, x_target):
    """Fit + predict one frozen Fix-7 candidate on one chronological batch,
    through that candidate's own frozen fitting path. Returns
    ``(margin_pred, total_pred)`` arrays, or ``(None, None)`` if the fit is
    not usable (never raises out to the caller)."""
    try:
        if candidate.family == "HGBR":
            cols = list(_FEATURES)
            train_df = pd.DataFrame(x_train, columns=cols)
            train_df["home_margin"] = y_margin
            train_df["total_points"] = y_total
            model = fd.JointScoreModel(numeric_features=cols, categorical_features=(), config=fd.MODEL_CONFIG)
            model.fit(train_df)
            pm, pt = model.predict_means(pd.DataFrame(x_target, columns=cols))
            return np.asarray(pm, dtype=float), np.asarray(pt, dtype=float)
        _, margin_pred, margin_ok, _ = mfs._fit_one_ridge_or_huber_target(
            candidate.family, candidate.hyperparameters, x_train, y_margin, x_target
        )
        _, total_pred, total_ok, _ = mfs._fit_one_ridge_or_huber_target(
            candidate.family, candidate.hyperparameters, x_train, y_total, x_target
        )
        if not (margin_ok and total_ok):
            return None, None
        return np.asarray(margin_pred, dtype=float), np.asarray(total_pred, dtype=float)
    except Exception:
        return None, None


def run_shadow_horizon(
    games: pd.DataFrame,
    horizon: str,
    shadow_root: Path,
    *,
    git_commit: str | None = None,
) -> dict:
    ledger = he.build_horizon_membership_ledger(games)
    matrix = ohf.build_official_horizon_matrix(games, horizon, ledger)
    matrix = matrix.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)

    target_cutoff = pd.to_datetime(matrix["target_cutoff_utc"], utc=True, errors="raise")
    result_available = pd.to_datetime(matrix["result_available_at_utc"], utc=True, errors="raise")
    feature_state_hash = he.HORIZON_FEATURE_SEMANTICS_VERSION

    batches: dict[pd.Timestamp, list[int]] = {}
    for i in range(len(matrix)):
        batches.setdefault(target_cutoff.iloc[i], []).append(i)

    written = 0
    noop = 0
    unavailable = 0
    for cutoff_i, positions in sorted(batches.items(), key=lambda kv: kv[0]):
        train_mask = (result_available < cutoff_i).to_numpy()
        x_train = matrix.loc[train_mask, _FEATURES].to_numpy(dtype=float)
        y_margin = matrix.loc[train_mask, "home_margin"].to_numpy(dtype=float)
        y_total = matrix.loc[train_mask, "total_points"].to_numpy(dtype=float)
        x_target = matrix.iloc[positions][_FEATURES].to_numpy(dtype=float)
        training_count = int(train_mask.sum())
        train_ids = tuple(sorted(matrix.loc[train_mask, "game_id"].astype(str)))
        training_membership_hash = ps._sha256_hex({"train_ids": list(train_ids)})

        for candidate in mfs.CANDIDATE_REGISTRY:
            if training_count >= _MIN_TRAIN:
                pm, pt = _predict_candidate(candidate, x_train, y_margin, y_total, x_target)
            else:
                pm = pt = None

            for slot, i in enumerate(positions):
                gid = str(matrix.at[i, "game_id"])
                if pm is None:
                    status = "SHADOW_FIT_UNAVAILABLE"
                    predicted_margin = predicted_total = None
                else:
                    status = "SHADOW_OOF"
                    predicted_margin = float(pm[slot])
                    predicted_total = float(pt[slot])
                prediction_payload = {
                    "status": status,
                    "predicted_margin": predicted_margin,
                    "predicted_total": predicted_total,
                    "candidate_family": candidate.family,
                    "candidate_complexity_rank": candidate.complexity_rank,
                    "candidate_spec_hash": mfs.compute_candidate_spec_hash(candidate),
                    "feature_columns": list(_FEATURES),
                    "training_game_count": training_count,
                    "training_membership_hash": training_membership_hash,
                    "min_training_games": _MIN_TRAIN,
                }
                record = {
                    "game_id": gid,
                    "horizon": horizon,
                    "candidate": candidate.name,
                    "target_cutoff_utc": str(cutoff_i),
                    "season": int(matrix.at[i, "season"]),
                    "week": None if pd.isna(matrix.at[i, "week"]) else str(matrix.at[i, "week"]),
                    "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "git_commit": git_commit,
                    "certified_baseline_sha": "d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7",
                    "feature_state_semantics_hash": feature_state_hash,
                    "candidate_registry_hash": mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY),
                    "prediction": prediction_payload,
                    "shadow_disclaimer": (
                        "SHADOW EVIDENCE ONLY -- no production-selection authority; "
                        "canonical production forecast remains RIDGE_ALPHA_100."
                    ),
                }
                result = ps.write_shadow_record(shadow_root, record)
                if result["status"] == "WRITTEN":
                    written += 1
                elif result["status"] == "IDEMPOTENT_NOOP":
                    noop += 1
                if status == "SHADOW_FIT_UNAVAILABLE":
                    unavailable += 1

    return {
        "horizon": horizon,
        "batches": len(batches),
        "candidates": [c.name for c in mfs.CANDIDATE_REGISTRY],
        "records_written": written,
        "records_idempotent_noop": noop,
        "shadow_fit_unavailable_rows": unavailable,
        "shadow_root": str(shadow_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", choices=("TUE", "FRI", "ALL"), default="ALL")
    parser.add_argument(
        "--games-parquet", type=str, default=None,
        help="Historical games parquet (defaults to the committed backfill.games estate).",
    )
    parser.add_argument(
        "--shadow-root", type=str, default=None,
        help="Shadow-ledger root (defaults to "
        "$NFL_MODEL_ARTIFACT_ROOT/production-2026/shadow-model-family-ledger).",
    )
    args = parser.parse_args()

    games_path = Path(args.games_parquet) if args.games_parquet else resolve("backfill.games")
    games = pd.read_parquet(games_path)

    if args.shadow_root:
        shadow_root = Path(args.shadow_root)
    else:
        shadow_root = artifact_root() / "production-2026" / ps.SHADOW_LEDGER_SUBDIR
    shadow_root.mkdir(parents=True, exist_ok=True)

    horizons = he.HORIZONS if args.horizon == "ALL" else (args.horizon,)
    git_commit = _git_commit()
    summary = {"horizons": []}
    for horizon in horizons:
        summary["horizons"].append(run_shadow_horizon(games, horizon, shadow_root, git_commit=git_commit))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
