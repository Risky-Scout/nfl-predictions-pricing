"""Fix 8: official card-scoped chronological Ridge OOF for the TUE/FRI
horizons, on the frozen six Elo features (Fix 6/Fix 7) and the merged Fix
7.1 V2 card-scoped horizon-as-of Elo semantics (:mod:`nfl_hybrid.features.horizon_elo`).

This is deliberately NOT :mod:`nfl_hybrid.evaluation.chronological_oof` run
again: Fix 3's own ``generate_oof_predictions`` derives each target's cutoff
as ``scheduled_kickoff_utc - 10 minutes`` (the repository's original
per-game "official snapshot" convention) and fits the incumbent
:class:`~nfl_hybrid.modern.joint_score.JointScoreModel` on Fix 3's own
five-state-family feature matrix. Fix 8's official TUE/FRI forecast cutoffs
are the card-scoped ones Fix 7.1 V2 already certified
(:func:`nfl_hybrid.features.horizon_elo.build_horizon_membership_ledger`),
and its frozen estimator is the ``RIDGE_ALPHA_100`` candidate Fix 7 selected
(:mod:`nfl_hybrid.selection.model_family_selection_2026`), fit on exactly
the six frozen Elo features -- not JointScoreModel, not the five-family
matrix. Everything downstream of "which rows form one fit batch, and which
games are eligible to train on which target" is reused from Fix 3 UNCHANGED
(:func:`nfl_hybrid.evaluation.chronological_oof.attach_outcomes_and_residuals`,
:func:`~nfl_hybrid.evaluation.chronological_oof.attach_expanding_oof_uncertainty`,
:func:`~nfl_hybrid.evaluation.chronological_oof.training_membership_hash`) --
this module only replaces WHICH cutoff a batch shares and WHICH
estimator/feature-set fits it, never the chronology invariants themselves.

Training-set eligibility per target-cutoff batch (Fix 8 contract section 4):
a training row must (a) belong to the SAME horizon's own eligible-target
population, (b) satisfy ``result_available_at_utc < target_cutoff_utc``
(STRICT), and (c) use its own historically-generated horizon-as-of feature
vector -- (a)+(c) are structural here, since ``matrix`` is already the one
horizon's own eligible-target matrix (:func:`build_official_horizon_matrix`,
mirroring Fix 7.1's own ``build_horizon_selection_matrix`` construction
exactly). ``result_available_at_utc`` reuses Fix 7.1 V2's own conservative
kickoff+5h floor (:data:`nfl_hybrid.features.horizon_elo.RESULT_AVAILABILITY_BASIS`)
-- the same policy already governing Elo update-event eligibility for these
exact card cutoffs -- rather than Fix 3's kickoff+5h-then-snap-to-next-Tue/Fri-
batch policy, which is redundant once the batch boundary itself IS a
Tue/Fri card cutoff.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from nfl_hybrid.data.availability import assert_available_before
from nfl_hybrid.evaluation.chronological_oof import (
    OOF_LEDGER_COLUMNS,
    attach_expanding_oof_uncertainty,
    attach_outcomes_and_residuals,
    training_membership_hash,
)
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features.feature_manifest import validate_no_banned_features
from nfl_hybrid.features.pregame_rolling import build_game_pregame_matrix
from nfl_hybrid.selection import feature_deduction_2026 as fd

SCHEMA_VERSION = "fix8-official-horizon-oof-v1"
MODEL_NAME = "RIDGE_ALPHA_100[margin,total]"

# Byte-identical to model_family_selection_2026.CANDIDATE_REGISTRY's
# RIDGE_ALPHA_100 entry (verified at orchestration Phase 1 against
# RIDGE_ALPHA_100_SPEC_HASH) -- constructed directly here rather than
# importing that module's private `_build_pipeline` helper.
RIDGE_PREPROCESSING = {"type": "StandardScaler", "with_mean": True, "with_std": True}
RIDGE_HYPERPARAMETERS = {"alpha": 100.0, "fit_intercept": True, "solver": "svd"}

ELO_FEATURE_COLUMNS: tuple[str, ...] = tuple(fd.FEATURE_GROUPS["ELO_STRENGTH"].columns)


@dataclass(frozen=True)
class OfficialHorizonOOFConfig:
    """Structural floors only -- nothing here is tuned, searched, or
    selected against held-out performance. ``min_training_games`` reuses
    Fix 3's own committed default (:class:`nfl_hybrid.evaluation.chronological_oof.ChronologicalOOFConfig`)
    verbatim; ``min_uncertainty_warmup``/rho-clip are Fix 3's own
    :func:`~nfl_hybrid.evaluation.chronological_oof.attach_expanding_oof_uncertainty`
    defaults, reused unchanged (rho clip [-0.95, 0.95] is hard-coded inside
    that reused function, not re-declared here)."""

    min_training_games: int = 48
    training_ids_cap: int = 250
    min_uncertainty_warmup: int = 32


def _build_ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("estimator", Ridge(**RIDGE_HYPERPARAMETERS)),
        ]
    )


def compute_official_model_config_hash(config: OfficialHorizonOOFConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "preprocessing": RIDGE_PREPROCESSING,
        "hyperparameters": RIDGE_HYPERPARAMETERS,
        "feature_columns": list(ELO_FEATURE_COLUMNS),
        "min_training_games": config.min_training_games,
        "training_ids_cap": config.training_ids_cap,
        "min_uncertainty_warmup": config.min_uncertainty_warmup,
        "result_availability_basis": he.RESULT_AVAILABILITY_BASIS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_official_horizon_matrix(games: pd.DataFrame, horizon: str, ledger: pd.DataFrame) -> pd.DataFrame:
    """The one horizon's eligible-target-only matrix: the frozen six Elo
    features (card-scoped horizon-as-of, Fix 7.1 V2), the two OOF targets
    (``home_margin``, ``total_points``), and ``result_available_at_utc`` for
    training-eligibility. Construction mirrors Fix 7.1's own
    ``build_horizon_selection_matrix`` exactly (same Elo state build, same
    carrier/forbidden-column checks) -- this function only adds the two
    scoring targets and the result-availability column Fix 7.1 itself never
    needed."""
    eligible_ids = he.eligible_game_ids(ledger, horizon)
    games = games.copy()
    games["game_id"] = games["game_id"].astype(str)
    games_eligible = games[games["game_id"].isin(eligible_ids)].reset_index(drop=True)

    elo_state = he.build_horizon_elo_state(games, horizon, membership_ledger=ledger)
    team_state = elo_state[
        ["game_id", "team_id", "elo_pregame_rating", "elo_pregame_win_probability", "elo_pregame_expected_margin"]
    ]
    carrier = ("season", "week", "home_score", "away_score", "season_type")
    fd.assert_no_forbidden_market_columns(carrier)

    pivoted = build_game_pregame_matrix(games_eligible, team_state, carrier_columns=carrier)
    pivoted["game_id"] = pivoted["game_id"].astype(str)

    missing = [c for c in ELO_FEATURE_COLUMNS if c not in pivoted.columns]
    if missing:
        raise ValueError(f"Pivoted matrix missing expected feature column(s): {missing}")
    fd.assert_no_forbidden_market_columns(ELO_FEATURE_COLUMNS)
    validate_no_banned_features(ELO_FEATURE_COLUMNS)

    pivoted["home_margin"] = pd.to_numeric(pivoted["home_score"], errors="coerce") - pd.to_numeric(
        pivoted["away_score"], errors="coerce"
    )
    pivoted["total_points"] = pd.to_numeric(pivoted["home_score"], errors="coerce") + pd.to_numeric(
        pivoted["away_score"], errors="coerce"
    )
    pivoted["season"] = pd.to_numeric(pivoted["season"], errors="raise").astype(int)
    for col in ELO_FEATURE_COLUMNS:
        pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce").astype(float)

    if pivoted[list(ELO_FEATURE_COLUMNS)].isna().sum().sum() != 0:
        raise ValueError("Unexpected missingness in the frozen six Elo features.")
    if len(pivoted) != len(eligible_ids):
        raise ValueError(f"Matrix row count mismatch: matrix={len(pivoted)} eligible={len(eligible_ids)}")

    h = horizon.lower()
    cutoff_col = ledger[["game_id", f"{h}_cutoff_utc"]].rename(columns={f"{h}_cutoff_utc": "target_cutoff_utc"})
    cutoff_col = cutoff_col.copy()
    cutoff_col["game_id"] = cutoff_col["game_id"].astype(str)
    pivoted = pivoted.merge(cutoff_col, on="game_id", how="left", validate="one_to_one")

    games_eligible = games_eligible.copy()
    games_eligible["result_available_at_utc"] = he.compute_result_available_at_utc(games_eligible)
    ra = games_eligible[["game_id", "result_available_at_utc"]].copy()
    ra["game_id"] = ra["game_id"].astype(str)
    pivoted = pivoted.merge(ra, on="game_id", how="left", validate="one_to_one")

    pivoted = pivoted.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    return pivoted


def generate_official_horizon_oof_predictions(
    matrix: pd.DataFrame,
    *,
    horizon: str,
    feature_columns: Sequence[str] = ELO_FEATURE_COLUMNS,
    config: OfficialHorizonOOFConfig | None = None,
    feature_state_hash: str = "",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Phase 1: card-scoped chronological OOF point predictions. One
    ``RIDGE_ALPHA_100`` fit per target-cutoff batch per target (margin,
    total) -- every row in ``matrix`` sharing one ``target_cutoff_utc``
    already shares one card, so this is "fit once per horizon cutoff batch,
    not once per game" (Fix 8 contract section 4) by construction, without
    a separate batching step.

    Never reads ``home_margin``/``total_points`` for the row being
    predicted -- only for rows already in the strictly-prior training mask.
    Outcomes are attached afterward by the caller, via Fix 3's own
    :func:`~nfl_hybrid.evaluation.chronological_oof.attach_outcomes_and_residuals`.
    """
    cfg = config or OfficialHorizonOOFConfig()
    feature_columns = list(feature_columns)
    model_config_hash = compute_official_model_config_hash(cfg)

    frame = matrix.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise ValueError("official horizon matrix has duplicate game_id rows.")

    target_cutoff = pd.to_datetime(frame["target_cutoff_utc"], utc=True, errors="raise")
    result_available = pd.to_datetime(frame["result_available_at_utc"], utc=True, errors="raise")

    batches: dict[pd.Timestamp, list[int]] = {}
    for i in range(len(frame)):
        batches.setdefault(target_cutoff.iloc[i], []).append(i)

    rows: list[dict[str, object]] = []
    paired_fits = 0
    individual_fits = 0
    for cutoff_i, positions in sorted(batches.items(), key=lambda kv: kv[0]):
        train_mask = (result_available < cutoff_i).to_numpy()
        train_ids = frame.loc[train_mask, "game_id"].tolist()
        training_count = len(train_ids)
        max_training_result_available = result_available.loc[train_mask].max() if training_count else pd.NaT
        membership_hash = training_membership_hash(train_ids)
        truncated = training_count > cfg.training_ids_cap
        stored_ids = () if truncated else tuple(sorted(train_ids))

        base_record = {
            "target_cutoff_utc": cutoff_i,
            "training_as_of_utc": cutoff_i,
            "max_training_result_available_at_utc": max_training_result_available,
            "result_availability_basis": he.RESULT_AVAILABILITY_BASIS,
            "training_game_count": training_count,
            "training_game_ids": stored_ids,
            "training_game_ids_truncated": truncated,
            "training_membership_hash": membership_hash,
            "model_config_hash": model_config_hash,
            "feature_state_hash": feature_state_hash,
        }

        if training_count < cfg.min_training_games:
            margin_pred = total_pred = None
        else:
            x_train = frame.loc[train_mask, feature_columns].to_numpy(dtype=float)
            y_margin = frame.loc[train_mask, "home_margin"].to_numpy(dtype=float)
            y_total = frame.loc[train_mask, "total_points"].to_numpy(dtype=float)
            x_target = frame.iloc[positions][feature_columns].to_numpy(dtype=float)
            with threadpool_limits(limits=1):
                margin_pipe = _build_ridge_pipeline().fit(x_train, y_margin)
                margin_pred = margin_pipe.predict(x_target)
                total_pipe = _build_ridge_pipeline().fit(x_train, y_total)
                total_pred = total_pipe.predict(x_target)
            paired_fits += 1
            individual_fits += 2

        for slot, i in enumerate(positions):
            record: dict[str, object] = {
                "game_id": frame.at[i, "game_id"],
                "season": int(frame.at[i, "season"]),
                "week": frame.at[i, "week"],
                "result_available_at_utc": result_available.iloc[i],
                **base_record,
            }
            if margin_pred is None:
                record["status"] = "MODEL_NOT_READY"
                record["predicted_margin"] = float("nan")
                record["predicted_total"] = float("nan")
            else:
                record["status"] = "OOF"
                record["predicted_margin"] = float(margin_pred[slot])
                record["predicted_total"] = float(total_pred[slot])
            rows.append(record)

    predictions = pd.DataFrame(rows, columns=list(OOF_LEDGER_COLUMNS))
    assert_available_before(
        predictions,
        available_at_column="max_training_result_available_at_utc",
        prediction_time_column="target_cutoff_utc",
        allow_equal=False,
    )
    fit_counts = {"horizon": horizon, "paired_fits": paired_fits, "individual_fits": individual_fits}
    return predictions, fit_counts


def build_official_horizon_oof(
    matrix: pd.DataFrame,
    *,
    horizon: str,
    feature_columns: Sequence[str] = ELO_FEATURE_COLUMNS,
    config: OfficialHorizonOOFConfig | None = None,
    feature_state_hash: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """End-to-end: card-scoped OOF predictions -> Fix 3's own outcome
    attachment -> Fix 3's own expanding same-horizon uncertainty (never
    pooled with the other horizon -- call this once per horizon)."""
    cfg = config or OfficialHorizonOOFConfig()
    predictions, fit_counts = generate_official_horizon_oof_predictions(
        matrix, horizon=horizon, feature_columns=feature_columns, config=cfg, feature_state_hash=feature_state_hash
    )
    residual_ledger = attach_outcomes_and_residuals(predictions, matrix)
    residual_ledger = attach_expanding_oof_uncertainty(residual_ledger, min_uncertainty_warmup=cfg.min_uncertainty_warmup)
    return predictions, residual_ledger, fit_counts
