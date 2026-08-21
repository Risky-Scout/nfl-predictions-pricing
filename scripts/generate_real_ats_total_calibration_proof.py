"""FIX 4 REPAIR (final hardening pass): real ATS + TOTAL chronological
calibration certification.

Proves, on the already-persisted real Fix 3 OOF ledger (never regenerated
here -- SHA256-verified read-only, see :func:`_hash_fix3_artifacts`) and the
real historical closing_t10 market lines actually available at each game's
own ``target_cutoff_utc``, that the Fix 4 chronological calibration engine
produces genuine, chronologically-valid, FULLY three-way calibrated (both
conditional AND push probability) ATS and TOTAL probabilities -- the
repository's PRIMARY production targets. Before accepting those metrics at
face value, this script also runs a leakage/settlement sanity audit (item 2)
against the exact 3,125 features Fix 3 actually trained on and against the
independent canonical T10 settlement targets.

Architecture (non-negotiable, matches the repair spec exactly):

    Fix 3 residual/uncertainty ledger (persisted; loaded, never regenerated)
        -> chronological_market_attachment.build_market_raw_ledger
           (real T-10 line, independently re-verified per row)
        -> chronological_calibration.generate_chronological_calibration
           (conditional, expanding-window, result-availability-gated)
        -> chronological_calibration.calibrate_chronological_push_probability
           (push mass, same fixed family, same chronological membership)
        -> independent re-verification of every Fix 4 invariant
        -> leakage/settlement sanity audit (descriptive only)
        -> corrected (conditional + three-way) diagnostics
        -> persisted ATS/TOTAL artifacts, explicit market partition
        -> outputs/real_ats_total_chronological_calibration_proof.json

Run:
    NFL_MODEL_DATA_ROOT=/path/to/data NFL_MODEL_ARTIFACT_ROOT=/path/to/artifacts \
        PYTHONPATH=src python scripts/generate_real_ats_total_calibration_proof.py

Scope (item 21): no feature selection, no predictive retraining, no
calibrator-family tournament, no BDL/Week1/TUE-FRI work, no commit, no PR.
This script only READS the already-persisted Fix 3 ledger/manifest and the
canonical market matrices, and writes the Fix 4 ATS/TOTAL proof artifacts --
never :func:`nfl_hybrid.evaluation.chronological_oof.generate_oof_predictions`
or any other Fix 3 training path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.external_data import resolve, resolve_for_write
from nfl_hybrid.evaluation.chronological_calibration import (
    CALIBRATOR_FAMILY,
    MARKET_ATS,
    MARKET_TOTAL,
    ChronologicalCalibrationConfig,
    calibrate_chronological_push_probability,
    compute_calibrator_config_hash,
    compute_push_calibration_config_hash,
    generate_chronological_calibration,
)
from nfl_hybrid.calibration.three_way import CalibrationConfig
from nfl_hybrid.evaluation.chronological_market_attachment import (
    CONSENSUS_METHOD,
    FORECAST_HORIZON,
    MAXIMUM_SNAPSHOT_LAG_MINUTES,
    MINIMUM_ELIGIBLE_BOOKS,
    assert_primary_markets_certified,
    build_market_raw_ledger,
    load_canonical_t10_settlement_columns,
    load_contributing_bookmaker_keys,
)
from nfl_hybrid.evaluation.week1_reliability import NOT_ESTIMABLE, binary_log_loss, brier, equal_mass_ece

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "real_ats_total_chronological_calibration_proof.json"

MIN_REQUIRED_CALIBRATION_SAMPLE_COUNT = 100
PRIMARY_MARKETS = [MARKET_ATS, MARKET_TOTAL]
N_AUDIT_TABLE_GAMES = 10

FIX3_ARTIFACT_KEYS = (
    "oof_chronological.predictions",
    "oof_chronological.residual_ledger",
    "oof_chronological.manifest",
)

_MARKET_LABELS = {
    MARKET_ATS: {
        "line_column": "home_spread",
        "upper_label": "home_cover",
        "lower_label": "away_cover",
        "raw_registry_key": "chronological_calibration.ats_raw",
        "ledger_registry_key": "chronological_calibration.ats_ledger",
        "manifest_registry_key": "chronological_calibration.ats_manifest",
        "settlement_label": ("home_cover", "away_cover"),
    },
    MARKET_TOTAL: {
        "line_column": "total_line",
        "upper_label": "over",
        "lower_label": "under",
        "raw_registry_key": "chronological_calibration.total_raw",
        "ledger_registry_key": "chronological_calibration.total_ledger",
        "manifest_registry_key": "chronological_calibration.total_manifest",
        "settlement_label": ("over", "under"),
    },
}


# ---------------------------------------------------------------------------
# Item 5: Fix 3 artifact immutability
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_fix3_artifacts() -> dict[str, str]:
    return {key: _sha256_file(resolve(key)) for key in FIX3_ARTIFACT_KEYS}


def _load_persisted_fix3_residual_ledger() -> pd.DataFrame:
    """Load ONLY -- this script never calls generate_oof_predictions or any
    other Fix 3 training path. If this key doesn't resolve, the correct
    behavior is to stop, not to regenerate the expensive OOF run."""
    return pd.read_parquet(resolve("oof_chronological.residual_ledger"))


def _load_fix3_manifest() -> dict[str, object]:
    return json.loads(resolve("oof_chronological.manifest").read_text(encoding="utf-8"))


def _fix3_1_lineage_fields(hashes: dict[str, str], feature_state_hash: str) -> dict[str, str]:
    """Artifact hygiene (Fix 4 v2): explicit, machine-readable Fix 3.1 lineage
    keys -- never rely on a caller having to dig them out of the generic
    fix3_artifact_sha256_before/after dict alone."""
    return {
        "source_fix3_1_feature_state_hash": feature_state_hash,
        "source_fix3_1_oof_predictions_sha256": hashes["oof_chronological.predictions"],
        "source_fix3_1_oof_residual_ledger_sha256": hashes["oof_chronological.residual_ledger"],
        "source_fix3_1_oof_manifest_sha256": hashes["oof_chronological.manifest"],
    }


# ---------------------------------------------------------------------------
# Item 2F: feature-leakage contract -- semantic provenance, not substring scan
# ---------------------------------------------------------------------------

# Traced from actual code, not assumed: nfl_hybrid.features.pregame_rolling.
# build_game_pregame_matrix pivots each state family's OWN (shift-then-rolled)
# team features onto a COPY of the full `games` table (`game_work = games.
# copy()`), then chronological_oof.build_oof_feature_matrix's per-family loop
# takes every column of that pivot starting with "home_"/"away_" -- which
# includes `games`' OWN native columns (re-attached by the merge, never
# shifted), not only the family's genuinely-engineered rolling features. This
# only matters for a games-native basename shared by the merge -- an
# engineered feature like "home_team_prior_games" has no native games column
# of that name and is correctly excluded.
_OUTCOME_BASENAMES = {"score"}
_CURRENT_MARKET_BASENAMES = {"moneyline_reference", "spread_reference", "spread_price_reference"}
_BENIGN_NATIVE_BASENAMES = {"team_id", "team_name", "rest_days", "qb_id", "qb_name", "coach"}


def _feature_leakage_audit(feature_columns: list[str]) -> dict[str, object]:
    games = pd.read_parquet(resolve("backfill.games"))
    native_home_away_columns = sorted(c for c in games.columns if c.startswith("home_") or c.startswith("away_"))
    native_set = set(native_home_away_columns)

    suspicious: list[dict[str, str]] = []
    for col in feature_columns:
        if "__" not in col:
            continue
        _family, rest = col.split("__", 1)
        if not (rest.startswith("home_") or rest.startswith("away_")):
            continue
        side, _sep, basename = rest.partition("_")
        native_col = f"{side}_{basename}"
        if native_col not in native_set:
            continue  # a genuinely engineered rolling feature, not a games-native passthrough
        if basename in _OUTCOME_BASENAMES:
            suspicious.append({"feature": col, "reason": "POSTGAME_OUTCOME_LEAKAGE", "native_source_column": native_col})
        elif basename in _CURRENT_MARKET_BASENAMES:
            suspicious.append(
                {"feature": col, "reason": "CURRENT_GAME_MARKET_LEAKAGE", "native_source_column": native_col}
            )
        elif basename in _BENIGN_NATIVE_BASENAMES:
            continue  # legitimately pregame-available identity/rest metadata (redundantly duplicated per family)
        else:
            suspicious.append(
                {
                    "feature": col,
                    "reason": "UNCLASSIFIED_NATIVE_GAMES_PASSTHROUGH_NEEDS_MANUAL_REVIEW",
                    "native_source_column": native_col,
                }
            )

    return {
        "n_features_examined": len(feature_columns),
        "n_suspicious": len(suspicious),
        "suspicious_features": suspicious,
        "native_games_home_away_columns": native_home_away_columns,
        "method": (
            "semantic provenance trace: games-native home_/away_ columns re-attached by "
            "build_game_pregame_matrix's merge onto `games`, cross-referenced per state family "
            "against that family's actual engineered output columns (not a substring scan -- an "
            "engineered feature sharing a games-native basename would still be correctly excluded "
            "if it were not literally a passthrough of that native column)"
        ),
    }


# ---------------------------------------------------------------------------
# Item 2A/B/C/D/E: settlement/class-balance/distribution/paired/audit-table
# ---------------------------------------------------------------------------


def _settlement_agreement_audit(raw: pd.DataFrame, settlement: pd.DataFrame) -> dict[str, object]:
    """Item 2A: Fix 4's own binary_target/push determination, independently
    cross-checked against nfl_hybrid.features.canonical_market_matrices' own
    settlement targets (target_t10_home_cover/ats_push or
    target_t10_over/total_push) for every row both sides can score."""
    joined = raw.merge(settlement, on="game_id", how="inner")
    joined = joined[joined["raw_status"] == "RAW_READY"]

    fix4_push = pd.to_numeric(joined["binary_target"], errors="coerce").isna()
    canonical_push = joined["canonical_target_push"].astype(int) == 1
    push_mismatches = joined.loc[fix4_push != canonical_push, "game_id"].tolist()

    non_push = joined[~fix4_push & ~canonical_push]
    fix4_upper = pd.to_numeric(non_push["binary_target"], errors="coerce").astype(int)
    canonical_upper = non_push["canonical_target_cover_or_over"].astype(int)
    cover_mismatches = non_push.loc[fix4_upper != canonical_upper, "game_id"].tolist()

    return {
        "n_compared": int(len(joined)),
        "n_push_label_mismatches": len(push_mismatches),
        "push_label_mismatch_game_ids": push_mismatches,
        "n_cover_or_over_label_mismatches": len(cover_mismatches),
        "cover_or_over_label_mismatch_game_ids": cover_mismatches,
        "agreement_rate": 1.0
        - (len(push_mismatches) + len(cover_mismatches)) / max(int(len(joined)), 1),
    }


def _class_balance(settlement_ready: pd.DataFrame, *, upper_label: str, lower_label: str) -> dict[str, object]:
    n = len(settlement_ready)
    push = int((settlement_ready["canonical_target_push"] == 1).sum())
    non_push = settlement_ready[settlement_ready["canonical_target_push"] != 1]
    upper = int((non_push["canonical_target_cover_or_over"] == 1).sum())
    lower = int((non_push["canonical_target_cover_or_over"] == 0).sum())
    return {
        "n": n,
        upper_label: upper,
        lower_label: lower,
        "push": push,
        f"{upper_label}_pct": round(100.0 * upper / n, 2) if n else None,
        f"{lower_label}_pct": round(100.0 * lower / n, 2) if n else None,
        "push_pct": round(100.0 * push / n, 2) if n else None,
    }


def _probability_distribution(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "count_below_0.10": int((v < 0.10).sum()),
        "count_above_0.90": int((v > 0.90).sum()),
    }


def _paired_market_comparison(scored: pd.DataFrame) -> dict[str, object]:
    """Item 2D: SANITY CHECK ONLY -- descriptive, paired, on the exact same
    scored rows, against the timestamp-aligned market consensus novig
    probability already in the canonical market attachment. Never used to
    tune/select anything."""
    y = pd.to_numeric(scored["binary_target"], errors="coerce").to_numpy(float)
    p_market = scored["market_t10_novig_probability"].to_numpy(float)
    p_raw = scored["raw_conditional_upper_probability"].to_numpy(float)
    p_cal = scored["calibrated_conditional_upper_probability"].to_numpy(float)
    return {
        "n_scored": int(len(y)),
        "market": {"log_loss": binary_log_loss(y, p_market), "brier": brier(y, p_market)},
        "model_raw": {"log_loss": binary_log_loss(y, p_raw), "brier": brier(y, p_raw)},
        "model_calibrated": {"log_loss": binary_log_loss(y, p_cal), "brier": brier(y, p_cal)},
    }


def _build_audit_table(
    residual_ledger: pd.DataFrame,
    ats_raw: pd.DataFrame,
    total_raw: pd.DataFrame,
    ats_settlement: pd.DataFrame,
    total_settlement: pd.DataFrame,
    *,
    n: int = N_AUDIT_TABLE_GAMES,
) -> list[dict[str, object]]:
    """Item 2E: a game-level audit table spread across the OOF window,
    selected to include favorite/underdog, integer/non-integer lines, and
    over/under outcomes."""
    ats_ready = ats_raw[ats_raw["raw_status"] == "RAW_READY"][["game_id", "home_spread", "raw_conditional_upper_probability", "binary_target"]]
    total_ready = total_raw[total_raw["raw_status"] == "RAW_READY"][["game_id", "total_line", "raw_conditional_upper_probability", "binary_target"]]
    both = ats_ready.merge(total_ready, on="game_id", suffixes=("_ats", "_total"))
    both = both.merge(
        residual_ledger[["game_id", "predicted_margin", "predicted_total", "margin_residual_sd_oof", "total_residual_sd_oof", "actual_margin", "actual_total"]],
        on="game_id",
        how="left",
    )
    both = both.merge(ats_settlement[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="left")
    both = both.sort_values("game_id").reset_index(drop=True)
    if both.empty:
        return []

    positions = np.linspace(0, len(both) - 1, num=min(n, len(both)), dtype=int)
    selected = both.iloc[positions].drop_duplicates(subset="game_id")

    rows = []
    for r in selected.itertuples(index=False):
        home_score = (r.actual_margin + r.actual_total) / 2.0
        away_score = (r.actual_total - r.actual_margin) / 2.0
        ats_push = pd.isna(r.binary_target_ats)
        total_push = pd.isna(r.binary_target_total)
        rows.append(
            {
                "game_id": r.game_id,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
                "final_home_score": float(home_score),
                "final_away_score": float(away_score),
                "actual_margin": float(r.actual_margin),
                "actual_total": float(r.actual_total),
                "home_spread": float(r.home_spread),
                "total_line": float(r.total_line),
                "predicted_margin": float(r.predicted_margin),
                "predicted_total": float(r.predicted_total),
                "margin_sd_oof": float(r.margin_residual_sd_oof),
                "total_sd_oof": float(r.total_residual_sd_oof),
                "raw_ats_conditional_probability": float(r.raw_conditional_upper_probability_ats),
                "ats_settlement": "push" if ats_push else ("home_cover" if int(r.binary_target_ats) == 1 else "away_cover"),
                "raw_total_conditional_probability": float(r.raw_conditional_upper_probability_total),
                "total_settlement": "push" if total_push else ("over" if int(r.binary_target_total) == 1 else "under"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Independent certification checks (items 13/14, unchanged by push repair)
# ---------------------------------------------------------------------------


def _three_class_target(binary_target: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(binary_target, errors="coerce").to_numpy(dtype=float)
    out = np.ones(len(numeric), dtype=int)
    out[numeric == 1.0] = 2
    out[numeric == 0.0] = 0
    return out


def _independent_certification_checks(
    raw: pd.DataFrame, row: pd.Series, *, market: str
) -> dict[str, bool]:
    target_game_id = str(row["game_id"])
    target_cutoff = pd.Timestamp(row["target_cutoff_utc"])
    calibration_ids = list(row["calibration_prediction_ids"])
    available_by_id = dict(zip(raw["game_id"], pd.to_datetime(raw["result_available_at_utc"], utc=True)))
    books_by_id = dict(zip(raw["game_id"], raw["eligible_books"]))
    returned_by_id = dict(zip(raw["game_id"], pd.to_datetime(raw["market_snapshot_returned_utc"], utc=True)))
    horizon_by_id = dict(zip(raw["game_id"], raw["forecast_horizon"]))
    cutoff_by_id = dict(zip(raw["game_id"], pd.to_datetime(raw["target_cutoff_utc"], utc=True)))

    return {
        "target_not_in_own_calibration_membership": target_game_id not in calibration_ids,
        "sample_count_matches_membership_list_length": len(calibration_ids) == int(row["calibration_sample_count"]),
        "every_member_result_available_before_target_cutoff": all(
            available_by_id[cid] < target_cutoff for cid in calibration_ids
        ),
        "every_member_market_snapshot_returned_before_own_cutoff": all(
            returned_by_id[cid] <= cutoff_by_id[cid] for cid in calibration_ids
        ),
        "every_member_has_at_least_minimum_books": all(
            books_by_id[cid] >= MINIMUM_ELIGIBLE_BOOKS for cid in calibration_ids
        ),
        "every_member_shares_target_forecast_horizon": all(
            horizon_by_id[cid] == row["forecast_horizon"] for cid in calibration_ids
        ),
        "max_calibration_result_available_before_target_cutoff": pd.Timestamp(
            row["max_calibration_result_available_at_utc"]
        )
        < target_cutoff,
        "target_market_matches_expected": market in (MARKET_ATS, MARKET_TOTAL),
    }


def _binary_diagnostics(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray) -> dict[str, object]:
    def _ece(fn_y, fn_p):
        value = equal_mass_ece(fn_y, fn_p)
        return value if value == NOT_ESTIMABLE else float(value)

    return {
        "n_scored": int(len(y)),
        "raw_conditional_log_loss": binary_log_loss(y, p_raw),
        "calibrated_conditional_log_loss": binary_log_loss(y, p_cal),
        "raw_conditional_brier": brier(y, p_raw),
        "calibrated_conditional_brier": brier(y, p_cal),
        "raw_conditional_ece": _ece(y, p_raw),
        "calibrated_conditional_ece": _ece(y, p_cal),
    }


def _three_way_descriptive_scores(
    three_class: np.ndarray, raw_probs: np.ndarray, calibrated_probs: np.ndarray
) -> dict[str, float]:
    eps = 1e-6
    one_hot = np.eye(3)[three_class]

    def log_loss(probs: np.ndarray) -> float:
        p_true = np.clip(probs[np.arange(len(probs)), three_class], eps, 1.0)
        return float(-np.mean(np.log(p_true)))

    def brier3(probs: np.ndarray) -> float:
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "n_scored": int(len(three_class)),
        "raw_three_way_log_loss": log_loss(raw_probs),
        "calibrated_three_way_log_loss": log_loss(calibrated_probs),
        "raw_three_way_brier": brier3(raw_probs),
        "calibrated_three_way_brier": brier3(calibrated_probs),
    }


def _certify_market(
    residual_ledger: pd.DataFrame, settlement: pd.DataFrame, *, market: str
) -> dict[str, object]:
    labels = _MARKET_LABELS[market]
    raw, coverage = build_market_raw_ledger(residual_ledger, market=market)

    n_examined = coverage["n_examined"]
    n_raw_ready = int((raw["raw_status"] == "RAW_READY").sum()) if len(raw) else 0

    if raw.empty or n_raw_ready == 0:
        return {"market": market, "coverage": coverage, "certification_status": "NO_RAW_READY_ROWS"}

    # Item 2A/B: settlement cross-check and class balance -- BEFORE trusting
    # any downstream metric.
    settlement_audit = _settlement_agreement_audit(raw, settlement)
    ready_joined = raw[raw["raw_status"] == "RAW_READY"].merge(settlement, on="game_id", how="inner")
    class_balance = _class_balance(ready_joined, upper_label=labels["upper_label"], lower_label=labels["lower_label"])

    calib_cfg = ChronologicalCalibrationConfig()
    conditional_ledger = generate_chronological_calibration(raw, config=calib_cfg)

    # Item 3: chronological push-mass calibration, on top of the already
    # chronologically-calibrated conditional probability.
    push_cfg = CalibrationConfig()
    ledger = calibrate_chronological_push_probability(
        raw, conditional_ledger, market=market, market_line=raw[labels["line_column"]], config=push_cfg
    )

    candidates = ledger[
        (ledger["calibration_status"] == "CALIBRATED")
        & (ledger["calibration_sample_count"] >= MIN_REQUIRED_CALIBRATION_SAMPLE_COUNT)
        & (~ledger["calibration_prediction_ids_truncated"])
    ]
    if candidates.empty:
        return {
            "market": market,
            "coverage": coverage,
            "settlement_audit": settlement_audit,
            "class_balance": class_balance,
            "raw": raw,
            "ledger": ledger,
            "certification_status": "NO_TARGET_REACHES_MIN_SAMPLE_COUNT",
        }

    row = candidates.sort_values("calibration_sample_count", ascending=False).iloc[0]
    checks = _independent_certification_checks(raw, row, market=market)
    certified = all(checks.values())

    calibrated_rows = ledger[ledger["calibration_status"] == "CALIBRATED"].merge(
        raw[["game_id", "binary_target"]], on="game_id", how="left"
    )
    scored = calibrated_rows[calibrated_rows["binary_target"].notna()].copy()
    n_pushes_excluded = int(calibrated_rows["binary_target"].isna().sum())
    y = pd.to_numeric(scored["binary_target"], errors="coerce").to_numpy(float)
    p_raw_cond = scored["raw_conditional_upper_probability"].to_numpy(float)
    p_cal_cond = scored["calibrated_conditional_upper_probability"].to_numpy(float)
    binary_diag = _binary_diagnostics(y, p_raw_cond, p_cal_cond) if len(y) else None
    if binary_diag is not None:
        binary_diag["n_pushes_excluded"] = n_pushes_excluded

    three_class = _three_class_target(calibrated_rows["binary_target"])
    raw_three = calibrated_rows[["raw_away_probability", "raw_push_probability", "raw_home_probability"]].to_numpy(float)
    cal_three = calibrated_rows[
        ["calibrated_away_probability", "calibrated_push_probability", "calibrated_home_probability"]
    ].to_numpy(float)
    three_way_diag = _three_way_descriptive_scores(three_class, raw_three, cal_three)

    push_diag = {
        "raw_mean_push_probability": float(calibrated_rows["raw_push_probability"].mean()),
        "calibrated_mean_push_probability": float(calibrated_rows["calibrated_push_probability"].mean()),
        "actual_push_rate": float(calibrated_rows["binary_target"].isna().mean()),
        "push_calibration_policy": str(calibrated_rows["push_calibration_policy"].mode().iloc[0]),
    }

    # Item 2D: paired sanity check against the market's own novig probability.
    scored_with_market = scored.merge(settlement[["game_id", "market_t10_novig_probability"]], on="game_id", how="left")
    paired = (
        _paired_market_comparison(scored_with_market)
        if scored_with_market["market_t10_novig_probability"].notna().any()
        else None
    )

    target_game_id = str(row["game_id"])
    line_value = float(raw.loc[raw["game_id"] == target_game_id, labels["line_column"]].iloc[0])
    provenance_row = raw.loc[raw["game_id"] == target_game_id].iloc[0]
    settlement_row = settlement.loc[settlement["game_id"] == target_game_id]

    # Item 6: team/line orientation, auditable without inferring from game_id.
    fix3_row = residual_ledger.loc[residual_ledger["game_id"].astype(str) == target_game_id].iloc[0]
    actual_margin = float(fix3_row["actual_margin"])
    actual_total = float(fix3_row["actual_total"])
    home_score = (actual_margin + actual_total) / 2.0
    away_score = (actual_total - actual_margin) / 2.0

    bookmaker_keys = load_contributing_bookmaker_keys(market, target_game_id)
    bookmaker_keys_sha256 = (
        hashlib.sha256(",".join(bookmaker_keys).encode("utf-8")).hexdigest() if bookmaker_keys else None
    )

    report = {
        "market": market,
        "coverage": coverage,
        "settlement_audit": settlement_audit,
        "class_balance": class_balance,
        "certification_status": "CALIBRATED_CERTIFIED" if certified else "INVARIANT_CHECK_FAILED",
        "independent_checks": checks,
        "target": {
            "game_id": target_game_id,
            "season": int(row["season"]),
            "week": int(row["week"]),
            "forecast_horizon": str(row["forecast_horizon"]),
            "target_cutoff_utc": str(row["target_cutoff_utc"]),
            "home_team_id": str(settlement_row["home_team_id"].iloc[0]) if len(settlement_row) else None,
            "away_team_id": str(settlement_row["away_team_id"].iloc[0]) if len(settlement_row) else None,
            "final_home_score": home_score,
            "final_away_score": away_score,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            labels["line_column"]: line_value,
            "ats_edge_actual_margin_plus_home_spread": (
                actual_margin + line_value if market == MARKET_ATS else None
            ),
            "market_snapshot_requested_utc": str(provenance_row["market_snapshot_requested_utc"]),
            "market_snapshot_returned_utc": str(provenance_row["market_snapshot_returned_utc"]),
            "market_snapshot_lag_minutes": float(provenance_row["market_snapshot_lag_minutes"]),
            "eligible_books": int(provenance_row["eligible_books"]),
            "consensus_method": CONSENSUS_METHOD,
            "market_t10_line_sd": (
                float(settlement_row["market_t10_line_sd"].iloc[0]) if len(settlement_row) else None
            ),
            "market_t10_novig_probability": (
                float(settlement_row["market_t10_novig_probability"].iloc[0]) if len(settlement_row) else None
            ),
            "market_last_update_utc": None,  # not carried by the canonical closing_t10 schema -- never fabricated
            "contributing_bookmaker_keys": list(bookmaker_keys) if bookmaker_keys else None,
            "contributing_bookmaker_keys_sha256": bookmaker_keys_sha256,
            "predicted_margin_or_total": float(fix3_row["predicted_margin" if market == MARKET_ATS else "predicted_total"]),
            "residual_sd_oof": float(fix3_row["margin_residual_sd_oof" if market == MARKET_ATS else "total_residual_sd_oof"]),
            "raw": {
                "upper": float(row["raw_home_probability"]),
                "push": float(row["raw_push_probability"]),
                "lower": float(row["raw_away_probability"]),
                "conditional_upper": float(row["raw_conditional_upper_probability"]),
            },
            "calibrated": {
                "upper": float(row["calibrated_home_probability"]),
                "push": float(row["calibrated_push_probability"]),
                "lower": float(row["calibrated_away_probability"]),
                "conditional_upper": float(row["calibrated_conditional_upper_probability"]),
            },
            "push_calibration_policy": str(row["push_calibration_policy"]),
            "push_calibration_membership_count": int(row["push_calibration_membership_count"]),
            "push_calibration_config_hash": str(row["push_calibration_config_hash"]),
            "calibration_sample_count": int(row["calibration_sample_count"]),
            "calibration_membership_hash": row["calibration_membership_hash"],
            "calibrator_family": row["calibrator_family"],
            "calibrator_config_hash": row["calibrator_config_hash"],
            "max_calibration_result_available_at_utc": str(row["max_calibration_result_available_at_utc"]),
        },
        "upper_label": labels["upper_label"],
        "lower_label": labels["lower_label"],
        "probability_distribution_raw_conditional": _probability_distribution(
            raw.loc[raw["raw_status"] == "RAW_READY", "raw_conditional_upper_probability"].to_numpy(float)
        ),
        "conditional_diagnostics": binary_diag,
        "three_way_descriptive_diagnostics": three_way_diag,
        "push_diagnostics": push_diag,
        "paired_market_comparison_sanity_check_only": paired,
        "n_raw_examined": n_examined,
        "n_raw_ready": n_raw_ready,
        "n_calibrated_rows": int((ledger["calibration_status"] == "CALIBRATED").sum()),
        "n_uncalibrated_insufficient_history": int(
            (ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").sum()
        ),
        "n_raw_unavailable": int((ledger["calibration_status"] == "RAW_UNAVAILABLE").sum()),
        "raw": raw,
        "ledger": ledger,
    }
    return report


def _semantic_rename_map(market: str) -> dict[str, str]:
    upper, lower = _MARKET_LABELS[market]["upper_label"], _MARKET_LABELS[market]["lower_label"]
    return {
        "raw_home_probability": f"raw_{upper}_probability",
        "raw_away_probability": f"raw_{lower}_probability",
        "raw_conditional_upper_probability": f"raw_conditional_{upper}_probability",
        "calibrated_home_probability": f"calibrated_{upper}_probability",
        "calibrated_away_probability": f"calibrated_{lower}_probability",
        "calibrated_conditional_upper_probability": f"calibrated_conditional_{upper}_probability",
    }


def _persist(
    market: str,
    raw: pd.DataFrame,
    ledger: pd.DataFrame,
    report: dict[str, object],
    *,
    fix3_1_lineage: dict[str, str],
) -> dict[str, str]:
    labels = _MARKET_LABELS[market]
    written: dict[str, str] = {}
    rename_map = _semantic_rename_map(market)

    raw_out = raw.rename(columns=rename_map)
    raw_path = resolve_for_write(labels["raw_registry_key"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out.to_parquet(raw_path, index=False)
    written[labels["raw_registry_key"]] = str(raw_path)

    ledger_out = ledger.rename(columns=rename_map)
    ledger_out.insert(0, "market", market)
    ledger_path = resolve_for_write(labels["ledger_registry_key"])
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_out.to_parquet(ledger_path, index=False)
    written[labels["ledger_registry_key"]] = str(ledger_path)

    manifest = {
        "artifact_scope": "FIX4_PROOF",
        "production_evidence": False,
        "full_historical_ledger": False,
        "final_model_frozen": False,
        "season_coverage": sorted(int(s) for s in raw["season"].dropna().unique().tolist()),
        "forecast_horizon": FORECAST_HORIZON,
        "primary_markets": PRIMARY_MARKETS,
        "market": market,
        "purpose": "real_ats_total_chronological_calibration_engine_proof",
        "market_source": "nfl_hybrid.features.odds_attachment / canonical_market_matrices (closing_t10)",
        "consensus_method": CONSENSUS_METHOD,
        "minimum_books": MINIMUM_ELIGIBLE_BOOKS,
        "maximum_snapshot_lag_minutes": MAXIMUM_SNAPSHOT_LAG_MINUTES,
        "push_probability_policy": "CHRONOLOGICAL_FIXED_PUSH_SCALE_FAMILY (three_way._fit_push_scales/_predict_push)",
        "calibrator_family": CALIBRATOR_FAMILY,
        "calibrator_config_hash": compute_calibrator_config_hash(ChronologicalCalibrationConfig()),
        "push_calibration_config_hash": compute_push_calibration_config_hash(CalibrationConfig(), market),
        "coverage": report["coverage"],
        "settlement_audit": report.get("settlement_audit"),
        "class_balance": report.get("class_balance"),
        "n_raw_ready": report["n_raw_ready"],
        "n_calibrated_rows": report["n_calibrated_rows"],
        "certification_status": report["certification_status"],
        **fix3_1_lineage,
    }
    manifest_path = resolve_for_write(labels["manifest_registry_key"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    written[labels["manifest_registry_key"]] = str(manifest_path)
    return written


def main() -> int:
    # Item 20's regression guard -- FIRST, before any pricing/calibration
    # work runs. MONEYLINE-only evidence can never satisfy this script.
    assert_primary_markets_certified(PRIMARY_MARKETS)

    # Item 5: hash the Fix 3 artifacts BEFORE this (read-only) certification
    # run touches anything.
    hashes_before = _hash_fix3_artifacts()

    residual_ledger = _load_persisted_fix3_residual_ledger()
    fix3_manifest = _load_fix3_manifest()
    feature_columns = list(fix3_manifest["feature_columns"])
    leakage_audit = _feature_leakage_audit(feature_columns)

    # Item 4 (Fix 4 v2 artifact hygiene): explicit, machine-readable Fix 3.1
    # lineage keys, from the SAME already-hashed/already-loaded artifacts
    # above -- never a second read or a recomputation.
    fix3_1_lineage = _fix3_1_lineage_fields(hashes_before, str(fix3_manifest["feature_state_hash"]))

    settlement_by_market = {m: load_canonical_t10_settlement_columns(m) for m in PRIMARY_MARKETS}

    reports: dict[str, dict[str, object]] = {}
    persisted_paths: dict[str, str] = {}
    for market in PRIMARY_MARKETS:
        report = _certify_market(residual_ledger, settlement_by_market[market], market=market)
        reports[market] = report
        if "raw" in report and "ledger" in report:
            persisted_paths.update(
                _persist(market, report["raw"], report["ledger"], report, fix3_1_lineage=fix3_1_lineage)
            )

    audit_table = []
    if "raw" in reports.get(MARKET_ATS, {}) and "raw" in reports.get(MARKET_TOTAL, {}):
        audit_table = _build_audit_table(
            residual_ledger,
            reports[MARKET_ATS]["raw"],
            reports[MARKET_TOTAL]["raw"],
            settlement_by_market[MARKET_ATS],
            settlement_by_market[MARKET_TOTAL],
        )

    # Item 5: hash again AFTER -- this script must be provably read-only.
    hashes_after = _hash_fix3_artifacts()
    fix3_oof_mutated_by_certification = hashes_before != hashes_after

    both_calibrated_certified = all(
        reports[m].get("certification_status") == "CALIBRATED_CERTIFIED" for m in PRIMARY_MARKETS
    )
    no_leakage = leakage_audit["n_suspicious"] == 0
    no_settlement_mismatch = all(
        reports[m].get("settlement_audit", {}).get("n_push_label_mismatches", 1) == 0
        and reports[m].get("settlement_audit", {}).get("n_cover_or_over_label_mismatches", 1) == 0
        for m in PRIMARY_MARKETS
    )
    both_certified = (
        both_calibrated_certified
        and no_leakage
        and no_settlement_mismatch
        and not fix3_oof_mutated_by_certification
    )

    payload = {
        "artifact": "real_ats_total_chronological_calibration_proof",
        "artifact_scope": "FIX4_PROOF",
        "production_evidence": False,
        "full_historical_ledger": False,
        "final_model_frozen": False,
        "season_coverage": sorted(int(s) for s in residual_ledger["season"].dropna().unique().tolist()),
        "forecast_horizon": FORECAST_HORIZON,
        "primary_markets": PRIMARY_MARKETS,
        "primary_targets_certified": [
            m for m in PRIMARY_MARKETS if reports[m].get("certification_status") == "CALIBRATED_CERTIFIED"
        ],
        **fix3_1_lineage,
        "purpose": "real_ats_total_chronological_calibration_engine_proof",
        "market_source": "nfl_hybrid.features.odds_attachment / canonical_market_matrices (closing_t10)",
        "consensus_method": CONSENSUS_METHOD,
        "minimum_books": MINIMUM_ELIGIBLE_BOOKS,
        "maximum_snapshot_lag_minutes": MAXIMUM_SNAPSHOT_LAG_MINUTES,
        "calibrator_family": CALIBRATOR_FAMILY,
        "feature_leakage_audit": leakage_audit,
        "audit_table_10_games": audit_table,
        "fix3_artifact_sha256_before": hashes_before,
        "fix3_artifact_sha256_after": hashes_after,
        "fix3_oof_mutated_by_certification": fix3_oof_mutated_by_certification,
        "development_process_incident": (
            "Fix3 proof artifact was unintentionally regenerated once during Fix4 development "
            "(a smoke-test run of scripts/generate_chronological_calibration_membership_proof.py "
            "after an API signature change); the regenerated content was spot-checked (row count, "
            "uncertainty_eligible split, first-5-row values) against a pre-regeneration read and "
            "found consistent, but no full pre-regeneration SHA256 exists, so byte-identity is NOT "
            "claimed. This final certification script never calls generate_oof_predictions and is "
            "hash-verified read-only for its own execution (see fix3_artifact_sha256_before/after)."
        ),
        "both_ats_and_total_certified": both_certified,
        "n_fix3_oof_rows_examined": int(len(residual_ledger)),
        "reports": {
            m: {k: v for k, v in reports[m].items() if k not in ("raw", "ledger")} for m in PRIMARY_MARKETS
        },
        "persisted_artifact_paths": persisted_paths,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for market in PRIMARY_MARKETS:
        r = reports[market]
        print(
            f"{market}: status={r.get('certification_status')} "
            f"n_examined={r['coverage']['n_examined']} "
            f"n_qualifying={r['coverage']['qualifies_valid_t10_line']} "
            f"n_raw_ready={r.get('n_raw_ready')} "
            f"n_calibrated={r.get('n_calibrated_rows')}"
        )
    print(f"feature_leakage_audit.n_suspicious={leakage_audit['n_suspicious']}")
    print(f"fix3_oof_mutated_by_certification={fix3_oof_mutated_by_certification}")
    print(f"both_ats_and_total_certified={both_certified}")
    return 0 if both_certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
