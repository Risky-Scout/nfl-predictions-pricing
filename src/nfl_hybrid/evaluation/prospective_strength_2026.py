"""Prospective 2026 model-strength promotion contract -- frozen rules +
automatic reporter (``PROSPECTIVE_2026_STRENGTH_V1``).

This module changes NOTHING about the certified model. It freezes, before
meaningful 2026 regular-season evidence accumulates, the exact rules that
will decide whether the remaining NFL model-strength statuses can be
promoted, and provides the reporter that applies those rules to the
immutable prospective ledgers.

What the reporter reads (Section 18) -- ONLY immutable records:

  - the immutable forecast ledger
    (``$NFL_MODEL_ARTIFACT_ROOT/production-2026/forecast-ledger/``),
  - the immutable evaluation ledger + attached ``*.result.json`` records
    (``.../production-2026/evaluation-ledger/``),
  - the prospective shadow model-family ledger
    (``.../production-2026/shadow-model-family-ledger/``),
  - the executable-price ledger, if activated
    (``.../production-2026/executable-price-ledger/``),
  - ``config/executable_books_2026.json`` (existence + hash only).

It NEVER reads a retrospective 2020-2025 evaluation artifact to score a
2026 promotion gate. Historical status is shown only as context
(:data:`CURRENT_LABELS`), never as a gate input.

Everything numeric a gate needs -- thresholds, bootstrap seed / reps /
cluster, ECE bin edges, sample-maturity boundaries -- is frozen here and
serialized into ``outputs/prospective_2026_strength_preregistration.json``
with a deterministic self-hash (:func:`build_preregistration_payload`,
:func:`preregistration_hash`). Observing 2026 results may never change any
of them.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from nfl_hybrid.evaluation.week1_reliability import (
    NOT_ESTIMABLE,
    auc_diagnostic,
    binary_log_loss,
    brier,
    calibration_slope_intercept,
)

SCHEMA_VERSION = "PROSPECTIVE_2026_STRENGTH_V1"

# ===========================================================================
# Section 1 -- canonical serialization + deterministic hashing.
# ===========================================================================
_HASH_FIELD = "prospective_2026_strength_preregistration_hash"


def canonical_json(payload: object) -> str:
    """The one canonical serialization for this contract (Section 1).
    ``sort_keys``, compact separators, ASCII-only, NO ``default=str``."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(payload: object) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ===========================================================================
# Section 3 -- sample-maturity firewall (unique completed game_id values).
# ===========================================================================
MATURITY_INSUFFICIENT = "INSUFFICIENT_PROSPECTIVE_SAMPLE"
MATURITY_DESCRIPTIVE = "DESCRIPTIVE_ONLY"
MATURITY_INTERIM = "INTERIM_EVIDENCE"
MATURITY_PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"

MATURITY_BOUNDARIES = {
    "insufficient_max": 63,          # 0-63    -> INSUFFICIENT_PROSPECTIVE_SAMPLE
    "descriptive_max": 127,          # 64-127  -> DESCRIPTIVE_ONLY
    "interim_max": 199,              # 128-199 -> INTERIM_EVIDENCE
    "promotion_eligible_min": 200,   # >=200   -> PROMOTION_ELIGIBLE
}
PROMOTION_ELIGIBLE_MIN_GAMES = MATURITY_BOUNDARIES["promotion_eligible_min"]


def sample_maturity(unique_completed_games: int) -> str:
    n = int(unique_completed_games)
    if n <= MATURITY_BOUNDARIES["insufficient_max"]:
        return MATURITY_INSUFFICIENT
    if n <= MATURITY_BOUNDARIES["descriptive_max"]:
        return MATURITY_DESCRIPTIVE
    if n <= MATURITY_BOUNDARIES["interim_max"]:
        return MATURITY_INTERIM
    return MATURITY_PROMOTION_ELIGIBLE


def _promotion_eligible(unique_completed_games: int) -> bool:
    return int(unique_completed_games) >= PROMOTION_ELIGIBLE_MIN_GAMES


# ===========================================================================
# Section 4 -- common inference method (game-cluster bootstrap). Frozen; the
# seed / reps / cluster / interval method may never change after observing
# 2026 results.
# ===========================================================================
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_CLUSTER = "game_id"
BOOTSTRAP_INTERVAL = "percentile_95"
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


def game_cluster_bootstrap(
    per_row_values: Sequence[float],
    game_ids: Sequence[str],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Cluster (game_id) bootstrap of ``statistic`` over ``per_row_values``.

    Unique games are resampled with replacement; every row belonging to a
    drawn game is carried together (Section 4). Returns the point estimate
    and a 95% percentile confidence interval. Deterministic for a fixed
    ``seed`` -- no RNG state escapes.
    """
    values = np.asarray(per_row_values, dtype=float)
    gids = np.asarray([str(g) for g in game_ids])
    if values.shape != gids.shape:
        raise ValueError("per_row_values and game_ids must be the same length")
    n = len(values)
    if n == 0:
        return {"n": 0, "point_estimate": NOT_ESTIMABLE, "ci_low": NOT_ESTIMABLE, "ci_high": NOT_ESTIMABLE,
                "reps": reps, "seed": seed}

    unique_games = np.unique(gids)
    rows_by_game = {g: np.where(gids == g)[0] for g in unique_games}
    point = float(statistic(values))

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    n_games = len(unique_games)
    for b in range(reps):
        drawn = rng.integers(0, n_games, size=n_games)
        idx = np.concatenate([rows_by_game[unique_games[d]] for d in drawn])
        boot[b] = statistic(values[idx])
    return {
        "n": int(n),
        "n_clusters": int(n_games),
        "point_estimate": point,
        "ci_low": float(np.percentile(boot, _CI_LOW_PCT)),
        "ci_high": float(np.percentile(boot, _CI_HIGH_PCT)),
        "reps": reps,
        "seed": seed,
    }


# ===========================================================================
# Section 5 -- fixed-bin ECE. 10 fixed equal-width bins; probability exactly
# 1.0 belongs to the final bin; no adaptive bins.
# ===========================================================================
ECE_BIN_EDGES: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))  # 0.0..1.0
ECE_N_BINS = 10


def fixed_bin_ece(y: Sequence[float], p: Sequence[float]) -> float | str:
    """Expected calibration error over 10 fixed equal-width bins
    ``[0.0,0.1) , [0.1,0.2) , ... , [0.9,1.0]``. ``p == 1.0`` is assigned to
    the final bin. Returns ``NOT_ESTIMABLE`` for an empty sample."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return NOT_ESTIMABLE
    # np.digitize with the interior edges: bin index in 0..9, then clip the
    # p==1.0 (index 10) case back into the final bin.
    idx = np.clip(np.digitize(p, ECE_BIN_EDGES[1:-1], right=False), 0, ECE_N_BINS - 1)
    ece = 0.0
    for b in range(ECE_N_BINS):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def sharpness(p: Sequence[float]) -> float | str:
    """Descriptive only (Sections 5, 7): standard deviation of the
    predicted probabilities."""
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return NOT_ESTIMABLE
    return float(p.std(ddof=0))


LN2 = math.log(2.0)  # 0.6931471805599453 -- the coin-flip reference (Section 7)

# ===========================================================================
# Section 6/7/8/9/11/15 -- frozen promotion thresholds.
# ===========================================================================
CALIBRATION_GATE = {
    "requires_all_four_streams_calibrated_lower_logloss": True,
    "requires_all_four_streams_calibrated_lower_brier": True,
    "requires_all_four_streams_calibrated_lower_ece": True,
    "all_four_pooled_logloss_delta_ci_upper_below": 0.0,
    "all_four_pooled_brier_delta_ci_upper_below": 0.0,
    "no_stream_adverse_logloss_point_estimate": True,
    "requires_promotion_eligible_maturity": True,
}
CALIBRATION_STREAM_EVIDENCE_LABELS = ("STRONG", "SUGGESTIVE", "MIXED", "INSUFFICIENT")

# Contract hardening item 5 -- the exact, frozen ATS/TOTAL calibration
# evidence label definitions. Applied to each market (ATS or TOTAL)
# separately over its unique completed eligible games and its two
# constituent horizon streams (TUE, FRI).
CALIBRATION_EVIDENCE_MIN_UNIQUE_GAMES = 64
CALIBRATION_EVIDENCE_STRONG_MIN_UNIQUE_GAMES = 200
CALIBRATION_STREAM_EVIDENCE_LABEL_DEFINITIONS = {
    "INSUFFICIENT": "fewer than 64 unique completed eligible games",
    "STRONG": (
        ">=200 unique completed games AND both constituent horizon streams improve raw on "
        "LL, Brier, and ECE AND pooled calibrated-minus-raw LL and Brier 95% CI upper bounds "
        "are <0 AND neither horizon has an adverse LL point estimate"
    ),
    "SUGGESTIVE": (
        ">=64 games, not STRONG, pooled calibrated-minus-raw LL <0 and Brier <0, and neither "
        "constituent horizon has an adverse LL point estimate"
    ),
    "MIXED": ">=64 games and neither STRONG nor SUGGESTIVE",
}

# Contract hardening item 6 -- the frozen definition of "materially
# inconsistent" for the ABSOLUTE_PROBABILITY_QUALITY gate. ATS-pooled and
# TOTAL-pooled are materially inconsistent with an ALL_FOUR MET_STRONGLY
# conclusion if EITHER market has ANY of these. Both markets must avoid all
# four for ALL_FOUR absolute quality to become MET_STRONGLY.
MATERIALLY_INCONSISTENT_IF_EITHER_MARKET = {
    "log_loss_at_or_above": LN2,           # 0.6931471805599453
    "brier_at_or_above": 0.250,
    "auc_at_or_below": 0.50,
    "ece_above": 0.075,
}

ABSOLUTE_PROBABILITY_QUALITY_GATE = {
    "pooled_log_loss_below": LN2,
    "pooled_log_loss_ci_upper_below": LN2,
    "pooled_brier_below": 0.250,
    "pooled_brier_ci_upper_below": 0.250,
    "pooled_ece_max": 0.050,
    "calibration_slope_range": [0.80, 1.20],
    "abs_calibration_intercept_max": 0.05,
    "auc_ci_lower_above": 0.50,
    "ats_and_total_not_materially_inconsistent": True,
    "materially_inconsistent_if_either_market": dict(MATERIALLY_INCONSISTENT_IF_EITHER_MARKET),
    "requires_promotion_eligible_maturity": True,
    "sharpness_is_descriptive_only": True,
}

POINT_FORECASTING_GATE = {
    "pooled_margin_rmse_beats_book_by_at_least": 0.25,
    "pooled_margin_squared_error_delta_ci_upper_below": 0.0,
    "pooled_total_rmse_beats_book_by_at_least": 0.25,
    "pooled_total_squared_error_delta_ci_upper_below": 0.0,
    "per_horizon_max_rmse_worse_than_book": 0.10,
    "requires_promotion_eligible_maturity": True,
}

SPORTSBOOK_PROBABILITY_EDGE_GATE = {
    "per_market_logloss_delta_max": -0.002,
    "per_market_logloss_delta_ci_upper_below": 0.0,
    "per_market_brier_delta_max": -0.001,
    "per_market_brier_delta_ci_upper_below": 0.0,
    "constituent_stream_logloss_delta_adverse_above": 0.002,
    "constituent_stream_brier_delta_adverse_above": 0.001,
    "no_ats_total_pooling_to_rescue": True,
    "requires_promotion_eligible_maturity": True,
}

BETTING_RULE_2026_V1 = {
    "markets": ["ATS", "TOTAL"],
    "horizons": ["TUE", "FRI"],
    "stake": 1.0,
    "stake_unit": "flat",
    "minimum_expected_value": 0.025,
    "max_wagers_per_game_horizon": {"ATS": 1, "TOTAL": 1},
    "push_pl_units": 0.0,
    "both_sides_qualify_behavior": "ABSTAIN",
    "both_sides_qualify_flag": "BETTING_INTEGRITY_CONFLICT",
    "no_kelly": True,
    "no_threshold_optimization": True,
    "no_chasing": True,
    "no_manual_overrides": True,
    "no_best_historical_cutoff": True,
    "no_strategy_switching": True,
    "no_post_cutoff_line_shopping": True,
}
BETTING_RULE_STATUS_NOT_ACTIVATED = "NOT_ACTIVATED_FOR_PROFITABILITY"
# Contract hardening item 8 -- deterministic self-hash of the frozen betting
# rule. The executable-book policy lock pins this exact value for the season.
BETTING_RULE_2026_V1_HASH = _sha256_hex(BETTING_RULE_2026_V1)

# Contract hardening item 7 -- the ONE frozen definition of a market-specific
# "statistically non-adverse" profitability result. A market with negative
# realized net units is never non-adverse.
PROFITABILITY_MARKET_NON_ADVERSE_RULE = {
    "roi_cluster_bootstrap_95_ci_upper_at_least": 0.0,
    "realized_net_units_at_least": 0.0,
    "negative_realized_net_units_is_adverse": True,
}

PROFITABILITY_GATE = {
    "executable_book_policy_frozen_before_first_wager": True,
    "all_wagers_used_forecast_time_executable_offers": True,
    "betting_rule_unchanged": True,
    "min_settled_non_push_wagers": 150,
    "min_unique_completed_prospective_games": 200,
    "min_realized_roi_pct": 2.0,
    "roi_cluster_bootstrap_ci_lower_above": 0.0,
    "net_units_above": 0.0,
    "median_closing_line_value_above_when_valid": 0.0,
    "ats_and_total_separately_positive_or_one_positive_other_non_adverse": True,
    "market_specific_non_adverse_rule": dict(PROFITABILITY_MARKET_NON_ADVERSE_RULE),
    "max_single_game_share_of_net_profit": 0.10,
    "no_integrity_violation": True,
    "any_persisted_betting_integrity_conflict_permanently_bars_met_strongly": True,
}

# Contract hardening item 4 -- the frozen replacement for the undefined
# phrase "adequately-sized scope". A TUE, FRI, pooled, first-third,
# middle-third, or final-third family scope is eligible for one-SE /
# selection conclusions only with at least this many unique completed
# game_id values in that scope. Anything smaller is INSUFFICIENT.
ADEQUATE_MODEL_FAMILY_SCOPE_MIN_UNIQUE_GAMES = 64
MODEL_FAMILY_SCOPES = ("TUE", "FRI", "pooled", "first_third", "middle_third", "final_third")

MODEL_FAMILY_STABILITY_GATE = {
    "ridge_family_selected_tue": True,
    "ridge_family_selected_fri": True,
    "ridge_family_selected_pooled": True,
    "ridge_in_one_se_set_every_adequate_scope": True,
    "hgbr_selected_in_no_adequate_scope": True,
    "ridge_alpha_100_selected_tue_fri_pooled": True,
    "bootstrap_p_ridge_family_selected_min": 0.80,
    "bootstrap_p_ridge_in_one_se_set_min": 0.90,
    "bootstrap_p_hgbr_selected_max": 0.05,
    "adequate_scope_min_unique_completed_games": ADEQUATE_MODEL_FAMILY_SCOPE_MIN_UNIQUE_GAMES,
    "adequate_scopes": list(MODEL_FAMILY_SCOPES),
    "smaller_scope_conclusion": "INSUFFICIENT",
    "requires_promotion_eligible_maturity": True,
}
FROZEN_FIX7_SHADOW_CANDIDATES = (
    "RIDGE_ALPHA_0_1",
    "RIDGE_ALPHA_1",
    "RIDGE_ALPHA_10",
    "RIDGE_ALPHA_100",
    "HUBER_FIXED",
    "HGBR_INCUMBENT",
)
SHADOW_SEASON_SEGMENTS = ("first_third", "middle_third", "final_third")

CHRONOLOGY_GATE = {
    "future_data_violations_max": 0,
    "cutoff_violations_max": 0,
    "outcome_contamination_max": 0,
    "mutable_forecast_violations_max": 0,
}
REPRODUCIBILITY_GATE = {
    "eligible_forecast_records_with_full_provenance_pct_min": 100.0,
    "unexplained_provenance_gaps_max": 0,
    "all_scorecard_inputs_traceable_to_immutable_records": True,
}
PRODUCTION_PIPELINE_GATE = {
    "scheduled_eligible_batches_success_or_documented_fail_closed_pct_min": 99.0,
    "silent_fallback_max": 0,
    "overwritten_forecasts_max": 0,
    "fabricated_market_states_max": 0,
    "numeric_calibrated_probability_when_not_calibrated_max": 0,
    "credential_leakage_max": 0,
}

# Provenance every genuinely-prospective forecast record must carry (Section 2).
# Contract hardening item 2 adds ``season`` + ``season_type`` -- the minimum
# operational provenance needed to freeze the 2026 REG+POST population and
# exclude preseason. A record that cannot prove its population membership is
# excluded (fail closed), never silently scored.
REQUIRED_FORECAST_PROVENANCE = (
    "game_id",
    "horizon",
    "season",
    "season_type",
    "target_cutoff_utc",
    "forecast_created_at_utc",
    "git_commit",
    "certified_baseline_sha",
    "operational_model_spec_hash",
    "feature_state_semantics_hash",
    "prediction_hash",
)
MARKET_RELATIVE_EXTRA_PROVENANCE = ("market_state_hash",)

# Contract hardening item 2 -- frozen prospective population. Prospective
# strength evidence is limited to season 2026, REG + POST only. PRESEASON
# never enters sample maturity, calibration scoring, sportsbook comparisons,
# model-family stability, profitability, or season-final completion.
PROSPECTIVE_SEASON = 2026
PROSPECTIVE_SEASON_TYPES = ("REG", "POST")
PRESEASON_SEASON_TYPE = "PRE"
PRESEASON_EXCLUDED_FROM = (
    "sample_maturity",
    "calibration_scoring",
    "sportsbook_comparisons",
    "model_family_stability",
    "profitability",
    "season_final_completion",
)

# Contract hardening item 3 (+ blocker-remediation) -- deterministic
# market-state provenance hash. SHA-256 over :func:`canonical_json` of the
# EXACT persisted ``{"ATS": <consensus market dict|null>, "TOTAL": <consensus
# market dict|null>}`` payload used for a forecast/horizon. The producer
# persists this explicit hash AT FORECAST TIME in both the forecast-ledger
# prediction payload and the evaluation-ledger record; the reporter
# reconstructs the payload from the immutable record alone, recomputes the
# hash, and requires exact equality before a row is used for point
# forecasting vs sportsbook, sportsbook probability edge, or betting
# evidence. There is NO derive-and-accept fallback: a market-relative row
# with no explicit persisted hash is ineligible. It never requires access to
# a later mutable raw market file.
MARKET_STATE_HASH_ALGORITHM = "SHA-256"
MARKET_STATE_HASH_RECOMPUTE_VERIFIED_BEFORE = (
    "POINT_FORECASTING_VS_SPORTSBOOK",
    "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE",
    "PROVEN_PROFITABLE_BETTING_MODEL",
)
# The frozen field order of a single ATS/TOTAL consensus market dict inside
# the hashed payload (documented; the hash itself is order-independent via
# canonical_json sort_keys).
MARKET_STATE_PAYLOAD_MARKETS = ("ATS", "TOTAL")
MARKET_STATE_PAYLOAD_MARKET_DICT_FIELDS = (
    "eligible_books",
    "consensus_line",
    "consensus_novig_probability",
    "bookmaker_keys",
    "selected_returned_snapshot_timestamps",
    "min_observation_age_hours",
    "max_observation_age_hours",
    "consensus_method",
)
MARKET_STATE_PAYLOAD_EXCLUDES = (
    "created_at_utc",
    "run_id",
    "git_commit",
    "prediction_probabilities",
    "result_or_outcome",
    "later_mutable_market_evidence",
)
MARKET_STATE_HASH_FORECAST_LEDGER_JSON_PATH = "prediction.market_state_hash"
MARKET_STATE_HASH_EVALUATION_LEDGER_JSON_PATH = "market_state_hash"

INTEGRITY_EXCLUSION_REASONS = (
    "FORECAST_IMMUTABILITY_VIOLATION",
    "SCHEMA_DRIFT",
    "IDENTIFIER_FAILURE",
    "HASH_MISMATCH",
    "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE",
    "FORECAST_OUTSIDE_EXECUTION_WINDOW",
    "OUT_OF_PROSPECTIVE_POPULATION",
)

# Contract hardening item 8 -- executable-book policy lock.
BETTING_POLICY_SUBDIR = "betting-policy"
EXECUTABLE_BOOK_POLICY_LOCK_FILENAME = "executable_book_policy_lock.json"
EXECUTABLE_BOOK_POLICY_LOCK_SCHEMA = "EXECUTABLE_BOOK_POLICY_LOCK_V1"
BETTING_INTEGRITY_CONFLICT = "BETTING_INTEGRITY_CONFLICT"

# Blocker-remediation item 3 -- append-only betting-policy integrity-event
# ledger. A conflicting wager attempt is durably recorded BEFORE the
# hard-stop is raised, so contamination is permanent and auditable across
# processes even if the lock file / config is later reverted. There is NO
# automatic clear-contamination path.
INTEGRITY_EVENTS_SUBDIR = "integrity-events"
BETTING_POLICY_INTEGRITY_EVENT_SCHEMA = "BETTING_POLICY_INTEGRITY_EVENT_V1"
CONFLICT_POLICY_HASH_MISMATCH = "POLICY_HASH_MISMATCH"
CONFLICT_BETTING_RULE_HASH_MISMATCH = "BETTING_RULE_HASH_MISMATCH"
BETTING_INTEGRITY_CONFLICT_EVENT_TYPES = (
    CONFLICT_POLICY_HASH_MISMATCH,
    CONFLICT_BETTING_RULE_HASH_MISMATCH,
)

# ===========================================================================
# Section 23 -- current evidence-supported labels. Historical/context only;
# NEVER a gate input. The reporter preserves these until prospective
# evidence legitimately promotes (or demotes) a row.
# ===========================================================================
CURRENT_LABELS = {
    "CHRONOLOGY_LEAKAGE_CONTROL": "MET_STRONGLY",
    "REPRODUCIBILITY_AUDITABILITY": "MET_STRONGLY",
    "MODEL_FAMILY_STABILITY": "MIXED",
    "CALIBRATION_IMPROVES_RAW_PROBABILITIES": "MET_STRONGLY",
    "ABSOLUTE_PROBABILITY_QUALITY": "MODERATE",
    "POINT_FORECASTING_VS_SPORTSBOOK": "NOT_MET",
    "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE": "NOT_DEMONSTRATED",
    "PRODUCTION_READY_PIPELINE": "MET_STRONGLY",
    "PROVEN_PROFITABLE_BETTING_MODEL": "NOT_ESTABLISHED",
}
CALIBRATION_HISTORICAL_NOTE = (
    "historical evidence; prospective confirmation pending"
)


# ===========================================================================
# Section 1 -- the frozen preregistration payload + self-hash.
# ===========================================================================
def build_preregistration_payload() -> dict:
    """The complete frozen contract. Serialized to
    ``outputs/prospective_2026_strength_preregistration.json`` with
    :func:`preregistration_hash` stored under
    ``prospective_2026_strength_preregistration_hash``.

    Returns a deep copy so a caller mutating the returned structure (e.g. a
    hash-sensitivity test) can never corrupt the module-level frozen
    constants it is assembled from."""
    return copy.deepcopy({
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Freeze, before meaningful 2026 regular-season evidence accumulates, "
            "the exact rules that determine whether the remaining NFL model-strength "
            "statuses can be promoted. Does not change the certified model."
        ),
        "immutable_scientific_baseline": {
            "tag": "v2026.1-fix8-certified",
            "commit": "d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7",
        },
        "immutable_review_ready_snapshot": {
            "tag": "v2026.1-review-ready",
            "commit": "a26bcb4b1ee6d727df9b6b884ee4fa1933006f62",
        },
        "implementation_base": {
            "branch": "cert/prospective-2026-strength-preregistration",
            "base_commit": "64ce7dbf5b12183145e6f96f0efdbca1279f3553",
            "content_addressed_feature_cache_commit": "41f7263fcf65e9cec73a4e8a563c0832c6618fb3",
            "calibration_fail_closed_commit": "8265e45",
        },
        "certified_production_model": {
            "feature_family": "ELO_ONLY",
            "estimator": "RIDGE_ALPHA_100",
            "features": [
                "home_elo_pregame_rating",
                "home_elo_pregame_win_probability",
                "home_elo_pregame_expected_margin",
                "away_elo_pregame_rating",
                "away_elo_pregame_win_probability",
                "away_elo_pregame_expected_margin",
            ],
            "horizon_state": "TUE/FRI card-scoped horizon-as-of state",
            "uncertainty": "Fix-8 same-horizon uncertainty",
            "probability_calibration": "Fix-8 ATS/TOTAL calibration",
            "no_feature_selection": True,
            "no_model_selection": True,
            "no_calibration_tuning": True,
            "no_betting_threshold_tuning": True,
            "no_retrospective_optimization": True,
        },
        "evidence_eligibility": {
            "prospective_only": True,
            # Contract hardening item 1: ``target_cutoff_utc`` is the
            # information as-of cutoff, NOT a requirement that the completed
            # forecast file itself be written before the cutoff.
            "target_cutoff_utc_is_information_as_of_cutoff": True,
            "information_limited_to_target_cutoff_utc": True,
            "every_source_observation_satisfies_certified_as_of_cutoff_rules": True,
            "market_observations_satisfy_certified_timestamp_rules_vs_target_cutoff": True,
            "forecast_generation_corresponds_to_certified_tue_fri_cutoff": True,
            "forecast_generation_within_certified_production_execution_window": True,
            "forecast_created_at_utc_may_be_at_or_after_target_cutoff_utc": True,
            "forecast_created_outside_permitted_execution_window_is_ineligible": True,
            "execution_window_semantics_source": (
                "nfl_hybrid.production.run_2026.is_within_due_window / current_or_recent_cutoff "
                "(the existing certified DST-aware due-run window; no second clock)"
            ),
            "required_forecast_provenance": list(REQUIRED_FORECAST_PROVENANCE),
            "market_relative_extra_provenance": list(MARKET_RELATIVE_EXTRA_PROVENANCE),
            "provenance_acceptance": {
                "certified_baseline_sha": (
                    "an explicit certified baseline SHA, OR -- equivalently -- the certified "
                    "operational_model_spec_hash + feature_state (horizon) semantics_hash pair, "
                    "which immutably pin the same certified commit (this is what the immutable "
                    "evaluation ledger stores)"
                ),
                "market_state_hash": (
                    "an explicit market_state_hash (legacy name: market_snapshot_hash) persisted "
                    "AT FORECAST TIME in BOTH the forecast-ledger prediction payload "
                    "(prediction.market_state_hash) and the evaluation-ledger record "
                    "(market_state_hash); the reporter reconstructs the exact {ATS, TOTAL} "
                    "consensus market payload from the immutable record alone, recomputes the "
                    "SHA-256, and requires EXACT equality. A missing, malformed, or mismatching "
                    "hash makes the row ineligible for every market-relative metric "
                    "(MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE / HASH_MISMATCH). There is no "
                    "derive-and-accept fallback."
                ),
            },
            "outcome_attached_via_immutable_result_attachment_path": True,
            "original_forecast_payload_never_rewritten": True,
            "integrity_exclusion_reasons": list(INTEGRITY_EXCLUSION_REASONS),
            "never_exclude_a_row_for_a_poor_prediction": True,
        },
        "prospective_population": {
            "season": PROSPECTIVE_SEASON,
            "season_types": list(PROSPECTIVE_SEASON_TYPES),
            "preseason_season_type": PRESEASON_SEASON_TYPE,
            "preseason_excluded_from": list(PRESEASON_EXCLUDED_FROM),
            "record_without_provable_population_membership_excluded": True,
            "out_of_population_exclusion_reason": "OUT_OF_PROSPECTIVE_POPULATION",
        },
        "market_state_provenance": {
            "hash_name": "market_state_hash",
            "legacy_hash_name": "market_snapshot_hash",
            "algorithm": MARKET_STATE_HASH_ALGORITHM,
            "canonical_bytes_over": (
                "canonical_json (sort_keys, compact, ASCII) of the EXACT persisted payload "
                '{"ATS": <persisted ATS consensus market dict or null>, '
                '"TOTAL": <persisted TOTAL consensus market dict or null>}, reconstructed from '
                "the immutable forecast/evaluation record alone"
            ),
            "payload_markets": list(MARKET_STATE_PAYLOAD_MARKETS),
            "payload_market_dict_fields": list(MARKET_STATE_PAYLOAD_MARKET_DICT_FIELDS),
            "payload_excludes": list(MARKET_STATE_PAYLOAD_EXCLUDES),
            "recompute_verified_before": list(MARKET_STATE_HASH_RECOMPUTE_VERIFIED_BEFORE),
            "mismatch_status": "HASH_MISMATCH",
            "no_later_mutable_raw_market_file_required_to_verify": True,
            "persisted_explicit_hash_required_at_forecast_time": True,
            "forecast_ledger_json_path": MARKET_STATE_HASH_FORECAST_LEDGER_JSON_PATH,
            "evaluation_ledger_json_path": MARKET_STATE_HASH_EVALUATION_LEDGER_JSON_PATH,
            "forecast_and_evaluation_ledger_hashes_must_agree": True,
            "reporter_reconstructs_payload_from_immutable_record_and_requires_exact_equality": True,
            "missing_explicit_hash_makes_market_relative_evidence_ineligible": True,
            "no_derive_and_accept_fallback": True,
        },
        "sample_maturity_firewall": {
            "unit": "unique_completed_game_id",
            "population": "season 2026, season_type in {REG, POST}; PRESEASON excluded",
            "boundaries": dict(MATURITY_BOUNDARIES),
            "labels": {
                "0_63": MATURITY_INSUFFICIENT,
                "64_127": MATURITY_DESCRIPTIVE,
                "128_199": MATURITY_INTERIM,
                "200_plus": MATURITY_PROMOTION_ELIGIBLE,
            },
            "season_final_confirmatory_status": "SEASON_FINAL_CONFIRMATORY",
            "season_final_confirmatory_rule": {
                "requires_canonical_2026_reg_post_schedule_population_available": True,
                "requires_every_eligible_reg_post_scheduled_game_has_final_result": True,
                "requires_no_remaining_reg_post_game_scheduled_or_unresolved": True,
                "never_inferred_from_calendar_date_or_sample_count_alone": True,
                "false_when_schedule_completeness_cannot_be_proven": True,
            },
            "no_edge_or_profitability_status_met_strongly_before_games": PROMOTION_ELIGIBLE_MIN_GAMES,
        },
        "common_inference_method": {
            "bootstrap_resamples": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "cluster": BOOTSTRAP_CLUSTER,
            "resample": "unique games with replacement; carry all associated TUE/FRI and ATS/TOTAL rows together",
            "interval": BOOTSTRAP_INTERVAL,
            "immutable_after_observing_2026": True,
        },
        "calibration_metrics": {
            "streams": ["ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI"],
            "push_semantics": "existing Fix-8 event/push semantics, exactly",
            "scored_rows": "non-push conditional-event rows only",
            "metrics": ["log_loss", "brier", "ece", "auc", "sharpness"],
            "ece_bin_edges": list(ECE_BIN_EDGES),
            "ece_n_bins": ECE_N_BINS,
            "ece_probability_one_in_final_bin": True,
            "ece_adaptive_bins": False,
            "calibration_intercept_slope": "reported diagnostically; no calibration refit for evaluation",
            "calibrators": "exact frozen production calibrators that generated the prospective probabilities",
            "calibration_not_ready_rows_excluded_from_calibrated_scoring": True,
            "fail_closed_no_numeric_calibrated_value_when_not_ready": True,
            "calibrated_gates_score_calibrated_rows_only": True,
            "no_raw_probability_fallback_in_calibrated_or_probability_gates": True,
            "absolute_probability_quality_scores_calibrated_rows_only": True,
            "sportsbook_probability_edge_scores_calibrated_rows_only": True,
            "point_forecast_metrics_still_use_the_point_forecast_not_a_probability": True,
            "calibrated_metric_sample_size_reported_separately_from_game_maturity": True,
            "numeric_calibrated_value_with_non_calibrated_status_is_a_production_integrity_violation": True,
        },
        "gates": {
            "CALIBRATION_IMPROVES_RAW_PROBABILITIES": CALIBRATION_GATE,
            "ABSOLUTE_PROBABILITY_QUALITY": ABSOLUTE_PROBABILITY_QUALITY_GATE,
            "POINT_FORECASTING_VS_SPORTSBOOK": POINT_FORECASTING_GATE,
            "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE": SPORTSBOOK_PROBABILITY_EDGE_GATE,
            "MODEL_FAMILY_STABILITY": MODEL_FAMILY_STABILITY_GATE,
            "PROVEN_PROFITABLE_BETTING_MODEL": PROFITABILITY_GATE,
            "CHRONOLOGY_LEAKAGE_CONTROL": CHRONOLOGY_GATE,
            "REPRODUCIBILITY_AUDITABILITY": REPRODUCIBILITY_GATE,
            "PRODUCTION_READY_PIPELINE": PRODUCTION_PIPELINE_GATE,
        },
        "calibration_stream_evidence_labels": list(CALIBRATION_STREAM_EVIDENCE_LABELS),
        "calibration_stream_evidence_label_definitions": dict(CALIBRATION_STREAM_EVIDENCE_LABEL_DEFINITIONS),
        "calibration_stream_evidence_min_unique_games": CALIBRATION_EVIDENCE_MIN_UNIQUE_GAMES,
        "calibration_stream_evidence_strong_min_unique_games": CALIBRATION_EVIDENCE_STRONG_MIN_UNIQUE_GAMES,
        "model_family_scope_adequacy": {
            "min_unique_completed_game_ids": ADEQUATE_MODEL_FAMILY_SCOPE_MIN_UNIQUE_GAMES,
            "scopes": list(MODEL_FAMILY_SCOPES),
            "below_threshold_conclusion": "INSUFFICIENT",
            "replaces_undefined_phrase": "adequately-sized scope",
        },
        "coin_flip_reference_log_loss": LN2,
        "shadow_model_family_ledger": {
            "purpose": "separate shadow-evidence path; canonical production forecast stays RIDGE_ALPHA_100 and never changes because of shadow output",
            "candidates": list(FROZEN_FIX7_SHADOW_CANDIDATES),
            "no_new_models": True,
            "features": "same six certified Elo features",
            "targets": "same margin / total targets",
            "training_rows": "only rows legally available to each TUE/FRI cutoff",
            "preprocessing": "same train-only preprocessing rules",
            "path": "$NFL_MODEL_ARTIFACT_ROOT/production-2026/shadow-model-family-ledger/",
            "identity": ["game_id", "horizon", "candidate", "target_cutoff_utc"],
            "write_semantics": "immutable first-write-wins",
            "shadow_failure_cannot_affect_production_forecast": True,
            "season_segments": list(SHADOW_SEASON_SEGMENTS),
            "family_complexity_selection_logic": "frozen Fix-7 logic (nfl_hybrid.selection.model_family_selection_2026)",
        },
        "betting_evidence_firewall": {
            "no_profitability_claim_from_consensus_lines_alone": True,
            "wager_requires_forecast_time_executable_offer": [
                "bookmaker_identifier", "market", "side", "line",
                "odds_american_or_decimal", "quote_timestamp", "snapshot_timestamp",
            ],
            "no_later_price_substitution": True,
            "no_closing_line_substitution": True,
            "no_synthetic_minus_110": True,
            "no_best_book_hindsight": True,
        },
        "betting_rule_2026_v1": BETTING_RULE_2026_V1,
        "betting_rule_2026_v1_hash": BETTING_RULE_2026_V1_HASH,
        "betting_rule_status": BETTING_RULE_STATUS_NOT_ACTIVATED,
        "profitability_market_non_adverse_rule": dict(PROFITABILITY_MARKET_NON_ADVERSE_RULE),
        "executable_book_configuration": {
            "config_path": "config/executable_books_2026.json",
            "absent_behavior": {
                "profitability_status": "NOT_ESTABLISHED",
                "reason": "EXECUTABLE_BOOK_POLICY_NOT_FROZEN",
            },
            "when_supplied": "fixed ordered list of eligible books OR another deterministic predeclared selection rule",
            "hash_locked_on_first_eligible_live_wager": True,
            "immutable_after_activation": True,
        },
        "executable_book_policy_lock": {
            "path": (
                "$NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/"
                "executable_book_policy_lock.json"
            ),
            "schema": EXECUTABLE_BOOK_POLICY_LOCK_SCHEMA,
            "written_only_on_first_genuinely_eligible_executable_wager": True,
            "first_write_wins": True,
            "contains_at_minimum": [
                "policy_hash",
                "betting_rule_2026_v1_hash",
                "first_eligible_wager_identity",
                "lock_timestamp_utc",
                "git_commit",
            ],
            "conflict_flag": BETTING_INTEGRITY_CONFLICT,
            "conflict_when_different_policy_hash_or_betting_rule_hash": True,
            "conflict_forbids_profitability_met_strongly_for_contaminated_estate": True,
            "not_created_by_this_preregistration": True,
            "integrity_event_ledger_path": (
                "$NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/integrity-events/"
            ),
            "integrity_event_schema": BETTING_POLICY_INTEGRITY_EVENT_SCHEMA,
            "conflict_types": list(BETTING_INTEGRITY_CONFLICT_EVENT_TYPES),
            "conflict_durably_recorded_before_hard_stop": True,
            "integrity_event_ledger_append_only_first_write_wins": True,
            "identical_repeated_conflict_is_idempotent_by_deterministic_identity": True,
            "integrity_event_contains_at_minimum": [
                "event_type", "observed_at_utc", "locked_policy_hash", "attempted_policy_hash",
                "locked_betting_rule_hash", "attempted_betting_rule_hash",
                "first_eligible_wager_identity", "attempted_wager_identity", "git_commit",
                "event_content_hash",
            ],
            "any_persisted_conflict_event_permanently_contaminates_2026_estate": True,
            "contamination_bars_profitability_met_strongly_forever": True,
            "reverting_config_or_restoring_original_hash_does_not_clear_contamination": True,
            "no_automatic_clear_contamination_path": True,
            "reporter_scans_integrity_event_ledger": True,
        },
        "reporter": {
            "module": "nfl_hybrid.evaluation.prospective_strength_2026",
            "cli": "scripts/report_2026_strength_scorecard.py",
            "reads_only": [
                "immutable forecast ledger",
                "immutable evaluation ledger",
                "attached result records",
                "shadow-model-family ledger",
                "executable-price ledger if activated",
            ],
            "must_not_read_retrospective_2020_2025_artifacts_for_gates": True,
            "outputs": [
                "$NFL_MODEL_ARTIFACT_ROOT/production-2026/prospective-strength/PROSPECTIVE_2026_STATUS_SCORECARD.json",
                "$NFL_MODEL_ARTIFACT_ROOT/production-2026/prospective-strength/PROSPECTIVE_2026_STATUS_SCORECARD.md",
            ],
            "empty_sample_never_produces_MET_or_MET_STRONGLY": True,
        },
        "current_evidence_supported_labels": dict(CURRENT_LABELS),
        "sportsbook_sign_orientation": {
            "convention": "existing certified market convention -- unchanged",
            "home_spread_minus_3_5_implies_sportsbook_home_margin_plus_3_5": True,
            "ats_model_probability_and_ats_novig_probability_same_event_side": True,
            "total_model_probability_and_total_novig_probability_same_over_under_orientation": True,
            "no_sign_flip_can_make_the_point_market_comparison_appear_favorable": True,
        },
        "invariance": {
            "no_change_to_certified_scientific_behavior": True,
            "no_scientific_refit": True,
            "no_change_to_existing_certified_market_semantics": True,
            "content_addressed_cache_reused_as_infrastructure_only": True,
            "calibration_fail_closed_consumed_never_weakened": True,
            "market_state_hash_persistence_is_operational_provenance_only": True,
            "calibrated_only_scoring_tightens_never_loosens_evidence": True,
            "betting_integrity_event_ledger_is_append_only_evidence_only": True,
        },
    })


def preregistration_hash(payload: dict | None = None) -> str:
    """Deterministic hash of the preregistration payload with the hash field
    itself omitted (Section 1)."""
    payload = payload if payload is not None else build_preregistration_payload()
    without_hash = {k: v for k, v in payload.items() if k != _HASH_FIELD}
    return _sha256_hex(without_hash)


def frozen_preregistration_document() -> dict:
    """The payload with its self-hash embedded -- exactly what is written to
    ``outputs/prospective_2026_strength_preregistration.json``."""
    payload = build_preregistration_payload()
    payload[_HASH_FIELD] = preregistration_hash(payload)
    return payload


# ===========================================================================
# Immutable-ledger loading. Every loader tolerates a missing directory and
# returns an empty result -- the reporter must never fail on an empty
# prospective estate (Section 19).
# ===========================================================================
def _iter_json_records(root: Path, *, suffix_skip: tuple[str, ...] = (".result.json", ".json.tmp")) -> list[dict]:
    if not root.exists():
        return []
    out: list[dict] = []
    for path in sorted(root.glob("**/*.json")):
        if any(path.name.endswith(s) for s in suffix_skip):
            continue
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_forecast_ledger(forecast_root: Path) -> list[dict]:
    return _iter_json_records(Path(forecast_root))


def load_attached_evaluation_records(evaluation_root: Path) -> list[dict]:
    """Evaluation-ledger forecasts that have a sibling ``*.result.json``.
    Mirrors :func:`nfl_hybrid.production.run_2026.load_prospective_records`
    but is self-contained so the reporter has no import-time dependency on
    the operational pipeline."""
    evaluation_root = Path(evaluation_root)
    if not evaluation_root.exists():
        return []
    records: list[dict] = []
    for forecast_path in sorted(evaluation_root.glob("*/*.json")):
        if forecast_path.name.endswith((".result.json", ".json.tmp")):
            continue
        result_path = forecast_path.with_suffix(".result.json")
        if not result_path.exists():
            continue
        try:
            forecast = json.loads(forecast_path.read_text())
            result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        records.append({**forecast, "result_record": result})
    return records


def load_shadow_ledger(shadow_root: Path) -> list[dict]:
    return _iter_json_records(Path(shadow_root))


def load_executable_price_ledger(price_root: Path) -> list[dict]:
    return _iter_json_records(Path(price_root))


def executable_book_policy_status(config_path: Path) -> dict:
    """Section 14. ``config/executable_books_2026.json`` presence + hash
    only -- never invents a sportsbook."""
    config_path = Path(config_path)
    if not config_path.is_file():
        return {
            "frozen": False,
            "profitability_status": "NOT_ESTABLISHED",
            "reason": "EXECUTABLE_BOOK_POLICY_NOT_FROZEN",
            "config_path": str(config_path),
        }
    try:
        content = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "frozen": False,
            "profitability_status": "NOT_ESTABLISHED",
            "reason": "EXECUTABLE_BOOK_POLICY_NOT_FROZEN",
            "detail": f"unreadable config: {exc}",
            "config_path": str(config_path),
        }
    return {
        "frozen": True,
        "config_path": str(config_path),
        "policy_hash": _sha256_hex(content),
        "profitability_status": "GATED_BY_EVIDENCE",
    }


# ===========================================================================
# Contract hardening item 8 -- season-long executable-book policy lock.
#
#   $NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/
#       executable_book_policy_lock.json
#
# This file is NOT created by the preregistration. It is written exactly
# once -- when the first genuinely eligible executable wager is recorded --
# and first-write-wins. Any later attempt to evaluate/write a wager under a
# different policy hash or a different BETTING_RULE_2026_V1 hash hard-stops
# with BETTING_INTEGRITY_CONFLICT, and profitability can never become
# MET_STRONGLY for that contaminated estate.
# ===========================================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BettingIntegrityConflict(RuntimeError):
    """A wager evaluated / written under a policy hash or betting-rule hash
    that differs from the season-locked executable-book policy. Never
    caught-and-substituted; profitability is permanently barred from
    MET_STRONGLY for the contaminated estate."""


def executable_book_policy_lock_path(operational_root: Path) -> Path:
    """``operational_root`` is the artifact root (the parent of
    ``production-2026``)."""
    return (Path(operational_root) / "production-2026" / BETTING_POLICY_SUBDIR
            / EXECUTABLE_BOOK_POLICY_LOCK_FILENAME)


def write_executable_book_policy_lock(
    operational_root: Path,
    *,
    policy_hash: str,
    first_eligible_wager_identity: dict,
    git_commit: str | None,
    betting_rule_hash: str = BETTING_RULE_2026_V1_HASH,
    lock_timestamp_utc: str | None = None,
) -> dict:
    """Write the season lock. First-write-wins. Call this ONLY when the first
    genuinely eligible executable wager is recorded -- never speculatively.
    A second call with the same ``policy_hash`` + ``betting_rule_hash`` is an
    idempotent no-op; a second call with a different one raises
    :class:`BettingIntegrityConflict`."""
    path = executable_book_policy_lock_path(operational_root)
    payload = {
        "schema": EXECUTABLE_BOOK_POLICY_LOCK_SCHEMA,
        "policy_hash": str(policy_hash),
        "betting_rule_2026_v1_hash": str(betting_rule_hash),
        "first_eligible_wager_identity": dict(first_eligible_wager_identity),
        "lock_timestamp_utc": lock_timestamp_utc or _utc_now_iso(),
        "git_commit": git_commit,
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if (existing.get("policy_hash") == payload["policy_hash"]
                and existing.get("betting_rule_2026_v1_hash") == payload["betting_rule_2026_v1_hash"]):
            return {"status": "IDEMPOTENT_NOOP", "path": str(path), "lock": existing}
        # Blocker remediation 3: durably record the conflict in the
        # append-only integrity-event ledger BEFORE the hard-stop is raised,
        # so contamination is permanent and auditable across processes even
        # if the lock file / config is later reverted.
        recorded: list[dict] = []
        if existing.get("policy_hash") != payload["policy_hash"]:
            recorded.append(record_betting_integrity_event(
                operational_root, event_type=CONFLICT_POLICY_HASH_MISMATCH,
                locked_policy_hash=existing.get("policy_hash"),
                attempted_policy_hash=payload["policy_hash"],
                locked_betting_rule_hash=existing.get("betting_rule_2026_v1_hash"),
                attempted_betting_rule_hash=payload["betting_rule_2026_v1_hash"],
                first_wager_identity=existing.get("first_eligible_wager_identity"),
                attempted_wager_identity=payload["first_eligible_wager_identity"],
                git_commit=git_commit,
            ))
        if existing.get("betting_rule_2026_v1_hash") != payload["betting_rule_2026_v1_hash"]:
            recorded.append(record_betting_integrity_event(
                operational_root, event_type=CONFLICT_BETTING_RULE_HASH_MISMATCH,
                locked_policy_hash=existing.get("policy_hash"),
                attempted_policy_hash=payload["policy_hash"],
                locked_betting_rule_hash=existing.get("betting_rule_2026_v1_hash"),
                attempted_betting_rule_hash=payload["betting_rule_2026_v1_hash"],
                first_wager_identity=existing.get("first_eligible_wager_identity"),
                attempted_wager_identity=payload["first_eligible_wager_identity"],
                git_commit=git_commit,
            ))
        raise BettingIntegrityConflict(
            f"{BETTING_INTEGRITY_CONFLICT}: locked policy_hash={existing.get('policy_hash')} "
            f"betting_rule_hash={existing.get('betting_rule_2026_v1_hash')} != attempted "
            f"policy_hash={payload['policy_hash']} betting_rule_hash={payload['betting_rule_2026_v1_hash']}; "
            f"{len(recorded)} integrity event(s) recorded (append-only; contamination is permanent)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.rename(path)
    return {"status": "WRITTEN", "path": str(path), "lock": payload}


def verify_executable_book_policy_lock(
    operational_root: Path,
    *,
    policy_hash: str | None = None,
    betting_rule_hash: str = BETTING_RULE_2026_V1_HASH,
) -> dict:
    """Assess the season lock for the profitability gate. A lock whose
    ``policy_hash`` (when one is supplied) or ``betting_rule_2026_v1_hash``
    differs from the current contract is a ``BETTING_INTEGRITY_CONFLICT``:
    profitability can never be MET_STRONGLY for that estate."""
    path = executable_book_policy_lock_path(operational_root)
    if not path.is_file():
        return {"locked": False, "conflict": False, "flag": None, "path": str(path)}
    try:
        lock = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"locked": True, "conflict": True, "flag": BETTING_INTEGRITY_CONFLICT,
                "detail": f"unreadable lock: {exc}", "path": str(path)}
    conflict = lock.get("betting_rule_2026_v1_hash") != betting_rule_hash
    if policy_hash is not None and lock.get("policy_hash") != policy_hash:
        conflict = True
    return {
        "locked": True,
        "conflict": bool(conflict),
        "flag": BETTING_INTEGRITY_CONFLICT if conflict else None,
        "lock_policy_hash": lock.get("policy_hash"),
        "lock_betting_rule_2026_v1_hash": lock.get("betting_rule_2026_v1_hash"),
        "first_eligible_wager_identity": lock.get("first_eligible_wager_identity"),
        "path": str(path),
    }


# ===========================================================================
# Blocker remediation 3 -- append-only betting-policy integrity-event ledger.
#
#   $NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/integrity-events/
#       <event_content_hash>.json
#
# A conflicting wager attempt is durably recorded here BEFORE the hard-stop
# is raised. Once ANY valid BETTING_INTEGRITY_CONFLICT event has ever been
# persisted for the 2026 estate, profitability can never become MET_STRONGLY
# -- reverting config/executable_books_2026.json, restoring the original
# policy hash, or reusing the original betting-rule hash on a later call can
# never clear it. There is NO automatic clear-contamination path.
# ===========================================================================
def betting_integrity_events_dir(operational_root: Path) -> Path:
    return (Path(operational_root) / "production-2026" / BETTING_POLICY_SUBDIR
            / INTEGRITY_EVENTS_SUBDIR)


def _none_or_str(value: object) -> str | None:
    return None if value is None else str(value)


def _betting_integrity_event_identity(
    *,
    event_type: str,
    locked_policy_hash: object,
    attempted_policy_hash: object,
    locked_betting_rule_hash: object,
    attempted_betting_rule_hash: object,
    first_wager_identity: object,
    attempted_wager_identity: object,
) -> dict:
    """The deterministic identity of a betting-integrity conflict event --
    everything EXCEPT wall-clock (``observed_at_utc``) and ``git_commit``, so
    an identical repeated conflict maps to the same append-only file and is
    idempotent."""
    return {
        "schema": BETTING_POLICY_INTEGRITY_EVENT_SCHEMA,
        "event_type": str(event_type),
        "locked_policy_hash": _none_or_str(locked_policy_hash),
        "attempted_policy_hash": _none_or_str(attempted_policy_hash),
        "locked_betting_rule_hash": _none_or_str(locked_betting_rule_hash),
        "attempted_betting_rule_hash": _none_or_str(attempted_betting_rule_hash),
        "first_eligible_wager_identity": dict(first_wager_identity)
        if isinstance(first_wager_identity, dict) else None,
        "attempted_wager_identity": dict(attempted_wager_identity)
        if isinstance(attempted_wager_identity, dict) else None,
    }


def record_betting_integrity_event(
    operational_root: Path,
    *,
    event_type: str,
    locked_policy_hash: str | None,
    attempted_policy_hash: str | None,
    locked_betting_rule_hash: str | None,
    attempted_betting_rule_hash: str | None,
    first_wager_identity: dict | None,
    attempted_wager_identity: dict | None = None,
    git_commit: str | None = None,
    observed_at_utc: str | None = None,
) -> dict:
    """Append-only, first-write-wins durable record of a betting-integrity
    conflict. Written BEFORE the hard-stop is raised. An identical repeated
    conflict (same deterministic identity) is an idempotent no-op. Once ANY
    such event exists for the 2026 estate, profitability can never become
    MET_STRONGLY -- there is no clear-contamination path."""
    if event_type not in BETTING_INTEGRITY_CONFLICT_EVENT_TYPES:
        raise ValueError(f"unknown betting-integrity event_type {event_type!r}")
    identity = _betting_integrity_event_identity(
        event_type=event_type,
        locked_policy_hash=locked_policy_hash, attempted_policy_hash=attempted_policy_hash,
        locked_betting_rule_hash=locked_betting_rule_hash,
        attempted_betting_rule_hash=attempted_betting_rule_hash,
        first_wager_identity=first_wager_identity, attempted_wager_identity=attempted_wager_identity,
    )
    event_content_hash = _sha256_hex(identity)
    path = betting_integrity_events_dir(operational_root) / f"{event_content_hash}.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        return {"status": "IDEMPOTENT_NOOP", "path": str(path),
                "event_content_hash": event_content_hash, "event": existing}
    payload = {
        **identity,
        "flag": BETTING_INTEGRITY_CONFLICT,
        "observed_at_utc": observed_at_utc or _utc_now_iso(),
        "git_commit": git_commit,
        "event_content_hash": event_content_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.rename(path)
    return {"status": "WRITTEN", "path": str(path),
            "event_content_hash": event_content_hash, "event": payload}


def load_betting_integrity_events(operational_root: Path) -> list[dict]:
    """Every persisted betting-policy integrity event. Tolerates a missing
    directory (returns ``[]``)."""
    root = betting_integrity_events_dir(operational_root)
    if not root.exists():
        return []
    out: list[dict] = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".json.tmp"):
            continue
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def betting_integrity_estate_status(
    operational_root: Path | None = None, *, events: list[dict] | None = None,
) -> dict:
    """Scan the append-only betting-policy integrity-event ledger. If ANY
    valid ``BETTING_INTEGRITY_CONFLICT`` event has ever been persisted for
    the 2026 estate, the estate is permanently contaminated: reverting the
    config, restoring the original policy hash, or reusing the original
    betting-rule hash on a later call can never clear it. Each event's
    deterministic content hash is re-verified so a hand-edited event can
    neither launder contamination away nor fabricate one."""
    if events is None:
        events = load_betting_integrity_events(operational_root) if operational_root is not None else []
    valid: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("event_type") not in BETTING_INTEGRITY_CONFLICT_EVENT_TYPES:
            continue
        try:
            identity = _betting_integrity_event_identity(
                event_type=ev.get("event_type"),
                locked_policy_hash=ev.get("locked_policy_hash"),
                attempted_policy_hash=ev.get("attempted_policy_hash"),
                locked_betting_rule_hash=ev.get("locked_betting_rule_hash"),
                attempted_betting_rule_hash=ev.get("attempted_betting_rule_hash"),
                first_wager_identity=ev.get("first_eligible_wager_identity"),
                attempted_wager_identity=ev.get("attempted_wager_identity"),
            )
        except Exception:
            continue
        if _sha256_hex(identity) != ev.get("event_content_hash"):
            continue
        valid.append(ev)
    return {
        "contaminated_estate": bool(valid),
        "conflict_event_count": len(valid),
        "conflict_events": valid,
        "no_automatic_clear_contamination_path": True,
    }


def market_non_adverse(roi_cluster_bootstrap_ci_upper: float | None, realized_net_units: float | None) -> bool:
    """Contract hardening item 7 -- the ONE frozen definition of a
    market-specific "statistically non-adverse" profitability result: the
    upper bound of that market's game_id-cluster bootstrap 95% ROI CI is
    >= 0 AND its realized net units are >= 0. A market with negative realized
    net units is NEVER non-adverse."""
    ci_upper = _num(roi_cluster_bootstrap_ci_upper)
    net = _num(realized_net_units)
    if ci_upper is None or net is None:
        return False
    return ci_upper >= 0.0 and net >= 0.0


def season_final_confirmatory(schedule_population: dict | None, completed_game_ids: set[str]) -> bool:
    """Contract hardening item 9 -- SEASON_FINAL_CONFIRMATORY is true ONLY
    when the canonical 2026 REG+POST schedule population is available, every
    eligible REG+POST scheduled game has a final result attached, and no
    remaining 2026 REG+POST game is scheduled / unresolved. Season completion
    is NEVER inferred from a calendar date or a sample count alone; if
    schedule completeness cannot be proven the answer is ``False``."""
    if not schedule_population or not schedule_population.get("available"):
        return False
    scheduled = {str(g) for g in (schedule_population.get("scheduled_reg_post_game_ids") or [])}
    if not scheduled:
        return False
    unresolved = {str(g) for g in (schedule_population.get("unresolved_reg_post_game_ids") or [])}
    if unresolved:
        return False
    return scheduled.issubset({str(g) for g in completed_game_ids})


# ===========================================================================
# Section 10 -- prospective shadow model-family ledger. Identity is
# (game_id, horizon, candidate, target_cutoff_utc); writes are immutable
# first-write-wins. A shadow record NEVER carries an outcome column and NEVER
# has production-selection authority -- it is a separate evidence path only.
# ===========================================================================
SHADOW_LEDGER_SUBDIR = "shadow-model-family-ledger"
_SHADOW_FORBIDDEN_FIELDS = frozenset({
    "actual_margin", "actual_total", "home_score", "away_score",
    "binary_outcome", "result", "winner",
})


class ShadowLedgerViolation(RuntimeError):
    """A shadow-ledger write that would leak an outcome, or silently change an
    existing immutable record. Never caught-and-substituted."""


def _shadow_identity_path(root: Path, game_id: str, horizon: str, candidate: str, target_cutoff_utc: str) -> Path:
    safe_cutoff = str(target_cutoff_utc).replace(":", "").replace("+", "_")
    return Path(root) / horizon / candidate / f"{game_id}__{safe_cutoff}.json"


def write_shadow_record(shadow_root: Path, record: dict) -> dict:
    """Immutable first-write-wins. ``record`` must contain ``game_id``,
    ``horizon``, ``candidate``, ``target_cutoff_utc`` and a deterministic
    ``prediction`` payload; it must NOT contain any outcome field."""
    leaked = _SHADOW_FORBIDDEN_FIELDS & set(record)
    if leaked:
        raise ShadowLedgerViolation(f"shadow record contains forbidden outcome field(s): {sorted(leaked)}")
    for field in ("game_id", "horizon", "candidate", "target_cutoff_utc", "prediction"):
        if field not in record:
            raise ShadowLedgerViolation(f"shadow record missing required field: {field!r}")
    if record["candidate"] not in FROZEN_FIX7_SHADOW_CANDIDATES:
        raise ShadowLedgerViolation(f"unknown shadow candidate {record['candidate']!r} (not a frozen Fix-7 candidate)")

    path = _shadow_identity_path(
        shadow_root, str(record["game_id"]), record["horizon"], record["candidate"], str(record["target_cutoff_utc"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_hash = _sha256_hex(record["prediction"])
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("prediction_hash") == prediction_hash:
            return {"status": "IDEMPOTENT_NOOP", "path": str(path), "prediction_hash": prediction_hash}
        raise ShadowLedgerViolation(
            f"shadow record already exists with a different payload for "
            f"{record['game_id']}/{record['horizon']}/{record['candidate']}/{record['target_cutoff_utc']}"
        )
    full = {**record, "prediction_hash": prediction_hash}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(full, indent=2, sort_keys=True, default=str))
    tmp.rename(path)
    return {"status": "WRITTEN", "path": str(path), "prediction_hash": prediction_hash}


# ===========================================================================
# Section 2 -- provenance / eligibility filtering.
# ===========================================================================
_SHA256_HEX_LEN = 64


def _looks_like_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == _SHA256_HEX_LEN and all(c in "0123456789abcdef" for c in text.lower())


def _record_population(rec: dict) -> tuple[object, object]:
    """The persisted ``(season, season_type)`` for a record, tolerating the
    evaluation-ledger shape (top-level) and the nested forecast-ledger shape
    (``prediction`` / ``prediction.prediction``). Contract hardening item 2."""
    pred = rec.get("prediction", {}) if isinstance(rec.get("prediction"), dict) else {}
    inner = pred.get("prediction", {}) if isinstance(pred.get("prediction"), dict) else {}
    fcast = rec.get("forecast", {}) if isinstance(rec.get("forecast"), dict) else {}
    season = rec.get("season")
    if season is None:
        season = pred.get("season")
    if season is None:
        season = inner.get("season")
    if season is None:
        season = fcast.get("season")
    season_type = (rec.get("season_type") or pred.get("season_type")
                   or inner.get("season_type") or fcast.get("season_type"))
    return season, season_type


def build_market_state_payload(rec: dict) -> dict:
    """The one deterministic market-state payload for a forecast record:

        {"ATS": <persisted ATS consensus market dict or None>,
         "TOTAL": <persisted TOTAL consensus market dict or None>}

    Reconstructed SOLELY from the immutable persisted record -- the
    ``markets[m]["market"]`` consensus sub-dict, exactly as stored. Volatile
    provenance (``created_at_utc``, ``run_id``, ``git_commit``), the model's
    prediction probabilities and any outcome are excluded by construction:
    only the consensus market sub-dict is carried. NEVER reads a later
    mutable raw market file."""
    markets = rec.get("markets") or rec.get("forecast", {}).get("markets") or {}
    payload: dict = {}
    for name in MARKET_STATE_PAYLOAD_MARKETS:
        entry = markets.get(name)
        snap = entry.get("market") if isinstance(entry, dict) else None
        payload[name] = snap if isinstance(snap, dict) else None
    return payload


def compute_market_state_hash(rec: dict) -> str | None:
    """Contract hardening item 3 (+ blocker remediation). Deterministic
    SHA-256 over :func:`canonical_json` of :func:`build_market_state_payload`
    -- the EXACT persisted ATS/TOTAL consensus market dicts, reconstructed
    from the immutable record alone. Returns ``None`` only when the record
    carries no persisted market state at all (both markets ``None``)."""
    payload = build_market_state_payload(rec)
    if all(v is None for v in payload.values()):
        return None
    return _sha256_hex(payload)


def _record_provenance(rec: dict) -> dict:
    """Flatten the provenance fields the reporter checks, tolerating both the
    evaluation-ledger shape ({forecast, markets, provenance, ...}) and a flat
    forecast record."""
    prov = dict(rec.get("provenance", {}))
    pred = rec.get("prediction", {}) if isinstance(rec.get("prediction"), dict) else {}
    op_hash = prov.get("operational_model_spec_hash") or pred.get("operational_model_spec_hash")
    fs_hash = prov.get("horizon_feature_semantics_hash") or pred.get("horizon_feature_semantics_hash") \
        or prov.get("feature_state_semantics_hash") or pred.get("feature_state_semantics_hash")
    # The certified baseline is pinned either by an explicit SHA or,
    # equivalently, by the certified operational-spec + feature-semantics
    # hash pair (both immutably identify the same certified commit). The
    # immutable evaluation ledger stores the hash pair; the forecast ledger
    # also stores the explicit SHA -- either is accepted.
    certified_pin = rec.get("certified_baseline_sha") or pred.get("certified_baseline_sha") \
        or (op_hash and fs_hash and f"HASHPIN:{op_hash}:{fs_hash}") or None
    # market_state_hash (legacy name market_snapshot_hash): an EXPLICIT hash
    # persisted at forecast time in the forecast-ledger prediction payload
    # (prediction.market_state_hash) or the evaluation-ledger record
    # (top-level market_state_hash / provenance.market_state_hash). Blocker
    # remediation 1: there is NO derive-and-accept fallback -- a
    # market-relative row with no explicit persisted hash is ineligible
    # (MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE). Integrity -- the persisted
    # hash matching the reporter's recompute over the immutable persisted
    # market state -- is verified in classify_evidence_eligibility (item 3).
    explicit_market_hash = (
        prov.get("market_state_hash") or prov.get("market_snapshot_hash")
        or pred.get("market_state_hash") or pred.get("market_snapshot_hash")
        or rec.get("market_state_hash") or rec.get("market_snapshot_hash")
    )
    market_state_hash = explicit_market_hash
    season, season_type = _record_population(rec)
    flat = {
        "game_id": rec.get("game_id"),
        "horizon": rec.get("horizon"),
        "season": season,
        "season_type": season_type,
        "target_cutoff_utc": rec.get("target_cutoff_utc"),
        "forecast_created_at_utc": rec.get("created_at_utc") or prov.get("created_at_utc")
        or pred.get("created_at_utc"),
        "git_commit": rec.get("git_commit") or prov.get("git_commit"),
        "certified_baseline_sha": certified_pin,
        "operational_model_spec_hash": op_hash,
        "feature_state_semantics_hash": fs_hash,
        "prediction_hash": rec.get("prediction_hash") or rec.get("content_hash"),
        "market_state_hash": market_state_hash,
        "explicit_market_state_hash": explicit_market_hash,
    }
    return flat




# Contract hardening item 1 -- reuse the certified production due-run window
# semantics; do NOT invent a second clock.
_DUE_WINDOW_HELPERS: tuple | None = None


def _due_window_helpers() -> tuple:
    global _DUE_WINDOW_HELPERS
    if _DUE_WINDOW_HELPERS is None:
        try:
            from nfl_hybrid.production.run_2026 import current_or_recent_cutoff, is_within_due_window
            _DUE_WINDOW_HELPERS = (is_within_due_window, current_or_recent_cutoff)
        except Exception:
            _DUE_WINDOW_HELPERS = (None, None)
    return _DUE_WINDOW_HELPERS


def _forecast_time_eligible(prov: dict) -> tuple[bool, str | None]:
    """Contract hardening item 1. ``target_cutoff_utc`` is the information
    as-of cutoff -- NOT a requirement that the completed forecast FILE be
    written before it. A forecast written seconds or minutes after the
    cutoff, using only cutoff-legal inputs, stays eligible. What must hold:
    forecast generation corresponds to the certified TUE/FRI cutoff and
    occurred inside the certified production due-run execution window
    (``forecast_created_at_utc`` may be >= ``target_cutoff_utc``). A forecast
    created outside the permitted execution window -- or whose creation maps
    to a different card cutoff than it claims -- is prospectively ineligible.
    Returns ``(eligible, reason_or_None)``."""
    is_within_due_window, current_or_recent_cutoff = _due_window_helpers()
    if is_within_due_window is None:
        return True, None  # production scheduling module unavailable -> cannot verify; do not punish
    horizon = prov.get("horizon")
    if horizon not in ("TUE", "FRI"):
        return False, "IDENTIFIER_FAILURE"
    try:
        import pandas as pd

        created = pd.Timestamp(prov.get("forecast_created_at_utc"))
        created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
        target_cutoff = pd.Timestamp(prov.get("target_cutoff_utc"))
        target_cutoff = target_cutoff.tz_localize("UTC") if target_cutoff.tzinfo is None else target_cutoff.tz_convert("UTC")
    except Exception:
        return False, "IDENTIFIER_FAILURE"
    try:
        due = bool(is_within_due_window(created, horizon)["due"])
        window_cutoff = pd.Timestamp(current_or_recent_cutoff(created, horizon))
    except Exception:
        return False, "FORECAST_OUTSIDE_EXECUTION_WINDOW"
    if not due or window_cutoff != target_cutoff:
        return False, "FORECAST_OUTSIDE_EXECUTION_WINDOW"
    return True, None


def _in_prospective_population(season: object, season_type: object) -> bool:
    """Contract hardening item 2. Season 2026, ``season_type`` in {REG, POST}
    only; PRESEASON never qualifies."""
    try:
        season_ok = int(season) == PROSPECTIVE_SEASON
    except (TypeError, ValueError):
        return False
    return season_ok and str(season_type or "").upper() in PROSPECTIVE_SEASON_TYPES


def classify_evidence_eligibility(records: list[dict]) -> dict:
    """Split raw records into ``eligible`` / ``excluded`` per Section 2 and
    the contract-hardening additions. A row is excluded ONLY for a documented
    integrity / eligibility reason -- never because its prediction was poor.

    Excludes: missing required provenance (``IDENTIFIER_FAILURE``); a
    market-relative row with no contemporaneous market evidence
    (``MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE``); a persisted market-state
    hash that does not match the reporter's recompute over the persisted
    market state (``HASH_MISMATCH``, item 3); a forecast generated outside
    the certified production execution window or mapped to the wrong card
    cutoff (``FORECAST_OUTSIDE_EXECUTION_WINDOW``, item 1); a record outside
    the frozen season-2026 REG+POST population (``OUT_OF_PROSPECTIVE_POPULATION``,
    item 2 -- PRESEASON never enters)."""
    eligible: list[dict] = []
    excluded: list[dict] = []
    for rec in records:
        prov = _record_provenance(rec)
        stub = {
            "game_id": prov.get("game_id"),
            "horizon": prov.get("horizon"),
            "target_cutoff_utc": prov.get("target_cutoff_utc"),
        }
        missing = [f for f in REQUIRED_FORECAST_PROVENANCE if not prov.get(f)]
        uses_market = _record_uses_market(rec)
        if uses_market and not prov.get("market_state_hash"):
            missing.append("market_state_hash")
        if missing:
            excluded.append({
                **stub,
                "reason": "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE"
                if missing == ["market_state_hash"]
                else "IDENTIFIER_FAILURE",
                "missing_fields": missing,
            })
            continue

        # Item 3 (+ blocker remediation) -- the EXPLICIT persisted
        # market_state_hash must reconstruct-and-recompute to EXACT equality
        # from the immutable record alone. An absent / malformed / mismatching
        # hash is a provenance failure (tamper / drift / no forecast-time
        # persistence), not a poor prediction. No derive-and-accept.
        if uses_market:
            explicit = prov.get("explicit_market_state_hash") or prov.get("market_state_hash")
            recomputed = compute_market_state_hash(rec)
            if (not explicit
                    or not _looks_like_sha256(str(explicit))
                    or recomputed is None
                    or str(explicit) != str(recomputed)):
                excluded.append({
                    **stub,
                    "reason": "HASH_MISMATCH",
                    "detail": (
                        "persisted market_state_hash is absent, malformed, or does not match the "
                        "reporter's recompute over the immutable persisted {ATS, TOTAL} market state"
                    ),
                    "explicit_market_state_hash": explicit,
                    "recomputed_market_state_hash": recomputed,
                })
                continue

        # Item 1 -- forecast-time (execution-window) eligibility.
        time_ok, time_reason = _forecast_time_eligible(prov)
        if not time_ok:
            excluded.append({**stub, "reason": time_reason})
            continue

        # Item 2 -- frozen 2026 REG+POST population (PRESEASON excluded).
        if not _in_prospective_population(prov.get("season"), prov.get("season_type")):
            excluded.append({
                **stub,
                "reason": "OUT_OF_PROSPECTIVE_POPULATION",
                "season": prov.get("season"),
                "season_type": prov.get("season_type"),
            })
            continue

        eligible.append(rec)
    return {"eligible": eligible, "excluded": excluded}


# Exclusion reasons that are scope / eligibility filters rather than
# provenance-coverage failures -- they do not, by themselves, demote an
# operational gate.
_NON_PROVENANCE_EXCLUSION_REASONS = frozenset({
    "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE",
    "FORECAST_OUTSIDE_EXECUTION_WINDOW",
    "OUT_OF_PROSPECTIVE_POPULATION",
})


def _record_uses_market(rec: dict) -> bool:
    markets = rec.get("markets") or rec.get("forecast", {}).get("markets") or {}
    for entry in markets.values():
        if isinstance(entry, dict) and entry.get("market") is not None:
            return True
    return False


def _completed_game_ids(records: list[dict]) -> set[str]:
    gids: set[str] = set()
    for rec in records:
        result = rec.get("result_record", {}).get("result") or rec.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("home_score") is None or result.get("away_score") is None:
            continue
        gids.add(str(rec.get("game_id")))
    return gids


def _unique_completed_games(records: list[dict]) -> int:
    return len(_completed_game_ids(records))


# ===========================================================================
# Stream row extraction for calibration / probability / point gates.
# ===========================================================================
_MARKETS = ("ATS", "TOTAL")
_HORIZONS = ("TUE", "FRI")


def _stream_rows(records: list[dict], market: str, horizon: str | None) -> dict:
    """Non-push graded rows for one ``market`` (optionally one ``horizon``).
    Returns aligned arrays: outcome ``y``, model raw/calibrated probability,
    market no-vig probability, per-game ids, and squared-error building
    blocks for the point gate."""
    y, p_raw, p_cal, p_market, gids = [], [], [], [], []
    m_sq_model, m_sq_market, t_sq_model, t_sq_market, pt_gids = [], [], [], [], []
    calibrated_status_violations: list[dict] = []
    for rec in records:
        if horizon is not None and rec.get("horizon") != horizon:
            continue
        result = rec.get("result_record", {}).get("result") or rec.get("result") or {}
        hs, as_ = result.get("home_score"), result.get("away_score")
        if hs is None or as_ is None:
            continue
        actual_margin = float(hs) - float(as_)
        actual_total = float(hs) + float(as_)
        gid = str(rec.get("game_id"))
        forecast = rec.get("forecast") or rec.get("prediction", {}).get("prediction") or {}

        entry = (rec.get("markets") or {}).get(market) or {}
        snap = entry.get("market")
        if snap is None:
            continue
        line = float(snap["consensus_line"])
        if market == "ATS":
            edge = actual_margin + line
            if forecast.get("predicted_margin") is not None:
                m_sq_model.append((float(forecast["predicted_margin"]) - actual_margin) ** 2)
                m_sq_market.append((-line - actual_margin) ** 2)
                pt_gids.append(gid)
        else:
            edge = actual_total - line
            if forecast.get("predicted_total") is not None:
                t_sq_model.append((float(forecast["predicted_total"]) - actual_total) ** 2)
                t_sq_market.append((line - actual_total) ** 2)
                pt_gids.append(gid)
        if abs(edge) < 1e-12:
            continue  # push -- excluded from binary scoring (Fix-8 semantics)
        outcome = 1 if edge > 0 else 0
        raw_p = entry.get("raw_conditional_upper_probability")
        novig = snap.get("consensus_novig_probability")
        if raw_p is None or novig is None:
            continue
        # Sections 5 / 24: a calibrated probability is used ONLY when the row
        # is genuinely CALIBRATED. A numeric ``calibrated_*`` value paired
        # with any non-CALIBRATED status is a fail-closed contract violation
        # in the producer -- the reporter refuses to score it (treats it as
        # absent) and counts it for the PRODUCTION_READY_PIPELINE gate.
        cal_status = str(entry.get("calibration_status", "")) or "UNKNOWN"
        cal_p = entry.get("calibrated_conditional_upper_probability")
        if cal_status != "CALIBRATED":
            if cal_p is not None:
                calibrated_status_violations.append({"game_id": gid, "market": market,
                                                     "calibration_status": cal_status})
            cal_p = None
        y.append(outcome)
        p_raw.append(float(raw_p))
        p_cal.append(float(cal_p) if cal_p is not None else np.nan)
        p_market.append(float(novig))
        gids.append(gid)
    return {
        "y": np.asarray(y, dtype=float),
        "p_raw": np.asarray(p_raw, dtype=float),
        "p_cal": np.asarray(p_cal, dtype=float),
        "p_market": np.asarray(p_market, dtype=float),
        "game_ids": gids,
        "margin_sq_model": np.asarray(m_sq_model, dtype=float),
        "margin_sq_market": np.asarray(m_sq_market, dtype=float),
        "total_sq_model": np.asarray(t_sq_model, dtype=float),
        "total_sq_market": np.asarray(t_sq_market, dtype=float),
        "point_game_ids": pt_gids,
        "calibrated_status_violations": calibrated_status_violations,
    }


def _num(x) -> float | None:
    if x is None or x == NOT_ESTIMABLE:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ===========================================================================
# Gate evaluators. Each returns a uniform dict; none can reach MET_STRONGLY
# without >= PROMOTION_ELIGIBLE maturity.
# ===========================================================================
def _blank_gate(name: str, maturity: str, n_games: int, reason: str) -> dict:
    return {
        "gate": name,
        "status": "INSUFFICIENT_PROSPECTIVE_SAMPLE" if maturity == MATURITY_INSUFFICIENT else "NOT_DEMONSTRATED",
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": None,
        "confidence_interval": None,
        "gate_booleans": {},
        "failure_reasons": [reason],
        "promotion_eligible": _promotion_eligible(n_games),
    }


def gate_calibration_improves_raw(records: list[dict], maturity: str, n_games: int) -> dict:
    name = "CALIBRATION_IMPROVES_RAW_PROBABILITIES"
    per_stream: dict[str, dict] = {}
    pooled_ll_delta_rows: list[float] = []
    pooled_br_delta_rows: list[float] = []
    pooled_gids: list[str] = []
    # Contract hardening item 5 -- per-market pooled deltas + unique games,
    # for the frozen ATS/TOTAL calibration evidence labels.
    per_market_ll_delta: dict[str, list[float]] = {m: [] for m in _MARKETS}
    per_market_br_delta: dict[str, list[float]] = {m: [] for m in _MARKETS}
    per_market_gids: dict[str, list[str]] = {m: [] for m in _MARKETS}
    any_rows = False
    for market in _MARKETS:
        for horizon in _HORIZONS:
            rows = _stream_rows(records, market, horizon)
            mask = np.isfinite(rows["p_cal"]) & (rows["p_cal"] > 0) & (rows["p_cal"] < 1)
            y = rows["y"][mask]
            praw = rows["p_raw"][mask]
            pcal = rows["p_cal"][mask]
            gids = [g for g, keep in zip(rows["game_ids"], mask) if keep]
            stream = f"{market}_{horizon}"
            if len(y) == 0:
                per_stream[stream] = {"n": 0, "status": "INSUFFICIENT"}
                continue
            any_rows = True
            ll_raw, ll_cal = binary_log_loss(y, praw), binary_log_loss(y, pcal)
            br_raw, br_cal = brier(y, praw), brier(y, pcal)
            ece_raw, ece_cal = fixed_bin_ece(y, praw), fixed_bin_ece(y, pcal)
            ll_rows = _pointwise_logloss(y, pcal) - _pointwise_logloss(y, praw)
            br_rows = (pcal - y) ** 2 - (praw - y) ** 2
            pooled_ll_delta_rows.extend(ll_rows.tolist())
            pooled_br_delta_rows.extend(br_rows.tolist())
            pooled_gids.extend(gids)
            per_market_ll_delta[market].extend(ll_rows.tolist())
            per_market_br_delta[market].extend(br_rows.tolist())
            per_market_gids[market].extend(gids)
            per_stream[stream] = {
                "n": int(len(y)),
                "log_loss_raw": ll_raw, "log_loss_calibrated": ll_cal,
                "brier_raw": br_raw, "brier_calibrated": br_cal,
                "ece_raw": ece_raw if ece_raw == NOT_ESTIMABLE else float(ece_raw),
                "ece_calibrated": ece_cal if ece_cal == NOT_ESTIMABLE else float(ece_cal),
                "calibrated_lower_log_loss": bool(ll_cal < ll_raw),
                "calibrated_lower_brier": bool(br_cal < br_raw),
                "calibrated_lower_ece": bool(_num(ece_cal) is not None and _num(ece_raw) is not None
                                             and _num(ece_cal) < _num(ece_raw)),
                "calibrated_minus_raw_log_loss_point": ll_cal - ll_raw,
            }

    if not any_rows:
        out = _blank_gate(name, maturity, n_games, "no calibrated stream rows yet")
        out["per_stream"] = per_stream
        return out

    ll_boot = game_cluster_bootstrap(pooled_ll_delta_rows, pooled_gids) if pooled_gids else None
    br_boot = game_cluster_bootstrap(pooled_br_delta_rows, pooled_gids) if pooled_gids else None
    streams_scored = [s for s in per_stream.values() if s.get("n", 0) > 0]
    all_four = len(streams_scored) == 4

    booleans = {
        "all_four_streams_scored": all_four,
        "all_four_calibrated_lower_log_loss": all_four and all(s["calibrated_lower_log_loss"] for s in streams_scored),
        "all_four_calibrated_lower_brier": all_four and all(s["calibrated_lower_brier"] for s in streams_scored),
        "all_four_calibrated_lower_ece": all_four and all(s["calibrated_lower_ece"] for s in streams_scored),
        "pooled_log_loss_delta_ci_upper_below_zero": bool(ll_boot and _num(ll_boot["ci_high"]) is not None
                                                         and ll_boot["ci_high"] < 0.0),
        "pooled_brier_delta_ci_upper_below_zero": bool(br_boot and _num(br_boot["ci_high"]) is not None
                                                       and br_boot["ci_high"] < 0.0),
        "no_stream_adverse_log_loss_point": all_four and all(
            s["calibrated_minus_raw_log_loss_point"] <= 0.0 for s in streams_scored
        ),
        "promotion_eligible_maturity": _promotion_eligible(n_games),
    }
    met = all(booleans.values())
    status = "MET_STRONGLY" if met else ("INTERIM_EVIDENCE" if maturity in (MATURITY_INTERIM, MATURITY_DESCRIPTIVE)
                                         else "NOT_DEMONSTRATED")
    failure_reasons = [k for k, v in booleans.items() if not v]
    return {
        "gate": name,
        "status": status,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {"pooled_calibrated_minus_raw_log_loss": ll_boot["point_estimate"] if ll_boot else None,
                           "pooled_calibrated_minus_raw_brier": br_boot["point_estimate"] if br_boot else None},
        "confidence_interval": {"pooled_log_loss_delta": [ll_boot["ci_low"], ll_boot["ci_high"]] if ll_boot else None,
                                "pooled_brier_delta": [br_boot["ci_low"], br_boot["ci_high"]] if br_boot else None},
        "gate_booleans": booleans,
        "failure_reasons": failure_reasons,
        "per_stream": per_stream,
        "ats_calibration_evidence": _market_calibration_evidence_label(
            per_stream, "ATS", per_market_ll_delta["ATS"], per_market_br_delta["ATS"], per_market_gids["ATS"]),
        "total_calibration_evidence": _market_calibration_evidence_label(
            per_stream, "TOTAL", per_market_ll_delta["TOTAL"], per_market_br_delta["TOTAL"], per_market_gids["TOTAL"]),
        "promotion_eligible": _promotion_eligible(n_games),
    }


def _pointwise_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _market_calibration_evidence_label(
    per_stream: dict, market: str,
    market_ll_delta_rows: list[float], market_br_delta_rows: list[float], market_gids: list[str],
) -> str:
    """Contract hardening item 5 -- the frozen ATS/TOTAL calibration evidence
    label for ONE market, over its unique completed eligible games and its
    two constituent horizon streams (TUE, FRI).

    INSUFFICIENT: fewer than 64 unique completed eligible games.
    STRONG:       >=200 unique completed games AND both constituent horizon
                  streams improve raw on LL, Brier, and ECE AND pooled
                  calibrated-minus-raw LL and Brier 95% CI upper bounds <0
                  AND neither horizon has an adverse LL point estimate.
    SUGGESTIVE:   >=64 games, not STRONG, pooled calibrated-minus-raw LL <0
                  and Brier <0, and neither constituent horizon has an
                  adverse LL point estimate.
    MIXED:        >=64 games and neither STRONG nor SUGGESTIVE.
    """
    streams = [v for k, v in per_stream.items() if k.startswith(market + "_") and v.get("n", 0) > 0]
    unique_games = len(set(market_gids))
    if unique_games < CALIBRATION_EVIDENCE_MIN_UNIQUE_GAMES or not streams:
        return "INSUFFICIENT"

    both_horizons = len(streams) == 2
    improves_all = both_horizons and all(
        s.get("calibrated_lower_log_loss") and s.get("calibrated_lower_brier") and s.get("calibrated_lower_ece")
        for s in streams
    )
    no_adverse_ll_point = all(s["calibrated_minus_raw_log_loss_point"] <= 0.0 for s in streams)

    ll_boot = game_cluster_bootstrap(market_ll_delta_rows, market_gids) if market_gids else None
    br_boot = game_cluster_bootstrap(market_br_delta_rows, market_gids) if market_gids else None
    ll_point = _num(ll_boot["point_estimate"]) if ll_boot else None
    br_point = _num(br_boot["point_estimate"]) if br_boot else None
    ll_ci_upper = _num(ll_boot["ci_high"]) if ll_boot else None
    br_ci_upper = _num(br_boot["ci_high"]) if br_boot else None

    if (
        unique_games >= CALIBRATION_EVIDENCE_STRONG_MIN_UNIQUE_GAMES
        and improves_all
        and ll_ci_upper is not None and ll_ci_upper < 0.0
        and br_ci_upper is not None and br_ci_upper < 0.0
        and no_adverse_ll_point
    ):
        return "STRONG"
    if (
        ll_point is not None and ll_point < 0.0
        and br_point is not None and br_point < 0.0
        and no_adverse_ll_point
    ):
        return "SUGGESTIVE"
    return "MIXED"


def gate_absolute_probability_quality(records: list[dict], maturity: str, n_games: int) -> dict:
    name = "ABSOLUTE_PROBABILITY_QUALITY"
    pooled_y, pooled_p, pooled_gids = [], [], []
    per_market_n = {"ATS": 0, "TOTAL": 0}
    per_market_rows_without_calibrated = {"ATS": 0, "TOTAL": 0}
    for market in _MARKETS:
        rows = _stream_rows(records, market, None)
        # Blocker remediation 2: this gate scores the CALIBRATED production
        # probability only. A row whose calibration_status != CALIBRATED
        # carries p_cal = NaN (fail-closed in _stream_rows) and is NOT
        # rescued with p_raw here.
        p_rows = rows["p_cal"]
        keep = np.isfinite(p_rows)
        per_market_rows_without_calibrated[market] = int((~keep).sum())
        pooled_y.extend(rows["y"][keep].tolist())
        pooled_p.extend(p_rows[keep].tolist())
        gids = [g for g, k in zip(rows["game_ids"], keep) if k]
        pooled_gids.extend(gids)
        per_market_n[market] = int(keep.sum())
    y = np.asarray(pooled_y, dtype=float)
    p = np.asarray(pooled_p, dtype=float)
    if len(y) == 0:
        return _blank_gate(name, maturity, n_games, "no CALIBRATED probability rows yet")

    ll = binary_log_loss(y, p)
    br = brier(y, p)
    ece = fixed_bin_ece(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    auc = auc_diagnostic(y, p)
    ll_boot = game_cluster_bootstrap(_pointwise_logloss(y, p).tolist(), pooled_gids)
    br_boot = game_cluster_bootstrap(((p - y) ** 2).tolist(), pooled_gids)
    auc_boot = _auc_cluster_bootstrap(y, p, pooled_gids)

    mi_inconsistent, mi_detail = _ats_total_materially_inconsistent(records)
    lo, hi = ABSOLUTE_PROBABILITY_QUALITY_GATE["calibration_slope_range"]
    booleans = {
        "pooled_log_loss_below_ln2": bool(ll < LN2),
        "pooled_log_loss_ci_upper_below_ln2": bool(_num(ll_boot["ci_high"]) is not None and ll_boot["ci_high"] < LN2),
        "pooled_brier_below_0_250": bool(br < 0.250),
        "pooled_brier_ci_upper_below_0_250": bool(_num(br_boot["ci_high"]) is not None and br_boot["ci_high"] < 0.250),
        "pooled_ece_at_most_0_050": bool(_num(ece) is not None and _num(ece) <= 0.050),
        "calibration_slope_in_range": bool(_num(slope) is not None and lo <= _num(slope) <= hi),
        "abs_calibration_intercept_at_most_0_05": bool(_num(intercept) is not None and abs(_num(intercept)) <= 0.05),
        "auc_ci_lower_above_0_50": bool(auc_boot is not None and _num(auc_boot["ci_low"]) is not None
                                       and auc_boot["ci_low"] > 0.50),
        "ats_total_not_materially_inconsistent": not mi_inconsistent,
        "promotion_eligible_maturity": _promotion_eligible(n_games),
    }
    met = all(booleans.values())
    status = "MET_STRONGLY" if met else ("MODERATE" if maturity != MATURITY_INSUFFICIENT else "NOT_DEMONSTRATED")
    return {
        "gate": name,
        "status": status,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {
            "pooled_log_loss": ll, "pooled_brier": br,
            "pooled_ece": _num(ece), "calibration_slope": _num(slope),
            "calibration_intercept": _num(intercept), "auc": _num(auc),
            "sharpness_descriptive": sharpness(p),
            "n_scored_rows": int(len(y)), "per_market_n": per_market_n,
            "n_calibrated_scored_rows": int(len(y)),
            "per_market_calibrated_scored_rows": dict(per_market_n),
            "per_market_rows_without_calibrated_probability": per_market_rows_without_calibrated,
            "ats_total_materially_inconsistent": mi_inconsistent,
            "ats_total_materially_inconsistent_detail": mi_detail,
        },
        "confidence_interval": {
            "pooled_log_loss": [ll_boot["ci_low"], ll_boot["ci_high"]],
            "pooled_brier": [br_boot["ci_low"], br_boot["ci_high"]],
            "auc": [auc_boot["ci_low"], auc_boot["ci_high"]] if auc_boot else None,
        },
        "gate_booleans": booleans,
        "failure_reasons": [k for k, v in booleans.items() if not v],
        "promotion_eligible": _promotion_eligible(n_games),
    }


def _auc_cluster_bootstrap(y: np.ndarray, p: np.ndarray, gids: list[str]) -> dict | None:
    if len(y) == 0:
        return None
    gid_arr = np.asarray([str(g) for g in gids])
    uniq = np.unique(gid_arr)
    rows_by_game = {g: np.where(gid_arr == g)[0] for g in uniq}
    point = auc_diagnostic(y, p)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_REPS):
        drawn = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_game[uniq[d]] for d in drawn])
        val = auc_diagnostic(y[idx], p[idx])
        if val != NOT_ESTIMABLE:
            boot.append(float(val))
    if not boot:
        return None
    return {
        "point_estimate": _num(point),
        "ci_low": float(np.percentile(boot, _CI_LOW_PCT)),
        "ci_high": float(np.percentile(boot, _CI_HIGH_PCT)),
    }


def _ats_total_materially_inconsistent(records: list[dict]) -> tuple[bool, dict]:
    """Contract hardening item 6 -- the frozen definition of "materially
    inconsistent" for the ABSOLUTE_PROBABILITY_QUALITY gate. ATS-pooled and
    TOTAL-pooled are materially inconsistent with an ALL_FOUR MET_STRONGLY
    conclusion if EITHER market has ANY of:

        log loss >= 0.6931471805599453 (ln 2)
        Brier    >= 0.250
        AUC      <= 0.50
        ECE      >  0.075

    Both ATS pooled and TOTAL pooled must avoid all four for ALL_FOUR
    absolute quality to become MET_STRONGLY. Returns ``(inconsistent, detail)``."""
    detail: dict = {}
    inconsistent = False
    for market in _MARKETS:
        rows = _stream_rows(records, market, None)
        # Blocker remediation 2: CALIBRATED rows only -- no p_raw fallback.
        p = rows["p_cal"]
        keep = np.isfinite(p)
        if keep.sum() == 0:
            detail[market] = {"n": 0}
            continue
        y = rows["y"][keep]
        pp = p[keep]
        ll = binary_log_loss(y, pp)
        br = brier(y, pp)
        ece = fixed_bin_ece(y, pp)
        auc = auc_diagnostic(y, pp)
        flags = {
            "log_loss_at_or_above_ln2": bool(_num(ll) is not None and ll >= LN2),
            "brier_at_or_above_0_250": bool(_num(br) is not None and br >= 0.250),
            "auc_at_or_below_0_50": bool(_num(auc) is not None and _num(auc) <= 0.50),
            "ece_above_0_075": bool(_num(ece) is not None and _num(ece) > 0.075),
        }
        detail[market] = {
            "n": int(keep.sum()),
            "log_loss": _num(ll), "brier": _num(br),
            "ece": _num(ece), "auc": _num(auc),
            "flags": flags,
        }
        if any(flags.values()):
            inconsistent = True
    return inconsistent, detail


def gate_point_forecasting_vs_sportsbook(records: list[dict], maturity: str, n_games: int) -> dict:
    name = "POINT_FORECASTING_VS_SPORTSBOOK"
    ats = _stream_rows(records, "ATS", None)
    tot = _stream_rows(records, "TOTAL", None)
    if len(ats["margin_sq_model"]) == 0 or len(tot["total_sq_model"]) == 0:
        return _blank_gate(name, maturity, n_games, "no paired point-forecast rows yet")

    m_delta = ats["margin_sq_model"] - ats["margin_sq_market"]
    t_delta = tot["total_sq_model"] - tot["total_sq_market"]
    m_rmse_model = float(np.sqrt(ats["margin_sq_model"].mean()))
    m_rmse_book = float(np.sqrt(ats["margin_sq_market"].mean()))
    t_rmse_model = float(np.sqrt(tot["total_sq_model"].mean()))
    t_rmse_book = float(np.sqrt(tot["total_sq_market"].mean()))
    m_boot = game_cluster_bootstrap(m_delta.tolist(), ats["point_game_ids"])
    t_boot = game_cluster_bootstrap(t_delta.tolist(), tot["point_game_ids"])
    per_horizon = _per_horizon_point_rmse(records)

    booleans = {
        "pooled_margin_rmse_beats_book_by_0_25": bool(m_rmse_model <= m_rmse_book - 0.25),
        "pooled_margin_sq_error_delta_ci_upper_below_zero": bool(_num(m_boot["ci_high"]) is not None
                                                                and m_boot["ci_high"] < 0.0),
        "pooled_total_rmse_beats_book_by_0_25": bool(t_rmse_model <= t_rmse_book - 0.25),
        "pooled_total_sq_error_delta_ci_upper_below_zero": bool(_num(t_boot["ci_high"]) is not None
                                                               and t_boot["ci_high"] < 0.0),
        "no_horizon_worse_than_book_by_more_than_0_10": per_horizon["within_tolerance"],
        "promotion_eligible_maturity": _promotion_eligible(n_games),
    }
    met = all(booleans.values())
    status = "MET_STRONGLY" if met else ("NOT_MET" if maturity != MATURITY_INSUFFICIENT else "NOT_DEMONSTRATED")
    return {
        "gate": name,
        "status": status,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {
            "margin_rmse_model": m_rmse_model, "margin_rmse_book": m_rmse_book,
            "total_rmse_model": t_rmse_model, "total_rmse_book": t_rmse_book,
            "margin_sq_error_delta": m_boot["point_estimate"], "total_sq_error_delta": t_boot["point_estimate"],
            "per_horizon": per_horizon["detail"],
        },
        "confidence_interval": {
            "margin_sq_error_delta": [m_boot["ci_low"], m_boot["ci_high"]],
            "total_sq_error_delta": [t_boot["ci_low"], t_boot["ci_high"]],
        },
        "gate_booleans": booleans,
        "failure_reasons": [k for k, v in booleans.items() if not v],
        "promotion_eligible": _promotion_eligible(n_games),
    }


def _per_horizon_point_rmse(records: list[dict]) -> dict:
    detail = {}
    ok = True
    for horizon in _HORIZONS:
        ats = _stream_rows(records, "ATS", horizon)
        tot = _stream_rows(records, "TOTAL", horizon)
        d = {}
        for label, model_sq, market_sq in (
            ("margin", ats["margin_sq_model"], ats["margin_sq_market"]),
            ("total", tot["total_sq_model"], tot["total_sq_market"]),
        ):
            if len(model_sq) == 0:
                d[label] = None
                continue
            rm = float(np.sqrt(model_sq.mean()))
            rb = float(np.sqrt(market_sq.mean()))
            d[label] = {"rmse_model": rm, "rmse_book": rb, "model_minus_book": rm - rb}
            if rm - rb > 0.10:
                ok = False
        detail[horizon] = d
    return {"within_tolerance": ok, "detail": detail}


def gate_sportsbook_probability_edge(records: list[dict], maturity: str, n_games: int) -> dict:
    name = "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE"
    per_market: dict[str, dict] = {}
    any_rows = False
    for market in _MARKETS:
        pooled_ll_delta, pooled_br_delta, gids = [], [], []
        rows_without_calibrated = 0
        constituents = {}
        for horizon in _HORIZONS:
            rows = _stream_rows(records, market, horizon)
            # Blocker remediation 2: the sportsbook probability-edge gate
            # scores the CALIBRATED production probability only -- no p_raw
            # fallback for a CALIBRATION_NOT_READY row.
            p = rows["p_cal"]
            keep = np.isfinite(p) & np.isfinite(rows["p_market"])
            rows_without_calibrated += int((np.isfinite(rows["p_market"]) & ~np.isfinite(p)).sum())
            y = rows["y"][keep]
            if len(y) == 0:
                constituents[f"{market}_{horizon}"] = {"n": 0}
                continue
            pm = rows["p_market"][keep]
            pmod = p[keep]
            g = [gg for gg, k in zip(rows["game_ids"], keep) if k]
            ll_delta_rows = _pointwise_logloss(y, pmod) - _pointwise_logloss(y, pm)
            br_delta_rows = (pmod - y) ** 2 - (pm - y) ** 2
            pooled_ll_delta.extend(ll_delta_rows.tolist())
            pooled_br_delta.extend(br_delta_rows.tolist())
            gids.extend(g)
            constituents[f"{market}_{horizon}"] = {
                "n": int(len(y)),
                "log_loss_delta": float(ll_delta_rows.mean()),
                "brier_delta": float(br_delta_rows.mean()),
            }
        if not gids:
            per_market[market] = {"n": 0, "status": "INSUFFICIENT", "constituents": constituents,
                                  "n_calibrated_scored_rows": 0,
                                  "rows_with_market_but_no_calibrated_probability": rows_without_calibrated}
            continue
        any_rows = True
        ll_boot = game_cluster_bootstrap(pooled_ll_delta, gids)
        br_boot = game_cluster_bootstrap(pooled_br_delta, gids)
        adverse = [
            s for s, v in constituents.items()
            if v.get("n", 0) > 0 and (v["log_loss_delta"] > 0.002 or v["brier_delta"] > 0.001)
        ]
        booleans = {
            "log_loss_delta_at_most_neg_0_002": bool(_num(ll_boot["point_estimate"]) is not None
                                                     and ll_boot["point_estimate"] <= -0.002),
            "log_loss_delta_ci_upper_below_zero": bool(_num(ll_boot["ci_high"]) is not None and ll_boot["ci_high"] < 0.0),
            "brier_delta_at_most_neg_0_001": bool(_num(br_boot["point_estimate"]) is not None
                                                 and br_boot["point_estimate"] <= -0.001),
            "brier_delta_ci_upper_below_zero": bool(_num(br_boot["ci_high"]) is not None and br_boot["ci_high"] < 0.0),
            "no_constituent_stream_adverse": len(adverse) == 0,
            "promotion_eligible_maturity": _promotion_eligible(n_games),
        }
        per_market[market] = {
            "n": int(len(gids)),
            "n_calibrated_scored_rows": int(len(gids)),
            "rows_with_market_but_no_calibrated_probability": rows_without_calibrated,
            "log_loss_delta": ll_boot["point_estimate"],
            "log_loss_delta_ci": [ll_boot["ci_low"], ll_boot["ci_high"]],
            "brier_delta": br_boot["point_estimate"],
            "brier_delta_ci": [br_boot["ci_low"], br_boot["ci_high"]],
            "adverse_constituents": adverse,
            "gate_booleans": booleans,
            "constituents": constituents,
        }

    if not any_rows:
        out = _blank_gate(name, maturity, n_games, "no probability-vs-market rows yet")
        out["per_market"] = per_market
        return out

    both_markets = all(m in per_market and per_market[m].get("n", 0) > 0 for m in _MARKETS)
    met = both_markets and all(all(per_market[m]["gate_booleans"].values()) for m in _MARKETS)
    status = "MET_STRONGLY" if met else ("NOT_DEMONSTRATED")
    reasons = []
    if not both_markets:
        reasons.append("both_markets_not_yet_scored")
    for m in _MARKETS:
        if per_market.get(m, {}).get("gate_booleans"):
            reasons += [f"{m}:{k}" for k, v in per_market[m]["gate_booleans"].items() if not v]
    return {
        "gate": name,
        "status": status,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {m: {"log_loss_delta": per_market[m].get("log_loss_delta"),
                               "brier_delta": per_market[m].get("brier_delta")}
                           for m in per_market},
        "confidence_interval": {m: {"log_loss_delta": per_market[m].get("log_loss_delta_ci"),
                                    "brier_delta": per_market[m].get("brier_delta_ci")}
                                for m in per_market},
        "gate_booleans": {"both_markets_scored": both_markets, "no_ats_total_pooling_to_rescue": True},
        "failure_reasons": reasons,
        "per_market": per_market,
        "promotion_eligible": _promotion_eligible(n_games),
    }


def gate_model_family_stability(shadow_records: list[dict], maturity: str, n_games: int) -> dict:
    name = "MODEL_FAMILY_STABILITY"
    n_shadow = len(shadow_records)
    out = _blank_gate(name, maturity, n_games, "prospective shadow model-family ledger not yet populated"
                      if n_shadow == 0 else "shadow evidence below PROMOTION_ELIGIBLE maturity")
    out["shadow_record_count"] = n_shadow
    out["candidates_tracked"] = list(FROZEN_FIX7_SHADOW_CANDIDATES)
    out["note"] = (
        "No shadow result has production-selection authority during 2026. "
        "Canonical production forecast stays RIDGE_ALPHA_100 regardless of this row."
    )
    if not _promotion_eligible(n_games) or n_shadow == 0:
        out["status"] = CURRENT_LABELS[name]  # preserve MIXED until prospective evidence legitimately changes it
    return out


def gate_profitability(price_records: list[dict], book_policy: dict, maturity: str, n_games: int,
                       settled_non_push_wagers: int, *, policy_lock: dict | None = None,
                       contaminated_estate: bool = False) -> dict:
    name = "PROVEN_PROFITABLE_BETTING_MODEL"
    policy_lock = policy_lock or {"locked": False, "conflict": False, "flag": None}
    # Contract hardening item 8 + blocker remediation 3 -- MET_STRONGLY is
    # permanently barred if the live lock conflicts OR if ANY conflict event
    # has ever been persisted in the append-only integrity-event ledger
    # (contamination survives a config / hash revert; there is no clear path).
    integrity_conflict = bool(policy_lock.get("conflict")) or bool(contaminated_estate)
    booleans = {
        "executable_book_policy_frozen": bool(book_policy.get("frozen")),
        "betting_rule_activated_for_profitability": False,  # Section 13: DISABLED until book policy frozen
        "min_settled_non_push_wagers": settled_non_push_wagers >= PROFITABILITY_GATE["min_settled_non_push_wagers"],
        "min_unique_completed_prospective_games": _promotion_eligible(n_games),
        "promotion_eligible_maturity": _promotion_eligible(n_games),
        "no_betting_integrity_conflict": not integrity_conflict,
        "no_persisted_betting_integrity_conflict_event": not bool(contaminated_estate),
    }
    status = CURRENT_LABELS[name]  # NOT_ESTABLISHED
    failure_reasons: list[str] = []
    if integrity_conflict:
        failure_reasons.append(BETTING_INTEGRITY_CONFLICT)
    if not book_policy.get("frozen"):
        failure_reasons.append(book_policy.get("reason", "EXECUTABLE_BOOK_POLICY_NOT_FROZEN"))
    else:
        failure_reasons.append("INSUFFICIENT_PROSPECTIVE_BETTING_EVIDENCE")
    return {
        "gate": name,
        "status": status,
        "betting_rule_status": BETTING_RULE_STATUS_NOT_ACTIVATED,
        "betting_rule_2026_v1_hash": BETTING_RULE_2026_V1_HASH,
        "betting_integrity_conflict": integrity_conflict,
        "betting_integrity_estate_contaminated": bool(contaminated_estate),
        "contaminated_estate_met_strongly_barred": integrity_conflict,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {"settled_non_push_wagers": settled_non_push_wagers,
                           "executable_price_ledger_rows": len(price_records)},
        "confidence_interval": None,
        "gate_booleans": booleans,
        "failure_reasons": failure_reasons,
        "executable_book_policy": book_policy,
        "executable_book_policy_lock": policy_lock,
        "market_specific_non_adverse_rule": dict(PROFITABILITY_MARKET_NON_ADVERSE_RULE),
        "promotion_eligible": _promotion_eligible(n_games),
    }


def gate_operational(name: str, manifests: list[dict], eligible: list[dict], excluded: list[dict],
                     maturity: str, n_games: int, *, calibrated_status_violations: int = 0) -> dict:
    """CHRONOLOGY_LEAKAGE_CONTROL / REPRODUCIBILITY_AUDITABILITY /
    PRODUCTION_READY_PIPELINE -- operational gates whose evidence is the
    ledger/manifest estate itself. They stay at their current MET_STRONGLY
    label only while zero violations are observed; any violation demotes."""
    violations: list[str] = []
    if name == "REPRODUCIBILITY_AUDITABILITY":
        if excluded:
            non_market = [e for e in excluded if e["reason"] not in _NON_PROVENANCE_EXCLUSION_REASONS]
            if non_market:
                violations.append(f"{len(non_market)} record(s) missing required provenance")
    if name == "PRODUCTION_READY_PIPELINE":
        bad = [m for m in manifests
               if m.get("status") not in ("SUCCESS",) and m.get("status") not in _DOCUMENTED_FAIL_CLOSED]
        if bad:
            violations.append(f"{len(bad)} run manifest(s) with an undocumented status")
        if calibrated_status_violations:
            violations.append(
                f"{calibrated_status_violations} row(s) carried a numeric calibrated_* probability "
                f"while calibration_status != CALIBRATED (fail-closed contract violation)"
            )
    status = CURRENT_LABELS[name] if not violations else "DEMOTED_VIOLATION_OBSERVED"
    return {
        "gate": name,
        "status": status,
        "sample_size": n_games,
        "evidence_stage": maturity,
        "point_estimate": {"run_manifests_seen": len(manifests),
                           "eligible_forecast_records": len(eligible),
                           "excluded_records": len(excluded)},
        "confidence_interval": None,
        "gate_booleans": {"zero_violations_observed": not violations},
        "failure_reasons": violations,
        "promotion_eligible": _promotion_eligible(n_games),
    }


_DOCUMENTED_FAIL_CLOSED = (
    "NOT_DUE", "SCHEDULE_UNAVAILABLE", "GAME_RESULT_SOURCE_UNAVAILABLE", "ELO_SOURCE_UNAVAILABLE",
    "MARKET_SOURCE_UNAVAILABLE", "MARKET_NOT_READY", "MODEL_NOT_READY", "UNCERTAINTY_NOT_READY",
    "CALIBRATION_NOT_READY", "SCHEMA_DRIFT", "IDENTIFIER_FAILURE", "HASH_MISMATCH",
    "FORECAST_IMMUTABILITY_VIOLATION",
)


# ===========================================================================
# Top-level scorecard assembly.
# ===========================================================================
def build_scorecard(
    *,
    forecast_records: list[dict] | None = None,
    evaluation_records: list[dict] | None = None,
    shadow_records: list[dict] | None = None,
    price_records: list[dict] | None = None,
    run_manifests: list[dict] | None = None,
    book_policy: dict | None = None,
    book_policy_lock: dict | None = None,
    integrity_events: list[dict] | None = None,
    schedule_population: dict | None = None,
    data_through_utc: str | None = None,
) -> dict:
    """Apply the frozen contract to whatever immutable prospective evidence
    exists. Never raises on an empty estate; never returns MET / MET_STRONGLY
    for a row with no supporting prospective evidence (Section 19).

    Contract hardening: ``evaluation_records`` are filtered to the frozen
    season-2026 REG+POST population (PRESEASON excluded) and to forecasts
    generated inside the certified production execution window before any
    scoring. ``book_policy_lock`` is the season executable-book policy lock
    assessment (:func:`verify_executable_book_policy_lock`); a
    ``BETTING_INTEGRITY_CONFLICT`` in it permanently bars profitability
    MET_STRONGLY. ``integrity_events`` is the append-only betting-policy
    integrity-event ledger (:func:`load_betting_integrity_events`); ANY valid
    persisted conflict event permanently contaminates the 2026 estate --
    profitability can never be MET_STRONGLY thereafter, even if the config or
    the original policy/betting-rule hash is later restored.
    ``schedule_population`` -- when supplied and complete -- is the only way
    ``SEASON_FINAL_CONFIRMATORY`` can be reported."""
    forecast_records = forecast_records or []
    evaluation_records = evaluation_records or []
    shadow_records = shadow_records or []
    price_records = price_records or []
    run_manifests = run_manifests or []
    book_policy = book_policy or {"frozen": False, "reason": "EXECUTABLE_BOOK_POLICY_NOT_FROZEN"}
    betting_estate = betting_integrity_estate_status(events=integrity_events or [])

    split = classify_evidence_eligibility(evaluation_records)
    eligible = split["eligible"]
    excluded = split["excluded"]
    completed_ids = _completed_game_ids(eligible)
    n_games = len(completed_ids)
    maturity = sample_maturity(n_games)
    prereg = frozen_preregistration_document()
    prereg_hash = prereg[_HASH_FIELD]

    settled_non_push = sum(1 for r in price_records if r.get("settled") and r.get("push") is False)

    # Fail-closed contract check across every stream (Sections 5 / 24): a
    # numeric ``calibrated_*`` probability paired with a non-CALIBRATED status
    # is a producer violation the PRODUCTION_READY_PIPELINE gate must catch.
    calibrated_status_violations = 0
    for _m in _MARKETS:
        calibrated_status_violations += len(_stream_rows(eligible, _m, None)["calibrated_status_violations"])

    rows = {
        "CALIBRATION_IMPROVES_RAW_PROBABILITIES": gate_calibration_improves_raw(eligible, maturity, n_games),
        "ABSOLUTE_PROBABILITY_QUALITY": gate_absolute_probability_quality(eligible, maturity, n_games),
        "POINT_FORECASTING_VS_SPORTSBOOK": gate_point_forecasting_vs_sportsbook(eligible, maturity, n_games),
        "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE": gate_sportsbook_probability_edge(eligible, maturity, n_games),
        "MODEL_FAMILY_STABILITY": gate_model_family_stability(shadow_records, maturity, n_games),
        "PROVEN_PROFITABLE_BETTING_MODEL": gate_profitability(
            price_records, book_policy, maturity, n_games, settled_non_push, policy_lock=book_policy_lock,
            contaminated_estate=betting_estate["contaminated_estate"]),
        "CHRONOLOGY_LEAKAGE_CONTROL": gate_operational("CHRONOLOGY_LEAKAGE_CONTROL", run_manifests, eligible, excluded, maturity, n_games),
        "REPRODUCIBILITY_AUDITABILITY": gate_operational("REPRODUCIBILITY_AUDITABILITY", run_manifests, eligible, excluded, maturity, n_games),
        "PRODUCTION_READY_PIPELINE": gate_operational(
            "PRODUCTION_READY_PIPELINE", run_manifests, eligible, excluded, maturity, n_games,
            calibrated_status_violations=calibrated_status_violations,
        ),
    }

    # Section 19 + 23. The reporter never *mints* MET / MET_STRONGLY from an
    # immature sample: a performance gate can only be promoted at
    # >= PROMOTION_ELIGIBLE maturity with every gate boolean satisfied.
    # Otherwise the row keeps its current evidence-supported label (Section
    # 23), which is historical context, not a fresh prospective claim.
    # Operational gates (chronology / reproducibility / pipeline) keep their
    # certified MET_STRONGLY label until a violation is *observed* in the
    # prospective estate, in which case they are demoted.
    reported: dict[str, dict] = {}
    for key, row in rows.items():
        current = CURRENT_LABELS[key]
        gate_status = row["status"]
        promoted = gate_status == "MET_STRONGLY" and bool(row.get("promotion_eligible"))
        if _is_operational(key):
            reported_status = "DEMOTED_VIOLATION_OBSERVED" if row.get("failure_reasons") else current
        elif promoted:
            reported_status = "MET_STRONGLY"
        elif key == "CALIBRATION_IMPROVES_RAW_PROBABILITIES":
            reported_status = f"{current} ({CALIBRATION_HISTORICAL_NOTE})"
        else:
            reported_status = current
            # defensive: a performance gate must never surface a bare
            # MET/MET_STRONGLY status without an actual promotion.
            if gate_status in ("MET", "MET_STRONGLY"):
                row["status"] = "NOT_DEMONSTRATED"
        row["reported_status"] = reported_status
        row["current_evidence_supported_label"] = current
        row["promotion_basis"] = (
            "prospective_gate_all_booleans_satisfied" if promoted
            else "preserved_current_evidence_supported_label_section_23"
        )
        row["preregistration_hash"] = prereg_hash
        row["data_through_utc"] = data_through_utc
        reported[key] = row

    # Contract hardening item 9 -- SEASON_FINAL_CONFIRMATORY is proven ONLY
    # from a complete canonical 2026 REG+POST schedule population, never from
    # a calendar date or a sample count.
    season_final = season_final_confirmatory(schedule_population, completed_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "preregistration_hash": prereg_hash,
        "data_through_utc": data_through_utc,
        "sample": {
            "unique_completed_prospective_games": n_games,
            "maturity": maturity,
            "population": {"season": PROSPECTIVE_SEASON, "season_types": list(PROSPECTIVE_SEASON_TYPES)},
            "season_final_confirmatory": "SEASON_FINAL_CONFIRMATORY" if season_final else None,
            "season_final_confirmatory_proven": bool(season_final),
            "eligible_evaluation_records": len(eligible),
            "excluded_records": excluded,
            "excluded_out_of_population": [e for e in excluded if e["reason"] == "OUT_OF_PROSPECTIVE_POPULATION"],
            "excluded_outside_execution_window": [
                e for e in excluded if e["reason"] == "FORECAST_OUTSIDE_EXECUTION_WINDOW"
            ],
            "forecast_ledger_records": len(forecast_records),
            "shadow_ledger_records": len(shadow_records),
            "executable_price_ledger_records": len(price_records),
            "run_manifests": len(run_manifests),
        },
        "betting_integrity_estate": {
            "contaminated_estate": betting_estate["contaminated_estate"],
            "conflict_event_count": betting_estate["conflict_event_count"],
            "conflict_event_types_seen": sorted({
                e.get("event_type") for e in betting_estate["conflict_events"]
            }),
            "no_automatic_clear_contamination_path": True,
        },
        "rows": reported,
        "current_evidence_supported_labels": dict(CURRENT_LABELS),
        "notes": [
            "Historical status shown for context only; never a gate input.",
            "No row is promoted merely because the reporter was added.",
            "Empty / immature sample can never yield MET or MET_STRONGLY.",
            "Population frozen: season 2026, season_type in {REG, POST}; PRESEASON excluded.",
            "SEASON_FINAL_CONFIRMATORY requires a proven-complete 2026 REG+POST schedule population.",
        ],
    }


def _is_operational(key: str) -> bool:
    return key in ("CHRONOLOGY_LEAKAGE_CONTROL", "REPRODUCIBILITY_AUDITABILITY", "PRODUCTION_READY_PIPELINE")


def render_scorecard_markdown(scorecard: dict) -> str:
    s = scorecard
    lines = [
        "# Prospective 2026 Status Scorecard",
        "",
        f"- schema: `{s['schema_version']}`",
        f"- preregistration hash: `{s['preregistration_hash']}`",
        f"- data through: `{s['data_through_utc'] or 'NONE (no prospective evidence attached yet)'}`",
        f"- unique completed prospective games: **{s['sample']['unique_completed_prospective_games']}**",
        f"- sample maturity: **{s['sample']['maturity']}**",
        "",
        "| Row | Reported status | Gate status | Evidence stage | Sample | Point estimate | 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, row in s["rows"].items():
        pe = row.get("point_estimate")
        ci = row.get("confidence_interval")
        lines.append(
            f"| {key} | {row['reported_status']} | {row['status']} | {row['evidence_stage']} "
            f"| {row['sample_size']} | {_fmt(pe)} | {_fmt(ci)} |"
        )
    lines += ["", "## Failure reasons (why each row is not promoted)", ""]
    for key, row in s["rows"].items():
        reasons = row.get("failure_reasons") or ["(promoted / no blocking reason)"]
        lines.append(f"- **{key}**: " + "; ".join(str(r) for r in reasons))
    lines += ["", "## Notes", ""] + [f"- {n}" for n in s["notes"]]
    return "\n".join(lines) + "\n"


def _fmt(obj) -> str:
    if obj is None:
        return "—"
    text = json.dumps(obj, sort_keys=True, default=str)
    return text if len(text) <= 90 else text[:87] + "…"
