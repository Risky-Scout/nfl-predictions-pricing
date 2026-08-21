"""Fix 4: the single authoritative chronological calibration framework.

A historical target prediction may be calibrated ONLY from earlier
chronological OOF predictions (Fix 3,
:mod:`nfl_hybrid.evaluation.chronological_oof`) whose own outcome was already
resolved/available before the target's own forecast cutoff:

    calibration_prediction.result_available_at_utc < target_cutoff_utc

never merely ``prediction_time < target_cutoff_utc``. This is exactly the
same result-availability correction Fix 3 already applies to its own
training-set eligibility (kickoff order is not resolution order -- see the
module docstring of :mod:`chronological_oof`); Fix 4 applies the identical
rule one layer up, to calibration-set eligibility.

Design, reused rather than reinvented (this fix runs no calibration-method
tournament and selects no new calibrator family):

    Fix 3 OOF residual/uncertainty ledger (the only calibration input)
        -> build_raw_probabilities   -- turn each eligible OOF row's Fix 3
                                         quantity (predicted_margin /
                                         margin_residual_sd_oof for MONEYLINE
                                         and ATS; predicted_total /
                                         total_residual_sd_oof for TOTAL) into
                                         a three-way (home, push, away)
                                         probability via
                                         :mod:`nfl_hybrid.pricing.distribution_pricing`
                                         (the one canonical pricing definition,
                                         itself a thin wrapper around
                                         :class:`nfl_hybrid.pricing.margin_surface.MarginSurface`,
                                         method="normal", fixed -- not
                                         selected), thresholded at the
                                         market's own line (margin 0 for
                                         MONEYLINE; the declared spread for
                                         ATS; the declared total for TOTAL --
                                         see :data:`_MARKET_QUANTITY_COLUMNS`
                                         and the REQUIRED ``market``/
                                         ``forecast_horizon``/``threshold``
                                         parameters -- ``market`` has no
                                         default; see item 1 below), and a
                                         nullable binary target via
                                         :func:`nfl_hybrid.labels.edge_to_nullable_binary`
                                         (the repository's one tie/push
                                         policy). This is the only
                                         market-specific stage: everything
                                         downstream of it consumes abstract
                                         (upper/push/lower) probabilities and
                                         a binary label, never a market
                                         concept.
        -> generate_chronological_calibration -- expanding-window chronological
                                         calibration: for each target row, fit
                                         the reused conditional calibrator
                                         (:func:`nfl_hybrid.calibration.three_way._fit_conditional_calibrator`,
                                         a logistic regression on
                                         logit(raw_conditional_upper_probability))
                                         on ONLY the eligible calibration set,
                                         then recombine with the raw push
                                         probability via
                                         (:func:`nfl_hybrid.calibration.three_way._recombine`)
                                         into a calibrated three-way
                                         probability -- the same three-way
                                         (lower/push/upper) representation
                                         already used by the existing
                                         calibration family, never a bespoke
                                         two-outcome one.

Nothing here selects a calibrator family, tunes the predictive model, selects
features, or runs a calibration-method tournament: the estimator
(``_fit_conditional_calibrator``) and its default configuration
(:class:`nfl_hybrid.calibration.three_way.CalibrationConfig`) are the
repository's existing, already-merged calibration family, reused verbatim
with a different (stricter, row-level) eligibility rule than its original
season-level one (:func:`nfl_hybrid.calibration.three_way.apply_expanding_three_way_calibration`
fits per test *season*; this module fits per unique *target forecast
cutoff*, mirroring Fix 3's own batching-by-shared-cutoff optimization).

No train=calibration fallback: when a target's eligible calibration sample is
empty, or the reused calibrator itself declines to fit (its own internal
floor: fewer than 100 labeled points, or fewer than 2 classes present -- see
``_fit_conditional_calibrator``), the row is marked
``UNCALIBRATED_INSUFFICIENT_HISTORY`` and its calibrated probability is left
NaN. The raw probability is always preserved regardless of calibration
status.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_hybrid.calibration.three_way import (
    CalibrationConfig,
    _fit_conditional_calibrator,
    _fit_push_scales,
    _predict_conditional,
    _predict_push,
    _recombine,
)
from nfl_hybrid.data import external_data
from nfl_hybrid.data.availability import assert_available_before
from nfl_hybrid.evaluation.chronological_oof import training_membership_hash
from nfl_hybrid.labels import edge_to_nullable_binary
from nfl_hybrid.pricing.distribution_pricing import price_ats_distribution, price_total_distribution
from nfl_hybrid.pricing.margin_surface import MarginSurface

# The reused calibrator family itself, named for the provenance record --
# never selected from a tournament (there is exactly one family in use).
CALIBRATOR_FAMILY = "three_way.logistic_on_logit_conditional_calibrator"

# Market semantics build_raw_probabilities supports. This is the whole
# abstraction Fix 4 needed to stop being implicitly moneyline-only: which
# Fix 3 quantity (margin or total) feeds the probability surface, and what
# threshold (line) the three-way outcome is evaluated against. The
# calibration machinery downstream of this (generate_chronological_calibration,
# and the reused three_way calibrator family) is already market-agnostic --
# it only ever sees abstract (upper/push/lower) probabilities and a nullable
# binary target, never "home"/"away" semantics -- so no change was needed
# there. "home"/"away" column names are kept (not renamed to "upper"/"lower")
# to avoid an unrelated schema churn across the ledger, tests, and the proof
# script; for ATS "home"/"away" mean home-covers/away-covers, and for TOTAL
# they mean OVER/UNDER, exactly the same discretized-normal three-way split
# used for moneyline, just with a different mean quantity and threshold.
MARKET_MONEYLINE = "MONEYLINE"  # away win / tie / home win, threshold = margin 0 (fixed)
MARKET_ATS = "ATS"  # away cover / push / home cover, threshold = declared spread
MARKET_TOTAL = "TOTAL"  # under / push / over, threshold = declared total
_VALID_MARKETS = (MARKET_MONEYLINE, MARKET_ATS, MARKET_TOTAL)

# market -> (mean-quantity column, OOF-uncertainty sigma column, actual-outcome
# column) read from the Fix 3 residual/uncertainty ledger. MONEYLINE and ATS
# are both margin markets; TOTAL is the only market that reads the total
# quantities instead of the margin ones.
_MARKET_QUANTITY_COLUMNS: dict[str, tuple[str, str, str]] = {
    MARKET_MONEYLINE: ("predicted_margin", "margin_residual_sd_oof", "actual_margin"),
    MARKET_ATS: ("predicted_margin", "margin_residual_sd_oof", "actual_margin"),
    MARKET_TOTAL: ("predicted_total", "total_residual_sd_oof", "actual_total"),
}

# Push-mass calibration policy labels (certification-blocker repair item 3).
# The default/fallback: push mass is left exactly as build_raw_probabilities
# computed it (the model-implied distributional push mass), never touched.
PUSH_CALIBRATION_POLICY_RAW = "RAW_DISTRIBUTIONAL_PUSH_MASS"
# Applied only where calibrate_chronological_push_probability() actually ran
# a fit -- the EXISTING fixed three_way._fit_push_scales/_predict_push
# family, on the identical prior-eligible (result_available_at_utc <
# target_cutoff_utc) membership Fix 4's conditional calibration already
# established. Never a new calibrator family, never season-level chronology.
PUSH_CALIBRATION_POLICY_CHRONOLOGICAL = "CHRONOLOGICAL_FIXED_PUSH_SCALE_FAMILY"

# nfl_hybrid.calibration.three_way's push-bucket logic (_push_bucket) keys
# off the repository's LEGACY market-name strings, not Fix 4's MARKET_*
# constants -- map between them here, in one place, rather than duplicating
# the legacy names throughout this module.
_LEGACY_PUSH_MARKET_NAME: dict[str, str] = {
    MARKET_MONEYLINE: "pregame_moneyline",
    MARKET_ATS: "pregame_ats",
    MARKET_TOTAL: "pregame_total",
}

RAW_CALIBRATION_INPUT_COLUMNS = (
    "game_id",
    "season",
    "week",
    # First-class row data (Fix 4 item 10), never inferred solely from
    # model_config_hash: the forecast horizon this row's prediction/market
    # line was actually captured at (e.g. "kickoff_minus_10_minutes"). Every
    # row of one build_raw_probabilities() call carries exactly the same
    # value, supplied by the caller -- this module does not infer it.
    "forecast_horizon",
    "target_cutoff_utc",
    "result_available_at_utc",
    "model_config_hash",
    "feature_state_hash",
    "raw_home_probability",
    "raw_push_probability",
    "raw_away_probability",
    "raw_conditional_upper_probability",
    "binary_target",
    "raw_status",
)

CALIBRATION_LEDGER_COLUMNS = (
    "game_id",
    "season",
    "week",
    "forecast_horizon",
    "target_cutoff_utc",
    # As with chronological_oof.py's own training_as_of_utc: the as-of
    # timestamp actually used to decide THIS row's calibration-set
    # eligibility. Currently always equal to target_cutoff_utc, kept as an
    # explicit, separately-named field rather than reusing target_cutoff_utc
    # for both meanings.
    "calibration_as_of_utc",
    "result_available_at_utc",
    "model_config_hash",
    "feature_state_hash",
    "calibration_sample_count",
    "calibration_prediction_ids",
    "calibration_prediction_ids_truncated",
    "calibration_membership_hash",
    "max_calibration_result_available_at_utc",
    "calibrator_family",
    "calibrator_config_hash",
    "raw_home_probability",
    "raw_push_probability",
    "raw_away_probability",
    "raw_conditional_upper_probability",
    "calibrated_home_probability",
    "calibrated_push_probability",
    "calibrated_away_probability",
    # The calibrator's own fitted conditional (non-push) output, BEFORE
    # recombination with the raw push mass -- item 12's "calibrated:
    # conditional_home_cover" / "conditional_over". NaN whenever
    # calibration_status != "CALIBRATED" (never derived post hoc from the
    # recombined three-way probabilities, which would double-apply the push
    # mass).
    "calibrated_conditional_upper_probability",
    "calibration_status",
)


@dataclass(frozen=True)
class ChronologicalCalibrationConfig:
    """The single fixed, explicitly declared calibrator/configuration for
    chronological calibration. Nothing here is tuned, searched, or selected
    against held-out performance -- ``calibration_config`` is the pre-existing
    :class:`CalibrationConfig` defaults already used by the merged three-way
    calibration family (:mod:`nfl_hybrid.calibration.three_way`)."""

    calibration_config: CalibrationConfig = field(default_factory=CalibrationConfig)
    # Lineage-style cap on how many calibration game_ids are enumerated in the
    # ledger row itself (mirrors chronological_oof.py's training_ids_cap).
    # The count and every timestamp are never truncated -- only the id list.
    calibration_ids_cap: int = 250
    # A target's own model_config_hash (from the Fix 3 OOF ledger) already
    # encodes target_cutoff_offset_minutes -- the one forecast-horizon knob
    # Fix 3 exposes -- so distinct model_config_hash values are distinct
    # forecast horizons. Refuse to silently pool them into one calibration
    # population unless explicitly configured to.
    pool_across_horizons: bool = False


@dataclass(frozen=True)
class ChronologicalCalibrationResult:
    raw: pd.DataFrame
    ledger: pd.DataFrame
    calibrator_family: str
    calibrator_config_hash: str


def compute_calibrator_config_hash(config: ChronologicalCalibrationConfig) -> str:
    payload = json.dumps(
        {
            "calibrator_family": CALIBRATOR_FAMILY,
            "calibration_config": asdict(config.calibration_config),
            "calibration_ids_cap": config.calibration_ids_cap,
            "pool_across_horizons": config.pool_across_horizons,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_raw_probabilities(
    residual_ledger: pd.DataFrame,
    *,
    market: str,
    forecast_horizon: str,
    threshold: pd.Series | np.ndarray | float | None = None,
    surface: MarginSurface | None = None,
) -> pd.DataFrame:
    """Turn each eligible row of a genuine Fix 3 OOF residual/uncertainty
    ledger (:func:`nfl_hybrid.evaluation.chronological_oof.run_chronological_oof`'s
    ``residual_ledger``, or the direct output of
    ``attach_expanding_oof_uncertainty``) into a raw three-way
    (home, push, away) probability, never a bespoke two-outcome one.

    ``market`` is a REQUIRED keyword-only argument with no default -- the Fix
    4 repair for a real defect: an earlier real-data proof script called this
    function without ``market=``, which silently priced MONEYLINE when ATS or
    TOTAL was intended. There is no longer a default to fall back to; omitting
    ``market`` is a ``TypeError`` at the Python call site, before any pricing
    logic runs (see ``tests/test_chronological_calibration.py::test_market_is_a_required_keyword_argument``).
    It selects which of the three supported market semantics this call
    prices, via :data:`_MARKET_QUANTITY_COLUMNS`:

    - ``MARKET_MONEYLINE``: away win / tie / home win. Threshold is fixed at
      margin 0 -- passing an explicit ``threshold`` is rejected.
    - ``MARKET_ATS``: away cover / push / home cover. ``threshold`` is
      REQUIRED -- the declared spread line, per row (a scalar broadcasts to
      every row, or an array-like the same length as ``residual_ledger``).
      Home covers when ``actual_margin + threshold > 0``, matching
      :mod:`nfl_hybrid.pricing.margin_surface`'s own documented convention.
    - ``MARKET_TOTAL``: under / push / over. ``threshold`` is REQUIRED -- the
      declared total line, per row. Over when ``actual_total > threshold``.
      Reads the *total* Fix 3 quantities (``predicted_total`` /
      ``total_residual_sd_oof`` / ``actual_total``), never the margin ones.

    ``forecast_horizon`` is likewise required (Fix 4 item 10): a caller-
    supplied label (e.g. ``"kickoff_minus_10_minutes"``) for the one forecast
    cutoff convention every row of this call was priced under. It is stamped
    onto every output row as first-class data -- not left implicit in
    ``model_config_hash`` alone -- so :func:`generate_chronological_calibration`
    can refuse to silently pool distinct horizons by default (see
    :attr:`ChronologicalCalibrationConfig.pool_across_horizons`).

    In every case the underlying probability surface is the single canonical
    :mod:`nfl_hybrid.pricing.distribution_pricing` definition
    (:func:`~nfl_hybrid.pricing.distribution_pricing.price_ats_distribution` /
    :func:`~nfl_hybrid.pricing.distribution_pricing.price_total_distribution`)
    -- :class:`nfl_hybrid.modern.joint_score.JointScoreModel`'s own
    continuity-corrected formula, extracted verbatim (not routed through
    :class:`nfl_hybrid.pricing.margin_surface.MarginSurface`'s PMF
    discretization, which is NOT numerically identical for an arbitrary
    non-integer line -- see that module's docstring). Exact parity with
    JointScoreModel holds for ANY finite real threshold, including
    fractional consensus lines a median-of-an-even-number-of-books
    calculation can produce, not only integers/half-points
    (``tests/test_distribution_pricing.py``). This function selects which
    mean quantity and which threshold feed it; it does not add a second
    pricing method.

    A row is eligible (``raw_status == "RAW_READY"``) only if Fix 3 actually
    produced a prediction for it (``status == "OOF"``) AND Fix 3's own
    expanding OOF uncertainty was available for it (``uncertainty_eligible``)
    -- the OOF-uncertainty sigma is read from Fix 3's own column, never from a
    fitted model's own training-residual sigma (the same rule Fix 3 itself
    enforces one layer down). Rows that never got a Fix 3 prediction
    (``INSUFFICIENT_WARMUP``) or that lack OOF uncertainty are marked
    ``RAW_UNAVAILABLE`` and carry NaN raw probabilities -- they are preserved
    in the output, never dropped, so the calibration ledger produced from
    this frame accounts for every input row.

    The binary calibration target is
    :func:`nfl_hybrid.labels.edge_to_nullable_binary` applied to the market's
    own decision edge (``actual_margin + threshold`` for MONEYLINE/ATS,
    ``actual_total - threshold`` for TOTAL) -- the repository's one tie/push
    policy: a tied/pushed game is ``pd.NA``, never class zero.
    """
    if market not in _VALID_MARKETS:
        raise ValueError(f"Unknown market {market!r}; expected one of {_VALID_MARKETS}.")
    if not forecast_horizon:
        raise ValueError("forecast_horizon is required and must be non-empty.")
    mean_col, sigma_col, actual_col = _MARKET_QUANTITY_COLUMNS[market]

    required = {
        "game_id", "season", "week", "target_cutoff_utc", "result_available_at_utc",
        "model_config_hash", "feature_state_hash", "status", mean_col,
        sigma_col, "uncertainty_eligible", actual_col,
    }
    missing = required - set(residual_ledger.columns)
    if missing:
        raise ValueError(
            f"residual_ledger is missing columns required to build raw "
            f"probabilities for market {market!r}: {sorted(missing)}"
        )

    surf = surface or MarginSurface(method="normal")
    frame = residual_ledger.reset_index(drop=True).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    n = len(frame)

    if market == MARKET_MONEYLINE:
        if threshold is not None:
            raise ValueError(
                "MARKET_MONEYLINE's threshold is fixed at margin 0; do not "
                "pass an explicit threshold (use MARKET_ATS for a declared "
                "spread line)."
            )
        threshold_arr = np.zeros(n, dtype=float)
    else:
        if threshold is None:
            raise ValueError(f"market {market!r} requires an explicit threshold (declared line per row).")
        if np.isscalar(threshold):
            threshold_arr = np.full(n, float(threshold))
        else:
            threshold_arr = pd.to_numeric(
                pd.Series(threshold).reset_index(drop=True), errors="coerce"
            ).to_numpy(float)
            if len(threshold_arr) != n:
                raise ValueError(
                    f"threshold has length {len(threshold_arr)}, expected {n} "
                    "(one per residual_ledger row)."
                )

    ready = (
        (frame["status"] == "OOF")
        & frame["uncertainty_eligible"].astype(bool)
        & frame[mean_col].notna()
        & frame[sigma_col].notna()
        & np.isfinite(threshold_arr)
    ).to_numpy()

    mean_value = pd.to_numeric(frame[mean_col], errors="coerce").to_numpy(float)
    sigma = pd.to_numeric(frame[sigma_col], errors="coerce").to_numpy(float)

    home = np.full(n, np.nan)
    push = np.full(n, np.nan)
    away = np.full(n, np.nan)
    cond = np.full(n, np.nan)
    if ready.any():
        # The single canonical distribution-pricing definition (Fix 4 item
        # 5): MONEYLINE and ATS route through price_ats_distribution (line ==
        # 0 for MONEYLINE, the declared spread for ATS); TOTAL routes through
        # price_total_distribution, which is itself documented as the same
        # core computation with the line negated and home/away relabeled
        # over/under. Never a bespoke third formula here.
        if market == MARKET_TOTAL:
            h, p, a = price_total_distribution(
                mean_value[ready], sigma[ready], threshold_arr[ready], surface=surf
            )
        else:
            h, p, a = price_ats_distribution(
                mean_value[ready], sigma[ready], threshold_arr[ready], surface=surf
            )
        home[ready], push[ready], away[ready] = h, p, a
        denom = h + a
        cond[ready] = np.where(denom > 0, h / denom, 0.5)

    actual = pd.to_numeric(frame[actual_col], errors="coerce")
    edge = (actual - threshold_arr) if market == MARKET_TOTAL else (actual + threshold_arr)

    out = pd.DataFrame(
        {
            "game_id": frame["game_id"],
            "season": frame["season"],
            "week": frame["week"],
            "forecast_horizon": forecast_horizon,
            "target_cutoff_utc": pd.to_datetime(frame["target_cutoff_utc"], utc=True, errors="raise"),
            "result_available_at_utc": pd.to_datetime(
                frame["result_available_at_utc"], utc=True, errors="raise"
            ),
            "model_config_hash": frame["model_config_hash"],
            "feature_state_hash": frame["feature_state_hash"],
            "raw_home_probability": home,
            "raw_push_probability": push,
            "raw_away_probability": away,
            "raw_conditional_upper_probability": cond,
            "binary_target": edge_to_nullable_binary(edge),
            "raw_status": np.where(ready, "RAW_READY", "RAW_UNAVAILABLE"),
        }
    )
    return out[list(RAW_CALIBRATION_INPUT_COLUMNS)]


def generate_chronological_calibration(
    raw: pd.DataFrame,
    *,
    config: ChronologicalCalibrationConfig | None = None,
    target_row_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Phase 2: strictly chronological calibration of raw probabilities.

    ``target_row_positions`` mirrors
    :func:`nfl_hybrid.evaluation.chronological_oof.generate_oof_predictions`'s
    parameter of the same name: restrict which rows (positions in the
    ``(target_cutoff_utc, game_id)``-sorted frame) actually get calibrated,
    while every row -- including ones outside this set -- remains eligible as
    calibration history for the ones that are. Omit it (the default) to
    calibrate every ``RAW_READY`` row.

    Rows are processed strictly in ``(target_cutoff_utc, game_id)`` order
    (re-sorted here regardless of input row order, so shuffled input never
    changes the result), and a candidate calibration prediction's ELIGIBILITY
    is decided by ``result_available_at_utc < target_cutoff_utc(G)`` -- the
    exact Fix 4 invariant, never ``prediction_time < target_cutoff_utc``.
    Because every row's own ``result_available_at_utc`` is always strictly
    after that same row's own ``target_cutoff_utc`` (Fix 3's own invariant,
    inherited unchanged by ``build_raw_probabilities``), a row can never
    enter its own calibration set -- self-exclusion is structural, not
    incidental.

    Batching (a performance optimization only, never a chronology change):
    every row that shares the exact same ``target_cutoff_utc`` also shares
    the exact same calibration-eligibility mask, so a same-slate batch of
    targets is calibrated from exactly one calibrator fit, mirroring Fix 3's
    own batching-by-shared-cutoff.

    No train=calibration fallback: if a batch's eligible calibration sample
    is empty, or the reused calibrator declines to fit (see module
    docstring), every row in that batch is marked
    ``UNCALIBRATED_INSUFFICIENT_HISTORY`` with a NaN calibrated probability
    -- never silently fit on the calibration set itself.
    """
    cfg = config or ChronologicalCalibrationConfig()
    calibrator_config_hash = compute_calibrator_config_hash(cfg)

    missing = set(RAW_CALIBRATION_INPUT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw calibration input is missing columns: {sorted(missing)}")

    frame = raw.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise ValueError("raw calibration input has duplicate game_id rows.")

    # Fix 4 item 10: forecast_horizon is the explicit, first-class row-level
    # pooling gate -- never inferred solely from model_config_hash (which is
    # also checked, as a second, independent signal: two rows sharing a
    # forecast_horizon label but disagreeing on model_config_hash are still a
    # real config mismatch worth catching).
    horizons = sorted(str(h) for h in frame["forecast_horizon"].dropna().unique())
    model_configs = sorted(str(h) for h in frame["model_config_hash"].dropna().unique())
    if (len(horizons) > 1 or len(model_configs) > 1) and not cfg.pool_across_horizons:
        raise ValueError(
            "raw calibration input mixes distinct forecast horizons "
            f"({horizons}) or model configs ({model_configs}) without "
            "pool_across_horizons=True; refusing to silently pool distinct "
            "forecast horizons into one calibration population."
        )

    target_cutoff = pd.to_datetime(frame["target_cutoff_utc"], utc=True, errors="raise")
    result_available = pd.to_datetime(frame["result_available_at_utc"], utc=True, errors="raise")
    if (result_available < target_cutoff).any():
        raise ValueError(
            "result_available_at_utc must never precede a row's own "
            "target_cutoff_utc (see chronological_oof's own invariant)."
        )

    ready_mask = (frame["raw_status"] == "RAW_READY").to_numpy()
    binary_numeric = pd.to_numeric(frame["binary_target"], errors="coerce").to_numpy(float)
    labeled_mask = ready_mask & np.isfinite(binary_numeric)
    conditional_upper = frame["raw_conditional_upper_probability"].to_numpy(dtype=float)
    push = frame["raw_push_probability"].to_numpy(dtype=float)
    home = frame["raw_home_probability"].to_numpy(dtype=float)
    away = frame["raw_away_probability"].to_numpy(dtype=float)

    positions = range(len(frame)) if target_row_positions is None else sorted(set(target_row_positions))

    # Only RAW_READY rows can ever be calibrated; group the rest for a
    # pass-through record below instead.
    batches: dict[pd.Timestamp, list[int]] = {}
    passthrough: list[int] = []
    for i in positions:
        if ready_mask[i]:
            batches.setdefault(target_cutoff.iloc[i], []).append(i)
        else:
            passthrough.append(i)

    rows: list[dict[str, object]] = []
    for cutoff_i, group_positions in sorted(batches.items(), key=lambda kv: kv[0]):
        train_mask = labeled_mask & (result_available < cutoff_i).to_numpy()
        train_ids = frame.loc[train_mask, "game_id"].tolist()
        count = len(train_ids)
        max_available = result_available.loc[train_mask].max() if count else pd.NaT
        membership_hash = training_membership_hash(train_ids)
        truncated = count > cfg.calibration_ids_cap
        stored_ids = () if truncated else tuple(sorted(train_ids))

        base_record = {
            "target_cutoff_utc": cutoff_i,
            "calibration_as_of_utc": cutoff_i,
            "calibration_sample_count": count,
            "calibration_prediction_ids": stored_ids,
            "calibration_prediction_ids_truncated": truncated,
            "calibration_membership_hash": membership_hash,
            "max_calibration_result_available_at_utc": max_available,
            "calibrator_family": CALIBRATOR_FAMILY,
            "calibrator_config_hash": calibrator_config_hash,
        }

        model = None
        if count > 0:
            x_train = conditional_upper[train_mask]
            y_train = binary_numeric[train_mask].astype(int)
            with threadpool_limits(limits=1):
                model = _fit_conditional_calibrator(x_train, y_train, cfg.calibration_config)

        if model is not None:
            group_cond = conditional_upper[group_positions]
            group_push = push[group_positions]
            calibrated_cond = _predict_conditional(model, group_cond, cfg.calibration_config)
            lower_arr, push_arr, upper_arr = _recombine(calibrated_cond, group_push)

        for slot, i in enumerate(group_positions):
            record: dict[str, object] = {
                "game_id": frame.at[i, "game_id"],
                "season": frame.at[i, "season"],
                "week": frame.at[i, "week"],
                "forecast_horizon": frame.at[i, "forecast_horizon"],
                "result_available_at_utc": result_available.iloc[i],
                "model_config_hash": frame.at[i, "model_config_hash"],
                "feature_state_hash": frame.at[i, "feature_state_hash"],
                "raw_home_probability": float(home[i]),
                "raw_push_probability": float(push[i]),
                "raw_away_probability": float(away[i]),
                "raw_conditional_upper_probability": float(conditional_upper[i]),
                **base_record,
            }
            if model is None:
                record["calibration_status"] = "UNCALIBRATED_INSUFFICIENT_HISTORY"
                record["calibrated_home_probability"] = float("nan")
                record["calibrated_push_probability"] = float("nan")
                record["calibrated_away_probability"] = float("nan")
                record["calibrated_conditional_upper_probability"] = float("nan")
            else:
                record["calibration_status"] = "CALIBRATED"
                record["calibrated_home_probability"] = float(upper_arr[slot])
                record["calibrated_push_probability"] = float(push_arr[slot])
                record["calibrated_away_probability"] = float(lower_arr[slot])
                record["calibrated_conditional_upper_probability"] = float(calibrated_cond[slot])
            rows.append(record)

    empty_hash = training_membership_hash([])
    for i in passthrough:
        rows.append(
            {
                "game_id": frame.at[i, "game_id"],
                "season": frame.at[i, "season"],
                "week": frame.at[i, "week"],
                "forecast_horizon": frame.at[i, "forecast_horizon"],
                "target_cutoff_utc": frame.at[i, "target_cutoff_utc"],
                "calibration_as_of_utc": frame.at[i, "target_cutoff_utc"],
                "result_available_at_utc": result_available.iloc[i],
                "model_config_hash": frame.at[i, "model_config_hash"],
                "feature_state_hash": frame.at[i, "feature_state_hash"],
                "calibration_sample_count": 0,
                "calibration_prediction_ids": (),
                "calibration_prediction_ids_truncated": False,
                "calibration_membership_hash": empty_hash,
                "max_calibration_result_available_at_utc": pd.NaT,
                "calibrator_family": CALIBRATOR_FAMILY,
                "calibrator_config_hash": calibrator_config_hash,
                "raw_home_probability": float("nan"),
                "raw_push_probability": float("nan"),
                "raw_away_probability": float("nan"),
                "raw_conditional_upper_probability": float("nan"),
                "calibrated_home_probability": float("nan"),
                "calibrated_push_probability": float("nan"),
                "calibrated_away_probability": float("nan"),
                "calibrated_conditional_upper_probability": float("nan"),
                "calibration_status": "RAW_UNAVAILABLE",
            }
        )

    ledger = pd.DataFrame(rows, columns=list(CALIBRATION_LEDGER_COLUMNS))
    ledger = ledger.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)

    # Self-check, using the repository's own existing availability-invariant
    # helper (nfl_hybrid.data.availability), not a bespoke one: every
    # CALIBRATED row's calibration set resolved strictly before that row's
    # own cutoff.
    calibrated_rows = ledger[ledger["calibration_status"] == "CALIBRATED"]
    assert_available_before(
        calibrated_rows,
        available_at_column="max_calibration_result_available_at_utc",
        prediction_time_column="target_cutoff_utc",
        allow_equal=False,
    )
    return ledger


def compute_push_calibration_config_hash(config: CalibrationConfig, market: str) -> str:
    payload = json.dumps(
        {"push_calibration_config": asdict(config), "market": market},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibrate_chronological_push_probability(
    raw: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    market: str,
    market_line: pd.Series | np.ndarray,
    config: CalibrationConfig | None = None,
) -> pd.DataFrame:
    """Certification-blocker repair item 3: chronologically calibrate the
    raw push probability too, so the final unconditional three-way
    distribution is fully calibrated, not just its conditional half.

    Reuses the EXISTING fixed push-calibration family verbatim
    (:func:`nfl_hybrid.calibration.three_way._fit_push_scales` /
    :func:`~._predict_push`) -- never
    :func:`~nfl_hybrid.calibration.three_way.apply_expanding_three_way_calibration`,
    whose season-level chronology is not Fix 4's row-level
    result-availability contract. For each row this engine already marked
    ``CALIBRATED``, the push-scale family is fit ONLY on the identical prior
    rows sharing that row's own ``target_cutoff_utc`` batch eligibility
    (``result_available_at_utc < target_cutoff_utc``, RAW_READY, same
    market/horizon by construction of ``raw``) -- exactly the same
    membership Fix 4's conditional calibration already established for that
    batch, not a separately-derived one. No future outcomes, no target
    outcome (self-exclusion is structural, identical to the conditional
    engine, for the identical reason), no cross-market or cross-horizon
    observations.

    Recombines the newly-calibrated push probability with the ALREADY
    calibrated conditional probability
    (``calibrated_conditional_upper_probability``) via the same
    :func:`~nfl_hybrid.calibration.three_way._recombine` rule the conditional
    engine itself uses -- never a second recombination formula.

    Rows the conditional engine left ``UNCALIBRATED_INSUFFICIENT_HISTORY`` or
    ``RAW_UNAVAILABLE`` are untouched: push calibration never calibrates a row
    the conditional engine didn't. Returns a NEW ledger (does not mutate the
    input); adds ``push_calibration_policy``,
    ``push_calibration_membership_count``, and ``push_calibration_config_hash``
    columns, and overwrites ``calibrated_push_probability`` /
    ``calibrated_home_probability`` / ``calibrated_away_probability`` for
    every row it actually calibrated push for.
    """
    if market not in _LEGACY_PUSH_MARKET_NAME:
        raise ValueError(f"Unknown market {market!r}; expected one of {sorted(_LEGACY_PUSH_MARKET_NAME)}.")
    cfg = config or CalibrationConfig()
    legacy_market = _LEGACY_PUSH_MARKET_NAME[market]
    config_hash = compute_push_calibration_config_hash(cfg, market)

    market_line_series = pd.to_numeric(pd.Series(market_line).reset_index(drop=True), errors="coerce")
    if len(market_line_series) != len(raw):
        raise ValueError(
            f"market_line has length {len(market_line_series)}, expected {len(raw)} (one per raw row, same order)."
        )
    # market_line is supplied against raw's ORIGINAL row order (the same
    # order it was passed as build_raw_probabilities' own threshold= -- raw's
    # own index is exactly that positional order, per build_raw_probabilities'
    # reset_index(drop=True)); reindex it to match the (target_cutoff_utc,
    # game_id)-sorted order generate_chronological_calibration itself uses.
    sort_positions = raw.sort_values(["target_cutoff_utc", "game_id"], kind="stable").index
    line = market_line_series.loc[sort_positions].reset_index(drop=True)

    frame = raw.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    frame["game_id"] = frame["game_id"].astype(str)

    target_cutoff = pd.to_datetime(frame["target_cutoff_utc"], utc=True, errors="raise")
    result_available = pd.to_datetime(frame["result_available_at_utc"], utc=True, errors="raise")
    ready_mask = (frame["raw_status"] == "RAW_READY").to_numpy()
    binary_numeric = pd.to_numeric(frame["binary_target"], errors="coerce").to_numpy(float)
    # A RAW_READY row with a null binary_target is exactly a push (see
    # nfl_hybrid.labels.edge_to_nullable_binary) -- the actual-push indicator
    # push-scale fitting needs, available for every RAW_READY row regardless
    # of whether it was "labeled" for conditional calibration (conditional
    # calibration's own labeled_mask deliberately EXCLUDES pushes; push
    # fitting needs them included).
    actual_push = (ready_mask & ~np.isfinite(binary_numeric)).astype(int)
    model_push = pd.to_numeric(frame["raw_push_probability"], errors="coerce").to_numpy(float)

    push_frame = pd.DataFrame(
        {"model_push_probability": model_push, "actual_push": actual_push, "market_line": line.to_numpy(float)}
    )

    out = ledger.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(drop=True).copy()
    out["game_id"] = out["game_id"].astype(str)
    ledger_pos_by_gid = {gid: pos for pos, gid in enumerate(out["game_id"])}

    calibrated_push = np.full(len(out), np.nan)
    membership_count = np.zeros(len(out), dtype=int)
    policy = np.full(len(out), PUSH_CALIBRATION_POLICY_RAW, dtype=object)

    batches: dict[pd.Timestamp, list[int]] = {}
    for i in range(len(frame)):
        if ready_mask[i]:
            batches.setdefault(target_cutoff.iloc[i], []).append(i)

    with threadpool_limits(limits=1):
        for cutoff_i, positions in sorted(batches.items(), key=lambda kv: kv[0]):
            # Which of these rows the conditional engine actually calibrated
            # (only those are eligible for push calibration too).
            calibratable = [
                i
                for i in positions
                if frame.at[i, "game_id"] in ledger_pos_by_gid
                and out.at[ledger_pos_by_gid[frame.at[i, "game_id"]], "calibration_status"] == "CALIBRATED"
            ]
            if not calibratable:
                continue
            train_mask = ready_mask & (result_available < cutoff_i).to_numpy()
            n_train = int(train_mask.sum())
            if n_train == 0:
                continue
            train = push_frame.loc[train_mask]
            global_scale, bucket_scales = _fit_push_scales(train, legacy_market, cfg)
            target_frame = push_frame.iloc[calibratable].reset_index(drop=True)
            predicted = _predict_push(target_frame, legacy_market, global_scale, bucket_scales, cfg)
            for slot, i in enumerate(calibratable):
                pos = ledger_pos_by_gid[frame.at[i, "game_id"]]
                calibrated_push[pos] = float(predicted[slot])
                membership_count[pos] = n_train
                policy[pos] = PUSH_CALIBRATION_POLICY_CHRONOLOGICAL

    out["push_calibration_policy"] = policy
    out["push_calibration_membership_count"] = membership_count
    out["push_calibration_config_hash"] = config_hash

    push_calibrated_mask = policy == PUSH_CALIBRATION_POLICY_CHRONOLOGICAL
    if push_calibrated_mask.any():
        cond = out.loc[push_calibrated_mask, "calibrated_conditional_upper_probability"].to_numpy(float)
        new_push = calibrated_push[push_calibrated_mask]
        lower, push_final, upper = _recombine(cond, new_push)
        out.loc[push_calibrated_mask, "calibrated_push_probability"] = push_final
        out.loc[push_calibrated_mask, "calibrated_home_probability"] = upper
        out.loc[push_calibrated_mask, "calibrated_away_probability"] = lower

    return out


def run_chronological_calibration(
    residual_ledger: pd.DataFrame,
    *,
    market: str,
    forecast_horizon: str,
    threshold: pd.Series | np.ndarray | float | None = None,
    config: ChronologicalCalibrationConfig | None = None,
    surface: MarginSurface | None = None,
    target_row_positions: Sequence[int] | None = None,
) -> ChronologicalCalibrationResult:
    """End-to-end orchestrator: Fix 3 OOF residual/uncertainty ledger -> raw
    three-way probabilities -> chronological calibration. The only entry
    point that wires both stages together; each stage is independently
    callable/testable above. ``market`` and ``forecast_horizon`` are
    required, with the identical no-default rationale as
    :func:`build_raw_probabilities` (Fix 4 item 1) -- this orchestrator must
    not reintroduce a silent-MONEYLINE default one level up."""
    cfg = config or ChronologicalCalibrationConfig()
    raw = build_raw_probabilities(
        residual_ledger, market=market, forecast_horizon=forecast_horizon, threshold=threshold, surface=surface
    )
    ledger = generate_chronological_calibration(raw, config=cfg, target_row_positions=target_row_positions)
    return ChronologicalCalibrationResult(
        raw=raw,
        ledger=ledger,
        calibrator_family=CALIBRATOR_FAMILY,
        calibrator_config_hash=compute_calibrator_config_hash(cfg),
    )


def persist_calibration_ledger(
    result: ChronologicalCalibrationResult,
    *,
    extra_manifest: Mapping[str, object] | None = None,
    root_override: str | Path | None = None,
) -> dict[str, Path]:
    """Persist the calibration raw-probability/ledger frames through the
    registered external-data artifact architecture
    (:mod:`nfl_hybrid.data.external_data`) -- never a hard-coded,
    machine-specific path. These are ``scope="generated"`` registry keys:
    they resolve under ``NFL_MODEL_ARTIFACT_ROOT`` (or ``root_override``),
    never ``NFL_MODEL_DATA_ROOT`` -- generated pipeline output is
    deliberately never written into the raw/canonical historical estate's own
    tree."""
    written: dict[str, Path] = {}

    raw_path = external_data.resolve_for_write(
        "chronological_calibration.raw_probabilities", root_override=root_override
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    result.raw.to_parquet(raw_path, index=False)
    written["chronological_calibration.raw_probabilities"] = raw_path

    ledger_path = external_data.resolve_for_write(
        "chronological_calibration.ledger", root_override=root_override
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    result.ledger.to_parquet(ledger_path, index=False)
    written["chronological_calibration.ledger"] = ledger_path

    manifest = {
        "calibrator_family": result.calibrator_family,
        "calibrator_config_hash": result.calibrator_config_hash,
        "n_raw_rows": int(len(result.raw)),
        "n_raw_ready": int((result.raw["raw_status"] == "RAW_READY").sum()),
        "n_ledger_rows": int(len(result.ledger)),
        "n_calibrated": int((result.ledger["calibration_status"] == "CALIBRATED").sum()),
        "n_uncalibrated_insufficient_history": int(
            (result.ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").sum()
        ),
        "n_raw_unavailable": int((result.ledger["calibration_status"] == "RAW_UNAVAILABLE").sum()),
        **dict(extra_manifest or {}),
    }
    manifest_path = external_data.resolve_for_write(
        "chronological_calibration.manifest", root_override=root_override
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    written["chronological_calibration.manifest"] = manifest_path

    return written
