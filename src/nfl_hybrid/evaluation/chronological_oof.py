"""Fix 3: the single authoritative chronological out-of-fold (OOF) prediction
and uncertainty engine.

Every historical target game gets its margin/total prediction from a model
trained only on games whose RESULT was already available before its own
forecast cutoff -- ``result_available_at_utc < target_cutoff_utc`` -- with
preprocessing fit on that same strictly-prior training set. This is a
correction to the module's first version, which used
``scheduled_kickoff_utc < target_cutoff_utc`` instead: kickoff order is not
result-availability order (a game can kick off earlier and still resolve, or
be confirmed, later than another game's forecast cutoff -- see
:func:`compute_result_available_at_utc`). The prediction is recorded before
the game's own outcome is ever attached (:func:`generate_oof_predictions`
never reads ``home_margin``/``total_points`` for the row it is predicting);
outcomes and residuals are attached afterward, in a separate pass
(:func:`attach_outcomes_and_residuals`).

Margin/total predictive uncertainty (residual SD, residual correlation) is
never estimated from a model's own fitted-training residuals -- that would be
the same leakage-adjacent mistake Fix 2 fixed for pregame state, one level up
the pipeline. :class:`nfl_hybrid.modern.joint_score.JointScoreModel` computes
``margin_sd_`` / ``total_sd_`` / ``residual_correlation_`` from its own
training residuals for its own internal probability math; this module never
reads those three attributes (see :meth:`JointScoreModel.predict_means`) and
instead estimates uncertainty in :func:`attach_expanding_oof_uncertainty` and
:func:`production_uncertainty` exclusively from this module's own OOF residual
ledger, expanding strictly backward in time.

Design, reused rather than reinvented:

    build_authoritative_pregame_state (Fix 2)     -- the only state source
        -> build_oof_feature_matrix                -- pivot to one game-level
                                                       feature row (this module)
        -> generate_oof_predictions                 -- expanding-window OOF
                                                       point predictions (this
                                                       module; reuses
                                                       JointScoreModel, the
                                                       same estimator already
                                                       used by
                                                       evaluation.walkforward)
        -> attach_outcomes_and_residuals            -- outcomes attached after
                                                       the prediction is fixed
        -> attach_expanding_oof_uncertainty         -- leave-future-out
                                                       residual SD/correlation

The estimator is fixed and declared once (:class:`ChronologicalOOFConfig`):
no feature selection, no hyperparameter search, no model-family tournament, no
probability calibration. Model fitting happens only as many times as there are
target games, each time on a fresh, unfitted model instance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_hybrid.data import external_data
from nfl_hybrid.data.availability import add_postgame_available_at, assert_available_before
from nfl_hybrid.features.pregame_rolling import build_game_pregame_matrix
from nfl_hybrid.features.pregame_state import (
    PregameStateBundle,
    build_authoritative_pregame_state,
)
from nfl_hybrid.modern.joint_score import JointScoreConfig, JointScoreModel

STATE_FAMILIES = ("epa", "opponent_adjusted", "qb", "elo", "market_history")

MODEL_NAME = "JointScoreModel[margin,total]"
MODEL_VERSION = "1.0"

# Conservative historical result-availability policy. Never a fabricated
# exact final-whistle timestamp -- two layers, each only able to DELAY
# eligibility, never advance it:
#   1. nfl_hybrid.data.availability.add_postgame_available_at (this
#      repository's own existing documented postgame-availability floor:
#      kickoff + duration_hours, reused unchanged -- already used by
#      augmented_matrix.py's historical_replay mode).
#   2. Snapped forward to the next production forecast-batch boundary
#      (Tuesday or Friday, 12:00 UTC -- mid-morning US Eastern, comfortably
#      after any Monday/Thursday night game's kickoff+floor). This mirrors
#      the batch cadence a live system would actually forecast on, at which
#      point every prior game -- including Monday Night Football, the
#      latest possible weekly kickoff -- is unambiguously final. It avoids
#      claiming a precise per-game completion time this repository's schema
#      cannot verify (no per-play UTC event time, no terminal-game marker;
#      see augmented_matrix.py's own ``_terminal_play_timestamp``).
RESULT_AVAILABILITY_DURATION_HOURS = 5.0
_BATCH_WEEKDAYS = (1, 4)  # Mon=0 ... Sun=6: Tuesday, Friday
_BATCH_ANCHOR_HOUR_UTC = 12

# Provenance label for how result_available_at_utc / max_training_result_available_at_utc
# were derived -- recorded on every ledger row so a reader never mistakes this
# for an observed final-whistle time. "HISTORICAL_CONSERVATIVE_BATCH" names
# the exact policy above: kickoff + RESULT_AVAILABILITY_DURATION_HOURS,
# snapped forward to the next Tue/Fri _BATCH_ANCHOR_HOUR_UTC boundary. There
# is currently exactly one derivation policy in use; this constant exists so
# a future second policy (e.g. a verified live finality feed) would be
# distinguishable in the ledger rather than silently indistinguishable from
# this conservative historical one.
RESULT_AVAILABILITY_BASIS = "HISTORICAL_CONSERVATIVE_BATCH"

OOF_LEDGER_COLUMNS = (
    "game_id",
    "season",
    "week",
    "target_cutoff_utc",
    # The as-of timestamp actually used to decide this batch's training-set
    # eligibility. Currently always equal to target_cutoff_utc (a batch's
    # training mask is evaluated against its own target cutoff), but kept as
    # an explicit, separately-named field rather than reusing
    # target_cutoff_utc for both meanings -- target_cutoff_utc is "when must
    # G's own forecast be ready", training_as_of_utc is "what as-of time did
    # the model-fit eligibility check actually use". The two concepts must
    # never be silently conflated even where their values coincide today.
    "training_as_of_utc",
    "result_available_at_utc",
    "max_training_result_available_at_utc",
    "result_availability_basis",
    "training_game_count",
    "training_game_ids",
    "training_game_ids_truncated",
    "training_membership_hash",
    "model_config_hash",
    "feature_state_hash",
    "status",
    "predicted_margin",
    "predicted_total",
)

RESIDUAL_LEDGER_COLUMNS = OOF_LEDGER_COLUMNS + (
    "actual_margin",
    "actual_total",
    "margin_residual",
    "total_residual",
)

UNCERTAINTY_LEDGER_COLUMNS = RESIDUAL_LEDGER_COLUMNS + (
    "margin_residual_sd_oof",
    "total_residual_sd_oof",
    "residual_correlation_oof",
    "uncertainty_eligible",
)


@dataclass(frozen=True)
class ChronologicalOOFConfig:
    """The single fixed, explicitly declared estimator/configuration for
    chronological OOF margin/total prediction.

    Nothing here is tuned, searched, or selected against held-out performance
    -- these are structural safety floors and the pre-existing
    :class:`JointScoreConfig` defaults already used elsewhere in this
    repository (:mod:`nfl_hybrid.evaluation.walkforward`).
    """

    model_config: JointScoreConfig = field(default_factory=JointScoreConfig)
    # Below this many strictly-prior games, a target is marked
    # INSUFFICIENT_WARMUP rather than fit on too little history.
    min_training_games: int = 48
    # Lineage-style cap on how many training game_ids are enumerated in the
    # ledger row itself (mirrors state_lineage.py's own truncation pattern).
    # The count and training_cutoff_utc are never truncated -- only the id list.
    training_ids_cap: int = 250
    # The repository's existing "official snapshot" convention
    # (governance/contracts.py: kickoff_minus_10_minutes).
    target_cutoff_offset_minutes: float = 10.0


@dataclass(frozen=True)
class ChronologicalOOFResult:
    predictions: pd.DataFrame
    residual_ledger: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_state_hash: str
    model_config_hash: str


def compute_model_config_hash(config: ChronologicalOOFConfig) -> str:
    payload = json.dumps(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_config": asdict(config.model_config),
            "min_training_games": config.min_training_games,
            "training_ids_cap": config.training_ids_cap,
            "target_cutoff_offset_minutes": config.target_cutoff_offset_minutes,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_feature_state_hash(feature_columns: Sequence[str], bundle: PregameStateBundle) -> str:
    transform_versions = sorted(
        {
            (str(lineage["transform_name"].iloc[0]), str(lineage["transform_version"].iloc[0]))
            for lineage in bundle.lineage.values()
            if len(lineage)
        }
    )
    payload = json.dumps(
        {"feature_columns": list(feature_columns), "transform_versions": transform_versions},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_membership_hash(game_ids: Sequence[str]) -> str:
    """Stable hash of a training set's membership, independent of input order."""
    joined = "|".join(sorted(str(g) for g in game_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _next_batch_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    """The next Tuesday-or-Friday 12:00 UTC boundary at or after ``ts``."""
    if pd.isna(ts):
        return ts
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    day_start = ts.normalize()
    for delta in range(8):  # a Tue and a Fri both occur within any 7-day span
        candidate = day_start + pd.Timedelta(days=delta, hours=_BATCH_ANCHOR_HOUR_UTC)
        if candidate.weekday() in _BATCH_WEEKDAYS and candidate >= ts:
            return candidate
    raise AssertionError("unreachable: no Tue/Fri boundary found within 8 days")


def compute_result_available_at_utc(
    games: pd.DataFrame,
    *,
    kickoff_column: str = "scheduled_kickoff_utc",
    duration_hours: float = RESULT_AVAILABILITY_DURATION_HOURS,
) -> pd.Series:
    """Conservative historical result-availability timestamp per game --
    never a fabricated exact final-whistle time (see the module docstring
    and the ``RESULT_AVAILABILITY_DURATION_HOURS`` comment above for the
    two-layer policy). Monotonic in ``kickoff_column``: this can only push a
    game's availability later, never earlier, than a naive
    kickoff-plus-duration floor."""
    floor = add_postgame_available_at(games, kickoff_column=kickoff_column, duration_hours=duration_hours)
    return floor.map(_next_batch_boundary)


def build_oof_feature_matrix(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    qb_game: pd.DataFrame | None = None,
    rolling_config=None,
    opponent_config=None,
    qb_config=None,
) -> tuple[pd.DataFrame, tuple[str, ...], PregameStateBundle]:
    """One game-level feature row per game, built exclusively from the Fix 2
    authoritative pregame-state bundle.

    No feature curation: every numeric home/away-pivoted column produced by
    every state family that was actually built is used, unchanged. This is a
    fixed, total-inclusion rule (declared once, here), not a selection
    procedure -- nothing is ranked, searched, or dropped for performance.

    Also attaches ``result_available_at_utc`` (via
    :func:`compute_result_available_at_utc`) -- required by
    :func:`generate_oof_predictions` for training eligibility. This is
    metadata, never a model feature: it is added to ``matrix`` directly, not
    through the per-family feature-column loop below.
    """
    if games["game_id"].astype(str).duplicated().any():
        raise ValueError("games table has duplicate game_id rows.")

    bundle = build_authoritative_pregame_state(
        games,
        play_by_play,
        qb_game=qb_game,
        rolling_config=rolling_config,
        opponent_config=opponent_config,
        qb_config=qb_config,
    )

    base = games.copy()
    base["game_id"] = base["game_id"].astype(str)
    base["scheduled_kickoff_utc"] = pd.to_datetime(
        base["scheduled_kickoff_utc"], utc=True, errors="raise"
    )
    base["home_margin"] = pd.to_numeric(base["home_score"], errors="coerce") - pd.to_numeric(
        base["away_score"], errors="coerce"
    )
    base["total_points"] = pd.to_numeric(base["home_score"], errors="coerce") + pd.to_numeric(
        base["away_score"], errors="coerce"
    )

    base["result_available_at_utc"] = compute_result_available_at_utc(base)

    matrix = base[
        [
            "game_id", "season", "week", "scheduled_kickoff_utc",
            "result_available_at_utc", "home_margin", "total_points",
        ]
    ].copy()

    feature_columns: list[str] = []
    for family in STATE_FAMILIES:
        state = bundle.state.get(family)
        if state is None or not len(state):
            continue
        pivoted = build_game_pregame_matrix(games, state)
        pivoted["game_id"] = pivoted["game_id"].astype(str)
        prefixed = [c for c in pivoted.columns if c.startswith("home_") or c.startswith("away_")]
        numeric = [c for c in prefixed if pd.api.types.is_numeric_dtype(pivoted[c])]
        renamed = {c: f"{family}__{c}" for c in numeric}
        family_frame = pivoted[["game_id"] + numeric].rename(columns=renamed)
        matrix = matrix.merge(family_frame, on="game_id", how="left", validate="one_to_one")
        feature_columns.extend(renamed.values())

    if not feature_columns:
        raise ValueError("No numeric pregame-state feature columns were produced by any family.")

    matrix[feature_columns] = matrix[feature_columns].astype("float64")
    matrix = matrix.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(
        drop=True
    )
    return matrix, tuple(feature_columns), bundle


def generate_oof_predictions(
    feature_matrix: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: ChronologicalOOFConfig | None = None,
    feature_state_hash: str = "",
    model_factory: Callable[[ChronologicalOOFConfig], JointScoreModel] | None = None,
    target_row_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Phase 1: chronological, expanding-window OOF point predictions.

    ``target_row_positions`` (optional): restrict which rows (positions in
    the chronologically-sorted frame) actually get a prediction generated,
    while every row -- including ones outside this set -- remains eligible as
    training history for the ones that are. This changes only which targets
    are *evaluated*, never which games are *eligible to train on* a given
    target; it exists so a representative real-data proof can exercise the
    full historical estate as a training pool without paying the cost of
    fitting a fresh model for every one of thousands of games. Omit it (the
    default) to generate a prediction for every row -- the actual production
    behavior.

    Games are processed strictly in ``(scheduled_kickoff_utc, game_id)``
    order (re-sorted here regardless of input row order, so shuffled input
    never changes the result), but a candidate training game's ELIGIBILITY
    is decided by ``result_available_at_utc < target_cutoff_utc(G)`` --
    *not* by kickoff order. Kickoff order is not result-availability order:
    a game can kick off earlier than another and still have its result
    become available later (weather delays, overtime, or simply a later
    production forecast-batch boundary -- see
    :func:`compute_result_available_at_utc`). ``feature_matrix`` must carry
    a ``result_available_at_utc`` column (:func:`build_oof_feature_matrix`
    attaches it); this raises if it is missing rather than silently falling
    back to kickoff. G itself, every future game, and every game whose
    result is not yet available by G's own cutoff are excluded by
    construction, not by convention.

    Batching (a performance optimization only, never a chronology change):
    every row that shares the exact same ``target_cutoff_utc`` also shares
    the exact same training mask (``result_available_at_utc < that cutoff``
    is evaluated against every OTHER row and does not depend on which row
    within the batch is being predicted). So a same-slate batch of games --
    the common NFL case, many games at one kickoff -- is fit exactly once
    and predicted together from that one fit, instead of once per game.
    This changes only how many times :meth:`JointScoreModel.fit` runs,
    never which games are eligible to train on which target: a game still
    trains only on games whose result was already available, and a
    batch-mate is still excluded from its own (shared) training set,
    exactly as it would be if fit individually -- true under the
    kickoff-order rule this replaces, and still true under the
    result-availability rule, since ``result_available_at_utc`` is always
    strictly after that same game's own ``target_cutoff_utc`` (it is
    computed from kickoff plus a positive floor, and ``target_cutoff_utc``
    is kickoff minus a positive offset).

    This function never reads ``home_margin`` / ``total_points`` for the row
    it is currently predicting. Outcomes are attached afterward, in
    :func:`attach_outcomes_and_residuals`, never here -- the prediction is
    already fixed by the time any outcome is consulted.
    """
    cfg = config or ChronologicalOOFConfig()
    model_config_hash = compute_model_config_hash(cfg)
    feature_columns = list(feature_columns)

    if "result_available_at_utc" not in feature_matrix.columns:
        raise ValueError(
            "feature_matrix must carry a result_available_at_utc column "
            "(see build_oof_feature_matrix / compute_result_available_at_utc); "
            "training eligibility is never inferred from scheduled_kickoff_utc alone."
        )

    frame = feature_matrix.sort_values(
        ["scheduled_kickoff_utc", "game_id"], kind="stable"
    ).reset_index(drop=True)
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise ValueError("feature matrix has duplicate game_id rows.")

    kickoff = pd.to_datetime(frame["scheduled_kickoff_utc"], utc=True, errors="raise")
    result_available = pd.to_datetime(frame["result_available_at_utc"], utc=True, errors="raise")
    if (result_available < kickoff).any():
        raise ValueError("result_available_at_utc must never precede scheduled_kickoff_utc.")
    offset = pd.Timedelta(minutes=cfg.target_cutoff_offset_minutes)
    target_cutoff = kickoff - offset

    factory = model_factory or (
        lambda c: JointScoreModel(feature_columns, (), c.model_config)
    )

    positions = range(len(frame)) if target_row_positions is None else sorted(set(target_row_positions))

    # Group requested positions by identical target_cutoff_utc. Positions are
    # visited in ascending order and ``frame`` is already sorted by
    # (kickoff, game_id), so this reproduces exactly the row order a
    # per-row loop would have produced -- batching is purely an
    # implementation detail of how many times a model gets fit.
    batches: dict[pd.Timestamp, list[int]] = {}
    for i in positions:
        batches.setdefault(target_cutoff.iloc[i], []).append(i)

    rows: list[dict[str, object]] = []
    for cutoff_i, group_positions in sorted(batches.items(), key=lambda kv: kv[0]):
        train_mask = (result_available < cutoff_i).to_numpy()
        train_ids = frame.loc[train_mask, "game_id"].tolist()
        training_count = len(train_ids)
        max_training_result_available = (
            result_available.loc[train_mask].max() if training_count else pd.NaT
        )
        membership_hash = training_membership_hash(train_ids)
        truncated = training_count > cfg.training_ids_cap
        stored_ids = () if truncated else tuple(sorted(train_ids))

        base_record = {
            "target_cutoff_utc": cutoff_i,
            # This batch's training mask was evaluated against cutoff_i --
            # that is the as-of eligibility timestamp, recorded explicitly
            # and separately from target_cutoff_utc (see OOF_LEDGER_COLUMNS).
            "training_as_of_utc": cutoff_i,
            "max_training_result_available_at_utc": max_training_result_available,
            "result_availability_basis": RESULT_AVAILABILITY_BASIS,
            "training_game_count": training_count,
            "training_game_ids": stored_ids,
            "training_game_ids_truncated": truncated,
            "training_membership_hash": membership_hash,
            "model_config_hash": model_config_hash,
            "feature_state_hash": feature_state_hash,
        }

        if training_count < cfg.min_training_games:
            margin_means = total_means = None
        else:
            train_frame = frame.loc[train_mask, feature_columns + ["home_margin", "total_points"]]
            model = factory(cfg)
            target_rows = frame.iloc[group_positions][feature_columns]
            with threadpool_limits(limits=1):
                model.fit(train_frame)
                margin_means, total_means = model.predict_means(target_rows)

        for slot, i in enumerate(group_positions):
            record: dict[str, object] = {
                "game_id": frame.at[i, "game_id"],
                "season": frame.at[i, "season"],
                "week": frame.at[i, "week"],
                "result_available_at_utc": result_available.iloc[i],
                **base_record,
            }
            if margin_means is None:
                record["status"] = "INSUFFICIENT_WARMUP"
                record["predicted_margin"] = float("nan")
                record["predicted_total"] = float("nan")
            else:
                record["status"] = "OOF"
                record["predicted_margin"] = float(margin_means[slot])
                record["predicted_total"] = float(total_means[slot])
            rows.append(record)

    predictions = pd.DataFrame(rows, columns=list(OOF_LEDGER_COLUMNS))
    # Self-check, using the repository's own existing availability-invariant
    # helper (nfl_hybrid.data.availability), not a bespoke one: every OOF
    # row's training set resolved strictly before that row's own cutoff.
    # NaT rows (INSUFFICIENT_WARMUP with zero training games) are skipped by
    # assert_available_before itself, not specially handled here.
    assert_available_before(
        predictions,
        available_at_column="max_training_result_available_at_utc",
        prediction_time_column="target_cutoff_utc",
        allow_equal=False,
    )
    return predictions


def attach_outcomes_and_residuals(
    predictions: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Phase 2: attach each target game's actual outcome and compute
    residuals. Strictly downstream of :func:`generate_oof_predictions` -- by
    the time this runs, every prediction was already fixed without ever
    consulting its own outcome."""
    outcomes = feature_matrix[["game_id", "home_margin", "total_points"]].rename(
        columns={"home_margin": "actual_margin", "total_points": "actual_total"}
    )
    outcomes["game_id"] = outcomes["game_id"].astype(str)
    ledger = predictions.merge(outcomes, on="game_id", how="left", validate="one_to_one")
    ledger["margin_residual"] = ledger["actual_margin"] - ledger["predicted_margin"]
    ledger["total_residual"] = ledger["actual_total"] - ledger["predicted_total"]
    return ledger[list(RESIDUAL_LEDGER_COLUMNS)]


def attach_expanding_oof_uncertainty(
    ledger: pd.DataFrame,
    *,
    min_uncertainty_warmup: int = 32,
) -> pd.DataFrame:
    """For every row, compute margin/total residual SD and residual
    correlation from OOF residuals of games whose RESULT was already
    available strictly before that row's own target cutoff --
    ``result_available_at_utc < target_cutoff_utc`` -- never this row's own
    residual, never a residual that had not yet resolved by that cutoff.

    A prediction occurring earlier is not enough: this deliberately does not
    walk the ledger in ``target_cutoff_utc`` processing order and assume
    "already visited" means "available" (that was the module's first,
    incorrect version -- kickoff/forecast order is not resolution order).
    Instead, OOF residuals are kept on their own timeline, sorted by
    ``result_available_at_utc``, and a residual only ever enters the pool
    for row *i* once its own ``result_available_at_utc`` is confirmed
    strictly before row *i*'s ``target_cutoff_utc`` -- via a merge over two
    independently-sorted sequences (target cutoffs ascending; resolution
    times ascending), which is exact regardless of how kickoff order and
    resolution order interleave.

    Never reads ``margin_sd_`` / ``total_sd_`` / ``residual_correlation_``
    from any fitted model -- those are fitted-training-residual quantities
    and are rejected by construction: this function only ever consumes the
    ``margin_residual`` / ``total_residual`` columns already produced by
    :func:`attach_outcomes_and_residuals`.
    """
    if min_uncertainty_warmup < 2:
        raise ValueError("min_uncertainty_warmup must be >= 2 (a sample SD needs >= 2 points).")

    ordered = ledger.sort_values(["target_cutoff_utc", "game_id"], kind="stable").reset_index(
        drop=True
    )
    n = len(ordered)
    margin_sd = np.full(n, np.nan)
    total_sd = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    eligible_flag = np.zeros(n, dtype=bool)

    # Resolution timeline: only OOF rows have a real residual, sorted by
    # when that residual actually becomes available -- never by
    # target_cutoff_utc / processing order. A game predicted "later" can
    # still resolve earlier than one predicted "earlier", and vice versa.
    oof = ordered[ordered["status"] == "OOF"].sort_values(
        "result_available_at_utc", kind="stable"
    )
    timeline_available = pd.to_datetime(
        oof["result_available_at_utc"], utc=True
    ).to_numpy()
    timeline_margin = oof["margin_residual"].to_numpy(dtype=float)
    timeline_total = oof["total_residual"].to_numpy(dtype=float)

    target_cutoff = pd.to_datetime(ordered["target_cutoff_utc"], utc=True).to_numpy()

    prior_margin: list[float] = []
    prior_total: list[float] = []
    j = 0
    for i in range(n):
        cutoff_i = target_cutoff[i]
        while j < len(timeline_available) and timeline_available[j] < cutoff_i:
            prior_margin.append(float(timeline_margin[j]))
            prior_total.append(float(timeline_total[j]))
            j += 1
        if len(prior_margin) >= min_uncertainty_warmup:
            m = np.asarray(prior_margin, dtype=float)
            t = np.asarray(prior_total, dtype=float)
            margin_sd[i] = float(np.std(m, ddof=1))
            total_sd[i] = float(np.std(t, ddof=1))
            corr[i] = float(np.clip(np.corrcoef(m, t)[0, 1], -0.95, 0.95)) if len(m) > 3 else 0.0
            eligible_flag[i] = True

    ordered["margin_residual_sd_oof"] = margin_sd
    ordered["total_residual_sd_oof"] = total_sd
    ordered["residual_correlation_oof"] = corr
    ordered["uncertainty_eligible"] = eligible_flag
    return ordered[list(UNCERTAINTY_LEDGER_COLUMNS)]


def production_uncertainty(
    ledger: pd.DataFrame,
    *,
    as_of_utc,
    min_uncertainty_warmup: int = 32,
) -> dict[str, object]:
    """Final production-artifact uncertainty: margin/total residual SD and
    residual correlation estimated from every eligible historical OOF
    residual whose RESULT was already available strictly before
    ``as_of_utc`` -- ``result_available_at_utc < as_of_utc``. A prediction
    occurring earlier than ``as_of_utc`` is not enough on its own: the game
    it belongs to must have actually resolved by then.

    Still never reads a fitted model's own training-residual attributes --
    only this module's own OOF residual ledger.
    """
    if min_uncertainty_warmup < 2:
        raise ValueError("min_uncertainty_warmup must be >= 2 (a sample SD needs >= 2 points).")

    as_of = pd.Timestamp(as_of_utc)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    eligible = ledger[(ledger["status"] == "OOF") & (ledger["result_available_at_utc"] < as_of)]
    if len(eligible) < min_uncertainty_warmup:
        return {
            "status": "INSUFFICIENT_WARMUP",
            "as_of_utc": str(as_of),
            "n_residuals": int(len(eligible)),
        }
    m = eligible["margin_residual"].to_numpy(dtype=float)
    t = eligible["total_residual"].to_numpy(dtype=float)
    return {
        "status": "OK",
        "as_of_utc": str(as_of),
        "n_residuals": int(len(eligible)),
        "margin_residual_sd": float(np.std(m, ddof=1)),
        "total_residual_sd": float(np.std(t, ddof=1)),
        "residual_correlation": float(np.clip(np.corrcoef(m, t)[0, 1], -0.95, 0.95)),
    }


def run_chronological_oof(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    qb_game: pd.DataFrame | None = None,
    config: ChronologicalOOFConfig | None = None,
    min_uncertainty_warmup: int = 32,
    rolling_config=None,
    opponent_config=None,
    qb_config=None,
    target_row_positions: Sequence[int] | None = None,
) -> ChronologicalOOFResult:
    """End-to-end orchestrator: Fix 2 authoritative pregame state -> game-level
    feature matrix -> chronological OOF predictions -> outcomes/residuals ->
    expanding OOF uncertainty. The only entry point that wires all five
    stages together; every stage is independently callable/testable above.

    ``target_row_positions`` passes through to :func:`generate_oof_predictions`
    unchanged -- see its docstring. Omit it for full production behavior
    (every game gets a prediction)."""
    cfg = config or ChronologicalOOFConfig()
    feature_matrix, feature_columns, bundle = build_oof_feature_matrix(
        games,
        play_by_play,
        qb_game=qb_game,
        rolling_config=rolling_config,
        opponent_config=opponent_config,
        qb_config=qb_config,
    )
    feature_state_hash = compute_feature_state_hash(feature_columns, bundle)
    predictions = generate_oof_predictions(
        feature_matrix,
        feature_columns=feature_columns,
        config=cfg,
        feature_state_hash=feature_state_hash,
        target_row_positions=target_row_positions,
    )
    residual_ledger = attach_outcomes_and_residuals(predictions, feature_matrix)
    residual_ledger = attach_expanding_oof_uncertainty(
        residual_ledger, min_uncertainty_warmup=min_uncertainty_warmup
    )
    return ChronologicalOOFResult(
        predictions=predictions,
        residual_ledger=residual_ledger,
        feature_columns=feature_columns,
        feature_state_hash=feature_state_hash,
        model_config_hash=compute_model_config_hash(cfg),
    )


def persist_oof_ledger(
    result: ChronologicalOOFResult,
    *,
    extra_manifest: Mapping[str, object] | None = None,
    root_override: str | Path | None = None,
) -> dict[str, Path]:
    """Persist the OOF predictions/residual ledger through the registered
    external-data artifact architecture (:mod:`nfl_hybrid.data.external_data`)
    -- never a hard-coded, machine-specific path. These are ``scope="generated"``
    registry keys: they resolve under ``NFL_MODEL_ARTIFACT_ROOT`` (or
    ``root_override``), never ``NFL_MODEL_DATA_ROOT`` -- generated pipeline
    output is deliberately never written into the raw/canonical historical
    estate's own tree."""
    written: dict[str, Path] = {}

    predictions_path = external_data.resolve_for_write(
        "oof_chronological.predictions", root_override=root_override
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(predictions_path, index=False)
    written["oof_chronological.predictions"] = predictions_path

    ledger_path = external_data.resolve_for_write(
        "oof_chronological.residual_ledger", root_override=root_override
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    result.residual_ledger.to_parquet(ledger_path, index=False)
    written["oof_chronological.residual_ledger"] = ledger_path

    manifest = {
        "feature_columns": list(result.feature_columns),
        "feature_state_hash": result.feature_state_hash,
        "model_config_hash": result.model_config_hash,
        "result_availability_basis": RESULT_AVAILABILITY_BASIS,
        "n_predictions": int(len(result.predictions)),
        "n_oof": int((result.predictions["status"] == "OOF").sum()),
        "n_insufficient_warmup": int(
            (result.predictions["status"] == "INSUFFICIENT_WARMUP").sum()
        ),
        **dict(extra_manifest or {}),
    }
    manifest_path = external_data.resolve_for_write(
        "oof_chronological.manifest", root_override=root_override
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    written["oof_chronological.manifest"] = manifest_path

    return written
