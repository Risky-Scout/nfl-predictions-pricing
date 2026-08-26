"""QB Depth-Chart As-Of V1 -- ``QB_DEPTH_CHART_QB1_ASOF_V1``.

Resolves, for each certified TUE/FRI card-scoped target cutoff
(:mod:`nfl_hybrid.features.horizon_elo`), the unique QB1 implied by the
latest nflverse rolling depth-chart snapshot that was defensibly available
at that cutoff. This is a narrower concept than "the quarterback who
ultimately started" -- no hindsight field, no injury data, no human
override, and no market information is ever consulted here. See
``qb_depthchart_asof_v1_semantics.json`` (frozen before this module was
written) for the full, hashed specification.

Source: the nflverse ``depth_charts`` *rolling-snapshot* schema block only
(``dt``/``team``/``gsis_id``/``pos_abb``/``pos_rank``) -- confirmed a full
32-team snapshot on every observed ``dt`` (never a partial delta), with a
genuine sub-day, timezone-aware timestamp (second precision, e.g.
``2026-08-25T07:39:23Z``). This is NOT the older season/week-tagged
``depth_charts`` block (``season``/``week``/``game_type``/``depth_team``/
``position``), which has zero 2025+ rows and no usable per-record
timestamp -- that block is never read by this module.
"""
from __future__ import annotations

import hashlib
import json
from datetime import time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_hybrid.data.provenance import dataframe_fingerprint, utc_now_iso
from nfl_hybrid.data.team_ids import canonical_team_id, try_canonical_team_id
from nfl_hybrid.features.horizon_elo import HORIZONS, build_horizon_membership_ledger

TRANSFORM_NAME = "resolve_depthchart_qb1_asof"
TRANSFORM_VERSION = "1.0"
SCHEMA_VERSION = "QB_DEPTH_CHART_QB1_ASOF_V1"
SOURCE_NAME = "NFLVERSE_ROLLING_DEPTH_CHART_SNAPSHOT"

REQUIRED_SNAPSHOT_COLUMNS: frozenset[str] = frozenset({"dt", "team", "gsis_id", "pos_abb", "pos_rank"})
QB_POSITION_VALUE = "QB"

STATUS_RESOLVED = "QB1_RESOLVED"
STATUS_NO_VALID_SNAPSHOT = "QB1_NO_VALID_SNAPSHOT"
STATUS_NO_QB_ROWS = "QB1_NO_QB_ROWS"
STATUS_AMBIGUOUS = "QB1_AMBIGUOUS_TOP_RANK"
STATUS_IDENTIFIER_UNRESOLVED = "QB1_IDENTIFIER_UNRESOLVED"
STATUS_SCHEMA_DRIFT = "QB1_SCHEMA_DRIFT"
STATUS_SOURCE_UNAVAILABLE = "QB1_SOURCE_UNAVAILABLE"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_RESOLVED,
    STATUS_NO_VALID_SNAPSHOT,
    STATUS_NO_QB_ROWS,
    STATUS_AMBIGUOUS,
    STATUS_IDENTIFIER_UNRESOLVED,
    STATUS_SCHEMA_DRIFT,
    STATUS_SOURCE_UNAVAILABLE,
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id", "team_id", "horizon", "target_cutoff_utc",
    "qb1_player_id", "qb1_source_snapshot", "qb1_source_available_at_utc", "qb1_source_hash",
    "qb1_resolution_status", "qb1_resolution_reason",
)

LIVE_OBSERVATION_LOG_SUBDIR = Path("live-observation-log") / "qb-depthchart"
LIVE_OBSERVATION_LOG_FILENAME = "observations.jsonl"

NY_ZONE = ZoneInfo("America/New_York")
PRECISION_EXACT_TIMESTAMP = "EXACT_TIMESTAMP"
PRECISION_DATE_ONLY = "DATE_ONLY"


def normalize_source_available_at(value: object, *, precision: str) -> pd.Timestamp:
    """Section 5's two historical-availability branches, implemented and
    independently testable. The live nflverse rolling-snapshot ``dt`` field
    was directly inspected (not assumed) and found to be a genuine
    timezone-aware timestamp with second precision (e.g.
    ``2026-08-25T07:39:23Z``) -- so :func:`prepare_snapshots` always uses
    ``PRECISION_EXACT_TIMESTAMP``. ``PRECISION_DATE_ONLY`` exists for a
    hypothetical future source that only publishes a calendar date; it must
    never be applied to a source that already provides an exact timestamp.
    """
    if precision == PRECISION_EXACT_TIMESTAMP:
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if precision == PRECISION_DATE_ONLY:
        source_date = pd.Timestamp(value).date()
        next_day = source_date + timedelta(days=1)
        local_midnight = pd.Timestamp.combine(next_day, time(0, 0)).tz_localize(NY_ZONE)
        return local_midnight.tz_convert("UTC")
    raise ValueError(f"Unknown timestamp precision: {precision!r}")


def _canonical_json_hash(payload: Any) -> str:
    """Canonical scientific/config hash. ``payload`` must already contain only
    ordinary JSON primitives (str/int/float/bool/None/list/dict) -- datetimes
    and any other non-primitive must be explicitly converted to a canonical
    UTC ISO8601 string (or other primitive) by the caller before this is
    invoked. Deliberately does NOT pass ``default=str``: a silent catch-all
    string coercion can absorb an unexpected/changed type (e.g. a
    ``pandas.Timestamp`` swapped in for a plain string) into the hash without
    ever raising, which defeats the purpose of a reproducibility hash. Let a
    non-primitive value raise ``TypeError`` here instead of hashing silently.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen V1 semantics (Section 3) -- persisted verbatim to
# qb_depthchart_asof_v1_semantics.json; no outcome data may influence this.
# ---------------------------------------------------------------------------
def qb_depthchart_asof_v1_semantics() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "forecast_horizons": list(HORIZONS),
        "card_cutoff_semantics": "certified Fix 7.1/Fix 8 card-scoped cutoffs (horizon_elo.build_horizon_membership_ledger)",
        "tue_cutoff_rule": "card_monday_date + 1 day, 12:00 America/New_York, DST-aware",
        "fri_cutoff_rule": "card_monday_date + 4 days, 12:00 America/New_York, DST-aware",
        "target_eligibility_rule": "target_cutoff_utc(card, horizon) < scheduled_kickoff_utc(game) (STRICT)",
        "selection_rule": "latest defensibly available snapshot at-or-before cutoff",
        "position_rule": "source-defined QB position only (pos_abb == 'QB')",
        "qb1_rule": (
            "use the source-provided depth ordering (pos_rank); the 'top' rank is the minimum pos_rank value "
            "present among that team's QB rows in the selected snapshot (observed to always be 1 in the live "
            "source, but not hardcoded to the literal integer 1)"
        ),
        "ambiguity_rule": "if a UNIQUE top-ranked QB cannot be resolved, abstain (QB1_AMBIGUOUS_TOP_RANK)",
        "fallback_rule": "NONE -- an unresolvable latest snapshot is never superseded by an older snapshot",
        "human_override": "DISABLED",
        "injury_override": "DISABLED",
        "actual_starter_fallback": "PROHIBITED",
        "market_fallback": "PROHIBITED",
        "target_game_participation_fallback": "PROHIBITED",
        "historical_availability_policy": (
            "Source timestamp precision was inspected directly (not assumed): the rolling-snapshot 'dt' field is "
            "a genuine timezone-aware ISO8601 timestamp with second precision (e.g. 2026-08-25T07:39:23Z), not a "
            "calendar date. Per the exact-timestamp branch, source_available_at_utc = dt, parsed to UTC and used "
            "directly -- the date-only 'next calendar day 00:00 America/New_York' conservative rule does NOT "
            "apply here (it is reserved for a date-only source, which this is not)."
        ),
        "live_availability_policy": (
            "For live 2026 production, a snapshot is available only when BOTH "
            "first_observed_at_utc <= target_cutoff_utc AND the source's own dt <= target_cutoff_utc. "
            "first_observed_at_utc comes from the append-only observation log "
            "(record_live_observation), never from dt alone and never backdated."
        ),
        "resolution_statuses": list(ALL_STATUSES),
        "required_snapshot_columns": sorted(REQUIRED_SNAPSHOT_COLUMNS),
        "no_actual_started_field": True,
        "no_target_game_statistics": True,
        "no_market_data": True,
    }


def compute_semantics_hash(semantics: dict | None = None) -> str:
    return _canonical_json_hash(semantics if semantics is not None else qb_depthchart_asof_v1_semantics())


# ---------------------------------------------------------------------------
# Additive V1.1 semantics (Section 7 pre-selection audit, 2026-08-25).
#
# ``compute_qb1_continuity`` (the qb1_changed_from_previous_resolved_same_horizon
# / qb1_continuity_cards indicators) was implemented after the original V1
# semantics freeze above and is not described by it. Per audit policy this is
# never folded into the V1 payload/hash after the fact -- V1's hash
# (``compute_semantics_hash()`` with no args) is frozen, historical
# provenance and stays exactly as it was. This V1.1 payload is strictly
# additive: it repeats V1 verbatim, records V1's own hash as
# ``supersedes.semantics_hash`` for traceability, and documents ONLY the
# continuity/change behavior. No performance/outcome data of any kind
# informed this payload.
# ---------------------------------------------------------------------------
SCHEMA_VERSION_V1_1 = "QB_DEPTH_CHART_QB1_ASOF_V1_1"


def qb_depthchart_asof_v1_1_semantics() -> dict:
    v1 = qb_depthchart_asof_v1_semantics()
    additive = dict(v1)
    additive["schema_version"] = SCHEMA_VERSION_V1_1
    additive["supersedes"] = {
        "schema_version": v1["schema_version"],
        "semantics_hash": compute_semantics_hash(v1),
        "relationship": "ADDITIVE_NO_REWRITE -- V1's own payload and hash are unchanged and remain valid, standalone historical provenance for the resolver behavior they describe",
    }
    additive["continuity_change_semantics"] = {
        "added_after_v1_freeze": True,
        "implemented_by": "qb_depthchart_asof.compute_qb1_continuity",
        "qb1_changed_from_previous_resolved_same_horizon": (
            "Boolean, defined only when BOTH the current card's qb1 and the immediately preceding "
            "chronological card's qb1 for the same (team_id, horizon) within the same season are "
            "QB1_RESOLVED; compares the two resolved qb1_player_id values only. Never compares against "
            "an actual starter, injury designation, human override, or market-implied value."
        ),
        "qb1_continuity_cards": (
            "Integer count of consecutive QB1_RESOLVED cards, ending at (and including) the current card, "
            "with an unchanged qb1_player_id, for the same (team_id, horizon, season); resets to 1 on a "
            "change or on the first-ever resolved card of the season."
        ),
        "missingness": (
            "qb1_changed_from_previous_resolved_same_horizon is missing (NA) when the current card is not "
            "QB1_RESOLVED, or the current card is QB1_RESOLVED but no immediately preceding QB1_RESOLVED "
            "card exists yet this season. qb1_continuity_cards is missing (NA) whenever the current card "
            "itself is not QB1_RESOLVED."
        ),
        "chronology_key": "(team_id, horizon, season), ordered by target_cutoff_utc -- never by game date or kickoff time.",
        "season_boundary_reset": "Both indicators reset at each season boundary; no continuity/change state carries over from a prior season.",
        "no_hindsight": "Only ever compares an already-QB1_RESOLVED PRIOR card's snapshot -- never the current or a future card's snapshot, and never an actual starter.",
    }
    return additive


def compute_semantics_v1_1_hash(semantics: dict | None = None) -> str:
    return _canonical_json_hash(semantics if semantics is not None else qb_depthchart_asof_v1_1_semantics())


# ---------------------------------------------------------------------------
# Snapshot preparation / schema-drift detection (Section 4/8).
# ---------------------------------------------------------------------------
def prepare_snapshots(depthchart_snapshots: pd.DataFrame | None) -> tuple[pd.DataFrame, bool]:
    """Normalize the rolling-snapshot frame. Returns ``(frame, schema_ok)``.

    ``schema_ok`` is False when the required rolling-schema columns are
    absent or ``dt`` cannot be parsed as a real timestamp for every row --
    the caller must then report ``QB1_SCHEMA_DRIFT`` rather than guess.
    """
    if depthchart_snapshots is None or len(depthchart_snapshots) == 0:
        return pd.DataFrame(), False
    missing = REQUIRED_SNAPSHOT_COLUMNS - set(depthchart_snapshots.columns)
    if missing:
        return depthchart_snapshots, False
    work = depthchart_snapshots.copy()
    work["dt_utc"] = pd.to_datetime(work["dt"], utc=True, errors="coerce")
    if work["dt_utc"].isna().any():
        return work, False
    work["team_canonical"] = work["team"].map(try_canonical_team_id)
    return work, True


def _hash_qb_group(qb_rows: pd.DataFrame) -> str:
    records = (
        qb_rows[["gsis_id", "pos_rank"]]
        .astype(object)
        .where(pd.notnull(qb_rows[["gsis_id", "pos_rank"]]), None)
        .sort_values(["pos_rank", "gsis_id"], key=lambda s: s.astype(str))
        .to_dict(orient="records")
    )
    return _canonical_json_hash(records)


def _resolve_team_dt_group(group: pd.DataFrame) -> dict:
    """Resolve QB1 for one already-selected (team, dt) snapshot slice."""
    qb_rows = group[group["pos_abb"] == QB_POSITION_VALUE]
    source_hash = _hash_qb_group(qb_rows)
    if qb_rows.empty:
        return {
            "qb1_player_id": None, "qb1_resolution_status": STATUS_NO_QB_ROWS,
            "qb1_resolution_reason": "no rows with pos_abb=='QB' in the latest valid snapshot for this team at or before cutoff",
            "qb1_source_hash": source_hash,
        }
    top_rank = qb_rows["pos_rank"].min()
    top = qb_rows[qb_rows["pos_rank"] == top_rank]
    if len(top) > 1:
        return {
            "qb1_player_id": None, "qb1_resolution_status": STATUS_AMBIGUOUS,
            "qb1_resolution_reason": f"{len(top)} QB rows tied at top rank {top_rank!r} in the latest valid snapshot",
            "qb1_source_hash": source_hash,
        }
    player_id = top.iloc[0]["gsis_id"]
    if pd.isna(player_id) or str(player_id).strip() == "":
        return {
            "qb1_player_id": None, "qb1_resolution_status": STATUS_IDENTIFIER_UNRESOLVED,
            "qb1_resolution_reason": "top-ranked QB row has a missing/blank gsis_id",
            "qb1_source_hash": source_hash,
        }
    return {
        "qb1_player_id": str(player_id), "qb1_resolution_status": STATUS_RESOLVED,
        "qb1_resolution_reason": "unique top-ranked QB resolved",
        "qb1_source_hash": source_hash,
    }


# ---------------------------------------------------------------------------
# Canonical resolver (Section 7).
# ---------------------------------------------------------------------------
def resolve_depthchart_qb1_asof(
    games: pd.DataFrame,
    depthchart_snapshots: pd.DataFrame | None,
    horizon: str,
    *,
    membership_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (game_id, team_id) for every side of every game eligible
    as a ``horizon`` target (per the certified card-scoped membership
    ledger). No actual-started field and no target-game statistic is read.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon!r}, expected one of {HORIZONS}")
    h = horizon.lower()

    ledger = membership_ledger if membership_ledger is not None else build_horizon_membership_ledger(games)
    ledger_h = ledger.loc[ledger[f"{h}_eligible"], ["game_id", f"{h}_cutoff_utc"]].rename(
        columns={f"{h}_cutoff_utc": "target_cutoff_utc"}
    )

    work_games = games.copy()
    work_games["game_id"] = work_games["game_id"].astype(str)
    work_games["home_team_id"] = work_games["home_team_id"].map(canonical_team_id)
    work_games["away_team_id"] = work_games["away_team_id"].map(canonical_team_id)
    work_games = work_games.merge(ledger_h, on="game_id", how="inner", validate="one_to_one")

    targets = pd.concat(
        [
            work_games[["game_id", "home_team_id", "target_cutoff_utc"]].rename(columns={"home_team_id": "team_id"}),
            work_games[["game_id", "away_team_id", "target_cutoff_utc"]].rename(columns={"away_team_id": "team_id"}),
        ],
        ignore_index=True,
    )
    targets["horizon"] = horizon
    if targets.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    prepared, schema_ok = prepare_snapshots(depthchart_snapshots)
    source_present = depthchart_snapshots is not None and len(depthchart_snapshots) > 0

    if not source_present:
        out = targets.copy()
        out["qb1_player_id"] = None
        out["qb1_source_snapshot"] = None
        out["qb1_source_available_at_utc"] = pd.NaT
        out["qb1_source_hash"] = None
        out["qb1_resolution_status"] = STATUS_SOURCE_UNAVAILABLE
        out["qb1_resolution_reason"] = "depth-chart snapshot source returned no data"
        return out[list(OUTPUT_COLUMNS)]

    if not schema_ok:
        out = targets.copy()
        out["qb1_player_id"] = None
        out["qb1_source_snapshot"] = None
        out["qb1_source_available_at_utc"] = pd.NaT
        out["qb1_source_hash"] = None
        out["qb1_resolution_status"] = STATUS_SCHEMA_DRIFT
        out["qb1_resolution_reason"] = f"required rolling-snapshot columns/timestamp not found: {sorted(REQUIRED_SNAPSHOT_COLUMNS)}"
        return out[list(OUTPUT_COLUMNS)]

    # Pre-resolve QB1 once per (team_canonical, dt_utc) group -- far cheaper
    # than re-scanning the full snapshot table per target.
    resolved_groups = (
        prepared.groupby(["team_canonical", "dt_utc"], dropna=True)
        .apply(_resolve_team_dt_group, include_groups=False)
        .apply(pd.Series)
        .reset_index()
        .sort_values("dt_utc", kind="stable")
    )

    targets_sorted = targets.sort_values("target_cutoff_utc", kind="stable").reset_index(drop=True)
    merged = pd.merge_asof(
        targets_sorted,
        resolved_groups.rename(columns={"team_canonical": "team_id"}),
        left_on="target_cutoff_utc", right_on="dt_utc",
        by="team_id", direction="backward", allow_exact_matches=True,
    )

    no_snapshot = merged["dt_utc"].isna()
    merged.loc[no_snapshot, "qb1_resolution_status"] = STATUS_NO_VALID_SNAPSHOT
    merged.loc[no_snapshot, "qb1_resolution_reason"] = merged.loc[no_snapshot, "target_cutoff_utc"].apply(
        lambda ts: f"no source snapshot at or before target_cutoff_utc={ts.isoformat()}"
    )
    merged.loc[no_snapshot, "qb1_player_id"] = None
    merged.loc[no_snapshot, "qb1_source_hash"] = None

    merged["qb1_source_snapshot"] = merged["dt_utc"].apply(lambda ts: ts.isoformat() if pd.notna(ts) else None)
    merged["qb1_source_available_at_utc"] = merged["dt_utc"]

    out = merged[list(OUTPUT_COLUMNS)].reset_index(drop=True)
    if out.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate QB1 as-of resolution rows produced.")
    return out


def compute_side_flags(resolved: pd.DataFrame) -> pd.DataFrame:
    """``{home,away}_qb1_resolved`` -- the two Section 14 candidate flags
    that require no chronology, one row per game for a single horizon's
    resolver output."""
    work = resolved.copy()
    work["qb1_resolved_flag"] = work["qb1_resolution_status"] == STATUS_RESOLVED
    return work[["game_id", "team_id", "horizon", "qb1_resolved_flag"]]


def compute_qb1_continuity(resolved: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Registered-candidate continuity/change indicators (Section 14) --
    proposal-with-working-code, not a selected Ridge feature. Compares each
    RESOLVED qb1 only against the previous chronologically RESOLVED qb1 for
    the same ``(team_id, horizon)`` -- never against an actual starter.
    Resets at season boundary. Unresolved current/prior state is flagged
    explicitly rather than silently defaulting.
    """
    season_by_game = games.set_index(games["game_id"].astype(str))["season"]
    work = resolved.copy()
    work["season"] = work["game_id"].map(season_by_game)
    work = work.sort_values(["team_id", "horizon", "season", "target_cutoff_utc"], kind="stable").reset_index(drop=True)

    prev_player = work.groupby(["team_id", "horizon", "season"])["qb1_player_id"].shift(1)
    prev_status = work.groupby(["team_id", "horizon", "season"])["qb1_resolution_status"].shift(1)

    both_resolved = (work["qb1_resolution_status"] == STATUS_RESOLVED) & (prev_status == STATUS_RESOLVED)
    work["qb1_continuity_prior_available"] = both_resolved
    work["qb1_changed_from_previous_resolved_same_horizon"] = pd.Series(pd.NA, index=work.index, dtype="boolean")
    work.loc[both_resolved, "qb1_changed_from_previous_resolved_same_horizon"] = (
        work.loc[both_resolved, "qb1_player_id"] != prev_player.loc[both_resolved]
    )

    # Optional Section 14 candidate: consecutive resolved cards with an
    # UNCHANGED qb1, ending at (and including) the current row. Missing
    # (pd.NA) whenever the current row itself is not RESOLVED.
    streak: list[object] = []
    running: dict[tuple, int] = {}
    for row in work.itertuples(index=False):
        key = (row.team_id, row.horizon, row.season)
        if row.qb1_resolution_status != STATUS_RESOLVED:
            running[key] = 0
            streak.append(pd.NA)
            continue
        changed = getattr(row, "qb1_changed_from_previous_resolved_same_horizon")
        if pd.isna(changed):  # no prior resolved card this season -- first observed streak of one
            running[key] = 1
        elif bool(changed):
            running[key] = 1
        else:
            running[key] = running.get(key, 0) + 1
        streak.append(running[key])
    work["qb1_continuity_cards"] = streak

    return work[
        [
            "game_id", "team_id", "horizon", "season",
            "qb1_resolution_status", "qb1_continuity_prior_available",
            "qb1_changed_from_previous_resolved_same_horizon", "qb1_continuity_cards",
        ]
    ]


# ---------------------------------------------------------------------------
# Live 2026 prospective observation log (Section 6). Append-only; never
# overwrites an earlier record; no credentials are ever included.
# ---------------------------------------------------------------------------
def record_live_observation(
    *,
    data_root: str | Path,
    snapshot_frame: pd.DataFrame,
    observed_at_utc: str | None = None,
) -> dict:
    ts = observed_at_utc or utc_now_iso()
    ts_utc = pd.Timestamp(ts)
    ts_utc = ts_utc.tz_localize("UTC") if ts_utc.tzinfo is None else ts_utc.tz_convert("UTC")

    log_path = Path(data_root) / LIVE_OBSERVATION_LOG_SUBDIR / LIVE_OBSERVATION_LOG_FILENAME
    if log_path.exists():
        existing = read_live_observation_log(data_root)
        if len(existing):
            prior_max = pd.to_datetime(existing["observed_at_utc"], utc=True, errors="coerce").max()
            if pd.notna(prior_max) and ts_utc < prior_max:
                raise ValueError(
                    "record_live_observation refuses to backdate the observation log: "
                    f"observed_at_utc={ts_utc.isoformat()} is earlier than the latest "
                    f"already-appended observation at {prior_max.isoformat()}. A later retrieval "
                    "may never overwrite or predate an earlier record's first_observed_at_utc."
                )

    record = {
        "observed_at_utc": ts,
        "source_snapshot_timestamp": str(snapshot_frame["dt"].max()) if "dt" in snapshot_frame.columns and len(snapshot_frame) else None,
        "source_content_hash": dataframe_fingerprint(snapshot_frame) if len(snapshot_frame) else None,
        "row_count": int(len(snapshot_frame)),
        "team_count": int(snapshot_frame["team"].nunique()) if "team" in snapshot_frame.columns else None,
        "schema_hash": _canonical_json_hash(sorted(snapshot_frame.columns.tolist())),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_live_observation_log(data_root: str | Path) -> pd.DataFrame:
    log_path = Path(data_root) / LIVE_OBSERVATION_LOG_SUBDIR / LIVE_OBSERVATION_LOG_FILENAME
    if not log_path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def live_snapshot_available_at_cutoff(
    observation_log: pd.DataFrame, target_cutoff_utc: pd.Timestamp
) -> pd.Timestamp | None:
    """Section 6's live-only additional gate: a snapshot is usable only when
    BOTH its own ``dt`` and this pipeline's ``first_observed_at_utc`` are
    at-or-before the cutoff. Returns the latest qualifying source timestamp,
    or ``None``."""
    if observation_log.empty:
        return None
    log = observation_log.copy()
    log["observed_at_utc"] = pd.to_datetime(log["observed_at_utc"], utc=True, errors="coerce")
    log["source_snapshot_timestamp"] = pd.to_datetime(log["source_snapshot_timestamp"], utc=True, errors="coerce")
    eligible = log[(log["observed_at_utc"] <= target_cutoff_utc) & (log["source_snapshot_timestamp"] <= target_cutoff_utc)]
    if eligible.empty:
        return None
    return eligible["source_snapshot_timestamp"].max()
