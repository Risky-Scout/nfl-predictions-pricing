"""QB Lagged Depth State V1 -- ``QB_LAGGED_DEPTH_STATE_V1``.

One source-agnostic feature contract, implemented over two structurally
different nflverse depth-chart sources (the old season/week-tagged block,
2020-2024, and the rolling timestamped snapshot block, 2025+), that are
NEVER claimed to be the same semantic version of anything. Both are reduced
to the identical concept:

    For target T at cutoff C, use the most recent source-defined QB depth
    state associated with a PRIOR COMPLETED CARD whose conservative
    state-availability timestamp is strictly before C. Never the target
    card's own current-week state.

This is a strictly narrower, laggier concept than
:mod:`nfl_hybrid.features.qb_depthchart_asof`'s
``QB_DEPTH_CHART_QB1_ASOF_V1`` (which resolves the latest snapshot at-or-
before the TARGET's own cutoff -- including same-card state). That module's
resolver is deliberately never called here; only its narrow, already
source-agnostic per-row resolution primitive
(``_resolve_team_dt_group``, min ``pos_rank`` among ``pos_abb=='QB'`` rows,
tie/blank-id -> abstain) is reused for the live side, because Section 3 of
the frozen task calls for resolving each card's own canonical state with
"the already-certified V1 resolver semantics."

See ``qb_lagged_depth_state_v1_semantics()`` (frozen before any outcome
data was read) for the full, hashed specification.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

import nfl_hybrid.features.qb_depthchart_asof as qda
from nfl_hybrid.data.team_ids import canonical_team_id, try_canonical_team_id
from nfl_hybrid.features.horizon_elo import (
    CARD_KEY,
    HORIZONS,
    RESULT_AVAILABILITY_DURATION_HOURS,
    build_horizon_membership_ledger,
)

TRANSFORM_NAME = "build_qb_lagged_depth_state"
TRANSFORM_VERSION = "1.0"
SCHEMA_VERSION = "QB_LAGGED_DEPTH_STATE_V1"

HISTORICAL_SOURCE_SEMANTIC_VERSION = "QB_HISTORICAL_DEPTH_STATE_V1"
LIVE_SOURCE_SEMANTIC_VERSION = "QB_LAGGED_DEPTH_STATE_LIVE_V1"
NOT_THIS_CONTRACT = "QB_DEPTH_CHART_QB1_ASOF_V1"  # explicitly a different, richer, non-lagged signal

# Old-schema `game_type` -> games-table `season_type`. SBBYE (a bye-round
# marker with no corresponding game row) is deliberately absent -- it can
# never join to a real card, so it is dropped rather than mapped.
GAME_TYPE_TO_SEASON_TYPE: dict[str, str] = {
    "REG": "REG", "WC": "POST", "DIV": "POST", "CON": "POST", "SB": "POST",
}

REQUIRED_HISTORICAL_COLUMNS: frozenset[str] = frozenset(
    {"season", "week", "game_type", "club_code", "position", "depth_team", "gsis_id", "dt"}
)
REQUIRED_LIVE_COLUMNS: frozenset[str] = frozenset({"dt", "team", "gsis_id", "pos_abb", "pos_rank"})

# --- Card-level ("Layer 1") resolution statuses -- ONE shared vocabulary for
# both adapters, so a schema/status comparison across sources is meaningful.
CARD_STATUS_RESOLVED = "CARD_QB1_RESOLVED"
CARD_STATUS_NO_SOURCE_DATA = "CARD_QB1_NO_SOURCE_DATA"
CARD_STATUS_NO_QB_ROWS = "CARD_QB1_NO_QB_ROWS"
CARD_STATUS_AMBIGUOUS = "CARD_QB1_AMBIGUOUS_TOP_RANK"
CARD_STATUS_IDENTIFIER_UNRESOLVED = "CARD_QB1_IDENTIFIER_UNRESOLVED"
CARD_STATUS_RANK_UNRESOLVED = "CARD_QB1_RANK_UNRESOLVED"

ALL_CARD_STATUSES: tuple[str, ...] = (
    CARD_STATUS_RESOLVED, CARD_STATUS_NO_SOURCE_DATA, CARD_STATUS_NO_QB_ROWS,
    CARD_STATUS_AMBIGUOUS, CARD_STATUS_IDENTIFIER_UNRESOLVED, CARD_STATUS_RANK_UNRESOLVED,
)

# --- Target-level ("Layer 2") resolution statuses.
TARGET_STATUS_RESOLVED = "LAGGED_QB1_RESOLVED"
TARGET_STATUS_NO_PRIOR_CARD = "LAGGED_NO_PRIOR_CARD"
TARGET_STATUS_NO_SOURCE_DATA = "LAGGED_NO_SOURCE_DATA"
TARGET_STATUS_PRIOR_CARD_UNRESOLVED = "LAGGED_PRIOR_CARD_UNRESOLVED"
TARGET_STATUS_STATE_NOT_YET_AVAILABLE = "LAGGED_STATE_NOT_YET_AVAILABLE"

ALL_TARGET_STATUSES: tuple[str, ...] = (
    TARGET_STATUS_RESOLVED, TARGET_STATUS_NO_PRIOR_CARD, TARGET_STATUS_NO_SOURCE_DATA,
    TARGET_STATUS_PRIOR_CARD_UNRESOLVED, TARGET_STATUS_STATE_NOT_YET_AVAILABLE,
)

# Section 4's required normalized intermediate schema -- both adapters must
# emit exactly this column set (see assert_card_schema_bridge).
NORMALIZED_CARD_STATE_COLUMNS: tuple[str, ...] = (
    "season", "source_card_week", "team_id", "qb1_player_id",
    "qb_state_available_at_utc", "qb_state_resolution_status", "source_semantic_version",
)

CARD_TABLE_COLUMNS: tuple[str, ...] = (
    "season", "source_card_week", "season_type", "card_end_utc", "card_order", "qb_state_available_at_utc",
)

TARGET_OUTPUT_COLUMNS: tuple[str, ...] = (
    "game_id", "team_id", "side", "horizon", "season", "target_cutoff_utc",
    "prior_card_week", "qb1_player_id", "source_semantic_version",
    "lagged_resolution_status",
    "qb_depth_changed", "qb_depth_continuity_cards", "qb_depth_state_missing",
)

CANDIDATE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{base}"
    for side in ("home", "away")
    for base in ("qb_depth_changed", "qb_depth_continuity_cards", "qb_depth_state_missing")
)


def _canonical_json_hash(payload: Any) -> str:
    """Identical recipe to ``qb_depthchart_asof._canonical_json_hash``:
    canonical-primitives-only JSON, no ``default=str`` (a non-primitive must
    raise, never be silently string-coerced into the hash)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Section 1: frozen V1 semantics -- persisted verbatim to
# qb_lagged_depth_state_v1_semantics.json before any outcome data existed.
# ---------------------------------------------------------------------------
def qb_lagged_depth_state_v1_semantics() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "not_the_same_contract_as": NOT_THIS_CONTRACT,
        "concept": (
            "For target T at cutoff C, use the most recent source-defined QB depth state associated with a "
            "PRIOR COMPLETED CARD whose conservative state-availability timestamp is strictly before C. Never "
            "the target card's own current-week state."
        ),
        "card_key": list(CARD_KEY),
        "card_end_utc_rule": "max(scheduled_kickoff_utc) over every game in (season, week, season_type)",
        "state_availability_rule": "card_end_utc + RESULT_AVAILABILITY_DURATION_HOURS (reused unchanged from horizon_elo)",
        "result_availability_duration_hours": RESULT_AVAILABILITY_DURATION_HOURS,
        "prior_card_definition": "the card with the largest source_card_week strictly less than the target's own card's week, within the SAME season (season boundary never crossed)",
        "availability_gate": "qb_state_available_at_utc < target_cutoff_utc (STRICT); a resolved prior-card state failing this gate is treated as missing, never floated to an earlier card",
        "same_card_state_prohibited": True,
        "cross_season_carryover_prohibited": True,
        "week1_of_season_has_no_prior_card": True,
        "historical_source": {
            "semantic_version": HISTORICAL_SOURCE_SEMANTIC_VERSION,
            "source": "nflverse depth_charts OLD season/week-tagged block (dt IS NULL), 2020-2024",
            "availability_rule": "POST_WEEK_AVAILABILITY (Phase-3 recovery audit, research/qb-historical-source-recovery-2026)",
            "position_field": "position == 'QB'",
            "rank_field": "depth_team (numeric-coerced; min = top rank)",
            "player_id_field": "gsis_id",
            "team_field": "club_code (canonicalized via try_canonical_team_id)",
            "game_type_to_season_type": dict(GAME_TYPE_TO_SEASON_TYPE),
            "game_type_dropped": ["SBBYE"],
        },
        "live_source": {
            "semantic_version": LIVE_SOURCE_SEMANTIC_VERSION,
            "source": "nflverse rolling timestamped depth-chart snapshot block (dt IS NOT NULL)",
            "canonical_weekly_state_rule": "for completed card W, per team, the latest valid rolling snapshot with source_snapshot_timestamp <= card_end_utc(W)",
            "resolution_reused_from": "qb_depthchart_asof._resolve_team_dt_group (min pos_rank among pos_abb=='QB' rows; tie/blank id -> abstain)",
            "current_card_rolling_snapshot_prohibited": True,
        },
        "unified_card_status_vocabulary": list(ALL_CARD_STATUSES),
        "target_status_vocabulary": list(ALL_TARGET_STATUSES),
        "feature_definitions": {
            "qb_depth_changed": (
                "1 if the current available PRIOR-CARD QB1 differs from the previous AVAILABLE prior-card QB1 "
                "for that team (skipping over any intervening unresolved/unavailable card -- 'available' is "
                "the operative word, not 'immediately preceding'); else 0."
            ),
            "qb_depth_continuity_cards": (
                "count of consecutive AVAILABLE prior-card states with the current QB1, ending at the latest "
                "available prior-card state; a gap (unresolved card) is skipped, not counted and not a reset "
                "trigger by itself -- only an actual QB1 change resets the streak."
            ),
            "qb_depth_state_missing": "1 if the current REQUIRED (immediately-prior-card) state cannot be resolved or is not yet available at the target cutoff; else 0.",
        },
        "missingness_encoding": {
            "rule": "if the target's own required prior-card state cannot be resolved (no prior card this season, no source data, unresolved card, or not-yet-available): qb_depth_changed=0, qb_depth_continuity_cards=0, qb_depth_state_missing=1; otherwise qb_depth_state_missing=0",
            "frozen_structurally_not_selected_from_outcomes": True,
        },
        "first_ever_resolved_card_convention": (
            "A card that is RESOLVED but has no earlier AVAILABLE (resolved) card this season to compare "
            "against is, by structural convention, unchanged (qb_depth_changed_at_card=0) with a continuity "
            "streak of 1 -- there is nothing for it to have changed FROM. Decided before any fit; documented "
            "here, not inferred from performance."
        ),
        "season_boundary_reset": True,
        "player_id_as_ridge_feature": "PROHIBITED",
        "actual_starter_fallback": "PROHIBITED",
        "highest_dropback_qb_fallback": "PROHIBITED",
        "target_game_statistics": "PROHIBITED",
        "market_data": "PROHIBITED",
        "injury_override": "DISABLED",
        "human_override": "DISABLED",
        "normalized_intermediate_columns": list(NORMALIZED_CARD_STATE_COLUMNS),
    }


def compute_semantics_hash(semantics: dict | None = None) -> str:
    return _canonical_json_hash(semantics if semantics is not None else qb_lagged_depth_state_v1_semantics())


# ---------------------------------------------------------------------------
# Card table -- the ONE authoritative source of (season, week, season_type)
# card identity, ordering, and conservative availability timestamp. Both
# adapters and the target resolver share this, never re-deriving it.
# ---------------------------------------------------------------------------
def build_card_table(games: pd.DataFrame) -> pd.DataFrame:
    work = games.copy()
    work["scheduled_kickoff_utc"] = pd.to_datetime(work["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if work["scheduled_kickoff_utc"].isna().any():
        raise ValueError("build_card_table requires a parseable scheduled_kickoff_utc for every game.")
    work["season"] = pd.to_numeric(work["season"], errors="raise").astype(int)
    work["week"] = pd.to_numeric(work["week"], errors="raise").astype(int)

    card_end = (
        work.groupby(list(CARD_KEY))["scheduled_kickoff_utc"].max().rename("card_end_utc").reset_index()
    )
    card_end = card_end.sort_values(["season", "card_end_utc"], kind="stable").reset_index(drop=True)
    card_end["card_order"] = card_end.groupby("season").cumcount() + 1
    card_end["qb_state_available_at_utc"] = card_end["card_end_utc"] + pd.Timedelta(
        hours=RESULT_AVAILABILITY_DURATION_HOURS
    )
    card_end = card_end.rename(columns={"week": "source_card_week"})
    return card_end[list(CARD_TABLE_COLUMNS)]


# ---------------------------------------------------------------------------
# Shared per-group QB1 resolution primitive (rank-column-agnostic).
# ---------------------------------------------------------------------------
def _resolve_qb1_from_group(qb_rows: pd.DataFrame, *, rank_col: str, id_col: str) -> tuple[str, str | None]:
    if qb_rows.empty:
        return CARD_STATUS_NO_QB_ROWS, None
    ranks = pd.to_numeric(qb_rows[rank_col], errors="coerce")
    valid = qb_rows.loc[ranks.notna()].copy()
    if valid.empty:
        return CARD_STATUS_RANK_UNRESOLVED, None
    valid["_rank"] = ranks.loc[valid.index]
    top_rank = valid["_rank"].min()
    top = valid[valid["_rank"] == top_rank]
    if len(top) > 1:
        return CARD_STATUS_AMBIGUOUS, None
    player_id = top.iloc[0][id_col]
    if pd.isna(player_id) or str(player_id).strip() == "":
        return CARD_STATUS_IDENTIFIER_UNRESOLVED, None
    return CARD_STATUS_RESOLVED, str(player_id)


# ---------------------------------------------------------------------------
# Section 2: historical normalization (old season/week-tagged source).
# ---------------------------------------------------------------------------
def normalize_historical_card_states(depth_charts_raw: pd.DataFrame, card_table: pd.DataFrame) -> pd.DataFrame:
    if depth_charts_raw is None or len(depth_charts_raw) == 0:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))
    missing = REQUIRED_HISTORICAL_COLUMNS - set(depth_charts_raw.columns)
    if missing:
        raise ValueError(f"Historical depth-chart source missing required column(s): {sorted(missing)}")

    old = depth_charts_raw[depth_charts_raw["dt"].isna()].copy()
    old = old.dropna(subset=["season", "week", "game_type", "club_code"])
    if old.empty:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))
    old["season"] = pd.to_numeric(old["season"], errors="coerce").astype("Int64")
    old["week"] = pd.to_numeric(old["week"], errors="coerce").astype("Int64")
    old = old.dropna(subset=["season", "week"])
    old["season"] = old["season"].astype(int)
    old["week"] = old["week"].astype(int)
    old["mapped_season_type"] = old["game_type"].map(GAME_TYPE_TO_SEASON_TYPE)
    old = old[old["mapped_season_type"].notna()].copy()  # drops SBBYE / unrecognized game_type
    old["team_canonical"] = old["club_code"].map(try_canonical_team_id)
    old = old[old["team_canonical"].notna()].copy()  # fail-closed: unmappable club_code is never guessed

    records: list[dict] = []
    for (season, week, season_type, team_id), group in old.groupby(
        ["season", "week", "mapped_season_type", "team_canonical"], dropna=False, sort=False
    ):
        qb_rows = group[group["position"] == "QB"]
        status, player_id = _resolve_qb1_from_group(qb_rows, rank_col="depth_team", id_col="gsis_id")
        records.append(
            {
                "season": season, "source_card_week": week, "season_type": season_type, "team_id": team_id,
                "qb1_player_id": player_id, "qb_state_resolution_status": status,
            }
        )
    resolved = pd.DataFrame.from_records(records)
    if resolved.empty:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))

    # Joined on the real card's own season_type too -- not just
    # (season, week) -- so a spurious game_type/week combo that shares a
    # week NUMBER with a genuine card of a DIFFERENT season_type (e.g. a
    # stray "REG" row at a week number games.parquet only has as POST) can
    # never collide with that unrelated real card.
    merged = resolved.merge(
        card_table[["season", "source_card_week", "season_type", "qb_state_available_at_utc"]],
        on=["season", "source_card_week", "season_type"], how="inner",
    )
    if merged.duplicated(["season", "source_card_week", "team_id"]).any():
        raise ValueError("Duplicate historical card-state rows produced (season/week/team not unique).")
    merged["source_semantic_version"] = HISTORICAL_SOURCE_SEMANTIC_VERSION
    return merged[list(NORMALIZED_CARD_STATE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 3: live rolling-source normalization to the same weekly concept.
# ---------------------------------------------------------------------------
def normalize_live_card_states(depth_charts_raw: pd.DataFrame, card_table: pd.DataFrame) -> pd.DataFrame:
    if depth_charts_raw is None or len(depth_charts_raw) == 0:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))
    missing = REQUIRED_LIVE_COLUMNS - set(depth_charts_raw.columns)
    if missing:
        raise ValueError(f"Live rolling depth-chart source missing required column(s): {sorted(missing)}")

    live = depth_charts_raw[depth_charts_raw["dt"].notna()].copy()
    if live.empty:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))
    live["dt_utc"] = pd.to_datetime(live["dt"], utc=True, errors="coerce")
    if live["dt_utc"].isna().any():
        raise ValueError("Live rolling depth-chart timestamps failed to parse for at least one row.")
    live["team_canonical"] = live["team"].map(try_canonical_team_id)
    live = live[live["team_canonical"].notna()].copy()
    if live.empty or card_table.empty:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))

    resolved_groups = (
        live.groupby(["team_canonical", "dt_utc"], dropna=True)
        .apply(lambda g: pd.Series(qda._resolve_team_dt_group(g)), include_groups=False)
        .reset_index()
        .rename(columns={"team_canonical": "team_id"})
        .sort_values("dt_utc", kind="stable")
        .reset_index(drop=True)
    )
    status_map = {
        qda.STATUS_RESOLVED: CARD_STATUS_RESOLVED,
        qda.STATUS_NO_QB_ROWS: CARD_STATUS_NO_QB_ROWS,
        qda.STATUS_AMBIGUOUS: CARD_STATUS_AMBIGUOUS,
        qda.STATUS_IDENTIFIER_UNRESOLVED: CARD_STATUS_IDENTIFIER_UNRESOLVED,
    }
    resolved_groups["qb1_resolution_status"] = resolved_groups["qb1_resolution_status"].map(status_map)

    teams = sorted(resolved_groups["team_id"].unique())
    if not teams:
        return pd.DataFrame(columns=list(NORMALIZED_CARD_STATE_COLUMNS))

    n_cards = len(card_table)
    queries = card_table.loc[card_table.index.repeat(len(teams))].reset_index(drop=True)
    queries["team_id"] = teams * n_cards
    queries = queries.sort_values("card_end_utc", kind="stable").reset_index(drop=True)
    # merge_asof requires identical datetime dtypes on both merge keys --
    # parquet-sourced timestamps may carry microsecond ("us") resolution
    # while pd.to_datetime defaults to nanosecond ("ns"); align explicitly
    # rather than let pandas raise.
    queries["card_end_utc"] = queries["card_end_utc"].astype("datetime64[ns, UTC]")
    resolved_groups["dt_utc"] = resolved_groups["dt_utc"].astype("datetime64[ns, UTC]")

    merged = pd.merge_asof(
        queries,
        resolved_groups[["team_id", "dt_utc", "qb1_player_id", "qb1_resolution_status"]],
        left_on="card_end_utc", right_on="dt_utc", by="team_id",
        direction="backward", allow_exact_matches=True,
    )
    no_snapshot = merged["dt_utc"].isna()
    merged.loc[no_snapshot, "qb1_resolution_status"] = CARD_STATUS_NO_SOURCE_DATA
    merged.loc[no_snapshot, "qb1_player_id"] = None
    merged = merged.rename(columns={"qb1_resolution_status": "qb_state_resolution_status"})
    merged["source_semantic_version"] = LIVE_SOURCE_SEMANTIC_VERSION
    if merged.duplicated(["season", "source_card_week", "team_id"]).any():
        raise ValueError("Duplicate live card-state rows produced (season/week/team not unique).")
    return merged[list(NORMALIZED_CARD_STATE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 4: structural bridge proof -- historical adapter output schema ==
# live adapter output schema.
# ---------------------------------------------------------------------------
class SchemaBridgeError(RuntimeError):
    """Raised when the historical and live adapters' output schemas diverge."""


def assert_card_schema_bridge(historical_states: pd.DataFrame, live_states: pd.DataFrame) -> dict:
    hist_cols = list(historical_states.columns)
    live_cols = list(live_states.columns)
    if hist_cols != list(NORMALIZED_CARD_STATE_COLUMNS):
        raise SchemaBridgeError(f"Historical adapter columns {hist_cols} != required {list(NORMALIZED_CARD_STATE_COLUMNS)}")
    if live_cols != list(NORMALIZED_CARD_STATE_COLUMNS):
        raise SchemaBridgeError(f"Live adapter columns {live_cols} != required {list(NORMALIZED_CARD_STATE_COLUMNS)}")
    hist_statuses = set(historical_states["qb_state_resolution_status"].unique())
    live_statuses = set(live_states["qb_state_resolution_status"].unique())
    bad_statuses = (hist_statuses | live_statuses) - set(ALL_CARD_STATUSES)
    if bad_statuses:
        raise SchemaBridgeError(f"Unrecognized card-status value(s) outside the unified vocabulary: {sorted(bad_statuses)}")
    return {
        "schema_equal": True,
        "columns": list(NORMALIZED_CARD_STATE_COLUMNS),
        "historical_row_count": int(len(historical_states)),
        "live_row_count": int(len(live_states)),
        "historical_statuses_observed": sorted(hist_statuses),
        "live_statuses_observed": sorted(live_statuses),
    }


# ---------------------------------------------------------------------------
# Section 5: card-chain continuity/change (source-agnostic -- operates on
# the normalized intermediate, never touches a snapshot timestamp).
# ---------------------------------------------------------------------------
def compute_card_chain_continuity(card_states: pd.DataFrame) -> pd.DataFrame:
    work = card_states.sort_values(["team_id", "season", "source_card_week"], kind="stable").reset_index(drop=True)
    resolved_mask = (work["qb_state_resolution_status"] == CARD_STATUS_RESOLVED).to_numpy()

    changed: list[int] = []
    continuity: list[int] = []
    running_player: dict[tuple, str] = {}
    running_streak: dict[tuple, int] = {}
    for row, is_resolved in zip(work.itertuples(index=False), resolved_mask):
        key = (row.team_id, row.season)
        if not is_resolved:
            changed.append(0)
            continuity.append(0)
            continue  # a gap is skipped, never a reset -- "previous AVAILABLE" state persists across it
        prev_player = running_player.get(key)
        if prev_player is None:
            changed_val, continuity_val = 0, 1
        elif row.qb1_player_id != prev_player:
            changed_val, continuity_val = 1, 1
        else:
            changed_val, continuity_val = 0, running_streak.get(key, 1) + 1
        changed.append(changed_val)
        continuity.append(continuity_val)
        running_player[key] = row.qb1_player_id
        running_streak[key] = continuity_val

    work["qb_depth_changed_at_card"] = changed
    work["qb_depth_continuity_cards_at_card"] = continuity
    return work


# ---------------------------------------------------------------------------
# Section 5 (target level): resolve each horizon-eligible target's required
# PRIOR-CARD state and derive the three frozen candidate features.
# ---------------------------------------------------------------------------
def resolve_lagged_target_state(
    games: pd.DataFrame,
    horizon: str,
    card_table: pd.DataFrame,
    card_states_with_chain: pd.DataFrame,
    *,
    membership_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    work_games["season"] = pd.to_numeric(work_games["season"], errors="raise").astype(int)
    work_games["week"] = pd.to_numeric(work_games["week"], errors="raise").astype(int)
    work_games = work_games.merge(ledger_h, on="game_id", how="inner", validate="one_to_one")

    card_lookup = card_table[["season", "source_card_week", "card_order"]].rename(columns={"source_card_week": "week"})
    work_games = work_games.merge(card_lookup, on=["season", "week"], how="left", validate="many_to_one")
    if work_games["card_order"].isna().any():
        raise ValueError("Some horizon-eligible game(s)' own card is missing from card_table.")

    home = work_games[["game_id", "home_team_id", "season", "card_order", "target_cutoff_utc"]].rename(
        columns={"home_team_id": "team_id"}
    )
    home["side"] = "home"
    away = work_games[["game_id", "away_team_id", "season", "card_order", "target_cutoff_utc"]].rename(
        columns={"away_team_id": "team_id"}
    )
    away["side"] = "away"
    targets = pd.concat([home, away], ignore_index=True)
    targets["horizon"] = horizon
    if targets.empty:
        return pd.DataFrame(columns=list(TARGET_OUTPUT_COLUMNS))

    targets["prior_card_order"] = targets["card_order"] - 1
    has_prior = (targets["prior_card_order"] >= 1).to_numpy()

    prior_card_map = card_table[["season", "card_order", "source_card_week"]].rename(
        columns={"card_order": "prior_card_order", "source_card_week": "prior_card_week"}
    )
    targets = targets.merge(prior_card_map, on=["season", "prior_card_order"], how="left")

    state_map = card_states_with_chain.rename(columns={"source_card_week": "prior_card_week"})[
        [
            "season", "prior_card_week", "team_id", "qb1_player_id", "qb_state_resolution_status",
            "qb_state_available_at_utc", "source_semantic_version",
            "qb_depth_changed_at_card", "qb_depth_continuity_cards_at_card",
        ]
    ]
    out = targets.merge(state_map, on=["season", "prior_card_week", "team_id"], how="left")

    out["lagged_resolution_status"] = TARGET_STATUS_RESOLVED
    out.loc[~has_prior, "lagged_resolution_status"] = TARGET_STATUS_NO_PRIOR_CARD

    status_isna = out["qb_state_resolution_status"].isna().to_numpy()
    no_state_row = has_prior & status_isna
    out.loc[no_state_row, "lagged_resolution_status"] = TARGET_STATUS_NO_SOURCE_DATA

    status_not_resolved = has_prior & ~status_isna & (out["qb_state_resolution_status"] != CARD_STATUS_RESOLVED).to_numpy()
    out.loc[status_not_resolved, "lagged_resolution_status"] = TARGET_STATUS_PRIOR_CARD_UNRESOLVED

    resolved_candidate = has_prior & ~status_isna & (out["qb_state_resolution_status"] == CARD_STATUS_RESOLVED).to_numpy()
    available_before_cutoff = (out["qb_state_available_at_utc"] < out["target_cutoff_utc"]).to_numpy()
    not_yet_available = resolved_candidate & ~available_before_cutoff
    out.loc[not_yet_available, "lagged_resolution_status"] = TARGET_STATUS_STATE_NOT_YET_AVAILABLE

    truly_resolved = resolved_candidate & available_before_cutoff
    out.loc[~truly_resolved, "qb1_player_id"] = None
    out.loc[~truly_resolved, "qb_depth_changed_at_card"] = pd.NA
    out.loc[~truly_resolved, "qb_depth_continuity_cards_at_card"] = pd.NA

    out["qb_depth_state_missing"] = (~truly_resolved).astype(int)
    out["qb_depth_changed"] = pd.to_numeric(out["qb_depth_changed_at_card"], errors="coerce").fillna(0).astype(int)
    out["qb_depth_continuity_cards"] = (
        pd.to_numeric(out["qb_depth_continuity_cards_at_card"], errors="coerce").fillna(0).astype(int)
    )

    if out.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate lagged target-state rows produced.")
    return out[list(TARGET_OUTPUT_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ridge-ready home/away pivoted feature matrix (Section 5's three columns
# per side; no player ID and no market data ever enter this frame).
# ---------------------------------------------------------------------------
def build_lagged_depth_feature_matrix(
    games: pd.DataFrame,
    horizon: str,
    card_table: pd.DataFrame,
    card_states_with_chain: pd.DataFrame,
    *,
    membership_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    resolved = resolve_lagged_target_state(
        games, horizon, card_table, card_states_with_chain, membership_ledger=membership_ledger
    )
    if resolved.empty:
        return pd.DataFrame(columns=["game_id", *CANDIDATE_FEATURE_COLUMNS])
    feature_cols = ["qb_depth_changed", "qb_depth_continuity_cards", "qb_depth_state_missing"]
    wide = resolved.pivot(index="game_id", columns="side", values=feature_cols)
    wide.columns = [f"{side}_{col}" for col, side in wide.columns]
    wide = wide.reset_index()
    missing_cols = [c for c in CANDIDATE_FEATURE_COLUMNS if c not in wide.columns]
    if missing_cols:
        raise ValueError(f"Feature matrix missing expected column(s) (one game side never eligible?): {missing_cols}")
    for col in CANDIDATE_FEATURE_COLUMNS:
        wide[col] = wide[col].astype(int)
    return wide[["game_id", *CANDIDATE_FEATURE_COLUMNS]]


# ---------------------------------------------------------------------------
# Section 6: structural coverage gate (no outcomes -- pure eligibility vs.
# resolvability counts).
# ---------------------------------------------------------------------------
def compute_structural_coverage(
    games: pd.DataFrame,
    horizon: str,
    card_table: pd.DataFrame,
    card_states_with_chain: pd.DataFrame,
    *,
    membership_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    resolved = resolve_lagged_target_state(
        games, horizon, card_table, card_states_with_chain, membership_ledger=membership_ledger
    )
    if resolved.empty:
        return pd.DataFrame(columns=["season", "horizon", "eligible", "usable", "missing", "coverage"])
    grouped = resolved.groupby("season").agg(
        eligible=("game_id", "size"), usable=("qb_depth_state_missing", lambda s: int((s == 0).sum()))
    ).reset_index()
    grouped["missing"] = grouped["eligible"] - grouped["usable"]
    grouped["coverage"] = grouped["usable"] / grouped["eligible"]
    grouped["horizon"] = horizon
    return grouped[["season", "horizon", "eligible", "usable", "missing", "coverage"]]
