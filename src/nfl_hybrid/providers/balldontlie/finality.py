"""Strict game finality + per-family completeness gates (Fix 5 Sections 8-10, 23).

Two separate concepts, deliberately never collapsed into one boolean:

- FINALITY: ``status_state == "final"``. Nothing else -- not elapsed kickoff
  time, not a non-null score, not status text containing digits, not the
  presence of stats/plays. If the provider's lifecycle value is missing or
  not one of the known states, this fails CLOSED (treated as not final).

- COMPLETENESS: a final game can still have an incomplete data family (a
  missing team_stats row, unreconciled PBP, ...). Each downstream feature
  family gets its own minimum-completion contract; the gates below combine
  finality with a per-family completeness flag rather than pretending one
  boolean can answer "can I use this game yet?" for every family at once.

Nothing here infers finality or completeness -- callers (canonical.py,
plays.py) compute the completeness booleans from data they actually
received and pass them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

FINAL_STATUS_STATE = "final"

# The full documented/observed status_state lifecycle. Anything outside this
# set -- including a missing value -- is treated as NOT final (fail closed),
# never inferred as final.
NON_FINAL_STATUS_STATES = frozenset(
    {
        "scheduled",
        "in_progress",
        "postponed",
        "canceled",
        "delayed",
        "suspended",
        "abandoned",
        "unknown",
    }
)

RESULT_AVAILABILITY_BASIS = "BDL_FIRST_TRUSTED_FINAL_OBSERVATION"

# -- reason codes (Section 23) ------------------------------------------------
ELIGIBLE_FINAL_SCORE = "ELIGIBLE_FINAL_SCORE"
ELIGIBLE_BOX_COMPLETE = "ELIGIBLE_BOX_COMPLETE"
ELIGIBLE_PBP_COMPLETE = "ELIGIBLE_PBP_COMPLETE"
ELIGIBLE_PLAYER_STATS_COMPLETE = "ELIGIBLE_PLAYER_STATS_COMPLETE"
INELIGIBLE_NOT_FINAL = "INELIGIBLE_NOT_FINAL"
INELIGIBLE_MISSING_SCORE = "INELIGIBLE_MISSING_SCORE"
INELIGIBLE_MISSING_TEAM_STATS = "INELIGIBLE_MISSING_TEAM_STATS"
INELIGIBLE_PBP_INCOMPLETE = "INELIGIBLE_PBP_INCOMPLETE"
INELIGIBLE_UNKNOWN_TEAM = "INELIGIBLE_UNKNOWN_TEAM"
INELIGIBLE_MISSING_PLAYER_STATS = "INELIGIBLE_MISSING_PLAYER_STATS"

# -- as-of eligibility reason codes (Section 4/24 of Fix 5's family-timestamp
# certification) -------------------------------------------------------------
ELIGIBLE_AS_OF = "ELIGIBLE_AS_OF"
INELIGIBLE_NOT_YET_AVAILABLE_AS_OF = "INELIGIBLE_NOT_YET_AVAILABLE_AS_OF"
INELIGIBLE_NEVER_OBSERVED_AVAILABLE = "INELIGIBLE_NEVER_OBSERVED_AVAILABLE"


def is_final(status_state: str | None) -> bool:
    """The ONLY finality rule. No score/time/stat-presence heuristics."""
    return status_state == FINAL_STATUS_STATE


@dataclass(frozen=True)
class CanonicalGameFinality:
    """The minimal fields a finality/completeness gate needs about one game."""

    game_id: str
    status_state: str | None
    home_team: str | None
    away_team: str | None
    home_score: float | None
    away_score: float | None


@dataclass(frozen=True)
class GateResult:
    eligible: bool
    reason_code: str


def _team_identity_ok(game: CanonicalGameFinality) -> bool:
    return bool(game.home_team) and bool(game.away_team)


def can_update_score_state(game: CanonicalGameFinality) -> GateResult:
    """Elo/score/moneyline-style state: FINAL + valid scores + known teams."""
    if not _team_identity_ok(game):
        return GateResult(False, INELIGIBLE_UNKNOWN_TEAM)
    if not is_final(game.status_state):
        return GateResult(False, INELIGIBLE_NOT_FINAL)
    if game.home_score is None or game.away_score is None or pd.isna(game.home_score) or pd.isna(game.away_score):
        return GateResult(False, INELIGIBLE_MISSING_SCORE)
    return GateResult(True, ELIGIBLE_FINAL_SCORE)


def can_update_box_state(game: CanonicalGameFinality, *, box_complete: bool) -> GateResult:
    """Team-box-stat feature family: everything score state requires, plus a
    reconciled one-row-per-team team_stats pair (Section 11)."""
    base = can_update_score_state(game)
    if not base.eligible:
        return base
    if not box_complete:
        return GateResult(False, INELIGIBLE_MISSING_TEAM_STATS)
    return GateResult(True, ELIGIBLE_BOX_COMPLETE)


def can_update_pbp_state(game: CanonicalGameFinality, *, pbp_complete: bool) -> GateResult:
    """PBP/EPA-eligible-family gate: everything score state requires, plus a
    conservative PBP completeness proof (Section 20). A final game with
    incomplete PBP must NOT update this family even though it may already be
    eligible for score/box state."""
    base = can_update_score_state(game)
    if not base.eligible:
        return base
    if not pbp_complete:
        return GateResult(False, INELIGIBLE_PBP_INCOMPLETE)
    return GateResult(True, ELIGIBLE_PBP_COMPLETE)


def can_update_player_stats_state(game: CanonicalGameFinality, *, player_stats_complete: bool) -> GateResult:
    """Player/QB box-stat feature family: everything score state requires,
    plus the player-stat completeness floor (``canonical.player_stats_completeness``:
    at least one player-stat row exists for the game -- see that function's
    docstring for why this is a floor, not a full-roster guarantee)."""
    base = can_update_score_state(game)
    if not base.eligible:
        return base
    if not player_stats_complete:
        return GateResult(False, INELIGIBLE_MISSING_PLAYER_STATS)
    return GateResult(True, ELIGIBLE_PLAYER_STATS_COMPLETE)


# -- result_available_at_utc (Section 9) --------------------------------------

RESULT_AVAILABILITY_LEDGER_COLUMNS = (
    "game_id",
    "result_available_at_utc",
    "result_availability_basis",
    "first_final_provider_status",
    "first_final_observed_at_utc",
)


def compute_result_available_at_utc(observations: pd.DataFrame) -> pd.DataFrame:
    """The first trusted ingestion observation, per game, at which
    ``status_state == "final"``.

    ``observations`` is an append-only ingestion log: one row per polling
    event, columns ``game_id``, ``status_state``, ``provider_status``
    (the raw human-readable status text, for traceability only -- never used
    to decide finality), and ``observed_at_utc`` (when THIS system received
    that poll response, never a provider-claimed final-whistle time BDL does
    not actually supply).

    Deterministic and idempotent: re-running this over the same (possibly
    longer, since new polls only ever append) observation log for a game
    that already has a recorded first-final observation returns the exact
    same timestamp -- a later poll, even one that re-confirms
    ``status_state == "final"``, can never overwrite it. A game with no
    FINAL observation yet is simply absent from the result (never a
    fabricated placeholder).
    """
    required = {"game_id", "status_state", "observed_at_utc"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    final_rows = observations.loc[observations["status_state"] == FINAL_STATUS_STATE].copy()
    if final_rows.empty:
        return pd.DataFrame(columns=RESULT_AVAILABILITY_LEDGER_COLUMNS)

    final_rows["observed_at_utc"] = pd.to_datetime(final_rows["observed_at_utc"], utc=True, errors="raise")
    final_rows = final_rows.sort_values(["game_id", "observed_at_utc"], kind="stable")
    first = final_rows.groupby("game_id", as_index=False, sort=False).first()

    result = pd.DataFrame(
        {
            "game_id": first["game_id"],
            "result_available_at_utc": first["observed_at_utc"],
            "result_availability_basis": RESULT_AVAILABILITY_BASIS,
            "first_final_provider_status": first["provider_status"] if "provider_status" in first.columns else pd.NA,
            "first_final_observed_at_utc": first["observed_at_utc"],
        }
    )
    return result.reset_index(drop=True)


# -- family-specific first-usable timestamps (Fix 5 certification, Sections
# 3-5) -------------------------------------------------------------------
#
# A single game-level `result_available_at_utc` (above) only answers "when
# did THIS game first go final" -- it is NOT sufficient to drive historical
# Tuesday/Friday replay, because a downstream family's own completeness gate
# can pass strictly LATER than plain finality: a game can go final Thursday
# night while its team-box, player-stat, or PBP completeness gate does not
# pass until a later poll (Friday, Saturday, ...). Treating
# `result_available_at_utc` as a stand-in for "PBP is usable" would let a
# game that finalized Thursday leak PBP-derived state into a Friday cutoff
# before PBP was actually complete -- exactly the contamination Section 3
# forbids.
#
# The functions below compute FOUR independent first-usable timestamps per
# game -- `score_available_at_utc`, `box_available_at_utc`,
# `player_stats_available_at_utc`, `pbp_available_at_utc` -- each defined as
# the first observation in an append-only polling log at which THAT
# family's own gate (`can_update_score_state` / `can_update_box_state` /
# `can_update_player_stats_state` / `can_update_pbp_state`) passed. Exactly
# like `compute_result_available_at_utc`, each is monotonic and immutable:
# once recorded, a later poll can never move an already-recorded family
# timestamp earlier or later, and one family's gate passing has zero effect
# on another family's recorded timestamp.

FAMILY_SCORE = "score"
FAMILY_BOX = "box"
FAMILY_PLAYER_STATS = "player_stats"
FAMILY_PBP = "pbp"

FAMILY_AVAILABILITY_COLUMNS = (
    "game_id",
    "score_available_at_utc",
    "box_available_at_utc",
    "player_stats_available_at_utc",
    "pbp_available_at_utc",
)

_FAMILY_OBSERVATION_REQUIRED_COLUMNS = frozenset(
    {
        "game_id",
        "observed_at_utc",
        "status_state",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "box_complete",
        "player_stats_complete",
        "pbp_complete",
    }
)

_FAMILY_TO_COLUMN = {
    FAMILY_SCORE: "score_available_at_utc",
    FAMILY_BOX: "box_available_at_utc",
    FAMILY_PLAYER_STATS: "player_stats_available_at_utc",
    FAMILY_PBP: "pbp_available_at_utc",
}


def _family_gate_eligible_columns(observations: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean Series per family (aligned to ``observations.index``):
    whether THIS single polling observation's own gate passes -- computed
    from the exact same gate functions the rest of this module uses, never
    a separate ad hoc rule."""
    eligible: dict[str, list[bool]] = {family: [] for family in _FAMILY_TO_COLUMN}
    for row in observations.itertuples(index=False):
        game = CanonicalGameFinality(
            game_id=row.game_id,
            status_state=row.status_state,
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=row.home_score,
            away_score=row.away_score,
        )
        eligible[FAMILY_SCORE].append(can_update_score_state(game).eligible)
        eligible[FAMILY_BOX].append(can_update_box_state(game, box_complete=bool(row.box_complete)).eligible)
        eligible[FAMILY_PLAYER_STATS].append(
            can_update_player_stats_state(game, player_stats_complete=bool(row.player_stats_complete)).eligible
        )
        eligible[FAMILY_PBP].append(can_update_pbp_state(game, pbp_complete=bool(row.pbp_complete)).eligible)
    return {family: pd.Series(values, index=observations.index) for family, values in eligible.items()}


def compute_family_available_at_utc(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute all four family-specific first-usable timestamps from an
    append-only polling log.

    ``observations`` columns, one row per polling event: ``game_id``,
    ``observed_at_utc``, ``status_state``, ``home_team``, ``away_team``,
    ``home_score``, ``away_score``, ``box_complete``,
    ``player_stats_complete``, ``pbp_complete`` -- the last three must be
    actual booleans for the rows they apply to (never ``NaN``, which is
    truthy).

    Returns one row per ``game_id`` seen in the log with
    ``score_available_at_utc`` / ``box_available_at_utc`` /
    ``player_stats_available_at_utc`` / ``pbp_available_at_utc``, each
    ``NaT`` if that family's own gate never passed anywhere in the log
    (never a fabricated placeholder).

    Deterministic and idempotent per family, independently: re-running this
    over a longer (append-only) log never changes an already-recorded
    family timestamp, and one family's completeness gate passing has zero
    effect on another family's recorded timestamp -- see
    ``tests/test_balldontlie_finality.py`` family-timestamp tests.
    """
    missing = _FAMILY_OBSERVATION_REQUIRED_COLUMNS - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    if observations.empty:
        return pd.DataFrame(columns=FAMILY_AVAILABILITY_COLUMNS)

    obs = observations.copy()
    obs["observed_at_utc"] = pd.to_datetime(obs["observed_at_utc"], utc=True, errors="raise")
    obs = obs.sort_values(["game_id", "observed_at_utc"], kind="stable").reset_index(drop=True)

    eligible_columns = _family_gate_eligible_columns(obs)

    result = pd.DataFrame({"game_id": pd.Series(obs["game_id"].unique())})
    for family, out_column in _FAMILY_TO_COLUMN.items():
        family_eligible = obs.loc[eligible_columns[family]]
        first = family_eligible.groupby("game_id", as_index=False, sort=False)["observed_at_utc"].first()
        first = first.rename(columns={"observed_at_utc": out_column})
        result = result.merge(first, on="game_id", how="left")

    return result.reindex(columns=list(FAMILY_AVAILABILITY_COLUMNS)).reset_index(drop=True)


def family_eligible_as_of(family_available_at_utc: Any, as_of_utc: Any) -> GateResult:
    """Section 4's shared as-of rule: eligible iff the family's own
    first-usable timestamp (from ``compute_family_available_at_utc``) exists
    and is ``<= as_of_utc``. Identical logic for all four families -- the
    family-specific work already happened when the timestamp was computed;
    this is only the monotonic comparison. Never infers eligibility from
    anything else (never from a later snapshot, never from
    ``result_available_at_utc``, never from any other family's timestamp)."""
    if family_available_at_utc is None or pd.isna(family_available_at_utc):
        return GateResult(False, INELIGIBLE_NEVER_OBSERVED_AVAILABLE)
    available = pd.Timestamp(family_available_at_utc)
    cutoff = pd.Timestamp(as_of_utc)
    if available.tzinfo is None:
        available = available.tz_localize("UTC")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    if available <= cutoff:
        return GateResult(True, ELIGIBLE_AS_OF)
    return GateResult(False, INELIGIBLE_NOT_YET_AVAILABLE_AS_OF)
