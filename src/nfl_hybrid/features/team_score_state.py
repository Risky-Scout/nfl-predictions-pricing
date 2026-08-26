"""Team Score State V1.1 -- ``TEAM_SCORE_STATE_V1_1_OPERATOR_FROZEN``.

Corrected, operator-contract rerun of ``TEAM_SCORE_STATE_V1``. The V0
attempt (see
``$NFL_MODEL_ARTIFACT_ROOT/team-score-state-v1-2026/invalidated-as-built-v0/``)
computed a per-team raw scoring margin (``points_for_avg - points_against_avg``)
and its raw pf/pa components -- an interpretation invented in the absence of a
written feature-level spec. This module instead implements the operator's own
explicit formulas verbatim: league-relative offense/defense deviations,
combined into home/away *expected* score-deviation matchup signals. V0 is
preserved as invalidated, non-authoritative provenance; nothing here reuses
its candidate predictions, selection freeze, or winner arithmetic.

Concept: for target T (a team side, horizon cutoff C):

``team_points_for_mean`` / ``team_points_against_mean`` -- same-season
expanding (cumulative, non-decayed) average points scored / allowed across
that team's own eligible prior REG+POST games this season, gated STRICT by
``result_available_at_utc(prior game) < target_cutoff_utc(T)`` (the same
conservative kickoff+5h floor, and the same chronological event-queue
architecture, already certified for
:func:`nfl_hybrid.features.horizon_elo.build_horizon_elo_state` -- reused here
because it is this codebase's own certified pattern for "replay only what a
target could actually have known by its own cutoff", not because Elo rating
math is involved).

``league_points_per_team_game_mean`` -- mean points scored over ALL eligible
team-game observations from the same season whose game results were
available before C (every prior game in the season contributes two
team-game observations -- its home team's points-for and its away team's
points-for -- to this single shared-per-cutoff league mean).

``offense_score_deviation = team_points_for_mean - league_points_per_team_game_mean``
``defense_allow_deviation = team_points_against_mean - league_points_per_team_game_mean``

Zero prior team games at C: both deviations are 0 and
``score_state_missing = 1``; otherwise ``score_state_missing = 0``. No
previous-season carry (state resets at each season boundary); REG+POST
contribute within the same season. A target's own game never updates its own
state. No market/QB/EPA/injury/weather information ever enters this module.

Matchup formulas (home H, away A, same cutoff C, same league mean):

``home_expected_score_deviation = 0.5 * (home_offense_score_deviation + away_defense_allow_deviation)``
``away_expected_score_deviation = 0.5 * (away_offense_score_deviation + home_defense_allow_deviation)``
``score_state_margin_signal = home_expected_score_deviation - away_expected_score_deviation``
``score_state_total_signal = home_expected_score_deviation + away_expected_score_deviation``
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.features.horizon_elo import (
    HORIZONS,
    RESULT_AVAILABILITY_BASIS,
    RESULT_AVAILABILITY_DURATION_HOURS,
    build_horizon_membership_ledger,
    compute_result_available_at_utc,
)

TRANSFORM_NAME = "build_team_score_state"
TRANSFORM_VERSION = "1.1"
SCHEMA_VERSION = "TEAM_SCORE_STATE_V1_1_OPERATOR_FROZEN"

STATE_COLUMNS: tuple[str, ...] = (
    "game_id", "team_id", "side", "horizon", "season", "target_cutoff_utc",
    "score_state_games_played", "team_points_for_mean", "team_points_against_mean",
    "league_points_per_team_game_mean", "offense_score_deviation", "defense_allow_deviation",
    "score_state_missing",
)

_PER_SIDE_COMPONENT_BASES: tuple[str, ...] = (
    "offense_score_deviation", "defense_allow_deviation", "score_state_missing",
)
PER_SIDE_COMPONENT_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{base}" for side in ("home", "away") for base in _PER_SIDE_COMPONENT_BASES
)

# Candidate B: ELO_PLUS_SCORE_SIGNALS -- the two matchup-level derived
# signals plus each side's missingness flag.
CANDIDATE_SIGNAL_COLUMNS: tuple[str, ...] = (
    "score_state_margin_signal", "score_state_total_signal",
    "home_score_state_missing", "away_score_state_missing",
)
# Candidate C: ELO_PLUS_SCORE_COMPONENTS -- the raw per-side offense/defense
# deviations plus each side's missingness flag. NOT a superset of candidate B
# (it omits the combined margin/total signals) -- deliberately, per the
# operator contract's exact candidate-set specification.
CANDIDATE_COMPONENT_COLUMNS: tuple[str, ...] = (
    "home_offense_score_deviation", "home_defense_allow_deviation",
    "away_offense_score_deviation", "away_defense_allow_deviation",
    "home_score_state_missing", "away_score_state_missing",
)

ALL_MATRIX_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(CANDIDATE_SIGNAL_COLUMNS + CANDIDATE_COMPONENT_COLUMNS)
)


def _canonical_json_hash(payload: Any) -> str:
    """Same recipe used throughout this codebase's frozen-semantics files:
    canonical-primitives-only JSON, no ``default=str`` (a non-primitive
    value must raise, never be silently string-coerced into the hash)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen V1.1 semantics -- persisted verbatim before any outcome-driven fit.
# ---------------------------------------------------------------------------
def team_score_state_v1_1_semantics() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "supersedes": "TEAM_SCORE_STATE_V1 (V0, invalidated -- see invalidated-as-built-v0/invalidation_manifest.json)",
        "concept": (
            "same-season expanding (cumulative, non-decayed) team points-for/points-against means, each "
            "expressed as a deviation from a shared same-season, same-cutoff league_points_per_team_game_mean, "
            "built only from game_result, gated by result_available_at_utc(earlier) < target_cutoff_utc(target) "
            "STRICT"
        ),
        "source": "game_result only (home_score/away_score)",
        "result_availability_rule": "kickoff + RESULT_AVAILABILITY_DURATION_HOURS (reused unchanged from horizon_elo)",
        "result_availability_basis": RESULT_AVAILABILITY_BASIS,
        "result_availability_duration_hours": RESULT_AVAILABILITY_DURATION_HOURS,
        "target_scope": "REG+POST",
        "update_eligibility_rule": (
            "result_available_at_utc(event) < target_cutoff_utc(target) (STRICT) -- identical gating discipline "
            "to build_horizon_elo_state; a target never updates itself and an unavailable/future result never "
            "updates a target's state"
        ),
        "season_boundary_reset": True,
        "cross_season_carryover_prohibited": True,
        "league_mean_definition": (
            "league_points_per_team_game_mean = mean points scored over ALL eligible team-game observations "
            "from the same season whose game results were available before C (every prior game contributes "
            "both its home team's and its away team's points-for as one team-game observation each); shared "
            "identically by every team evaluated at the same cutoff C"
        ),
        "offense_score_deviation_formula": "team_points_for_mean - league_points_per_team_game_mean",
        "defense_allow_deviation_formula": "team_points_against_mean - league_points_per_team_game_mean",
        "missingness_rule": (
            "if a team has zero prior eligible same-season games at a target's cutoff: score_state_missing=1, "
            "offense_score_deviation=0, defense_allow_deviation=0 (team_points_for_mean/team_points_against_mean "
            "also 0); else score_state_missing=0"
        ),
        "missingness_frozen_structurally_not_selected_from_outcomes": True,
        "matchup_formula_home_expected_score_deviation": "0.5 * (home_offense_score_deviation + away_defense_allow_deviation)",
        "matchup_formula_away_expected_score_deviation": "0.5 * (away_offense_score_deviation + home_defense_allow_deviation)",
        "matchup_formula_score_state_margin_signal": "home_expected_score_deviation - away_expected_score_deviation",
        "matchup_formula_score_state_total_signal": "home_expected_score_deviation + away_expected_score_deviation",
        "no_home_field_adjustment_in_matchup_formulas": True,
        "candidate_B_ELO_PLUS_SCORE_SIGNALS": (
            "adds score_state_margin_signal, score_state_total_signal, home_score_state_missing, "
            "away_score_state_missing -- exactly these four, no raw PF/PA averages, no per-team historical "
            "margin averages"
        ),
        "candidate_C_ELO_PLUS_SCORE_COMPONENTS": (
            "adds home_offense_score_deviation, home_defense_allow_deviation, away_offense_score_deviation, "
            "away_defense_allow_deviation, home_score_state_missing, away_score_state_missing -- exactly these "
            "six; deliberately NOT a superset of candidate B (omits the combined margin/total signals), and "
            "never the raw PF/PA averages"
        ),
        "no_market_data": True,
        "no_qb_epa_injury_weather": True,
        "no_actual_starter_or_hindsight_field": True,
        "no_play_by_play_or_drive_data": True,
        "no_feature_search": True,
        "state_columns": list(STATE_COLUMNS),
    }


def compute_semantics_hash(semantics: dict | None = None) -> str:
    return _canonical_json_hash(semantics if semantics is not None else team_score_state_v1_1_semantics())


# ---------------------------------------------------------------------------
# Canonical state builder -- same chronological event-queue pattern as
# build_horizon_elo_state, adapted to (a) a per-(team, season) running
# (games, pf, pa) accumulator and (b) a per-season league-wide running
# (team_games, points_sum) accumulator shared by every team evaluated at the
# same cutoff.
# ---------------------------------------------------------------------------
def _apply_event(state: dict[tuple, dict], league: dict[int, dict], event: dict) -> None:
    season = event["season"]
    for team_id, points_for, points_against in (
        (event["home_team_id"], event["home_score"], event["away_score"]),
        (event["away_team_id"], event["away_score"], event["home_score"]),
    ):
        key = (team_id, season)
        acc = state.setdefault(key, {"games": 0, "pf": 0.0, "pa": 0.0})
        acc["games"] += 1
        acc["pf"] += points_for
        acc["pa"] += points_against

    lacc = league.setdefault(season, {"team_games": 0, "points_sum": 0.0})
    lacc["team_games"] += 2
    lacc["points_sum"] += event["home_score"] + event["away_score"]


def build_team_score_state(
    games: pd.DataFrame, horizon: str, *, membership_ledger: pd.DataFrame | None = None
) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon!r}, expected one of {HORIZONS}")
    h = horizon.lower()

    work = games.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["scheduled_kickoff_utc"] = pd.to_datetime(work["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if work["scheduled_kickoff_utc"].isna().any():
        raise ValueError("build_team_score_state requires a parseable scheduled_kickoff_utc for every game.")
    work["season"] = pd.to_numeric(work["season"], errors="raise").astype(int)
    work["home_team_id"] = work["home_team_id"].map(canonical_team_id)
    work["away_team_id"] = work["away_team_id"].map(canonical_team_id)
    if work["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id rows in games table.")
    work["result_available_at_utc"] = compute_result_available_at_utc(work)

    ledger = membership_ledger if membership_ledger is not None else build_horizon_membership_ledger(games)
    ledger_h = ledger[["game_id", f"{h}_cutoff_utc", f"{h}_eligible"]].rename(
        columns={f"{h}_cutoff_utc": "target_cutoff_utc", f"{h}_eligible": "eligible_for_horizon"}
    )
    work = work.merge(ledger_h, on="game_id", how="left", validate="one_to_one")
    if work["eligible_for_horizon"].isna().any():
        raise ValueError("Membership ledger missing rows for some games in this population.")

    work = work.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)

    state: dict[tuple, dict] = {}
    league: dict[int, dict] = {}
    pending: deque[dict] = deque()
    rows: list[dict] = []

    for game in work.itertuples(index=False):
        if bool(game.eligible_for_horizon):
            cutoff = game.target_cutoff_utc
            while pending and pending[0]["result_available_at_utc"] < cutoff:  # STRICT
                _apply_event(state, league, pending.popleft())

            season = int(game.season)
            lacc = league.get(season)
            league_mean = (lacc["points_sum"] / lacc["team_games"]) if lacc and lacc["team_games"] else 0.0

            for team_id, side in ((game.home_team_id, "home"), (game.away_team_id, "away")):
                acc = state.get((team_id, season))
                if acc is None or acc["games"] == 0:
                    games_played = 0
                    pf_mean = pa_mean = 0.0
                    offense_dev = defense_dev = 0.0
                    missing = 1
                else:
                    games_played = acc["games"]
                    pf_mean = acc["pf"] / acc["games"]
                    pa_mean = acc["pa"] / acc["games"]
                    offense_dev = pf_mean - league_mean
                    defense_dev = pa_mean - league_mean
                    missing = 0
                rows.append(
                    {
                        "game_id": game.game_id, "team_id": team_id, "side": side, "horizon": horizon,
                        "season": season, "target_cutoff_utc": cutoff,
                        "score_state_games_played": games_played,
                        "team_points_for_mean": pf_mean, "team_points_against_mean": pa_mean,
                        "league_points_per_team_game_mean": league_mean,
                        "offense_score_deviation": offense_dev, "defense_allow_deviation": defense_dev,
                        "score_state_missing": missing,
                    }
                )

        home_score, away_score = game.home_score, game.away_score
        if pd.notna(home_score) and pd.notna(away_score):
            pending.append(
                {
                    "season": int(game.season), "home_team_id": game.home_team_id, "away_team_id": game.away_team_id,
                    "home_score": float(home_score), "away_score": float(away_score),
                    "result_available_at_utc": game.result_available_at_utc,
                }
            )

    output = pd.DataFrame(rows)
    if len(output) and output.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate team score-state rows produced.")
    return output[list(STATE_COLUMNS)] if len(output) else pd.DataFrame(columns=list(STATE_COLUMNS))


# ---------------------------------------------------------------------------
# Ridge-ready home/away pivoted feature matrix -- exactly the columns the
# operator contract's candidate B/C need (matchup margin/total signals, raw
# per-side offense/defense deviations, per-side missingness), plus the two
# home/away expected-score-deviation intermediates for transparency/testing.
# No raw PF/PA averages, no per-team historical margin averages, no market
# data, ever in this frame.
# ---------------------------------------------------------------------------
def build_team_score_feature_matrix(
    games: pd.DataFrame, horizon: str, *, membership_ledger: pd.DataFrame | None = None
) -> pd.DataFrame:
    state = build_team_score_state(games, horizon, membership_ledger=membership_ledger)
    output_columns = [
        "game_id", *PER_SIDE_COMPONENT_COLUMNS,
        "home_expected_score_deviation", "away_expected_score_deviation",
        "score_state_margin_signal", "score_state_total_signal",
    ]
    if state.empty:
        return pd.DataFrame(columns=output_columns)

    component_cols = ["offense_score_deviation", "defense_allow_deviation", "score_state_missing"]
    wide = state.pivot(index="game_id", columns="side", values=component_cols)
    wide.columns = [f"{side}_{col}" for col, side in wide.columns]
    wide = wide.reset_index()
    missing_cols = [c for c in PER_SIDE_COMPONENT_COLUMNS if c not in wide.columns]
    if missing_cols:
        raise ValueError(f"Feature matrix missing expected column(s): {missing_cols}")
    for col in PER_SIDE_COMPONENT_COLUMNS:
        if col.endswith("score_state_missing"):
            wide[col] = wide[col].astype(int)
        else:
            wide[col] = wide[col].astype(float)

    wide["home_expected_score_deviation"] = 0.5 * (
        wide["home_offense_score_deviation"] + wide["away_defense_allow_deviation"]
    )
    wide["away_expected_score_deviation"] = 0.5 * (
        wide["away_offense_score_deviation"] + wide["home_defense_allow_deviation"]
    )
    wide["score_state_margin_signal"] = wide["home_expected_score_deviation"] - wide["away_expected_score_deviation"]
    wide["score_state_total_signal"] = wide["home_expected_score_deviation"] + wide["away_expected_score_deviation"]

    return wide[output_columns]
