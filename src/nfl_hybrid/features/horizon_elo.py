"""Fix 7.1 V2: card-scoped horizon-as-of Elo pregame state -- TUE/FRI
operational semantics, corrected.

V1 (superseded, NON-CERTIFIED -- preserved under
``NFL_MODEL_ARTIFACT_ROOT/invalidated/fix71-per-game-horizon-membership-2026-08-24/``)
computed each game's own horizon cutoff independently from THAT game's own
kickoff, floored backward until strictly earlier. That let a game whose own
kickoff preceded its week's real Friday-noon batch (a Thursday/Thanksgiving/
Christmas game) silently borrow the PREVIOUS week's Friday cutoff instead of
being excluded as an ineligible Friday-forecast target. A real production
Friday batch forecasts a CARD (one week's slate) as a single unit; it never
re-derives a different cutoff per game within that slate.

V2 fixes this: the horizon cutoff is computed once per CARD
``(season, week, season_type)``, from that card's own earliest kickoff, and
applied identically to every game in the card
(:func:`build_horizon_membership_ledger`, the one authoritative source).
A game whose own kickoff is not strictly after its card's horizon cutoff is
NOT a valid forecast target for that horizon -- it is excluded from the
horizon's target population entirely, never floated to a different card's
cutoff.

Target eligibility (may this game get a supervised prediction/training row
for horizon H) is kept STRICTLY separate from Elo update-event eligibility
(may an earlier game's already-known result update the engine before THIS
target is predicted): the latter is governed purely by
``result_available_at_utc(earlier) < target_cutoff_utc(this target)``
(STRICT), over every REG+POST game regardless of whether that earlier game
was itself ever a valid target for this horizon
(:func:`build_horizon_elo_state`).
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict
from datetime import date, time, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_hybrid.data.availability import add_postgame_available_at
from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.legacy.elo import EloConfig, EloContext, LegacyElo

TRANSFORM_NAME = "build_horizon_elo_state"
TRANSFORM_VERSION = "2.0"
SCHEMA_VERSION = "fix7-1-horizon-asof-elo-v2-card-scoped"
HORIZON_FEATURE_SEMANTICS_VERSION = "HORIZON_CUTOFF_ASOF_ELO_V2_CARD_SCOPED"

NY_ZONE = ZoneInfo("America/New_York")
HORIZONS: tuple[str, ...] = ("TUE", "FRI")
# Days after the card's Monday (America/New_York local calendar).
_HORIZON_DAY_OFFSET: dict[str, int] = {"TUE": 1, "FRI": 4}
_HORIZON_ANCHOR_HOUR_LOCAL = 12
CARD_KEY: tuple[str, ...] = ("season", "week", "season_type")

RESULT_AVAILABILITY_DURATION_HOURS = 5.0
RESULT_AVAILABILITY_BASIS = "HISTORICAL_CONSERVATIVE_FLOOR_KICKOFF_PLUS_5H"

ELIGIBLE_REASON = "ELIGIBLE"
INELIGIBLE_REASON = "CUTOFF_NOT_BEFORE_KICKOFF"

REQUIRED_GAME_COLUMNS = {
    "game_id", "season", "week", "season_type",
    "home_team_id", "away_team_id", "scheduled_kickoff_utc", "home_score", "away_score",
}


# ---------------------------------------------------------------------------
# Card-scoped horizon membership -- the ONE authoritative source. No other
# function in this module (or any caller) may derive a cutoff any other way.
# ---------------------------------------------------------------------------
def _card_monday_date(card_earliest_kickoff_utc: pd.Timestamp) -> date:
    local = card_earliest_kickoff_utc.tz_convert(NY_ZONE)
    d = local.date()
    return d - timedelta(days=d.weekday())


def _card_noon_cutoff_utc(monday_date: date, days_after_monday: int) -> pd.Timestamp:
    dt = monday_date + timedelta(days=days_after_monday)
    return pd.Timestamp.combine(dt, time(_HORIZON_ANCHOR_HOUR_LOCAL, 0)).tz_localize(NY_ZONE).tz_convert("UTC")


def compute_result_available_at_utc(games: pd.DataFrame) -> pd.Series:
    """Conservative historical result-availability timestamp per game --
    reused unchanged from the existing repository policy (kickoff + 5h
    floor). Never a fabricated exact final-whistle time."""
    return add_postgame_available_at(
        games, kickoff_column="scheduled_kickoff_utc", duration_hours=RESULT_AVAILABILITY_DURATION_HOURS
    )


def build_horizon_membership_ledger(games: pd.DataFrame) -> pd.DataFrame:
    """One row per ``game_id``: its card, that card's Monday, both horizons'
    cutoffs, and both horizons' eligibility booleans/reasons. Card
    membership is always the canonical ``(season, week, season_type)`` --
    never a per-game re-derivation.

    Eligibility: ``target_cutoff_utc(card, horizon) < scheduled_kickoff_utc``
    (STRICT). An ineligible game is simply excluded from that horizon's
    target population -- it is never floated to a different card's cutoff.
    """
    missing = REQUIRED_GAME_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing required columns: {sorted(missing)}")

    work = games.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["scheduled_kickoff_utc"] = pd.to_datetime(work["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if work["scheduled_kickoff_utc"].isna().any():
        raise ValueError("Horizon membership requires a parseable scheduled_kickoff_utc for every game.")
    if work["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id rows in games table.")

    card_earliest = (
        work.groupby(list(CARD_KEY))["scheduled_kickoff_utc"].min().rename("card_earliest_kickoff_utc").reset_index()
    )
    card_earliest["card_monday_date"] = card_earliest["card_earliest_kickoff_utc"].apply(_card_monday_date)
    for horizon, offset in _HORIZON_DAY_OFFSET.items():
        card_earliest[f"{horizon.lower()}_cutoff_utc"] = card_earliest["card_monday_date"].apply(
            lambda d, off=offset: _card_noon_cutoff_utc(d, off)
        )

    ledger = work.merge(
        card_earliest[list(CARD_KEY) + ["card_monday_date"] + [f"{h.lower()}_cutoff_utc" for h in HORIZONS]],
        on=list(CARD_KEY), how="left", validate="many_to_one",
    )
    for horizon in HORIZONS:
        h = horizon.lower()
        eligible = ledger[f"{h}_cutoff_utc"] < ledger["scheduled_kickoff_utc"]
        ledger[f"{h}_eligible"] = eligible
        ledger[f"{h}_reason"] = eligible.map({True: ELIGIBLE_REASON, False: INELIGIBLE_REASON})

    keep_cols = ["game_id"] + list(CARD_KEY) + ["scheduled_kickoff_utc", "card_monday_date"]
    for horizon in HORIZONS:
        h = horizon.lower()
        keep_cols += [f"{h}_cutoff_utc", f"{h}_eligible", f"{h}_reason"]
    return ledger[keep_cols].reset_index(drop=True)


def compute_horizon_membership_ledger_hash(ledger: pd.DataFrame) -> str:
    cols = ["game_id"] + list(CARD_KEY) + ["scheduled_kickoff_utc"]
    for horizon in HORIZONS:
        h = horizon.lower()
        cols += [f"{h}_cutoff_utc", f"{h}_eligible", f"{h}_reason"]
    ordered = ledger[cols].sort_values("game_id").reset_index(drop=True)
    records = ordered.astype(object).where(pd.notnull(ordered), None).to_dict(orient="records")
    payload = json.dumps({"schema_version": SCHEMA_VERSION, "columns": cols, "rows": records}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eligible_game_ids(ledger: pd.DataFrame, horizon: str) -> set[str]:
    h = horizon.lower()
    return set(ledger.loc[ledger[f"{h}_eligible"], "game_id"])


# ---------------------------------------------------------------------------
# Elo event replay: ALL REG+POST games participate as UPDATE EVENTS
# regardless of target eligibility; only horizon-eligible games are ever
# PREDICTED (get a row). Never conflate the two.
# ---------------------------------------------------------------------------
def _prepare(games: pd.DataFrame) -> pd.DataFrame:
    work = games.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["scheduled_kickoff_utc"] = pd.to_datetime(work["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if work["scheduled_kickoff_utc"].isna().any():
        raise ValueError("Elo pregame state requires a parseable scheduled_kickoff_utc for every game.")
    if work["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id rows in games table.")
    work["home_team_id"] = work["home_team_id"].map(canonical_team_id)
    work["away_team_id"] = work["away_team_id"].map(canonical_team_id)
    work["neutral_site"] = work["neutral_site"].fillna(False).astype(bool) if "neutral_site" in work else False
    work["playoff"] = work["playoff"].fillna(False).astype(bool) if "playoff" in work else False
    return work


def build_horizon_elo_state(
    games: pd.DataFrame,
    horizon: str,
    *,
    config: EloConfig | None = None,
    membership_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (game_id, team_id) -- ONLY for games eligible as a
    ``horizon`` target (per :func:`build_horizon_membership_ledger`). Every
    REG+POST row in ``games`` still participates in the underlying Elo
    UPDATE-EVENT stream (chronological, ``result_available_at_utc <
    target_cutoff_utc`` STRICT gates each update), whether or not it is
    itself ever predicted -- an ineligible game's own result can still
    update Elo for a later eligible target once available.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon!r}, expected one of {HORIZONS}")
    h = horizon.lower()

    work = _prepare(games)
    work["result_available_at_utc"] = compute_result_available_at_utc(work)

    ledger = membership_ledger if membership_ledger is not None else build_horizon_membership_ledger(games)
    ledger_h = ledger[["game_id", f"{h}_cutoff_utc", f"{h}_eligible"]].rename(
        columns={f"{h}_cutoff_utc": "target_cutoff_utc", f"{h}_eligible": "eligible_for_horizon"}
    )
    work = work.merge(ledger_h, on="game_id", how="left", validate="one_to_one")
    if work["eligible_for_horizon"].isna().any():
        raise ValueError("Membership ledger missing rows for some games in this population.")

    work = work.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)

    elo = LegacyElo(config=config or EloConfig())
    # The season the CURRENT ratings conceptually represent -- noted either
    # when an update is APPLIED or when an ELIGIBLE target is about to be
    # PREDICTED (never for an ineligible game, which gets neither).
    current_ratings_season: int | None = None
    pending: deque[dict] = deque()
    rows: list[dict[str, object]] = []

    def _note_season(season: int) -> None:
        nonlocal current_ratings_season
        if current_ratings_season is not None and season != current_ratings_season:
            elo.regress_to_mean()
        current_ratings_season = season

    def _apply(entry: dict) -> None:
        _note_season(entry["season"])
        elo.update(
            entry["home_team_id"], entry["away_team_id"],
            entry["home_score"], entry["away_score"],
            entry["context"], completed=True,
        )

    for game in work.itertuples(index=False):
        eligible = bool(game.eligible_for_horizon)
        if eligible:
            cutoff = game.target_cutoff_utc
            # STRICT: result_available_at_utc < target_cutoff_utc.
            while pending and pending[0]["result_available_at_utc"] < cutoff:
                _apply(pending.popleft())
            _note_season(int(game.season))

            context = EloContext(neutral_site=bool(game.neutral_site), playoff=bool(game.playoff))
            prediction = elo.predict(game.home_team_id, game.away_team_id, context)

            for team_id, side, rating, win_probability, expected_margin in (
                (
                    game.home_team_id, "home",
                    prediction.home_rating, prediction.home_win_probability, prediction.expected_home_margin,
                ),
                (
                    game.away_team_id, "away",
                    prediction.away_rating, 1.0 - prediction.home_win_probability, -prediction.expected_home_margin,
                ),
            ):
                rows.append(
                    {
                        "game_id": game.game_id, "team_id": team_id, "season": int(game.season),
                        "week": int(game.week), "season_type": game.season_type,
                        "event_time": game.scheduled_kickoff_utc,
                        "horizon": horizon,
                        "target_cutoff_utc": cutoff,
                        "eligible_for_horizon": True,
                        "result_available_at_utc": game.result_available_at_utc,
                        "result_availability_basis": RESULT_AVAILABILITY_BASIS,
                        "elo_pregame_side": side,
                        "elo_pregame_rating": rating,
                        "elo_pregame_win_probability": win_probability,
                        "elo_pregame_expected_margin": expected_margin,
                    }
                )

        home_score = game.home_score
        away_score = game.away_score
        played = pd.notna(home_score) and pd.notna(away_score)
        if played:
            # Enqueued regardless of this game's OWN target eligibility --
            # its result may still update Elo for a LATER eligible target.
            ctx = EloContext(neutral_site=bool(game.neutral_site), playoff=bool(game.playoff))
            pending.append(
                {
                    "season": int(game.season), "home_team_id": game.home_team_id, "away_team_id": game.away_team_id,
                    "home_score": float(home_score), "away_score": float(away_score), "context": ctx,
                    "result_available_at_utc": game.result_available_at_utc,
                }
            )

    output = pd.DataFrame(rows)
    if len(output) and output.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate Elo pregame rows produced.")
    return output


# ---------------------------------------------------------------------------
# Elo config fingerprint (unchanged Elo math -- fingerprinted, not modified).
# ---------------------------------------------------------------------------
def compute_elo_config_hash(config: EloConfig | None = None) -> str:
    payload = json.dumps({"elo_config": asdict(config or EloConfig())}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# V2 non-negotiable semantics fingerprint -- frozen preregistration content,
# not a fitted spec: no candidate, no hyperparameter, no 2024 result enters
# this.
# ---------------------------------------------------------------------------
def horizon_semantics_spec_v2(
    *, fix6_feature_manifest_hash: str, fix6_frozen_feature_columns: Sequence[str], elo_config_hash: str
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "horizon_feature_semantics_version": HORIZON_FEATURE_SEMANTICS_VERSION,
        "ordered_six_features": list(fix6_frozen_feature_columns),
        "fix6_feature_manifest_hash": fix6_feature_manifest_hash,
        "elo_config_hash": elo_config_hash,
        "football_event_order": "scheduled_kickoff_utc ascending, deterministic game_id tie-break",
        "elo_update_eligibility_rule": "result_available_at_utc(event) < target_cutoff_utc(target) (STRICT)",
        "result_availability_basis": RESULT_AVAILABILITY_BASIS,
        "result_availability_duration_hours": RESULT_AVAILABILITY_DURATION_HOURS,
        "target_never_updates_itself": True,
        "unavailable_or_future_result_never_updates_target_state": True,
        "postseason_results_update_elo_when_chronologically_available": True,
        "postseason_state_carries_into_next_season_then_regress_to_mean": True,
        "target_scope": "REG+POST",
        "card_membership_key": list(CARD_KEY),
        "card_monday_algorithm": (
            "card_monday_date = earliest scheduled_kickoff_utc in (season, week, season_type) "
            "-> America/New_York local calendar date -> minus weekday() days"
        ),
        "tue_cutoff_rule": "card_monday_date + 1 day, 12:00 America/New_York, DST-aware",
        "fri_cutoff_rule": "card_monday_date + 4 days, 12:00 America/New_York, DST-aware",
        "target_eligibility_rule": "target_cutoff_utc(card, horizon) < scheduled_kickoff_utc(game) (STRICT)",
        "ineligible_target_handling": (
            "excluded entirely from that horizon's target population; never floated to a different card's cutoff"
        ),
        "target_vs_update_event_distinction": (
            "target eligibility (does this game get a prediction/training row for horizon H) is independent of "
            "update-event eligibility (does an earlier game's known result update the engine before a target is "
            "predicted); an ineligible target's own result may still update Elo for a later eligible target once "
            "available"
        ),
        "no_game_id_special_cases": True,
        "no_market_data": True,
        "no_ats_total": True,
        "no_roi_clv_profit": True,
        "no_2025_or_2026_outcomes_in_certification": True,
    }


def compute_horizon_feature_semantics_hash_v2(
    *, fix6_feature_manifest_hash: str, fix6_frozen_feature_columns: Sequence[str], elo_config_hash: str
) -> str:
    payload = json.dumps(
        horizon_semantics_spec_v2(
            fix6_feature_manifest_hash=fix6_feature_manifest_hash,
            fix6_frozen_feature_columns=fix6_frozen_feature_columns,
            elo_config_hash=elo_config_hash,
        ),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
