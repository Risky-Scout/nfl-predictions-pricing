"""BDL raw plays -> canonical, provider-neutral play-by-play (Sections 15, 20).

Field names verified live (game 423945, 2025 week 1) on 2026-08-21:
``id`` (string), ``game`` (nested full game object), ``type_slug``,
``type_abbreviation``, ``type_text``, ``text``, ``short_text``,
``away_score``, ``home_score``, ``scoring_play``, ``period``,
``clock_display``, ``team`` (the possessing team), ``start_yard_line``,
``start_down``, ``start_distance``, ``start_yards_to_endzone``,
``end_yard_line``, ``end_down``, ``end_distance``, ``end_yards_to_endzone``,
``end_down_distance_text``, ``end_short_down_distance_text``,
``end_possession_text``, ``stat_yardage``, ``home_win_probability``,
``wallclock`` (ISO timestamp).

Critically: there is no ``epa``, ``wpa``, ``success``, or ``cpoe`` field
anywhere on this payload. ``home_win_probability`` IS present but is a
provider-native win-probability figure, not nflverse's ``wpa`` -- no
validated transform between the two exists, so it is never relabeled as
one. See ``parity.py`` / ``docs/BDL_PARITY_MATRIX.md`` for the full
EPA/advanced-feature audit (Section 16).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

import pandas as pd

from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.providers.balldontlie.team_crosswalk import (
    TeamCrosswalkError,
    canonical_team_from_team_object,
)

PROVIDER_NAME = "balldontlie"

CANONICAL_PLAY_COLUMNS = (
    "provider", "provider_game_id", "provider_play_id", "sequence",
    "period", "clock_display", "wallclock_utc",
    "possession_team",
    "down", "distance", "yardline_100",
    "end_down", "end_distance", "end_yardline_100",
    "play_type_slug", "play_type_abbreviation", "play_type_text",
    "description", "short_description",
    "stat_yardage", "scoring_play",
    "home_score", "away_score",
    "provider_home_win_probability",
    "collected_at_utc", "raw_payload_hash",
)


def _payload_hash(raw: dict[str, Any]) -> str:
    encoded = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PlaysNormalizationReport:
    provider_game_id: int
    raw_row_count: int
    canonical_row_count: int
    exact_duplicate_play_ids: tuple[str, ...] = field(default_factory=tuple)
    conflicting_play_ids: tuple[str, ...] = field(default_factory=tuple)
    foreign_game_play_ids: tuple[str, ...] = field(default_factory=tuple)
    period_ordering_violations: int = 0

    @property
    def clean(self) -> bool:
        return not (self.conflicting_play_ids or self.foreign_game_play_ids or self.period_ordering_violations)


def normalize_plays(
    raw_plays: list[dict[str, Any]],
    *,
    provider_game_id: int,
    retrieved_at_utc: str | None = None,
) -> tuple[pd.DataFrame, PlaysNormalizationReport]:
    """Normalize one game's raw plays into canonical PBP rows.

    Dedup/reject rules (Section 15), applied per ``provider_play_id``:

    - identical raw payload repeated across pages -> kept once, counted in
      ``exact_duplicate_play_ids``.
    - same id, DIFFERENT raw payload -> both excluded from the canonical
      output, id recorded in ``conflicting_play_ids`` (never silently pick
      one version).
    - a play whose own ``game.id`` does not match ``provider_game_id`` ->
      excluded, recorded in ``foreign_game_play_ids``.

    Final row order is deterministic: sorted by (``period``, ``wallclock``,
    ``provider_play_id``), never provider response order. A play in which
    ``period`` decreases while wallclock time increases relative to the
    previous row is impossible and increments
    ``period_ordering_violations`` (the row is still kept -- this is a
    reported anomaly, not a silent drop, since it does not indicate which
    of the two rows is wrong).
    """
    retrieved = retrieved_at_utc or utc_now_iso()

    by_id: dict[str, list[dict[str, Any]]] = {}
    foreign: list[str] = []
    for raw in raw_plays:
        play_id = str(raw.get("id"))
        game_obj = raw.get("game") or {}
        if game_obj.get("id") is not None and int(game_obj["id"]) != int(provider_game_id):
            foreign.append(play_id)
            continue
        by_id.setdefault(play_id, []).append(raw)

    exact_duplicates: list[str] = []
    conflicting: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for play_id, variants in by_id.items():
        if len(variants) == 1:
            accepted[play_id] = variants[0]
            continue
        hashes = {_payload_hash(v) for v in variants}
        if len(hashes) == 1:
            exact_duplicates.append(play_id)
            accepted[play_id] = variants[0]
        else:
            conflicting.append(play_id)  # excluded entirely -- never guess which is right

    rows: list[dict[str, Any]] = []
    for play_id, raw in accepted.items():
        team_obj = raw.get("team")
        try:
            possession_team = canonical_team_from_team_object(team_obj) if team_obj else None
        except TeamCrosswalkError:
            possession_team = None

        rows.append(
            {
                "provider": PROVIDER_NAME,
                "provider_game_id": provider_game_id,
                "provider_play_id": play_id,
                "period": pd.to_numeric(raw.get("period"), errors="coerce"),
                "clock_display": raw.get("clock_display"),
                "wallclock_utc": pd.to_datetime(raw.get("wallclock"), utc=True, errors="coerce"),
                "possession_team": possession_team,
                "down": pd.to_numeric(raw.get("start_down"), errors="coerce"),
                "distance": pd.to_numeric(raw.get("start_distance"), errors="coerce"),
                "yardline_100": pd.to_numeric(raw.get("start_yards_to_endzone"), errors="coerce"),
                "end_down": pd.to_numeric(raw.get("end_down"), errors="coerce"),
                "end_distance": pd.to_numeric(raw.get("end_distance"), errors="coerce"),
                "end_yardline_100": pd.to_numeric(raw.get("end_yards_to_endzone"), errors="coerce"),
                "play_type_slug": raw.get("type_slug"),
                "play_type_abbreviation": raw.get("type_abbreviation"),
                "play_type_text": raw.get("type_text"),
                "description": raw.get("text"),
                "short_description": raw.get("short_text"),
                "stat_yardage": pd.to_numeric(raw.get("stat_yardage"), errors="coerce"),
                "scoring_play": bool(raw.get("scoring_play")) if raw.get("scoring_play") is not None else pd.NA,
                "home_score": pd.to_numeric(raw.get("home_score"), errors="coerce"),
                "away_score": pd.to_numeric(raw.get("away_score"), errors="coerce"),
                "provider_home_win_probability": pd.to_numeric(raw.get("home_win_probability"), errors="coerce"),
                "collected_at_utc": retrieved,
                "raw_payload_hash": _payload_hash(raw),
            }
        )

    frame = pd.DataFrame(rows, columns=list(CANONICAL_PLAY_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values(
            ["period", "wallclock_utc", "provider_play_id"], kind="stable"
        ).reset_index(drop=True)
        frame["sequence"] = range(1, len(frame) + 1)

        violations = 0
        prev_period = None
        prev_wallclock = None
        for period, wallclock in zip(frame["period"], frame["wallclock_utc"]):
            if prev_period is not None and pd.notna(period) and pd.notna(prev_period) and pd.notna(wallclock) and pd.notna(prev_wallclock):
                if period < prev_period and wallclock > prev_wallclock:
                    violations += 1
            prev_period, prev_wallclock = period, wallclock
    else:
        violations = 0

    report = PlaysNormalizationReport(
        provider_game_id=provider_game_id,
        raw_row_count=len(raw_plays),
        canonical_row_count=len(frame),
        exact_duplicate_play_ids=tuple(exact_duplicates),
        conflicting_play_ids=tuple(conflicting),
        foreign_game_play_ids=tuple(foreign),
        period_ordering_violations=violations,
    )
    return frame, report


@dataclass(frozen=True)
class PBPCompletenessReport:
    provider_game_id: int
    game_final: bool
    play_count: int
    nonempty: bool
    unique_play_ids: bool
    normalization_clean: bool
    terminal_score_reconciled: bool | None  # None = unavailable to check
    pbp_complete: bool


def pbp_completeness(
    plays: pd.DataFrame,
    normalization_report: PlaysNormalizationReport,
    *,
    game_final: bool,
    canonical_home_score: float | None,
    canonical_away_score: float | None,
) -> PBPCompletenessReport:
    """Conservative completeness contract (Section 20). ``pbp_complete`` is
    true only when the game is FINAL, pagination/normalization found no
    unresolved anomaly, at least one play was returned, play ids are
    unique, and -- when the game's own canonical score is known -- the
    terminal play's score state matches it exactly. A game whose PBP lacks
    any score-bearing terminal play leaves ``terminal_score_reconciled`` as
    ``None`` (unavailable), which does NOT by itself fail completeness --
    inventing a reconciliation this schema cannot support would be worse
    than leaving it explicitly unknown."""
    play_count = len(plays)
    nonempty = play_count > 0
    unique_ids = plays["provider_play_id"].is_unique if nonempty else False
    normalization_clean = normalization_report.clean

    terminal_reconciled: bool | None = None
    if nonempty and pd.notna(canonical_home_score) and pd.notna(canonical_away_score):
        ordered = plays.dropna(subset=["home_score", "away_score"])
        if len(ordered):
            terminal = ordered.iloc[-1]
            terminal_reconciled = (
                float(terminal["home_score"]) == float(canonical_home_score)
                and float(terminal["away_score"]) == float(canonical_away_score)
            )

    pbp_complete = bool(
        game_final
        and nonempty
        and unique_ids
        and normalization_clean
        and (terminal_reconciled is not False)
    )
    return PBPCompletenessReport(
        provider_game_id=normalization_report.provider_game_id,
        game_final=game_final,
        play_count=play_count,
        nonempty=nonempty,
        unique_play_ids=unique_ids,
        normalization_clean=normalization_clean,
        terminal_score_reconciled=terminal_reconciled,
        pbp_complete=pbp_complete,
    )
