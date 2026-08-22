"""BDL team id/abbreviation -> canonical NFL team crosswalk.

The table below was captured from a live, authenticated ``GET /teams`` call
against the current production BDL NFL API on 2026-08-21 (all 32 current
franchises, ids 1-33 with id 2 unused/retired by the provider). It is a
frozen, hand-verified registry -- not derived at import time from a network
call -- so the crosswalk is available offline and its content is auditable
in a diff.

Every ``canonical_team`` value is resolved through
:func:`nfl_hybrid.data.team_ids.canonical_team_id`, the SAME crosswalk Fix
2/Fix 3 already use for the historical estate, so a BDL-sourced team id
always lands on the identical canonical code a historical nflverse row
would use for the same franchise. The one alias BDL needs that the
historical estate's own providers didn't already require is
``WSH -> WAS`` (BDL abbreviates Washington as ``WSH``); that alias already
existed in the shared registry before Fix 5.
"""
from __future__ import annotations

from dataclasses import dataclass

from nfl_hybrid.data.team_ids import canonical_team_id

# (bdl_team_id, bdl_abbreviation, full_name) -- verified live, 2026-08-21.
_BDL_TEAMS: tuple[tuple[int, str, str], ...] = (
    (1, "NE", "New England Patriots"),
    (3, "BUF", "Buffalo Bills"),
    (4, "NYJ", "New York Jets"),
    (5, "MIA", "Miami Dolphins"),
    (6, "BAL", "Baltimore Ravens"),
    (7, "PIT", "Pittsburgh Steelers"),
    (8, "CLE", "Cleveland Browns"),
    (9, "CIN", "Cincinnati Bengals"),
    (10, "HOU", "Houston Texans"),
    (11, "TEN", "Tennessee Titans"),
    (12, "IND", "Indianapolis Colts"),
    (13, "JAX", "Jacksonville Jaguars"),
    (14, "KC", "Kansas City Chiefs"),
    (15, "DEN", "Denver Broncos"),
    (16, "LV", "Las Vegas Raiders"),
    (17, "LAC", "Los Angeles Chargers"),
    (18, "PHI", "Philadelphia Eagles"),
    (19, "DAL", "Dallas Cowboys"),
    (20, "NYG", "New York Giants"),
    (21, "WSH", "Washington Commanders"),
    (22, "GB", "Green Bay Packers"),
    (23, "MIN", "Minnesota Vikings"),
    (24, "CHI", "Chicago Bears"),
    (25, "DET", "Detroit Lions"),
    (26, "NO", "New Orleans Saints"),
    (27, "ATL", "Atlanta Falcons"),
    (28, "TB", "Tampa Bay Buccaneers"),
    (29, "CAR", "Carolina Panthers"),
    (30, "SF", "San Francisco 49ers"),
    (31, "SEA", "Seattle Seahawks"),
    (32, "LAR", "Los Angeles Rams"),
    (33, "ARI", "Arizona Cardinals"),
)

EXPECTED_FRANCHISE_COUNT = 32


class TeamCrosswalkError(ValueError):
    """Fail-closed: an unknown/duplicate/ambiguous team identity."""


@dataclass(frozen=True)
class CrosswalkEntry:
    bdl_team_id: int
    bdl_abbreviation: str
    canonical_team: str
    full_name: str


def _build_registry() -> dict[int, CrosswalkEntry]:
    registry: dict[int, CrosswalkEntry] = {}
    seen_abbrev: dict[str, int] = {}
    seen_canonical: dict[str, int] = {}
    for bdl_id, abbrev, full_name in _BDL_TEAMS:
        if bdl_id in registry:
            raise TeamCrosswalkError(f"Duplicate BDL team id in crosswalk table: {bdl_id}")
        canonical = canonical_team_id(abbrev)
        if canonical is None:
            raise TeamCrosswalkError(f"Unknown NFL team identifier from BDL: {abbrev!r}")
        if abbrev in seen_abbrev:
            raise TeamCrosswalkError(f"Duplicate BDL abbreviation in crosswalk table: {abbrev!r}")
        if canonical in seen_canonical:
            raise TeamCrosswalkError(
                f"Ambiguous crosswalk: BDL ids {seen_canonical[canonical]} and {bdl_id} both "
                f"map to canonical team {canonical!r}."
            )
        seen_abbrev[abbrev] = bdl_id
        seen_canonical[canonical] = bdl_id
        registry[bdl_id] = CrosswalkEntry(
            bdl_team_id=bdl_id,
            bdl_abbreviation=abbrev,
            canonical_team=canonical,
            full_name=full_name,
        )
    if len(registry) != EXPECTED_FRANCHISE_COUNT:
        raise TeamCrosswalkError(
            f"BDL team crosswalk has {len(registry)} entries, expected {EXPECTED_FRANCHISE_COUNT}."
        )
    return registry


_REGISTRY: dict[int, CrosswalkEntry] = _build_registry()


def registry() -> dict[int, CrosswalkEntry]:
    return dict(_REGISTRY)


def canonical_team_for_bdl_id(bdl_team_id: int | None) -> str:
    """Fail closed: raises on an unknown/None BDL team id rather than
    guessing or returning a null/placeholder team."""
    if bdl_team_id is None:
        raise TeamCrosswalkError("BDL team id is missing (None) -- cannot resolve a canonical team.")
    entry = _REGISTRY.get(int(bdl_team_id))
    if entry is None:
        raise TeamCrosswalkError(f"Unknown BDL team id: {bdl_team_id!r} (not in the {EXPECTED_FRANCHISE_COUNT}-franchise crosswalk).")
    return entry.canonical_team


def canonical_team_from_team_object(team: dict | None) -> str:
    """Resolve a canonical team from a BDL nested ``team`` object (present on
    games/plays/stats/injuries/roster payloads as ``{"id": ..., "abbreviation": ...}``).

    Prefers the numeric id (stable across a franchise's own abbreviation
    changes); falls back to abbreviation only if id is absent, and still
    fails closed if neither resolves.
    """
    if not isinstance(team, dict):
        raise TeamCrosswalkError(f"Expected a BDL team object, got {type(team).__name__}")
    team_id = team.get("id")
    if team_id is not None:
        return canonical_team_for_bdl_id(team_id)
    abbreviation = team.get("abbreviation")
    canonical = canonical_team_id(abbreviation) if abbreviation else None
    if canonical is None:
        raise TeamCrosswalkError(f"BDL team object has neither a resolvable id nor abbreviation: {team!r}")
    return canonical


def completeness_report() -> dict[str, object]:
    """All 32 current NFL franchises present, one canonical team per BDL id,
    no duplicate/ambiguous mapping. Non-raising; for reporting."""
    canonical_teams = {entry.canonical_team for entry in _REGISTRY.values()}
    return {
        "bdl_franchise_count": len(_REGISTRY),
        "expected_franchise_count": EXPECTED_FRANCHISE_COUNT,
        "distinct_canonical_teams": len(canonical_teams),
        "complete": len(_REGISTRY) == EXPECTED_FRANCHISE_COUNT == len(canonical_teams),
    }
