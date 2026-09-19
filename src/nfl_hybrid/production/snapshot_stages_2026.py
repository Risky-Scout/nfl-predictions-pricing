"""OPEN / MID / CLOSE -- the 2026 pregame snapshot stage contract.

WHAT THIS IS. Three immutable pregame snapshots per game, defined purely as
OPERATIONAL market-observation instants:

    OPEN   the first valid sportsbook market observed for that game once its
           board becomes available. Observational: its instant is discovered,
           never computed, and once recorded it is never moved.
    MID    Friday 12:00 America/New_York for games that have not yet closed.
           For a Thursday game whose CLOSE falls before that Friday noon, the
           temporal midpoint between that game's OPEN and its CLOSE instead,
           so the stage still lands strictly pregame.
    CLOSE  exactly 60 minutes before that individual game's scheduled kickoff.
           This is the snapshot published publicly.

WHAT THIS IS NOT. A stage is not a model, a model family, a feature set, a
calibration state or a horizon in the certified TUE/FRI sense. Nothing here
predicts anything, fits anything, reads a market or names a bookmaker. This
module answers exactly one question -- "which instant is stage S for this
game?" -- so that the existing certified science can be asked for a forecast
AT that instant without any of it being duplicated or renamed.

THE TWO RULES THAT MADE THIS NON-OBVIOUS.

1. MID is anchored on the CARD, not on the game. An NFL week is one card
   spanning Thursday night through Monday night, and ``_card_monday_date``
   answers for whatever kickoff it is handed. Handed a Monday-night kickoff it
   returns THAT Monday -- the Monday *after* the card the game belongs to --
   which would put the game's Friday-noon anchor a week late and, for a
   Monday-nighter, after its own kickoff. So MID is derived from the card's
   earliest kickoff, exactly as ``horizon_elo`` derives the certified TUE/FRI
   cutoffs, rather than from the individual game.

2. Every stage must be STRICTLY pregame, and that is enforced rather than
   assumed. A post-kickoff observation is never OPEN, never MID and never
   CLOSE; a game whose board was first seen too late simply has no valid OPEN,
   which is recorded as missing rather than back-dated.

TIME MATH IS BORROWED, NOT RESTATED. ``NY_ZONE``, ``_card_monday_date`` and
``_card_noon_cutoff_utc`` come from :mod:`nfl_hybrid.features.horizon_elo`,
the one place the repository keeps DST-aware Eastern card arithmetic. There is
no second timezone implementation here and no hard-coded EDT/EST offset, so a
stage resolves correctly across the November transition for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nfl_hybrid.features import horizon_elo as he

STAGE_OPEN = "OPEN"
STAGE_MID = "MID"
STAGE_CLOSE = "CLOSE"

# The capture-manifest horizon label a point-in-time stage observation is
# archived under. Named here so the archive reader and the capture writer
# cannot drift; it is the provider-capture horizon, NOT one of the three
# operational stages above.
STAGE_CAPTURE_HORIZON = "STAGE"

# Publication order, and the order a season progresses through them.
SNAPSHOT_STAGES: tuple[str, ...] = (STAGE_OPEN, STAGE_MID, STAGE_CLOSE)

# CLOSE is per-GAME: 60 minutes before that game's own scheduled kickoff, not
# a card-wide instant. Two games on the same card close at different times.
CLOSE_LEAD = pd.Timedelta(minutes=60)

# MID's normal anchor: Friday 12:00 America/New_York of the game's own card.
# The weekday offset is expressed against the card Monday the same way
# horizon_elo expresses TUE (+1) and FRI (+4), so the two cannot drift.
MID_ANCHOR_DAYS_AFTER_CARD_MONDAY = 4

# Snapshot instants are floored to whole seconds. The MID midpoint is the only
# derived instant that can land on a fraction, and a stage instant becomes part
# of a forecast-of-record identity, so it must be exactly reproducible.
STAGE_INSTANT_RESOLUTION = "s"


class SnapshotStageError(RuntimeError):
    """Fail-closed stage error.

    Raised rather than returning a best guess whenever a stage cannot be
    established: an unusable kickoff, a missing OPEN a Thursday MID depends
    on, or a candidate instant that is not strictly pregame. A stage that
    cannot be proved is missing, and missing stays missing.
    """


def _as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise SnapshotStageError(f"instant is not a usable timestamp: {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _floor(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor(STAGE_INSTANT_RESOLUTION)


def validate_stage(stage: str) -> str:
    if stage not in SNAPSHOT_STAGES:
        raise SnapshotStageError(f"unknown snapshot stage {stage!r}; expected one of {SNAPSHOT_STAGES}")
    return stage


def is_strictly_pregame(instant, scheduled_kickoff_utc) -> bool:
    """Whether ``instant`` precedes kickoff. Kickoff itself is NOT pregame."""
    return _as_utc(instant) < _as_utc(scheduled_kickoff_utc)


def require_strictly_pregame(instant, scheduled_kickoff_utc, *, what: str) -> pd.Timestamp:
    instant_utc = _as_utc(instant)
    kickoff = _as_utc(scheduled_kickoff_utc)
    if instant_utc >= kickoff:
        raise SnapshotStageError(
            f"{what} at {instant_utc.isoformat()} is not strictly pregame for a kickoff at "
            f"{kickoff.isoformat()} -- a post-kickoff observation is never a pregame snapshot"
        )
    return instant_utc


# ---------------------------------------------------------------------------
# CLOSE -- per game, purely a function of that game's kickoff.
# ---------------------------------------------------------------------------
def close_cutoff_utc(scheduled_kickoff_utc) -> pd.Timestamp:
    """Exactly 60 minutes before this game's scheduled kickoff."""
    kickoff = _as_utc(scheduled_kickoff_utc)
    return require_strictly_pregame(_floor(kickoff - CLOSE_LEAD), kickoff, what="CLOSE")


# ---------------------------------------------------------------------------
# MID -- Friday noon Eastern of the game's CARD, or the OPEN..CLOSE midpoint
# for a game that has already closed by then.
# ---------------------------------------------------------------------------
def card_monday_date(card_earliest_kickoff_utc):
    """The card's Monday, via horizon_elo's one implementation.

    Must be handed the CARD's earliest kickoff, not an individual game's --
    see rule 1 in the module docstring.
    """
    return he._card_monday_date(_as_utc(card_earliest_kickoff_utc))


def mid_anchor_utc(card_earliest_kickoff_utc) -> pd.Timestamp:
    """Friday 12:00 America/New_York for this card, DST-aware."""
    monday = card_monday_date(card_earliest_kickoff_utc)
    return _floor(he._card_noon_cutoff_utc(monday, MID_ANCHOR_DAYS_AFTER_CARD_MONDAY))


def mid_uses_midpoint(scheduled_kickoff_utc, card_earliest_kickoff_utc) -> bool:
    """Whether this game closes at or before its card's Friday noon anchor.

    True for the Thursday game of a card: Friday noon is no longer pregame for
    it, so the anchor cannot be its MID.
    """
    return mid_anchor_utc(card_earliest_kickoff_utc) >= close_cutoff_utc(scheduled_kickoff_utc)


def mid_cutoff_utc(
    *, scheduled_kickoff_utc, card_earliest_kickoff_utc, open_observed_at_utc=None
) -> pd.Timestamp:
    """This game's MID instant.

    Friday noon Eastern when that is still strictly before the game's CLOSE.
    Otherwise -- the Thursday case -- the temporal midpoint between the game's
    OPEN and its CLOSE, which requires OPEN to have actually been observed.
    A Thursday game with no recorded OPEN has no MID and says so, rather than
    inventing an anchor.
    """
    kickoff = _as_utc(scheduled_kickoff_utc)
    close = close_cutoff_utc(kickoff)
    anchor = mid_anchor_utc(card_earliest_kickoff_utc)

    if anchor < close:
        return require_strictly_pregame(anchor, kickoff, what="MID")

    if open_observed_at_utc is None:
        raise SnapshotStageError(
            f"this game closes at {close.isoformat()}, at or before its card's Friday noon anchor "
            f"{anchor.isoformat()}, so MID is the OPEN..CLOSE midpoint -- but no OPEN has been "
            "observed yet, and a midpoint is never invented from an assumed opening instant"
        )

    open_at = require_strictly_pregame(open_observed_at_utc, kickoff, what="OPEN")
    if open_at >= close:
        raise SnapshotStageError(
            f"OPEN at {open_at.isoformat()} is not before CLOSE at {close.isoformat()} -- "
            "there is no interval for a midpoint MID to fall inside"
        )

    midpoint = _floor(open_at + (close - open_at) / 2)
    if not (open_at < midpoint < close):
        raise SnapshotStageError(
            f"the OPEN..CLOSE interval {open_at.isoformat()}..{close.isoformat()} is too short to "
            "contain a distinct midpoint MID"
        )
    return require_strictly_pregame(midpoint, kickoff, what="MID")


# ---------------------------------------------------------------------------
# OPEN -- observational. Discovered, validated, then frozen.
# ---------------------------------------------------------------------------
def validate_open_observation(observed_at_utc, *, scheduled_kickoff_utc) -> pd.Timestamp:
    """The first valid market observation, checked for being a legitimate OPEN.

    A market first seen at or after kickoff is not an opening line for this
    game: the board it belongs to is no longer pregame.
    """
    return _floor(
        require_strictly_pregame(observed_at_utc, scheduled_kickoff_utc, what="OPEN observation")
    )


def first_open_observation(observed_instants, *, scheduled_kickoff_utc) -> pd.Timestamp:
    """The EARLIEST strictly-pregame observation among candidates.

    "First valid market observed" is the earliest one, so a later observation
    can never be relabelled as the opening line. Every candidate must itself
    be pregame; a post-kickoff candidate fails the whole resolution closed
    rather than being quietly dropped, because its presence means the caller
    handed in observations it had not filtered.
    """
    candidates = [
        validate_open_observation(instant, scheduled_kickoff_utc=scheduled_kickoff_utc)
        for instant in observed_instants
    ]
    if not candidates:
        raise SnapshotStageError(
            "no market observation supplied -- a game whose board has not been seen has no OPEN, "
            "and OPEN is never back-dated to a board that was never observed"
        )
    return min(candidates)


# ---------------------------------------------------------------------------
# All three stages for one game.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameSnapshotSchedule:
    """The three pregame instants for one game, with how MID was derived."""

    game_id: str
    scheduled_kickoff_utc: pd.Timestamp
    open_observed_at_utc: pd.Timestamp | None
    mid_cutoff_utc: pd.Timestamp | None
    close_cutoff_utc: pd.Timestamp
    mid_basis: str

    def cutoff_for(self, stage: str) -> pd.Timestamp | None:
        return {
            STAGE_OPEN: self.open_observed_at_utc,
            STAGE_MID: self.mid_cutoff_utc,
            STAGE_CLOSE: self.close_cutoff_utc,
        }[validate_stage(stage)]

    def resolved_stages(self) -> tuple[str, ...]:
        return tuple(s for s in SNAPSHOT_STAGES if self.cutoff_for(s) is not None)


MID_BASIS_FRIDAY_ANCHOR = "CARD_FRIDAY_NOON_ET"
MID_BASIS_OPEN_CLOSE_MIDPOINT = "OPEN_CLOSE_MIDPOINT"
MID_BASIS_UNRESOLVED = "UNRESOLVED_NO_OPEN"


def resolve_game_snapshots(
    *, game_id: str, scheduled_kickoff_utc, card_earliest_kickoff_utc, open_observed_at_utc=None
) -> GameSnapshotSchedule:
    """Every stage instant for one game.

    CLOSE always resolves from the kickoff alone. OPEN is whatever was
    observed, validated as strictly pregame, or ``None`` when the board has
    not been seen. MID resolves from the card's Friday anchor, or from the
    OPEN..CLOSE midpoint when the game closes first -- and is left ``None``,
    explicitly unresolved, when that midpoint depends on an OPEN that does not
    exist yet.
    """
    kickoff = _as_utc(scheduled_kickoff_utc)
    close = close_cutoff_utc(kickoff)
    open_at = (
        None
        if open_observed_at_utc is None
        else validate_open_observation(open_observed_at_utc, scheduled_kickoff_utc=kickoff)
    )

    midpoint_based = mid_uses_midpoint(kickoff, card_earliest_kickoff_utc)
    if midpoint_based and open_at is None:
        mid, basis = None, MID_BASIS_UNRESOLVED
    else:
        mid = mid_cutoff_utc(
            scheduled_kickoff_utc=kickoff,
            card_earliest_kickoff_utc=card_earliest_kickoff_utc,
            open_observed_at_utc=open_at,
        )
        basis = MID_BASIS_OPEN_CLOSE_MIDPOINT if midpoint_based else MID_BASIS_FRIDAY_ANCHOR

    return GameSnapshotSchedule(
        game_id=game_id,
        scheduled_kickoff_utc=kickoff,
        open_observed_at_utc=open_at,
        mid_cutoff_utc=mid,
        close_cutoff_utc=close,
        mid_basis=basis,
    )


def card_earliest_kickoff_utc(scheduled_kickoffs) -> pd.Timestamp:
    """The card's earliest kickoff -- the anchor every MID on it is derived
    from. Fails closed on an empty card rather than returning a sentinel."""
    instants = [_as_utc(value) for value in scheduled_kickoffs]
    if not instants:
        raise SnapshotStageError("a card with no kickoffs has no earliest kickoff to anchor MID on")
    return min(instants)
