"""Acceptance proofs for the OPEN / MID / CLOSE pregame snapshot contract.

Everything here is pure calendar arithmetic over synthetic kickoffs. No
market, no model, no provider, no artifact root, no network.

The 2026 weeks used below are real NFL shapes, expressed in UTC as the
canonical population stores them:

    Thursday night  8:15pm ET  ->  next day 00:15Z
    Sunday early    1:00pm ET  ->  same day 17:00Z
    Sunday night    8:20pm ET  ->  next day 00:20Z
    Monday night    8:15pm ET  ->  next day 00:15Z

so a "Thursday" game's UTC date is Friday and a "Monday" game's is Tuesday --
which is exactly why MID is anchored on the card rather than on the game.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.production import snapshot_stages_2026 as st

# One card: 2026 week 2. Thu Sep 17 ET, Sun Sep 20 ET, Mon Sep 21 ET.
THU_KICKOFF = pd.Timestamp("2026-09-18T00:15:00Z")  # Thu 8:15pm ET Sep 17
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00Z")  # Sun 1:00pm ET
SUN_NIGHT_KICKOFF = pd.Timestamp("2026-09-21T00:20:00Z")  # Sun 8:20pm ET
MON_KICKOFF = pd.Timestamp("2026-09-22T00:15:00Z")  # Mon 8:15pm ET Sep 21
CARD = (THU_KICKOFF, SUN_EARLY_KICKOFF, SUN_NIGHT_KICKOFF, MON_KICKOFF)
CARD_EARLIEST = THU_KICKOFF

# Friday 12:00 ET of that card's week, during EDT (UTC-4).
CARD_FRIDAY_NOON = pd.Timestamp("2026-09-18T16:00:00Z")


# ===========================================================================
# CLOSE -- kickoff minus exactly 60 minutes, per game.
# ===========================================================================
@pytest.mark.parametrize("kickoff", CARD)
def test_close_resolves_to_kickoff_minus_exactly_sixty_minutes(kickoff):
    close = st.close_cutoff_utc(kickoff)
    assert kickoff - close == pd.Timedelta(minutes=60)
    assert close == kickoff - pd.Timedelta(minutes=60)


def test_close_is_per_game_not_card_wide():
    """Two games on one card close at different instants."""
    closes = {st.close_cutoff_utc(k) for k in CARD}
    assert len(closes) == len(CARD)


def test_close_is_strictly_pregame():
    for kickoff in CARD:
        assert st.is_strictly_pregame(st.close_cutoff_utc(kickoff), kickoff)


def test_close_survives_the_dst_transition():
    """A November game (EST) and a September game (EDT) both close exactly 60
    minutes out -- the rule is wall-clock independent."""
    november = pd.Timestamp("2026-11-09T01:15:00Z")  # Sun 8:15pm EST Nov 8
    assert november - st.close_cutoff_utc(november) == pd.Timedelta(minutes=60)


# ===========================================================================
# MID -- Friday noon ET normally; OPEN..CLOSE midpoint for a Thursday game.
# ===========================================================================
def test_mid_anchor_is_friday_noon_eastern_of_the_card():
    assert st.mid_anchor_utc(CARD_EARLIEST) == CARD_FRIDAY_NOON
    assert st.mid_anchor_utc(CARD_EARLIEST).tz_convert(he.NY_ZONE).hour == 12
    assert st.mid_anchor_utc(CARD_EARLIEST).tz_convert(he.NY_ZONE).weekday() == 4  # Friday


@pytest.mark.parametrize("kickoff", [SUN_EARLY_KICKOFF, SUN_NIGHT_KICKOFF, MON_KICKOFF])
def test_normal_sunday_and_monday_games_use_the_friday_anchor(kickoff):
    schedule = st.resolve_game_snapshots(
        game_id="G", scheduled_kickoff_utc=kickoff, card_earliest_kickoff_utc=CARD_EARLIEST
    )
    assert schedule.mid_cutoff_utc == CARD_FRIDAY_NOON
    assert schedule.mid_basis == st.MID_BASIS_FRIDAY_ANCHOR
    assert st.is_strictly_pregame(schedule.mid_cutoff_utc, kickoff)


def test_a_monday_night_game_is_anchored_on_its_own_card_not_the_next_one():
    """The bug this guards against: a Monday-night kickoff's OWN card Monday is
    the Monday AFTER its card, which would place its Friday anchor a week late
    and after its own kickoff."""
    assert he._card_monday_date(MON_KICKOFF) != he._card_monday_date(CARD_EARLIEST)

    naive_anchor = st.mid_anchor_utc(MON_KICKOFF)  # what per-game derivation gives
    assert naive_anchor > MON_KICKOFF  # ... and it is post-kickoff, i.e. unusable

    schedule = st.resolve_game_snapshots(
        game_id="MNF", scheduled_kickoff_utc=MON_KICKOFF, card_earliest_kickoff_utc=CARD_EARLIEST
    )
    assert schedule.mid_cutoff_utc == CARD_FRIDAY_NOON
    assert st.is_strictly_pregame(schedule.mid_cutoff_utc, MON_KICKOFF)


def test_the_thursday_game_closes_before_the_friday_anchor():
    assert st.close_cutoff_utc(THU_KICKOFF) < CARD_FRIDAY_NOON
    assert st.mid_uses_midpoint(THU_KICKOFF, CARD_EARLIEST) is True


@pytest.mark.parametrize("kickoff", [SUN_EARLY_KICKOFF, MON_KICKOFF])
def test_non_thursday_games_do_not_use_the_midpoint(kickoff):
    assert st.mid_uses_midpoint(kickoff, CARD_EARLIEST) is False


def test_a_thursday_game_uses_the_open_to_close_midpoint():
    open_at = pd.Timestamp("2026-09-14T14:00:00Z")
    close = st.close_cutoff_utc(THU_KICKOFF)  # 2026-09-17T23:15:00Z

    schedule = st.resolve_game_snapshots(
        game_id="TNF",
        scheduled_kickoff_utc=THU_KICKOFF,
        card_earliest_kickoff_utc=CARD_EARLIEST,
        open_observed_at_utc=open_at,
    )

    assert schedule.mid_basis == st.MID_BASIS_OPEN_CLOSE_MIDPOINT
    assert schedule.mid_cutoff_utc == open_at + (close - open_at) / 2
    assert open_at < schedule.mid_cutoff_utc < close


def test_the_thursday_midpoint_is_still_strictly_pregame():
    schedule = st.resolve_game_snapshots(
        game_id="TNF",
        scheduled_kickoff_utc=THU_KICKOFF,
        card_earliest_kickoff_utc=CARD_EARLIEST,
        open_observed_at_utc=pd.Timestamp("2026-09-14T14:00:00Z"),
    )
    assert st.is_strictly_pregame(schedule.mid_cutoff_utc, THU_KICKOFF)
    assert schedule.mid_cutoff_utc < schedule.close_cutoff_utc


def test_a_thursday_mid_is_explicitly_unresolved_without_an_open():
    """Missing evidence stays missing: no assumed opening instant, no
    back-dated midpoint."""
    schedule = st.resolve_game_snapshots(
        game_id="TNF", scheduled_kickoff_utc=THU_KICKOFF, card_earliest_kickoff_utc=CARD_EARLIEST
    )
    assert schedule.mid_cutoff_utc is None
    assert schedule.mid_basis == st.MID_BASIS_UNRESOLVED
    assert schedule.resolved_stages() == (st.STAGE_CLOSE,)

    with pytest.raises(st.SnapshotStageError, match="no OPEN has been observed"):
        st.mid_cutoff_utc(
            scheduled_kickoff_utc=THU_KICKOFF, card_earliest_kickoff_utc=CARD_EARLIEST
        )


def test_an_open_at_or_after_close_leaves_no_room_for_a_midpoint():
    close = st.close_cutoff_utc(THU_KICKOFF)
    with pytest.raises(st.SnapshotStageError, match="no interval"):
        st.mid_cutoff_utc(
            scheduled_kickoff_utc=THU_KICKOFF,
            card_earliest_kickoff_utc=CARD_EARLIEST,
            open_observed_at_utc=close,
        )


def test_mid_survives_the_dst_transition():
    """A November card's Friday anchor is noon EST (17:00Z), not noon EDT
    (16:00Z) -- proving the offset is never hard-coded."""
    november_card_earliest = pd.Timestamp("2026-11-06T01:15:00Z")  # Thu 8:15pm EST Nov 5
    anchor = st.mid_anchor_utc(november_card_earliest)
    assert anchor.tz_convert(he.NY_ZONE).hour == 12
    assert anchor == pd.Timestamp("2026-11-06T17:00:00Z")
    assert anchor != pd.Timestamp("2026-11-06T16:00:00Z")


# ===========================================================================
# OPEN -- first valid observed market, never a later one.
# ===========================================================================
def test_open_is_the_first_valid_observed_market():
    observations = [
        pd.Timestamp("2026-09-16T12:00:00Z"),
        pd.Timestamp("2026-09-14T14:00:00Z"),  # earliest
        pd.Timestamp("2026-09-15T09:30:00Z"),
    ]
    assert st.first_open_observation(
        observations, scheduled_kickoff_utc=SUN_EARLY_KICKOFF
    ) == pd.Timestamp("2026-09-14T14:00:00Z")


def test_a_later_observation_is_never_relabelled_as_the_opening_line():
    early = pd.Timestamp("2026-09-14T14:00:00Z")
    late = pd.Timestamp("2026-09-19T14:00:00Z")
    assert st.first_open_observation([early, late], scheduled_kickoff_utc=SUN_EARLY_KICKOFF) == early
    assert st.first_open_observation([late, early], scheduled_kickoff_utc=SUN_EARLY_KICKOFF) == early


def test_open_cannot_be_established_from_no_observation():
    with pytest.raises(st.SnapshotStageError, match="no market observation"):
        st.first_open_observation([], scheduled_kickoff_utc=SUN_EARLY_KICKOFF)


def test_an_unobserved_open_is_recorded_as_missing_not_invented():
    schedule = st.resolve_game_snapshots(
        game_id="G", scheduled_kickoff_utc=SUN_EARLY_KICKOFF, card_earliest_kickoff_utc=CARD_EARLIEST
    )
    assert schedule.open_observed_at_utc is None
    assert st.STAGE_OPEN not in schedule.resolved_stages()


# ===========================================================================
# No post-kickoff observation is ever a pregame stage.
# ===========================================================================
@pytest.mark.parametrize(
    "offset", [pd.Timedelta(0), pd.Timedelta(seconds=1), pd.Timedelta(hours=3)]
)
def test_no_post_kickoff_observation_can_become_open(offset):
    with pytest.raises(st.SnapshotStageError, match="not strictly pregame"):
        st.validate_open_observation(
            SUN_EARLY_KICKOFF + offset, scheduled_kickoff_utc=SUN_EARLY_KICKOFF
        )


def test_kickoff_itself_is_not_pregame():
    assert st.is_strictly_pregame(SUN_EARLY_KICKOFF, SUN_EARLY_KICKOFF) is False
    assert st.is_strictly_pregame(SUN_EARLY_KICKOFF - pd.Timedelta(seconds=1), SUN_EARLY_KICKOFF)


def test_a_post_kickoff_candidate_fails_open_resolution_closed():
    """Not silently dropped: its presence means the caller handed in
    observations it had not filtered."""
    with pytest.raises(st.SnapshotStageError, match="not strictly pregame"):
        st.first_open_observation(
            [pd.Timestamp("2026-09-14T14:00:00Z"), SUN_EARLY_KICKOFF + pd.Timedelta(hours=1)],
            scheduled_kickoff_utc=SUN_EARLY_KICKOFF,
        )


@pytest.mark.parametrize("stage", st.SNAPSHOT_STAGES)
def test_every_resolved_stage_of_every_card_game_is_strictly_pregame(stage):
    for kickoff in CARD:
        schedule = st.resolve_game_snapshots(
            game_id="G",
            scheduled_kickoff_utc=kickoff,
            card_earliest_kickoff_utc=CARD_EARLIEST,
            open_observed_at_utc=pd.Timestamp("2026-09-14T14:00:00Z"),
        )
        cutoff = schedule.cutoff_for(stage)
        assert cutoff is not None
        assert st.is_strictly_pregame(cutoff, kickoff)


def test_the_three_stages_are_chronologically_ordered():
    for kickoff in CARD:
        schedule = st.resolve_game_snapshots(
            game_id="G",
            scheduled_kickoff_utc=kickoff,
            card_earliest_kickoff_utc=CARD_EARLIEST,
            open_observed_at_utc=pd.Timestamp("2026-09-14T14:00:00Z"),
        )
        assert (
            schedule.open_observed_at_utc
            < schedule.mid_cutoff_utc
            < schedule.close_cutoff_utc
            < kickoff
        )


# ===========================================================================
# Stage naming and card anchoring hygiene.
# ===========================================================================
def test_the_stage_vocabulary_is_exactly_three_operational_stages():
    assert st.SNAPSHOT_STAGES == ("OPEN", "MID", "CLOSE")
    with pytest.raises(st.SnapshotStageError, match="unknown snapshot stage"):
        st.validate_stage("TUE")
    with pytest.raises(st.SnapshotStageError, match="unknown snapshot stage"):
        st.validate_stage("open")


def test_the_card_anchor_is_the_earliest_kickoff():
    assert st.card_earliest_kickoff_utc(CARD) == THU_KICKOFF
    assert st.card_earliest_kickoff_utc(reversed(CARD)) == THU_KICKOFF
    with pytest.raises(st.SnapshotStageError, match="no kickoffs"):
        st.card_earliest_kickoff_utc([])


@pytest.mark.parametrize("bad", [None, "", "not-a-timestamp", float("nan")])
def test_an_unusable_kickoff_fails_closed(bad):
    with pytest.raises(Exception):
        st.close_cutoff_utc(bad)


def test_the_stage_contract_borrows_the_one_eastern_implementation():
    """No second timezone implementation and no hard-coded seasonal offset."""
    import re
    from pathlib import Path

    code = Path(st.__file__).read_text(encoding="utf-8").split('"""', 2)[-1]
    for forbidden in (
        r"ZoneInfo\(",
        r"US/Eastern",
        r"\bEDT\b",
        r"\bEST\b",
        r"Timedelta\(hours=4\)",
        r"Timedelta\(hours=5\)",
    ):
        assert not re.search(forbidden, code), forbidden
    assert "he._card_monday_date" in code
    assert "he._card_noon_cutoff_utc" in code


def test_stage_instants_are_whole_seconds_and_reproducible():
    """A stage instant becomes part of a forecast identity, so a fractional
    midpoint would make the identity unreproducible."""
    schedule = st.resolve_game_snapshots(
        game_id="TNF",
        scheduled_kickoff_utc=THU_KICKOFF,
        card_earliest_kickoff_utc=CARD_EARLIEST,
        # An odd-second OPEN forces a half-second raw midpoint.
        open_observed_at_utc=pd.Timestamp("2026-09-14T14:00:01Z"),
    )
    assert schedule.mid_cutoff_utc.microsecond == 0
    assert schedule.mid_cutoff_utc.nanosecond == 0
    again = st.resolve_game_snapshots(
        game_id="TNF",
        scheduled_kickoff_utc=THU_KICKOFF,
        card_earliest_kickoff_utc=CARD_EARLIEST,
        open_observed_at_utc=pd.Timestamp("2026-09-14T14:00:01Z"),
    )
    assert again.mid_cutoff_utc == schedule.mid_cutoff_utc
