"""Finality rule, completeness gates, result_available_at_utc, and the
Tuesday/Friday availability scenarios (Sections 8-10, 23-24).
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.providers.balldontlie.finality import (
    RESULT_AVAILABILITY_BASIS,
    CanonicalGameFinality,
    ELIGIBLE_AS_OF,
    ELIGIBLE_BOX_COMPLETE,
    ELIGIBLE_FINAL_SCORE,
    ELIGIBLE_PBP_COMPLETE,
    ELIGIBLE_PLAYER_STATS_COMPLETE,
    INELIGIBLE_MISSING_PLAYER_STATS,
    INELIGIBLE_MISSING_SCORE,
    INELIGIBLE_MISSING_TEAM_STATS,
    INELIGIBLE_NEVER_OBSERVED_AVAILABLE,
    INELIGIBLE_NOT_FINAL,
    INELIGIBLE_NOT_YET_AVAILABLE_AS_OF,
    INELIGIBLE_PBP_INCOMPLETE,
    INELIGIBLE_UNKNOWN_TEAM,
    can_update_box_state,
    can_update_pbp_state,
    can_update_player_stats_state,
    can_update_score_state,
    compute_family_available_at_utc,
    compute_result_available_at_utc,
    family_eligible_as_of,
    is_final,
)


def _game(**overrides) -> CanonicalGameFinality:
    defaults = dict(
        game_id="2026_05_CLE_BAL",
        status_state="final",
        home_team="BAL",
        away_team="CLE",
        home_score=30.0,
        away_score=10.0,
    )
    defaults.update(overrides)
    return CanonicalGameFinality(**defaults)


# -- finality rule (Section 8) --------------------------------------------------

@pytest.mark.parametrize(
    "status_state",
    ["scheduled", "in_progress", "postponed", "canceled", "delayed", "suspended", "abandoned", "unknown", None, "garbage"],
)
def test_non_final_states_are_not_final(status_state):
    assert is_final(status_state) is False


def test_final_is_final():
    assert is_final("final") is True


def test_in_progress_with_realistic_score_still_not_final():
    """An in-progress game with realistic scores/stats/PBP must still be
    excluded -- finality never inferred from a non-null score."""
    game = _game(status_state="in_progress", home_score=17.0, away_score=14.0)
    result = can_update_score_state(game)
    assert result.eligible is False
    assert result.reason_code == INELIGIBLE_NOT_FINAL


# -- gates + reason codes (Section 23) -------------------------------------------

def test_can_update_score_state_eligible():
    result = can_update_score_state(_game())
    assert result.eligible is True
    assert result.reason_code == ELIGIBLE_FINAL_SCORE


def test_can_update_score_state_missing_score():
    result = can_update_score_state(_game(home_score=None))
    assert result.eligible is False
    assert result.reason_code == INELIGIBLE_MISSING_SCORE


def test_can_update_score_state_unknown_team():
    result = can_update_score_state(_game(home_team=None))
    assert result.eligible is False
    assert result.reason_code == INELIGIBLE_UNKNOWN_TEAM


def test_can_update_box_state_eligible_and_ineligible():
    ok = can_update_box_state(_game(), box_complete=True)
    assert ok.eligible is True
    assert ok.reason_code == ELIGIBLE_BOX_COMPLETE

    missing = can_update_box_state(_game(), box_complete=False)
    assert missing.eligible is False
    assert missing.reason_code == INELIGIBLE_MISSING_TEAM_STATS

    not_final = can_update_box_state(_game(status_state="scheduled"), box_complete=True)
    assert not_final.reason_code == INELIGIBLE_NOT_FINAL


def test_can_update_pbp_state_eligible_and_ineligible():
    ok = can_update_pbp_state(_game(), pbp_complete=True)
    assert ok.eligible is True
    assert ok.reason_code == ELIGIBLE_PBP_COMPLETE

    incomplete = can_update_pbp_state(_game(), pbp_complete=False)
    assert incomplete.eligible is False
    assert incomplete.reason_code == INELIGIBLE_PBP_INCOMPLETE


# -- result_available_at_utc (Section 9) -----------------------------------------

def test_first_final_observation_wins_never_overwritten_by_later_poll():
    observations = pd.DataFrame(
        [
            {"game_id": "G1", "status_state": "in_progress", "provider_status": "Q3", "observed_at_utc": "2026-09-11T00:00:00Z"},
            {"game_id": "G1", "status_state": "final", "provider_status": "Final", "observed_at_utc": "2026-09-11T03:30:00Z"},
            {"game_id": "G1", "status_state": "final", "provider_status": "Final", "observed_at_utc": "2026-09-12T09:00:00Z"},
        ]
    )
    ledger = compute_result_available_at_utc(observations)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["result_available_at_utc"] == pd.Timestamp("2026-09-11T03:30:00Z")
    assert row["result_availability_basis"] == RESULT_AVAILABILITY_BASIS


def test_idempotent_recomputation_over_growing_log():
    base = [
        {"game_id": "G1", "status_state": "in_progress", "provider_status": "Q3", "observed_at_utc": "2026-09-11T00:00:00Z"},
        {"game_id": "G1", "status_state": "final", "provider_status": "Final", "observed_at_utc": "2026-09-11T03:30:00Z"},
    ]
    first = compute_result_available_at_utc(pd.DataFrame(base))
    grown = base + [
        {"game_id": "G1", "status_state": "final", "provider_status": "Final", "observed_at_utc": "2026-09-13T00:00:00Z"}
    ]
    second = compute_result_available_at_utc(pd.DataFrame(grown))
    assert first.iloc[0]["result_available_at_utc"] == second.iloc[0]["result_available_at_utc"]


def test_no_final_observation_yields_empty_not_fabricated():
    observations = pd.DataFrame(
        [{"game_id": "G2", "status_state": "scheduled", "provider_status": "Sched", "observed_at_utc": "2026-09-11T00:00:00Z"}]
    )
    ledger = compute_result_available_at_utc(observations)
    assert ledger.empty


# -- Tuesday/Friday availability scenarios (Section 24) ---------------------------

def test_scenario_a_thursday_game_in_progress_at_friday_cutoff_not_eligible():
    game = _game(status_state="in_progress", home_score=17.0, away_score=14.0)
    result = can_update_score_state(game)
    assert result.eligible is False
    assert result.reason_code == INELIGIBLE_NOT_FINAL


def test_scenario_b_final_before_friday_and_score_complete_is_eligible():
    game = _game(status_state="final")
    result = can_update_score_state(game)
    assert result.eligible is True
    assert result.reason_code == ELIGIBLE_FINAL_SCORE


def test_scenario_c_final_but_pbp_incomplete_score_ok_pbp_blocked():
    game = _game(status_state="final")
    score_result = can_update_score_state(game)
    pbp_result = can_update_pbp_state(game, pbp_complete=False)
    assert score_result.eligible is True
    assert pbp_result.eligible is False
    assert pbp_result.reason_code == INELIGIBLE_PBP_INCOMPLETE


def test_can_update_player_stats_state_eligible_and_ineligible():
    ok = can_update_player_stats_state(_game(), player_stats_complete=True)
    assert ok.eligible is True
    assert ok.reason_code == ELIGIBLE_PLAYER_STATS_COMPLETE

    missing = can_update_player_stats_state(_game(), player_stats_complete=False)
    assert missing.eligible is False
    assert missing.reason_code == INELIGIBLE_MISSING_PLAYER_STATS

    not_final = can_update_player_stats_state(_game(status_state="in_progress"), player_stats_complete=True)
    assert not_final.reason_code == INELIGIBLE_NOT_FINAL


def test_scenario_d_pbp_completing_later_does_not_retroactively_change_earlier_gate():
    """The gate function is pure/stateless: calling it again with
    pbp_complete=True does not (and structurally cannot) reach back and
    mutate a decision already made and recorded at an earlier cutoff --
    each call is independent, so an orchestrator that persisted the
    Scenario C result already has an immutable record of it."""
    game = _game(status_state="final")
    friday_decision = can_update_pbp_state(game, pbp_complete=False)
    assert friday_decision.eligible is False

    later_decision = can_update_pbp_state(game, pbp_complete=True)
    assert later_decision.eligible is True
    # The two calls are independent GateResult values; the first is
    # unaffected by the second having been made.
    assert friday_decision.eligible is False


# -- family-specific first-usable timestamps (Fix 5 certification, Sections 3-5) --

_THURSDAY_KICKOFF_FINAL = "2026-09-10T23:45:00Z"  # game goes final Thursday night
_FRIDAY_CUTOFF = "2026-09-11T12:00:00Z"
_SATURDAY_PBP_COMPLETE = "2026-09-12T09:00:00Z"


def _poll(
    game_id="2026_02_CLE_BAL",
    observed_at_utc="2026-09-10T20:00:00Z",
    status_state="in_progress",
    home_team="BAL",
    away_team="CLE",
    home_score=None,
    away_score=None,
    box_complete=False,
    player_stats_complete=False,
    pbp_complete=False,
):
    return {
        "game_id": game_id,
        "observed_at_utc": observed_at_utc,
        "status_state": status_state,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "box_complete": box_complete,
        "player_stats_complete": player_stats_complete,
        "pbp_complete": pbp_complete,
    }


def test_family_timestamps_independent_when_completeness_lags_finality():
    """Scenario C/D at the family-timestamp level: score/box/player_stats go
    final+complete Thursday night, but PBP completeness doesn't land until
    Saturday -- pbp_available_at_utc must reflect Saturday, not Thursday,
    even though the game's own finality happened Thursday."""
    observations = pd.DataFrame(
        [
            _poll(observed_at_utc="2026-09-10T20:00:00Z", status_state="in_progress", home_score=17.0, away_score=14.0),
            _poll(
                observed_at_utc=_THURSDAY_KICKOFF_FINAL,
                status_state="final",
                home_score=30.0,
                away_score=10.0,
                box_complete=True,
                player_stats_complete=True,
                pbp_complete=False,
            ),
            _poll(
                observed_at_utc=_SATURDAY_PBP_COMPLETE,
                status_state="final",
                home_score=30.0,
                away_score=10.0,
                box_complete=True,
                player_stats_complete=True,
                pbp_complete=True,
            ),
        ]
    )
    ledger = compute_family_available_at_utc(observations)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["score_available_at_utc"] == pd.Timestamp(_THURSDAY_KICKOFF_FINAL)
    assert row["box_available_at_utc"] == pd.Timestamp(_THURSDAY_KICKOFF_FINAL)
    assert row["player_stats_available_at_utc"] == pd.Timestamp(_THURSDAY_KICKOFF_FINAL)
    assert row["pbp_available_at_utc"] == pd.Timestamp(_SATURDAY_PBP_COMPLETE)


def test_scenario_a_in_progress_at_friday_cutoff_all_families_ineligible():
    observations = pd.DataFrame(
        [_poll(observed_at_utc="2026-09-10T20:00:00Z", status_state="in_progress", home_score=17.0, away_score=14.0)]
    )
    ledger = compute_family_available_at_utc(observations)
    row = ledger.iloc[0]
    for column in ("score_available_at_utc", "box_available_at_utc", "player_stats_available_at_utc", "pbp_available_at_utc"):
        assert pd.isna(row[column])
        result = family_eligible_as_of(row[column], _FRIDAY_CUTOFF)
        assert result.eligible is False
        assert result.reason_code == INELIGIBLE_NEVER_OBSERVED_AVAILABLE


def test_scenario_b_score_complete_before_friday_is_eligible_friday():
    observations = pd.DataFrame(
        [
            _poll(
                observed_at_utc=_THURSDAY_KICKOFF_FINAL,
                status_state="final",
                home_score=30.0,
                away_score=10.0,
                box_complete=True,
                player_stats_complete=True,
                pbp_complete=False,
            )
        ]
    )
    ledger = compute_family_available_at_utc(observations)
    row = ledger.iloc[0]
    result = family_eligible_as_of(row["score_available_at_utc"], _FRIDAY_CUTOFF)
    assert result.eligible is True
    assert result.reason_code == ELIGIBLE_AS_OF


def test_scenario_c_and_d_pbp_completing_after_friday_cannot_contaminate_friday_state():
    """The exact invariant Section 3/5 exists to certify: a later poll where
    pbp_complete becomes True must NOT make PBP appear available at the
    earlier Friday cutoff merely because the game's result_available_at_utc
    (or score/box family timestamps) landed Thursday night. score/Elo IS
    eligible Friday; PBP is NOT eligible Friday; PBP becomes eligible only
    once evaluated as-of the Saturday timestamp it actually completed at."""
    observations = pd.DataFrame(
        [
            _poll(
                observed_at_utc=_THURSDAY_KICKOFF_FINAL,
                status_state="final",
                home_score=30.0,
                away_score=10.0,
                box_complete=True,
                player_stats_complete=True,
                pbp_complete=False,
            ),
            _poll(
                observed_at_utc=_SATURDAY_PBP_COMPLETE,
                status_state="final",
                home_score=30.0,
                away_score=10.0,
                box_complete=True,
                player_stats_complete=True,
                pbp_complete=True,
            ),
        ]
    )
    ledger = compute_family_available_at_utc(observations)
    row = ledger.iloc[0]

    score_friday = family_eligible_as_of(row["score_available_at_utc"], _FRIDAY_CUTOFF)
    pbp_friday = family_eligible_as_of(row["pbp_available_at_utc"], _FRIDAY_CUTOFF)
    assert score_friday.eligible is True
    assert pbp_friday.eligible is False
    assert pbp_friday.reason_code == INELIGIBLE_NOT_YET_AVAILABLE_AS_OF

    # D: evaluated as-of a cutoff AT/AFTER the actual PBP completion moment,
    # PBP becomes eligible -- not any earlier.
    pbp_after_saturday = family_eligible_as_of(row["pbp_available_at_utc"], _SATURDAY_PBP_COMPLETE)
    assert pbp_after_saturday.eligible is True
    assert pbp_after_saturday.reason_code == ELIGIBLE_AS_OF

    pbp_one_second_before_saturday = family_eligible_as_of(
        row["pbp_available_at_utc"], pd.Timestamp(_SATURDAY_PBP_COMPLETE) - pd.Timedelta(seconds=1)
    )
    assert pbp_one_second_before_saturday.eligible is False


def test_scenario_e_later_poll_never_moves_family_timestamps_backward_or_overwrites():
    base_observations = [
        _poll(observed_at_utc="2026-09-10T20:00:00Z", status_state="in_progress", home_score=17.0, away_score=14.0),
        _poll(
            observed_at_utc=_THURSDAY_KICKOFF_FINAL,
            status_state="final",
            home_score=30.0,
            away_score=10.0,
            box_complete=True,
            player_stats_complete=True,
            pbp_complete=False,
        ),
    ]
    first_ledger = compute_family_available_at_utc(pd.DataFrame(base_observations))
    first_row = first_ledger.iloc[0]

    grown_observations = base_observations + [
        # A later poll that reconfirms an already-eligible family AND
        # newly satisfies pbp_complete.
        _poll(
            observed_at_utc=_SATURDAY_PBP_COMPLETE,
            status_state="final",
            home_score=30.0,
            away_score=10.0,
            box_complete=True,
            player_stats_complete=True,
            pbp_complete=True,
        ),
        # An even later poll -- must not move anything, including PBP,
        # further forward than its true first-eligible observation.
        _poll(
            observed_at_utc="2026-09-13T00:00:00Z",
            status_state="final",
            home_score=30.0,
            away_score=10.0,
            box_complete=True,
            player_stats_complete=True,
            pbp_complete=True,
        ),
    ]
    grown_ledger = compute_family_available_at_utc(pd.DataFrame(grown_observations))
    grown_row = grown_ledger.iloc[0]

    # score/box/player_stats timestamps recorded from the first log are
    # bit-for-bit identical after the log grows -- never overwritten.
    assert grown_row["score_available_at_utc"] == first_row["score_available_at_utc"]
    assert grown_row["box_available_at_utc"] == first_row["box_available_at_utc"]
    assert grown_row["player_stats_available_at_utc"] == first_row["player_stats_available_at_utc"]
    # pbp_available_at_utc is recorded as the FIRST poll where pbp_complete
    # was true (Saturday), not the later reconfirming poll (Tuesday).
    assert grown_row["pbp_available_at_utc"] == pd.Timestamp(_SATURDAY_PBP_COMPLETE)


def test_family_timestamps_idempotent_over_growing_log():
    base = [
        _poll(observed_at_utc="2026-09-10T20:00:00Z", status_state="in_progress", home_score=17.0, away_score=14.0),
        _poll(
            observed_at_utc=_THURSDAY_KICKOFF_FINAL,
            status_state="final",
            home_score=30.0,
            away_score=10.0,
            box_complete=True,
            player_stats_complete=True,
            pbp_complete=True,
        ),
    ]
    first = compute_family_available_at_utc(pd.DataFrame(base))
    grown = base + [
        _poll(
            observed_at_utc="2026-09-13T00:00:00Z",
            status_state="final",
            home_score=30.0,
            away_score=10.0,
            box_complete=True,
            player_stats_complete=True,
            pbp_complete=True,
        )
    ]
    second = compute_family_available_at_utc(pd.DataFrame(grown))
    for column in ("score_available_at_utc", "box_available_at_utc", "player_stats_available_at_utc", "pbp_available_at_utc"):
        assert first.iloc[0][column] == second.iloc[0][column]


def test_family_timestamps_no_eligible_observation_is_nat_not_fabricated():
    observations = pd.DataFrame(
        [_poll(observed_at_utc="2026-09-10T20:00:00Z", status_state="scheduled")]
    )
    ledger = compute_family_available_at_utc(observations)
    row = ledger.iloc[0]
    for column in ("score_available_at_utc", "box_available_at_utc", "player_stats_available_at_utc", "pbp_available_at_utc"):
        assert pd.isna(row[column])


def test_family_eligible_as_of_never_observed_available():
    result = family_eligible_as_of(None, _FRIDAY_CUTOFF)
    assert result.eligible is False
    assert result.reason_code == INELIGIBLE_NEVER_OBSERVED_AVAILABLE

    result_nat = family_eligible_as_of(pd.NaT, _FRIDAY_CUTOFF)
    assert result_nat.eligible is False
    assert result_nat.reason_code == INELIGIBLE_NEVER_OBSERVED_AVAILABLE


def test_compute_family_available_at_utc_requires_expected_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        compute_family_available_at_utc(pd.DataFrame([{"game_id": "G1"}]))
