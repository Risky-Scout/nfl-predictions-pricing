"""CI-safe finality tests (Step 1 remediation). Synthetic fixtures only.

Verified finality (live) is evidence-only and fails closed. A historical availability
assumption exists ONLY in historical_replay mode and is never reported as verified.
"""

import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import resolve_finality_before_asof


def _game(gid, kickoff, **kw):
    row = {"game_id": gid, "scheduled_kickoff_utc": kickoff}
    row.update(kw)
    return row


def _resolve(rows, as_of, mode="live"):
    return resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc=as_of, mode=mode).set_index("game_id")


# ---- verified finality (live) ------------------------------------------- #
def test_in_progress_with_populated_final_score_excluded():
    r = _resolve([_game("A", "2024-11-10T18:00:00Z", game_status="FINAL", home_score=24, away_score=17,
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-10T20:55:00Z")
    assert not r.loc["A", "verified_final_before_asof"]
    assert r.loc["A", "finality_status"] == "IN_PROGRESS_AT_ASOF"


def test_same_game_after_completion_eligible():
    r = _resolve([_game("A", "2024-11-10T18:00:00Z", game_status="FINAL",
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-10T21:10:00Z")
    assert r.loc["A", "verified_final_before_asof"]
    assert r.loc["A", "finality_status"] == "FINAL_BEFORE_ASOF"
    assert pd.Timestamp(r.loc["A", "finality_evidence_timestamp"]) <= pd.Timestamp("2024-11-10T21:10:00Z")


def test_synthetic_duration_rejection_unknown_in_live():
    # kickoff 13:00, as_of 18:30 (>5h elapsed), final scores populated, NO status/end/terminal
    r = _resolve([_game("C", "2024-11-10T18:00:00Z", home_score=24, away_score=17)], "2024-11-10T23:30:00Z")
    assert not r.loc["C", "verified_final_before_asof"]
    assert r.loc["C", "finality_status"] == "UNKNOWN_FINALITY"
    assert not r.loc["C", "historically_available_before_asof"]  # not available in live mode


def test_suspended_game_excluded_regardless_of_duration():
    r = _resolve([_game("S", "2024-11-10T18:00:00Z", game_status="SUSPENDED", home_score=24, away_score=17,
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-11T12:00:00Z")
    assert not r.loc["S", "verified_final_before_asof"]
    assert r.loc["S", "finality_status"] == "EXPLICIT_NON_FINAL"


def test_final_status_but_completion_after_asof_excluded():
    r = _resolve([_game("E", "2024-11-10T18:00:00Z", game_status="FINAL",
                        completion_timestamp_utc="2024-11-11T00:00:00Z")], "2024-11-10T22:00:00Z")
    assert not r.loc["E", "verified_final_before_asof"]


def test_post_asof_kickoff_excluded():
    r = _resolve([_game("F", "2024-11-12T18:00:00Z", game_status="SCHEDULED")], "2024-11-10T12:00:00Z")
    assert r.loc["F", "finality_status"] == "POST_ASOF_KICKOFF"
    assert not r.loc["F", "verified_final_before_asof"]


# ---- historical availability (replay only) ------------------------------ #
def test_historical_mode_separates_availability_from_verified():
    # no verified evidence, but in historical_replay the documented convention applies
    r = _resolve([_game("H", "2024-11-10T18:00:00Z", home_score=24, away_score=17)],
                 "2024-11-11T12:00:00Z", mode="historical_replay")
    assert r.loc["H", "historically_available_before_asof"]          # available under the convention
    assert not r.loc["H", "verified_final_before_asof"]              # but NOT verified finality
    assert r.loc["H", "availability_status"] == "HISTORICAL_AVAILABILITY_ASSUMPTION"
    assert r.loc["H", "availability_source"] == "documented_training_policy"
    assert r.loc["H", "finality_status"] == "HISTORICAL_AVAILABILITY_ASSUMPTION"


def test_historical_availability_absent_within_5h_window():
    # even in replay mode, a game < 5h old is not yet available
    r = _resolve([_game("H", "2024-11-10T18:00:00Z", home_score=24, away_score=17)],
                 "2024-11-10T21:00:00Z", mode="historical_replay")
    assert not r.loc["H", "historically_available_before_asof"]


def test_live_mode_never_uses_availability_assumption():
    r = _resolve([_game("H", "2024-11-10T18:00:00Z", home_score=24, away_score=17)],
                 "2024-11-11T12:00:00Z", mode="live")
    assert not r.loc["H", "historically_available_before_asof"]
    assert r.loc["H", "availability_status"] == ""


def test_determinism():
    rows = [_game("A", "2024-11-10T18:00:00Z", game_status="FINAL", completion_timestamp_utc="2024-11-10T21:08:00Z"),
            _game("B", "2024-11-10T18:00:00Z", home_score=1, away_score=0)]
    a = resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc="2024-11-10T22:00:00Z", mode="historical_replay")
    b = resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc="2024-11-10T22:00:00Z", mode="historical_replay")
    pd.testing.assert_frame_equal(a, b)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        resolve_finality_before_asof(pd.DataFrame([_game("A", "2024-11-10T18:00:00Z")]),
                                     None, as_of_utc="2024-11-10T22:00:00Z", mode="bogus")
