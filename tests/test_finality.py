"""CI-safe finality resolution tests (Step 1). Synthetic fixtures only -- no backfill.

Covers: in-progress game with populated final score, same game after completion,
final score without finality evidence, explicit non-final status, final-but-completion-
after-as-of, late-window Sunday, no-partial-aggregation (resolver level), determinism.
"""

import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import resolve_finality_before_asof


def _game(gid, kickoff, **kw):
    row = {"game_id": gid, "scheduled_kickoff_utc": kickoff}
    row.update(kw)
    return row


def _resolve(rows, as_of):
    return resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc=as_of).set_index("game_id")


def test_A_in_progress_with_populated_final_score_excluded():
    # kickoff 13:00, completes 16:08, predict at 15:55; historical row already has scores
    r = _resolve([_game("A", "2024-11-10T18:00:00Z", game_status="FINAL", home_score=24, away_score=17,
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-10T20:55:00Z")
    assert not r.loc["A", "is_final_before_asof"]
    assert r.loc["A", "finality_status"] == "IN_PROGRESS_AT_ASOF"
    assert "completion_after_asof" in r.loc["A", "exclusion_reason"]


def test_B_same_game_after_completion_eligible():
    r = _resolve([_game("A", "2024-11-10T18:00:00Z", game_status="FINAL",
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-10T21:10:00Z")
    assert r.loc["A", "is_final_before_asof"]
    assert r.loc["A", "finality_status"] == "FINAL_BEFORE_ASOF"
    assert pd.Timestamp(r.loc["A", "completion_timestamp_utc"]) <= pd.Timestamp("2024-11-10T21:10:00Z")


def test_C_final_score_without_finality_evidence_unknown():
    r = _resolve([_game("C", "2024-11-10T18:00:00Z", home_score=24, away_score=17)], "2024-11-10T23:00:00Z")
    assert not r.loc["C", "is_final_before_asof"]
    assert r.loc["C", "finality_status"] == "UNKNOWN_FINALITY"


def test_D_explicit_non_final_status_excluded():
    r = _resolve([_game("D", "2024-11-10T18:00:00Z", game_status="IN_PROGRESS", home_score=24, away_score=17,
                        completion_timestamp_utc="2024-11-10T21:08:00Z")], "2024-11-10T23:00:00Z")
    assert not r.loc["D", "is_final_before_asof"]
    assert r.loc["D", "finality_status"] == "EXPLICIT_NON_FINAL"


def test_E_final_status_but_completion_after_asof_excluded():
    r = _resolve([_game("E", "2024-11-10T18:00:00Z", game_status="FINAL",
                        completion_timestamp_utc="2024-11-11T00:00:00Z")], "2024-11-10T22:00:00Z")
    assert not r.loc["E", "is_final_before_asof"]


def test_H_late_window_sunday_early_game_excluded():
    # early game still running at the late game's pre-kickoff as_of
    r = _resolve([
        # early game kicked off 18:00, still running (completes 21:30) at as_of 21:15
        _game("early", "2024-11-10T18:00:00Z", game_status="FINAL", completion_timestamp_utc="2024-11-10T21:30:00Z"),
        _game("late", "2024-11-10T21:25:00Z", game_status="SCHEDULED"),
    ], "2024-11-10T21:15:00Z")
    assert not r.loc["early", "is_final_before_asof"]  # in progress (completes after as_of)
    assert not r.loc["late", "is_final_before_asof"]   # not yet kicked off / scheduled


def test_G_in_progress_game_is_not_eligible():
    # an in-progress game must never enter the eligible (aggregated) set
    r = _resolve([_game("live", "2024-11-10T18:00:00Z", game_status="LIVE",
                        completion_timestamp_utc="2024-11-10T22:00:00Z")], "2024-11-10T19:30:00Z")
    assert not r.loc["live", "is_final_before_asof"]


def test_post_asof_kickoff_excluded():
    r = _resolve([_game("future", "2024-11-12T18:00:00Z", game_status="SCHEDULED")], "2024-11-10T12:00:00Z")
    assert r.loc["future", "finality_status"] == "POST_ASOF_KICKOFF"
    assert not r.loc["future", "is_final_before_asof"]


def test_J_determinism():
    rows = [_game("A", "2024-11-10T18:00:00Z", game_status="FINAL", completion_timestamp_utc="2024-11-10T21:08:00Z"),
            _game("B", "2024-11-10T18:00:00Z", home_score=1, away_score=0)]
    a = resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc="2024-11-10T22:00:00Z")
    b = resolve_finality_before_asof(pd.DataFrame(rows), None, as_of_utc="2024-11-10T22:00:00Z")
    pd.testing.assert_frame_equal(a, b)
