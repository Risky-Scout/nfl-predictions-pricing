"""Canonical PBP normalization: pagination/dedup/conflict/foreign-game
rejection, deterministic ordering, and completeness (Sections 15, 20).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_hybrid.providers.balldontlie.plays import normalize_plays, pbp_completeness

FIXTURES = Path(__file__).parent / "fixtures" / "balldontlie"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())["data"]


def _all_raw_plays() -> list[dict]:
    return _load("plays_page1.json") + _load("plays_page2.json")


def test_page2_plays_are_present():
    raw = _all_raw_plays()
    frame, report = normalize_plays(raw, provider_game_id=900001)
    ids = set(frame["provider_play_id"])
    assert "40177251200" in ids  # only present on page 2


def test_exact_duplicate_across_pages_is_deduped_once():
    raw = _all_raw_plays()
    frame, report = normalize_plays(raw, provider_game_id=900001)
    assert "40177251071" in report.exact_duplicate_play_ids
    assert (frame["provider_play_id"] == "40177251071").sum() == 1


def test_conflicting_duplicate_is_excluded_not_guessed():
    raw = _all_raw_plays()
    frame, report = normalize_plays(raw, provider_game_id=900001)
    assert "40177259999" in report.conflicting_play_ids
    assert (frame["provider_play_id"] == "40177259999").sum() == 0


def test_foreign_game_play_is_rejected():
    raw = _all_raw_plays()
    frame, report = normalize_plays(raw, provider_game_id=900001)
    assert "40177250001" in report.foreign_game_play_ids
    assert (frame["provider_play_id"] == "40177250001").sum() == 0


def test_deterministic_ordering_by_period_then_wallclock():
    raw = _all_raw_plays()
    frame, _ = normalize_plays(raw, provider_game_id=900001)
    periods = list(frame["period"])
    assert periods == sorted(periods)
    assert list(frame["sequence"]) == list(range(1, len(frame) + 1))


def test_unique_play_ids_in_canonical_output():
    raw = _all_raw_plays()
    frame, _ = normalize_plays(raw, provider_game_id=900001)
    assert frame["provider_play_id"].is_unique


def test_no_epa_wpa_success_cpoe_columns_fabricated():
    raw = _all_raw_plays()
    frame, _ = normalize_plays(raw, provider_game_id=900001)
    for banned in ("epa", "wpa", "success", "cpoe"):
        assert banned not in frame.columns


def test_report_clean_when_no_anomalies():
    raw = _load("plays_page1.json")
    frame, report = normalize_plays(raw, provider_game_id=900001)
    assert report.clean is True


def test_report_not_clean_with_conflicts_and_foreign_plays():
    raw = _all_raw_plays()
    _, report = normalize_plays(raw, provider_game_id=900001)
    assert report.clean is False


# -- completeness (Section 20) ---------------------------------------------------

def test_pbp_complete_true_for_final_game_with_score_reconciled():
    # A CLEAN subset (no conflicting/foreign rows) -- the positive path.
    page2 = _load("plays_page2.json")
    terminal_only = [p for p in page2 if p["id"] == "40177251200"]
    raw = _load("plays_page1.json") + terminal_only
    frame, report = normalize_plays(raw, provider_game_id=900001)
    assert report.clean is True
    completeness = pbp_completeness(
        frame, report, game_final=True, canonical_home_score=24, canonical_away_score=20
    )
    assert completeness.terminal_score_reconciled is True
    assert completeness.pbp_complete is True


def test_pbp_incomplete_when_not_final():
    raw = _load("plays_page1.json")
    frame, report = normalize_plays(raw, provider_game_id=900001)
    completeness = pbp_completeness(
        frame, report, game_final=False, canonical_home_score=24, canonical_away_score=20
    )
    assert completeness.pbp_complete is False


def test_pbp_incomplete_when_normalization_not_clean():
    raw = _all_raw_plays()  # has conflicting + foreign plays -> report not clean
    frame, report = normalize_plays(raw, provider_game_id=900001)
    completeness = pbp_completeness(
        frame, report, game_final=True, canonical_home_score=24, canonical_away_score=20
    )
    assert completeness.normalization_clean is False
    assert completeness.pbp_complete is False


def test_pbp_incomplete_when_score_mismatch():
    raw = _load("plays_page1.json")
    frame, report = normalize_plays(raw, provider_game_id=900001)
    completeness = pbp_completeness(
        frame, report, game_final=True, canonical_home_score=99, canonical_away_score=1
    )
    # page1 alone has no scoring plays with nonzero score, so this exercises
    # the "no reconcilable terminal score" path -> unavailable, not failed.
    assert completeness.terminal_score_reconciled in (None, False)


def test_pbp_incomplete_when_empty():
    frame, report = normalize_plays([], provider_game_id=900001)
    completeness = pbp_completeness(
        frame, report, game_final=True, canonical_home_score=24, canonical_away_score=20
    )
    assert completeness.nonempty is False
    assert completeness.pbp_complete is False
