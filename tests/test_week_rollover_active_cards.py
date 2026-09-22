"""Proofs that a week rollover cannot orphan the previous Monday-night stages.

THE PROVEN GAP. A card becomes "current" on its own TUE cutoff, but the
previous card is not finished then. In 2026 the sweep advanced from Week 2 to
Week 3 between Sep 20 23:47Z and Sep 21 05:50Z, while Week-2's Monday-night
game ``2026_02_NYG_LAR`` did not close until ``2026-09-21T23:15:00Z`` --
roughly 17 hours later. A sweep that only ever looks at the current card walks
away from a CLOSE that has not happened yet, and once it does, no scheduled
run can ever record it. Week 2 had to be repaired by hand.

Hermetic: ``tmp_path`` archives of synthetic captures built by the real
capture hashing helpers, synthetic games, and the real entry point. No
network, no provider, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.production import snapshot_execution_2026 as ex
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl
from nfl_hybrid.production import snapshot_stages_2026 as st

from test_bdl_market_bridge import _game_row, _odds_row, write_capture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS = ("draftkings", "fanduel", "caesars")


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "_stage_entry_rollover", REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stage_entry_rollover"] = module
    spec.loader.exec_module(module)
    return module


entry = _load_entry()

# The real 2026 shape. Week 2's Monday-nighter closes at 23:15Z on the Monday
# AFTER Week 3 has already become the current card.
W2_SUNDAY = pd.Timestamp("2026-09-20T17:00:00Z")
W2_MONDAY = pd.Timestamp("2026-09-22T00:15:00Z")  # Mon 8:15pm ET Sep 21
W2_MONDAY_CLOSE = pd.Timestamp("2026-09-21T23:15:00Z")
W3_THURSDAY = pd.Timestamp("2026-09-25T00:15:00Z")
W3_SUNDAY = pd.Timestamp("2026-09-27T17:00:00Z")

MONDAY_MORNING = pd.Timestamp("2026-09-21T05:50:00Z")  # the observed rollover
AFTER_MONDAY_CLOSE = pd.Timestamp("2026-09-21T23:30:00Z")

W2_SUN_GAME = "2026_02_HOU_KC"
W2_MON_GAME = "2026_02_BUF_NYJ"
W3_THU_GAME = "2026_03_HOU_KC"
W3_SUN_GAME = "2026_03_BUF_NYJ"


def _games() -> pd.DataFrame:
    rows = []
    for week, pairs in (
        (2, [(("HOU", "KC"), W2_SUNDAY), (("BUF", "NYJ"), W2_MONDAY)]),
        (3, [(("HOU", "KC"), W3_THURSDAY), (("BUF", "NYJ"), W3_SUNDAY)]),
    ):
        for (away, home), kickoff in pairs:
            rows.append(
                {
                    "game_id": f"2026_{week:02d}_{away}_{home}",
                    "season": 2026, "week": week, "season_type": "REG",
                    "home_team_id": home, "away_team_id": away,
                    "scheduled_kickoff_utc": kickoff,
                    "home_score": None, "away_score": None, "neutral_site": False,
                }
            )
    return pd.DataFrame(rows)


def _archive(data_root: Path, observed: str, *, week: int) -> Path:
    stamp = pd.Timestamp(observed).strftime("%Y%m%dT%H%M%SZ")
    staging = data_root / ".staging" / f"w{week}-{stamp}"
    manifest = write_capture(
        staging,
        horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=observed,
        games=[
            _game_row(1, 14, 10, season=2026, week=week),  # HOU @ KC
            _game_row(2, 4, 3, season=2026, week=week),    # BUF @ NYJ
        ],
        odds_pages=[[_odds_row(g, b) for g in (1, 2) for b in BOOKS]],
        odds_received=[observed],
        season=2026,
        week=week,
    )
    final = (
        data_root / ex.STAGE_ARCHIVE_NAMESPACE / "season=2026" / f"week={week:02d}"
        / f"horizon={st.STAGE_CAPTURE_HORIZON}" / f"capture={stamp}"
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(manifest.parent), str(final))
    return final / "manifest.json"


@pytest.fixture
def estate(tmp_path):
    data = tmp_path / "data"
    # Week-2 evidence spanning its OPEN, MID and the Monday CLOSE.
    for observed in (
        "2026-09-15T01:24:12Z", "2026-09-18T15:40:05Z",
        "2026-09-20T15:55:00Z", "2026-09-21T23:05:00Z",
    ):
        _archive(data, observed, week=2)
    # Week-3 evidence already accumulating on the Monday morning.
    for observed in ("2026-09-21T05:00:00Z", "2026-09-21T23:20:00Z"):
        _archive(data, observed, week=3)
    return {"data": data, "artifacts": tmp_path / "artifacts", "games": _games()}


def _sweep(estate, as_of, **kw):
    """A DEFAULT scheduled sweep -- no operator override of any kind."""
    return entry.run(
        stage=kw.pop("stage", "ALL"),
        as_of=str(as_of),
        operational_root=estate["artifacts"],
        market_capture_manifest=None,
        data_root=estate["data"],
        games=estate["games"],
        **kw,
    )


def _weeks_processed(result) -> list[str]:
    return [card["week"] for card in result["active_cards"]]


def _recorded(estate, *, week, game_id, stage):
    return pl.read_snapshot(
        estate["artifacts"], season=2026, week=week, game_id=game_id, stage=stage
    )


# ===========================================================================
# 1. The exact observed failure.
# ===========================================================================
def test_the_card_really_does_roll_over_before_the_monday_close(estate):
    """Pins the premise: on Monday morning the current card is already
    Week 3, while Week 2's Monday CLOSE is still 17 hours away."""
    _, info = entry.resolve_current_card(estate["games"], MONDAY_MORNING)
    assert info["week"] == "3"
    assert st.close_cutoff_utc(W2_MONDAY) == W2_MONDAY_CLOSE
    assert W2_MONDAY_CLOSE > MONDAY_MORNING


def test_monday_morning_sweep_processes_both_cards(estate):
    result = _sweep(estate, MONDAY_MORNING)
    assert _weeks_processed(result) == ["2", "3"]
    roles = [card.get("role") for card in result["active_cards"]]
    assert roles == ["PREVIOUS_STILL_ACTIVE", "CURRENT"]


def test_next_week_open_is_not_delayed_by_the_overlap(estate):
    """Requirement 3: Week-3 evidence keeps accumulating on Monday morning."""
    result = _sweep(estate, MONDAY_MORNING)
    assert result["status"] == "OK"
    week3 = [c for c in result["cards"] if c["card"]["week"] == "3"][0]
    assert week3["open_observations"], "Week-3 OPEN must still be discovered"
    assert _recorded(estate, week=3, game_id=W3_THU_GAME, stage=st.STAGE_OPEN) is not None


def test_the_monday_close_is_recorded_automatically_after_it_becomes_due(estate):
    """THE REGRESSION. No operator override, no --season/--week."""
    _sweep(estate, MONDAY_MORNING)
    assert _recorded(estate, week=2, game_id=W2_MON_GAME, stage=st.STAGE_CLOSE) is None

    result = _sweep(estate, AFTER_MONDAY_CLOSE)
    assert result["status"] == "OK"

    recorded = _recorded(estate, week=2, game_id=W2_MON_GAME, stage=st.STAGE_CLOSE)
    assert recorded is not None
    assert pd.Timestamp(recorded["snapshot_at_utc"]) == W2_MONDAY_CLOSE


def test_no_operator_historical_override_is_needed(estate):
    """The sweep that fixes it passes no repair flag at all."""
    result = _sweep(estate, AFTER_MONDAY_CLOSE)
    assert result["card"].get("mode") != "HISTORICAL_REPAIR"
    for card in result["active_cards"]:
        assert card.get("mode") != "HISTORICAL_REPAIR"


# ===========================================================================
# 2. The previous card drops out once it is finished.
# ===========================================================================
def test_a_finished_previous_card_is_no_longer_active(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    # Everything resolvable for Week 2 is now recorded.
    later = _sweep(estate, pd.Timestamp("2026-09-23T12:00:00Z"))
    assert _weeks_processed(later) == ["3"]


def test_activity_is_decided_by_the_ledger_not_the_clock(estate):
    card = estate["games"][estate["games"]["week"] == 2][["game_id", "scheduled_kickoff_utc"]]
    assert entry.card_still_stage_active(
        card, season=2026, week="2", operational_root=estate["artifacts"]
    )
    _sweep(estate, AFTER_MONDAY_CLOSE)
    assert not entry.card_still_stage_active(
        card, season=2026, week="2", operational_root=estate["artifacts"]
    )


def test_only_one_card_back_is_ever_considered(estate):
    """A backlog sweeper would quietly re-open weeks closed deliberately."""
    _, current = entry.resolve_current_card(estate["games"], pd.Timestamp("2026-09-29T12:00:00Z"))
    previous = entry._previous_card(estate["games"], current["tue_cutoff_utc"])
    assert previous is not None
    assert previous[1]["week"] == "3"


def test_the_first_card_of_the_season_has_no_previous(estate):
    _, first = entry.resolve_current_card(estate["games"], W2_SUNDAY)
    assert entry._previous_card(estate["games"], first["tue_cutoff_utc"]) is None


def test_open_alone_never_keeps_a_card_active(estate):
    """OPEN depends on observed evidence that may never exist, so an
    unobservable OPEN must not pin a finished card open forever."""
    card = estate["games"][estate["games"]["week"] == 2][["game_id", "scheduled_kickoff_utc"]]
    _sweep(estate, AFTER_MONDAY_CLOSE)
    for game_id in (W2_SUN_GAME, W2_MON_GAME):
        path = pl.snapshot_path(
            estate["artifacts"], season=2026, week=2, game_id=game_id, stage=st.STAGE_OPEN
        )
        if path.is_file():
            path.unlink()
    assert not entry.card_still_stage_active(
        card, season=2026, week="2", operational_root=estate["artifacts"]
    )


# ===========================================================================
# 3. Ordinary weeks with no overlap behave exactly as before.
# ===========================================================================
def test_a_week_with_nothing_outstanding_processes_one_card(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    result = _sweep(estate, pd.Timestamp("2026-09-24T12:00:00Z"))
    assert _weeks_processed(result) == ["3"]
    assert result["card"]["week"] == "3"


def test_the_single_card_result_shape_is_preserved(estate):
    """Readers that only ever expected one card still work."""
    _sweep(estate, AFTER_MONDAY_CLOSE)
    result = _sweep(estate, pd.Timestamp("2026-09-24T12:00:00Z"))
    for key in ("status", "as_of_utc", "card", "stage_archive", "open_observations", "stages"):
        assert key in result
    assert result["card"]["week"] == "3"


def test_no_current_card_is_still_reported(estate):
    empty = estate["games"].iloc[0:0]
    result = entry.run(
        stage="ALL", as_of="2026-09-24T12:00:00Z",
        operational_root=estate["artifacts"], market_capture_manifest=None,
        data_root=estate["data"], games=empty,
    )
    assert result["status"] == "NO_CURRENT_CARD"
    assert result["stages"] == []


def test_historical_repair_mode_is_unaffected(estate):
    result = _sweep(estate, AFTER_MONDAY_CLOSE, historical_season=2026, historical_week=2)
    assert _weeks_processed(result) == ["2"]
    assert result["card"]["mode"] == "HISTORICAL_REPAIR"


# ===========================================================================
# 4. Guards, idempotence and no duplicated work.
# ===========================================================================
def test_repeated_sweeps_are_idempotent(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    root = pl.ledger_root(estate["artifacts"])
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.json"))}

    _sweep(estate, AFTER_MONDAY_CLOSE)
    assert {p: p.read_bytes() for p in sorted(root.rglob("*.json"))} == before


def test_already_recorded_stages_stay_skipped(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    again = _sweep(estate, AFTER_MONDAY_CLOSE)
    reasons = {
        reason
        for stage_result in again["stages"]
        for reason in stage_result["skipped"].values()
    }
    assert ex.SKIP_ALREADY_RECORDED in reasons
    assert all(s["due_batches"] == 0 for s in again["stages"])


def test_processing_two_cards_creates_no_duplicate_artifacts(estate):
    result = _sweep(estate, AFTER_MONDAY_CLOSE)
    archives = {
        card["card"]["week"]: card["stage_archive"]["archived_capture_count"]
        for card in result["cards"]
    }
    # Each card reads only its OWN week's archive; nothing is copied.
    assert archives == {"2": 4, "3": 2}
    captures = list((estate["data"] / ex.STAGE_ARCHIVE_NAMESPACE).rglob("manifest.json"))
    assert len(captures) == 6


def test_a_genuine_conflict_still_raises(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    recorded = _recorded(estate, week=2, game_id=W2_MON_GAME, stage=st.STAGE_CLOSE)
    with pytest.raises(pl.SnapshotImmutabilityViolation, match="immutable"):
        pl.record_snapshot(estate["artifacts"], {**recorded, "model_home_margin": 99.0})


def test_close_remains_kickoff_minus_sixty_on_both_cards(estate):
    _sweep(estate, pd.Timestamp("2026-09-28T12:00:00Z"))
    for week, game_id, kickoff in (
        (2, W2_MON_GAME, W2_MONDAY), (3, W3_SUN_GAME, W3_SUNDAY)
    ):
        recorded = _recorded(estate, week=week, game_id=game_id, stage=st.STAGE_CLOSE)
        if recorded is not None:
            assert pd.Timestamp(recorded["snapshot_at_utc"]) == kickoff - pd.Timedelta(minutes=60)


def test_no_lookahead_on_the_previous_cards_close(estate):
    """The Monday CLOSE must price from evidence observed before it."""
    _sweep(estate, AFTER_MONDAY_CLOSE)
    captures = ex.archived_stage_captures(estate["data"], season=2026, week=2)
    chosen = ex.newest_capture_at_or_before(captures, W2_MONDAY_CLOSE)
    assert chosen.observed_at_utc == pd.Timestamp("2026-09-21T23:05:00Z")
    assert chosen.observed_at_utc <= W2_MONDAY_CLOSE


def test_the_three_book_floor_is_unchanged():
    from nfl_hybrid.evaluation import raw_market_reconstruction as rmr

    assert rmr.MINIMUM_FRESH_COHERENT_BOOKS == 3


# ===========================================================================
# 5. Publication: the sweep entry point still cannot publish anything.
# ===========================================================================
def test_the_entry_point_has_no_publication_path():
    import ast

    source = (REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = (
        {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    )
    for forbidden in (
        "publish_wizard_nfl_local", "publish_2026_current_week",
        "write_latest", "publish_current_week_feed",
    ):
        assert not any(forbidden in n for n in names), forbidden


def test_processing_the_previous_card_writes_no_public_tree(estate):
    _sweep(estate, AFTER_MONDAY_CLOSE)
    assert not (estate["artifacts"] / "public").exists()


def test_the_due_close_is_available_to_the_publisher(estate):
    """Requirement 7: processing the upcoming card must not suppress the
    previous card's due CLOSE -- it is recorded and readable."""
    _sweep(estate, AFTER_MONDAY_CLOSE)
    recorded = _recorded(estate, week=2, game_id=W2_MON_GAME, stage=st.STAGE_CLOSE)
    assert recorded is not None
    assert recorded["snapshot_stage"] == st.STAGE_CLOSE
    assert pd.Timestamp(recorded["snapshot_at_utc"]) == W2_MONDAY_CLOSE


def test_the_feed_publishes_close_ahead_of_open_and_mid():
    """The publisher's preference order is unchanged: a CLOSE always wins."""
    spec = importlib.util.spec_from_file_location(
        "_feed_pref", REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py"
    )
    feed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feed)
    assert feed.PUBLICATION_PREFERENCE[0] == st.STAGE_CLOSE
