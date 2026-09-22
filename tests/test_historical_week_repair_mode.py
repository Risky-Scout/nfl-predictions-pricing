"""Proofs for explicit historical-week stage repair.

WHY IT EXISTS. The sweep asks which card is CURRENT. That is right for
unattended operation and wrong for repair: Week-2's defective OPEN rows were
quarantined for correction while Week 2 was current, the card advanced to
Week 3 before the corrected rows were written, and from that moment no
ordinary sweep would ever look at Week 2 again. The gap could not be filled.

Naming a season/week selects that card instead. ONLY the selection changes --
every stage rule, cutoff, market gate and ledger guard downstream is the same
code the scheduled sweep runs.

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
        "_stage_entry_repair", REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stage_entry_repair"] = module
    spec.loader.exec_module(module)
    return module


entry = _load_entry()

# Two cards a week apart. Week 2 is the one that has gone by; Week 3 is
# current. Kickoffs are Sunday, so every card shares one CLOSE group.
WEEK2_KICKOFF = pd.Timestamp("2026-09-20T17:00:00Z")
WEEK3_KICKOFF = pd.Timestamp("2026-09-27T17:00:00Z")
NOW = pd.Timestamp("2026-09-24T12:00:00Z")  # after Week 2, before Week 3 kicks off


def _games() -> pd.DataFrame:
    rows = []
    for week, kickoff, teams in (
        (2, WEEK2_KICKOFF, [("HOU", "KC"), ("BUF", "NYJ")]),
        (3, WEEK3_KICKOFF, [("HOU", "KC"), ("BUF", "NYJ")]),
    ):
        for away, home in teams:
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


def _archive(data_root: Path, observed: str, *, week: int, books=BOOKS) -> Path:
    """One archived STAGE capture at the real production archive path."""
    stamp = pd.Timestamp(observed).strftime("%Y%m%dT%H%M%SZ")
    staging = data_root / ".staging" / f"w{week}-{stamp}"
    manifest = write_capture(
        staging,
        horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=observed,
        games=[
            _game_row(1, 14, 10, season=2026, week=week),   # HOU @ KC
            _game_row(2, 4, 3, season=2026, week=week),     # BUF @ NYJ
        ],
        odds_pages=[[_odds_row(g, b) for g in (1, 2) for b in books]],
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
    """Week-2 archive spanning its OPEN, MID and CLOSE; Week-3 archive too."""
    data = tmp_path / "data"
    for observed in ("2026-09-15T01:24:12Z", "2026-09-18T15:40:05Z", "2026-09-20T15:55:00Z"):
        _archive(data, observed, week=2)
    _archive(data, "2026-09-24T11:00:00Z", week=3)
    return {"data": data, "artifacts": tmp_path / "artifacts", "games": _games()}


def _run(estate, **kw):
    kw.setdefault("stage", st.STAGE_OPEN)
    kw.setdefault("as_of", str(NOW))
    return entry.run(
        operational_root=estate["artifacts"],
        market_capture_manifest=None,
        data_root=estate["data"],
        games=estate["games"],
        **kw,
    )


# ===========================================================================
# 1. Default current-card behaviour is unchanged.
# ===========================================================================
def test_the_default_mode_still_resolves_the_current_card(estate):
    card, info = entry.resolve_current_card(estate["games"], NOW)
    assert info["status"] == "OK"
    assert info["week"] == "3"
    assert set(card["game_id"]) == {"2026_03_HOU_KC", "2026_03_BUF_NYJ"}
    assert "mode" not in info


def test_the_default_run_processes_the_current_week(estate):
    result = _run(estate, dry_run=True)
    assert result["card"]["week"] == "3"
    assert result["card"].get("mode") is None


def test_week_three_behaviour_is_untouched_by_a_week_two_repair(estate):
    before = _run(estate, dry_run=True)
    _run(estate, historical_season=2026, historical_week=2)
    after = _run(estate, dry_run=True)

    assert before["card"] == after["card"] == {**before["card"]}
    assert after["card"]["week"] == "3"
    # Week-3 stages remain unwritten by the Week-2 repair.
    for game_id in ("2026_03_HOU_KC", "2026_03_BUF_NYJ"):
        assert pl.read_snapshot(
            estate["artifacts"], season=2026, week=3, game_id=game_id, stage=st.STAGE_OPEN
        ) is None


# ===========================================================================
# 2. Explicit historical selection works.
# ===========================================================================
def test_an_explicit_week_selects_that_card(estate):
    card, info = entry.resolve_historical_card(estate["games"], season=2026, week=2)
    assert info["status"] == "OK"
    assert info["mode"] == "HISTORICAL_REPAIR"
    assert info["week"] == "2"
    assert set(card["game_id"]) == {"2026_02_HOU_KC", "2026_02_BUF_NYJ"}


def test_an_unknown_week_is_reported_not_guessed(estate):
    _, info = entry.resolve_historical_card(estate["games"], season=2026, week=17)
    assert info["status"] == "NO_SUCH_CARD"

    result = _run(estate, historical_season=2026, historical_week=17)
    assert result["status"] == "NO_SUCH_CARD"
    assert result["stages"] == []


def test_season_and_week_must_be_given_together(estate):
    with pytest.raises(SystemExit, match="BOTH --season and --week"):
        _run(estate, historical_season=2026)
    with pytest.raises(SystemExit, match="BOTH --season and --week"):
        _run(estate, historical_week=2)


def test_the_cli_exposes_the_repair_flags():
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    for flag in ("--season", "--week", "--dry-run"):
        assert flag in out


# ===========================================================================
# 3. Historical OPEN resolves from the archive, at the right instant.
# ===========================================================================
def test_historical_open_uses_the_earliest_archived_observation(estate):
    result = _run(estate, historical_season=2026, historical_week=2)
    assert result["status"] == "OK"
    observed = set(result["open_observations"].values())
    assert observed == {"2026-09-15 01:24:12+00:00"}


def test_the_repair_reads_only_that_weeks_archive(estate):
    result = _run(estate, historical_season=2026, historical_week=2, dry_run=True)
    # Three Week-2 captures; the Week-3 one is not in scope.
    assert result["stage_archive"]["archived_capture_count"] == 3
    assert result["stage_archive"]["usable_capture_count"] == 3


def test_historical_open_is_written_to_that_weeks_ledger(estate):
    _run(estate, historical_season=2026, historical_week=2)
    for game_id in ("2026_02_HOU_KC", "2026_02_BUF_NYJ"):
        recorded = pl.read_snapshot(
            estate["artifacts"], season=2026, week=2, game_id=game_id, stage=st.STAGE_OPEN
        )
        assert recorded is not None
        assert pd.Timestamp(recorded["snapshot_at_utc"]) == pd.Timestamp("2026-09-15T01:24:12Z")


# ===========================================================================
# 4. MID / CLOSE keep their cutoffs and their no-lookahead rule.
# ===========================================================================
def test_historical_mid_and_close_resolve_from_pre_cutoff_evidence(estate):
    for stage in (st.STAGE_MID, st.STAGE_CLOSE):
        result = _run(estate, stage=stage, historical_season=2026, historical_week=2)
        assert result["status"] == "OK", (stage, result)
        assert result["stages"][0]["due_batches"] == 1


def test_close_is_still_kickoff_minus_sixty(estate):
    assert WEEK2_KICKOFF - st.close_cutoff_utc(WEEK2_KICKOFF) == pd.Timedelta(minutes=60)
    _run(estate, stage=st.STAGE_CLOSE, historical_season=2026, historical_week=2)
    recorded = pl.read_snapshot(
        estate["artifacts"], season=2026, week=2, game_id="2026_02_HOU_KC", stage=st.STAGE_CLOSE
    )
    assert pd.Timestamp(recorded["snapshot_at_utc"]) == WEEK2_KICKOFF - pd.Timedelta(minutes=60)


def test_the_capture_chosen_for_a_stage_never_postdates_its_cutoff(estate):
    captures = ex.archived_stage_captures(estate["data"], season=2026, week=2)
    close = st.close_cutoff_utc(WEEK2_KICKOFF)
    chosen = ex.newest_capture_at_or_before(captures, close)
    assert chosen.observed_at_utc == pd.Timestamp("2026-09-20T15:55:00Z")
    assert chosen.observed_at_utc <= close


def test_no_lookahead_when_nothing_predates_the_cutoff(estate):
    captures = ex.archived_stage_captures(estate["data"], season=2026, week=2)
    assert ex.newest_capture_at_or_before(captures, pd.Timestamp("2026-09-14T00:00:00Z")) is None


def test_the_three_book_floor_is_unchanged():
    from nfl_hybrid.evaluation import raw_market_reconstruction as rmr

    assert rmr.MINIMUM_FRESH_COHERENT_BOOKS == 3


def test_the_thursday_midpoint_rule_is_unchanged():
    thu = pd.Timestamp("2026-09-18T00:15:00Z")
    open_at = pd.Timestamp("2026-09-15T01:24:12Z")
    schedule = st.resolve_game_snapshots(
        game_id="TNF", scheduled_kickoff_utc=thu,
        card_earliest_kickoff_utc=thu, open_observed_at_utc=open_at,
    )
    close = st.close_cutoff_utc(thu)
    assert schedule.mid_basis == st.MID_BASIS_OPEN_CLOSE_MIDPOINT
    assert schedule.mid_cutoff_utc == open_at + (close - open_at) / 2
    assert schedule.mid_cutoff_utc < close < thu


# ===========================================================================
# 5. Ledger guards survive.
# ===========================================================================
def test_an_already_correct_stage_is_skipped(estate):
    first = _run(estate, historical_season=2026, historical_week=2)
    assert first["stages"][0]["due_batches"] == 1

    second = _run(estate, historical_season=2026, historical_week=2)
    assert second["stages"][0]["due_batches"] == 0
    assert set(second["stages"][0]["skipped"].values()) == {ex.SKIP_ALREADY_RECORDED}


def test_a_second_identical_repair_is_idempotent(estate):
    _run(estate, historical_season=2026, historical_week=2)
    root = pl.ledger_root(estate["artifacts"])
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.json"))}

    _run(estate, historical_season=2026, historical_week=2)
    assert {p: p.read_bytes() for p in sorted(root.rglob("*.json"))} == before


def test_a_genuine_conflict_still_raises(estate):
    """The immutability guard is untouched by repair mode."""
    _run(estate, historical_season=2026, historical_week=2)
    recorded = pl.read_snapshot(
        estate["artifacts"], season=2026, week=2, game_id="2026_02_HOU_KC", stage=st.STAGE_OPEN
    )
    with pytest.raises(pl.SnapshotImmutabilityViolation, match="immutable"):
        pl.record_snapshot(estate["artifacts"], {**recorded, "model_home_margin": 99.0})


def test_a_missing_historical_stage_can_be_written(estate):
    """The whole point: a gap left by the quarantine is fillable."""
    assert pl.read_snapshot(
        estate["artifacts"], season=2026, week=2, game_id="2026_02_HOU_KC", stage=st.STAGE_OPEN
    ) is None
    _run(estate, historical_season=2026, historical_week=2)
    assert pl.read_snapshot(
        estate["artifacts"], season=2026, week=2, game_id="2026_02_HOU_KC", stage=st.STAGE_OPEN
    ) is not None


# ===========================================================================
# 6. Repair cannot touch the public feed.
# ===========================================================================
def test_the_dry_run_writes_nothing(estate):
    result = _run(estate, historical_season=2026, historical_week=2, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["wrote_nothing"] is True
    assert result["stages"][0]["due_batches"] == 1
    assert not pl.ledger_root(estate["artifacts"]).exists()


def test_the_dry_run_names_what_it_would_write(estate):
    result = _run(estate, historical_season=2026, historical_week=2, dry_run=True)
    would = result["stages"][0]["would_write"]
    assert len(would) == 1
    assert sorted(would[0]["game_ids"]) == ["2026_02_BUF_NYJ", "2026_02_HOU_KC"]


def test_repair_never_creates_a_public_tree(estate):
    _run(estate, historical_season=2026, historical_week=2)
    assert not (estate["artifacts"] / "public").exists()


def test_repair_writes_only_under_the_requested_week(estate):
    _run(estate, historical_season=2026, historical_week=2)
    written = sorted(
        p.relative_to(pl.ledger_root(estate["artifacts"]))
        for p in pl.ledger_root(estate["artifacts"]).rglob("*.json")
    )
    assert written, "expected the repair to write something"
    for path in written:
        assert path.parts[0] == "season=2026"
        assert path.parts[1] == "week=02"


def test_the_entry_point_contains_no_publication_path():
    """Structural: repairing a past week cannot disturb the current feed
    because this entry point cannot publish at all, in either mode."""
    import ast

    source = (REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    for forbidden in ("publish_wizard_nfl_local", "publish_2026_current_week", "write_latest",
                      "publish_current_week_feed"):
        assert not any(forbidden in n for n in names), forbidden


def test_the_sweep_orchestrator_never_passes_repair_flags():
    """The scheduled path stays current-card only; repair is operator-driven."""
    sweep = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    start = sweep.index("run_2026_stage_snapshots.py")
    invocation = sweep[start : sweep.index("sweep_exit=", start)]
    assert "--season" not in invocation
    assert "--week" not in invocation
    assert "--dry-run" not in invocation
