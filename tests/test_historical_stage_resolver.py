"""Proofs for historical stage resolution and the audited OPEN recovery.

THE DEFECT. ``discover_open_instant`` saw only the capture a sweep happened to
be holding, so "the first valid market observed" was computed as "the earliest
observation in THIS capture" -- always roughly now. OPEN therefore advanced
every sweep and collided with the immutable performance ledger. The same
single-capture input made a historical MID or CLOSE unreachable: a capture
observed today is correctly refused by ``observation_not_after_utc`` for a
cutoff that has already passed.

Hermetic: ``tmp_path`` archives of synthetic captures built by the real
capture hashing helpers, and the real resolver functions. No network, no
provider, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.production import snapshot_execution_2026 as ex
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl
from nfl_hybrid.production import snapshot_stages_2026 as st

from test_bdl_market_bridge import _odds_row, write_capture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS = ("draftkings", "fanduel", "caesars")
CANONICAL_GAME = "2026_01_HOU_KC"
KICKOFF = pd.Timestamp("2026-09-13T17:00:00Z")


def _load_recovery():
    spec = importlib.util.spec_from_file_location(
        "_recover_stage_open", REPO_ROOT / "scripts" / "recover_stage_open_snapshots.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_recover_stage_open"] = module
    spec.loader.exec_module(module)
    return module


recovery = _load_recovery()


def _archive_capture(data_root: Path, observed: str, *, books=BOOKS, season=2026, week=1) -> Path:
    """One archived STAGE capture, at the real production archive path.

    ``write_capture`` names its own ``capture=<fixed>`` directory, so the
    built capture is relocated to the stamped path the archive reader globs.
    """
    import shutil

    stamp = pd.Timestamp(observed).strftime("%Y%m%dT%H%M%SZ")
    staging = data_root / ".staging" / stamp
    manifest = write_capture(
        staging,
        horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=observed,
        odds_pages=[[_odds_row(1, book) for book in books]],
        odds_received=[observed],
        season=season,
        week=week,
    )
    final = (
        data_root
        / ex.STAGE_ARCHIVE_NAMESPACE
        / f"season={season}"
        / f"week={week:02d}"
        / f"horizon={st.STAGE_CAPTURE_HORIZON}"
        / f"capture={stamp}"
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(manifest.parent), str(final))
    return final / "manifest.json"


@pytest.fixture
def archive(tmp_path) -> Path:
    """Three observations: 01:24, 11:59, 22:57 -- the live Week-2 shape."""
    root = tmp_path / "data"
    for observed in ("2026-09-10T01:24:12Z", "2026-09-11T11:59:02Z", "2026-09-12T22:57:54Z"):
        _archive_capture(root, observed)
    return root


# ===========================================================================
# 1. OPEN uses accumulated history and stays deterministic.
# ===========================================================================
def test_the_archive_is_listed_chronologically(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    assert len(captures) == 3
    assert [c.observed_at_utc for c in captures] == sorted(c.observed_at_utc for c in captures)
    assert captures[0].observed_at_utc == pd.Timestamp("2026-09-10T01:24:12Z")


def test_open_resolves_to_the_earliest_archived_observation(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    accumulated, provenance = ex.accumulated_stage_quotes(captures)

    assert provenance["usable_capture_count"] == 3
    assert provenance["rejected_capture_count"] == 0
    assert ex.discover_open_instant(
        accumulated, game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF
    ) == pd.Timestamp("2026-09-10T01:24:12Z")


def test_open_does_not_move_when_a_later_capture_is_added(archive):
    """The regression: with the old single-capture resolver every sweep
    produced a later OPEN."""
    before = ex.discover_open_instant(
        ex.accumulated_stage_quotes(ex.archived_stage_captures(archive, season=2026, week=1))[0],
        game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF,
    )
    _archive_capture(archive, "2026-09-13T04:42:22Z")
    after = ex.discover_open_instant(
        ex.accumulated_stage_quotes(ex.archived_stage_captures(archive, season=2026, week=1))[0],
        game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF,
    )
    assert before == after == pd.Timestamp("2026-09-10T01:24:12Z")


def test_the_newest_capture_alone_would_have_given_a_later_open(archive):
    """Pins the defect itself, so the fix cannot silently regress."""
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    newest_only, _ = ex.accumulated_stage_quotes([captures[-1]])
    assert ex.discover_open_instant(
        newest_only, game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF
    ) == pd.Timestamp("2026-09-12T22:57:54Z")


def test_an_unusable_archived_capture_is_counted_not_fatal(archive):
    """Parses well enough to be listed, then fails certified validation."""
    import shutil
    staging = archive / ".staging" / "corrupt"
    manifest = write_capture(
        staging, horizon=bridge.STAGE_HORIZON, nominal_cutoff_utc="2026-09-10T02:00:00Z",
        odds_pages=[[_odds_row(1, b) for b in BOOKS]], odds_received=["2026-09-10T02:00:00Z"],
        season=2026, week=1, corrupt_manifest_hash=True,
    )
    final = (
        archive / ex.STAGE_ARCHIVE_NAMESPACE / "season=2026" / "week=01"
        / f"horizon={st.STAGE_CAPTURE_HORIZON}" / "capture=20260910T020000Z"
    )
    shutil.move(str(manifest.parent), str(final))
    captures = ex.archived_stage_captures(archive, season=2026, week=1)

    accumulated, provenance = ex.accumulated_stage_quotes(captures)
    assert provenance["rejected_capture_count"] == 1
    assert accumulated is not None
    assert ex.discover_open_instant(
        accumulated, game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF
    ) == pd.Timestamp("2026-09-10T01:24:12Z")


def test_an_empty_archive_yields_no_open(tmp_path):
    accumulated, provenance = ex.accumulated_stage_quotes([])
    assert accumulated is None
    assert provenance["usable_capture_count"] == 0


# ===========================================================================
# 2. MID / CLOSE select only evidence at or before their cutoff.
# ===========================================================================
def test_the_selected_capture_never_postdates_the_cutoff(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    cutoff = pd.Timestamp("2026-09-11T16:00:00Z")
    chosen = ex.newest_capture_at_or_before(captures, cutoff)
    assert chosen is not None
    assert chosen.observed_at_utc == pd.Timestamp("2026-09-11T11:59:02Z")
    assert chosen.observed_at_utc <= cutoff
    assert chosen.observed_at_utc != pd.Timestamp("2026-09-12T22:57:54Z")


def test_the_freshest_eligible_capture_is_chosen(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    chosen = ex.newest_capture_at_or_before(captures, pd.Timestamp("2026-09-12T23:59:00Z"))
    assert chosen.observed_at_utc == pd.Timestamp("2026-09-12T22:57:54Z")


def test_a_capture_exactly_at_the_cutoff_is_eligible(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    chosen = ex.newest_capture_at_or_before(captures, pd.Timestamp("2026-09-11T11:59:02Z"))
    assert chosen.observed_at_utc == pd.Timestamp("2026-09-11T11:59:02Z")


def test_no_lookahead_when_nothing_predates_the_cutoff(archive):
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    assert ex.newest_capture_at_or_before(captures, pd.Timestamp("2026-09-09T00:00:00Z")) is None


def test_the_selected_capture_passes_the_no_lookahead_bridge_rule(archive):
    """End to end: the chosen capture is one the certified bridge accepts for
    that cutoff, so observation_not_after_utc is satisfied by construction
    rather than relaxed."""
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    cutoff = pd.Timestamp("2026-09-12T00:00:00Z")
    chosen = ex.newest_capture_at_or_before(captures, cutoff)
    validated = bridge.validate_capture_manifest(
        chosen.manifest_path,
        expected_horizon=bridge.STAGE_HORIZON,
        allowed_horizons=bridge.STAGE_HORIZONS,
        observation_not_after_utc=cutoff,
    )
    assert validated.horizon == bridge.STAGE_HORIZON


def test_a_later_capture_is_still_refused_for_a_past_cutoff(archive):
    """The rule itself is untouched -- this is what made historical MID/CLOSE
    unreachable before, and it still fires when asked improperly."""
    captures = ex.archived_stage_captures(archive, season=2026, week=1)
    with pytest.raises(bridge.BdlMarketBridgeError, match="refusing to look ahead"):
        bridge.validate_capture_manifest(
            captures[-1].manifest_path,
            expected_horizon=bridge.STAGE_HORIZON,
            allowed_horizons=bridge.STAGE_HORIZONS,
            observation_not_after_utc=pd.Timestamp("2026-09-11T00:00:00Z"),
        )


# ===========================================================================
# 3. Cutoff definitions and the book floor are unchanged.
# ===========================================================================
def test_close_is_still_kickoff_minus_sixty(archive):
    assert KICKOFF - st.close_cutoff_utc(KICKOFF) == pd.Timedelta(minutes=60)


def test_the_thursday_midpoint_rule_is_unchanged():
    thu = pd.Timestamp("2026-09-18T00:15:00Z")
    open_at = pd.Timestamp("2026-09-14T14:00:00Z")
    schedule = st.resolve_game_snapshots(
        game_id="TNF", scheduled_kickoff_utc=thu,
        card_earliest_kickoff_utc=thu, open_observed_at_utc=open_at,
    )
    close = st.close_cutoff_utc(thu)
    assert schedule.mid_basis == st.MID_BASIS_OPEN_CLOSE_MIDPOINT
    assert schedule.mid_cutoff_utc == open_at + (close - open_at) / 2
    assert schedule.mid_cutoff_utc < close < thu


def test_the_three_book_floor_is_unchanged():
    assert rmr.MINIMUM_FRESH_COHERENT_BOOKS == 3


def test_a_two_book_archive_yields_no_open(tmp_path):
    """Accumulating history does not lower the bar."""
    root = tmp_path / "data"
    _archive_capture(root, "2026-09-10T01:24:12Z", books=("draftkings", "fanduel"))
    accumulated, _ = ex.accumulated_stage_quotes(
        ex.archived_stage_captures(root, season=2026, week=1)
    )
    assert ex.discover_open_instant(
        accumulated, game_id=CANONICAL_GAME, scheduled_kickoff_utc=KICKOFF
    ) is None


def test_open_is_never_claimed_at_or_after_kickoff(archive):
    _archive_capture(archive, "2026-09-10T01:24:12Z")
    accumulated, _ = ex.accumulated_stage_quotes(
        ex.archived_stage_captures(archive, season=2026, week=1)
    )
    early_kickoff = pd.Timestamp("2026-09-10T01:00:00Z")
    assert ex.discover_open_instant(
        accumulated, game_id=CANONICAL_GAME, scheduled_kickoff_utc=early_kickoff
    ) is None


# ===========================================================================
# 4. An already-recorded stage is complete, not rewritten.
# ===========================================================================
def _card() -> pd.DataFrame:
    return pd.DataFrame([{"game_id": CANONICAL_GAME, "scheduled_kickoff_utc": KICKOFF}])


def _snapshot(stage, *, at):
    return {
        "season": 2026, "week": 1, "game_id": CANONICAL_GAME, "snapshot_stage": stage,
        "snapshot_at_utc": str(at), "scheduled_kickoff_utc": str(KICKOFF),
        "model_home_margin": 3.0, "model_total": 44.0,
        "market_consensus_home_spread": -3.0, "market_consensus_total": 44.5,
        "market_book_quotes": [{"bookmaker_key": "draftkings", "line": -3.0}],
        "market_observed_at_utc": str(at), "model_tag": "t", "model_sha": "s",
        "production_git_commit": "a" * 40, "active_calibrator_source": "BASELINE",
        "active_calibrator_candidate_id": None, "active_calibrator_seed_sha256": "b" * 64,
        "capture_manifest_sha256": "c" * 64, "forecast_run_id": "r", "forecast_prediction_hash": "d" * 64,
    }


def test_an_already_recorded_stage_is_skipped(tmp_path):
    pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))
    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_OPEN, card=_card(), as_of_utc=KICKOFF,
        open_observations={CANONICAL_GAME: pd.Timestamp("2026-09-10T01:24:12Z")},
        operational_root=tmp_path, season=2026, week=1,
    )
    assert batches == []
    assert skipped == {CANONICAL_GAME: ex.SKIP_ALREADY_RECORDED}


def test_the_recorded_row_is_not_rewritten(tmp_path):
    pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))
    path = pl.snapshot_path(tmp_path, season=2026, week=1, game_id=CANONICAL_GAME, stage=st.STAGE_OPEN)
    before = path.read_bytes()

    ex.plan_stage_batches(
        stage=st.STAGE_OPEN, card=_card(), as_of_utc=KICKOFF,
        open_observations={CANONICAL_GAME: pd.Timestamp("2026-09-12T22:57:54Z")},
        operational_root=tmp_path, season=2026, week=1,
    )
    assert path.read_bytes() == before


def test_a_stage_not_yet_recorded_is_still_planned(tmp_path):
    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_OPEN, card=_card(), as_of_utc=KICKOFF,
        open_observations={CANONICAL_GAME: pd.Timestamp("2026-09-10T01:24:12Z")},
        operational_root=tmp_path, season=2026, week=1,
    )
    assert len(batches) == 1
    assert skipped == {}


def test_without_a_ledger_root_the_skip_is_not_applied(tmp_path):
    """Backward compatible: callers that pass no root behave as before."""
    pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))
    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_OPEN, card=_card(), as_of_utc=KICKOFF,
        open_observations={CANONICAL_GAME: pd.Timestamp("2026-09-10T01:24:12Z")},
    )
    assert len(batches) == 1
    assert ex.SKIP_ALREADY_RECORDED not in skipped.values()


def test_a_genuine_contradiction_still_raises(tmp_path):
    """The immutability guard stays active for what it is meant to catch."""
    pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))
    with pytest.raises(pl.SnapshotImmutabilityViolation, match="immutable"):
        pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, at="2026-09-12T22:57:54Z"))


def test_an_identical_re_record_is_still_an_idempotent_noop(tmp_path):
    snap = _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z")
    assert pl.record_snapshot(tmp_path, snap).status == "WRITTEN"
    assert pl.record_snapshot(tmp_path, snap).status == "IDEMPOTENT_NOOP"


# ===========================================================================
# 5. Week-2 style recovery: dry run identifies, apply preserves.
# ===========================================================================
@pytest.fixture
def recovery_estate(tmp_path, archive):
    """A ledger holding one LATE OPEN row, plus the archive that proves the
    true one -- the production shape."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pl.record_snapshot(artifacts, _snapshot(st.STAGE_OPEN, at="2026-09-12T22:57:54Z"))
    games = pd.DataFrame(
        [{
            "game_id": CANONICAL_GAME, "season": 2026, "week": 1, "season_type": "REG",
            "home_team_id": "KC", "away_team_id": "HOU", "scheduled_kickoff_utc": KICKOFF,
            "home_score": None, "away_score": None, "neutral_site": False,
        }]
    )
    return {"artifacts": artifacts, "data": archive, "games": games}


def _plan(estate):
    return recovery.plan(
        artifact_root=estate["artifacts"], data_root=estate["data"],
        season=2026, week=1, as_of_utc=KICKOFF, games=estate["games"],
    )


def test_the_dry_run_identifies_the_defective_open(recovery_estate):
    decision = _plan(recovery_estate)
    action = decision["actions"][0]
    assert action["status"] == recovery.STATUS_SUPERSEDE
    assert pd.Timestamp(action["recorded_open"]) == pd.Timestamp("2026-09-12T22:57:54Z")
    assert action["archive_open"] == "2026-09-10 01:24:12+00:00"
    assert decision["status_counts"][recovery.STATUS_SUPERSEDE] == 1


def test_the_dry_run_mutates_nothing(recovery_estate):
    before = {p: p.read_bytes() for p in sorted(recovery_estate["artifacts"].rglob("*.json"))}
    _plan(recovery_estate)
    assert {p: p.read_bytes() for p in sorted(recovery_estate["artifacts"].rglob("*.json"))} == before
    assert not (recovery_estate["artifacts"] / "invalidated").exists()


def test_a_correct_row_is_reported_correct_and_left_alone(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    for p in artifacts.rglob("*__OPEN.json"):
        p.unlink()
    pl.record_snapshot(artifacts, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))

    decision = _plan(recovery_estate)
    assert decision["actions"][0]["status"] == recovery.STATUS_CORRECT


def test_a_missing_row_is_reported_missing(recovery_estate):
    for p in recovery_estate["artifacts"].rglob("*__OPEN.json"):
        p.unlink()
    decision = _plan(recovery_estate)
    assert decision["actions"][0]["status"] == recovery.STATUS_MISSING_ROW


def test_quarantine_preserves_the_original_intact(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    original = pl.snapshot_path(
        artifacts, season=2026, week=1, game_id=CANONICAL_GAME, stage=st.STAGE_OPEN
    ).read_bytes()

    decision = _plan(recovery_estate)
    result = recovery.quarantine(decision, artifact_root=artifacts)

    preserved = Path(result["moved_to_invalidated"][0])
    assert preserved.read_bytes() == original
    assert preserved.parent.parent.name == "invalidated"
    assert preserved.parent.name == recovery.DEFAULT_DEFECT_SLUG


def test_quarantine_removes_the_defective_row_from_the_live_ledger(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    recovery.quarantine(_plan(recovery_estate), artifact_root=artifacts)
    assert pl.read_snapshot(
        artifacts, season=2026, week=1, game_id=CANONICAL_GAME, stage=st.STAGE_OPEN
    ) is None


def test_quarantine_records_the_supersession_provenance(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    recovery.quarantine(_plan(recovery_estate), artifact_root=artifacts)

    manifest = json.loads(
        (artifacts / "invalidated" / recovery.DEFAULT_DEFECT_SLUG / recovery.SUPERSESSION_MANIFEST)
        .read_text()
    )
    link = manifest["superseded"][CANONICAL_GAME]
    assert link[recovery.SUPERSEDES_KEY].startswith("invalidated/")
    assert pd.Timestamp(link["invalidated_open"]) == pd.Timestamp("2026-09-12T22:57:54Z")
    assert pd.Timestamp(link["archive_open"]) == pd.Timestamp("2026-09-10T01:24:12Z")
    assert "accumulated" in manifest["invalidation_reason"]


def test_the_corrected_open_is_then_writable(recovery_estate):
    """After quarantine the true OPEN can be recorded, which is what the next
    ordinary sweep does."""
    artifacts = recovery_estate["artifacts"]
    recovery.quarantine(_plan(recovery_estate), artifact_root=artifacts)
    result = pl.record_snapshot(artifacts, _snapshot(st.STAGE_OPEN, at="2026-09-10T01:24:12Z"))
    assert result.status == "WRITTEN"


def test_recovery_is_idempotent(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    first = recovery.quarantine(_plan(recovery_estate), artifact_root=artifacts)
    second = recovery.quarantine(_plan(recovery_estate), artifact_root=artifacts)
    assert first["moved_to_invalidated"]
    assert second["moved_to_invalidated"] == []
    assert Path(first["moved_to_invalidated"][0]).is_file()


def test_recovery_refuses_to_overwrite_preserved_evidence(recovery_estate):
    artifacts = recovery_estate["artifacts"]
    decision = _plan(recovery_estate)
    destination = Path(decision["invalidated_dir"])
    destination.mkdir(parents=True)
    (destination / f"{CANONICAL_GAME}__OPEN.json").write_text('{"different": true}')

    with pytest.raises(recovery.RecoveryRefused, match="refusing to overwrite preserved"):
        recovery.quarantine(decision, artifact_root=artifacts)


def test_recovery_refuses_when_the_archive_is_empty(recovery_estate, tmp_path):
    with pytest.raises(recovery.RecoveryRefused, match="no archived STAGE captures"):
        recovery.plan(
            artifact_root=recovery_estate["artifacts"], data_root=tmp_path / "empty",
            season=2026, week=1, as_of_utc=KICKOFF, games=recovery_estate["games"],
        )


def test_the_invalidated_tree_is_protected_from_pruning():
    spec = importlib.util.spec_from_file_location(
        "_pruner_check", REPO_ROOT / "scripts" / "prune_replaceable_artifacts.py"
    )
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)
    assert "invalidated" in pruner.PROTECTED_NAMESPACES
