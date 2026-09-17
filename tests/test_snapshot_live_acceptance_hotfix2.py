"""Regression proofs for the two defects live run 35170192215 exposed.

Both are cases the previous hotfix's tests were too weak to catch, so each
section below states plainly what the old test missed.

Hermetic: tmp_path estates, synthetic captures built by the real capture
hashing helpers, and a stub interpreter for the shell path. No network, no
server, no provider.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.production import snapshot_stages_2026 as st

from test_bdl_market_bridge import CUTOFF, write_capture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage_entry = _load(REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py", "_stage_entry_hotfix2")


# ===========================================================================
# 1. Zero due means no-op, regardless of any old artifact-side latest.json.
#
#    WHAT THE OLD TEST MISSED: it built an estate with NO latest.json, so the
#    guard's second condition was vacuously true and the bug was invisible.
#    The live estate carries one from a previous publication, so the guard
#    never fired and every idle sweep drove into assembly and exit 10.
# ===========================================================================
STUB = """#!/usr/bin/env bash
set -euo pipefail
if [ "${1}" = "-c" ]; then exec @PY@ "$@"; fi
case "${1}" in
*/refresh_bdl_2026_games_evidence.py) echo refreshed ;;
*/update_2026_games_population.py) echo population ;;
*/run_2026_stage_snapshots.py)
  echo '{"status": "OK", "stages": [{"stage": "OPEN", "due_batches": @DUE@, "executions": []}]}' ;;
*/generate_2026_recalibration_candidate.py) echo recalibration ;;
*/attach_2026_results_from_population.py) echo attached ;;
*/report_2026_snapshot_performance.py) echo reported ;;
*/publish_2026_current_week.py)
  # The live failure mode: with nothing executed, assembly correctly refuses.
  # If the sweep ever invokes this on a zero-due firing, the test fails.
  echo "FEED_ASSEMBLER_WAS_INVOKED" >&2
  echo 'FAIL_CLOSED: no game in the current week has a publishable snapshot yet' >&2
  exit 2 ;;
*/publish_wizard_nfl_local.py) echo published ;;
*/prune_replaceable_artifacts.py) echo pruned ;;
*) echo "unexpected: ${1}" >&2; exit 97 ;;
esac
"""

EXISTING_FEED = b'{"schema_version": "wizard-nfl-pricing-v2", "games": []}\n'


def _estate(tmp_path: Path, *, due: int, with_existing_feed: bool) -> dict:
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    feed_path = home / "artifacts" / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    if with_existing_feed:
        feed_path.parent.mkdir(parents=True, exist_ok=True)
        feed_path.write_bytes(EXISTING_FEED)

    stub = home / "venv" / "bin" / "python"
    stub.write_text(STUB.replace("@PY@", sys.executable).replace("@DUE@", str(due)), encoding="utf-8")
    stub.chmod(0o755)

    return {"home": home, "feed": feed_path}


def _run_sweep(estate: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SWEEP), "--stage", "ALL", "--skip-capture", "--dry-run-publish", *extra],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "WIZARD_NFL_HOME": str(estate["home"]),
            "WIZARD_NFL_WEB_DIR": str(estate["home"].parent / "web"),
        },
    )


def test_zero_due_no_ops_even_when_an_old_feed_exists(tmp_path):
    """The exact live condition. The assembler is rigged to fail if called."""
    estate = _estate(tmp_path, due=0, with_existing_feed=True)
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert "stage_snapshot_status=OK" in result.stdout


def test_the_feed_assembler_is_never_invoked_on_a_zero_due_sweep(tmp_path):
    estate = _estate(tmp_path, due=0, with_existing_feed=True)
    result = _run_sweep(estate)
    assert "FEED_ASSEMBLER_WAS_INVOKED" not in (result.stdout + result.stderr)
    assert "=== 7. current-week publication ===" not in result.stdout


def test_the_existing_feed_is_left_byte_identical(tmp_path):
    estate = _estate(tmp_path, due=0, with_existing_feed=True)
    _run_sweep(estate)
    assert estate["feed"].read_bytes() == EXISTING_FEED


def test_zero_due_still_no_ops_when_no_feed_exists(tmp_path):
    """The case the old test covered stays covered."""
    estate = _estate(tmp_path, due=0, with_existing_feed=False)
    result = _run_sweep(estate)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert not estate["feed"].exists()


def test_the_no_op_still_completes_the_work_that_precedes_it(tmp_path):
    """Evidence refresh, population update and recalibration are not skipped
    -- only assembly and publication are."""
    estate = _estate(tmp_path, due=0, with_existing_feed=True)
    result = _run_sweep(estate)
    assert "games_evidence_refresh=OK" in result.stdout
    assert "games_population=OK" in result.stdout
    assert "recalibration=OK" in result.stdout


def test_a_real_executed_batch_still_proceeds_to_publication(tmp_path):
    """The no-op must not swallow a genuine run. With a due batch the sweep
    reaches assembly -- and here the rigged assembler makes it fail closed,
    which is exactly the behaviour that must survive."""
    estate = _estate(tmp_path, due=2, with_existing_feed=True)
    result = _run_sweep(estate)

    assert result.returncode == 10, result.stdout + result.stderr
    assert "FEED_ASSEMBLER_WAS_INVOKED" in result.stderr
    assert "snapshot_action=NOOP_NOTHING_DUE" not in result.stdout


def test_the_guard_reads_only_the_due_count(tmp_path):
    """Nothing about the filesystem can change a zero-due verdict. Checked
    against the CODE, not the comments explaining why."""
    text = SWEEP.read_text(encoding="utf-8")
    guard = text[text.index('if [ "${due_batches}" -eq 0 ]'):]
    guard = guard[: guard.index("\nfi")]
    code = "\n".join(line for line in guard.splitlines() if not line.strip().startswith("#"))
    assert "latest.json" not in code
    assert "-f " not in code
    assert "-e " not in code


# ===========================================================================
# 2. A STAGE capture must actually be loadable by OPEN discovery.
#
#    WHAT THE OLD TESTS MISSED: they proved validate_capture_manifest accepts
#    a STAGE capture when asked with STAGE semantics. Nothing proved the
#    DISCOVERY path asked with those semantics -- and it did not, so a
#    COMPLETE capture with 125 observations produced open_observations = {}
#    and all 16 games UNRESOLVED_STAGE.
# ===========================================================================
STAGE_OBSERVED_AT = "2026-09-08T15:30:00Z"
SWEEP_AS_OF = "2026-09-08T15:45:00Z"


@pytest.fixture
def stage_capture(tmp_path) -> Path:
    return write_capture(
        tmp_path / "capture", horizon=bridge.STAGE_HORIZON, nominal_cutoff_utc=STAGE_OBSERVED_AT
    )


# The CANONICAL game_id the bridge derives, not the raw provider id.
CANONICAL_GAME_ID = "2026_01_HOU_KC"


@pytest.fixture
def card() -> pd.DataFrame:
    """The capture's own game, kicking off well after the observation."""
    return pd.DataFrame(
        [{"game_id": CANONICAL_GAME_ID, "scheduled_kickoff_utc": pd.Timestamp("2026-09-08T23:00:00Z")}]
    )


def test_a_stage_capture_yields_non_empty_quotes(tmp_path, stage_capture):
    """The regression. Through the DISCOVERY load path, not the batch path."""
    quotes = stage_entry.load_stage_quotes(
        stage_capture, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    )
    assert quotes is not None
    assert not quotes.empty
    assert set(quotes["bookmaker_key"]) == {"draftkings", "fanduel", "betmgm"}


def test_open_discovery_returns_real_observations_not_an_empty_dict(tmp_path, stage_capture, card):
    quotes = stage_entry.load_stage_quotes(
        stage_capture, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    )
    observations = stage_entry.discover_open_observations(card, quotes=quotes)
    assert observations != {}
    assert set(observations) == {CANONICAL_GAME_ID}
    assert observations[CANONICAL_GAME_ID] == pd.Timestamp(STAGE_OBSERVED_AT)


def test_a_resolved_open_produces_a_due_batch(tmp_path, stage_capture, card):
    """End of the chain: a discoverable board becomes an executable stage."""
    from nfl_hybrid.production import snapshot_execution_2026 as ex

    quotes = stage_entry.load_stage_quotes(
        stage_capture, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    )
    observations = stage_entry.discover_open_observations(card, quotes=quotes)

    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_OPEN, card=card, as_of_utc=SWEEP_AS_OF, open_observations=observations
    )
    assert len(batches) == 1
    assert batches[0].game_ids == (CANONICAL_GAME_ID,)
    assert skipped == {}


def test_without_the_stage_semantics_the_same_capture_is_rejected(tmp_path, stage_capture):
    """Pins the root cause: the DEFAULT contract refuses it, which is why
    discovery must ask explicitly rather than rely on defaults."""
    from nfl_hybrid.production import run_2026 as prod

    evidence = prod.evaluate_live_market_source(
        stage_capture, artifact_root_path=tmp_path / "artifacts"
    )
    assert evidence["registered"] is False
    assert evidence["status"] == prod.LIVE_MARKET_INVALID


def test_a_stage_capture_is_not_accepted_by_a_certified_caller(tmp_path, stage_capture):
    """A stage observation can never be priced as a certified TUE/FRI card."""
    with pytest.raises(bridge.BdlMarketBridgeError):
        bridge.validate_capture_manifest(stage_capture, expected_horizon="TUE")
    with pytest.raises(bridge.BdlMarketBridgeError):
        bridge.validate_capture_manifest(stage_capture)  # default PRODUCTION_HORIZONS


def test_a_certified_capture_is_unaffected_by_the_discovery_change(tmp_path):
    """And the certified path still validates exactly as before."""
    manifest = write_capture(tmp_path / "certified", horizon="TUE", nominal_cutoff_utc=CUTOFF)
    capture = bridge.validate_capture_manifest(
        manifest, expected_horizon="TUE", expected_target_cutoff_utc=CUTOFF
    )
    assert capture.horizon == "TUE"


def test_a_smoke_capture_is_still_refused_by_stage_discovery(tmp_path):
    manifest = write_capture(tmp_path / "smoke", horizon="SMOKE", nominal_cutoff_utc=STAGE_OBSERVED_AT)
    assert stage_entry.load_stage_quotes(
        manifest, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    ) is None


def test_an_incomplete_stage_capture_is_refused(tmp_path):
    manifest = write_capture(
        tmp_path / "incomplete", horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=STAGE_OBSERVED_AT, status="INCOMPLETE",
    )
    assert stage_entry.load_stage_quotes(
        manifest, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    ) is None


def test_a_corrupt_stage_manifest_is_refused(tmp_path):
    manifest = write_capture(
        tmp_path / "corrupt", horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=STAGE_OBSERVED_AT, corrupt_manifest_hash=True,
    )
    assert stage_entry.load_stage_quotes(
        manifest, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    ) is None


def test_an_observation_taken_after_the_sweep_instant_is_refused(tmp_path):
    """Matters for a replay with an explicit past --as-of: a capture from
    later must not inform an earlier instant."""
    manifest = write_capture(
        tmp_path / "future", horizon=bridge.STAGE_HORIZON, nominal_cutoff_utc="2026-09-08T16:30:00Z"
    )
    assert stage_entry.load_stage_quotes(
        manifest, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    ) is None


def test_open_is_not_fabricated_when_no_priceable_market_exists(tmp_path, card):
    """Too few books for a certified consensus -- no OPEN, not a guessed one."""
    manifest = write_capture(
        tmp_path / "thin",
        horizon=bridge.STAGE_HORIZON,
        nominal_cutoff_utc=STAGE_OBSERVED_AT,
        odds_pages=[[__import__("test_bdl_market_bridge")._odds_row(1, "draftkings")]],
    )
    quotes = stage_entry.load_stage_quotes(
        manifest, not_after_utc=SWEEP_AS_OF, artifact_root_path=tmp_path / "artifacts"
    )
    # Either the capture is refused outright, or it loads but yields no OPEN.
    observations = {} if quotes is None else stage_entry.discover_open_observations(card, quotes=quotes)
    assert observations == {}


def test_discovery_and_the_batch_path_share_one_stage_contract():
    """Stated once, not twice: the publisher reuses the sweep's loader rather
    than restating the expectations."""
    publisher = (REPO_ROOT / "scripts" / "publish_2026_current_week.py").read_text(encoding="utf-8")
    assert "_stage.load_stage_quotes" in publisher
    assert "evaluate_live_market_source" not in publisher

    sweep_entry = (REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py").read_text(encoding="utf-8")
    assert sweep_entry.count("bridge.STAGE_HORIZONS") == 1
