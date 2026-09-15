"""Proofs that ONE as-of instant governs a certified card run end to end.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
``ops/wizard/run_certified_card.sh`` built its ``--as-of`` arguments in stage
3 and forwarded them only to the card. Stage 2's preflight was therefore
always asked about "now". Preflight cross-checks the supplied capture against
the certified cutoff it resolves from the instant it is given, so a Week-1
replay was judged against the CURRENT card's cutoff
(``2026-09-15T16:00:00Z``) while the preserved capture correctly carried
``2026-09-08T16:00:00Z``. The cutoffs disagreed, capture validation failed,
and a perfectly good capture was reported as
``live_2026_market_source_unregistered`` -- a blocker naming a missing market
source when the real fault was that the two stages were asked about two
different weeks.

Everything here is HERMETIC: ``tmp_path`` estates, synthetic manifests built
by the real capture hashing helpers, and a stub interpreter that only records
how it was invoked. No network, no real capture, no SSH, nothing published.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.production import run_2026 as prod

# Reuse the bridge suite's synthetic capture builder rather than writing a
# second one: it produces a COMPLETE capture whose manifest and per-page
# hashes are computed the way the real capture script computes them, so the
# bridge's own validation is exercised for real. Its default cutoff already
# IS the Week-1 TUE cutoff.
from test_bdl_market_bridge import CUTOFF, write_capture  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFIED_SCRIPT = REPO_ROOT / "ops" / "wizard" / "run_certified_card.sh"
CARD_ENTRYPOINT = REPO_ROOT / "scripts" / "run_2026_production_card.py"

# The exact Week-1 TUE replay under way, and the cutoff the preserved
# certified capture carries.
WEEK1_AS_OF = "2026-09-08T16:05:00Z"
WEEK1_CUTOFF = pd.Timestamp(CUTOFF)
# What "now" resolved to instead while preflight was never told the as-of.
CURRENT_CARD_CUTOFF = pd.Timestamp("2026-09-15T16:00:00Z")


def _card_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_run_2026_production_card", CARD_ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_2026_production_card"] = module
    spec.loader.exec_module(module)
    return module


card = _card_module()


# ===========================================================================
# PART 1 -- the entrypoint resolves the cutoff from the as-of it is given.
#
# These call the REAL cmd_preflight and capture the cutoff it hands to
# run_preflight, so they prove the value the orchestrator's argument actually
# controls rather than restating the shell script's text.
# ===========================================================================
def _preflight_cutoff(monkeypatch, as_of: str | None, horizon: str = "TUE") -> pd.Timestamp | None:
    seen: dict = {}

    def _fake_run_preflight(**kwargs):
        seen.update(kwargs)
        return {"overall_status": "READY", "production_run_ready": True}

    monkeypatch.setattr(prod, "run_preflight", _fake_run_preflight)
    args = argparse.Namespace(
        horizon=horizon, as_of=as_of, market_capture_manifest="/nonexistent/manifest.json"
    )
    card.cmd_preflight(args)
    return seen["expected_target_cutoff_utc"]


def test_the_week1_replay_as_of_resolves_the_week1_cutoff(monkeypatch):
    """The whole point of the fix: given the replay instant, preflight judges
    the capture against Week-1's cutoff, not this week's."""
    assert _preflight_cutoff(monkeypatch, WEEK1_AS_OF) == WEEK1_CUTOFF


def test_an_omitted_as_of_keeps_the_existing_current_card_behaviour(monkeypatch):
    """No as-of means "now", exactly as before -- the scheduled TUE/FRI runs
    are unchanged by this fix."""
    monkeypatch.setattr(prod, "utc_now", lambda: prod._as_utc("2026-09-15T03:00:00Z"))
    assert _preflight_cutoff(monkeypatch, None) == CURRENT_CARD_CUTOFF


def test_the_cutoff_is_horizon_aware_not_hardcoded(monkeypatch):
    """Cutoff semantics are untouched: the certified resolver still decides,
    and it still answers differently for the two horizons."""
    tue = _preflight_cutoff(monkeypatch, WEEK1_AS_OF, horizon="TUE")
    fri = _preflight_cutoff(monkeypatch, WEEK1_AS_OF, horizon="FRI")
    assert tue == WEEK1_CUTOFF
    assert fri != tue
    assert tue == prod.current_or_recent_cutoff(prod._as_utc(WEEK1_AS_OF), "TUE")
    assert fri == prod.current_or_recent_cutoff(prod._as_utc(WEEK1_AS_OF), "FRI")


# ===========================================================================
# PART 2 -- a capture judged against the wrong cutoff STILL fails closed.
#
# The fix must make the correct replay pass, without making a genuinely
# mismatched capture pass.
# ===========================================================================
def _week1_capture(directory: Path) -> Path:
    return write_capture(directory)


def test_the_week1_capture_validates_against_the_week1_cutoff(tmp_path):
    bridge.validate_capture_manifest(
        _week1_capture(tmp_path / "capture"),
        expected_season=2026,
        expected_week=1,
        expected_horizon="TUE",
        expected_target_cutoff_utc=WEEK1_CUTOFF,
    )


def test_the_same_capture_fails_closed_against_this_weeks_cutoff(tmp_path):
    """This is the exact failure the two dry-runs hit, reproduced: the capture
    is sound, the cutoff it is judged against is the wrong week."""
    with pytest.raises(Exception) as excinfo:
        bridge.validate_capture_manifest(
            _week1_capture(tmp_path / "capture"),
            expected_season=2026,
            expected_week=1,
            expected_horizon="TUE",
            expected_target_cutoff_utc=CURRENT_CARD_CUTOFF,
        )
    assert "cutoff" in str(excinfo.value).lower()


def test_a_cutoff_mismatch_surfaces_as_the_unregistered_market_blocker(tmp_path):
    """And it surfaces through preflight's live-market evidence exactly the
    way the failed runs reported it."""
    evidence = prod.evaluate_live_market_source(
        _week1_capture(tmp_path / "capture"),
        expected_season=2026,
        expected_week=1,
        expected_horizon="TUE",
        expected_target_cutoff_utc=CURRENT_CARD_CUTOFF,
        artifact_root_path=tmp_path / "artifacts",
    )
    assert evidence["registered"] is False
    assert evidence["status"] == prod.LIVE_MARKET_INVALID


# ===========================================================================
# PART 3 -- the orchestrator asks BOTH stages about the same instant.
#
# The script is EXECUTED against a synthetic estate whose interpreter is a
# stub that records its argv, so these assert observed behaviour rather than
# grepping the source for a string that could drift.
# ===========================================================================
STUB_PY = r"""#!/usr/bin/env bash
# Records one file per invocation, one argument per line, then answers with
# the minimum each stage's caller parses out of it.
set -euo pipefail
n=$(find "${STUB_INVOCATION_DIR}" -name '*.args' | wc -l | tr -d ' ')
printf '%s\n' "$@" > "${STUB_INVOCATION_DIR}/$(printf '%03d' "${n}").args"

if [ "${1}" = "-c" ]; then exec @REAL_PYTHON@ "$@"; fi

script="${1}"; shift
case "${script}" in
*/verify_capture_manifest_integrity.py)
    echo "capture_manifest_integrity=OK" ;;
*/run_2026_production_card.py)
    if printf '%s\n' "$@" | grep -qx -- '--preflight'; then
        echo '{"overall_status": "READY", "production_run_ready": true}'
    else
        echo '{"status": "SUCCESS", "run_id": "stub-run-0001"}'
    fi ;;
*/export_wizard_nfl_pricing.py)
    out=""
    while [ $# -gt 0 ]; do
        if [ "${1}" = "--output" ]; then out="${2}"; fi
        shift
    done
    mkdir -p "$(dirname "${out}")"
    echo '{}' > "${out}" ;;
*/publish_wizard_nfl_local.py)
    echo "published=DRY_RUN" ;;
*)
    echo "stub interpreter got an unexpected script: ${script}" >&2; exit 97 ;;
esac
"""


@pytest.fixture
def estate(tmp_path) -> dict:
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state", "staging"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    invocations = tmp_path / "invocations"
    invocations.mkdir()

    stub = home / "venv" / "bin" / "python"
    stub.write_text(STUB_PY.replace("@REAL_PYTHON@", sys.executable), encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = {
        **os.environ,
        "WIZARD_NFL_HOME": str(home),
        "WIZARD_NFL_WEB_DIR": str(tmp_path / "web"),
        "STUB_INVOCATION_DIR": str(invocations),
    }
    return {
        "env": env,
        "invocations": invocations,
        "manifest": _week1_capture(tmp_path / "capture"),
    }


def _run_orchestrator(estate: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            str(CERTIFIED_SCRIPT),
            "--horizon",
            "TUE",
            "--capture-manifest",
            str(estate["manifest"]),
            "--dry-run-publish",
            *extra,
        ],
        capture_output=True,
        text=True,
        env=estate["env"],
    )


def _card_invocations(estate: dict) -> list[list[str]]:
    """Every invocation of the production card entrypoint, in order."""
    found = []
    for path in sorted(estate["invocations"].glob("*.args")):
        argv = path.read_text(encoding="utf-8").splitlines()
        if argv and argv[0].endswith("run_2026_production_card.py"):
            found.append(argv)
    return found


def _as_of_of(argv: list[str]) -> str | None:
    return argv[argv.index("--as-of") + 1] if "--as-of" in argv else None


def test_the_as_of_reaches_preflight(estate):
    result = _run_orchestrator(estate, "--as-of", WEEK1_AS_OF)
    assert result.returncode == 0, result.stdout + result.stderr

    preflight = [argv for argv in _card_invocations(estate) if "--preflight" in argv]
    assert len(preflight) == 1
    assert _as_of_of(preflight[0]) == WEEK1_AS_OF


def test_preflight_and_the_card_receive_the_same_as_of(estate):
    """The defect was not a missing flag so much as two stages disagreeing.
    Both must be asked about one instant."""
    result = _run_orchestrator(estate, "--as-of", WEEK1_AS_OF)
    assert result.returncode == 0, result.stdout + result.stderr

    invocations = _card_invocations(estate)
    assert len(invocations) == 2, invocations
    preflight, run = invocations
    assert "--preflight" in preflight
    assert "--preflight" not in run
    assert _as_of_of(preflight) == _as_of_of(run) == WEEK1_AS_OF


def test_preflight_runs_before_the_card_and_is_never_bypassed(estate):
    result = _run_orchestrator(estate, "--as-of", WEEK1_AS_OF)
    assert result.returncode == 0, result.stdout + result.stderr

    invocations = _card_invocations(estate)
    assert "--preflight" in invocations[0]
    assert "preflight_overall_status=READY" in result.stdout


def test_an_omitted_as_of_passes_the_flag_to_neither_stage(estate):
    """Unchanged scheduled behaviour: with no as-of, neither stage is pinned
    and both resolve "now" for themselves."""
    result = _run_orchestrator(estate)
    assert result.returncode == 0, result.stdout + result.stderr

    invocations = _card_invocations(estate)
    assert len(invocations) == 2
    assert all("--as-of" not in argv for argv in invocations)


def test_both_stages_still_receive_the_same_single_capture(estate):
    """The as-of change must not disturb the other thing both stages must
    agree about."""
    _run_orchestrator(estate, "--as-of", WEEK1_AS_OF)
    manifests = {
        argv[argv.index("--market-capture-manifest") + 1] for argv in _card_invocations(estate)
    }
    assert manifests == {str(estate["manifest"])}


def test_a_not_ready_preflight_still_blocks_the_card(estate):
    """Fail-closed ordering is untouched: nothing runs, exports or publishes
    after a preflight that is not READY."""
    stub = Path(estate["env"]["WIZARD_NFL_HOME"]) / "venv" / "bin" / "python"
    stub.write_text(
        stub.read_text(encoding="utf-8").replace(
            '{"overall_status": "READY", "production_run_ready": true}',
            '{"overall_status": "BLOCKED_ON_LIVE_INPUTS", "production_run_ready": false}',
        ),
        encoding="utf-8",
    )

    result = _run_orchestrator(estate, "--as-of", WEEK1_AS_OF)
    assert result.returncode == 4
    assert "FAIL CLOSED" in result.stdout + result.stderr
    assert len(_card_invocations(estate)) == 1
    assert "certified_card_status=OK" not in result.stdout


def test_the_as_of_is_constructed_once_and_reused(estate):
    """A second definition would be a second chance to drift apart."""
    text = CERTIFIED_SCRIPT.read_text(encoding="utf-8")
    assert text.count("as_of_args=()") == 1
    assert text.count('"${as_of_args[@]}"') == 2
    assert text.index("as_of_args=()") < text.index("--preflight")
