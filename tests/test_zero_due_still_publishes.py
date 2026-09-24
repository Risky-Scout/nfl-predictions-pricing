"""Proofs that an idle sweep still publishes the current week.

THE DEFECT. PR #58 taught the publisher to emit a valid empty envelope for a
week that has not closed anything yet, so the page could advance off the
Week-1 card it had been serving since ``2026-09-15T05:14:43Z``. Run
``35896133421`` ran that exact code, succeeded, and changed nothing:

    due_batches=0
    snapshot_action=NOOP_NOTHING_DUE
    stage_snapshot_status=OK

and never printed ``=== 7. current-week publication ===``. The zero-due guard
in ``run_stage_snapshots.sh`` returned before the publication stage, so the
empty-envelope path was unreachable on precisely the sweeps that needed it.

THE RULE. "Nothing new to execute" and "nothing to publish" are different
facts. A zero-due sweep is still a no-op in the stage ledger -- no stage is
invented, no capture is taken -- but it now continues into the ordinary
publication stage and lets the publisher decide what the current week shows.

Hermetic: ``tmp_path`` estates and a stub interpreter for the shell path, plus
real calls to the publisher and the card resolver where those are the subject.
No network, no provider, no server, nothing published.
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
import yaml

from nfl_hybrid.production import snapshot_stages_2026 as st

from test_close_only_publication import (  # noqa: E402
    MONDAY,
    OPEN_AT,
    SUNDAY,
    _publish,
    _rollover_games,
    feed as exporter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


entry = _load(REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py", "_entry_zero_due")


# --- the shell harness -----------------------------------------------------
# One stub interpreter stands in for every entrypoint the sweep shells out to.
# The publisher branch records its argv and is configurable, because what the
# sweep must do depends entirely on what the publisher says.
STUB = """#!/usr/bin/env bash
set -euo pipefail
if [ "${1}" = "-c" ]; then exec @PY@ "$@"; fi
entry="${1}"
case "${entry}" in
*/refresh_bdl_2026_games_evidence.py) echo refreshed ;;
*/update_2026_games_population.py) echo population ;;
*/run_2026_stage_snapshots.py)
  echo '{"status": "OK", "stages": [{"stage": "OPEN", "due_batches": @DUE@, "executions": []}]}' ;;
*/generate_2026_recalibration_candidate.py) echo recalibration ;;
*/attach_2026_results_from_population.py) echo attached ;;
*/report_2026_snapshot_performance.py) echo reported ;;
*/publish_2026_current_week.py)
  printf '%s\\n' "$@" >> "@ARGV_LOG@"
  echo PUBLISHER_INVOKED
  out=""
  while [ $# -gt 0 ]; do
    if [ "${1}" = "--output" ]; then out="${2}"; shift 2; else shift; fi
  done
  if [ "@PUBLISH_EXIT@" -ne 0 ]; then
    echo "FAIL_CLOSED: @PUBLISH_FAILURE@" >&2
    exit @PUBLISH_EXIT@
  fi
  mkdir -p "$(dirname "${out}")"
  printf '%s\\n' '@PUBLISHED_BODY@' > "${out}"
  echo "publication_status=@PUBLICATION_STATUS@" ;;
*/publish_wizard_nfl_local.py) echo published ;;
*/prune_replaceable_artifacts.py) echo pruned ;;
*) echo "unexpected: ${entry}" >&2; exit 97 ;;
esac
"""

# What the page was stuck on.
STALE_WEEK_1 = (
    '{"schema_version": "wizard-nfl-pricing-v2", "season": 2026, "week": 1, '
    '"horizon": "TUE", "generated_at_utc": "2026-09-15T05:14:43Z", '
    '"games": [{"game_id": "2026_01_HOU_KC"}]}'
)
# What the publisher emits for a current week with nothing closed yet.
CURRENT_WEEK_EMPTY = (
    '{"schema_version": "wizard-nfl-pricing-v2", "season": 2026, "week": 3, '
    '"horizon": "TUE", "generated_at_utc": "2026-09-23T16:25:00Z", "games": []}'
)
# And once that week's games start closing.
CURRENT_WEEK_CUMULATIVE = (
    '{"schema_version": "wizard-nfl-pricing-v2", "season": 2026, "week": 3, '
    '"horizon": "TUE", "generated_at_utc": "2026-09-27T21:00:00Z", '
    '"games": [{"game_id": "2026_03_HOU_KC"}, {"game_id": "2026_03_BUF_NYJ"}]}'
)


def _estate(
    tmp_path: Path,
    *,
    due: int,
    existing_feed: str | None = STALE_WEEK_1,
    publish_exit: int = 0,
    publish_failure: str = "an already-closed game is never mutated",
    published_body: str = CURRENT_WEEK_EMPTY,
    publication_status: str = "OK_AWAITING_FIRST_CLOSE",
) -> dict:
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    feed_path = home / "artifacts" / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    if existing_feed is not None:
        feed_path.parent.mkdir(parents=True, exist_ok=True)
        feed_path.write_text(existing_feed + "\n", encoding="utf-8")

    argv_log = tmp_path / "publisher-argv.txt"
    stub = home / "venv" / "bin" / "python"
    body = STUB
    for token, value in (
        ("@PY@", sys.executable),
        ("@DUE@", str(due)),
        ("@ARGV_LOG@", str(argv_log)),
        ("@PUBLISH_EXIT@", str(publish_exit)),
        ("@PUBLISH_FAILURE@", publish_failure),
        ("@PUBLISHED_BODY@", published_body),
        ("@PUBLICATION_STATUS@", publication_status),
    ):
        body = body.replace(token, value)
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)

    return {"home": home, "feed": feed_path, "argv_log": argv_log}


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


def _publisher_argv(estate: dict) -> list[str]:
    if not estate["argv_log"].exists():
        return []
    return estate["argv_log"].read_text(encoding="utf-8").split()


def _served(estate: dict) -> dict:
    return json.loads(estate["feed"].read_text(encoding="utf-8"))


def _without_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


# ===========================================================================
# 1. Zero due no longer terminates the sweep before publication.
# ===========================================================================
def test_a_zero_due_sweep_reaches_the_publication_stage(tmp_path):
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "due_batches=0" in result.stdout
    assert "=== 7. current-week publication ===" in result.stdout
    assert "PUBLISHER_INVOKED" in result.stdout


def test_no_early_exit_stands_between_the_due_count_and_publication():
    """Against the CODE, not the comment explaining it. The recalibration
    failure exits in between are untouched; what must be gone is the clean
    `exit 0` that made publication unreachable."""
    text = SWEEP.read_text(encoding="utf-8")
    span = text[
        text.index('echo "due_batches=') : text.index("=== 7. current-week publication ===")
    ]
    assert "exit 0" not in _without_comments(span)


def test_the_sweep_still_runs_everything_that_precedes_publication(tmp_path):
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    for marker in (
        "games_evidence_refresh=OK",
        "games_population=OK",
        "stage_sweep=OK",
        "recalibration=OK",
        "season_reporting=OK",
    ):
        assert marker in result.stdout


# ===========================================================================
# 2. Zero due, a new publication week, nothing closed yet.
# ===========================================================================
def test_stage_execution_is_still_reported_as_a_no_op(tmp_path):
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    assert "stage_execution=NOOP_NOTHING_DUE" in result.stdout
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert "stage_snapshot_status=OK" in result.stdout
    assert "snapshot_action=EXECUTED" not in result.stdout


def test_the_public_payload_advances_to_the_current_week(tmp_path):
    """The whole point: the page stops serving Week 1."""
    estate = _estate(tmp_path, due=0)
    assert _served(estate)["week"] == 1

    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "publication_status=OK_AWAITING_FIRST_CLOSE" in result.stdout
    served = _served(estate)
    assert served["week"] == 3
    assert served["games"] == []


def test_the_sweep_reports_the_bytes_it_published(tmp_path):
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)
    assert "PUBLISHED_SHA256=" in result.stdout


def test_no_stage_is_fabricated_to_force_a_publication(tmp_path):
    """Publication happens because the publisher can describe the week, not
    because the sweep invented an execution for it to describe."""
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    assert '"due_batches": 0' in result.stdout
    assert "=== 3. market capture SKIPPED ===" in result.stdout
    argv = _publisher_argv(estate)
    assert "--market-capture-manifest" not in argv


def test_the_real_publisher_returns_awaiting_first_close_for_an_empty_week(tmp_path):
    """The shell proofs above stub the publisher; this pins that the status
    and shape they stub are the ones the real exporter produces."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    forecast_dir.mkdir(parents=True)
    estate = {
        "root": tmp_path,
        "card": pd.DataFrame(
            [
                {"game_id": "SUN", "scheduled_kickoff_utc": SUNDAY},
                {"game_id": "MON", "scheduled_kickoff_utc": MONDAY},
            ]
        ),
        "forecast_dir": forecast_dir,
        "observations": {"SUN": OPEN_AT, "MON": OPEN_AT},
        "output": tmp_path / "public" / "latest.json",
    }

    result = _publish(estate)

    assert result["status"] == "OK_AWAITING_FIRST_CLOSE"
    assert json.loads(estate["output"].read_text())["games"] == []


# ===========================================================================
# 3. Repeated idle sweeps stay safe and idempotent.
#
#    Reaching publication every 15 minutes must not turn into 96 new files a
#    day. There is one latest.json and it is rewritten in place.
# ===========================================================================
def test_repeated_zero_due_sweeps_all_succeed(tmp_path):
    estate = _estate(tmp_path, due=0)
    for _ in range(3):
        result = _run_sweep(estate)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout


def test_an_unchanged_current_week_stays_byte_identical(tmp_path):
    estate = _estate(tmp_path, due=0)
    _run_sweep(estate)
    published = estate["feed"].read_bytes()

    for _ in range(3):
        _run_sweep(estate)
        assert estate["feed"].read_bytes() == published


def test_repeated_zero_due_sweeps_create_no_extra_artifacts(tmp_path):
    estate = _estate(tmp_path, due=0)
    artifacts = estate["home"] / "artifacts"

    _run_sweep(estate)
    after_first = sorted(p.relative_to(artifacts) for p in artifacts.rglob("*"))
    for _ in range(3):
        _run_sweep(estate)

    assert sorted(p.relative_to(artifacts) for p in artifacts.rglob("*")) == after_first
    assert [p.name for p in estate["feed"].parent.iterdir()] == ["latest.json"]


# ===========================================================================
# 4. Zero due with CLOSE snapshots already recorded.
# ===========================================================================
def test_a_zero_due_sweep_publishes_the_cumulative_close_feed(tmp_path):
    """Between two CLOSE instants nothing is due, but the games that have
    already closed must keep being served."""
    estate = _estate(
        tmp_path,
        due=0,
        published_body=CURRENT_WEEK_CUMULATIVE,
        publication_status="OK",
    )
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert [game["game_id"] for game in _served(estate)["games"]] == [
        "2026_03_HOU_KC",
        "2026_03_BUF_NYJ",
    ]


# ===========================================================================
# 5. OPEN and MID are still never exposed.
# ===========================================================================
def test_publication_remains_close_only():
    assert exporter.PUBLICATION_PREFERENCE == (st.STAGE_CLOSE,)


def test_the_sweep_gives_the_publisher_no_way_to_select_a_stage(tmp_path):
    estate = _estate(tmp_path, due=0)
    _run_sweep(estate)

    argv = _publisher_argv(estate)
    assert argv, "the publisher was never invoked"
    assert "--stage" not in argv
    assert "OPEN" not in argv
    assert "MID" not in argv


def test_reaching_publication_did_not_add_a_market_capture_to_the_publisher():
    """Handing the publisher a live board is how OPEN/MID would leak into the
    public feed. The sweep never has, and still does not."""
    text = SWEEP.read_text(encoding="utf-8")
    call = text[
        text.index("publish_2026_current_week.py") : text.index("publish_exit=$?")
    ]
    assert "capture_args" not in call
    assert "market-capture-manifest" not in call


# ===========================================================================
# 6. Genuine corruption still fails closed.
# ===========================================================================
def test_a_failing_publisher_still_fails_a_zero_due_sweep_closed(tmp_path):
    estate = _estate(tmp_path, due=0, publish_exit=2)
    result = _run_sweep(estate)

    assert result.returncode == 10, result.stdout + result.stderr
    assert "FAIL CLOSED: current-week feed assembly failed (exit 2)" in result.stderr
    assert "stage_snapshot_status=OK" not in result.stdout


def test_a_failed_publication_leaves_the_previous_feed_untouched(tmp_path):
    estate = _estate(tmp_path, due=0, publish_exit=2)
    _run_sweep(estate)
    assert _served(estate)["week"] == 1


def test_the_real_publisher_still_raises_on_a_mutated_close(tmp_path):
    """The exit-2 the shell test rigs is a real failure mode, not a fiction."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    forecast_dir.mkdir(parents=True)
    estate = {
        "root": tmp_path,
        "card": pd.DataFrame([{"game_id": "SUN", "scheduled_kickoff_utc": SUNDAY}]),
        "forecast_dir": forecast_dir,
        "observations": {"SUN": OPEN_AT},
        "output": tmp_path / "public" / "latest.json",
    }

    from test_close_only_publication import _write

    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    _write(estate, "SUN", st.STAGE_CLOSE, margin=99.0)

    with pytest.raises(exporter.WizardExportError):
        _publish(estate, generated_at="2026-09-20T16:30:00Z")


# ===========================================================================
# 7. Historical repair still cannot publish the public feed.
# ===========================================================================
def test_the_sweep_never_runs_the_stage_entrypoint_in_repair_mode():
    text = SWEEP.read_text(encoding="utf-8")
    call = text[
        text.index("run_2026_stage_snapshots.py") : text.index("sweep_exit=$?")
    ]
    assert "--season" not in call
    assert "--week" not in call


def test_the_publisher_refuses_a_historical_week_override(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "publish_2026_current_week.py"),
            "--artifact-root", str(tmp_path),
            "--output", str(tmp_path / "latest.json"),
            "--season", "2026", "--week", "2",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --season 2026 --week 2" in result.stderr
    assert not (tmp_path / "latest.json").exists()


# ===========================================================================
# 8. The PR #57 rollover rule is untouched.
# ===========================================================================
def test_the_previous_week_stays_public_until_its_last_kickoff():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-21T05:50:00Z")
    )
    assert info["week"] == "2"


def test_the_handover_still_happens_and_now_actually_reaches_the_page():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-23T01:38:00Z")
    )
    assert info["week"] == "3"


# ===========================================================================
# 9. An ordinary executed sweep behaves exactly as before.
# ===========================================================================
def test_an_executed_sweep_publishes_and_reports_executed(tmp_path):
    estate = _estate(
        tmp_path, due=2, published_body=CURRENT_WEEK_CUMULATIVE, publication_status="OK"
    )
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=EXECUTED" in result.stdout
    assert "NOOP_NOTHING_DUE" not in result.stdout
    assert _served(estate)["week"] == 3


def test_an_executed_sweep_still_fails_closed_when_assembly_fails(tmp_path):
    estate = _estate(tmp_path, due=2, publish_exit=2)
    result = _run_sweep(estate)

    assert result.returncode == 10, result.stdout + result.stderr
    assert "FAIL CLOSED: current-week feed assembly failed (exit 2)" in result.stderr


def test_a_failed_stage_execution_never_reaches_publication(tmp_path):
    """The no-op relaxation must not relax the real gate: a due stage that
    fails still exits 4 before anything is published."""
    estate = _estate(tmp_path, due=2)
    stub = estate["home"] / "venv" / "bin" / "python"
    stub.write_text(
        stub.read_text(encoding="utf-8").replace(
            "*/run_2026_stage_snapshots.py)\n  echo",
            "*/run_2026_stage_snapshots.py)\n  exit 1 ;;\n*/never_run.py)\n  echo",
        ),
        encoding="utf-8",
    )
    result = _run_sweep(estate)

    assert result.returncode == 4, result.stdout + result.stderr
    assert _publisher_argv(estate) == []
    assert _served(estate)["week"] == 1


# ===========================================================================
# The workflow verifier follows the bytes, not the stage ledger.
# ===========================================================================
@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_the_verifier_runs_whenever_bytes_were_published(workflow):
    steps = workflow["jobs"]["snapshot-sweep"]["steps"]
    verify = next(s for s in steps if "verify_public_nfl_feed.py" in str(s.get("run", "")))
    assert "published_sha256 != ''" in verify["if"]
    assert "snapshot_action" not in verify["if"]


def test_the_verifier_is_still_skipped_when_nothing_was_published(workflow):
    """An empty published_sha256 -- the sweep failed before publication, or
    was a dry run -- must not send the verifier at a stale public URL.

    The publish half of the condition now reads the resolve job's single
    resolved mode rather than the raw dispatch input, because on a scheduled
    firing that input does not exist; see
    tests/test_scheduled_publication_mode.py."""
    steps = workflow["jobs"]["snapshot-sweep"]["steps"]
    verify = next(s for s in steps if "verify_public_nfl_feed.py" in str(s.get("run", "")))
    assert "needs.resolve.outputs.publish_mode == 'publish'" in verify["if"]
