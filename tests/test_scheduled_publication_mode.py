"""One publish decision, and a scheduled run makes it the publishing one.

THE DEFECT. ``inputs`` exists only for ``workflow_dispatch``, so on a
scheduled firing ``inputs.publish`` is empty. Two jobs read that emptiness in
opposite directions: the certified job took it to mean ``publish``, and the
snapshot sweep took it to mean ``dry_run``::

    PUBLISH_MODE: ${{ inputs.publish }}
    [ "${PUBLISH_MODE:-dry_run}" != "publish" ] && args="... --dry-run-publish"

    10|Scheduled sweeps therefore always ran ``--dry-run-publish`` and the
public-bytes verifier, gated on ``inputs.publish == 'publish'``, never ran at
all. Run ``36039577444`` is the proof: on the PR #60 merge commit it reached
stage 7, the assembler returned ``OK_AWAITING_FIRST_CLOSE`` for 2026 week 3,
the deployment validator accepted the empty card -- and reported::

    "status": "DRY_RUN"

so the served ``latest.json`` stayed on the Week-1 card from 2026-09-15. The
scheduled cadence is the only thing that runs unattended, and it could not
    20|publish.

THE RULE. The effective publish mode is resolved once, in the resolve job,
and every downstream job reads that one output. An absent input means a
schedule, and a schedule publishes. A ``workflow_dispatch`` always carries
the choice and is still obeyed, dry run included.

Hermetic: the workflow's own shell is extracted and executed, the real ops
sweep runs against a ``tmp_path`` estate with a stub interpreter for the
provider-facing entrypoints, and the real deployment and verification code
    30|runs unchanged. No network, no provider, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from nfl_hybrid.production import snapshot_stages_2026 as st

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"
SWEEP = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load(REPO_ROOT / "scripts" / "verify_public_nfl_feed.py", "_mode_verifier")
exporter = _load(REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py", "_mode_exporter")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step(workflow: dict, job: str, step_id: str) -> dict:
    return next(s for s in workflow["jobs"][job]["steps"] if s.get("id") == step_id)


def _step_named(workflow: dict, job: str, needle: str) -> dict:
    return next(s for s in workflow["jobs"][job]["steps"] if needle in str(s.get("run", "")))


# ===========================================================================
# 1-3, 6. The single decision, executed rather than read.
# ===========================================================================
def _run_gate(workflow: dict, tmp_path: Path, *, publish_input: str | None, ref: str = "refs/heads/main") -> dict:
    """Execute the resolve job's gate step exactly as Actions would, and
    return the GITHUB_OUTPUT it produced."""
    script = tmp_path / "gate.sh"
    script.write_text(_step(workflow, "resolve", "gate")["run"], encoding="utf-8")
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")

    env = {
        "PATH": os.environ["PATH"],
        "MODE": "",
        "DAILY_DUE": "false",
        "CERTIFIED_DUE": "false",
        "GITHUB_REF": ref,
        "GITHUB_OUTPUT": str(output),
    }
    # A scheduled firing has no inputs context at all, so the variable is
    # genuinely absent rather than set to "".
    if publish_input is not None:
        env["PUBLISH_INPUT"] = publish_input

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    return {
        "returncode": result.returncode,
        "stderr": result.stderr,
        "outputs": dict(
            line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
        ),
    }


def test_a_scheduled_firing_resolves_to_publish(workflow, tmp_path):
    """THE regression. An absent input is a schedule, and a schedule
    publishes."""
    gate = _run_gate(workflow, tmp_path, publish_input=None)

    assert gate["returncode"] == 0, gate["stderr"]
    assert gate["outputs"]["publish_mode"] == "publish"


def test_a_manual_publish_resolves_to_publish(workflow, tmp_path):
    gate = _run_gate(workflow, tmp_path, publish_input="publish")

    assert gate["returncode"] == 0, gate["stderr"]
    assert gate["outputs"]["publish_mode"] == "publish"


def test_a_manual_dry_run_resolves_to_dry_run(workflow, tmp_path):
    """The manual rehearsal is still a rehearsal."""
    gate = _run_gate(workflow, tmp_path, publish_input="dry_run")

    assert gate["returncode"] == 0, gate["stderr"]
    assert gate["outputs"]["publish_mode"] == "dry_run"


def test_the_schedule_does_not_depend_on_the_input_existing(workflow, tmp_path):
    """Unset and empty are the same firing -- a schedule -- and both publish.
    The old code distinguished them only by accident of shell defaulting."""
    unset = _run_gate(workflow, tmp_path, publish_input=None)
    empty = _run_gate(workflow, tmp_path, publish_input="")

    assert unset["outputs"]["publish_mode"] == empty["outputs"]["publish_mode"] == "publish"


def test_an_unrecognised_publish_input_fails_closed(workflow, tmp_path):
    """Neither a schedule nor a choice: refused rather than guessed at."""
    gate = _run_gate(workflow, tmp_path, publish_input="PUBLISH")

    assert gate["returncode"] != 0
    assert "unrecognised publish input" in gate["stderr"]
    assert "publish_mode" not in gate["outputs"]


def test_the_resolved_mode_is_the_only_publish_decision(workflow):
    """No job may re-derive the mode for itself, which is how the two jobs
    came to disagree in the first place."""
    text = WORKFLOW.read_text(encoding="utf-8")
    gate_run = _step(workflow, "resolve", "gate")["run"]

    # Exactly one reference to the raw input in the whole file, and it is the
    # one that feeds the resolver.
    assert text.count("inputs.publish }}") == 1
    assert "PUBLISH_INPUT: ${{ inputs.publish }}" in text
    assert "inputs.publish ||" not in text
    assert "PUBLISH_MODE:-" not in text

    assert "publish_mode=publish" in gate_run
    assert "publish_mode=dry_run" in gate_run
    assert workflow["jobs"]["resolve"]["outputs"]["publish_mode"] == (
        "${{ steps.gate.outputs.publish_mode }}"
    )


@pytest.mark.parametrize("job", ["certified-production", "snapshot-sweep"])
def test_both_deploying_jobs_read_the_same_resolved_mode(workflow, job):
    assert workflow["jobs"][job]["env"]["PUBLISH_MODE"] == (
        "${{ needs.resolve.outputs.publish_mode }}"
    )
    assert "resolve" in workflow["jobs"][job]["needs"]


@pytest.mark.parametrize(
    "job,step_needle",
    [
        ("certified-production", "run_certified_card.sh"),
        ("snapshot-sweep", "run_stage_snapshots.sh"),
    ],
)
def test_both_jobs_withhold_publication_on_exactly_the_same_test(workflow, job, step_needle):
    run = _step_named(workflow, job, step_needle)["run"]
    assert '[ "${PUBLISH_MODE}" = "dry_run" ]' in run


# ===========================================================================
# The workflow's own argument construction, executed.
# ===========================================================================
def _sweep_args(workflow: dict, tmp_path: Path, *, publish_mode: str, as_of: str = "") -> str:
    """Run the arg-building prologue of the sweep step and report the args it
    would hand to run_stage_snapshots.sh."""
    run = _step(workflow, "snapshot-sweep", "sweep")["run"]
    prologue = run.split("# WIZARD_NFL_HOME must be resolved", 1)[0]
    script = tmp_path / f"args-{publish_mode}.sh"
    script.write_text(prologue + '\nprintf "%s" "${args}"\n', encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "AS_OF": as_of, "PUBLISH_MODE": publish_mode},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_publishing_sweep_is_not_given_the_dry_run_flag(workflow, tmp_path):
    assert "--dry-run-publish" not in _sweep_args(workflow, tmp_path, publish_mode="publish")


def test_a_dry_run_sweep_is_still_given_the_dry_run_flag(workflow, tmp_path):
    assert "--dry-run-publish" in _sweep_args(workflow, tmp_path, publish_mode="dry_run")


def test_the_as_of_override_is_unaffected_by_the_mode(workflow, tmp_path):
    args = _sweep_args(workflow, tmp_path, publish_mode="publish", as_of="2026-09-24T18:00:00Z")
    assert "--as-of '2026-09-24T18:00:00Z'" in args


# ===========================================================================
# 4, 7-9. The real sweep, end to end, in each mode.
#
#   Only the provider-facing and model-facing entrypoints are stubbed. The
#   deployment step runs the REAL publish_wizard_nfl_local.py, so the atomic
#   write, the contract validation and the read-back are all genuine.
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
  out=""
  while [ $# -gt 0 ]; do
    if [ "${1}" = "--output" ]; then out="${2}"; shift 2; else shift; fi
  done
  mkdir -p "$(dirname "${out}")"
  printf '%s\\n' '@ASSEMBLED@' > "${out}"
  echo "publication_status=OK_AWAITING_FIRST_CLOSE" ;;
*/publish_wizard_nfl_local.py)
  shift
  exec @PY@ "@REPO@/scripts/publish_wizard_nfl_local.py" "$@" ;;
*/prune_replaceable_artifacts.py) echo pruned ;;
*) echo "unexpected: ${1}" >&2; exit 97 ;;
esac
"""

# What the assembler writes for the current week before its first CLOSE.
EMPTY_WEEK_3 = json.dumps(
    {
        "schema_version": "wizard-nfl-pricing-v2",
        "season": 2026,
        "week": 3,
        "horizon": "TUE",
        "generated_at_utc": "2026-09-24T18:39:26.672957Z",
        "games": [],
    },
    indent=2,
    sort_keys=True,
)
# The stale card the public page was stuck on.
STALE_WEEK_1 = json.dumps(
    {
        "schema_version": "wizard-nfl-pricing-v2",
        "season": 2026,
        "week": 1,
        "horizon": "TUE",
        "generated_at_utc": "2026-09-15T05:14:43.247767Z",
        "games": [
            {
                "game_id": "2026_01_DEN_KC",
                "kickoff_utc": "2026-09-15T00:20:00Z",
                "away_team": "DEN",
                "home_team": "KC",
                "predicted_home_margin": 5.8,
                "predicted_game_total": 46.1,
                "market_home_spread": -3.5,
                "market_total": 44.0,
                "market_as_of_utc": "2026-09-14T12:30:00Z",
                "market_ats_book_count": 6,
                "market_total_book_count": 5,
            }
        ],
    },
    indent=2,
    sort_keys=True,
)


def _estate(tmp_path: Path, *, due: int = 0, assembled: str = EMPTY_WEEK_3) -> dict:
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    # The nginx-served directory, already holding the stale published card.
    served = tmp_path / "web"
    served.mkdir(exist_ok=True)
    (served / "latest.json").write_text(STALE_WEEK_1 + "\n", encoding="utf-8")

    stub = home / "venv" / "bin" / "python"
    body = STUB
    for token, value in (
        ("@PY@", sys.executable),
        ("@REPO@", str(REPO_ROOT)),
        ("@DUE@", str(due)),
        ("@ASSEMBLED@", assembled),
    ):
        body = body.replace(token, value)
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)

    return {"home": home, "served": served / "latest.json", "web_dir": served}


def _run_sweep(estate: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SWEEP), "--stage", "ALL", "--skip-capture", *extra],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "WIZARD_NFL_HOME": str(estate["home"]),
            "WIZARD_NFL_WEB_DIR": str(estate["web_dir"]),
        },
    )


def _published_sha(result: subprocess.CompletedProcess) -> str:
    matches = re.findall(r"^PUBLISHED_SHA256=(\S+)$", result.stdout, re.MULTILINE)
    return matches[-1] if matches else ""


def test_a_scheduled_zero_due_sweep_publishes_the_current_week(tmp_path):
    """The exact production firing, with the flag the workflow now omits."""
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage_execution=NOOP_NOTHING_DUE" in result.stdout
    assert "=== 7. current-week publication ===" in result.stdout
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert "stage_snapshot_status=OK" in result.stdout


def test_that_sweep_actually_writes_the_served_card(tmp_path):
    estate = _estate(tmp_path, due=0)
    assert json.loads(estate["served"].read_text())["week"] == 1

    _run_sweep(estate)

    served = json.loads(estate["served"].read_text())
    assert served["week"] == 3
    assert served["games"] == []


def test_that_sweep_emits_a_non_empty_published_sha_matching_the_served_bytes(tmp_path):
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    published = _published_sha(result)
    assert len(published) == 64
    assert published == sha256(estate["served"].read_bytes()).hexdigest()


def test_the_deployment_step_reports_published_not_dry_run(tmp_path):
    """The exact field that read DRY_RUN in run 36039577444."""
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate)

    assert '"status": "PUBLISHED"' in result.stdout
    assert '"status": "DRY_RUN"' not in result.stdout


def test_a_manual_dry_run_sweep_makes_no_public_write(tmp_path):
    estate = _estate(tmp_path, due=0)
    before = estate["served"].read_bytes()

    result = _run_sweep(estate, "--dry-run-publish")

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "DRY_RUN"' in result.stdout
    assert estate["served"].read_bytes() == before
    assert json.loads(estate["served"].read_text())["week"] == 1
    # No retained previous, no temp file: nothing was touched at all.
    assert sorted(p.name for p in estate["web_dir"].iterdir()) == ["latest.json"]


def test_a_manual_dry_run_still_validates_and_still_reports_a_sha(tmp_path):
    """Withholding the write is the only difference; a dry run is still a
    full rehearsal that would catch a malformed card."""
    estate = _estate(tmp_path, due=0)
    result = _run_sweep(estate, "--dry-run-publish")

    assert "=== 7. current-week publication ===" in result.stdout
    assert len(_published_sha(result)) == 64


def test_a_scheduled_sweep_with_work_due_publishes_and_reports_executed(tmp_path):
    """due_batches > 0 behaves as before, except the publication is real."""
    estate = _estate(tmp_path, due=2)
    result = _run_sweep(estate)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=EXECUTED" in result.stdout
    assert "NOOP_NOTHING_DUE" not in result.stdout
    assert json.loads(estate["served"].read_text())["week"] == 3


def test_repeated_scheduled_sweeps_stay_idempotent(tmp_path):
    """96 publishing sweeps a day must not become 96 artifacts a day."""
    estate = _estate(tmp_path, due=0)
    _run_sweep(estate)
    published = estate["served"].read_bytes()

    for _ in range(3):
        result = _run_sweep(estate)
        assert result.returncode == 0, result.stdout + result.stderr
        assert estate["served"].read_bytes() == published

    # The existing atomic path: one served card plus the one retained copy.
    assert sorted(p.name for p in estate["web_dir"].iterdir()) == [
        "latest.json",
        "latest.json.prev",
    ]


def test_a_malformed_card_still_fails_a_publishing_sweep_closed(tmp_path):
    """Publishing by default must not mean publishing anything."""
    broken = json.dumps({"schema_version": "wizard-nfl-pricing-v2", "games": []})
    estate = _estate(tmp_path, due=0, assembled=broken)
    result = _run_sweep(estate)

    assert result.returncode != 0
    assert "stage_snapshot_status=OK" not in result.stdout
    assert json.loads(estate["served"].read_text())["week"] == 1


def test_publication_uses_the_existing_store_and_creates_no_new_one(tmp_path):
    estate = _estate(tmp_path, due=0)
    _run_sweep(estate)

    text = SWEEP.read_text(encoding="utf-8")
    assert 'public_json="${NFL_MODEL_ARTIFACT_ROOT}/public/wizardofodds/nfl-pricing/latest.json"' in text
    assert (
        estate["home"] / "artifacts" / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    ).is_file()


# ===========================================================================
# 5. The public-bytes verifier runs, and passes, on a scheduled firing.
# ===========================================================================
def _evaluate_if(expression: str, context: dict) -> bool:
    """Evaluate the small GitHub expression grammar these two conditions use:
    ``${{ a.b.c == 'x' && d.e != '' }}``."""
    body = expression.strip()
    assert body.startswith("${{") and body.endswith("}}"), body
    result = True
    for clause in body[3:-2].split("&&"):
        match = re.fullmatch(r"\s*([\w.\-]+)\s*(==|!=)\s*'([^']*)'\s*", clause)
        assert match, clause
        reference, operator, literal = match.groups()
        assert reference in context, f"unmodelled reference {reference}"
        actual = context[reference]
        result = result and (actual == literal if operator == "==" else actual != literal)
    return result


def test_the_verifier_condition_fires_for_a_scheduled_publication(workflow, tmp_path):
    """Composed from real values: the mode the gate actually resolved for a
    schedule, and the sha the real sweep actually published."""
    gate = _run_gate(workflow, tmp_path, publish_input=None)
    estate = _estate(tmp_path / "estate", due=0)
    published = _published_sha(_run_sweep(estate))

    condition = _step_named(workflow, "snapshot-sweep", "verify_public_nfl_feed.py")["if"]

    assert _evaluate_if(
        condition,
        {
            "needs.resolve.outputs.publish_mode": gate["outputs"]["publish_mode"],
            "steps.sweep.outputs.published_sha256": published,
        },
    )


@pytest.mark.parametrize(
    "publish_mode,published_sha,expected",
    [
        ("publish", "2f5591cc", True),
        ("publish", "", False),
        ("dry_run", "2f5591cc", False),
        ("dry_run", "", False),
    ],
)
def test_the_verifier_condition_truth_table(workflow, publish_mode, published_sha, expected):
    condition = _step_named(workflow, "snapshot-sweep", "verify_public_nfl_feed.py")["if"]
    assert (
        _evaluate_if(
            condition,
            {
                "needs.resolve.outputs.publish_mode": publish_mode,
                "steps.sweep.outputs.published_sha256": published_sha,
            },
        )
        is expected
    )


def test_the_certified_verifier_condition_also_fires_on_a_schedule(workflow, tmp_path):
    gate = _run_gate(workflow, tmp_path, publish_input=None)
    condition = _step_named(workflow, "certified-production", "verify_public_nfl_feed.py")["if"]

    assert _evaluate_if(
        condition, {"needs.resolve.outputs.publish_mode": gate["outputs"]["publish_mode"]}
    )


def test_the_verifier_is_asked_to_compare_against_the_published_sha(workflow):
    step = _step_named(workflow, "snapshot-sweep", "verify_public_nfl_feed.py")
    assert '--expect-sha256 "${{ steps.sweep.outputs.published_sha256 }}"' in step["run"]


def test_the_verifier_passes_on_the_bytes_the_sweep_published(tmp_path, monkeypatch):
    """End of the chain: the sha the sweep emits is the sha the verifier
    accepts when the public URL serves those exact bytes."""
    estate = _estate(tmp_path, due=0)
    published = _published_sha(_run_sweep(estate))
    served_bytes = estate["served"].read_bytes()
    page_html = (REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html").read_bytes()

    monkeypatch.setattr(
        verifier,
        "_fetch",
        lambda url, *, timeout: (
            200,
            {"Content-Type": "application/json"},
            served_bytes if url.endswith("latest.json") else page_html,
        ),
    )

    report = verifier.verify(expect_sha256=published, expect_week=3)
    assert report["status"] == "VERIFIED"
    assert report["feed"]["sha256"] == published
    assert report["feed"]["game_count"] == 0


def test_the_verifier_still_fails_when_the_stale_card_is_served(tmp_path, monkeypatch):
    estate = _estate(tmp_path, due=0)
    published = _published_sha(_run_sweep(estate))
    stale = STALE_WEEK_1.encode("utf-8") + b"\n"
    page_html = (REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html").read_bytes()

    monkeypatch.setattr(
        verifier,
        "_fetch",
        lambda url, *, timeout: (
            200,
            {"Content-Type": "application/json"},
            stale if url.endswith("latest.json") else page_html,
        ),
    )

    with pytest.raises(verifier.PublicVerificationError, match="serving a different card"):
        verifier.verify(expect_sha256=published)


# ===========================================================================
# 10-12. What must not have moved.
# ===========================================================================
def test_publication_is_still_close_only():
    assert exporter.PUBLICATION_PREFERENCE == (st.STAGE_CLOSE,)


def test_no_stage_is_named_in_a_published_card(tmp_path):
    estate = _estate(tmp_path, due=0)
    _run_sweep(estate)
    served = estate["served"].read_text()
    for token in ("OPEN", "MID", "CLOSE", "snapshot_stage"):
        assert token not in served


def test_the_manual_dry_run_choice_still_exists(workflow):
    """Removing the rehearsal would be a different kind of regression."""
    publish = workflow[True]["workflow_dispatch"]["inputs"]["publish"]
    assert publish["options"] == ["dry_run", "publish"]
    assert publish["default"] == "dry_run"
    assert publish["required"] is True


def test_the_deployment_gate_and_the_publish_mode_stay_separate(workflow):
    """Publishing by default must not let a non-main ref deploy: that is the
    ref gate's job, and it is untouched."""
    for job in ("daily-maintenance", "certified-production", "snapshot-sweep"):
        assert "github.ref == 'refs/heads/main'" in workflow["jobs"][job]["if"]
    gate_run = _step(workflow, "resolve", "gate")["run"]
    assert 'if [ "${GITHUB_REF}" = "refs/heads/main" ]' in gate_run


def test_a_non_main_ref_still_runs_nothing(workflow, tmp_path):
    gate = _run_gate(workflow, tmp_path, publish_input=None, ref="refs/heads/some-branch")

    assert gate["outputs"]["deployment_allowed"] == "false"
    assert gate["outputs"]["run_snapshot_sweep"] == "false"
    assert gate["outputs"]["run_certified"] == "false"
    assert gate["outputs"]["run_daily"] == "false"


def test_the_sweep_control_flow_from_pr59_is_unchanged():
    text = SWEEP.read_text(encoding="utf-8")
    span = text[
        text.index('echo "due_batches=') : text.index("=== 7. current-week publication ===")
    ]
    code = "\n".join(line for line in span.splitlines() if not line.strip().startswith("#"))
    assert "exit 0" not in code
    assert 'echo "stage_execution=NOOP_NOTHING_DUE"' in text


def test_the_sweep_still_takes_no_historical_week_override():
    text = SWEEP.read_text(encoding="utf-8")
    call = text[text.index("run_2026_stage_snapshots.py") : text.index("sweep_exit=$?")]
    assert "--season" not in call
    assert "--week" not in call
