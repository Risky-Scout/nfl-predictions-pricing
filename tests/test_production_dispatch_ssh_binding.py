"""Production dispatch values cross SSH as data, or the step fails closed.

ops/wizard/remote.sh cmd_exec is unchanged: each argument of `remote.sh exec`
is one line of the script it pipes to `bash -s`. The runner therefore has to
reject a horizon, as-of, manifest, or capture sha256 before that pipe starts.
An accepted value is one `name=value` assignment argument. It is not spliced
into a double-quoted remote program.

The SSH client in these tests is a double on PATH. It records the script it
would have sent and runs that script locally. It does not open a connection.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"
BINDER = REPO_ROOT / "ops" / "wizard" / "bind_dispatch_argv.sh"
REMOTE = REPO_ROOT / "ops" / "wizard" / "remote.sh"

# The official Week-1 TUE capture path the certified job is allowed to accept.
MANIFEST = (
    "/home/wizard/nfl-production-2026/data/live-observation-log/"
    "balldontlie-2026/season=2026/week=01/horizon=TUE/"
    "capture=20260908T153615Z/manifest.json"
)
STAGE_MANIFEST = MANIFEST.replace("/horizon=TUE/", "/horizon=STAGE/")
SHA = "59d1b46e488a1e80c695cb77081f80b346f29a808422439187aabcd8981b10ab"
AS_OF = "2026-09-08T16:05:00Z"
SWEEP_AS_OF = "2026-09-24T18:00:00Z"
# A quote, a command substitution, and a newline. Any of those, expanded
# inside the remote program, would run `id` before the real command.
PAYLOAD = "'; $(id)\n"

EXPECTED_CRONS = (
    "20 11 * * *",
    "5 16 * * 2,5",
    "5 17 * * 2,5",
    "25 0-10 * 9,10,11,12,1 0",
    "25 2-21 * 9,10,11,12,1 1",
    "25 3-23 * 9,10,11,12,1 2",
    "25 * * 9,10,11,12,1 3",
    "25 0-21 * 9,10,11,12,1 4",
    "25 3-23 * 9,10,11,12,1 5",
    "25 * * 9,10,11,12,1 6",
    "*/15 11-21 * 9,10,11,12,1 0",
    "*/15 22-23 * 9,10,11,12,1 0",
    "*/15 0-1 * 9,10,11,12,1 1",
    "*/15 22-23 * 9,10,11,12,1 1",
    "*/15 0-2 * 9,10,11,12,1 2",
    "*/15 22-23 * 9,10,11,12,1 4",
    "*/15 0-2 * 9,10,11,12,1 5",
)
EXPECTED_SECRETS = {
    "WIZARD_SSH_HOST",
    "WIZARD_SSH_PORT",
    "WIZARD_SSH_USER",
    "WIZARD_SSH_PRIVATE_KEY",
    "WIZARD_SSH_KNOWN_HOSTS",
    "WIZARD_NFL_HOME",
    "WIZARD_NFL_WEB_DIR",
    "BALLDONTLIE_API_KEY",
    "THE_ODDS_API_KEY",
}


def _workflow() -> dict:
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # YAML 1.1 resolves the bare key `on` to the boolean True.
    if True in parsed:
        parsed["on"] = parsed.pop(True)
    return parsed


@pytest.fixture(scope="module")
def workflow() -> dict:
    return _workflow()


def _step(workflow: dict, job: str, step_id: str) -> dict:
    return next(step for step in workflow["jobs"][job]["steps"] if step.get("id") == step_id)


def _call(function: str, value: str) -> subprocess.CompletedProcess[str]:
    script = f". ops/wizard/bind_dispatch_argv.sh\n{function} \"$VALUE\"\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "VALUE": value},
    )


def _assign(name: str, value: str) -> subprocess.CompletedProcess[str]:
    script = (
        ". ops/wizard/bind_dispatch_argv.sh\n"
        'dispatch_checked_assignment "$NAME" "$VALUE"\n'
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/tmp"),
            "NAME": name,
            "VALUE": value,
        },
    )


def _install_doubles(tmp_path: Path) -> dict[str, Path]:
    """An ssh that records stdin and a bash that records the remote argv.

    `id` and `python` record that they ran. The workflow step's PATH sees
    ssh, id, and python. The bash double is visible only to the script the
    ssh double executes, so the step itself still runs under /bin/bash.
    """
    bin_dir = tmp_path / "bin"
    remote_bin = tmp_path / "remote-bin"
    bin_dir.mkdir()
    remote_bin.mkdir()
    stdin_path = tmp_path / "ssh-stdin"
    argv_path = tmp_path / "remote-argv"
    id_marker = tmp_path / "id-ran"
    python_marker = tmp_path / "python-ran"

    (bin_dir / "ssh").write_text(
        "#!/bin/bash\n"
        f'cat > "{stdin_path}"\n'
        f'export PATH="{remote_bin}:/usr/bin:/bin"\n'
        f'exec /bin/bash -s < "{stdin_path}"\n',
        encoding="utf-8",
    )
    (bin_dir / "id").write_text(
        "#!/bin/bash\n"
        f'printf ran >> "{id_marker}"\n'
        "printf 'uid=injected\\n'\n",
        encoding="utf-8",
    )
    (bin_dir / "python").write_text(
        "#!/bin/bash\n"
        f'printf ran >> "{python_marker}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "python3").write_text(
        "#!/bin/bash\n"
        f'printf ran >> "{python_marker}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (remote_bin / "bash").write_text(
        "#!/bin/bash\n"
        'target="${1:-}"\n'
        "case \"$target\" in\n"
        "  *create_official_capture.sh|*run_certified_card.sh|*run_stage_snapshots.sh)\n"
        f'    printf \'%s\\n\' "$@" > "{argv_path}"\n'
        "    case \"$target\" in\n"
        "      *create_official_capture.sh)\n"
        "        printf 'OFFICIAL_CAPTURE_MANIFEST=%s\\n' \"${STUB_CAPTURE_MANIFEST}\"\n"
        "        ;;\n"
        "      *run_certified_card.sh)\n"
        "        printf '%s\\n' \\\n"
        "          'certified_card_published=NO_PUBLIC_WRITE_BY_DESIGN' \\\n"
        "          'exported_sha256=abc' \\\n"
        "          'CERTIFIED_RUN_ID=run-1' \\\n"
        "          'capture_manifest_sha256=def'\n"
        "        ;;\n"
        "      *run_stage_snapshots.sh)\n"
        "        printf '%s\\n' \\\n"
        "          'stage_snapshot_status=OK' \\\n"
        "          'PUBLISHED_SHA256=abc' \\\n"
        "          'snapshot_action=executed'\n"
        "        ;;\n"
        "    esac\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        'exec /bin/bash "$@"\n',
        encoding="utf-8",
    )
    for path in (
        bin_dir / "ssh",
        bin_dir / "id",
        bin_dir / "python",
        bin_dir / "python3",
        remote_bin / "bash",
    ):
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return {
        "path": bin_dir,
        "stdin": stdin_path,
        "argv": argv_path,
        "id_marker": id_marker,
        "python_marker": python_marker,
    }


def _run_step(tmp_path: Path, run: str, **overrides: str) -> dict:
    doubles = _install_doubles(tmp_path)
    script = tmp_path / "step.sh"
    script.write_text(run, encoding="utf-8")
    github_output = tmp_path / "github_output"
    github_output.write_text("", encoding="utf-8")
    env = {
        "PATH": f"{doubles['path']}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "LANG": "C",
        "WIZARD_SSH_HOST": "wizard.example",
        "WIZARD_SSH_PORT": "22",
        "WIZARD_SSH_USER": "wizard",
        "WIZARD_NFL_HOME": "/home/wizard/nfl-production-2026",
        "WIZARD_NFL_WEB_DIR": "/var/www/wizard-nfl",
        "BALLDONTLIE_API_KEY": "bdl-test",
        "THE_ODDS_API_KEY": "odds-test",
        "DEFAULT_WIZARD_NFL_HOME_LEAF": "nfl-production-2026",
        "GITHUB_OUTPUT": str(github_output),
        "STUB_CAPTURE_MANIFEST": MANIFEST,
        "HORIZON": "TUE",
        "AS_OF": AS_OF,
        "SUPPLIED_MANIFEST": "",
        "MANIFEST": MANIFEST,
        "EXPECTED_CAPTURE_SHA256": SHA,
        "PUBLISH_MODE": "publish",
        "FORCE_HORIZON": "",
    }
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    argv_lines: list[str] = []
    if doubles["argv"].is_file():
        argv_lines = doubles["argv"].read_text(encoding="utf-8").splitlines()
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": github_output.read_text(encoding="utf-8"),
        "stdin": doubles["stdin"].read_text(encoding="utf-8") if doubles["stdin"].is_file() else None,
        "argv": " ".join(argv_lines[1:]) if argv_lines else None,
        "id_ran": doubles["id_marker"].is_file(),
        "python_ran": doubles["python_marker"].is_file(),
    }


# ---------------------------------------------------------------------------
# Grammar. Nothing is trimmed, cased, or coerced into a nearby accepted value.
# ---------------------------------------------------------------------------
def test_validate_dispatch_horizon_accepts_only_tue_and_fri():
    for value in ("TUE", "FRI"):
        result = _call("validate_dispatch_horizon", value)
        assert result.returncode == 0, result.stderr
    for value in ("", "tue", "Fri", "MON", "DAILY", "TUE ", " TUE", "STAGE", "SMOKE", PAYLOAD):
        result = _call("validate_dispatch_horizon", value)
        assert result.returncode != 0, value
        assert "FAIL CLOSED" in result.stderr


def test_validate_dispatch_as_of_accepts_the_documented_utc_instant_only():
    result = _call("validate_dispatch_as_of", AS_OF)
    assert result.returncode == 0, result.stderr

    rejected = (
        "2026-09-08",
        "2026-09-08T16:05:00",
        "2026-09-08 16:05:00Z",
        "2026-09-08T16:05:00+00:00",
        "2026-09-08T12:05:00-04:00",
        "2026-09-08T16:05:00z",
        "2026-02-31T00:00:00Z",
        "2026-04-31T12:00:00Z",
        "2026-13-01T00:00:00Z",
        "2026-09-08T24:05:00Z",
        PAYLOAD,
    )
    for value in rejected:
        result = _call("validate_dispatch_as_of", value)
        assert result.returncode != 0, value
        assert "FAIL CLOSED" in result.stderr


def test_validate_dispatch_manifest_accepts_only_an_official_absolute_capture():
    for value in (MANIFEST, STAGE_MANIFEST):
        result = _call("validate_dispatch_manifest", value)
        assert result.returncode == 0, (value, result.stderr)

    relative = MANIFEST.lstrip("/")
    home = MANIFEST.replace("/home/wizard", "$HOME", 1)
    traversal = MANIFEST.replace("/data/", "/data/../data/", 1)
    one_digit_week = MANIFEST.replace("/week=01/", "/week=1/", 1)
    for value in (relative, home, traversal, one_digit_week, PAYLOAD):
        result = _call("validate_dispatch_manifest", value)
        assert result.returncode != 0, value
        assert "FAIL CLOSED" in result.stderr


def test_sha256_case_is_preserved():
    mixed = "aB" * 32
    assert len(mixed) == 64
    assert mixed != mixed.lower()
    assert mixed != mixed.upper()

    result = _assign("capture_sha256", mixed)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"capture_sha256={mixed}\n"

    quoted = subprocess.run(
        ["bash", "-c", "printf '%q' \"$VALUE\""],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "VALUE": mixed},
    )
    assert quoted.stdout == mixed


def test_checked_assignment_is_one_name_value_word():
    result = _assign("horizon", "TUE")
    assert result.stdout == "horizon=TUE\n"
    result = _assign("as_of", AS_OF)
    assert result.stdout == f"as_of={AS_OF}\n"
    result = _assign("manifest", MANIFEST)
    assert result.stdout == f"manifest={MANIFEST}\n"
    result = _assign("capture_sha256", SHA)
    assert result.stdout == f"capture_sha256={SHA}\n"


# ---------------------------------------------------------------------------
# The workflow steps. Valid input reaches the remote program as data.
# ---------------------------------------------------------------------------
def test_valid_capture_remote_argv(workflow, tmp_path):
    ran = _run_step(tmp_path, _step(workflow, "certified-production", "capture")["run"])

    assert ran["returncode"] == 0, ran["stderr"]
    assert ran["argv"] == f"--horizon TUE --as-of {AS_OF}"
    assert ran["stdin"] is not None
    lines = ran["stdin"].splitlines()
    assert lines[0] == "set -euo pipefail"
    assert "horizon=TUE" in lines
    assert f"as_of={AS_OF}" in lines
    assert 'bash "${OPS}/create_official_capture.sh" --horizon "${horizon}" "${as_of_args[@]}"' in lines
    assert f"manifest={MANIFEST}" in ran["output"]
    assert not ran["id_ran"]


def test_valid_certified_card_remote_argv(workflow, tmp_path):
    ran = _run_step(tmp_path, _step(workflow, "certified-production", "certified")["run"])

    assert ran["returncode"] == 0, ran["stderr"]
    assert ran["argv"] == (
        f"--horizon TUE --capture-manifest {MANIFEST} --capture-sha256 {SHA} --as-of {AS_OF}"
    )
    lines = ran["stdin"].splitlines()
    assert f"manifest={MANIFEST}" in lines
    assert f"capture_sha256={SHA}" in lines
    assert "--capture-manifest \"${manifest}\"" in ran["stdin"]
    assert not ran["id_ran"]


def test_valid_sweep_remote_argv(workflow, tmp_path):
    ran = _run_step(
        tmp_path,
        _step(workflow, "snapshot-sweep", "sweep")["run"],
        AS_OF=SWEEP_AS_OF,
        PUBLISH_MODE="publish",
    )

    assert ran["returncode"] == 0, ran["stderr"]
    assert ran["argv"] == f"--stage ALL --as-of {SWEEP_AS_OF}"
    assert f"as_of={SWEEP_AS_OF}" in ran["stdin"].splitlines()
    assert 'bash "${OPS}/run_stage_snapshots.sh" --stage ALL "${as_of_args[@]}"' in ran["stdin"]
    assert not ran["id_ran"]


def test_dry_run_sweep_with_empty_as_of_omits_the_flag(workflow, tmp_path):
    ran = _run_step(
        tmp_path,
        _step(workflow, "snapshot-sweep", "sweep")["run"],
        AS_OF="",
        PUBLISH_MODE="dry_run",
    )

    assert ran["returncode"] == 0, ran["stderr"]
    assert ran["argv"] == "--stage ALL --dry-run-publish"
    assert "as_of=" in ran["stdin"].splitlines()
    assert "--as-of" not in ran["argv"]
    assert not ran["id_ran"]


# ---------------------------------------------------------------------------
# The same payload, on every dispatch input that would otherwise reach SSH.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "job,step_id,field",
    [
        ("certified-production", "capture", "HORIZON"),
        ("certified-production", "capture", "AS_OF"),
        ("certified-production", "capture", "SUPPLIED_MANIFEST"),
        ("certified-production", "certified", "MANIFEST"),
        ("certified-production", "certified", "AS_OF"),
        ("snapshot-sweep", "sweep", "AS_OF"),
    ],
)
def test_payload_fails_closed_before_ssh(workflow, tmp_path, job, step_id, field):
    ran = _run_step(
        tmp_path,
        _step(workflow, job, step_id)["run"],
        **{field: PAYLOAD},
    )

    assert ran["returncode"] != 0
    assert "FAIL CLOSED" in ran["stderr"]
    assert ran["stdin"] is None
    assert ran["argv"] is None
    assert not ran["id_ran"]
    assert "manifest=" not in ran["output"]


def test_a_poisoned_capture_manifest_is_not_written_to_github_output(workflow, tmp_path):
    ran = _run_step(
        tmp_path,
        _step(workflow, "certified-production", "capture")["run"],
        HORIZON="TUE",
        AS_OF=AS_OF,
        SUPPLIED_MANIFEST="",
        STUB_CAPTURE_MANIFEST=PAYLOAD,
    )

    assert ran["returncode"] != 0
    assert "FAIL CLOSED" in ran["stderr"]
    assert ran["stdin"] is not None
    assert "manifest=" not in ran["output"]
    assert PAYLOAD not in ran["output"]
    assert not ran["id_ran"]


@pytest.mark.parametrize("field", ["FORCE_HORIZON", "AS_OF"])
def test_resolve_rejects_the_payload_before_python_runs(workflow, tmp_path, field):
    overrides = {"FORCE_HORIZON": "", "AS_OF": ""}
    overrides[field] = PAYLOAD
    ran = _run_step(
        tmp_path,
        _step(workflow, "resolve", "decide")["run"],
        **overrides,
    )

    assert ran["returncode"] != 0
    assert "FAIL CLOSED" in ran["stderr"]
    assert not ran["python_ran"]
    assert ran["stdin"] is None
    assert not ran["id_ran"]


def test_resolve_reaches_python_for_a_real_horizon_and_instant(workflow, tmp_path):
    ran = _run_step(
        tmp_path,
        _step(workflow, "resolve", "decide")["run"],
        FORCE_HORIZON="FRI",
        AS_OF=AS_OF,
    )

    assert ran["returncode"] == 0, ran["stderr"]
    assert ran["python_ran"]
    assert ran["stdin"] is None


# ---------------------------------------------------------------------------
# The destination, the schedule, and the SSH wrapper stay as they were.
# ---------------------------------------------------------------------------
def test_three_jobs_still_use_wizard_production(workflow):
    for job in ("daily-maintenance", "certified-production", "snapshot-sweep"):
        assert workflow["jobs"][job]["environment"] == "wizard-production"
    for job in ("resolve", "report"):
        assert "environment" not in workflow["jobs"][job]


def test_secrets_and_crons_remain(workflow):
    text = WORKFLOW.read_text(encoding="utf-8")
    found = set(re.findall(r"secrets\.([A-Z0-9_]+)", text))
    assert found == EXPECTED_SECRETS
    crons = tuple(entry["cron"] for entry in workflow["on"]["schedule"])
    assert crons == EXPECTED_CRONS
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert list(inputs) == [
        "mode",
        "horizon",
        "market_capture_manifest",
        "market_capture_sha256",
        "as_of",
        "publish",
    ]


def test_remote_sh_still_pipes_each_argument_to_bash_s_and_does_not_bind():
    text = REMOTE.read_text(encoding="utf-8")
    assert "printf '%s\\n' 'set -euo pipefail' \"$@\"" in text
    assert "'bash -s'" in text
    assert "bind_dispatch_argv" not in text


def test_summary_binds_horizon_and_manifest_outside_the_run_body(workflow):
    summary = next(
        step for step in workflow["jobs"]["certified-production"]["steps"] if step.get("name") == "Summary"
    )
    body = summary["run"]
    assert "${{ needs.resolve.outputs.certified_horizon }}" not in body
    assert "${{ steps.capture.outputs.manifest }}" not in body
    assert "HORIZON_LABEL" in body
    assert "MANIFEST_LABEL" in body
    assert summary["env"]["HORIZON_LABEL"] == "${{ needs.resolve.outputs.certified_horizon }}"
    assert summary["env"]["MANIFEST_LABEL"] == "${{ steps.capture.outputs.manifest }}"


def test_binder_is_sourced_not_executed_and_is_executable():
    mode = BINDER.stat().st_mode
    assert mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", str(BINDER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not a remote command" in result.stderr
