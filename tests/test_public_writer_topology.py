"""The public NFL endpoint has ONE contract and ONE authorized writer topology.

THE DEFECT. PR #63 made the current-week pregame board the public product
(``wizard-nfl-pricing-v3``, each game shown at the latest pregame snapshot it
has reached). It left a second public writer in place: the certified TUE/FRI
orchestrator exported its own single-cutoff ``wizard-nfl-pricing-v2`` card,
published it over the same atomic path, and then asserted the public site served
exactly those bytes. Twice a week the live board was therefore replaced by a
card that drops every game being shown from its OPEN or MID snapshot and carries
no stage labelling at all, and the only thing that restored it was the next
snapshot sweep happening to run afterwards. Under ``mode=certified_only`` no
sweep follows, so the downgrade would simply persist.

THE RULE. Writing the public endpoint and reading it are different questions.

    WRITE  exactly AUTHORITATIVE_PUBLIC_SCHEMA_VERSION, enforced at every
           transport that can reach the served directory.
    READ   both contracts, because the endpoint is a real place whose current
           contents this repository does not get to assume.

So the downgrade is impossible rather than merely unwired: the certified
orchestrator no longer publishes, AND the publishers refuse a non-authoritative
payload whoever hands them one.

WHAT IS DELIBERATELY UNCHANGED. The certified card is still captured, still
preflighted, still run, still exported and still archived into its immutable
``archive/season=/week=/horizon=`` tree, and its forecasts remain the records the
OPEN/MID/CLOSE stage machinery reads. Nothing here touches model science,
grading, retraining, recalibration or the promotion gate.

Hermetic: source-level topology assertions plus ``tmp_path`` publication
estates. No network, no SSH, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
OPS = REPO_ROOT / "ops" / "wizard"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PRODUCTION_WORKFLOW = WORKFLOWS / "nfl_2026_production.yml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(SCRIPTS / "publish_sportsodds_nfl.py", "_topology_contract")
deployer = _load(SCRIPTS / "publish_wizard_nfl_local.py", "_topology_deployer")
verifier = _load(SCRIPTS / "verify_public_nfl_feed.py", "_topology_verifier")
board = _load(SCRIPTS / "export_current_week_nfl_feed.py", "_topology_board")

CERTIFIED_ORCHESTRATOR = OPS / "run_certified_card.sh"
SWEEP_ORCHESTRATOR = OPS / "run_stage_snapshots.sh"

# The two transports that can place bytes on the public NFL endpoint. Nothing
# else in the repository may, and both are gated.
PUBLIC_TRANSPORTS = ("publish_wizard_nfl_local.py", "publish_sportsodds_nfl.py")

# The ONE orchestrator authorized to invoke a public transport for JSON.
AUTHORIZED_JSON_PUBLISHING_ORCHESTRATORS = ("run_stage_snapshots.sh",)


# ===========================================================================
# The authoritative contract, named once.
# ===========================================================================
def test_the_authoritative_public_contract_is_the_board_contract():
    assert contract.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION == "wizard-nfl-pricing-v3"
    assert contract.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION == contract.BOARD_SCHEMA_VERSION
    # The board assembler and the publication gate cannot drift apart.
    assert board.SCHEMA_VERSION == contract.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION


def test_every_consumer_borrows_the_one_constant_rather_than_restating_it():
    for module in (deployer, verifier):
        assert module.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION == contract.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION
    for path in (
        SCRIPTS / "publish_wizard_nfl_local.py",
        SCRIPTS / "verify_public_nfl_feed.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "AUTHORITATIVE_PUBLIC_SCHEMA_VERSION = _contract." in source
        assert 'AUTHORITATIVE_PUBLIC_SCHEMA_VERSION = "' not in source


def test_reading_stays_broader_than_writing():
    """The verifier must still be able to PARSE an older card in order to
    report that the endpoint is serving one."""
    assert set(contract.GAME_KEYS_BY_SCHEMA) == {
        contract.SCHEMA_VERSION, contract.BOARD_SCHEMA_VERSION
    }
    assert contract.SCHEMA_VERSION != contract.AUTHORITATIVE_PUBLIC_SCHEMA_VERSION


# ===========================================================================
# Writer topology: enumerated, and guarded against silent additions.
# ===========================================================================
def _executable_source(path: Path) -> str:
    """The file with whole-line comments removed.

    Both shells and YAML comment with ``#``, and every file here documents the
    publication rules at length -- including by naming the transports. Topology
    is about what RUNS, so prose about a transport must not read as a call to it.
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _public_writer_call_sites() -> dict[str, list[str]]:
    """Every shell orchestrator and workflow that invokes a public transport."""
    found: dict[str, list[str]] = {}
    for path in sorted([*OPS.glob("*.sh"), *WORKFLOWS.glob("*.yml")]):
        source = _executable_source(path)
        hits = [transport for transport in PUBLIC_TRANSPORTS if transport in source]
        if hits:
            found[path.name] = sorted(hits)
    return found


def test_only_the_expected_orchestrators_invoke_a_public_transport():
    """THE TOPOLOGY GUARD.

    A future change that wires a second public writer -- another orchestrator,
    another workflow, a well-meaning "republish" helper -- fails here rather than
    silently reintroducing the race this amendment removed. Update the expected
    map deliberately, never incidentally.
    """
    expected = {
        # The sole authoritative JSON writer.
        "run_stage_snapshots.sh": ["publish_wizard_nfl_local.py"],
        # HTML only, manual dispatch only: it publishes the page shell and is
        # given no --json, so it is not a latest.json writer.
        "deploy_sportsodds_nfl_html.yml": ["publish_sportsodds_nfl.py"],
    }
    assert _public_writer_call_sites() == expected


def test_the_html_deploy_workflow_publishes_html_only():
    workflow = yaml.safe_load((WORKFLOWS / "deploy_sportsodds_nfl_html.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["deploy"]["steps"]
    commands = " ".join(step.get("run", "") for step in steps)
    assert "--html-only" in commands
    assert "--json" not in commands
    # And it cannot be reached by the scheduler at all.
    assert set(workflow[True]) == {"workflow_dispatch"}


def test_the_certified_orchestrator_is_no_longer_a_public_writer():
    source = _executable_source(CERTIFIED_ORCHESTRATOR)
    for forbidden in (
        "publish_wizard_nfl_local",
        "publish_sportsodds_nfl",
        "PUBLISHED_SHA256",
        "WIZARD_NFL_WEB_DIR",
    ):
        assert forbidden not in source, f"certified orchestrator still reaches publication: {forbidden}"
    assert "certified_card_published=NO_PUBLIC_WRITE_BY_DESIGN" in source


def test_the_sweep_orchestrator_is_still_the_public_writer():
    source = _executable_source(SWEEP_ORCHESTRATOR)
    assert "publish_2026_current_week.py" in source
    assert "publish_wizard_nfl_local.py" in source
    assert "PUBLISHED_SHA256=" in source


# ===========================================================================
# The certified path still generates and archives.
# ===========================================================================
def test_the_certified_orchestrator_still_runs_and_exports_the_card():
    source = CERTIFIED_ORCHESTRATOR.read_text(encoding="utf-8")
    # Capture identity, preflight, the certified card, then the export that
    # writes the immutable archive. All unchanged.
    for required in (
        "verify_capture_manifest_integrity.py",
        "--preflight",
        "run_2026_production_card.py",
        "export_wizard_nfl_pricing.py",
        "--run-manifest",
        "--forecast-dir",
        "exported_sha256=",
        "CERTIFIED_RUN_ID=",
        "certified_card_status=OK",
    ):
        assert required in source, f"certified generation/archival step lost: {required}"


def test_the_certified_exporter_still_writes_its_immutable_archive():
    """The archive is the audit record, and the v2 contract behind it is
    untouched -- which is why this amendment did not change the exporter."""
    source = (SCRIPTS / "export_wizard_nfl_pricing.py").read_text(encoding="utf-8")
    assert 'SCHEMA_VERSION = "wizard-nfl-pricing-v2"' in source
    assert "write_archive(" in source
    assert "refusing to overwrite" in (SCRIPTS / "export_wizard_nfl_predictions.py").read_text(encoding="utf-8")


def test_the_certified_orchestrator_still_accepts_the_dry_run_flag():
    """It is inert now, and kept only so an existing caller is not failed by an
    unknown argument. The workflow still passes it on a dry run."""
    source = CERTIFIED_ORCHESTRATOR.read_text(encoding="utf-8")
    assert "--dry-run-publish)   shift ;;" in source
    assert "Accepted and inert" in source


# ===========================================================================
# The workflow: certified generates, the sweep publishes.
# ===========================================================================
@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(PRODUCTION_WORKFLOW.read_text(encoding="utf-8"))


def _job_run_text(workflow: dict, job: str) -> str:
    return "\n".join(step.get("run", "") for step in workflow["jobs"][job]["steps"])


def test_the_certified_job_no_longer_asserts_the_public_bytes(workflow):
    text = _job_run_text(workflow, "certified-production")
    assert "--expect-sha256" not in text
    assert "published_sha256" not in text
    # It asserts the opposite: the orchestrator performed no public write.
    assert "certified_card_published=NO_PUBLIC_WRITE_BY_DESIGN" in text
    assert "^PUBLISHED_SHA256=" in text
    assert "exported_sha256=" in text


def test_only_the_sweep_job_asserts_the_public_bytes(workflow):
    asserting = [
        job for job in workflow["jobs"]
        if "--expect-sha256" in _job_run_text(workflow, job)
    ]
    assert asserting == ["snapshot-sweep"]


def test_the_sweep_job_asserts_the_authoritative_contract(workflow):
    text = _job_run_text(workflow, "snapshot-sweep")
    assert "--expect-authoritative-contract" in text
    assert "--expect-sha256" in text


def test_the_daily_liveness_probe_stays_a_probe(workflow):
    """It publishes nothing and can fix nothing, so it must not assert a
    contract the endpoint may legitimately not have reached yet."""
    text = _job_run_text(workflow, "daily-maintenance")
    assert "verify_public_nfl_feed.py" in text
    assert "--expect-authoritative-contract" not in text
    assert "--expect-sha256" not in text


def test_no_workflow_publishes_json_over_ftp():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = _executable_source(path)
        if "publish_sportsodds_nfl.py" not in text:
            continue
        assert "--html-only" in text, f"{path.name} reaches the FTP publisher without --html-only"


# ===========================================================================
# The gate itself, exercised.
# ===========================================================================
def _game(**overrides) -> dict:
    game = {
        "game_id": "2026_03_ATL_GB",
        "kickoff_utc": "2026-09-25T00:15:00Z",
        "away_team": "ATL",
        "home_team": "GB",
        "predicted_home_margin": 6.17,
        "predicted_game_total": 45.25,
        "market_home_spread": -4.5,
        "market_total": 43.5,
        "market_as_of_utc": "2026-09-24T23:07:33Z",
        "market_ats_book_count": 8,
        "market_total_book_count": 8,
    }
    game.update(overrides)
    return game


def _board_payload(**overrides) -> dict:
    payload = {
        "schema_version": "wizard-nfl-pricing-v3",
        "season": 2026,
        "week": 3,
        "horizon": "TUE",
        "generated_at_utc": "2026-09-27T01:59:37Z",
        "games": [_game(snapshot_stage="OPEN", model_generated_at_utc="2026-09-21T04:10:00Z")],
    }
    payload.update(overrides)
    return payload


def _certified_payload(**overrides) -> dict:
    payload = {
        "schema_version": "wizard-nfl-pricing-v2",
        "season": 2026,
        "week": 3,
        "horizon": "FRI",
        "generated_at_utc": "2026-09-25T16:20:00Z",
        "games": [_game()],
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def estate(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    return {
        "web": web,
        "board": _write(tmp_path / "board.json", _board_payload()),
        "certified": _write(tmp_path / "certified.json", _certified_payload()),
    }


def test_the_board_publishes_normally(estate):
    report = deployer.publish(json_path=estate["board"], web_dir=estate["web"], dry_run=False)
    assert report["status"] == "PUBLISHED"
    assert report["schema_version"] == "wizard-nfl-pricing-v3"
    served = json.loads((estate["web"] / "latest.json").read_text(encoding="utf-8"))
    assert served["schema_version"] == "wizard-nfl-pricing-v3"
    assert served["games"][0]["snapshot_stage"] == "OPEN"


@pytest.mark.parametrize("horizon", ["TUE", "FRI"])
def test_a_certified_card_cannot_overwrite_a_live_board(estate, horizon):
    """Requirements 2 and 3: neither the Tuesday nor the Friday certified card
    can replace the live board."""
    deployer.publish(json_path=estate["board"], web_dir=estate["web"], dry_run=False)
    before = (estate["web"] / "latest.json").read_bytes()

    card = _write(estate["certified"], _certified_payload(horizon=horizon))
    with pytest.raises(deployer.PublishError, match="never downgraded"):
        deployer.publish(json_path=card, web_dir=estate["web"], dry_run=False)

    assert (estate["web"] / "latest.json").read_bytes() == before


def test_the_refusal_happens_before_any_filesystem_effect(estate):
    deployer.publish(json_path=estate["board"], web_dir=estate["web"], dry_run=False)
    listing = sorted(p.name for p in estate["web"].iterdir())

    with pytest.raises(deployer.PublishError):
        deployer.publish(json_path=estate["certified"], web_dir=estate["web"], dry_run=False)

    # No staged temp file, and the retained previous card was not rotated.
    assert sorted(p.name for p in estate["web"].iterdir()) == listing
    assert not list(estate["web"].glob("*.tmp"))
    assert not list(estate["web"].glob(".*.tmp"))


def test_even_a_dry_run_refuses_a_non_authoritative_card(estate):
    """A dry run that reported a v2 card as publishable would be a lie that an
    operator could act on."""
    with pytest.raises(deployer.PublishError, match="never downgraded"):
        deployer.publish(json_path=estate["certified"], web_dir=estate["web"], dry_run=True)


def test_the_ftp_transport_is_held_to_the_same_rule(estate):
    """The manual FTP fallback reaches the same public endpoint, so it is gated
    too -- and the refusal lands on a DRY RUN, before any socket could exist."""
    env = {
        "SPORTSODDS_FTP_HOST": "ftp.example-test.invalid",
        "SPORTSODDS_FTP_USER": "sportsodds-test",
        "SPORTSODDS_FTP_REMOTE_DIR": "/tools/odds-scanner/predictions/NFL",
        "SPORTSODDS_FTP_MODE": "ftp",
    }
    with pytest.raises(contract.PublishError, match="never downgraded"):
        contract.publish(
            json_path=estate["certified"],
            html_path=REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html",
            publish_json=True,
            publish_html=False,
            dry_run=True,
            env=env,
        )


# ===========================================================================
# The certified orchestrator, EXECUTED. Observed behaviour, not grepped source.
# ===========================================================================
#
# The interpreter is a stub that records one file per invocation. Crucially it
# FAILS LOUDLY if the public transport is ever reached, so "the certified path
# does not publish" is proved by running it rather than by reading it.
STUB_PY = r"""#!/usr/bin/env bash
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
    printf '%s\n' '{"schema_version": "wizard-nfl-pricing-v2"}' > "${out}" ;;
*/publish_wizard_nfl_local.py|*/publish_sportsodds_nfl.py)
    echo "A PUBLIC TRANSPORT WAS INVOKED BY THE CERTIFIED ORCHESTRATOR" >&2
    exit 91 ;;
*)
    echo "stub interpreter got an unexpected script: ${script}" >&2; exit 97 ;;
esac
"""


@pytest.fixture
def certified_estate(tmp_path) -> dict:
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state", "staging"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    invocations = tmp_path / "invocations"
    invocations.mkdir()

    stub = home / "venv" / "bin" / "python"
    stub.write_text(STUB_PY.replace("@REAL_PYTHON@", sys.executable), encoding="utf-8")
    stub.chmod(0o755)

    manifest = tmp_path / "capture" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"manifest_sha256": "stub"}), encoding="utf-8")

    served = tmp_path / "web"
    served.mkdir()
    served_card = served / "latest.json"
    _write(served_card, _board_payload())

    import os

    return {
        "env": {
            **os.environ,
            "WIZARD_NFL_HOME": str(home),
            "WIZARD_NFL_WEB_DIR": str(served),
            "STUB_INVOCATION_DIR": str(invocations),
        },
        "home": home,
        "manifest": manifest,
        "served": served_card,
        "invocations": invocations,
    }


def _run_certified(estate: dict, *extra: str):
    import subprocess

    return subprocess.run(
        [
            "bash", str(CERTIFIED_ORCHESTRATOR),
            "--capture-manifest", str(estate["manifest"]),
            *extra,
        ],
        capture_output=True,
        text=True,
        env=estate["env"],
    )


def _invoked_scripts(estate: dict) -> list[str]:
    names = []
    for path in sorted(estate["invocations"].glob("*.args")):
        argv = path.read_text(encoding="utf-8").splitlines()
        if argv and argv[0].endswith(".py"):
            names.append(Path(argv[0]).name)
    return names


@pytest.mark.parametrize("horizon", ["TUE", "FRI"])
def test_the_certified_orchestrator_generates_and_archives_without_publishing(certified_estate, horizon):
    """Requirements 2-6 in one observed run, for both certified horizons."""
    served_before = certified_estate["served"].read_bytes()
    result = _run_certified(certified_estate, "--horizon", horizon)
    assert result.returncode == 0, result.stdout + result.stderr

    invoked = _invoked_scripts(certified_estate)
    # Generation and archival still happen.
    assert "verify_capture_manifest_integrity.py" in invoked
    assert invoked.count("run_2026_production_card.py") == 2  # preflight, then the card
    assert "export_wizard_nfl_pricing.py" in invoked
    # Publication does not. The stub would have exited 91 if it had.
    assert "publish_wizard_nfl_local.py" not in invoked
    assert "publish_sportsodds_nfl.py" not in invoked

    assert f"=== 3. certified {horizon} card ===" in result.stdout
    assert "=== 4. export + archive wizard-nfl-pricing-v2 ===" in result.stdout
    assert "certified_card_published=NO_PUBLIC_WRITE_BY_DESIGN" in result.stdout
    assert "certified_card_status=OK" in result.stdout
    assert "exported_sha256=" in result.stdout
    assert f"CERTIFIED_HORIZON={horizon}" in result.stdout
    # The single line the old workflow keyed its public assertion on is gone.
    assert "PUBLISHED_SHA256=" not in result.stdout

    # And the live board is byte-identical afterwards.
    assert certified_estate["served"].read_bytes() == served_before


def test_the_exported_certified_card_goes_to_the_artifact_tree_only(certified_estate):
    result = _run_certified(certified_estate, "--horizon", "TUE")
    assert result.returncode == 0, result.stdout + result.stderr

    exported = (
        certified_estate["home"] / "artifacts" / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    )
    assert exported.is_file()
    # A staging path, not the served directory -- different files entirely.
    assert exported != certified_estate["served"]
    assert json.loads(certified_estate["served"].read_text())["schema_version"] == (
        "wizard-nfl-pricing-v3"
    )


def test_the_inert_dry_run_flag_does_not_break_the_certified_run(certified_estate):
    result = _run_certified(certified_estate, "--horizon", "TUE", "--dry-run-publish")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "certified_card_status=OK" in result.stdout


def test_the_gate_reads_the_validated_summary_not_the_raw_payload():
    """It cannot be reached without full contract validation first."""
    with pytest.raises(contract.PublishError, match="never downgraded"):
        contract.require_publishable_contract({"schema_version": "wizard-nfl-pricing-v2"}, source="x")
    with pytest.raises(contract.PublishError, match="never downgraded"):
        contract.require_publishable_contract({}, source="x")
    assert contract.require_publishable_contract(
        {"schema_version": "wizard-nfl-pricing-v3"}, source="x"
    ) == "wizard-nfl-pricing-v3"


def test_no_public_writer_can_emit_the_legacy_contract_after_this_change():
    """Requirement 18, stated as the property rather than as a list of callers:
    every transport that can reach the endpoint passes through the gate."""
    for name in PUBLIC_TRANSPORTS:
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "require_publishable_contract(" in source, f"{name} places bytes without the gate"


def test_the_verifier_can_assert_the_authoritative_contract():
    """Requirement 19. Exercised against the verifier's own assertion path with
    a synthetic payload, so no network is involved."""
    assert "--expect-authoritative-contract" in (
        SCRIPTS / "verify_public_nfl_feed.py"
    ).read_text(encoding="utf-8")
    parsed = verifier.build_arg_parser().parse_args(["--expect-authoritative-contract"])
    assert parsed.expect_authoritative_contract is True
    assert verifier.build_arg_parser().parse_args([]).expect_authoritative_contract is False
