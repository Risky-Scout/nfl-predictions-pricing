"""Regression proofs for the three live snapshot-sweep failures, and the
recalibration-authorization question they surfaced.

Live runs 35124840706 and 35128396794 failed for three independent
orchestration reasons. Each section below pins one of them, plus section 4's
conclusion about whether the legacy preregistration can block promotion.

Hermetic: YAML/bash structure where the subject is configuration, real calls
where the subject is Python. No network, no server, no provider.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.evaluation import prospective_strength_2026 as ps
from nfl_hybrid.production import recalibration_2026 as rc

from test_nfl_2026_recalibration_promotion import (  # noqa: E402
    _seed,
    _write_baseline,
    _write_candidate,
    _write_maturity_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"
SWEEP = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"
CAPTURE_SH = REPO_ROOT / "ops" / "wizard" / "create_official_capture.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture_mod = _load(REPO_ROOT / "scripts" / "capture_bdl_2026_asof.py", "_capture_hotfix")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sweep_step(workflow) -> dict:
    return next(s for s in workflow["jobs"]["snapshot-sweep"]["steps"] if s.get("id") == "sweep")


@pytest.fixture(scope="module")
def sweep_text() -> str:
    return SWEEP.read_text(encoding="utf-8")


# ===========================================================================
# 1. The sweep forwards provider credentials to the remote process.
#
#    Live: BDLConfigError: BALLDONTLIE key is missing.
#          games_evidence_refresh=PROVIDER_UNAVAILABLE
# ===========================================================================
def test_the_sweep_step_receives_the_provider_secrets(sweep_step):
    assert sweep_step["env"]["BALLDONTLIE_API_KEY"] == "${{ secrets.BALLDONTLIE_API_KEY }}"
    assert sweep_step["env"]["THE_ODDS_API_KEY"] == "${{ secrets.THE_ODDS_API_KEY }}"


def test_the_sweep_exports_the_provider_secrets_into_the_remote_process(sweep_step):
    """Having them in the job env is not enough -- the work happens over SSH,
    so they must reach the far side."""
    run = sweep_step["run"]
    assert "export BALLDONTLIE_API_KEY=" in run
    assert "export THE_ODDS_API_KEY=" in run
    assert "run_stage_snapshots.sh" in run


def test_the_sweep_uses_the_same_secrets_as_the_jobs_that_already_worked(workflow):
    """No new secret was invented: the same wizard-production names the daily
    and certified jobs already use."""
    def secret_names(job: str) -> set[str]:
        names = set()
        for step in workflow["jobs"][job]["steps"]:
            for key, value in (step.get("env") or {}).items():
                if "secrets." in str(value):
                    names.add(key)
        return names

    sweep = secret_names("snapshot-sweep")
    assert {"BALLDONTLIE_API_KEY", "THE_ODDS_API_KEY"} <= sweep
    assert sweep <= secret_names("daily-maintenance") | secret_names("certified-production")


def test_no_secret_is_written_to_disk_or_committed(sweep_step):
    run = sweep_step["run"]
    for leak in (">", "tee", "echo $BALLDONTLIE", "cat <<"):
        assert f"{leak} /tmp/bdl" not in run
    assert "BALLDONTLIE_API_KEY" not in SWEEP.read_text(encoding="utf-8")
    assert "THE_ODDS_API_KEY" not in SWEEP.read_text(encoding="utf-8")


def test_the_server_home_is_resolved_on_the_server(sweep_step):
    """The fallback mentions $HOME; resolving it on the runner would point at
    the runner's home directory."""
    run = sweep_step["run"]
    assert "OPS=" in run
    assert "home_leaf=" not in run


# ===========================================================================
# 2. Stage observations are point-in-time, not forced through TUE/FRI.
#
#    Live: target_cutoff_utc = 2026-09-15T16:00:00Z, ERROR (off-window),
#          market_capture=UNAVAILABLE on a Wednesday sweep.
# ===========================================================================
WEDNESDAY = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)
FRIDAY_NOON = datetime(2026, 9, 18, 16, 0, tzinfo=timezone.utc)
GAME_MINUS_60 = datetime(2026, 9, 20, 16, 0, tzinfo=timezone.utc)
CERTIFIED_TUE_CUTOFF = datetime(2026, 9, 15, 16, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "label,instant",
    [
        ("wednesday OPEN discovery", WEDNESDAY),
        ("friday MID", FRIDAY_NOON),
        ("per-game CLOSE", GAME_MINUS_60),
    ],
)
def test_a_stage_observation_succeeds_outside_every_certified_window(label, instant):
    horizon, downgrade = capture_mod.resolve_effective_horizon(
        capture_mod.STAGE_HORIZON, instant, instant, False
    )
    assert horizon == capture_mod.STAGE_HORIZON, label
    assert downgrade is None, label


def test_a_certified_capture_outside_its_window_still_fails_closed():
    """The whole point of the separate horizon: the certified gate is not
    relaxed, it is simply not the gate a stage goes through."""
    with pytest.raises(capture_mod.OffWindowError, match="outside the window"):
        capture_mod.resolve_effective_horizon("TUE", WEDNESDAY, CERTIFIED_TUE_CUTOFF, False)


def test_a_certified_capture_inside_its_window_is_unchanged():
    assert capture_mod.resolve_effective_horizon(
        "TUE", CERTIFIED_TUE_CUTOFF, CERTIFIED_TUE_CUTOFF, False
    ) == ("TUE", None)


def test_a_stage_is_never_a_production_card_horizon():
    """A stage capture can never be priced as a certified card, and a
    certified card can never be priced from a stage observation."""
    assert bridge.PRODUCTION_HORIZONS == ("TUE", "FRI")
    assert capture_mod.STAGE_HORIZON not in bridge.PRODUCTION_HORIZONS
    assert bridge.STAGE_HORIZONS == (bridge.STAGE_HORIZON,)


def test_a_stage_is_not_smoke_relabelled():
    assert capture_mod.STAGE_HORIZON != "SMOKE"
    assert capture_mod.resolve_effective_horizon("SMOKE", WEDNESDAY, None, False) == ("SMOKE", None)


def test_the_sweep_asks_for_a_stage_observation_not_a_certified_capture(sweep_text):
    assert "--stage-observation" in sweep_text
    assert "--allow-off-window-smoke" not in sweep_text


def test_the_certified_orchestrator_never_asks_for_a_stage_observation():
    assert "--stage-observation" not in (
        REPO_ROOT / "ops" / "wizard" / "run_certified_card.sh"
    ).read_text(encoding="utf-8")


def test_the_capture_script_refuses_a_stage_without_an_explicit_instant(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "capture_bdl_2026_asof.py"),
            "--season", "2026", "--week", "2", "--season-type", "REG",
            "--horizon", capture_mod.STAGE_HORIZON, "--data-root", str(tmp_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "requires --nominal-cutoff" in result.stderr


@pytest.mark.parametrize(
    "offset,expected",
    [
        (timedelta(hours=1), "in the future"),
        (-capture_mod.STAGE_MAX_OBSERVATION_LAG - timedelta(hours=1), "never back-dated"),
    ],
)
def test_a_stage_instant_must_be_recent_and_not_in_the_future(tmp_path, offset, expected):
    instant = datetime.now(timezone.utc) + offset
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "capture_bdl_2026_asof.py"),
            "--season", "2026", "--week", "2", "--season-type", "REG",
            "--horizon", capture_mod.STAGE_HORIZON,
            "--nominal-cutoff", instant.isoformat().replace("+00:00", "Z"),
            "--data-root", str(tmp_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert expected in result.stderr


def test_the_ops_capture_script_keeps_certified_and_stage_modes_separate():
    text = CAPTURE_SH.read_text(encoding="utf-8")
    assert "STAGE_OBSERVATION" in text
    assert 'capture_horizon="STAGE"' in text
    # The certified branch still passes the resolved card cutoff verbatim.
    assert 'capture_horizon="${HORIZON}"' in text


def test_the_sweep_parses_the_capture_scripts_own_output_key(sweep_text):
    """The sweep previously grepped for a key the capture script never emits,
    so every successful capture was silently dropped."""
    assert "OFFICIAL_CAPTURE_MANIFEST=" in sweep_text
    assert "s/^capture_manifest=//p" not in sweep_text
    assert "OFFICIAL_CAPTURE_MANIFEST=" in CAPTURE_SH.read_text(encoding="utf-8")


# --- the bridge rule that lets a stage observation be priced --------------
def _stage_capture(tmp_path, *, nominal_cutoff: str):
    from test_bdl_market_bridge import write_capture

    return write_capture(tmp_path, horizon=bridge.STAGE_HORIZON, nominal_cutoff_utc=nominal_cutoff)


def test_a_stage_observation_may_be_priced_at_a_later_stage_cutoff(tmp_path):
    """Observed at 15:40, priced at a 16:00 CLOSE -- the normal case."""
    manifest = _stage_capture(tmp_path, nominal_cutoff="2026-09-08T15:40:00Z")
    capture = bridge.validate_capture_manifest(
        manifest,
        expected_horizon=bridge.STAGE_HORIZON,
        allowed_horizons=bridge.STAGE_HORIZONS,
        observation_not_after_utc="2026-09-08T16:00:00Z",
    )
    assert capture.horizon == bridge.STAGE_HORIZON


def test_a_stage_observation_after_its_own_cutoff_is_refused(tmp_path):
    """Lookahead: pricing a snapshot from a market observed after its
    instant."""
    manifest = _stage_capture(tmp_path, nominal_cutoff="2026-09-08T16:30:00Z")
    with pytest.raises(bridge.BdlMarketBridgeError, match="refusing to look ahead"):
        bridge.validate_capture_manifest(
            manifest,
            expected_horizon=bridge.STAGE_HORIZON,
            allowed_horizons=bridge.STAGE_HORIZONS,
            observation_not_after_utc="2026-09-08T16:00:00Z",
        )


def test_a_stage_capture_cannot_be_priced_as_a_certified_card(tmp_path):
    manifest = _stage_capture(tmp_path, nominal_cutoff="2026-09-08T15:40:00Z")
    with pytest.raises(bridge.BdlMarketBridgeError):
        bridge.validate_capture_manifest(manifest, expected_horizon="TUE")


def test_the_certified_cutoff_rule_is_exactly_as_strict_as_before(tmp_path):
    """The certified path still demands EQUALITY, not "at or before"."""
    from test_bdl_market_bridge import CUTOFF, write_capture

    manifest = write_capture(tmp_path, horizon="TUE", nominal_cutoff_utc=CUTOFF)
    bridge.validate_capture_manifest(
        manifest, expected_horizon="TUE", expected_target_cutoff_utc=CUTOFF
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="nominal_cutoff_utc"):
        bridge.validate_capture_manifest(
            manifest, expected_horizon="TUE", expected_target_cutoff_utc="2026-09-08T17:00:00Z"
        )


# ===========================================================================
# 3. A zero-due sweep is a clean no-op that publishes nothing.
#
#    Live: FAIL_CLOSED "no game in the current week has a publishable
#          snapshot yet", exit 10, on a firing with due_batches=0.
# ===========================================================================
def test_the_sweep_counts_due_batches_before_deciding_to_publish(sweep_text):
    assert "due_batches=" in sweep_text
    assert sweep_text.index("due_batches=") < sweep_text.index("publish_2026_current_week.py")


def test_a_zero_due_sweep_exits_clean_without_touching_latest_json(sweep_text):
    block = sweep_text[sweep_text.index("NOOP_NOTHING_DUE") - 800: sweep_text.index("=== 6.")]
    assert "snapshot_action=NOOP_NOTHING_DUE" in block
    assert "stage_snapshot_status=OK" in block
    assert "exit 0" in block
    # The no-op returns BEFORE assembly, so latest.json is never written.
    assert sweep_text.index("NOOP_NOTHING_DUE") < sweep_text.index("publish_2026_current_week.py")


def test_the_no_op_is_narrow_and_never_masks_a_real_failure(sweep_text):
    """A stage that WAS due and failed already exited 4 above; the no-op asks
    only whether zero were due."""
    assert sweep_text.index("exit 4") < sweep_text.index("due_batches=")
    assert "due_batches" not in sweep_text[: sweep_text.index("exit 4")]


def test_a_sweep_that_executed_something_still_publishes_and_can_fail_closed(sweep_text):
    assert "exit 10" in sweep_text
    assert "FAIL CLOSED: current-week feed assembly failed" in sweep_text
    assert "snapshot_action=EXECUTED" in sweep_text


def test_dry_run_never_publishes(sweep_text):
    assert "--dry-run-publish" in sweep_text
    assert "publish_args+=(--dry-run)" in sweep_text


def test_the_workflow_only_verifies_public_bytes_when_something_was_published(workflow):
    steps = workflow["jobs"]["snapshot-sweep"]["steps"]
    verify = next(s for s in steps if "verify_public_nfl_feed.py" in str(s.get("run", "")))
    assert "snapshot_action == 'EXECUTED'" in verify["if"]


def test_a_zero_due_sweep_runs_end_to_end_and_exits_zero(tmp_path):
    """The real script, on a real (synthetic) estate with nothing due and no
    published state. Exit 0, the declared status, and no latest.json."""
    home = tmp_path / "nfl-production-2026"
    for sub in ("repo/scripts", "venv/bin", "logs", "artifacts", "state"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    stub = home / "venv" / "bin" / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${1}" = "-c" ]; then exec ' + sys.executable + ' "$@"; fi\n'
        'case "${1}" in\n'
        "*/refresh_bdl_2026_games_evidence.py) echo refreshed ;;\n"
        "*/update_2026_games_population.py) echo population ;;\n"
        "*/run_2026_stage_snapshots.py)\n"
        '  echo \'{"status": "OK", "stages": [{"stage": "OPEN", "due_batches": 0, "executions": []}]}\' ;;\n'
        "*/generate_2026_recalibration_candidate.py) echo recalibration ;;\n"
        '*) echo "unexpected: ${1}" >&2; exit 97 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SWEEP), "--stage", "ALL", "--skip-capture", "--dry-run-publish"],
        capture_output=True, text=True,
        env={
            **__import__("os").environ,
            "WIZARD_NFL_HOME": str(home),
            "WIZARD_NFL_WEB_DIR": str(tmp_path / "web"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot_action=NOOP_NOTHING_DUE" in result.stdout
    assert "stage_snapshot_status=OK" in result.stdout
    assert not (home / "artifacts" / "public").exists()


# ===========================================================================
# 4. The legacy preregistration does NOT block promotion.
#
#    CONCLUSION: no fix is required. promotion_authorization() is recorded as
#    provenance on every decision, and PROMOTION_NOT_AUTHORIZED is only ever
#    a status INSIDE that provenance document -- never a decision status of
#    evaluate_promotion or promote_candidate, which branch on the operational
#    policy and the maturity floor alone.
# ===========================================================================
def test_the_legacy_authorization_is_still_not_authorized():
    authorization = rc.promotion_authorization(REPO_ROOT)
    assert authorization["authorized"] is False
    assert authorization["status"] == rc.NOT_AUTHORIZED


def test_promotion_not_authorized_is_never_a_promotion_decision_status():
    """It is provenance, not a verdict. If it ever became a decision status
    the automatic contract would silently stop working."""
    decisions = {
        rc.DECISION_PROMOTED, rc.DECISION_ELIGIBLE, rc.DECISION_NOT_YET_MATURE,
        rc.DECISION_NO_CANDIDATE, rc.DECISION_ALREADY_ACTIVE, rc.DECISION_POLICY_DISABLED,
    }
    assert rc.NOT_AUTHORIZED not in decisions


@pytest.fixture
def mature_estate(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir(parents=True)
    _write_baseline(root)
    return root


def test_a_mature_candidate_is_eligible_despite_the_legacy_wording(mature_estate):
    _write_maturity_ledger(mature_estate, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    _write_candidate(mature_estate, _seed())

    decision = rc.evaluate_promotion(
        mature_estate, repo_root=REPO_ROOT, operational_root=mature_estate
    )
    assert decision["status"] == rc.DECISION_ELIGIBLE
    assert decision["eligible"] is True
    # Recorded, and demonstrably not obeyed.
    assert decision["scientific_refit_authorization"]["authorized"] is False
    assert "no_scientific_refit" not in decision["reason"]
    assert "authoriz" not in decision["reason"].lower()


def test_that_eligible_candidate_actually_promotes(mature_estate):
    _write_maturity_ledger(mature_estate, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    candidate = _write_candidate(mature_estate, _seed())

    result = rc.promote_candidate(
        mature_estate, repo_root=REPO_ROOT, operational_root=mature_estate
    )
    assert result["status"] == rc.DECISION_PROMOTED
    assert rc.resolve_active_calibrator(mature_estate).candidate_id == candidate


def test_the_two_hundred_game_firewall_is_untouched(mature_estate):
    """One game short is still NOT_YET_MATURE. The conclusion above must not
    be mistaken for loosening the floor."""
    _write_maturity_ledger(mature_estate, ps.PROMOTION_ELIGIBLE_MIN_GAMES - 1)
    _write_candidate(mature_estate, _seed())

    decision = rc.evaluate_promotion(
        mature_estate, repo_root=REPO_ROOT, operational_root=mature_estate
    )
    assert decision["status"] == rc.DECISION_NOT_YET_MATURE
    assert decision["eligible"] is False
    assert decision["minimum_prospective_games"] == 200


def test_automatic_promotion_is_enabled_in_the_operational_policy():
    policy = json.loads((REPO_ROOT / "config" / "recalibration_promotion_2026.json").read_text())
    assert policy["automatic_promotion_enabled"] is True
    assert policy["minimum_prospective_maturity"]["derived_from_symbol"] == "PROMOTION_ELIGIBLE_MIN_GAMES"
