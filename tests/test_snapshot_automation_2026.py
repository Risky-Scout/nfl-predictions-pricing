"""Proofs that the snapshot contract is actually SCHEDULED, not just callable.

Covers the last link in the chain: the existing workflow and ops stack reach
the stage execution entrypoint, on a DST-safe schedule, with bounded storage
and with recalibration still gated by the existing maturity firewall.

Structural where the subject is YAML or bash, behavioural where the subject is
Python. Nothing here contacts a server, a provider or the network.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"
SWEEP_SCRIPT = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"
DAILY_SCRIPT = REPO_ROOT / "ops" / "wizard" / "run_daily_maintenance.sh"
PRUNER = REPO_ROOT / "scripts" / "prune_replaceable_artifacts.py"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sweep_text() -> str:
    return SWEEP_SCRIPT.read_text(encoding="utf-8")


# ===========================================================================
# The sweep is a job in the EXISTING workflow, not a parallel scheduler.
# ===========================================================================
def test_there_is_still_exactly_one_production_workflow():
    workflows = sorted(p.name for p in (REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert "nfl_2026_production.yml" in workflows
    for name in workflows:
        assert "snapshot" not in name and "stage" not in name


def test_the_sweep_is_a_job_in_the_existing_production_workflow(workflow):
    assert "snapshot-sweep" in workflow["jobs"]
    assert list(workflow["jobs"]) == [
        "resolve",
        "daily-maintenance",
        "certified-production",
        "snapshot-sweep",
        "report",
    ]


def test_the_sweep_runs_after_the_certified_card_rather_than_racing_it(workflow):
    """Both republish latest.json, so on a Tuesday the certified card must
    land first and the sweep must reflect it."""
    needs = workflow["jobs"]["snapshot-sweep"]["needs"]
    assert "certified-production" in needs
    condition = workflow["jobs"]["snapshot-sweep"]["if"]
    assert "needs.certified-production.result == 'success'" in condition
    assert "needs.certified-production.result == 'skipped'" in condition


def test_the_sweep_is_main_only_and_uses_the_wizard_environment(workflow):
    job = workflow["jobs"]["snapshot-sweep"]
    assert job["environment"] == "wizard-production"
    assert "refs/heads/main" in job["if"]


def test_a_failed_sweep_fails_the_workflow(workflow):
    report = workflow["jobs"]["report"]
    assert "snapshot-sweep" in report["needs"]
    assert "needs.snapshot-sweep.result" in report["steps"][0]["run"]


def test_the_daily_pass_is_unchanged_and_still_runs():
    """Daily maintenance continues: the sweep adds to it, never replaces it."""
    text = DAILY_SCRIPT.read_text(encoding="utf-8")
    assert "refresh_bdl_2026_games_evidence.py" in text
    assert "update_2026_games_population.py" in text
    assert "generate_2026_recalibration_candidate.py --promote-if-eligible" in text
    assert 'echo "daily_maintenance_status=OK"' in text


# ===========================================================================
# Scheduling is DST-safe and the cron is never the gate.
# ===========================================================================
def test_the_sweep_has_its_own_polling_crons(workflow):
    crons = [entry["cron"] for entry in workflow[True]["schedule"]]
    # The two pre-existing certified firings and the daily one survive.
    assert "20 11 * * *" in crons
    assert "5 16 * * 2,5" in crons
    assert "5 17 * * 2,5" in crons
    # Plus polling: dense on game days, hourly otherwise.
    sweep = [c for c in crons if c.startswith("*/15") or c.startswith("25 ")]
    assert len(sweep) == 2


def test_the_polling_crons_are_bounded_to_the_season(workflow):
    """Not every 15 minutes all year -- storage and runner minutes stay
    bounded."""
    crons = [entry["cron"] for entry in workflow[True]["schedule"]]
    for cron in [c for c in crons if c.startswith("*/15") or c.startswith("25 ")]:
        month_field = cron.split()[3]
        assert month_field != "*", cron
        assert set(month_field.split(",")) <= {"1", "9", "10", "11", "12"}


def test_no_sweep_cron_pins_a_wall_clock_hour(workflow):
    """The stage instants come from America/New_York arithmetic and each
    game's kickoff, so the sweep crons deliberately do NOT name an hour the
    way the DST-redundant certified pair has to."""
    crons = [entry["cron"] for entry in workflow[True]["schedule"]]
    for cron in [c for c in crons if c.startswith("*/15") or c.startswith("25 ")]:
        assert cron.split()[1] == "*", cron


def test_stage_dueness_is_decided_by_the_sweep_not_the_gate(workflow):
    gate = next(
        step for step in workflow["jobs"]["resolve"]["steps"] if step.get("id") == "gate"
    )["run"]
    assert "run_snapshot_sweep" in gate

    # The gate says only whether the sweep MAY look; it never computes a stage
    # instant itself. Checked against the CODE, not the comments explaining it.
    code = "\n".join(
        line for line in gate.splitlines() if not line.strip().startswith("#")
    )
    for stage_logic in ("CLOSE", "OPEN", "MID", "kickoff", "60"):
        assert stage_logic not in code, stage_logic


def test_the_snapshot_only_dispatch_mode_exists(workflow):
    assert "snapshot_only" in workflow[True]["workflow_dispatch"]["inputs"]["mode"]["options"]


def test_nothing_is_published_unless_the_operator_asked(workflow_text, sweep_text):
    assert "--dry-run-publish" in workflow_text
    assert "--dry-run-publish" in sweep_text


def test_the_public_verifier_proves_the_published_bytes(workflow):
    steps = workflow["jobs"]["snapshot-sweep"]["steps"]
    verify = [s for s in steps if "verify_public_nfl_feed.py" in str(s.get("run", ""))]
    assert len(verify) == 1
    assert "--expect-sha256" in verify[0]["run"]


# ===========================================================================
# The sweep orchestrator honours the point-in-time contract, in order.
# ===========================================================================
def test_the_sweep_refreshes_then_retrains_then_snapshots_then_publishes(sweep_text):
    order = [
        "refresh_bdl_2026_games_evidence.py",
        "update_2026_games_population.py",
        "create_official_capture.sh",
        "run_2026_stage_snapshots.py",
        "generate_2026_recalibration_candidate.py",
        "attach_2026_results_from_population.py",
        "report_2026_snapshot_performance.py",
        "publish_2026_current_week.py",
        "publish_wizard_nfl_local.py",
        "prune_replaceable_artifacts.py",
    ]
    positions = [sweep_text.index(token) for token in order]
    assert positions == sorted(positions), dict(zip(order, positions, strict=True))


def test_the_sweep_reuses_the_daily_entrypoints_rather_than_restating_them(sweep_text):
    """The snapshot path and the daily path must never disagree about the
    population or the active calibrator."""
    daily = DAILY_SCRIPT.read_text(encoding="utf-8")
    for shared in (
        "refresh_bdl_2026_games_evidence.py",
        "update_2026_games_population.py",
        "generate_2026_recalibration_candidate.py",
    ):
        assert shared in daily
        assert shared in sweep_text


def test_the_sweep_uses_one_as_of_for_every_stage(sweep_text):
    assert sweep_text.count("as_of_args=()") == 1
    assert sweep_text.index("as_of_args=()") < sweep_text.index("run_2026_stage_snapshots.py")


def test_the_sweep_fails_closed_with_distinct_exit_codes(sweep_text):
    """Every distinct failure mode gets its own code, so an operator reading
    a workflow log knows which stage refused without opening the script."""
    import re

    codes = {int(m) for m in re.findall(r"\bexit (\d+)\b", sweep_text)} - {0}
    # 2 usage, 3 evidence/population, 4 stage execution, 5-9 the five
    # recalibration failure classes, 10 feed assembly.
    assert codes == {2, 3, 4, 5, 6, 7, 8, 9, 10}, sorted(codes)


def test_the_sweep_publishes_nothing_when_a_stage_fails(sweep_text):
    stage_block = sweep_text[sweep_text.index("=== 4."): sweep_text.index("=== 5.")]
    assert "nothing is published" in stage_block
    assert "exit 4" in stage_block


def test_recalibration_promotion_still_uses_the_existing_policy_entrypoint(sweep_text):
    assert "generate_2026_recalibration_candidate.py --promote-if-eligible" in sweep_text
    # No bypass flag, no threshold restated in the orchestrator.
    assert "PROMOTION_ELIGIBLE_MIN_GAMES" not in sweep_text
    assert "--force-promote" not in sweep_text
    assert "200" not in sweep_text


def test_the_maturity_firewall_is_still_single_sourced():
    from nfl_hybrid.evaluation import prospective_strength_2026 as ps

    policy = json.loads((REPO_ROOT / "config" / "recalibration_promotion_2026.json").read_text())
    minimum = policy["minimum_prospective_maturity"]
    assert minimum["derived_from_symbol"] == "PROMOTION_ELIGIBLE_MIN_GAMES"
    assert minimum["numeric_threshold_duplicated_here"] is False
    assert ps.PROMOTION_ELIGIBLE_MIN_GAMES == 200


# ===========================================================================
# Bounded storage: the pruner refuses to touch evidence.
# ===========================================================================
def _quotes_dir(root: Path, digest: str) -> Path:
    path = root / "live-market-2026" / "balldontlie" / "season=2026" / "week=02" / "horizon=TUE" / f"manifest_sha256={digest}"
    path.mkdir(parents=True)
    (path / "bookmaker_quotes.parquet").write_bytes(b"x" * 1024)
    return path


def _age(path: Path, days: int) -> None:
    old = time.time() - days * 86400
    import os

    os.utime(path, (old, old))


def test_the_pruner_only_ever_targets_the_re_materializable_namespace(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pruner", PRUNER)
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)

    assert pruner.PRUNABLE_NAMESPACES == ("live-market-2026",)
    for protected in ("production-2026", "public", "live-observation-log", "games-population-2026"):
        assert protected in pruner.PROTECTED_NAMESPACES
        with pytest.raises(pruner.PruneRefused, match="refusing to delete"):
            pruner._assert_prunable(tmp_path / protected / "anything", tmp_path)


def test_the_pruner_keeps_anything_a_run_manifest_references(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pruner", PRUNER)
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)

    referenced = _quotes_dir(tmp_path, "a" * 64)
    orphan = _quotes_dir(tmp_path, "b" * 64)
    _age(referenced, 200)
    _age(orphan, 200)

    manifests = tmp_path / "production-2026" / "run-manifests"
    manifests.mkdir(parents=True)
    (manifests / "r1.json").write_text(
        json.dumps({"input_hashes": {"live_market_capture_sha256": "a" * 64}})
    )

    decision = pruner.plan(tmp_path, retention_days=45, now_epoch=time.time())
    assert [entry["path"] for entry in decision["prunable"]] == [str(orphan)]
    assert any(entry["reason"] == "REFERENCED_BY_RUN_MANIFEST" for entry in decision["kept"])


def test_the_pruner_keeps_anything_inside_the_retention_window(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pruner", PRUNER)
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)

    recent = _quotes_dir(tmp_path, "c" * 64)
    _age(recent, 3)
    decision = pruner.plan(tmp_path, retention_days=45, now_epoch=time.time())
    assert decision["prunable"] == []
    assert decision["kept"][0]["reason"] == "WITHIN_RETENTION_WINDOW"


def test_the_pruner_is_a_dry_run_unless_asked(tmp_path):
    orphan = _quotes_dir(tmp_path, "d" * 64)
    _age(orphan, 200)
    result = subprocess.run(
        [sys.executable, str(PRUNER), "--artifact-root", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "DRY_RUN"
    assert orphan.exists()


def test_the_pruner_deletes_only_when_applied(tmp_path):
    orphan = _quotes_dir(tmp_path, "e" * 64)
    _age(orphan, 200)
    result = subprocess.run(
        [sys.executable, str(PRUNER), "--artifact-root", str(tmp_path), "--apply"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "APPLIED"
    assert not orphan.exists()


def test_an_unreadable_manifest_is_not_permission_to_delete(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pruner", PRUNER)
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)

    _age(_quotes_dir(tmp_path, "f" * 64), 200)
    manifests = tmp_path / "production-2026" / "run-manifests"
    manifests.mkdir(parents=True)
    (manifests / "broken.json").write_text("{not json")
    with pytest.raises(pruner.PruneRefused, match="unreadable"):
        pruner.plan(tmp_path, retention_days=45, now_epoch=time.time())


def test_the_sweep_prunes_dry_by_default(sweep_text):
    prune_block = sweep_text[sweep_text.index("=== 8."):]
    assert "--apply" in prune_block
    assert "PRUNE_APPLY" in prune_block
    assert 'prune_args+=(--apply)' in prune_block


# ===========================================================================
# The entrypoint the workflow calls is a real no-op when nothing is due.
# ===========================================================================
def test_the_stage_entrypoint_exits_zero_with_no_current_card(tmp_path):
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "_stage_entry", REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"
    )
    entry = importlib.util.module_from_spec(spec)
    sys.modules["_stage_entry"] = entry
    spec.loader.exec_module(entry)

    empty = pd.DataFrame(
        columns=[
            "game_id", "season", "week", "season_type", "home_team_id", "away_team_id",
            "scheduled_kickoff_utc", "home_score", "away_score", "neutral_site",
        ]
    )
    result = entry.run(
        stage="ALL", as_of="2026-09-20T16:05:00Z", operational_root=tmp_path,
        market_capture_manifest=None, games=empty,
    )
    assert result["status"] == "NO_CURRENT_CARD"
    assert result["stages"] == []


def test_the_stage_entrypoint_claims_no_open_without_market_evidence(tmp_path):
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "_stage_entry2", REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    card = pd.DataFrame(
        [{"game_id": "G1", "scheduled_kickoff_utc": pd.Timestamp("2026-09-20T17:00:00Z")}]
    )
    assert entry.discover_open_observations(card, quotes=None) == {}
    assert entry.discover_open_observations(card, quotes=pd.DataFrame()) == {}
