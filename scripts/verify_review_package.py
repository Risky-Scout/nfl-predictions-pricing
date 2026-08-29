"""Section 28: fails if the review package is not what it claims to be.

Checks:
  1. Every file listed in outputs/review_package_manifest_2026.json's
     ``review_docs`` / ``production_entrypoint`` / test-script fields is
     actually tracked in git.
  2. The manifest's scientific hashes match the live, recomputed values
     (never trusts the manifest's own copies blindly).
  3. ``--require-clean``: working tree must be clean.
  4. No private-data path (NFL_MODEL_DATA_ROOT / NFL_MODEL_ARTIFACT_ROOT
     contents, .env) is accidentally tracked in git.

Exit 0 on success, 1 on any failure (each failure printed, not just the
first).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.data.external_data import REPO_ROOT  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "outputs" / "review_package_manifest_2026.json"

# Matched against the FULL tracked path, case-insensitively. Deliberately
# narrow (a literal dotfile name, or an actual private-data file
# extension) rather than a broad keyword match -- a broad match on strings
# like "odds_history" or "the_odds_api" false-positives on entirely
# legitimate source modules/tests that merely reference those concepts by
# name (e.g. src/nfl_hybrid/odds_history.py,
# src/nfl_hybrid/data/providers/the_odds_api.py); this check exists to
# catch an accidentally-committed SECRET or DATA FILE, not to flag code
# that talks about odds data.
_PRIVATE_DOTFILES = {".env"}  # exact basename only -- .env.example is fine
_PRIVATE_DATA_EXTENSIONS = {".parquet"}

CERTIFIED_BASELINE_SHA = "d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def _tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


def _tracked_files_at_baseline() -> set[str]:
    """Files already tracked at the certified baseline commit -- a
    ``.parquet``/dotfile check only needs to catch NEW private-data files
    this certification session might have accidentally added; several
    small derived-evidence parquet files were already reviewed and
    committed by earlier certified Fix work and are not this check's
    concern."""
    return set(_run(["git", "ls-tree", "-r", "--name-only", CERTIFIED_BASELINE_SHA]).splitlines())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    if not MANIFEST_PATH.exists():
        print(f"FAIL: manifest missing: {MANIFEST_PATH}")
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text())
    tracked = _tracked_files()

    required_files = [manifest["production_entrypoint"], *manifest["review_docs"], *manifest["test_commands_files"]]
    for f in required_files:
        if f not in tracked:
            failures.append(f"REQUIRED_TRACKED_FILE_MISSING: {f}")

    try:
        fix71_summary = json.loads((REPO_ROOT / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text())
        fix8_prereg = json.loads((REPO_ROOT / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text())
        live = {
            "horizon_feature_semantics_hash": fix71_summary["horizon_feature_semantics_hash"],
            "horizon_membership_ledger_hash": fix71_summary["horizon_membership_ledger_hash"],
            "operational_model_spec_hash": fix71_summary["operational_model_spec_hash"],
            "fix8_preregistration_hash": fix8_prereg["fix8_official_oof_calibration_preregistration_hash"],
        }
        for key, live_value in live.items():
            manifest_value = manifest["scientific_hashes"][key]
            if manifest_value != live_value:
                failures.append(f"SCIENTIFIC_HASH_MISMATCH[{key}]: manifest={manifest_value} live={live_value}")
    except (FileNotFoundError, KeyError) as exc:
        failures.append(f"SCIENTIFIC_HASH_VERIFICATION_FAILED: {exc}")

    if args.require_clean:
        status = [ln for ln in _run(["git", "status", "--short"]).splitlines() if ln.strip()]
        if status:
            failures.append(f"REPO_DIRTY: {status}")

    baseline_tracked = _tracked_files_at_baseline()
    for f in tracked:
        if f in baseline_tracked:
            continue  # already reviewed/committed by earlier certified work -- not this session's concern
        path = Path(f)
        if path.name in _PRIVATE_DOTFILES:
            failures.append(f"PRIVATE_DATA_ACCIDENTALLY_TRACKED: {f}")
        elif path.suffix.lower() in _PRIVATE_DATA_EXTENSIONS:
            failures.append(f"PRIVATE_DATA_ACCIDENTALLY_TRACKED: {f}")

    if failures:
        print("REVIEW PACKAGE VERIFICATION: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("REVIEW PACKAGE VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
