"""Audit every non-Git runtime asset 2026 production depends on, classify how
each one could be recovered, and emit a size/hash manifest.

This is the evidence behind the server-consolidation decision (removing the
dependency on the operator's Mac). It answers three questions with data rather
than assumption:

  1. WHAT does production actually read and write outside git, under
     ``NFL_MODEL_DATA_ROOT``, ``NFL_MODEL_ARTIFACT_ROOT`` and
     ``NFL_LIVE_DATA_ROOT``?
  2. WHICH of those can be deterministically rebuilt or re-downloaded on a
     fresh host, and which are IRREPLACEABLE production evidence that must be
     migrated rather than recreated?
  3. Do two hosts hold the same bytes? (``--compare`` diffs two manifests, so
     a migration can be verified BEFORE anything is deleted anywhere.)

READ-ONLY. It never downloads, rebuilds, migrates, deletes or publishes
anything, and it never opens a network connection. Python standard library
only, so it runs on the Wizard server without the project's dependencies
installed.

RECOVERY CLASSES
  ``REBUILDABLE_PUBLIC``
      Re-downloadable from a free public source with an in-repo producer
      (the nflverse backfill estate: ``scripts/pull_pbp_2020_2025.py``).
      Losing it costs time, not evidence.
  ``REGENERABLE_FROM_ESTATE``
      Deterministically recomputable on any host that holds the estate, by a
      certified in-repo script. Not migrated by choice, only by convenience.
  ``IRREPLACEABLE_PAID_HISTORY``
      Purchased, timestamped historical market snapshots. A vendor will not
      re-serve a point-in-time snapshot that has already passed; re-buying is
      not equivalent because the certified matrices were built from THESE
      bytes. MUST be migrated.
  ``IRREPLACEABLE_LIVE_EVIDENCE``
      Immutable point-in-time captures and immutable production ledgers: the
      BallDontLie observation log, the forecast/evaluation ledgers and the run
      manifests. Physically impossible to recreate -- the instant they record
      has passed. MUST be migrated.
  ``IRREPLACEABLE_CERTIFIED``
      The frozen production calibration seed. Regenerable in principle, but
      its content hash is part of the certified contract, so a regenerated
      copy is a different artifact and would break certification. MUST be
      migrated.
  ``IN_GIT``
      Already versioned in this repository; nothing to migrate.

Usage:
  python scripts/audit_production_runtime_assets.py
  python scripts/audit_production_runtime_assets.py --hash --output /tmp/mac_manifest.json
  python scripts/audit_production_runtime_assets.py --compare /tmp/mac_manifest.json /tmp/wizard_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

SCHEMA_VERSION = "nfl-production-runtime-asset-audit-v1"

DATA_ROOT_ENV = "NFL_MODEL_DATA_ROOT"
ARTIFACT_ROOT_ENV = "NFL_MODEL_ARTIFACT_ROOT"
LIVE_ROOT_ENV = "NFL_LIVE_DATA_ROOT"

REBUILDABLE_PUBLIC = "REBUILDABLE_PUBLIC"
REGENERABLE_FROM_ESTATE = "REGENERABLE_FROM_ESTATE"
IRREPLACEABLE_PAID_HISTORY = "IRREPLACEABLE_PAID_HISTORY"
IRREPLACEABLE_LIVE_EVIDENCE = "IRREPLACEABLE_LIVE_EVIDENCE"
IRREPLACEABLE_CERTIFIED = "IRREPLACEABLE_CERTIFIED"

MIGRATION_REQUIRED_CLASSES = (
    IRREPLACEABLE_PAID_HISTORY,
    IRREPLACEABLE_LIVE_EVIDENCE,
    IRREPLACEABLE_CERTIFIED,
)


class AssetSpec:
    """One audited runtime asset tree."""

    __slots__ = ("name", "root_env", "relative", "recovery_class", "producer", "why")

    def __init__(self, name, root_env, relative, recovery_class, producer, why):
        self.name = name
        self.root_env = root_env
        self.relative = relative
        self.recovery_class = recovery_class
        self.producer = producer
        self.why = why


# ---------------------------------------------------------------------------
# The audited estate. Declared explicitly rather than discovered, so an asset
# that is MISSING is reported as missing instead of silently omitted -- the
# whole point of the audit is to find what a fresh host does not yet have.
# ---------------------------------------------------------------------------
ASSETS: tuple[AssetSpec, ...] = (
    # -- NFL_MODEL_DATA_ROOT: the private historical estate ------------------
    AssetSpec(
        "backfill.nflverse_estate", DATA_ROOT_ENV, "backfill-2020-2025",
        REBUILDABLE_PUBLIC, "scripts/pull_pbp_2020_2025.py",
        "nflverse play-by-play/box-score/roster/schedule pulls and their canonical tables. "
        "Free public source with an in-repo producer; a fresh host can re-download it.",
    ),
    AssetSpec(
        "odds_history.2020_2023", DATA_ROOT_ENV, "odds-api-history-2020-2023",
        IRREPLACEABLE_PAID_HISTORY, "scripts/capture_lines.py (forward-only)",
        "Purchased point-in-time Odds API snapshots. The certified 2020-2023 market matrices "
        "were built from exactly these bytes; a past snapshot cannot be re-served.",
    ),
    AssetSpec(
        "odds_history.2024_confirmation", DATA_ROOT_ENV, "odds-api-history-2024-confirmation",
        IRREPLACEABLE_PAID_HISTORY, "scripts/capture_lines.py (forward-only)",
        "Purchased point-in-time Odds API snapshots backing the 2024 confirmation run.",
    ),
    AssetSpec(
        "odds_history.2025_final_test", DATA_ROOT_ENV, "odds-api-history-2025-final-test",
        IRREPLACEABLE_PAID_HISTORY, "scripts/capture_lines.py (forward-only)",
        "Purchased point-in-time Odds API snapshots backing the 2025 final test.",
    ),
    AssetSpec(
        "feature_store.canonical_market_matrices", DATA_ROOT_ENV, "feature-store-2020-2025",
        REGENERABLE_FROM_ESTATE, "examples/build_canonical_market_matrices.py",
        "Canonical closing_t10 ATS/TOTAL market matrices. Deterministic from the odds history "
        "plus the backfill estate.",
    ),
    AssetSpec(
        "live_observation_log.balldontlie_2026", DATA_ROOT_ENV,
        "live-observation-log/balldontlie-2026",
        IRREPLACEABLE_LIVE_EVIDENCE, "scripts/capture_bdl_2026_asof.py (forward-only)",
        "The official TUE/FRI pregame BallDontLie captures, including the Week-1 capture whose "
        "manifest hash is a certified input. Each records a market at an instant that has passed.",
    ),
    AssetSpec(
        "live_observation_log.balldontlie_2026_games_evidence", DATA_ROOT_ENV,
        "live-observation-log/balldontlie-2026-games-evidence",
        IRREPLACEABLE_LIVE_EVIDENCE, "scripts/refresh_bdl_2026_games_evidence.py (forward-only)",
        "Daily 2026 schedule/results evidence captures. Forward-only: a fresh host starts "
        "accumulating its own from the day it takes over, but existing captures cannot be recreated.",
    ),

    # -- NFL_LIVE_DATA_ROOT --------------------------------------------------
    AssetSpec(
        "live_provider_cache.balldontlie", LIVE_ROOT_ENV, ".",
        IRREPLACEABLE_LIVE_EVIDENCE, "nfl_hybrid.providers.balldontlie (forward-only)",
        "Immutable raw BallDontLie provider snapshot cache.",
    ),

    # -- NFL_MODEL_ARTIFACT_ROOT: generated output ---------------------------
    AssetSpec(
        "certified.production_calibration_seed", ARTIFACT_ROOT_ENV,
        "fix8-official-oof-calibration-2026/production_calibration_seed.json",
        IRREPLACEABLE_CERTIFIED, "scripts/run_fix8_official_oof_calibration.py",
        "The FROZEN Fix-8 production calibrator. Its hash is part of the certified contract, so a "
        "regenerated copy is a different artifact even when the math is identical.",
    ),
    AssetSpec(
        "certified.fix8_calibration_evidence", ARTIFACT_ROOT_ENV,
        "fix8-official-oof-calibration-2026",
        REGENERABLE_FROM_ESTATE, "scripts/run_fix8_official_oof_calibration.py",
        "The rest of the Fix-8 calibration evidence tree (ledgers, manifests). Deterministic from "
        "the estate; migrated for convenience and provenance rather than necessity.",
    ),
    AssetSpec(
        "oof_chronological", ARTIFACT_ROOT_ENV, "chronological-oof-2020-2025",
        REGENERABLE_FROM_ESTATE, "scripts/generate_fix3_1_proof_scale_oof.py",
        "Chronological OOF predictions and residual ledger.",
    ),
    AssetSpec(
        "chronological_calibration", ARTIFACT_ROOT_ENV, "chronological-calibration-2020-2025",
        REGENERABLE_FROM_ESTATE, "scripts/build_pricing_calibration.py",
        "Per-market chronological calibration ledgers and manifests.",
    ),
    AssetSpec(
        "feature_deduction_2026", ARTIFACT_ROOT_ENV, "feature-deduction-2026",
        REGENERABLE_FROM_ESTATE, "scripts/run_fix6_feature_deduction.py",
        "Frozen feature-selection evidence for the six ELO_STRENGTH features.",
    ),
    AssetSpec(
        "production.forecast_ledger", ARTIFACT_ROOT_ENV, "production-2026/forecast-ledger",
        IRREPLACEABLE_LIVE_EVIDENCE, "src/nfl_hybrid/production/run_2026.py (append-only)",
        "Immutable per-game production forecasts. Write-once by identity; a rerun cannot recreate "
        "a forecast that was made at a past cutoff.",
    ),
    AssetSpec(
        "production.evaluation_ledger", ARTIFACT_ROOT_ENV, "production-2026/evaluation-ledger",
        IRREPLACEABLE_LIVE_EVIDENCE, "src/nfl_hybrid/production/run_2026.py (append-only)",
        "Immutable prospective evaluation records and their attached results.",
    ),
    AssetSpec(
        "production.run_manifests", ARTIFACT_ROOT_ENV, "production-2026/run-manifests",
        IRREPLACEABLE_LIVE_EVIDENCE, "src/nfl_hybrid/production/run_2026.py (append-only)",
        "One manifest per attempted production batch, including fail-closed attempts.",
    ),
    AssetSpec(
        "production.recalibration_candidates", ARTIFACT_ROOT_ENV,
        "production-2026/recalibration-candidates",
        REGENERABLE_FROM_ESTATE, "scripts/generate_2026_recalibration_candidate.py",
        "Recalibration candidate seeds and manifests. Regenerated daily from the same evidence; "
        "never the active certified calibrator.",
    ),
    AssetSpec(
        "games_population_2026", ARTIFACT_ROOT_ENV, "games-population-2026",
        REGENERABLE_FROM_ESTATE, "scripts/update_2026_games_population.py",
        "The durable canonical BallDontLie 2026 games population. Rebuildable by replaying the "
        "BallDontLie evidence captures, which ARE irreplaceable.",
    ),
    AssetSpec(
        "live_market_2026", ARTIFACT_ROOT_ENV, "live-market-2026",
        REGENERABLE_FROM_ESTATE, "nfl_hybrid.data.bdl_market_bridge.materialize_bookmaker_quotes",
        "Materialized bookmaker quotes, keyed by capture manifest hash. Rebuildable by "
        "re-materializing the captures, which ARE irreplaceable.",
    ),
    AssetSpec(
        "public.wizard_nfl_pricing", ARTIFACT_ROOT_ENV, "public/wizardofodds/nfl-pricing",
        REGENERABLE_FROM_ESTATE, "scripts/export_wizard_nfl_pricing.py",
        "The exported wizard-nfl-pricing-v2 cards. Deterministically re-exportable from the run "
        "manifests and forecast ledger.",
    ),
)


def _roots(overrides: dict[str, str | None]) -> dict[str, str | None]:
    return {
        env: (overrides.get(env) or os.environ.get(env) or None)
        for env in (DATA_ROOT_ENV, ARTIFACT_ROOT_ENV, LIVE_ROOT_ENV)
    }


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _walk(path: Path, *, with_hashes: bool) -> dict:
    """Size/count (and optionally per-file hash) of one asset tree.

    Symlinks are never followed: a symlinked tree would be counted on the host
    that holds the link and again on the host that holds the target, which
    would make a migration comparison lie.
    """
    if path.is_file():
        entry = {"kind": "file", "file_count": 1, "bytes": path.stat().st_size}
        if with_hashes:
            entry["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}}
        return entry

    file_count = 0
    total_bytes = 0
    files: dict[str, dict] = {}
    for current_dir, dir_names, file_names in os.walk(path, followlinks=False):
        dir_names[:] = [d for d in dir_names if not os.path.islink(os.path.join(current_dir, d))]
        for name in sorted(file_names):
            full = Path(current_dir) / name
            if full.is_symlink() or not full.is_file():
                continue
            size = full.stat().st_size
            file_count += 1
            total_bytes += size
            if with_hashes:
                files[str(full.relative_to(path))] = {"bytes": size, "sha256": _sha256_file(full)}
    entry = {"kind": "dir", "file_count": file_count, "bytes": total_bytes}
    if with_hashes:
        entry["files"] = files
    return entry


def audit(*, root_overrides: dict[str, str | None] | None = None, with_hashes: bool = False) -> dict:
    roots = _roots(root_overrides or {})

    assets: list[dict] = []
    for spec in ASSETS:
        root = roots.get(spec.root_env)
        record = {
            "name": spec.name,
            "root_env": spec.root_env,
            "relative_path": spec.relative,
            "recovery_class": spec.recovery_class,
            "migration_required": spec.recovery_class in MIGRATION_REQUIRED_CLASSES,
            "producer": spec.producer,
            "why": spec.why,
        }
        if not root:
            record.update({"status": "ROOT_UNSET", "path": None, "file_count": 0, "bytes": 0})
            assets.append(record)
            continue
        path = (Path(root) / spec.relative).resolve() if spec.relative != "." else Path(root).resolve()
        record["path"] = str(path)
        if not path.exists():
            record.update({"status": "ABSENT", "file_count": 0, "bytes": 0})
        else:
            record.update({"status": "PRESENT", **_walk(path, with_hashes=with_hashes)})
        assets.append(record)

    by_class: dict[str, dict] = {}
    for record in assets:
        bucket = by_class.setdefault(
            record["recovery_class"], {"asset_count": 0, "present": 0, "file_count": 0, "bytes": 0}
        )
        bucket["asset_count"] += 1
        bucket["present"] += 1 if record["status"] == "PRESENT" else 0
        bucket["file_count"] += record.get("file_count", 0)
        bucket["bytes"] += record.get("bytes", 0)

    migration = [
        r for r in assets if r["migration_required"] and r["status"] == "PRESENT" and r["file_count"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "roots": roots,
        "hashed": bool(with_hashes),
        "assets": assets,
        "summary_by_recovery_class": by_class,
        "migration_required_assets": [r["name"] for r in migration],
        "migration_required_bytes": sum(r["bytes"] for r in migration),
        "absent_assets": [r["name"] for r in assets if r["status"] == "ABSENT"],
    }


def compare(source: dict, target: dict) -> dict:
    """Compare two audit manifests -- typically the Mac's and the Wizard
    server's -- asset by asset and, when both were produced with ``--hash``,
    file by file.

    Deliberately ASYMMETRIC: it asks whether the target holds everything the
    source holds, which is the only question that matters before deleting the
    source. Extra content on the target is reported but never a failure.
    """
    source_assets = {a["name"]: a for a in source.get("assets", [])}
    target_assets = {a["name"]: a for a in target.get("assets", [])}
    hashed = bool(source.get("hashed") and target.get("hashed"))

    results = []
    complete = True
    for name, src in sorted(source_assets.items()):
        tgt = target_assets.get(name)
        entry = {
            "name": name,
            "recovery_class": src["recovery_class"],
            "migration_required": src["migration_required"],
            "source_status": src["status"],
            "source_file_count": src.get("file_count", 0),
            "source_bytes": src.get("bytes", 0),
            "target_status": (tgt or {}).get("status", "NOT_AUDITED"),
            "target_file_count": (tgt or {}).get("file_count", 0),
            "target_bytes": (tgt or {}).get("bytes", 0),
        }
        if src["status"] != "PRESENT" or src.get("file_count", 0) == 0:
            entry["verdict"] = "NOTHING_TO_MIGRATE"
        elif tgt is None or tgt.get("status") != "PRESENT":
            entry["verdict"] = "MISSING_ON_TARGET"
        elif hashed:
            src_files = src.get("files", {})
            tgt_files = tgt.get("files", {})
            missing = sorted(set(src_files) - set(tgt_files))
            mismatched = sorted(
                name_ for name_ in set(src_files) & set(tgt_files)
                if src_files[name_]["sha256"] != tgt_files[name_]["sha256"]
            )
            entry["missing_files"] = missing[:50]
            entry["missing_file_count"] = len(missing)
            entry["mismatched_files"] = mismatched[:50]
            entry["mismatched_file_count"] = len(mismatched)
            entry["verdict"] = "HASH_VERIFIED" if not missing and not mismatched else "INCOMPLETE_OR_CORRUPT"
        elif tgt.get("file_count", 0) < src.get("file_count", 0) or tgt.get("bytes", 0) < src.get("bytes", 0):
            entry["verdict"] = "INCOMPLETE_ON_TARGET"
        else:
            entry["verdict"] = "SIZE_VERIFIED_ONLY"

        if entry["migration_required"] and entry["verdict"] not in (
            "NOTHING_TO_MIGRATE", "HASH_VERIFIED", "SIZE_VERIFIED_ONLY"
        ):
            complete = False
        results.append(entry)

    return {
        "schema_version": SCHEMA_VERSION + "-comparison",
        "hash_verified": hashed,
        "migration_complete": complete,
        "safe_to_delete_source": bool(
            complete and hashed
        ),
        "safe_to_delete_source_reason": (
            "every migration-required asset is byte-identical on the target"
            if complete and hashed
            else (
                "one or more migration-required assets are missing, incomplete or corrupt on the target"
                if not complete
                else "both manifests must be produced with --hash before any source deletion is authorized"
            )
        ),
        "assets": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=None, help=f"Override ${DATA_ROOT_ENV}.")
    parser.add_argument("--artifact-root", default=None, help=f"Override ${ARTIFACT_ROOT_ENV}.")
    parser.add_argument("--live-root", default=None, help=f"Override ${LIVE_ROOT_ENV}.")
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Hash every file (slower). Required before any source deletion is authorized.",
    )
    parser.add_argument("--output", default=None, help="Write the manifest here instead of stdout.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SOURCE_MANIFEST", "TARGET_MANIFEST"),
        default=None,
        help="Compare two previously written manifests instead of auditing this host.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.compare:
        source = json.loads(Path(args.compare[0]).read_text())
        target = json.loads(Path(args.compare[1]).read_text())
        report = compare(source, target)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered)
        print(rendered)
        return 0 if report["migration_complete"] else 2

    report = audit(
        root_overrides={
            DATA_ROOT_ENV: args.data_root,
            ARTIFACT_ROOT_ENV: args.artifact_root,
            LIVE_ROOT_ENV: args.live_root,
        },
        with_hashes=args.hash,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
        print(f"wrote {args.output}", file=sys.stderr)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
