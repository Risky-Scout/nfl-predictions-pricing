"""Bounded storage: prune only what is genuinely replaceable.

WHY A PRUNER AT ALL. Per-game CLOSE snapshots mean the sweep runs often, and
each execution materializes an intermediate live-market quotes artifact keyed
by capture hash. Those accumulate. Left alone the estate grows without bound;
deleted carelessly it destroys evidence that cannot be recovered at any price.

WHAT IS NEVER TOUCHED, unconditionally:

    forecast-ledger/          the forecast of record. Irreplaceable.
    evaluation-ledger/        the same, for evaluation.
    run-manifests/            what ran, with what inputs.
    snapshot-performance-ledger/  the season record and its results.
    published-close-state/    which game was frozen at which bytes.
    recalibration*/           candidates, pointer, promotion events, policy lock.
    fix8-official-oof-calibration-2026/  the certified baseline seed.
    games-population-2026/    the durable canonical population.
    public/                   exported cards and their immutable archives.
    live-observation-log/     raw provider captures (under the DATA root,
                              which this tool never even looks at).

WHAT IS ELIGIBLE, and only after proving it is stale:

    live-market-2026/<provider>/.../manifest_sha256=<hash>/
        the materialized per-book quotes parquet for one capture. Pure
        derivation: re-materializable from the raw capture that produced it,
        which lives in the immutable observation log and is not touched here.

FAIL-CLOSED AND DRY BY DEFAULT. Nothing is deleted without ``--apply``.
Nothing is deleted whose capture hash is still referenced by any run manifest
inside the retention window, and nothing younger than the retention age. A
path that does not resolve to an eligible namespace is refused rather than
skipped, so a mis-typed root can never turn this into a general-purpose
deleter.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The ONE prunable namespace. Adding to this list is a deliberate act.
PRUNABLE_NAMESPACES = ("live-market-2026",)

# Namespaces that must never be reachable from a prune decision. Checked
# explicitly so a future refactor of the path layout cannot quietly widen the
# blast radius.
PROTECTED_NAMESPACES = (
    "production-2026",
    "fix8-official-oof-calibration-2026",
    "games-population-2026",
    "public",
    "live-observation-log",
    "chronological-oof-2020-2025",
    "chronological-calibration-2020-2025",
    "invalidated",
)

DEFAULT_RETENTION_DAYS = 45


class PruneRefused(RuntimeError):
    """Fail-closed pruning error."""


def referenced_capture_hashes(artifact_root: Path) -> set[str]:
    """Every capture hash any run manifest says it priced from.

    A materialized artifact that a recorded run used is part of that run's
    reproducibility story for as long as we keep it, so it is never pruned
    while referenced.
    """
    manifests = artifact_root / "production-2026" / "run-manifests"
    hashes: set[str] = set()
    if not manifests.is_dir():
        return hashes
    for path in sorted(manifests.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # An unreadable manifest is not permission to delete anything.
            raise PruneRefused(f"run manifest is unreadable, refusing to prune: {path}")
        value = (manifest.get("input_hashes") or {}).get("live_market_capture_sha256")
        if isinstance(value, str) and value:
            hashes.add(value)
    return hashes


def _assert_prunable(path: Path, artifact_root: Path) -> None:
    relative = path.relative_to(artifact_root)
    if not relative.parts or relative.parts[0] not in PRUNABLE_NAMESPACES:
        raise PruneRefused(
            f"{path} is not inside a prunable namespace {PRUNABLE_NAMESPACES} -- refusing to delete"
        )
    if any(part in PROTECTED_NAMESPACES for part in relative.parts):
        raise PruneRefused(f"{path} touches a protected namespace -- refusing to delete")


def plan(artifact_root: Path, *, retention_days: int, now_epoch: float) -> dict:
    artifact_root = Path(artifact_root)
    referenced = referenced_capture_hashes(artifact_root)
    cutoff_epoch = now_epoch - retention_days * 86400

    prunable, kept = [], []
    for namespace in PRUNABLE_NAMESPACES:
        root = artifact_root / namespace
        if not root.is_dir():
            continue
        for directory in sorted(p for p in root.rglob("manifest_sha256=*") if p.is_dir()):
            digest = directory.name.split("=", 1)[1]
            age_ok = directory.stat().st_mtime < cutoff_epoch
            if digest in referenced:
                kept.append({"path": str(directory), "reason": "REFERENCED_BY_RUN_MANIFEST"})
            elif not age_ok:
                kept.append({"path": str(directory), "reason": "WITHIN_RETENTION_WINDOW"})
            else:
                _assert_prunable(directory, artifact_root)
                prunable.append({
                    "path": str(directory),
                    "bytes": sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()),
                })
    return {
        "artifact_root": str(artifact_root),
        "retention_days": retention_days,
        "referenced_capture_hashes": len(referenced),
        "prunable": prunable,
        "kept": kept,
        "reclaimable_bytes": sum(entry["bytes"] for entry in prunable),
    }


def apply(decision: dict, artifact_root: Path) -> list[str]:
    removed = []
    for entry in decision["prunable"]:
        path = Path(entry["path"])
        _assert_prunable(path, Path(artifact_root))
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def main(argv: list[str] | None = None) -> int:
    import time

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument("--apply", action="store_true", help="Actually delete. Omitted, this is a dry run.")
    args = parser.parse_args(argv)

    try:
        decision = plan(
            Path(args.artifact_root), retention_days=args.retention_days, now_epoch=time.time()
        )
        decision["applied"] = apply(decision, Path(args.artifact_root)) if args.apply else []
    except PruneRefused as exc:
        print(json.dumps({"status": "REFUSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    decision["status"] = "APPLIED" if args.apply else "DRY_RUN"
    print(json.dumps(decision, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
