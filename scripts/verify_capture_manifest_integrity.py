"""Resolve and verify the integrity of an official BallDontLie capture manifest.

A capture manifest has TWO distinct SHA256 integrity objects, and confusing
them is what produced the Week-1 TUE "hash discrepancy". They are different by
construction, so no capture can ever have one value for both:

  MANIFEST_CONTENT_SHA256
      ``sha256(deterministic_json(body))`` where ``body`` is the manifest
      WITHOUT its own ``manifest_sha256`` key, serialized compactly
      (``separators=(",", ":")``, ``sort_keys=True``, ``ensure_ascii=True``).
      This is what ``scripts/capture_bdl_2026_asof.py`` computes and then
      stores INSIDE the file as ``manifest_sha256``. It is insensitive to
      pretty-printing, so it identifies the CONTENT of the capture.

  MANIFEST_FILE_SHA256
      ``sha256sum manifest.json`` -- the bytes on disk. The file is written
      with ``indent=2`` AND it contains the ``manifest_sha256`` field that did
      not exist when MANIFEST_CONTENT_SHA256 was computed, so a
      self-referential hash cannot cover itself. This identifies the FILE.

This script does three things, and fails closed on any of them:

  1. SELF-VERIFICATION (always). Recompute MANIFEST_CONTENT_SHA256 from the
     body and compare it with the ``manifest_sha256`` the file carries. This
     is the check that detects a MUTATED capture, and nothing in production
     performed it before. It needs no external reference value at all.
  2. DECLARATION RESOLUTION. When the caller declares a hash, report which of
     the two objects it is -- ``MANIFEST_FILE_SHA256`` or
     ``MANIFEST_CONTENT_SHA256`` -- so a correctly recorded value is never
     rejected merely for referring to the other object, and an unrecognised
     value is never accepted.
  3. EVIDENCE. Emit both objects so the run record retains them verbatim.

It NEVER writes, reformats, normalizes or repairs the manifest. The capture is
immutable evidence; this only reads it.

Exit codes:
  0  integrity OK (self-verified, and any declared hash resolved to an object)
  2  usage error
  3  FAIL CLOSED -- self-verification failed, or a declared hash matched
     neither object

Usage:
  python scripts/verify_capture_manifest_integrity.py MANIFEST
  python scripts/verify_capture_manifest_integrity.py MANIFEST --declared-sha256 <hex>
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

SCHEMA_VERSION = "nfl-capture-manifest-integrity-v1"

SELF_HASH_KEY = "manifest_sha256"

OBJECT_FILE = "MANIFEST_FILE_SHA256"
OBJECT_CONTENT = "MANIFEST_CONTENT_SHA256"
OBJECT_NONE_DECLARED = "NONE_DECLARED"
OBJECT_NO_MATCH = "NO_MATCH"


def deterministic_json(obj: object) -> str:
    """Byte-for-byte the serialization ``scripts/capture_bdl_2026_asof.py``
    hashes. Kept identical on purpose: if these two ever diverge, every
    capture's self-hash stops verifying, which is exactly the loud failure
    that should happen."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def manifest_content_sha256(document: dict) -> str:
    """Recompute the capture's own content hash from a loaded manifest.

    The capture script inserts ``manifest_sha256`` AFTER hashing, so the hash
    is taken over the body without that key."""
    body = {key: value for key, value in document.items() if key != SELF_HASH_KEY}
    return sha256(deterministic_json(body).encode("utf-8")).hexdigest()


def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path, *, declared_sha256: str | None = None) -> dict:
    """Both integrity objects, the self-verification outcome, and which object
    a declared hash refers to. Never raises for a merely failing manifest --
    the caller decides, from ``integrity``, whether to proceed."""
    manifest_path = Path(manifest_path)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "capture_manifest_path": str(manifest_path),
    }

    if not manifest_path.is_file():
        return {
            **report,
            "integrity": "FAIL_CLOSED",
            "detail": f"capture manifest not found: {manifest_path}",
        }

    on_disk = file_sha256(manifest_path)
    report["manifest_file_sha256"] = on_disk

    try:
        document = json.loads(manifest_path.read_text())
    except ValueError as exc:
        return {
            **report,
            "integrity": "FAIL_CLOSED",
            "detail": f"capture manifest is not valid JSON: {exc}",
        }
    if not isinstance(document, dict):
        return {
            **report,
            "integrity": "FAIL_CLOSED",
            "detail": f"capture manifest is a {type(document).__name__}, not a JSON object",
        }

    recorded = document.get(SELF_HASH_KEY)
    recomputed = manifest_content_sha256(document)
    report.update(
        {
            "manifest_content_sha256": recomputed,
            "manifest_recorded_content_sha256": recorded,
            "capture_status": document.get("status"),
            "season": document.get("season"),
            "week": document.get("week"),
            "horizon": document.get("horizon"),
            "nominal_cutoff_utc": document.get("nominal_cutoff_utc"),
            "capture_uuid": document.get("capture_uuid"),
        }
    )

    if recorded is None:
        report["self_verified"] = False
        report["self_verification_detail"] = (
            f"the manifest carries no {SELF_HASH_KEY!r} field, so its content cannot be "
            "self-verified; it was not produced by scripts/capture_bdl_2026_asof.py"
        )
    elif str(recorded) != recomputed:
        report["self_verified"] = False
        report["self_verification_detail"] = (
            f"the manifest records {SELF_HASH_KEY}={recorded} but its body hashes to {recomputed}; "
            "the capture content was altered after it was written"
        )
    else:
        report["self_verified"] = True
        report["self_verification_detail"] = (
            f"{SELF_HASH_KEY} recomputes exactly; the capture content is intact"
        )

    if declared_sha256 is None or declared_sha256 == "":
        report["declared_sha256"] = None
        report["declared_sha256_object"] = OBJECT_NONE_DECLARED
    else:
        declared = str(declared_sha256).strip().lower()
        report["declared_sha256"] = declared
        if declared == on_disk:
            report["declared_sha256_object"] = OBJECT_FILE
        elif declared == recomputed:
            report["declared_sha256_object"] = OBJECT_CONTENT
        else:
            report["declared_sha256_object"] = OBJECT_NO_MATCH

    ok = bool(report["self_verified"]) and report["declared_sha256_object"] != OBJECT_NO_MATCH
    report["integrity"] = "OK" if ok else "FAIL_CLOSED"
    if report["declared_sha256_object"] == OBJECT_NO_MATCH:
        report["detail"] = (
            f"declared sha256 {report['declared_sha256']} matches neither integrity object of "
            f"{manifest_path}: file={on_disk} content={recomputed}. Refusing to run a certified "
            "card against a capture whose declared identity cannot be resolved."
        )
    elif not report["self_verified"]:
        report["detail"] = str(report["self_verification_detail"])
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", help="Path to the capture's manifest.json.")
    parser.add_argument(
        "--declared-sha256",
        default=None,
        help=(
            "The hash the caller believes identifies this capture. Resolved to whichever integrity "
            "object it is; a value matching neither fails closed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = verify(Path(args.manifest), declared_sha256=args.declared_sha256)

    # key=value lines first so the shell orchestrator can parse them without a
    # JSON dependency; the full report follows on stdout as evidence.
    print(f"capture_manifest={report['capture_manifest_path']}")
    print(f"capture_manifest_sha256={report.get('manifest_file_sha256', '')}")
    print(f"capture_manifest_content_sha256={report.get('manifest_content_sha256', '')}")
    print(f"capture_manifest_self_verified={str(report.get('self_verified', False)).lower()}")
    print(f"capture_manifest_declared_object={report.get('declared_sha256_object', OBJECT_NO_MATCH)}")
    print(f"capture_manifest_integrity={report['integrity']}")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))

    if report["integrity"] != "OK":
        print(f"FAIL CLOSED: {report.get('detail', 'capture manifest integrity check failed')}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
