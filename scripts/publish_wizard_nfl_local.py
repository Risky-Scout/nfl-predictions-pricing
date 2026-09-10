"""Publish one ``wizard-nfl-pricing-v2`` card onto the Wizard server's own
nginx-served NFL directory, atomically. Runs ON the Wizard server.

LOCAL FILESYSTEM TRANSPORT ONLY. This is the authoritative publication path
and it opens no socket at all: no FTP, no FTPS, no HTTP upload. The workflow
reaches the server over the already-proven SSH channel and then runs this
script there, so the bytes are placed by a local ``os.replace`` -- a genuinely
atomic rename within one filesystem, which the FTP transport in
``scripts/publish_sportsodds_nfl.py`` can only approximate. That FTP script is
left in place unchanged as a manual fallback; nothing in the authoritative
workflow calls it.

It generates no prediction, fits nothing, calibrates nothing, reads no
credential, and never edits the NFL HTML shell already deployed on the server.

VALIDATION HAPPENS BEFORE ANYTHING IS PLACED
  The card is validated with the EXISTING published-contract validator
  (:func:`scripts.publish_sportsodds_nfl.load_and_validate_json`) -- schema
  version, exact top-level and per-game key sets, season, horizon, instants,
  finite non-boolean numerics, the >=3-book floors, game_id uniqueness. There
  is no second contract here: the ``wizard-nfl-pricing-v2`` schema is frozen
  and this script only re-uses its gate. An invalid card is never staged and
  never placed.

ATOMIC PLACEMENT
  1. the validated bytes are written to a unique temporary file INSIDE the
     served directory (same filesystem -- a cross-filesystem rename is not
     atomic, so staging elsewhere and moving in would defeat the point);
  2. the temporary file is fsynced and its permissions set for nginx to read;
  3. ``os.replace`` renames it onto ``latest.json``.
  A reader therefore sees either the complete previous card or the complete
  new card, never a partial file and never a missing one.

  The previous card is retained as ``latest.json.prev`` before replacement, so
  a bad publication can be reverted on the server without a rerun.

FAIL CLOSED
  An unresolved web directory, a missing web directory, a non-writable web
  directory, a card that fails the published contract, or a post-write
  read-back that does not byte-match aborts with a non-zero exit and leaves
  the currently published card untouched.

Usage (on the Wizard server):
  python scripts/publish_wizard_nfl_local.py --json <artifact_root>/public/wizardofodds/nfl-pricing/latest.json
  python scripts/publish_wizard_nfl_local.py --json <path> --web-dir /var/www/.../NFL --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The FTP publisher is a sibling script, not an importable package module, so
# its validator is loaded by explicit path -- the same pattern the v2 exporter
# uses to reuse the v1 exporter. Importing it defines functions only; its
# ``main`` is guarded and no connection is ever opened by this script.
_FTP_PUBLISHER_PATH = REPO_ROOT / "scripts" / "publish_sportsodds_nfl.py"
_FTP_PUBLISHER_MODULE = "_sportsodds_nfl_publisher"


def _load_contract_validator():
    existing = sys.modules.get(_FTP_PUBLISHER_MODULE)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_FTP_PUBLISHER_MODULE, _FTP_PUBLISHER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the published-contract validator from {_FTP_PUBLISHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FTP_PUBLISHER_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_FTP_PUBLISHER_MODULE, None)
        raise
    return module


_contract = _load_contract_validator()

PublishError = _contract.PublishError
load_and_validate_json = _contract.load_and_validate_json

PUBLIC_JSON_NAME = "latest.json"
PREVIOUS_JSON_NAME = "latest.json.prev"

# The published card is served to the public by nginx: readable by everyone,
# writable only by the deploying account.
PUBLIC_FILE_MODE = 0o644


def _fail(message: str) -> None:
    raise PublishError(message)


def resolve_web_dir(explicit: str | None) -> Path:
    """The nginx-served NFL directory, never guessed.

    ``--web-dir`` wins; otherwise ``WIZARD_NFL_WEB_DIR`` from the environment
    (set by ``ops/wizard/nfl_production_layout.sh``, which itself reads the
    record written by ``ops/wizard/resolve_web_root.sh``). With neither, this
    fails closed rather than inventing a destination -- publishing a public
    card into a directory nginx does not serve is indistinguishable from not
    publishing at all, except that it reports success.
    """
    raw = explicit or os.environ.get("WIZARD_NFL_WEB_DIR")
    if not raw or not str(raw).strip():
        _fail(
            "the nginx-served NFL web directory is unresolved: pass --web-dir or export "
            "WIZARD_NFL_WEB_DIR (see ops/wizard/resolve_web_root.sh). Refusing to guess a "
            "publication destination."
        )
    web_dir = Path(str(raw).strip())
    if not web_dir.is_absolute():
        _fail(f"the web directory must be an absolute path: {web_dir}")
    if not web_dir.is_dir():
        _fail(
            f"the web directory does not exist: {web_dir}. This script never creates the served "
            "directory -- a directory nginx has not been configured to serve would silently swallow "
            "the published card."
        )
    return web_dir


def publish(
    *,
    json_path: Path,
    web_dir: Path,
    dry_run: bool = False,
) -> dict:
    payload_bytes, summary = load_and_validate_json(json_path)
    digest = sha256(payload_bytes).hexdigest()

    target = web_dir / PUBLIC_JSON_NAME
    previous = web_dir / PREVIOUS_JSON_NAME

    existing_digest = sha256(target.read_bytes()).hexdigest() if target.is_file() else None

    report = {
        "status": "DRY_RUN" if dry_run else "PUBLISHED",
        "source": str(json_path),
        "web_dir": str(web_dir),
        "target": str(target),
        "bytes": len(payload_bytes),
        "sha256": digest,
        "previous_sha256": existing_digest,
        "unchanged": existing_digest == digest,
        **summary,
    }
    if dry_run:
        return report

    if not os.access(web_dir, os.W_OK | os.X_OK):
        _fail(f"the web directory is not writable by {os.getlogin() if hasattr(os, 'getlogin') else 'this account'}: {web_dir}")

    # The temporary file lives in the SERVED directory so the rename below is
    # a same-filesystem rename, which is atomic. Staging on another filesystem
    # and moving in would degrade to a copy and could expose a partial file.
    tmp = web_dir / f".{PUBLIC_JSON_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, PUBLIC_FILE_MODE)

        if target.is_file():
            # Retained BEFORE the replacement, so a revert is always possible
            # without rerunning the pipeline.
            try:
                previous.write_bytes(target.read_bytes())
                os.chmod(previous, PUBLIC_FILE_MODE)
                report["previous_retained"] = str(previous)
            except OSError as exc:
                _fail(f"could not retain the currently published card as {previous}: {exc}")

        os.replace(tmp, target)
    except PublishError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        _fail(f"atomic publication failed: {exc}")

    # Read back from the served path: the proof is what is now on disk, not
    # what this process believes it wrote.
    published = target.read_bytes()
    if sha256(published).hexdigest() != digest:
        _fail(
            f"post-publication read-back of {target} does not match the validated bytes "
            "(sha256 mismatch); the served card is not the card that was validated"
        )
    report["verified_sha256"] = sha256(published).hexdigest()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", required=True, help="The exported wizard-nfl-pricing-v2 card to publish.")
    parser.add_argument(
        "--web-dir",
        default=None,
        help="The nginx-served NFL directory. Defaults to $WIZARD_NFL_WEB_DIR; never guessed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the card and report the intended destination without writing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        web_dir = resolve_web_dir(args.web_dir)
        report = publish(json_path=Path(args.json), web_dir=web_dir, dry_run=args.dry_run)
    except PublishError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
