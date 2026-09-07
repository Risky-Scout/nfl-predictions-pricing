"""Publish the NFL Predictive Pricing page and its public feed over FTP/FTPS.

PUBLICATION TRANSPORT ONLY. This script uploads two already-created files --
the static page ``web/sportsodds/nfl/index.html`` and one
``wizard-nfl-pricing-v2`` JSON card written by
``scripts/export_wizard_nfl_pricing.py`` -- to the SportsOdds web host that
serves the public ``/tools/odds-scanner/predictions/NFL/`` page.

It does NOT generate predictions, fit or calibrate a model, compute or alter a
predicted margin/total, capture BallDontLie data, reconstruct a market, call
any sportsbook or any other HTTP API, read a legacy season output artifact, or
write anything into the repository. Python standard library only.

Credentials live in the environment, never here and never on disk:

    SPORTSODDS_FTP_HOST        host name of the SportsOdds FTP endpoint
    SPORTSODDS_FTP_USER        FTP account user
    SPORTSODDS_FTP_PASSWORD    FTP account password (never printed, never
                               logged, never written to a file, and redacted
                               from this module's own object reprs)
    SPORTSODDS_FTP_REMOTE_DIR  AUTHORITATIVE remote directory -- see below
    SPORTSODDS_FTP_MODE        exactly ``ftp`` or ``ftps``
    SPORTSODDS_FTP_PORT        optional, default 21

The remote directory is configuration, not something to be inferred. The
public page lives at ``/tools/odds-scanner/predictions/NFL/`` and nginx serves
it from a document root on the SportsOdds host, but the FTP account may well be
chrooted so that root's absolute filesystem path does not exist from the FTP
session's point of view. Nothing here derives, guesses or prepends a server
root: ``SPORTSODDS_FTP_REMOTE_DIR`` is used exactly as supplied.

Remote file names are fixed and are never taken from the command line:

    latest.json    the public wizard-nfl-pricing-v2 feed the page fetches
    index.html     the page itself

Every upload lands on a unique temporary remote name first and is then renamed
onto its final name, so a reader either sees the previous complete file or the
new complete file -- best-effort atomic replacement over a protocol that
offers no true atomic write. For the default both-file publication the JSON is
validated and uploaded first and the page second, so the page never goes live
pointing at a feed that is not there yet; a later data-only refresh is
``--json-only``.

The JSON is fully validated locally BEFORE any socket is opened: schema
version, exact top-level key set, exact per-game key set, season, horizon,
timestamps, finite non-boolean numerics, eligible-book floors and game_id
uniqueness. Malformed data is never uploaded. ``--dry-run`` validates
everything and prints the intended target while making no network connection
at all.

Usage:
    python scripts/publish_sportsodds_nfl.py --json <path/to/latest.json>
    python scripts/publish_sportsodds_nfl.py --json <path> --dry-run
    python scripts/publish_sportsodds_nfl.py --json <path> --json-only
    python scripts/publish_sportsodds_nfl.py --html-only
"""
from __future__ import annotations

import argparse
import ftplib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# published contract -- mirrored from scripts/export_wizard_nfl_pricing.py.
# This module is a transport gate, so it re-checks the contract itself rather
# than importing the exporter: the file on disk is what will be uploaded, and
# it is validated as data, whatever produced it.
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "wizard-nfl-pricing-v2"
SUPPORTED_SEASON = 2026
ALLOWED_HORIZONS = ("TUE", "FRI")
MINIMUM_ELIGIBLE_BOOKS = 3

TOP_LEVEL_KEYS = ("schema_version", "season", "week", "horizon", "generated_at_utc", "games")
GAME_KEYS = (
    "game_id",
    "kickoff_utc",
    "away_team",
    "home_team",
    "predicted_home_margin",
    "predicted_game_total",
    "market_home_spread",
    "market_total",
    "market_as_of_utc",
    "market_ats_book_count",
    "market_total_book_count",
)
NUMERIC_GAME_KEYS = ("predicted_home_margin", "predicted_game_total", "market_home_spread", "market_total")
BOOK_COUNT_KEYS = ("market_ats_book_count", "market_total_book_count")
INSTANT_GAME_KEYS = ("kickoff_utc", "market_as_of_utc")
TEAM_KEYS = ("away_team", "home_team")

# --------------------------------------------------------------------------- #
# transport configuration
# --------------------------------------------------------------------------- #
ENV_HOST = "SPORTSODDS_FTP_HOST"
ENV_USER = "SPORTSODDS_FTP_USER"
ENV_PASSWORD = "SPORTSODDS_FTP_PASSWORD"
ENV_REMOTE_DIR = "SPORTSODDS_FTP_REMOTE_DIR"
ENV_MODE = "SPORTSODDS_FTP_MODE"
ENV_PORT = "SPORTSODDS_FTP_PORT"

REQUIRED_ENV_VARS = (ENV_HOST, ENV_USER, ENV_PASSWORD, ENV_REMOTE_DIR, ENV_MODE)

MODE_FTP = "ftp"
MODE_FTPS = "ftps"
ALLOWED_FTP_MODES = (MODE_FTP, MODE_FTPS)

DEFAULT_FTP_PORT = 21
FTP_TIMEOUT_SECONDS = 30

REMOTE_JSON_NAME = "latest.json"
REMOTE_HTML_NAME = "index.html"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML_SOURCE = REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html"

_REDACTED = "<redacted>"


class PublishError(RuntimeError):
    """Fail-closed publication error. Never carries the password, and never
    leaves a malformed file uploaded: validation completes before any socket
    is opened, and a failure mid-publication exits non-zero."""


def _fail(message: str) -> None:
    raise PublishError(message)


# --------------------------------------------------------------------------- #
# JSON validation -- entirely local, always before the network
# --------------------------------------------------------------------------- #
def _reject_json_constant(name: str) -> None:
    """``json.loads`` accepts the JavaScript-only literals ``NaN``,
    ``Infinity`` and ``-Infinity`` by default and hands back float nan/inf.
    A public price is never one of those, so the parse itself refuses them."""
    _fail(f"public JSON contains the non-finite literal {name} -- refusing to publish")


def _validate_numeric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a numeric value, got {value!r}")
    if not math.isfinite(value):
        _fail(f"{label} must be finite, got {value!r}")
    return float(value)


def _validate_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer, got {value!r}")
    return value


def _validate_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string, got {value!r}")
    return value


def _validate_instant(value: Any, *, label: str) -> datetime:
    """Absolute instants only: a timezone-naive or unparseable timestamp is
    rejected rather than assumed to be UTC."""
    text = _validate_non_empty_str(value, label=label).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        _fail(f"{label} is not a parseable timestamp: {value!r}")
        raise AssertionError("unreachable")  # pragma: no cover
    if parsed.tzinfo is None:
        _fail(f"{label} is not timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _validate_exact_keys(payload: Mapping[str, Any], expected: Sequence[str], *, label: str) -> None:
    """Exact key set: a missing public field and an unexpected extra public
    field are both schema drift, and neither is published."""
    actual = set(payload.keys())
    missing = [key for key in expected if key not in actual]
    if missing:
        _fail(f"{label} is missing required key(s) {missing}")
    unexpected = sorted(actual.difference(expected))
    if unexpected:
        _fail(f"{label} carries unexpected key(s) {unexpected} -- {SCHEMA_VERSION} is a closed contract")


def validate_public_payload(payload: Any) -> dict:
    """Validate one decoded wizard-nfl-pricing-v2 card. Returns a small
    summary used only for operator output; raises PublishError on anything
    that must not be published."""
    if not isinstance(payload, dict):
        _fail(f"public JSON must be a JSON object, got {type(payload).__name__}")

    _validate_exact_keys(payload, TOP_LEVEL_KEYS, label="public JSON")

    if payload["schema_version"] != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION!r}, got {payload['schema_version']!r}")

    season = _validate_int(payload["season"], label="season")
    if season != SUPPORTED_SEASON:
        _fail(f"season must be {SUPPORTED_SEASON}, got {season}")

    week = _validate_int(payload["week"], label="week")
    if week < 1:
        _fail(f"week must be a positive integer NFL week, got {week}")

    horizon = payload["horizon"]
    if horizon not in ALLOWED_HORIZONS:
        _fail(f"horizon must be one of {ALLOWED_HORIZONS}, got {horizon!r}")

    generated_at = _validate_instant(payload["generated_at_utc"], label="generated_at_utc")

    games = payload["games"]
    if not isinstance(games, list):
        _fail(f"games must be a list, got {type(games).__name__}")
    if not games:
        _fail("games is empty -- refusing to publish a pricing card with no games")

    seen_game_ids: set[str] = set()
    for index, game in enumerate(games):
        position = f"games[{index}]"
        if not isinstance(game, dict):
            _fail(f"{position} must be a JSON object, got {type(game).__name__}")
        _validate_exact_keys(game, GAME_KEYS, label=position)

        game_id = _validate_non_empty_str(game["game_id"], label=f"{position}.game_id")
        if game_id in seen_game_ids:
            _fail(f"duplicate game_id {game_id!r} in games -- refusing to publish an ambiguous card")
        seen_game_ids.add(game_id)

        for key in TEAM_KEYS:
            _validate_non_empty_str(game[key], label=f"{position}.{key}")
        if game["home_team"] == game["away_team"]:
            _fail(f"{position}: home_team equals away_team ({game['home_team']!r})")

        for key in NUMERIC_GAME_KEYS:
            _validate_numeric(game[key], label=f"{position}.{key}")

        for key in BOOK_COUNT_KEYS:
            count = _validate_int(game[key], label=f"{position}.{key}")
            if count < MINIMUM_ELIGIBLE_BOOKS:
                _fail(
                    f"{position}.{key} is {count}, below the certified minimum of "
                    f"{MINIMUM_ELIGIBLE_BOOKS} eligible books"
                )

        for key in INSTANT_GAME_KEYS:
            _validate_instant(game[key], label=f"{position}.{key}")

    return {
        "season": season,
        "week": week,
        "horizon": horizon,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "game_count": len(games),
    }


def load_and_validate_json(path: Path) -> tuple[bytes, dict]:
    """Read the card as bytes (exactly what will be uploaded -- the published
    bytes are never re-serialized here) and validate its decoded form."""
    if not path.is_file():
        _fail(f"public JSON not found: {path}")
    payload_bytes = path.read_bytes()
    if not payload_bytes.strip():
        _fail(f"public JSON is empty: {path}")
    try:
        decoded = json.loads(payload_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    except UnicodeDecodeError as exc:
        _fail(f"public JSON is not valid UTF-8: {path} ({exc})")
        raise AssertionError("unreachable")  # pragma: no cover
    except json.JSONDecodeError as exc:
        _fail(f"public JSON is not valid JSON: {path} ({exc})")
        raise AssertionError("unreachable")  # pragma: no cover
    summary = validate_public_payload(decoded)
    return payload_bytes, summary


# --------------------------------------------------------------------------- #
# HTML validation -- the page must be the self-contained page this repo owns
# --------------------------------------------------------------------------- #
_REQUIRED_HTML_MARKERS = (
    "<title>NFL Predictive Pricing | Wizard of Odds</title>",
    "NFL Predictive Pricing",
    "./latest.json",
)


def load_and_validate_html(path: Path) -> bytes:
    if not path.is_file():
        _fail(f"page source not found: {path}")
    payload_bytes = path.read_bytes()
    if not payload_bytes.strip():
        _fail(f"page source is empty: {path}")
    try:
        text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"page source is not valid UTF-8: {path} ({exc})")
        raise AssertionError("unreachable")  # pragma: no cover
    missing = [marker for marker in _REQUIRED_HTML_MARKERS if marker not in text]
    if missing:
        _fail(f"page source {path} does not look like the NFL Predictive Pricing page (missing {missing})")
    return payload_bytes


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class FtpConfig:
    """Transport settings. ``__repr__``/``__str__`` redact the password so it
    cannot reach a log line, a traceback frame summary or an operator's
    terminal through an accidental print of this object."""

    __slots__ = ("host", "port", "user", "password", "remote_dir", "mode")

    def __init__(self, *, host: str, port: int, user: str, password: str, remote_dir: str, mode: str) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.remote_dir = remote_dir
        self.mode = mode

    def __repr__(self) -> str:
        return (
            f"FtpConfig(host={self.host!r}, port={self.port!r}, user={self.user!r}, "
            f"remote_dir={self.remote_dir!r}, mode={self.mode!r}, password={_REDACTED})"
        )

    __str__ = __repr__

    def remote_path(self, filename: str) -> str:
        return f"{self.remote_dir.rstrip('/')}/{filename}"


def _validate_remote_dir(raw: str) -> str:
    """``SPORTSODDS_FTP_REMOTE_DIR`` is authoritative and is used verbatim. No
    server filesystem root is inferred and no document-root prefix is ever
    prepended; the only rejections are values that cannot address a directory
    or that would smuggle a second FTP command."""
    remote_dir = raw.strip()
    if not remote_dir:
        _fail(f"{ENV_REMOTE_DIR} must not be blank -- the remote directory is configuration, never inferred")
    if any(character in remote_dir for character in ("\r", "\n", "\0")):
        _fail(f"{ENV_REMOTE_DIR} must not contain control characters")
    if "\\" in remote_dir:
        _fail(f"{ENV_REMOTE_DIR} must use forward slashes, got {remote_dir!r}")
    components = [component for component in remote_dir.split("/") if component]
    if not components:
        _fail(f"{ENV_REMOTE_DIR} must name at least one directory component, got {remote_dir!r}")
    for component in components:
        if component in (".", ".."):
            _fail(f"{ENV_REMOTE_DIR} must not contain '.' or '..' components, got {remote_dir!r}")
    return remote_dir.rstrip("/") or "/"


def _validate_port(raw: str | None) -> int:
    if raw is None or not raw.strip():
        return DEFAULT_FTP_PORT
    text = raw.strip()
    if not text.isdigit():
        _fail(f"{ENV_PORT} must be a positive integer port, got {raw!r}")
    port = int(text)
    if not 1 <= port <= 65535:
        _fail(f"{ENV_PORT} must be between 1 and 65535, got {port}")
    return port


def load_config(env: Mapping[str, str] | None = None, *, require_password: bool = True) -> FtpConfig:
    """Build the transport configuration from the environment.

    ``require_password=False`` is the dry-run path: it validates every other
    required setting and never inspects, prints or transmits the password
    contents, because a dry run opens no connection to authenticate against."""
    environment = os.environ if env is None else env

    def is_configured(name: str) -> bool:
        value = environment.get(name)
        if name == ENV_PASSWORD:
            # Presence only. The contents are never inspected, so a password
            # made entirely of spaces is still the operator's business.
            return bool(value)
        return bool((value or "").strip())

    required = REQUIRED_ENV_VARS if require_password else tuple(n for n in REQUIRED_ENV_VARS if n != ENV_PASSWORD)
    missing = [name for name in required if not is_configured(name)]
    if missing:
        _fail(f"missing required environment variable(s): {', '.join(missing)}")

    # Exact membership, deliberately un-normalised: "FTP", " ftp" and "ftpes"
    # are configuration mistakes, not dialects to be guessed at.
    mode = environment.get(ENV_MODE) or ""
    if mode not in ALLOWED_FTP_MODES:
        _fail(f"{ENV_MODE} must be exactly one of {ALLOWED_FTP_MODES}, got {mode!r}")

    return FtpConfig(
        host=(environment.get(ENV_HOST) or "").strip(),
        port=_validate_port(environment.get(ENV_PORT)),
        user=(environment.get(ENV_USER) or "").strip(),
        password=environment.get(ENV_PASSWORD) or "",
        remote_dir=_validate_remote_dir(environment.get(ENV_REMOTE_DIR) or ""),
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# FTP transport
# --------------------------------------------------------------------------- #
def connect(config: FtpConfig):
    """Open and authenticate one control connection. ``ftp`` uses
    ``ftplib.FTP``; ``ftps`` uses ``ftplib.FTP_TLS`` and switches the data
    channel to TLS with ``prot_p()`` after login. Both use a timeout so a
    silent host cannot hang a scheduled publication forever."""
    if config.mode == MODE_FTPS:
        client = ftplib.FTP_TLS(timeout=FTP_TIMEOUT_SECONDS)
    elif config.mode == MODE_FTP:
        client = ftplib.FTP(timeout=FTP_TIMEOUT_SECONDS)
    else:  # pragma: no cover - load_config already enforces the allowlist
        _fail(f"unsupported FTP mode {config.mode!r}")

    client.connect(host=config.host, port=config.port, timeout=FTP_TIMEOUT_SECONDS)
    client.login(user=config.user, passwd=config.password)
    if config.mode == MODE_FTPS:
        client.prot_p()
    return client


def ensure_remote_dir(client, remote_dir: str) -> None:
    """Walk to the configured remote directory, creating only the components
    that are missing. The path is taken from configuration exactly as given:
    an absolute value is walked from the session root, a relative value from
    the login directory, and no server root is ever assumed."""
    if remote_dir.startswith("/"):
        client.cwd("/")
    for component in [part for part in remote_dir.split("/") if part]:
        try:
            client.cwd(component)
        except ftplib.error_perm:
            client.mkd(component)
            client.cwd(component)


def _temporary_remote_name(final_name: str) -> str:
    return f".{final_name}.publish-{uuid.uuid4().hex}.tmp"


def upload_bytes(client, payload_bytes: bytes, final_name: str) -> str:
    """Best-effort atomic replacement: store to a unique temporary remote name
    inside the target directory, then rename onto the fixed public name, so a
    reader never sees a half-written file. The temporary file is removed if the
    store or the rename fails."""
    temporary_name = _temporary_remote_name(final_name)
    try:
        client.storbinary(f"STOR {temporary_name}", BytesIO(payload_bytes))
        try:
            client.rename(temporary_name, final_name)
        except ftplib.error_perm:
            # Some servers refuse RNTO onto an existing path; fall back to
            # delete-then-rename, which is still far shorter than a full upload.
            client.delete(final_name)
            client.rename(temporary_name, final_name)
    except ftplib.all_errors:
        _discard_remote(client, temporary_name)
        raise
    return temporary_name


def _discard_remote(client, name: str) -> None:
    try:
        client.delete(name)
    except Exception:  # pragma: no cover - cleanup is best effort by design
        pass


# --------------------------------------------------------------------------- #
# publication
# --------------------------------------------------------------------------- #
def _print(stream, message: str) -> None:
    print(message, file=stream)


def _describe_plan(config: FtpConfig, uploads: Sequence[tuple[str, Path]], stream) -> None:
    _print(stream, f"host: {config.host}:{config.port}")
    _print(stream, f"mode: {config.mode}")
    _print(stream, f"user: {config.user}")
    _print(stream, f"password: {_REDACTED}")
    _print(stream, f"remote dir: {config.remote_dir}")
    for remote_name, source_path in uploads:
        _print(stream, f"remote target: {config.remote_path(remote_name)} <- {source_path}")


def publish(
    *,
    json_path: Path | None,
    html_path: Path,
    publish_json: bool,
    publish_html: bool,
    dry_run: bool,
    env: Mapping[str, str] | None = None,
    stream=None,
) -> int:
    """Validate locally, then (unless this is a dry run) upload. Returns the
    process exit status: 0 only when everything requested actually landed."""
    out = sys.stdout if stream is None else stream

    if not publish_json and not publish_html:  # pragma: no cover - CLI cannot express this
        _fail("nothing to publish")

    config = load_config(env, require_password=not dry_run)

    # Everything is validated before a socket exists, and for the default
    # both-file publication everything is validated before EITHER upload.
    uploads: list[tuple[str, Path, bytes]] = []
    json_summary: dict | None = None
    if publish_json:
        if json_path is None:
            _fail("--json <PATH> is required when publishing the public JSON feed")
        payload_bytes, json_summary = load_and_validate_json(json_path)
        uploads.append((REMOTE_JSON_NAME, json_path, payload_bytes))
    if publish_html:
        uploads.append((REMOTE_HTML_NAME, html_path, load_and_validate_html(html_path)))

    if json_summary is not None:
        _print(
            out,
            "validated public JSON: season={season} week={week} horizon={horizon} "
            "generated_at_utc={generated_at_utc} games={game_count}".format(**json_summary),
        )
    _print(out, f"validated page source: {html_path}" if publish_html else "page publication skipped (--json-only)")

    plan = [(remote_name, source_path) for remote_name, source_path, _ in uploads]

    if dry_run:
        _print(out, "DRY RUN: no network connection will be opened")
        _describe_plan(config, plan, out)
        return 0

    _describe_plan(config, plan, out)

    client = connect(config)
    try:
        ensure_remote_dir(client, config.remote_dir)
        for remote_name, source_path, payload_bytes in uploads:
            try:
                upload_bytes(client, payload_bytes, remote_name)
            except ftplib.all_errors as exc:
                _fail(f"upload of {config.remote_path(remote_name)} failed: {exc}")
            _print(out, f"published {config.remote_path(remote_name)} ({len(payload_bytes)} bytes)")
    finally:
        try:
            client.quit()
        except Exception:  # pragma: no cover - a dead control channel is not a publication failure
            try:
                client.close()
            except Exception:
                pass

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the NFL Predictive Pricing page and public feed to SportsOdds over FTP/FTPS.",
        epilog=(
            "Credentials and the remote directory come from the environment: "
            f"{', '.join(REQUIRED_ENV_VARS)} (plus optional {ENV_PORT}). "
            f"Remote file names are fixed: {REMOTE_JSON_NAME} and {REMOTE_HTML_NAME}."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help=f"path to the {SCHEMA_VERSION} card to publish as {REMOTE_JSON_NAME}",
    )
    parser.add_argument(
        "--html",
        dest="html_path",
        default=str(DEFAULT_HTML_SOURCE),
        help=f"page source to publish as {REMOTE_HTML_NAME} (default: {DEFAULT_HTML_SOURCE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the intended target without opening any network connection",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--html-only", action="store_true", help=f"publish only {REMOTE_HTML_NAME}")
    selection.add_argument("--json-only", action="store_true", help=f"publish only {REMOTE_JSON_NAME}")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    publish_json = not args.html_only
    publish_html = not args.json_only

    try:
        return publish(
            json_path=Path(args.json_path) if args.json_path else None,
            html_path=Path(args.html_path),
            publish_json=publish_json,
            publish_html=publish_html,
            dry_run=args.dry_run,
        )
    except PublishError as exc:
        print(f"PUBLISH FAILED: {exc}", file=sys.stderr)
        return 2
    except ftplib.all_errors as exc:
        print(f"PUBLISH FAILED: FTP transport error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
