"""Verify the PUBLIC NFL Predictive Pricing page and feed over HTTPS.

Runs on the GitHub-hosted runner, from outside the server, against the real
public NFL predictions URL on the Wizard SportsOdds host (see
:data:`DEFAULT_PAGE_URL`, overridable with ``--page-url``).

It is a read-only observer: it fetches the page and its ``latest.json`` feed,
checks the page still points at the feed, and validates the feed against the
frozen published contract. It publishes nothing, deploys nothing, writes
nothing to the server, uses no credential and never calls BallDontLie or any
sportsbook. Python standard library only (``urllib``), so it needs no
dependency install on the verifying runner.

WHY VERIFY FROM OUTSIDE
  A server-side atomic rename proves the bytes landed on the served path. It
  does not prove the public site actually serves them -- a caching layer, a
  wrong document root, a permissions problem or an nginx rule can all sit
  between the file and the reader. This check closes that gap by reading what
  the public reads.

THE FEED IS VALIDATED WITH THE EXISTING CONTRACT
  :func:`scripts.publish_sportsodds_nfl.validate_public_payload` -- the same
  frozen ``wizard-nfl-pricing-v2`` gate the publisher applies. No second
  schema is defined here.

OPTIONAL EXACT-CARD PROOF
  ``--expect-sha256`` asserts the served feed's bytes hash to exactly the card
  the run published, which is what turns "a valid card is live" into "OUR card
  is live". ``--expect-generated-at``/``--expect-horizon``/``--expect-week``
  assert the card's identity fields for the same reason at a coarser grain.

DAILY vs CERTIFIED USE
  The daily maintenance pass runs this with no expectations: it is a liveness
  and contract check of whatever is currently published. The certified TUE/FRI
  pass runs it with ``--expect-sha256`` immediately after deployment.

Usage:
  python scripts/verify_public_nfl_feed.py
  python scripts/verify_public_nfl_feed.py --expect-sha256 <hex> --require-fresh-hours 96
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PAGE_URL = "https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/NFL/"  # pragma: allowlist secret
DEFAULT_JSON_NAME = "latest.json"
USER_AGENT = "nfl-predictions-pricing-public-verifier/1"

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
validate_public_payload = _contract.validate_public_payload
PublishError = _contract.PublishError


class PublicVerificationError(RuntimeError):
    """The public page or feed did not verify. Always a hard failure -- a
    deployment that cannot be observed publicly is not a deployment."""


def _fetch(url: str, *, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
            return response.getcode(), dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        raise PublicVerificationError(f"GET {url} -> HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise PublicVerificationError(f"GET {url} failed: {exc.reason}") from exc
    except OSError as exc:
        raise PublicVerificationError(f"GET {url} failed: {exc}") from exc


def verify(
    *,
    page_url: str = DEFAULT_PAGE_URL,
    json_name: str = DEFAULT_JSON_NAME,
    expect_sha256: str | None = None,
    expect_generated_at: str | None = None,
    expect_horizon: str | None = None,
    expect_week: int | None = None,
    require_fresh_hours: float | None = None,
    timeout: float = 30.0,
    now_utc: datetime | None = None,
) -> dict:
    page_url = page_url if page_url.endswith("/") else page_url + "/"
    feed_url = page_url + json_name

    page_status, _page_headers, page_bytes = _fetch(page_url, timeout=timeout)
    page_text = page_bytes.decode("utf-8", errors="replace")
    # The page must still be the NFL pricing shell AND must still point at the
    # feed this script verified. A page that stopped referencing latest.json
    # is publishing nothing, however valid the feed is.
    page_checks = {
        "http_status": page_status,
        "bytes": len(page_bytes),
        "references_feed": (json_name in page_text),
        "is_nfl_pricing_page": ("NFL Predictive Pricing" in page_text),
    }
    if not page_checks["is_nfl_pricing_page"]:
        raise PublicVerificationError(
            f"{page_url} does not look like the NFL Predictive Pricing page (marker text absent)"
        )
    if not page_checks["references_feed"]:
        raise PublicVerificationError(f"{page_url} no longer references its {json_name} feed")

    feed_status, feed_headers, feed_bytes = _fetch(feed_url, timeout=timeout)
    digest = sha256(feed_bytes).hexdigest()
    try:
        payload = json.loads(feed_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicVerificationError(f"{feed_url} is not valid UTF-8 JSON: {exc}") from exc

    try:
        summary = validate_public_payload(payload)
    except PublishError as exc:
        raise PublicVerificationError(f"{feed_url} failed the wizard-nfl-pricing-v2 contract: {exc}") from exc

    if expect_sha256 and digest.lower() != expect_sha256.strip().lower():
        raise PublicVerificationError(
            f"{feed_url} sha256={digest} does not match the card this run published "
            f"(expected {expect_sha256.strip().lower()}); the public site is serving a different card"
        )
    for label, expected, actual in (
        ("generated_at_utc", expect_generated_at, summary.get("generated_at_utc")),
        ("horizon", expect_horizon, summary.get("horizon")),
        ("week", expect_week, summary.get("week")),
    ):
        if expected is not None and str(expected) != str(actual):
            raise PublicVerificationError(
                f"{feed_url} {label}={actual!r} does not match the expected {expected!r}"
            )

    age_hours = None
    generated_at = summary.get("generated_at_utc")
    if generated_at:
        stamp = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        now = now_utc or datetime.now(timezone.utc)
        age_hours = (now - stamp).total_seconds() / 3600.0
        if require_fresh_hours is not None and age_hours > require_fresh_hours:
            raise PublicVerificationError(
                f"{feed_url} was generated {age_hours:.1f}h ago, older than the required "
                f"{require_fresh_hours:.1f}h"
            )

    return {
        "status": "VERIFIED",
        "page_url": page_url,
        "feed_url": feed_url,
        "page": page_checks,
        "feed": {
            "http_status": feed_status,
            "bytes": len(feed_bytes),
            "sha256": digest,
            "content_type": feed_headers.get("Content-Type"),
            "age_hours": None if age_hours is None else round(age_hours, 3),
            **summary,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    parser.add_argument("--expect-sha256", default=None, help="Assert the served feed is exactly this card.")
    parser.add_argument("--expect-generated-at", default=None)
    parser.add_argument("--expect-horizon", default=None, choices=[None, "TUE", "FRI"])
    parser.add_argument("--expect-week", type=int, default=None)
    parser.add_argument(
        "--require-fresh-hours",
        type=float,
        default=None,
        help="Fail when the served card's generated_at_utc is older than this many hours.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = verify(
            page_url=args.page_url,
            json_name=args.json_name,
            expect_sha256=args.expect_sha256,
            expect_generated_at=args.expect_generated_at,
            expect_horizon=args.expect_horizon,
            expect_week=args.expect_week,
            require_fresh_hours=args.require_fresh_hours,
            timeout=args.timeout,
        )
    except PublicVerificationError as exc:
        print(json.dumps({"status": "PUBLIC_VERIFICATION_FAILED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
