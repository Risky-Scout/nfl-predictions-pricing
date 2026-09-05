"""Bridge from ONE explicitly supplied, immutable BallDontLie TUE/FRI capture
to the certified Fix-8 canonical ``bookmaker_quotes`` contract.

    frozen BDL capture  ->  canonical bookmaker_quotes  ->  existing certified
    Fix-8 reconstruction (:mod:`nfl_hybrid.evaluation.raw_market_reconstruction`)

This module is market-input PLUMBING ONLY. It introduces no consensus rule,
no de-vig, no aggregation, no freshness/minimum-book policy and no vendor
inclusion/exclusion policy -- every one of those already lives, unchanged, in
the certified Fix-8 path (median-of-devigged-per-book quotes across eligible
books, >=3 books, 48-hour maximum age, same-snapshot coherence, no synthetic
prices). All this file does is turn raw captured bytes into the eight
certified columns those rules already consume.

WHAT IT READS
  A capture produced by ``scripts/capture_bdl_2026_asof.py`` (schema
  ``bdl-2026-asof-capture-v1``): ``manifest.json`` plus the verbatim
  ``games.pNNN.json`` / ``odds_current.pNNN.json`` response bodies beside it.
  Never the network -- there is no HTTP client here and no import of one.
  Never the legacy Week-1 pricing artifacts (``outputs/season_2026/
  lines_wk1.csv``, ``card_wk1.csv``, ``predictions_wk01.csv``,
  ``run_manifest_wk1.json``, ``scripts/build_week1_2026_lines.py``), which
  are not the certified public production source.

CAPTURE SELECTION IS NEVER AUTOMATIC
  The exact manifest path is supplied explicitly by the caller
  (``--market-capture-manifest``). Nothing here scans a directory, sorts by
  mtime/name/``capture_started_at_utc``/UUID, or picks "the latest COMPLETE
  capture". A capture that was not named explicitly is never used.

REAL BDL ODDS SCHEMA (frozen; verified from an immutable smoke capture)
  Envelope ``{"data": [...], "meta": {...}}``; each odds row carries
  ``game_id``, ``vendor``, ``spread_home_value``, ``spread_away_value``,
  ``spread_home_odds``, ``spread_away_odds``, ``total_value``,
  ``total_over_odds``, ``total_under_odds`` and ``updated_at``. Point values
  arrive as strings ("-3.5"), prices as American integers (-105). The raw
  payload ALSO carries ``moneyline_home_odds``/``moneyline_away_odds``;
  moneyline is deliberately NOT part of the certified market contract and is
  not emitted here (a future enhancement, not this task).

The observed smoke coverage (16 games x 8 vendors) proved schema capability
only. Nothing here hard-codes a game count, a vendor count or a vendor name:
production validates whatever the official capture actually contains.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_hybrid.data.external_data import artifact_root
from nfl_hybrid.providers.balldontlie import canonical as bdl_canonical

# The capture schema this bridge understands. A capture written by any other
# schema version is refused rather than best-effort parsed.
SUPPORTED_CAPTURE_SCHEMA_VERSION = "bdl-2026-asof-capture-v1"

# Logical sources that MUST have succeeded for certified CURRENT-market
# pricing. ``odds_opening`` is deliberately absent: the certified Fix-8
# reconstruction prices from current two-sided quotes and has no opening-line
# dependency, so requiring one here would invent a gate that production does
# not have.
REQUIRED_LOGICAL_SOURCES: tuple[str, ...] = ("games", "odds_current")

GAMES_LOGICAL_NAME = "games"
ODDS_CURRENT_LOGICAL_NAME = "odds_current"

# Production captures are TUE/FRI only. SMOKE is schema evidence, never an
# official production market.
PRODUCTION_HORIZONS: tuple[str, ...] = ("TUE", "FRI")
PRODUCTION_SEASON_TYPES: tuple[str, ...] = ("REG", "POST")

MARKET_SPREADS = "spreads"
MARKET_TOTALS = "totals"

# The certified canonical schema consumed by the existing Fix-8
# reconstruction (identical to the historical raw Odds API stores'
# ``bookmaker_quotes.parquet``). Never extended here.
BOOKMAKER_QUOTE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "bookmaker_key",
    "returned_snapshot_utc",
    "market_last_update",
    "market",
    "outcome_key",
    "point",
    "price_decimal",
)

# Where a materialized live-market artifact goes: under the GENERATED
# artifact root (never NFL_MODEL_DATA_ROOT, never the git repo), keyed by the
# capture's own immutable identity so re-materializing the same capture is
# deterministic and materializing a different capture can never collide.
LIVE_MARKET_ARTIFACT_NAMESPACE = "live-market-2026/balldontlie"
QUOTES_FILENAME = "bookmaker_quotes.parquet"
QUOTES_IDENTITY_FILENAME = "bookmaker_quotes.identity.json"


class BdlMarketBridgeError(RuntimeError):
    """Fail-closed: the supplied capture cannot be turned into certified
    market rows without fabricating something. Never downgraded to a warning,
    never partially satisfied -- a caller that sees this must treat the live
    market source as unavailable."""


# ---------------------------------------------------------------------------
# Odds conversion. Actual supplied prices only -- never assumed, never filled.
# ---------------------------------------------------------------------------
def american_to_decimal(value: Any) -> float:
    """Convert an ACTUAL American price to decimal odds.

        A > 0  ->  1 + A / 100
        A < 0  ->  1 + 100 / abs(A)

    Rejects ``0``, ``None``, booleans, non-numeric values and non-finite
    values. There is no default price: a missing or unusable price is an
    error, never a silently substituted -110. The certified downstream
    contract requires ``price_decimal > 1.0``, which every accepted input
    here satisfies by construction."""
    if isinstance(value, bool):
        raise BdlMarketBridgeError(f"American odds must be numeric, got boolean {value!r}")
    if value is None:
        raise BdlMarketBridgeError("American odds are missing (null)")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise BdlMarketBridgeError("American odds are missing (empty string)")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise BdlMarketBridgeError(f"American odds are non-numeric: {value!r}") from exc
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise BdlMarketBridgeError(f"American odds are non-numeric: {value!r}")

    if not math.isfinite(numeric):
        raise BdlMarketBridgeError(f"American odds are not finite: {value!r}")
    if numeric == 0:
        raise BdlMarketBridgeError("American odds of 0 are not a real price")

    decimal = 1.0 + (numeric / 100.0 if numeric > 0 else 100.0 / abs(numeric))
    if not (math.isfinite(decimal) and decimal > 1.0):
        raise BdlMarketBridgeError(f"American odds {value!r} did not produce a decimal price > 1.0")
    return decimal


def _required_point(value: Any, field: str) -> float:
    """Parse a REQUIRED BDL point value (spread/total). BDL sends these as
    strings ("-3.5", "44.5"); numbers are accepted too. Missing, non-numeric
    or non-finite fails closed -- a point is never invented."""
    if isinstance(value, bool):
        raise BdlMarketBridgeError(f"{field} must be numeric, got boolean {value!r}")
    if value is None:
        raise BdlMarketBridgeError(f"{field} is missing")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise BdlMarketBridgeError(f"{field} is missing (empty string)")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise BdlMarketBridgeError(f"{field} is non-numeric: {value!r}") from exc
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise BdlMarketBridgeError(f"{field} is non-numeric: {value!r}")
    if not math.isfinite(numeric):
        raise BdlMarketBridgeError(f"{field} is not finite: {value!r}")
    return numeric


# ---------------------------------------------------------------------------
# Manifest validation. Everything below runs BEFORE a single odds row is read.
# ---------------------------------------------------------------------------
def _deterministic_json(obj: Any) -> str:
    """Byte-identical to ``scripts/capture_bdl_2026_asof.deterministic_json``
    -- the encoding the capture's own ``manifest_sha256`` was computed over.
    Reimplemented here rather than imported so this library module never
    depends on a script, and so the capture script stays untouched."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def manifest_body_sha256(manifest: dict) -> str:
    """Recompute the capture's ``manifest_sha256`` over the manifest body
    excluding the hash field itself, exactly as the capture script did."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return _sha256_hex(_deterministic_json(body).encode("utf-8"))


def _as_utc(value: Any, field: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise BdlMarketBridgeError(f"{field} is unparseable: {value!r}") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise BdlMarketBridgeError(f"{field} is missing/unparseable: {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@dataclass(frozen=True)
class ValidatedCapture:
    """A capture that has passed every manifest gate. Holds the identity the
    materialized artifact is keyed by, and the PHYSICAL directory the raw
    pages were found in (``manifest_path.parent``) -- never the manifest's
    recorded ``capture_dir`` string, which is an absolute path from the
    capturing host and need not exist here."""

    manifest_path: Path
    capture_dir: Path
    manifest: dict
    manifest_sha256: str
    season: int
    week: int
    horizon: str
    season_type: str
    nominal_cutoff_utc: pd.Timestamp


def _request_records(manifest: dict, logical_name: str) -> list[dict]:
    records = manifest.get("requests")
    if not isinstance(records, list):
        raise BdlMarketBridgeError("capture manifest has no 'requests' list")
    out = []
    for record in records:
        if isinstance(record, dict) and record.get("logical_name") == logical_name:
            out.append(record)
    return out


def _verify_raw_file(capture_dir: Path, record: dict) -> bytes:
    """Verify one recorded request's persisted response bytes against the
    manifest's own ``response_sha256`` and ``response_byte_count``, and return
    them. A missing page, a byte-count drift or a hash drift all fail closed."""
    filename = record.get("output_filename")
    if not filename:
        raise BdlMarketBridgeError(
            f"capture request {record.get('logical_name')!r} page {record.get('page_index')!r} "
            "recorded no output_filename"
        )
    path = capture_dir / str(filename)
    if not path.is_file():
        raise BdlMarketBridgeError(f"capture is missing the raw response file {filename!r} at {path}")
    data = path.read_bytes()

    expected_hash = record.get("response_sha256")
    if not expected_hash:
        raise BdlMarketBridgeError(f"capture manifest recorded no response_sha256 for {filename!r}")
    actual_hash = _sha256_hex(data)
    if actual_hash != expected_hash:
        raise BdlMarketBridgeError(
            f"response hash mismatch for {filename!r}: manifest={expected_hash} actual={actual_hash}"
        )

    expected_bytes = record.get("response_byte_count")
    if expected_bytes is not None and int(expected_bytes) != len(data):
        raise BdlMarketBridgeError(
            f"response byte-count mismatch for {filename!r}: manifest={expected_bytes} actual={len(data)}"
        )
    return data


def _http_ok(record: dict) -> bool:
    status = record.get("http_status")
    return isinstance(status, int) and 200 <= status < 300


def validate_capture_manifest(
    manifest_path: str | Path,
    *,
    expected_season: int | None = None,
    expected_week: int | None = None,
    expected_horizon: str | None = None,
    expected_target_cutoff_utc: pd.Timestamp | str | None = None,
) -> ValidatedCapture:
    """Validate ONE explicitly supplied capture manifest, fail-closed.

    Enforced unconditionally: the capture schema version; ``status ==
    "COMPLETE"``; a production ``horizon`` (SMOKE refused); ``requested_horizon
    == horizon`` with no recorded downgrade (a TUE/FRI request that the
    capture script forced to SMOKE can never re-enter as production); a
    production ``season_type``; the recomputed ``manifest_sha256``; and, for
    every required source (``games``, ``odds_current``), that the source
    succeeded and that each of its persisted pages matches the manifest's own
    ``response_sha256``/``response_byte_count``.

    The ``expected_*`` arguments are the REQUESTING production run's own
    identity. When supplied they must match exactly -- in particular
    ``expected_target_cutoff_utc`` must equal the manifest's
    ``nominal_cutoff_utc``, so a capture frozen for one certified cutoff can
    never be priced into a different one."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise BdlMarketBridgeError(f"market capture manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as exc:
        raise BdlMarketBridgeError(f"market capture manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise BdlMarketBridgeError(f"market capture manifest is not a JSON object: {manifest_path}")

    schema_version = manifest.get("schema_version")
    if schema_version != SUPPORTED_CAPTURE_SCHEMA_VERSION:
        raise BdlMarketBridgeError(
            f"unsupported capture schema_version {schema_version!r} "
            f"(expected {SUPPORTED_CAPTURE_SCHEMA_VERSION!r})"
        )

    missing_keys = [
        key
        for key in (
            "status", "season", "week", "season_type", "horizon", "requested_horizon",
            "nominal_cutoff_utc", "required_source_ok", "requests", "manifest_sha256",
        )
        if key not in manifest
    ]
    if missing_keys:
        raise BdlMarketBridgeError(f"capture manifest is missing required field(s): {sorted(missing_keys)}")

    if manifest["status"] != "COMPLETE":
        raise BdlMarketBridgeError(
            f"capture status is {manifest['status']!r}; only a COMPLETE capture may price production"
        )

    horizon = str(manifest["horizon"])
    if horizon not in PRODUCTION_HORIZONS:
        raise BdlMarketBridgeError(
            f"capture horizon {horizon!r} is not a production horizon {PRODUCTION_HORIZONS}; "
            "SMOKE captures are schema evidence only and are never an official production market"
        )
    requested_horizon = str(manifest["requested_horizon"])
    if requested_horizon != horizon:
        raise BdlMarketBridgeError(
            f"capture horizon was downgraded: requested_horizon={requested_horizon!r} but horizon={horizon!r}"
        )
    if manifest.get("horizon_downgrade_reason"):
        raise BdlMarketBridgeError(
            f"capture records a horizon downgrade: {manifest['horizon_downgrade_reason']!r}"
        )

    season_type = str(manifest["season_type"])
    if season_type not in PRODUCTION_SEASON_TYPES:
        raise BdlMarketBridgeError(
            f"capture season_type {season_type!r} is not valid for production {PRODUCTION_SEASON_TYPES}"
        )

    try:
        season = int(manifest["season"])
        week = int(manifest["week"])
    except (TypeError, ValueError) as exc:
        raise BdlMarketBridgeError(
            f"capture season/week are not integers: {manifest['season']!r}/{manifest['week']!r}"
        ) from exc

    nominal_cutoff_utc = _as_utc(manifest["nominal_cutoff_utc"], "capture nominal_cutoff_utc")

    if expected_season is not None and season != int(expected_season):
        raise BdlMarketBridgeError(f"capture season {season} != requested season {int(expected_season)}")
    if expected_week is not None and week != int(expected_week):
        raise BdlMarketBridgeError(f"capture week {week} != requested week {int(expected_week)}")
    if expected_horizon is not None and horizon != str(expected_horizon):
        raise BdlMarketBridgeError(f"capture horizon {horizon!r} != requested horizon {str(expected_horizon)!r}")
    if expected_target_cutoff_utc is not None:
        target = _as_utc(expected_target_cutoff_utc, "production target_cutoff_utc")
        if nominal_cutoff_utc != target:
            raise BdlMarketBridgeError(
                f"capture nominal_cutoff_utc {nominal_cutoff_utc.isoformat()} != "
                f"production target_cutoff_utc {target.isoformat()}"
            )

    recorded_hash = str(manifest["manifest_sha256"])
    recomputed_hash = manifest_body_sha256(manifest)
    if recorded_hash != recomputed_hash:
        raise BdlMarketBridgeError(
            f"manifest hash mismatch: recorded={recorded_hash} recomputed={recomputed_hash}"
        )

    required_source_ok = manifest["required_source_ok"]
    if not isinstance(required_source_ok, dict):
        raise BdlMarketBridgeError("capture manifest 'required_source_ok' is not an object")
    for source in REQUIRED_LOGICAL_SOURCES:
        if not required_source_ok.get(source):
            raise BdlMarketBridgeError(f"capture did not successfully acquire required source {source!r}")

    capture_dir = manifest_path.parent
    for source in REQUIRED_LOGICAL_SOURCES:
        records = _request_records(manifest, source)
        if not records:
            raise BdlMarketBridgeError(f"capture manifest has no request records for {source!r}")
        for record in records:
            if not _http_ok(record):
                raise BdlMarketBridgeError(
                    f"capture request for {source!r} page {record.get('page_index')!r} "
                    f"did not succeed (http_status={record.get('http_status')!r})"
                )
            _verify_raw_file(capture_dir, record)

    return ValidatedCapture(
        manifest_path=manifest_path,
        capture_dir=capture_dir,
        manifest=manifest,
        manifest_sha256=recorded_hash,
        season=season,
        week=week,
        horizon=horizon,
        season_type=season_type,
        nominal_cutoff_utc=nominal_cutoff_utc,
    )


# ---------------------------------------------------------------------------
# Raw page reading. Every page is re-verified against the manifest before use.
# ---------------------------------------------------------------------------
def _envelope_records(data: bytes, filename: str) -> list[dict]:
    try:
        body = json.loads(data)
    except ValueError as exc:
        raise BdlMarketBridgeError(f"capture page {filename!r} is not valid JSON") from exc
    if not isinstance(body, dict):
        raise BdlMarketBridgeError(f"capture page {filename!r} is not a JSON object envelope")
    rows = body.get("data")
    if rows is None:
        raise BdlMarketBridgeError(f"capture page {filename!r} has no 'data' array")
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise BdlMarketBridgeError(f"capture page {filename!r} has a non-list 'data' envelope")
    for row in rows:
        if not isinstance(row, dict):
            raise BdlMarketBridgeError(f"capture page {filename!r} contains a non-object row")
    return rows


def read_games_rows(capture: ValidatedCapture) -> list[dict]:
    """Every raw ``/games`` row across the capture's ``games.pNNN.json``
    pages, in page order."""
    rows: list[dict] = []
    for record in _request_records(capture.manifest, GAMES_LOGICAL_NAME):
        data = _verify_raw_file(capture.capture_dir, record)
        rows.extend(_envelope_records(data, str(record["output_filename"])))
    return rows


def read_odds_pages(capture: ValidatedCapture) -> list[tuple[pd.Timestamp, list[dict]]]:
    """``[(returned_snapshot_utc, rows), ...]`` -- one entry per
    ``odds_current.pNNN.json`` page.

    ``returned_snapshot_utc`` is that page's OWN
    ``request.response_received_at_utc`` from the manifest request record
    whose ``output_filename`` is this page. Never
    ``capture_completed_at_utc``, never ``capture_started_at_utc``, never a
    filesystem mtime, never the current clock -- a page is stamped with the
    instant its own response actually arrived."""
    pages: list[tuple[pd.Timestamp, list[dict]]] = []
    for record in _request_records(capture.manifest, ODDS_CURRENT_LOGICAL_NAME):
        received = record.get("response_received_at_utc")
        if not received:
            raise BdlMarketBridgeError(
                f"odds page {record.get('output_filename')!r} has no response_received_at_utc"
            )
        snapshot = _as_utc(received, "response_received_at_utc")
        data = _verify_raw_file(capture.capture_dir, record)
        pages.append((snapshot, _envelope_records(data, str(record["output_filename"]))))
    return pages


# ---------------------------------------------------------------------------
# Game-id mapping. Never inferred from the numeric BDL odds game_id.
# ---------------------------------------------------------------------------
def build_game_id_map(capture: ValidatedCapture) -> dict[int, str]:
    """``{BDL /games row id -> canonical production game_id}``.

    The canonical id comes from the EXISTING BallDontLie canonical mapping
    (:func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games` and the
    existing canonical team crosswalk), which already emits the production
    convention ``{season}_{week:02d}_{away}_{home}``. No second game-id
    implementation is introduced here.

    Fails closed on a games row without a usable id, on two rows claiming the
    same BDL id, and on a canonical game-id collision (two distinct BDL games
    normalizing to the same production game_id)."""
    raw_games = read_games_rows(capture)
    try:
        normalized = bdl_canonical.normalize_games(raw_games, season_type_hint=capture.season_type)
    except bdl_canonical.CanonicalizationError as exc:
        raise BdlMarketBridgeError(f"capture games could not be canonicalized: {exc}") from exc

    mapping: dict[int, str] = {}
    canonical_owner: dict[str, int] = {}
    for provider_game_id, canonical_game_id in zip(
        normalized["provider_game_id"], normalized["game_id"], strict=True
    ):
        if provider_game_id is None or pd.isna(provider_game_id):
            raise BdlMarketBridgeError("capture games response contains a row with no BDL game id")
        bdl_id = int(provider_game_id)
        canonical_game_id = str(canonical_game_id)
        if bdl_id in mapping and mapping[bdl_id] != canonical_game_id:
            raise BdlMarketBridgeError(
                f"BDL game id {bdl_id} maps to more than one canonical game_id: "
                f"{mapping[bdl_id]!r} and {canonical_game_id!r}"
            )
        owner = canonical_owner.get(canonical_game_id)
        if owner is not None and owner != bdl_id:
            raise BdlMarketBridgeError(
                f"canonical game-id collision: BDL games {owner} and {bdl_id} both normalize to "
                f"{canonical_game_id!r}"
            )
        mapping[bdl_id] = canonical_game_id
        canonical_owner[canonical_game_id] = bdl_id
    if not mapping:
        raise BdlMarketBridgeError("capture games response contained no games")
    return mapping


def _resolve_game_id(mapping: dict[int, str], raw_game_id: Any) -> str:
    if raw_game_id is None or isinstance(raw_game_id, bool):
        raise BdlMarketBridgeError(f"odds row has no usable game_id: {raw_game_id!r}")
    try:
        bdl_id = int(str(raw_game_id).strip())
    except (TypeError, ValueError) as exc:
        raise BdlMarketBridgeError(f"odds row game_id is not an integer id: {raw_game_id!r}") from exc
    canonical_game_id = mapping.get(bdl_id)
    if canonical_game_id is None:
        raise BdlMarketBridgeError(
            f"odds row references BDL game {bdl_id}, which has no matching row in the capture's "
            "own games response"
        )
    return canonical_game_id


# ---------------------------------------------------------------------------
# BDL odds row -> the four certified rows.
# ---------------------------------------------------------------------------
def _vendor(row: dict) -> str:
    """The BDL ``vendor`` passed through VERBATIM as ``bookmaker_key``. No
    inclusion/exclusion policy, no renaming, no special weighting -- for any
    vendor, including exchange-style venues. A sportsbook-only vendor policy
    would be a separate scientific decision, not plumbing."""
    vendor = row.get("vendor")
    if vendor is None or isinstance(vendor, bool):
        raise BdlMarketBridgeError(f"odds row has no vendor: {vendor!r}")
    text = str(vendor).strip()
    if not text:
        raise BdlMarketBridgeError("odds row has an empty vendor")
    return text


def certified_rows_for_odds_row(
    row: dict, *, game_id: str, returned_snapshot_utc: pd.Timestamp
) -> list[dict]:
    """Exactly four certified rows for one BDL odds row: spread home, spread
    away, total over, total under. Never three, never five -- a missing side
    or price fails the whole row closed rather than fabricating the other
    half of a market."""
    bookmaker_key = _vendor(row)
    market_last_update = _as_utc(row.get("updated_at"), "odds row updated_at")

    spread_home_point = _required_point(row.get("spread_home_value"), "spread_home_value")
    spread_away_point = _required_point(row.get("spread_away_value"), "spread_away_value")
    if not math.isclose(spread_home_point + spread_away_point, 0.0, abs_tol=1e-6):
        raise BdlMarketBridgeError(
            f"spread points are not opposites: home={spread_home_point} away={spread_away_point}"
        )
    total_point = _required_point(row.get("total_value"), "total_value")

    shared = {
        "game_id": game_id,
        "bookmaker_key": bookmaker_key,
        "returned_snapshot_utc": returned_snapshot_utc,
        "market_last_update": market_last_update,
    }
    return [
        {**shared, "market": MARKET_SPREADS, "outcome_key": "home",
         "point": spread_home_point, "price_decimal": american_to_decimal(row.get("spread_home_odds"))},
        {**shared, "market": MARKET_SPREADS, "outcome_key": "away",
         "point": spread_away_point, "price_decimal": american_to_decimal(row.get("spread_away_odds"))},
        {**shared, "market": MARKET_TOTALS, "outcome_key": "over",
         "point": total_point, "price_decimal": american_to_decimal(row.get("total_over_odds"))},
        {**shared, "market": MARKET_TOTALS, "outcome_key": "under",
         "point": total_point, "price_decimal": american_to_decimal(row.get("total_under_odds"))},
    ]


def build_bookmaker_quotes(capture: ValidatedCapture) -> pd.DataFrame:
    """The canonical ``bookmaker_quotes`` frame for one validated capture,
    with exactly the certified columns and nothing else.

    Chronology is NOT re-implemented here: the certified Fix-8 rules
    (``market_last_update <= returned_snapshot_utc <= target_cutoff_utc``, the
    48-hour per-book maximum age, the >=3 eligible-book minimum) are applied
    downstream, unchanged, by
    :mod:`nfl_hybrid.evaluation.raw_market_reconstruction`."""
    game_id_map = build_game_id_map(capture)
    rows: list[dict] = []
    for returned_snapshot_utc, odds_rows in read_odds_pages(capture):
        for odds_row in odds_rows:
            game_id = _resolve_game_id(game_id_map, odds_row.get("game_id"))
            rows.append(
                {
                    "_source_row": odds_row,
                    "_game_id": game_id,
                    "_snapshot": returned_snapshot_utc,
                }
            )
    if not rows:
        raise BdlMarketBridgeError("capture odds_current pages contained no odds rows")

    certified: list[dict] = []
    for entry in rows:
        certified.extend(
            certified_rows_for_odds_row(
                entry["_source_row"],
                game_id=entry["_game_id"],
                returned_snapshot_utc=entry["_snapshot"],
            )
        )

    frame = pd.DataFrame(certified, columns=list(BOOKMAKER_QUOTE_COLUMNS))
    frame["game_id"] = frame["game_id"].astype(str)
    frame["bookmaker_key"] = frame["bookmaker_key"].astype(str)
    frame["market"] = frame["market"].astype(str)
    frame["outcome_key"] = frame["outcome_key"].astype(str)
    frame["returned_snapshot_utc"] = pd.to_datetime(frame["returned_snapshot_utc"], utc=True)
    frame["market_last_update"] = pd.to_datetime(frame["market_last_update"], utc=True)
    frame["point"] = frame["point"].astype(float)
    frame["price_decimal"] = frame["price_decimal"].astype(float)
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Materialization, keyed by immutable capture identity.
# ---------------------------------------------------------------------------
def quotes_content_hash(quotes: pd.DataFrame) -> str:
    """Order-independent content hash of a canonical quotes frame. Computed
    over the frame's own values (not the parquet bytes, which are not
    byte-stable across writer versions), so "the same capture materialized
    twice" is provably the same content."""
    ordered = quotes[list(BOOKMAKER_QUOTE_COLUMNS)].copy()
    ordered["returned_snapshot_utc"] = ordered["returned_snapshot_utc"].map(lambda t: pd.Timestamp(t).isoformat())
    ordered["market_last_update"] = ordered["market_last_update"].map(lambda t: pd.Timestamp(t).isoformat())
    ordered["point"] = ordered["point"].map(lambda v: repr(float(v)))
    ordered["price_decimal"] = ordered["price_decimal"].map(lambda v: repr(float(v)))
    records = sorted(
        [tuple(str(v) for v in record) for record in ordered.itertuples(index=False, name=None)]
    )
    return _sha256_hex(_deterministic_json(records).encode("utf-8"))


def live_quotes_artifact_dir(capture: ValidatedCapture, *, artifact_root_path: Path | None = None) -> Path:
    """The deterministic location for this capture's materialized quotes:
    ``$NFL_MODEL_ARTIFACT_ROOT/live-market-2026/balldontlie/
    season=YYYY/week=WW/horizon=H/manifest_sha256=<hash>/``.

    Keyed by the capture's own immutable manifest hash, so re-materializing
    the same capture always lands on the same path and a different capture
    can never overwrite it. Outside git by construction (the artifact root is
    a private, git-ignored estate root); live market data is never written
    into the repository."""
    root = artifact_root_path if artifact_root_path is not None else artifact_root()
    return (
        Path(root)
        / LIVE_MARKET_ARTIFACT_NAMESPACE
        / f"season={capture.season}"
        / f"week={capture.week:02d}"
        / f"horizon={capture.horizon}"
        / f"manifest_sha256={capture.manifest_sha256}"
    )


def materialize_bookmaker_quotes(
    capture: ValidatedCapture,
    quotes: pd.DataFrame,
    *,
    artifact_root_path: Path | None = None,
) -> Path:
    """Write (or confirm) this capture's ``bookmaker_quotes.parquet`` at its
    identity-keyed location and return the path.

    Deterministic and idempotent: re-materializing the same capture rewrites
    nothing and returns the same path. FAILS CLOSED if an artifact already
    exists at this immutable identity but carries different content -- a
    capture's materialized market is never silently replaced."""
    target_dir = live_quotes_artifact_dir(capture, artifact_root_path=artifact_root_path)
    quotes_path = target_dir / QUOTES_FILENAME
    identity_path = target_dir / QUOTES_IDENTITY_FILENAME
    content_hash = quotes_content_hash(quotes)

    identity = {
        "manifest_sha256": capture.manifest_sha256,
        "season": capture.season,
        "week": capture.week,
        "horizon": capture.horizon,
        "season_type": capture.season_type,
        "nominal_cutoff_utc": capture.nominal_cutoff_utc.isoformat(),
        "row_count": int(len(quotes)),
        "content_sha256": content_hash,
    }

    if identity_path.is_file():
        try:
            existing = json.loads(identity_path.read_text())
        except ValueError as exc:
            raise BdlMarketBridgeError(
                f"existing live-market artifact identity at {identity_path} is unreadable"
            ) from exc
        if existing != identity:
            raise BdlMarketBridgeError(
                f"FAIL CLOSED: a different live-market artifact already exists at the same immutable "
                f"capture identity {capture.manifest_sha256}: existing={existing} new={identity}"
            )
        if not quotes_path.is_file():
            raise BdlMarketBridgeError(
                f"live-market artifact identity exists at {identity_path} but its "
                f"{QUOTES_FILENAME} is missing"
            )
        return quotes_path

    target_dir.mkdir(parents=True, exist_ok=True)
    quotes[list(BOOKMAKER_QUOTE_COLUMNS)].to_parquet(quotes_path, index=False)
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True))
    return quotes_path


@dataclass(frozen=True)
class LiveMarketSource:
    """The materialized, re-read live market source plus the provenance a
    production run records for it. ``quotes`` is what the certified Fix-8
    reconstruction actually consumes."""

    capture: ValidatedCapture
    quotes_path: Path
    quotes: pd.DataFrame
    content_sha256: str

    def provenance(self) -> dict:
        return {
            "provider": bdl_canonical.PROVIDER_NAME,
            "manifest_path": str(self.capture.manifest_path),
            "manifest_sha256": self.capture.manifest_sha256,
            "season": self.capture.season,
            "week": self.capture.week,
            "horizon": self.capture.horizon,
            "season_type": self.capture.season_type,
            "nominal_cutoff_utc": self.capture.nominal_cutoff_utc.isoformat(),
            "quotes_path": str(self.quotes_path),
            "quotes_content_sha256": self.content_sha256,
            "quote_row_count": int(len(self.quotes)),
            "game_count": int(self.quotes["game_id"].nunique()),
            "bookmaker_count": int(self.quotes["bookmaker_key"].nunique()),
        }


def load_live_market_source(
    manifest_path: str | Path,
    *,
    expected_season: int | None = None,
    expected_week: int | None = None,
    expected_horizon: str | None = None,
    expected_target_cutoff_utc: pd.Timestamp | str | None = None,
    artifact_root_path: Path | None = None,
) -> LiveMarketSource:
    """The one entry point a production run uses: validate the explicitly
    supplied capture, build the certified rows, materialize them at their
    immutable identity, and re-read the materialized artifact.

    Returning the RE-READ frame is deliberate -- a production run's market
    source is the artifact on disk, so what was written is what gets priced."""
    capture = validate_capture_manifest(
        manifest_path,
        expected_season=expected_season,
        expected_week=expected_week,
        expected_horizon=expected_horizon,
        expected_target_cutoff_utc=expected_target_cutoff_utc,
    )
    quotes = build_bookmaker_quotes(capture)
    quotes_path = materialize_bookmaker_quotes(capture, quotes, artifact_root_path=artifact_root_path)
    reread = pd.read_parquet(quotes_path)
    reread["returned_snapshot_utc"] = pd.to_datetime(reread["returned_snapshot_utc"], utc=True)
    reread["market_last_update"] = pd.to_datetime(reread["market_last_update"], utc=True)
    content_hash = quotes_content_hash(reread)
    if content_hash != quotes_content_hash(quotes):
        raise BdlMarketBridgeError(
            f"materialized live-market artifact at {quotes_path} does not match the rows built from "
            "the capture"
        )
    return LiveMarketSource(
        capture=capture, quotes_path=quotes_path, quotes=reread, content_sha256=content_hash
    )
