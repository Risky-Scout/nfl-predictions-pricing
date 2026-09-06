"""wizard-nfl-pricing-v2 -- public NFL Predictive Pricing exporter.

PUBLISHING TRANSLATION ONLY. Reads ONE already-created certified 2026 TUE/FRI
production run -- an EXPLICIT run manifest (written by
``src/nfl_hybrid/production/run_2026.py::build_run_manifest``) plus the
EXPLICIT forecast-ledger horizon directory it produced -- and writes the
public ``wizard-nfl-pricing-v2`` JSON contract consumed by the NFL Predictive
Pricing page.

This script does NOT generate predictions, fit any model, compute or alter a
predicted margin/total, calibrate anything, capture BallDontLie data, read
prospective QB-shadow data, reconstruct a certified market, call any external
API, or read any environment variable or credential. It never guesses which
run is "current": both the manifest and the forecast directory are explicit
CLI arguments, and there is no "find latest" behaviour anywhere in this
module (no mtime scan, no newest-directory scan, no legacy fallback path).

The v1 exporter (``scripts/export_wizard_nfl_predictions.py``) already froze
the run-selection, identity, numeric, archive and atomic-latest discipline
this exporter needs; its helpers are loaded and reused verbatim below rather
than reimplemented. v1 itself is never modified and its public schema is
untouched -- v2 is a strictly separate contract with its own schema_version,
its own key order, and its own output tree.

Frozen source field mapping (do not rediscover):

    public game_id                 <- top-level game_id
    public kickoff_utc             <- prediction.scheduled_kickoff_utc
    public away_team               <- prediction.away_team_id   (already a
    public home_team               <- prediction.home_team_id      canonical
                                      NFL abbreviation -- no further mapping)
    public predicted_home_margin   <- prediction.prediction.predicted_margin
    public predicted_game_total    <- prediction.prediction.predicted_total
    public market_home_spread      <- prediction.markets.ATS.market.consensus_line
    public market_total            <- prediction.markets.TOTAL.market.consensus_line
    public market_ats_book_count   <- prediction.markets.ATS.market.eligible_books
    public market_total_book_count <- prediction.markets.TOTAL.market.eligible_books
    public market_as_of_utc        <- max instant across the UNION of
                                      prediction.markets.ATS.market
                                          .selected_returned_snapshot_timestamps
                                      and prediction.markets.TOTAL.market
                                          .selected_returned_snapshot_timestamps
    public season                  <- prediction.season
    public week                    <- prediction.week
    public horizon                 <- top-level horizon
    public generated_at_utc        <- run_created_at_utc (the ONE authoritative
                                      shared run-creation timestamp; never
                                      created_at_utc, mtime, target_cutoff_utc,
                                      or the export's own clock)

Publication gate. A game is priced only when its point forecast is genuinely
a forecast and both certified markets are genuinely certified:

    prediction.prediction.model_status == "OOF"
    predicted_margin present, numeric, finite, non-boolean
    predicted_total  present, numeric, finite, non-boolean
    markets.ATS.market   present and fully valid
    markets.TOTAL.market present and fully valid
    markets.ATS.status   exactly one of ALLOWED_MARKET_STATUSES
    markets.TOTAL.status exactly one of ALLOWED_MARKET_STATUSES

The model_status gate is required in ADDITION to the finiteness checks, not
implied by them: a non-OOF row is an abstention, and an abstention can sit
behind a market entry that looks usable (UNCERTAINTY_NOT_READY, or even
CALIBRATED) while its point forecast is null or a placeholder. Anything short
of the whole gate fails the card closed -- no partial pricing board.

The market-status allowlist is an additional SCHEMA-DRIFT guard, not a
replacement for raw-data validation and not a requirement that the status be
``OK``. CALIBRATION_NOT_READY and UNCERTAINTY_NOT_READY describe calibration
and uncertainty machinery this raw pricing page does not publish, so both stay
publishable whenever the model row is OOF, the point forecasts are finite and
the certified market payload independently validates. Anything else -- an
unrecognised status, a missing or null status, a wrong-cased ``ok``, a padded
`` OK ``, or an explicit MARKET_NOT_READY / MARKET_SOURCE_UNAVAILABLE -- fails
the card closed. Those last two already fail for lack of a ``market``
consensus payload; the allowlist rejects them a second time so that a future
producer which starts attaching a payload to a not-ready entry cannot quietly
become publishable.

``market_home_spread`` is emitted in the certified sportsbook notation exactly
as captured (-3.5 means the HOME team is -3.5, +3.5 means the home team is
+3.5, 0 is a pick'em); this exporter never inverts it. ``predicted_home_margin``
stays signed from the HOME perspective. No model or market value is ever
rounded here -- the page formats presentation values.

Every derived pricing quantity (model fair home spread, sportsbook implied
home margin, ATS home edge and its side, total edge and its side, model
winner, market favourite) is a pure function of the fields above and is
deliberately NOT persisted: it belongs to presentation logic. Moneyline,
Kelly, EV, bet recommendations and confidence grades are outside v2 scope and
are not read here at all.

The production output tree is ``$NFL_MODEL_ARTIFACT_ROOT/public/wizardofodds/
nfl-pricing/`` -- ``latest.json`` plus ``archive/season=<YYYY>/week=<NN>/
horizon=<TUE|FRI>.json``. That root is expanded by the CALLER and passed in as
an explicit ``--output`` path; this module never expands it, and generated
prediction JSON is never committed.

Usage:
    python scripts/export_wizard_nfl_pricing.py \\
        --run-manifest <root>/production-2026/run-manifests/<run_id>.json \\
        --forecast-dir <root>/production-2026/forecast-ledger/<TUE|FRI> \\
        --output <root>/public/wizardofodds/nfl-pricing/latest.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "wizard-nfl-pricing-v2"

# --------------------------------------------------------------------------- #
# v1 helper reuse. The v1 publishing exporter is a sibling script, not an
# importable package module, so it is loaded by explicit path -- exactly the
# way its own focused tests load it. Loading it has no side effect (constants
# and function definitions only, guarded __main__), reads nothing, and cannot
# change v1 behaviour: every symbol below is bound read-only and v1's own
# module globals stay untouched.
# --------------------------------------------------------------------------- #
_V1_EXPORTER_PATH = Path(__file__).resolve().parent / "export_wizard_nfl_predictions.py"


def _load_v1_exporter():
    spec = importlib.util.spec_from_file_location("_wizard_nfl_predictions_v1_exporter", _V1_EXPORTER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the v1 exporter helpers from {_V1_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v1 = _load_v1_exporter()

# Shared fail-closed error type: a v2 failure is the same class of failure as
# a v1 failure and callers catch one exception, not two.
WizardExportError = _v1.WizardExportError
_fail = _v1._fail

_canonical_json = _v1._canonical_json
_sha256_hex = _v1._sha256_hex
_parse_utc_instant = _v1._parse_utc_instant
_format_utc_z = _v1._format_utc_z
_validate_season = _v1._validate_season
_parse_week = _v1._parse_week
_validate_team = _v1._validate_team
_validate_numeric = _v1._validate_numeric

load_run_manifest = _v1.load_run_manifest
select_run_forecasts = _v1.select_run_forecasts
serialize_card = _v1.serialize_card
archive_path = _v1.archive_path
write_archive = _v1.write_archive
write_latest = _v1.write_latest

ALLOWED_HORIZONS = _v1.ALLOWED_HORIZONS
SUPPORTED_SEASONS = _v1.SUPPORTED_SEASONS

# ATS and TOTAL are independently certified markets. Both are mandatory for
# v2 and their book counts are persisted separately -- never collapsed into a
# single market_book_count by min/max/mean/union/intersection.
MARKET_ATS = "ATS"
MARKET_TOTAL = "TOTAL"
CERTIFIED_MARKETS = (MARKET_ATS, MARKET_TOTAL)

# Same floor the repo's abstention policy and market-attachment gates enforce
# (nfl_hybrid.governance.abstention / MINIMUM_ELIGIBLE_BOOKS): fewer than
# three eligible books is not a publishable consensus.
MINIMUM_ELIGIBLE_BOOKS = 3

# The ONLY model status whose point forecast may be publicly priced. Every
# other status the production run can write (MODEL_NOT_READY and friends) is
# an abstention: run_2026 counts exactly `status == "OOF"` rows as forecasts
# and everything else as an abstention, and an abstaining row's
# predicted_margin/predicted_total are written as null. A finiteness check
# alone is NOT a substitute for this gate -- see _require_model_ready.
REQUIRED_MODEL_STATUS = "OOF"

# The only market-entry statuses whose certified payload may be published.
# Membership is decided by exact equality, so a missing status, a null status,
# a wrong-cased "ok", a padded " OK ", a non-string, and any status the
# production run does not currently write are all rejected. MARKET_NOT_READY
# and MARKET_SOURCE_UNAVAILABLE are deliberately absent: they are rejected
# explicitly here as well as by their missing ``market`` payload.
ALLOWED_MARKET_STATUSES = ("OK", "CALIBRATION_NOT_READY", "UNCERTAINTY_NOT_READY")

_REQUIRED_MARKET_FIELDS = ("consensus_line", "eligible_books", "selected_returned_snapshot_timestamps")

TOP_LEVEL_KEY_ORDER = ("schema_version", "season", "week", "horizon", "generated_at_utc", "games")
GAME_KEY_ORDER = (
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


# --------------------------------------------------------------------------- #
# certified market access -- read-only. A record whose market entry carries no
# ``market`` consensus payload was never certified for that market
# (MARKET_NOT_READY / MARKET_SOURCE_UNAVAILABLE), so the whole card fails
# closed: v2 does not publish a partial pricing board. An entry that DOES
# carry a payload must additionally declare a recognised status and then have
# every field it exposes validate -- three independent gates, checked payload
# first so a not-ready entry keeps its precise diagnostic.
# --------------------------------------------------------------------------- #
def _require_allowed_market_status(entry: dict, *, game_id: str, market_name: str) -> str:
    """Schema-drift guard, NOT a requirement that the status be ``OK``. Exact
    equality against the allowlist, so a missing/null/non-string status and any
    case or whitespace variant are all rejected rather than normalised."""
    status = entry.get("status")
    if status not in ALLOWED_MARKET_STATUSES:
        _fail(
            f"{game_id}: markets.{market_name}.status is {status!r}, which is not one of the "
            f"publishable statuses {ALLOWED_MARKET_STATUSES} -- refusing to price an entry whose "
            "certification state this exporter does not recognise"
        )
    return status


def _certified_market(prediction: dict, *, game_id: str, market_name: str) -> dict:
    markets = prediction.get("markets")
    if not isinstance(markets, dict):
        _fail(
            f"{game_id}: forecast record has no markets payload -- both certified "
            f"{MARKET_ATS} and {MARKET_TOTAL} markets are mandatory for {SCHEMA_VERSION}"
        )
    entry = markets.get(market_name)
    if not isinstance(entry, dict):
        _fail(f"{game_id}: certified {market_name} market is missing from the forecast record")
    market = entry.get("market")
    if not isinstance(market, dict):
        _fail(
            f"{game_id}: certified {market_name} market snapshot is missing "
            f"(markets.{market_name}.market) -- refusing to publish a partial pricing board"
        )
    _require_allowed_market_status(entry, game_id=game_id, market_name=market_name)
    missing = [f for f in _REQUIRED_MARKET_FIELDS if f not in market]
    if missing:
        _fail(f"{game_id}: markets.{market_name}.market is missing required field(s) {missing}")
    return market


def _require_model_ready(inner: dict, *, game_id: str) -> str:
    """Require an OOF point forecast. This gate is mandatory EVEN THOUGH the
    finiteness checks below exist, because the two failure modes are
    independent: a market entry can be UNCERTAINTY_NOT_READY (or even fully
    CALIBRATED) while the model row behind it is an abstention whose
    predicted_margin/predicted_total is null or otherwise unusable. Gating on
    the numbers alone would let a future non-null placeholder on an abstaining
    row become a published price."""
    status = inner.get("model_status")
    if status != REQUIRED_MODEL_STATUS:
        _fail(
            f"{game_id}: prediction.prediction.model_status is {status!r}, not "
            f"{REQUIRED_MODEL_STATUS!r} -- a non-{REQUIRED_MODEL_STATUS} model row is an "
            "abstention and is never publicly priced"
        )
    return status


def _validate_book_count(value: Any, *, game_id: str, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{game_id}: {field_name} must be an integer eligible-book count, got {value!r}")
    if value < MINIMUM_ELIGIBLE_BOOKS:
        _fail(
            f"{game_id}: {field_name} is {value}, below the certified minimum of "
            f"{MINIMUM_ELIGIBLE_BOOKS} eligible books"
        )
    return value


def _selected_snapshot_instants(market: dict, *, game_id: str, market_name: str) -> list[datetime]:
    """The certified snapshot instants actually SELECTED into this market's
    consensus. FAILS CLOSED unless the collection exists as a non-empty list
    of timezone-aware, parseable instants."""
    field = f"markets.{market_name}.market.selected_returned_snapshot_timestamps"
    raw = market.get("selected_returned_snapshot_timestamps")
    if not isinstance(raw, list):
        _fail(f"{game_id}: {field} must be a list of timestamps, got {raw!r}")
    if not raw:
        _fail(f"{game_id}: {field} is empty -- market as-of instant cannot be established")
    return [
        _parse_utc_instant(value, field_name=f"{game_id}: {field}[{index}]")
        for index, value in enumerate(raw)
    ]


def _market_as_of_utc(ats_market: dict, total_market: dict, *, game_id: str) -> str:
    """The maximum instant across the UNION of the ATS and TOTAL selected
    snapshot timestamps. There is no fallback: run_created_at_utc,
    target_cutoff_utc, created_at_utc, filesystem mtime and the current clock
    are never substituted."""
    instants = _selected_snapshot_instants(ats_market, game_id=game_id, market_name=MARKET_ATS)
    instants += _selected_snapshot_instants(total_market, game_id=game_id, market_name=MARKET_TOTAL)
    return _format_utc_z(max(instants))


# --------------------------------------------------------------------------- #
# per-game public translation. Only the frozen eleven keys ever enter the
# returned dict: no predicted_winner, no market_favorite, no model_fair_spread,
# no ats_side/ats_edge_points, no total_side/total_edge_points, no
# winner_agreement, no moneyline/Kelly/EV/recommendation/confidence grade.
# --------------------------------------------------------------------------- #
def _build_public_game(record: dict) -> tuple[datetime, dict]:
    game_id = record.get("game_id")
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        _fail(f"{game_id}: forecast record has no prediction payload")

    kickoff_dt = _parse_utc_instant(
        prediction.get("scheduled_kickoff_utc"), field_name=f"{game_id}: scheduled_kickoff_utc",
    )

    home_team = _validate_team(prediction.get("home_team_id"), game_id=game_id, side="home")
    away_team = _validate_team(prediction.get("away_team_id"), game_id=game_id, side="away")
    if home_team == away_team:
        _fail(f"{game_id}: home_team_id equals away_team_id ({home_team!r})")

    inner = prediction.get("prediction")
    if not isinstance(inner, dict):
        _fail(f"{game_id}: forecast record has no nested prediction.prediction payload")
    _require_model_ready(inner, game_id=game_id)
    margin = _validate_numeric(inner.get("predicted_margin"), game_id=game_id, field_name="predicted_margin")
    total = _validate_numeric(inner.get("predicted_total"), game_id=game_id, field_name="predicted_total")

    ats_market = _certified_market(prediction, game_id=game_id, market_name=MARKET_ATS)
    total_market = _certified_market(prediction, game_id=game_id, market_name=MARKET_TOTAL)

    market_home_spread = _validate_numeric(
        ats_market.get("consensus_line"), game_id=game_id,
        field_name=f"markets.{MARKET_ATS}.market.consensus_line",
    )
    market_total = _validate_numeric(
        total_market.get("consensus_line"), game_id=game_id,
        field_name=f"markets.{MARKET_TOTAL}.market.consensus_line",
    )
    ats_book_count = _validate_book_count(
        ats_market.get("eligible_books"), game_id=game_id,
        field_name=f"markets.{MARKET_ATS}.market.eligible_books",
    )
    total_book_count = _validate_book_count(
        total_market.get("eligible_books"), game_id=game_id,
        field_name=f"markets.{MARKET_TOTAL}.market.eligible_books",
    )
    market_as_of_utc = _market_as_of_utc(ats_market, total_market, game_id=game_id)

    public_game = {
        "game_id": game_id,
        "kickoff_utc": _format_utc_z(kickoff_dt),
        "away_team": away_team,
        "home_team": home_team,
        "predicted_home_margin": margin,
        "predicted_game_total": total,
        "market_home_spread": market_home_spread,
        "market_total": market_total,
        "market_as_of_utc": market_as_of_utc,
        "market_ats_book_count": ats_book_count,
        "market_total_book_count": total_book_count,
    }
    assert tuple(public_game.keys()) == GAME_KEY_ORDER
    return kickoff_dt, public_game


# --------------------------------------------------------------------------- #
# card assembly -- identical run-integrity discipline to v1: the manifest's
# own run_id selects the forecasts, per-record prediction_hash is re-verified,
# the batch hash must reproduce the manifest's declared value, and mixed
# season/week/horizon/run_created_at_utc or a duplicate game_id fails closed.
# --------------------------------------------------------------------------- #
def build_public_card(manifest: dict, forecast_dir: Path) -> dict:
    selected = select_run_forecasts(manifest, forecast_dir)
    if not selected:
        _fail(
            f"run {manifest['run_id']!r} status=SUCCESS but selected zero forecasts -- "
            "season/week cannot be established for an empty pricing card"
        )

    seasons: set[int] = set()
    weeks: set[Any] = set()
    prediction_hashes: list[str] = []
    dated_games: list[tuple[datetime, dict]] = []

    for record in selected:
        prediction = record["prediction"]

        stored_hash = record.get("prediction_hash")
        if not isinstance(stored_hash, str) or not stored_hash:
            _fail(f"{record.get('game_id')}: forecast record has no prediction_hash")
        if _sha256_hex(prediction) != stored_hash:
            _fail(f"{record.get('game_id')}: prediction_hash does not match its own prediction payload")
        prediction_hashes.append(stored_hash)

        seasons.add(prediction.get("season"))
        weeks.add(prediction.get("week"))

        dated_games.append(_build_public_game(record))

    if len(seasons) > 1:
        _fail(f"mixed season across selected forecasts for run {manifest['run_id']!r}: {sorted(seasons, key=str)}")
    if len(weeks) > 1:
        _fail(f"mixed week across selected forecasts for run {manifest['run_id']!r}: {sorted(weeks, key=str)}")

    recomputed_batch_hash = _sha256_hex({"prediction_hashes": sorted(prediction_hashes)})
    if recomputed_batch_hash != manifest["output_hashes"]["forecast_batch_hash"]:
        _fail(
            f"recomputed forecast_batch_hash {recomputed_batch_hash} does not match run manifest "
            f"output_hashes.forecast_batch_hash {manifest['output_hashes']['forecast_batch_hash']}"
        )

    season = _validate_season(next(iter(seasons)))
    week = _parse_week(next(iter(weeks)))

    dated_games.sort(key=lambda pair: (pair[0], pair[1]["game_id"]))
    games = [game for _, game in dated_games]

    generated_at_utc = _format_utc_z(
        _parse_utc_instant(manifest["run_created_at_utc"], field_name="run_created_at_utc"),
    )

    card = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "horizon": manifest["horizon"],
        "generated_at_utc": generated_at_utc,
        "games": games,
    }
    assert tuple(card.keys()) == TOP_LEVEL_KEY_ORDER
    return card


# --------------------------------------------------------------------------- #
# output: immutable archive first, then atomic latest.json (v1 discipline,
# v1 helpers, separate nfl-pricing output tree)
# --------------------------------------------------------------------------- #
def run_export(*, run_manifest_path: Path, forecast_dir: Path, output_path: Path) -> dict:
    """The whole pipeline: manifest validation -> card completeness ->
    forecast + certified-market validation -> deterministic serialization ->
    immutable archive -> atomic latest.json. Raises WizardExportError (never
    partially writing anything, never overwriting a differing archive, never
    touching latest.json when the archive step fails) on any failure."""
    manifest = load_run_manifest(run_manifest_path)
    if not forecast_dir.is_dir():
        _fail(f"forecast-dir does not exist or is not a directory: {forecast_dir}")
    card = build_public_card(manifest, forecast_dir)
    payload_bytes = serialize_card(card)

    archive_root = output_path.parent / "archive"
    archived_path = write_archive(
        archive_root, season=card["season"], week=card["week"], horizon=card["horizon"],
        payload_bytes=payload_bytes,
    )
    latest = write_latest(output_path, payload_bytes)
    return {
        "status": "OK",
        "run_id": manifest["run_id"],
        "season": card["season"],
        "week": card["week"],
        "horizon": card["horizon"],
        "generated_at_utc": card["generated_at_utc"],
        "game_count": len(card["games"]),
        "archive_path": str(archived_path),
        "latest_path": str(latest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-manifest", required=True, type=str, help="explicit run-manifest JSON path (no auto-selection)")
    parser.add_argument("--forecast-dir", required=True, type=str, help="explicit forecast-ledger/<horizon> directory")
    parser.add_argument("--output", required=True, type=str, help="explicit nfl-pricing latest.json output path")
    args = parser.parse_args(argv)

    try:
        result = run_export(
            run_manifest_path=Path(args.run_manifest),
            forecast_dir=Path(args.forecast_dir),
            output_path=Path(args.output),
        )
    except WizardExportError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
