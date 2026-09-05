"""Focused tests for the certified 2026 BallDontLie -> Fix-8 market bridge.

Every fixture here is SYNTHETIC and compact, written with the verified real
BDL capture schema (``bdl-2026-asof-capture-v1``: ``{"data": [...], "meta":
{...}}`` envelopes; odds rows carrying ``game_id``/``vendor``/
``spread_home_value``/``spread_away_value``/``spread_home_odds``/
``spread_away_odds``/``total_value``/``total_over_odds``/``total_under_odds``/
``updated_at``). No network access, no real capture is read, no model is fit,
no historical bootstrap runs, and no official forecast is produced.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.production import run_2026 as prod

REPO_ROOT = Path(__file__).resolve().parents[1]

# The certified cutoff a synthetic TUE capture is frozen for, and quote
# timestamps that satisfy the existing chronology
# (market_last_update <= returned_snapshot_utc <= target_cutoff_utc) with
# room to spare inside the certified 48-hour freshness window.
CUTOFF = "2026-09-08T16:00:00Z"
SNAPSHOT = "2026-09-08T15:30:00Z"
UPDATED = "2026-09-08T15:26:01.575Z"

# Real BDL team ids from the frozen 32-franchise crosswalk (KC / HOU / BUF /
# NYJ). Never a made-up id -- the bridge resolves teams through the existing
# canonical crosswalk.
KC, HOU, BUF, NYJ = 14, 10, 3, 4


# ---------------------------------------------------------------------------
# Synthetic capture construction (mirrors the real capture layout exactly).
# ---------------------------------------------------------------------------
def _game_row(bdl_id: int, home_id: int, away_id: int, *, season: int = 2026, week: int = 1) -> dict:
    return {
        "id": bdl_id,
        "season": season,
        "week": week,
        "postseason": False,
        "date": "2026-09-13T17:00:00Z",
        "home_team": {"id": home_id},
        "visitor_team": {"id": away_id},
        "home_team_score": None,
        "visitor_team_score": None,
        "status": "Scheduled",
        "status_state": "pre",
    }


def _odds_row(
    bdl_game_id: int,
    vendor: str,
    *,
    spread_home_value: object = "-3.5",
    spread_away_value: object = "3.5",
    spread_home_odds: object = -105,
    spread_away_odds: object = -115,
    total_value: object = "44.5",
    total_over_odds: object = -108,
    total_under_odds: object = -112,
    updated_at: object = UPDATED,
    **overrides: object,
) -> dict:
    row = {
        "game_id": bdl_game_id,
        "id": 900000 + bdl_game_id,
        "vendor": vendor,
        "spread_home_value": spread_home_value,
        "spread_away_value": spread_away_value,
        "spread_home_odds": spread_home_odds,
        "spread_away_odds": spread_away_odds,
        "total_value": total_value,
        "total_over_odds": total_over_odds,
        "total_under_odds": total_under_odds,
        # Present in the real payload; deliberately NOT part of the certified
        # contract this bridge emits.
        "moneyline_home_odds": -170,
        "moneyline_away_odds": 145,
        "updated_at": updated_at,
    }
    row.update(overrides)
    for key, value in list(row.items()):
        if value is _ABSENT:
            del row[key]
    return row


class _Absent:
    """Sentinel meaning "this field is entirely absent from the payload"."""


_ABSENT = _Absent()


def _drop(row: dict, field: str) -> dict:
    """A copy of ``row`` with ``field`` entirely absent (not null)."""
    return {k: v for k, v in row.items() if k != field}


def _envelope(rows: list[dict]) -> bytes:
    return json.dumps({"data": rows, "meta": {"next_cursor": None, "per_page": 100}}).encode("utf-8")


def _request_record(logical_name: str, filename: str, data: bytes, page_index: int | None, received: str) -> dict:
    return {
        "logical_name": logical_name,
        "endpoint": f"https://api.balldontlie.io/nfl/v1/{logical_name}",
        "method": "GET",
        "query_parameters": {"season": 2026, "week": 1},
        "request_started_at_utc": received,
        "response_received_at_utc": received,
        "http_status": 200,
        "response_byte_count": len(data),
        "response_sha256": sha256(data).hexdigest(),
        "output_filename": filename,
        "page_index": page_index,
        "attempts": 1,
        "error": None,
    }


def _deterministic_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def write_capture(
    root: Path,
    *,
    games: list[dict] | None = None,
    odds_pages: list[list[dict]] | None = None,
    odds_received: list[str] | None = None,
    season: int = 2026,
    week: int = 1,
    season_type: str = "REG",
    horizon: str = "TUE",
    requested_horizon: str | None = None,
    status: str = "COMPLETE",
    nominal_cutoff_utc: str = CUTOFF,
    downgrade_reason: str | None = None,
    required_source_ok: dict | None = None,
    corrupt_manifest_hash: bool = False,
    corrupt_page_bytes: str | None = None,
    delete_page: str | None = None,
    schema_version: str = bridge.SUPPORTED_CAPTURE_SCHEMA_VERSION,
) -> Path:
    """Write a synthetic capture directory and return its ``manifest.json``
    path. Hashes are computed exactly the way the real capture script does,
    so the bridge's verification is exercised for real."""
    games = games if games is not None else [_game_row(1, KC, HOU)]
    odds_pages = odds_pages if odds_pages is not None else [
        [_odds_row(1, vendor) for vendor in ("draftkings", "fanduel", "betmgm")]
    ]
    odds_received = odds_received or [SNAPSHOT] * len(odds_pages)

    capture_dir = root / "capture=20260908T153000Z"
    capture_dir.mkdir(parents=True, exist_ok=True)

    requests: list[dict] = []
    games_bytes = _envelope(games)
    (capture_dir / "games.p001.json").write_bytes(games_bytes)
    requests.append(_request_record("games", "games.p001.json", games_bytes, 1, SNAPSHOT))

    for index, (page, received) in enumerate(zip(odds_pages, odds_received, strict=True), start=1):
        filename = f"odds_current.p{index:03d}.json"
        data = _envelope(page)
        (capture_dir / filename).write_bytes(data)
        requests.append(_request_record("odds_current", filename, data, index, received))

    manifest_body = {
        "schema_version": schema_version,
        "status": status,
        "capture_dir": "/some/other/host/path/that/does/not/exist",
        "season": season,
        "week": week,
        "season_type": season_type,
        "season_type_code": 2,
        "horizon": horizon,
        "requested_horizon": requested_horizon if requested_horizon is not None else horizon,
        "horizon_downgrade_reason": downgrade_reason,
        "allow_off_window_smoke": False,
        "nominal_cutoff_utc": nominal_cutoff_utc,
        "nominal_cutoff_provided": True,
        "capture_started_at_utc": "2026-09-08T15:29:00Z",
        "capture_completed_at_utc": "2026-09-08T15:31:00Z",
        "git_commit": "0" * 40,
        "capture_uuid": "11111111-1111-1111-1111-111111111111",
        "base_url": "https://api.balldontlie.io/nfl/v1",
        "request_count": len(requests),
        "success_count": len(requests),
        "failure_count": 0,
        "required_source_ok": required_source_ok
        if required_source_ok is not None
        else {name: True for name in (
            "games", "rosters", "injuries", "fantasy_qb_projections",
            "odds_current", "odds_opening", "player_props",
        )},
        "pagination_notes": {},
        "target_games": {"count": len(games)},
        "eligibility_summary": {},
        "requests": requests,
        "scientific_model_files_changed": False,
    }
    manifest_body["manifest_sha256"] = sha256(
        _deterministic_json(manifest_body).encode("utf-8")
    ).hexdigest()
    if corrupt_manifest_hash:
        manifest_body["manifest_sha256"] = "f" * 64

    manifest_path = capture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_body, indent=2, sort_keys=True))

    if corrupt_page_bytes:
        (capture_dir / corrupt_page_bytes).write_bytes(_envelope([]) + b" ")
    if delete_page:
        (capture_dir / delete_page).unlink()
    return manifest_path


def _write_fake_historical_estate(monkeypatch, tmp_path: Path) -> int:
    """Three disjoint synthetic historical stores (one season range each,
    matching the real estate's disjointness), and a ``resolve`` that points at
    them. Returns the total row count written."""
    total = 0
    for index, key in enumerate(rmr.RAW_ODDS_HISTORY_KEYS):
        season = 2020 + index
        rows = pd.DataFrame([
            {"game_id": f"{season}_01_HOU_KC", "bookmaker_key": bk, "market": "spreads", "outcome_key": side,
             "returned_snapshot_utc": pd.Timestamp(f"{season}-09-03T15:30:00Z"),
             "market_last_update": pd.Timestamp(f"{season}-09-03T15:00:00Z"),
             "point": -3.5 if side == "home" else 3.5, "price_decimal": 1.91}
            for bk in ("bookA", "bookB", "bookC") for side in ("home", "away")
        ])
        store = tmp_path / key
        store.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(store / "bookmaker_quotes.parquet", index=False)
        total += len(rows)
    monkeypatch.setattr(rmr.external_data, "resolve", lambda key, root_override=None: tmp_path / key)
    return total


def _executable_source(*relative_paths: str) -> str:
    """The EXECUTABLE source of the given repo files -- every comment and
    string literal (so every docstring) removed. Lets a test assert "this
    module never references X" without a prose mention of X in a docstring
    counting as a reference."""
    import io
    import tokenize

    out: list[str] = []
    for relative in relative_paths:
        text = (REPO_ROOT / relative).read_text()
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(token.string)
    return " ".join(out)


def _validate(manifest_path: Path, **kwargs) -> bridge.ValidatedCapture:
    return bridge.validate_capture_manifest(manifest_path, **kwargs)


def _quotes(manifest_path: Path) -> pd.DataFrame:
    return bridge.build_bookmaker_quotes(_validate(manifest_path))


# ===========================================================================
# 1-7: canonical row construction and field mapping.
# ===========================================================================
def test_valid_capture_creates_canonical_bookmaker_quotes_rows(tmp_path):
    quotes = _quotes(write_capture(tmp_path))
    assert list(quotes.columns) == list(bridge.BOOKMAKER_QUOTE_COLUMNS)
    assert not quotes.empty
    assert set(quotes["market"]) == {"spreads", "totals"}
    assert set(quotes["game_id"]) == {"2026_01_HOU_KC"}


def test_one_odds_row_creates_exactly_four_certified_rows(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings")]]))
    assert len(quotes) == 4
    assert sorted(zip(quotes["market"], quotes["outcome_key"])) == [
        ("spreads", "away"), ("spreads", "home"), ("totals", "over"), ("totals", "under"),
    ]


def test_vendor_becomes_bookmaker_key_verbatim(tmp_path):
    vendors = ["draftkings", "fanduel", "betmgm", "kalshi", "polymarket"]
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, v) for v in vendors]]))
    # Passed through verbatim: no renaming, and no vendor is excluded or
    # specially weighted -- kalshi/polymarket included exactly like the rest.
    assert set(quotes["bookmaker_key"]) == set(vendors)
    for vendor in vendors:
        assert len(quotes[quotes["bookmaker_key"] == vendor]) == 4


def test_home_spread_mapping(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings")]]))
    row = quotes[(quotes["market"] == "spreads") & (quotes["outcome_key"] == "home")].iloc[0]
    assert row["point"] == -3.5


def test_away_spread_mapping(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings")]]))
    row = quotes[(quotes["market"] == "spreads") & (quotes["outcome_key"] == "away")].iloc[0]
    assert row["point"] == 3.5


def test_total_over_mapping(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings")]]))
    row = quotes[(quotes["market"] == "totals") & (quotes["outcome_key"] == "over")].iloc[0]
    assert row["point"] == 44.5


def test_total_under_mapping_shares_the_same_point(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings")]]))
    over = quotes[(quotes["market"] == "totals") & (quotes["outcome_key"] == "over")].iloc[0]
    under = quotes[(quotes["market"] == "totals") & (quotes["outcome_key"] == "under")].iloc[0]
    assert under["point"] == over["point"] == 44.5


# ===========================================================================
# 8-13: actual supplied American prices -> decimal.
# ===========================================================================
def _price(quotes: pd.DataFrame, market: str, outcome: str) -> float:
    return float(quotes[(quotes["market"] == market) & (quotes["outcome_key"] == outcome)].iloc[0]["price_decimal"])


def test_actual_home_spread_price_converted(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", spread_home_odds=-105)]]))
    assert _price(quotes, "spreads", "home") == pytest.approx(1 + 100 / 105)


def test_actual_away_spread_price_converted(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", spread_away_odds=-115)]]))
    assert _price(quotes, "spreads", "away") == pytest.approx(1 + 100 / 115)


def test_actual_over_price_converted(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", total_over_odds=-108)]]))
    assert _price(quotes, "totals", "over") == pytest.approx(1 + 100 / 108)


def test_actual_under_price_converted(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", total_under_odds=+120)]]))
    assert _price(quotes, "totals", "under") == pytest.approx(2.2)


def test_positive_american_odds_conversion():
    assert bridge.american_to_decimal(150) == pytest.approx(2.5)
    assert bridge.american_to_decimal(100) == pytest.approx(2.0)


def test_negative_american_odds_conversion():
    assert bridge.american_to_decimal(-200) == pytest.approx(1.5)
    assert bridge.american_to_decimal(-110) == pytest.approx(1 + 100 / 110)


# ===========================================================================
# 14-19: fail-closed on malformed rows. Nothing is ever filled in.
# ===========================================================================
def test_zero_american_odds_rejected():
    with pytest.raises(bridge.BdlMarketBridgeError, match="0 are not a real price"):
        bridge.american_to_decimal(0)


@pytest.mark.parametrize("bad", [None, True, False, "", "even", float("nan"), float("inf"), [], {}])
def test_non_numeric_or_non_finite_american_odds_rejected(bad):
    with pytest.raises(bridge.BdlMarketBridgeError):
        bridge.american_to_decimal(bad)


def test_missing_price_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", spread_home_odds=_ABSENT)]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="missing"):
        _quotes(manifest)


def test_missing_vendor_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_drop(_odds_row(1, "draftkings"), "vendor")]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="no vendor"):
        _quotes(manifest)


def test_empty_vendor_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_odds_row(1, "   ")]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="empty vendor"):
        _quotes(manifest)


def test_missing_game_id_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_drop(_odds_row(1, "draftkings"), "game_id")]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="no usable game_id"):
        _quotes(manifest)


def test_missing_updated_at_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", updated_at=_ABSENT)]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="updated_at"):
        _quotes(manifest)


def test_spread_points_not_opposite_rejected(tmp_path):
    manifest = write_capture(
        tmp_path, odds_pages=[[_odds_row(1, "draftkings", spread_home_value="-3.5", spread_away_value="4.0")]]
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="not opposites"):
        _quotes(manifest)


def test_missing_total_value_rejected(tmp_path):
    manifest = write_capture(tmp_path, odds_pages=[[_odds_row(1, "draftkings", total_value=_ABSENT)]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="total_value"):
        _quotes(manifest)


# ===========================================================================
# 20-22: game-id mapping goes through the capture's own games response.
# ===========================================================================
def test_odds_game_id_maps_through_games_response(tmp_path):
    manifest = write_capture(
        tmp_path,
        games=[_game_row(1, KC, HOU), _game_row(2, BUF, NYJ)],
        odds_pages=[[_odds_row(1, "draftkings"), _odds_row(2, "draftkings")]],
    )
    quotes = _quotes(manifest)
    # The production convention {season}_{week:02d}_{away}_{home}, resolved
    # through the existing canonical BDL mapping -- never inferred from the
    # numeric BDL odds game_id.
    assert set(quotes["game_id"]) == {"2026_01_HOU_KC", "2026_01_NYJ_BUF"}


def test_unknown_bdl_game_rejected(tmp_path):
    manifest = write_capture(
        tmp_path, games=[_game_row(1, KC, HOU)], odds_pages=[[_odds_row(4242, "draftkings")]]
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="no matching row in the capture"):
        _quotes(manifest)


def test_canonical_game_collision_rejected(tmp_path):
    # Two distinct BDL games normalizing to the same production game_id.
    manifest = write_capture(
        tmp_path, games=[_game_row(1, KC, HOU), _game_row(2, KC, HOU)], odds_pages=[[_odds_row(1, "draftkings")]]
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="canonical game-id collision"):
        _quotes(manifest)


def test_zero_matching_games_fails_closed(tmp_path):
    manifest = write_capture(tmp_path, games=[], odds_pages=[[_odds_row(1, "draftkings")]])
    with pytest.raises(bridge.BdlMarketBridgeError, match="no games"):
        _quotes(manifest)


# ===========================================================================
# 23-32: manifest validation.
# ===========================================================================
def test_manifest_must_be_complete(tmp_path):
    manifest = write_capture(tmp_path, status="INCOMPLETE")
    with pytest.raises(bridge.BdlMarketBridgeError, match="only a COMPLETE capture"):
        _validate(manifest)


def test_smoke_capture_rejected(tmp_path):
    manifest = write_capture(tmp_path, horizon="SMOKE", requested_horizon="SMOKE")
    with pytest.raises(bridge.BdlMarketBridgeError, match="not a production horizon"):
        _validate(manifest)


def test_horizon_downgrade_rejected(tmp_path):
    manifest = write_capture(
        tmp_path, horizon="SMOKE", requested_horizon="TUE",
        downgrade_reason="requested TUE but run outside the capture window; forced to SMOKE",
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="not a production horizon|downgrade"):
        _validate(manifest)


def test_downgrade_reason_alone_rejected(tmp_path):
    manifest = write_capture(tmp_path, horizon="TUE", requested_horizon="TUE", downgrade_reason="anything")
    with pytest.raises(bridge.BdlMarketBridgeError, match="records a horizon downgrade"):
        _validate(manifest)


def test_season_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path, season=2025)
    with pytest.raises(bridge.BdlMarketBridgeError, match="season 2025 != requested season 2026"):
        _validate(manifest, expected_season=2026)


def test_week_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path, week=2)
    with pytest.raises(bridge.BdlMarketBridgeError, match="week 2 != requested week 1"):
        _validate(manifest, expected_week=1)


def test_horizon_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path, horizon="FRI", requested_horizon="FRI")
    with pytest.raises(bridge.BdlMarketBridgeError, match="!= requested horizon"):
        _validate(manifest, expected_horizon="TUE")


def test_nominal_cutoff_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path)
    with pytest.raises(bridge.BdlMarketBridgeError, match="!= production target_cutoff_utc"):
        _validate(manifest, expected_target_cutoff_utc="2026-09-11T16:00:00Z")


def test_nominal_cutoff_match_accepted(tmp_path):
    capture = _validate(
        write_capture(tmp_path),
        expected_season=2026, expected_week=1, expected_horizon="TUE", expected_target_cutoff_utc=CUTOFF,
    )
    assert capture.nominal_cutoff_utc == pd.Timestamp(CUTOFF)


def test_manifest_hash_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path, corrupt_manifest_hash=True)
    with pytest.raises(bridge.BdlMarketBridgeError, match="manifest hash mismatch"):
        _validate(manifest)


def test_response_hash_mismatch_rejected(tmp_path):
    manifest = write_capture(tmp_path, corrupt_page_bytes="odds_current.p001.json")
    with pytest.raises(bridge.BdlMarketBridgeError, match="response hash mismatch"):
        _validate(manifest)


def test_missing_odds_page_rejected(tmp_path):
    manifest = write_capture(tmp_path, delete_page="odds_current.p001.json")
    with pytest.raises(bridge.BdlMarketBridgeError, match="missing the raw response file"):
        _validate(manifest)


def test_unsupported_schema_version_rejected(tmp_path):
    manifest = write_capture(tmp_path, schema_version="bdl-2027-something-v9")
    with pytest.raises(bridge.BdlMarketBridgeError, match="unsupported capture schema_version"):
        _validate(manifest)


def test_failed_required_source_rejected(tmp_path):
    manifest = write_capture(
        tmp_path, required_source_ok={"games": True, "odds_current": False, "odds_opening": True}
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="required source 'odds_current'"):
        _validate(manifest)


def test_opening_odds_are_not_required(tmp_path):
    # Certified CURRENT-market pricing has no opening-line dependency, and
    # none is invented here.
    manifest = write_capture(
        tmp_path, required_source_ok={"games": True, "odds_current": True, "odds_opening": False}
    )
    assert _validate(manifest).horizon == "TUE"


# ===========================================================================
# 33-34: returned_snapshot_utc provenance.
# ===========================================================================
def test_response_received_at_used_as_returned_snapshot(tmp_path):
    page_a_received = "2026-09-08T15:30:00Z"
    page_b_received = "2026-09-08T15:35:00Z"
    manifest = write_capture(
        tmp_path,
        games=[_game_row(1, KC, HOU), _game_row(2, BUF, NYJ)],
        odds_pages=[[_odds_row(1, "draftkings")], [_odds_row(2, "draftkings")]],
        odds_received=[page_a_received, page_b_received],
    )
    quotes = _quotes(manifest)
    by_game = quotes.groupby("game_id")["returned_snapshot_utc"].first()
    # Each page carries ITS OWN request record's response_received_at_utc.
    assert by_game["2026_01_HOU_KC"] == pd.Timestamp(page_a_received)
    assert by_game["2026_01_NYJ_BUF"] == pd.Timestamp(page_b_received)


def test_filesystem_mtime_and_capture_timestamps_not_used(tmp_path):
    manifest_path = write_capture(tmp_path)
    page = manifest_path.parent / "odds_current.p001.json"
    # Move the file's mtime far away from every timestamp in the manifest.
    import os

    os.utime(page, (1_600_000_000, 1_600_000_000))
    quotes = _quotes(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    snapshots = set(quotes["returned_snapshot_utc"])
    # ONLY the request record's own response_received_at_utc -- never the
    # file's mtime, never the capture-level start/finish timestamps, never
    # the current clock.
    assert snapshots == {pd.Timestamp(SNAPSHOT)}
    assert pd.Timestamp(manifest["capture_completed_at_utc"]) not in snapshots
    assert pd.Timestamp(manifest["capture_started_at_utc"]) not in snapshots
    assert pd.Timestamp(1_600_000_000, unit="s", tz="UTC") not in snapshots
    assert pd.Timestamp.now(tz="UTC").floor("min") not in snapshots


# ===========================================================================
# 35-39: the EXISTING certified Fix-8 gates, applied to bridged rows.
# ===========================================================================
def _targets(game_id: str = "2026_01_HOU_KC") -> pd.DataFrame:
    return pd.DataFrame({"game_id": [game_id], "target_cutoff_utc": [pd.Timestamp(CUTOFF)]})


def _reconstruct(quotes: pd.DataFrame, market: str = rmr.MARKET_SPREADS):
    coherent = rmr.build_coherent_book_observations(quotes, market)
    return rmr.reconstruct_market_at_cutoffs(coherent, _targets(), market=market)


def test_future_market_update_rejected_by_certified_chronology(tmp_path):
    # market_last_update AFTER returned_snapshot_utc violates the existing
    # certified ordering rule; no future quote enters the forecast.
    quotes = _quotes(write_capture(
        tmp_path,
        odds_pages=[[_odds_row(1, v, updated_at="2026-09-08T15:59:00Z") for v in ("dk", "fd", "mgm")]],
        odds_received=["2026-09-08T15:30:00Z"],
    ))
    assert (quotes["market_last_update"] > quotes["returned_snapshot_utc"]).all()
    assert _reconstruct(quotes).consensus.empty


def test_stale_book_beyond_48h_rejected_by_certified_path(tmp_path):
    stale_snapshot = "2026-09-05T15:30:00Z"  # ~72h before the cutoff
    quotes = _quotes(write_capture(
        tmp_path,
        odds_pages=[[_odds_row(1, v, updated_at="2026-09-05T15:00:00Z") for v in ("dk", "fd", "mgm")]],
        odds_received=[stale_snapshot],
    ))
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert not coherent.empty  # coherent, but stale
    result = _reconstruct(quotes)
    assert result.consensus.empty
    assert result.coverage["ge3_before_age_filter"] == 1
    assert result.coverage["market_ready"] == 0


def test_fewer_than_three_coherent_books_fails_certified_readiness(tmp_path):
    quotes = _quotes(write_capture(
        tmp_path, odds_pages=[[_odds_row(1, v) for v in ("draftkings", "fanduel")]]
    ))
    assert _reconstruct(quotes).consensus.empty


def test_three_or_more_coherent_books_succeeds(tmp_path):
    quotes = _quotes(write_capture(
        tmp_path, odds_pages=[[_odds_row(1, v) for v in ("draftkings", "fanduel", "betmgm")]]
    ))
    for market in (rmr.MARKET_SPREADS, rmr.MARKET_TOTALS):
        result = _reconstruct(quotes, market)
        assert len(result.consensus) == 1
        row = result.consensus.iloc[0]
        assert row["eligible_books"] == 3
        # The certified aggregation is reused verbatim -- not reimplemented.
        assert row["consensus_method"] == rmr.CONSENSUS_METHOD
    assert _reconstruct(quotes, rmr.MARKET_SPREADS).consensus.iloc[0]["consensus_line"] == -3.5
    assert _reconstruct(quotes, rmr.MARKET_TOTALS).consensus.iloc[0]["consensus_line"] == 44.5


def test_no_synthetic_minus_110_introduced(tmp_path):
    quotes = _quotes(write_capture(tmp_path, odds_pages=[[
        _odds_row(1, "draftkings", spread_home_odds=-105, spread_away_odds=-115,
                  total_over_odds=-108, total_under_odds=-112),
    ]]))
    minus_110_decimal = 1 + 100 / 110
    assert not any(abs(p - minus_110_decimal) < 1e-9 for p in quotes["price_decimal"])
    assert sorted(round(p, 10) for p in quotes["price_decimal"]) == sorted(
        round(1 + 100 / a, 10) for a in (105, 115, 108, 112)
    )
    assert (quotes["price_decimal"] > 1.0).all()


def test_moneyline_not_added_to_certified_contract(tmp_path):
    quotes = _quotes(write_capture(tmp_path))
    assert set(quotes["market"]) <= {"spreads", "totals"}
    assert "h2h" not in set(quotes["market"])
    assert set(quotes["outcome_key"]) <= {"home", "away", "over", "under"}


# ===========================================================================
# Materialization: deterministic, identity-keyed, outside git.
# ===========================================================================
def test_materialization_is_deterministic_and_idempotent(tmp_path):
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    manifest = write_capture(tmp_path / "cap")
    first = bridge.load_live_market_source(manifest, artifact_root_path=aroot)
    second = bridge.load_live_market_source(manifest, artifact_root_path=aroot)
    assert first.quotes_path == second.quotes_path
    assert first.content_sha256 == second.content_sha256
    assert first.capture.manifest_sha256 in str(first.quotes_path)
    assert REPO_ROOT not in first.quotes_path.parents


def test_differing_artifact_at_same_identity_fails_closed(tmp_path):
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    manifest = write_capture(tmp_path / "cap")
    capture = _validate(manifest)
    quotes = bridge.build_bookmaker_quotes(capture)
    bridge.materialize_bookmaker_quotes(capture, quotes, artifact_root_path=aroot)
    tampered = quotes.copy()
    tampered.loc[0, "price_decimal"] = 1.99
    with pytest.raises(bridge.BdlMarketBridgeError, match="FAIL CLOSED"):
        bridge.materialize_bookmaker_quotes(capture, tampered, artifact_root_path=aroot)


# ===========================================================================
# 40: historical safety.
# ===========================================================================
def test_historical_reconstruction_unchanged_with_no_2026_manifest(monkeypatch, tmp_path):
    """A 2020-2025 reconstruction works with no 2026 capture anywhere: the
    live source is an OPTIONAL argument, never a globally required key."""
    assert "odds_history.2026" not in rmr.RAW_ODDS_HISTORY_KEYS
    assert rmr.LIVE_MARKET_SOURCE_STORE not in rmr.RAW_ODDS_HISTORY_KEYS

    total_rows = _write_fake_historical_estate(monkeypatch, tmp_path)

    baseline = rmr.load_raw_bookmaker_quotes()
    assert len(baseline) == total_rows
    assert set(baseline["source_store"]) == set(rmr.RAW_ODDS_HISTORY_KEYS)

    coherent = rmr.build_coherent_book_observations(baseline, rmr.MARKET_SPREADS)
    targets = pd.DataFrame({
        "game_id": ["2020_01_HOU_KC"], "target_cutoff_utc": [pd.Timestamp("2020-09-03T16:00:00Z")],
    })
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market=rmr.MARKET_SPREADS)
    assert len(result.consensus) == 1
    assert result.consensus.iloc[0]["eligible_books"] == 3


def test_live_quotes_are_additive_to_the_historical_estate(monkeypatch, tmp_path):
    total_rows = _write_fake_historical_estate(monkeypatch, tmp_path)
    live = _quotes(write_capture(tmp_path / "cap"))
    combined = rmr.load_raw_bookmaker_quotes(live_quotes=live)
    assert set(combined["source_store"]) == set(rmr.RAW_ODDS_HISTORY_KEYS) | {rmr.LIVE_MARKET_SOURCE_STORE}
    assert len(combined[combined["source_store"] == rmr.LIVE_MARKET_SOURCE_STORE]) == len(live)
    # Purely additive: every historical row is still present, unchanged.
    assert len(combined) == total_rows + len(live)


# ===========================================================================
# 41-42: production preflight is evidence-based, never optimistic.
# ===========================================================================
def test_production_blocked_without_explicit_live_capture():
    evidence = prod.evaluate_live_market_source(None)
    assert evidence["registered"] is False
    assert evidence["status"] == prod.LIVE_MARKET_NOT_SUPPLIED

    readiness = prod.summarize_preflight_readiness(
        infra_blocking=[], schedule_2026_available=True, live_2026_market_source_registered=False,
    )
    assert readiness["overall_status"] == "BLOCKED_ON_LIVE_INPUTS"
    assert prod.LIVE_MARKET_2026_BLOCKER in readiness["blocking_problems"]


def test_invalid_supplied_capture_still_blocks_production(tmp_path):
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    smoke = write_capture(tmp_path / "cap", horizon="SMOKE", requested_horizon="SMOKE")
    evidence = prod.evaluate_live_market_source(smoke, artifact_root_path=aroot)
    assert evidence["registered"] is False
    assert evidence["status"] == prod.LIVE_MARKET_INVALID


def test_valid_live_capture_clears_only_the_market_source_blocker(tmp_path):
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    manifest = write_capture(
        tmp_path / "cap",
        odds_pages=[[_odds_row(1, v) for v in ("draftkings", "fanduel", "betmgm")]],
    )
    evidence = prod.evaluate_live_market_source(
        manifest, expected_season=2026, expected_week=1, expected_horizon="TUE",
        expected_target_cutoff_utc=CUTOFF, artifact_root_path=aroot,
    )
    assert evidence["registered"] is True
    assert evidence["status"] == prod.LIVE_MARKET_OK
    assert evidence["provenance"]["manifest_sha256"]

    # ONLY the market-source blocker is cleared. A host with no 2026 schedule
    # is still BLOCKED_ON_LIVE_INPUTS, and an infra blocker still wins.
    readiness = prod.summarize_preflight_readiness(
        infra_blocking=[], schedule_2026_available=False, live_2026_market_source_registered=True,
    )
    assert readiness["overall_status"] == "BLOCKED_ON_LIVE_INPUTS"
    assert readiness["blocking_problems"] == [prod.SCHEDULE_2026_BLOCKER]

    ready = prod.summarize_preflight_readiness(
        infra_blocking=[], schedule_2026_available=True, live_2026_market_source_registered=True,
    )
    assert ready["overall_status"] == "READY"

    still_not_ready = prod.summarize_preflight_readiness(
        infra_blocking=["hash_mismatch"], schedule_2026_available=True, live_2026_market_source_registered=True,
    )
    assert still_not_ready["overall_status"] == "NOT_READY"


def test_preflight_accepts_an_explicit_manifest_argument(tmp_path):
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    manifest = write_capture(tmp_path / "cap")
    result = prod.run_preflight(
        artifact_root_path=aroot, market_capture_manifest=manifest,
        expected_season=2026, expected_week=1, expected_horizon="TUE", expected_target_cutoff_utc=CUTOFF,
    )
    assert result["live_2026_market_source_registered"] is True
    assert result["checks"]["live_2026_market_source"]["status"] == prod.LIVE_MARKET_OK

    default_result = prod.run_preflight(artifact_root_path=aroot)
    assert default_result["live_2026_market_source_registered"] is False
    assert default_result["checks"]["live_2026_market_source"]["status"] == prod.LIVE_MARKET_NOT_SUPPLIED


def test_cli_threads_the_explicit_manifest_through(monkeypatch, tmp_path):
    """``--market-capture-manifest`` reaches both the preflight and the
    horizon batch; omitting it passes ``None``, leaving the historical
    behaviour of every existing invocation unchanged."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_2026_production_card_under_test", REPO_ROOT / "scripts/run_2026_production_card.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    seen: dict[str, object] = {}
    monkeypatch.setattr(cli.prod, "run_preflight", lambda **kw: seen.update(preflight=kw) or {"production_run_ready": False})
    monkeypatch.setattr(cli.prod, "run_horizon_batch", lambda **kw: seen.update(batch=kw) or {"status": "SUCCESS"})

    manifest = str(write_capture(tmp_path / "cap"))
    monkeypatch.setattr(
        "sys.argv", ["prog", "--preflight", "--horizon", "TUE", "--market-capture-manifest", manifest]
    )
    cli.main()
    assert seen["preflight"]["market_capture_manifest"] == manifest
    assert seen["preflight"]["expected_horizon"] == "TUE"
    assert seen["preflight"]["expected_target_cutoff_utc"] is not None

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--horizon", "TUE", "--as-of", "2026-09-08T17:00:00Z", "--market-capture-manifest", manifest],
    )
    cli.main()
    assert seen["batch"]["market_capture_manifest"] == manifest

    monkeypatch.setattr("sys.argv", ["prog", "--horizon", "TUE", "--as-of", "2026-09-08T17:00:00Z"])
    cli.main()
    assert seen["batch"]["market_capture_manifest"] is None


def test_capture_is_never_auto_selected(tmp_path):
    """Two captures exist side by side; nothing picks one. There is no
    scan/sort/"latest" path anywhere in the bridge or the production runner."""
    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    older = write_capture(tmp_path / "older")
    newer = write_capture(tmp_path / "newer")
    assert older.exists() and newer.exists()

    # Two captures on disk and neither is chosen: with no explicit manifest
    # there is simply no live market source.
    evidence = prod.evaluate_live_market_source(None, artifact_root_path=aroot)
    assert evidence["registered"] is False
    assert evidence["source"] is None

    # The bridge has no directory-scanning primitive at all, so there is no
    # "latest by mtime / name / capture_started_at / UUID / first COMPLETE"
    # path for it to take.
    tokens = _executable_source("src/nfl_hybrid/data/bdl_market_bridge.py").split()
    for forbidden in ("glob", "rglob", "iterdir", "scandir", "listdir", "walk", "getmtime", "st_mtime"):
        assert forbidden not in tokens
    # ...and it never reads the capture-level timestamps a "pick the latest"
    # heuristic would sort on.
    for forbidden in ("capture_started_at_utc", "capture_completed_at_utc", "capture_uuid"):
        assert forbidden not in tokens


# ===========================================================================
# 43: no network access.
# ===========================================================================
def test_no_balldontlie_http_request_occurs(monkeypatch, tmp_path):
    """Structural + behavioural: the bridge imports no HTTP client, and a
    full build with every socket/requests entry point poisoned still
    succeeds."""
    import socket

    def _explode(*args, **kwargs):
        raise AssertionError("the certified market bridge must never perform a network call")

    monkeypatch.setattr(socket, "socket", _explode)
    monkeypatch.setattr(socket, "create_connection", _explode)
    try:
        import requests

        monkeypatch.setattr(requests.Session, "request", _explode)
        monkeypatch.setattr(requests, "get", _explode)
    except ImportError:
        pass

    aroot = tmp_path / "artifacts"
    aroot.mkdir()
    source = bridge.load_live_market_source(write_capture(tmp_path / "cap"), artifact_root_path=aroot)
    assert not source.quotes.empty

    tokens = _executable_source("src/nfl_hybrid/data/bdl_market_bridge.py").split()
    for forbidden in ("requests", "urllib", "urlopen", "http", "httpx", "socket", "BDLConfig", "client"):
        assert forbidden not in tokens


# ===========================================================================
# 44-46: read-only invariants.
# ===========================================================================
def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_capture_script_unchanged():
    assert _sha256_file(REPO_ROOT / "scripts/capture_bdl_2026_asof.py") == (
        "0a571a8d9ea9254057d7fdca8819c7d2c762bb052ec195b4b6df4bebfe2dc6ca"
    ), "scripts/capture_bdl_2026_asof.py is pinned by SHA-256 by the Sep 8/Sep 11 automation"


def test_qb_shadow_files_unchanged():
    assert _sha256_file(REPO_ROOT / "scripts/update_v2026_7_qb_projection_shock_shadow.py") == (
        "3bb20644d18911ed48d6f1b235508a35c6b265eab6b5e2aef748b6a0246af75e"
    )
    assert _sha256_file(REPO_ROOT / "outputs/v2026_7_prospective_qb_projection_shock_preregistration.json") == (
        "aabe03e7a601036fe2e40cf6df91114e90a1a9939e6693b86f37988e4642d6a2"
    )


def test_model_feature_contract_unchanged():
    from nfl_hybrid.evaluation.official_horizon_oof import ELO_FEATURE_COLUMNS

    assert ELO_FEATURE_COLUMNS == (
        "home_elo_pregame_rating",
        "home_elo_pregame_win_probability",
        "home_elo_pregame_expected_margin",
        "away_elo_pregame_rating",
        "away_elo_pregame_win_probability",
        "away_elo_pregame_expected_margin",
    )


def test_certified_market_contract_unchanged():
    assert rmr.CONSENSUS_METHOD == "median_of_devigged_per_book_quotes_across_eligible_books"
    assert rmr.MINIMUM_FRESH_COHERENT_BOOKS == 3
    assert rmr.FRESHNESS_MAX_AGE_HOURS == 48.0
    assert bridge.BOOKMAKER_QUOTE_COLUMNS == (
        "game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update",
        "market", "outcome_key", "point", "price_decimal",
    )


def test_legacy_week1_pricing_path_not_used():
    source = _executable_source(
        "src/nfl_hybrid/data/bdl_market_bridge.py",
        "src/nfl_hybrid/production/run_2026.py",
        "src/nfl_hybrid/evaluation/raw_market_reconstruction.py",
        "scripts/run_2026_production_card.py",
    )
    for legacy in ("lines_wk1", "card_wk1", "predictions_wk01", "run_manifest_wk1",
                   "build_week1_2026_lines", "season_2026"):
        assert legacy not in source
