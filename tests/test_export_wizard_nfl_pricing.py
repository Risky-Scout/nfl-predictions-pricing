"""Focused tests for the wizard-nfl-pricing-v2 public pricing exporter.

Every fixture is a SYNTHETIC certified forecast-of-record JSON + synthetic
run-manifest JSON constructed directly inside pytest tmp_path directories --
never a real Week 1 forecast, never live BallDontLie or The Odds API, never
2024/2025 outcomes, never the legacy ``outputs/season_2026`` tree. This file
never runs the real production pipeline, never fits a model, and never
performs a real export.

Test numbering in the section comments follows the v2 directive's checklist.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO_ROOT / "scripts" / "export_wizard_nfl_pricing.py"
_V1_MOD_PATH = _REPO_ROOT / "scripts" / "export_wizard_nfl_predictions.py"

_spec = importlib.util.spec_from_file_location("export_wizard_nfl_pricing", _MOD_PATH)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

DEFAULT_RUN_ID = "20260908T160000Z__TUE__abcd1234"
DEFAULT_RUN_CREATED = "2026-09-08T15:30:00Z"
DEFAULT_CUTOFF = "2026-09-08T16:00:00+00:00"

# The ATS collection's newest instant is deliberately EARLIER than the TOTAL
# collection's newest instant, and neither collection is sorted -- so the
# expected market_as_of_utc can only come from a max over the UNION.
DEFAULT_ATS_SNAPSHOTS = ["2026-09-08T15:40:00Z", "2026-09-08T15:50:00Z"]
DEFAULT_TOTAL_SNAPSHOTS = ["2026-09-08T15:55:00Z", "2026-09-08T15:45:00Z"]
DEFAULT_MARKET_AS_OF = "2026-09-08T15:55:00Z"

# Field names the exporter must never invent, derive, or persist.
BANNED_PUBLIC_FIELDS = (
    "predicted_winner", "market_favorite", "model_fair_spread", "ats_side", "ats_edge_points",
    "total_side", "total_edge_points", "winner_agreement", "moneyline", "kelly", "expected_value",
    "bet_recommendation", "confidence_grade", "market_book_count",
)


# --------------------------------------------------------------------------- #
# synthetic fixture builders
# --------------------------------------------------------------------------- #
def _certified_market(*, consensus_line, eligible_books, snapshot_timestamps, drop_fields=()):
    """A synthetic CERTIFIED consensus market payload, shaped exactly like
    ``run_2026._consensus_entry``. Deliberately carries NO per-book American
    price, no -110 synthetic vig, and no moneyline/h2h data -- v2 must not
    need any of that."""
    market = {
        "eligible_books": eligible_books,
        "consensus_line": consensus_line,
        "consensus_novig_probability": 0.5,
        "bookmaker_keys": ["book_a", "book_b", "book_c"],
        "selected_returned_snapshot_timestamps": list(snapshot_timestamps)
        if isinstance(snapshot_timestamps, (list, tuple)) else snapshot_timestamps,
        "min_observation_age_hours": 0.5,
        "max_observation_age_hours": 1.5,
        "consensus_method": "median_of_devigged_per_book_quotes_across_eligible_books",
    }
    for field in drop_fields:
        market.pop(field, None)
    return market


_UNSET = object()  # lets a test pass an explicitly invalid None collection


def _markets_payload(
    *,
    ats_line=-3.0, ats_books=5, ats_snapshots=_UNSET, ats_drop_fields=(),
    total_line=45.5, total_books=4, total_snapshots=_UNSET, total_drop_fields=(),
    ats_entry="certified", total_entry="certified",
):
    """``ats_entry`` / ``total_entry`` select the shape of the market entry:

    ``"certified"``   -- normal entry carrying a ``market`` consensus payload
    ``"not_ready"``   -- production MARKET_NOT_READY entry (no ``market`` key)
    ``"absent"``      -- the market key is missing from ``markets`` entirely
    """
    payload = {}
    specs = (
        ("ATS", ats_entry, ats_line, ats_books,
         DEFAULT_ATS_SNAPSHOTS if ats_snapshots is _UNSET else ats_snapshots, ats_drop_fields),
        ("TOTAL", total_entry, total_line, total_books,
         DEFAULT_TOTAL_SNAPSHOTS if total_snapshots is _UNSET else total_snapshots, total_drop_fields),
    )
    for name, entry_kind, line, books, snapshots, drop_fields in specs:
        if entry_kind == "absent":
            continue
        if entry_kind == "not_ready":
            payload[name] = {"status": "MARKET_NOT_READY"}
            continue
        payload[name] = {
            "status": "OK",
            "market": _certified_market(
                consensus_line=line, eligible_books=books,
                snapshot_timestamps=snapshots, drop_fields=drop_fields,
            ),
            "raw_conditional_upper_probability": 0.55,
            "calibration_status": "CALIBRATED",
        }
    return payload


def _prediction_payload(
    *, game_id, horizon="TUE", target_cutoff_utc=DEFAULT_CUTOFF, season=2026, week="1", season_type="REG",
    home_team_id="KC", away_team_id="BUF", scheduled_kickoff_utc="2026-09-08T17:00:00Z",
    predicted_margin=3.4, predicted_total=46.8, model_status="OOF", markets=None, markets_kwargs=None,
):
    return {
        "game_id": game_id, "horizon": horizon, "target_cutoff_utc": target_cutoff_utc,
        "season": season, "week": week, "season_type": season_type,
        "home_team_id": home_team_id, "away_team_id": away_team_id,
        "scheduled_kickoff_utc": scheduled_kickoff_utc,
        "prediction": {
            "model_status": model_status, "predicted_margin": predicted_margin,
            "predicted_total": predicted_total, "model_config_hash": "cfg-hash",
        },
        "markets": markets if markets is not None else _markets_payload(**(markets_kwargs or {})),
        "market_state_hash": "market-hash",
        "certified_baseline_sha": "cert-sha", "horizon_feature_semantics_hash": "feat-hash",
        "operational_model_spec_hash": "spec-hash", "fix8_preregistration_hash": "prereg-hash",
    }


def _forecast_record(
    *, game_id, horizon="TUE", target_cutoff_utc=DEFAULT_CUTOFF, run_id=DEFAULT_RUN_ID,
    run_created_at_utc=DEFAULT_RUN_CREATED, created_at_utc="2026-09-08T16:00:05+00:00",
    prediction_kwargs=None,
):
    prediction = _prediction_payload(
        game_id=game_id, horizon=horizon, target_cutoff_utc=target_cutoff_utc, **(prediction_kwargs or {}),
    )
    record = {
        "game_id": game_id, "horizon": horizon, "target_cutoff_utc": target_cutoff_utc,
        "created_at_utc": created_at_utc, "git_commit": "deadbeef", "run_id": run_id,
        "run_created_at_utc": run_created_at_utc,
        "certified_baseline_tag": "certified-tag", "certified_baseline_sha": "cert-sha",
        "prediction": prediction,
    }
    record["prediction_hash"] = exp._sha256_hex(prediction)
    return record


def _write_record(forecast_dir: Path, record: dict, *, allow_nan: bool = False) -> Path:
    forecast_dir.mkdir(parents=True, exist_ok=True)
    safe_cutoff = str(record["target_cutoff_utc"]).replace(":", "").replace("+", "_")
    path = forecast_dir / f"{record['game_id']}__{safe_cutoff}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=allow_nan))
    return path


def _manifest(
    *, records, run_id=DEFAULT_RUN_ID, horizon="TUE", run_created_at_utc=DEFAULT_RUN_CREATED,
    status="SUCCESS", game_count=None, forecast_batch_hash_override=None,
):
    hashes = sorted(r["prediction_hash"] for r in records)
    batch_hash = forecast_batch_hash_override or exp._sha256_hex({"prediction_hashes": hashes})
    return {
        "run_id": run_id, "started_at_utc": run_created_at_utc, "completed_at_utc": "2026-09-08T16:00:10+00:00",
        "run_created_at_utc": run_created_at_utc,
        "horizon": horizon, "target_cutoff_utc": DEFAULT_CUTOFF,
        "git_commit": "deadbeef", "certified_model_tag": "certified-tag", "certified_model_sha": "cert-sha",
        "source_readiness": {}, "game_count": game_count if game_count is not None else len(records),
        "forecast_count": len(records), "abstention_count": 0,
        "market_ready_counts": {"ATS": len(records), "TOTAL": len(records)},
        "calibration_ready_counts": {"ATS": len(records), "TOTAL": len(records)},
        "input_hashes": {}, "output_hashes": {"forecast_batch_hash": batch_hash},
        "status": status, "detail": "",
    }


def _write_manifest(manifest_dir: Path, manifest: dict) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def _single_game_card(tmp_path, *, horizon="TUE", prediction_kwargs=None, allow_nan=False, game_count=None):
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    record = _forecast_record(game_id="G1", horizon=horizon, prediction_kwargs=prediction_kwargs)
    _write_record(forecast_dir, record, allow_nan=allow_nan)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=[record], horizon=horizon, game_count=game_count),
    )
    return manifest_path, forecast_dir, record


def _valid_two_game_card(tmp_path, *, horizon="TUE", run_id=DEFAULT_RUN_ID, run_created_at_utc=DEFAULT_RUN_CREATED):
    """G1 kicks off AFTER G2 -- exercises kickoff-ascending ordering."""
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    g1 = _forecast_record(
        game_id="G1", horizon=horizon, run_id=run_id, run_created_at_utc=run_created_at_utc,
        prediction_kwargs=dict(
            home_team_id="KC", away_team_id="BUF", scheduled_kickoff_utc="2026-09-08T17:00:00Z",
            predicted_margin=3.4, predicted_total=46.8,
            markets_kwargs=dict(ats_line=-3.0, ats_books=5, total_line=45.5, total_books=4),
        ),
    )
    g2 = _forecast_record(
        game_id="G2", horizon=horizon, run_id=run_id, run_created_at_utc=run_created_at_utc,
        prediction_kwargs=dict(
            home_team_id="HOU", away_team_id="NE", scheduled_kickoff_utc="2026-09-08T13:00:00Z",
            predicted_margin=-2.1, predicted_total=41.0,
            markets_kwargs=dict(ats_line=1.5, ats_books=6, total_line=42.5, total_books=3),
        ),
    )
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests",
        _manifest(records=[g1, g2], run_id=run_id, horizon=horizon, run_created_at_utc=run_created_at_utc),
    )
    return manifest_path, forecast_dir, [g1, g2]


def _run(manifest_path: Path, forecast_dir: Path, output_path: Path) -> dict:
    return exp.run_export(run_manifest_path=manifest_path, forecast_dir=forecast_dir, output_path=output_path)


def _card(manifest_path: Path, forecast_dir: Path) -> dict:
    return exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


# --------------------------------------------------------------------------- #
# 1-2: valid TUE / FRI pricing cards. 3-4: exact key sets and key order.
# 5-10: field mapping. 12-13: as-of / generated-at provenance. 31: ordering.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", ["TUE", "FRI"])
def test_valid_pricing_card_exports_exact_public_schema(tmp_path, horizon):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path, horizon=horizon)
    output_path = tmp_path / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    result = _run(manifest_path, forecast_dir, output_path)
    assert result["status"] == "OK"

    card = json.loads(output_path.read_text())

    # item 3: exact top-level keys, exact order
    assert set(card.keys()) == {"schema_version", "season", "week", "horizon", "generated_at_utc", "games"}
    assert list(card.keys()) == ["schema_version", "season", "week", "horizon", "generated_at_utc", "games"]
    assert card["schema_version"] == "wizard-nfl-pricing-v2"
    assert card["season"] == 2026
    assert card["week"] == 1 and isinstance(card["week"], int)
    assert card["horizon"] == horizon
    # item 13: generated_at_utc is the shared run_created_at_utc
    assert card["generated_at_utc"] == "2026-09-08T15:30:00Z"

    # item 31: kickoff-ascending -- G2 (13:00) precedes G1 (17:00)
    assert [g["game_id"] for g in card["games"]] == ["G2", "G1"]

    g1 = card["games"][1]
    expected_keys = [
        "game_id", "kickoff_utc", "away_team", "home_team",
        "predicted_home_margin", "predicted_game_total",
        "market_home_spread", "market_total", "market_as_of_utc",
        "market_ats_book_count", "market_total_book_count",
    ]
    # item 4: exact per-game keys, exact order
    assert set(g1.keys()) == set(expected_keys)
    assert list(g1.keys()) == expected_keys

    assert g1["kickoff_utc"] == "2026-09-08T17:00:00Z"
    assert g1["home_team"] == "KC"
    assert g1["away_team"] == "BUF"
    assert g1["predicted_home_margin"] == 3.4          # item 5
    assert g1["predicted_game_total"] == 46.8          # item 6
    assert g1["market_home_spread"] == -3.0            # item 7
    assert g1["market_total"] == 45.5                  # item 8
    assert g1["market_ats_book_count"] == 5            # item 9
    assert g1["market_total_book_count"] == 4          # item 10
    assert g1["market_as_of_utc"] == DEFAULT_MARKET_AS_OF  # item 12

    g2 = card["games"][0]
    assert g2["market_home_spread"] == 1.5
    assert g2["market_total"] == 42.5
    assert g2["market_ats_book_count"] == 6
    assert g2["market_total_book_count"] == 3


def test_games_with_identical_kickoff_sort_by_game_id(tmp_path):
    """Item 31 (tie-break half): identical kickoff instants order by game_id."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    same_kickoff = "2026-09-08T17:00:00Z"
    gb = _forecast_record(game_id="GB", prediction_kwargs=dict(
        scheduled_kickoff_utc=same_kickoff, home_team_id="SEA", away_team_id="LAR"))
    ga = _forecast_record(game_id="GA", prediction_kwargs=dict(
        scheduled_kickoff_utc=same_kickoff, home_team_id="KC", away_team_id="BUF"))
    _write_record(forecast_dir, gb)
    _write_record(forecast_dir, ga)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[gb, ga]))
    assert [g["game_id"] for g in _card(manifest_path, forecast_dir)["games"]] == ["GA", "GB"]


def test_ordering_is_deterministic_regardless_of_filesystem_write_order(tmp_path):
    """Item 31: two ledger directories holding the same run's forecasts,
    written in opposite order, produce byte-identical cards."""
    first, first_dir, records = _valid_two_game_card(tmp_path / "a")
    second_dir = tmp_path / "b" / "forecast-ledger" / "TUE"
    for record in reversed(records):
        _write_record(second_dir, record)
    second = _write_manifest(tmp_path / "b" / "run-manifests", _manifest(records=records))
    assert exp.serialize_card(_card(first, first_dir)) == exp.serialize_card(_card(second, second_dir))


# --------------------------------------------------------------------------- #
# 11: no single market_book_count. 14-17: no derived field is ever emitted.
# --------------------------------------------------------------------------- #
def test_no_single_market_book_count_field_exists(tmp_path):
    """Item 11: ATS and TOTAL counts stay independent -- no collapsed
    market_book_count, and neither count equals a min/max/mean of the two."""
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text()
    assert "market_book_count" not in text
    card = json.loads(text)
    for game in card["games"]:
        assert "market_book_count" not in game
        assert game["market_ats_book_count"] != game["market_total_book_count"]
        assert isinstance(game["market_ats_book_count"], int)
        assert isinstance(game["market_total_book_count"], int)


def test_no_derived_or_out_of_scope_field_is_emitted(tmp_path):
    """Items 14-17: no ATS edge/side, no total edge/side, no predicted_winner,
    no market_favorite, and no moneyline/Kelly/EV/recommendation/grade."""
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text()
    card = json.loads(text)
    for banned in BANNED_PUBLIC_FIELDS:
        assert banned not in text
        for game in card["games"]:
            assert banned not in game
    for game in card["games"]:
        assert not any("edge" in key or "side" in key or "winner" in key or "favorite" in key
                       for key in game)


def test_no_internal_market_or_qb_shadow_field_leaks_into_public_json(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    record = _forecast_record(game_id="G1")
    record["prediction"]["qb_projection_shock_fri"] = 12.5  # rogue field: never read, never emitted
    record["prediction_hash"] = exp._sha256_hex(record["prediction"])
    _write_record(forecast_dir, record)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[record]))
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text()
    for banned in (
        "consensus_line", "eligible_books", "selected_returned_snapshot_timestamps",
        "consensus_novig_probability", "bookmaker_keys", "consensus_method", "market_state_hash",
        "raw_conditional_upper_probability", "calibration_status", "qb_projection_shock",
        "model_config_hash", "prediction_hash", "run_id",
    ):
        assert banned not in text


# --------------------------------------------------------------------------- #
# 12: market as-of rule
# --------------------------------------------------------------------------- #
def test_market_as_of_is_max_over_union_of_ats_and_total_snapshots(tmp_path):
    """The winning instant must be able to come from EITHER collection."""
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(
        markets_kwargs=dict(
            ats_snapshots=["2026-09-08T15:59:00Z", "2026-09-08T15:10:00Z"],
            total_snapshots=["2026-09-08T15:20:00Z"],
        ),
    ))
    assert _card(manifest_path, forecast_dir)["games"][0]["market_as_of_utc"] == "2026-09-08T15:59:00Z"


def test_market_as_of_normalizes_offset_timestamps_to_utc_z(tmp_path):
    """A +02:00 instant that is genuinely the latest is emitted in UTC Z form."""
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(
        markets_kwargs=dict(
            ats_snapshots=["2026-09-08T18:05:00+02:00"],  # 16:05Z
            total_snapshots=["2026-09-08T15:55:00Z"],
        ),
    ))
    assert _card(manifest_path, forecast_dir)["games"][0]["market_as_of_utc"] == "2026-09-08T16:05:00Z"


def test_market_as_of_never_falls_back_to_any_other_timestamp(tmp_path):
    """Item 12: the emitted as-of is a genuine market instant, never
    run_created_at_utc / target_cutoff_utc / created_at_utc / now()."""
    manifest_path, forecast_dir, record = _single_game_card(tmp_path)
    as_of = _card(manifest_path, forecast_dir)["games"][0]["market_as_of_utc"]
    assert as_of == DEFAULT_MARKET_AS_OF
    assert as_of != record["run_created_at_utc"]
    assert as_of != record["created_at_utc"]
    assert as_of != record["target_cutoff_utc"]
    assert as_of != record["prediction"]["target_cutoff_utc"]


# --------------------------------------------------------------------------- #
# 13: generated_at provenance
# --------------------------------------------------------------------------- #
def test_generated_at_utc_comes_from_run_created_at_utc(tmp_path):
    manifest_path, forecast_dir, records = _valid_two_game_card(tmp_path)
    for record in records:
        assert record["created_at_utc"] != record["run_created_at_utc"]
    assert _card(manifest_path, forecast_dir)["generated_at_utc"] == "2026-09-08T15:30:00Z"


@pytest.mark.parametrize("bad_value", [None, "", "not-a-timestamp", "2026-09-08T15:30:00"])
def test_invalid_run_created_at_utc_fails_closed(tmp_path, bad_value):
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["run_created_at_utc"] = bad_value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(exp.WizardExportError):
        _card(manifest_path, forecast_dir)


# --------------------------------------------------------------------------- #
# 18-19: both certified markets are mandatory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry_kind", ["absent", "not_ready"])
def test_missing_ats_market_fails_closed(tmp_path, entry_kind):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(ats_entry=entry_kind)),
    )
    with pytest.raises(exp.WizardExportError, match="ATS"):
        _card(manifest_path, forecast_dir)


@pytest.mark.parametrize("entry_kind", ["absent", "not_ready"])
def test_missing_total_market_fails_closed(tmp_path, entry_kind):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(total_entry=entry_kind)),
    )
    with pytest.raises(exp.WizardExportError, match="TOTAL"):
        _card(manifest_path, forecast_dir)


def test_missing_markets_payload_entirely_fails_closed(tmp_path):
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(markets={}))
    with pytest.raises(exp.WizardExportError, match="ATS"):
        _card(manifest_path, forecast_dir)


def test_partial_pricing_board_is_never_published(tmp_path):
    """One certified game plus one market-less game must abort the WHOLE card
    -- not silently drop the game and publish a partial board."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    good = _forecast_record(game_id="G1")
    bad = _forecast_record(game_id="G2", prediction_kwargs=dict(
        scheduled_kickoff_utc="2026-09-08T13:00:00Z", home_team_id="HOU", away_team_id="NE",
        markets_kwargs=dict(total_entry="not_ready"),
    ))
    _write_record(forecast_dir, good)
    _write_record(forecast_dir, bad)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[good, bad]))
    output_path = tmp_path / "out" / "latest.json"
    with pytest.raises(exp.WizardExportError):
        _run(manifest_path, forecast_dir, output_path)
    assert not output_path.exists()
    assert not (output_path.parent / "archive").exists()


@pytest.mark.parametrize("market_name,drop_kwarg", [("ATS", "ats_drop_fields"), ("TOTAL", "total_drop_fields")])
@pytest.mark.parametrize("field", ["consensus_line", "eligible_books", "selected_returned_snapshot_timestamps"])
def test_certified_market_missing_a_required_field_fails_closed(tmp_path, market_name, drop_kwarg, field):
    """Items 18-20: each certified market must carry consensus_line,
    eligible_books AND selected_returned_snapshot_timestamps."""
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs={drop_kwarg: (field,)}),
    )
    with pytest.raises(exp.WizardExportError, match=field):
        _card(manifest_path, forecast_dir)


# --------------------------------------------------------------------------- #
# 20-21: snapshot-timestamp collection validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwarg", ["ats_snapshots", "total_snapshots"])
@pytest.mark.parametrize("bad_collection", [[], None, "2026-09-08T15:55:00Z", {}, 0])
def test_missing_or_non_list_timestamp_collection_fails_closed(tmp_path, kwarg, bad_collection):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs={kwarg: bad_collection}),
    )
    with pytest.raises(exp.WizardExportError, match="selected_returned_snapshot_timestamps"):
        _card(manifest_path, forecast_dir)


@pytest.mark.parametrize("kwarg", ["ats_snapshots", "total_snapshots"])
@pytest.mark.parametrize("bad_timestamp", [
    "not-a-timestamp",
    "2026-09-08T15:55:00",   # naive -- not timezone-aware
    "",
    None,
    1757347200,
])
def test_invalid_snapshot_timestamp_fails_closed(tmp_path, kwarg, bad_timestamp):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs={kwarg: ["2026-09-08T15:50:00Z", bad_timestamp]}),
    )
    with pytest.raises(exp.WizardExportError, match="selected_returned_snapshot_timestamps"):
        _card(manifest_path, forecast_dir)


# --------------------------------------------------------------------------- #
# 22-23: independent book-count floors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("books", [2, 1, 0, -1])
def test_ats_book_count_below_three_fails_closed(tmp_path, books):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(ats_books=books)),
    )
    with pytest.raises(exp.WizardExportError, match="eligible books"):
        _card(manifest_path, forecast_dir)


@pytest.mark.parametrize("books", [2, 1, 0, -1])
def test_total_book_count_below_three_fails_closed(tmp_path, books):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(total_books=books)),
    )
    with pytest.raises(exp.WizardExportError, match="eligible books"):
        _card(manifest_path, forecast_dir)


def test_exactly_three_eligible_books_is_accepted(tmp_path):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(ats_books=3, total_books=3)),
    )
    game = _card(manifest_path, forecast_dir)["games"][0]
    assert game["market_ats_book_count"] == 3
    assert game["market_total_book_count"] == 3


@pytest.mark.parametrize("kwarg", ["ats_books", "total_books"])
@pytest.mark.parametrize("bad_books", [None, "5", 5.0, 4.5, True, float("nan")])
def test_non_integer_book_count_fails_closed(tmp_path, kwarg, bad_books):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs={kwarg: bad_books}), allow_nan=True,
    )
    with pytest.raises(exp.WizardExportError, match="eligible-book count"):
        _card(manifest_path, forecast_dir)


# --------------------------------------------------------------------------- #
# 24-25: numeric semantics -- finite, non-null, never boolean, never rounded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["predicted_margin", "predicted_total"])
@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), float("-inf"), True, "3.5"])
def test_non_finite_model_value_fails_closed(tmp_path, field, bad_value):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs={field: bad_value}, allow_nan=True,
    )
    with pytest.raises(exp.WizardExportError, match=field):
        _card(manifest_path, forecast_dir)


@pytest.mark.parametrize("kwarg,expected_field", [
    ("ats_line", "ATS.market.consensus_line"),
    ("total_line", "TOTAL.market.consensus_line"),
])
@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), float("-inf"), True, "-3.5"])
def test_non_finite_market_value_fails_closed(tmp_path, kwarg, expected_field, bad_value):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs={kwarg: bad_value}), allow_nan=True,
    )
    with pytest.raises(exp.WizardExportError, match=expected_field):
        _card(manifest_path, forecast_dir)


def test_model_and_market_values_are_never_rounded(tmp_path):
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(
        predicted_margin=3.456789123, predicted_total=44.987654321,
        markets_kwargs=dict(ats_line=-2.750000001, total_line=45.499999999),
    ))
    game = _card(manifest_path, forecast_dir)["games"][0]
    assert game["predicted_home_margin"] == 3.456789123
    assert game["predicted_game_total"] == 44.987654321
    assert game["market_home_spread"] == -2.750000001
    assert game["market_total"] == 45.499999999


@pytest.mark.parametrize("ats_line,expected", [(-3.5, -3.5), (3.5, 3.5), (0.0, 0.0), (-0.5, -0.5)])
def test_market_home_spread_is_never_inverted(tmp_path, ats_line, expected):
    """Certified sportsbook notation is emitted verbatim: -3.5 means the HOME
    team is -3.5, +3.5 means the home team is +3.5, 0 is a pick'em."""
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(markets_kwargs=dict(ats_line=ats_line)),
    )
    assert _card(manifest_path, forecast_dir)["games"][0]["market_home_spread"] == expected


def test_predicted_home_margin_stays_signed_from_home_perspective(tmp_path):
    manifest_path, forecast_dir, _ = _single_game_card(
        tmp_path, prediction_kwargs=dict(predicted_margin=-6.25),
    )
    assert _card(manifest_path, forecast_dir)["games"][0]["predicted_home_margin"] == -6.25


# --------------------------------------------------------------------------- #
# 26-30: run-integrity fail-closed cases
# --------------------------------------------------------------------------- #
def test_duplicate_game_id_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", target_cutoff_utc=DEFAULT_CUTOFF)
    g1_dup = _forecast_record(game_id="G1", target_cutoff_utc="2026-09-08T16:05:00+00:00")
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g1_dup)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g1_dup], game_count=2))
    with pytest.raises(exp.WizardExportError, match="duplicate"):
        _card(manifest_path, forecast_dir)


def test_mixed_run_id_fails_closed(tmp_path):
    """A foreign run's forecast is never mixed into this run's card, so the
    manifest's declared game_count cannot be satisfied -- FAIL CLOSED."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    ours = _forecast_record(game_id="G1", run_id=DEFAULT_RUN_ID)
    foreign = _forecast_record(game_id="G2", run_id="some-other-run-id")
    _write_record(forecast_dir, ours)
    _write_record(forecast_dir, foreign)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=[ours, foreign], game_count=2),
    )
    with pytest.raises(exp.WizardExportError, match="game_count"):
        _card(manifest_path, forecast_dir)


def test_mixed_run_created_at_utc_within_one_run_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", run_created_at_utc="2026-09-08T15:30:00Z")
    g2 = _forecast_record(game_id="G2", run_created_at_utc="2026-09-08T15:31:00Z")
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2]))
    with pytest.raises(exp.WizardExportError, match="run_created_at_utc"):
        _card(manifest_path, forecast_dir)


def test_mixed_season_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", prediction_kwargs=dict(season=2026))
    g2 = _forecast_record(game_id="G2", prediction_kwargs=dict(season=2027))
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2]))
    with pytest.raises(exp.WizardExportError, match="season"):
        _card(manifest_path, forecast_dir)


def test_mixed_week_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", prediction_kwargs=dict(week="1"))
    g2 = _forecast_record(game_id="G2", prediction_kwargs=dict(week="2"))
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2]))
    with pytest.raises(exp.WizardExportError, match="week"):
        _card(manifest_path, forecast_dir)


def test_mixed_horizon_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", horizon="TUE")
    g2 = _forecast_record(game_id="G2", horizon="FRI")
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2], horizon="TUE"))
    with pytest.raises(exp.WizardExportError, match="horizon"):
        _card(manifest_path, forecast_dir)


@pytest.mark.parametrize("bad_horizon", ["SMOKE", "PRE", "TEST", "WED"])
def test_only_tue_fri_horizon_is_accepted(tmp_path, bad_horizon):
    forecast_dir = tmp_path / "forecast-ledger" / bad_horizon
    record = _forecast_record(game_id="G1", horizon=bad_horizon)
    _write_record(forecast_dir, record)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=[record], horizon=bad_horizon),
    )
    with pytest.raises(exp.WizardExportError, match="horizon"):
        exp.load_run_manifest(manifest_path)


def test_manifest_non_success_status_fails_closed(tmp_path):
    manifest_path, _, _ = _single_game_card(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "MARKET_NOT_READY"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(exp.WizardExportError, match="SUCCESS"):
        exp.load_run_manifest(manifest_path)


def test_tampered_prediction_hash_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    record = _forecast_record(game_id="G1")
    record["prediction"]["markets"]["ATS"]["market"]["consensus_line"] = -14.0  # tampered after hashing
    _write_record(forecast_dir, record)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[record]))
    with pytest.raises(exp.WizardExportError, match="prediction_hash"):
        _card(manifest_path, forecast_dir)


def test_forecast_batch_hash_mismatch_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    record = _forecast_record(game_id="G1")
    _write_record(forecast_dir, record)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=[record], forecast_batch_hash_override="0" * 64),
    )
    with pytest.raises(exp.WizardExportError, match="forecast_batch_hash"):
        _card(manifest_path, forecast_dir)


def test_only_season_2026_is_accepted(tmp_path):
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(season=2025))
    with pytest.raises(exp.WizardExportError, match="season"):
        _card(manifest_path, forecast_dir)


# --------------------------------------------------------------------------- #
# 32-35: archive immutability, idempotency, atomic latest
# --------------------------------------------------------------------------- #
def test_immutable_archive_writes_once(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    result = _run(manifest_path, forecast_dir, output_path)
    archive_file = Path(result["archive_path"])
    assert archive_file == output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    assert archive_file.is_file()
    assert json.loads(archive_file.read_text()) == json.loads(output_path.read_text())


def test_identical_archive_rerun_is_idempotent_success(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl-pricing" / "latest.json"
    first = _run(manifest_path, forecast_dir, output_path)
    before = Path(first["archive_path"]).read_bytes()
    second = _run(manifest_path, forecast_dir, output_path)
    assert second["status"] == "OK"
    assert Path(second["archive_path"]).read_bytes() == before


def test_differing_archive_fails_closed_and_leaves_archive_untouched(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl-pricing" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    archive_file = output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    before = archive_file.read_bytes()

    card = _card(manifest_path, forecast_dir)
    card["games"][0]["market_home_spread"] = 99.0  # a genuinely different card, same season/week/horizon
    with pytest.raises(exp.WizardExportError, match="different content"):
        exp.write_archive(
            output_path.parent / "archive", season=2026, week=1, horizon="TUE",
            payload_bytes=exp.serialize_card(card),
        )
    assert archive_file.read_bytes() == before


def test_latest_is_not_written_when_the_archive_step_fails(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl-pricing" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    before_latest = output_path.read_bytes()

    archive_file = output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    archive_file.write_bytes(b'{"schema_version": "poisoned"}\n')

    with pytest.raises(exp.WizardExportError):
        _run(manifest_path, forecast_dir, output_path)
    assert output_path.read_bytes() == before_latest
    assert not output_path.with_name(output_path.name + ".tmp").exists()


def test_latest_replacement_is_atomic_with_no_leftover_tmp(tmp_path, monkeypatch):
    """Item 35: latest.json is replaced via a same-directory temp file and
    os.replace, so a reader never observes a partial file and no .tmp is left
    behind."""
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl-pricing" / "latest.json"

    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src, dst, *args, **kwargs):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy_replace)
    _run(manifest_path, forecast_dir, output_path)

    assert replaced == [(str(output_path) + ".tmp", str(output_path))]
    assert output_path.is_file()
    assert not output_path.with_name(output_path.name + ".tmp").exists()
    assert json.loads(output_path.read_text())["schema_version"] == "wizard-nfl-pricing-v2"


# --------------------------------------------------------------------------- #
# 36: explicit run selection only -- no legacy path, no "latest" discovery
# --------------------------------------------------------------------------- #
def test_legacy_outputs_season_2026_path_is_never_read(tmp_path, monkeypatch):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    reads: list[Path] = []
    real_read_text, real_read_bytes = Path.read_text, Path.read_bytes

    def spy_read_text(self, *args, **kwargs):
        reads.append(Path(self))
        return real_read_text(self, *args, **kwargs)

    def spy_read_bytes(self, *args, **kwargs):
        reads.append(Path(self))
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.chdir(_REPO_ROOT)
    monkeypatch.setattr(Path, "read_text", spy_read_text)
    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")

    assert reads, "the exporter must actually read the explicit inputs"
    legacy_root = _REPO_ROOT / "outputs"
    for path in reads:
        resolved = path.resolve()
        assert legacy_root not in resolved.parents
        assert "season_2026" not in str(resolved)
        # every read is inside the explicitly supplied inputs or the explicit output tree
        assert str(resolved).startswith(str(tmp_path.resolve()))


def test_source_contains_no_latest_discovery_or_legacy_path():
    src = _MOD_PATH.read_text()
    for banned in ("outputs/season_2026", "st_mtime", "getmtime", "max(paths", "iterdir()", "rglob("):
        assert banned not in src


def test_cli_requires_all_three_explicit_arguments(tmp_path):
    for argv in (
        ["--run-manifest", "m.json"],
        ["--forecast-dir", "d"],
        ["--output", "o.json"],
        ["--run-manifest", "m.json", "--forecast-dir", "d"],
    ):
        with pytest.raises(SystemExit):
            exp.main(argv)


def test_cli_runs_end_to_end_and_fails_closed_with_exit_code_2(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "out" / "latest.json"
    assert exp.main([
        "--run-manifest", str(manifest_path),
        "--forecast-dir", str(forecast_dir),
        "--output", str(output_path),
    ]) == 0
    assert output_path.is_file()

    bad_manifest, bad_dir, _ = _single_game_card(
        tmp_path / "bad", prediction_kwargs=dict(markets_kwargs=dict(ats_entry="absent")),
    )
    bad_output = tmp_path / "bad-out" / "latest.json"
    assert exp.main([
        "--run-manifest", str(bad_manifest),
        "--forecast-dir", str(bad_dir),
        "--output", str(bad_output),
    ]) == 2
    assert not bad_output.exists()


def test_missing_manifest_or_forecast_dir_fails_closed(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    with pytest.raises(exp.WizardExportError, match="run manifest not found"):
        _run(tmp_path / "nope.json", forecast_dir, tmp_path / "out" / "latest.json")
    with pytest.raises(exp.WizardExportError, match="forecast-dir"):
        _run(manifest_path, tmp_path / "nope", tmp_path / "out" / "latest.json")


# --------------------------------------------------------------------------- #
# 37-39: no synthetic -110, no moneyline, no external API, no credential
# --------------------------------------------------------------------------- #
def test_synthetic_minus_110_price_is_not_required(tmp_path):
    """Item 37: the certified market fixture carries no per-book American
    price and no synthetic -110 vig, and the export still succeeds."""
    manifest_path, forecast_dir, record = _single_game_card(tmp_path)
    market = record["prediction"]["markets"]["ATS"]["market"]
    for absent in ("price", "american_odds", "home_price", "away_price", "vig", "hold", "odds"):
        assert absent not in market
    card = _card(manifest_path, forecast_dir)
    assert card["games"][0]["market_home_spread"] == -3.0
    for key in card["games"][0]:
        assert not any(token in key for token in ("price", "odds", "vig", "hold", "probability"))


def test_moneyline_market_is_not_required(tmp_path):
    """Item 38: only ATS and TOTAL exist in the fixture -- no H2H/moneyline
    entry at all -- and the export still succeeds and emits none."""
    manifest_path, forecast_dir, record = _single_game_card(tmp_path)
    assert set(record["prediction"]["markets"]) == {"ATS", "TOTAL"}
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text().lower()
    for banned in ("moneyline", "money_line", "h2h", "ml_"):
        assert banned not in text


def test_no_external_api_call_is_possible(tmp_path, monkeypatch):
    """Item 39: with every socket constructor poisoned, a full export still
    succeeds -- proving no network I/O occurs on the export path."""
    def poisoned(*args, **kwargs):
        raise AssertionError("the pricing exporter must never open a network connection")

    monkeypatch.setattr(socket, "socket", poisoned)
    monkeypatch.setattr(socket, "create_connection", poisoned)
    monkeypatch.setattr(socket, "getaddrinfo", poisoned)
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    assert _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")["status"] == "OK"


def test_source_references_no_network_client_or_credential():
    src = _MOD_PATH.read_text()
    for banned in (
        "os.environ", "getenv", "THE_ODDS_API_KEY", "BALLDONTLIE_API_KEY",
        "requests.", "urllib.request", "httpx", "socket.", "http" + "://", "http" + "s://",
    ):
        assert banned not in src


def test_export_succeeds_with_no_credentials_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    monkeypatch.delenv("NFL_MODEL_ARTIFACT_ROOT", raising=False)
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    assert _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")["status"] == "OK"


def test_no_model_fitting_or_market_reconstruction_module_is_imported(tmp_path):
    src = _MOD_PATH.read_text()
    for banned in ("nfl_hybrid", "sklearn", "pandas", "numpy", "import pd", "run_2026"):
        assert f"import {banned}" not in src
        assert f"from {banned}" not in src


# --------------------------------------------------------------------------- #
# 40: v1 exporter behaviour unchanged
# --------------------------------------------------------------------------- #
def _load_v1_fresh():
    spec = importlib.util.spec_from_file_location("_v1_probe", _V1_MOD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1_exporter_contract_is_unchanged(tmp_path):
    v1 = _load_v1_fresh()
    assert v1.SCHEMA_VERSION == "wizard-nfl-predictions-v1"
    assert v1.TOP_LEVEL_KEY_ORDER == (
        "schema_version", "season", "week", "horizon", "generated_at_utc", "games",
    )
    assert v1.GAME_KEY_ORDER == (
        "game_id", "kickoff_utc", "away_team", "home_team", "predicted_home_margin", "predicted_game_total",
    )
    # v2 binds v1's helpers read-only; it never rewrites v1's module globals.
    assert exp._v1.SCHEMA_VERSION == v1.SCHEMA_VERSION
    assert exp._v1.GAME_KEY_ORDER == v1.GAME_KEY_ORDER
    assert exp._v1.TOP_LEVEL_KEY_ORDER == v1.TOP_LEVEL_KEY_ORDER
    assert exp.SCHEMA_VERSION == "wizard-nfl-pricing-v2"
    assert exp.GAME_KEY_ORDER != v1.GAME_KEY_ORDER


def test_v1_export_of_the_same_certified_run_still_emits_only_the_v1_schema(tmp_path):
    """The v2 fixtures carry full certified market payloads; v1 must still
    ignore them completely and publish its own frozen six-field games."""
    v1 = _load_v1_fresh()
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    result = v1.run_export(run_manifest_path=manifest_path, forecast_dir=forecast_dir, output_path=output_path)
    assert result["status"] == "OK"

    card = json.loads(output_path.read_text())
    assert card["schema_version"] == "wizard-nfl-predictions-v1"
    assert list(card.keys()) == list(v1.TOP_LEVEL_KEY_ORDER)
    for game in card["games"]:
        assert list(game.keys()) == list(v1.GAME_KEY_ORDER)
    text = output_path.read_text()
    for banned in ("market_home_spread", "market_total", "market_as_of_utc",
                   "market_ats_book_count", "market_total_book_count", "consensus_line", "ATS"):
        assert banned not in text


def test_v1_and_v2_agree_on_the_shared_fields_and_use_separate_output_trees(tmp_path):
    v1 = _load_v1_fresh()
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    v1_out = tmp_path / "public" / "wizardofodds" / "nfl" / "latest.json"
    v2_out = tmp_path / "public" / "wizardofodds" / "nfl-pricing" / "latest.json"
    v1.run_export(run_manifest_path=manifest_path, forecast_dir=forecast_dir, output_path=v1_out)
    _run(manifest_path, forecast_dir, v2_out)

    v1_card = json.loads(v1_out.read_text())
    v2_card = json.loads(v2_out.read_text())
    assert v1_out.parent != v2_out.parent
    for key in ("season", "week", "horizon", "generated_at_utc"):
        assert v1_card[key] == v2_card[key]
    for v1_game, v2_game in zip(v1_card["games"], v2_card["games"]):
        for key in v1.GAME_KEY_ORDER:
            assert v1_game[key] == v2_game[key]


# --------------------------------------------------------------------------- #
# derived page rules -- TEST ONLY. These are NEVER persisted; the assertions
# below prove the published fields are sufficient for the page to compute them
# and that the sign conventions are the ones the page expects.
# --------------------------------------------------------------------------- #
def _derive(game: dict) -> dict:
    margin = game["predicted_home_margin"]
    spread = game["market_home_spread"]
    total_edge = game["predicted_game_total"] - game["market_total"]
    ats_edge = margin + spread
    return {
        "model_fair_home_spread": -margin,
        "sportsbook_implied_home_margin": -spread,
        "ats_home_edge": ats_edge,
        "ats_side": "HOME ATS" if ats_edge > 0 else ("AWAY ATS" if ats_edge < 0 else "NO ATS EDGE"),
        "total_edge": total_edge,
        "total_side": "OVER" if total_edge > 0 else ("UNDER" if total_edge < 0 else "NO TOTAL EDGE"),
        "model_winner": "HOME" if margin > 0 else ("AWAY" if margin < 0 else "PICK'EM"),
        "market_favorite": "HOME" if -spread > 0 else ("AWAY" if -spread < 0 else "PICK'EM"),
    }


@pytest.mark.parametrize(
    "margin,spread,total,market_total,expected",
    [
        # home favoured by the model beyond the number -> HOME ATS, model HOME, market HOME
        (7.0, -3.0, 48.0, 45.0, dict(model_fair_home_spread=-7.0, sportsbook_implied_home_margin=3.0,
                                     ats_home_edge=4.0, ats_side="HOME ATS", total_edge=3.0,
                                     total_side="OVER", model_winner="HOME", market_favorite="HOME")),
        # market prices the home team higher than the model -> AWAY ATS, UNDER
        (1.0, -6.5, 40.0, 44.5, dict(model_fair_home_spread=-1.0, sportsbook_implied_home_margin=6.5,
                                     ats_home_edge=-5.5, ats_side="AWAY ATS", total_edge=-4.5,
                                     total_side="UNDER", model_winner="HOME", market_favorite="HOME")),
        # exact agreement -> no edge either way; pick'em on both sides
        (0.0, 0.0, 44.0, 44.0, dict(model_fair_home_spread=0.0, sportsbook_implied_home_margin=0.0,
                                    ats_home_edge=0.0, ats_side="NO ATS EDGE", total_edge=0.0,
                                    total_side="NO TOTAL EDGE", model_winner="PICK'EM",
                                    market_favorite="PICK'EM")),
        # model likes the away team, market has the away team favoured
        (-4.0, 2.5, 39.5, 41.0, dict(model_fair_home_spread=4.0, sportsbook_implied_home_margin=-2.5,
                                     ats_home_edge=-1.5, ats_side="AWAY ATS", total_edge=-1.5,
                                     total_side="UNDER", model_winner="AWAY", market_favorite="AWAY")),
    ],
)
def test_page_derivable_values_are_computable_from_published_fields_only(
    tmp_path, margin, spread, total, market_total, expected,
):
    manifest_path, forecast_dir, _ = _single_game_card(tmp_path, prediction_kwargs=dict(
        predicted_margin=margin, predicted_total=total,
        markets_kwargs=dict(ats_line=spread, total_line=market_total),
    ))
    game = _card(manifest_path, forecast_dir)["games"][0]
    assert _derive(game) == expected
    for derived_key in expected:
        assert derived_key not in game


def test_serialized_card_is_deterministic_and_json_strict(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    card = _card(manifest_path, forecast_dir)
    first = exp.serialize_card(card)
    assert first == exp.serialize_card(_card(manifest_path, forecast_dir))
    assert first.endswith(b"\n")
    assert json.loads(first.decode("utf-8")) == card
    # allow_nan=False: a non-finite value could never be serialized even if it
    # somehow reached this point
    card["games"][0]["market_total"] = float("nan")
    with pytest.raises(ValueError):
        exp.serialize_card(card)


def test_builtins_open_is_never_used_to_read_outside_the_explicit_inputs(tmp_path, monkeypatch):
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")
    for path in opened:
        assert str(_REPO_ROOT / "outputs") not in path
