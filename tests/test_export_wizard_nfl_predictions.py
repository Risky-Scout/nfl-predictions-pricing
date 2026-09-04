"""Focused tests for the wizard-nfl-predictions-v1 publishing exporter.

All fixtures are synthetic forecast-of-record JSON + synthetic run-manifest
JSON constructed directly inside pytest tmp_path directories -- never real
Week 1 forecasts, never live BallDontLie, never 2024/2025 outcomes, never
``/tmp/wizard-nfl-predictions-v1.json``. This file never runs the real
production pipeline and never performs a real export.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_wizard_nfl_predictions.py"
_spec = importlib.util.spec_from_file_location("export_wizard_nfl_predictions", _MOD_PATH)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

DEFAULT_RUN_ID = "20260908T160000Z__TUE__abcd1234"
DEFAULT_RUN_CREATED = "2026-09-08T15:30:00Z"
DEFAULT_CUTOFF = "2026-09-08T16:00:00+00:00"


# --------------------------------------------------------------------------- #
# synthetic fixture builders
# --------------------------------------------------------------------------- #
def _prediction_payload(
    *, game_id, horizon="TUE", target_cutoff_utc=DEFAULT_CUTOFF, season=2026, week="1", season_type="REG",
    home_team_id="KC", away_team_id="BUF", scheduled_kickoff_utc="2026-09-08T17:00:00Z",
    predicted_margin=3.4, predicted_total=46.8, model_status="OOF",
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
        "markets": {
            "ATS": {"status": "OK", "market": {"consensus_line": -3.0, "consensus_novig_probability": 0.55}},
            "TOTAL": {"status": "OK", "market": {"consensus_line": 45.5, "consensus_novig_probability": 0.5}},
        },
        "market_state_hash": "market-hash",
        "certified_baseline_sha": "cert-sha", "horizon_feature_semantics_hash": "feat-hash",
        "operational_model_spec_hash": "spec-hash", "fix8_preregistration_hash": "prereg-hash",
    }


def _forecast_record(
    *, game_id, horizon="TUE", target_cutoff_utc=DEFAULT_CUTOFF, run_id=DEFAULT_RUN_ID,
    run_created_at_utc=DEFAULT_RUN_CREATED, created_at_utc="2026-09-08T16:00:05+00:00",
    prediction_kwargs=None, prediction_hash_override=None,
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
    record["prediction_hash"] = prediction_hash_override or exp._sha256_hex(prediction)
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


def _valid_two_game_card(tmp_path, *, horizon="TUE", run_id=DEFAULT_RUN_ID, run_created_at_utc=DEFAULT_RUN_CREATED):
    """G1 kicks off AFTER G2 -- exercises kickoff-ascending sort."""
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    g1 = _forecast_record(
        game_id="G1", horizon=horizon, run_id=run_id, run_created_at_utc=run_created_at_utc,
        prediction_kwargs=dict(home_team_id="KC", away_team_id="BUF", scheduled_kickoff_utc="2026-09-08T17:00:00Z",
                                predicted_margin=3.4, predicted_total=46.8),
    )
    g2 = _forecast_record(
        game_id="G2", horizon=horizon, run_id=run_id, run_created_at_utc=run_created_at_utc,
        prediction_kwargs=dict(home_team_id="HOU", away_team_id="NE", scheduled_kickoff_utc="2026-09-08T13:00:00Z",
                                predicted_margin=-2.1, predicted_total=41.0),
    )
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest = _manifest(records=[g1, g2], run_id=run_id, horizon=horizon, run_created_at_utc=run_created_at_utc)
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    return manifest_path, forecast_dir, [g1, g2]


def _run(manifest_path: Path, forecast_dir: Path, output_path: Path) -> dict:
    return exp.run_export(run_manifest_path=manifest_path, forecast_dir=forecast_dir, output_path=output_path)


# --------------------------------------------------------------------------- #
# 1-2: valid TUE / FRI exports exact schema. 5-9: field mapping. 30: sort.
# 41-42: exact key sets.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", ["TUE", "FRI"])
def test_valid_run_exports_exact_public_schema(tmp_path, horizon):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path, horizon=horizon)
    result = _run(manifest_path, forecast_dir, tmp_path / "public" / "nfl" / "latest.json")
    assert result["status"] == "OK"

    card = json.loads((tmp_path / "public" / "nfl" / "latest.json").read_text())
    assert set(card.keys()) == {"schema_version", "season", "week", "horizon", "generated_at_utc", "games"}
    assert list(card.keys()) == ["schema_version", "season", "week", "horizon", "generated_at_utc", "games"]
    assert card["schema_version"] == "wizard-nfl-predictions-v1"
    assert card["season"] == 2026
    assert card["week"] == 1 and isinstance(card["week"], int)
    assert card["horizon"] == horizon
    assert card["generated_at_utc"] == "2026-09-08T15:30:00Z"

    # sort: G2 (13:00) before G1 (17:00) -- item 30
    assert [g["game_id"] for g in card["games"]] == ["G2", "G1"]

    g1 = card["games"][1]
    assert set(g1.keys()) == {"game_id", "kickoff_utc", "away_team", "home_team",
                               "predicted_home_margin", "predicted_game_total"}
    assert list(g1.keys()) == ["game_id", "kickoff_utc", "away_team", "home_team",
                                "predicted_home_margin", "predicted_game_total"]
    assert g1["kickoff_utc"] == "2026-09-08T17:00:00Z"
    assert g1["home_team"] == "KC"
    assert g1["away_team"] == "BUF"
    assert g1["predicted_home_margin"] == 3.4
    assert g1["predicted_game_total"] == 46.8
    assert "predicted_winner" not in g1  # item 10


def test_games_with_identical_kickoff_sort_by_game_id(tmp_path):
    horizon = "TUE"
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    same_kickoff = "2026-09-08T17:00:00Z"
    gb = _forecast_record(game_id="GB", horizon=horizon,
                           prediction_kwargs=dict(scheduled_kickoff_utc=same_kickoff, home_team_id="SEA", away_team_id="LAR"))
    ga = _forecast_record(game_id="GA", horizon=horizon,
                           prediction_kwargs=dict(scheduled_kickoff_utc=same_kickoff, home_team_id="KC", away_team_id="BUF"))
    _write_record(forecast_dir, gb)
    _write_record(forecast_dir, ga)
    manifest = _manifest(records=[gb, ga], horizon=horizon)
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    result = _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")
    card = json.loads(Path(result["latest_path"]).read_text())
    assert [g["game_id"] for g in card["games"]] == ["GA", "GB"]


# --------------------------------------------------------------------------- #
# 3-4: generated_at_utc provenance
# --------------------------------------------------------------------------- #
def test_generated_at_utc_comes_from_run_created_at_utc_not_created_at_utc(tmp_path):
    manifest_path, forecast_dir, records = _valid_two_game_card(tmp_path, run_created_at_utc="2026-09-08T15:30:00Z")
    # per-game created_at_utc is deliberately a totally different instant
    for r in records:
        assert r["created_at_utc"] != r["run_created_at_utc"]
    card = exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)
    assert card["generated_at_utc"] == "2026-09-08T15:30:00Z"


# --------------------------------------------------------------------------- #
# 11-13: week/season/horizon acceptance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("week_value,expected", [(1, 1), ("1", 1), ("01", 1), (17, 17)])
def test_public_week_is_integer_from_valid_representations(tmp_path, week_value, expected):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(week=week_value))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    card = exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)
    assert card["week"] == expected and isinstance(card["week"], int) and not isinstance(card["week"], bool)


@pytest.mark.parametrize("bad_week", [None, "Week 1", "1.5", 1.5, True])
def test_invalid_week_representation_fails_closed(tmp_path, bad_week):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(week=bad_week))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_only_season_2026_is_accepted(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(season=2025))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="season"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


@pytest.mark.parametrize("bad_horizon", ["SMOKE", "PRE", "TEST", "WED"])
def test_only_tue_fri_horizon_is_accepted(tmp_path, bad_horizon):
    forecast_dir = tmp_path / "forecast-ledger" / bad_horizon
    rec = _forecast_record(game_id="G1", horizon=bad_horizon)
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec], horizon=bad_horizon))
    with pytest.raises(exp.WizardExportError, match="horizon"):
        exp.load_run_manifest(manifest_path)


# --------------------------------------------------------------------------- #
# 14-18: run-consistency fail-closed cases
# --------------------------------------------------------------------------- #
def test_mixed_run_id_present_but_expected_count_unmet_fails_closed(tmp_path):
    """A directory holding both this run's forecast and another run's
    forecast: the foreign record is excluded from selection (never mixed
    in), and because the manifest declared game_count=2 for ITS OWN run,
    completeness cannot be proven with only 1 genuinely matching record --
    FAIL CLOSED (items 14 + 32)."""
    horizon = "TUE"
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    ours = _forecast_record(game_id="G1", horizon=horizon, run_id=DEFAULT_RUN_ID)
    foreign = _forecast_record(game_id="G2", horizon=horizon, run_id="some-other-run-id")
    _write_record(forecast_dir, ours)
    _write_record(forecast_dir, foreign)
    # manifest declares this run produced 2 forecasts, but only 1 file truly belongs to it
    manifest = _manifest(records=[ours, foreign], run_id=DEFAULT_RUN_ID, horizon=horizon, game_count=2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    with pytest.raises(exp.WizardExportError, match="game_count"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_foreign_run_id_file_is_excluded_when_completeness_still_holds(tmp_path):
    """Item 31: a wrong-run file coexisting in the same horizon directory
    (the normal, expected production shape) is excluded from the card --
    the export still succeeds and contains ONLY the correct run's game."""
    horizon = "TUE"
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    ours = _forecast_record(game_id="G1", horizon=horizon, run_id=DEFAULT_RUN_ID)
    foreign = _forecast_record(game_id="G2", horizon=horizon, run_id="an-earlier-run-id")
    _write_record(forecast_dir, ours)
    _write_record(forecast_dir, foreign)
    manifest = _manifest(records=[ours], run_id=DEFAULT_RUN_ID, horizon=horizon, game_count=1)
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    card = exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)
    assert [g["game_id"] for g in card["games"]] == ["G1"]


def test_mixed_run_created_at_utc_within_same_run_id_fails_closed(tmp_path):
    horizon = "TUE"
    forecast_dir = tmp_path / "forecast-ledger" / horizon
    g1 = _forecast_record(game_id="G1", horizon=horizon, run_created_at_utc="2026-09-08T15:30:00Z")
    g2 = _forecast_record(game_id="G2", horizon=horizon, run_created_at_utc="2026-09-08T15:31:00Z")  # same run_id, different instant
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest = _manifest(records=[g1, g2], horizon=horizon, run_created_at_utc="2026-09-08T15:30:00Z")
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    with pytest.raises(exp.WizardExportError, match="run_created_at_utc"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_mixed_horizon_within_same_run_id_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", horizon="TUE")
    g2 = _forecast_record(game_id="G2", horizon="FRI")  # same run_id, different horizon
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest = _manifest(records=[g1, g2], horizon="TUE")
    manifest_path = _write_manifest(tmp_path / "run-manifests", manifest)
    with pytest.raises(exp.WizardExportError, match="horizon"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_mixed_season_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", prediction_kwargs=dict(season=2026))
    g2 = _forecast_record(game_id="G2", prediction_kwargs=dict(season=2027))
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2]))
    with pytest.raises(exp.WizardExportError, match="season"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_mixed_week_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", prediction_kwargs=dict(week="1"))
    g2 = _forecast_record(game_id="G2", prediction_kwargs=dict(week="2"))
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g2)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g2]))
    with pytest.raises(exp.WizardExportError, match="week"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


# --------------------------------------------------------------------------- #
# 19: duplicate game_id
# --------------------------------------------------------------------------- #
def test_duplicate_game_id_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    g1 = _forecast_record(game_id="G1", target_cutoff_utc=DEFAULT_CUTOFF)
    g1_dup = _forecast_record(game_id="G1", target_cutoff_utc="2026-09-08T16:05:00+00:00")  # different filename, same game_id
    _write_record(forecast_dir, g1)
    _write_record(forecast_dir, g1_dup)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[g1, g1_dup], game_count=2))
    with pytest.raises(exp.WizardExportError, match="duplicate"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


# --------------------------------------------------------------------------- #
# 20-28: per-field validation fail-closed cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_value", [None, "", "   "])
def test_missing_team_fails_closed(tmp_path, bad_value):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(home_team_id=bad_value))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="home_team_id"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_identical_home_and_away_team_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(home_team_id="KC", away_team_id="KC"))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="equals"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_missing_kickoff_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(scheduled_kickoff_utc=None))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="scheduled_kickoff_utc"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


@pytest.mark.parametrize("bad_kickoff", ["not-a-timestamp", "2026-09-08T17:00:00"])  # unparseable / naive
def test_invalid_timestamp_fails_closed(tmp_path, bad_kickoff):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(scheduled_kickoff_utc=bad_kickoff))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_missing_margin_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_margin=None))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="predicted_margin"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_missing_total_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_total=None))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="predicted_total"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_nan_margin_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_margin=float("nan")))
    _write_record(forecast_dir, rec, allow_nan=True)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="finite"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_infinity_total_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_total=float("inf")))
    _write_record(forecast_dir, rec, allow_nan=True)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="finite"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_boolean_rejected_as_numeric_prediction(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_margin=True))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="numeric"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_predictions_are_not_rounded(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_margin=3.456789123, predicted_total=44.987654321))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    card = exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)
    assert card["games"][0]["predicted_home_margin"] == 3.456789123
    assert card["games"][0]["predicted_game_total"] == 44.987654321


# --------------------------------------------------------------------------- #
# 32: manifest count mismatch (plain, no foreign-run complication)
# --------------------------------------------------------------------------- #
def test_manifest_forecast_count_mismatch_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1")
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec], game_count=2))
    with pytest.raises(exp.WizardExportError, match="game_count"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_tampered_prediction_hash_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1")
    rec["prediction"]["prediction"]["predicted_margin"] = 999.0  # tamper AFTER hash was computed
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    with pytest.raises(exp.WizardExportError, match="prediction_hash"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_forecast_batch_hash_mismatch_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1")
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=[rec], forecast_batch_hash_override="0" * 64),
    )
    with pytest.raises(exp.WizardExportError, match="forecast_batch_hash"):
        exp.build_public_card(exp.load_run_manifest(manifest_path), forecast_dir)


def test_manifest_non_success_status_fails_closed(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1")
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec], status="MODEL_NOT_READY"))
    with pytest.raises(exp.WizardExportError, match="SUCCESS"):
        exp.load_run_manifest(manifest_path)


# --------------------------------------------------------------------------- #
# 33-37: archive + latest.json write discipline
# --------------------------------------------------------------------------- #
def test_immutable_archive_writes_once(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    result = _run(manifest_path, forecast_dir, output_path)
    archive_file = Path(result["archive_path"])
    assert archive_file == output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    assert archive_file.is_file()
    assert json.loads(archive_file.read_text()) == json.loads(output_path.read_text())


def test_identical_archive_rerun_is_idempotent_success(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    first = _run(manifest_path, forecast_dir, output_path)
    before = Path(first["archive_path"]).read_bytes()
    second = _run(manifest_path, forecast_dir, output_path)  # exact same inputs
    assert second["status"] == "OK"
    assert Path(second["archive_path"]).read_bytes() == before


def test_differing_archive_rerun_fails_closed_and_leaves_archive_untouched(tmp_path):
    manifest_path, forecast_dir, records = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    archive_file = output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    before = archive_file.read_bytes()

    # Build a DIFFERENT valid card for the exact same season/week/horizon
    # (a hypothetical re-run whose numbers changed) and attempt to archive it.
    manifest2, forecast_dir2, _ = _valid_two_game_card(
        tmp_path / "second", run_id="different-run-id-000",
    )
    card = exp.build_public_card(exp.load_run_manifest(manifest2), forecast_dir2)
    card["games"][0]["predicted_home_margin"] = 999.0
    payload = exp.serialize_card(card)
    with pytest.raises(exp.WizardExportError, match="different content"):
        exp.write_archive(output_path.parent / "archive", season=2026, week=1, horizon="TUE", payload_bytes=payload)
    assert archive_file.read_bytes() == before  # never overwritten


def test_latest_not_written_when_archive_fails(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    before_latest = output_path.read_bytes()

    # Poison the archive file so ANY subsequent export for this season/week/horizon must fail closed.
    archive_file = output_path.parent / "archive" / "season=2026" / "week=01" / "horizon=TUE.json"
    archive_file.write_bytes(b'{"schema_version": "poisoned"}\n')

    with pytest.raises(exp.WizardExportError):
        _run(manifest_path, forecast_dir, output_path)
    assert output_path.read_bytes() == before_latest  # latest.json untouched by the failed attempt
    assert not output_path.with_name(output_path.name + ".tmp").exists()  # no leftover partial write


def test_latest_replacement_is_atomic_no_leftover_tmp(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "public" / "nfl" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    assert output_path.is_file()
    assert not (output_path.parent / (output_path.name + ".tmp")).exists()


# --------------------------------------------------------------------------- #
# 38-40: no market / QB-shadow / credential surface
# --------------------------------------------------------------------------- #
def test_no_market_fields_enter_public_json(tmp_path):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text()
    for banned in ("markets", "consensus_line", "consensus_novig_probability", "market_state_hash", "ATS", "TOTAL"):
        assert banned not in text


def test_no_qb_shadow_fields_enter_public_json(tmp_path):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1")
    rec["prediction"]["qb_projection_shock_fri"] = 12.5  # rogue field, must never be read/emitted
    rec["prediction"]["home_qb_projection_change"] = 4.0
    rec["prediction_hash"] = exp._sha256_hex(rec["prediction"])  # keep the fixture internally consistent
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    output_path = tmp_path / "out" / "latest.json"
    _run(manifest_path, forecast_dir, output_path)
    text = output_path.read_text()
    assert "qb_projection_shock" not in text
    assert "qb_projection_change" not in text


def test_no_credential_environment_variables_referenced_in_source():
    src = _MOD_PATH.read_text()
    for banned in ("os.environ", "getenv", "THE_ODDS_API_KEY", "BALLDONTLIE_API_KEY", "requests.", "urllib.request", "http://", "https://"):
        assert banned not in src


def test_export_succeeds_with_empty_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    result = _run(manifest_path, forecast_dir, tmp_path / "out" / "latest.json")
    assert result["status"] == "OK"


# --------------------------------------------------------------------------- #
# CLI smoke (explicit args only -- no "find latest" flag exists at all)
# --------------------------------------------------------------------------- #
def test_cli_requires_all_three_explicit_arguments(tmp_path, capsys):
    with pytest.raises(SystemExit):
        exp.main(["--run-manifest", str(tmp_path / "m.json")])


def test_cli_runs_end_to_end(tmp_path, capsys):
    manifest_path, forecast_dir, _ = _valid_two_game_card(tmp_path)
    output_path = tmp_path / "out" / "latest.json"
    rc = exp.main([
        "--run-manifest", str(manifest_path),
        "--forecast-dir", str(forecast_dir),
        "--output", str(output_path),
    ])
    assert rc == 0
    assert output_path.is_file()


def test_cli_fail_closed_exit_code(tmp_path, capsys):
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    rec = _forecast_record(game_id="G1", prediction_kwargs=dict(predicted_margin=None))
    _write_record(forecast_dir, rec)
    manifest_path = _write_manifest(tmp_path / "run-manifests", _manifest(records=[rec]))
    rc = exp.main([
        "--run-manifest", str(manifest_path),
        "--forecast-dir", str(forecast_dir),
        "--output", str(tmp_path / "out" / "latest.json"),
    ])
    assert rc == 2
    assert not (tmp_path / "out" / "latest.json").exists()
