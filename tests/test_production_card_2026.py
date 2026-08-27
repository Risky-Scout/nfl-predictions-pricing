"""Focused + integration tests for the 2026 production pipeline (Sections
12-24 of the FINAL REVIEW CERTIFICATION).

Almost all tests in this file are HERMETIC: they use a small synthetic,
deterministic games population (``_synthetic_games``) and, where a
calibration seed is needed, a synthetic seed payload written under
``tmp_path`` -- never ``resolve("backfill.games")`` or the real
``$NFL_MODEL_ARTIFACT_ROOT``. They pass with both ``NFL_MODEL_DATA_ROOT``
and ``NFL_MODEL_ARTIFACT_ROOT`` unset (this is exactly what GitHub-hosted
CI runs with).

The exception is ``TestRealHistoricalIntegration``, which proves the same
behaviors end-to-end against the real, already-elapsed 2024 week-3 TUE/FRI
card from the certified 2020-2025 population. It is marked
``extended_data`` and skips cleanly (with a precise reason) when
``NFL_MODEL_DATA_ROOT``/``backfill.games`` is unavailable -- see
``tests/test_chronological_oof_extended.py`` for the identical repo
convention this follows. It still runs wherever the private data estate is
available (locally, or on a properly provisioned self-hosted runner).

Every write in this file goes through ``operational_root``/
``artifact_root_path`` overrides pointing at a pytest ``tmp_path`` -- this
file NEVER writes into the real
``$NFL_MODEL_ARTIFACT_ROOT/production-2026/`` ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nfl_hybrid.calibration.three_way import CalibrationConfig, _predict_conditional
from nfl_hybrid.data.external_data import ExternalDataUnavailableError, artifact_root, resolve
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.production import run_2026 as prod

# ===========================================================================
# Synthetic (hermetic) fixtures.
# ===========================================================================
_SYNTHETIC_SEASON = 2023
_SYNTHETIC_TEAMS = ["BUF", "MIA", "NE", "NYJ", "KC", "LAC", "DEN", "LV"]
# 15 weeks x 4 games/week: by the last two weeks, >=48 strictly-prior games
# exist for both TUE and FRI (min_training_games=48), so those weeks get
# real status="OOF" predictions -- exercising the full certified point-
# forecast + market/calibration path, not just the fail-closed one.
_SYNTHETIC_WEEKS = 15


def _synthetic_games(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_num = 0
    for week in range(1, _SYNTHETIC_WEEKS + 1):
        monday = pd.Timestamp(f"{_SYNTHETIC_SEASON}-09-04") + pd.Timedelta(weeks=week - 1)
        for pair in range(4):
            home, away = _SYNTHETIC_TEAMS[pair * 2], _SYNTHETIC_TEAMS[pair * 2 + 1]
            kickoff = monday + pd.Timedelta(days=6, hours=17)  # Sunday, after both TUE and FRI cutoffs
            rows.append({
                "game_id": f"{_SYNTHETIC_SEASON}_{week:02d}_{away}_{home}_{game_num}",
                "season": _SYNTHETIC_SEASON, "week": week, "season_type": "REG",
                "home_team_id": home, "away_team_id": away,
                "scheduled_kickoff_utc": kickoff.tz_localize("UTC"),
                "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                "neutral_site": False,
            })
            game_num += 1
    return pd.DataFrame(rows)


def _last_card_cutoffs(games: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """The real TUE/FRI cutoffs for the LAST card in ``games``, computed via
    the certified membership-ledger algorithm itself -- never hand-derived
    date arithmetic that could silently drift from the real cutoff rule."""
    ledger = he.build_horizon_membership_ledger(games)
    last_week = int(pd.to_numeric(ledger["week"], errors="coerce").max())
    last_card = ledger[ledger["week"] == last_week]
    return {
        "TUE": pd.Timestamp(last_card["tue_cutoff_utc"].iloc[0]),
        "FRI": pd.Timestamp(last_card["fri_cutoff_utc"].iloc[0]),
    }


def _write_synthetic_calibration_seed(root: Path, *, fitted: bool = True) -> Path:
    """Writes a tiny, schema-correct ``production_calibration_seed.json``
    (same shape as the real certified file: one entry per stream, each with
    a ``conditional_calibrator_state`` and ``push_scale_state``) under
    ``root`` -- exercises the SAME production loading/application code
    (:func:`prod.apply_frozen_conditional_calibrator`,
    :func:`prod.price_and_calibrate`) with synthetic-but-representative
    numbers, never a mock of that code."""
    seed_dir = root / "fix8-official-oof-calibration-2026"
    seed_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "conditional_calibrator_state": (
            {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]} if fitted else {"fitted": False}
        ),
        "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}},
    }
    seed = {stream: entry for stream in ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")}
    path = seed_dir / "production_calibration_seed.json"
    path.write_text(json.dumps(seed, indent=2))
    return path


@pytest.fixture
def synthetic_games() -> pd.DataFrame:
    return _synthetic_games()


# ===========================================================================
# Preflight (Section 17). Hermetic: the calibration-seed override and
# secret-leak checks need no real data at all. The overall READY/
# READY_WAITING_FOR_FIRST_DUE_CUTOFF verdict genuinely requires the real
# backfill.games source (schedule_source is never overridable -- it is a
# read-only live data feed, not a generated artifact) and is therefore
# covered in TestRealHistoricalIntegration below instead.
# ===========================================================================
def test_preflight_calibration_seed_check_respects_artifact_root_path_override(tmp_path):
    """The bug this test guards against: run_preflight(artifact_root_path=...)
    must resolve the Fix-8 calibration seed under the SAME injected root it
    uses for the writable-directory checks, never a bare artifact_root()
    call that requires the real NFL_MODEL_ARTIFACT_ROOT to even be set."""
    _write_synthetic_calibration_seed(tmp_path)
    result = prod.run_preflight(artifact_root_path=tmp_path)
    assert result["checks"]["fix8_calibration_seed"]["status"] == "OK"
    assert result["checks"]["fix8_calibration_seed"]["path"].startswith(str(tmp_path))


def test_preflight_missing_calibration_seed_reported_not_crashed(tmp_path):
    result = prod.run_preflight(artifact_root_path=tmp_path)  # tmp_path deliberately empty
    assert result["checks"]["fix8_calibration_seed"]["status"] == "MISSING"
    assert "calibration_seed_missing" in result["blocking_problems"]
    assert result["overall_status"] == "NOT_READY"


def test_preflight_never_crashes_and_never_leaks_secrets_with_no_env_roots(tmp_path, monkeypatch):
    """Runs with both external-data env vars explicitly unset (the real
    GitHub-hosted CI condition) -- must degrade gracefully to NOT_READY
    statuses, never raise, and never print a secret VALUE."""
    monkeypatch.delenv("NFL_MODEL_DATA_ROOT", raising=False)
    monkeypatch.delenv("NFL_MODEL_ARTIFACT_ROOT", raising=False)
    result = prod.run_preflight(artifact_root_path=tmp_path)
    dumped = json.dumps(result)
    for name in prod.env_var_names_from_env_example():
        assert f'"{name}": "SET"' in dumped or f'"{name}": "UNSET"' in dumped
    import os
    for name, value in os.environ.items():
        if name in prod.env_var_names_from_env_example() and value:
            assert value not in dumped


def test_preflight_writable_dirs_isolated_from_real_root(tmp_path):
    prod.run_preflight(artifact_root_path=tmp_path)
    assert (tmp_path / "production-2026" / "forecast-ledger").is_dir()
    assert not (tmp_path / "production-2026" / "forecast-ledger" / "TUE").exists()


# ===========================================================================
# DST-aware TUE/FRI due-run logic (Section 13) -- already fully hermetic,
# unchanged.
# ===========================================================================
def test_due_window_true_at_exact_cutoff_and_false_just_before_and_after():
    at_cutoff = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:00:00Z"), "TUE")  # a real Tuesday, EST (UTC-5)
    assert at_cutoff["due"] is True
    just_before = prod.is_within_due_window(pd.Timestamp("2026-01-06T16:59:59Z"), "TUE")
    assert just_before["due"] is False
    just_after_window = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:20:00Z"), "TUE")
    assert just_after_window["due"] is False


def test_due_window_is_dst_aware_across_edt_and_est():
    edt_due = prod.is_within_due_window(pd.Timestamp("2026-08-25T16:00:00Z"), "TUE")  # EDT, UTC-4
    assert edt_due["due"] is True
    est_due = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:00:00Z"), "TUE")  # EST, UTC-5
    assert est_due["due"] is True
    assert prod.is_within_due_window(pd.Timestamp("2026-01-06T16:00:00Z"), "TUE")["due"] is False


def test_due_window_wrong_weekday_never_due():
    monday = prod.is_within_due_window(pd.Timestamp("2026-08-24T16:00:00Z"), "TUE")
    assert monday["due"] is False


def test_run_due_outside_window_is_not_due_and_writes_no_forecasts(tmp_path, synthetic_games):
    off_window_as_of = pd.Timestamp("2026-08-24T12:00:00Z")  # a Monday, not due for either horizon
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=off_window_as_of, force=False, operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "NOT_DUE"
    assert manifest["forecast_count"] == 0


def test_manual_horizon_refuses_future_cutoff(tmp_path, synthetic_games):
    # This week's TUE cutoff has not occurred yet relative to as_of -- must
    # never fabricate a future market snapshot, even under force=True.
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=pd.Timestamp("2026-08-24T12:00:00Z"), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "NOT_DUE"
    assert "fabricate" in manifest["detail"]
    assert manifest["forecast_count"] == 0


# ===========================================================================
# TUE / FRI batch mechanics: immutable write, evaluation-ledger schema
# (Sections 12, 14, 18) -- hermetic, synthetic games.
# ===========================================================================
def test_tue_batch_succeeds_and_writes_forecasts(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["horizon"] == "TUE"
    assert manifest["target_cutoff_utc"] == str(cutoffs["TUE"])
    assert manifest["game_count"] > 0
    # This synthetic population's last card crosses min_training_games=48,
    # so its games are real OOF-ready, not abstentions.
    assert manifest["forecast_count"] == manifest["game_count"]
    assert manifest["abstention_count"] == 0

    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    written = list(ledger_dir.glob("*.json"))
    assert len(written) == manifest["game_count"]
    sample = json.loads(written[0].read_text())
    assert sample["horizon"] == "TUE"
    assert sample["prediction"]["certified_baseline_sha"] == prod.CERTIFIED_SHA
    assert sample["prediction"]["operational_model_spec_hash"] == manifest["input_hashes"]["operational_model_spec_hash"]


def test_fri_batch_succeeds_independently_of_tue(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="FRI", as_of_utc=cutoffs["FRI"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["horizon"] == "FRI"
    assert manifest["target_cutoff_utc"] == str(cutoffs["FRI"])
    assert manifest["game_count"] > 0
    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "FRI"
    assert len(list(ledger_dir.glob("*.json"))) == manifest["game_count"]


def test_evaluation_ledger_has_no_outcome_columns(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    eval_dir = tmp_path / "production-2026" / "evaluation-ledger" / "TUE"
    written = list(eval_dir.glob("*.json"))
    assert written
    for path in written:
        record = json.loads(path.read_text())
        assert prod._FORBIDDEN_EVALUATION_FIELDS.isdisjoint(record.keys())
        assert "provenance" in record and "forecast" in record and "markets" in record


# ===========================================================================
# Idempotent rerun + conflicting-payload hard stop (Sections 14, 24 items
# 7-8) -- hermetic, synthetic games.
# ===========================================================================
def test_idempotent_rerun_is_a_noop_with_identical_hash(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    as_of = cutoffs["TUE"] + pd.Timedelta(minutes=1)
    first = prod.run_horizon_batch(horizon="TUE", as_of_utc=as_of, force=True, operational_root=tmp_path, games=synthetic_games)
    rerun = prod.run_horizon_batch(horizon="TUE", as_of_utc=as_of, force=True, operational_root=tmp_path, games=synthetic_games)
    assert rerun["status"] == "SUCCESS"
    assert rerun["output_hashes"] == first["output_hashes"]
    # Two distinct manifests were still appended (append-only run manifest,
    # never edited/overwritten) even though no forecast content changed.
    assert rerun["run_id"] != first["run_id"]
    manifest_dir = tmp_path / "production-2026" / "run-manifests"
    assert (manifest_dir / f"{first['run_id']}.json").exists()
    assert (manifest_dir / f"{rerun['run_id']}.json").exists()


def test_conflicting_payload_is_a_hard_stop_not_an_overwrite(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    existing_path = next(ledger_dir.glob("*.json"))
    existing = json.loads(existing_path.read_text())
    before = existing_path.read_text()

    conflicting_record = {
        "game_id": existing["game_id"], "horizon": existing["horizon"], "target_cutoff_utc": existing["target_cutoff_utc"],
        "created_at_utc": prod.utc_now().isoformat(), "git_commit": existing["git_commit"], "run_id": "conflict-test",
        "certified_baseline_tag": prod.CERTIFIED_TAG, "certified_baseline_sha": prod.CERTIFIED_SHA,
        "prediction": {**existing["prediction"], "prediction": {"deliberately": "different-payload"}},
    }
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.write_forecast(tmp_path / "production-2026" / "forecast-ledger", conflicting_record)
    assert excinfo.value.status == "FORECAST_IMMUTABILITY_VIOLATION"
    # The original file on disk is byte-for-byte untouched.
    assert existing_path.read_text() == before


def test_run_id_collision_is_identifier_failure(tmp_path):
    manifest_root = tmp_path / "run-manifests"
    manifest = {"run_id": "dup-run-id", "status": "SUCCESS"}
    prod.write_run_manifest(manifest_root, manifest)
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.write_run_manifest(manifest_root, manifest)
    assert excinfo.value.status == "IDENTIFIER_FAILURE"


# ===========================================================================
# Fail-closed statuses that don't require a full pipeline success (Section
# 16) -- hermetic, synthetic games.
# ===========================================================================
def test_hash_mismatch_when_certified_sources_missing(tmp_path, synthetic_games):
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "outputs").mkdir(parents=True)
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path / "hash-check", repo_root=fake_repo, games=synthetic_games,
    )
    assert manifest["status"] == "HASH_MISMATCH"
    assert manifest["forecast_count"] == 0


def test_schedule_unavailable_when_no_card_at_cutoff(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    empty_games = synthetic_games.iloc[0:0]
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path / "schedule-check", games=empty_games,
    )
    assert manifest["status"] == "SCHEDULE_UNAVAILABLE"
    assert manifest["forecast_count"] == 0


def test_elo_source_unavailable_on_malformed_games(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    broken_games = synthetic_games.drop(columns=["scheduled_kickoff_utc"])
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path / "elo-check", games=broken_games,
    )
    assert manifest["status"] == "ELO_SOURCE_UNAVAILABLE"
    assert manifest["forecast_count"] == 0


def test_model_not_ready_when_oof_pipeline_raises(tmp_path, synthetic_games, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated OOF pipeline failure")

    monkeypatch.setattr(prod.ohf, "build_official_horizon_oof", _raise)
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path / "model-check", games=synthetic_games,
    )
    assert manifest["status"] == "MODEL_NOT_READY"
    assert manifest["forecast_count"] == 0


def test_market_source_unavailable_reported_per_game(tmp_path, synthetic_games, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated raw odds store outage")

    monkeypatch.setattr(prod.rmr, "load_raw_bookmaker_quotes", _raise)
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path / "market-outage-check", games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"  # point forecasts still produced -- markets degrade, never the whole batch
    assert manifest["market_ready_counts"] == {"ATS": 0, "TOTAL": 0}
    ledger_dir = tmp_path / "market-outage-check" / "production-2026" / "forecast-ledger" / "TUE"
    sample = json.loads(next(ledger_dir.glob("*.json")).read_text())
    assert sample["prediction"]["markets"]["ATS"]["status"] == "MARKET_SOURCE_UNAVAILABLE"
    assert sample["prediction"]["markets"]["TOTAL"]["status"] == "MARKET_SOURCE_UNAVAILABLE"


# ===========================================================================
# Market/calibration readiness -- direct unit test of price_and_calibrate
# (Section 12), the "OK"/CALIBRATED path. Hermetic: a hand-built residual
# ledger + market consensus + calibration seed matching the certified
# schemas exactly, exercising the REAL production function, not a mock.
# ===========================================================================
def test_price_and_calibrate_reports_ok_when_market_and_calibration_seed_present():
    residual_ledger = pd.DataFrame({
        "game_id": ["G1", "G2", "G3"], "season": [2023, 2023, 2023], "week": [10, 10, 10],
        "target_cutoff_utc": pd.to_datetime(["2023-11-07T17:00:00Z"] * 3, utc=True),
        "result_available_at_utc": pd.to_datetime(["2023-11-13T00:00:00Z"] * 3, utc=True),
        "model_config_hash": ["h"] * 3, "feature_state_hash": ["f"] * 3,
        "status": ["OOF"] * 3, "uncertainty_eligible": [True] * 3,
        "predicted_margin": [3.0, -2.0, 1.0], "predicted_total": [45.0, 42.0, 48.0],
        "actual_margin": [np.nan] * 3, "actual_total": [np.nan] * 3,
        "margin_residual_sd_oof": [7.0, 7.0, 7.0], "total_residual_sd_oof": [9.0, 9.0, 9.0],
    })

    def _consensus(lines):
        return pd.DataFrame({
            "game_id": ["G1", "G2", "G3"], "consensus_line": lines,
            "consensus_novig_probability": [0.5, 0.52, 0.48], "eligible_books": [5, 5, 5],
            "bookmaker_keys": [["a", "b"]] * 3, "selected_returned_snapshot_timestamps": [["2023-11-07T12:00:00Z"]] * 3,
            "min_observation_age_hours": [1.0, 1.0, 1.0], "max_observation_age_hours": [2.0, 2.0, 2.0],
            "consensus_method": ["median"] * 3,
        })

    market_consensus = {"ATS": _consensus([-3.5, 2.0, -1.0]), "TOTAL": _consensus([44.5, 41.0, 47.5])}
    calibration_seed = {
        "ATS_TUE": {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
                    "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
        "TOTAL_TUE": {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.05]], "intercept": [0.01]},
                      "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
    }
    priced = prod.price_and_calibrate(residual_ledger, horizon="TUE", market_consensus=market_consensus, calibration_seed=calibration_seed)
    assert set(priced.keys()) == {"ATS_TUE", "TOTAL_TUE"}
    for stream, frame in priced.items():
        assert (frame["calibration_status"] == "CALIBRATED").all()
        assert frame["raw_conditional_upper_probability"].between(0, 1).all()
        assert frame["calibrated_conditional_upper_probability"].between(0, 1).all()
        assert frame["calibrated_upper_probability"].between(0, 1).all()


def test_price_and_calibrate_omits_stream_when_no_market_consensus_supplied():
    residual_ledger = pd.DataFrame({
        "game_id": ["G1"], "season": [2023], "week": [10],
        "target_cutoff_utc": pd.to_datetime(["2023-11-07T17:00:00Z"], utc=True),
        "result_available_at_utc": pd.to_datetime(["2023-11-13T00:00:00Z"], utc=True),
        "model_config_hash": ["h"], "feature_state_hash": ["f"], "status": ["OOF"], "uncertainty_eligible": [True],
        "predicted_margin": [3.0], "predicted_total": [45.0], "actual_margin": [np.nan], "actual_total": [np.nan],
        "margin_residual_sd_oof": [7.0], "total_residual_sd_oof": [9.0],
    })
    priced = prod.price_and_calibrate(residual_ledger, horizon="TUE", market_consensus={}, calibration_seed={})
    assert priced == {}  # no market -> nothing priced, never a synthetic line


# ===========================================================================
# Result attachment cannot mutate a forecast (Section 19) -- hermetic;
# every test builds its OWN prerequisite evaluation-ledger record, never
# relying on another test's execution order.
# ===========================================================================
def _write_prerequisite_evaluation_record(evaluation_root: Path, *, game_id: str = "G1", horizon: str = "TUE") -> dict:
    record = {
        "game_id": game_id, "horizon": horizon, "target_cutoff_utc": "2023-11-07T17:00:00+00:00",
        "season": 2023, "forecast": {"predicted_margin": 3.0, "predicted_total": 45.0},
        "markets": {}, "market_state": {"snapshot_available": False, "error": None},
        "provenance": {"git_commit": "deadbeef", "run_id": "test-run"},
    }
    prod.write_evaluation_record(evaluation_root, record)
    return record


def test_result_attachment_never_mutates_forecast_or_evaluation_record(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    record = _write_prerequisite_evaluation_record(evaluation_root)
    eval_path = prod._identity_path(evaluation_root, record["game_id"], record["horizon"], record["target_cutoff_utc"])
    before_eval_text = eval_path.read_text()

    result = {"home_score": 20.0, "away_score": 17.0}
    result_source_hash = prod._sha256_hex({"game_id": record["game_id"], "result": result})
    result_path = prod.attach_result(
        evaluation_root, game_id=record["game_id"], horizon=record["horizon"], target_cutoff_utc=record["target_cutoff_utc"],
        result=result, result_available_at_utc=pd.Timestamp("2023-11-08T00:00:00Z"), attachment_run_time=prod.utc_now(),
        result_source_hash=result_source_hash,
    )
    assert result_path.exists()
    attached = json.loads(result_path.read_text())
    assert attached["result"] == result
    assert attached["result_source_hash"] == result_source_hash
    assert "result_attached_at_utc" in attached
    # Original evaluation record is byte-for-byte untouched.
    assert eval_path.read_text() == before_eval_text


def test_result_attachment_rejects_result_not_yet_available(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    record = _write_prerequisite_evaluation_record(evaluation_root)
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.attach_result(
            evaluation_root, game_id=record["game_id"], horizon=record["horizon"], target_cutoff_utc=record["target_cutoff_utc"],
            result={"home_score": 20.0, "away_score": 17.0}, result_available_at_utc=prod.utc_now(),
            attachment_run_time=prod.utc_now() - pd.Timedelta(seconds=1), result_source_hash="deadbeef",
        )
    assert excinfo.value.status == "IDENTIFIER_FAILURE"


def test_result_attachment_requires_forecast_present(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.attach_result(
            evaluation_root, game_id="NO_SUCH_GAME", horizon="TUE", target_cutoff_utc="2023-11-07T17:00:00+00:00",
            result={"home_score": 20.0, "away_score": 17.0}, result_available_at_utc=pd.Timestamp("2023-11-08T00:00:00Z"),
            attachment_run_time=prod.utc_now(), result_source_hash="deadbeef",
        )
    assert excinfo.value.status == "IDENTIFIER_FAILURE"


def test_result_attachment_idempotent_and_conflict(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    record = _write_prerequisite_evaluation_record(evaluation_root)
    kwargs = dict(
        game_id=record["game_id"], horizon=record["horizon"], target_cutoff_utc=record["target_cutoff_utc"],
        result_available_at_utc=pd.Timestamp("2023-11-08T00:00:00Z"), attachment_run_time=prod.utc_now(),
        result_source_hash="deadbeef",
    )
    p1 = prod.attach_result(evaluation_root, result={"home_score": 20.0, "away_score": 17.0}, **kwargs)
    p2 = prod.attach_result(evaluation_root, result={"home_score": 20.0, "away_score": 17.0}, **kwargs)
    assert p1 == p2
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.attach_result(evaluation_root, result={"home_score": 99.0, "away_score": 1.0}, **kwargs)
    assert excinfo.value.status == "FORECAST_IMMUTABILITY_VIOLATION"


# ===========================================================================
# Prospective performance reporter (Section 20) -- hermetic; builds its own
# synthetic evaluation-ledger + attached results directly, no real games or
# real market data needed.
# ===========================================================================
def _write_scored_record(evaluation_root: Path, *, game_id: str, horizon: str, margin: float, total: float,
                          home_score: float, away_score: float) -> None:
    record = {
        "game_id": game_id, "horizon": horizon, "target_cutoff_utc": "2023-11-07T17:00:00+00:00",
        "season": 2023, "forecast": {"predicted_margin": margin, "predicted_total": total},
        "markets": {}, "market_state": {"snapshot_available": False, "error": None},
        "provenance": {"git_commit": "deadbeef", "run_id": "test-run"},
    }
    prod.write_evaluation_record(evaluation_root, record)
    result = {"home_score": home_score, "away_score": away_score}
    prod.attach_result(
        evaluation_root, game_id=game_id, horizon=horizon, target_cutoff_utc=record["target_cutoff_utc"],
        result=result, result_available_at_utc=pd.Timestamp("2023-11-08T00:00:00Z"), attachment_run_time=prod.utc_now(),
        result_source_hash=prod._sha256_hex({"game_id": game_id, "result": result}),
    )


def test_reporter_insufficient_sample_on_empty_ledger(tmp_path):
    records = prod.load_prospective_records(tmp_path / "does-not-exist")
    report = prod.compute_prospective_performance(records)
    assert report["status"] == "INSUFFICIENT_PROSPECTIVE_SAMPLE"
    assert report["n"] == 0


def test_reporter_insufficient_sample_below_floor(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    for i in range(5):  # well below MIN_PROSPECTIVE_SAMPLE
        _write_scored_record(
            evaluation_root, game_id=f"G{i}", horizon="TUE", margin=float(i - 2), total=44.0,
            home_score=20.0 + i, away_score=17.0,
        )
    records = prod.load_prospective_records(evaluation_root)
    report = prod.compute_prospective_performance(records)
    assert report["status"] == "INSUFFICIENT_PROSPECTIVE_SAMPLE"
    assert report["n"] == 5


def test_reporter_metrics_computable_with_a_lowered_sample_floor(tmp_path):
    evaluation_root = tmp_path / "production-2026" / "evaluation-ledger"
    for i in range(10):
        _write_scored_record(
            evaluation_root, game_id=f"G{i}", horizon="TUE", margin=float(i % 7 - 3), total=45.0,
            home_score=20.0 + (i % 5), away_score=17.0,
        )
    records = prod.load_prospective_records(evaluation_root)
    assert len(records) == 10
    report = prod.compute_prospective_performance(records, min_sample=1)
    assert report["status"] == "OK"
    assert report["n"] == 10
    # No market snapshots were supplied, so every ATS/TOTAL stream stays
    # below its own per-stream floor -- margin/total RMSE for the model
    # itself is still descriptive, market-vs-model comparisons correctly
    # report insufficient sample rather than a fabricated market number.
    for stream, detail in report["streams"].items():
        assert detail["status"] in ("OK", "INSUFFICIENT_PROSPECTIVE_SAMPLE")


# ===========================================================================
# Frozen calibration seed application -- verified against three_way.py's own
# apply-path (_predict_conditional on a reconstructed LogisticRegression),
# never trusted from the thin sigmoid reimplementation alone. Hermetic: a
# synthetic seed_state dict, no file I/O, no real artifact root.
# ===========================================================================
def test_frozen_calibrator_matches_three_way_apply_path_synthetic():
    rng = np.random.default_rng(0)
    p_sample = rng.uniform(0.05, 0.95, size=25)
    synthetic_states = [
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]}},
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[-1.4]], "intercept": [0.31]}},
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.0]], "intercept": [0.0]}},
    ]
    for state in synthetic_states:
        cond_state = state["conditional_calibrator_state"]
        via_reimplementation = prod.apply_frozen_conditional_calibrator(p_sample, state)

        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array(cond_state["coefficient"], dtype=float)
        model.intercept_ = np.array(cond_state["intercept"], dtype=float)
        model.n_features_in_ = 1
        via_three_way_apply_path = _predict_conditional(model, p_sample, CalibrationConfig())

        assert np.allclose(via_reimplementation, via_three_way_apply_path, rtol=1e-9, atol=1e-9)


def test_frozen_calibrator_passthrough_when_not_fitted():
    p_sample = np.array([0.1, 0.5, 0.9])
    out = prod.apply_frozen_conditional_calibrator(p_sample, {"conditional_calibrator_state": {"fitted": False}})
    assert np.allclose(out, p_sample, atol=1e-6)


# ===========================================================================
# Real historical integration test (Section 22) -- extended_data, skipped
# cleanly when the private data estate is unavailable. Same repo convention
# as tests/test_chronological_oof_extended.py.
# ===========================================================================
def _try_resolve(key: str):
    try:
        return resolve(key)
    except ExternalDataUnavailableError:
        return None


_REAL_GAMES_PATH = _try_resolve("backfill.games")


@pytest.mark.extended_data
@pytest.mark.skipif(
    _REAL_GAMES_PATH is None, reason="NFL_MODEL_DATA_ROOT not configured or backfill.games missing (extended_data)",
)
class TestRealHistoricalIntegration:
    """Proves end-to-end, on a real already-elapsed TUE and FRI card from
    the certified 2020-2025 population (2024 season, week 3 -- chosen
    arbitrarily, no cherry-picking of outcomes since nothing here scores a
    strategy): preflight, a TUE run, a FRI run, the immutable
    forecast-ledger write, real market/calibration readiness, result
    attachment, and prospective reporting -- against the real certified
    pipeline and the real Fix-8 calibration seed, not synthetic stand-ins.

    Two full historical batches (TUE, FRI) each refit the full certified
    Ridge-per-cutoff-batch OOF machinery over the whole 2020-2025
    population (~131 batches x2 targets), so this class is not fast (order
    of a minute or two per batch) -- that cost is the certified pipeline's
    own, not something this test adds.
    """

    TUE_AS_OF = pd.Timestamp("2024-09-17T16:01:00Z")
    TUE_CUTOFF = pd.Timestamp("2024-09-17T16:00:00Z")
    FRI_AS_OF = pd.Timestamp("2024-09-20T16:01:00Z")
    FRI_CUTOFF = pd.Timestamp("2024-09-20T16:00:00Z")

    @pytest.fixture(scope="class")
    def games_population(self) -> pd.DataFrame:
        return pd.read_parquet(resolve("backfill.games"))

    @pytest.fixture(scope="class")
    def operational_root(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("production-2026-real-fixture")

    @pytest.fixture(scope="class")
    def tue_manifest(self, operational_root, games_population) -> dict:
        return prod.run_horizon_batch(
            horizon="TUE", as_of_utc=self.TUE_AS_OF, force=True, operational_root=operational_root, games=games_population,
        )

    @pytest.fixture(scope="class")
    def fri_manifest(self, operational_root, games_population) -> dict:
        return prod.run_horizon_batch(
            horizon="FRI", as_of_utc=self.FRI_AS_OF, force=True, operational_root=operational_root, games=games_population,
        )

    def test_preflight_reports_ready_waiting_or_ready(self):
        # Deliberately NO artifact_root_path override here -- this test
        # verifies genuine real-environment readiness (including the real
        # Fix-8 calibration seed), unlike the hermetic preflight tests
        # above which specifically exercise the artifact_root_path
        # override in isolation. run_preflight's own writable-directory
        # probes are harmless (a file is written then immediately
        # unlinked) and never touch the forecast/evaluation ledgers.
        result = prod.run_preflight()
        assert result["overall_status"] in ("READY", "READY_WAITING_FOR_FIRST_DUE_CUTOFF")
        assert result["infra_ready"] is True
        assert result["checks"]["certified_hashes"]["status"] == "MATCH"
        assert result["checks"]["fix8_calibration_seed"]["status"] == "OK"

    def test_real_calibration_seed_round_trips_through_production_loader(self):
        """Narrow, real-artifact-only assertion: the actual certified seed
        file on this machine has the schema apply_frozen_conditional_calibrator
        expects, and produces the same output as the independently
        reconstructed three_way apply-path -- byte-for-byte against the
        real certified evidence, not a synthetic stand-in."""
        seed_path = artifact_root() / "fix8-official-oof-calibration-2026" / "production_calibration_seed.json"
        if not seed_path.is_file():
            pytest.skip("real production_calibration_seed.json not present under NFL_MODEL_ARTIFACT_ROOT (extended_data)")
        seed = json.loads(seed_path.read_text())
        rng = np.random.default_rng(0)
        p_sample = rng.uniform(0.05, 0.95, size=25)
        for stream, state in seed.items():
            cond_state = state["conditional_calibrator_state"]
            if not cond_state.get("fitted"):
                continue
            via_reimplementation = prod.apply_frozen_conditional_calibrator(p_sample, state)
            model = LogisticRegression()
            model.classes_ = np.array([0, 1])
            model.coef_ = np.array(cond_state["coefficient"], dtype=float)
            model.intercept_ = np.array(cond_state["intercept"], dtype=float)
            model.n_features_in_ = 1
            via_three_way_apply_path = _predict_conditional(model, p_sample, CalibrationConfig())
            assert np.allclose(via_reimplementation, via_three_way_apply_path, rtol=1e-9, atol=1e-9), stream

    def test_tue_batch_succeeds_and_writes_forecasts(self, tue_manifest, operational_root):
        assert tue_manifest["status"] == "SUCCESS"
        assert tue_manifest["horizon"] == "TUE"
        assert tue_manifest["target_cutoff_utc"] == str(self.TUE_CUTOFF)
        assert tue_manifest["game_count"] > 0
        assert tue_manifest["forecast_count"] == tue_manifest["game_count"]
        assert tue_manifest["abstention_count"] == 0
        # Real historical bookmaker history should cover most, not
        # necessarily all, of a 2024 week-3 card.
        assert tue_manifest["market_ready_counts"]["ATS"] > 0
        assert tue_manifest["calibration_ready_counts"]["ATS"] > 0

        ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
        written = list(ledger_dir.glob("*.json"))
        assert len(written) == tue_manifest["game_count"]
        sample = json.loads(written[0].read_text())
        assert sample["horizon"] == "TUE"
        assert sample["prediction"]["certified_baseline_sha"] == prod.CERTIFIED_SHA

    def test_fri_batch_succeeds_independently_of_tue(self, fri_manifest, operational_root):
        assert fri_manifest["status"] == "SUCCESS"
        assert fri_manifest["horizon"] == "FRI"
        assert fri_manifest["target_cutoff_utc"] == str(self.FRI_CUTOFF)
        assert fri_manifest["game_count"] > 0
        ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "FRI"
        assert len(list(ledger_dir.glob("*.json"))) == fri_manifest["game_count"]

    def test_per_game_records_carry_explicit_market_readiness_statuses(self, tue_manifest, operational_root):
        ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
        statuses_seen = set()
        for path in ledger_dir.glob("*.json"):
            record = json.loads(path.read_text())
            for market in ("ATS", "TOTAL"):
                statuses_seen.add(record["prediction"]["markets"][market]["status"])
        assert "OK" in statuses_seen
        assert statuses_seen <= {"OK", "MARKET_NOT_READY", "UNCERTAINTY_NOT_READY", "CALIBRATION_NOT_READY", "MARKET_SOURCE_UNAVAILABLE"}

    def _attach_all_real_results(self, operational_root: Path, horizon: str, games_population: pd.DataFrame) -> None:
        evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
        eval_dir = evaluation_root / horizon
        for path in eval_dir.glob("*.json"):
            record = json.loads(path.read_text())
            game_id = record["game_id"]
            game_row = games_population[games_population["game_id"].astype(str) == game_id].iloc[0]
            if pd.isna(game_row["home_score"]) or pd.isna(game_row["away_score"]):
                continue
            result_available_at_utc = prod._as_utc(game_row["scheduled_kickoff_utc"]) + pd.Timedelta(hours=5)
            result = {"home_score": float(game_row["home_score"]), "away_score": float(game_row["away_score"])}
            prod.attach_result(
                evaluation_root, game_id=game_id, horizon=horizon, target_cutoff_utc=record["target_cutoff_utc"],
                result=result, result_available_at_utc=result_available_at_utc, attachment_run_time=prod.utc_now(),
                result_source_hash=prod._sha256_hex({"game_id": game_id, "result": result}),
            )

    def test_reporter_on_one_small_historical_card_never_overstates_sample_maturity(
        self, tue_manifest, fri_manifest, operational_root, games_population,
    ):
        self._attach_all_real_results(operational_root, "TUE", games_population)
        self._attach_all_real_results(operational_root, "FRI", games_population)
        evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
        records = prod.load_prospective_records(evaluation_root)
        assert len(records) > 0
        report = prod.compute_prospective_performance(records)
        assert report["status"] in ("OK", "INSUFFICIENT_PROSPECTIVE_SAMPLE")
        if report["status"] == "OK":
            for stream, detail in report["streams"].items():
                assert detail["n"] < 30, f"{stream} n={detail['n']} implausibly large for a single historical card"
