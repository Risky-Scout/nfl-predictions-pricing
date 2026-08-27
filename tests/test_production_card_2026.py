"""Section 22: historical integration test for the 2026 production pipeline.

Proves end-to-end, on a real already-elapsed TUE and FRI card from the
certified 2020-2025 population (2024 season, week 3 -- chosen arbitrarily,
no cherry-picking of outcomes since nothing here scores a strategy):
preflight, a TUE run, a FRI run, the immutable forecast-ledger write, an
idempotent rerun, a conflicting-payload hard stop, market/calibration
readiness, result attachment, and prospective reporting.

Every write in this file goes through ``operational_root``/
``artifact_root_path`` overrides pointing at a pytest ``tmp_path`` --
this test NEVER writes into the real
``$NFL_MODEL_ARTIFACT_ROOT/production-2026/`` ledger. The certified hash
sources, the frozen calibration seed, the real historical games population,
and the raw market estate are read from their real (certified, read-only)
locations, exactly as production does.

Two full historical batches (TUE, FRI) each refit the full certified
Ridge-per-cutoff-batch OOF machinery over the whole 2020-2025 population
(~131 batches x2 targets), so this module is not fast (order of a minute or
two) -- that cost is the certified pipeline's own, not something this test
adds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from nfl_hybrid.calibration.three_way import CalibrationConfig, _predict_conditional
from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.production import run_2026 as prod

# A real, already-elapsed 2024 week-3 card. TUE/FRI cutoffs independently
# confirmed against he.build_horizon_membership_ledger before writing this
# test (see the task's own preparatory exploration) -- not asserted again
# here, since the pipeline itself recomputes them from the certified
# membership-ledger algorithm on every run.
TUE_AS_OF = pd.Timestamp("2024-09-17T16:01:00Z")
TUE_CUTOFF = pd.Timestamp("2024-09-17T16:00:00Z")
FRI_AS_OF = pd.Timestamp("2024-09-20T16:01:00Z")
FRI_CUTOFF = pd.Timestamp("2024-09-20T16:00:00Z")


@pytest.fixture(scope="module")
def games_population() -> pd.DataFrame:
    return pd.read_parquet(resolve("backfill.games"))


@pytest.fixture(scope="module")
def operational_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("production-2026-fixture")


@pytest.fixture(scope="module")
def tue_manifest(operational_root, games_population) -> dict:
    return prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root, games=games_population,
    )


@pytest.fixture(scope="module")
def fri_manifest(operational_root, games_population) -> dict:
    return prod.run_horizon_batch(
        horizon="FRI", as_of_utc=FRI_AS_OF, force=True, operational_root=operational_root, games=games_population,
    )


# ===========================================================================
# Preflight (against the real certified inputs; only the writable-directory
# probe is pointed at an isolated tmp root).
# ===========================================================================
def test_preflight_reports_ready_waiting_or_ready(tmp_path):
    result = prod.run_preflight(artifact_root_path=tmp_path)
    assert result["overall_status"] in ("READY", "READY_WAITING_FOR_FIRST_DUE_CUTOFF")
    assert result["infra_ready"] is True
    assert result["checks"]["certified_hashes"]["status"] == "MATCH"
    assert result["checks"]["fix8_calibration_seed"]["status"] == "OK"
    # No secret VALUES anywhere in the preflight output -- only SET/UNSET.
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
# DST-aware TUE/FRI due-run logic (Section 13).
# ===========================================================================
def test_due_window_true_at_exact_cutoff_and_false_just_before_and_after():
    at_cutoff = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:00:00Z"), "TUE")  # a real Tuesday, EST (UTC-5)
    assert at_cutoff["due"] is True
    just_before = prod.is_within_due_window(pd.Timestamp("2026-01-06T16:59:59Z"), "TUE")
    assert just_before["due"] is False
    just_after_window = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:20:00Z"), "TUE")
    assert just_after_window["due"] is False


def test_due_window_is_dst_aware_across_edt_and_est():
    # 2026-08-25 is a real Tuesday during EDT (UTC-4): noon local = 16:00 UTC.
    edt_due = prod.is_within_due_window(pd.Timestamp("2026-08-25T16:00:00Z"), "TUE")
    assert edt_due["due"] is True
    # 2026-01-06 is a real Tuesday during EST (UTC-5): noon local = 17:00 UTC.
    est_due = prod.is_within_due_window(pd.Timestamp("2026-01-06T17:00:00Z"), "TUE")
    assert est_due["due"] is True
    # The EDT offset does NOT apply in January -- 16:00 UTC in January is 11:00 EST, before the window.
    assert prod.is_within_due_window(pd.Timestamp("2026-01-06T16:00:00Z"), "TUE")["due"] is False


def test_due_window_wrong_weekday_never_due():
    monday = prod.is_within_due_window(pd.Timestamp("2026-08-24T16:00:00Z"), "TUE")
    assert monday["due"] is False


def test_run_due_outside_window_is_not_due_and_writes_no_forecasts(operational_root, games_population):
    off_window_as_of = pd.Timestamp("2026-08-24T12:00:00Z")  # a Monday, real "now"-like value
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=off_window_as_of, force=False, operational_root=operational_root,
        games=games_population,
    )
    assert manifest["status"] == "NOT_DUE"
    assert manifest["forecast_count"] == 0


def test_manual_horizon_refuses_future_cutoff(operational_root, games_population):
    # Monday before this week's TUE cutoff -- must never fabricate a future snapshot.
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=pd.Timestamp("2026-08-24T12:00:00Z"), force=True,
        operational_root=operational_root, games=games_population,
    )
    assert manifest["status"] == "NOT_DUE"
    assert manifest["forecast_count"] == 0


# ===========================================================================
# TUE / FRI historical batches: immutable write, market/calibration
# readiness (Sections 12, 14, 16).
# ===========================================================================
def test_tue_batch_succeeds_and_writes_forecasts(tue_manifest, operational_root):
    assert tue_manifest["status"] == "SUCCESS"
    assert tue_manifest["horizon"] == "TUE"
    assert tue_manifest["target_cutoff_utc"] == str(TUE_CUTOFF)
    assert tue_manifest["game_count"] > 0
    assert tue_manifest["forecast_count"] == tue_manifest["game_count"]
    assert tue_manifest["abstention_count"] == 0
    # Real historical bookmaker history should cover most, not necessarily
    # all, of a 2024 week-3 card.
    assert tue_manifest["market_ready_counts"]["ATS"] > 0
    assert tue_manifest["calibration_ready_counts"]["ATS"] > 0

    ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
    written = list(ledger_dir.glob("*.json"))
    assert len(written) == tue_manifest["game_count"]
    sample = json.loads(written[0].read_text())
    assert sample["horizon"] == "TUE"
    assert sample["prediction"]["certified_baseline_sha"] == prod.CERTIFIED_SHA
    assert sample["prediction"]["operational_model_spec_hash"] == tue_manifest["input_hashes"]["operational_model_spec_hash"]


def test_fri_batch_succeeds_independently_of_tue(fri_manifest, operational_root):
    assert fri_manifest["status"] == "SUCCESS"
    assert fri_manifest["horizon"] == "FRI"
    assert fri_manifest["target_cutoff_utc"] == str(FRI_CUTOFF)
    assert fri_manifest["game_count"] > 0
    ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "FRI"
    assert len(list(ledger_dir.glob("*.json"))) == fri_manifest["game_count"]


def test_per_game_records_carry_explicit_market_readiness_statuses(tue_manifest, operational_root):
    """Every eligible game gets a persisted forecast, including ones where
    the market/calibration layer is not ready -- never silently dropped,
    never silently substituted."""
    ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
    statuses_seen = set()
    for path in ledger_dir.glob("*.json"):
        record = json.loads(path.read_text())
        for market in ("ATS", "TOTAL"):
            statuses_seen.add(record["prediction"]["markets"][market]["status"])
    # At least one fully-ready stream (this card has real historical market
    # coverage) -- the exact mix of ready/not-ready games is descriptive,
    # not asserted row-by-row (no cherry-picking of which games "should"
    # have coverage).
    assert "OK" in statuses_seen
    assert statuses_seen <= {"OK", "MARKET_NOT_READY", "UNCERTAINTY_NOT_READY", "CALIBRATION_NOT_READY", "MARKET_SOURCE_UNAVAILABLE"}


def test_evaluation_ledger_has_no_outcome_columns(tue_manifest, operational_root):
    eval_dir = operational_root / "production-2026" / "evaluation-ledger" / "TUE"
    for path in eval_dir.glob("*.json"):
        record = json.loads(path.read_text())
        assert prod._FORBIDDEN_EVALUATION_FIELDS.isdisjoint(record.keys())
        assert "provenance" in record and "forecast" in record and "markets" in record


# ===========================================================================
# Idempotent rerun + conflicting-payload hard stop (Sections 14, 24 items
# 7-8).
# ===========================================================================
def test_idempotent_rerun_is_a_noop_with_identical_hash(tue_manifest, operational_root, games_population):
    rerun_manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root, games=games_population,
    )
    assert rerun_manifest["status"] == "SUCCESS"
    assert rerun_manifest["output_hashes"] == tue_manifest["output_hashes"]
    # Two distinct manifests were still appended (append-only run manifest,
    # never edited/overwritten) even though no forecast content changed.
    assert rerun_manifest["run_id"] != tue_manifest["run_id"]
    manifest_dir = operational_root / "production-2026" / "run-manifests"
    assert (manifest_dir / f"{tue_manifest['run_id']}.json").exists()
    assert (manifest_dir / f"{rerun_manifest['run_id']}.json").exists()


def test_conflicting_payload_is_a_hard_stop_not_an_overwrite(tue_manifest, operational_root):
    ledger_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
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
        prod.write_forecast(operational_root / "production-2026" / "forecast-ledger", conflicting_record)
    assert excinfo.value.status == "FORECAST_IMMUTABILITY_VIOLATION"
    # The original file on disk is byte-for-byte untouched.
    assert existing_path.read_text() == before


def test_run_id_collision_is_identifier_failure(operational_root):
    manifest_root = operational_root / "production-2026" / "run-manifests" / "collision-check"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": "dup-run-id", "status": "SUCCESS"}
    prod.write_run_manifest(manifest_root, manifest)
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.write_run_manifest(manifest_root, manifest)
    assert excinfo.value.status == "IDENTIFIER_FAILURE"


# ===========================================================================
# Fail-closed statuses that don't require a full expensive pipeline run
# (Section 16).
# ===========================================================================
def test_hash_mismatch_when_certified_sources_missing(tmp_path, operational_root, games_population):
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "outputs").mkdir(parents=True)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root / "hash-check",
        repo_root=fake_repo, games=games_population,
    )
    assert manifest["status"] == "HASH_MISMATCH"
    assert manifest["forecast_count"] == 0


def test_schedule_unavailable_when_no_card_at_cutoff(operational_root, games_population):
    empty_games = games_population.iloc[0:0]
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root / "schedule-check",
        games=empty_games,
    )
    assert manifest["status"] == "SCHEDULE_UNAVAILABLE"
    assert manifest["forecast_count"] == 0


def test_elo_source_unavailable_on_malformed_games(operational_root, games_population):
    broken_games = games_population.drop(columns=["scheduled_kickoff_utc"])
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root / "elo-check",
        games=broken_games,
    )
    assert manifest["status"] == "ELO_SOURCE_UNAVAILABLE"
    assert manifest["forecast_count"] == 0


def test_model_not_ready_when_oof_pipeline_raises(operational_root, games_population, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated OOF pipeline failure")

    monkeypatch.setattr(prod.ohf, "build_official_horizon_oof", _raise)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root / "model-check",
        games=games_population,
    )
    assert manifest["status"] == "MODEL_NOT_READY"
    assert manifest["forecast_count"] == 0


def test_market_source_unavailable_reported_per_game(operational_root, games_population, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated raw odds store outage")

    monkeypatch.setattr(prod.rmr, "load_raw_bookmaker_quotes", _raise)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=TUE_AS_OF, force=True, operational_root=operational_root / "market-outage-check",
        games=games_population,
    )
    assert manifest["status"] == "SUCCESS"  # point forecasts still produced -- markets degrade, never the whole batch
    assert manifest["market_ready_counts"] == {"ATS": 0, "TOTAL": 0}
    ledger_dir = operational_root / "market-outage-check" / "production-2026" / "forecast-ledger" / "TUE"
    sample = json.loads(next(ledger_dir.glob("*.json")).read_text())
    assert sample["prediction"]["markets"]["ATS"]["status"] == "MARKET_SOURCE_UNAVAILABLE"
    assert sample["prediction"]["markets"]["TOTAL"]["status"] == "MARKET_SOURCE_UNAVAILABLE"


# ===========================================================================
# Result attachment cannot mutate a forecast (Section 19).
# ===========================================================================
def test_result_attachment_never_mutates_forecast_or_evaluation_record(tue_manifest, operational_root, games_population):
    evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
    eval_dir = evaluation_root / "TUE"
    forecast_dir = operational_root / "production-2026" / "forecast-ledger" / "TUE"
    sample_eval_path = next(eval_dir.glob("*.json"))
    sample_eval = json.loads(sample_eval_path.read_text())
    game_id = sample_eval["game_id"]
    before_eval_text = sample_eval_path.read_text()
    forecast_path = next(p for p in forecast_dir.glob("*.json") if json.loads(p.read_text())["game_id"] == game_id)
    before_forecast_text = forecast_path.read_text()

    game_row = games_population[games_population["game_id"].astype(str) == game_id].iloc[0]
    result_available_at_utc = prod._as_utc(game_row["scheduled_kickoff_utc"]) + pd.Timedelta(hours=5)
    result = {"home_score": float(game_row["home_score"]), "away_score": float(game_row["away_score"])}
    result_source_hash = prod._sha256_hex({"game_id": game_id, "result": result})

    result_path = prod.attach_result(
        evaluation_root, game_id=game_id, horizon="TUE", target_cutoff_utc=sample_eval["target_cutoff_utc"],
        result=result, result_available_at_utc=result_available_at_utc, attachment_run_time=prod.utc_now(),
        result_source_hash=result_source_hash,
    )
    assert result_path.exists()
    attached = json.loads(result_path.read_text())
    assert attached["result"] == result
    assert attached["result_source_hash"] == result_source_hash
    assert "result_attached_at_utc" in attached

    # Original forecast/evaluation files are byte-for-byte untouched.
    assert sample_eval_path.read_text() == before_eval_text
    assert forecast_path.read_text() == before_forecast_text


def test_result_attachment_rejects_result_not_yet_available(operational_root):
    evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
    eval_dir = evaluation_root / "TUE"
    sample_eval = json.loads(next(eval_dir.glob("*.json")).read_text())
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.attach_result(
            evaluation_root, game_id=sample_eval["game_id"], horizon="TUE",
            target_cutoff_utc=sample_eval["target_cutoff_utc"], result={"home_score": 20.0, "away_score": 17.0},
            result_available_at_utc=prod.utc_now(), attachment_run_time=prod.utc_now() - pd.Timedelta(seconds=1),
            result_source_hash="deadbeef",
        )
    assert excinfo.value.status == "IDENTIFIER_FAILURE"


def _attach_all_real_results(operational_root: Path, horizon: str, games_population: pd.DataFrame) -> None:
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


# ===========================================================================
# Prospective performance reporter (Section 20).
# ===========================================================================
def test_reporter_insufficient_sample_on_empty_ledger(tmp_path):
    records = prod.load_prospective_records(tmp_path / "does-not-exist")
    report = prod.compute_prospective_performance(records)
    assert report["status"] == "INSUFFICIENT_PROSPECTIVE_SAMPLE"
    assert report["n"] == 0


def test_reporter_on_one_small_historical_card_never_overstates_sample_maturity(
    tue_manifest, fri_manifest, operational_root, games_population,
):
    """A single TUE+FRI card's ~30 attached records happens to clear the
    module's default overall-sample floor (MIN_PROSPECTIVE_SAMPLE=20 rows
    with a result attached), so the top-level ``status`` can legitimately
    read ``OK``. What must still be reported honestly is that no individual
    ATS/TOTAL stream has anywhere near a mature sample from one week alone
    -- every per-stream ``n`` here is a small fraction of a real production
    floor (dozens of weeks), which is exactly why the reporter tracks
    per-stream sample sizes separately rather than only a pooled total."""
    _attach_all_real_results(operational_root, "TUE", games_population)
    _attach_all_real_results(operational_root, "FRI", games_population)
    evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
    records = prod.load_prospective_records(evaluation_root)
    assert len(records) > 0
    report = prod.compute_prospective_performance(records)
    assert report["status"] in ("OK", "INSUFFICIENT_PROSPECTIVE_SAMPLE")
    if report["status"] == "OK":
        for stream, detail in report["streams"].items():
            assert detail["n"] < 30, f"{stream} n={detail['n']} implausibly large for a single historical card"


def test_reporter_metrics_computable_with_a_lowered_sample_floor(tue_manifest, fri_manifest, operational_root, games_population):
    """Same fixture, with an explicit (test-only) lower min_sample so the
    metric-computation code path itself is exercised and its output shape
    verified -- the production default floor is untouched. Attaches results
    itself (idempotently) rather than relying on another test's ordering."""
    _attach_all_real_results(operational_root, "TUE", games_population)
    _attach_all_real_results(operational_root, "FRI", games_population)
    evaluation_root = operational_root / "production-2026" / "evaluation-ledger"
    records = prod.load_prospective_records(evaluation_root)
    report = prod.compute_prospective_performance(records, min_sample=1)
    assert report["status"] == "OK"
    for stream, detail in report["streams"].items():
        assert detail["status"] in ("OK", "INSUFFICIENT_PROSPECTIVE_SAMPLE")
        if detail["status"] == "OK":
            assert detail["n"] > 0
            assert "model_log_loss" in detail and "market_no_vig_log_loss" in detail
            assert "log_loss_paired_delta" in detail and "brier_paired_delta" in detail
            assert np.isfinite(detail["model_log_loss"])
            assert np.isfinite(detail["market_no_vig_log_loss"])


# ===========================================================================
# Frozen calibration seed application -- verified against three_way.py's own
# apply-path (_predict_conditional on a reconstructed LogisticRegression),
# never trusted from the thin sigmoid reimplementation alone (per the
# certification GROUNDING's verification requirement).
# ===========================================================================
def test_frozen_calibrator_matches_three_way_apply_path():
    seed_path = artifact_root() / "fix8-official-oof-calibration-2026" / "production_calibration_seed.json"
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
