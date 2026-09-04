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
# secret-leak checks need no real data at all. The overall READY /
# BLOCKED_ON_LIVE_INPUTS verdict against the real environment (real
# backfill.games -- schedule_source is never overridable, it is a read-only
# live data feed) is covered in TestRealHistoricalIntegration below; the
# pure readiness decision itself (all four required-input combinations) is
# covered hermetically by TestPreflightReadinessSemantics.
# ===========================================================================


class TestPreflightReadinessSemantics:
    """`summarize_preflight_readiness` -- infrastructure readiness is
    reported separately from live-input readiness, and `production_run_ready`
    is true ONLY when infra is ready AND every required live 2026 input
    (schedule, registered market source) is actually available. No
    `READY_WAITING_FOR_FIRST_DUE_CUTOFF` status is ever emitted, and a
    blocked verdict always names its blockers."""

    def test_both_live_inputs_available_and_infra_ready_is_production_run_ready(self):
        r = prod.summarize_preflight_readiness(
            infra_blocking=[], schedule_2026_available=True, live_2026_market_source_registered=True,
        )
        assert r["production_run_ready"] is True
        assert r["overall_status"] == "READY"
        assert r["blocking_problems"] == []

    def test_schedule_2026_unavailable_blocks_production_run_ready(self):
        r = prod.summarize_preflight_readiness(
            infra_blocking=[], schedule_2026_available=False, live_2026_market_source_registered=True,
        )
        assert r["production_run_ready"] is False
        assert r["overall_status"] == "BLOCKED_ON_LIVE_INPUTS"
        assert "schedule_2026_unavailable" in r["blocking_problems"]

    def test_live_market_source_unregistered_blocks_production_run_ready(self):
        r = prod.summarize_preflight_readiness(
            infra_blocking=[], schedule_2026_available=True, live_2026_market_source_registered=False,
        )
        assert r["production_run_ready"] is False
        assert r["overall_status"] == "BLOCKED_ON_LIVE_INPUTS"
        assert "live_2026_market_source_unregistered" in r["blocking_problems"]

    def test_infra_blocker_forces_not_ready_regardless_of_live_inputs(self):
        r = prod.summarize_preflight_readiness(
            infra_blocking=["calibration_seed_missing"],
            schedule_2026_available=True, live_2026_market_source_registered=True,
        )
        assert r["infra_ready"] is False
        assert r["production_run_ready"] is False
        assert r["overall_status"] == "NOT_READY"
        assert "calibration_seed_missing" in r["blocking_problems"]

    def test_no_misleading_ready_waiting_status_ever_emitted(self):
        for sched in (True, False):
            for mkt in (True, False):
                for infra in ([], ["x"]):
                    r = prod.summarize_preflight_readiness(
                        infra_blocking=infra, schedule_2026_available=sched,
                        live_2026_market_source_registered=mkt,
                    )
                    assert r["overall_status"] in ("READY", "BLOCKED_ON_LIVE_INPUTS", "NOT_READY")
                    assert r["overall_status"] != "READY_WAITING_FOR_FIRST_DUE_CUTOFF"
                    if r["overall_status"] != "READY":
                        assert r["production_run_ready"] is False
                        assert r["blocking_problems"], "a non-READY verdict must name its blockers"

    def test_run_preflight_hermetic_never_claims_production_ready_without_live_market(self, tmp_path):
        # live_2026_market_source_registered is structurally False (no
        # odds_history.2026 key exists), so a hermetic run can never be
        # production_run_ready no matter what the schedule check returns.
        _write_synthetic_calibration_seed(tmp_path)
        result = prod.run_preflight(artifact_root_path=tmp_path)
        assert result["live_2026_market_source_registered"] is False
        assert result["production_run_ready"] is False
        assert result["overall_status"] in ("BLOCKED_ON_LIVE_INPUTS", "NOT_READY")
        assert "live_2026_market_source_unregistered" in result["blocking_problems"]

    def test_no_live_2026_market_key_registered_in_rmr(self):
        from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
        assert not any("2026" in key for key in rmr.RAW_ODDS_HISTORY_KEYS)
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


def test_preseason_games_are_excluded_from_the_production_card(tmp_path, synthetic_games):
    """PRESEASON never enters the card. A live 2026 schedule source (BDL /
    nflverse) exposes preseason games; feeding them to run_horizon_batch
    must produce ZERO PRESEASON forecasts and leave the REG card identical
    to a preseason-free run."""
    cutoffs = _last_card_cutoffs(synthetic_games)
    as_of = cutoffs["TUE"] + pd.Timedelta(minutes=1)

    pre = synthetic_games.copy()
    pre["season_type"] = "PRE"
    pre["game_id"] = pre["game_id"].astype(str) + "_PRE"
    mixed = pd.concat([synthetic_games, pre], ignore_index=True)

    reg_only = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=as_of, force=True,
        operational_root=tmp_path / "reg", games=synthetic_games,
    )
    with_pre = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=as_of, force=True,
        operational_root=tmp_path / "mixed", games=mixed,
    )
    assert with_pre["status"] == "SUCCESS"
    assert with_pre["game_count"] == reg_only["game_count"]

    for sub in ("forecast-ledger", "evaluation-ledger"):
        for path in (tmp_path / "mixed" / "production-2026" / sub / "TUE").glob("*.json"):
            rec = json.loads(path.read_text())
            payload = rec.get("prediction", rec)
            assert payload.get("season_type") in ("REG", "POST")
            assert not str(rec["game_id"]).endswith("_PRE")


def test_live_schedule_missing_season_type_fails_closed(tmp_path, synthetic_games):
    """A schedule source that cannot prove it excludes preseason (no
    season_type column) is a fail-closed SCHEDULE_UNAVAILABLE, never a
    silent 'assume REG' pass."""
    cutoffs = _last_card_cutoffs(synthetic_games)
    no_type = synthetic_games.drop(columns=["season_type"])
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=no_type,
    )
    assert manifest["status"] == "SCHEDULE_UNAVAILABLE"


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
# Public forecast-of-record identity metadata hardening (additive plumbing
# only: home_team_id / away_team_id / scheduled_kickoff_utc / one shared
# run_created_at_utc). No model feature, fit, calibration, or prediction-path
# change -- see PRODUCTION FORECAST PUBLIC IDENTITY METADATA HARDENING.
# ===========================================================================
def _identity_games(rows: list[dict]) -> pd.DataFrame:
    """Minimal canonical-games-population rows for resolve_forecast_identity
    unit tests -- deliberately opaque game_id values (no team codes embedded)
    so a passing test can only mean the identity fields were read from the
    matching row, never parsed out of game_id."""
    return pd.DataFrame(rows)


def test_resolve_forecast_identity_persists_home_team_id_from_canonical_population():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    identity = prod.resolve_forecast_identity(games, "G1")
    assert identity["home_team_id"] == "KC"


def test_resolve_forecast_identity_persists_away_team_id_from_canonical_population():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    identity = prod.resolve_forecast_identity(games, "G1")
    assert identity["away_team_id"] == "BUF"


def test_resolve_forecast_identity_persists_scheduled_kickoff_utc_from_canonical_population():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    identity = prod.resolve_forecast_identity(games, "G1")
    assert identity["scheduled_kickoff_utc"] == "2026-09-13T17:00:00Z"


def test_resolve_forecast_identity_never_parses_game_id():
    # game_id carries NO team codes or kickoff information at all -- the
    # correct home/away/kickoff values can only have come from the row.
    games = _identity_games([{"game_id": "opaque-id-000001", "home_team_id": "ZZZ", "away_team_id": "YYY",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-11-01T18:00:00Z")}])
    identity = prod.resolve_forecast_identity(games, "opaque-id-000001")
    assert identity == {
        "home_team_id": "ZZZ", "away_team_id": "YYY", "scheduled_kickoff_utc": "2026-11-01T18:00:00Z",
    }
    assert "resolve_forecast_identity"  # sanity: no game_id string-parsing helper is invoked
    import inspect
    src = inspect.getsource(prod.resolve_forecast_identity)
    assert "game_id.split" not in src and "game_id[" not in src


@pytest.mark.parametrize("bad_home", [None, np.nan, "", "   "])
def test_resolve_forecast_identity_missing_home_team_fails_closed(bad_home):
    games = _identity_games([{"game_id": "G1", "home_team_id": bad_home, "away_team_id": "BUF",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"


@pytest.mark.parametrize("bad_away", [None, np.nan, "", "  "])
def test_resolve_forecast_identity_missing_away_team_fails_closed(bad_away):
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": bad_away,
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"


def test_resolve_forecast_identity_identical_home_and_away_fails_closed():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "KC",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"
    assert "equals" in excinfo.value.detail


def test_resolve_forecast_identity_missing_kickoff_fails_closed():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": None}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"
    assert "scheduled_kickoff_utc" in excinfo.value.detail


@pytest.mark.parametrize("bad_kickoff", ["not-a-timestamp", "2026-09-13T17:00:00"])  # unparseable / naive
def test_resolve_forecast_identity_invalid_kickoff_fails_closed(bad_kickoff):
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": bad_kickoff}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"


def test_resolve_forecast_identity_kickoff_persisted_in_utc():
    # a non-UTC but timezone-AWARE kickoff must be normalized to UTC.
    eastern_kickoff = pd.Timestamp("2026-09-13T13:00:00").tz_localize("America/New_York")
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": eastern_kickoff}])
    identity = prod.resolve_forecast_identity(games, "G1")
    assert identity["scheduled_kickoff_utc"] == "2026-09-13T17:00:00Z"


def test_resolve_forecast_identity_missing_columns_fail_closed():
    games = pd.DataFrame([{"game_id": "G1"}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "G1")
    assert excinfo.value.status == "SCHEMA_DRIFT"


def test_resolve_forecast_identity_unknown_game_id_fails_closed():
    games = _identity_games([{"game_id": "G1", "home_team_id": "KC", "away_team_id": "BUF",
                               "scheduled_kickoff_utc": pd.Timestamp("2026-09-13T17:00:00Z")}])
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.resolve_forecast_identity(games, "does-not-exist")
    assert excinfo.value.status == "SCHEMA_DRIFT"


def test_run_horizon_batch_writes_identity_fields_from_canonical_population(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    checked = 0
    for path in ledger_dir.glob("*.json"):
        record = json.loads(path.read_text())
        pred = record["prediction"]
        game_row = synthetic_games[synthetic_games["game_id"].astype(str) == record["game_id"]].iloc[0]
        assert pred["home_team_id"] == game_row["home_team_id"]
        assert pred["away_team_id"] == game_row["away_team_id"]
        expected_kickoff = prod._as_utc(game_row["scheduled_kickoff_utc"]).isoformat().replace("+00:00", "Z")
        assert pred["scheduled_kickoff_utc"] == expected_kickoff
        # existing prediction field paths are unchanged
        assert isinstance(pred["prediction"]["predicted_margin"], (float, type(None)))
        assert isinstance(pred["prediction"]["predicted_total"], (float, type(None)))
        checked += 1
    assert checked == manifest["game_count"]


def test_one_run_created_at_utc_shared_by_every_forecast_and_the_run_manifest(tmp_path, synthetic_games):
    cutoffs = _last_card_cutoffs(synthetic_games)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    assert "run_created_at_utc" in manifest and manifest["run_created_at_utc"]

    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    records = [json.loads(p.read_text()) for p in ledger_dir.glob("*.json")]
    assert records
    run_created_values = {r["run_created_at_utc"] for r in records}
    run_ids = {r["run_id"] for r in records}
    # exactly one authoritative run-level creation timestamp for the whole batch
    assert run_created_values == {manifest["run_created_at_utc"]}
    # run_id consistency is unchanged by this fix
    assert run_ids == {manifest["run_id"]}
    # it is the SAME instant as the pre-existing started_at_utc, not a second
    # competing timestamp
    assert manifest["run_created_at_utc"] == manifest["started_at_utc"]


def test_predicted_margin_and_total_match_independent_certified_recomputation(tmp_path, synthetic_games):
    """Proves items 14/15/19: the identity/timestamp plumbing changes nothing
    about the scientific prediction. Recomputes predictions via the SAME
    certified functions run_horizon_batch calls (Elo membership -> official
    horizon matrix -> official horizon OOF), entirely independently of
    run_horizon_batch, and diffs against what got persisted."""
    from nfl_hybrid.certification import final_review_2026 as cert
    from nfl_hybrid.evaluation import official_horizon_oof as ohf

    cutoffs = _last_card_cutoffs(synthetic_games)
    as_of = cutoffs["TUE"] + pd.Timedelta(minutes=1)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=as_of, force=True, operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"

    fix71_summary = json.loads((prod.REPO_ROOT / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text())
    fix8_prereg = json.loads((prod.REPO_ROOT / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text())
    hash_checks = cert.verify_certified_hashes(fix71_summary, fix8_prereg)
    feature_state_hash = hash_checks["horizon_feature_semantics_hash"]

    games_df = prod.filter_reg_post(synthetic_games)
    membership_ledger = he.build_horizon_membership_ledger(games_df)
    matrix = ohf.build_official_horizon_matrix(games_df, "TUE", membership_ledger)
    # the frozen six certified Elo features are exactly the OOF model's input
    # columns (default feature_columns=ELO_FEATURE_COLUMNS below); the new
    # public-identity fields are not, and never become, one of them (Section
    # 8: no OOF feature contract expansion). home_team_id/away_team_id were
    # already present in this matrix as pre-existing pivot/carrier columns,
    # unrelated to and untouched by this fix -- only ELO_FEATURE_COLUMNS
    # membership matters for "is this a model predictor."
    assert set(ohf.ELO_FEATURE_COLUMNS).isdisjoint({"home_team_id", "away_team_id", "scheduled_kickoff_utc"})
    assert len(ohf.ELO_FEATURE_COLUMNS) == 6
    assert set(ohf.ELO_FEATURE_COLUMNS).issubset(matrix.columns)

    predictions, _residual_ledger, _fit_counts = ohf.build_official_horizon_oof(
        matrix, horizon="TUE", feature_state_hash=feature_state_hash,
    )
    predictions = predictions.copy()
    predictions["target_cutoff_utc"] = pd.to_datetime(predictions["target_cutoff_utc"], utc=True)
    independent_by_key = {
        (str(r["game_id"]), str(r["target_cutoff_utc"])): r
        for _, r in predictions.iterrows()
    }

    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    checked = 0
    for path in ledger_dir.glob("*.json"):
        record = json.loads(path.read_text())
        key = (record["game_id"], record["target_cutoff_utc"])
        independent = independent_by_key[key]
        persisted = record["prediction"]["prediction"]
        exp_margin = None if pd.isna(independent["predicted_margin"]) else float(independent["predicted_margin"])
        exp_total = None if pd.isna(independent["predicted_total"]) else float(independent["predicted_total"])
        assert persisted["predicted_margin"] == exp_margin
        assert persisted["predicted_total"] == exp_total
        checked += 1
    assert checked == manifest["game_count"]


def test_identical_immutable_replay_with_identity_fields_is_a_noop(tmp_path):
    record = {
        "game_id": "G1", "horizon": "TUE", "target_cutoff_utc": "2026-09-08T16:00:00+00:00",
        "created_at_utc": prod.utc_now().isoformat(), "git_commit": "abc123", "run_id": "run-1",
        "run_created_at_utc": "2026-09-08T00:00:00+00:00",
        "certified_baseline_tag": prod.CERTIFIED_TAG, "certified_baseline_sha": prod.CERTIFIED_SHA,
        "prediction": {
            "game_id": "G1", "horizon": "TUE", "target_cutoff_utc": "2026-09-08T16:00:00+00:00",
            "season": 2026, "week": "1", "season_type": "REG",
            "home_team_id": "KC", "away_team_id": "BUF", "scheduled_kickoff_utc": "2026-09-13T17:00:00Z",
            "prediction": {"model_status": "OOF", "predicted_margin": 1.5, "predicted_total": 44.0,
                           "model_config_hash": "h"},
            "markets": {}, "market_state_hash": None,
            "certified_baseline_sha": prod.CERTIFIED_SHA, "horizon_feature_semantics_hash": "h",
            "operational_model_spec_hash": "h", "fix8_preregistration_hash": "h",
        },
    }
    ledger_root = tmp_path / "forecast-ledger"
    first = prod.write_forecast(ledger_root, record)
    assert first.status == "WRITTEN"

    replay = {**record, "created_at_utc": prod.utc_now().isoformat(), "git_commit": "def456", "run_id": "run-2",
              "run_created_at_utc": "2026-09-09T00:00:00+00:00"}  # volatile provenance differs; deterministic payload identical
    second = prod.write_forecast(ledger_root, replay)
    assert second.status == "IDEMPOTENT_NOOP"
    assert second.prediction_hash == first.prediction_hash


def test_differing_identity_metadata_triggers_immutability_violation(tmp_path):
    base_prediction = {
        "game_id": "G1", "horizon": "TUE", "target_cutoff_utc": "2026-09-08T16:00:00+00:00",
        "season": 2026, "week": "1", "season_type": "REG",
        "home_team_id": "KC", "away_team_id": "BUF", "scheduled_kickoff_utc": "2026-09-13T17:00:00Z",
        "prediction": {"model_status": "OOF", "predicted_margin": 1.5, "predicted_total": 44.0,
                       "model_config_hash": "h"},
        "markets": {}, "market_state_hash": None,
        "certified_baseline_sha": prod.CERTIFIED_SHA, "horizon_feature_semantics_hash": "h",
        "operational_model_spec_hash": "h", "fix8_preregistration_hash": "h",
    }
    record = {
        "game_id": "G1", "horizon": "TUE", "target_cutoff_utc": "2026-09-08T16:00:00+00:00",
        "created_at_utc": prod.utc_now().isoformat(), "git_commit": "abc123", "run_id": "run-1",
        "run_created_at_utc": "2026-09-08T00:00:00+00:00",
        "certified_baseline_tag": prod.CERTIFIED_TAG, "certified_baseline_sha": prod.CERTIFIED_SHA,
        "prediction": base_prediction,
    }
    ledger_root = tmp_path / "forecast-ledger"
    prod.write_forecast(ledger_root, record)

    conflicting = {**record, "run_id": "run-2",
                   "prediction": {**base_prediction, "home_team_id": "DEN"}}  # only identity metadata differs
    with pytest.raises(prod.ProductionHardStop) as excinfo:
        prod.write_forecast(ledger_root, conflicting)
    assert excinfo.value.status == "FORECAST_IMMUTABILITY_VIOLATION"


_PINNED_FILE_SHA256 = {
    "scripts/capture_bdl_2026_asof.py": "0a571a8d9ea9254057d7fdca8819c7d2c762bb052ec195b4b6df4bebfe2dc6ca",
    "scripts/update_v2026_7_qb_projection_shock_shadow.py": "3bb20644d18911ed48d6f1b235508a35c6b265eab6b5e2aef748b6a0246af75e",
    "outputs/v2026_7_prospective_qb_projection_shock_preregistration.json": "aabe03e7a601036fe2e40cf6df91114e90a1a9939e6693b86f37988e4642d6a2",
}


def test_pinned_bdl_and_qb_shadow_files_remain_unchanged():
    import hashlib

    for rel_path, expected_sha256 in _PINNED_FILE_SHA256.items():
        full_path = prod.REPO_ROOT / rel_path
        digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert digest == expected_sha256, f"{rel_path} content changed -- this fix must not touch pinned files"


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
# CALIBRATION FAIL-CLOSED HARDENING -- when the frozen Fix-8 production
# calibration seed is missing/unfitted, price_and_calibrate must NOT
# recombine raw conditional/push probabilities into any ``calibrated_``-named
# field. Every ``calibrated_*`` probability is NaN, calibration_status is
# CALIBRATION_NOT_READY, and the raw_* probabilities stay available for
# diagnostics/provenance. Hermetic: hand-built residual ledger + consensus +
# seed, exercising the REAL production functions (no mocks of them).
# ===========================================================================
_CALIBRATED_PROBABILITY_FIELDS = (
    "calibrated_conditional_upper_probability",
    "calibrated_lower_probability",
    "calibrated_push_probability",
    "calibrated_upper_probability",
)
_RAW_PROBABILITY_FIELDS = (
    "raw_home_probability", "raw_push_probability", "raw_away_probability", "raw_conditional_upper_probability",
)


def _fail_closed_residual_ledger() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1", "G2", "G3"], "season": [2023, 2023, 2023], "week": [10, 10, 10],
        "target_cutoff_utc": pd.to_datetime(["2023-11-07T17:00:00Z"] * 3, utc=True),
        "result_available_at_utc": pd.to_datetime(["2023-11-13T00:00:00Z"] * 3, utc=True),
        "model_config_hash": ["h"] * 3, "feature_state_hash": ["f"] * 3,
        "status": ["OOF"] * 3, "uncertainty_eligible": [True] * 3,
        "predicted_margin": [3.0, -2.0, 1.0], "predicted_total": [45.0, 42.0, 48.0],
        "actual_margin": [np.nan] * 3, "actual_total": [np.nan] * 3,
        "margin_residual_sd_oof": [7.0, 7.0, 7.0], "total_residual_sd_oof": [9.0, 9.0, 9.0],
    })


def _fail_closed_consensus(lines) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["G1", "G2", "G3"], "consensus_line": lines,
        "consensus_novig_probability": [0.5, 0.52, 0.48], "eligible_books": [5, 5, 5],
        "bookmaker_keys": [["a", "b"]] * 3, "selected_returned_snapshot_timestamps": [["2023-11-07T12:00:00Z"]] * 3,
        "min_observation_age_hours": [1.0, 1.0, 1.0], "max_observation_age_hours": [2.0, 2.0, 2.0],
        "consensus_method": ["median"] * 3,
    })


@pytest.mark.parametrize("calibration_seed", [
    pytest.param({}, id="seed-entirely-absent"),
    pytest.param(
        {stream: {"conditional_calibrator_state": {"fitted": False},
                  "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}}
         for stream in ("ATS_TUE", "TOTAL_TUE")},
        id="seed-present-but-unfitted",
    ),
])
def test_price_and_calibrate_fail_closed_when_frozen_seed_unavailable(calibration_seed):
    market_consensus = {
        "ATS": _fail_closed_consensus([-3.5, 2.0, -1.0]), "TOTAL": _fail_closed_consensus([44.5, 41.0, 47.5]),
    }
    priced = prod.price_and_calibrate(
        _fail_closed_residual_ledger(), horizon="TUE", market_consensus=market_consensus,
        calibration_seed=calibration_seed,
    )
    assert set(priced.keys()) == {"ATS_TUE", "TOTAL_TUE"}
    for stream, frame in priced.items():
        # missing/unfitted seed => CALIBRATION_NOT_READY for every row
        assert (frame["calibration_status"] == "CALIBRATION_NOT_READY").all(), stream
        # every calibrated_* probability field is NaN -- raw values are NEVER
        # recombined into a calibrated_-named field when the calibrator is gone
        for field in _CALIBRATED_PROBABILITY_FIELDS:
            assert frame[field].isna().all(), f"{stream}.{field} must be NaN when the frozen calibrator is unavailable"
        # raw probabilities remain available (diagnostics/provenance)
        for field in _RAW_PROBABILITY_FIELDS:
            assert frame[field].notna().all(), f"{stream}.{field} must stay populated"
            assert frame[field].between(0.0, 1.0).all(), f"{stream}.{field}"


def test_run_horizon_batch_market_payload_fail_closed_without_frozen_seed(tmp_path, synthetic_games, monkeypatch):
    """End-to-end through the REAL ``run_horizon_batch``: a batch where the
    market snapshot IS available but the frozen calibration seed is not.

    The tiny synthetic OOF population never reaches raw-pricing readiness, so
    ``price_and_calibrate`` is wrapped to (a) call the REAL function and (b)
    force ``raw_status = RAW_READY`` on its output -- the calibration
    fail-closed masking (``calibrated_* == NaN``, ``calibration_status``) is
    left EXACTLY as the real function produced it. This isolates the
    ``run_horizon_batch`` market-payload branch: every entry that reaches
    pricing must be CALIBRATION_NOT_READY with all ``calibrated_*``
    probabilities None, raw probabilities present, NO calibration_ready_count
    increment, and the canonical margin/total point forecast still written."""
    cutoffs = _last_card_cutoffs(synthetic_games)

    monkeypatch.setattr(prod.rmr, "load_raw_bookmaker_quotes", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(prod.rmr, "build_coherent_book_observations", lambda quotes, market: pd.DataFrame())

    def _synthetic_reconstruct(coherent, targets, *, market):
        consensus = targets[["game_id", "target_cutoff_utc"]].copy()
        consensus["consensus_line"] = -2.5 if market == prod.rmr.MARKET_SPREADS else 44.5
        consensus["line_sd"] = 0.0
        consensus["consensus_novig_probability"] = 0.5
        consensus["probability_sd"] = 0.0
        consensus["eligible_books"] = 5
        consensus["bookmaker_keys"] = [["a", "b"]] * len(consensus)
        consensus["selected_returned_snapshot_timestamps"] = [["2024-01-01T00:00:00Z"]] * len(consensus)
        consensus["min_observation_age_hours"] = 1.0
        consensus["max_observation_age_hours"] = 2.0
        consensus["consensus_method"] = "median"
        consensus = consensus[list(prod.rmr.CONSENSUS_COLUMNS)]
        return prod.rmr.MarketReconstructionResult(market=market, consensus=consensus, coverage={})

    monkeypatch.setattr(prod.rmr, "reconstruct_market_at_cutoffs", _synthetic_reconstruct)
    # Point the seed lookup at an empty dir so this holds even when a real
    # NFL_MODEL_ARTIFACT_ROOT with a real seed is configured on the box.
    monkeypatch.setattr(prod, "artifact_root", lambda: tmp_path)

    real_price_and_calibrate = prod.price_and_calibrate

    def _raw_ready_price_and_calibrate(*args, **kwargs):
        priced = real_price_and_calibrate(*args, **kwargs)
        for frame in priced.values():
            frame["raw_status"] = "RAW_READY"
            frame["raw_home_probability"] = frame["raw_home_probability"].fillna(0.45)
            frame["raw_away_probability"] = frame["raw_away_probability"].fillna(0.45)
            frame["raw_push_probability"] = frame["raw_push_probability"].fillna(0.10)
            frame["raw_conditional_upper_probability"] = frame["raw_conditional_upper_probability"].fillna(0.55)
        return priced

    monkeypatch.setattr(prod, "price_and_calibrate", _raw_ready_price_and_calibrate)

    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["market_ready_counts"]["ATS"] > 0  # market snapshot genuinely present
    # no calibrated_ready_count increment for a non-calibrated stream
    assert manifest["calibration_ready_counts"] == {"ATS": 0, "TOTAL": 0}

    ledger_dir = tmp_path / "production-2026" / "forecast-ledger" / "TUE"
    priced_entries = 0
    for path in ledger_dir.glob("*.json"):
        record = json.loads(path.read_text())
        for market in ("ATS", "TOTAL"):
            entry = record["prediction"]["markets"][market]
            if entry["status"] != "CALIBRATION_NOT_READY":
                continue
            priced_entries += 1
            assert entry["calibration_status"] == "CALIBRATION_NOT_READY"
            for field in _CALIBRATED_PROBABILITY_FIELDS:
                assert entry[field] is None, f"{market}.{field} must be None without a frozen calibrator"
            for field in _RAW_PROBABILITY_FIELDS:
                assert 0.0 <= entry[field] <= 1.0, f"{market}.{field}"
    assert priced_entries > 0  # the fail-closed pricing path was actually exercised

    any_record = json.loads(next(ledger_dir.glob("*.json")).read_text())
    assert any_record["prediction"]["prediction"]["predicted_margin"] is not None
    assert any_record["prediction"]["prediction"]["predicted_total"] is not None


def _mk_market_batch(tmp_path, synthetic_games, monkeypatch):
    """Run a REAL run_horizon_batch with a synthetic-but-complete market
    consensus available (mirrors the fail-closed market test's setup)."""
    cutoffs = _last_card_cutoffs(synthetic_games)
    monkeypatch.setattr(prod.rmr, "load_raw_bookmaker_quotes", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(prod.rmr, "build_coherent_book_observations", lambda quotes, market: pd.DataFrame())

    def _synthetic_reconstruct(coherent, targets, *, market):
        consensus = targets[["game_id", "target_cutoff_utc"]].copy()
        consensus["consensus_line"] = -2.5 if market == prod.rmr.MARKET_SPREADS else 44.5
        consensus["line_sd"] = 0.0
        consensus["consensus_novig_probability"] = 0.5
        consensus["probability_sd"] = 0.0
        consensus["eligible_books"] = 5
        consensus["bookmaker_keys"] = [["a", "b"]] * len(consensus)
        consensus["selected_returned_snapshot_timestamps"] = [["2024-01-01T00:00:00Z"]] * len(consensus)
        consensus["min_observation_age_hours"] = 1.0
        consensus["max_observation_age_hours"] = 2.0
        consensus["consensus_method"] = "median"
        consensus = consensus[list(prod.rmr.CONSENSUS_COLUMNS)]
        return prod.rmr.MarketReconstructionResult(market=market, consensus=consensus, coverage={})

    monkeypatch.setattr(prod.rmr, "reconstruct_market_at_cutoffs", _synthetic_reconstruct)
    monkeypatch.setattr(prod, "artifact_root", lambda: tmp_path)
    manifest = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["market_ready_counts"]["ATS"] > 0
    return tmp_path


def test_market_state_hash_persisted_at_forecast_time_and_ledgers_agree(tmp_path, synthetic_games, monkeypatch):
    from nfl_hybrid.evaluation import prospective_strength_2026 as ps

    root = _mk_market_batch(tmp_path, synthetic_games, monkeypatch)
    fdir = root / "production-2026" / "forecast-ledger" / "TUE"
    edir = root / "production-2026" / "evaluation-ledger"

    forecast_by_game = {}
    for path in fdir.glob("*.json"):
        rec = json.loads(path.read_text())
        h = rec["prediction"]["market_state_hash"]
        assert isinstance(h, str) and len(h) == 64  # explicit, persisted at forecast time
        forecast_by_game[rec["game_id"]] = h

    checked = 0
    for path in edir.glob("*/*.json"):
        if path.name.endswith((".result.json", ".json.tmp")):
            continue
        erec = json.loads(path.read_text())
        gid = erec["game_id"]
        # forecast-ledger and evaluation-ledger hashes agree
        assert erec["market_state_hash"] == forecast_by_game[gid]
        # the reporter's recompute over the immutable record matches exactly
        assert ps.compute_market_state_hash(erec) == erec["market_state_hash"]
        # volatile provenance is not in the hashed payload
        payload = ps.build_market_state_payload(erec)
        for mkt in ("ATS", "TOTAL"):
            if payload[mkt] is not None:
                assert "created_at_utc" not in payload[mkt]
                assert "run_id" not in payload[mkt]
        # tampering a persisted consensus field breaks the recompute
        erec["markets"]["ATS"]["market"]["consensus_line"] = 999.0
        assert ps.compute_market_state_hash(erec) != erec["market_state_hash"]
        checked += 1
    assert checked > 0


def test_market_state_hash_is_idempotent_across_reruns(tmp_path, synthetic_games, monkeypatch):
    root = _mk_market_batch(tmp_path, synthetic_games, monkeypatch)
    fdir = root / "production-2026" / "forecast-ledger" / "TUE"
    first = {json.loads(p.read_text())["game_id"]: json.loads(p.read_text())["prediction"]["market_state_hash"]
             for p in fdir.glob("*.json")}
    cutoffs = _last_card_cutoffs(synthetic_games)
    rerun = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1), force=True,
        operational_root=tmp_path, games=synthetic_games,
    )
    assert rerun["status"] == "SUCCESS"
    second = {json.loads(p.read_text())["game_id"]: json.loads(p.read_text())["prediction"]["market_state_hash"]
              for p in fdir.glob("*.json")}
    assert first == second and len(first) > 0


def test_calibrated_path_values_unchanged_by_fail_closed_hardening():
    """A valid frozen seed still produces the exact previously verified
    calibrated values -- byte-for-byte identical to reconstructing them from
    three_way.py's own apply-path primitives (apply_frozen_conditional_calibrator
    + _predict_push + _recombine). The fail-closed masking must not perturb a
    genuinely CALIBRATED row."""
    market_consensus = {
        "ATS": _fail_closed_consensus([-3.5, 2.0, -1.0]), "TOTAL": _fail_closed_consensus([44.5, 41.0, 47.5]),
    }
    calibration_seed = {
        "ATS_TUE": {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
                    "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
        "TOTAL_TUE": {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.05]], "intercept": [0.01]},
                      "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
    }
    priced = prod.price_and_calibrate(
        _fail_closed_residual_ledger(), horizon="TUE", market_consensus=market_consensus,
        calibration_seed=calibration_seed,
    )
    cfg = CalibrationConfig()
    for stream, frame in priced.items():
        assert (frame["calibration_status"] == "CALIBRATED").all(), stream
        seed_entry = calibration_seed[stream]
        expected_cond = prod.apply_frozen_conditional_calibrator(
            frame["raw_conditional_upper_probability"].to_numpy(float), seed_entry,
        )
        push_state = seed_entry["push_scale_state"]
        legacy_market = prod._LEGACY_PUSH_MARKET_NAME[stream.split("_")[0]]
        push_frame = pd.DataFrame({
            "model_push_probability": frame["raw_push_probability"].to_numpy(float),
            "market_line": frame["market_line"].to_numpy(float),
        })
        expected_push_in = prod._predict_push(
            push_frame, legacy_market, push_state["global_scale"], push_state["bucket_scales"], cfg,
        )
        exp_lower, exp_push, exp_upper = prod._recombine(expected_cond, expected_push_in)
        np.testing.assert_array_equal(
            frame["calibrated_conditional_upper_probability"].to_numpy(float), expected_cond,
        )
        np.testing.assert_array_equal(frame["calibrated_lower_probability"].to_numpy(float), exp_lower)
        np.testing.assert_array_equal(frame["calibrated_push_probability"].to_numpy(float), exp_push)
        np.testing.assert_array_equal(frame["calibrated_upper_probability"].to_numpy(float), exp_upper)


def _complete_seed_entry(coef: float = 0.08, intercept: float = -0.02) -> dict:
    return {
        "conditional_calibrator_state": {"fitted": True, "coefficient": [[coef]], "intercept": [intercept]},
        "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}},
    }


# ===========================================================================
# FULL-SEED FAIL-CLOSED HARDENING -- a stream is CALIBRATED only when EVERY
# frozen calibration component the certified three_way apply path needs is
# present and valid. A fitted conditional calibrator alone is NOT enough:
# an absent/malformed push_scale_state must also fail closed, and raw_push
# must never stand in for a missing frozen push calibrator.
# ===========================================================================
@pytest.mark.parametrize("seed_entry", [
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]}},
        id="push_scale_state-missing-entirely",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
         "push_scale_state": {"bucket_scales": {"NO_PUSH": 0.0}}},
        id="push_scale_state-missing-global_scale",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
         "push_scale_state": {"global_scale": 1.0}},
        id="push_scale_state-missing-bucket_scales",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
         "push_scale_state": {"global_scale": None, "bucket_scales": {"NO_PUSH": 0.0}}},
        id="push_scale_state-null-global_scale",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]], "intercept": [-0.02]},
         "push_scale_state": {"global_scale": 1.0, "bucket_scales": None}},
        id="push_scale_state-null-bucket_scales",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "intercept": [-0.02]},
         "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
        id="conditional-missing-coefficient",
    ),
    pytest.param(
        {"conditional_calibrator_state": {"fitted": True, "coefficient": [[0.08]]},
         "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}}},
        id="conditional-missing-intercept",
    ),
])
def test_price_and_calibrate_fail_closed_when_frozen_seed_incomplete(seed_entry):
    """Any missing/malformed frozen calibration component -> CALIBRATION_NOT_READY
    with every calibrated_* probability NaN; raw probabilities stay available
    and raw_push is NEVER used as a substitute push calibrator."""
    market_consensus = {
        "ATS": _fail_closed_consensus([-3.5, 2.0, -1.0]), "TOTAL": _fail_closed_consensus([44.5, 41.0, 47.5]),
    }
    calibration_seed = {"ATS_TUE": seed_entry, "TOTAL_TUE": seed_entry}
    priced = prod.price_and_calibrate(
        _fail_closed_residual_ledger(), horizon="TUE", market_consensus=market_consensus,
        calibration_seed=calibration_seed,
    )
    assert set(priced.keys()) == {"ATS_TUE", "TOTAL_TUE"}
    for stream, frame in priced.items():
        assert (frame["calibration_status"] == "CALIBRATION_NOT_READY").all(), stream
        for field in _CALIBRATED_PROBABILITY_FIELDS:
            assert frame[field].isna().all(), f"{stream}.{field} must be NaN for an incomplete frozen seed"
        for field in _RAW_PROBABILITY_FIELDS:
            assert frame[field].notna().all(), f"{stream}.{field} must stay populated"
            assert frame[field].between(0.0, 1.0).all(), f"{stream}.{field}"


def test_frozen_stream_calibration_ready_truth_table():
    ready = _complete_seed_entry()
    assert prod._frozen_stream_calibration_ready(ready) is True
    assert prod._frozen_stream_calibration_ready(None) is False
    assert prod._frozen_stream_calibration_ready({}) is False
    assert prod._frozen_stream_calibration_ready(
        {"conditional_calibrator_state": {"fitted": False}, "push_scale_state": ready["push_scale_state"]}
    ) is False
    # fitted conditional but no push state -> NOT ready (the gap this fix closes)
    assert prod._frozen_stream_calibration_ready(
        {"conditional_calibrator_state": ready["conditional_calibrator_state"]}
    ) is False
    assert prod._frozen_stream_calibration_ready(
        {"conditional_calibrator_state": ready["conditional_calibrator_state"],
         "push_scale_state": {"bucket_scales": {"NO_PUSH": 0.0}}}
    ) is False
    assert prod._frozen_stream_calibration_ready(
        {"conditional_calibrator_state": ready["conditional_calibrator_state"],
         "push_scale_state": {"global_scale": 1.0}}
    ) is False


def test_price_and_calibrate_calibrated_path_byte_identical_for_complete_seed():
    """A complete valid Fix-8-shaped seed still produces the exact same
    CALIBRATED values as before the full-seed readiness check was added --
    byte-for-byte against three_way.py's own apply-path primitives."""
    market_consensus = {
        "ATS": _fail_closed_consensus([-3.5, 2.0, -1.0]), "TOTAL": _fail_closed_consensus([44.5, 41.0, 47.5]),
    }
    calibration_seed = {"ATS_TUE": _complete_seed_entry(0.08, -0.02), "TOTAL_TUE": _complete_seed_entry(0.05, 0.01)}
    priced = prod.price_and_calibrate(
        _fail_closed_residual_ledger(), horizon="TUE", market_consensus=market_consensus,
        calibration_seed=calibration_seed,
    )
    cfg = CalibrationConfig()
    for stream, frame in priced.items():
        assert (frame["calibration_status"] == "CALIBRATED").all(), stream
        seed_entry = calibration_seed[stream]
        expected_cond = prod.apply_frozen_conditional_calibrator(
            frame["raw_conditional_upper_probability"].to_numpy(float), seed_entry,
        )
        push_state = seed_entry["push_scale_state"]
        legacy_market = prod._LEGACY_PUSH_MARKET_NAME[stream.split("_")[0]]
        push_frame = pd.DataFrame({
            "model_push_probability": frame["raw_push_probability"].to_numpy(float),
            "market_line": frame["market_line"].to_numpy(float),
        })
        expected_push_in = prod._predict_push(
            push_frame, legacy_market, push_state["global_scale"], push_state["bucket_scales"], cfg,
        )
        exp_lower, exp_push, exp_upper = prod._recombine(expected_cond, expected_push_in)
        np.testing.assert_array_equal(
            frame["calibrated_conditional_upper_probability"].to_numpy(float), expected_cond,
        )
        np.testing.assert_array_equal(frame["calibrated_lower_probability"].to_numpy(float), exp_lower)
        np.testing.assert_array_equal(frame["calibrated_push_probability"].to_numpy(float), exp_push)
        np.testing.assert_array_equal(frame["calibrated_upper_probability"].to_numpy(float), exp_upper)


def test_calibration_fail_closed_hardening_leaves_certified_hashes_untouched():
    """The hardening is an operational-only change: the certified scientific
    hash constants and the certified hash gate still verify unchanged."""
    fix71 = json.loads(
        (prod.REPO_ROOT / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text()
    )
    fix8 = json.loads(
        (prod.REPO_ROOT / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text()
    )
    checks = prod.cert.verify_certified_hashes(fix71, fix8)
    assert checks["horizon_feature_semantics_hash"] == prod.cert.CERTIFIED_HORIZON_FEATURE_SEMANTICS_HASH
    assert checks["horizon_membership_ledger_hash"] == prod.cert.CERTIFIED_HORIZON_MEMBERSHIP_LEDGER_HASH
    assert checks["operational_model_spec_hash"] == prod.cert.CERTIFIED_OPERATIONAL_MODEL_SPEC_HASH
    assert checks["fix8_preregistration_hash"] == prod.cert.CERTIFIED_FIX8_PREREGISTRATION_HASH


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

    def test_preflight_separates_infra_readiness_from_live_input_readiness(self):
        # Deliberately NO artifact_root_path override here -- this test
        # verifies genuine real-environment readiness (including the real
        # Fix-8 calibration seed), unlike the hermetic preflight tests
        # above which specifically exercise the artifact_root_path
        # override in isolation. run_preflight's own writable-directory
        # probes are harmless (a file is written then immediately
        # unlinked) and never touch the forecast/evaluation ledgers.
        result = prod.run_preflight()
        assert result["infra_ready"] is True
        assert result["checks"]["certified_hashes"]["status"] == "MATCH"
        assert result["checks"]["fix8_calibration_seed"]["status"] == "OK"
        # A misleading "waiting" status must never appear.
        assert result["overall_status"] in ("READY", "BLOCKED_ON_LIVE_INPUTS")
        assert result["overall_status"] != "READY_WAITING_FOR_FIRST_DUE_CUTOFF"
        if result["production_run_ready"]:
            assert result["overall_status"] == "READY"
            assert result["schedule_2026_available"] is True
            assert result["live_2026_market_source_registered"] is True
        else:
            # Infra ready, but at least one required live 2026 input is not
            # -- a hard block, with the missing inputs named explicitly.
            assert result["overall_status"] == "BLOCKED_ON_LIVE_INPUTS"
            assert result["blocking_problems"]
            if not result["schedule_2026_available"]:
                assert "schedule_2026_unavailable" in result["blocking_problems"]
            if not result["live_2026_market_source_registered"]:
                assert "live_2026_market_source_unregistered" in result["blocking_problems"]

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
