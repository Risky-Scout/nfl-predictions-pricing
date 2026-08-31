"""Hermetic tests for the frozen prospective-2026 strength contract
(:mod:`nfl_hybrid.evaluation.prospective_strength_2026`) and its reporter.

Every test here runs with ``NFL_MODEL_DATA_ROOT`` and
``NFL_MODEL_ARTIFACT_ROOT`` unset -- no private estate, no network. The
synthetic evaluation-ledger records are hand-built dicts; nothing reads a
real forecast/evaluation ledger or a retrospective 2020-2025 artifact.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from nfl_hybrid.evaluation import prospective_strength_2026 as ps

CERT_SHA = "d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7"

# Contract hardening item 1 -- horizon-aware certified card cutoffs. 2026-09-08
# is a Tuesday, 2026-09-11 the same card's Friday; NY noon = 16:00 UTC (EDT).
# A forecast FILE written a few seconds after the cutoff, inside the certified
# production due-run window, stays eligible.
_CUTOFF_BY_HORIZON = {"TUE": "2026-09-08T16:00:00+00:00", "FRI": "2026-09-11T16:00:00+00:00"}
_CREATED_BY_HORIZON = {"TUE": "2026-09-08T16:00:05+00:00", "FRI": "2026-09-11T16:00:05+00:00"}


# ---------------------------------------------------------------------------
# Record factory -- an evaluation-ledger-shaped record with a fully valid
# provenance block and an attached result, unless overridden.
# ---------------------------------------------------------------------------
def _mk_record(
    game_id: str,
    *,
    horizon: str = "TUE",
    market: str = "ATS",
    season: int | None = 2026,
    season_type: str | None = "REG",
    home_score: int | None = 21,
    away_score: int | None = 17,
    predicted_margin: float | None = 3.0,
    predicted_total: float | None = 44.0,
    consensus_line: float = -2.5,
    consensus_novig_probability: float = 0.52,
    raw_p: float | None = 0.60,
    calibrated_p: float | None = 0.58,
    calibration_status: str = "CALIBRATED",
    drop_provenance: str | None = None,
    with_market_snapshot_hash: bool = True,
    target_cutoff_utc: str | None = None,
    created_at_utc: str | None = None,
    tamper_market_after_hash: bool = False,
) -> dict:
    target_cutoff_utc = target_cutoff_utc or _CUTOFF_BY_HORIZON.get(horizon, _CUTOFF_BY_HORIZON["TUE"])
    created_at_utc = created_at_utc or _CREATED_BY_HORIZON.get(horizon, _CREATED_BY_HORIZON["TUE"])

    market_entry: dict = {"status": "OK"}
    if consensus_line is not None:
        market_entry["market"] = {
            "consensus_line": consensus_line,
            "consensus_novig_probability": consensus_novig_probability,
        }
    if raw_p is not None:
        market_entry["raw_conditional_upper_probability"] = raw_p
    market_entry["calibrated_conditional_upper_probability"] = calibrated_p
    market_entry["calibration_status"] = calibration_status

    provenance = {
        "git_commit": "deadbeef" * 5,
        "operational_model_spec_hash": "op" * 20,
        "horizon_feature_semantics_hash": "fs" * 20,
        "created_at_utc": created_at_utc,
    }
    if drop_provenance and drop_provenance in provenance:
        provenance.pop(drop_provenance)

    rec = {
        "game_id": game_id,
        "horizon": horizon,
        "season": season,
        "season_type": season_type,
        "target_cutoff_utc": target_cutoff_utc,
        "created_at_utc": created_at_utc,
        "git_commit": "deadbeef" * 5,
        "certified_baseline_sha": CERT_SHA,
        "content_hash": f"hash-{game_id}-{horizon}-{market}",
        "forecast": {"predicted_margin": predicted_margin, "predicted_total": predicted_total},
        "markets": {market: market_entry},
        "provenance": provenance,
        "result_record": {"result": {"home_score": home_score, "away_score": away_score}},
    }
    if drop_provenance == "certified_baseline_sha":
        rec.pop("certified_baseline_sha")
    if drop_provenance == "created_at_utc":
        rec.pop("created_at_utc")
        provenance.pop("created_at_utc", None)
    if drop_provenance == "season":
        rec.pop("season")
    if drop_provenance == "season_type":
        rec.pop("season_type")

    # Contract hardening item 3 -- persist a real deterministic market-state
    # hash over the exact market state on the record, so the reporter's
    # recompute verifies (and a later tamper is caught).
    if with_market_snapshot_hash and market_entry.get("market") is not None:
        provenance["market_state_hash"] = ps.compute_market_state_hash(rec)
    if tamper_market_after_hash and market_entry.get("market") is not None:
        market_entry["market"]["consensus_line"] = float(consensus_line) + 1.0
    return rec


def _many(n: int, **kw) -> list[dict]:
    return [_mk_record(f"G{i:04d}", **kw) for i in range(n)]


# ===========================================================================
# 1. Preregistration hash determinism.
# ===========================================================================
def test_preregistration_hash_is_deterministic():
    hashes = {ps.preregistration_hash() for _ in range(5)}
    assert len(hashes) == 1
    doc = ps.frozen_preregistration_document()
    assert doc[ps._HASH_FIELD] == ps.preregistration_hash(doc)
    # hash is computed with the hash field omitted
    without = {k: v for k, v in doc.items() if k != ps._HASH_FIELD}
    assert ps.preregistration_hash(doc) == ps._sha256_hex(without)


def test_preregistration_matches_committed_file():
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "prospective_2026_strength_preregistration.json"
    committed = json.loads(path.read_text())
    assert committed["schema_version"] == "PROSPECTIVE_2026_STRENGTH_V1"
    assert committed[ps._HASH_FIELD] == ps.preregistration_hash()
    # the committed file is exactly the canonical serialization
    assert path.read_text().rstrip("\n") == ps.canonical_json(ps.frozen_preregistration_document())


def test_preregistration_hash_is_sensitive_to_a_threshold_change():
    doc = ps.build_preregistration_payload()
    base = ps.preregistration_hash(doc)
    doc["gates"]["ABSOLUTE_PROBABILITY_QUALITY"]["pooled_brier_below"] = 0.249
    assert ps.preregistration_hash(doc) != base


# ===========================================================================
# 2. Sample-maturity boundaries (Section 3).
# ===========================================================================
@pytest.mark.parametrize(
    "n, expected",
    [
        (0, ps.MATURITY_INSUFFICIENT),
        (63, ps.MATURITY_INSUFFICIENT),
        (64, ps.MATURITY_DESCRIPTIVE),
        (127, ps.MATURITY_DESCRIPTIVE),
        (128, ps.MATURITY_INTERIM),
        (199, ps.MATURITY_INTERIM),
        (200, ps.MATURITY_PROMOTION_ELIGIBLE),
        (500, ps.MATURITY_PROMOTION_ELIGIBLE),
    ],
)
def test_sample_maturity_boundaries(n, expected):
    assert ps.sample_maturity(n) == expected
    assert ps._promotion_eligible(n) == (n >= 200)


def test_unique_completed_games_ignores_unfinished_and_dupes():
    recs = [
        _mk_record("A", horizon="TUE"),
        _mk_record("A", horizon="FRI"),          # same game, other horizon -> still one game
        _mk_record("B", home_score=None, away_score=None),  # not completed
        _mk_record("C"),
    ]
    assert ps._unique_completed_games(recs) == 2  # A and C


# ===========================================================================
# 3. Game-cluster bootstrap (Section 4).
# ===========================================================================
def test_game_cluster_bootstrap_is_deterministic_and_seed_frozen():
    vals = list(np.linspace(-1, 1, 40))
    gids = [f"G{i//2}" for i in range(40)]  # 2 rows per game
    a = ps.game_cluster_bootstrap(vals, gids)
    b = ps.game_cluster_bootstrap(vals, gids)
    assert a == b
    assert a["seed"] == ps.BOOTSTRAP_SEED == 20260829
    assert a["reps"] == ps.BOOTSTRAP_REPS == 10_000
    assert a["n_clusters"] == 20
    assert a["ci_low"] <= a["point_estimate"] <= a["ci_high"]


def test_game_cluster_bootstrap_carries_rows_of_a_drawn_game_together():
    # one cluster dominates: every row of game "BIG" is +100, all others 0.
    vals = [0.0] * 10 + [100.0] * 5
    gids = [f"S{i}" for i in range(10)] + ["BIG"] * 5
    out = ps.game_cluster_bootstrap(vals, gids)
    # resampling clusters (not rows) with a heavy 5-row cluster gives a wide CI
    assert out["ci_high"] > out["point_estimate"] > out["ci_low"]
    assert out["n_clusters"] == 11


def test_game_cluster_bootstrap_empty():
    out = ps.game_cluster_bootstrap([], [])
    assert out["point_estimate"] == ps.NOT_ESTIMABLE


# ===========================================================================
# 4. Fixed-bin ECE (Section 5).
# ===========================================================================
def test_fixed_bin_ece_uses_ten_fixed_bins_and_puts_one_in_final_bin():
    assert ps.ECE_N_BINS == 10
    assert ps.ECE_BIN_EDGES == tuple(round(0.1 * i, 1) for i in range(11))
    # perfectly-calibrated within each fixed bin -> 0
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0, 0.05, 0.95])
    # bin 0: y mean 0 == p mean 0.025-ish? Not exactly; just assert p==1.0 handled.
    ece = ps.fixed_bin_ece(y, p)
    assert isinstance(ece, float) and ece >= 0.0
    # a p of exactly 1.0 with y==1 contributes 0 error in the final bin
    y2 = np.array([1, 1, 1])
    p2 = np.array([1.0, 1.0, 1.0])
    assert ps.fixed_bin_ece(y2, p2) == 0.0


def test_fixed_bin_ece_no_adaptive_bins():
    # 12 points all in [0.30,0.40): one fixed bin, error = |ybar - pbar|
    p = np.full(12, 0.35)
    y = np.array([1] * 6 + [0] * 6)
    assert ps.fixed_bin_ece(y, p) == pytest.approx(abs(0.5 - 0.35))


def test_fixed_bin_ece_empty_is_not_estimable():
    assert ps.fixed_bin_ece([], []) == ps.NOT_ESTIMABLE


# ===========================================================================
# 5. No empty-sample promotion (Section 19).
# ===========================================================================
_PERFORMANCE_ROWS = (
    "CALIBRATION_IMPROVES_RAW_PROBABILITIES",
    "ABSOLUTE_PROBABILITY_QUALITY",
    "POINT_FORECASTING_VS_SPORTSBOOK",
    "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE",
    "MODEL_FAMILY_STABILITY",
    "PROVEN_PROFITABLE_BETTING_MODEL",
)


def test_empty_estate_scorecard_never_promotes_and_never_raises():
    sc = ps.build_scorecard()
    assert sc["sample"]["unique_completed_prospective_games"] == 0
    assert sc["sample"]["maturity"] == ps.MATURITY_INSUFFICIENT
    for key in _PERFORMANCE_ROWS:
        row = sc["rows"][key]
        assert row["status"] not in ("MET", "MET_STRONGLY")
        assert row["reported_status"] not in ("MET", "MET_STRONGLY")
    assert sc["preregistration_hash"] == ps.preregistration_hash()


def test_small_sample_below_promotion_eligible_cannot_promote_performance_rows():
    # 150 completed games (INTERIM_EVIDENCE) with *excellent* fake numbers.
    recs = []
    for i in range(150):
        recs += [
            _mk_record(f"G{i:04d}", horizon="TUE", market="ATS", raw_p=0.55, calibrated_p=0.9, home_score=30, away_score=10),
            _mk_record(f"G{i:04d}", horizon="FRI", market="TOTAL", raw_p=0.55, calibrated_p=0.9, home_score=30, away_score=10),
        ]
    sc = ps.build_scorecard(evaluation_records=recs)
    assert sc["sample"]["maturity"] == ps.MATURITY_INTERIM
    for key in _PERFORMANCE_ROWS:
        assert sc["rows"][key]["status"] not in ("MET", "MET_STRONGLY")
        assert sc["rows"][key]["reported_status"] not in ("MET", "MET_STRONGLY")


# ===========================================================================
# 6. Reporter provenance / eligibility (Section 2).
# ===========================================================================
def test_missing_required_provenance_excludes_row_and_it_is_not_scored():
    good = _mk_record("GOOD")
    bad = _mk_record("BAD", drop_provenance="operational_model_spec_hash")
    split = ps.classify_evidence_eligibility([good, bad])
    assert [r["game_id"] for r in split["eligible"]] == ["GOOD"]
    assert split["excluded"][0]["game_id"] == "BAD"
    assert "operational_model_spec_hash" in split["excluded"][0]["missing_fields"]
    # the excluded row never reaches the maturity count
    sc = ps.build_scorecard(evaluation_records=[good, bad])
    assert sc["sample"]["unique_completed_prospective_games"] == 1


def test_market_relative_row_without_market_snapshot_hash_is_excluded():
    rec = _mk_record("NOHASH", with_market_snapshot_hash=False)
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE"


def test_real_evaluation_ledger_shape_is_accepted_via_hash_pin_and_persisted_market_state_hash():
    """The immutable run_2026 evaluation ledger stores the certified
    operational + feature-semantics hash pair (not an explicit baseline SHA)
    and -- blocker remediation 1 -- an EXPLICIT top-level market_state_hash
    persisted at forecast time over the exact {ATS, TOTAL} consensus market
    payload. Both must be accepted as valid provenance, and the reporter's
    recompute over the immutable record must match exactly."""
    rec = {
        "game_id": "2026_02_KC_BUF",
        "horizon": "FRI",
        "season": 2026,
        "season_type": "REG",
        "target_cutoff_utc": "2026-09-11T16:00:00+00:00",
        "content_hash": "c" * 64,
        "forecast": {"model_status": "OOF", "predicted_margin": 2.0, "predicted_total": 45.0},
        "markets": {
            "ATS": {
                "status": "OK",
                "raw_conditional_upper_probability": 0.55,
                "calibrated_conditional_upper_probability": 0.54,
                "calibration_status": "CALIBRATED",
                "market": {
                    "eligible_books": 3,
                    "consensus_line": -1.5,
                    "consensus_novig_probability": 0.51,
                    "bookmaker_keys": ["book_a", "book_b", "book_c"],
                    "selected_returned_snapshot_timestamps": ["2026-09-11T15:40:00+00:00"],
                    "min_observation_age_hours": 1.0,
                    "max_observation_age_hours": 6.0,
                    "consensus_method": "median",
                },
            }
        },
        "provenance": {
            "git_commit": "abc123" * 6,
            "created_at_utc": "2026-09-11T16:00:02+00:00",
            "operational_model_spec_hash": "3b230bfeee3279c0e3ed9b6a7118931c1a5cf08203155be011f362c66ee8d722",
            "horizon_feature_semantics_hash": "bf0b136dd9b7c7f3741617b6c088926e539406d60b82563839eb1825de9fc72d",
        },
        "result_record": {"result": {"home_score": 24, "away_score": 20}},
    }
    # persisted-at-forecast-time explicit hash (top-level, per the contract)
    rec["market_state_hash"] = ps.compute_market_state_hash(rec)
    split = ps.classify_evidence_eligibility([rec])
    assert [r["game_id"] for r in split["eligible"]] == ["2026_02_KC_BUF"]
    assert split["excluded"] == []
    prov = ps._record_provenance(rec)
    assert prov["certified_baseline_sha"].startswith("HASHPIN:")
    assert prov["market_state_hash"] == rec["market_state_hash"]
    # a structured snapshot with NO explicit persisted hash is now ineligible
    no_hash = {k: v for k, v in rec.items() if k != "market_state_hash"}
    only = ps.classify_evidence_eligibility([no_hash])
    assert only["eligible"] == []
    assert only["excluded"][0]["reason"] == "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE"


def test_a_row_is_never_excluded_for_a_poor_prediction():
    terrible = _mk_record("TERRIBLE", predicted_margin=-999.0, raw_p=0.001, calibrated_p=0.001,
                           home_score=3, away_score=45)
    split = ps.classify_evidence_eligibility([terrible])
    assert [r["game_id"] for r in split["eligible"]] == ["TERRIBLE"]
    assert split["excluded"] == []


# ===========================================================================
# 7. CALIBRATION_NOT_READY / no numeric calibrated_* when not CALIBRATED.
# ===========================================================================
def test_calibration_not_ready_rows_do_not_enter_calibrated_scoring():
    # calibrated_p present numerically but status is CALIBRATION_NOT_READY:
    # such a row must not be counted in the calibrated-stream comparison.
    ready = _many(30, horizon="TUE", market="ATS", calibration_status="CALIBRATED",
                  raw_p=0.6, calibrated_p=0.58)
    not_ready = [
        _mk_record(f"NR{i:04d}", horizon="TUE", market="ATS",
                   calibration_status="CALIBRATION_NOT_READY", raw_p=0.6, calibrated_p=None)
        for i in range(30)
    ]
    gate = ps.gate_calibration_improves_raw(ready + not_ready, ps.MATURITY_PROMOTION_ELIGIBLE, 300)
    assert gate["per_stream"]["ATS_TUE"]["n"] == 30  # only the CALIBRATED rows


def test_numeric_calibrated_probability_ignored_when_status_not_calibrated():
    rows = [
        _mk_record(f"X{i:04d}", horizon="TUE", market="ATS",
                   calibration_status="CALIBRATION_NOT_READY", raw_p=0.6, calibrated_p=0.99)
        for i in range(40)
    ]
    gate = ps.gate_calibration_improves_raw(rows, ps.MATURITY_PROMOTION_ELIGIBLE, 300)
    # the 0.99 "calibrated" values must never be scored
    assert gate["per_stream"]["ATS_TUE"] == {"n": 0, "status": "INSUFFICIENT"}
    assert gate["status"] != "MET_STRONGLY"


# ===========================================================================
# 8. Absolute-quality gate (Section 7).
# ===========================================================================
def test_absolute_quality_gate_requires_promotion_eligible_maturity():
    recs = _many(120, horizon="TUE", market="ATS")
    gate = ps.gate_absolute_probability_quality(recs, ps.MATURITY_DESCRIPTIVE, 100)
    assert gate["gate_booleans"]["promotion_eligible_maturity"] is False
    assert gate["status"] != "MET_STRONGLY"


def test_absolute_quality_gate_reports_auc_and_slope_intercept():
    # deterministic labelled rows with signal so AUC/slope are estimable
    rng = np.random.default_rng(0)
    recs = []
    for i in range(240):
        p = float(np.clip(rng.beta(2, 2), 0.02, 0.98))
        y = 1 if rng.random() < p else 0
        recs.append(_mk_record(f"Q{i:04d}", horizon="TUE", market="ATS",
                               home_score=(24 if y else 10), away_score=(10 if y else 24),
                               consensus_line=0.0, raw_p=p, calibrated_p=p))
    gate = ps.gate_absolute_probability_quality(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 240)
    pe = gate["point_estimate"]
    assert pe["auc"] is not None
    assert "calibration_slope" in pe and "calibration_intercept" in pe
    assert pe["sharpness_descriptive"] != ps.NOT_ESTIMABLE
    assert set(gate["gate_booleans"]) >= {
        "pooled_log_loss_below_ln2", "auc_ci_lower_above_0_50", "calibration_slope_in_range",
    }


# ===========================================================================
# 9. Point-forecast market gate (Section 8).
# ===========================================================================
def test_point_gate_negative_when_model_worse_than_book():
    # model margin always off by 20; book exact -> model clearly worse.
    recs = []
    for i in range(240):
        recs += [
            _mk_record(f"P{i:04d}", horizon="TUE", market="ATS",
                       home_score=20, away_score=20, predicted_margin=20.0, consensus_line=0.0),
            _mk_record(f"P{i:04d}", horizon="FRI", market="TOTAL",
                       home_score=20, away_score=20, predicted_total=80.0, consensus_line=40.0),
        ]
    gate = ps.gate_point_forecasting_vs_sportsbook(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 240)
    assert gate["status"] in ("NOT_MET", "NOT_DEMONSTRATED")
    assert gate["gate_booleans"]["pooled_margin_rmse_beats_book_by_0_25"] is False


# ===========================================================================
# 10. Probability-market edge gate (Section 9) -- no ATS/TOTAL pooling.
# ===========================================================================
def test_probability_edge_gate_scores_markets_separately():
    recs = []
    for i in range(220):
        recs += [
            _mk_record(f"E{i:04d}", horizon="TUE", market="ATS", raw_p=0.6, calibrated_p=0.6,
                       consensus_novig_probability=0.5, home_score=24, away_score=10, consensus_line=0.0),
            _mk_record(f"E{i:04d}", horizon="FRI", market="TOTAL", raw_p=0.6, calibrated_p=0.6,
                       consensus_novig_probability=0.5, home_score=24, away_score=10, consensus_line=30.0),
        ]
    gate = ps.gate_sportsbook_probability_edge(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 220)
    assert set(gate["per_market"]) == {"ATS", "TOTAL"}
    assert gate["gate_booleans"]["no_ats_total_pooling_to_rescue"] is True
    # each market has its own booleans dict
    for m in ("ATS", "TOTAL"):
        assert "log_loss_delta_ci_upper_below_zero" in gate["per_market"][m]["gate_booleans"]


# ===========================================================================
# 11. Family-stability gate (Sections 10-11).
# ===========================================================================
def test_family_stability_preserved_mixed_without_shadow_evidence():
    gate = ps.gate_model_family_stability([], ps.MATURITY_PROMOTION_ELIGIBLE, 300)
    assert gate["status"] == "MIXED"
    assert gate["shadow_record_count"] == 0
    assert "production-selection authority" in gate["note"]
    assert "RIDGE_ALPHA_100" in gate["note"]


def test_family_stability_never_met_strongly_below_promotion_eligible():
    gate = ps.gate_model_family_stability([{"candidate": "RIDGE_ALPHA_100"}], ps.MATURITY_INTERIM, 150)
    assert gate["status"] != "MET_STRONGLY"


# ===========================================================================
# 12. Profitability disabled without executable-book policy (Sections 13-15).
# ===========================================================================
def test_profitability_disabled_without_executable_book_policy():
    gate = ps.gate_profitability([], {"frozen": False, "reason": "EXECUTABLE_BOOK_POLICY_NOT_FROZEN"},
                                 ps.MATURITY_PROMOTION_ELIGIBLE, 300, 0)
    assert gate["status"] == "NOT_ESTABLISHED"
    assert gate["betting_rule_status"] == ps.BETTING_RULE_STATUS_NOT_ACTIVATED
    assert gate["gate_booleans"]["executable_book_policy_frozen"] is False
    assert "EXECUTABLE_BOOK_POLICY_NOT_FROZEN" in gate["failure_reasons"]


def test_executable_book_policy_status_absent_file(tmp_path):
    status = ps.executable_book_policy_status(tmp_path / "does_not_exist.json")
    assert status["frozen"] is False
    assert status["reason"] == "EXECUTABLE_BOOK_POLICY_NOT_FROZEN"
    assert status["profitability_status"] == "NOT_ESTABLISHED"


def test_executable_book_policy_status_present_file_hashed(tmp_path):
    cfg = tmp_path / "executable_books_2026.json"
    cfg.write_text(json.dumps({"ordered_books": ["book_a", "book_b"]}))
    status = ps.executable_book_policy_status(cfg)
    assert status["frozen"] is True
    assert status["policy_hash"] == ps._sha256_hex({"ordered_books": ["book_a", "book_b"]})


def test_profitability_gate_stays_not_established_even_with_frozen_policy_and_no_wagers():
    gate = ps.gate_profitability([], {"frozen": True, "policy_hash": "abc"},
                                 ps.MATURITY_PROMOTION_ELIGIBLE, 300, settled_non_push_wagers=0)
    assert gate["status"] == "NOT_ESTABLISHED"
    assert gate["gate_booleans"]["min_settled_non_push_wagers"] is False


# ===========================================================================
# 13. Immutable shadow records (Section 10).
# ===========================================================================
def _shadow_rec(gid="G1", horizon="TUE", cand="RIDGE_ALPHA_100", cutoff="2026-09-08T16:00:00+00:00", pm=1.0):
    return {
        "game_id": gid, "horizon": horizon, "candidate": cand, "target_cutoff_utc": cutoff,
        "prediction": {"status": "SHADOW_OOF", "predicted_margin": pm, "predicted_total": 44.0},
    }


def test_shadow_record_first_write_wins_and_is_idempotent(tmp_path):
    root = tmp_path / "shadow"
    r1 = ps.write_shadow_record(root, _shadow_rec())
    assert r1["status"] == "WRITTEN"
    r2 = ps.write_shadow_record(root, _shadow_rec())
    assert r2["status"] == "IDEMPOTENT_NOOP"


def test_shadow_record_conflicting_payload_is_a_violation(tmp_path):
    root = tmp_path / "shadow"
    ps.write_shadow_record(root, _shadow_rec(pm=1.0))
    with pytest.raises(ps.ShadowLedgerViolation):
        ps.write_shadow_record(root, _shadow_rec(pm=2.0))


def test_shadow_record_rejects_outcome_leak(tmp_path):
    root = tmp_path / "shadow"
    rec = _shadow_rec()
    rec["home_score"] = 21
    with pytest.raises(ps.ShadowLedgerViolation):
        ps.write_shadow_record(root, rec)


def test_shadow_record_rejects_unknown_candidate(tmp_path):
    root = tmp_path / "shadow"
    with pytest.raises(ps.ShadowLedgerViolation):
        ps.write_shadow_record(root, _shadow_rec(cand="XGBOOST_V2"))


# ===========================================================================
# 14. Shadow failure cannot affect the production forecast.
# ===========================================================================
def test_shadow_failure_row_is_isolated_and_never_a_production_forecast(tmp_path):
    root = tmp_path / "shadow"
    rec = _shadow_rec()
    rec["prediction"] = {"status": "SHADOW_FIT_UNAVAILABLE", "predicted_margin": None, "predicted_total": None}
    out = ps.write_shadow_record(root, rec)
    assert out["status"] == "WRITTEN"
    # nothing in the shadow ledger feeds the canonical rows other than
    # MODEL_FAMILY_STABILITY, and that row can never gain production authority.
    sc = ps.build_scorecard(shadow_records=ps.load_shadow_ledger(root))
    assert sc["rows"]["MODEL_FAMILY_STABILITY"]["reported_status"] == "MIXED"
    assert "PROVEN_PROFITABLE_BETTING_MODEL" in sc["rows"]


# ===========================================================================
# 15. No 2020-2025 artifact contamination (Section 18/24).
# ===========================================================================
def test_reporter_module_never_references_retrospective_artifacts():
    src = inspect.getsource(ps)
    for banned in ("fix7_model_family_selection_summary", "fix8_official_oof_calibration_summary",
                   "real_ats_total_chronological_calibration_proof", "walkforward_2025",
                   "phase_c_model", "market_relative_probability_scorecard",
                   "confirmation_2024", "final_test_2025"):
        assert banned not in src


def test_build_scorecard_only_consumes_passed_in_ledgers():
    # No positional/keyword path reads disk: a call with everything empty
    # returns a well-formed scorecard.
    sc = ps.build_scorecard(
        forecast_records=[], evaluation_records=[], shadow_records=[],
        price_records=[], run_manifests=[], book_policy={"frozen": False},
    )
    assert sc["schema_version"] == "PROSPECTIVE_2026_STRENGTH_V1"
    assert set(sc["rows"]) == set(ps.CURRENT_LABELS)


# ===========================================================================
# 16. Content-addressed feature cache remains untouched (Section 24).
# ===========================================================================
def test_content_addressed_feature_cache_contract_is_intact():
    from nfl_hybrid.selection import feature_deduction_2026 as fdmod

    assert hasattr(fdmod, "_games_content_fingerprint")
    src = inspect.getsource(fdmod.build_candidate_matrix)
    assert "_games_content_fingerprint(games)" in src
    assert "id(games)" not in src


# ===========================================================================
# 17. Reporter markdown render + data-through timestamp.
# ===========================================================================
def test_render_markdown_and_data_through_timestamp():
    sc = ps.build_scorecard(evaluation_records=_many(4), data_through_utc="2026-10-01T00:00:00+00:00")
    md = ps.render_scorecard_markdown(sc)
    assert "Prospective 2026 Status Scorecard" in md
    assert "2026-10-01T00:00:00+00:00" in md
    for key in ps.CURRENT_LABELS:
        assert key in md
    assert sc["rows"]["ABSOLUTE_PROBABILITY_QUALITY"]["data_through_utc"] == "2026-10-01T00:00:00+00:00"


# ===========================================================================
# 18. Shadow model-family runner -- historical-fixture integration only
#     (Section 20). Uses a self-contained synthetic already-elapsed season.
# ===========================================================================
def _load_shadow_runner():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_2026_shadow_model_family.py"
    spec = importlib.util.spec_from_file_location("run_2026_shadow_model_family", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_elapsed_games(seed: int = 7) -> "object":
    import pandas as pd

    rng = np.random.default_rng(seed)
    teams = ["BUF", "MIA", "NE", "NYJ", "KC", "LAC", "DEN", "LV"]
    rows = []
    n = 0
    for week in range(1, 19):  # 18 weeks x 4 games -> late cards clear the 48-game training floor
        monday = pd.Timestamp("2023-09-04") + pd.Timedelta(weeks=week - 1)
        for pair in range(4):
            home, away = teams[pair * 2], teams[pair * 2 + 1]
            rows.append({
                "game_id": f"2023_{week:02d}_{away}_{home}_{n}",
                "season": 2023, "week": week, "season_type": "REG",
                "home_team_id": home, "away_team_id": away,
                "scheduled_kickoff_utc": (monday + pd.Timedelta(days=6, hours=17)).tz_localize("UTC"),
                "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                "neutral_site": False,
            })
            n += 1
    return pd.DataFrame(rows)


def test_shadow_runner_writes_immutable_ledger_and_never_leaks_outcomes(tmp_path):
    runner = _load_shadow_runner()
    games = _synthetic_elapsed_games()
    shadow_root = tmp_path / "shadow"

    summary = runner.run_shadow_horizon(games, "TUE", shadow_root, git_commit="testsha")
    assert summary["records_written"] > 0
    assert set(summary["candidates"]) == set(ps.FROZEN_FIX7_SHADOW_CANDIDATES)

    written = ps.load_shadow_ledger(shadow_root)
    assert written, "expected shadow ledger records on disk"
    for rec in written:
        assert rec["horizon"] == "TUE"
        assert rec["candidate"] in ps.FROZEN_FIX7_SHADOW_CANDIDATES
        assert not (ps._SHADOW_FORBIDDEN_FIELDS & set(rec))          # no outcome columns
        assert "home_score" not in rec and "away_score" not in rec
        assert rec["prediction"]["status"] in ("SHADOW_OOF", "SHADOW_FIT_UNAVAILABLE")
        assert rec["certified_baseline_sha"] == CERT_SHA

    # at least one real SHADOW_OOF prediction was produced (training floor cleared)
    assert any(r["prediction"]["status"] == "SHADOW_OOF" for r in written)

    # re-run is idempotent -- immutable first-write-wins
    again = runner.run_shadow_horizon(games, "TUE", shadow_root, git_commit="testsha")
    assert again["records_written"] == 0
    assert again["records_idempotent_noop"] == summary["records_written"] + summary["records_idempotent_noop"]


def test_shadow_runner_does_not_write_any_forecast_or_evaluation_ledger(tmp_path):
    runner = _load_shadow_runner()
    games = _synthetic_elapsed_games()
    operational_root = tmp_path / "artifact-root"
    shadow_root = operational_root / "production-2026" / ps.SHADOW_LEDGER_SUBDIR
    runner.run_shadow_horizon(games, "FRI", shadow_root, git_commit="testsha")

    base = operational_root / "production-2026"
    assert not (base / "forecast-ledger").exists()
    assert not (base / "evaluation-ledger").exists()
    assert not (base / "run-manifests").exists()
    assert (base / ps.SHADOW_LEDGER_SUBDIR).exists()


def test_shadow_ledger_feeds_only_the_family_row_and_grants_no_authority(tmp_path):
    runner = _load_shadow_runner()
    games = _synthetic_elapsed_games()
    shadow_root = tmp_path / "shadow"
    runner.run_shadow_horizon(games, "TUE", shadow_root, git_commit="testsha")
    shadow_records = ps.load_shadow_ledger(shadow_root)

    sc = ps.build_scorecard(shadow_records=shadow_records)
    # shadow evidence present but sample immature -> family row stays MIXED,
    # every other row is exactly what it would be with no shadow evidence.
    assert sc["rows"]["MODEL_FAMILY_STABILITY"]["reported_status"] == "MIXED"
    assert sc["rows"]["MODEL_FAMILY_STABILITY"]["shadow_record_count"] == len(shadow_records)
    assert sc["rows"]["PROVEN_PROFITABLE_BETTING_MODEL"]["reported_status"] == "NOT_ESTABLISHED"


# ===========================================================================
# CONTRACT HARDENING (items 1-11)
# ===========================================================================
import pathlib  # noqa: E402

_PREREG = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "outputs" / "prospective_2026_strength_preregistration.json").read_text()
)


# --- item 1: forecast-time (execution-window) eligibility -------------------
def test_forecast_written_shortly_after_cutoff_but_in_window_stays_eligible():
    # default factory creates the file 5s AFTER target_cutoff_utc, inside the
    # certified TUE noon..noon+20m production due-run window.
    tue = _mk_record("AFTER_TUE", horizon="TUE")
    fri = _mk_record("AFTER_FRI", horizon="FRI")
    assert tue["created_at_utc"] > tue["target_cutoff_utc"]
    split = ps.classify_evidence_eligibility([tue, fri])
    assert sorted(r["game_id"] for r in split["eligible"]) == ["AFTER_FRI", "AFTER_TUE"]
    assert split["excluded"] == []


def test_forecast_created_outside_execution_window_is_ineligible():
    late = _mk_record("LATE", horizon="TUE", created_at_utc="2026-09-08T18:30:00+00:00")  # ~2.5h late
    split = ps.classify_evidence_eligibility([late])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "FORECAST_OUTSIDE_EXECUTION_WINDOW"


def test_forecast_generation_mapped_to_a_later_card_is_ineligible():
    # claims week-1 TUE cutoff but was actually generated during a *later*
    # TUE window -- i.e. it could have used data past its declared cutoff.
    wrong = _mk_record(
        "WRONG_CARD", horizon="TUE",
        target_cutoff_utc="2026-09-08T16:00:00+00:00",
        created_at_utc="2026-09-15T16:00:05+00:00",
    )
    split = ps.classify_evidence_eligibility([wrong])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "FORECAST_OUTSIDE_EXECUTION_WINDOW"


def test_execution_window_semantics_are_frozen_in_prereg():
    ee = _PREREG["evidence_eligibility"]
    assert ee["forecast_created_at_utc_may_be_at_or_after_target_cutoff_utc"] is True
    assert ee["forecast_created_outside_permitted_execution_window_is_ineligible"] is True
    assert ee["target_cutoff_utc_is_information_as_of_cutoff"] is True
    assert "is_within_due_window" in ee["execution_window_semantics_source"]


# --- item 2: frozen 2026 REG+POST population -------------------------------
def test_preseason_records_never_enter_any_analysis():
    recs = []
    for i in range(220):
        recs += [
            _mk_record(f"PRE{i:04d}", horizon="TUE", market="ATS", season_type="PRE",
                       raw_p=0.55, calibrated_p=0.95, home_score=40, away_score=3),
            _mk_record(f"PRE{i:04d}", horizon="FRI", market="TOTAL", season_type="PRE",
                       raw_p=0.55, calibrated_p=0.95, home_score=40, away_score=3),
        ]
    sc = ps.build_scorecard(evaluation_records=recs, data_through_utc="2026-12-01T00:00:00+00:00")
    assert sc["sample"]["unique_completed_prospective_games"] == 0
    assert sc["sample"]["maturity"] == ps.MATURITY_INSUFFICIENT
    assert len(sc["sample"]["excluded_out_of_population"]) == len(recs)
    for key in _PERFORMANCE_ROWS:
        assert sc["rows"][key]["reported_status"] not in ("MET", "MET_STRONGLY")


def test_non_2026_season_is_out_of_population():
    rec = _mk_record("OLD", season=2025)
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "OUT_OF_PROSPECTIVE_POPULATION"


def test_reg_and_post_both_qualify():
    split = ps.classify_evidence_eligibility([
        _mk_record("R", season_type="REG"), _mk_record("P", season_type="POST"),
    ])
    assert sorted(r["game_id"] for r in split["eligible"]) == ["P", "R"]


def test_missing_season_type_is_identifier_failure():
    rec = _mk_record("NOSTYPE", drop_provenance="season_type")
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "IDENTIFIER_FAILURE"
    assert "season_type" in split["excluded"][0]["missing_fields"]


def test_population_freeze_is_in_prereg():
    pop = _PREREG["prospective_population"]
    assert pop["season"] == 2026
    assert pop["season_types"] == ["REG", "POST"]
    assert set(pop["preseason_excluded_from"]) == {
        "sample_maturity", "calibration_scoring", "sportsbook_comparisons",
        "model_family_stability", "profitability", "season_final_completion",
    }
    assert "season" in _PREREG["evidence_eligibility"]["required_forecast_provenance"]
    assert "season_type" in _PREREG["evidence_eligibility"]["required_forecast_provenance"]


# --- item 3: market-state provenance hash --------------------------------
def test_market_state_hash_is_computed_from_the_record_alone_and_deterministic():
    rec = _mk_record("MSH")
    h1 = ps.compute_market_state_hash(rec)
    h2 = ps.compute_market_state_hash(json.loads(json.dumps(rec)))
    assert h1 == h2 and len(h1) == 64
    # the reporter accepts the persisted hash without any external market file
    assert ps._record_provenance(rec)["market_state_hash"] == h1


def test_tampered_persisted_market_state_is_hash_mismatch():
    tampered = _mk_record("TAMPER", tamper_market_after_hash=True)
    split = ps.classify_evidence_eligibility([tampered])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "HASH_MISMATCH"


def test_market_state_hash_change_is_detected_for_every_market_relative_row():
    clean = _mk_record("CLEAN", market="ATS")
    assert ps.classify_evidence_eligibility([clean])["eligible"]
    # mutate the persisted consensus payload AFTER the hash was frozen
    clean["markets"]["ATS"]["market"]["consensus_novig_probability"] = 0.99
    split = ps.classify_evidence_eligibility([clean])
    assert split["excluded"][0]["reason"] == "HASH_MISMATCH"


def test_market_state_provenance_is_frozen_in_prereg():
    msp = _PREREG["market_state_provenance"]
    assert msp["algorithm"] == "SHA-256"
    assert msp["mismatch_status"] == "HASH_MISMATCH"
    assert msp["no_later_mutable_raw_market_file_required_to_verify"] is True
    assert set(msp["recompute_verified_before"]) == {
        "POINT_FORECASTING_VS_SPORTSBOOK", "DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE",
        "PROVEN_PROFITABLE_BETTING_MODEL",
    }


# --- item 4: frozen "adequately-sized" -> 64 -----------------------------
def test_adequate_model_family_scope_threshold_is_frozen_at_64():
    assert ps.ADEQUATE_MODEL_FAMILY_SCOPE_MIN_UNIQUE_GAMES == 64
    mfa = _PREREG["model_family_scope_adequacy"]
    assert mfa["min_unique_completed_game_ids"] == 64
    assert set(mfa["scopes"]) == {"TUE", "FRI", "pooled", "first_third", "middle_third", "final_third"}
    assert mfa["below_threshold_conclusion"] == "INSUFFICIENT"
    assert _PREREG["gates"]["MODEL_FAMILY_STABILITY"]["adequate_scope_min_unique_completed_games"] == 64


# --- item 5: frozen ATS/TOTAL calibration evidence labels ---------------
def test_calibration_evidence_label_insufficient_below_64_unique_games():
    recs = (_many(20, horizon="TUE", market="ATS", raw_p=0.6, calibrated_p=0.55)
            + _many(20, horizon="FRI", market="ATS", raw_p=0.6, calibrated_p=0.55))
    gate = ps.gate_calibration_improves_raw(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 300)
    assert gate["ats_calibration_evidence"] == "INSUFFICIENT"


def test_calibration_evidence_label_is_always_from_the_frozen_set():
    rng = np.random.default_rng(3)
    recs = []
    for i in range(90):
        p = float(np.clip(rng.beta(2, 2), 0.05, 0.95))
        y = 1 if rng.random() < p else 0
        for hz in ("TUE", "FRI"):
            recs.append(_mk_record(f"CE{i:04d}", horizon=hz, market="ATS", consensus_line=0.0,
                                   home_score=(24 if y else 10), away_score=(10 if y else 24),
                                   raw_p=p, calibrated_p=float(np.clip(p + 0.01, 0.02, 0.98))))
    gate = ps.gate_calibration_improves_raw(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 300)
    assert gate["ats_calibration_evidence"] in ps.CALIBRATION_STREAM_EVIDENCE_LABELS


def test_calibration_stream_evidence_label_definitions_frozen_in_prereg():
    defs = _PREREG["calibration_stream_evidence_label_definitions"]
    assert set(defs) == {"INSUFFICIENT", "STRONG", "SUGGESTIVE", "MIXED"}
    assert "fewer than 64" in defs["INSUFFICIENT"]
    assert ">=200" in defs["STRONG"] and "95% CI upper bounds" in defs["STRONG"]
    assert ">=64" in defs["SUGGESTIVE"]
    assert _PREREG["calibration_stream_evidence_min_unique_games"] == 64
    assert _PREREG["calibration_stream_evidence_strong_min_unique_games"] == 200


# --- item 6: define "materially inconsistent" --------------------------
def _clean_ats(n, tag):
    """Well-separated, well-calibrated ATS rows: log loss << ln2, Brier tiny,
    AUC = 1, fixed-bin ECE = 0.05 (<= 0.075). None of the four
    materially-inconsistent conditions is met."""
    out = []
    for i in range(n):
        y = i % 2
        p = 0.95 if y else 0.05
        out.append(_mk_record(f"{tag}{i:04d}", market="ATS", consensus_line=0.0,
                              home_score=(24 if y else 10), away_score=(10 if y else 24),
                              raw_p=p, calibrated_p=p))
    return out


def test_materially_inconsistent_when_one_market_is_at_the_coin_flip_reference():
    total_recs = []
    for i in range(80):
        over = i % 2 == 0
        total_recs.append(_mk_record(
            f"MI{i:04d}", market="TOTAL", consensus_line=40.0, calibration_status="CALIBRATED",
            calibrated_p=0.5, raw_p=0.5,
            home_score=(30 if over else 10), away_score=(25 if over else 10),
        ))
    inconsistent, detail = ps._ats_total_materially_inconsistent(total_recs)
    assert inconsistent is True
    assert detail["TOTAL"]["flags"]["log_loss_at_or_above_ln2"] is True


def test_not_materially_inconsistent_when_both_markets_are_clean():
    inconsistent, detail = ps._ats_total_materially_inconsistent(_clean_ats(160, "CLN"))
    assert inconsistent is False
    assert not any(detail["ATS"]["flags"].values())


def test_materially_inconsistent_definition_frozen_in_prereg():
    mi = _PREREG["gates"]["ABSOLUTE_PROBABILITY_QUALITY"]["materially_inconsistent_if_either_market"]
    assert mi["log_loss_at_or_above"] == pytest.approx(0.6931471805599453)
    assert mi["brier_at_or_above"] == 0.25
    assert mi["auc_at_or_below"] == 0.5
    assert mi["ece_above"] == 0.075


# --- item 7: profitability "non-adverse" ------------------------------
def test_market_non_adverse_rule():
    assert ps.market_non_adverse(0.0, 0.0) is True
    assert ps.market_non_adverse(0.10, 3.0) is True
    assert ps.market_non_adverse(0.10, -0.01) is False       # negative net units -> never non-adverse
    assert ps.market_non_adverse(-0.01, 5.0) is False        # ROI CI upper < 0
    assert ps.market_non_adverse(None, 1.0) is False
    rule = _PREREG["profitability_market_non_adverse_rule"]
    assert rule["roi_cluster_bootstrap_95_ci_upper_at_least"] == 0.0
    assert rule["realized_net_units_at_least"] == 0.0
    assert rule["negative_realized_net_units_is_adverse"] is True


# --- item 8: executable-book policy lock -----------------------------
_WAGER_ID = {"game_id": "2026_05_KC_BUF", "horizon": "TUE", "market": "ATS", "side": "HOME", "wager_seq": 1}


def test_executable_book_policy_lock_first_write_wins(tmp_path):
    r1 = ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1")
    assert r1["status"] == "WRITTEN"
    assert r1["lock"]["betting_rule_2026_v1_hash"] == ps.BETTING_RULE_2026_V1_HASH
    for field in ("policy_hash", "betting_rule_2026_v1_hash", "first_eligible_wager_identity",
                  "lock_timestamp_utc", "git_commit"):
        assert field in r1["lock"]
    # identical re-write is an idempotent no-op
    r2 = ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha2")
    assert r2["status"] == "IDEMPOTENT_NOOP"
    # a different policy hash hard-stops
    with pytest.raises(ps.BettingIntegrityConflict):
        ps.write_executable_book_policy_lock(
            tmp_path, policy_hash="policy_b", first_eligible_wager_identity=_WAGER_ID, git_commit="sha3")


def test_executable_book_policy_lock_betting_rule_hash_conflict(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1",
        betting_rule_hash="STALE_BETTING_RULE_HASH")
    verdict = ps.verify_executable_book_policy_lock(tmp_path, policy_hash="policy_a")
    assert verdict["locked"] is True
    assert verdict["conflict"] is True
    assert verdict["flag"] == "BETTING_INTEGRITY_CONFLICT"


def test_verify_lock_absent_is_not_a_conflict(tmp_path):
    verdict = ps.verify_executable_book_policy_lock(tmp_path, policy_hash="anything")
    assert verdict == {"locked": False, "conflict": False, "flag": None,
                       "path": str(ps.executable_book_policy_lock_path(tmp_path))}


def test_profitability_gate_barred_by_betting_integrity_conflict():
    gate = ps.gate_profitability(
        [], {"frozen": True, "policy_hash": "policy_b"}, ps.MATURITY_PROMOTION_ELIGIBLE, 300, 200,
        policy_lock={"locked": True, "conflict": True, "flag": "BETTING_INTEGRITY_CONFLICT"},
    )
    assert gate["status"] == "NOT_ESTABLISHED"
    assert gate["gate_booleans"]["no_betting_integrity_conflict"] is False
    assert "BETTING_INTEGRITY_CONFLICT" in gate["failure_reasons"]
    assert gate["contaminated_estate_met_strongly_barred"] is True


def test_executable_book_policy_lock_frozen_in_prereg_and_no_config_file():
    lock = _PREREG["executable_book_policy_lock"]
    assert lock["path"].endswith("production-2026/betting-policy/executable_book_policy_lock.json")
    assert lock["first_write_wins"] is True
    assert lock["conflict_flag"] == "BETTING_INTEGRITY_CONFLICT"
    assert lock["conflict_forbids_profitability_met_strongly_for_contaminated_estate"] is True
    assert set(lock["contains_at_minimum"]) == {
        "policy_hash", "betting_rule_2026_v1_hash", "first_eligible_wager_identity",
        "lock_timestamp_utc", "git_commit",
    }
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert not (repo_root / "config" / "executable_books_2026.json").exists()


# --- item 9: season-final confirmatory rule -------------------------
def test_season_final_confirmatory_helper_semantics():
    sched = {"available": True, "scheduled_reg_post_game_ids": ["A", "B", "C"],
             "unresolved_reg_post_game_ids": []}
    assert ps.season_final_confirmatory(sched, {"A", "B", "C"}) is True
    assert ps.season_final_confirmatory(sched, {"A", "B"}) is False            # a game still lacks a result
    assert ps.season_final_confirmatory({**sched, "unresolved_reg_post_game_ids": ["C"]}, {"A", "B", "C"}) is False
    assert ps.season_final_confirmatory({**sched, "available": False}, {"A", "B", "C"}) is False
    assert ps.season_final_confirmatory(None, {"A", "B", "C"}) is False
    assert ps.season_final_confirmatory({"available": True, "scheduled_reg_post_game_ids": []}, set()) is False


def test_scorecard_does_not_infer_season_final_from_sample_count_or_calendar():
    recs = []
    for i in range(210):
        recs += [_mk_record(f"SF{i:04d}", horizon="TUE", market="ATS"),
                 _mk_record(f"SF{i:04d}", horizon="FRI", market="TOTAL")]
    sc = ps.build_scorecard(evaluation_records=recs, data_through_utc="2027-03-01T00:00:00+00:00")
    assert sc["sample"]["maturity"] == ps.MATURITY_PROMOTION_ELIGIBLE
    assert sc["sample"]["season_final_confirmatory"] is None
    assert sc["sample"]["season_final_confirmatory_proven"] is False


def test_scorecard_reports_season_final_only_with_a_proven_complete_schedule():
    recs, ids = [], []
    for i in range(205):
        gid = f"SFC{i:04d}"
        ids.append(gid)
        recs.append(_mk_record(gid, horizon="TUE", market="ATS"))
    sc = ps.build_scorecard(
        evaluation_records=recs,
        schedule_population={"available": True, "scheduled_reg_post_game_ids": ids,
                             "unresolved_reg_post_game_ids": []},
        data_through_utc="2027-03-01T00:00:00+00:00",
    )
    assert sc["sample"]["season_final_confirmatory"] == "SEASON_FINAL_CONFIRMATORY"
    assert sc["sample"]["season_final_confirmatory_proven"] is True


def test_season_final_confirmatory_rule_frozen_in_prereg():
    rule = _PREREG["sample_maturity_firewall"]["season_final_confirmatory_rule"]
    assert rule["requires_canonical_2026_reg_post_schedule_population_available"] is True
    assert rule["never_inferred_from_calendar_date_or_sample_count_alone"] is True
    assert rule["false_when_schedule_completeness_cannot_be_proven"] is True


# --- item 10: sportsbook sign / event orientation --------------------
def test_home_spread_minus_3_5_implies_sportsbook_home_margin_plus_3_5():
    # actual home margin +10; a -3.5 home spread => book-implied home margin +3.5.
    rec = _mk_record("SPREAD", market="ATS", consensus_line=-3.5, home_score=27, away_score=17,
                     predicted_margin=8.0)
    rows = ps._stream_rows([rec], "ATS", None)
    actual_margin = 27 - 17
    implied_home_margin = 3.5
    assert rows["margin_sq_market"][0] == pytest.approx((implied_home_margin - actual_margin) ** 2)
    # home covers a -3.5 line at +10 -> ATS upper event realised
    assert rows["y"][0] == 1


def test_ats_model_and_novig_probability_share_one_event_side():
    recs = [
        _mk_record(f"ATSO{i:04d}", market="ATS", consensus_line=-3.5, home_score=30, away_score=10,
                   raw_p=0.80, calibrated_p=None, consensus_novig_probability=0.78)
        for i in range(40)
    ]
    rows = ps._stream_rows(recs, "ATS", None)
    assert set(rows["y"].tolist()) == {1}  # home always covers
    ll_model = ps.binary_log_loss(rows["y"], rows["p_raw"])
    ll_novig = ps.binary_log_loss(rows["y"], rows["p_market"])
    ll_novig_flipped = ps.binary_log_loss(rows["y"], 1.0 - rows["p_market"])
    assert ll_model < ps.LN2 and ll_novig < ps.LN2           # same orientation, both informative
    assert ll_novig_flipped > ps.LN2                         # a sign flip is unambiguously worse


def test_total_model_and_novig_probability_share_one_over_under_orientation():
    recs = [
        _mk_record(f"TOTO{i:04d}", market="TOTAL", consensus_line=40.0, home_score=30, away_score=24,
                   raw_p=0.80, calibrated_p=None, consensus_novig_probability=0.77)
        for i in range(40)
    ]
    rows = ps._stream_rows(recs, "TOTAL", None)
    assert set(rows["y"].tolist()) == {1}  # total 54 > 40 -> OVER
    ll_model = ps.binary_log_loss(rows["y"], rows["p_raw"])
    ll_novig = ps.binary_log_loss(rows["y"], rows["p_market"])
    assert ll_model < ps.LN2 and ll_novig < ps.LN2
    assert ps.binary_log_loss(rows["y"], 1.0 - rows["p_market"]) > ps.LN2


def test_no_sign_flip_can_make_the_point_market_comparison_favorable():
    good = _mk_record("PT", market="ATS", consensus_line=-3.5, home_score=30, away_score=10,
                      predicted_margin=20.0)   # model exactly right; book far off
    flipped = _mk_record("PTF", market="ATS", consensus_line=-3.5, home_score=30, away_score=10,
                         predicted_margin=-20.0)  # same magnitude, wrong sign
    g = ps._stream_rows([good], "ATS", None)
    f = ps._stream_rows([flipped], "ATS", None)
    assert g["margin_sq_model"][0] < g["margin_sq_market"][0]     # correct model beats the book
    assert f["margin_sq_model"][0] > f["margin_sq_market"][0]     # flipping the sign only makes it worse


def test_sportsbook_sign_orientation_is_frozen_in_prereg():
    so = _PREREG["sportsbook_sign_orientation"]
    assert so["home_spread_minus_3_5_implies_sportsbook_home_margin_plus_3_5"] is True
    assert so["ats_model_probability_and_ats_novig_probability_same_event_side"] is True
    assert so["total_model_probability_and_total_novig_probability_same_over_under_orientation"] is True
    assert so["no_sign_flip_can_make_the_point_market_comparison_appear_favorable"] is True


# --- item 11: preregistration hash recompute ------------------------
def test_preregistration_hash_changed_and_is_self_consistent():
    # Negative regression only: the pre-remediation frozen hash. The blocker
    # remediation (forecast-time market_state_hash persistence, calibrated-only
    # scoring, permanent betting-integrity contamination) intentionally moved
    # the self-hash off this value.
    old = "e592fb86d1d56c3e6fe27ef9003c538c2610ea1c7da78d937c75718c6f05c1f8"
    new = ps.preregistration_hash()
    assert new != old
    assert _PREREG[ps._HASH_FIELD] == new
    doc = ps.frozen_preregistration_document()
    assert doc[ps._HASH_FIELD] == ps.preregistration_hash(doc)


# ===========================================================================
# BLOCKER REMEDIATION 1 -- forecast-time market_state_hash persistence +
# exact recompute verification (no derive-and-accept).
# ===========================================================================
def _full_market_dict(*, consensus_line=-2.5, novig=0.52, books=("book_a", "book_b", "book_c"),
                      snaps=("2026-09-08T15:40:00+00:00",)):
    return {
        "eligible_books": len(books),
        "consensus_line": consensus_line,
        "consensus_novig_probability": novig,
        "bookmaker_keys": list(books),
        "selected_returned_snapshot_timestamps": list(snaps),
        "min_observation_age_hours": 1.0,
        "max_observation_age_hours": 6.0,
        "consensus_method": "median",
    }


def _persisted_market_rec(game_id="MS", *, market="ATS", **market_kw):
    """An evaluation-ledger-shaped record carrying a full consensus market
    dict AND the explicit forecast-time market_state_hash persisted top
    level (exactly as run_2026 now writes it)."""
    rec = {
        "game_id": game_id, "horizon": "TUE", "season": 2026, "season_type": "REG",
        "target_cutoff_utc": _CUTOFF_BY_HORIZON["TUE"],
        "created_at_utc": _CREATED_BY_HORIZON["TUE"],
        "git_commit": "deadbeef" * 5, "certified_baseline_sha": CERT_SHA,
        "content_hash": f"hash-{game_id}",
        "forecast": {"predicted_margin": 3.0, "predicted_total": 44.0},
        "markets": {market: {
            "status": "OK",
            "raw_conditional_upper_probability": 0.60,
            "calibrated_conditional_upper_probability": 0.58,
            "calibration_status": "CALIBRATED",
            "market": _full_market_dict(**market_kw),
        }},
        "provenance": {
            "git_commit": "deadbeef" * 5,
            "operational_model_spec_hash": "op" * 20,
            "horizon_feature_semantics_hash": "fs" * 20,
            "created_at_utc": _CREATED_BY_HORIZON["TUE"],
            "run_id": "run-0001",
        },
        "result_record": {"result": {"home_score": 21, "away_score": 17}},
    }
    rec["market_state_hash"] = ps.compute_market_state_hash(rec)
    return rec


def test_market_state_payload_shape_and_exclusions_are_frozen():
    assert ps.MARKET_STATE_PAYLOAD_MARKETS == ("ATS", "TOTAL")
    rec = _persisted_market_rec("SHAPE")
    payload = ps.build_market_state_payload(rec)
    assert set(payload) == {"ATS", "TOTAL"}
    assert payload["TOTAL"] is None
    assert set(payload["ATS"]) == set(ps.MARKET_STATE_PAYLOAD_MARKET_DICT_FIELDS)
    for volatile in ("created_at_utc", "run_id", "git_commit"):
        assert volatile not in payload["ATS"]


def test_recompute_agrees_with_the_persisted_forecast_time_hash():
    rec = _persisted_market_rec("AGREE")
    assert ps.compute_market_state_hash(rec) == rec["market_state_hash"]
    split = ps.classify_evidence_eligibility([rec])
    assert [r["game_id"] for r in split["eligible"]] == ["AGREE"]


def test_changing_consensus_line_triggers_hash_mismatch():
    rec = _persisted_market_rec("LINE")
    rec["markets"]["ATS"]["market"]["consensus_line"] = -1.5
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "HASH_MISMATCH"


def test_changing_bookmaker_keys_triggers_hash_mismatch():
    rec = _persisted_market_rec("BOOKS")
    rec["markets"]["ATS"]["market"]["bookmaker_keys"] = ["book_a", "book_z"]
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "HASH_MISMATCH"


def test_changing_selected_snapshot_timestamps_triggers_hash_mismatch():
    rec = _persisted_market_rec("SNAPS")
    rec["markets"]["ATS"]["market"]["selected_returned_snapshot_timestamps"] = ["2026-09-08T15:55:00+00:00"]
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "HASH_MISMATCH"


def test_created_at_utc_and_run_id_do_not_alter_market_state_hash():
    a = _persisted_market_rec("VOLA")
    b = _persisted_market_rec("VOLA")
    b["created_at_utc"] = "2026-12-31T23:59:59+00:00"
    b["provenance"]["created_at_utc"] = "2026-12-31T23:59:59+00:00"
    b["provenance"]["run_id"] = "run-9999"
    assert ps.compute_market_state_hash(a) == ps.compute_market_state_hash(b)


def test_missing_explicit_market_state_hash_makes_market_relative_evidence_ineligible():
    rec = _persisted_market_rec("NOPE")
    rec.pop("market_state_hash")
    split = ps.classify_evidence_eligibility([rec])
    assert split["eligible"] == []
    assert split["excluded"][0]["reason"] == "MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE"


def test_market_state_provenance_forecast_time_persistence_frozen_in_prereg():
    msp = _PREREG["market_state_provenance"]
    assert msp["persisted_explicit_hash_required_at_forecast_time"] is True
    assert msp["forecast_ledger_json_path"] == "prediction.market_state_hash"
    assert msp["evaluation_ledger_json_path"] == "market_state_hash"
    assert msp["forecast_and_evaluation_ledger_hashes_must_agree"] is True
    assert msp["reporter_reconstructs_payload_from_immutable_record_and_requires_exact_equality"] is True
    assert msp["missing_explicit_hash_makes_market_relative_evidence_ineligible"] is True
    assert msp["no_derive_and_accept_fallback"] is True
    assert set(msp["payload_markets"]) == {"ATS", "TOTAL"}
    assert "created_at_utc" in msp["payload_excludes"] and "run_id" in msp["payload_excludes"]
    assert not hasattr(ps, "_derive_market_snapshot_hash")


# ===========================================================================
# BLOCKER REMEDIATION 2 -- calibrated gates score CALIBRATED rows only; no
# raw-probability fallback.
# ===========================================================================
def test_calibration_not_ready_excellent_raw_cannot_enter_absolute_quality():
    good = _many(220, horizon="TUE", market="ATS", raw_p=0.55, calibrated_p=0.55,
                 home_score=24, away_score=10, consensus_line=0.0)
    only_good = ps.gate_absolute_probability_quality(good, ps.MATURITY_PROMOTION_ELIGIBLE, 220)
    # inject CALIBRATION_NOT_READY rows with an artificially perfect raw prob
    poisoned = good + [
        _mk_record(f"NR{i:04d}", horizon="FRI", market="ATS",
                   calibration_status="CALIBRATION_NOT_READY", raw_p=1.0, calibrated_p=None,
                   home_score=24, away_score=10, consensus_line=0.0)
        for i in range(220)
    ]
    with_poison = ps.gate_absolute_probability_quality(poisoned, ps.MATURITY_PROMOTION_ELIGIBLE, 440)
    # identical calibrated arithmetic -- the not-ready rows contributed nothing
    assert with_poison["point_estimate"]["n_calibrated_scored_rows"] == \
        only_good["point_estimate"]["n_calibrated_scored_rows"] == 220
    assert with_poison["point_estimate"]["pooled_log_loss"] == only_good["point_estimate"]["pooled_log_loss"]
    assert with_poison["point_estimate"]["pooled_ece"] == only_good["point_estimate"]["pooled_ece"]
    assert with_poison["point_estimate"]["auc"] == only_good["point_estimate"]["auc"]
    assert with_poison["point_estimate"]["per_market_rows_without_calibrated_probability"]["ATS"] == 220


def test_calibration_not_ready_excellent_raw_cannot_enter_sportsbook_probability_edge():
    good = []
    for i in range(220):
        good += [
            _mk_record(f"E{i:04d}", horizon="TUE", market="ATS", raw_p=0.6, calibrated_p=0.6,
                       consensus_novig_probability=0.5, home_score=24, away_score=10, consensus_line=0.0),
            _mk_record(f"E{i:04d}", horizon="FRI", market="TOTAL", raw_p=0.6, calibrated_p=0.6,
                       consensus_novig_probability=0.5, home_score=24, away_score=10, consensus_line=30.0),
        ]
    base = ps.gate_sportsbook_probability_edge(good, ps.MATURITY_PROMOTION_ELIGIBLE, 220)
    poisoned = good + [
        _mk_record(f"NR{i:04d}", horizon="TUE", market="ATS",
                   calibration_status="CALIBRATION_NOT_READY", raw_p=1.0, calibrated_p=None,
                   consensus_novig_probability=0.5, home_score=24, away_score=10, consensus_line=0.0)
        for i in range(220)
    ]
    with_poison = ps.gate_sportsbook_probability_edge(poisoned, ps.MATURITY_PROMOTION_ELIGIBLE, 440)
    assert with_poison["per_market"]["ATS"]["n_calibrated_scored_rows"] == \
        base["per_market"]["ATS"]["n_calibrated_scored_rows"]
    assert with_poison["per_market"]["ATS"]["log_loss_delta"] == base["per_market"]["ATS"]["log_loss_delta"]
    assert with_poison["per_market"]["ATS"]["brier_delta"] == base["per_market"]["ATS"]["brier_delta"]
    assert with_poison["per_market"]["ATS"]["rows_with_market_but_no_calibrated_probability"] == 220
    # the CALIBRATION_NOT_READY rows with a perfect raw prob changed nothing
    assert with_poison["status"] == base["status"]
    assert with_poison["per_market"]["ATS"]["gate_booleans"] == base["per_market"]["ATS"]["gate_booleans"]


def test_calibrated_gates_scored_rows_reported_separately_from_maturity():
    # 210 completed games but only ATS calibrated -> TOTAL calibrated sample is 0
    recs = _many(210, horizon="TUE", market="ATS", raw_p=0.55, calibrated_p=0.55)
    gate = ps.gate_sportsbook_probability_edge(recs, ps.MATURITY_PROMOTION_ELIGIBLE, 210)
    assert gate["per_market"]["TOTAL"]["n_calibrated_scored_rows"] == 0
    assert gate["status"] != "MET_STRONGLY"


def test_calibrated_only_scoring_frozen_in_prereg():
    cm = _PREREG["calibration_metrics"]
    assert cm["calibrated_gates_score_calibrated_rows_only"] is True
    assert cm["no_raw_probability_fallback_in_calibrated_or_probability_gates"] is True
    assert cm["absolute_probability_quality_scores_calibrated_rows_only"] is True
    assert cm["sportsbook_probability_edge_scores_calibrated_rows_only"] is True
    assert cm["calibrated_metric_sample_size_reported_separately_from_game_maturity"] is True


# ===========================================================================
# BLOCKER REMEDIATION 3 -- durable, append-only, permanent betting-policy
# integrity contamination.
# ===========================================================================
def test_conflicting_policy_writes_integrity_event_then_hard_stops(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1")
    with pytest.raises(ps.BettingIntegrityConflict):
        ps.write_executable_book_policy_lock(
            tmp_path, policy_hash="policy_b", first_eligible_wager_identity=_WAGER_ID, git_commit="sha2")
    events = ps.load_betting_integrity_events(tmp_path)
    assert len(events) == 1
    assert events[0]["event_type"] == ps.CONFLICT_POLICY_HASH_MISMATCH
    assert events[0]["locked_policy_hash"] == "policy_a"
    assert events[0]["attempted_policy_hash"] == "policy_b"
    for field in ("event_type", "observed_at_utc", "locked_policy_hash", "attempted_policy_hash",
                  "locked_betting_rule_hash", "attempted_betting_rule_hash",
                  "first_eligible_wager_identity", "attempted_wager_identity", "git_commit",
                  "event_content_hash"):
        assert field in events[0]
    status = ps.betting_integrity_estate_status(tmp_path)
    assert status["contaminated_estate"] is True
    assert status["conflict_event_count"] == 1


def test_repeated_identical_conflict_is_idempotent(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1")
    for _ in range(3):
        with pytest.raises(ps.BettingIntegrityConflict):
            ps.write_executable_book_policy_lock(
                tmp_path, policy_hash="policy_b", first_eligible_wager_identity=_WAGER_ID, git_commit="sha9")
    assert len(ps.load_betting_integrity_events(tmp_path)) == 1


def test_reverting_to_original_policy_does_not_clear_contamination(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1")
    with pytest.raises(ps.BettingIntegrityConflict):
        ps.write_executable_book_policy_lock(
            tmp_path, policy_hash="policy_b", first_eligible_wager_identity=_WAGER_ID, git_commit="sha2")
    # a later call reusing the ORIGINAL policy hash is an idempotent no-op on
    # the lock, but the persisted conflict event still contaminates the estate
    noop = ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha3")
    assert noop["status"] == "IDEMPOTENT_NOOP"
    assert ps.betting_integrity_estate_status(tmp_path)["contaminated_estate"] is True


def test_different_betting_rule_hash_also_records_and_contaminates(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1",
        betting_rule_hash="RULE_HASH_1")
    with pytest.raises(ps.BettingIntegrityConflict):
        ps.write_executable_book_policy_lock(
            tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha2",
            betting_rule_hash="RULE_HASH_2")
    events = ps.load_betting_integrity_events(tmp_path)
    assert [e["event_type"] for e in events] == [ps.CONFLICT_BETTING_RULE_HASH_MISMATCH]
    assert ps.betting_integrity_estate_status(tmp_path)["contaminated_estate"] is True


def test_reporter_sees_persisted_conflict_across_a_fresh_reload(tmp_path):
    ps.write_executable_book_policy_lock(
        tmp_path, policy_hash="policy_a", first_eligible_wager_identity=_WAGER_ID, git_commit="sha1")
    with pytest.raises(ps.BettingIntegrityConflict):
        ps.write_executable_book_policy_lock(
            tmp_path, policy_hash="policy_b", first_eligible_wager_identity=_WAGER_ID, git_commit="sha2")
    # fresh read from disk -- nothing carried in memory
    events = ps.load_betting_integrity_events(tmp_path)
    sc = ps.build_scorecard(integrity_events=events)
    prof = sc["rows"]["PROVEN_PROFITABLE_BETTING_MODEL"]
    assert sc["betting_integrity_estate"]["contaminated_estate"] is True
    assert prof["betting_integrity_estate_contaminated"] is True
    assert prof["gate_booleans"]["no_persisted_betting_integrity_conflict_event"] is False
    assert "BETTING_INTEGRITY_CONFLICT" in prof["failure_reasons"]


def test_profitability_met_strongly_permanently_barred_for_contaminated_estate():
    events = [{
        "schema": ps.BETTING_POLICY_INTEGRITY_EVENT_SCHEMA,
        "event_type": ps.CONFLICT_POLICY_HASH_MISMATCH,
        "locked_policy_hash": "policy_a", "attempted_policy_hash": "policy_b",
        "locked_betting_rule_hash": ps.BETTING_RULE_2026_V1_HASH,
        "attempted_betting_rule_hash": ps.BETTING_RULE_2026_V1_HASH,
        "first_eligible_wager_identity": dict(_WAGER_ID), "attempted_wager_identity": dict(_WAGER_ID),
    }]
    events[0]["event_content_hash"] = ps._sha256_hex({
        k: events[0][k] for k in (
            "schema", "event_type", "locked_policy_hash", "attempted_policy_hash",
            "locked_betting_rule_hash", "attempted_betting_rule_hash",
            "first_eligible_wager_identity", "attempted_wager_identity")
    })
    gate = ps.gate_profitability(
        [], {"frozen": True, "policy_hash": "policy_a"}, ps.MATURITY_PROMOTION_ELIGIBLE, 300, 200,
        policy_lock={"locked": True, "conflict": False, "flag": None},
        contaminated_estate=ps.betting_integrity_estate_status(events=events)["contaminated_estate"],
    )
    assert gate["status"] == "NOT_ESTABLISHED"
    assert gate["betting_integrity_estate_contaminated"] is True
    assert gate["contaminated_estate_met_strongly_barred"] is True
    assert gate["gate_booleans"]["no_betting_integrity_conflict"] is False


def test_hand_edited_integrity_event_cannot_launder_or_fabricate_contamination():
    # a fabricated event whose content hash does not verify is ignored
    bogus = [{
        "event_type": ps.CONFLICT_POLICY_HASH_MISMATCH, "locked_policy_hash": "x",
        "attempted_policy_hash": "y", "event_content_hash": "0" * 64,
    }]
    assert ps.betting_integrity_estate_status(events=bogus)["contaminated_estate"] is False


def test_integrity_event_ledger_frozen_in_prereg():
    lock = _PREREG["executable_book_policy_lock"]
    assert lock["integrity_event_ledger_path"].endswith("betting-policy/integrity-events/")
    assert lock["integrity_event_schema"] == "BETTING_POLICY_INTEGRITY_EVENT_V1"
    assert set(lock["conflict_types"]) == {"POLICY_HASH_MISMATCH", "BETTING_RULE_HASH_MISMATCH"}
    assert lock["conflict_durably_recorded_before_hard_stop"] is True
    assert lock["any_persisted_conflict_event_permanently_contaminates_2026_estate"] is True
    assert lock["no_automatic_clear_contamination_path"] is True
    assert lock["reverting_config_or_restoring_original_hash_does_not_clear_contamination"] is True
    assert _PREREG["gates"]["PROVEN_PROFITABLE_BETTING_MODEL"][
        "any_persisted_betting_integrity_conflict_permanently_bars_met_strongly"] is True
