"""Fix 4 repair: tests for the real ATS/TOTAL market attachment
(:mod:`nfl_hybrid.evaluation.chronological_market_attachment`).

Covers: the item 20 MONEYLINE-only regression guard, the item 4 spread-sign
convention (proven against hand-built known values, never against the
canonical function's own internal math), and the item 15 explicit per-reason
coverage/rejection accounting -- built on a small synthetic canonical-market
parquet (never the real private data estate; that's exercised separately by
the extended real-data proof).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.evaluation.chronological_calibration import MARKET_ATS, MARKET_MONEYLINE, MARKET_TOTAL
from nfl_hybrid.evaluation.chronological_market_attachment import (
    FORECAST_HORIZON,
    MAXIMUM_SNAPSHOT_LAG_MINUTES,
    MINIMUM_ELIGIBLE_BOOKS,
    assert_primary_markets_certified,
    attach_market_to_residual_ledger,
    build_market_raw_ledger,
    load_canonical_t10_market,
)
from nfl_hybrid.evaluation.chronological_oof import UNCERTAINTY_LEDGER_COLUMNS

# ---------------------------------------------------------------------------
# Item 20: the exact regression guard for the original real-data-proof drift.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "primary_markets",
    [
        [MARKET_MONEYLINE],
        [MARKET_ATS],
        [MARKET_TOTAL],
        [MARKET_MONEYLINE, MARKET_ATS],
        [MARKET_MONEYLINE, MARKET_TOTAL],
        [],
    ],
)
def test_certification_is_rejected_without_both_ats_and_total(primary_markets):
    with pytest.raises(ValueError):
        assert_primary_markets_certified(primary_markets)


@pytest.mark.parametrize(
    "primary_markets",
    [
        [MARKET_ATS, MARKET_TOTAL],
        [MARKET_TOTAL, MARKET_ATS],
        [MARKET_ATS, MARKET_TOTAL, MARKET_MONEYLINE],
    ],
)
def test_certification_accepts_ats_and_total_together(primary_markets):
    assert_primary_markets_certified(primary_markets)  # must not raise


# ---------------------------------------------------------------------------
# Fixtures: a tiny synthetic canonical closing_t10 market matrix, laid out at
# exactly the registered relative path so external_data.resolve(...,
# root_override=tmp_path) finds it -- never the real private data estate.
# ---------------------------------------------------------------------------

_ATS_REL_PATH = "feature-store-2020-2025/compact-canonical-2020-2023/pregame_ats_market_augmented_canonical_t10.parquet"
_TOTAL_REL_PATH = "feature-store-2020-2025/compact-canonical-2020-2023/pregame_total_market_augmented_canonical_t10.parquet"


def _write_market_parquet(root: Path, relative_path: str, frame: pd.DataFrame) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _synthetic_ats_market(root: Path, *, n: int = 5) -> pd.DataFrame:
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "game_id": [f"G{i:03d}" for i in range(n)],
            "season": 2023,
            "week": 5,
            "market_t10_consensus_line": [-3.5, -3.0, 0.0, 3.0, 7.0][:n],
            "market_t10_eligible_books": [5, 5, 5, 5, 5][:n],
            "market_t10_requested_snapshot_utc": [cutoff + pd.Timedelta(minutes=10)] * n,
            "market_t10_returned_snapshot_utc": [cutoff - pd.Timedelta(minutes=5)] * n,
            "market_t10_snapshot_lag_minutes": [3.0] * n,
        }
    )
    _write_market_parquet(root, _ATS_REL_PATH, frame)
    return frame


def _residual_ledger_row(game_id: str, target_cutoff: pd.Timestamp, *, actual_margin: float) -> dict:
    return {
        "game_id": game_id,
        "season": 2023,
        "week": 5,
        "target_cutoff_utc": target_cutoff,
        "training_as_of_utc": target_cutoff,
        "result_available_at_utc": target_cutoff + pd.Timedelta(hours=4),
        "max_training_result_available_at_utc": target_cutoff - pd.Timedelta(days=1),
        "result_availability_basis": "HISTORICAL_CONSERVATIVE_BATCH",
        "training_game_count": 60,
        "training_game_ids": (),
        "training_game_ids_truncated": False,
        "training_membership_hash": "h",
        "model_config_hash": "mc1",
        "feature_state_hash": "fs1",
        "status": "OOF",
        "predicted_margin": 1.0,
        "predicted_total": 44.0,
        "actual_margin": actual_margin,
        "actual_total": 44.0,
        "margin_residual": actual_margin - 1.0,
        "total_residual": 0.0,
        "margin_residual_sd_oof": 7.0,
        "total_residual_sd_oof": 7.0,
        "residual_correlation_oof": 0.0,
        "uncertainty_eligible": True,
    }


# ---------------------------------------------------------------------------
# Item 4: spread-sign convention, proven against hand-built known values.
# ---------------------------------------------------------------------------


def test_negative_home_spread_means_home_is_favored_and_grading_matches(tmp_path):
    """-3.5 (row 0 of the synthetic fixture) means the home team is favored
    by 3.5. A home team that wins by only 2 (actual_margin=2) does NOT cover
    -3.5; a home team that wins by 7 does."""
    _synthetic_ats_market(tmp_path, n=1)
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")

    # Home wins by 2: actual_margin + home_spread = 2 + (-3.5) = -1.5 < 0 -> away covers.
    ledger_loses = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=2.0)])
    raw_loses, coverage = build_market_raw_ledger(ledger_loses, market=MARKET_ATS, root_override=str(tmp_path))
    assert coverage["qualifies_valid_t10_line"] == 1
    assert raw_loses.loc[0, "binary_target"] == 0  # away (line) covers -- home did not cover
    assert raw_loses.loc[0, "home_spread"] == -3.5

    # Home wins by 7: actual_margin + home_spread = 7 + (-3.5) = 3.5 > 0 -> home covers.
    ledger_covers = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=7.0)])
    raw_covers, _ = build_market_raw_ledger(ledger_covers, market=MARKET_ATS, root_override=str(tmp_path))
    assert raw_covers.loc[0, "binary_target"] == 1


def test_positive_home_spread_means_home_is_underdog(tmp_path):
    """+3.0 (row 3) means the home team is an underdog getting 3 points. A
    home team that loses by exactly 2 still covers (2-point loss + 3 = +1 >
    0); a home team that loses by 5 does not (-5 + 3 = -2 < 0)."""
    frame = pd.DataFrame(
        {
            "game_id": ["G000"],
            "season": 2023,
            "week": 5,
            "market_t10_consensus_line": [3.0],
            "market_t10_eligible_books": [5],
            "market_t10_requested_snapshot_utc": [pd.Timestamp("2023-10-01", tz="UTC") + pd.Timedelta(minutes=10)],
            "market_t10_returned_snapshot_utc": [pd.Timestamp("2023-10-01", tz="UTC") - pd.Timedelta(minutes=5)],
            "market_t10_snapshot_lag_minutes": [3.0],
        }
    )
    _write_market_parquet(tmp_path, _ATS_REL_PATH, frame)
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")

    ledger = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=-2.0)])  # home loses by 2
    raw, _ = build_market_raw_ledger(ledger, market=MARKET_ATS, root_override=str(tmp_path))
    assert raw.loc[0, "binary_target"] == 1  # home (getting 3) still covers

    ledger_loses_big = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=-5.0)])
    raw_loses, _ = build_market_raw_ledger(ledger_loses_big, market=MARKET_ATS, root_override=str(tmp_path))
    assert raw_loses.loc[0, "binary_target"] == 0


# ---------------------------------------------------------------------------
# Item 15: explicit, counted rejection reasons -- nothing silently dropped.
# ---------------------------------------------------------------------------


def test_coverage_report_counts_every_rejection_reason(tmp_path):
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "game_id": ["OK", "TOO_FEW_BOOKS", "STALE_LAG", "AFTER_CUTOFF"],
            "season": 2023,
            "week": 5,
            "market_t10_consensus_line": [-3.0, -3.0, -3.0, -3.0],
            "market_t10_eligible_books": [5, 2, 5, 5],
            "market_t10_requested_snapshot_utc": [cutoff + pd.Timedelta(minutes=10)] * 4,
            "market_t10_returned_snapshot_utc": [
                cutoff - pd.Timedelta(minutes=5),
                cutoff - pd.Timedelta(minutes=5),
                cutoff - pd.Timedelta(minutes=5),
                cutoff + pd.Timedelta(minutes=1),  # AFTER the target cutoff
            ],
            "market_t10_snapshot_lag_minutes": [3.0, 3.0, 9.0, 3.0],  # STALE_LAG exceeds the 5-minute cap
        }
    )
    _write_market_parquet(tmp_path, _ATS_REL_PATH, frame)

    ledger = pd.DataFrame(
        [
            _residual_ledger_row("OK", cutoff, actual_margin=1.0),
            _residual_ledger_row("TOO_FEW_BOOKS", cutoff, actual_margin=1.0),
            _residual_ledger_row("STALE_LAG", cutoff, actual_margin=1.0),
            _residual_ledger_row("AFTER_CUTOFF", cutoff, actual_margin=1.0),
            _residual_ledger_row("NO_MARKET_ROW", cutoff, actual_margin=1.0),
        ]
    )

    result = attach_market_to_residual_ledger(ledger, market=MARKET_ATS, root_override=str(tmp_path))
    assert result.coverage["n_examined"] == 5
    assert result.coverage["rejected_no_market_match"] == 1
    assert result.coverage["rejected_too_few_books"] == 1
    assert result.coverage["rejected_stale_or_invalid_snapshot_lag"] == 1
    assert result.coverage["rejected_snapshot_after_target_cutoff"] == 1
    assert result.coverage["qualifies_valid_t10_line"] == 1
    assert list(result.attached["game_id"]) == ["OK"]


def test_attached_rows_carry_forecast_horizon_and_provenance(tmp_path):
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")
    _synthetic_ats_market(tmp_path, n=1)
    ledger = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=1.0)])
    raw, coverage = build_market_raw_ledger(ledger, market=MARKET_ATS, root_override=str(tmp_path))
    assert coverage["qualifies_valid_t10_line"] == 1
    assert raw.loc[0, "forecast_horizon"] == FORECAST_HORIZON
    assert raw.loc[0, "eligible_books"] == 5
    assert raw.loc[0, "consensus_method"]
    assert pd.isna(raw.loc[0, "market_last_update_utc"])  # not fabricated -- never present in this schema


def test_empty_market_source_produces_zero_qualifying_rows_not_an_error(tmp_path):
    frame = pd.DataFrame(
        {
            "game_id": pd.Series([], dtype=str),
            "season": pd.Series([], dtype=int),
            "week": pd.Series([], dtype=int),
            "market_t10_consensus_line": pd.Series([], dtype=float),
            "market_t10_eligible_books": pd.Series([], dtype=int),
            "market_t10_requested_snapshot_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
            "market_t10_returned_snapshot_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
            "market_t10_snapshot_lag_minutes": pd.Series([], dtype=float),
        }
    )
    _write_market_parquet(tmp_path, _ATS_REL_PATH, frame)
    cutoff = pd.Timestamp("2023-10-01", tz="UTC")
    ledger = pd.DataFrame([_residual_ledger_row("G000", cutoff, actual_margin=1.0)])
    raw, coverage = build_market_raw_ledger(ledger, market=MARKET_ATS, root_override=str(tmp_path))
    assert raw.empty
    assert coverage["qualifies_valid_t10_line"] == 0
    assert coverage["rejected_no_market_match"] == 1


# ---------------------------------------------------------------------------
# Basic loader sanity
# ---------------------------------------------------------------------------


def test_load_canonical_t10_market_rejects_unsupported_market(tmp_path):
    with pytest.raises(ValueError):
        load_canonical_t10_market(MARKET_MONEYLINE, root_override=str(tmp_path))


def test_load_canonical_t10_market_normalizes_expected_columns(tmp_path):
    _synthetic_ats_market(tmp_path, n=3)
    out = load_canonical_t10_market(MARKET_ATS, root_override=str(tmp_path))
    for col in ("game_id", "market", "line", "market_snapshot_returned_utc", "eligible_books", "consensus_method"):
        assert col in out.columns
    assert (out["market"] == MARKET_ATS).all()


def test_attach_market_rejects_moneyline():
    ledger = pd.DataFrame([_residual_ledger_row("G000", pd.Timestamp("2023-10-01", tz="UTC"), actual_margin=1.0)])
    with pytest.raises(ValueError):
        attach_market_to_residual_ledger(ledger, market=MARKET_MONEYLINE)


def test_residual_ledger_row_fixture_matches_uncertainty_ledger_columns():
    row = _residual_ledger_row("G000", pd.Timestamp("2023-10-01", tz="UTC"), actual_margin=1.0)
    assert set(UNCERTAINTY_LEDGER_COLUMNS) <= set(row.keys())
