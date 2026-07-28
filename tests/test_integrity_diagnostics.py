from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.integrity import (
    SCHEMA_CONTRACTS,
    audit_compact_targets,
)


def test_exact_schema_contract_counts():
    assert SCHEMA_CONTRACTS["pregame_moneyline"].football_feature_count == 33
    assert SCHEMA_CONTRACTS["pregame_moneyline"].augmented_feature_count == 42
    assert SCHEMA_CONTRACTS["pregame_ats"].football_feature_count == 31
    assert SCHEMA_CONTRACTS["pregame_ats"].augmented_feature_count == 44
    assert SCHEMA_CONTRACTS["pregame_total"].football_feature_count == 36
    assert SCHEMA_CONTRACTS["pregame_total"].augmented_feature_count == 45


def _base_ids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2021, 2021, 2021],
            "home_team_id": ["KC", "BUF", "DAL"],
            "away_team_id": ["BUF", "KC", "PHI"],
        }
    )


def _valid_frames() -> dict[str, pd.DataFrame]:
    ids = _base_ids()

    moneyline = ids.copy()
    moneyline["target_home_margin"] = [3.0, -4.0, 0.0]
    moneyline["target_home_win"] = [1.0, 0.0, np.nan]
    moneyline["target_tie"] = [0.0, 0.0, 1.0]
    moneyline["market_home_ml_novig_prob"] = [0.6, 0.4, 0.5]

    ats = ids.copy()
    ats["target_home_margin"] = [3.0, -4.0, 3.0]
    ats["market_home_spread"] = [-2.5, 3.5, -3.0]
    ats["market_implied_margin"] = -ats["market_home_spread"]
    ats["target_margin_residual"] = (
        ats["target_home_margin"] + ats["market_home_spread"]
    )
    residual = ats["target_margin_residual"]
    ats["target_home_cover"] = np.where(
        residual > 0,
        1.0,
        np.where(residual < 0, 0.0, np.nan),
    )
    ats["target_ats_push"] = (residual == 0).astype(float)
    ats["market_home_cover_novig_prob"] = [0.51, 0.49, 0.5]

    total = ids.copy()
    total["target_total_points"] = [47.0, 40.0, 44.0]
    total["market_total_line"] = [45.5, 42.5, 44.0]
    total["target_total_residual"] = (
        total["target_total_points"] - total["market_total_line"]
    )
    residual = total["target_total_residual"]
    total["target_over"] = np.where(
        residual > 0,
        1.0,
        np.where(residual < 0, 0.0, np.nan),
    )
    total["target_total_push"] = (residual == 0).astype(float)
    total["market_over_novig_prob"] = [0.52, 0.48, 0.5]

    return {
        "pregame_moneyline_market_augmented.parquet": moneyline,
        "pregame_ats_market_augmented.parquet": ats,
        "pregame_total_market_augmented.parquet": total,
    }


def _patch_parquet(monkeypatch, frames):
    def fake_read_parquet(path, *args, **kwargs):
        name = Path(path).name
        return frames[name].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def test_target_audit_passes_valid_formulas(tmp_path, monkeypatch):
    frames = _valid_frames()
    _patch_parquet(monkeypatch, frames)
    summary, failures = audit_compact_targets(
        tmp_path / "compact",
        tmp_path / "output",
    )
    assert summary["mismatches"].fillna(0).sum() == 0
    assert failures.empty


def test_target_audit_detects_ats_sign_error(tmp_path, monkeypatch):
    frames = _valid_frames()
    ats = frames["pregame_ats_market_augmented.parquet"]
    ats.loc[0, "target_margin_residual"] = (
        ats.loc[0, "target_home_margin"]
        - ats.loc[0, "market_home_spread"]
    )
    _patch_parquet(monkeypatch, frames)

    with pytest.raises(ValueError, match="Target/source audit failed"):
        audit_compact_targets(
            tmp_path / "compact",
            tmp_path / "output",
        )
