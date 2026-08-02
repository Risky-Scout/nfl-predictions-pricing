"""``scripts/build_pricing_calibration.build_outcomes`` target encoding.

Drives the ACTUAL ``build_outcomes`` (loaded from the script file, never
re-implemented) over a synthetic ``games`` frame injected via a monkeypatched
``pd.read_parquet``. Proves the three market targets go through the canonical
``edge_to_nullable_binary`` helper: ties / ATS pushes / total pushes and any
missing or non-finite edge become ``pd.NA`` (Int8), never class zero, and one
market's null never erases a resolved target in another market.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.labels import edge_to_nullable_binary

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_pricing_calibration.py"


def _load_build_script():
    """Load the exact production script file without copying any of its logic."""
    spec = importlib.util.spec_from_file_location("build_pricing_calibration_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_script = _load_build_script()


# (label, home_score, away_score, home_spread_reference, total_line_reference,
#  expected home_win, expected home_cover, expected over). NA sentinel == pd.NA.
# All rows use season 2022 (in TEST_SEASONS) so none are filtered out.
NA = pd.NA
ROWS = [
    # label,                 hs_score, as_score, spread, total,     hw,  hc,  ov
    ("home_win",             24,       17,       -3.0,   38.0,      1,   1,   1),
    ("away_win",             17,       24,        3.0,   50.0,      0,   0,   0),
    # tie nulls ONLY moneyline; cover & total stay resolved (cross-market independence)
    ("game_tie",             20,       20,       -1.0,   39.0,      NA,  0,   1),
    ("home_ats_cover",       21,       20,        3.0,   39.0,      1,   1,   1),
    ("away_ats_cover",       21,       20,       -5.0,   39.0,      1,   0,   1),
    # ATS push nulls ONLY spread
    ("ats_push",             24,       21,       -3.0,   44.0,      1,   NA,  1),
    ("over_row",             30,       28,       -1.0,   50.0,      1,   1,   1),
    ("under_row",            10,        7,       -1.0,   40.0,      1,   1,   0),
    # total push nulls ONLY total
    ("total_push",           24,       18,       -3.0,   42.0,      1,   1,   NA),
    # missing scores feed every market -> all three null
    ("missing_home_score",   np.nan,   17,       -3.0,   38.0,      NA,  NA,  NA),
    ("missing_away_score",   24,       np.nan,   -3.0,   38.0,      NA,  NA,  NA),
    # missing spread nulls ONLY spread; moneyline & total resolved
    ("missing_spread",       24,       17,       np.nan, 38.0,      1,   NA,  1),
    # missing total line nulls ONLY total; moneyline & spread resolved
    ("missing_total_line",   24,       17,       -3.0,   np.nan,    1,   1,   NA),
    # non-finite edges (via +/-inf total line) -> total null; other markets resolved
    ("pos_inf_edge",         24,       17,       -3.0,   -np.inf,   1,   1,   NA),
    ("neg_inf_edge",         24,       17,       -3.0,    np.inf,   1,   1,   NA),
    # tolerance flows through build_outcomes exactly as in test_labels.py.
    # tp == 0 (0-0 scores) so ov == -total_line is exact (no float cancellation).
    ("pos_edge_inside_tol",  0,        0,        -1.0,   -0.5e-9,   NA,  0,   NA),
    ("neg_edge_inside_tol",  0,        0,        -1.0,    0.5e-9,   NA,  0,   NA),
    ("pos_edge_outside_tol", 0,        0,        -1.0,   -2.0e-9,   NA,  0,   1),
    ("neg_edge_outside_tol", 0,        0,        -1.0,    2.0e-9,   NA,  0,   0),
]


@pytest.fixture()
def outcomes(monkeypatch):
    games = pd.DataFrame(
        {
            "game_id": [f"g_{label}" for (label, *_rest) in ROWS],
            "season": 2022,
            "home_score": [r[1] for r in ROWS],
            "away_score": [r[2] for r in ROWS],
            "home_spread_reference": [r[3] for r in ROWS],
            "total_line_reference": [r[4] for r in ROWS],
        },
        index=[r[0] for r in ROWS],  # meaningful index we assert is preserved
    )
    monkeypatch.setattr(build_script.pd, "read_parquet", lambda *a, **k: games)
    return build_script.build_outcomes()


def _expected():
    return {r[0]: {"home_win": r[5], "home_cover": r[6], "over": r[7]} for r in ROWS}


def test_targets_are_nullable_int8(outcomes):
    for col in ("home_win", "home_cover", "over"):
        assert outcomes[col].dtype == "Int8", col


def test_source_index_is_preserved(outcomes):
    assert list(outcomes.index) == [r[0] for r in ROWS]


def test_game_id_column_passes_through(outcomes):
    assert list(outcomes["game_id"]) == [f"g_{r[0]}" for r in ROWS]


@pytest.mark.parametrize("label", [r[0] for r in ROWS])
def test_each_row_exact_targets(outcomes, label):
    exp = _expected()[label]
    row = outcomes.loc[label]
    for col in ("home_win", "home_cover", "over"):
        got, want = row[col], exp[col]
        if want is pd.NA:
            assert pd.isna(got), f"{label}.{col}: expected pd.NA, got {got!r}"
        else:
            assert not pd.isna(got), f"{label}.{col}: expected {want}, got NA"
            assert int(got) == want, f"{label}.{col}: expected {want}, got {got!r}"


def test_ties_and_pushes_are_na(outcomes):
    assert pd.isna(outcomes.loc["game_tie", "home_win"])       # moneyline tie
    assert pd.isna(outcomes.loc["ats_push", "home_cover"])     # ATS push
    assert pd.isna(outcomes.loc["total_push", "over"])         # total push


def test_nonfinite_edges_are_na(outcomes):
    assert pd.isna(outcomes.loc["pos_inf_edge", "over"])
    assert pd.isna(outcomes.loc["neg_inf_edge", "over"])


def test_one_market_null_does_not_erase_another(outcomes):
    # tie: moneyline null, spread+total resolved
    assert pd.isna(outcomes.loc["game_tie", "home_win"])
    assert not pd.isna(outcomes.loc["game_tie", "home_cover"])
    assert not pd.isna(outcomes.loc["game_tie", "over"])
    # missing spread: spread null, moneyline+total resolved
    assert pd.isna(outcomes.loc["missing_spread", "home_cover"])
    assert not pd.isna(outcomes.loc["missing_spread", "home_win"])
    assert not pd.isna(outcomes.loc["missing_spread", "over"])
    # total push: total null, moneyline+spread resolved
    assert pd.isna(outcomes.loc["total_push", "over"])
    assert not pd.isna(outcomes.loc["total_push", "home_win"])
    assert not pd.isna(outcomes.loc["total_push", "home_cover"])


def test_tolerance_matches_label_policy(outcomes):
    # inside +/-1e-9 -> null; outside -> resolved, exactly as edge_to_nullable_binary
    assert pd.isna(outcomes.loc["pos_edge_inside_tol", "over"])
    assert pd.isna(outcomes.loc["neg_edge_inside_tol", "over"])
    assert int(outcomes.loc["pos_edge_outside_tol", "over"]) == 1
    assert int(outcomes.loc["neg_edge_outside_tol", "over"]) == 0


def test_build_script_uses_canonical_function_object():
    # The script must import the ONE canonical helper, not a private re-implementation.
    assert build_script.edge_to_nullable_binary is edge_to_nullable_binary
