"""Canonical binary-label policy (``nfl_hybrid.labels.edge_to_nullable_binary``).

Exercises the SINGLE source-of-truth tie/push encoding directly, on a Series with
a non-default index. A zero edge (tie / push) and any non-finite / missing edge
must be ``pd.NA`` -- never class zero -- and the boundary at exactly ``+/-1e-9`` is
inclusive-null. Regression guard for the R1 tie/push mislabelling defect.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.labels import edge_to_nullable_binary

# (edge value, expected nullable label). NA sentinel is represented by pd.NA.
CASES = [
    (2.0, 1),
    (-2.0, 0),
    (0.0, pd.NA),
    (0.5e-9, pd.NA),
    (-0.5e-9, pd.NA),
    (1.0e-9, pd.NA),     # boundary: |edge| == tol -> null (inclusive)
    (-1.0e-9, pd.NA),    # boundary: |edge| == tol -> null (inclusive)
    (2.0e-9, 1),         # just outside tol -> resolved
    (-2.0e-9, 0),        # just outside tol -> resolved
    (np.nan, pd.NA),
    (pd.NA, pd.NA),
    (np.inf, pd.NA),
    (-np.inf, pd.NA),
]


def _series():
    """The full case vector as one Series with a deliberately non-default index."""
    values = [v for v, _ in CASES]
    index = list(range(100, 100 + 10 * len(CASES), 10))
    return pd.Series(values, index=index, dtype=object), index


def test_dtype_is_nullable_int8():
    s, _ = _series()
    out = edge_to_nullable_binary(s)
    assert out.dtype == "Int8"


def test_index_is_preserved_exactly():
    s, index = _series()
    out = edge_to_nullable_binary(s)
    assert list(out.index) == index


def test_exact_values_for_every_case():
    s, _ = _series()
    out = edge_to_nullable_binary(s)
    for pos, (edge, expected) in enumerate(CASES):
        got = out.iloc[pos]
        if expected is pd.NA:
            assert got is pd.NA or pd.isna(got), f"{edge!r} should be pd.NA, got {got!r}"
        else:
            assert not pd.isna(got), f"{edge!r} should be {expected}, got NA"
            assert int(got) == expected, f"{edge!r} should be {expected}, got {got!r}"


def test_boundary_values_at_plus_minus_1e9_are_null():
    out = edge_to_nullable_binary(pd.Series([1.0e-9, -1.0e-9], index=["a", "b"]))
    assert out.loc["a"] is pd.NA
    assert out.loc["b"] is pd.NA


def test_just_outside_tolerance_resolves():
    out = edge_to_nullable_binary(pd.Series([2.0e-9, -2.0e-9], index=["a", "b"]))
    assert int(out.loc["a"]) == 1
    assert int(out.loc["b"]) == 0


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf, pd.NA])
def test_nonfinite_and_missing_are_null(nonfinite):
    out = edge_to_nullable_binary(pd.Series([nonfinite], index=[7], dtype=object))
    assert out.loc[7] is pd.NA
    assert out.dtype == "Int8"
    assert list(out.index) == [7]
