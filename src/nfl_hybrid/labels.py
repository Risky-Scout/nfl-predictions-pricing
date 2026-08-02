"""Canonical binary-label policy (single source of truth).

Every production consumer that turns a numeric decision *edge* into a binary
classification target imports :func:`edge_to_nullable_binary` from here. There is
exactly one function body in the repository that implements this policy; do NOT
re-derive the greater-than / less-than tie/push handling anywhere else.

Ties and betting pushes are EXCLUDED binary outcomes, never class zero:

* a tied game (``home_margin == 0``) is **null** for the no-tie moneyline
  classifier -- it is neither a home win (1) nor an away win (0);
* an ATS push (``home_margin + home_spread == 0``) is **null** for the no-push
  ATS cover classifier;
* a total push (``total_points - total_line == 0``) is **null** for the no-push
  total (over/under) classifier.

The very same tied / pushed rows remain fully usable for continuous margin and
total *regression*, which has no tie/push concept -- only the binary label is
withheld. Excluding them here (as ``pd.NA``) prevents mislabelling a tie or push
as a class-zero loss, which would silently bias every calibration metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Symmetric dead-zone half-width. ``|edge| <= _EDGE_TOLERANCE`` is treated as an
# exact tie/push (null), so floating-point noise around zero never flips a label.
_EDGE_TOLERANCE = 1e-9


def edge_to_nullable_binary(edge: object) -> pd.Series:
    """Convert a numeric decision edge into a nullable binary target (``Int8``).

    ``edge > +tol -> 1``; ``edge < -tol -> 0``; ``|edge| <= tol`` (a tie / push)
    or a non-finite / missing edge (``NaN``, ``pd.NA``, ``+inf``, ``-inf``) ->
    ``pd.NA``, with ``tol = _EDGE_TOLERANCE``. Deliberately NOT a Boolean
    comparison cast to int: a zero edge must be null, not class zero (that is
    exactly the tie/push mislabelling this policy repairs).

    The result is a pandas nullable :class:`Int8` Series whose index is preserved
    exactly from a Series input.
    """
    s = pd.to_numeric(pd.Series(edge), errors="coerce")
    values = s.to_numpy(dtype="float64")
    out = pd.array([pd.NA] * len(values), dtype="Int8")
    finite = np.isfinite(values)
    out[finite & (values > _EDGE_TOLERANCE)] = 1
    out[finite & (values < -_EDGE_TOLERANCE)] = 0
    return pd.Series(out, index=s.index)
