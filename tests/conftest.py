"""Session-wide test setup.

Force every BLAS/OpenMP backend single-threaded before numpy/scipy/sklearn
are ever imported by a test module. This is belt-and-suspenders alongside the
``threadpoolctl.threadpool_limits(limits=1)`` scope already wrapped around
every model fit/predict call in
:mod:`nfl_hybrid.evaluation.chronological_oof` -- that guards production
usage too; this guards the test process even before that module is imported,
and covers any other library's own internal parallelism. Env vars must be set
here, before any of those libraries are imported anywhere in the test
session, because most BLAS backends only read them once, at first load.

Fix 3 (chronological OOF) fits many small models across many focused tests;
oversubscribed BLAS/OpenMP thread pools (one pool spun up per fit, contending
with every other pool) were the dominant cost, not the arithmetic itself.
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"
