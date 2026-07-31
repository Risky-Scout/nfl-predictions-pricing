"""Deterministically (re)build the fundamental shadow artifact from the frozen
challenger spec (JointScoreModel + EPA) using only permitted data through 2024.

No model selection, no hyperparameter change. A clean rebuild yields numerically
identical parameters (same features, same training rows, fixed model random_state).
The .joblib is gitignored (binary); this committed script + the recorded hashes make
it reproducible. Run: PYTHONPATH=src python scripts/build_fundamental_shadow.py
"""

from __future__ import annotations

import subprocess

import pandas as pd

from nfl_hybrid.features.augmented_matrix import (
    FROZEN_FEATURES,
    build_augmented_feature_matrix,
)
from nfl_hybrid.pricing.fundamental_shadow import build_shadow_artifact


def main() -> None:
    games = pd.read_parquet("data/backfill_2020_2025/canonical/games.parquet")
    pbp = pd.read_parquet("data/backfill_2020_2025/raw/pbp.parquet")
    matrix, manifest = build_augmented_feature_matrix(games, pbp)
    # single authoritative contract: training matrix must expose exactly the frozen list
    assert manifest["all_features"] == FROZEN_FEATURES, "training features drifted from FROZEN_FEATURES"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = build_shadow_artifact(matrix, FROZEN_FEATURES, code_commit=commit)
    print("shadow artifact:", meta["artifact"], "| cutoff", meta["training_cutoff_season"],
          "| features", meta["n_features"], "| model_sha256", meta["model_sha256"][:12])


if __name__ == "__main__":
    main()
