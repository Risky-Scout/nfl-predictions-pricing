"""Transparent fundamental SHADOW price (Acceptance Gate C).

The repository has a frozen challenger *specification* (JointScoreModel + EPA, the
C4 candidate from the pre-registered EPA tournament) but no serialized inference
artifact. Per Section 14, a deterministic shadow artifact is built from that frozen
spec and permitted pre-2025 data only — no new model selection, no hyperparameter
change. The shadow is a *monitoring* estimate; production stays the market baseline
until the live gate promotes a market.

Distinct concepts are never conflated:
  market_probability   — de-vigged consensus market
  fundamental_probability — this shadow model (NULL when its inputs are unavailable)
  production_probability  — market baseline (governance), unless a challenger is promoted
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHADOW_DIR = Path("config/shadow_fundamental_2026")
FEATURES_JSON = SHADOW_DIR / "features.json"
MODEL_PATH = SHADOW_DIR / "jointscore_epa.joblib"
META_PATH = SHADOW_DIR / "metadata.json"


def build_shadow_artifact(matrix: pd.DataFrame, features: list[str], *, code_commit: str = "unknown") -> dict:
    """Deterministically fit the frozen challenger (JointScoreModel + EPA features) on
    seasons <= 2024 and serialize it. No selection, no tuning."""
    import joblib
    from nfl_hybrid.modern.joint_score import JointScoreModel

    train = matrix[matrix["season"] <= 2023].copy()
    calib = matrix[matrix["season"] == 2024].copy()
    model = JointScoreModel(numeric_features=features)
    model.fit(train, calibration=calib)

    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURES_JSON.write_text(json.dumps(features, indent=2))
    meta = {
        "artifact": "shadow_fundamental_2026.v1",
        "spec": "JointScoreModel + EPA (frozen C4 challenger)",
        "training_cutoff_season": 2024,
        "holdout_excluded": 2025,
        "n_features": len(features),
        "features": features,
        "random_seed": int(model.config.random_state),
        "train_seasons": "2020-2023", "calibration_season": 2024,
        "code_commit": code_commit,
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def load_shadow():
    import joblib
    if not MODEL_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_JSON.read_text())
    meta = json.loads(META_PATH.read_text())
    return model, features, meta


def score_shadow(features_frame: pd.DataFrame) -> pd.DataFrame | None:
    """Return per-game fundamental probabilities, or None if the artifact/features
    are unavailable. Missing inputs yield NULL, never the market probability."""
    model, features, _ = load_shadow()
    if model is None or features is None:
        return None
    missing = [c for c in features if c not in features_frame.columns]
    if missing:
        return None
    pred = model.predict_markets(features_frame)
    return pd.DataFrame({
        "game_id": features_frame["game_id"].astype(str).to_numpy() if "game_id" in features_frame else np.arange(len(pred)),
        "fundamental_home_win": pred["home_win_probability_no_tie"].to_numpy(),
        "fundamental_home_cover": pred["home_cover_probability_no_push"].to_numpy(),
        "fundamental_over": pred["over_probability_no_push"].to_numpy(),
    })


def attach_shadow_columns(card: pd.DataFrame, fundamental: pd.DataFrame | None, *, status: str, artifact_version: str | None) -> pd.DataFrame:
    """Attach distinct fundamental / production columns. When unavailable, the
    fundamental column stays NULL (never the market probability)."""
    out = card.copy()
    market_col = {"moneyline": "fundamental_home_win", "ats": "fundamental_home_cover", "total": "fundamental_over"}
    if fundamental is not None:
        fmap = fundamental.set_index("game_id")
        vals = []
        for _, row in out.iterrows():
            col = market_col.get(row["market"])
            gid = str(row["game_id"])
            vals.append(float(fmap.loc[gid, col]) if (gid in fmap.index and col) else np.nan)
        out["fundamental_probability"] = vals
    else:
        out["fundamental_probability"] = np.nan  # NULL, not market
    out["fundamental_input_status"] = status
    out["fundamental_artifact_version"] = artifact_version or "NO_FROZEN_FUNDAMENTAL_ARTIFACT"
    # production stays the market baseline (governance); never the shadow
    out["production_probability"] = out["market_fair_probability"]
    out["production_source"] = "MARKET_BASELINE"
    out["market_fundamental_difference"] = out["fundamental_probability"] - out["market_fair_probability"]
    return out
