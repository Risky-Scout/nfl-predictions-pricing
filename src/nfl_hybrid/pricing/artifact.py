"""Load the frozen pricing-calibration artifact for the runtime path.

The weekly path calls :func:`load_pricing_artifact` **once** and reuses the
returned object (surface tables are precomputed lookups). No fitting happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json

import pandas as pd

from nfl_hybrid.pricing.margin_surface import MarginSurface

DEFAULT_ARTIFACT = "config/pricing_calibration_2026.json"


@dataclass
class PricingArtifact:
    version: str
    artifact_sha256: str
    code_commit: str
    training_cutoff_season: int
    margin_surface: MarginSurface
    margin_sigma: float
    total_sigma: float
    devig_consensus: dict
    raw: dict

    # -- convenience pricing (delegates to the surface) --------------------- #
    def cover(self, closing_spread: float, cover_line: float) -> tuple[float, float, float]:
        return self.margin_surface.cover_probabilities(closing_spread, cover_line, self.margin_sigma)

    def moneyline(self, closing_spread: float) -> tuple[float, float, float]:
        return self.margin_surface.moneyline_probabilities(closing_spread, self.margin_sigma)


@lru_cache(maxsize=4)
def load_pricing_artifact(path: str = DEFAULT_ARTIFACT) -> PricingArtifact:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Pricing calibration artifact not found at {path}. "
            "Run scripts/build_pricing_calibration.py (offline path) first."
        )
    art = json.loads(p.read_text())
    ms_cfg = art["margin_surface"]
    method = ms_cfg.get("selected", "normal")
    if method == "empirical":
        pmf_csv = ms_cfg["pmf_export_csv"]
        pmf = pd.read_csv(pmf_csv)
        surface = MarginSurface(
            method="empirical", pmf_table=pmf,
            bucket_width=float(ms_cfg.get("bucket_width", 2.0)),
            tail_cutoff=int(ms_cfg.get("tail_cutoff", 21)),
        )
    else:
        surface = MarginSurface(method="normal")
    return PricingArtifact(
        version=art.get("artifact_version", "unknown"),
        artifact_sha256=art.get("artifact_sha256", "unknown"),
        code_commit=art.get("code_commit", "unknown"),
        training_cutoff_season=int(art.get("training_cutoff_season", 2024)),
        margin_surface=surface,
        margin_sigma=float(art["runtime_defaults"]["margin_sigma_value"]),
        total_sigma=float(art["runtime_defaults"]["total_sigma_value"]),
        devig_consensus={k: art["runtime_defaults"][k] for k in art["runtime_defaults"]
                         if k.endswith(("_devig", "_consensus"))},
        raw=art,
    )
