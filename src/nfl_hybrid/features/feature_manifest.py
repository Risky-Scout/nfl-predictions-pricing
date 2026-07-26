from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json
import re

import pandas as pd
import yaml


GLOBAL_BANNED_EXACT = frozenset(
    {
        "home_score",
        "away_score",
        "score_home",
        "score_away",
        "score_total",
        "total_points",
        "home_margin",
        "margin_of_victory",
        "home_win",
        "away_win",
        "tie",
        "home_cover",
        "away_cover",
        "over",
        "under",
        "ats_push",
        "total_push",
        "home_ats_target",
    }
)

GLOBAL_BANNED_PATTERNS = (
    r"(^|_)target($|_)",
    r"(^|_)label($|_)",
    r"(^|_)outcome($|_)",
    r"(^|_)winner($|_)",
    r"(^|_)final($|_)",
    r"(^|_)post($|_)",
    r"_post$",
    r"^post_",
    r"^score_",
    r"_score$",
    r"margin_of_victory",
    r"current_game",
    r"observed_weather",
)


@dataclass(frozen=True)
class FeatureManifest:
    market: str
    horizon: str
    target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    football_features: tuple[str, ...]
    market_features: tuple[str, ...]
    max_football_features: int
    max_market_augmented_features: int
    banned_exact: tuple[str, ...] = ()
    banned_patterns: tuple[str, ...] = ()

    @property
    def market_augmented_features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.football_features + self.market_features))


def load_feature_manifest(path: str | Path) -> FeatureManifest:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    required = {
        "market",
        "horizon",
        "target_columns",
        "metadata_columns",
        "football_features",
        "market_features",
        "max_football_features",
        "max_market_augmented_features",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"Feature manifest {manifest_path} is missing: {sorted(missing)}"
        )

    manifest = FeatureManifest(
        market=str(payload["market"]),
        horizon=str(payload["horizon"]),
        target_columns=tuple(payload["target_columns"]),
        metadata_columns=tuple(payload["metadata_columns"]),
        football_features=tuple(payload["football_features"]),
        market_features=tuple(payload["market_features"]),
        max_football_features=int(payload["max_football_features"]),
        max_market_augmented_features=int(
            payload["max_market_augmented_features"]
        ),
        banned_exact=tuple(payload.get("banned_exact", ())),
        banned_patterns=tuple(payload.get("banned_patterns", ())),
    )
    _validate_manifest_definition(manifest, manifest_path)
    return manifest


def _validate_manifest_definition(
    manifest: FeatureManifest,
    path: Path | None = None,
) -> None:
    label = str(path) if path is not None else manifest.market

    if len(manifest.football_features) > manifest.max_football_features:
        raise ValueError(
            f"{label}: football feature count exceeds configured cap."
        )
    if (
        len(manifest.market_augmented_features)
        > manifest.max_market_augmented_features
    ):
        raise ValueError(
            f"{label}: market-augmented feature count exceeds configured cap."
        )
    duplicates = [
        name
        for name in manifest.market_augmented_features
        if manifest.market_augmented_features.count(name) > 1
    ]
    if duplicates:
        raise ValueError(f"{label}: duplicate features: {sorted(set(duplicates))}")

    validate_no_banned_features(
        manifest.market_augmented_features,
        manifest=manifest,
    )


def validate_no_banned_features(
    columns: Iterable[str],
    *,
    manifest: FeatureManifest | None = None,
) -> None:
    selected = tuple(str(column) for column in columns)
    banned_exact = set(GLOBAL_BANNED_EXACT)
    banned_patterns = list(GLOBAL_BANNED_PATTERNS)

    if manifest is not None:
        banned_exact.update(manifest.banned_exact)
        banned_patterns.extend(manifest.banned_patterns)

    violations: list[str] = []
    for column in selected:
        lowered = column.lower()
        if lowered in {value.lower() for value in banned_exact}:
            violations.append(column)
            continue
        if any(re.search(pattern, lowered) for pattern in banned_patterns):
            violations.append(column)

    if violations:
        raise ValueError(
            "Banned or postgame feature(s) selected: "
            + ", ".join(sorted(set(violations)))
        )


def build_manifest_matrix(
    engineered: pd.DataFrame,
    manifest: FeatureManifest,
    *,
    include_market: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = (
        manifest.market_augmented_features
        if include_market
        else manifest.football_features
    )
    cap = (
        manifest.max_market_augmented_features
        if include_market
        else manifest.max_football_features
    )

    validate_no_banned_features(features, manifest=manifest)

    required = tuple(
        dict.fromkeys(
            manifest.metadata_columns
            + manifest.target_columns
            + features
        )
    )
    missing = [column for column in required if column not in engineered.columns]
    if missing:
        raise ValueError(
            f"{manifest.market}: engineered table is missing: {missing}"
        )
    if len(features) > cap:
        raise ValueError(
            f"{manifest.market}: resolved feature count {len(features)} exceeds {cap}."
        )

    matrix = engineered.loc[:, required].copy()
    if matrix["game_id"].duplicated().any():
        raise ValueError(f"{manifest.market}: duplicate game_id rows.")

    feature_frame = matrix.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    )
    fully_missing = [
        column for column in features if feature_frame[column].isna().all()
    ]
    if fully_missing:
        raise ValueError(
            f"{manifest.market}: fully missing feature(s): {fully_missing}"
        )
    constant = [
        column
        for column in features
        if feature_frame[column].nunique(dropna=True) <= 1
    ]
    if constant:
        raise ValueError(
            f"{manifest.market}: constant feature(s): {constant}"
        )
    return matrix, features


def resolved_manifest_record(
    manifest: FeatureManifest,
    features: Iterable[str],
    *,
    include_market: bool,
) -> dict[str, object]:
    selected = tuple(features)
    payload = {
        "market": manifest.market,
        "horizon": manifest.horizon,
        "variant": "market_augmented" if include_market else "football_only",
        "feature_count": len(selected),
        "features": list(selected),
        "target_columns": list(manifest.target_columns),
        "metadata_columns": list(manifest.metadata_columns),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload
