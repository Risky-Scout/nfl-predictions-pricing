from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKETS = ("pregame_moneyline", "pregame_ats", "pregame_total")
ROLES = {
    "CANONICAL_ANCHOR",
    "CHALLENGER_FEATURE",
    "QUALITY_GATE",
    "AUDIT_ONLY",
    "PROHIBITED",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_roles(path: str | Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != set(MARKETS):
        raise ValueError("Role config must define all three markets.")
    for market, mapping in payload.items():
        if set(mapping) != ROLES:
            raise ValueError(f"{market} has invalid role names.")
        seen: dict[str, str] = {}
        for role, columns in mapping.items():
            for column in columns:
                if column in seen:
                    raise ValueError(
                        f"{market}.{column} has multiple roles."
                    )
                seen[column] = role
    return payload


def _market_like(column: str) -> bool:
    return (
        column.startswith(("market_", "spread_", "ats_t10_"))
        or column == "home_favorite_flag"
    )


def _add_ats_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    line = pd.to_numeric(output["market_t10_consensus_line"], errors="raise")
    magnitude = line.abs()
    fraction = np.mod(magnitude, 1.0)
    output["ats_t10_spread_magnitude"] = magnitude
    output["ats_t10_spread_distance_to_3"] = (magnitude - 3.0).abs()
    output["ats_t10_spread_distance_to_7"] = (magnitude - 7.0).abs()
    output["ats_t10_spread_integer_flag"] = np.isclose(
        fraction, 0.0, atol=1e-9
    ).astype(int)
    output["ats_t10_spread_half_point_flag"] = np.isclose(
        fraction, 0.5, atol=1e-9
    ).astype(int)
    output["ats_t10_home_favorite_flag"] = (line < 0).astype(int)
    return output


def _add_t10_targets(
    frame: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    """
    Derive settlement targets from the same verified T-10 line used by
    the estimator and market baseline.

    Legacy targets remain in the parquet for audit comparison, but the
    canonical manifest points only to these T-10-aligned targets.
    """
    output = frame.copy()

    if market == "pregame_ats":
        required = {
            "target_home_margin",
            "market_t10_consensus_line",
        }
        missing = sorted(required - set(output.columns))
        if missing:
            raise ValueError(
                f"ATS T-10 target derivation missing columns: {missing}"
            )

        margin = pd.to_numeric(
            output["target_home_margin"],
            errors="raise",
        )
        line = pd.to_numeric(
            output["market_t10_consensus_line"],
            errors="raise",
        )
        residual = margin + line
        push = np.isclose(
            residual,
            0.0,
            atol=1e-9,
            rtol=0.0,
        )

        output["target_t10_home_cover"] = (
            residual > 1e-9
        ).astype(int)
        output["target_t10_ats_push"] = push.astype(int)
        output["target_t10_margin_residual"] = residual

    elif market == "pregame_total":
        required = {
            "target_total_points",
            "market_t10_consensus_line",
        }
        missing = sorted(required - set(output.columns))
        if missing:
            raise ValueError(
                f"Total T-10 target derivation missing columns: {missing}"
            )

        points = pd.to_numeric(
            output["target_total_points"],
            errors="raise",
        )
        line = pd.to_numeric(
            output["market_t10_consensus_line"],
            errors="raise",
        )
        residual = points - line
        push = np.isclose(
            residual,
            0.0,
            atol=1e-9,
            rtol=0.0,
        )

        output["target_t10_over"] = (
            residual > 1e-9
        ).astype(int)
        output["target_t10_total_push"] = push.astype(int)
        output["target_t10_total_residual"] = residual

    return output


def canonicalize(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    market: str,
    roles: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if market not in MARKETS:
        raise ValueError(f"Unsupported market: {market}")
    if frame["game_id"].duplicated().any():
        raise ValueError(f"{market} has duplicate game_id values.")

    output = (
        _add_ats_features(frame)
        if market == "pregame_ats"
        else frame.copy()
    )
    output = _add_t10_targets(output, market)
    original = list(manifest.get("features", []))
    if not original:
        raise ValueError(f"{market} manifest has no features.")

    missing = sorted(set(original) - set(output.columns))
    if missing:
        raise ValueError(f"{market} manifest columns missing: {missing}")

    mapping = roles[market]
    lookup = {
        column: role
        for role, columns in mapping.items()
        for column in columns
    }
    unknown = sorted(
        column
        for column in original
        if _market_like(column) and column not in lookup
    )
    if unknown:
        raise ValueError(
            f"{market} has unclassified market fields: {unknown}"
        )

    anchors = mapping["CANONICAL_ANCHOR"]
    approved = anchors + mapping["CHALLENGER_FEATURE"]
    missing_approved = sorted(set(approved) - set(output.columns))
    if missing_approved:
        raise ValueError(
            f"{market} missing canonical fields: {missing_approved}"
        )

    football = [column for column in original if not _market_like(column)]
    features = list(dict.fromkeys(football + approved))
    blocked = (
        set(mapping["PROHIBITED"])
        | set(mapping["QUALITY_GATE"])
        | set(mapping["AUDIT_ONLY"])
    )
    leaked = sorted(set(features) & blocked)
    if leaked:
        raise ValueError(f"{market} blocked estimator fields: {leaked}")

    if market == "pregame_ats":
        target_columns = [
            "target_t10_home_cover",
            "target_t10_ats_push",
            "target_t10_margin_residual",
            "target_home_margin",
        ]
    elif market == "pregame_total":
        target_columns = [
            "target_t10_over",
            "target_t10_total_push",
            "target_t10_total_residual",
            "target_total_points",
        ]
    else:
        target_columns = list(
            manifest.get(
                "target_columns",
                [
                    "target_home_win",
                    "target_tie",
                    "target_home_margin",
                ],
            )
        )

    result_manifest = dict(manifest)
    result_manifest.update(
        {
            "variant": "market_augmented_canonical_t10",
            "rows": int(len(output)),
            "features": features,
            "feature_count": len(features),
            "target_columns": target_columns,
            "market_feature_roles": mapping,
            "canonical_market_anchors": anchors,
            "approved_market_challenger_features":
                mapping["CHALLENGER_FEATURE"],
            "quality_gate_columns": mapping["QUALITY_GATE"],
            "audit_only_columns": mapping["AUDIT_ONLY"],
            "prohibited_columns": mapping["PROHIBITED"],
        }
    )

    audit = {
        "market": market,
        "rows": int(len(output)),
        "games": int(output["game_id"].nunique()),
        "original_features": len(original),
        "canonical_features": len(features),
        "canonical_anchors": len(anchors),
        "blocked_features_in_estimator": len(set(features) & blocked),
        "unclassified_market_features": len(unknown),
        "duplicate_game_ids": int(output["game_id"].duplicated().sum()),
    }
    return output, result_manifest, audit


def build_all(
    *,
    enriched_root: str | Path,
    output_root: str | Path,
    role_config: str | Path,
) -> pd.DataFrame:
    source = Path(enriched_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    role_path = Path(role_config).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    roles = load_roles(role_path)
    audits = []

    for market in MARKETS:
        stem = f"{market}_market_augmented_odds_enriched"
        frame_path = source / f"{stem}.parquet"
        manifest_path = source / f"{stem}.manifest.json"
        frame = pd.read_parquet(frame_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        canonical, out_manifest, audit = canonicalize(
            frame, manifest, market=market, roles=roles
        )
        out_stem = f"{market}_market_augmented_canonical_t10"
        out_frame = destination / f"{out_stem}.parquet"
        out_manifest_path = destination / f"{out_stem}.manifest.json"
        canonical.to_parquet(out_frame, index=False)
        out_manifest.update(
            {
                "source_frame": str(frame_path),
                "source_manifest": str(manifest_path),
                "source_frame_sha256": _sha256(frame_path),
                "source_manifest_sha256": _sha256(manifest_path),
                "role_config_sha256": _sha256(role_path),
            }
        )
        out_manifest_path.write_text(
            json.dumps(out_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audits.append(audit)

    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(
        destination / "canonical_matrix_audit.csv", index=False
    )
    (destination / "market_semantic_lineage.json").write_text(
        json.dumps(roles, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_frame
