from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from nfl_hybrid.features.feature_manifest import (
    build_manifest_matrix,
    load_feature_manifest,
    resolved_manifest_record,
)
from nfl_hybrid.features.market_compact import (
    engineer_compact_market_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact, target-specific NFL modeling matrices from the "
            "Stage 2 feature warehouse. The full warehouse is never passed "
            "directly into a model."
        )
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config/features"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_root = args.feature_root.expanduser().resolve()
    config_root = args.config_root.expanduser().resolve()

    warehouse_path = feature_root / "modeling_matrix_stage2_qb.parquet"
    if not warehouse_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 warehouse: {warehouse_path}")

    warehouse = pd.read_parquet(warehouse_path)
    print("=" * 80)
    print("FEATURE WAREHOUSE")
    print("=" * 80)
    print(f"Rows:    {len(warehouse):,}")
    print(f"Columns: {len(warehouse.columns):,}")
    print("Classification: FEATURE_WAREHOUSE_ONLY")
    print("Direct model training from this table is prohibited.")

    engineered = engineer_compact_market_features(warehouse)
    print()
    print("=" * 80)
    print("COMPACT ENGINEERED SURFACE")
    print("=" * 80)
    print(f"Rows:    {len(engineered):,}")
    print(f"Columns: {len(engineered.columns):,}")

    output_root = feature_root / "compact"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for manifest_path in sorted(config_root.glob("pregame_*.yaml")):
        manifest = load_feature_manifest(manifest_path)
        for include_market in (False, True):
            matrix, features = build_manifest_matrix(
                engineered,
                manifest,
                include_market=include_market,
            )
            variant = "market_augmented" if include_market else "football_only"
            filename = f"{manifest.market}_{variant}.parquet"
            output_path = output_root / filename
            matrix.to_parquet(output_path, index=False)

            record = resolved_manifest_record(
                manifest,
                features,
                include_market=include_market,
            )
            record_path = output_root / f"{manifest.market}_{variant}.manifest.json"
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary_rows.append(
                {
                    "market": manifest.market,
                    "variant": variant,
                    "rows": len(matrix),
                    "feature_count": len(features),
                    "total_columns": len(matrix.columns),
                    "manifest_sha256": record["sha256"],
                    "path": str(output_path),
                }
            )
            print(
                f"{manifest.market:22s} {variant:18s} "
                f"rows={len(matrix):,} features={len(features):2d}"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_root / "compact_matrix_summary.csv"
    summary.to_csv(summary_path, index=False)

    assert len(summary) == 6
    assert summary["rows"].eq(len(warehouse)).all()
    assert summary["feature_count"].max() < 60

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(summary.to_string(index=False))
    print()
    print("TARGET-SPECIFIC COMPACT MATRIX BUILD PASSED")


if __name__ == "__main__":
    main()
