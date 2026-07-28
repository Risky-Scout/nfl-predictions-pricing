from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from nfl_hybrid.features.canonical_market_matrices import (
    canonicalize,
    load_roles,
)
from nfl_hybrid.features.odds_attachment import (
    OddsAttachmentConfig,
    attach_to_compact,
    build_market_odds_features,
    model_features,
)


MARKETS = (
    "pregame_moneyline",
    "pregame_ats",
    "pregame_total",
)

EXPECTED_FREEZE_SHA = (
    "7811079a5fc8371e30f7941117dbf875"
    "f1bb6359214e76c01bd4dc16d73847e0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--odds-root", type=Path, required=True)
    parser.add_argument("--prior-canonical-root", type=Path, required=True)
    parser.add_argument("--enriched-output-root", type=Path, required=True)
    parser.add_argument("--canonical-output-root", type=Path, required=True)
    parser.add_argument("--combined-output-root", type=Path, required=True)
    parser.add_argument("--role-config", type=Path, required=True)
    parser.add_argument("--frozen-spec", type=Path, required=True)
    parser.add_argument("--access-log-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frozen_spec = json.loads(
        args.frozen_spec.read_text(encoding="utf-8")
    )

    if frozen_spec["status"] != "FROZEN_FOR_2025_FINAL_TEST":
        raise ValueError("2025 model specification is not frozen.")

    if frozen_spec["freeze_sha256"] != EXPECTED_FREEZE_SHA:
        raise ValueError("Frozen 2025 model specification hash mismatch.")

    if frozen_spec["holdout_access"]["2025_accessed"]:
        raise ValueError(
            "Frozen specification already records 2025 access."
        )

    for output_root in (
        args.enriched_output_root,
        args.canonical_output_root,
        args.combined_output_root,
    ):
        shutil.rmtree(output_root, ignore_errors=True)
        output_root.mkdir(parents=True, exist_ok=True)

    args.access_log_root.mkdir(parents=True, exist_ok=True)

    consensus_path = (
        args.odds_root / "consensus_by_horizon.parquet"
    )
    movement_path = (
        args.odds_root
        / "opening_closing_market_features.parquet"
    )

    consensus = pd.read_parquet(consensus_path)
    movement = pd.read_parquet(movement_path)

    attachment_config = OddsAttachmentConfig(
        development_seasons=(2025,),
        minimum_books=3,
        maximum_snapshot_lag_minutes=5.0,
    )

    roles = load_roles(args.role_config)
    audit_rows = []

    for market in MARKETS:
        print(f"\nBuilding {market}...")

        compact_stem = f"{market}_market_augmented"
        compact_path = (
            args.compact_root / f"{compact_stem}.parquet"
        )
        compact_manifest_path = (
            args.compact_root / f"{compact_stem}.manifest.json"
        )

        compact = pd.read_parquet(compact_path)
        compact_manifest = json.loads(
            compact_manifest_path.read_text(encoding="utf-8")
        )

        odds_features = build_market_odds_features(
            consensus,
            movement,
            market,
            attachment_config,
        )

        enriched = attach_to_compact(
            compact,
            odds_features,
            market,
            attachment_config,
        )

        if len(enriched) != 285:
            raise ValueError(
                f"{market}: expected 285 rows, found {len(enriched)}"
            )

        added_features = list(model_features(market))

        enriched_manifest = dict(compact_manifest)
        enriched_manifest.update(
            {
                "variant":
                    "market_augmented_odds_enriched_2025_final_test",
                "rows": 285,
                "seasons": [2025],
                "features": list(
                    dict.fromkeys(
                        list(compact_manifest["features"])
                        + added_features
                    )
                ),
                "new_odds_features": added_features,
                "frozen_model_spec_sha256":
                    frozen_spec["freeze_sha256"],
            }
        )

        enriched_stem = (
            f"{market}_market_augmented_"
            "odds_enriched_2025_final_test"
        )

        enriched.to_parquet(
            args.enriched_output_root
            / f"{enriched_stem}.parquet",
            index=False,
        )

        (
            args.enriched_output_root
            / f"{enriched_stem}.manifest.json"
        ).write_text(
            json.dumps(
                enriched_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        canonical, canonical_manifest, audit = canonicalize(
            enriched,
            enriched_manifest,
            market=market,
            roles=roles,
        )

        canonical_stem = (
            f"{market}_market_augmented_canonical_t10"
        )

        canonical.to_parquet(
            args.canonical_output_root
            / f"{canonical_stem}.parquet",
            index=False,
        )

        (
            args.canonical_output_root
            / f"{canonical_stem}.manifest.json"
        ).write_text(
            json.dumps(
                canonical_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        prior_path = (
            args.prior_canonical_root
            / f"{canonical_stem}.parquet"
        )
        prior_manifest_path = (
            args.prior_canonical_root
            / f"{canonical_stem}.manifest.json"
        )

        prior = pd.read_parquet(prior_path)
        prior_manifest = json.loads(
            prior_manifest_path.read_text(encoding="utf-8")
        )

        if (
            prior_manifest["features"]
            != canonical_manifest["features"]
        ):
            raise ValueError(
                f"{market}: frozen feature manifests differ."
            )

        combined = pd.concat(
            [prior, canonical],
            ignore_index=True,
            sort=False,
        ).sort_values(
            ["season", "week", "game_id"],
            kind="stable",
        ).reset_index(drop=True)

        if len(combined) != 1693:
            raise ValueError(
                f"{market}: expected 1,693 combined rows, "
                f"found {len(combined)}"
            )

        if combined["game_id"].nunique() != 1693:
            raise ValueError(
                f"{market}: combined game IDs are not unique."
            )

        expected_seasons = {
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
        }

        if set(combined["season"]) != expected_seasons:
            raise ValueError(
                f"{market}: unexpected combined seasons."
            )

        combined.to_parquet(
            args.combined_output_root
            / f"{canonical_stem}.parquet",
            index=False,
        )

        combined_manifest = dict(canonical_manifest)
        combined_manifest.update(
            {
                "variant":
                    "market_augmented_canonical_t10_"
                    "2020_2025_final_test",
                "rows": 1693,
                "seasons": [
                    2020,
                    2021,
                    2022,
                    2023,
                    2024,
                    2025,
                ],
                "prior_rows": int(len(prior)),
                "final_test_rows": int(len(canonical)),
                "frozen_model_spec_sha256":
                    frozen_spec["freeze_sha256"],
            }
        )

        (
            args.combined_output_root
            / f"{canonical_stem}.manifest.json"
        ).write_text(
            json.dumps(
                combined_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        audit_rows.append(audit)

    audit_frame = pd.DataFrame(audit_rows)

    audit_frame.to_csv(
        args.canonical_output_root
        / "canonical_2025_final_test_audit.csv",
        index=False,
    )

    assert (audit_frame["rows"] == 285).all()
    assert (audit_frame["games"] == 285).all()
    assert (audit_frame["duplicate_game_ids"] == 0).all()
    assert (
        audit_frame["blocked_features_in_estimator"] == 0
    ).all()
    assert (
        audit_frame["unclassified_market_features"] == 0
    ).all()

    access_record = {
        "event": "2025_FINAL_TEST_MATRIX_BUILD",
        "frozen_model_spec_sha256":
            frozen_spec["freeze_sha256"],
        "2024_accessed": True,
        "2025_accessed": True,
        "retuning_permitted": False,
        "feature_changes_permitted": False,
        "calibration_changes_permitted": False,
    }

    with (
        args.access_log_root / "holdout_access_log.jsonl"
    ).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(access_record, sort_keys=True) + "\n"
        )

    print("\n" + "=" * 110)
    print("2025 FINAL-TEST MATRIX AUDIT")
    print("=" * 110)
    print(audit_frame.to_string(index=False))
    print("\n2025 FINAL-TEST MATRICES BUILT")


if __name__ == "__main__":
    main()
