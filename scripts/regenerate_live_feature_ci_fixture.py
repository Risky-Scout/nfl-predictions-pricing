#!/usr/bin/env python3
"""Manually (re)generate the committed core live-feature CI fixture.

This script is the ONLY sanctioned way to change the golden expected outputs. It
never runs during CI and refuses to touch the committed files unless ``--overwrite``
is passed. It writes deterministic ``games.csv`` / ``pbp.csv`` inputs, freezes the
golden ``expected_features.csv`` from the *accepted historical training path*
(``build_augmented_feature_matrix``), and records a ``fixture_manifest.json`` of
hashes and target metadata.

    PYTHONPATH=src python scripts/regenerate_live_feature_ci_fixture.py --overwrite

If a future change intentionally moves a golden value, run this with ``--overwrite``
and review the printed old/new hashes and the resulting diff before committing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(REPO, "tests", "fixtures", "live_features")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, FIXTURE_DIR)

import factory  # noqa: E402  (deterministic fixture factory)
from nfl_hybrid.features.augmented_matrix import (  # noqa: E402
    FROZEN_FEATURES,
    build_augmented_feature_matrix,
    build_live_augmented_features,
)

FIXTURE_VERSION = "1.0.0"
GAMES_CSV = os.path.join(FIXTURE_DIR, "games.csv")
PBP_CSV = os.path.join(FIXTURE_DIR, "pbp.csv")
EXPECTED_CSV = os.path.join(FIXTURE_DIR, "expected_features.csv")
MANIFEST_JSON = os.path.join(FIXTURE_DIR, "fixture_manifest.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "<absent>"
    with open(path, "rb") as handle:
        return _sha256_bytes(handle.read())


def _target_frame(games: pd.DataFrame, gid: str) -> pd.DataFrame:
    t = games[games["game_id"].astype(str) == gid].copy()
    t["home_spread"] = pd.to_numeric(t["home_spread_reference"], errors="coerce")
    t["total_line"] = pd.to_numeric(t["total_line_reference"], errors="coerce")
    return t


def _as_of_before(target: pd.DataFrame, minutes: int = 10) -> str:
    ko = pd.to_datetime(target["scheduled_kickoff_utc"].iloc[0], utc=True)
    return (ko - pd.Timedelta(minutes=minutes)).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Required: actually write the committed fixture files.",
    )
    args = parser.parse_args()

    if os.environ.get("CI"):
        print("Refusing to run inside CI (CI env var set). This is a manual tool.")
        return 2
    if not args.overwrite:
        print("Refusing to run without --overwrite. This is a manual, deterministic tool.")
        print("Old hashes:")
        print(f"  games.csv            {_sha256_file(GAMES_CSV)}")
        print(f"  pbp.csv              {_sha256_file(PBP_CSV)}")
        print(f"  expected_features.csv{_sha256_file(EXPECTED_CSV)}")
        return 1

    old = {
        "games.csv": _sha256_file(GAMES_CSV),
        "pbp.csv": _sha256_file(PBP_CSV),
        "expected_features.csv": _sha256_file(EXPECTED_CSV),
    }

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # 1. Deterministic synthetic inputs.
    games = factory.build_games()[factory.GAMES_COLUMNS]
    pbp = factory.build_pbp()[factory.PBP_COLUMNS]
    games.to_csv(GAMES_CSV, index=False)
    pbp.to_csv(PBP_CSV, index=False)

    # 2. Read back exactly as the tests do, then freeze golden from the TRAINING path.
    g = pd.read_csv(GAMES_CSV)
    p = pd.read_csv(PBP_CSV)
    completed = g[g["home_score"].notna()].copy()
    matrix, _ = build_augmented_feature_matrix(completed, p)

    rows = []
    snapshot_hashes = {}
    as_of = {}
    for gid in factory.GOLDEN_TARGETS:
        hist = matrix[matrix["game_id"].astype(str) == gid]
        if len(hist) != 1:
            raise SystemExit(f"training path did not yield exactly one row for {gid}")
        row = {"game_id": gid}
        for col in FROZEN_FEATURES:
            row[col] = hist.iloc[0][col]
        rows.append(row)

        target = _target_frame(g, gid)
        asof = _as_of_before(target)
        as_of[gid] = asof
        _, status = build_live_augmented_features(
            g, target, p, mode="historical_replay", as_of_utc=asof
        )
        snapshot_hashes[gid] = status["feature_snapshot_hash"]

    expected = pd.DataFrame(rows)[["game_id"] + FROZEN_FEATURES]
    expected.to_csv(EXPECTED_CSV, index=False)

    # 3. Manifest (no volatile creation timestamp; deterministic content only).
    frozen_hash = _sha256_bytes(json.dumps(FROZEN_FEATURES).encode())
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "creation_reason": (
            "Step 2: run core live/training feature-parity and leakage tests in "
            "normal CI on committed synthetic fixtures instead of skipping when the "
            "private 2020-2025 backfill parquets are absent."
        ),
        "target_game_ids": list(factory.GOLDEN_TARGETS),
        "target_as_of_utc": as_of,
        "frozen_features": list(FROZEN_FEATURES),
        "frozen_features_sha256": frozen_hash,
        "expected_snapshot_hashes": snapshot_hashes,
        "games_csv_sha256": _sha256_file(GAMES_CSV),
        "pbp_csv_sha256": _sha256_file(PBP_CSV),
        "expected_features_csv_sha256": _sha256_file(EXPECTED_CSV),
    }
    with open(MANIFEST_JSON, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    new = {
        "games.csv": manifest["games_csv_sha256"],
        "pbp.csv": manifest["pbp_csv_sha256"],
        "expected_features.csv": manifest["expected_features_csv_sha256"],
    }
    print(f"Wrote fixture v{FIXTURE_VERSION} to {FIXTURE_DIR}")
    print(f"  games rows={len(games)} pbp rows={len(pbp)} golden targets={len(expected)}")
    for name in ("games.csv", "pbp.csv", "expected_features.csv"):
        print(f"  {name:22s} {old[name]}  ->  {new[name]}")
    print("Snapshot hashes:")
    for gid, h in snapshot_hashes.items():
        print(f"  {gid:20s} {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
