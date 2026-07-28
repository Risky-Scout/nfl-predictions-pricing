
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.features.qb_pregame import (
    actual_starter_candidates,
    build_game_qb_matrix,
    build_qb_pregame_team_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe quarterback priors, starter-mixture uncertainty, "
            "and the Phase B stage-two modeling matrix."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    feature_root.mkdir(parents=True, exist_ok=True)

    games_path = data_root / "canonical" / "games.parquet"
    qb_game_path = feature_root / "qb_game_efficiency.parquet"
    stage_one_path = feature_root / "modeling_matrix_stage1.parquet"

    for path in (games_path, qb_game_path, stage_one_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    games = pd.read_parquet(games_path)
    qb_game = pd.read_parquet(qb_game_path)
    stage_one = pd.read_parquet(stage_one_path)

    print("=" * 80)
    print("INPUTS")
    print("=" * 80)
    print(f"Games:                 {len(games):,}")
    print(f"QB-game rows:          {len(qb_game):,}")
    print(f"Stage-one rows:        {len(stage_one):,}")
    print(f"Stage-one fields:      {len(stage_one.columns):,}")

    assert len(games) == 1_693
    assert len(stage_one) == 1_693
    assert qb_game["game_id"].nunique() == 1_693

    candidates = actual_starter_candidates(games)
    assert len(candidates) == 3_386
    assert candidates["starter_probability"].eq(1.0).all()

    print()
    print("=" * 80)
    print("BUILD LEAKAGE-SAFE QB PRIORS")
    print("=" * 80)

    qb_team = build_qb_pregame_team_features(
        qb_game,
        games,
        starter_candidates=candidates,
    )

    assert len(qb_team) == 3_386
    assert not qb_team.duplicated(["game_id", "team_id"]).any()
    assert qb_team.groupby("game_id")["team_id"].nunique().eq(2).all()

    reliability_columns = [
        column
        for column in qb_team.columns
        if column.endswith("_reliability")
    ]
    uncertainty_columns = [
        column
        for column in qb_team.columns
        if column.endswith("_sd")
    ]
    assert reliability_columns
    assert uncertainty_columns

    reliability = qb_team[reliability_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    uncertainty = qb_team[uncertainty_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    assert reliability.notna().all().all()
    assert reliability.ge(0).all().all()
    assert reliability.le(1).all().all()
    assert uncertainty.notna().all().all()
    assert uncertainty.gt(0).all().all()

    first_team_games = (
        qb_team.sort_values(
            ["team_id", "event_time", "game_id"],
            kind="stable",
        )
        .groupby("team_id", as_index=False)
        .head(1)
    )
    assert len(first_team_games) == 32
    assert first_team_games[reliability_columns].eq(0).all().all()

    numeric_qb_columns = [
        column
        for column in qb_team.select_dtypes(
            include=[np.number, "bool"]
        ).columns
        if column.startswith("qb_")
    ]
    assert numeric_qb_columns
    assert np.isfinite(
        qb_team[numeric_qb_columns].to_numpy(dtype=float)
    ).all()

    print(f"QB team rows:          {len(qb_team):,}")
    print(f"QB team fields:        {len(qb_team.columns):,}")
    print(f"Reliability fields:    {len(reliability_columns):,}")
    print(f"Uncertainty fields:    {len(uncertainty_columns):,}")

    print()
    print("=" * 80)
    print("BUILD GAME-LEVEL QB MATRIX")
    print("=" * 80)

    qb_game_matrix = build_game_qb_matrix(games, qb_team)

    assert len(qb_game_matrix) == 1_693
    assert qb_game_matrix["game_id"].nunique() == 1_693
    assert not qb_game_matrix["game_id"].duplicated().any()

    candidate_qb_columns = [
        column
        for column in qb_game_matrix.columns
        if (
            column.startswith("home_qb_")
            or column.startswith("away_qb_")
        )
    ]
    assert candidate_qb_columns

    # Stage one already contains game metadata such as home_qb_id,
    # away_qb_id, home_qb_name, and away_qb_name. Merge only fields
    # that are genuinely new to prevent pandas _x/_y suffix collisions.
    duplicate_qb_columns = sorted(
        set(candidate_qb_columns)
        & set(stage_one.columns)
    )

    qb_feature_columns = [
        column
        for column in candidate_qb_columns
        if column not in stage_one.columns
    ]
    assert qb_feature_columns

    print(
        "Existing QB metadata retained from Stage 1:",
        duplicate_qb_columns,
    )
    print(
        "New QB prior/mixture fields to attach:",
        len(qb_feature_columns),
    )

    qb_only = qb_game_matrix[
        ["game_id"] + qb_feature_columns
    ].copy()

    stage_two = stage_one.merge(
        qb_only,
        on="game_id",
        how="left",
        validate="1:1",
    )

    assert len(stage_two) == 1_693
    assert stage_two["game_id"].nunique() == 1_693
    assert not stage_two["game_id"].duplicated().any()

    missing_new_qb_values = (
        stage_two[qb_feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    assert missing_new_qb_values.eq(0).all(), (
        "Missing values found in newly attached QB fields:\n"
        f"{missing_new_qb_values[missing_new_qb_values.gt(0)].to_string()}"
    )

    # Existing metadata must remain unsuffixed.
    required_existing_qb_metadata = {
        "home_qb_id",
        "away_qb_id",
        "home_qb_name",
        "away_qb_name",
    }

    assert required_existing_qb_metadata.issubset(
        stage_two.columns
    )

    assert not any(
        column.endswith("_x") or column.endswith("_y")
        for column in stage_two.columns
    )

    print(f"QB game rows:          {len(qb_game_matrix):,}")
    print(f"QB game fields:        {len(qb_game_matrix.columns):,}")
    print(f"New QB fields:         {len(qb_feature_columns):,}")
    print(f"Stage-two fields:      {len(stage_two.columns):,}")

    outputs = {
        "qb_team_pregame_features.parquet": qb_team,
        "qb_game_matrix.parquet": qb_game_matrix,
        "modeling_matrix_stage2_qb.parquet": stage_two,
    }
    for filename, frame in outputs.items():
        output_path = feature_root / filename
        frame.to_parquet(output_path, index=False)
        print(
            f"{filename:42s} "
            f"rows={len(frame):,} "
            f"columns={len(frame.columns):,} "
            f"size={output_path.stat().st_size / 1_000_000:.2f} MB"
        )

    audit_rows: list[dict[str, object]] = []
    for column in qb_team.select_dtypes(
        include=[np.number, "bool"]
    ).columns:
        values = pd.to_numeric(qb_team[column], errors="coerce")
        audit_rows.append(
            {
                "column": column,
                "rows": len(values),
                "non_null_count": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["missing_rate", "column"],
        ascending=[False, True],
    )
    audit_path = feature_root / "qb_pregame_feature_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Audit: {audit_path}")

    print()
    print("Average QB reliability by season:")
    summary = (
        qb_team.assign(
            average_qb_reliability=reliability.mean(axis=1)
        )
        .groupby("season")["average_qb_reliability"]
        .agg(["count", "mean", "min", "max"])
    )
    print(summary.to_string())

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print("QB PRIOR AND STARTER-MIXTURE FEATURE BUILD PASSED")


if __name__ == "__main__":
    main()
