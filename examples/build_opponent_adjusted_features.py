from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.features.opponent_pregame import (
    build_game_opponent_adjusted_matrix,
    build_opponent_adjusted_team_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe opponent-adjusted NFL team features "
            "and merge them into the pregame modeling matrix."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    feature_root.mkdir(parents=True, exist_ok=True)

    games_path = data_root / "canonical" / "games.parquet"
    team_game_path = (
        feature_root
        / "advanced_team_game_efficiency.parquet"
    )
    base_matrix_path = (
        feature_root
        / "pregame_game_matrix.parquet"
    )

    for path in (
        games_path,
        team_game_path,
        base_matrix_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {path}"
            )

    games = pd.read_parquet(games_path)
    team_game = pd.read_parquet(team_game_path)
    base_matrix = pd.read_parquet(base_matrix_path)

    print("=" * 80)
    print("INPUTS")
    print("=" * 80)
    print(f"Games:              {len(games):,}")
    print(f"Team-game rows:     {len(team_game):,}")
    print(f"Base matrix rows:   {len(base_matrix):,}")
    print(f"Base matrix fields: {len(base_matrix.columns):,}")

    assert len(games) == 1_693
    assert len(team_game) == 3_386
    assert len(base_matrix) == 1_693

    print()
    print("=" * 80)
    print("BUILD OPPONENT-ADJUSTED TEAM FEATURES")
    print("=" * 80)

    team_strength = build_opponent_adjusted_team_features(
        team_game,
        games,
    )

    assert len(team_strength) == 3_386
    assert not team_strength.duplicated(
        ["game_id", "team_id"]
    ).any()

    assert (
        team_strength.groupby("game_id")["team_id"]
        .nunique()
        .eq(2)
        .all()
    )

    reliability_columns = [
        column
        for column in team_strength.columns
        if column.endswith("_reliability")
    ]

    exposure_columns = [
        column
        for column in team_strength.columns
        if column.endswith("_exposure")
    ]

    uncertainty_columns = [
        column
        for column in team_strength.columns
        if column.endswith("_sd")
    ]

    assert reliability_columns
    assert exposure_columns
    assert uncertainty_columns

    reliability = team_strength[reliability_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    exposure = team_strength[exposure_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    uncertainty = team_strength[uncertainty_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    assert reliability.notna().all().all()
    assert reliability.ge(0).all().all()
    assert reliability.le(1).all().all()

    assert exposure.notna().all().all()
    assert exposure.ge(0).all().all()

    assert uncertainty.notna().all().all()
    assert uncertainty.gt(0).all().all()

    first_team_games = (
        team_strength
        .sort_values(
            ["team_id", "event_time", "game_id"],
            kind="stable",
        )
        .groupby("team_id", as_index=False)
        .head(1)
    )

    assert len(first_team_games) == 32

    # A team's first observed game cannot use its own future exposure.
    assert (
        first_team_games[reliability_columns]
        .eq(0)
        .all()
        .all()
    )

    numeric_oa_columns = [
        column
        for column in team_strength.select_dtypes(
            include=[np.number, "bool"]
        ).columns
        if column.startswith("oa_")
    ]

    assert numeric_oa_columns
    assert np.isfinite(
        team_strength[numeric_oa_columns]
        .to_numpy(dtype=float)
    ).all()

    print(f"Rows:                {len(team_strength):,}")
    print(f"Columns:             {len(team_strength.columns):,}")
    print(f"Reliability fields:  {len(reliability_columns):,}")
    print(f"Exposure fields:     {len(exposure_columns):,}")
    print(f"Uncertainty fields:  {len(uncertainty_columns):,}")

    print()
    print("=" * 80)
    print("BUILD ONE-ROW-PER-GAME OPPONENT MATRIX")
    print("=" * 80)

    game_strength = build_game_opponent_adjusted_matrix(
        games,
        team_strength,
    )

    assert len(game_strength) == 1_693
    assert game_strength["game_id"].nunique() == 1_693
    assert not game_strength["game_id"].duplicated().any()

    opponent_game_columns = [
        column
        for column in game_strength.columns
        if (
            column.startswith("home_oa_")
            or column.startswith("away_oa_")
        )
    ]

    assert opponent_game_columns

    opponent_only = game_strength[
        ["game_id"] + opponent_game_columns
    ].copy()

    stage_one_matrix = base_matrix.merge(
        opponent_only,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    assert len(stage_one_matrix) == 1_693
    assert stage_one_matrix["game_id"].nunique() == 1_693
    assert not stage_one_matrix["game_id"].duplicated().any()

    assert (
        stage_one_matrix[opponent_game_columns]
        .notna()
        .all()
        .all()
    )

    print(f"Opponent matrix rows:    {len(game_strength):,}")
    print(f"Opponent matrix fields:  {len(game_strength.columns):,}")
    print(f"Stage-one rows:          {len(stage_one_matrix):,}")
    print(f"Stage-one fields:        {len(stage_one_matrix.columns):,}")
    print(f"New opponent fields:     {len(opponent_game_columns):,}")

    print()
    print("=" * 80)
    print("CREATE FEATURE AUDIT")
    print("=" * 80)

    audit_rows: list[dict[str, object]] = []

    numeric_columns = team_strength.select_dtypes(
        include=[np.number, "bool"]
    ).columns

    for column in numeric_columns:
        values = pd.to_numeric(
            team_strength[column],
            errors="coerce",
        )

        audit_rows.append(
            {
                "column": column,
                "rows": len(values),
                "non_null_count": int(
                    values.notna().sum()
                ),
                "missing_rate": float(
                    values.isna().mean()
                ),
                "mean": float(values.mean()),
                "standard_deviation": float(
                    values.std()
                ),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["missing_rate", "column"],
        ascending=[False, True],
    )

    outputs = {
        "opponent_adjusted_team_features.parquet": (
            team_strength
        ),
        "opponent_adjusted_game_matrix.parquet": (
            game_strength
        ),
        "modeling_matrix_stage1.parquet": (
            stage_one_matrix
        ),
    }

    for filename, frame in outputs.items():
        output_path = feature_root / filename
        frame.to_parquet(
            output_path,
            index=False,
        )

        print(
            f"{filename:48s} "
            f"rows={len(frame):,} "
            f"columns={len(frame.columns):,} "
            f"size={output_path.stat().st_size / 1_000_000:.2f} MB"
        )

    audit_path = (
        feature_root
        / "opponent_adjusted_feature_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    print(f"Audit: {audit_path}")

    print()
    print("Average reliability by season:")

    reliability_summary = (
        team_strength.assign(
            average_reliability=(
                reliability.mean(axis=1)
            )
        )
        .groupby("season")["average_reliability"]
        .agg(["count", "mean", "min", "max"])
    )

    print(reliability_summary.to_string())

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print("OPPONENT-ADJUSTED FEATURE BUILD PASSED")


if __name__ == "__main__":
    main()
