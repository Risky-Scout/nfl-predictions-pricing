from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompactSchemaContract:
    market: str
    metadata_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    football_feature_count: int
    market_feature_count: int
    market_features: tuple[str, ...]

    @property
    def augmented_feature_count(self) -> int:
        return self.football_feature_count + self.market_feature_count


COMMON_METADATA = (
    "game_id",
    "season",
    "home_team_id",
    "away_team_id",
)

SCHEMA_CONTRACTS: dict[str, CompactSchemaContract] = {
    "pregame_moneyline": CompactSchemaContract(
        market="pregame_moneyline",
        metadata_columns=COMMON_METADATA,
        target_columns=(
            "target_home_win",
            "target_tie",
            "target_home_margin",
        ),
        football_feature_count=33,
        market_feature_count=9,
        market_features=(
            "market_home_ml_novig_prob",
            "market_home_ml_raw_prob",
            "market_moneyline_hold",
            "market_implied_margin",
            "market_total_line",
            "market_implied_home_points",
            "market_implied_away_points",
            "market_spread_source_disagreement",
            "market_total_source_disagreement",
        ),
    ),
    "pregame_ats": CompactSchemaContract(
        market="pregame_ats",
        metadata_columns=COMMON_METADATA,
        target_columns=(
            "target_home_cover",
            "target_ats_push",
            "target_margin_residual",
            "target_home_margin",
        ),
        football_feature_count=31,
        market_feature_count=13,
        market_features=(
            "market_home_spread",
            "market_home_cover_novig_prob",
            "market_spread_hold",
            "market_total_line",
            "market_implied_margin",
            "spread_magnitude",
            "spread_distance_to_3",
            "spread_distance_to_7",
            "spread_integer_flag",
            "spread_half_point_flag",
            "home_favorite_flag",
            "market_spread_source_disagreement",
            "market_total_source_disagreement",
        ),
    ),
    "pregame_total": CompactSchemaContract(
        market="pregame_total",
        metadata_columns=COMMON_METADATA,
        target_columns=(
            "target_over",
            "target_total_push",
            "target_total_residual",
            "target_total_points",
        ),
        football_feature_count=36,
        market_feature_count=9,
        market_features=(
            "market_total_line",
            "market_over_novig_prob",
            "market_total_hold",
            "market_implied_home_points",
            "market_implied_away_points",
            "market_implied_margin",
            "market_home_spread",
            "market_total_source_disagreement",
            "market_spread_source_disagreement",
        ),
    ),
}


def _load_manifest(
    compact_root: Path,
    market: str,
    variant: str,
) -> dict[str, object]:
    path = compact_root / f"{market}_{variant}.manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _required_columns(manifest: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            tuple(manifest["metadata_columns"])
            + tuple(manifest["target_columns"])
            + tuple(manifest["features"])
        )
    )


def review_compact_schemas(
    compact_root: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    compact_path = Path(compact_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for market, contract in SCHEMA_CONTRACTS.items():
        for variant in ("football_only", "market_augmented"):
            manifest = _load_manifest(compact_path, market, variant)
            matrix_path = compact_path / f"{market}_{variant}.parquet"
            if not matrix_path.exists():
                raise FileNotFoundError(matrix_path)
            frame = pd.read_parquet(matrix_path)

            expected_feature_count = (
                contract.football_feature_count
                if variant == "football_only"
                else contract.augmented_feature_count
            )
            features = tuple(manifest["features"])
            required = _required_columns(manifest)
            missing = sorted(set(required) - set(frame.columns))
            unexpected = sorted(set(frame.columns) - set(required))
            duplicate_games = int(frame["game_id"].duplicated().sum())
            all_null_features = sorted(
                feature
                for feature in features
                if feature in frame and frame[feature].isna().all()
            )
            constant_features = sorted(
                feature
                for feature in features
                if feature in frame
                and frame[feature].nunique(dropna=True) <= 1
            )

            manifest_metadata_matches = (
                tuple(manifest["metadata_columns"])
                == contract.metadata_columns
            )
            manifest_targets_match = (
                tuple(manifest["target_columns"])
                == contract.target_columns
            )
            feature_count_matches = len(features) == expected_feature_count
            market_features_present = (
                True
                if variant == "football_only"
                else set(contract.market_features).issubset(features)
            )

            status = "PASS"
            reasons: list[str] = []
            checks = {
                "missing_columns": missing,
                "unexpected_columns": unexpected,
                "duplicate_games": duplicate_games,
                "all_null_features": all_null_features,
                "constant_features": constant_features,
                "manifest_metadata_matches": manifest_metadata_matches,
                "manifest_targets_match": manifest_targets_match,
                "feature_count_matches": feature_count_matches,
                "market_features_present": market_features_present,
            }
            for key, value in checks.items():
                bad = (
                    bool(value)
                    if key
                    in {
                        "missing_columns",
                        "unexpected_columns",
                        "duplicate_games",
                        "all_null_features",
                        "constant_features",
                    }
                    else not bool(value)
                )
                if bad:
                    status = "FAIL"
                    reasons.append(key)

            rows.append(
                {
                    "market": market,
                    "variant": variant,
                    "rows": len(frame),
                    "total_columns": len(frame.columns),
                    "feature_count": len(features),
                    "expected_feature_count": expected_feature_count,
                    "target_count": len(manifest["target_columns"]),
                    "metadata_count": len(manifest["metadata_columns"]),
                    "duplicate_game_ids": duplicate_games,
                    "missing_columns": json.dumps(missing),
                    "unexpected_columns": json.dumps(unexpected),
                    "all_null_features": json.dumps(all_null_features),
                    "constant_features": json.dumps(constant_features),
                    "status": status,
                    "failure_reasons": ",".join(reasons),
                }
            )
            if status != "PASS":
                failures.append(f"{market}/{variant}: {', '.join(reasons)}")

    review = pd.DataFrame(rows)
    review.to_csv(output_path / "compact_schema_review.csv", index=False)
    (output_path / "compact_schema_contract.json").write_text(
        json.dumps(
            {
                market: {
                    "metadata_columns": list(contract.metadata_columns),
                    "target_columns": list(contract.target_columns),
                    "football_feature_count": contract.football_feature_count,
                    "market_feature_count": contract.market_feature_count,
                    "market_features": list(contract.market_features),
                }
                for market, contract in SCHEMA_CONTRACTS.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise ValueError("Compact schema review failed: " + " | ".join(failures))
    return review


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _same_numeric(
    observed: pd.Series,
    expected: pd.Series,
    *,
    tolerance: float,
) -> pd.Series:
    observed_numeric = _numeric(observed)
    expected_numeric = _numeric(expected)
    both_missing = observed_numeric.isna() & expected_numeric.isna()
    both_present_close = (
        observed_numeric.notna()
        & expected_numeric.notna()
        & np.isclose(
            observed_numeric.to_numpy(float),
            expected_numeric.to_numpy(float),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
    )
    return pd.Series(
        both_missing.to_numpy() | both_present_close,
        index=observed.index,
    )


def _record_mismatches(
    failures: list[pd.DataFrame],
    frame: pd.DataFrame,
    *,
    market: str,
    check: str,
    observed_column: str,
    expected: pd.Series,
    tolerance: float,
) -> int:
    matches = _same_numeric(
        frame[observed_column],
        expected,
        tolerance=tolerance,
    )
    mismatch = ~matches
    if not mismatch.any():
        return 0

    detail_columns = [
        column
        for column in (
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_home_margin",
            "target_total_points",
            "market_home_spread",
            "market_total_line",
            "market_implied_margin",
            observed_column,
        )
        if column in frame.columns
    ]
    detail = frame.loc[mismatch, detail_columns].copy()
    detail["market"] = market
    detail["check"] = check
    detail["observed_column"] = observed_column
    detail["expected_value"] = expected.loc[mismatch].to_numpy()
    failures.append(detail)
    return int(mismatch.sum())


def _american_implied_probability(values: pd.Series) -> pd.Series:
    odds = _numeric(values)
    probability = pd.Series(np.nan, index=odds.index, dtype=float)
    positive = odds.gt(0)
    negative_or_zero = odds.le(0) & odds.notna()
    probability.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)
    probability.loc[negative_or_zero] = (
        -odds.loc[negative_or_zero]
        / (-odds.loc[negative_or_zero] + 100.0)
    )
    return probability


def _devig(first: pd.Series, second: pd.Series) -> pd.Series:
    first_raw = _american_implied_probability(first)
    second_raw = _american_implied_probability(second)
    return first_raw / (first_raw + second_raw)


def audit_compact_targets(
    compact_root: str | Path,
    output_root: str | Path,
    *,
    warehouse_path: str | Path | None = None,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compact_path = Path(compact_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failure_frames: list[pd.DataFrame] = []

    moneyline = pd.read_parquet(
        compact_path / "pregame_moneyline_market_augmented.parquet"
    )
    margin = _numeric(moneyline["target_home_margin"])
    expected_tie = margin.eq(0).astype(float)
    expected_home_win = pd.Series(
        np.where(
            margin.gt(0),
            1.0,
            np.where(margin.lt(0), 0.0, np.nan),
        ),
        index=moneyline.index,
    )
    ml_checks = {
        "home_win_from_margin": _record_mismatches(
            failure_frames,
            moneyline,
            market="pregame_moneyline",
            check="home_win_from_margin",
            observed_column="target_home_win",
            expected=expected_home_win,
            tolerance=tolerance,
        ),
        "tie_from_margin": _record_mismatches(
            failure_frames,
            moneyline,
            market="pregame_moneyline",
            check="tie_from_margin",
            observed_column="target_tie",
            expected=expected_tie,
            tolerance=tolerance,
        ),
    }
    summary_rows.append(
        {
            "market": "pregame_moneyline",
            "rows": len(moneyline),
            "push_or_tie_rows": int(expected_tie.sum()),
            "mismatches": sum(ml_checks.values()),
            "checks": json.dumps(ml_checks, sort_keys=True),
        }
    )

    ats = pd.read_parquet(
        compact_path / "pregame_ats_market_augmented.parquet"
    )
    home_margin = _numeric(ats["target_home_margin"])
    home_spread = _numeric(ats["market_home_spread"])
    ats_residual = home_margin + home_spread
    expected_push = ats_residual.eq(0).astype(float)
    expected_cover = pd.Series(
        np.where(
            ats_residual.gt(0),
            1.0,
            np.where(ats_residual.lt(0), 0.0, np.nan),
        ),
        index=ats.index,
    )
    ats_checks = {
        "market_implied_margin_equals_negative_home_spread": _record_mismatches(
            failure_frames,
            ats,
            market="pregame_ats",
            check="market_implied_margin_equals_negative_home_spread",
            observed_column="market_implied_margin",
            expected=-home_spread,
            tolerance=tolerance,
        ),
        "margin_residual_equals_margin_plus_home_spread": _record_mismatches(
            failure_frames,
            ats,
            market="pregame_ats",
            check="margin_residual_equals_margin_plus_home_spread",
            observed_column="target_margin_residual",
            expected=ats_residual,
            tolerance=tolerance,
        ),
        "home_cover_from_residual": _record_mismatches(
            failure_frames,
            ats,
            market="pregame_ats",
            check="home_cover_from_residual",
            observed_column="target_home_cover",
            expected=expected_cover,
            tolerance=tolerance,
        ),
        "ats_push_from_residual": _record_mismatches(
            failure_frames,
            ats,
            market="pregame_ats",
            check="ats_push_from_residual",
            observed_column="target_ats_push",
            expected=expected_push,
            tolerance=tolerance,
        ),
    }
    summary_rows.append(
        {
            "market": "pregame_ats",
            "rows": len(ats),
            "push_or_tie_rows": int(expected_push.sum()),
            "mismatches": sum(ats_checks.values()),
            "checks": json.dumps(ats_checks, sort_keys=True),
        }
    )

    total = pd.read_parquet(
        compact_path / "pregame_total_market_augmented.parquet"
    )
    total_points = _numeric(total["target_total_points"])
    total_line = _numeric(total["market_total_line"])
    total_residual = total_points - total_line
    expected_total_push = total_residual.eq(0).astype(float)
    expected_over = pd.Series(
        np.where(
            total_residual.gt(0),
            1.0,
            np.where(total_residual.lt(0), 0.0, np.nan),
        ),
        index=total.index,
    )
    total_checks = {
        "total_residual_equals_points_minus_line": _record_mismatches(
            failure_frames,
            total,
            market="pregame_total",
            check="total_residual_equals_points_minus_line",
            observed_column="target_total_residual",
            expected=total_residual,
            tolerance=tolerance,
        ),
        "over_from_total_residual": _record_mismatches(
            failure_frames,
            total,
            market="pregame_total",
            check="over_from_total_residual",
            observed_column="target_over",
            expected=expected_over,
            tolerance=tolerance,
        ),
        "total_push_from_residual": _record_mismatches(
            failure_frames,
            total,
            market="pregame_total",
            check="total_push_from_residual",
            observed_column="target_total_push",
            expected=expected_total_push,
            tolerance=tolerance,
        ),
    }
    summary_rows.append(
        {
            "market": "pregame_total",
            "rows": len(total),
            "push_or_tie_rows": int(expected_total_push.sum()),
            "mismatches": sum(total_checks.values()),
            "checks": json.dumps(total_checks, sort_keys=True),
        }
    )

    probability_checks = [
        (moneyline, "pregame_moneyline", "market_home_ml_novig_prob"),
        (ats, "pregame_ats", "market_home_cover_novig_prob"),
        (total, "pregame_total", "market_over_novig_prob"),
    ]
    for frame, market, column in probability_checks:
        values = _numeric(frame[column])
        invalid = values.isna() | values.le(0) | values.ge(1)
        if invalid.any():
            detail = frame.loc[
                invalid,
                [
                    column_name
                    for column_name in (
                        "game_id",
                        "season",
                        "home_team_id",
                        "away_team_id",
                        column,
                    )
                    if column_name in frame.columns
                ],
            ].copy()
            detail["market"] = market
            detail["check"] = "no_vig_probability_in_open_unit_interval"
            detail["observed_column"] = column
            detail["expected_value"] = np.nan
            failure_frames.append(detail)

    if warehouse_path is None:
        inferred = compact_path.parent / "modeling_matrix_stage2_qb.parquet"
        warehouse = inferred if inferred.exists() else None
    else:
        requested = Path(warehouse_path).expanduser().resolve()
        warehouse = requested if requested.exists() else None

    source_mapping_status = "SKIPPED_NO_WAREHOUSE"
    source_mapping_mismatches = 0
    if warehouse is not None:
        source = pd.read_parquet(warehouse)
        required_source = {
            "game_id",
            "home_moneyline_reference",
            "away_moneyline_reference",
            "home_spread_reference",
            "home_spread_price_reference",
            "away_spread_price_reference",
            "total_line_reference",
            "over_price_reference",
            "under_price_reference",
        }
        missing_source = sorted(required_source - set(source.columns))
        if missing_source:
            source_mapping_status = "SKIPPED_MISSING_SOURCE_COLUMNS"
            summary_rows.append(
                {
                    "market": "source_market_mapping",
                    "rows": len(source),
                    "push_or_tie_rows": np.nan,
                    "mismatches": np.nan,
                    "checks": json.dumps(
                        {"missing_source_columns": missing_source},
                        sort_keys=True,
                    ),
                }
            )
        else:
            source_mapping_status = "PASS"
            source = source.loc[:, sorted(required_source)].copy()
            comparisons = [
                (
                    moneyline,
                    "pregame_moneyline",
                    "market_home_ml_novig_prob",
                    _devig(
                        source["home_moneyline_reference"],
                        source["away_moneyline_reference"],
                    ),
                ),
                (
                    ats,
                    "pregame_ats",
                    "market_home_spread",
                    _numeric(source["home_spread_reference"]),
                ),
                (
                    ats,
                    "pregame_ats",
                    "market_home_cover_novig_prob",
                    _devig(
                        source["home_spread_price_reference"],
                        source["away_spread_price_reference"],
                    ),
                ),
                (
                    total,
                    "pregame_total",
                    "market_total_line",
                    _numeric(source["total_line_reference"]),
                ),
                (
                    total,
                    "pregame_total",
                    "market_over_novig_prob",
                    _devig(
                        source["over_price_reference"],
                        source["under_price_reference"],
                    ),
                ),
            ]
            source_indexed = source.set_index("game_id")
            for frame, market, observed_column, expected_source in comparisons:
                expected_lookup = pd.Series(
                    expected_source.to_numpy(),
                    index=source["game_id"].to_numpy(),
                )
                expected = frame["game_id"].map(expected_lookup)
                count = _record_mismatches(
                    failure_frames,
                    frame,
                    market=market,
                    check=f"{observed_column}_matches_stage2_source",
                    observed_column=observed_column,
                    expected=expected,
                    tolerance=tolerance,
                )
                source_mapping_mismatches += count
            if source_mapping_mismatches:
                source_mapping_status = "FAIL"
            summary_rows.append(
                {
                    "market": "source_market_mapping",
                    "rows": len(source_indexed),
                    "push_or_tie_rows": np.nan,
                    "mismatches": source_mapping_mismatches,
                    "checks": json.dumps(
                        {
                            "status": source_mapping_status,
                            "warehouse": str(warehouse),
                        },
                        sort_keys=True,
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "market",
                "check",
                "observed_column",
                "expected_value",
            ]
        )
    )
    summary.to_csv(output_path / "target_audit.csv", index=False)
    failures.to_csv(output_path / "target_audit_failures.csv", index=False)

    total_mismatches = int(
        pd.to_numeric(summary["mismatches"], errors="coerce").fillna(0).sum()
    )
    if total_mismatches or not failures.empty:
        raise ValueError(
            f"Target/source audit failed with {max(total_mismatches, len(failures))} "
            "mismatch rows. See target_audit_failures.csv."
        )
    return summary, failures
