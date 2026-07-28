from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "horizon",
    "requested_snapshot_utc",
    "returned_snapshot_utc",
    "event_id",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market",
    "market_last_update",
    "outcome_key",
    "price_decimal",
    "price_american",
    "point",
}

BOOK_GROUP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "horizon",
    "requested_snapshot_utc",
    "returned_snapshot_utc",
    "bookmaker_key",
    "bookmaker_title",
    "market",
]

SNAPSHOT_MARKET_COLUMNS = [
    "game_id",
    "season",
    "week",
    "horizon",
    "requested_snapshot_utc",
    "returned_snapshot_utc",
    "market",
]

OUTCOME_ALIASES = {
    "home": "home",
    "home_team": "home",
    "homewin": "home",
    "home_win": "home",
    "1": "home",
    "away": "away",
    "away_team": "away",
    "awaywin": "away",
    "away_win": "away",
    "2": "away",
    "tie": "tie",
    "draw": "tie",
    "x": "tie",
    "over": "over",
    "o": "over",
    "under": "under",
    "u": "under",
}


@dataclass(frozen=True)
class QuoteFreshnessConfig:
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023)
    closing_horizon: str = "closing_t10"
    maximum_snapshot_lag_minutes: float = 5.0
    maximum_quote_age_minutes: float = 15.0
    minimum_books: int = 3
    minimum_books_for_outlier_detection: int = 5
    robust_z_threshold: float = 6.0
    zero_mad_line_tolerance: float = 1.5
    zero_mad_probability_tolerance: float = 0.10
    spread_pair_tolerance: float = 0.25
    total_pair_tolerance: float = 0.25
    future_timestamp_tolerance_seconds: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "QuoteFreshnessConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["seasons"] = tuple(payload["seasons"])
        return cls(**payload)


def _normalize_outcome(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return OUTCOME_ALIASES.get(key, key)


def _join_reasons(reasons: Iterable[str]) -> str:
    return "|".join(sorted(set(reason for reason in reasons if reason)))


def _robust_outlier_mask(
    values: pd.Series,
    *,
    threshold: float,
    zero_mad_tolerance: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(False, index=values.index)
    usable = numeric.dropna()

    if len(usable) < 2:
        return result

    median = float(usable.median())
    absolute_deviation = (usable - median).abs()
    mad = float(absolute_deviation.median())

    if math.isclose(mad, 0.0, abs_tol=1e-12):
        result.loc[usable.index] = (
            absolute_deviation > zero_mad_tolerance
        )
    else:
        robust_z = 0.6744897501960817 * absolute_deviation / mad
        result.loc[usable.index] = robust_z > threshold

    return result


def _required_outcomes(market: str) -> set[str]:
    if market in {"h2h", "spreads"}:
        return {"home", "away"}
    if market == "totals":
        return {"over", "under"}
    raise ValueError(f"Unsupported market: {market}")


def _book_market_summary(
    group: pd.DataFrame,
    config: QuoteFreshnessConfig,
) -> dict[str, Any]:
    first = group.iloc[0]
    market = str(first["market"])
    reasons: list[str] = []

    outcomes = group["normalized_outcome"].tolist()
    outcome_set = set(outcomes)

    if group["normalized_outcome"].duplicated().any():
        reasons.append("DUPLICATE_OUTCOME")

    unknown_outcomes = sorted(
        outcome_set - {"home", "away", "tie", "over", "under"}
    )
    if unknown_outcomes:
        reasons.append("UNKNOWN_OUTCOME")

    required = _required_outcomes(market)
    if not required <= outcome_set:
        reasons.append("INCOMPLETE_MARKET")

    if group["invalid_price"].any():
        reasons.append("INVALID_PRICE")

    if group["future_market_update"].any():
        reasons.append("FUTURE_MARKET_UPDATE")

    if group["future_bookmaker_update"].any():
        reasons.append("FUTURE_BOOKMAKER_UPDATE")

    if group["stale_market_quote"].any():
        reasons.append("STALE_MARKET_QUOTE")

    if group["invalid_snapshot_lag"].any():
        reasons.append("INVALID_SNAPSHOT_LAG")

    point_reference = math.nan
    first_probability = math.nan
    hold = math.nan
    tie_available = "tie" in outcome_set

    implied = {
        row.normalized_outcome: 1.0 / float(row.price_decimal)
        for row in group.itertuples(index=False)
        if (
            row.normalized_outcome
            in {"home", "away", "tie", "over", "under"}
            and float(row.price_decimal) > 1.0
        )
    }

    if not reasons or set(reasons) <= {
        "STALE_MARKET_QUOTE",
        "FUTURE_MARKET_UPDATE",
        "FUTURE_BOOKMAKER_UPDATE",
        "INVALID_SNAPSHOT_LAG",
    }:
        if required <= set(implied):
            hold = float(sum(implied.values()) - 1.0)

            if market == "h2h":
                denominator = implied["home"] + implied["away"]
                if denominator <= 0:
                    reasons.append("INVALID_IMPLIED_PROBABILITY")
                else:
                    first_probability = implied["home"] / denominator

            elif market == "spreads":
                denominator = implied["home"] + implied["away"]
                if denominator <= 0:
                    reasons.append("INVALID_IMPLIED_PROBABILITY")
                else:
                    first_probability = implied["home"] / denominator

                home_points = pd.to_numeric(
                    group.loc[
                        group["normalized_outcome"].eq("home"),
                        "point",
                    ],
                    errors="coerce",
                )
                away_points = pd.to_numeric(
                    group.loc[
                        group["normalized_outcome"].eq("away"),
                        "point",
                    ],
                    errors="coerce",
                )

                if (
                    len(home_points) != 1
                    or len(away_points) != 1
                    or home_points.isna().any()
                    or away_points.isna().any()
                ):
                    reasons.append("INVALID_SPREAD_POINTS")
                else:
                    home_point = float(home_points.iloc[0])
                    away_point = float(away_points.iloc[0])
                    if (
                        abs(home_point + away_point)
                        > config.spread_pair_tolerance
                    ):
                        reasons.append("INCONSISTENT_SPREAD_PAIR")
                    point_reference = home_point

            elif market == "totals":
                denominator = implied["over"] + implied["under"]
                if denominator <= 0:
                    reasons.append("INVALID_IMPLIED_PROBABILITY")
                else:
                    first_probability = implied["over"] / denominator

                over_points = pd.to_numeric(
                    group.loc[
                        group["normalized_outcome"].eq("over"),
                        "point",
                    ],
                    errors="coerce",
                )
                under_points = pd.to_numeric(
                    group.loc[
                        group["normalized_outcome"].eq("under"),
                        "point",
                    ],
                    errors="coerce",
                )

                if (
                    len(over_points) != 1
                    or len(under_points) != 1
                    or over_points.isna().any()
                    or under_points.isna().any()
                ):
                    reasons.append("INVALID_TOTAL_POINTS")
                else:
                    over_point = float(over_points.iloc[0])
                    under_point = float(under_points.iloc[0])
                    if (
                        abs(over_point - under_point)
                        > config.total_pair_tolerance
                    ):
                        reasons.append("INCONSISTENT_TOTAL_PAIR")
                    point_reference = over_point

    output = {
        column: first[column]
        for column in BOOK_GROUP_COLUMNS
    }
    output.update(
        {
            "event_id": first["event_id"],
            "quote_rows": int(len(group)),
            "outcome_count": int(group["normalized_outcome"].nunique()),
            "tie_available": bool(tie_available),
            "maximum_market_quote_age_minutes": float(
                group["market_quote_age_minutes"].max()
            ),
            "maximum_bookmaker_quote_age_minutes": float(
                group["bookmaker_quote_age_minutes"].max()
            ),
            "snapshot_lag_minutes": float(
                group["snapshot_lag_minutes"].max()
            ),
            "point_reference": point_reference,
            "first_novig_probability": first_probability,
            "hold": hold,
            "pre_outlier_eligible": not reasons,
            "pre_outlier_exclusion_reason": _join_reasons(reasons),
        }
    )
    return output


def prepare_quote_rows(
    quotes: pd.DataFrame,
    config: QuoteFreshnessConfig,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(quotes.columns))
    if missing:
        raise ValueError(f"Bookmaker quotes missing columns: {missing}")

    frame = quotes[
        quotes["season"].isin(config.seasons)
    ].copy()

    for column in (
        "kickoff_utc",
        "requested_snapshot_utc",
        "returned_snapshot_utc",
        "bookmaker_last_update",
        "market_last_update",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
            errors="raise",
        )

    frame["price_decimal"] = pd.to_numeric(
        frame["price_decimal"],
        errors="coerce",
    )
    frame["point"] = pd.to_numeric(
        frame["point"],
        errors="coerce",
    )
    frame["normalized_outcome"] = frame["outcome_key"].map(
        _normalize_outcome
    )

    frame["snapshot_lag_minutes"] = (
        frame["requested_snapshot_utc"]
        - frame["returned_snapshot_utc"]
    ).dt.total_seconds() / 60.0

    frame["market_quote_age_minutes"] = (
        frame["returned_snapshot_utc"]
        - frame["market_last_update"]
    ).dt.total_seconds() / 60.0

    frame["bookmaker_quote_age_minutes"] = (
        frame["returned_snapshot_utc"]
        - frame["bookmaker_last_update"]
    ).dt.total_seconds() / 60.0

    future_tolerance_minutes = (
        config.future_timestamp_tolerance_seconds / 60.0
    )

    frame["future_market_update"] = (
        frame["market_quote_age_minutes"]
        < -future_tolerance_minutes
    )
    frame["future_bookmaker_update"] = (
        frame["bookmaker_quote_age_minutes"]
        < -future_tolerance_minutes
    )
    frame["stale_market_quote"] = (
        frame["market_quote_age_minutes"]
        > config.maximum_quote_age_minutes
    )
    frame["invalid_snapshot_lag"] = (
        (frame["snapshot_lag_minutes"] < -future_tolerance_minutes)
        | (
            frame["snapshot_lag_minutes"]
            > config.maximum_snapshot_lag_minutes
        )
    )
    frame["invalid_price"] = (
        frame["price_decimal"].isna()
        | ~np.isfinite(frame["price_decimal"])
        | (frame["price_decimal"] <= 1.0)
    )

    return frame


def build_book_market_audit(
    prepared_quotes: pd.DataFrame,
    config: QuoteFreshnessConfig,
) -> pd.DataFrame:
    summaries = [
        _book_market_summary(group, config)
        for _, group in prepared_quotes.groupby(
            BOOK_GROUP_COLUMNS,
            sort=False,
            dropna=False,
        )
    ]
    books = pd.DataFrame(summaries)

    books["line_outlier"] = False
    books["probability_outlier"] = False

    for _, index in books.groupby(
        SNAPSHOT_MARKET_COLUMNS,
        sort=False,
        dropna=False,
    ).groups.items():
        eligible_index = books.loc[index].index[
            books.loc[index, "pre_outlier_eligible"]
        ]

        if len(eligible_index) < config.minimum_books_for_outlier_detection:
            continue

        probability_mask = _robust_outlier_mask(
            books.loc[eligible_index, "first_novig_probability"],
            threshold=config.robust_z_threshold,
            zero_mad_tolerance=
                config.zero_mad_probability_tolerance,
        )
        books.loc[
            probability_mask[probability_mask].index,
            "probability_outlier",
        ] = True

        market = str(books.loc[eligible_index[0], "market"])
        if market in {"spreads", "totals"}:
            line_mask = _robust_outlier_mask(
                books.loc[eligible_index, "point_reference"],
                threshold=config.robust_z_threshold,
                zero_mad_tolerance=
                    config.zero_mad_line_tolerance,
            )
            books.loc[
                line_mask[line_mask].index,
                "line_outlier",
            ] = True

    def final_reason(row: pd.Series) -> str:
        reasons = [
            value
            for value in str(
                row["pre_outlier_exclusion_reason"]
            ).split("|")
            if value
        ]
        if bool(row["line_outlier"]):
            reasons.append("LINE_OUTLIER")
        if bool(row["probability_outlier"]):
            reasons.append("PROBABILITY_OUTLIER")
        return _join_reasons(reasons)

    books["exclusion_reason"] = books.apply(
        final_reason,
        axis=1,
    )
    books["eligible"] = books["exclusion_reason"].eq("")
    return books


def build_freshness_consensus(
    book_audit: pd.DataFrame,
    config: QuoteFreshnessConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for key, group in book_audit.groupby(
        SNAPSHOT_MARKET_COLUMNS,
        sort=False,
        dropna=False,
    ):
        eligible = group[group["eligible"]].copy()
        raw_books = int(group["bookmaker_key"].nunique())
        eligible_books = int(
            eligible["bookmaker_key"].nunique()
        )

        if eligible_books < config.minimum_books:
            continue

        key_values = dict(zip(SNAPSHOT_MARKET_COLUMNS, key))
        market = str(key_values["market"])
        line_values = pd.to_numeric(
            eligible["point_reference"],
            errors="coerce",
        )
        probability_values = pd.to_numeric(
            eligible["first_novig_probability"],
            errors="raise",
        )
        hold_values = pd.to_numeric(
            eligible["hold"],
            errors="raise",
        )

        rows.append(
            {
                **key_values,
                "eligible_books": eligible_books,
                "raw_books": raw_books,
                "excluded_books": raw_books - eligible_books,
                "consensus_line": (
                    math.nan
                    if market == "h2h"
                    else float(line_values.median())
                ),
                "line_sd": (
                    math.nan
                    if market == "h2h"
                    else float(line_values.std(ddof=0))
                ),
                "consensus_home_or_over_novig_probability":
                    float(probability_values.median()),
                "probability_sd": float(
                    probability_values.std(ddof=0)
                ),
                "median_hold": float(hold_values.median()),
                "maximum_market_quote_age_minutes": float(
                    eligible[
                        "maximum_market_quote_age_minutes"
                    ].max()
                ),
                "median_market_quote_age_minutes": float(
                    eligible[
                        "maximum_market_quote_age_minutes"
                    ].median()
                ),
                "tie_books": int(
                    eligible["tie_available"].sum()
                )
                if market == "h2h"
                else 0,
            }
        )

    output_columns = (
        SNAPSHOT_MARKET_COLUMNS
        + [
            "eligible_books",
            "raw_books",
            "excluded_books",
            "consensus_line",
            "line_sd",
            "consensus_home_or_over_novig_probability",
            "probability_sd",
            "median_hold",
            "maximum_market_quote_age_minutes",
            "median_market_quote_age_minutes",
            "tie_books",
        ]
    )

    if not rows:
        return pd.DataFrame(columns=output_columns)

    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values(
            ["season", "week", "game_id", "horizon", "market"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def compare_consensus(
    current: pd.DataFrame,
    qualified: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["game_id", "horizon", "market"]
    required = {
        *keys,
        "eligible_books",
        "consensus_line",
        "line_sd",
        "consensus_home_or_over_novig_probability",
        "probability_sd",
        "median_hold",
    }
    missing = sorted(required - set(current.columns))
    if missing:
        raise ValueError(
            f"Current consensus missing columns: {missing}"
        )

    old = current[list(required)].copy().rename(
        columns={
            column: f"original_{column}"
            for column in required
            if column not in keys
        }
    )
    new = qualified[
        keys
        + [
            "eligible_books",
            "consensus_line",
            "line_sd",
            "consensus_home_or_over_novig_probability",
            "probability_sd",
            "median_hold",
            "maximum_market_quote_age_minutes",
        ]
    ].copy().rename(
        columns={
            column: f"qualified_{column}"
            for column in (
                "eligible_books",
                "consensus_line",
                "line_sd",
                "consensus_home_or_over_novig_probability",
                "probability_sd",
                "median_hold",
                "maximum_market_quote_age_minutes",
            )
        }
    )

    comparison = old.merge(
        new,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    comparison["book_count_delta"] = (
        comparison["qualified_eligible_books"]
        - comparison["original_eligible_books"]
    )
    comparison["line_delta"] = (
        comparison["qualified_consensus_line"]
        - comparison["original_consensus_line"]
    )
    comparison["probability_delta"] = (
        comparison[
            "qualified_consensus_home_or_over_novig_probability"
        ]
        - comparison[
            "original_consensus_home_or_over_novig_probability"
        ]
    )
    comparison["hold_delta"] = (
        comparison["qualified_median_hold"]
        - comparison["original_median_hold"]
    )
    return comparison


def run_quote_freshness_audit(
    *,
    quotes_path: str | Path,
    current_consensus_path: str | Path,
    output_root: str | Path,
    config: QuoteFreshnessConfig,
) -> dict[str, pd.DataFrame]:
    quotes_path = Path(quotes_path).expanduser().resolve()
    current_consensus_path = (
        Path(current_consensus_path).expanduser().resolve()
    )
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_quotes = pd.read_parquet(quotes_path)
    prepared = prepare_quote_rows(raw_quotes, config)
    books = build_book_market_audit(prepared, config)
    consensus = build_freshness_consensus(books, config)

    current_consensus = pd.read_parquet(
        current_consensus_path
    )
    current_consensus = current_consensus[
        current_consensus["season"].isin(config.seasons)
    ].copy()

    comparison = compare_consensus(
        current_consensus,
        consensus,
    )

    status_columns = BOOK_GROUP_COLUMNS + [
        "eligible",
        "exclusion_reason",
        "line_outlier",
        "probability_outlier",
        "maximum_market_quote_age_minutes",
        "maximum_bookmaker_quote_age_minutes",
        "snapshot_lag_minutes",
    ]
    quote_audit = prepared.merge(
        books[status_columns],
        on=BOOK_GROUP_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    exclusions = books[~books["eligible"]].copy()

    coverage_by_book = (
        books.groupby(
            ["bookmaker_key", "bookmaker_title", "market"],
            as_index=False,
        )
        .agg(
            book_market_snapshots=("game_id", "size"),
            eligible_snapshots=("eligible", "sum"),
            stale_snapshots=(
                "exclusion_reason",
                lambda values: values.str.contains(
                    "STALE_MARKET_QUOTE",
                    regex=False,
                ).sum(),
            ),
            future_update_snapshots=(
                "exclusion_reason",
                lambda values: values.str.contains(
                    "FUTURE_",
                    regex=False,
                ).sum(),
            ),
            incomplete_snapshots=(
                "exclusion_reason",
                lambda values: values.str.contains(
                    "INCOMPLETE_MARKET",
                    regex=False,
                ).sum(),
            ),
            outlier_snapshots=(
                "exclusion_reason",
                lambda values: values.str.contains(
                    "OUTLIER",
                    regex=False,
                ).sum(),
            ),
            median_market_quote_age_minutes=(
                "maximum_market_quote_age_minutes",
                "median",
            ),
        )
    )
    coverage_by_book["eligible_rate"] = (
        coverage_by_book["eligible_snapshots"]
        / coverage_by_book["book_market_snapshots"]
    )

    tie_availability = (
        books[books["market"].eq("h2h")]
        .groupby(
            ["season", "horizon", "game_id"],
            as_index=False,
        )
        .agg(
            raw_books=("bookmaker_key", "nunique"),
            eligible_books=("eligible", "sum"),
            raw_tie_books=("tie_available", "sum"),
            eligible_tie_books=(
                "tie_available",
                lambda values: int(
                    values[
                        books.loc[values.index, "eligible"]
                    ].sum()
                ),
            ),
        )
    )

    closing_coverage = (
        consensus[
            consensus["horizon"].eq(config.closing_horizon)
        ]
        .groupby(["season", "market"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            median_books=("eligible_books", "median"),
            minimum_books=("eligible_books", "min"),
            median_quote_age_minutes=(
                "median_market_quote_age_minutes",
                "median",
            ),
            maximum_quote_age_minutes=(
                "maximum_market_quote_age_minutes",
                "max",
            ),
        )
    )

    quote_audit.to_parquet(
        output_root / "quote_freshness_audit.parquet",
        index=False,
    )
    exclusions.to_csv(
        output_root / "quote_exclusions.csv",
        index=False,
    )
    consensus.to_parquet(
        output_root / "freshness_qualified_consensus.parquet",
        index=False,
    )
    comparison.to_csv(
        output_root / "consensus_comparison.csv",
        index=False,
    )
    coverage_by_book.to_csv(
        output_root / "quote_coverage_by_book.csv",
        index=False,
    )
    tie_availability.to_csv(
        output_root / "tie_quote_availability.csv",
        index=False,
    )
    closing_coverage.to_csv(
        output_root / "closing_freshness_coverage.csv",
        index=False,
    )

    manifest = {
        "status": "PASS",
        "config": asdict(config),
        "raw_quote_rows": int(len(raw_quotes)),
        "audited_quote_rows": int(len(prepared)),
        "book_market_snapshots": int(len(books)),
        "eligible_book_market_snapshots": int(
            books["eligible"].sum()
        ),
        "excluded_book_market_snapshots": int(
            (~books["eligible"]).sum()
        ),
        "qualified_consensus_rows": int(len(consensus)),
        "qualified_closing_rows": int(
            consensus["horizon"].eq(
                config.closing_horizon
            ).sum()
        ),
        "future_market_update_rows": int(
            prepared["future_market_update"].sum()
        ),
        "future_bookmaker_update_rows": int(
            prepared["future_bookmaker_update"].sum()
        ),
        "stale_market_quote_rows": int(
            prepared["stale_market_quote"].sum()
        ),
        "new_api_requests": 0,
        "source_quotes": str(quotes_path),
        "source_consensus": str(current_consensus_path),
    }
    (
        output_root / "quote_freshness_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    included = quote_audit[quote_audit["eligible"]]
    if included["future_market_update"].any():
        raise ValueError("Included quotes contain future market updates.")
    if included["stale_market_quote"].any():
        raise ValueError("Included quotes contain stale market updates.")
    if (
        consensus["eligible_books"] < config.minimum_books
    ).any():
        raise ValueError("Consensus contains too few books.")

    return {
        "book_audit": books,
        "consensus": consensus,
        "comparison": comparison,
        "coverage_by_book": coverage_by_book,
        "closing_coverage": closing_coverage,
    }
