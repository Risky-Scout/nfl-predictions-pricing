"""Fix 8 sections 6-7: raw timestamped Odds API bookmaker-history market
reconstruction for the official TUE/FRI card cutoffs.

Reads the RAW per-book snapshot store directly
(``odds_history.{2020_2023,2024_confirmation,2025_final_test}``'s own
``bookmaker_quotes.parquet``) -- never the pre-generated opening/closing_t10
horizon exports (:mod:`nfl_hybrid.odds_history`'s ``consensus_by_horizon.parquet``
/ ``opening_closing_market_features.parquet``, or
:mod:`nfl_hybrid.features.canonical_market_matrices`'s canonical closing_t10
matrices), which are fixed to a different snapshot cadence (opening_7d /
opening_72h / opening_24h / opening_6h / opening_60m / closing_t10) and a
5-minute snapshot-lag rule Fix 8 is explicitly forbidden from reusing.

The de-vig formula and the median-of-devigged-per-book-quotes consensus
method ARE reused verbatim from the existing historical-backfill pipeline
(:func:`nfl_hybrid.odds_history._no_vig`, the same computation
:func:`nfl_hybrid.odds_history.build_consensus` already applies to its own
fixed snapshot buckets) -- applied here to a different row selection: for
each official card cutoff, independently per bookmaker, the freshest
COMPLETE COHERENT quote (both sides present, from the same
game_id/bookmaker_key/returned_snapshot_utc/market_last_update, matching
points, both decimal prices > 1.0) satisfying
``market_last_update <= returned_snapshot_utc <= target_cutoff_utc`` and a
48-hour freshness window, applied BEFORE counting books (a book whose only
qualifying observation is stale never counts toward the >=3-book minimum).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_hybrid.data import external_data
from nfl_hybrid.odds_history import _no_vig

RAW_ODDS_HISTORY_KEYS: tuple[str, ...] = (
    "odds_history.2020_2023",
    "odds_history.2024_confirmation",
    "odds_history.2025_final_test",
)

MARKET_SPREADS = "spreads"
MARKET_TOTALS = "totals"
_VALID_RAW_MARKETS = (MARKET_SPREADS, MARKET_TOTALS)

FRESHNESS_MAX_AGE_HOURS = 48.0
MINIMUM_FRESH_COHERENT_BOOKS = 3
CONSENSUS_METHOD = "median_of_devigged_per_book_quotes_across_eligible_books"

CONSENSUS_COLUMNS = (
    "game_id",
    "target_cutoff_utc",
    "eligible_books",
    "consensus_line",
    "line_sd",
    "consensus_novig_probability",
    "probability_sd",
    "bookmaker_keys",
    "selected_returned_snapshot_timestamps",
    "min_observation_age_hours",
    "max_observation_age_hours",
    "consensus_method",
)


def load_raw_bookmaker_quotes(*, root_override: str | None = None) -> pd.DataFrame:
    """Union of all three raw historical Odds API snapshot stores. Each
    store covers a disjoint season range (2020-2023 / 2024 confirmation /
    2025 final-test), so game_id collisions across stores are not expected;
    a duplicate (game_id, bookmaker_key, returned_snapshot_utc,
    market_last_update, market, outcome_key) row is still de-duplicated
    defensively below."""
    frames = []
    for key in RAW_ODDS_HISTORY_KEYS:
        odds_dir = external_data.resolve(key, root_override=root_override)
        path = odds_dir / "bookmaker_quotes.parquet"
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["source_store"] = key
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = combined["game_id"].astype(str)
    combined["bookmaker_key"] = combined["bookmaker_key"].astype(str)
    combined["returned_snapshot_utc"] = pd.to_datetime(combined["returned_snapshot_utc"], utc=True, errors="coerce")
    combined["market_last_update"] = pd.to_datetime(combined["market_last_update"], utc=True, errors="coerce")
    dedup_cols = ["game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update", "market", "outcome_key"]
    combined = combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    return combined


def build_coherent_book_observations(quotes: pd.DataFrame, market: str) -> pd.DataFrame:
    """One row per (game_id, bookmaker_key, returned_snapshot_utc,
    market_last_update) COMPLETE COHERENT two-sided quote. A coherent
    spread observation has both home and away outcomes from the same
    snapshot event, points that are exact opposites, and both decimal
    prices > 1.0; a coherent total observation has both over and under from
    the same event, an identical total point, and both decimal prices >
    1.0. Cross-snapshot side pairing (e.g. home from one fetch, away from a
    later one) is structurally impossible here -- both sides must share the
    same ``(game_id, bookmaker_key, returned_snapshot_utc,
    market_last_update)`` group key."""
    if market not in _VALID_RAW_MARKETS:
        raise ValueError(f"Unknown raw market {market!r}; expected one of {_VALID_RAW_MARKETS}.")

    sub = quotes[quotes["market"] == market].copy()
    wanted_outcomes = ("home", "away") if market == MARKET_SPREADS else ("over", "under")
    sub = sub[sub["outcome_key"].isin(wanted_outcomes)]
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update",
                "line", "home_or_over_price_decimal", "away_or_under_price_decimal",
                "home_or_over_novig_probability", "away_or_under_novig_probability", "hold",
            ]
        )

    group_cols = ["game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update"]
    rows: list[dict[str, object]] = []
    for keys, frame in sub.groupby(group_cols, sort=False, dropna=False):
        outcomes = {r.outcome_key: r for r in frame.itertuples()}
        if not set(wanted_outcomes).issubset(outcomes):
            continue
        first_key, second_key = wanted_outcomes
        first, second = outcomes[first_key], outcomes[second_key]
        if not (np.isfinite(first.price_decimal) and np.isfinite(second.price_decimal)):
            continue
        if not (first.price_decimal > 1.0 and second.price_decimal > 1.0):
            continue
        if not (np.isfinite(first.point) and np.isfinite(second.point)):
            continue
        if market == MARKET_SPREADS:
            if not np.isclose(first.point, -second.point, atol=1e-6):
                continue
        else:
            if not np.isclose(first.point, second.point, atol=1e-6):
                continue

        home_p, away_p, hold = _no_vig(float(first.price_decimal), float(second.price_decimal))
        rows.append(
            {
                "game_id": keys[0],
                "bookmaker_key": keys[1],
                "returned_snapshot_utc": keys[2],
                "market_last_update": keys[3],
                "line": float(first.point),
                "home_or_over_price_decimal": float(first.price_decimal),
                "away_or_under_price_decimal": float(second.price_decimal),
                "home_or_over_novig_probability": home_p,
                "away_or_under_novig_probability": away_p,
                "hold": hold,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class MarketReconstructionResult:
    market: str
    consensus: pd.DataFrame
    coverage: dict[str, int]


def reconstruct_market_at_cutoffs(
    coherent: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    market: str,
    max_age_hours: float = FRESHNESS_MAX_AGE_HOURS,
    minimum_books: int = MINIMUM_FRESH_COHERENT_BOOKS,
) -> MarketReconstructionResult:
    """For each row of ``targets`` (``game_id``, ``target_cutoff_utc`` --
    one row per horizon-eligible target, regardless of OOF readiness):
    select, independently per bookmaker, the freshest coherent observation
    satisfying ``market_last_update <= returned_snapshot_utc <=
    target_cutoff_utc`` and ``0 <= age_hours <= max_age_hours`` (age applied
    BEFORE counting books -- a book whose only qualifying observation is
    stale never counts). Require >= ``minimum_books`` distinct bookmakers,
    then take the median line and median per-book no-vig probability across
    those qualifying books (the reused Fix-4/backfill consensus method).

    Returns four explicit reconciliation stage counts (Fix 8 contract
    section 7): ``eligible`` (targets examined), ``coherent_leq_cutoff``
    (targets with >=1 coherent book whose snapshot arrived at or before
    cutoff, ignoring the market_last_update ordering/age checks),
    ``ge3_before_age_filter`` (targets with >=3 distinct qualifying books
    once the market_last_update<=returned_snapshot_utc<=cutoff ordering
    check is applied, BEFORE discarding stale (>48h) observations), and
    ``market_ready`` (the final gate, after also requiring freshness)."""
    targets = targets[["game_id", "target_cutoff_utc"]].copy()
    targets["game_id"] = targets["game_id"].astype(str)
    targets["target_cutoff_utc"] = pd.to_datetime(targets["target_cutoff_utc"], utc=True, errors="raise")
    n_eligible = int(targets["game_id"].nunique())

    if coherent.empty:
        coverage = {
            "eligible": n_eligible, "coherent_leq_cutoff": 0, "ge3_before_age_filter": 0, "market_ready": 0,
            "minimum_books": minimum_books, "max_age_hours": max_age_hours,
        }
        return MarketReconstructionResult(market=market, consensus=pd.DataFrame(columns=CONSENSUS_COLUMNS), coverage=coverage)

    merged = targets.merge(coherent, on="game_id", how="inner")
    merged["age_hours"] = (
        merged["target_cutoff_utc"] - merged["returned_snapshot_utc"]
    ).dt.total_seconds() / 3600.0

    leq_cutoff = merged[merged["returned_snapshot_utc"] <= merged["target_cutoff_utc"]]
    n_coherent_leq_cutoff = int(leq_cutoff["game_id"].nunique())

    ordering_ok = merged[
        (merged["market_last_update"] <= merged["returned_snapshot_utc"])
        & (merged["returned_snapshot_utc"] <= merged["target_cutoff_utc"])
    ].copy()
    ordering_ok = ordering_ok.sort_values(
        ["game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update"], kind="stable"
    )
    freshest_before_age = ordering_ok.groupby(["game_id", "bookmaker_key"], sort=False, as_index=False).tail(1)
    counts_before_age = freshest_before_age.groupby("game_id")["bookmaker_key"].nunique()
    n_ge3_before_age = int((counts_before_age >= minimum_books).sum())

    fresh = ordering_ok[(ordering_ok["age_hours"] >= 0.0) & (ordering_ok["age_hours"] <= max_age_hours)]
    fresh = fresh.sort_values(
        ["game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update"], kind="stable"
    )
    freshest = fresh.groupby(["game_id", "bookmaker_key"], sort=False, as_index=False).tail(1)
    book_counts = freshest.groupby("game_id")["bookmaker_key"].nunique()
    ready_ids = book_counts[book_counts >= minimum_books].index

    ready = freshest[freshest["game_id"].isin(ready_ids)]
    consensus_rows: list[dict[str, object]] = []
    for game_id, frame in ready.groupby("game_id", sort=False):
        consensus_rows.append(
            {
                "game_id": game_id,
                "eligible_books": int(frame["bookmaker_key"].nunique()),
                "consensus_line": float(frame["line"].median()),
                "line_sd": float(frame["line"].std(ddof=0)) if len(frame) > 1 else 0.0,
                "consensus_novig_probability": float(frame["home_or_over_novig_probability"].median()),
                "probability_sd": float(frame["home_or_over_novig_probability"].std(ddof=0)) if len(frame) > 1 else 0.0,
                "bookmaker_keys": tuple(sorted(frame["bookmaker_key"].unique())),
                "selected_returned_snapshot_timestamps": tuple(
                    sorted(str(t) for t in frame["returned_snapshot_utc"].unique())
                ),
                "min_observation_age_hours": float(frame["age_hours"].min()),
                "max_observation_age_hours": float(frame["age_hours"].max()),
                "consensus_method": CONSENSUS_METHOD,
            }
        )
    consensus = pd.DataFrame(consensus_rows, columns=[c for c in CONSENSUS_COLUMNS if c != "target_cutoff_utc"])
    if not consensus.empty:
        consensus = consensus.merge(targets, on="game_id", how="left")
        consensus = consensus[list(CONSENSUS_COLUMNS)]
    else:
        consensus = pd.DataFrame(columns=CONSENSUS_COLUMNS)

    coverage = {
        "eligible": n_eligible,
        "coherent_leq_cutoff": n_coherent_leq_cutoff,
        "ge3_before_age_filter": n_ge3_before_age,
        "market_ready": int(len(consensus)),
        "minimum_books": minimum_books,
        "max_age_hours": max_age_hours,
    }
    return MarketReconstructionResult(market=market, consensus=consensus, coverage=coverage)
