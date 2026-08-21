"""Fix 4 repair, items 2/3/4/15/20: attach the REAL historical T-10 market
line (ATS spread / TOTAL line) actually available at each Fix 3 OOF row's own
``target_cutoff_utc`` -- a derived, auditable join on top of the Fix 3
chronological OOF ledger, never a mutation of it.

Reuses the EXISTING canonical closing_t10 market attachment
(:mod:`nfl_hybrid.features.odds_attachment` /
:mod:`nfl_hybrid.features.canonical_market_matrices`, registered under
``canonical_market.pregame_ats_t10_2020_2023`` /
``canonical_market.pregame_total_t10_2020_2023`` in
:mod:`nfl_hybrid.data.external_data`) rather than inventing a second
consensus rule: that pipeline already builds one T-10 consensus line per game
from the median of de-vigged per-book quotes across eligible books, already
enforces >=3 eligible books and a <=5 minute snapshot lag at build time (see
``OddsAttachmentConfig`` in :mod:`nfl_hybrid.features.odds_attachment`), and
already raises if any row violates either gate -- so every row in the
registered parquet already passed those two checks once. This module does
NOT trust that alone: it independently re-derives eligible_books>=3,
snapshot-lag<=5min, and (the genuinely new invariant Fix 3/Fix 4 introduces)
``market_snapshot_returned_utc <= target_cutoff_utc`` per row, against the
actual Fix 3 ``target_cutoff_utc`` of the game being priced -- not merely
against kickoff, which is a materially looser bound (Fix 3's own
result-availability correction is exactly this same distinction one layer
up).

``market_last_update_utc`` (item 3's third, conditional as-of check) is not a
field the canonical closing_t10 consensus schema carries -- it has
``requested_snapshot_utc`` / ``returned_snapshot_utc`` only, at the
per-book-consensus level, not a per-bookmaker last-update timestamp folded
into the consensus row. This module does not fabricate one; every attached
row's coverage/manifest metadata says so explicitly rather than silently
omitting the check.

Fixed spread convention (item 4), inspected and proven, not assumed --
``market_t10_consensus_line`` for ATS follows exactly the convention already
enforced in :mod:`nfl_hybrid.features.canonical_market_matrices`
(``_add_t10_targets``: ``residual = target_home_margin + line``, home covers
iff ``residual > 0``) and :mod:`nfl_hybrid.pricing.margin_surface` (module
docstring): a NEGATIVE line means the home team is favored (e.g. -3.5 = home
favored by 3.5), and home covers iff ``actual_margin + home_spread > 0`` --
see ``tests/test_chronological_market_attachment.py``'s sign tests, which
independently confirm this against a hand-built example before any real data
is processed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from nfl_hybrid.data import external_data
from nfl_hybrid.evaluation.chronological_calibration import (
    MARKET_ATS,
    MARKET_TOTAL,
    build_raw_probabilities,
)

FORECAST_HORIZON = "kickoff_minus_10_minutes"
MINIMUM_ELIGIBLE_BOOKS = 3
MAXIMUM_SNAPSHOT_LAG_MINUTES = 5.0

# Never a second consensus rule: this is the ONLY place Fix 4 names which
# registered canonical-market dataset backs each market. Both keys are the
# existing closing_t10 attachment (OddsAttachmentConfig's own >=3
# books/<=5min gate already applied once at build time).
_CANONICAL_MARKET_KEYS: dict[str, str] = {
    MARKET_ATS: "canonical_market.pregame_ats_t10_2020_2023",
    MARKET_TOTAL: "canonical_market.pregame_total_t10_2020_2023",
}

CONSENSUS_METHOD = "median_of_devigged_per_book_quotes_across_eligible_books"


@dataclass(frozen=True)
class MarketAttachmentResult:
    """One market's attachment outcome: the qualifying joined rows, plus an
    explicit, non-silent rejection-reason breakdown (item 15) -- nothing here
    is dropped without a counted, named reason."""

    market: str
    attached: pd.DataFrame
    coverage: dict[str, int]
    market_last_update_available: bool = False


def load_canonical_t10_market(market: str, *, root_override: str | None = None) -> pd.DataFrame:
    """Load and normalize the existing canonical closing_t10 ATS or TOTAL
    market matrix. Never invents a second consensus rule -- this is a column
    rename/normalize only, reading the real, already-gated parquet."""
    if market not in _CANONICAL_MARKET_KEYS:
        raise ValueError(
            f"No canonical T-10 market source is registered for market {market!r}; "
            f"supported: {sorted(_CANONICAL_MARKET_KEYS)}."
        )
    path = external_data.resolve(_CANONICAL_MARKET_KEYS[market], root_override=root_override)
    frame = pd.read_parquet(path)

    required = {
        "game_id", "season", "week",
        "market_t10_consensus_line", "market_t10_eligible_books",
        "market_t10_requested_snapshot_utc", "market_t10_returned_snapshot_utc",
        "market_t10_snapshot_lag_minutes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical T-10 {market} market matrix is missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str),
            "season": frame["season"],
            "week": frame["week"],
            "market": market,
            "line": pd.to_numeric(frame["market_t10_consensus_line"], errors="coerce"),
            "market_snapshot_requested_utc": pd.to_datetime(
                frame["market_t10_requested_snapshot_utc"], utc=True, errors="coerce"
            ),
            "market_snapshot_returned_utc": pd.to_datetime(
                frame["market_t10_returned_snapshot_utc"], utc=True, errors="coerce"
            ),
            "market_snapshot_lag_minutes": pd.to_numeric(frame["market_t10_snapshot_lag_minutes"], errors="coerce"),
            "eligible_books": pd.to_numeric(frame["market_t10_eligible_books"], errors="coerce"),
        }
    )
    out["consensus_method"] = CONSENSUS_METHOD
    # Not a field the canonical closing_t10 schema carries -- never
    # fabricated; downstream code/manifests must say so explicitly (see
    # MarketAttachmentResult.market_last_update_available).
    out["market_last_update_utc"] = pd.NaT
    if out["game_id"].duplicated().any():
        raise ValueError(f"Canonical T-10 {market} market matrix has duplicate game_id rows.")
    return out


def assert_primary_markets_certified(primary_markets: Sequence[str]) -> None:
    """Fix 4 item 20 -- the exact regression guard against the original real-
    data-proof drift: a Fix 4 certification run must never be satisfiable by
    MONEYLINE-only evidence. Raises unless BOTH ``MARKET_ATS`` and
    ``MARKET_TOTAL`` are present in ``primary_markets``; MONEYLINE may be
    present alongside them (it's auxiliary, not forbidden), but can never
    stand in for either."""
    markets = set(primary_markets)
    missing = {MARKET_ATS, MARKET_TOTAL} - markets
    if missing:
        raise ValueError(
            "Fix 4 certification requires BOTH ATS and TOTAL in primary_markets "
            f"(got {sorted(markets)}, missing {sorted(missing)}). MONEYLINE-only "
            "evidence can never satisfy Fix 4 acceptance -- this is the exact "
            "regression guard for the original real-data-proof drift (Fix 4 "
            "item 20)."
        )


def attach_market_to_residual_ledger(
    residual_ledger: pd.DataFrame,
    *,
    market: str,
    root_override: str | None = None,
    minimum_eligible_books: int = MINIMUM_ELIGIBLE_BOOKS,
    maximum_snapshot_lag_minutes: float = MAXIMUM_SNAPSHOT_LAG_MINUTES,
) -> MarketAttachmentResult:
    """Join the Fix 3 residual ledger (unmutated -- this returns a NEW,
    derived frame, never writes back to ``residual_ledger``) to the real
    closing_t10 market line, independently re-verifying every Fix 4 gate
    per row rather than trusting the upstream build once, and reporting an
    explicit, counted rejection reason (item 15) for every row that does not
    qualify."""
    if market not in (MARKET_ATS, MARKET_TOTAL):
        raise ValueError(f"attach_market_to_residual_ledger supports ATS/TOTAL only, got {market!r}.")

    market_frame = load_canonical_t10_market(market, root_override=root_override)

    ledger = residual_ledger.copy()
    ledger["game_id"] = ledger["game_id"].astype(str)
    if ledger["game_id"].duplicated().any():
        raise ValueError("residual_ledger has duplicate game_id rows.")
    target_cutoff = pd.to_datetime(ledger["target_cutoff_utc"], utc=True, errors="raise")
    ledger = ledger.assign(target_cutoff_utc=target_cutoff)

    n_examined = len(ledger)
    # season/week are already authoritative on the Fix 3 ledger; drop the
    # market frame's copies rather than let pandas suffix them into
    # season_x/season_y (which would silently break every downstream column
    # reference, including build_raw_probabilities' own required-columns
    # check).
    market_for_merge = market_frame.drop(columns=["season", "week"])
    joined = ledger.merge(market_for_merge, on="game_id", how="left", validate="one_to_one")

    has_match = joined["market"].notna()
    n_no_match = int((~has_match).sum())

    too_few_books = has_match & (joined["eligible_books"] < minimum_eligible_books)
    n_too_few_books = int(too_few_books.sum())

    lag = joined["market_snapshot_lag_minutes"]
    lag_out_of_bounds = has_match & ((lag < 0) | (lag > maximum_snapshot_lag_minutes) | lag.isna())
    n_lag_out_of_bounds = int(lag_out_of_bounds.sum())

    returned = joined["market_snapshot_returned_utc"]
    snapshot_after_cutoff = has_match & (returned.isna() | (returned > joined["target_cutoff_utc"]))
    n_snapshot_after_cutoff = int(snapshot_after_cutoff.sum())

    line_missing = has_match & joined["line"].isna()
    n_line_missing = int(line_missing.sum())

    qualifies = (
        has_match
        & (joined["eligible_books"] >= minimum_eligible_books)
        & (lag >= 0)
        & (lag <= maximum_snapshot_lag_minutes)
        & returned.notna()
        & (returned <= joined["target_cutoff_utc"])
        & joined["line"].notna()
    )
    n_qualifies = int(qualifies.sum())

    coverage = {
        "n_examined": n_examined,
        "rejected_no_market_match": n_no_match,
        "rejected_too_few_books": n_too_few_books,
        "rejected_stale_or_invalid_snapshot_lag": n_lag_out_of_bounds,
        "rejected_snapshot_after_target_cutoff": n_snapshot_after_cutoff,
        "rejected_missing_line": n_line_missing,
        "qualifies_valid_t10_line": n_qualifies,
    }

    attached = joined[qualifies].copy()
    attached["forecast_horizon"] = FORECAST_HORIZON
    attached = attached.reset_index(drop=True)

    return MarketAttachmentResult(market=market, attached=attached, coverage=coverage)


# game_id / provenance columns copied from the market-attachment join onto
# the build_raw_probabilities() output, keyed by row order (both frames share
# the same row order -- attached.reset_index(drop=True) above, and
# build_raw_probabilities preserves residual_ledger's row order verbatim).
# "forecast_horizon" is NOT included here -- build_raw_probabilities() already
# stamps it onto its own output; duplicating it here would collide on concat.
_PROVENANCE_COLUMNS = (
    "market",
    "market_snapshot_requested_utc",
    "market_snapshot_returned_utc",
    "market_snapshot_lag_minutes",
    "eligible_books",
    "consensus_method",
    "market_last_update_utc",
    "line",
)


def build_market_raw_ledger(
    residual_ledger: pd.DataFrame,
    *,
    market: str,
    root_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fix 4 items 7/8: the real ATS or TOTAL raw ledger -- attach the real
    T-10 line (:func:`attach_market_to_residual_ledger`), then price it
    through the one Fix 4 pricing engine
    (:func:`nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities`,
    ``market=market, threshold=<the real attached line>``). Returns
    ``(raw_ledger, coverage)`` -- coverage is the same explicit rejection-
    reason breakdown as :func:`attach_market_to_residual_ledger`."""
    result = attach_market_to_residual_ledger(residual_ledger, market=market, root_override=root_override)
    attached = result.attached

    if attached.empty:
        return attached, result.coverage

    threshold = attached["line"].to_numpy(float)
    raw = build_raw_probabilities(
        attached, market=market, forecast_horizon=FORECAST_HORIZON, threshold=threshold
    )
    provenance = attached[list(_PROVENANCE_COLUMNS)].reset_index(drop=True)
    combined = pd.concat([raw.reset_index(drop=True), provenance], axis=1)
    line_column = "home_spread" if market == MARKET_ATS else "total_line"
    combined = combined.rename(columns={"line": line_column})
    return combined, result.coverage


def load_canonical_t10_settlement_columns(market: str, *, root_override: str | None = None) -> pd.DataFrame:
    """Certification-blocker repair item 2 (A/C/D/E/6): the canonical T10
    settlement/team-identity/novig columns needed for the leakage/settlement
    sanity audit -- reads the EXACT SAME registered parquet
    :func:`load_canonical_t10_market` reads (never a second consensus rule,
    never rebuilt historical odds). ``canonical_target_cover_or_over`` /
    ``canonical_target_push`` are :mod:`nfl_hybrid.features.canonical_market_matrices`'s
    own settlement targets (``target_t10_home_cover``/``target_t10_ats_push``
    for ATS, ``target_t10_over``/``target_t10_total_push`` for TOTAL) --
    the independent ground truth Fix 4's own settlement must be cross-checked
    against."""
    if market not in _CANONICAL_MARKET_KEYS:
        raise ValueError(f"No canonical T-10 market source is registered for market {market!r}.")
    path = external_data.resolve(_CANONICAL_MARKET_KEYS[market], root_override=root_override)
    frame = pd.read_parquet(path)

    if market == MARKET_ATS:
        target_cover_col, target_push_col, actual_col = (
            "target_t10_home_cover",
            "target_t10_ats_push",
            "target_home_margin",
        )
    else:
        target_cover_col, target_push_col, actual_col = (
            "target_t10_over",
            "target_t10_total_push",
            "target_total_points",
        )
    required = {
        "game_id", "home_team_id", "away_team_id",
        "market_t10_consensus_line", "market_t10_novig_probability", "market_t10_line_sd",
        target_cover_col, target_push_col, actual_col,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical T-10 {market} market matrix is missing settlement columns: {sorted(missing)}")

    out = frame[list(required)].copy()
    out = out.rename(
        columns={
            target_cover_col: "canonical_target_cover_or_over",
            target_push_col: "canonical_target_push",
            actual_col: "canonical_actual_margin_or_total",
        }
    )
    out["game_id"] = out["game_id"].astype(str)
    return out


_LEGACY_ODDS_API_MARKET_KEY = {MARKET_ATS: "spreads", MARKET_TOTAL: "totals"}


def load_contributing_bookmaker_keys(
    market: str,
    game_id: str,
    *,
    horizon: str = "closing_t10",
    root_override: str | None = None,
) -> tuple[str, ...] | None:
    """Certification-blocker repair item 7: the sorted, unique bookmaker
    keys that contributed to one game's closing_t10 consensus, read from the
    already-fetched raw per-book snapshot store
    (``odds_history.2020_2023``'s ``bookmaker_quotes.parquet``) -- never a
    second historical-odds fetch or rebuild; this file already exists as a
    side effect of the same backfill run that produced the consensus this
    module attaches. Returns ``None`` (never fabricates an empty/placeholder
    list) if that raw per-book store isn't available on this machine or has
    no rows for this game/market/horizon."""
    legacy_market = _LEGACY_ODDS_API_MARKET_KEY.get(market)
    if legacy_market is None:
        raise ValueError(f"load_contributing_bookmaker_keys supports ATS/TOTAL only, got {market!r}.")
    try:
        odds_dir = external_data.resolve("odds_history.2020_2023", root_override=root_override)
    except external_data.ExternalDataUnavailableError:
        return None
    quotes_path = odds_dir / "bookmaker_quotes.parquet"
    if not quotes_path.is_file():
        return None
    quotes = pd.read_parquet(quotes_path, columns=["game_id", "horizon", "market", "bookmaker_key"])
    sub = quotes[
        (quotes["game_id"].astype(str) == str(game_id))
        & (quotes["horizon"] == horizon)
        & (quotes["market"] == legacy_market)
    ]
    if sub.empty:
        return None
    return tuple(sorted(sub["bookmaker_key"].astype(str).unique()))
