"""THE canonical market-pricing path (Acceptance Gate A).

One implementation for: quote validation -> two-sided pairing -> per-quote
freshness -> de-vig -> reference-point selection -> consensus -> audit. Both the
live weekly path and historical replay call this, so train and serve cannot
diverge. De-vig and consensus methods are read from the frozen calibration
artifact, never hard-coded here.

Input quote schema (one row per book-outcome), matching the odds flattener:
  provider_event_id, game_id, bookmaker_id, market_type, outcome_side,
  line_value, raw_implied_probability, price_decimal,
  bookmaker_last_update_utc, commence_time_utc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nfl_hybrid.pricing.devig import devig_pair
from nfl_hybrid.pricing.weekly import QuoteAgeConfig, quote_is_fresh

# market -> (side_a is the reported probability side, side_b is its complement)
MARKET_SIDES = {
    "moneyline": ("home", "away"),
    "spread": ("home", "away"),
    "total": ("over", "under"),
}
STATUS_ORDER = ["NO_PRICE", "INCOMPLETE_MARKET", "INSUFFICIENT_BOOKS", "STALE_QUOTES",
                "INDICATIVE_ONLY", "VALID_MARKET"]


@dataclass
class MarketPrice:
    game_id: str
    market: str
    reference_point: float | None
    fair_probability: float
    status: str
    audit: dict = field(default_factory=dict)


def _consensus(values: np.ndarray, method: str) -> float:
    if len(values) == 0:
        return float("nan")
    if method == "median":
        return float(np.median(values))
    if method == "sharpest_book_anchor":  # runtime has no anchor table -> equal mean
        return float(np.mean(values))
    return float(np.mean(values))  # equal_mean (default)


def _select_reference_point(sub: pd.DataFrame, market: str) -> float | None:
    """Deterministic single reference contract point (mode of |line|). Moneyline
    has no point. This never mixes -2.5 and -3 into one consensus."""
    if market == "moneyline":
        return None
    pts = pd.to_numeric(sub["line_value"], errors="coerce").abs().dropna()
    if pts.empty:
        return None
    return float(pts.round(1).mode().iloc[0])


def price_one_market(
    quotes: pd.DataFrame,
    *,
    market: str,
    as_of_utc: str,
    devig_method: str,
    consensus_method: str,
    quote_age_cfg: QuoteAgeConfig | None = None,
    minimum_books: int = 3,
) -> MarketPrice:
    """Price a single (game, market) from its book quotes. Returns fair prob for the
    'a' side (home / over) with a full freshness + consensus audit."""
    cfg = quote_age_cfg or QuoteAgeConfig()
    side_a, side_b = MARKET_SIDES[market]
    now = pd.Timestamp(as_of_utc)
    gid = str(quotes["game_id"].iloc[0]) if len(quotes) else ""

    q = quotes.copy()
    q["commence_ts"] = pd.to_datetime(q.get("commence_time_utc"), utc=True, errors="coerce")
    q["book_ts"] = pd.to_datetime(q.get("bookmaker_last_update_utc"), utc=True, errors="coerce")
    q["age_min"] = (now - q["book_ts"]).dt.total_seconds() / 60.0
    q["mtk_min"] = (q["commence_ts"] - now).dt.total_seconds() / 60.0

    # reject malformed / future / post-kickoff / non-finite quotes up front
    q = q[q["market_type"] == market]
    q = q[q["outcome_side"].isin([side_a, side_b])]
    q = q[np.isfinite(pd.to_numeric(q["raw_implied_probability"], errors="coerce"))]
    q = q[q["book_ts"].notna()]
    q = q[q["age_min"] >= 0]  # no future-dated quotes
    # post-kickoff => mtk<0 => reject
    total_books = q["bookmaker_id"].nunique()
    q = q[(q["mtk_min"].isna()) | (q["mtk_min"] > 0)]

    ref = _select_reference_point(q, market)
    if ref is not None:
        q = q[np.isclose(pd.to_numeric(q["line_value"], errors="coerce").abs(), ref, atol=1e-6)]

    # per-quote freshness BEFORE pairing/consensus (a fresh book cannot rescue a stale one)
    fresh_mask = np.array(
        [quote_is_fresh(a, m, cfg) for a, m in zip(q["age_min"], q["mtk_min"])], dtype=bool
    )
    stale_books = int(q.loc[~fresh_mask, "bookmaker_id"].nunique()) if len(q) else 0
    q = q.loc[fresh_mask] if len(q) else q

    # pair two sides within each book at the reference point, then de-vig
    fair = []
    kept_books = []
    for book, bg in q.groupby("bookmaker_id"):
        a = bg[bg["outcome_side"] == side_a]["raw_implied_probability"]
        b = bg[bg["outcome_side"] == side_b]["raw_implied_probability"]
        if len(a) == 1 and len(b) == 1:  # complete, non-duplicated pair only
            qa, _ = devig_pair(float(a.iloc[0]), float(b.iloc[0]), devig_method)
            if np.isfinite(qa):
                fair.append(qa); kept_books.append(book)

    fair = np.array(fair, float)
    n_books = len(fair)
    ages = q[q["bookmaker_id"].isin(kept_books)]["age_min"].to_numpy()
    audit = {
        "reference_point": ref,
        "books_total": int(total_books),
        "books_retained": n_books,
        "books_rejected_stale": stale_books,
        "newest_quote_age_min": float(np.min(ages)) if len(ages) else None,
        "oldest_quote_age_min": float(np.max(ages)) if len(ages) else None,
        "median_quote_age_min": float(np.median(ages)) if len(ages) else None,
        "consensus_dispersion": float(np.std(fair)) if n_books > 1 else 0.0,
        "freshness_max_age_min": (cfg.max_age_near_minutes
                                  if (len(q) and (q["mtk_min"].min() <= cfg.near_kickoff_hours * 60))
                                  else cfg.max_age_far_minutes),
        "devig_method": devig_method,
        "consensus_method": consensus_method,
    }

    if n_books == 0:
        status = "NO_PRICE" if total_books == 0 else "STALE_QUOTES"
        return MarketPrice(gid, market, ref, float("nan"), status, audit)
    prob = _consensus(fair, consensus_method)
    if n_books < minimum_books:
        status = "INDICATIVE_ONLY"  # priced but below coverage -> not actionable
    else:
        status = "VALID_MARKET"
    audit["fair_probability"] = prob
    return MarketPrice(gid, market, ref, prob, status, audit)


def price_game_markets(
    quotes: pd.DataFrame,
    *,
    as_of_utc: str,
    artifact,
    quote_age_cfg: QuoteAgeConfig | None = None,
    minimum_books: int = 3,
) -> dict[str, MarketPrice]:
    """Price moneyline/spread/total for one game using the artifact-selected methods."""
    dc = getattr(artifact, "devig_consensus", {}) if artifact is not None else {}
    method_for = {
        "moneyline": ("moneyline_devig", "moneyline_consensus"),
        "spread": ("ats_devig", "ats_consensus"),
        "total": ("total_devig", "total_consensus"),
    }
    out = {}
    for market, (dv_key, cons_key) in method_for.items():
        dv = dc.get(dv_key, "proportional")
        cons = dc.get(cons_key, "equal_mean")
        sub = quotes[quotes["market_type"] == market]
        if sub.empty:
            out[market] = MarketPrice(str(quotes["game_id"].iloc[0]) if len(quotes) else "",
                                      market, None, float("nan"), "INCOMPLETE_MARKET",
                                      {"devig_method": dv, "consensus_method": cons})
            continue
        out[market] = price_one_market(
            sub, market=market, as_of_utc=as_of_utc, devig_method=dv,
            consensus_method=cons, quote_age_cfg=quote_age_cfg, minimum_books=minimum_books,
        )
    return out
