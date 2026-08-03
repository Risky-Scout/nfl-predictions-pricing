"""Authoritative exact market-contract utility (contract integrity, not a model).

R2 root cause this module repairs: the EPA tournament fitted, predicted and
scored on one contract (the Spreadspoke *reference* line) while the benchmark it
was graded against used a *different* contract (a purchased closing line), and
the benchmark itself paired a median line with a median probability computed over
*different* quoted points. A probability generated for home -2.5 could be scored
against a label/price of home -3, and a benchmark point of -3 could carry a
probability aggregated across -2.5 and -3 quotes.

Everything that must agree on a single betting contract lives here exactly once:

* :func:`canonical_decimal_text` -- deterministic point/snapshot formatting;
* :func:`make_market_contract_id` -- deterministic SHA-256 contract identity;
* :func:`build_real_closing_benchmark` -- exact-point consensus benchmark
  (one actual quoted point per market, probability paired to *that* point);
* :func:`bind_closing_contract` -- replace reference lines with closing lines
  *before* feature-matrix construction, retaining originals for audit;
* :func:`validate_prediction_contract` -- prove prediction/outcome/benchmark
  contract identity (IDs and points) before evaluation, failing closed.

There is exactly ONE implementation of each of these behaviours in the
repository; scripts are thin I/O wrappers over this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical identifiers / constants
# --------------------------------------------------------------------------- #
AGGREGATION_METHOD = "exact_point_consensus_v1"
MARKET_SOURCE_CLOSING = "REAL-CLOSING"

PROXY_AGGREGATION_METHOD = "reference_proxy_v1"
MARKET_SOURCE_PROXY = "REFERENCE-LINE PROXY"
PROXY_MARGIN_SD = 13.5  # reference-spread normal approximation sd (pre-existing method)

# Enumerated, non-free-form contract exclusion reasons for the coverage audit.
CONTRACT_INCLUDED = "INCLUDED"
NO_BENCHMARK_ROW = "NO_BENCHMARK_ROW"
INCOMPLETE_MONEYLINE_CONTRACT = "INCOMPLETE_MONEYLINE_CONTRACT"
INCOMPLETE_SPREAD_CONTRACT = "INCOMPLETE_SPREAD_CONTRACT"
INCOMPLETE_TOTAL_CONTRACT = "INCOMPLETE_TOTAL_CONTRACT"
DUPLICATE_CONTRACT = "DUPLICATE_CONTRACT"
INVALID_CONTRACT_VALUE = "INVALID_CONTRACT_VALUE"

VALID_EXCLUSION_REASONS = frozenset(
    {
        NO_BENCHMARK_ROW,
        INCOMPLETE_MONEYLINE_CONTRACT,
        INCOMPLETE_SPREAD_CONTRACT,
        INCOMPLETE_TOTAL_CONTRACT,
        DUPLICATE_CONTRACT,
        INVALID_CONTRACT_VALUE,
    }
)

_POINT_TOL = 1e-9


class ContractError(ValueError):
    """Deterministic contract-integrity failure (fail closed, never a warning)."""


# --------------------------------------------------------------------------- #
# 4.1 Canonical decimal formatting
# --------------------------------------------------------------------------- #
def canonical_decimal_text(value: object) -> str | None:
    """Deterministic, locale-independent shortest round-trip decimal text.

    ``-0.0`` -> ``"0"``; ``-3.0`` -> ``"-3"``; ``44.5`` -> ``"44.5"``. A missing
    or non-finite value (``None``/``NaN``/``inf``) -> ``None``. Never collapses two
    genuinely distinct points: the shortest round-trip repr is exact for the
    half-point grid these markets quote on.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f == 0.0:  # covers -0.0
        return "0"
    # ``repr`` gives the shortest string that round-trips to the same float;
    # Decimal.normalize() then strips a trailing ".0"/zeros deterministically.
    d = Decimal(repr(f)).normalize()
    text = format(d, "f")  # plain decimal, never scientific notation
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text


# --------------------------------------------------------------------------- #
# 4.2 Deterministic contract ID
# --------------------------------------------------------------------------- #
def make_market_contract_id(
    *,
    game_id: object,
    market_type: str,
    outcome_side: str,
    line_value: object,
    market_source: str,
    snapshot_minutes_to_kickoff: object,
    aggregation_method: str,
) -> str:
    """SHA-256 over a canonical, stably-ordered contract payload.

    ``line_value`` is ``None`` for moneyline (canonicalised to JSON ``null``). The
    ID changes iff any of game/market/side/point/snapshot/source/aggregation
    changes; ``-0.0`` and ``0.0`` canonicalise identically.
    """
    payload = {
        "game_id": str(game_id),
        "market_type": str(market_type),
        "outcome_side": str(outcome_side),
        "line_value": canonical_decimal_text(line_value),
        "market_source": str(market_source),
        "snapshot_minutes_to_kickoff": canonical_decimal_text(
            snapshot_minutes_to_kickoff
        ),
        "aggregation_method": str(aggregation_method),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Exact-point consensus benchmark
# --------------------------------------------------------------------------- #
_QUOTE_REQUIRED = (
    "game_id",
    "market_type",
    "outcome_side",
    "bookmaker_id",
    "minutes_to_kickoff",
    "devig_probability",
)
_CONTRACT_MARKETS = ("moneyline", "spread", "total")


def _closing_snapshot(odds: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the closing snapshot per game (smallest positive mtk).

    Preserves the current R2 definition of the closing snapshot; the time-horizon
    rule is intentionally *not* redesigned here.
    """
    o = odds[odds["market_type"].isin(_CONTRACT_MARKETS)].copy()
    o["mtk"] = pd.to_numeric(o["minutes_to_kickoff"], errors="coerce")
    o = o[o["mtk"] > 0]  # exclude in-play / non-positive
    if o.empty:
        return o.assign(_closing_mtk=pd.Series(dtype=float))
    closing_snap = o.groupby("game_id")["mtk"].transform("min")
    closing = o[o["mtk"] == closing_snap].copy()
    closing["_closing_mtk"] = closing["mtk"]
    return closing


def _validate_and_dedup_quotes(closing: pd.DataFrame) -> pd.DataFrame:
    """Validate closing quote rows and collapse only *identical* duplicates.

    Fails closed on missing required fields, non-finite / out-of-(0,1)
    probabilities, non-finite spread/total points, missing bookmaker IDs, or
    conflicting duplicate quotes for one (game, snapshot, market, side, book).
    """
    missing = [c for c in _QUOTE_REQUIRED if c not in closing.columns]
    if missing:
        raise ContractError(f"closing quotes missing required columns: {missing}")
    if closing.empty:
        return closing

    c = closing.copy()
    c["devig_probability"] = pd.to_numeric(c["devig_probability"], errors="coerce")
    c["line_value"] = pd.to_numeric(c.get("line_value"), errors="coerce")

    prob = c["devig_probability"].to_numpy(dtype=float)
    if not np.isfinite(prob).all():
        raise ContractError("closing quote has non-finite devig_probability")
    if not ((prob > 0.0) & (prob < 1.0)).all():
        raise ContractError(
            "closing quote devig_probability outside open interval (0, 1)"
        )
    if c["bookmaker_id"].isna().any() or (
        c["bookmaker_id"].astype(str).str.strip() == ""
    ).any():
        raise ContractError("closing quote has missing bookmaker_id")

    needs_line = c["market_type"].isin(("spread", "total"))
    if not np.isfinite(c.loc[needs_line, "line_value"].to_numpy(dtype=float)).all():
        raise ContractError("spread/total closing quote has non-finite line_value")

    key = ["game_id", "market_type", "outcome_side", "bookmaker_id"]
    # Exact duplicates (all contract + probability fields identical) collapse to one.
    c = c.drop_duplicates(subset=key + ["line_value", "devig_probability"])
    # Anything still sharing the key now differs in line/probability -> conflict.
    conflict = c.duplicated(subset=key, keep=False)
    if conflict.any():
        bad = c.loc[conflict, key].drop_duplicates().to_dict("records")
        raise ContractError(
            f"conflicting duplicate closing quotes for {bad[:3]} "
            f"({len(bad)} conflicting keys)"
        )
    return c


def _select_consensus_point(sub: pd.DataFrame) -> tuple[float, int, int]:
    """Select one actual quoted point by unique-book count with deterministic ties.

    Returns ``(selected_point, unique_books_at_point, candidate_point_count)``.
    Tie-break: most unique books -> nearest to the median of eligible quoted
    points -> numerically smaller. The selected point is always an actual quote.
    """
    books_at = sub.groupby("line_value")["bookmaker_id"].nunique()
    candidate_count = int(len(books_at))
    max_books = int(books_at.max())
    tied = sorted(float(p) for p in books_at.index[books_at == max_books])
    if len(tied) == 1:
        selected = tied[0]
    else:
        eligible_points = sorted(float(p) for p in books_at.index)
        median_pt = float(np.median(eligible_points))
        selected = min(tied, key=lambda p: (abs(p - median_pt), p))
    return selected, max_books, candidate_count


def _point_market_consensus(
    game_quotes: pd.DataFrame, *, market_type: str, outcome_side: str
) -> dict | None:
    """Exact-point consensus for a spread/total market of one game.

    The probability is the median de-vigged probability among ONLY the quotes at
    the exact selected point -- never a median across mixed points.
    """
    sub = game_quotes[
        (game_quotes["market_type"] == market_type)
        & (game_quotes["outcome_side"] == outcome_side)
    ]
    if sub.empty:
        return None
    point, n_books, candidate_count = _select_consensus_point(sub)
    at_point = sub[np.isclose(sub["line_value"].to_numpy(dtype=float), point, atol=_POINT_TOL)]
    prob = float(at_point["devig_probability"].median())
    return {
        "point": point,
        "probability": prob,
        "consensus_books": n_books,
        "candidate_point_count": candidate_count,
    }


def _moneyline_consensus(game_quotes: pd.DataFrame) -> dict | None:
    sub = game_quotes[
        (game_quotes["market_type"] == "moneyline")
        & (game_quotes["outcome_side"] == "home")
    ]
    if sub.empty:
        return None
    return {
        "probability": float(sub["devig_probability"].median()),
        "consensus_books": int(sub["bookmaker_id"].nunique()),
    }


BENCHMARK_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_team_id",
    "away_team_id",
    "market_ml_home_probability",
    "market_cover_home_probability",
    "market_over_probability",
    "closing_home_spread",
    "closing_total_line",
    "closing_minutes_to_kickoff",
    "market_source",
    "aggregation_method",
    "moneyline_contract_id",
    "spread_contract_id",
    "total_contract_id",
    "moneyline_consensus_books",
    "spread_consensus_books",
    "total_consensus_books",
    "spread_candidate_point_count",
    "total_candidate_point_count",
)


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: pd.DataFrame  # one row per fully-covered game
    per_game_status: pd.DataFrame  # game_id, complete flag, exclusion reason
    input_row_count: int


def build_real_closing_benchmark(odds: pd.DataFrame, games: pd.DataFrame) -> BenchmarkResult:
    """Build the exact-point REAL-CLOSING benchmark (one row per covered game).

    ``games`` provides identity columns (game_id, season, week, home/away team).
    Every selected spread/total point is an actual quote and every probability is
    paired to that exact selected point.
    """
    dev = games[["game_id", "season", "week", "home_team_id", "away_team_id"]].copy()
    dev["game_id"] = dev["game_id"].astype(str)
    if dev["game_id"].duplicated().any():
        raise ContractError("games frame has duplicate game_id rows")

    closing = _closing_snapshot(odds)
    closing["game_id"] = closing["game_id"].astype(str)
    closing = _validate_and_dedup_quotes(closing)

    ident = dev.set_index("game_id")
    rows: list[dict] = []
    status: list[dict] = []
    for gid, gq in closing.groupby("game_id"):
        if gid not in ident.index:
            continue  # odds for a game not in the target schedule -> ignore
        snap = float(gq["_closing_mtk"].iloc[0])
        ml = _moneyline_consensus(gq)
        sp = _point_market_consensus(gq, market_type="spread", outcome_side="home")
        to = _point_market_consensus(gq, market_type="total", outcome_side="over")

        reason = None
        if ml is None:
            reason = INCOMPLETE_MONEYLINE_CONTRACT
        elif sp is None:
            reason = INCOMPLETE_SPREAD_CONTRACT
        elif to is None:
            reason = INCOMPLETE_TOTAL_CONTRACT
        if reason is not None:
            status.append({"game_id": gid, "contract_complete": False, "exclusion_reason": reason})
            continue

        info = ident.loc[gid]
        rows.append(
            {
                "game_id": gid,
                "season": int(info["season"]),
                "week": info["week"],
                "home_team_id": info["home_team_id"],
                "away_team_id": info["away_team_id"],
                "market_ml_home_probability": ml["probability"],
                "market_cover_home_probability": sp["probability"],
                "market_over_probability": to["probability"],
                "closing_home_spread": sp["point"],
                "closing_total_line": to["point"],
                "closing_minutes_to_kickoff": snap,
                "market_source": MARKET_SOURCE_CLOSING,
                "aggregation_method": AGGREGATION_METHOD,
                "moneyline_contract_id": make_market_contract_id(
                    game_id=gid, market_type="moneyline", outcome_side="home",
                    line_value=None, market_source=MARKET_SOURCE_CLOSING,
                    snapshot_minutes_to_kickoff=snap, aggregation_method=AGGREGATION_METHOD,
                ),
                "spread_contract_id": make_market_contract_id(
                    game_id=gid, market_type="spread", outcome_side="home",
                    line_value=sp["point"], market_source=MARKET_SOURCE_CLOSING,
                    snapshot_minutes_to_kickoff=snap, aggregation_method=AGGREGATION_METHOD,
                ),
                "total_contract_id": make_market_contract_id(
                    game_id=gid, market_type="total", outcome_side="over",
                    line_value=to["point"], market_source=MARKET_SOURCE_CLOSING,
                    snapshot_minutes_to_kickoff=snap, aggregation_method=AGGREGATION_METHOD,
                ),
                "moneyline_consensus_books": ml["consensus_books"],
                "spread_consensus_books": sp["consensus_books"],
                "total_consensus_books": to["consensus_books"],
                "spread_candidate_point_count": sp["candidate_point_count"],
                "total_candidate_point_count": to["candidate_point_count"],
            }
        )
        status.append({"game_id": gid, "contract_complete": True, "exclusion_reason": CONTRACT_INCLUDED})

    benchmark = pd.DataFrame(rows, columns=list(BENCHMARK_COLUMNS))
    validate_benchmark(benchmark)
    return BenchmarkResult(
        benchmark=benchmark.reset_index(drop=True),
        per_game_status=pd.DataFrame(status, columns=["game_id", "contract_complete", "exclusion_reason"]),
        input_row_count=int(len(closing)),
    )


def validate_benchmark(benchmark: pd.DataFrame) -> None:
    """Fail closed unless the benchmark satisfies every contract invariant."""
    missing = [c for c in BENCHMARK_COLUMNS if c not in benchmark.columns]
    if missing:
        raise ContractError(f"benchmark missing columns: {missing}")
    if benchmark.empty:
        raise ContractError("benchmark is empty")
    if benchmark["game_id"].duplicated().any():
        raise ContractError("benchmark has more than one row per game_id")
    for col in (
        "market_ml_home_probability",
        "market_cover_home_probability",
        "market_over_probability",
    ):
        p = benchmark[col].to_numpy(dtype=float)
        if not np.isfinite(p).all() or not ((p > 0.0) & (p < 1.0)).all():
            raise ContractError(f"benchmark {col} not all finite in (0, 1)")
    for col in ("closing_home_spread", "closing_total_line"):
        if not np.isfinite(benchmark[col].to_numpy(dtype=float)).all():
            raise ContractError(f"benchmark {col} has a non-finite point")
    for col in ("moneyline_contract_id", "spread_contract_id", "total_contract_id"):
        if benchmark[col].isna().any() or (benchmark[col].astype(str).str.len() == 0).any():
            raise ContractError(f"benchmark {col} has an empty contract id")
    if (benchmark["market_source"] != MARKET_SOURCE_CLOSING).any():
        raise ContractError("benchmark market_source must be REAL-CLOSING")
    if (benchmark["aggregation_method"] != AGGREGATION_METHOD).any():
        raise ContractError("benchmark aggregation_method must be exact_point_consensus_v1")


# --------------------------------------------------------------------------- #
# Pre-matrix contract binding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContractBinding:
    games: pd.DataFrame  # games copy with closing lines bound + provenance columns
    audit: pd.DataFrame  # per-game inclusion/exclusion audit


def _proxy_ml_probability(home_spread: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return 1.0 - norm.cdf((0.0 - (-home_spread)) / PROXY_MARGIN_SD)


# --------------------------------------------------------------------------- #
# Batch 2B: authoritative closing-era two-state contract validator
#
# This is the SINGLE implementation of the 2022-2024 two-state policy. Both the
# runner (``_validate_season_contract_sources``) and the binder
# (``bind_closing_contract``) import and reuse it, so the state machine is never
# duplicated and no closing-era row can bypass validation.
# --------------------------------------------------------------------------- #
CLOSING_ERA_STATE_COLUMNS = (
    "game_id",
    "season",
    "contract_included",
    "contract_exclusion_reason",
    "market_contract_source",
    "moneyline_contract_id",
    "spread_contract_id",
    "total_contract_id",
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
)
_STATE_ID_COLUMNS = (
    "moneyline_contract_id",
    "spread_contract_id",
    "total_contract_id",
)
_STATE_PROB_COLUMNS = (
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
)


def _text_is_missing(values: pd.Series) -> pd.Series:
    """A text field is missing iff pandas-null OR blank after a string strip.

    Sentinel-looking strings such as ``"None"``/``"nan"``/``"NULL"``/``"0"`` are
    NOT treated as missing unless the underlying value is actually null.
    """
    as_string = values.astype("string")
    stripped = as_string.str.strip()
    return as_string.isna() | (stripped == "")


def _is_actual_bool(value) -> bool:
    return isinstance(value, (bool, np.bool_))


def _strict_boolean_series(
    values: pd.Series,
    *,
    context: str,
    column: str,
    game_ids: pd.Series | None = None,
) -> pd.Series:
    """Return a nullable-Boolean Series, rejecting anything that is not an actual
    Boolean scalar.

    Null values, the strings ``"True"``/``"False"``/``"0"``/``"1"``, the integers
    ``0``/``1`` and any float are all rejected rather than coerced. This helper
    deliberately uses NO ``.fillna(False)``, ``.astype(bool)``, truthiness, string
    parsing or integer coercion. On failure it names the context, column, offending
    rows, game ids, values and Python type names.
    """
    ids = (
        game_ids
        if game_ids is not None
        else pd.Series([""] * len(values), index=values.index)
    )
    null_rows = []
    bad_rows = []
    for pos in range(len(values)):
        value = values.iloc[pos]
        if pd.isna(value):
            null_rows.append(pos)
        elif not _is_actual_bool(value):
            bad_rows.append(pos)
    if null_rows or bad_rows:
        lines = []
        for pos in null_rows:
            lines.append(
                f"row {pos} game_id={ids.iloc[pos]!s} "
                f"value=<null/missing Boolean>"
            )
        for pos in bad_rows:
            value = values.iloc[pos]
            lines.append(
                f"row {pos} game_id={ids.iloc[pos]!s} "
                f"value={value!r} type={type(value).__name__}"
            )
        raise ValueError(
            f"{context}: column '{column}' must be an actual Boolean for every "
            "row; null/missing Boolean and non-Boolean values are rejected:\n"
            + "\n".join(lines)
        )
    # Every value is now a real bool/np.bool_; build an explicit nullable-Boolean
    # Series without astype(bool)/fillna.
    return pd.Series(
        [bool(value) for value in values],
        index=values.index,
        dtype="boolean",
    )


def validate_closing_era_contract_states(
    frame: pd.DataFrame,
    *,
    context: str,
    closing_seasons=(2022, 2023, 2024),
) -> dict:
    """Authoritative two-state closing-era contract validator (single source).

    Every game whose season is in ``closing_seasons`` MUST be exactly one of:

    STATE A -- included exact-closing contract:
        * ``contract_included`` is actual Boolean ``True``;
        * ``market_contract_source == REAL-CLOSING``;
        * ``contract_exclusion_reason`` is null/blank (or the canonical
          ``CONTRACT_INCLUDED`` "not-excluded" marker the binder stamps);
        * all three contract ids are nonempty;
        * all three tournament probabilities are finite and strictly inside
          ``(0, 1)``.

    STATE B -- audited exclusion:
        * ``contract_included`` is actual Boolean ``False``;
        * ``contract_exclusion_reason == NO_BENCHMARK_ROW``;
        * ``market_contract_source`` is null/blank;
        * all three contract ids are null/blank;
        * all three tournament probabilities are null.

    Any other combination raises BEFORE the caller filters, fits, scores or writes
    output. There is NO ``~audited_excluded`` bypass -- excluded closing-era rows
    are validated, never exempted. Pre-``closing_seasons`` rows are ignored here
    (their season/source policy is the caller's responsibility) but the whole
    frame's ``contract_included`` column must still be strictly Boolean.

    Returns ``{closing_era_rows, included_exact_contract_rows,
    audited_excluded_rows}``.
    """
    missing_columns = sorted(set(CLOSING_ERA_STATE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{context}: missing closing-era contract-state columns "
            f"{missing_columns}"
        )

    season = pd.to_numeric(frame["season"], errors="coerce")
    if bool(season.isna().any()):
        bad = frame.loc[season.isna().to_numpy(dtype=bool), ["game_id", "season"]]
        raise ValueError(
            f"{context}: non-numeric seasons at rows {list(bad.index)} "
            f"game_ids={list(bad['game_id'])} values={list(bad['season'])}"
        )

    game_id = frame["game_id"].astype("string")
    gid_missing = (game_id.isna() | (game_id.str.strip() == "")).to_numpy(
        dtype=bool, na_value=True
    )
    if gid_missing.any():
        rows = [int(i) for i, m in zip(range(len(frame)), gid_missing) if m]
        raise ValueError(
            f"{context}: missing/blank game_id at rows {rows}"
        )

    # Strict Boolean for EVERY row (missing/malformed contract_included fails).
    strict_included = _strict_boolean_series(
        frame["contract_included"],
        context=context,
        column="contract_included",
        game_ids=frame["game_id"],
    )
    strict_np = np.array([bool(value) for value in strict_included], dtype=bool)

    season_np = season.to_numpy(dtype=float)
    closing_np = np.isin(season_np, np.array(list(closing_seasons), dtype=float))
    included_np = strict_np & closing_np
    excluded_np = (~strict_np) & closing_np

    failures = []
    for pos in np.nonzero(closing_np)[0]:
        pos = int(pos)
        gid = game_id.iloc[pos]
        yr = season.iloc[pos]
        if strict_np[pos]:
            # ---- STATE A ---- #
            src = frame["market_contract_source"].iloc[pos]
            if pd.isna(src) or str(src).strip() != MARKET_SOURCE_CLOSING:
                failures.append(
                    (gid, yr, "market_contract_source", src, "A",
                     f"must equal {MARKET_SOURCE_CLOSING} for an included row")
                )
            reason = frame["contract_exclusion_reason"].iloc[pos]
            if not (
                pd.isna(reason)
                or str(reason).strip() in ("", CONTRACT_INCLUDED)
            ):
                failures.append(
                    (gid, yr, "contract_exclusion_reason", reason, "A",
                     "must be null/blank for an included row")
                )
            for col in _STATE_ID_COLUMNS:
                value = frame[col].iloc[pos]
                if pd.isna(value) or str(value).strip() == "":
                    failures.append(
                        (gid, yr, col, value, "A",
                         "must be a nonempty contract id for an included row")
                    )
            for col in _STATE_PROB_COLUMNS:
                value = frame[col].iloc[pos]
                fnum = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(fnum) or not np.isfinite(float(fnum)) or not (
                    0.0 < float(fnum) < 1.0
                ):
                    failures.append(
                        (gid, yr, col, value, "A",
                         "must be finite and strictly inside (0, 1) for an "
                         "included row")
                    )
        else:
            # ---- STATE B (audited exclusion) ---- #
            reason = frame["contract_exclusion_reason"].iloc[pos]
            if pd.isna(reason) or str(reason).strip() != NO_BENCHMARK_ROW:
                failures.append(
                    (gid, yr, "contract_exclusion_reason", reason, "B",
                     f"must equal {NO_BENCHMARK_ROW} for an audited exclusion")
                )
            src = frame["market_contract_source"].iloc[pos]
            if not (pd.isna(src) or str(src).strip() == ""):
                failures.append(
                    (gid, yr, "market_contract_source", src, "B",
                     "must be null/blank for an audited exclusion")
                )
            for col in _STATE_ID_COLUMNS:
                value = frame[col].iloc[pos]
                if not (pd.isna(value) or str(value).strip() == ""):
                    failures.append(
                        (gid, yr, col, value, "B",
                         "must be null/blank for an audited exclusion")
                    )
            for col in _STATE_PROB_COLUMNS:
                value = frame[col].iloc[pos]
                if not pd.isna(value):
                    failures.append(
                        (gid, yr, col, value, "B",
                         "must be null for an audited exclusion")
                    )

    if failures:
        lines = [
            f"[State {state}] game_id={gid} season={yr} field={field} "
            f"value={value!r} -> {rule}"
            for (gid, yr, field, value, state, rule) in failures
        ]
        raise ValueError(
            f"{context}: closing-era contract-state violations "
            "(each 2022-2024 game must be State A included or State B audited "
            "exclusion):\n" + "\n".join(lines)
        )

    closing_era_rows = int(closing_np.sum())
    included_rows = int(included_np.sum())
    excluded_rows = int(excluded_np.sum())
    if included_rows + excluded_rows != closing_era_rows:
        raise ValueError(
            f"{context}: internal closing-era state partition inconsistency "
            f"(included={included_rows} + excluded={excluded_rows} != "
            f"closing_era={closing_era_rows})"
        )
    return {
        "closing_era_rows": closing_era_rows,
        "included_exact_contract_rows": included_rows,
        "audited_excluded_rows": excluded_rows,
    }


def bind_closing_contract(
    games: pd.DataFrame,
    benchmark: pd.DataFrame,
    per_game_status: pd.DataFrame | None = None,
    *,
    closing_seasons: tuple[int, ...] = (2022, 2023, 2024),
) -> ContractBinding:
    """Replace reference lines with closing lines BEFORE matrix construction.

    Matched (benchmark-covered) rows get ``home_spread_reference`` /
    ``total_line_reference`` overwritten with the exact closing points, the
    exact market probabilities, contract IDs, and ``market_contract_source ==
    REAL-CLOSING``. Original references are retained in ``original_*_reference``.
    Rows in ``closing_seasons`` without a complete benchmark row are marked for
    exclusion. Rows in seasons before ``closing_seasons`` keep the reference line
    as a clearly-labelled ``REFERENCE-LINE PROXY`` training-only contract.
    """
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    if g["game_id"].duplicated().any():
        raise ContractError("games frame has duplicate game_id rows before binding")

    g["original_home_spread_reference"] = pd.to_numeric(
        g["home_spread_reference"], errors="coerce"
    )
    g["original_total_line_reference"] = pd.to_numeric(
        g["total_line_reference"], errors="coerce"
    )

    bench = benchmark.copy()
    bench["game_id"] = bench["game_id"].astype(str)
    if bench["game_id"].duplicated().any():
        raise ContractError("benchmark has duplicate game_id rows before binding")

    bench_cols = [
        "game_id",
        "market_ml_home_probability",
        "market_cover_home_probability",
        "market_over_probability",
        "closing_home_spread",
        "closing_total_line",
        "closing_minutes_to_kickoff",
        "moneyline_contract_id",
        "spread_contract_id",
        "total_contract_id",
    ]
    merged = g.merge(
        bench[bench_cols], on="game_id", how="left", validate="one_to_one", indicator=True
    )

    matched = merged["_merge"] == "both"
    season = pd.to_numeric(merged["season"], errors="coerce")
    in_closing_seasons = season.isin(closing_seasons)

    # Provenance defaults.
    merged["market_contract_source"] = ""
    merged["tournament_market_ml_home_probability"] = np.nan
    merged["tournament_market_cover_home_probability"] = np.nan
    merged["tournament_market_over_probability"] = np.nan
    merged["contract_included"] = False
    merged["contract_exclusion_reason"] = NO_BENCHMARK_ROW
    merged["benchmark_merge_status"] = merged["_merge"].astype(str)

    # ---- matched REAL-CLOSING rows: bind the exact closing contract ---- #
    m_idx = merged.index[matched]
    merged.loc[m_idx, "home_spread_reference"] = merged.loc[m_idx, "closing_home_spread"]
    merged.loc[m_idx, "total_line_reference"] = merged.loc[m_idx, "closing_total_line"]
    merged.loc[m_idx, "market_contract_source"] = MARKET_SOURCE_CLOSING
    merged.loc[m_idx, "tournament_market_ml_home_probability"] = merged.loc[
        m_idx, "market_ml_home_probability"
    ]
    merged.loc[m_idx, "tournament_market_cover_home_probability"] = merged.loc[
        m_idx, "market_cover_home_probability"
    ]
    merged.loc[m_idx, "tournament_market_over_probability"] = merged.loc[
        m_idx, "market_over_probability"
    ]
    merged.loc[m_idx, "contract_included"] = True
    merged.loc[m_idx, "contract_exclusion_reason"] = CONTRACT_INCLUDED

    # ---- pre-closing-season rows: REFERENCE-LINE PROXY training-only ---- #
    proxy_mask = (~in_closing_seasons) & (~matched)
    p_idx = merged.index[proxy_mask]
    ref_spread = merged.loc[p_idx, "original_home_spread_reference"].to_numpy(dtype=float)
    ref_total = merged.loc[p_idx, "original_total_line_reference"].to_numpy(dtype=float)
    merged.loc[p_idx, "market_contract_source"] = MARKET_SOURCE_PROXY
    merged.loc[p_idx, "tournament_market_ml_home_probability"] = _proxy_ml_probability(ref_spread)
    merged.loc[p_idx, "tournament_market_cover_home_probability"] = 0.5
    merged.loc[p_idx, "tournament_market_over_probability"] = 0.5
    # Proxy rows are NEVER scored test rows; contract_included stays False and the
    # reason marks them as proxy (not an eligible closing contract).
    merged.loc[p_idx, "contract_exclusion_reason"] = "REFERENCE-LINE PROXY (pre-closing training only)"
    merged.loc[p_idx, "moneyline_contract_id"] = [
        make_market_contract_id(
            game_id=gid, market_type="moneyline", outcome_side="home", line_value=None,
            market_source=MARKET_SOURCE_PROXY, snapshot_minutes_to_kickoff=None,
            aggregation_method=PROXY_AGGREGATION_METHOD,
        )
        for gid in merged.loc[p_idx, "game_id"]
    ]
    merged.loc[p_idx, "spread_contract_id"] = [
        make_market_contract_id(
            game_id=gid, market_type="spread", outcome_side="home", line_value=sv,
            market_source=MARKET_SOURCE_PROXY, snapshot_minutes_to_kickoff=None,
            aggregation_method=PROXY_AGGREGATION_METHOD,
        )
        for gid, sv in zip(merged.loc[p_idx, "game_id"], ref_spread)
    ]
    merged.loc[p_idx, "total_contract_id"] = [
        make_market_contract_id(
            game_id=gid, market_type="total", outcome_side="over", line_value=tv,
            market_source=MARKET_SOURCE_PROXY, snapshot_minutes_to_kickoff=None,
            aggregation_method=PROXY_AGGREGATION_METHOD,
        )
        for gid, tv in zip(merged.loc[p_idx, "game_id"], ref_total)
    ]

    # ---- closing-season rows without a benchmark: excluded (NO_BENCHMARK_ROW) ---- #
    if per_game_status is not None and len(per_game_status):
        pgs = per_game_status.copy()
        pgs["game_id"] = pgs["game_id"].astype(str)
        reason_map = dict(zip(pgs["game_id"], pgs["exclusion_reason"]))
        excl = merged.index[in_closing_seasons & (~matched)]
        merged.loc[excl, "contract_exclusion_reason"] = [
            reason_map.get(gid, NO_BENCHMARK_ROW) if reason_map.get(gid, NO_BENCHMARK_ROW) != CONTRACT_INCLUDED
            else NO_BENCHMARK_ROW
            for gid in merged.loc[excl, "game_id"]
        ]

    merged = merged.drop(columns=["_merge"])

    # ---- Batch 2B: authoritative two-state closing-era validation ---- #
    # Supersedes the earlier narrow proxy-only guard. Every 2022-2024 row must be
    # either a complete included exact-closing contract (State A) or a complete
    # audited exclusion (State B); ANY other combination -- a closing-era proxy, an
    # excluded row still carrying ids/probabilities/a source, a malformed
    # ``contract_included`` -- fails closed HERE, before the frame can reach the
    # matrix. The validator is the single implementation of the state machine.
    validate_closing_era_contract_states(
        merged,
        context="bind_closing_contract",
        closing_seasons=(2022, 2023, 2024),
    )

    audit = merged[
        ["game_id", "season", "week", "contract_included", "contract_exclusion_reason", "benchmark_merge_status"]
    ].copy()
    return ContractBinding(games=merged, audit=audit)


# --------------------------------------------------------------------------- #
# Strict prediction/outcome/benchmark contract validation before evaluation
# --------------------------------------------------------------------------- #
def _recompute_ids(row: pd.Series, snapshot: object) -> tuple[str, str, str]:
    ml = make_market_contract_id(
        game_id=row["game_id"], market_type="moneyline", outcome_side="home",
        line_value=None, market_source=MARKET_SOURCE_CLOSING,
        snapshot_minutes_to_kickoff=snapshot, aggregation_method=AGGREGATION_METHOD,
    )
    sp = make_market_contract_id(
        game_id=row["game_id"], market_type="spread", outcome_side="home",
        line_value=row["home_spread"], market_source=MARKET_SOURCE_CLOSING,
        snapshot_minutes_to_kickoff=snapshot, aggregation_method=AGGREGATION_METHOD,
    )
    to = make_market_contract_id(
        game_id=row["game_id"], market_type="total", outcome_side="over",
        line_value=row["total_line"], market_source=MARKET_SOURCE_CLOSING,
        snapshot_minutes_to_kickoff=snapshot, aggregation_method=AGGREGATION_METHOD,
    )
    return ml, sp, to


def validate_prediction_contract(
    predictions: pd.DataFrame, benchmark: pd.DataFrame
) -> None:
    """Prove prediction == outcome == benchmark contract identity, row by row.

    For every scored row the moneyline/spread/total contract IDs recomputed from
    the *prediction* frame's own point columns must equal both the IDs carried on
    the prediction row and the benchmark's IDs, and the points must be equal
    within 1e-9. Any mismatch raises a :class:`ContractError` naming the game,
    market, both IDs and both points. Never logs-and-continues, never drops a row.
    """
    required = [
        "game_id", "home_spread", "total_line", "closing_minutes_to_kickoff",
        "moneyline_contract_id", "spread_contract_id", "total_contract_id",
    ]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ContractError(f"prediction frame missing contract columns: {missing}")

    bench = benchmark.set_index(benchmark["game_id"].astype(str))
    if bench.index.duplicated().any():
        raise ContractError("benchmark has duplicate game_id rows in validation")

    for _, row in predictions.iterrows():
        gid = str(row["game_id"])
        if gid not in bench.index:
            raise ContractError(
                f"game {gid}: scored row has no benchmark contract row"
            )
        b = bench.loc[gid]
        snapshot = row["closing_minutes_to_kickoff"]
        rc_ml, rc_sp, rc_to = _recompute_ids(row, snapshot)

        checks = [
            ("moneyline", rc_ml, row["moneyline_contract_id"], b["moneyline_contract_id"], None, None),
            ("spread", rc_sp, row["spread_contract_id"], b["spread_contract_id"],
             float(row["home_spread"]), float(b["closing_home_spread"])),
            ("total", rc_to, row["total_contract_id"], b["total_contract_id"],
             float(row["total_line"]), float(b["closing_total_line"])),
        ]
        for market, recomputed, pred_id, bench_id, pred_pt, bench_pt in checks:
            if not (recomputed == str(pred_id) == str(bench_id)):
                raise ContractError(
                    f"contract-id mismatch game={gid} market={market}: "
                    f"prediction_id={pred_id} benchmark_id={bench_id} "
                    f"recomputed_from_prediction_point={recomputed} "
                    f"prediction_point={pred_pt} benchmark_point={bench_pt}"
                )
            if pred_pt is not None and abs(pred_pt - bench_pt) > _POINT_TOL:
                raise ContractError(
                    f"contract-point mismatch game={gid} market={market}: "
                    f"prediction_point={pred_pt} benchmark_point={bench_pt} "
                    f"prediction_id={pred_id} benchmark_id={bench_id}"
                )
