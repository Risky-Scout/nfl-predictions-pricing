"""Run the pre-registered 2026 EPA tournament, bound to EXACT closing contracts.

R2 correctness contract (see ``nfl_hybrid.markets.exact_contract``): the exact
closing spread/total points are bound onto the games BEFORE the single augmented
feature-matrix build, so market features, market-residual targets, ATS/total
binary labels, every candidate probability, the benchmark probability, the graded
outcome and the displayed point all share ONE betting contract. There is no
post-prediction line overwrite. Every scored 2022-2024 row passes an exact
contract-identity validation before evaluation.

The functions here are importable and I/O-free so the runner is unit-tested
without touching private data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.evaluation.market_relative import evaluate_market_relative, MarketRelativeConfig
from nfl_hybrid.features.augmented_matrix import build_augmented_feature_matrix
from nfl_hybrid.labels import edge_to_nullable_binary
from nfl_hybrid.markets.exact_contract import (
    MARKET_SOURCE_CLOSING,
    MARKET_SOURCE_PROXY,
    bind_closing_contract,
    validate_benchmark,
    validate_closing_era_contract_states,
    validate_prediction_contract,
)
from nfl_hybrid.selection.epa_tournament import (
    CANDIDATES,
    _ordered_game_keys,
    run_walk_forward,
)

OUT = Path("outputs")
BASE = Path("data/backfill_2020_2025")
TEST_SEASONS = [2022, 2023, 2024]

# Batch 2B: authoritative, hard-coded season/contract-source policy. Seasons
# before 2022 may only use REFERENCE-LINE PROXY (training history); 2022-2024 may
# only use REAL-CLOSING. These are NOT derived from the data and deliberately
# exclude 2025 (broader 2025 validation is out of scope for this batch).
CLOSING_CONTRACT_START_SEASON = 2022

CLOSING_CONTRACT_SEASONS = {
    2022,
    2023,
    2024,
}
MARKET_MAP = {"home_win": "moneyline", "home_cover": "ats", "over": "total"}
MARKETS = tuple(MARKET_MAP.values())  # ("moneyline", "ats", "total")
EXPECTED_MARKETS = {"moneyline", "ats", "total"}
TOURNAMENT_MARKET_PROB_COLS = [
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
]
CONTRACT_META_COLS = [
    "game_id",
    "market_contract_source",
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
    "moneyline_contract_id",
    "spread_contract_id",
    "total_contract_id",
    "closing_home_spread",
    "closing_total_line",
    "closing_minutes_to_kickoff",
    "original_home_spread_reference",
    "original_total_line_reference",
    "contract_included",
    "contract_exclusion_reason",
    "benchmark_merge_status",
]
_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Pipeline (importable, no I/O)
# --------------------------------------------------------------------------- #
def build_contract_bound_matrix(games, pbp, benchmark, per_game_status=None):
    """Bind the exact closing contract BEFORE the single matrix build.

    Returns ``(matrix, manifest, binding)`` where ``matrix`` carries the exact
    contract metadata and every REAL-CLOSING row's ``home_spread``/``total_line``
    equal the bound closing points (asserted, tolerance ``1e-9``), with ATS/total
    labels proven to have been built from those closing points.
    """
    validate_benchmark(benchmark)
    binding = bind_closing_contract(games, benchmark, per_game_status)
    contract_games = binding.games

    # The one and only augmented matrix build for this tournament run.
    matrix, manifest = build_augmented_feature_matrix(contract_games, pbp)

    matrix = _attach_contract_metadata(matrix, contract_games)
    _assert_labels_match_closing_contract(matrix)
    return matrix, manifest, binding


def _attach_contract_metadata(matrix, contract_games):
    meta = contract_games[CONTRACT_META_COLS].copy()
    meta["game_id"] = meta["game_id"].astype(str)
    m = matrix.copy()
    m["game_id"] = m["game_id"].astype(str)
    out = m.merge(meta, on="game_id", how="left", validate="one_to_one")

    real = out["market_contract_source"] == MARKET_SOURCE_CLOSING
    if real.any():
        sub = out[real]
        dspread = np.abs(
            sub["home_spread"].to_numpy(float) - sub["closing_home_spread"].to_numpy(float)
        )
        dtotal = np.abs(
            sub["total_line"].to_numpy(float) - sub["closing_total_line"].to_numpy(float)
        )
        if (dspread > _TOL).any() or (dtotal > _TOL).any():
            raise ValueError(
                "matrix home_spread/total_line do not equal the bound closing points"
            )
    return out


def _assert_labels_match_closing_contract(matrix):
    """Prove ATS/total labels were built from the closing points (not relabelled)."""
    real = matrix["market_contract_source"] == MARKET_SOURCE_CLOSING
    if not real.any():
        return
    sub = matrix[real]
    expected_ats = edge_to_nullable_binary(
        sub["home_margin"].to_numpy(float) + sub["closing_home_spread"].to_numpy(float)
    ).reset_index(drop=True)
    expected_tot = edge_to_nullable_binary(
        sub["total_points"].to_numpy(float) - sub["closing_total_line"].to_numpy(float)
    ).reset_index(drop=True)
    if not _nullable_equal(sub["home_cover"].reset_index(drop=True), expected_ats):
        raise ValueError("home_cover label does not match the bound closing spread")
    if not _nullable_equal(sub["over"].reset_index(drop=True), expected_tot):
        raise ValueError("over label does not match the bound closing total")


def _nullable_equal(a: pd.Series, b: pd.Series) -> bool:
    a = a.to_numpy(dtype=object)
    b = b.to_numpy(dtype=object)
    for x, y in zip(a, b):
        xn, yn = pd.isna(x), pd.isna(y)
        if xn and yn:
            continue
        if xn != yn or x != y:
            return False
    return True


def _nullable_changed_count(a: pd.Series, b: pd.Series) -> int:
    a = a.to_numpy(dtype=object)
    b = b.to_numpy(dtype=object)
    changed = 0
    for x, y in zip(a, b):
        xn, yn = pd.isna(x), pd.isna(y)
        if xn and yn:
            continue
        if xn != yn or x != y:
            changed += 1
    return changed


def _validate_season_contract_sources(
    matrix: pd.DataFrame,
) -> dict:
    """Batch 2B: fail closed on any season/contract-source contamination.

    Enforces the authoritative policy BEFORE any tournament filtering:
      * seasons < 2022 must carry REFERENCE-LINE PROXY;
      * seasons in {2022, 2023, 2024} must be one of exactly two states, checked
        by the single authoritative validator
        :func:`nfl_hybrid.markets.exact_contract.validate_closing_era_contract_states`
        -- State A (included exact REAL-CLOSING contract) or State B (audited
        exclusion with a blank/null source and ``NO_BENCHMARK_ROW`` reason).

    The closing-era two-state machine is NOT reimplemented here: audited
    exclusions are validated (never exempted via a ``~audited_excluded`` bypass),
    and a malformed excluded row -- a closing-era proxy, a missing/wrong exclusion
    reason, a stray contract id or probability, a non-Boolean ``contract_included``
    -- raises before any filtering, fitting or scoring. No invalid row is silently
    removed.
    """
    required = {
        "game_id",
        "season",
        "market_contract_source",
    }

    missing_columns = sorted(
        required - set(matrix.columns)
    )

    if missing_columns:
        raise ValueError(
            "season/source validation missing required columns "
            f"{missing_columns}"
        )

    season = pd.to_numeric(
        matrix["season"],
        errors="coerce",
    )

    if season.isna().any():
        bad = matrix.loc[
            season.isna(),
            [
                "game_id",
                "season",
                "market_contract_source",
            ],
        ]

        raise ValueError(
            "season/source validation found non-numeric seasons:\n"
            + bad.head(50).to_string(index=False)
        )

    # Authoritative closing-era two-state validation (State A / State B). This is
    # the SINGLE implementation of the state machine; its exceptions are never
    # caught. A valid audited exclusion is NOT a source-validation failure.
    state_report = validate_closing_era_contract_states(
        matrix,
        context="EPA tournament season/source validation",
        closing_seasons=(
            2022,
            2023,
            2024,
        ),
    )

    # Pre-2022 rows keep the existing training-history rule: REFERENCE-LINE PROXY
    # only. NA sources are treated as "not proxy" without any fillna/astype(bool).
    season_np = season.to_numpy(dtype=float)
    pre_closing_np = season_np < CLOSING_CONTRACT_START_SEASON
    source_stripped = matrix["market_contract_source"].astype("string").str.strip()
    is_proxy_np = np.array(
        [value is True for value in (source_stripped == MARKET_SOURCE_PROXY).to_numpy(dtype=object)],
        dtype=bool,
    )

    invalid_pre_closing = pre_closing_np & (~is_proxy_np)
    if invalid_pre_closing.any():
        bad = matrix.loc[
            invalid_pre_closing,
            [
                "game_id",
                "season",
                "market_contract_source",
            ],
        ].copy()
        bad["expected_source"] = MARKET_SOURCE_PROXY
        raise ValueError(
            "invalid season/contract-source mapping:\n"
            + bad.head(50).to_string(index=False)
        )

    pre_closing_proxy_rows = int((pre_closing_np & is_proxy_np).sum())
    return {
        "pre_closing_proxy_rows": pre_closing_proxy_rows,
        "closing_era_real_rows": state_report["included_exact_contract_rows"],
        "included_exact_contract_rows": state_report["included_exact_contract_rows"],
        "audited_excluded_rows": state_report["audited_excluded_rows"],
        "validated_rows": pre_closing_proxy_rows + state_report["closing_era_rows"],
    }


def select_tournament_matrix(matrix):
    """Keep pre-closing REFERENCE-LINE PROXY training rows + REAL-CLOSING rows.

    Closing-season (2022-2024) rows without a complete REAL-CLOSING contract are
    dropped from BOTH fitting and testing (never silently proxied), but only after
    the full-history feature matrix has already been built.

    Batch 2B: the season/contract-source policy is validated BEFORE any filtering
    (a closing-era proxy row or a pre-2022 real-closing row fails closed here), and
    the retained rows are selected explicitly by exact (season-range, source) pair
    -- never by a lenient ``source.isin([...])`` that would admit the wrong source
    in the wrong season.
    """
    # Validate first: fail closed on any contaminated season/source mapping AND on
    # any closing-era row that is not a complete State A / State B contract.
    source_report = _validate_season_contract_sources(matrix)

    season_np = pd.to_numeric(
        matrix["season"],
        errors="raise",
    ).to_numpy(dtype=float)

    source_stripped = matrix["market_contract_source"].astype("string").str.strip()
    is_proxy_np = np.array(
        [v is True for v in (source_stripped == MARKET_SOURCE_PROXY).to_numpy(dtype=object)],
        dtype=bool,
    )
    is_real_closing_np = np.array(
        [v is True for v in (source_stripped == MARKET_SOURCE_CLOSING).to_numpy(dtype=object)],
        dtype=bool,
    )

    # The authoritative validator has already proven contract_included holds only
    # actual Booleans, so select with an exact identity test -- never a loose
    # astype(bool)/fillna that could read a string or number as True.
    included_np = np.array(
        [
            (value is True)
            or (isinstance(value, np.bool_) and bool(value))
            for value in matrix["contract_included"]
        ],
        dtype=bool,
    )

    pre_closing_np = season_np < CLOSING_CONTRACT_START_SEASON
    closing_era_np = np.isin(
        season_np, np.array(list(CLOSING_CONTRACT_SEASONS), dtype=float)
    )

    valid_pre_closing_training = pre_closing_np & is_proxy_np
    # Audited exclusions (contract_included False) are dropped ONLY after the
    # authoritative validator above has fully validated them.
    valid_closing_era = closing_era_np & included_np & is_real_closing_np

    selected = matrix.loc[
        valid_pre_closing_training
        | valid_closing_era
    ].copy()

    return selected.reset_index(drop=True)


def assert_complete_scorecard(scorecard):
    """Deliverable 3: require exactly five candidates x three markets = 15 pairs.

    Every registered candidate must contribute exactly one pooled row for each of
    the three markets, with no missing, extra, or duplicated (candidate, market)
    pair. This validated scorecard is the single source of truth the summary's
    candidate/market counts are derived from (see :func:`build_summary`).
    """
    expected = {(c, m) for c in CANDIDATES for m in MARKETS}
    n_expected = len(CANDIDATES) * len(MARKETS)
    if len(scorecard) != n_expected:
        raise ValueError(
            f"scorecard must have exactly {len(CANDIDATES)} candidates x "
            f"{len(MARKETS)} markets = {n_expected} pooled rows, got {len(scorecard)}"
        )
    pairs = list(zip(scorecard["candidate"], scorecard["market_name"]))
    got = set(pairs)
    if len(pairs) != len(got):
        raise ValueError("scorecard has duplicate (candidate, market) rows")
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            "scorecard (candidate, market) coverage mismatch; "
            f"missing={missing} extra={extra}"
        )
    return scorecard


def _validate_candidate_market_pairs(scorecard: pd.DataFrame) -> dict:
    """Issue 2: prove the scorecard holds exactly the 15 registered candidate x
    market pairs (5 candidates x {moneyline, ats, total}) with no missing, extra,
    or duplicated pair. Returns the validated candidate names and pair count so the
    summary derives its counts from actual scored content, never a module constant.
    """
    required_columns = {"candidate", "market_name"}
    missing_columns = sorted(required_columns - set(scorecard.columns))
    if missing_columns:
        raise RuntimeError(
            "EPA tournament scorecard missing required columns: "
            f"{missing_columns}"
        )

    pairs = scorecard[["candidate", "market_name"]].astype(str)

    duplicate_mask = pairs.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = (
            pairs.loc[duplicate_mask].drop_duplicates().to_dict("records")
        )
        raise RuntimeError(
            "EPA tournament scorecard contains duplicate "
            f"candidate-market pairs: {duplicates}"
        )

    expected_pairs = {
        (candidate, market)
        for candidate in CANDIDATES
        for market in EXPECTED_MARKETS
    }

    actual_pairs = set(pairs.itertuples(index=False, name=None))

    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(
            "EPA tournament candidate-market coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )

    candidate_names = sorted(pairs["candidate"].unique().tolist())

    return {
        "candidate_names": candidate_names,
        "candidate_market_count": len(actual_pairs),
    }


def _validate_candidate_game_coverage(
    results: dict[str, pd.DataFrame],
    tournament_matrix: pd.DataFrame,
    *,
    test_seasons,
) -> dict:
    """Batch 2A defense-in-depth: prove every candidate result carries the EXACT
    ordered requested-season game grid before any contract validation or scoring.

    Reuses the canonical :func:`_ordered_game_keys` helper (never a second
    game-key implementation). Fails closed on a missing, extra, duplicated, or
    reordered game, and on any per-season count that differs from the expected
    grid. Returns the expected/per-candidate counts for a later batch; the caller
    must not inner-join or filter candidate rows to make them match.
    """
    requested_seasons = [
        int(value)
        for value in test_seasons
    ]

    if len(requested_seasons) != len(
        set(requested_seasons)
    ):
        raise ValueError(
            "EPA tournament duplicate requested "
            "test seasons"
        )

    matrix_season = pd.to_numeric(
        tournament_matrix["season"],
        errors="coerce",
    )

    if matrix_season.isna().any():
        bad_rows = tournament_matrix.index[
            matrix_season.isna()
        ].tolist()

        raise ValueError(
            "EPA tournament expected-game grid "
            "contains non-numeric seasons at rows "
            f"{bad_rows[:20]}"
        )

    expected_parts = []

    for season in requested_seasons:
        part = tournament_matrix.loc[
            matrix_season == season
        ].copy()

        if part.empty:
            raise RuntimeError(
                "EPA tournament expected-game grid "
                f"has zero rows for test season {season}"
            )

        expected_parts.append(part)

    expected_frame = pd.concat(
        expected_parts,
        ignore_index=True,
    )

    expected_keys = _ordered_game_keys(
        expected_frame,
        context=(
            "EPA tournament expected exact-contract "
            "test rows"
        ),
    )

    if not expected_keys:
        raise RuntimeError(
            "EPA tournament has zero expected "
            "exact-contract test games"
        )

    expected_counts = (
        expected_frame.assign(
            _season_numeric=pd.to_numeric(
                expected_frame["season"],
                errors="raise",
            ).astype(int)
        )
        .groupby("_season_numeric")
        .size()
        .to_dict()
    )

    expected_counts = {
        int(key): int(value)
        for key, value in expected_counts.items()
    }

    candidate_counts = {}

    for name in CANDIDATES:
        if name not in results:
            raise RuntimeError(
                "EPA tournament missing candidate "
                f"result {name}"
            )

        actual_keys = _ordered_game_keys(
            results[name],
            context=(
                f"EPA tournament result for {name}"
            ),
        )

        if actual_keys != expected_keys:
            expected_set = set(expected_keys)
            actual_set = set(actual_keys)

            missing = sorted(
                expected_set - actual_set
            )

            extra = sorted(
                actual_set - expected_set
            )

            order_mismatch = (
                not missing
                and not extra
                and actual_keys != expected_keys
            )

            raise RuntimeError(
                f"{name}: EPA tournament game "
                "coverage mismatch; "
                f"missing={missing[:20]}, "
                f"extra={extra[:20]}, "
                f"order_mismatch={order_mismatch}"
            )

        actual_frame = results[name].copy()

        actual_frame["_season_numeric"] = (
            pd.to_numeric(
                actual_frame["season"],
                errors="raise",
            ).astype(int)
        )

        counts = (
            actual_frame
            .groupby("_season_numeric")
            .size()
            .to_dict()
        )

        counts = {
            int(key): int(value)
            for key, value in counts.items()
        }

        if counts != expected_counts:
            raise RuntimeError(
                f"{name}: per-season game counts "
                "do not match the expected grid; "
                f"expected={expected_counts}, "
                f"actual={counts}"
            )

        candidate_counts[name] = counts

    return {
        "expected_game_count": len(
            expected_keys
        ),
        "expected_games_by_season": (
            expected_counts
        ),
        "candidate_games_by_season": (
            candidate_counts
        ),
    }


def run_tournament(matrix, features, benchmark, *, test_seasons=TEST_SEASONS,
                   bootstrap_repetitions=2000):
    """Walk-forward all candidates on the contract-bound matrix and score them.

    No spread/total point is assigned, replaced or corrected after prediction; the
    market-probability frame is built directly from the same contract-bound
    prediction rows, and every candidate's rows pass exact contract validation
    before ``evaluate_market_relative`` is called.
    """
    tour = select_tournament_matrix(matrix)
    results = run_walk_forward(
        tour,
        features,
        test_seasons=test_seasons,
        market_prob_cols=TOURNAMENT_MARKET_PROB_COLS,
        expected_test_contract_source=MARKET_SOURCE_CLOSING,
        require_complete=True,
    )
    # Issue 2: every registered candidate must be scored; a missing one is a silent
    # omission from the scorecard, so fail closed naming it rather than skipping. A
    # registered candidate may never disappear from ``results``.
    expected_names = set(CANDIDATES)
    actual_names = set(results)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(
            "EPA tournament candidate set mismatch; "
            f"missing={missing}, extra={extra}"
        )
    # Batch 2A defense-in-depth: every candidate must carry the exact ordered
    # requested-season game grid BEFORE any contract validation or evaluation. A
    # coverage mismatch fails closed here; no candidate is scored on a reshaped
    # game set (report retained for a later batch).
    game_coverage_report = (
        _validate_candidate_game_coverage(
            results,
            tour,
            test_seasons=test_seasons,
        )
    )
    all_rows = []
    for name in CANDIDATES:
        pred = results[name].reset_index(drop=True)
        # Contract validator runs BEFORE evaluation; a mismatch raises and no
        # scoring proceeds. No inner merge, no line overwrite.
        validate_prediction_contract(pred, benchmark)
        market_probs = pd.DataFrame(
            {
                "market_ml_home_probability": pred["tournament_market_ml_home_probability"].to_numpy(),
                "market_cover_home_probability": pred["tournament_market_cover_home_probability"].to_numpy(),
                "market_over_probability": pred["tournament_market_over_probability"].to_numpy(),
                "market_source": MARKET_SOURCE_CLOSING,
            }
        )
        res = evaluate_market_relative(
            pred, config=MarketRelativeConfig(bootstrap_repetitions=bootstrap_repetitions),
            market_probabilities=market_probs,
        )
        prob = res["probability_scorecard"]
        pooled = prob[prob["segment"] == "pooled"].copy()
        pooled.insert(0, "candidate", name)
        pooled.insert(1, "benchmark", MARKET_SOURCE_CLOSING)
        all_rows.append(pooled)

    scorecard = pd.concat(all_rows, ignore_index=True)
    scorecard["market_name"] = scorecard["market"].map(MARKET_MAP)
    # Issue 2: fail closed unless the scorecard is exactly the 15 registered
    # candidate x market pairs (no missing, extra, or duplicate pair) before any
    # promotion logic or return.
    pair_report = _validate_candidate_market_pairs(scorecard)
    scorecard["promotion"] = scorecard.apply(_decide_promotion, axis=1)
    return scorecard, results


def _decide_promotion(row):
    ll = row["log_loss_gain_vs_market"] > 0
    br = row["brier_gain_vs_market"] >= 0
    if row["market_name"] in ("ats", "total"):
        acc = bool(row["beats_breakeven_ci"])
        return "PROVISIONAL_ONLY" if (ll and br and acc) else "RETAIN_BASELINE"
    return "PROVISIONAL_ONLY" if (ll and br) else "RETAIN_BASELINE"


def build_coverage_audit(matrix, binding):
    """One row per 2022-2024 matrix game with its inclusion/exclusion reason."""
    matrix_ids = set(matrix["game_id"].astype(str))
    audit = binding.audit.copy()
    audit["game_id"] = audit["game_id"].astype(str)
    season = pd.to_numeric(audit["season"], errors="coerce")
    audit = audit[season.isin(TEST_SEASONS) & audit["game_id"].isin(matrix_ids)]
    return audit.reset_index(drop=True)


def build_summary(matrix, binding, scorecard, benchmark, *, input_row_count=None):
    # Issue 2: the reported candidate/market counts are derived from the VALIDATED
    # scorecard (proven exactly 5 x 3 = 15 unique candidate-market pairs), never
    # from module constants or an unvalidated row count, so the summary can never
    # disagree with what was actually scored.
    pair_report = _validate_candidate_market_pairs(scorecard)
    n_candidates = len(pair_report["candidate_names"])
    n_candidate_markets = int(pair_report["candidate_market_count"])

    real = matrix["market_contract_source"] == MARKET_SOURCE_CLOSING
    dev = matrix[pd.to_numeric(matrix["season"], errors="coerce").isin(TEST_SEASONS)]
    dev_real = dev[dev["market_contract_source"] == MARKET_SOURCE_CLOSING]

    total_by_season = {int(s): int(n) for s, n in dev.groupby("season").size().items()}
    exact_by_season = {int(s): int(n) for s, n in dev_real.groupby("season").size().items()}
    excluded_by_season = {
        s: total_by_season.get(s, 0) - exact_by_season.get(s, 0) for s in total_by_season
    }
    coverage_by_season = {
        s: round(exact_by_season.get(s, 0) / total_by_season[s], 4)
        for s in total_by_season if total_by_season[s]
    }

    sub = matrix[real]
    spread_disagree = int(
        (np.abs(sub["original_home_spread_reference"].to_numpy(float)
                - sub["closing_home_spread"].to_numpy(float)) > _TOL).sum()
    )
    total_disagree = int(
        (np.abs(sub["original_total_line_reference"].to_numpy(float)
                - sub["closing_total_line"].to_numpy(float)) > _TOL).sum()
    )
    ref_ats = edge_to_nullable_binary(
        sub["home_margin"].to_numpy(float) + sub["original_home_spread_reference"].to_numpy(float)
    ).reset_index(drop=True)
    ref_tot = edge_to_nullable_binary(
        sub["total_points"].to_numpy(float) - sub["original_total_line_reference"].to_numpy(float)
    ).reset_index(drop=True)
    ats_labels_changed = _nullable_changed_count(sub["home_cover"].reset_index(drop=True), ref_ats)
    total_labels_changed = _nullable_changed_count(sub["over"].reset_index(drop=True), ref_tot)

    for s in TEST_SEASONS:
        if exact_by_season.get(s, 0) == 0:
            raise ValueError(f"test season {s} has zero exact-contract scored rows")

    return {
        "contract_policy": "exact-closing-bound-before-matrix-v1",
        "benchmark_aggregation": "exact_point_consensus_v1",
        "post_prediction_line_overwrite": False,
        "contract_validation_passed": True,
        "test_seasons": TEST_SEASONS,
        "benchmark": "REAL-CLOSING (exact-point de-vigged closing, 2022-2024)",
        "n_candidates": n_candidates,
        "n_candidate_markets": n_candidate_markets,
        "n_promoted": int((scorecard["promotion"] == "PROVISIONAL_ONLY").sum()),
        "promoted": scorecard[scorecard["promotion"] == "PROVISIONAL_ONLY"][
            ["candidate", "market_name"]
        ].to_dict("records"),
        "total_matrix_games_by_season": total_by_season,
        "exact_contract_games_by_season": exact_by_season,
        "excluded_games_by_season": excluded_by_season,
        "coverage_fraction_by_season": coverage_by_season,
        "selected_spread_point_disagreement_count": spread_disagree,
        "selected_total_point_disagreement_count": total_disagree,
        "games_reference_spread_differs_from_closing": spread_disagree,
        "games_reference_total_differs_from_closing": total_disagree,
        "ats_labels_changed_by_closing_contract": ats_labels_changed,
        "total_labels_changed_by_closing_contract": total_labels_changed,
        "benchmark_input_row_count": int(input_row_count) if input_row_count is not None else None,
        "benchmark_output_game_count": int(len(benchmark)),
    }


# --------------------------------------------------------------------------- #
# I/O entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    games = pd.read_parquet(BASE / "canonical" / "games.parquet")
    pbp = pd.read_parquet(BASE / "raw" / "pbp.parquet")
    benchmark = pd.read_parquet(OUT / "real_closing_benchmark_2022_2024.parquet")

    input_row_count = None
    summ_path = OUT / "real_closing_benchmark_summary.json"
    if summ_path.exists():
        try:
            input_row_count = json.loads(summ_path.read_text()).get("benchmark_input_row_count")
        except (ValueError, OSError):
            input_row_count = None

    matrix, manifest, binding = build_contract_bound_matrix(games, pbp, benchmark)
    features = manifest["all_features"]
    print(f"features ({len(features)}): {features}")

    scorecard, _ = run_tournament(matrix, features, benchmark)

    cols = [
        "candidate", "market_name", "n", "model_brier", "market_brier", "brier_gain_vs_market",
        "model_log_loss", "market_log_loss", "log_loss_gain_vs_market",
        "pick_accuracy", "pick_accuracy_ci_lower", "pick_accuracy_ci_upper",
        "beats_breakeven_ci", "promotion",
    ]
    OUT.mkdir(exist_ok=True)
    scorecard[cols].to_csv(OUT / "epa_tournament_scorecard.csv", index=False)

    coverage = build_coverage_audit(matrix, binding)
    coverage.to_csv(OUT / "epa_tournament_contract_coverage.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n=== FULL SCORECARD (every candidate x market, vs REAL-CLOSING) ===")
    print("all scored rows passed exact-contract validation")
    print(scorecard[cols].to_string(index=False))
    promoted = scorecard[scorecard["promotion"] == "PROVISIONAL_ONLY"]
    print(f"\nPROMOTED (PROVISIONAL_ONLY): {len(promoted)} of {len(scorecard)} candidate-markets")

    summary = build_summary(matrix, binding, scorecard, benchmark, input_row_count=input_row_count)
    (OUT / "epa_tournament_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nsummary:", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
