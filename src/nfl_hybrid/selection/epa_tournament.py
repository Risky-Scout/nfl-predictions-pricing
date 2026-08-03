"""Pre-registered 2026 EPA candidate tournament (dev seasons 2022-2024).

Five candidates (see docs/model-selection/candidates-2026/registry.md), each
producing per-game no-tie/no-push probabilities for moneyline, ATS and totals,
scored against the REAL-CLOSING de-vigged market. Walk-forward: each test season
trains only on prior seasons. 2025 is never passed in.

Kept intentionally small and explicit so a skeptic can audit every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from nfl_hybrid.modern.joint_score import JointScoreModel

PROB_COLS = [
    "home_win_probability_no_tie",
    "home_cover_probability_no_push",
    "over_probability_no_push",
]
_EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _market_baseline_probs(test: pd.DataFrame, mkt_col, *, market: str) -> np.ndarray:
    """Validated per-test-row market-baseline fallback for a degenerate target.

    When a market's nullable target (pd.NA for ties/pushes) leaves zero valid
    fitting rows or a single observed class, no discriminative classifier can be
    fit. The candidate then returns the reference-market probability column for
    that market — one value per test row, in test-row order. Never silently
    substitutes 0.50: a missing, wrong-length, non-finite, or out-of-bounds
    baseline raises a clear market-specific error instead.
    """
    if mkt_col is None:
        raise ValueError(
            f"{market}: nullable-target fallback needs a market-baseline column, "
            "but no market_prob_col_map was supplied."
        )
    if mkt_col not in test.columns:
        raise ValueError(
            f"{market}: market-baseline column '{mkt_col}' is missing from the test frame."
        )
    p = test[mkt_col].to_numpy(dtype=float)
    if p.shape[0] != len(test):
        raise ValueError(
            f"{market}: market-baseline column '{mkt_col}' length {p.shape[0]} "
            f"!= len(test) {len(test)}."
        )
    if not np.isfinite(p).all():
        raise ValueError(
            f"{market}: market-baseline column '{mkt_col}' contains non-finite values."
        )
    if not ((p >= 0.0) & (p <= 1.0)).all():
        raise ValueError(
            f"{market}: market-baseline column '{mkt_col}' has values outside [0, 1]."
        )
    return _clip(p)


def _margin_to_probs(pred_margin, pred_total, home_spread, total_line, margin_sd, total_sd):
    home_win = 1.0 - norm.cdf((0.0 - pred_margin) / margin_sd)
    ats_edge = pred_margin + home_spread
    cover = norm.cdf(ats_edge / margin_sd)
    over = norm.cdf((pred_total - total_line) / total_sd)
    return _clip(home_win), _clip(cover), _clip(over)


def _prep(frame, features):
    x = frame[features].to_numpy(float)
    return np.nan_to_num(x, nan=0.0)


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def candidate_market_residual(train, test, features):
    """C1: ridge on EPA/rest predicting margin & total residuals vs market."""
    xtr, xte = _prep(train, features), _prep(test, features)
    margin_resid = train["home_margin"].to_numpy(float) - (-train["home_spread"].to_numpy(float))
    total_resid = train["total_points"].to_numpy(float) - train["total_line"].to_numpy(float)

    m = Ridge(alpha=10.0).fit(xtr, margin_resid)
    t = Ridge(alpha=10.0).fit(xtr, total_resid)
    pred_margin = -test["home_spread"].to_numpy(float) + m.predict(xte)
    pred_total = test["total_line"].to_numpy(float) + t.predict(xte)

    margin_sd = max(float(np.std(margin_resid - m.predict(xtr), ddof=1)), 1.0)
    total_sd = max(float(np.std(total_resid - t.predict(xtr), ddof=1)), 1.0)
    hw, cov, ov = _margin_to_probs(
        pred_margin, pred_total, test["home_spread"].to_numpy(float),
        test["total_line"].to_numpy(float), margin_sd, total_sd,
    )
    return pd.DataFrame(
        {PROB_COLS[0]: hw, PROB_COLS[1]: cov, PROB_COLS[2]: ov, "predicted_margin": pred_margin},
        index=test.index,
    )


def _isotonic_gbm(xtr, ytr, xcal, ycal, xte):
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, random_state=42,
    )
    clf.fit(xtr, ytr)
    raw_te = clf.predict_proba(xte)[:, 1]
    # Cases C/D: zero valid calibration rows or a single calibration class leave
    # isotonic undefined -> return clipped *uncalibrated* GBM probabilities. Guard
    # BEFORE predict_proba(xcal), which raises on an empty calibration array.
    if len(ycal) == 0 or np.unique(ycal).size < 2:
        return _clip(raw_te)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(xcal)[:, 1], ycal)
    return _clip(iso.predict(raw_te))


def candidate_gbm(train, test, features, calibration_season, market_prob_col_map=None):
    """C2: GBM per market, isotonic-calibrated on the prior season only.

    Targets are nullable (pd.NA for ties/pushes). Each market independently
    filters its own valid (0/1) fit and calibration rows; predictions are still
    produced for every test row. Degenerate-target fallbacks (per market):

      * zero valid fit rows OR one fit class -> validated market baseline;
      * both fit classes but zero/one-class calibration -> uncalibrated GBM
        (handled inside ``_isotonic_gbm``, which then skips isotonic);
      * both fit and calibration classes -> GBM + isotonic (unchanged).

    ``market_prob_col_map`` names the [ml, ats, total] reference-probability
    columns used only as the degenerate-target fallback baseline.
    """
    cal = train[train["season"] == calibration_season]
    fit = train[train["season"] != calibration_season]
    if len(cal) < 30 or len(fit) < 50:
        cal, fit = train, train
    xfit, xcal, xte = _prep(fit, features), _prep(cal, features), _prep(test, features)
    baselines = market_prob_col_map if market_prob_col_map is not None else [None, None, None]
    out = {}
    for col, target, mkt_col in zip(PROB_COLS, ["home_win", "home_cover", "over"], baselines):
        fit_valid = fit[target].isin([0, 1]).to_numpy()
        yfit = fit.loc[fit_valid, target]
        # Cases A/B: no valid fit rows or a single observed class -> cannot fit a
        # discriminative GBM; fall back to the validated market baseline.
        if fit_valid.sum() == 0 or yfit.nunique() < 2:
            out[col] = _market_baseline_probs(test, mkt_col, market=target)
            continue
        cal_valid = cal[target].isin([0, 1]).to_numpy()
        out[col] = _isotonic_gbm(
            xfit[fit_valid], yfit.to_numpy(int),
            xcal[cal_valid], cal.loc[cal_valid, target].to_numpy(int),
            xte,
        )
    return pd.DataFrame(out, index=test.index)


def candidate_logistic(train, test, features, market_prob_col_map=None):
    """C3: standardized per-market logistic regression (interpretable floor).

    Each market drops its own nullable (tie/push) label rows before fitting
    preprocessing + classifier, but scores every test row. With zero valid fit
    rows or a single observed class the classifier is undefined, so the market
    returns its validated baseline instead. ``market_prob_col_map`` names the
    [ml, ats, total] reference-probability fallback columns.
    """
    xtr, xte = _prep(train, features), _prep(test, features)
    baselines = market_prob_col_map if market_prob_col_map is not None else [None, None, None]
    out = {}
    for col, target, mkt_col in zip(PROB_COLS, ["home_win", "home_cover", "over"], baselines):
        valid = train[target].isin([0, 1]).to_numpy()
        ytr = train.loc[valid, target]
        if valid.sum() == 0 or ytr.nunique() < 2:
            out[col] = _market_baseline_probs(test, mkt_col, market=target)
            continue
        scaler = StandardScaler().fit(xtr[valid])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(
            scaler.transform(xtr[valid]), ytr.to_numpy(int)
        )
        out[col] = _clip(clf.predict_proba(scaler.transform(xte))[:, 1])
    return pd.DataFrame(out, index=test.index)


def candidate_jointscore_epa(train, test, features, calibration_season):
    """C4: JointScoreModel with market + EPA numeric features."""
    model = JointScoreModel(numeric_features=list(features))
    cal = train[train["season"] == calibration_season]
    fit = train[train["season"] != calibration_season]
    if len(cal) < 30 or len(fit) < 50:
        fit, cal = train, train
    model.fit(fit, calibration=cal)
    pred = model.predict_markets(test)
    return pd.DataFrame(
        {
            PROB_COLS[0]: _clip(pred["home_win_probability_no_tie"].to_numpy()),
            PROB_COLS[1]: _clip(pred["home_cover_probability_no_push"].to_numpy()),
            PROB_COLS[2]: _clip(pred["over_probability_no_push"].to_numpy()),
            "predicted_margin": pred["predicted_margin"].to_numpy(),
        },
        index=test.index,
    )


def candidate_stacked(train, test, features, calibration_season, market_prob_col_map):
    """C5: logistic meta over C1-C4 OOF probs + reference market prob meta-feature.

    Base models are fit on all-but-last training season and predict the held-out
    last training season -> meta-training features (leakage-safe blend holdout).
    Base models are then refit on all train to predict the test season.
    """
    seasons = sorted(train["season"].unique())
    holdout = seasons[-1]
    base_fit = train[train["season"] != holdout]
    meta_train = train[train["season"] == holdout]
    if len(base_fit) < 50 or len(meta_train) < 30:
        return None  # not enough seasons to stack cleanly

    def base_probs(tr, te, cal_season):
        c1 = candidate_market_residual(tr, te, features)
        c2 = candidate_gbm(tr, te, features, cal_season, market_prob_col_map)
        c3 = candidate_logistic(tr, te, features, market_prob_col_map)
        c4 = candidate_jointscore_epa(tr, te, features, cal_season)
        return c1, c2, c3, c4

    out = {}
    for col, target, mkt_col in zip(PROB_COLS, ["home_win", "home_cover", "over"], market_prob_col_map):
        # meta-train features
        cal_inner = sorted(base_fit["season"].unique())[-1]
        mc1, mc2, mc3, mc4 = base_probs(base_fit, meta_train, cal_inner)
        meta_x = np.column_stack([
            mc1[col].to_numpy(), mc2[col].to_numpy(), mc3[col].to_numpy(), mc4[col].to_numpy(),
            meta_train[mkt_col].to_numpy(float),
        ])
        # Exclude this meta-target's nullable (tie/push) rows before fitting; never
        # cast the nullable target to int before filtering.
        meta_valid = meta_train[target].isin([0, 1]).to_numpy()
        meta_y = meta_train.loc[meta_valid, target].to_numpy(int)
        # Zero valid meta rows or a single meta class -> the meta-learner is
        # undefined. Return one validated market-baseline probability per TEST row
        # (never the previous [:0] empty / training-frame array).
        if meta_valid.sum() == 0 or np.unique(meta_y).size < 2:
            out[col] = _market_baseline_probs(test, mkt_col, market=target)
            continue
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(
            np.nan_to_num(meta_x[meta_valid]), meta_y
        )
        # test features from base models refit on all train
        tc1, tc2, tc3, tc4 = base_probs(train, test, calibration_season)
        test_x = np.column_stack([
            tc1[col].to_numpy(), tc2[col].to_numpy(), tc3[col].to_numpy(), tc4[col].to_numpy(),
            test[mkt_col].to_numpy(float),
        ])
        out[col] = _clip(meta.predict_proba(np.nan_to_num(test_x))[:, 1])
    return pd.DataFrame(out, index=test.index)


CANDIDATES = ("C1_market_residual", "C2_gbm", "C3_logistic", "C4_jointscore_epa", "C5_stacked")

# Exact-contract metadata carried untouched from the matrix through prediction so
# the pre-evaluation validator can prove market/outcome/prediction identity. These
# columns are never fitted on or altered by any candidate.
CONTRACT_COLS = (
    "market_contract_source",
    "moneyline_contract_id",
    "spread_contract_id",
    "total_contract_id",
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
    "closing_home_spread",
    "closing_total_line",
    "closing_minutes_to_kickoff",
)
_CONTRACT_POINT_TOL = 1e-9


def _assert_test_contract(test: pd.DataFrame, expected_source: str) -> None:
    """Fail closed unless every test row carries a complete, matching contract.

    Used only when ``run_walk_forward`` is called with an explicit expected
    test-contract source (the real-closing tournament). Requires unique game IDs,
    all contract columns, non-null IDs, finite line values, finite probabilities
    in (0, 1), the exact expected source, and ``home_spread == closing_home_spread``
    / ``total_line == closing_total_line``.
    """
    if test["game_id"].duplicated().any():
        raise ValueError("contract-strict walk-forward: duplicate test game_id rows")
    missing = [c for c in CONTRACT_COLS if c not in test.columns]
    if missing:
        raise ValueError(f"contract-strict walk-forward: test frame missing {missing}")
    id_cols = ["moneyline_contract_id", "spread_contract_id", "total_contract_id"]
    if test[id_cols].isna().any().any():
        raise ValueError("contract-strict walk-forward: null contract id in test frame")
    if (test[id_cols].astype(str).apply(lambda s: s.str.len() == 0)).any().any():
        raise ValueError("contract-strict walk-forward: empty contract id in test frame")
    for c in ("home_spread", "total_line", "closing_home_spread", "closing_total_line",
              "closing_minutes_to_kickoff"):
        if not np.isfinite(test[c].to_numpy(dtype=float)).all():
            raise ValueError(f"contract-strict walk-forward: non-finite {c} in test frame")
    for c in ("tournament_market_ml_home_probability",
              "tournament_market_cover_home_probability",
              "tournament_market_over_probability"):
        p = test[c].to_numpy(dtype=float)
        if not (np.isfinite(p).all() and ((p > 0.0) & (p < 1.0)).all()):
            raise ValueError(f"contract-strict walk-forward: {c} not all finite in (0, 1)")
    if (test["market_contract_source"].astype(str) != expected_source).any():
        raise ValueError(
            "contract-strict walk-forward: a test row is not "
            f"{expected_source!r} (proxy/reference rows may never be scored)."
        )
    dspread = np.abs(test["home_spread"].to_numpy(float) - test["closing_home_spread"].to_numpy(float))
    dtotal = np.abs(test["total_line"].to_numpy(float) - test["closing_total_line"].to_numpy(float))
    if (dspread > _CONTRACT_POINT_TOL).any() or (dtotal > _CONTRACT_POINT_TOL).any():
        raise ValueError(
            "contract-strict walk-forward: home_spread/total_line diverge from the "
            "bound closing points."
        )


def _ordered_game_keys(
    frame: pd.DataFrame,
    *,
    context: str,
) -> list[tuple[int, str]]:
    """Canonical ordered (season, game_id) key list for a frame, failing closed.

    The single game-key normalization used by both ``run_walk_forward`` and the
    runner: it requires ``season``/``game_id`` columns, coerces season to int,
    normalizes game_id to a stripped string, and rejects missing/empty game IDs,
    non-numeric seasons, or any duplicate season/game key. Row order is preserved
    exactly so callers can assert exact ordered equality against an expected grid.
    """
    required = {"season", "game_id"}
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(
            f"{context}: missing game-key columns {missing}"
        )

    season = pd.to_numeric(
        frame["season"],
        errors="coerce",
    )

    game_id = frame["game_id"].astype("string")

    if season.isna().any():
        bad_rows = frame.index[
            season.isna()
        ].tolist()

        raise ValueError(
            f"{context}: non-numeric season values "
            f"at rows {bad_rows[:20]}"
        )

    if game_id.isna().any():
        bad_rows = frame.index[
            game_id.isna()
        ].tolist()

        raise ValueError(
            f"{context}: missing game_id values "
            f"at rows {bad_rows[:20]}"
        )

    normalized_game_id = game_id.str.strip()

    if (normalized_game_id == "").any():
        bad_rows = frame.index[
            normalized_game_id == ""
        ].tolist()

        raise ValueError(
            f"{context}: empty game_id values "
            f"at rows {bad_rows[:20]}"
        )

    key_frame = pd.DataFrame(
        {
            "season": season.astype(int),
            "game_id": normalized_game_id.astype(str),
        },
        index=frame.index,
    )

    duplicate_mask = key_frame.duplicated(
        subset=["season", "game_id"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            key_frame.loc[
                duplicate_mask,
                ["season", "game_id"],
            ]
            .drop_duplicates()
            .sort_values(
                ["season", "game_id"],
                kind="stable",
            )
            .itertuples(index=False, name=None)
        )

        raise ValueError(
            f"{context}: duplicate season/game keys "
            f"{list(duplicates)[:20]}"
        )

    return list(
        key_frame[
            ["season", "game_id"]
        ].itertuples(index=False, name=None)
    )


def run_walk_forward(matrix, features, *, test_seasons, market_prob_cols,
                     expected_test_contract_source=None, require_complete=False):
    """Return {candidate: predictions_df} pooled over test seasons.

    ``market_prob_cols`` maps [ml, ats, total] market-probability columns present
    in ``matrix`` (used as the C5 meta-feature and the degenerate-target fallback
    baseline).

    ``expected_test_contract_source`` validates CONTRACT SOURCE / METADATA only:
    when given (e.g. ``"REAL-CLOSING"``), every scored *test* row must carry a
    complete matching exact contract (see :func:`_assert_test_contract`) and the
    returned prediction frames carry the exact contract metadata untouched.
    Training rows are not source-restricted (pre-closing proxy history is allowed).
    When it is ``None`` the contract columns are simply propagated if present and no
    source restriction is applied. This flag NEVER controls completeness.

    ``require_complete`` is the INDEPENDENT completeness contract (default
    ``False``): when true, every requested test season must be processed and every
    registered candidate must complete successfully for it -- a requested season
    may never be silently skipped, a registered candidate may never be silently
    dropped, and on return every requested season is proven to have been produced
    by every registered candidate with a nonempty pooled result. It is orthogonal
    to ``expected_test_contract_source``: contract validation does not imply
    completeness and completeness does not imply contract validation.
    """
    # Contract-source validation and completeness enforcement are independent.
    # ``validate_contract`` gates ONLY the contract-metadata checks; it must never
    # decide season/candidate completeness.
    validate_contract = expected_test_contract_source is not None

    requested_seasons = [int(value) for value in test_seasons]
    if len(requested_seasons) != len(set(requested_seasons)):
        raise ValueError(
            "run_walk_forward: duplicate requested test seasons"
        )

    processed_seasons: list[int] = []
    candidate_processed_seasons = {name: set() for name in CANDIDATES}
    # Batch 2A: the exact ordered (season, game_id) grid every candidate must
    # reproduce, accumulated once per fully-completed season (never per candidate).
    expected_pooled_keys: list[tuple[int, str]] = []

    results = {name: [] for name in CANDIDATES}
    for test_season in requested_seasons:
        train = matrix[matrix["season"] < test_season].copy()
        test = matrix[matrix["season"] == test_season].copy()
        # Issue 1: never SILENTLY skip a requested historical test season. When
        # completeness is required an unscoreable requested season is a hidden
        # failure, so fail closed naming the season; otherwise keep lenient skip.
        if len(test) == 0:
            if require_complete:
                raise ValueError(
                    "run_walk_forward: test season "
                    f"{test_season} has zero eligible rows"
                )
            continue
        if len(train) < 100:
            if require_complete:
                raise ValueError(
                    "run_walk_forward: insufficient historical training rows "
                    f"for test season {test_season}: "
                    f"train={len(train)}, required=100"
                )
            continue
        if validate_contract:
            _assert_test_contract(test, expected_test_contract_source)
        # Batch 2A: the exact ordered game grid this scored season must produce.
        # Derived from the eligible test frame before any candidate result is
        # accepted; also proves the test frame itself has unique season/game keys.
        expected_season_keys = _ordered_game_keys(
            test,
            context=(
                "run_walk_forward expected test rows "
                f"for season {test_season}"
            ),
        )
        cal_season = sorted(train["season"].unique())[-1]

        preds = {
            "C1_market_residual": candidate_market_residual(train, test, features),
            "C2_gbm": candidate_gbm(train, test, features, cal_season, market_prob_cols),
            "C3_logistic": candidate_logistic(train, test, features, market_prob_cols),
            "C4_jointscore_epa": candidate_jointscore_epa(train, test, features, cal_season),
            "C5_stacked": candidate_stacked(train, test, features, cal_season, market_prob_cols),
        }
        # Issue 2: the seasonal predicted set must cover the registry EXACTLY, so a
        # registered candidate can never be silently dropped (nor an unregistered
        # one appear) for a scored season.
        if require_complete:
            expected_names = set(CANDIDATES)
            actual_names = set(preds)
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                raise RuntimeError(
                    "run_walk_forward: seasonal candidate set mismatch "
                    f"for test season {test_season}; "
                    f"missing={missing}, extra={extra}"
                )
        keep = ["game_id", "season", "week", "home_win", "home_cover", "over",
                "home_margin", "total_points", "home_spread", "total_line"]
        keep += [c for c in CONTRACT_COLS if c in test.columns]
        for name in CANDIDATES:
            pred = preds[name]
            if pred is None:
                # Issue 2: a registered candidate that produces nothing for a scored
                # season is a silent omission -> fail closed when completeness is
                # required.
                if require_complete:
                    raise RuntimeError(
                        f"{name}: candidate returned None "
                        f"for test season {test_season}"
                    )
                continue
            # Candidates return one row per test row indexed by test.index; refuse
            # anything that is not an exact, unique-game realignment of the test set.
            if len(pred) != len(test) or not pred.index.equals(test.index):
                raise ValueError(
                    f"{name}: candidate output is not one aligned row per test row"
                )
            merged = pd.concat(
                [test[keep].reset_index(drop=True), pred.reset_index(drop=True)], axis=1
            )
            if validate_contract and merged["game_id"].duplicated().any():
                raise ValueError(f"{name}: duplicate game_id after candidate join")
            # Batch 2A: the merged candidate rows must be the EXACT ordered game
            # grid of the eligible test frame -- no missing, extra, duplicated, or
            # reordered game. This is an invariant check in addition to the
            # length/index guards above (a candidate must never silently reshape
            # the scored game set).
            actual_season_keys = _ordered_game_keys(
                merged,
                context=(
                    f"{name} candidate result "
                    f"for season {test_season}"
                ),
            )
            if actual_season_keys != expected_season_keys:
                expected_set = set(expected_season_keys)
                actual_set = set(actual_season_keys)
                missing = sorted(expected_set - actual_set)
                extra = sorted(actual_set - expected_set)
                order_mismatch = (
                    not missing
                    and not extra
                    and actual_season_keys != expected_season_keys
                )
                raise RuntimeError(
                    f"{name}: candidate game coverage mismatch "
                    f"for test season {test_season}; "
                    f"missing={missing[:20]}, "
                    f"extra={extra[:20]}, "
                    f"order_mismatch={order_mismatch}"
                )
            results[name].append(merged)
            # Only record a candidate/season as processed AFTER its result has been
            # successfully validated and appended.
            candidate_processed_seasons[name].add(int(test_season))

        # Mark the season processed only once EVERY registered candidate has
        # successfully appended a result for it (never on partial coverage).
        if all(int(test_season) in candidate_processed_seasons[name] for name in CANDIDATES):
            processed_seasons.append(int(test_season))
            # Batch 2A: append this season's expected keys to the pooled grid
            # EXACTLY ONCE (only on full completion), preserving requested-season
            # order and, within a season, the original test row order.
            expected_pooled_keys.extend(expected_season_keys)

    out = {name: pd.concat(frames, ignore_index=True)
           for name, frames in results.items() if frames}

    if require_complete:
        # First: every requested season was processed, no more and no fewer.
        missing_seasons = sorted(set(requested_seasons) - set(processed_seasons))
        extra_seasons = sorted(set(processed_seasons) - set(requested_seasons))
        if missing_seasons or extra_seasons:
            raise RuntimeError(
                "run_walk_forward: requested test seasons were not "
                "completely processed; "
                f"missing={missing_seasons}, extra={extra_seasons}"
            )

        # Second: every registered candidate covered every requested season.
        for name in CANDIDATES:
            missing = sorted(set(requested_seasons) - candidate_processed_seasons[name])
            extra = sorted(candidate_processed_seasons[name] - set(requested_seasons))
            if missing or extra:
                raise RuntimeError(
                    f"{name}: completed-season mismatch; "
                    f"missing={missing}, extra={extra}"
                )

        # Third: the pooled candidate set is exactly the registry (never trust the
        # initial dict keys as proof that a candidate produced data).
        expected_names = set(CANDIDATES)
        actual_names = set(out)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise RuntimeError(
                "run_walk_forward: pooled candidate set mismatch; "
                f"missing={missing}, extra={extra}"
            )

        # Fourth: every pooled result actually holds rows.
        for name in CANDIDATES:
            if out[name].empty:
                raise RuntimeError(
                    f"run_walk_forward: empty pooled result for {name}"
                )

        # Fifth (Batch 2A): every pooled candidate must reproduce the COMPLETE
        # requested-season game grid in exact order -- no missing, extra,
        # duplicated, or reordered game across the pooled result.
        for name in CANDIDATES:
            actual_pooled_keys = _ordered_game_keys(
                out[name],
                context=(
                    f"{name} pooled candidate result"
                ),
            )
            if actual_pooled_keys != expected_pooled_keys:
                expected_set = set(expected_pooled_keys)
                actual_set = set(actual_pooled_keys)
                missing = sorted(expected_set - actual_set)
                extra = sorted(actual_set - expected_set)
                order_mismatch = (
                    not missing
                    and not extra
                    and actual_pooled_keys != expected_pooled_keys
                )
                raise RuntimeError(
                    f"{name}: pooled candidate game "
                    "coverage mismatch; "
                    f"missing={missing[:20]}, "
                    f"extra={extra[:20]}, "
                    f"order_mismatch={order_mismatch}"
                )
    return out
