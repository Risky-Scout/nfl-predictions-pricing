"""Fix 6: Week 1 prior + compact production feature deduction (auto-mode
revision).

Two deliverables only (see ``docs/FEATURE_DEDUCTION_2026.md`` and
``docs/WEEK1_PRIOR_2026.md``): a Week 1/early-season prior for team state
(:mod:`nfl_hybrid.features.week1_prior`), and a compact, interpretable,
2026-live-reproducible production feature set, measured with the FIXED
incumbent :class:`~nfl_hybrid.modern.joint_score.JointScoreModel`.
Model-family selection and hyperparameter tuning are explicitly out of
scope -- this module freezes the *input* to that later stage.

Because only ``game_result`` and ``elo_inputs`` are Fix-5 production-eligible
(:mod:`nfl_hybrid.providers.balldontlie.parity`), and because a fresh
live-parity/registry audit (this revision) additionally excludes
``neutral_site`` (not in Fix 5's live-mapped field set) and ``division_game``
(no static team/division registry exists anywhere in this repository), the
candidate feature universe is exactly 7 groups / 25 features / 1 core group
/ 6 non-core candidate groups -- see :data:`FEATURE_GROUPS`,
:data:`CORE_GROUPS`, :data:`CANDIDATE_GROUP_ORDER`. No play-by-play data is
read anywhere in this module.

**Auto-mode hard gates** (:class:`HardGateFailure`): every STOP condition
named in the auto-mode operating contract is wired as code here, not left
as a convention -- see each raising call site below and
``scripts/run_fix6_feature_deduction.py`` for the repo/branch/dirty-tree
gate (checked before this module is ever imported for real work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Sequence
import json

import numpy as np
import pandas as pd

from nfl_hybrid.data import external_data
from nfl_hybrid.evaluation import chronological_market_attachment as cma
from nfl_hybrid.features.elo_state import build_elo_pregame_state
from nfl_hybrid.features.feature_manifest import validate_no_banned_features
from nfl_hybrid.features.pregame_rolling import (
    PregameRollingConfig,
    build_game_pregame_matrix,
    build_team_pregame_features,
)
from nfl_hybrid.features.week1_prior import (
    GAME_RESULT_METRICS,
    Week1PriorConfig,
    apply_week1_blend,
    build_game_result_team_game,
    compute_week1_prior_config_hash,
)
from nfl_hybrid.evaluation.metrics import probability_metrics, regression_metrics
from nfl_hybrid.labels import edge_to_nullable_binary
from nfl_hybrid.modern.joint_score import JointScoreConfig, JointScoreModel
from nfl_hybrid.providers.balldontlie.parity import assert_production_eligible


class HardGateFailure(RuntimeError):
    """Raised when an auto-mode hard-STOP condition fires. Never caught and
    retried -- the caller (script) reports it and stops."""


# ---------------------------------------------------------------------------
# Section 1: development / holdout policy (fail-closed 2025/2026 firewall).
# ---------------------------------------------------------------------------
SELECTION_MAX_SEASON = 2024
SELECTION_FORBIDDEN_SEASONS = (2025, 2026)


class ForbiddenSeasonError(HardGateFailure):
    """Raised when Fix 6 selection/confirmation code is asked to touch a
    season outside the locked development/holdout policy."""


def assert_selection_season_allowed(seasons: Sequence[int] | pd.Series) -> None:
    values = {int(s) for s in pd.Series(list(seasons)).dropna().unique()}
    bad = {s for s in values if s > SELECTION_MAX_SEASON} | (values & set(SELECTION_FORBIDDEN_SEASONS))
    if bad:
        raise ForbiddenSeasonError(
            f"Season(s) {sorted(bad)} are forbidden for Fix 6 selection/confirmation "
            f"(selection_max_season={SELECTION_MAX_SEASON}, "
            f"forbidden_seasons={SELECTION_FORBIDDEN_SEASONS})."
        )


def enforce_2025_firewall(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """The ONE place raw ``games`` (which may physically contain 2025/2026
    rows) is ever filtered down to the selection-eligible estate. Returns
    ``(selection_games, firewall_counts)`` -- the counts are exact, not a
    blanket "0 rows seen" claim: the firewall necessarily reads the raw
    rows in order to remove them."""
    seasons = pd.to_numeric(games["season"], errors="raise")
    raw_rows_season_2025_seen = int((seasons == 2025).sum())
    selection_games = games[seasons <= SELECTION_MAX_SEASON].copy()
    if len(selection_games) == 0:
        raise HardGateFailure("enforce_2025_firewall: no rows remain after filtering.")
    assert int(pd.to_numeric(selection_games["season"], errors="raise").max()) == SELECTION_MAX_SEASON
    assert_selection_season_allowed(selection_games["season"])
    counts = {
        "raw_rows_season_2025_seen_by_firewall": raw_rows_season_2025_seen,
        "rows_season_ge_2025_passed_beyond_firewall": 0,
        "rows_season_ge_2025_used_for_feature_building": 0,
        "rows_season_ge_2025_used_for_model_fit": 0,
        "rows_season_ge_2025_used_for_metrics": 0,
    }
    return selection_games, counts


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_max_season: int
    validate_season: int


INNER_FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec("A", 2020, 2021),
    FoldSpec("B", 2021, 2022),
    FoldSpec("C", 2022, 2023),
)
# Technical fold name unchanged; its ROLE is LOCKED_POST_EXPOSURE_AUDIT, not
# an untouched/pristine holdout -- see run_outer_confirmation's docstring
# and OUTER_2024_ROLE below. A prior Fix 6 pass already evaluated this fold
# once and reported its result.
OUTER_FOLD = FoldSpec("OUTER", 2023, 2024)
OUTER_2024_PREVIOUSLY_EXPOSED = True
OUTER_2024_ROLE = "LOCKED_POST_EXPOSURE_AUDIT"


def assert_fold_seasons_allowed(fold: FoldSpec) -> None:
    assert_selection_season_allowed([fold.train_max_season, fold.validate_season])


# ---------------------------------------------------------------------------
# Section 6: hard market-line exclusion (target game's own current market
# state may never enter the fundamental predictive feature vector).
# ---------------------------------------------------------------------------
FORBIDDEN_MARKET_COLUMNS = frozenset(
    {
        "home_moneyline_reference",
        "away_moneyline_reference",
        "nflverse_spread_line_home_favored_positive",
        "home_spread_reference",
        "home_spread_price_reference",
        "away_spread_price_reference",
        "total_line_reference",
        "over_price_reference",
        "under_price_reference",
        "spreadspoke_home_spread_reference",
        "spreadspoke_total_line_reference",
        "spread_reference_difference",
        "total_reference_difference",
        "home_spread",
        "total_line",
        "home_spread_price",
        "away_spread_price",
        "over_price",
        "under_price",
        "no_vig_probability",
        "line_timestamp_utc",
        "line_reference_type",
    }
)


def assert_no_forbidden_market_columns(columns: Sequence[str]) -> None:
    bad = sorted(set(columns) & FORBIDDEN_MARKET_COLUMNS)
    if bad:
        raise HardGateFailure(f"Forbidden target-market column(s) requested as a feature: {bad}")


# ---------------------------------------------------------------------------
# Section 4/5/7: production-candidate feature groups -- pre-registered,
# fixed order, hash-enforced immutable after the first real model fit.
# ---------------------------------------------------------------------------
def _pivoted(bases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_{base}" for side in ("home", "away") for base in bases)


@dataclass(frozen=True)
class FeatureGroupDef:
    name: str
    columns: tuple[str, ...]
    kind: str  # "pivoted" (home_/away_ team state) or "carrier" (native games column)
    source_family: str
    interpretation: str


FEATURE_GROUPS: dict[str, FeatureGroupDef] = {
    "ELO_STRENGTH": FeatureGroupDef(
        name="ELO_STRENGTH",
        columns=_pivoted(("elo_pregame_rating", "elo_pregame_win_probability", "elo_pregame_expected_margin")),
        kind="pivoted",
        source_family="elo_inputs",
        interpretation="Pregame Elo rating / win probability / expected margin (Fix 2 elo_state; EXACT parity).",
    ),
    "SCORING_MARGIN": FeatureGroupDef(
        name="SCORING_MARGIN",
        columns=_pivoted(("margin__week1_blended", "margin__last8_mean")),
        kind="pivoted",
        source_family="game_result",
        interpretation="Team scoring-margin state: Week1-blended within-season average, 8-game recency average.",
    ),
    "SCORING_OFFENSE": FeatureGroupDef(
        name="SCORING_OFFENSE",
        columns=_pivoted(("points_scored__week1_blended", "points_scored__last8_mean")),
        kind="pivoted",
        source_family="game_result",
        interpretation="Team points-scored state (offense proxy): Week1-blended, 8-game recency average.",
    ),
    "SCORING_DEFENSE": FeatureGroupDef(
        name="SCORING_DEFENSE",
        columns=_pivoted(("points_allowed__week1_blended", "points_allowed__last8_mean")),
        kind="pivoted",
        source_family="game_result",
        interpretation="Team points-allowed state (defense proxy): Week1-blended, 8-game recency average.",
    ),
    "WIN_RATE": FeatureGroupDef(
        name="WIN_RATE",
        columns=_pivoted(("win__week1_blended", "win__last8_mean")),
        kind="pivoted",
        source_family="game_result",
        interpretation="Team win-rate state: Week1-blended within-season rate, 8-game recency rate.",
    ),
    "REST": FeatureGroupDef(
        name="REST",
        columns=_pivoted(("days_since_last_game",)),
        kind="pivoted",
        source_family="game_result",
        interpretation=(
            "Days of rest since each team's prior game, computed by the same shared "
            "shift-based transform (build_team_pregame_features) from each team's own "
            "chronological kickoff sequence -- DERIVED_FROM_EXACT_SOURCE. Replaces the "
            "native home_rest_days/away_rest_days historical-only columns, which have no "
            "proven identical live transform."
        ),
    ),
    "SEASON_PHASE": FeatureGroupDef(
        name="SEASON_PHASE",
        columns=("week",),
        kind="carrier",
        source_family="game_result",
        interpretation="Regular-season week number (native canonical schedule field, DERIVED_FROM_EXACT_SOURCE).",
    ),
}

# Fixed order: core first, then candidates in this literal order -- used for
# every tie-break in forward/backward selection. Never reordered after
# Phase 2 preregistration.
CANDIDATE_GROUP_ORDER: tuple[str, ...] = (
    "ELO_STRENGTH",
    "SCORING_MARGIN",
    "SCORING_OFFENSE",
    "SCORING_DEFENSE",
    "WIN_RATE",
    "REST",
    "SEASON_PHASE",
)
CORE_GROUPS: tuple[str, ...] = ("ELO_STRENGTH",)

# Exact counts asserted against the live registry in Phase 2 -- a mismatch
# is itself a hard STOP before Phase 4's first fit (Section 3 of the plan).
EXPECTED_TOTAL_GROUP_COUNT = 7
EXPECTED_CORE_GROUP_COUNT = 1
EXPECTED_NONCORE_CANDIDATE_GROUP_COUNT = 6
EXPECTED_CANDIDATE_FEATURE_COUNT = 25


def assert_registry_counts_match_expected() -> dict[str, int]:
    total_groups = len(FEATURE_GROUPS)
    core_groups = len(CORE_GROUPS)
    noncore_groups = len(CANDIDATE_GROUP_ORDER) - len(CORE_GROUPS)
    feature_count = sum(len(FEATURE_GROUPS[name].columns) for name in CANDIDATE_GROUP_ORDER)
    actual = {
        "total_group_count": total_groups,
        "core_group_count": core_groups,
        "noncore_candidate_group_count": noncore_groups,
        "candidate_feature_count": feature_count,
    }
    expected = {
        "total_group_count": EXPECTED_TOTAL_GROUP_COUNT,
        "core_group_count": EXPECTED_CORE_GROUP_COUNT,
        "noncore_candidate_group_count": EXPECTED_NONCORE_CANDIDATE_GROUP_COUNT,
        "candidate_feature_count": EXPECTED_CANDIDATE_FEATURE_COUNT,
    }
    if actual != expected:
        raise HardGateFailure(f"Registry counts {actual} do not match expected {expected}.")
    return actual


# Families/columns this revision deliberately excludes -- recorded here so
# the candidate inventory report shows WHY, not just that they are absent.
BLOCKED_LIVE_PARITY_FAMILIES: tuple[str, ...] = (
    "team_box_official",
    "team_box_pbp_derived (play volume / rates)",
    "qb_player_stats_boxscore",
    "epa_success_cpoe (team + QB)",
    "success_rate",
    "pass_rush_efficiency (yards/play, completion%, ypc -- non-EPA)",
    "turnovers",
    "play_volume (drives, pace, total plays)",
    "roster_depth",
    "injuries",
)
BLOCKED_NO_LIVE_REGISTRY_COLUMNS: tuple[dict[str, str], ...] = (
    {
        "column": "division_game",
        "reason": (
            "No static team->division registry exists anywhere in this repository "
            "(src/nfl_hybrid/data/team_ids.py has only a name-alias table, no "
            "conference/division data; grep for division_map/TEAM_DIVISION/AFC_EAST "
            "etc. across src/ returns zero hits). The native `division_game` column is "
            "historical-only (src/nfl_hybrid/data/providers/nflverse.py reads it "
            "straight from the nflverse feed) with no proven identical live "
            "reconstruction, so it cannot be classified DERIVED_FROM_EXACT_SOURCE."
        ),
    },
    {
        "column": "neutral_site",
        "reason": (
            "Not in Fix 5's live-mapped field set: "
            "nfl_hybrid.providers.balldontlie.canonical.GAMES_COLUMNS has no "
            "neutral_site entry, and parity.py's elo_inputs live_definition only "
            "lists game_id/season/week/kickoff_utc/home_team/away_team/home_score/"
            "away_score. Used only internally (defaulted to False when absent) by "
            "nfl_hybrid.features.elo_state for Elo home-field adjustment -- not "
            "independently live-provable as a standalone Fix 6 feature."
        ),
    },
)
BLOCKED_HIGH_VALUE_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "blocked_family": "market_history (prior-game ATS/OU outcome rolling state)",
        "historical_incremental_signal": (
            "Not measured in Fix 6 (excluded before evaluation per Section 0: eligibility must be "
            "proven before use, never assumed)."
        ),
        "reason_blocked": (
            "Fix 5's BDL parity work (docs/BDL_PARITY_MATRIX.md) validates box-score/PBP families "
            "only; it does not cover the T10/closing-line market-data provider. No 2026 live-parity "
            "contract has been proven for reproducing historical home_spread_reference/"
            "total_line_reference as-of a past date, so this family cannot be admitted here even "
            "though it is already lagged and leakage-safe by construction "
            "(nfl_hybrid.features.market_history)."
        ),
        "parity_work_required": (
            "Run a BDL/T10-style overlap backtest for the market-line provider's live 2026 feed "
            "against the historical home_spread_reference/total_line_reference estate, and add it "
            "to a parity matrix the way docs/BDL_PARITY_MATRIX.md does for box/PBP data."
        ),
    },
    {
        "blocked_family": "epa/opponent_adjusted/qb pregame-state families "
        "(nfl_hybrid.features.pregame_state.STATE_FAMILIES)",
        "historical_incremental_signal": (
            "Not measured in Fix 6 (excluded before evaluation) -- these are the state families the "
            "existing Fix 2/3 pregame-state pipeline already computes, but all of them ultimately "
            "read PBP/QB-box columns Fix 5 marked APPROXIMATE_NOT_APPROVED or UNAVAILABLE."
        ),
        "reason_blocked": (
            "Every source column these families read (EPA, success, CPOE, PBP-derived play volume, "
            "QB box counting stats) is Fix-5 blocked -- see BLOCKED_LIVE_PARITY_FAMILIES above."
        ),
        "parity_work_required": (
            "Resolve the specific gaps documented in docs/BDL_PARITY_MATRIX.md per family before "
            "any of these families can re-enter a future feature-selection round."
        ),
    },
)


def group_source_family_check(group_names: Sequence[str]) -> None:
    """Fail-closed Fix-5 parity gate for every requested group."""
    for name in group_names:
        if name not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group: {name!r}")
        assert_production_eligible(FEATURE_GROUPS[name].source_family)


def compute_candidate_group_registry_hash(
    groups: dict[str, FeatureGroupDef] | None = None,
    order: tuple[str, ...] | None = None,
) -> str:
    groups = groups if groups is not None else FEATURE_GROUPS
    order = order if order is not None else CANDIDATE_GROUP_ORDER
    payload = json.dumps(
        {
            "order": list(order),
            "core_groups": list(CORE_GROUPS),
            "groups": {
                name: {
                    "columns": list(groups[name].columns),
                    "kind": groups[name].kind,
                    "source_family": groups[name].source_family,
                }
                for name in order
            },
        },
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


# Hash-enforced registry freeze: set once by freeze_candidate_registry() at
# the end of Phase 2, before Phase 4's first real fit. _fit_predict_fold
# recomputes the live registry hash on EVERY call and requires exact
# equality -- this, not the boolean flag, is the authoritative check.
_FROZEN_REGISTRY_HASH: str | None = None
_REGISTRY_FROZEN: bool = False


def freeze_candidate_registry() -> str:
    global _FROZEN_REGISTRY_HASH, _REGISTRY_FROZEN
    assert_registry_counts_match_expected()
    _FROZEN_REGISTRY_HASH = compute_candidate_group_registry_hash()
    _REGISTRY_FROZEN = True
    return _FROZEN_REGISTRY_HASH


def _assert_registry_unchanged() -> None:
    if _FROZEN_REGISTRY_HASH is None:
        raise HardGateFailure("Candidate registry was never frozen (freeze_candidate_registry() not called).")
    live_hash = compute_candidate_group_registry_hash()
    if live_hash != _FROZEN_REGISTRY_HASH:
        raise HardGateFailure(
            f"Candidate group registry changed after freeze: frozen={_FROZEN_REGISTRY_HASH}, live={live_hash}."
        )


# ---------------------------------------------------------------------------
# Team-state construction (Elo, unmodified + Week1-blended game-result state
# + the shared days_since_last_game rest transform).
# ---------------------------------------------------------------------------
_TEAM_STATE_CACHE: dict[tuple[int, str, int], pd.DataFrame] = {}
_MATRIX_CACHE: dict[tuple[int, tuple[str, ...], str], tuple[pd.DataFrame, list[str]]] = {}


def build_production_team_state(
    games: pd.DataFrame,
    *,
    week1_config: Week1PriorConfig,
    rolling_config: PregameRollingConfig | None = None,
) -> pd.DataFrame:
    """One row per (game_id, team_id): every production-eligible team-state
    column this module can build -- Elo (untouched) + Week1-blended
    game-result state + shared days_since_last_game. No PBP is read."""
    assert_production_eligible("elo_inputs")
    assert_production_eligible("game_result")

    cache_key = (id(games), compute_week1_prior_config_hash(week1_config), len(games))
    cached = _TEAM_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rcfg = rolling_config or PregameRollingConfig(windows=(8,))

    elo_state = build_elo_pregame_state(games)
    elo_cols = ["game_id", "team_id", "elo_pregame_rating", "elo_pregame_win_probability", "elo_pregame_expected_margin"]

    team_game = build_game_result_team_game(games)
    team_pregame_raw = build_team_pregame_features(
        team_game, games, config=rcfg, metric_columns=list(GAME_RESULT_METRICS)
    )
    team_pregame = apply_week1_blend(team_pregame_raw, team_game, config=week1_config)

    metric_cols = [f"{m}__week1_blended" for m in GAME_RESULT_METRICS] + [
        f"{m}__last8_mean" for m in GAME_RESULT_METRICS
    ]
    keep_cols = ["game_id", "team_id"] + metric_cols + ["days_since_last_game"]

    combined = elo_state[elo_cols].merge(
        team_pregame[keep_cols], on=["game_id", "team_id"], how="inner", validate="one_to_one"
    )
    _TEAM_STATE_CACHE[cache_key] = combined
    return combined


ALWAYS_CARRIER_COLUMNS: tuple[str, ...] = ("season", "week", "home_score", "away_score")


def build_candidate_matrix(
    games: pd.DataFrame,
    group_names: Sequence[str],
    *,
    week1_config: Week1PriorConfig,
    rolling_config: PregameRollingConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Regular-season, development-only (season <= 2024) game-level feature
    matrix for exactly the requested groups, plus training targets
    (``home_margin``, ``total_points``) and fold-filtering metadata
    (``season``, ``week``). ``feature_columns`` never includes
    ``home_score``/``away_score`` or any market-line column."""
    group_source_family_check(group_names)

    seasons = pd.to_numeric(games["season"], errors="raise")
    assert_selection_season_allowed(seasons)

    cache_key = (id(games), tuple(sorted(group_names)), compute_week1_prior_config_hash(week1_config))
    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached[0].copy(), list(cached[1])

    reg_games = games[games["season_type"] == "REG"].reset_index(drop=True)
    if reg_games.empty:
        raise ValueError("No REG season games in the provided games table.")

    team_state = build_production_team_state(reg_games, week1_config=week1_config, rolling_config=rolling_config)

    group_carrier = [c for name in group_names for c in FEATURE_GROUPS[name].columns if FEATURE_GROUPS[name].kind == "carrier"]
    carrier = tuple(dict.fromkeys(list(ALWAYS_CARRIER_COLUMNS) + group_carrier))
    assert_no_forbidden_market_columns(carrier)

    pivoted = build_game_pregame_matrix(reg_games, team_state, carrier_columns=carrier)

    feature_columns: list[str] = []
    for name in group_names:
        feature_columns.extend(FEATURE_GROUPS[name].columns)
    feature_columns = list(dict.fromkeys(feature_columns))

    missing = [c for c in feature_columns if c not in pivoted.columns]
    if missing:
        raise ValueError(f"Pivoted matrix missing expected feature column(s): {missing}")

    assert_no_forbidden_market_columns(feature_columns)
    validate_no_banned_features(feature_columns)

    pivoted["home_margin"] = pd.to_numeric(pivoted["home_score"], errors="coerce") - pd.to_numeric(
        pivoted["away_score"], errors="coerce"
    )
    pivoted["total_points"] = pd.to_numeric(pivoted["home_score"], errors="coerce") + pd.to_numeric(
        pivoted["away_score"], errors="coerce"
    )
    pivoted["season"] = pd.to_numeric(pivoted["season"], errors="raise").astype(int)
    for col in feature_columns:
        pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce").astype(float)

    _MATRIX_CACHE[cache_key] = (pivoted.copy(), list(feature_columns))
    return pivoted, feature_columns


# ---------------------------------------------------------------------------
# Section 3: fixed measuring instrument.
# ---------------------------------------------------------------------------
MODEL_CONFIG = JointScoreConfig()


def compute_model_config_hash(config: JointScoreConfig = MODEL_CONFIG) -> str:
    from dataclasses import asdict

    payload = json.dumps({"model_config": asdict(config)}, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _fit_predict_fold(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    fold: FoldSpec,
    *,
    model_config: JointScoreConfig = MODEL_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, JointScoreModel]:
    """SEASON_FORWARD_FEATURE_SELECTION_PROXY -- NOT the production
    Tuesday/Friday chronological OOF machinery
    (nfl_hybrid.evaluation.chronological_oof); the final frozen spec will be
    regenerated through that engine in a later stage.

    Fits ONE model on ``season <= fold.train_max_season`` and predicts ONCE
    on ``season == fold.validate_season`` -- a single fixed fit per fold.
    Validation-season feature values may still legitimately reflect that
    season's own already-completed prior games (the shift-based rolling
    engine already guarantees this correctly), but the fitted estimator
    itself is never updated within the validation season, and no
    target/future game can affect its own feature vector.

    Hash-enforced registry-immutability gate: every call re-verifies the
    live candidate-group registry still matches what was frozen in Phase 2.
    """
    _assert_registry_unchanged()
    assert_fold_seasons_allowed(fold)
    train = matrix[matrix["season"] <= fold.train_max_season]
    validate = matrix[matrix["season"] == fold.validate_season]
    if train.empty or validate.empty:
        raise ValueError(f"Fold {fold.name}: empty train ({len(train)}) or validate ({len(validate)}) set.")

    model = JointScoreModel(numeric_features=list(feature_columns), categorical_features=(), config=model_config)
    model.fit(train)
    pred_margin, pred_total = model.predict_means(validate)

    predictions = validate[["game_id", "season", "week", "home_margin", "total_points"]].copy()
    predictions["pred_margin"] = pred_margin
    predictions["pred_total"] = pred_total
    predictions["fold"] = fold.name
    return predictions, validate, model


def _combined_regression_metric(df: pd.DataFrame) -> dict:
    margin = regression_metrics(df["home_margin"].to_numpy(), df["pred_margin"].to_numpy())
    total = regression_metrics(df["total_points"].to_numpy(), df["pred_total"].to_numpy())
    return {"margin": margin, "total": total, "primary_score": (margin["rmse"] + total["rmse"]) / 2.0}


def _per_fold_primary_scores(preds: pd.DataFrame, folds: Sequence[FoldSpec]) -> dict[str, float]:
    return {
        f.name: _combined_regression_metric(preds[preds["fold"] == f.name])["primary_score"]
        for f in folds
        if (preds["fold"] == f.name).any()
    }


def _evaluate_group_set_across_folds(
    games: pd.DataFrame,
    group_names: Sequence[str],
    week1_config: Week1PriorConfig,
    folds: Sequence[FoldSpec],
) -> tuple[float, pd.DataFrame, list[str]]:
    matrix, feature_columns = build_candidate_matrix(games, group_names, week1_config=week1_config)
    preds = []
    for fold in folds:
        prediction, _validate, _model = _fit_predict_fold(matrix, feature_columns, fold)
        preds.append(prediction)
    pooled = pd.concat(preds, ignore_index=True)
    primary_score = _combined_regression_metric(pooled)["primary_score"]
    return primary_score, pooled, feature_columns


# ---------------------------------------------------------------------------
# Section 7: fixed primary-metric formula + analytic paired SE (no
# bootstrap, no RNG, no seed).
# ---------------------------------------------------------------------------
def _paired_delta_se(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> tuple[float, float]:
    """Paired per-game loss comparison over the SAME validate rows (both
    specs share the identical game universe for a given fold set).

    ``L_i = 0.5 * (margin_error_i^2 + total_error_i^2)``
    ``delta_i = L_b_i - L_a_i``  (negative => b is better than a)
    ``SE(delta) = sample_std(delta, ddof=1) / sqrt(n)``

    Returns ``(mean_delta, se_delta)``. Purely analytic -- no seed, no
    resampling, deterministic by construction.
    """
    merged = preds_a.merge(
        preds_b[["game_id", "fold", "pred_margin", "pred_total"]],
        on=["game_id", "fold"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    n = len(merged)
    if n == 0:
        raise ValueError("No paired predictions to compare.")

    margin_actual = merged["home_margin"].to_numpy(dtype=float)
    total_actual = merged["total_points"].to_numpy(dtype=float)
    l_a = 0.5 * (
        (margin_actual - merged["pred_margin_a"].to_numpy(dtype=float)) ** 2
        + (total_actual - merged["pred_total_a"].to_numpy(dtype=float)) ** 2
    )
    l_b = 0.5 * (
        (margin_actual - merged["pred_margin_b"].to_numpy(dtype=float)) ** 2
        + (total_actual - merged["pred_total_b"].to_numpy(dtype=float)) ** 2
    )
    delta = l_b - l_a
    mean_delta = float(delta.mean())
    se_delta = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_delta, se_delta


# ---------------------------------------------------------------------------
# Section 3: Week1 blend-constant k grid search -- paired, exact tie-break.
# ---------------------------------------------------------------------------
K_GRID: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
# K_SEARCH_SPEC != CORE_GROUPS: k only affects *_week1_blended columns,
# which don't exist in ELO_STRENGTH alone -- a literal "search on core"
# would make k unidentifiable (every k scores identically). This is the
# minimum spec that actually exercises k, documented as a deliberate,
# evidence-based deviation (see the plan's "One interpretation flagged for
# operator review").
K_SEARCH_SPEC: tuple[str, ...] = ("ELO_STRENGTH", "SCORING_MARGIN", "SCORING_OFFENSE", "SCORING_DEFENSE", "WIN_RATE")


def run_k_grid_search(
    games: pd.DataFrame,
    *,
    neutral_fallback: dict[str, float] | None = None,
    folds: Sequence[FoldSpec] = INNER_FOLDS,
    group_names: Sequence[str] = K_SEARCH_SPEC,
) -> tuple[float, list[dict], int]:
    assert_selection_season_allowed([f.train_max_season for f in folds] + [f.validate_season for f in folds])
    scores: dict[float, float] = {}
    preds_by_k: dict[float, pd.DataFrame] = {}
    fit_count = 0
    for k in K_GRID:
        config = Week1PriorConfig(k=k) if neutral_fallback is None else Week1PriorConfig(k=k, neutral_fallback=dict(neutral_fallback))
        score, pooled, _features = _evaluate_group_set_across_folds(games, group_names, config, folds)
        fit_count += len(folds)
        scores[k] = score
        preds_by_k[k] = pooled

    best_k = min(scores, key=lambda k: scores[k])
    trace: list[dict] = []
    within_one_se = [best_k]
    for k in K_GRID:
        if k == best_k:
            trace.append({"k": k, "primary_score": scores[k], "mean_delta_vs_best": 0.0, "se_delta_vs_best": 0.0, "is_best": True})
            continue
        mean_delta, se_delta = _paired_delta_se(preds_by_k[best_k], preds_by_k[k])
        is_within = mean_delta <= se_delta
        if is_within:
            within_one_se.append(k)
        trace.append(
            {
                "k": k,
                "primary_score": scores[k],
                "mean_delta_vs_best": mean_delta,
                "se_delta_vs_best": se_delta,
                "is_best": False,
                "within_one_se_of_best": is_within,
            }
        )

    selected_k = max(within_one_se)
    trace.append({"selected_k": selected_k, "within_one_se_set": sorted(within_one_se)})
    return selected_k, trace, fit_count


# ---------------------------------------------------------------------------
# Section 7: deterministic forward-then-backward group ablation.
# ---------------------------------------------------------------------------
def run_group_ablation(
    games: pd.DataFrame,
    week1_config: Week1PriorConfig,
    *,
    folds: Sequence[FoldSpec] = INNER_FOLDS,
    core_groups: Sequence[str] = CORE_GROUPS,
    candidate_order: Sequence[str] = CANDIDATE_GROUP_ORDER,
) -> tuple[list[str], list[dict], int]:
    assert_selection_season_allowed([f.train_max_season for f in folds] + [f.validate_season for f in folds])

    candidate_pool = [g for g in candidate_order if g not in core_groups]
    retained: list[str] = list(core_groups)
    trace: list[dict] = []
    fit_count = 0
    verdict_by_group: dict[str, str] = {g: "REJECTED_NO_OOS_GAIN" for g in candidate_pool}

    current_score, current_preds, _ = _evaluate_group_set_across_folds(games, retained, week1_config, folds)
    current_fold_scores = _per_fold_primary_scores(current_preds, folds)
    fit_count += len(folds)
    trace.append({"step": "baseline", "groups": list(retained), "primary_score": current_score})

    # --- Forward pass: each round, evaluate EVERY remaining candidate group
    # against the current spec; among those that pass, add the single best.
    remaining = list(candidate_pool)
    while remaining:
        round_results = []
        for name in remaining:
            candidate_groups = retained + [name]
            cand_score, cand_preds, _ = _evaluate_group_set_across_folds(games, candidate_groups, week1_config, folds)
            fit_count += len(folds)
            cand_fold_scores = _per_fold_primary_scores(cand_preds, folds)
            wins = sum(1 for f in folds if cand_fold_scores.get(f.name, float("inf")) < current_fold_scores.get(f.name, float("inf")))
            mean_delta, se_delta = _paired_delta_se(current_preds, cand_preds)  # delta = L_candidate - L_current
            passes = (wins >= 2) and (mean_delta + se_delta < 0)
            round_results.append(
                {
                    "group": name,
                    "primary_score": cand_score,
                    "preds": cand_preds,
                    "fold_scores": cand_fold_scores,
                    "wins_of_folds": wins,
                    "n_folds": len(folds),
                    "mean_delta": mean_delta,
                    "se_delta": se_delta,
                    "passes": passes,
                }
            )
            trace.append(
                {
                    "step": "forward_round",
                    "group": name,
                    "baseline_primary_score": current_score,
                    "candidate_primary_score": cand_score,
                    "wins_of_folds": wins,
                    "n_folds": len(folds),
                    "mean_delta": mean_delta,
                    "se_delta": se_delta,
                    "verdict": "CANDIDATE_PASSES_ROUND" if passes else "REJECTED_NO_OOS_GAIN",
                }
            )

        passing = [r for r in round_results if r["passes"]]
        if not passing:
            break

        def _sort_key(r: dict) -> tuple[float, int, int]:
            return (r["primary_score"], len(FEATURE_GROUPS[r["group"]].columns), candidate_order.index(r["group"]))

        chosen = min(passing, key=_sort_key)
        retained.append(chosen["group"])
        verdict_by_group[chosen["group"]] = "RETAINED"
        current_score, current_preds, current_fold_scores = chosen["primary_score"], chosen["preds"], chosen["fold_scores"]
        remaining = [g for g in remaining if g != chosen["group"]]
        trace.append({"step": "forward_selected", "group": chosen["group"], "primary_score": current_score})

    # --- Backward ablation: paired, signed, no abs(), continuous updates,
    # fixed registry order, capped at 3 sweeps with a hard STOP if the 3rd
    # sweep still removes a group (never silently reported as converged).
    converged = False
    removed_this_sweep: list[str] = []
    for sweep in range(1, 4):
        removed_this_sweep = []
        for name in [g for g in candidate_order if g in retained and g not in core_groups]:
            reduced_groups = [g for g in retained if g != name]
            red_score, red_preds, _ = _evaluate_group_set_across_folds(games, reduced_groups, week1_config, folds)
            fit_count += len(folds)
            mean_delta, se_delta = _paired_delta_se(current_preds, red_preds)  # delta = L_reduced - L_current
            remove = mean_delta <= se_delta
            trace.append(
                {
                    "step": "backward",
                    "sweep": sweep,
                    "group": name,
                    "with_group_primary_score": current_score,
                    "without_group_primary_score": red_score,
                    "mean_delta": mean_delta,
                    "se_delta": se_delta,
                    "verdict": "REJECTED_REDUNDANT" if remove else "RETAINED",
                }
            )
            if remove:
                retained = reduced_groups
                current_score, current_preds = red_score, red_preds
                verdict_by_group[name] = "REJECTED_REDUNDANT"
                removed_this_sweep.append(name)
        if not removed_this_sweep:
            converged = True
            break

    if not converged:
        raise HardGateFailure(
            f"BACKWARD_ABLATION_DID_NOT_CONVERGE: sweep 3 still removed {removed_this_sweep}."
        )

    trace.append({"step": "final_verdicts", "verdict_by_group": dict(verdict_by_group), "retained": list(retained)})
    return retained, trace, fit_count


# ---------------------------------------------------------------------------
# Section 10: LOCKED_POST_EXPOSURE_2024_AUDIT acceptance -- exact
# predeclared numbers, frozen before this function is ever called for real.
# ---------------------------------------------------------------------------
OUTER_PRIMARY_TOLERANCE = 0.02
OUTER_MARGIN_TOLERANCE = 0.05
OUTER_TOTAL_TOLERANCE = 0.05
OUTER_WEEK1_TOLERANCE = 0.10
OUTER_WEEKS2_4_TOLERANCE = 0.10
OUTER_WEEKS5PLUS_TOLERANCE = 0.05


def _segment_labels(week: pd.Series) -> np.ndarray:
    w = pd.to_numeric(week, errors="coerce")
    return np.select([w == 1, (w >= 2) & (w <= 4), w >= 5], ["week1", "weeks2_4", "weeks5_plus"], default="unknown")


MINIMUM_ELIGIBLE_BOOKS = cma.MINIMUM_ELIGIBLE_BOOKS
MAXIMUM_SNAPSHOT_LAG_MINUTES = cma.MAXIMUM_SNAPSHOT_LAG_MINUTES

_CONFIRMATION_MARKET_KEYS = {
    "ATS": "canonical_market.pregame_ats_t10_2020_2024_confirmation",
    "TOTAL": "canonical_market.pregame_total_t10_2020_2024_confirmation",
}


def _load_canonical_t10_2020_2024_confirmation(market: str) -> pd.DataFrame:
    """Fix-6-local read of the NEW 2020-2024-confirmation canonical T10
    export (registered in external_data.py, additive, Fix 4 untouched) --
    same normalization contract and quality gate as
    chronological_market_attachment.load_canonical_t10_market, applied
    independently here since that function is hardwired to the 2020-2023
    registry keys (a Fix-4-owned constant this module does not edit)."""
    path = external_data.resolve(_CONFIRMATION_MARKET_KEYS[market])
    frame = pd.read_parquet(path)
    required = {
        "game_id", "season", "week",
        "market_t10_consensus_line", "market_t10_eligible_books", "market_t10_snapshot_lag_minutes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise HardGateFailure(f"Canonical T10 2020-2024-confirmation {market} matrix missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str),
            "season": frame["season"],
            "week": frame["week"],
            "line": pd.to_numeric(frame["market_t10_consensus_line"], errors="coerce"),
            "eligible_books": pd.to_numeric(frame["market_t10_eligible_books"], errors="coerce"),
            "snapshot_lag_minutes": pd.to_numeric(frame["market_t10_snapshot_lag_minutes"], errors="coerce"),
        }
    )
    gated = out[
        (out["eligible_books"] >= MINIMUM_ELIGIBLE_BOOKS) & (out["snapshot_lag_minutes"] <= MAXIMUM_SNAPSHOT_LAG_MINUTES)
    ]
    return gated


def _secondary_market_diagnostics(model: JointScoreModel, validate: pd.DataFrame, season: int) -> dict:
    """Raw ATS/TOTAL conditional log loss/Brier -- a SECONDARY diagnostic
    only, computed strictly after prediction, using CANONICAL T10 market
    lines (never never home_spread_reference/total_line_reference). Never
    used to select features or fit the model. For season <= 2023 (inner
    folds) reads Fix 4's own existing registered 2020-2023 canonical T10
    matrices directly (read-only reuse, no Fix-4 file edited); for season
    2024 reads the new Fix-6-registered 2020-2024-confirmation export. If
    coverage is unavailable -- including NFL_MODEL_DATA_ROOT not being
    configured at all, e.g. a CI-safe synthetic-fixture run -- returns
    status="UNAVAILABLE_NO_CANONICAL_T10_COVERAGE", never a silent fallback
    to the generic reference columns and never a hard crash (this
    diagnostic is strictly secondary/non-gating)."""
    try:
        if season <= 2023:
            ats = cma.load_canonical_t10_market("ATS")
            total = cma.load_canonical_t10_market("TOTAL")
            ats = ats[(ats["eligible_books"] >= MINIMUM_ELIGIBLE_BOOKS) & (ats["snapshot_lag_minutes"] <= MAXIMUM_SNAPSHOT_LAG_MINUTES)]
            total = total[(total["eligible_books"] >= MINIMUM_ELIGIBLE_BOOKS) & (total["snapshot_lag_minutes"] <= MAXIMUM_SNAPSHOT_LAG_MINUTES)]
        else:
            ats = _load_canonical_t10_2020_2024_confirmation("ATS")
            total = _load_canonical_t10_2020_2024_confirmation("TOTAL")
    except external_data.ExternalDataUnavailableError:
        return {"status": "UNAVAILABLE_NO_CANONICAL_T10_COVERAGE"}

    if ats.empty or total.empty:
        return {"status": "UNAVAILABLE_NO_CANONICAL_T10_COVERAGE"}

    ats_frame = validate.merge(ats[["game_id", "line"]].rename(columns={"line": "home_spread"}), on="game_id", how="inner")
    total_frame = validate.merge(total[["game_id", "line"]].rename(columns={"line": "total_line"}), on="game_id", how="inner")
    if ats_frame.empty or total_frame.empty:
        return {"status": "UNAVAILABLE_NO_CANONICAL_T10_COVERAGE"}

    ats_frame = ats_frame.assign(total_line=np.nan)
    total_frame = total_frame.assign(home_spread=np.nan)

    ats_raw = model.raw_probabilities(ats_frame)
    total_raw = model.raw_probabilities(total_frame)

    home_cover = pd.to_numeric(
        edge_to_nullable_binary(ats_frame["home_margin"] + ats_frame["home_spread"]), errors="coerce"
    )
    over = pd.to_numeric(
        edge_to_nullable_binary(total_frame["total_points"] - total_frame["total_line"]), errors="coerce"
    )
    return {
        "status": "OK",
        "ats": probability_metrics(home_cover.to_numpy(dtype=float), ats_raw["raw_home_cover_probability_no_push"].to_numpy()),
        "total": probability_metrics(over.to_numpy(dtype=float), total_raw["raw_over_probability_no_push"].to_numpy()),
        "ats_coverage_n": len(ats_frame),
        "total_coverage_n": len(total_frame),
    }


def run_outer_confirmation(
    games: pd.DataFrame,
    *,
    compact_groups: Sequence[str],
    week1_config: Week1PriorConfig,
    core_groups: Sequence[str] = CORE_GROUPS,
    fold: FoldSpec = OUTER_FOLD,
) -> dict:
    """LOCKED_POST_EXPOSURE_2024_AUDIT -- ONE run of the frozen compact spec
    vs the core baseline, train <= 2023, validate 2024. NOT an independent
    untouched holdout: a prior Fix 6 pass already evaluated this fold once
    and reported its result (``outer_2024_previously_exposed=True``). Still
    useful as a locked stress/audit period because every rule below is
    frozen before THIS run touches 2024 data, and the spec is never revised
    afterward regardless of outcome."""
    assert_fold_seasons_allowed(fold)

    compact_matrix, compact_features = build_candidate_matrix(games, compact_groups, week1_config=week1_config)
    core_matrix, core_features = build_candidate_matrix(games, core_groups, week1_config=week1_config)

    compact_pred, compact_validate, compact_model = _fit_predict_fold(compact_matrix, compact_features, fold)
    core_pred, core_validate, core_model = _fit_predict_fold(core_matrix, core_features, fold)

    compact_metrics = _combined_regression_metric(compact_pred)
    core_metrics = _combined_regression_metric(core_pred)

    compact_pred = compact_pred.assign(segment=_segment_labels(compact_pred["week"]))
    core_pred = core_pred.assign(segment=_segment_labels(core_pred["week"]))

    segment_tolerance = {
        "week1": OUTER_WEEK1_TOLERANCE,
        "weeks2_4": OUTER_WEEKS2_4_TOLERANCE,
        "weeks5_plus": OUTER_WEEKS5PLUS_TOLERANCE,
    }
    segment_report: dict[str, dict] = {}
    passed_segments = True
    for segment, tolerance in segment_tolerance.items():
        c = compact_pred[compact_pred["segment"] == segment]
        b = core_pred[core_pred["segment"] == segment]
        if c.empty or b.empty:
            segment_report[segment] = {"status": "NO_DATA"}
            continue
        c_metric = _combined_regression_metric(c)["primary_score"]
        b_metric = _combined_regression_metric(b)["primary_score"]
        relative_change = (c_metric - b_metric) / b_metric if b_metric else float("nan")
        ok = bool(relative_change <= tolerance)
        passed_segments = passed_segments and ok
        segment_report[segment] = {
            "compact_primary_score": c_metric,
            "baseline_primary_score": b_metric,
            "relative_change": relative_change,
            "tolerance": tolerance,
            "pass": ok,
        }

    relative_primary_change = (compact_metrics["primary_score"] - core_metrics["primary_score"]) / core_metrics["primary_score"]
    relative_margin_change = (compact_metrics["margin"]["rmse"] - core_metrics["margin"]["rmse"]) / core_metrics["margin"]["rmse"]
    relative_total_change = (compact_metrics["total"]["rmse"] - core_metrics["total"]["rmse"]) / core_metrics["total"]["rmse"]

    primary_pass = bool(relative_primary_change <= OUTER_PRIMARY_TOLERANCE)
    margin_pass = bool(relative_margin_change <= OUTER_MARGIN_TOLERANCE)
    total_pass = bool(relative_total_change <= OUTER_TOTAL_TOLERANCE)
    overall_pass = bool(primary_pass and margin_pass and total_pass and passed_segments)

    secondary = {
        "compact": _secondary_market_diagnostics(compact_model, compact_validate, fold.validate_season),
        "core_baseline": _secondary_market_diagnostics(core_model, core_validate, fold.validate_season),
    }

    return {
        "fold": fold.name,
        "outer_2024_previously_exposed": OUTER_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_role": OUTER_2024_ROLE,
        "outer_2024_used_for_selection": False,
        "outer_2024_used_for_revised_spec_adjustment": False,
        "compact_groups": list(compact_groups),
        "core_groups": list(core_groups),
        "compact_metrics": compact_metrics,
        "core_metrics": core_metrics,
        "relative_primary_change": relative_primary_change,
        "relative_margin_change": relative_margin_change,
        "relative_total_change": relative_total_change,
        "primary_pass": primary_pass,
        "margin_pass": margin_pass,
        "total_pass": total_pass,
        "segment_report": segment_report,
        "overall_pass": overall_pass,
        "secondary_market_diagnostics": secondary,
    }


# ---------------------------------------------------------------------------
# Manifest hashing.
# ---------------------------------------------------------------------------
def compute_feature_manifest_hash(feature_columns: Sequence[str]) -> str:
    payload = json.dumps({"feature_columns": sorted(feature_columns)}, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def compute_selection_config_hash(
    *,
    inner_folds: Sequence[FoldSpec] = INNER_FOLDS,
    outer_fold: FoldSpec = OUTER_FOLD,
    k_grid: Sequence[float] = K_GRID,
    core_groups: Sequence[str] = CORE_GROUPS,
    candidate_order: Sequence[str] = CANDIDATE_GROUP_ORDER,
) -> str:
    payload = json.dumps(
        {
            "inner_folds": [(f.name, f.train_max_season, f.validate_season) for f in inner_folds],
            "outer_fold": (outer_fold.name, outer_fold.train_max_season, outer_fold.validate_season),
            "outer_2024_role": OUTER_2024_ROLE,
            "k_grid": list(k_grid),
            "outer_primary_tolerance": OUTER_PRIMARY_TOLERANCE,
            "outer_margin_tolerance": OUTER_MARGIN_TOLERANCE,
            "outer_total_tolerance": OUTER_TOTAL_TOLERANCE,
            "outer_week1_tolerance": OUTER_WEEK1_TOLERANCE,
            "outer_weeks2_4_tolerance": OUTER_WEEKS2_4_TOLERANCE,
            "outer_weeks5plus_tolerance": OUTER_WEEKS5PLUS_TOLERANCE,
            "backward_ablation_sweep_cap": 3,
            "core_groups": list(core_groups),
            "candidate_group_order": list(candidate_order),
            "selection_max_season": SELECTION_MAX_SEASON,
            "selection_forbidden_seasons": list(SELECTION_FORBIDDEN_SEASONS),
        },
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Section 14: fit-budget planning, checked BEFORE running anything expensive.
# Recomputed for the revised stepwise forward pass + capped backward sweeps.
# ---------------------------------------------------------------------------
def plan_fit_budget(
    *,
    n_inner_folds: int = len(INNER_FOLDS),
    n_k_grid: int = len(K_GRID),
    n_candidate_groups: int = len(CANDIDATE_GROUP_ORDER) - len(CORE_GROUPS),
) -> dict:
    k_grid_fits = n_k_grid * n_inner_folds
    ablation_baseline_fits = n_inner_folds
    # Stepwise forward: round sizes shrink each time a group is added (worst
    # case every group eventually added): n + (n-1) + ... + 1 evaluations.
    forward_evals = sum(range(1, n_candidate_groups + 1))
    ablation_forward_fits = forward_evals * n_inner_folds
    # Backward: up to 3 sweeps, each sweep re-checks up to n_candidate_groups.
    ablation_backward_fits = 3 * n_candidate_groups * n_inner_folds
    outer_fits = 2  # compact spec + core baseline
    total = k_grid_fits + ablation_baseline_fits + ablation_forward_fits + ablation_backward_fits + outer_fits
    return {
        "candidate_group_count": len(CANDIDATE_GROUP_ORDER),
        "core_group_count": len(CORE_GROUPS),
        "noncore_candidate_group_count": n_candidate_groups,
        "eligible_feature_count": sum(len(FEATURE_GROUPS[g].columns) for g in CANDIDATE_GROUP_ORDER),
        "k_grid_fits": k_grid_fits,
        "ablation_baseline_fits": ablation_baseline_fits,
        "ablation_forward_fits_worst_case": ablation_forward_fits,
        "ablation_backward_fits_worst_case": ablation_backward_fits,
        "outer_confirmation_fits": outer_fits,
        "planned_total_fits_worst_case": total,
        "fit_budget_ok": total <= 200,
    }
