"""Week 1 / early-season prior for game-result-derived team state (Fix 6).

This module answers one question: "what state exists before the first
current-season result?" for the team-state metrics this repository can build
*solely* from the ``games`` table's own final-score/identity columns (the
``game_result`` family -- the only feature family, alongside ``elo_inputs``,
Fix 5 classifies as production-eligible; see
:mod:`nfl_hybrid.providers.balldontlie.parity`).

It adds no new rolling-window math: :func:`build_game_result_team_game` is a
thin adapter (structurally identical to
:func:`nfl_hybrid.features.market_history.build_market_history_pregame_state`'s
``_team_market_outcomes`` pattern) that hands a minimal, game-result-only
``team_game`` frame to the existing, unmodified
:func:`nfl_hybrid.features.pregame_rolling.build_team_pregame_features`.  The
Elo family already has its own proven Week-1 prior --
:func:`nfl_hybrid.legacy.elo.LegacyElo.regress_to_mean`, fired once per
season boundary by
:func:`nfl_hybrid.features.elo_state.build_elo_pregame_state` -- and is left
untouched; this module exists only for the metrics that engine does not
cover (scoring margin, points scored/allowed, win rate).

Blend contract (auto-mode revision, closes every ambiguity the first pass
left open)::

    w(n) = n / (n + k)
    blended = w(n) * current_season_state + (1 - w(n)) * prior_reference

``n`` is ``team_season_prior_games`` (already emitted by
``build_team_pregame_features`` -- the count of that team's *strictly prior*
games in the current season; 0 at Week 1). At ``n=0`` this is an EXPLICIT
branch -- ``blended = prior_reference`` -- never a reliance on
``0 * NaN == 0`` (false in IEEE arithmetic). ``prior_reference`` is that
team's own full prior-season average if one exists
(``PRIOR_SEASON_AVAILABLE``), else a single declared, FIXED, untuned neutral
value (``NEUTRAL_FALLBACK`` -- see :data:`DEFAULT_NEUTRAL_FALLBACK` and
:data:`NEUTRAL_PRIOR_SOURCE`; never derived from this repository's outcome
data). ``k`` is one global constant for the whole blend mechanism (not
tuned per feature).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.labels import edge_to_nullable_binary

TRANSFORM_NAME = "build_game_result_team_game"
TRANSFORM_VERSION = "1.0"

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
}

# The exact, closed set of metrics this module blends. Adding a metric here
# requires it to be re-derivable solely from `games`' own score/identity
# columns -- nothing else.
GAME_RESULT_METRICS = ("points_scored", "points_allowed", "margin", "win")

PRIOR_SEASON_AVAILABLE = "PRIOR_SEASON_AVAILABLE"
NEUTRAL_FALLBACK = "NEUTRAL_FALLBACK"

# Fixed, predeclared, UNTUNED_ENGINEERING_PRIOR constants -- never estimated
# from this repository's 2020-2024 outcome data, never searched/tuned. No
# predeclared league-average-points constant already existed anywhere in
# this repository (constants.py, config/*.yaml, priors/*, spreadsheet_
# baselines/* all checked, zero hits) -- 22.0 is a round, publicly-known
# modern-NFL-era scoring average, declared once here and never adjusted in
# response to any observed metric.
NEUTRAL_MARGIN = 0.0
NEUTRAL_WIN_RATE = 0.5
NEUTRAL_POINTS_SCORED = 22.0
NEUTRAL_POINTS_ALLOWED = 22.0
NEUTRAL_PRIOR_SOURCE = "fixed untuned engineering constant; not estimated from 2020-2024 data"

DEFAULT_NEUTRAL_FALLBACK: dict[str, float] = {
    "points_scored": NEUTRAL_POINTS_SCORED,
    "points_allowed": NEUTRAL_POINTS_ALLOWED,
    "margin": NEUTRAL_MARGIN,
    "win": NEUTRAL_WIN_RATE,
}


def build_game_result_team_game(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id): that team's own game-result metrics.

    ``points_scored``/``points_allowed``/``margin`` come directly from
    ``home_score``/``away_score``; ``win`` is
    :func:`nfl_hybrid.labels.edge_to_nullable_binary` applied to the margin
    (a tie is excluded -- ``pd.NA`` -- never scored as a loss, the
    repository's single tie/push policy). Nothing here reads any column
    outside :data:`REQUIRED_GAME_COLUMNS`.
    """
    missing = REQUIRED_GAME_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing game-result-required columns: {sorted(missing)}")

    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    g["home_team_id"] = g["home_team_id"].map(canonical_team_id)
    g["away_team_id"] = g["away_team_id"].map(canonical_team_id)

    home_score = pd.to_numeric(g["home_score"], errors="coerce")
    away_score = pd.to_numeric(g["away_score"], errors="coerce")
    home_margin = home_score - away_score

    home_rows = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team_id": g["home_team_id"],
            "opponent_id": g["away_team_id"],
            "season": g["season"],
            "points_scored": home_score,
            "points_allowed": away_score,
            "margin": home_margin,
            "win": edge_to_nullable_binary(home_margin),
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team_id": g["away_team_id"],
            "opponent_id": g["home_team_id"],
            "season": g["season"],
            "points_scored": away_score,
            "points_allowed": home_score,
            "margin": -home_margin,
            "win": edge_to_nullable_binary(-home_margin),
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


@dataclass(frozen=True)
class Week1PriorConfig:
    """Frozen Week-1/early-season blend configuration.

    ``k`` is the single global blend constant (``w(n) = n / (n + k)``),
    selected once via inner chronological folds and frozen before the
    LOCKED_POST_EXPOSURE_2024_AUDIT. ``neutral_fallback`` maps each metric in
    :data:`GAME_RESULT_METRICS` to its declared, FIXED, untuned neutral
    value (default: :data:`DEFAULT_NEUTRAL_FALLBACK` -- never derived from
    this repository's outcome data; no search/tuning is ever performed
    around these values).
    """

    k: float
    neutral_fallback: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NEUTRAL_FALLBACK))
    neutral_prior_source: str = NEUTRAL_PRIOR_SOURCE
    metrics: tuple[str, ...] = GAME_RESULT_METRICS

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be > 0.")
        missing = set(self.metrics) - set(self.neutral_fallback)
        if missing:
            raise ValueError(f"neutral_fallback missing metric(s): {sorted(missing)}")


def compute_week1_prior_config_hash(config: Week1PriorConfig) -> str:
    payload = json.dumps(
        {
            "k": config.k,
            "neutral_fallback": dict(sorted(config.neutral_fallback.items())),
            "neutral_prior_source": config.neutral_prior_source,
            "metrics": list(config.metrics),
            "transform_name": TRANSFORM_NAME,
            "transform_version": TRANSFORM_VERSION,
        },
        sort_keys=True,
        default=str,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _prior_season_end_values(team_game: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Each team's own full-season average per metric, one row per
    (team_id, season) -- computed from that season's complete history (not
    shifted; a season's own final average may only ever inform a *later*
    season's Week 1, never games within itself -- enforced by the caller's
    ``season + 1`` join, not by this function)."""
    season_summary = (
        team_game.groupby(["team_id", "season"], sort=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    return season_summary


def apply_week1_blend(
    team_pregame: pd.DataFrame,
    team_game: pd.DataFrame,
    *,
    config: Week1PriorConfig,
) -> pd.DataFrame:
    """Attach ``{metric}__week1_blended`` and ``prior_status`` to a
    game-result pregame-state frame already built by
    :func:`nfl_hybrid.features.pregame_rolling.build_team_pregame_features`
    (over :func:`build_game_result_team_game`'s output).

    ``team_pregame`` must already carry ``team_id``, ``season``,
    ``team_season_prior_games``, and ``{metric}__season_mean`` for every
    metric in ``config.metrics``. ``team_game`` is the *raw* (unshifted)
    game-result team-game frame -- used only to compute each team's
    complete prior-season average, never joined onto its own season.

    Blend is computed with an explicit branch, never ``0 * NaN``:
    ``n=0`` always yields ``prior_reference`` outright; for ``n>0`` rows
    where the current-season metric is unexpectedly still missing (e.g.
    every prior game this season was a tie), the documented fail-closed
    fallback is also ``prior_reference`` -- never a fabricated value.
    """
    missing_cols = {"team_id", "season", "team_season_prior_games"} - set(team_pregame.columns)
    if missing_cols:
        raise ValueError(f"team_pregame missing required columns: {sorted(missing_cols)}")
    for metric in config.metrics:
        if f"{metric}__season_mean" not in team_pregame.columns:
            raise ValueError(f"team_pregame missing {metric}__season_mean.")

    season_summary = _prior_season_end_values(team_game, config.metrics)
    prior_reference = season_summary.copy()
    prior_reference["season"] = prior_reference["season"] + 1
    rename = {metric: f"{metric}__prior_season_end_value" for metric in config.metrics}
    prior_reference = prior_reference.rename(columns=rename)

    out = team_pregame.merge(
        prior_reference,
        on=["team_id", "season"],
        how="left",
        validate="many_to_one",
    )

    prior_available = pd.Series(True, index=out.index)
    for metric in config.metrics:
        col = f"{metric}__prior_season_end_value"
        prior_available &= out[col].notna()
    out["prior_status"] = np.where(prior_available, PRIOR_SEASON_AVAILABLE, NEUTRAL_FALLBACK)

    n_games = pd.to_numeric(out["team_season_prior_games"], errors="coerce").to_numpy(dtype=float)
    weight = n_games / (n_games + float(config.k))

    for metric in config.metrics:
        prior_col = f"{metric}__prior_season_end_value"
        season_mean = pd.to_numeric(out[f"{metric}__season_mean"], errors="coerce")
        prior_reference_value = out[prior_col].fillna(config.neutral_fallback[metric]).to_numpy(dtype=float)
        current_available = season_mean.notna().to_numpy()
        current_value = season_mean.to_numpy(dtype=float)

        blended_current = weight * current_value + (1.0 - weight) * prior_reference_value
        blended = np.where(
            n_games == 0,
            prior_reference_value,  # explicit n=0 branch -- never 0 * NaN
            np.where(
                current_available,
                blended_current,
                prior_reference_value,  # n>0 but no current evidence yet (e.g. all-tie season-to-date): documented fail-closed fallback, never a fabricated value
            ),
        )
        out[f"{metric}__week1_blended"] = blended

    out = out.drop(columns=[f"{metric}__prior_season_end_value" for metric in config.metrics])
    return out
