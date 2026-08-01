"""Deterministic synthetic fixture for the Week 1 shadow reliability evaluator.

Builds a small, self-contained (matrix, games, odds) triple with the exact
columns the evaluator consumes — no backfill parquet, no purchased odds, no
network. Week 1 rows deliberately carry NaN season-to-date EPA features to
exercise the imputation regime; later weeks are fully populated.

Everything is generated from a fixed seed so the fixture (and every hash derived
from it) is byte-stable across machines and Python versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_hybrid.features.augmented_matrix import FROZEN_FEATURES

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
WEEKS = (1, 2, 3, 4, 5)
TEAMS = [f"T{n:02d}" for n in range(16)]
_SEED = 424242

# season-to-date EPA features that are unavailable (NaN) in Week 1
SEASON_MEAN_FEATURES = [c for c in FROZEN_FEATURES if c.endswith("season_mean")]


def _american_from_prob(p: float, vig: float = 0.045) -> int:
    """A vigged American price implying probability ~ p (+ half the overround)."""
    q = min(max(p + vig / 2.0, 0.02), 0.98)
    dec = 1.0 / q
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))


def build_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    rows = []
    kickoff = pd.Timestamp("2020-09-10T00:00:00Z")
    for season in SEASONS:
        strengths = {t: rng.normal(0, 6.0) for t in TEAMS}
        for week in WEEKS:
            order = list(TEAMS)
            rng.shuffle(order)
            for gi in range(8):
                home, away = order[2 * gi], order[2 * gi + 1]
                kickoff = kickoff + pd.Timedelta(hours=6)
                edge = strengths[home] - strengths[away] + 2.4  # home-field
                home_spread = float(np.clip(np.round(-edge), -14, 14))
                total_line = float(np.round(44 + rng.normal(0, 3)))
                margin = int(np.round(-home_spread + rng.normal(0, 12)))
                total_pts = int(max(20, np.round(total_line + rng.normal(0, 10))))
                home_score = int(max(0, round((total_pts + margin) / 2)))
                away_score = int(max(0, round((total_pts - margin) / 2)))
                margin = home_score - away_score
                total_pts = home_score + away_score
                rec = {
                    "game_id": f"{season}_{week:02d}_{gi:02d}",
                    "season": int(season), "week": int(week),
                    "home_team_id": home, "away_team_id": away,
                    "scheduled_kickoff_utc": kickoff.isoformat(),
                    "home_score": home_score, "away_score": away_score,
                    "home_margin": margin, "total_points": total_pts,
                    "home_spread": home_spread, "total_line": total_line,
                    "home_win": int(margin > 0),
                    "home_cover": int(margin + home_spread > 0),
                    "over": int(total_pts > total_line),
                }
                for c in FROZEN_FEATURES:
                    if c == "home_spread":
                        rec[c] = home_spread
                    elif c == "total_line":
                        rec[c] = total_line
                    elif c in SEASON_MEAN_FEATURES and week == 1:
                        rec[c] = np.nan  # season-to-date unavailable in Week 1
                    elif c in ("home_short_week", "away_short_week"):
                        rec[c] = 0.0
                    elif c in ("home_rest_days", "away_rest_days"):
                        rec[c] = 7.0
                    elif c == "rest_diff":
                        rec[c] = 0.0
                    else:
                        rec[c] = float(rng.normal(0, 0.08))
                rows.append(rec)
    return pd.DataFrame(rows)


def build_games(matrix: pd.DataFrame) -> pd.DataFrame:
    """Schedule + reference prices implying a de-viggable two-way market."""
    rng = np.random.default_rng(_SEED + 1)
    rows = []
    for _, r in matrix.iterrows():
        # true-ish market probabilities from the spread/total
        p_home = float(np.clip(1.0 / (1.0 + np.exp(r["home_spread"] / 6.0)), 0.05, 0.95))
        p_cover = float(np.clip(0.5 + rng.normal(0, 0.03), 0.05, 0.95))
        p_over = float(np.clip(0.5 + rng.normal(0, 0.03), 0.05, 0.95))
        rows.append({
            "game_id": r["game_id"], "season": r["season"], "week": r["week"],
            "home_team_id": r["home_team_id"], "away_team_id": r["away_team_id"],
            "scheduled_kickoff_utc": r["scheduled_kickoff_utc"],
            "home_spread_reference": r["home_spread"], "total_line_reference": r["total_line"],
            "home_moneyline_reference": _american_from_prob(p_home),
            "away_moneyline_reference": _american_from_prob(1.0 - p_home),
            "home_spread_price_reference": _american_from_prob(p_cover),
            "away_spread_price_reference": _american_from_prob(1.0 - p_cover),
            "over_price_reference": _american_from_prob(p_over),
            "under_price_reference": _american_from_prob(1.0 - p_over),
        })
    return pd.DataFrame(rows)


def build_odds(games: pd.DataFrame) -> pd.DataFrame:
    """Small multi-book closing-odds frame for contract-matching tests.

    game 0: two books at the reference spread/total/moneyline in one closing
    snapshot (MATCHED consensus = book mean), a third book quoting a *different*
    spread point (must be excluded), plus a post-kickoff and a live quote (both
    rejected). game 1: only a mismatched-point spread quote (CONTRACT_MISMATCH).
    """
    rows = []

    def add(gid, book, mtype, side, line, mtk, dp, live=False):
        rows.append({
            "game_id": gid, "bookmaker_id": book, "market_type": mtype,
            "outcome_side": side, "line_value": line, "minutes_to_kickoff": mtk,
            "is_live": live, "devig_probability": dp,
        })

    g0 = games.iloc[0]
    gid0 = g0["game_id"]
    ref_spread = float(g0["home_spread_reference"])
    ref_total = float(g0["total_line_reference"])
    # one closing snapshot (mtk == 18) shared across books, as in real snapshots
    add(gid0, "bookA", "spread", "home", ref_spread, 18.0, 0.52)
    add(gid0, "bookB", "spread", "home", ref_spread, 18.0, 0.54)
    add(gid0, "bookC", "spread", "home", ref_spread + 1.0, 18.0, 0.61)  # different point -> excluded
    add(gid0, "bookA", "total", "over", ref_total, 18.0, 0.49)
    add(gid0, "bookB", "total", "over", ref_total, 18.0, 0.51)
    add(gid0, "bookA", "moneyline", "home", np.nan, 18.0, 0.58)
    add(gid0, "bookB", "moneyline", "home", np.nan, 18.0, 0.60)
    add(gid0, "bookA", "spread", "home", ref_spread, -5.0, 0.99)          # post-kickoff -> excluded
    add(gid0, "bookA", "spread", "home", ref_spread, 30.0, 0.30, live=True)  # live -> excluded

    g1 = games.iloc[1]
    gid1 = g1["game_id"]
    add(gid1, "bookA", "spread", "home", float(g1["home_spread_reference"]) + 2.0, 18.0, 0.7)
    return pd.DataFrame(rows)


def build_all():
    matrix = build_matrix()
    games = build_games(matrix)
    odds = build_odds(games)
    return matrix, games, odds
