"""Deterministic synthetic fixture factory for the core live-feature CI tests.

This module builds a *small, fully synthetic* NFL-like schedule and play-by-play
that exercise the real production feature pipeline

    aggregate_advanced_team_game
        -> build_team_pregame_features (shift-then-roll)
        -> build_game_pregame_matrix
        -> _diff_features / _rest_features
        -> build_augmented_feature_matrix (training path)
        -> build_live_augmented_features (live path)
        -> resolve_finality_before_asof

with no private backfill, no purchased odds, no provider payloads and no
copyrighted records. Every identifier, timestamp, score and play value is a
hand-chosen deterministic constant so the golden expected outputs are stable
and reproducible on any supported Python / pandas version.

The factory is *setup only*. It is used by:
  * ``scripts/regenerate_live_feature_ci_fixture.py`` to (re)write the committed
    ``games.csv`` / ``pbp.csv`` and the golden ``expected_features.csv``;
It is deliberately NOT imported by the CI tests, which load the committed CSVs.

Design (two synthetic seasons, four teams: BUF, MIA, NYJ, NE):

  Season 2021, weeks 1-5 (weekly Sunday schedule), all verified FINAL except one
  designated in-progress/unknown game X.
  Season 2022, weeks 1-4. Week 3 kicks on a Thursday (short-week flag). Week 4 is
  a genuinely upcoming target with no outcome and no play-by-play.

Three golden target-game regimes (from different weeks/seasons):
  A = 2021 wk 5  NYJ vs BUF  -> complete last-four + valid season-to-date history
  B = 2022 wk 1  BUF vs MIA  -> early-season / season-boundary missingness
                                (2022 season-to-date features are NaN)
  C = 2022 wk 3  BUF vs NE   -> mid-season, short-week rest flag exercised

Plus:
  D = 2022 wk 4  BUF vs MIA  -> upcoming target with NO outcome and NO pbp
  X = 2021 wk 4  MIA vs BUF  -> explicit non-final ("IN_PROGRESS"), populated
                                retrospective scores, PARTIAL pbp, no completion
                                timestamp: excluded in live mode, available under
                                the historical-replay availability assumption.
"""

from __future__ import annotations

import pandas as pd

TEAMS = ("BUF", "MIA", "NYJ", "NE")

# Deterministic per-team base offensive EPA/play (small, distinct, no randomness).
TEAM_BASE = {"BUF": 0.12, "MIA": 0.02, "NYJ": -0.06, "NE": 0.06}

# Deterministic dropback / rush play EPA offsets. Each list sums to zero so the
# per-team-game means are exactly the intended targets.
_DROPBACK_OFFSETS = (0.30, 0.10, -0.10, -0.30, 0.20, -0.20)  # 6 plays, mean 0
_RUSH_OFFSETS = (0.15, -0.15, 0.05, -0.05)                   # 4 plays, mean 0
# Partial (in-progress) game: fewer plays but still zero-sum.
_DROPBACK_OFFSETS_PARTIAL = (0.20, -0.20)
_RUSH_OFFSETS_PARTIAL = (0.10, -0.10)

# game_id  = SEASON_WEEK_AWAY_HOME  (nflverse-style)
# each tuple: (season, week, home, away, kickoff_utc, kind)
#   kind in {"final", "inprogress", "upcoming"}
_SCHEDULE = [
    # 2021 season -------------------------------------------------------------
    (2021, 1, "BUF", "MIA", "2021-09-12T17:00:00Z", "final"),
    (2021, 1, "NYJ", "NE", "2021-09-12T20:25:00Z", "final"),
    (2021, 2, "BUF", "NYJ", "2021-09-19T17:00:00Z", "final"),
    (2021, 2, "MIA", "NE", "2021-09-19T20:25:00Z", "final"),
    (2021, 3, "BUF", "NE", "2021-09-26T17:00:00Z", "final"),
    (2021, 3, "MIA", "NYJ", "2021-09-26T20:25:00Z", "final"),
    (2021, 4, "MIA", "BUF", "2021-10-03T17:00:00Z", "inprogress"),  # X
    (2021, 4, "NE", "NYJ", "2021-10-03T20:25:00Z", "final"),
    (2021, 5, "NYJ", "BUF", "2021-10-10T17:00:00Z", "final"),       # Target A
    (2021, 5, "NE", "MIA", "2021-10-10T20:25:00Z", "final"),
    # 2022 season -------------------------------------------------------------
    (2022, 1, "BUF", "MIA", "2022-09-11T17:00:00Z", "final"),       # Target B
    (2022, 1, "NYJ", "NE", "2022-09-11T20:25:00Z", "final"),
    (2022, 2, "BUF", "NYJ", "2022-09-18T17:00:00Z", "final"),
    (2022, 2, "MIA", "NE", "2022-09-18T20:25:00Z", "final"),
    (2022, 3, "BUF", "NE", "2022-09-22T17:00:00Z", "final"),        # Target C (Thu)
    (2022, 3, "MIA", "NYJ", "2022-09-22T20:25:00Z", "final"),
    (2022, 4, "BUF", "MIA", "2022-10-02T17:00:00Z", "upcoming"),    # Target D
]

# The three frozen golden-parity target game_ids, in canonical order.
TARGET_A = "2021_05_BUF_NYJ"
TARGET_B = "2022_01_MIA_BUF"
TARGET_C = "2022_03_NE_BUF"
GOLDEN_TARGETS = (TARGET_A, TARGET_B, TARGET_C)
# Non-golden live-only target (no outcome / no pbp) and the excluded in-progress game.
TARGET_D = "2022_04_MIA_BUF"
INPROGRESS_GAME = "2021_04_BUF_MIA"


def game_id_for(season: int, week: int, home: str, away: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def _off_target(team: str, season: int, week: int) -> float:
    """Deterministic per-team-game offensive dropback-EPA target."""
    form = ((week % 3) - 1) * 0.03
    season_delta = (season - 2021) * 0.02
    return round(TEAM_BASE[team] + form + season_delta, 6)


def _team_plays(game_id, season, week, home, away, posteam, defteam, kickoff, partial):
    """Deterministic offensive plays for one team in one game."""
    off = _off_target(posteam, season, week)
    rush = round(off - 0.05, 6)
    dbo = _DROPBACK_OFFSETS_PARTIAL if partial else _DROPBACK_OFFSETS
    ruo = _RUSH_OFFSETS_PARTIAL if partial else _RUSH_OFFSETS
    ko = pd.Timestamp(kickoff)
    rows = []
    play_seq = 0
    for offset in dbo:
        epa = round(off + offset, 6)
        rows.append(
            {
                "game_id": game_id, "season": season, "week": week,
                "home_team": home, "away_team": away,
                "posteam": posteam, "defteam": defteam,
                "play_type": "pass", "qb_dropback": 1, "pass_attempt": 1,
                "rush_attempt": 0, "scramble": 0, "no_play": 0, "qb_kneel": 0,
                "down": 1, "game_seconds_remaining": 1800,
                "epa": epa, "success": 1.0 if epa > 0 else 0.0,
                "passing_yards": 8.0, "rushing_yards": 0.0,
                "start_time": (ko + pd.Timedelta(seconds=30 * play_seq)).isoformat(),
            }
        )
        play_seq += 1
    for offset in ruo:
        epa = round(rush + offset, 6)
        rows.append(
            {
                "game_id": game_id, "season": season, "week": week,
                "home_team": home, "away_team": away,
                "posteam": posteam, "defteam": defteam,
                "play_type": "run", "qb_dropback": 0, "pass_attempt": 0,
                "rush_attempt": 1, "scramble": 0, "no_play": 0, "qb_kneel": 0,
                "down": 1, "game_seconds_remaining": 1800,
                "epa": epa, "success": 1.0 if epa > 0 else 0.0,
                "passing_yards": 0.0, "rushing_yards": 4.0,
                "start_time": (ko + pd.Timedelta(seconds=30 * play_seq)).isoformat(),
            }
        )
        play_seq += 1
    return rows


def build_games() -> pd.DataFrame:
    """Deterministic synthetic schedule table (one row per game)."""
    rows = []
    for season, week, home, away, kickoff, kind in _SCHEDULE:
        gid = game_id_for(season, week, home, away)
        home_spread = round(-((TEAM_BASE[home] - TEAM_BASE[away]) * 20 + 2.5), 1)
        total_line = round(44.0 + week * 0.5, 1)
        ko = pd.Timestamp(kickoff)
        if kind == "upcoming":
            home_score = ""
            away_score = ""
            status = "SCHEDULED"
            completion = ""
        else:
            # deterministic retrospective scores (populated even for in-progress X)
            home_score = 20 + (week % 4) * 3
            away_score = 17 + (season - 2021) * 3
            if kind == "inprogress":
                status = "IN_PROGRESS"
                completion = ""  # no finality evidence
            else:
                status = "FINAL"
                completion = (ko + pd.Timedelta(hours=3, minutes=30)).isoformat()
        rows.append(
            {
                "game_id": gid,
                "season": season,
                "week": week,
                "home_team_id": home,
                "away_team_id": away,
                "scheduled_kickoff_utc": ko.isoformat(),
                "home_spread_reference": home_spread,
                "total_line_reference": total_line,
                "home_score": home_score,
                "away_score": away_score,
                "game_status": status,
                "completion_timestamp_utc": completion,
            }
        )
    return pd.DataFrame(rows)


def build_pbp() -> pd.DataFrame:
    """Deterministic synthetic play-by-play (upcoming target D has none)."""
    rows = []
    for season, week, home, away, kickoff, kind in _SCHEDULE:
        if kind == "upcoming":
            continue  # target D has no play-by-play
        gid = game_id_for(season, week, home, away)
        partial = kind == "inprogress"
        for posteam, defteam in ((home, away), (away, home)):
            rows.extend(
                _team_plays(gid, season, week, home, away, posteam, defteam, kickoff, partial)
            )
    return pd.DataFrame(rows)


# Column order used when writing the committed CSV fixtures.
GAMES_COLUMNS = [
    "game_id", "season", "week", "home_team_id", "away_team_id",
    "scheduled_kickoff_utc", "home_spread_reference", "total_line_reference",
    "home_score", "away_score", "game_status", "completion_timestamp_utc",
]
PBP_COLUMNS = [
    "game_id", "season", "week", "home_team", "away_team", "posteam", "defteam",
    "play_type", "qb_dropback", "pass_attempt", "rush_attempt", "scramble",
    "no_play", "qb_kneel", "down", "game_seconds_remaining",
    "epa", "success", "passing_yards", "rushing_yards", "start_time",
]
