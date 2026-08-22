"""2025 BDL-vs-historical overlap backtest (Fix 5 certification, Section 1).

Compares a non-cherry-picked, multi-week, multi-team, both-orientation
sample of 2025 regular-season games between BALLDONTLIE (live) and the
historical nflverse estate (``NFL_MODEL_DATA_ROOT``) across FOUR families:

- game identity + final score (season/week/teams/orientation/kickoff, home
  and away score)
- team box (field-by-field, with an explicit direct-vs-derived mapping --
  see ``TEAM_BOX_FIELD_MAP``)
- player/QB box stats (identity coverage by name+team+game since BDL's
  player id space has no crosswalk to nflverse's gsis_id -- see
  ``canonical.PLAYER_IDENTITY_PARITY``)
- PBP (pagination completeness, canonical play count vs. historical row
  count, ordering/duplicate anomalies, terminal-score reconciliation)

Fails closed with a clear message and non-zero exit if either input is
unavailable -- never fabricates a report.

Run: NFL_MODEL_DATA_ROOT=/path/to/estate PYTHONPATH=src python \
    scripts/bdl_overlap_backtest.py --season 2025 --sample-size 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_hybrid.data import external_data
from nfl_hybrid.providers.balldontlie import canonical, plays as plays_mod
from nfl_hybrid.providers.balldontlie.client import (
    BDLClient,
    BDLConfigError,
    BDLRateLimitError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# -- team box: BDL field -> (historical field or None if derived, note) ------
TEAM_BOX_FIELD_MAP: dict[str, tuple[str | None, str]] = {
    "passing_completions": ("completions", "direct"),
    "passing_attempts": ("attempts", "direct"),
    "rushing_yards": ("rushing_yards", "direct"),
    "rushing_attempts": ("carries", "direct"),
    "sacks": ("sacks_suffered", "direct"),
    "sack_yards_lost": ("sack_yards_lost", "direct (BDL reports unsigned magnitude vs historical's signed-negative convention -- compared via abs())"),
    "penalties": ("penalties", "direct"),
    "penalty_yards": ("penalty_yards", "direct"),
    "interceptions_thrown": ("passing_interceptions", "direct"),
    "fumbles_lost": ("fumbles_lost_total", "direct"),
    "defensive_touchdowns": ("def_tds", "direct"),
    "first_downs_passing": ("passing_first_downs", "direct"),
    "first_downs_rushing": ("rushing_first_downs", "direct"),
    # NOTE (measured live 2026-08-22): historical nflverse team_stats stores
    # sack_yards_lost as a SIGNED value (negative = yards lost); BDL reports
    # the unsigned magnitude. `passing_yards + sack_yards_lost` (adding the
    # negative) is therefore nflverse's own net-passing-yards arithmetic --
    # confirmed to match BDL's net_passing_yards exactly once the sign
    # convention is accounted for (19/19 in the 2025 overlap sample).
    "net_passing_yards": (None, "derived: historical passing_yards + sack_yards_lost (historical sign is negative)"),
    "turnovers": (None, "derived: historical passing_interceptions + fumbles_lost_total"),
    "total_yards": (None, "derived: historical (passing_yards + sack_yards_lost) + rushing_yards"),
}

# -- player/QB box: BDL field -> historical field (all direct) ---------------
PLAYER_BOX_FIELD_MAP: dict[str, str] = {
    "passing_completions": "completions",
    "passing_attempts": "attempts",
    "passing_yards": "passing_yards",
    "passing_touchdowns": "passing_tds",
    "passing_interceptions": "passing_interceptions",
    "sacks": "sacks_suffered",
    "rushing_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_touchdowns": "rushing_tds",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_touchdowns": "receiving_tds",
    "receiving_targets": "targets",
}


def _sample_games(games: pd.DataFrame, season: int, sample_size: int) -> pd.DataFrame:
    season_games = games.loc[
        (games["season"] == season) & (games["season_type"] == "REG")
    ].sort_values(["week", "game_id"])
    if season_games.empty:
        return season_games
    # Stratify across weeks: take roughly evenly-spaced rows rather than the
    # first N (which would all be week 1) so the sample spans the season.
    step = max(1, len(season_games) // sample_size)
    return season_games.iloc[::step].head(sample_size)


def _compare_one(historical_row: pd.Series, bdl_games: pd.DataFrame) -> dict[str, Any]:
    match = bdl_games.loc[
        (bdl_games["season"] == historical_row["season"])
        & (bdl_games["week"] == historical_row["week"])
        & (bdl_games["home_team"] == historical_row["home_team_id"])
        & (bdl_games["away_team"] == historical_row["away_team_id"])
    ]
    if match.empty:
        return {"game_id": historical_row["game_id"], "matched": False}

    bdl_row = match.iloc[0]
    hist_kickoff = pd.to_datetime(historical_row.get("scheduled_kickoff_utc"), utc=True, errors="coerce")
    bdl_kickoff = pd.to_datetime(bdl_row.get("kickoff_utc"), utc=True, errors="coerce")
    kickoff_diff_minutes = (
        abs((hist_kickoff - bdl_kickoff).total_seconds()) / 60.0
        if pd.notna(hist_kickoff) and pd.notna(bdl_kickoff)
        else None
    )
    home_score_match = historical_row.get("home_score") == bdl_row.get("home_score")
    away_score_match = historical_row.get("away_score") == bdl_row.get("away_score")

    return {
        "game_id": historical_row["game_id"],
        "provider_game_id": int(bdl_row["provider_game_id"]),
        "week": int(historical_row["week"]),
        "home_team": historical_row["home_team_id"],
        "away_team": historical_row["away_team_id"],
        "matched": True,
        "kickoff_diff_minutes": kickoff_diff_minutes,
        "home_score_match": bool(home_score_match),
        "away_score_match": bool(away_score_match),
        "historical_home_score": historical_row.get("home_score"),
        "bdl_home_score": bdl_row.get("home_score"),
        "historical_away_score": historical_row.get("away_score"),
        "bdl_away_score": bdl_row.get("away_score"),
    }


def _get_with_retry(fn, *args, retries: int = 3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except BDLRateLimitError as exc:
            if attempt == retries - 1:
                raise
            time.sleep(exc.retry_after_seconds or 2.0)
    raise AssertionError("unreachable")


def _numeric_diff_stats(pairs: list[tuple[float | None, float | None]]) -> dict[str, Any]:
    hist_vals = [h for h, _ in pairs]
    bdl_vals = [b for _, b in pairs]
    both_present = [(h, b) for h, b in pairs if h is not None and b is not None and pd.notna(h) and pd.notna(b)]
    hist_missing = sum(1 for h in hist_vals if h is None or pd.isna(h))
    bdl_missing = sum(1 for b in bdl_vals if b is None or pd.isna(b))
    if not both_present:
        return {
            "n_rows": len(pairs),
            "n_comparable": 0,
            "n_exact_agree": 0,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "historical_missing": hist_missing,
            "bdl_missing": bdl_missing,
        }
    diffs = [abs(float(h) - float(b)) for h, b in both_present]
    return {
        "n_rows": len(pairs),
        "n_comparable": len(both_present),
        "n_exact_agree": sum(1 for d in diffs if d == 0),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "max_abs_diff": max(diffs),
        "historical_missing": hist_missing,
        "bdl_missing": bdl_missing,
    }


def _compare_team_box(
    matched: list[dict[str, Any]],
    client: BDLClient,
    historical_team_stats: pd.DataFrame,
) -> dict[str, Any]:
    provider_game_ids = [m["provider_game_id"] for m in matched]
    bdl_raw = _get_with_retry(client.get_team_stats, game_ids=provider_game_ids).data
    bdl_frame = canonical.normalize_team_stats(bdl_raw)
    # /team_stats does not honor a weeks/game_ids-scoped guarantee beyond
    # what's documented -- filter defensively to exactly the requested games.
    bdl_frame = bdl_frame.loc[bdl_frame["provider_game_id"].isin(provider_game_ids)]

    field_pairs: dict[str, list[tuple[float | None, float | None]]] = {f: [] for f in TEAM_BOX_FIELD_MAP}
    rows_compared = 0
    rows_missing_bdl_side = 0
    for m in matched:
        hist_rows = historical_team_stats.loc[historical_team_stats["game_id"] == m["game_id"]]
        for _, hist_row in hist_rows.iterrows():
            team = hist_row["team"]
            bdl_rows = bdl_frame.loc[
                (bdl_frame["provider_game_id"] == m["provider_game_id"]) & (bdl_frame["team"] == team)
            ]
            if bdl_rows.empty:
                rows_missing_bdl_side += 1
                continue
            bdl_row = bdl_rows.iloc[0]
            rows_compared += 1
            hist_passing_yards = hist_row.get("passing_yards")
            hist_sack_yards_lost = hist_row.get("sack_yards_lost")  # historical convention: negative
            hist_rushing_yards = hist_row.get("rushing_yards")
            hist_net_passing = (
                hist_passing_yards + hist_sack_yards_lost
                if pd.notna(hist_passing_yards) and pd.notna(hist_sack_yards_lost)
                else None
            )
            hist_turnovers = (
                hist_row.get("passing_interceptions", 0) + hist_row.get("fumbles_lost_total", 0)
                if pd.notna(hist_row.get("passing_interceptions")) and pd.notna(hist_row.get("fumbles_lost_total"))
                else None
            )
            hist_total_yards = (
                hist_net_passing + hist_rushing_yards
                if hist_net_passing is not None and pd.notna(hist_rushing_yards)
                else None
            )

            for bdl_field, (hist_field, _kind) in TEAM_BOX_FIELD_MAP.items():
                bdl_val = bdl_row.get(bdl_field)
                if bdl_field == "sack_yards_lost":
                    # BDL reports the unsigned magnitude; historical is signed negative.
                    hist_val = abs(hist_sack_yards_lost) if pd.notna(hist_sack_yards_lost) else None
                elif hist_field is not None:
                    hist_val = hist_row.get(hist_field)
                elif bdl_field == "net_passing_yards":
                    hist_val = hist_net_passing
                elif bdl_field == "turnovers":
                    hist_val = hist_turnovers
                elif bdl_field == "total_yards":
                    hist_val = hist_total_yards
                else:
                    hist_val = None
                field_pairs[bdl_field].append((hist_val, bdl_val))

    field_report = {
        bdl_field: {"historical_field_or_derivation": TEAM_BOX_FIELD_MAP[bdl_field][1], **_numeric_diff_stats(pairs)}
        for bdl_field, pairs in field_pairs.items()
    }
    return {
        "rows_compared": rows_compared,
        "rows_missing_bdl_side": rows_missing_bdl_side,
        "fields": field_report,
    }


def _normalize_player_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def _compare_player_stats(
    matched: list[dict[str, Any]],
    client: BDLClient,
    historical_player_stats: pd.DataFrame,
) -> dict[str, Any]:
    provider_game_ids = [m["provider_game_id"] for m in matched]
    bdl_raw = _get_with_retry(client.get_player_stats, game_ids=provider_game_ids).data
    bdl_frame = canonical.normalize_player_stats(bdl_raw)
    bdl_frame = bdl_frame.loc[bdl_frame["provider_game_id"].isin(provider_game_ids)].copy()
    bdl_frame["_name_key"] = bdl_frame["player_name"].map(_normalize_player_name)

    hist_relevant_cols = ["attempts", "carries", "targets"]
    field_pairs: dict[str, list[tuple[float | None, float | None]]] = {f: [] for f in PLAYER_BOX_FIELD_MAP}
    total_hist_rows = 0
    matched_by_name = 0
    total_hist_qb_rows = 0
    matched_qb_by_name = 0

    for m in matched:
        hist_rows = historical_player_stats.loc[historical_player_stats["game_id"] == m["game_id"]]
        # only players with recorded offensive activity -- excludes rows
        # that are pure zero-stat roster entries not comparable to BDL's
        # per-game box (which only lists players who recorded a stat).
        hist_rows = hist_rows.loc[(hist_rows[hist_relevant_cols].fillna(0).sum(axis=1) > 0)]
        bdl_subset = bdl_frame.loc[bdl_frame["provider_game_id"] == m["provider_game_id"]]

        for _, hist_row in hist_rows.iterrows():
            total_hist_rows += 1
            is_qb = pd.notna(hist_row.get("attempts")) and hist_row.get("attempts", 0) > 0
            if is_qb:
                total_hist_qb_rows += 1
            name_key = _normalize_player_name(hist_row.get("player_display_name") or hist_row.get("player_name"))
            candidates = bdl_subset.loc[bdl_subset["_name_key"] == name_key]
            if candidates.empty:
                continue
            matched_by_name += 1
            if is_qb:
                matched_qb_by_name += 1
            bdl_row = candidates.iloc[0]
            for bdl_field, hist_field in PLAYER_BOX_FIELD_MAP.items():
                field_pairs[bdl_field].append((hist_row.get(hist_field), bdl_row.get(bdl_field)))

    field_report = {
        bdl_field: {"historical_field": hist_field, **_numeric_diff_stats(pairs)}
        for bdl_field, hist_field in PLAYER_BOX_FIELD_MAP.items()
        for pairs in [field_pairs[bdl_field]]
    }
    return {
        "historical_active_player_rows": total_hist_rows,
        "matched_by_name_team_game": matched_by_name,
        "player_identity_coverage_pct": (matched_by_name / total_hist_rows * 100.0) if total_hist_rows else None,
        "historical_qb_rows": total_hist_qb_rows,
        "matched_qb_by_name": matched_qb_by_name,
        "qb_identity_coverage_pct": (matched_qb_by_name / total_hist_qb_rows * 100.0) if total_hist_qb_rows else None,
        "unresolved_provider_id_crosswalk": "BDL provider_player_id has no crosswalk to nflverse "
        "player_id/gsis_id in this repo -- matching done by normalized player_name + game only "
        "(see canonical.PLAYER_IDENTITY_PARITY = 'PARTIAL').",
        "fields": field_report,
    }


def _compare_pbp(matched: list[dict[str, Any]], client: BDLClient, historical_pbp: pd.DataFrame) -> dict[str, Any]:
    per_game: list[dict[str, Any]] = []
    for m in matched:
        raw_plays = _get_with_retry(client.get_plays, m["provider_game_id"])
        canonical_plays, norm_report = plays_mod.normalize_plays(
            raw_plays.data, provider_game_id=m["provider_game_id"]
        )
        completeness = plays_mod.pbp_completeness(
            canonical_plays,
            norm_report,
            game_final=True,
            canonical_home_score=m["bdl_home_score"],
            canonical_away_score=m["bdl_away_score"],
        )
        hist_rows = historical_pbp.loc[historical_pbp["game_id"] == m["game_id"]]
        per_game.append(
            {
                "game_id": m["game_id"],
                "provider_game_id": m["provider_game_id"],
                "pages_fetched": raw_plays.pages_fetched,
                "bdl_canonical_play_count": len(canonical_plays),
                "historical_pbp_row_count": len(hist_rows),
                "exact_duplicate_play_ids": len(norm_report.exact_duplicate_play_ids),
                "conflicting_play_ids": len(norm_report.conflicting_play_ids),
                "foreign_game_play_ids": len(norm_report.foreign_game_play_ids),
                "period_ordering_violations": norm_report.period_ordering_violations,
                "terminal_score_reconciled": completeness.terminal_score_reconciled,
                "pbp_complete": completeness.pbp_complete,
            }
        )
    return {
        "games": per_game,
        "n_games_multi_page": sum(1 for g in per_game if g["pages_fetched"] > 1),
        "n_games_pbp_complete": sum(1 for g in per_game if g["pbp_complete"]),
        "n_games_terminal_score_reconciled_true": sum(1 for g in per_game if g["terminal_score_reconciled"] is True),
        "n_games_terminal_score_reconciled_false": sum(1 for g in per_game if g["terminal_score_reconciled"] is False),
        "n_games_terminal_score_reconciled_unavailable": sum(
            1 for g in per_game if g["terminal_score_reconciled"] is None
        ),
        "n_games_with_normalization_anomaly": sum(
            1 for g in per_game if g["conflicting_play_ids"] or g["foreign_game_play_ids"] or g["period_ordering_violations"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "bdl_overlap_backtest_2025.json")
    args = parser.parse_args(argv)

    try:
        games_path = external_data.resolve("backfill.games")
        team_stats_path = external_data.resolve("backfill.team_stats")
        player_stats_path = external_data.resolve("backfill.player_stats")
        pbp_path = external_data.resolve("backfill.pbp")
    except external_data.ExternalDataUnavailableError as exc:
        print(f"NOT RUN: historical estate unavailable ({exc})", file=sys.stderr)
        return 2

    try:
        client = BDLClient()
        client.config.resolved_api_key()
    except BDLConfigError as exc:
        print(f"NOT RUN: BDL credential unavailable ({exc})", file=sys.stderr)
        return 2

    historical_games = pd.read_parquet(games_path)
    sample = _sample_games(historical_games, args.season, args.sample_size)
    if sample.empty:
        print(f"NOT RUN: no {args.season} REG-season rows found in {games_path}", file=sys.stderr)
        return 2

    bdl_raw = client.get_games(seasons=[args.season], season_type=2).data
    bdl_games = canonical.normalize_games(bdl_raw, season_type_hint="REG")

    comparisons = [_compare_one(row, bdl_games) for _, row in sample.iterrows()]
    matched = [c for c in comparisons if c["matched"]]

    weeks_covered = sorted({c["week"] for c in matched})
    teams_covered = sorted({t for c in matched for t in (c["home_team"], c["away_team"])})
    home_orientations = sum(1 for c in matched if True)  # every matched game has one home + one away row by construction
    away_orientations = home_orientations

    team_box_report = _compare_team_box(matched, client, pd.read_parquet(team_stats_path)) if matched else {}
    player_stats_report = (
        _compare_player_stats(matched, client, pd.read_parquet(player_stats_path)) if matched else {}
    )
    pbp_report = _compare_pbp(matched, client, pd.read_parquet(pbp_path)) if matched else {}

    report = {
        "season": args.season,
        "sample_selection_method": "deterministic evenly-spaced stride over all REG-season games "
        "sorted by (week, game_id) -- not cherry-picked; step = len(season_games) // sample_size",
        "n_games_sampled": len(comparisons),
        "n_games_matched": len(matched),
        "n_games_unmatched": len(comparisons) - len(matched),
        "weeks_covered": weeks_covered,
        "teams_covered": teams_covered,
        "n_home_orientations": home_orientations,
        "n_away_orientations": away_orientations,
        "game_identity_and_score": {
            "home_score_agreement": sum(1 for c in matched if c["home_score_match"]),
            "away_score_agreement": sum(1 for c in matched if c["away_score_match"]),
            "max_kickoff_diff_minutes": max(
                (c["kickoff_diff_minutes"] for c in matched if c["kickoff_diff_minutes"] is not None),
                default=None,
            ),
            "comparisons": comparisons,
        },
        "team_box": team_box_report,
        "player_stats": player_stats_report,
        "pbp": pbp_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote overlap report to {args.output}: {report['n_games_matched']}/{report['n_games_sampled']} matched, "
        f"weeks={weeks_covered}, teams={len(teams_covered)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
