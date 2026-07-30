from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_HORIZON_MINUTES = (10080, 4320, 1440, 360, 60, 15)


def build_prediction_horizons(
    games: pd.DataFrame,
    *,
    horizon_minutes: Iterable[int] = DEFAULT_HORIZON_MINUTES,
) -> pd.DataFrame:
    required = {"game_id", "scheduled_kickoff_utc"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        kickoff = pd.Timestamp(game.scheduled_kickoff_utc)
        if pd.isna(kickoff):
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        else:
            kickoff = kickoff.tz_convert("UTC")
        for minutes in horizon_minutes:
            rows.append(
                {
                    "game_id": game.game_id,
                    "scheduled_kickoff_utc": kickoff,
                    "horizon_minutes": int(minutes),
                    "requested_snapshot_utc": kickoff - pd.Timedelta(minutes=int(minutes)),
                }
            )
    return pd.DataFrame(rows)


def match_odds_to_games(
    odds: pd.DataFrame,
    games: pd.DataFrame,
    *,
    kickoff_tolerance_hours: float = 6.0,
) -> pd.DataFrame:
    """Attach canonical game IDs using teams and scheduled time.

    Team identities are mandatory. Time is used to resolve preseason or
    rescheduled-game ambiguity. Unmatched rows retain a null game_id and a
    diagnostic status.
    """
    required_odds = {
        "provider_event_id", "home_team_id", "away_team_id", "commence_time_utc"
    }
    required_games = {
        "game_id", "home_team_id", "away_team_id", "scheduled_kickoff_utc"
    }
    if required_odds - set(odds.columns):
        raise ValueError(f"Odds missing: {sorted(required_odds - set(odds.columns))}")
    if required_games - set(games.columns):
        raise ValueError(f"Games missing: {sorted(required_games - set(games.columns))}")

    out = odds.copy()
    out["game_id"] = pd.NA
    out["game_match_status"] = "unmatched"
    tolerance = pd.Timedelta(hours=float(kickoff_tolerance_hours))

    game_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in games.groupby(["home_team_id", "away_team_id"], dropna=False):
        game_lookup[key] = group.copy()

    for event_id, event_rows in out.groupby("provider_event_id", dropna=False):
        row = event_rows.iloc[0]
        key = (row["home_team_id"], row["away_team_id"])
        candidates = game_lookup.get(key)
        if candidates is None or candidates.empty:
            out.loc[event_rows.index, "game_match_status"] = "team_pair_not_found"
            continue

        commence = pd.Timestamp(row["commence_time_utc"])
        kickoff = pd.to_datetime(candidates["scheduled_kickoff_utc"], utc=True, errors="coerce")
        deltas = (kickoff - commence).abs()
        viable = candidates[deltas <= tolerance]
        if len(viable) == 1:
            game_id = viable.iloc[0]["game_id"]
            out.loc[event_rows.index, "game_id"] = game_id
            out.loc[event_rows.index, "game_match_status"] = "matched"
        elif len(viable) > 1:
            nearest_index = deltas.loc[viable.index].idxmin()
            out.loc[event_rows.index, "game_id"] = candidates.loc[nearest_index, "game_id"]
            out.loc[event_rows.index, "game_match_status"] = "matched_nearest_ambiguous"
        else:
            out.loc[event_rows.index, "game_match_status"] = "kickoff_outside_tolerance"
    return out


_MATCHED_STATUSES = ("matched", "matched_nearest_ambiguous")


def require_matched_events(
    matched_odds: pd.DataFrame,
    *,
    allowed_statuses: tuple[str, ...] = _MATCHED_STATUSES,
) -> pd.DataFrame:
    """Raise if any priced event failed to match a canonical game.

    Before purchased quotes are swapped in as a benchmark, a silent
    ``game_id = NA`` would quietly drop games and bias the comparison. This
    validator fails loudly instead, listing the offending provider event ids and
    their statuses.
    """

    if "game_match_status" not in matched_odds.columns:
        raise ValueError("Input has no game_match_status; run match_odds_to_games first.")
    bad = matched_odds[~matched_odds["game_match_status"].isin(allowed_statuses)]
    if len(bad):
        offenders = (
            bad[["provider_event_id", "home_team_id", "away_team_id", "game_match_status"]]
            .drop_duplicates()
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"{bad['provider_event_id'].nunique()} event(s) did not match a "
            f"canonical game: {offenders}"
        )
    return matched_odds


def add_market_consensus(odds: pd.DataFrame) -> pd.DataFrame:
    """Add consensus line/probability without deleting bookmaker observations."""
    out = odds.copy()
    keys = ["game_id", "snapshot_utc", "market_type", "outcome_side"]
    grouped = out.groupby(keys, dropna=False)
    out["consensus_line"] = grouped["line_value"].transform("median")
    out["consensus_price_american"] = grouped["price_american"].transform("median")
    out["market_line_std"] = grouped["line_value"].transform("std")
    out["market_probability_std"] = grouped["raw_implied_probability"].transform("std")
    out["number_of_books_reporting"] = grouped["bookmaker_id"].transform("nunique")
    return out


def devig_two_way_groups(odds: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative de-vig within each book, market, line, and snapshot."""
    out = odds.copy()
    out["market_pair_line"] = np.select(
        [
            out["market_type"].eq("spread"),
            out["market_type"].eq("total"),
        ],
        [
            pd.to_numeric(out["line_value"], errors="coerce").abs(),
            pd.to_numeric(out["line_value"], errors="coerce"),
        ],
        default=0.0,
    )
    group_cols = [
        "provider_event_id", "bookmaker_id", "market_type", "snapshot_utc",
        "market_pair_line",
    ]
    overround = out.groupby(group_cols, dropna=False)["raw_implied_probability"].transform("sum")
    out["bookmaker_overround"] = overround
    out["devig_probability"] = np.where(
        overround > 0,
        out["raw_implied_probability"] / overround,
        np.nan,
    )
    return out
