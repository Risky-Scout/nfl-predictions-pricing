from __future__ import annotations

import numpy as np
import pandas as pd


def crosswalk_spreadspoke_to_nflverse(
    spreadspoke_games: pd.DataFrame,
    nflverse_games: pd.DataFrame,
    *,
    date_tolerance_days: int = 2,
) -> pd.DataFrame:
    """Create a provider-ID crosswalk using franchise IDs and game dates.

    nflverse `game_id` is the internal canonical game key for 2020 onward.
    Spreadspoke remains an independent continuity/reference source.
    """
    required = {"game_id", "season", "home_team_id", "away_team_id", "schedule_date"}
    for name, frame in (
        ("spreadspoke_games", spreadspoke_games),
        ("nflverse_games", nflverse_games),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    tolerance = pd.Timedelta(days=int(date_tolerance_days))
    rows: list[dict[str, object]] = []

    nfl = nflverse_games.copy()
    nfl["schedule_date"] = pd.to_datetime(nfl["schedule_date"], errors="coerce")
    spread = spreadspoke_games.copy()
    spread["schedule_date"] = pd.to_datetime(spread["schedule_date"], errors="coerce")

    lookup = {
        key: group
        for key, group in nfl.groupby(
            ["season", "home_team_id", "away_team_id"],
            dropna=False,
        )
    }

    for row in spread.itertuples(index=False):
        key = (row.season, row.home_team_id, row.away_team_id)
        candidates = lookup.get(key)
        status = "unmatched"
        canonical_game_id = pd.NA
        date_delta_days = np.nan
        candidate_count = 0
        if candidates is not None and len(candidates):
            deltas = (
                pd.to_datetime(candidates["schedule_date"], errors="coerce")
                - pd.Timestamp(row.schedule_date)
            ).abs()
            viable = candidates[deltas <= tolerance]
            candidate_count = len(viable)
            if len(viable):
                nearest_idx = deltas.loc[viable.index].idxmin()
                canonical_game_id = candidates.loc[nearest_idx, "game_id"]
                date_delta_days = float(deltas.loc[nearest_idx] / pd.Timedelta(days=1))
                status = "matched" if len(viable) == 1 else "matched_nearest_ambiguous"
            else:
                status = "date_outside_tolerance"
        else:
            status = "team_pair_not_found"

        rows.append(
            {
                "canonical_game_id": canonical_game_id,
                "nflverse_game_id": canonical_game_id,
                "spreadspoke_game_id": row.game_id,
                "season": row.season,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "spreadspoke_schedule_date": row.schedule_date,
                "date_delta_days": date_delta_days,
                "candidate_count": candidate_count,
                "match_status": status,
            }
        )
    return pd.DataFrame(rows)


def enrich_games_with_spreadspoke(
    nflverse_games: pd.DataFrame,
    spreadspoke_games: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Add Spreadspoke reference fields without overwriting nflverse fields."""
    matched = crosswalk[crosswalk["canonical_game_id"].notna()].copy()
    spread_columns = [
        "game_id",
        "home_spread_reference",
        "total_line_reference",
        "temperature_f_observed",
        "wind_mph_observed",
        "humidity_observed",
        "weather_detail_observed",
        "stadium_name",
        "line_reference_type",
        "source_retrieved_at_utc",
    ]
    available = [column for column in spread_columns if column in spreadspoke_games.columns]
    spread = spreadspoke_games[available].copy()
    spread = spread.rename(
        columns={
            "game_id": "spreadspoke_game_id",
            "home_spread_reference": "spreadspoke_home_spread_reference",
            "total_line_reference": "spreadspoke_total_line_reference",
            "temperature_f_observed": "spreadspoke_temperature_f_observed",
            "wind_mph_observed": "spreadspoke_wind_mph_observed",
            "humidity_observed": "spreadspoke_humidity_observed",
            "weather_detail_observed": "spreadspoke_weather_detail_observed",
            "stadium_name": "spreadspoke_stadium_name",
            "line_reference_type": "spreadspoke_line_reference_type",
            "source_retrieved_at_utc": "spreadspoke_retrieved_at_utc",
        }
    )
    link = matched[
        ["canonical_game_id", "spreadspoke_game_id", "match_status", "date_delta_days"]
    ].merge(spread, on="spreadspoke_game_id", how="left")
    out = nflverse_games.merge(
        link,
        left_on="game_id",
        right_on="canonical_game_id",
        how="left",
    )
    out["spread_reference_difference"] = (
        pd.to_numeric(out.get("home_spread_reference"), errors="coerce")
        - pd.to_numeric(out.get("spreadspoke_home_spread_reference"), errors="coerce")
    )
    out["total_reference_difference"] = (
        pd.to_numeric(out.get("total_line_reference"), errors="coerce")
        - pd.to_numeric(out.get("spreadspoke_total_line_reference"), errors="coerce")
    )
    return out
