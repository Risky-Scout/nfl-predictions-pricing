from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import requests

from nfl_hybrid.data.team_ids import canonical_team_id


HORIZONS_MINUTES = {
    "opening_7d": 10080,
    "opening_72h": 4320,
    "opening_24h": 1440,
    "opening_6h": 360,
    "opening_60m": 60,
    "closing_t10": 10,
}

TEAM_ALIASES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
}


@dataclass(frozen=True)
class BackfillConfig:
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023)
    minimum_books: int = 3
    kickoff_tolerance_minutes: int = 180
    estimated_cost_per_request: int = 30
    minimum_interval_seconds: float = 0.35
    maximum_requests: int | None = None
    plan_only: bool = False


def _team_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    try:
        return canonical_team_id(text)
    except ValueError:
        return None


def _find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing expected column. Tried: {names}")


def normalize_games(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "game_id": _find_column(frame, ("game_id",)),
        "season": _find_column(frame, ("season",)),
        "week": _find_column(frame, ("week",)),
        "kickoff_utc": _find_column(
            frame, (
            "scheduled_kickoff_utc",
            "kickoff_utc",
            "game_datetime_utc",
            "start_time_utc",
        )
        ),
        "home_team_id": _find_column(
            frame, ("home_team_id", "home_team", "home_team_abbr")
        ),
        "away_team_id": _find_column(
            frame, ("away_team_id", "away_team", "away_team_abbr")
        ),
    }
    output = frame[list(columns.values())].copy()
    output.columns = list(columns)
    output["kickoff_utc"] = pd.to_datetime(
        output["kickoff_utc"], utc=True, errors="raise"
    )
    output["home_team_id"] = output["home_team_id"].map(_team_id)
    output["away_team_id"] = output["away_team_id"].map(_team_id)
    if output[["home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError("Unmappable team identifier in canonical games.")
    if output["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id in canonical games.")
    return output


def build_snapshot_plan(
    games: pd.DataFrame,
    config: BackfillConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = games[games["season"].isin(config.seasons)]
    for game in selected.itertuples(index=False):
        for horizon, minutes in HORIZONS_MINUTES.items():
            rows.append(
                {
                    "game_id": game.game_id,
                    "season": int(game.season),
                    "week": game.week,
                    "kickoff_utc": game.kickoff_utc,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "horizon": horizon,
                    "requested_snapshot_utc": (
                        game.kickoff_utc - timedelta(minutes=minutes)
                    ),
                }
            )
    return pd.DataFrame(rows)


def decimal_to_american(price: float) -> float:
    value = float(price)
    if value <= 1.0 or not math.isfinite(value):
        return math.nan
    return round((value - 1.0) * 100.0) if value >= 2.0 else round(
        -100.0 / (value - 1.0)
    )


def _snapshot_cache_path(cache_root: Path, timestamp: str) -> Path:
    digest = hashlib.sha256(timestamp.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{digest}.json"


def fetch_snapshot(
    timestamp: str,
    cache_root: Path,
    *,
    minimum_interval_seconds: float,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _snapshot_cache_path(cache_root, timestamp)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    key = os.environ.get("THE_ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("THE_ODDS_API_KEY is not active.")

    response = None
    for attempt in range(6):
        response = requests.get(
            "https://api.the-odds-api.com/v4/historical/"
            "sports/americanfootball_nfl/odds",
            params={
                "apiKey": key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
                "date": timestamp,
            },
            timeout=60,
        )
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(min(30.0, 0.75 * (2 ** attempt)))
            continue
        break
    if response is None or not response.ok:
        status = None if response is None else response.status_code
        text = "" if response is None else response.text[:1000]
        raise RuntimeError(f"Historical odds request failed: {status}: {text}")

    payload = {
        "requested_timestamp": timestamp,
        "request_cost": response.headers.get("x-requests-last"),
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "response": response.json(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    time.sleep(minimum_interval_seconds)
    return payload


def match_events(
    games: pd.DataFrame,
    events: list[dict[str, Any]],
    tolerance_minutes: int,
) -> dict[str, dict[str, Any]]:
    normalized = []
    for event in events:
        home = _team_id(event.get("home_team"))
        away = _team_id(event.get("away_team"))
        commence = pd.to_datetime(
            event.get("commence_time"), utc=True, errors="coerce"
        )
        if home and away and pd.notna(commence):
            normalized.append((event, home, away, commence))

    output: dict[str, dict[str, Any]] = {}
    for game in games.itertuples(index=False):
        candidates = [
            item for item in normalized
            if item[1] == game.home_team_id and item[2] == game.away_team_id
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: abs(
                (item[3] - game.kickoff_utc).total_seconds()
            )
        )
        event, _, _, commence = candidates[0]
        delta = abs((commence - game.kickoff_utc).total_seconds()) / 60.0
        if delta <= tolerance_minutes:
            output[str(game.game_id)] = event
    return output


def flatten_event(
    game: pd.Series,
    horizon: str,
    requested: pd.Timestamp,
    returned: str,
    event: dict[str, Any],
) -> list[dict[str, object]]:
    home_id = _team_id(event.get("home_team"))
    away_id = _team_id(event.get("away_team"))
    rows: list[dict[str, object]] = []
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            if market_key not in {"h2h", "spreads", "totals"}:
                continue
            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", ""))
                if market_key in {"h2h", "spreads"}:
                    team = _team_id(name)
                    if team == home_id:
                        outcome_key = "home"
                    elif team == away_id:
                        outcome_key = "away"
                    elif name.lower() in {"draw", "tie"}:
                        outcome_key = "tie"
                    else:
                        continue
                else:
                    outcome_key = name.lower()
                    if outcome_key not in {"over", "under"}:
                        continue
                price = float(outcome["price"])
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "season": int(game["season"]),
                        "week": game["week"],
                        "kickoff_utc": game["kickoff_utc"],
                        "horizon": horizon,
                        "requested_snapshot_utc": requested,
                        "returned_snapshot_utc": returned,
                        "event_id": event.get("id"),
                        "bookmaker_key": bookmaker.get("key"),
                        "bookmaker_title": bookmaker.get("title"),
                        "bookmaker_last_update": bookmaker.get("last_update"),
                        "market": market_key,
                        "market_last_update": market.get("last_update"),
                        "outcome_key": outcome_key,
                        "price_decimal": price,
                        "price_american": decimal_to_american(price),
                        "point": outcome.get("point", np.nan),
                    }
                )
    return rows


def _no_vig(a: float, b: float) -> tuple[float, float, float]:
    raw_a, raw_b = 1.0 / float(a), 1.0 / float(b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total, total - 1.0


def build_consensus(quotes: pd.DataFrame) -> pd.DataFrame:
    book_rows: list[dict[str, object]] = []
    group_cols = [
        "game_id", "season", "week", "horizon",
        "requested_snapshot_utc", "returned_snapshot_utc",
        "bookmaker_key", "market",
    ]
    for keys, frame in quotes.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        outcomes = {row.outcome_key: row for row in frame.itertuples()}
        market = base["market"]
        if market == "h2h" and {"home", "away"}.issubset(outcomes):
            home_p, away_p, hold = _no_vig(
                outcomes["home"].price_decimal,
                outcomes["away"].price_decimal,
            )
            line = np.nan
        elif market == "spreads" and {"home", "away"}.issubset(outcomes):
            home_p, away_p, hold = _no_vig(
                outcomes["home"].price_decimal,
                outcomes["away"].price_decimal,
            )
            line = float(outcomes["home"].point)
        elif market == "totals" and {"over", "under"}.issubset(outcomes):
            home_p, away_p, hold = _no_vig(
                outcomes["over"].price_decimal,
                outcomes["under"].price_decimal,
            )
            line = float(outcomes["over"].point)
        else:
            continue
        book_rows.append(
            {
                **base,
                "line": line,
                "home_or_over_novig_probability": home_p,
                "away_or_under_novig_probability": away_p,
                "hold": hold,
            }
        )

    books = pd.DataFrame(book_rows)
    rows = []
    aggregate_cols = [
        "game_id", "season", "week", "horizon",
        "requested_snapshot_utc", "returned_snapshot_utc", "market",
    ]
    for keys, frame in books.groupby(aggregate_cols, sort=False):
        line = pd.to_numeric(frame["line"], errors="coerce")
        prob = pd.to_numeric(
            frame["home_or_over_novig_probability"], errors="coerce"
        )
        rows.append(
            {
                **dict(zip(aggregate_cols, keys)),
                "eligible_books": int(frame["bookmaker_key"].nunique()),
                "consensus_line": (
                    float(line.median()) if line.notna().any() else np.nan
                ),
                "line_sd": (
                    float(line.std(ddof=0)) if line.notna().any() else np.nan
                ),
                "consensus_home_or_over_novig_probability": float(
                    prob.median()
                ),
                "probability_sd": float(prob.std(ddof=0)),
                "median_hold": float(frame["hold"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_movement_features(
    consensus: pd.DataFrame,
    minimum_books: int,
) -> pd.DataFrame:
    order = {
        "opening_7d": 0, "opening_72h": 1, "opening_24h": 2,
        "opening_6h": 3, "opening_60m": 4, "closing_t10": 5,
    }
    frame = consensus[consensus["eligible_books"] >= minimum_books].copy()
    frame["order"] = frame["horizon"].map(order)
    rows = []
    for (game_id, market), group in frame.groupby(
        ["game_id", "market"], sort=False
    ):
        closing = group[group["horizon"] == "closing_t10"]
        if closing.empty:
            continue
        close = closing.iloc[-1]
        opening = group[group["horizon"] != "closing_t10"].sort_values(
            "order"
        )
        open_row = None if opening.empty else opening.iloc[0]
        rows.append(
            {
                "game_id": game_id,
                "season": int(close["season"]),
                "week": close["week"],
                "market": market,
                "closing_snapshot_utc": close["returned_snapshot_utc"],
                "closing_eligible_books": int(close["eligible_books"]),
                "closing_line": close["consensus_line"],
                "closing_home_or_over_novig_probability": close[
                    "consensus_home_or_over_novig_probability"
                ],
                "closing_line_sd": close["line_sd"],
                "closing_probability_sd": close["probability_sd"],
                "opening_available": open_row is not None,
                "opening_horizon": (
                    None if open_row is None else open_row["horizon"]
                ),
                "opening_line": (
                    np.nan if open_row is None else open_row["consensus_line"]
                ),
                "opening_home_or_over_novig_probability": (
                    np.nan if open_row is None
                    else open_row[
                        "consensus_home_or_over_novig_probability"
                    ]
                ),
                "line_movement": (
                    np.nan if open_row is None
                    else close["consensus_line"] - open_row["consensus_line"]
                ),
                "probability_movement": (
                    np.nan if open_row is None
                    else close[
                        "consensus_home_or_over_novig_probability"
                    ] - open_row[
                        "consensus_home_or_over_novig_probability"
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_backfill(
    games_path: str | Path,
    output_root: str | Path,
    config: BackfillConfig,
) -> dict[str, object]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    games = normalize_games(pd.read_parquet(games_path))
    plan = build_snapshot_plan(games, config)
    plan.to_parquet(output / "historical_snapshot_plan.parquet", index=False)

    unique = sorted(plan["requested_snapshot_utc"].unique())
    summary: dict[str, object] = {
        "seasons": list(config.seasons),
        "games": int(plan["game_id"].nunique()),
        "game_horizon_rows": int(len(plan)),
        "unique_api_requests": len(unique),
        "estimated_credits": len(unique) * config.estimated_cost_per_request,
        "plan_only": config.plan_only,
    }
    if config.plan_only:
        (output / "historical_backfill_plan.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    if (
        config.maximum_requests is not None
        and len(unique) > config.maximum_requests
    ):
        raise RuntimeError(
            f"Plan needs {len(unique)} requests; limit is "
            f"{config.maximum_requests}."
        )

    quote_rows, audit_rows, unmatched_rows = [], [], []
    cache_root = output / "raw_snapshots"
    for number, timestamp in enumerate(unique, start=1):
        ts = pd.Timestamp(timestamp)
        request_iso = ts.isoformat().replace("+00:00", "Z")
        payload = fetch_snapshot(
            request_iso,
            cache_root,
            minimum_interval_seconds=config.minimum_interval_seconds,
        )
        response = payload["response"]
        group = plan[plan["requested_snapshot_utc"] == timestamp]
        matches = match_events(
            group,
            response.get("data", []),
            config.kickoff_tolerance_minutes,
        )
        returned = response["timestamp"]

        for game_id, game_group in group.groupby("game_id", sort=False):
            game = game_group.iloc[0]
            event = matches.get(str(game_id))
            if event is None:
                for horizon in game_group["horizon"]:
                    unmatched_rows.append(
                        {
                            "game_id": game_id,
                            "season": int(game["season"]),
                            "week": game["week"],
                            "horizon": horizon,
                            "requested_snapshot_utc": timestamp,
                            "reason": "EVENT_NOT_FOUND_AT_SNAPSHOT",
                        }
                    )
                continue
            for horizon in game_group["horizon"]:
                quote_rows.extend(
                    flatten_event(game, horizon, ts, returned, event)
                )

        audit_rows.append(
            {
                "request_number": number,
                "requested_snapshot_utc": timestamp,
                "returned_snapshot_utc": returned,
                "events_returned": len(response.get("data", [])),
                "games_planned": int(group["game_id"].nunique()),
                "games_matched": len(matches),
                "request_cost": payload.get("request_cost"),
                "requests_remaining": payload.get("requests_remaining"),
            }
        )
        if number % 25 == 0:
            print(
                f"requests={number}/{len(unique)} "
                f"remaining={payload.get('requests_remaining')}"
            )

    quotes = pd.DataFrame(quote_rows)
    audit = pd.DataFrame(audit_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    quotes.to_parquet(output / "bookmaker_quotes.parquet", index=False)
    audit.to_csv(output / "snapshot_request_audit.csv", index=False)
    unmatched.to_csv(output / "unmatched_game_snapshots.csv", index=False)

    consensus = build_consensus(quotes)
    movement = build_movement_features(consensus, config.minimum_books)
    consensus.to_parquet(output / "consensus_by_horizon.parquet", index=False)
    movement.to_parquet(
        output / "opening_closing_market_features.parquet", index=False
    )

    summary.update(
        {
            "quote_rows": int(len(quotes)),
            "consensus_rows": int(len(consensus)),
            "movement_rows": int(len(movement)),
            "unmatched_rows": int(len(unmatched)),
            "actual_credits_recorded": int(
                pd.to_numeric(audit["request_cost"], errors="coerce")
                .fillna(0).sum()
            ),
        }
    )
    (output / "historical_backfill_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
