from __future__ import annotations

import pandas as pd


def assert_available_before(
    frame: pd.DataFrame,
    *,
    available_at_column: str,
    prediction_time_column: str,
    allow_equal: bool = True,
) -> None:
    available = pd.to_datetime(frame[available_at_column], utc=True, errors="coerce")
    prediction = pd.to_datetime(frame[prediction_time_column], utc=True, errors="coerce")
    valid = available <= prediction if allow_equal else available < prediction
    violation = available.notna() & prediction.notna() & ~valid
    if violation.any():
        sample = frame.loc[
            violation,
            [available_at_column, prediction_time_column],
        ].head(10)
        raise ValueError(
            f"{int(violation.sum())} rows violate as-of availability:\n{sample}"
        )


def add_postgame_available_at(
    games: pd.DataFrame,
    *,
    kickoff_column: str = "scheduled_kickoff_utc",
    duration_hours: float = 5.0,
) -> pd.Series:
    kickoff = pd.to_datetime(games[kickoff_column], utc=True, errors="coerce")
    return kickoff + pd.Timedelta(hours=float(duration_hours))
