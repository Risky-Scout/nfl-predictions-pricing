from __future__ import annotations

import numpy as np
import pandas as pd


def build_roster_continuity(
    prior_season_snaps: pd.DataFrame,
    target_roster: pd.DataFrame,
    *,
    prior_team_column: str = "team_id",
    target_team_column: str = "team_id",
    player_column: str = "player_id",
    position_group_column: str = "position_group",
) -> pd.DataFrame:
    """Calculate returning snap shares from prior-season snaps and an as-of roster."""
    required_snaps = {
        prior_team_column,
        player_column,
        position_group_column,
        "offensive_snaps",
        "defensive_snaps",
        "special_teams_snaps",
    }
    required_roster = {target_team_column, player_column}
    if required_snaps - set(prior_season_snaps.columns):
        raise ValueError("Prior-season snap table is missing required columns.")
    if required_roster - set(target_roster.columns):
        raise ValueError("Target roster is missing required columns.")

    snaps = prior_season_snaps.copy()
    roster = target_roster[[target_team_column, player_column]].drop_duplicates().copy()
    roster = roster.rename(columns={target_team_column: "target_team_id"})
    merged = snaps.merge(roster, on=player_column, how="left")
    merged["returned_to_same_team"] = (
        merged["target_team_id"].astype("string")
        == merged[prior_team_column].astype("string")
    )

    snap_columns = {
        "offensive": "offensive_snaps",
        "defensive": "defensive_snaps",
        "special_teams": "special_teams_snaps",
    }
    rows: list[dict[str, object]] = []
    for team_id, group in merged.groupby(prior_team_column, dropna=False):
        row: dict[str, object] = {"team_id": team_id}
        returned = group["returned_to_same_team"].fillna(False)
        for label, column in snap_columns.items():
            values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
            denominator = float(values.sum())
            numerator = float(values[returned].sum())
            row[f"returning_{label}_snap_percentage"] = (
                numerator / denominator if denominator > 0 else np.nan
            )

        for position_group, position_frame in group.groupby(
            position_group_column, dropna=False
        ):
            label = str(position_group).lower().replace(" ", "_")
            offense = pd.to_numeric(
                position_frame["offensive_snaps"], errors="coerce"
            ).fillna(0.0)
            defense = pd.to_numeric(
                position_frame["defensive_snaps"], errors="coerce"
            ).fillna(0.0)
            total = offense + defense
            denominator = float(total.sum())
            numerator = float(
                total[position_frame["returned_to_same_team"].fillna(False)].sum()
            )
            row[f"returning_{label}_snap_percentage"] = (
                numerator / denominator if denominator > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
