"""Generic pregame-state lineage primitives, shared by every state family.

These functions compute *lineage* only -- which prior games generated a given
row's state -- never the state values themselves. They are deliberately
state-family-agnostic so each family (EPA, opponent-adjusted, QB, Elo, market
history) reuses the same lineage code instead of re-deriving its own notion of
"what did this depend on".

Two lineage shapes are distinguished, matching how the underlying builders
actually compute state:

- :func:`sequential_prior_game_lineage` -- for a family whose value at each row
  depends only on that same entity's own strictly-earlier rows (the
  shift-then-roll team builders, and the sequential Elo state machine). Direct,
  exactly enumerable lineage.
- :func:`snapshot_prior_game_lineage` -- for a family fit from every league
  row strictly before a snapshot date (the opponent-adjustment and QB ridge
  regressions, which pool every team's history, not just the target team's).
  Lineage here is leaguewide by construction; the full id list is capped for
  size and the row is marked truncated rather than silently dropped.
"""
from __future__ import annotations

import pandas as pd

LINEAGE_COLUMNS = (
    "state_family",
    "game_id",
    "team_id",
    "target_kickoff_utc",
    "contributing_game_ids",
    "contributing_game_count",
    "contributing_ids_truncated",
    "latest_contributing_kickoff_utc",
    "contributing_scope",
    "transform_name",
    "transform_version",
)


def sequential_prior_game_lineage(
    long_frame: pd.DataFrame,
    *,
    state_family: str,
    transform_name: str,
    transform_version: str,
    entity_col: str = "team_id",
    game_id_col: str = "game_id",
    event_time_col: str = "event_time",
    window: int | None = None,
) -> pd.DataFrame:
    """Direct lineage: each row's contributors are that same entity's own
    strictly-earlier rows (all of them, or the last ``window`` if the state
    family is itself windowed).

    ``long_frame`` must already carry one row per (entity, game) with a
    parseable ``event_time_col`` -- this computes lineage only, never state.
    """
    required = {entity_col, game_id_col, event_time_col}
    missing = required - set(long_frame.columns)
    if missing:
        raise ValueError(f"Lineage input missing columns: {sorted(missing)}")

    work = long_frame[[entity_col, game_id_col, event_time_col]].copy()
    work[game_id_col] = work[game_id_col].astype(str)
    work[event_time_col] = pd.to_datetime(work[event_time_col], utc=True, errors="coerce")
    if work[event_time_col].isna().any():
        raise ValueError("Lineage input has unparseable event times.")
    if work.duplicated([entity_col, game_id_col]).any():
        raise ValueError("Lineage input has duplicate (entity, game) rows.")

    work = work.sort_values([entity_col, event_time_col, game_id_col], kind="stable").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for entity, frame in work.groupby(entity_col, sort=False):
        frame = frame.sort_values([event_time_col, game_id_col], kind="stable").reset_index(drop=True)
        ids = frame[game_id_col].tolist()
        times = frame[event_time_col].tolist()
        for i in range(len(frame)):
            prior_ids = ids[:i]
            prior_times = times[:i]
            if window is not None:
                prior_ids = prior_ids[-window:]
                prior_times = prior_times[-window:]
            rows.append(
                {
                    "state_family": state_family,
                    "game_id": ids[i],
                    "team_id": entity,
                    "target_kickoff_utc": times[i],
                    "contributing_game_ids": tuple(prior_ids),
                    "contributing_game_count": len(prior_ids),
                    "contributing_ids_truncated": False,
                    "latest_contributing_kickoff_utc": max(prior_times) if prior_times else pd.NaT,
                    "contributing_scope": "team_direct",
                    "transform_name": transform_name,
                    "transform_version": transform_version,
                }
            )
    return pd.DataFrame(rows, columns=list(LINEAGE_COLUMNS))


def snapshot_prior_game_lineage(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    state_family: str,
    transform_name: str,
    transform_version: str,
    game_id_col: str = "game_id",
    team_col: str = "team_id",
    snapshot_col: str = "snapshot_date",
    event_time_col: str = "event_time",
    max_ids_stored: int = 250,
) -> pd.DataFrame:
    """Leaguewide lineage: every row sharing a snapshot date is fit from the
    same "every history row strictly before this snapshot" set. When that set
    exceeds ``max_ids_stored`` the id list is stored empty and
    ``contributing_ids_truncated`` is set -- the count and time bound are never
    truncated, only the enumerated id list.
    """
    rows: list[dict[str, object]] = []
    for snapshot_date, current in candidates.groupby(snapshot_col, sort=True):
        prior = history.loc[history[snapshot_col] < snapshot_date]
        unique_ids = sorted(prior[game_id_col].astype(str).unique().tolist())
        truncated = len(unique_ids) > max_ids_stored
        stored_ids = () if truncated else tuple(unique_ids)
        latest = prior[event_time_col].max() if len(prior) else pd.NaT
        for row in current.itertuples(index=False):
            rows.append(
                {
                    "state_family": state_family,
                    "game_id": str(getattr(row, game_id_col)),
                    "team_id": getattr(row, team_col),
                    "target_kickoff_utc": getattr(row, event_time_col),
                    "contributing_game_ids": stored_ids,
                    "contributing_game_count": len(unique_ids),
                    "contributing_ids_truncated": truncated,
                    "latest_contributing_kickoff_utc": latest,
                    "contributing_scope": "leaguewide_prior_to_snapshot",
                    "transform_name": transform_name,
                    "transform_version": transform_version,
                }
            )
    return pd.DataFrame(rows, columns=list(LINEAGE_COLUMNS))


def verify_pregame_state_lineage(lineage: pd.DataFrame, games: pd.DataFrame) -> None:
    """Invariants every state-family lineage table must satisfy:

    - a game never lists itself as its own contributor;
    - every referenced game_id is a known game_id (no phantom lineage);
    - no contributing game's kickoff is at or after the target game's kickoff
      -- this excludes both the target game and any future game, independent
      of and in addition to the id-based self-reference check above.

    Rows whose id list was truncated (see :func:`snapshot_prior_game_lineage`)
    skip the id-level checks -- there is nothing enumerated to check -- but the
    count/latest-kickoff fields are never truncated, so callers who need a hard
    future-exclusion guarantee on a truncated family should additionally check
    ``latest_contributing_kickoff_utc < target_kickoff_utc`` directly.
    """
    kickoff = pd.to_datetime(games["scheduled_kickoff_utc"], utc=True, errors="coerce")
    kickoff_by_id = dict(zip(games["game_id"].astype(str), kickoff))

    for row in lineage.itertuples(index=False):
        gid = str(row.game_id)
        target_kickoff = kickoff_by_id.get(gid)
        latest_contrib = row.latest_contributing_kickoff_utc
        if pd.notna(latest_contrib) and pd.notna(target_kickoff) and latest_contrib >= target_kickoff:
            raise ValueError(
                f"{row.state_family}: game {gid} (kickoff {target_kickoff}) has a contributing "
                f"game at or after its own kickoff ({latest_contrib})."
            )
        if row.contributing_ids_truncated:
            continue
        contributing = row.contributing_game_ids
        if gid in contributing:
            raise ValueError(f"{row.state_family}: game {gid} lists itself as a contributing prior game.")
        unknown = sorted(set(contributing) - set(kickoff_by_id))
        if unknown:
            raise ValueError(f"{row.state_family}: game {gid} lineage references unknown game_id(s): {unknown}")
        if pd.isna(target_kickoff):
            continue
        for cid in contributing:
            c_kickoff = kickoff_by_id.get(cid)
            if pd.notna(c_kickoff) and c_kickoff >= target_kickoff:
                raise ValueError(
                    f"{row.state_family}: game {gid} (kickoff {target_kickoff}) lists contributing "
                    f"game {cid} (kickoff {c_kickoff}) which is not strictly before it."
                )
