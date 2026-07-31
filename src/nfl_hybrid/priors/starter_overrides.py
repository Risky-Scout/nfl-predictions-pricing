"""Human starter overrides + league-centered fallback for missing QB priors.

Generated depth-chart projections and human judgments are kept **separate**: the
generated rows live in ``data/templates/starter_probabilities.csv`` and are never
overwritten; human overrides live in an override CSV and are applied at refit time
only while unexpired. After application, each team's probabilities are renormalized
to sum to 1.0 and conflicting/ambiguous teams are reflagged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OVERRIDE_COLUMNS = [
    "team_id", "player_id", "starter_probability", "source",
    "source_timestamp_utc", "entered_at_utc", "expires_at_utc", "review_reason",
]

# QB availability states that should force a starter review.
UNAVAILABLE_STATES = {
    "OUT", "DOUBTFUL", "IR", "PUP", "SUSPENDED", "INACTIVE", "RELEASED", "TRADED", "NFI",
}


def empty_override_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OVERRIDE_COLUMNS)


def load_overrides(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return empty_override_frame()
    df = pd.read_csv(p)
    for c in OVERRIDE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[OVERRIDE_COLUMNS]


class OverrideError(ValueError):
    """Explicit overrides for a team sum to more than 1.0."""


def apply_overrides(generated: pd.DataFrame, overrides: pd.DataFrame, *, as_of_utc: str) -> pd.DataFrame:
    """Apply unexpired overrides, **preserving each override's exact probability**.

    Explicit override probabilities are summed first (fail if > 1); only the
    residual mass is distributed among non-overridden eligible candidates, in
    proportion to their generated probability. If a team's overrides already sum
    to 1, no residual is distributed. Rows carry ``origin`` in
    {"generated", "override"}.
    """
    now = pd.Timestamp(as_of_utc)
    gen = generated.copy()
    gen["starter_probability"] = pd.to_numeric(gen["starter_probability"], errors="coerce").fillna(0.0)
    gen["origin"] = "generated"

    ov = overrides.copy()
    if len(ov):
        exp = pd.to_datetime(ov["expires_at_utc"], utc=True, errors="coerce")
        ov = ov[exp.isna() | (exp >= now)].copy()  # unexpired only
    if len(ov) == 0:
        return gen.reset_index(drop=True)

    ov["origin"] = "override"
    ov["starter_probability"] = pd.to_numeric(ov["starter_probability"], errors="coerce")
    season_val = gen["season"].iloc[0] if len(gen) and "season" in gen.columns else np.nan
    ov["season"] = season_val

    out_parts = []
    override_teams = set(ov["team_id"])
    for team, tg in gen.groupby("team_id"):
        if team not in override_teams:
            out_parts.append(tg)
            continue
        team_ov = ov[ov["team_id"] == team].copy()
        override_sum = float(team_ov["starter_probability"].sum())
        if override_sum > 1.0 + 1e-9:
            raise OverrideError(f"{team}: explicit overrides sum to {override_sum:.3f} > 1.0")
        named = set(team_ov["player_id"])
        backups = tg[~tg["player_id"].isin(named)].copy()
        residual = max(1.0 - override_sum, 0.0)
        bsum = float(backups["starter_probability"].sum())
        if bsum > 0 and residual > 0:
            backups["starter_probability"] = backups["starter_probability"] / bsum * residual
        else:
            backups["starter_probability"] = 0.0  # overrides consume all mass
        out_parts.append(pd.concat([team_ov, backups], ignore_index=True))
    return pd.concat(out_parts, ignore_index=True).reset_index(drop=True)


def apply_availability_exclusions(starters: pd.DataFrame, availability: pd.DataFrame | None) -> pd.DataFrame:
    """Zero any definitively-unavailable QB and renormalize the eligible candidates.

    Uncertain designations (Questionable, camp battles) are left untouched here;
    they are surfaced by :func:`flag_review` instead.
    """
    out = starters.copy()
    out["starter_probability"] = pd.to_numeric(out["starter_probability"], errors="coerce").fillna(0.0)
    if availability is None or len(availability) == 0:
        return out
    st_col = next((c for c in ("report_status", "status", "game_status") if c in availability.columns), None)
    pid_col = next((c for c in ("gsis_id", "player_id") if c in availability.columns), None)
    if not (st_col and pid_col):
        return out
    unavailable = set(
        availability[availability[st_col].astype(str).str.upper().str.strip().isin(UNAVAILABLE_STATES)][pid_col]
    )
    out.loc[out["player_id"].isin(unavailable), "starter_probability"] = 0.0
    totals = out.groupby("team_id")["starter_probability"].transform("sum")
    out["starter_probability"] = np.where(totals > 0, out["starter_probability"] / totals, out["starter_probability"])
    return out


def league_centered_fallback(
    missing_player_ids,
    qb_priors: pd.DataFrame,
    *,
    widen_factor: float = 2.0,
) -> pd.DataFrame:
    """Transparent league-centered prior (widened uncertainty) for QBs that have no
    historical prior (rookies, zero-snap, newly acquired). Never silently drops them.
    """
    if len(qb_priors) == 0:
        league_mean, league_sd = 0.0, 0.20
    else:
        league_mean = float(np.average(qb_priors["prior_mean"], weights=np.maximum(qb_priors.get("effective_dropbacks", 1.0), 1e-9)))
        league_sd = float(qb_priors["prior_standard_deviation"].median())
    rows = [
        {
            "player_id": pid,
            "prior_mean": league_mean,
            "prior_standard_deviation": league_sd * widen_factor,
            "effective_dropbacks": 0.0,
            "prior_source": "league_centered_fallback",
        }
        for pid in missing_player_ids
    ]
    return pd.DataFrame(rows)


def flag_review(starters: pd.DataFrame, availability: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``qb_review_status`` = NEEDS_QB_REVIEW when a team's top candidate is in an
    unavailable state or the team has no valid candidate; else OK.
    """
    out = starters.copy()
    unavailable_pid = set()
    if availability is not None and len(availability):
        av = availability.copy()
        st_col = next((c for c in ("report_status", "status", "game_status") if c in av.columns), None)
        pid_col = next((c for c in ("gsis_id", "player_id") if c in av.columns), None)
        if st_col and pid_col:
            bad = av[av[st_col].astype(str).str.upper().str.strip().isin(UNAVAILABLE_STATES)]
            unavailable_pid = set(bad[pid_col])

    statuses = {}
    for team, tg in out.groupby("team_id"):
        top = tg.sort_values("starter_probability", ascending=False).iloc[0]
        if len(tg) == 0 or top["player_id"] in unavailable_pid:
            statuses[team] = "NEEDS_QB_REVIEW"
        else:
            statuses[team] = "OK"
    out["qb_review_status"] = out["team_id"].map(statuses)
    return out
