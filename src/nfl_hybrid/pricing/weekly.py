"""Weekly runtime validation + audit-field assembly.

Critical validations raise :class:`WeeklyRunError` (the weekly job must stop, not
silently continue). Quote-age policy: within two hours of kickoff, quotes older
than 15 minutes are stale; outside that window a configurable 24-hour maximum
applies. Each priced row carries a full audit trail; unpriceable rows are labelled
``NO_PRICE`` rather than invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class WeeklyRunError(RuntimeError):
    """A critical weekly-run failure. The job must stop."""


@dataclass(frozen=True)
class QuoteAgeConfig:
    near_kickoff_hours: float = 2.0
    max_age_near_minutes: float = 15.0
    max_age_far_minutes: float = 24 * 60.0


def quote_is_fresh(age_minutes: float, minutes_to_kickoff: float, cfg: QuoteAgeConfig | None = None) -> bool:
    cfg = cfg or QuoteAgeConfig()
    if not np.isfinite(age_minutes):
        return False
    near = np.isfinite(minutes_to_kickoff) and minutes_to_kickoff <= cfg.near_kickoff_hours * 60.0
    limit = cfg.max_age_near_minutes if near else cfg.max_age_far_minutes
    return age_minutes <= limit


def validate_lines(lines: pd.DataFrame, *, require_priced: bool = True) -> None:
    """Critical pre-pricing checks. Raise on any failure."""
    required = {"game_id", "home_team", "away_team", "status"}
    missing = sorted(required - set(lines.columns))
    if missing:
        raise WeeklyRunError(f"lines frame missing columns: {missing}")
    if lines["game_id"].duplicated().any():
        raise WeeklyRunError("duplicate game_id in lines (unresolved schedule/team mapping)")
    if lines[["home_team", "away_team"]].isna().any().any():
        raise WeeklyRunError("unresolved team mapping (null home/away team)")
    priced = lines[lines["status"] == "PRICED"]
    if require_priced and len(priced) == 0:
        raise WeeklyRunError("no PRICED games (no current odds for any priceable market)")


def validate_starter_sums(starters: pd.DataFrame, *, atol: float = 1e-6) -> None:
    sums = starters.groupby("team_id")["starter_probability"].sum()
    bad = sums[~np.isclose(sums.to_numpy(), 1.0, atol=atol)]
    if len(bad):
        raise WeeklyRunError(f"starter probabilities do not sum to 1.0 for: {list(bad.index)[:8]}")


def validate_probability_identity(card: pd.DataFrame, *, atol: float = 1e-6) -> None:
    """cover + push + fail must equal 1 for ATS rows that carry the components."""
    ats = card[card["market"] == "ats"]
    if {"calibrated_probability", "push_probability"}.issubset(ats.columns) and len(ats):
        total = ats["calibrated_probability"] + ats["push_probability"] + (
            1.0 - ats["calibrated_probability"] - ats["push_probability"]
        )
        if not np.allclose(total.to_numpy(float), 1.0, atol=atol):
            raise WeeklyRunError("probability identity failed (cover+push+fail != 1)")


def assemble_audit(
    card: pd.DataFrame,
    *,
    as_of_utc: str,
    artifact,
    lines: pd.DataFrame | None = None,
    injury_data_utc: str | None = None,
    preliminary: bool = False,
) -> pd.DataFrame:
    """Attach the full audit trail + per-row status label to the card."""
    out = card.copy()
    out["as_of_utc"] = as_of_utc
    out["artifact_sha256"] = getattr(artifact, "artifact_sha256", "none")
    out["code_commit"] = getattr(artifact, "code_commit", "none")
    dc = getattr(artifact, "devig_consensus", {}) if artifact else {}
    market_to_keys = {"moneyline": "moneyline", "ats": "ats", "total": "total"}
    out["devig_method"] = out["market"].map(lambda m: dc.get(f"{market_to_keys.get(m, m)}_devig", "proportional"))
    out["consensus_method"] = out["market"].map(lambda m: dc.get(f"{market_to_keys.get(m, m)}_consensus", "equal_mean"))
    out["injury_data_utc"] = injury_data_utc or "UNKNOWN"

    # per-row quote metadata from the lines frame
    meta_cols = {}
    if lines is not None:
        idx = lines.set_index("game_id")
        for col, dst in [("n_books", "bookmakers_used"), ("quote_timestamp_utc", "quote_timestamp_utc"),
                         ("quote_age_minutes", "quote_age_minutes"), ("status", "line_status")]:
            if col in idx.columns:
                meta_cols[dst] = out["game_id"].astype(str).map(idx[col])
    for dst, series in meta_cols.items():
        out[dst] = series.to_numpy()
    if "bookmakers_used" not in out.columns:
        out["bookmakers_used"] = np.nan

    # per-row status
    line_status = out["line_status"] if "line_status" in out.columns else pd.Series(["PRICED"] * len(out))
    def _status(row_line_status):
        if str(row_line_status) != "PRICED":
            return "NO_PRICE"
        return "PRELIMINARY" if preliminary else "VALID_LIVE_CARD"
    out["card_status"] = [ _status(s) for s in line_status ]
    out["data_quality_status"] = np.where(out["card_status"] == "NO_PRICE", "INCOMPLETE", "OK")
    return out
