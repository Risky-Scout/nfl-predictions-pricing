"""Populate data/templates/ with REAL 2020-2025 data for the 2026 prior refit.

Everything is derived from the already-pulled nflverse backfill (PBP, team_stats,
snap_counts, weekly_rosters, depth_charts, schedules, players). Every row carries
``as_of_utc`` / ``available_at_utc`` (postgame availability = season end) and a
shared ``source_manifest_hash``. Coordinator/play-caller identities are external
only and are documented, never invented; head coaches come from schedules.

Outputs (long/wide per the prior-engine schemas):
- team_metric_history.csv       (EmpiricalBayesTeamPrior)
- quarterback_history.csv        (QuarterbackPriorBuilder)
- starter_probabilities.csv      (starter_mixture; 2026 rows flagged for review)
- offseason_roster_history.csv   (RosterAdjustmentModel)
- coach_history.csv              (HierarchicalCoachPrior)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.external_data import ENV_VAR, ExternalDataUnavailableError, resolve

TEMPLATES = Path("data/templates")
SEASONS = list(range(2020, 2026))


def _season_available_at(season: int) -> str:
    """Team/QB season stats are fully available after the season ends."""
    return f"{season + 1}-02-15T00:00:00Z"


def _manifest_hash() -> str:
    h = hashlib.sha256()
    for name in ("pbp", "team_stats", "snap_counts", "weekly_rosters", "depth_charts"):
        try:
            p = resolve(f"backfill.manifest_{name}")
        except ExternalDataUnavailableError:
            continue
        h.update(p.read_bytes())
    return h.hexdigest()[:32]


# --------------------------------------------------------------------------- #
# 1. team_metric_history (long format)
# --------------------------------------------------------------------------- #
def build_team_metric_history(pbp: pd.DataFrame, src_hash: str) -> pd.DataFrame:
    p = pbp.copy()
    for c in ["epa", "success", "down", "yardline_100", "qb_dropback", "rush_attempt",
              "pass_attempt", "sack", "interception", "fumble_lost", "special_teams_play"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    scrim = p[p["play_type"].isin(["pass", "run"])].copy()

    rows: list[dict] = []

    def add(team_col, season, metric, mask_frame, value_series, prefix):
        v = value_series[mask_frame]
        v = v[np.isfinite(v)]
        if len(v):
            rows.append(dict(entity_id=team, season=int(season), metric=f"{prefix}{metric}",
                             value=float(v.mean()), sample_size=int(len(v))))

    for (team, season), g in scrim.groupby(["posteam", "season"]):
        if pd.isna(team):
            continue
        dbk = g["qb_dropback"] == 1
        rush = g["rush_attempt"] == 1
        early = g["down"].isin([1, 2])
        late = g["down"].isin([3, 4])
        rz = g["yardline_100"] <= 20
        specs = [
            ("offense_epa_per_play", g["epa"], np.ones(len(g), bool)),
            ("offense_success_rate", g["success"], np.ones(len(g), bool)),
            ("offense_pass_epa", g["epa"], dbk.to_numpy()),
            ("offense_rush_epa", g["epa"], rush.to_numpy()),
            ("offense_early_down_epa", g["epa"], early.to_numpy()),
            ("offense_late_down_epa", g["epa"], late.to_numpy()),
            ("offense_red_zone_epa", g["epa"], rz.to_numpy()),
            ("offense_explosive_rate", (g["epa"] > 1.0).astype(float), np.ones(len(g), bool)),
            ("offense_sack_rate", g["sack"], dbk.to_numpy()),
            ("offense_int_rate", g["interception"], dbk.to_numpy()),
            ("offense_fumble_lost_rate", g["fumble_lost"], np.ones(len(g), bool)),
        ]
        for metric, series, mask in specs:
            s = pd.to_numeric(series, errors="coerce").to_numpy(float)
            m = np.asarray(mask) & np.isfinite(s)
            if m.sum() > 0:
                rows.append(dict(entity_id=team, season=int(season), metric=metric,
                                 value=float(s[m].mean()), sample_size=int(m.sum())))
        # pace: offensive plays per game
        n_games = g["game_id"].nunique()
        rows.append(dict(entity_id=team, season=int(season), metric="offense_pace_plays_per_game",
                         value=float(len(g) / max(n_games, 1)), sample_size=int(len(g))))

    # defense allowed (group by defteam)
    for (team, season), g in scrim.groupby(["defteam", "season"]):
        if pd.isna(team):
            continue
        for metric, series in [("defense_allowed_epa_per_play", g["epa"]),
                               ("defense_allowed_success_rate", g["success"])]:
            s = pd.to_numeric(series, errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if len(s):
                rows.append(dict(entity_id=team, season=int(season), metric=metric,
                                 value=float(s.mean()), sample_size=int(len(s))))

    # special teams EPA (all ST plays)
    if "special_teams_play" in p.columns:
        st = p[p["special_teams_play"] == 1]
        for (team, season), g in st.groupby(["posteam", "season"]):
            if pd.isna(team):
                continue
            s = pd.to_numeric(g["epa"], errors="coerce").to_numpy(float)
            s = s[np.isfinite(s)]
            if len(s):
                rows.append(dict(entity_id=team, season=int(season), metric="special_teams_epa_per_play",
                                 value=float(s.mean()), sample_size=int(len(s))))

    frame = pd.DataFrame(rows)
    frame["available_at_utc"] = frame["season"].map(_season_available_at)
    frame["source_manifest_hash"] = src_hash
    return frame.sort_values(["entity_id", "season", "metric"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. quarterback_history
# --------------------------------------------------------------------------- #
def build_quarterback_history(pbp: pd.DataFrame, players: pd.DataFrame, src_hash: str) -> pd.DataFrame:
    p = pbp.copy()
    p["epa"] = pd.to_numeric(p["epa"], errors="coerce")
    p["qb_dropback"] = pd.to_numeric(p["qb_dropback"], errors="coerce")
    db = p[(p["qb_dropback"] == 1) & p["passer_player_id"].notna()].copy()

    per = db.groupby(["passer_player_id", "season"]).agg(
        value=("epa", "mean"), dropbacks=("epa", "size"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    per = per[per["dropbacks"] >= 20]  # meaningful sample

    # recency_dropbacks = dropbacks thrown in later seasons (staleness measure)
    per = per.sort_values(["player_id", "season"])
    per["recency_dropbacks"] = (
        per.groupby("player_id")["dropbacks"].transform(lambda s: s[::-1].cumsum().shift(1)[::-1])
    ).fillna(0.0)

    # years_experience from players rookie_year (fallback: first season seen)
    rookie = None
    for col in ("rookie_year", "entry_year", "draft_year"):
        if col in players.columns:
            rookie = players[["gsis_id", col]].rename(columns={"gsis_id": "player_id", col: "rookie_year"})
            break
    if rookie is None and "gsis_id" in players.columns:
        rookie = pd.DataFrame({"player_id": [], "rookie_year": []})
    if rookie is not None:
        per = per.merge(rookie, on="player_id", how="left")
        per["years_experience"] = (per["season"] - pd.to_numeric(per["rookie_year"], errors="coerce") + 1)
    else:
        per["years_experience"] = np.nan
    first_seen = per.groupby("player_id")["season"].transform("min")
    per["years_experience"] = per["years_experience"].fillna(per["season"] - first_seen + 1).clip(lower=0)

    per["available_at_utc"] = per["season"].map(_season_available_at)
    per["source_manifest_hash"] = src_hash
    return per[["player_id", "season", "value", "dropbacks", "recency_dropbacks",
                "years_experience", "available_at_utc", "source_manifest_hash"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. starter_probabilities (historical + 2026 projection)
# --------------------------------------------------------------------------- #
def build_starter_probabilities(pbp: pd.DataFrame, depth: pd.DataFrame, src_hash: str) -> pd.DataFrame:
    p = pbp.copy()
    p["qb_dropback"] = pd.to_numeric(p["qb_dropback"], errors="coerce")
    db = p[(p["qb_dropback"] == 1) & p["passer_player_id"].notna()]
    # historical: dominant passer per (team, season) = starter, prob by dropback share
    share = db.groupby(["posteam", "season", "passer_player_id"]).size().reset_index(name="db")
    share["total"] = share.groupby(["posteam", "season"])["db"].transform("sum")
    share["starter_probability"] = share["db"] / share["total"]
    hist = share[share["starter_probability"] >= 0.10].copy()
    hist = hist.rename(columns={"posteam": "team_id", "passer_player_id": "player_id"})
    hist["as_of_utc"] = hist["season"].map(_season_available_at)
    hist["source"] = "historical_dropback_share"
    hist["needs_human_review"] = False
    hist = hist[["team_id", "player_id", "starter_probability", "as_of_utc", "source", "season", "needs_human_review"]]

    # 2026 projection from the latest available depth chart (rank-based).
    proj = build_projected_starters(depth, target_season=2026, as_of_utc="2026-08-01T00:00:00Z")
    out = pd.concat([hist, proj], ignore_index=True)
    out["source_manifest_hash"] = src_hash
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. offseason_roster_history (returning snap shares by unit + residual target)
# --------------------------------------------------------------------------- #
RANK_WEIGHTS = {1: 0.85, 2: 0.12, 3: 0.03}


def _rank_from_depth(qb: pd.DataFrame) -> pd.Series:
    """Robust integer QB depth rank. Uses depth_team (the populated column),
    falling back to pos_rank, then to first-seen order. Normalizes strings/floats."""
    rank = pd.Series(np.nan, index=qb.index, dtype=float)
    for col in ("depth_team", "pos_rank", "depth_chart_order"):
        if col in qb.columns:
            cand = pd.to_numeric(qb[col], errors="coerce")
            rank = rank.where(rank.notna(), cand)
    return rank


def build_projected_starters(depth: pd.DataFrame, *, target_season: int, as_of_utc: str) -> pd.DataFrame:
    """Rank-based projected starter probabilities for every team from the latest
    depth chart. Ranks are normalized; per-team probabilities sum to 1.0."""
    if depth is None or len(depth) == 0:
        return pd.DataFrame()
    d = depth.copy()
    pos_col = next((c for c in ("position", "pos_abb", "depth_position", "pos") if c in d.columns), None)
    pid_col = next((c for c in ("gsis_id", "player_id", "player_gsis_id") if c in d.columns), None)
    team_col = next((c for c in ("club_code", "team", "recent_team") if c in d.columns), None)
    name_col = next((c for c in ("full_name", "player_name", "football_name") if c in d.columns), None)
    if not (pos_col and pid_col and team_col):
        return pd.DataFrame()
    d["season_num"] = pd.to_numeric(d["season"], errors="coerce")
    d["week_num"] = pd.to_numeric(d.get("week", 0), errors="coerce").fillna(0)
    latest_season = int(d["season_num"].max())
    qb = d[(d[pos_col].astype(str).str.upper() == "QB") & (d["season_num"] == latest_season)].copy()
    if qb.empty:
        return pd.DataFrame()
    # keep each player's most recent week, then their best (lowest) rank
    qb["_rank_raw"] = _rank_from_depth(qb)
    qb = qb.sort_values(["week_num"], ascending=False).drop_duplicates([team_col, pid_col])

    rows = []
    for team, tg in qb.groupby(team_col):
        tg = tg.copy()
        # normalize ranks: dense-rank the raw ranks so 1,2,3 are contiguous even with gaps
        tg["_rank"] = tg["_rank_raw"].rank(method="first").astype(int)
        tg = tg.sort_values("_rank").head(4)
        weights = tg["_rank"].map(lambda r: RANK_WEIGHTS.get(int(r), 0.01)).to_numpy(float)
        if weights.sum() <= 0:
            weights = np.ones(len(tg))
        probs = weights / weights.sum()  # per-team probabilities sum to 1.0
        for (_, r), p in zip(tg.iterrows(), probs):
            rows.append(dict(
                team_id=team, player_id=r[pid_col],
                player_name=r.get(name_col) if name_col else None,
                starter_probability=float(p), rank=int(r["_rank"]),
                as_of_utc=as_of_utc, source="depth_chart_rank",
                season=target_season, needs_human_review=True,
            ))
    return pd.DataFrame(rows)


_UNIT_COLS = {
    "returning_offensive_snap_percentage": "offense_snaps",
    "returning_defensive_snap_percentage": "defense_snaps",
    "returning_special_teams_snap_percentage": "st_snaps",
}


def build_offseason_roster_history(snaps: pd.DataFrame, team_hist: pd.DataFrame, src_hash: str) -> pd.DataFrame:
    s = snaps.copy()
    pid = next((c for c in ("pfr_player_id", "player_id", "gsis_id") if c in s.columns), None)
    team_col = next((c for c in ("team", "club_code", "recent_team") if c in s.columns), None)
    unit_map = {c: s[c] for c in ("offense_snaps", "defense_snaps", "st_snaps") if c in s.columns}
    if not (pid and team_col and unit_map):
        return pd.DataFrame()
    for c in unit_map:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)
    # player-season snaps by unit and team
    agg = s.groupby([pid, team_col, "season"]).agg(
        {c: "sum" for c in unit_map}
    ).reset_index()

    rows = []
    for team in agg[team_col].dropna().unique():
        tg = agg[agg[team_col] == team]
        for target in range(2021, 2026):
            prev = tg[tg["season"] == target - 1]
            cur_players = set(tg[tg["season"] == target][pid])
            if prev.empty or not cur_players:
                continue
            row = dict(team_id=team, target_season=int(target), as_of_utc=f"{target}-08-01T00:00:00Z")
            for out_col, snap_col in _UNIT_COLS.items():
                if snap_col in prev.columns:
                    total = prev[snap_col].sum()
                    ret = prev[prev[pid].isin(cur_players)][snap_col].sum()
                    row[out_col] = float(ret / total) if total > 0 else np.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    # remaining schema columns not derivable from snaps alone -> documented NaN placeholders
    for col in ["returning_qb_dropback_percentage", "returning_receiver_target_percentage",
                "returning_rushing_attempt_percentage", "returning_offensive_line_snap_percentage",
                "returning_pass_rush_snap_percentage", "returning_secondary_snap_percentage",
                "starter_changes_offense", "starter_changes_defense"]:
        if col not in frame.columns:
            frame[col] = np.nan

    # target: early-season (target season) offense EPA residual vs prior-season offense EPA
    off = team_hist[team_hist["metric"] == "offense_epa_per_play"][["entity_id", "season", "value"]]
    off = off.rename(columns={"entity_id": "team_id"})
    frame = frame.merge(off.rename(columns={"season": "target_season", "value": "cur_off"}),
                        on=["team_id", "target_season"], how="left")
    prev_off = off.copy(); prev_off["target_season"] = prev_off["season"] + 1
    frame = frame.merge(prev_off[["team_id", "target_season", "value"]].rename(columns={"value": "prev_off"}),
                        on=["team_id", "target_season"], how="left")
    frame["early_season_performance_residual"] = frame["cur_off"] - frame["prev_off"]
    frame = frame.drop(columns=["cur_off", "prev_off"])
    frame["source_manifest_hash"] = src_hash
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. coach_history (head coaches from schedules; unit = team-level residual)
# --------------------------------------------------------------------------- #
def build_coach_history(schedules: pd.DataFrame, team_hist: pd.DataFrame, src_hash: str) -> pd.DataFrame:
    hc = next((c for c in ("home_coach", "home_head_coach") if c in schedules.columns), None)
    ac = next((c for c in ("away_coach", "away_head_coach") if c in schedules.columns), None)
    if not (hc and ac):
        return pd.DataFrame()
    home_team = next((c for c in ("home_team", "home_team_id") if c in schedules.columns), None)
    away_team = next((c for c in ("away_team", "away_team_id") if c in schedules.columns), None)
    if not (home_team and away_team):
        return pd.DataFrame()
    off = team_hist[team_hist["metric"] == "offense_epa_per_play"].set_index(["entity_id", "season"])["value"]
    # aggregate at coach-season-team level to residual vs league mean
    long = []
    for side_coach, side_team in [(hc, home_team), (ac, away_team)]:
        tmp = schedules[["season", side_coach, side_team]].dropna()
        tmp.columns = ["season", "coach_id", "team_id"]
        long.append(tmp)
    cs = pd.concat(long, ignore_index=True).drop_duplicates(["season", "coach_id", "team_id"])
    cs = cs.merge(off.rename("off_epa"), left_on=["team_id", "season"], right_index=True, how="left")
    league = cs.groupby("season")["off_epa"].transform("mean")
    cs["residual_value"] = cs["off_epa"] - league
    games = schedules.groupby([ "season"]).size()  # not per coach; approximate below
    # games per coach-season
    gpc = pd.concat([
        schedules[["season", hc]].rename(columns={hc: "coach_id"}),
        schedules[["season", ac]].rename(columns={ac: "coach_id"}),
    ], ignore_index=True).dropna().groupby(["season", "coach_id"]).size().reset_index(name="games")
    cs = cs.merge(gpc, on=["season", "coach_id"], how="left")
    cs["unit"] = "offense"
    cs = cs.dropna(subset=["residual_value"])
    cs["source_manifest_hash"] = src_hash
    return cs[["coach_id", "unit", "residual_value", "games", "season", "team_id", "source_manifest_hash"]].reset_index(drop=True)


def main() -> None:
    import json
    src_hash = _manifest_hash()
    pbp = pd.read_parquet(resolve("backfill.pbp"))
    team_stats = pd.read_parquet(resolve("backfill.team_stats"))
    snaps = pd.read_parquet(resolve("backfill.snap_counts"))
    depth = pd.read_parquet(resolve("backfill.depth_charts"))
    games = pd.read_parquet(resolve("backfill.nflverse_games"))
    from nfl_hybrid.data.providers.nflverse import NflverseAdapter
    try:
        players = NflverseAdapter().load_players().data
    except Exception:
        players = pd.DataFrame()

    TEMPLATES.mkdir(parents=True, exist_ok=True)
    team_hist = build_team_metric_history(pbp, src_hash)
    team_hist.to_csv(TEMPLATES / "team_metric_history.csv", index=False)
    print(f"team_metric_history: {len(team_hist)} rows, {team_hist['metric'].nunique()} metrics, seasons {sorted(team_hist['season'].unique())}")

    qb = build_quarterback_history(pbp, players, src_hash)
    qb.to_csv(TEMPLATES / "quarterback_history.csv", index=False)
    print(f"quarterback_history: {len(qb)} rows, {qb['player_id'].nunique()} QBs")

    starters = build_starter_probabilities(pbp, depth, src_hash)
    starters.to_csv(TEMPLATES / "starter_probabilities.csv", index=False)
    n2026 = int((starters["season"] == 2026).sum())
    print(f"starter_probabilities: {len(starters)} rows ({n2026} projected 2026 rows needing review)")

    roster = build_offseason_roster_history(snaps, team_hist, src_hash)
    roster.to_csv(TEMPLATES / "offseason_roster_history.csv", index=False)
    print(f"offseason_roster_history: {len(roster)} rows")

    coach = build_coach_history(games, team_hist, src_hash)
    if len(coach):
        coach.to_csv(TEMPLATES / "coach_history.csv", index=False)
    print(f"coach_history: {len(coach)} rows, {coach['coach_id'].nunique() if len(coach) else 0} coaches")

    (TEMPLATES / "PROVENANCE.json").write_text(json.dumps({
        "source_manifest_hash": src_hash,
        "seasons": SEASONS,
        "generated_from": f"{ENV_VAR}/backfill-2020-2025 (nflverse PBP/team_stats/snaps/depth_charts/schedules)",
        "note": "Coordinators/play-callers are external-only and not included; head coaches from schedules.",
    }, indent=2))
    print("wrote data/templates/PROVENANCE.json")


if __name__ == "__main__":
    main()
