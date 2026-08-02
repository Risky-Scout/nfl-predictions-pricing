"""Augmented pregame feature matrix: market lines + game-level EPA matchup.

This is an *orchestration* over the existing builders, not a parallel feature
stack:

    aggregate_advanced_team_game (postgame team-game facts)
        -> build_team_pregame_features (leakage-safe shift-then-roll)
        -> build_game_pregame_matrix (home/away pivot)

From the wide pregame pivot it selects a curated set of base EPA metrics and
constructs opponent-adjusted matchup differentials (home offense vs away
defense, and the reverse), rolling 4-game and season-to-date, plus rest-days /
short-week from the schedule. These EPA features are merged **alongside** the
market features (``home_spread``, ``total_line``) -- augmenting, never replacing.

Leakage is enforced by the underlying builder (each metric is shifted one game
before any rolling/expanding/EWMA calc) and verified explicitly by
:func:`leakage_report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_hybrid.data.availability import add_postgame_available_at, assert_available_before
from nfl_hybrid.features.pbp_advanced import aggregate_advanced_team_game
from nfl_hybrid.features.pregame_rolling import (
    PregameRollingConfig,
    build_game_pregame_matrix,
    build_team_pregame_features,
)

# Curated base metrics (kept small on purpose: multiple-comparison discipline).
BASE_METRICS = (
    "offense_epa_per_play",
    "offense_success_rate",
    "offense_dropback_epa",
    "offense_rush_epa",
    "offense_explosive_pass_rate",
    "defense_allowed_epa_per_play",
    "defense_allowed_success_rate",
)
WINDOWS = ("last4_mean", "season_mean")
MARKET_FEATURES = ("home_spread", "total_line")


def _diff_features(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Opponent-adjusted matchup differentials from home/away pregame columns."""

    out = pd.DataFrame(index=matrix.index)
    names: list[str] = []
    for win in WINDOWS:
        # offense EPA vs opponent defense-allowed EPA (higher = home edge)
        h_off = f"home_offense_epa_per_play__{win}"
        a_def = f"away_defense_allowed_epa_per_play__{win}"
        a_off = f"away_offense_epa_per_play__{win}"
        h_def = f"home_defense_allowed_epa_per_play__{win}"
        if {h_off, a_def, a_off, h_def}.issubset(matrix.columns):
            home_matchup = matrix[h_off] - matrix[a_def]
            away_matchup = matrix[a_off] - matrix[h_def]
            out[f"epa_matchup_home_{win}"] = home_matchup
            out[f"epa_matchup_away_{win}"] = away_matchup
            out[f"epa_net_edge_{win}"] = home_matchup - away_matchup
            names += [
                f"epa_matchup_home_{win}",
                f"epa_matchup_away_{win}",
                f"epa_net_edge_{win}",
            ]
        # success-rate net edge
        h_sr, a_sr = f"home_offense_success_rate__{win}", f"away_offense_success_rate__{win}"
        if {h_sr, a_sr}.issubset(matrix.columns):
            out[f"success_rate_net_{win}"] = matrix[h_sr] - matrix[a_sr]
            names.append(f"success_rate_net_{win}")
        # pass/run efficiency split net edge
        h_db, a_db = f"home_offense_dropback_epa__{win}", f"away_offense_dropback_epa__{win}"
        h_ru, a_ru = f"home_offense_rush_epa__{win}", f"away_offense_rush_epa__{win}"
        if {h_db, a_db}.issubset(matrix.columns):
            out[f"dropback_epa_net_{win}"] = matrix[h_db] - matrix[a_db]
            names.append(f"dropback_epa_net_{win}")
        if {h_ru, a_ru}.issubset(matrix.columns):
            out[f"rush_epa_net_{win}"] = matrix[h_ru] - matrix[a_ru]
            names.append(f"rush_epa_net_{win}")
    return out, names


def _rest_features(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rest-days and short-week flags per team from schedule kickoff times."""

    g = games[["game_id", "season", "home_team_id", "away_team_id", "scheduled_kickoff_utc"]].copy()
    g["kickoff"] = pd.to_datetime(g["scheduled_kickoff_utc"], utc=True, errors="coerce")
    long = pd.concat(
        [
            g[["game_id", "season", "home_team_id", "kickoff"]].rename(
                columns={"home_team_id": "team_id"}
            ).assign(side="home"),
            g[["game_id", "season", "away_team_id", "kickoff"]].rename(
                columns={"away_team_id": "team_id"}
            ).assign(side="away"),
        ]
    )
    long = long.sort_values(["team_id", "kickoff"], kind="stable")
    long["prev_kickoff"] = long.groupby("team_id")["kickoff"].shift(1)
    long["rest_days"] = (long["kickoff"] - long["prev_kickoff"]).dt.total_seconds() / 86400.0
    long["short_week"] = (long["rest_days"] < 6.0).astype("Int64")
    home = long[long["side"] == "home"].set_index("game_id")[["rest_days", "short_week"]]
    away = long[long["side"] == "away"].set_index("game_id")[["rest_days", "short_week"]]
    out = pd.DataFrame(index=games.index)
    out["game_id"] = games["game_id"].to_numpy()
    out["home_rest_days"] = out["game_id"].map(home["rest_days"])
    out["away_rest_days"] = out["game_id"].map(away["rest_days"])
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    out["home_short_week"] = out["game_id"].map(home["short_week"]).astype(float)
    out["away_short_week"] = out["game_id"].map(away["short_week"]).astype(float)
    return out.drop(columns=["game_id"]), [
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_short_week",
        "away_short_week",
    ]


# --- Canonical label policy (single source of truth) ------------------------- #
# Ties and betting pushes are EXCLUDED binary outcomes, never class zero:
#   * a tied game is null for the no-tie moneyline classifier;
#   * an ATS/total push is null for the no-push cover/over classifier.
# The same rows remain valid for continuous margin/total regression, which has no
# tie/push concept. Do NOT re-derive this tie/push handling in other modules;
# reuse ``_edge_to_nullable_binary`` (via ``_outcome_columns``) instead.
_EDGE_TOLERANCE = 1e-9


def _edge_to_nullable_binary(edge: object) -> pd.Series:
    """Convert a numeric decision edge into a nullable binary target (``Int8``).

    ``edge > +tol -> 1``; ``edge < -tol -> 0``; ``|edge| <= tol`` (a tie / push)
    or a non-finite / missing edge -> ``pd.NA``, with ``tol = _EDGE_TOLERANCE``.
    Deliberately NOT a Boolean comparison cast to int: a zero edge must be null,
    not class zero (that is exactly the tie/push mislabelling this repairs).
    """
    s = pd.to_numeric(pd.Series(edge), errors="coerce")
    values = s.to_numpy(dtype="float64")
    out = pd.array([pd.NA] * len(values), dtype="Int8")
    finite = np.isfinite(values)
    out[finite & (values > _EDGE_TOLERANCE)] = 1
    out[finite & (values < -_EDGE_TOLERANCE)] = 0
    return pd.Series(out, index=s.index)


def _outcome_columns(games: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import norm

    g = games.copy()
    g["home_spread"] = pd.to_numeric(g["home_spread_reference"], errors="coerce")
    g["total_line"] = pd.to_numeric(g["total_line_reference"], errors="coerce")
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    # Ties/pushes -> pd.NA (excluded), never class zero. See the label policy above.
    g["home_win"] = _edge_to_nullable_binary(g["home_margin"])
    g["home_cover"] = _edge_to_nullable_binary(g["home_margin"] + g["home_spread"])
    g["over"] = _edge_to_nullable_binary(g["total_points"] - g["total_line"])
    g["legacy_expected_margin"] = -g["home_spread"]
    g["legacy_expected_total"] = g["total_line"]
    g["legacy_home_win_probability"] = 1.0 - norm.cdf((0.0 - (-g["home_spread"])) / 13.5)
    return g


# Exact frozen 19-feature contract (order matters; must match the shadow artifact).
FROZEN_FEATURES = [
    "home_spread", "total_line",
    "epa_matchup_home_last4_mean", "epa_matchup_away_last4_mean", "epa_net_edge_last4_mean",
    "success_rate_net_last4_mean", "dropback_epa_net_last4_mean", "rush_epa_net_last4_mean",
    "epa_matchup_home_season_mean", "epa_matchup_away_season_mean", "epa_net_edge_season_mean",
    "success_rate_net_season_mean", "dropback_epa_net_season_mean", "rush_epa_net_season_mean",
    "home_rest_days", "away_rest_days", "rest_diff", "home_short_week", "away_short_week",
]
_ID_COLS = {"game_id", "team_id", "opponent_id", "season", "week"}


_NONFINAL_STATUSES = {"IN_PROGRESS", "INPROGRESS", "LIVE", "HALFTIME", "DELAYED",
                      "SCHEDULED", "PREGAME", "SUSPENDED", "POSTPONED"}
_FINAL_STATUSES = {"FINAL", "COMPLETED", "CLOSED", "FINAL_OT", "F"}
# The repository's documented postgame-availability policy (data.availability). Used ONLY
# in historical_replay mode as a labelled availability convention -- never as verified
# finality. The value is the existing default (kickoff + 5h).
_HISTORICAL_AVAILABILITY_HOURS = 5.0


def _terminal_play_timestamp(play_by_play, gid):
    """Return a genuine terminal-play UTC timestamp only if the schema clearly proves the
    game ended. This repository's play-by-play has no per-play UTC event time and no
    terminal-game marker, so this returns None (limitation recorded)."""
    return None


def resolve_finality_before_asof(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame | None,
    *,
    as_of_utc: str,
    mode: str = "live",
) -> pd.DataFrame:
    """Resolve, per game, VERIFIED finality vs a historical availability assumption.

    Two distinct concepts are returned and never conflated:
      - ``verified_final_before_asof`` — True only with real evidence observed at/before
        ``as_of``: explicit final status + completion timestamp; explicit final status +
        verified terminal-play timestamp; a schema completion/terminal timestamp; or an
        official final snapshot. Non-null scores or kickoff time alone are NOT evidence.
      - ``historically_available_before_asof`` — set only in ``mode="historical_replay"``
        via the repository's documented postgame-availability policy (kickoff + 5h),
        labelled ``HISTORICAL_AVAILABILITY_ASSUMPTION``. It NEVER sets verified finality
        and is never used in ``live`` mode.

    ``live`` mode fails closed: absent verified evidence -> ``UNKNOWN_FINALITY`` (excluded).
    """
    if mode not in ("live", "historical_replay"):
        raise ValueError("mode must be 'live' or 'historical_replay'")
    now = pd.Timestamp(as_of_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    g = games.copy()
    kickoff = pd.to_datetime(g["scheduled_kickoff_utc"], utc=True, errors="coerce")

    status_col = next((c for c in ("game_status", "status") if c in g.columns), None)
    status = g[status_col].astype(str).str.upper().str.strip() if status_col else None
    comp_col = next((c for c in ("completion_timestamp_utc", "game_end_utc", "end_time_utc") if c in g.columns), None)
    completion = pd.to_datetime(g[comp_col], utc=True, errors="coerce") if comp_col else pd.Series(pd.NaT, index=g.index)

    rows = []
    for i in g.index:
        gid = str(g.loc[i, "game_id"])
        ko = kickoff.loc[i]
        st = status.loc[i] if status_col else None
        comp = completion.loc[i]
        verified = False
        fstatus, fsource, fevidence_ts, reason = "UNKNOWN_FINALITY", "none", pd.NaT, "no_finality_evidence"

        if pd.notna(ko) and ko > now:
            fstatus, fsource, reason = "POST_ASOF_KICKOFF", "kickoff", "kickoff_after_asof"
        elif st is not None and st in _NONFINAL_STATUSES:
            fstatus, fsource, reason = "EXPLICIT_NON_FINAL", "explicit_status", f"status={st}"
        elif st is not None and st in _FINAL_STATUSES and pd.notna(comp):
            if comp <= now:
                verified, fstatus, fsource, fevidence_ts, reason = True, "FINAL_BEFORE_ASOF", "explicit_status_and_completion_timestamp", comp, ""
            else:
                fstatus, fsource, reason = "IN_PROGRESS_AT_ASOF", "explicit_status_and_completion_timestamp", "completion_after_asof"
        elif pd.notna(comp):
            if comp <= now:
                verified, fstatus, fsource, fevidence_ts, reason = True, "FINAL_BEFORE_ASOF", "completion_timestamp", comp, ""
            else:
                fstatus, fsource, reason = "IN_PROGRESS_AT_ASOF", "completion_timestamp", "completion_after_asof"
        else:
            tp = _terminal_play_timestamp(play_by_play, gid)
            if tp is not None and pd.Timestamp(tp) <= now:
                verified, fstatus, fsource, fevidence_ts, reason = True, "FINAL_BEFORE_ASOF", "verified_terminal_play", pd.Timestamp(tp), ""

        # historical availability assumption (replay mode only; never verified finality)
        available = False
        avail_status, avail_source, avail_ts = "", "none", pd.NaT
        if mode == "historical_replay" and not verified and pd.notna(ko) and ko <= now:
            postgame = ko + pd.Timedelta(hours=_HISTORICAL_AVAILABILITY_HOURS)
            if postgame <= now:
                available, avail_status, avail_source, avail_ts = (
                    True, "HISTORICAL_AVAILABILITY_ASSUMPTION", "documented_training_policy", postgame)
                if fstatus == "UNKNOWN_FINALITY":
                    fstatus = "HISTORICAL_AVAILABILITY_ASSUMPTION"

        rows.append({
            "game_id": gid,
            "verified_final_before_asof": verified,
            "historically_available_before_asof": available,
            "finality_status": fstatus,
            "availability_status": avail_status,
            "finality_source": fsource,
            "finality_evidence_timestamp": fevidence_ts,
            "availability_source": avail_source,
            "availability_timestamp": avail_ts,
            "exclusion_reason": reason,
            "scheduled_kickoff_utc": ko,
        })
    return pd.DataFrame(rows)


def build_live_augmented_features(
    historical_games: pd.DataFrame,
    target_games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    as_of_utc: str,
    mode: str = "live",
    rolling_config: PregameRollingConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the frozen 19 pregame features for upcoming ``target_games``.

    Leakage-safe by construction: each upcoming game is inserted into its teams'
    rolling sequences as a metrics-NaN placeholder, and the shared shift-then-roll
    builder uses only *prior* completed games. The target's own (unknown) result and
    play-by-play never enter its features. Returns (features, status_report) with one
    row per target ``game_id`` and columns in exactly ``FROZEN_FEATURES`` order.
    """
    import hashlib
    import json

    cfg = rolling_config or PregameRollingConfig(windows=(4,))
    now = pd.Timestamp(as_of_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    tg = target_games.copy()
    tg["scheduled_kickoff_utc"] = pd.to_datetime(tg["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if tg["game_id"].duplicated().any():
        raise ValueError("duplicate target game_id rows")
    if (tg["scheduled_kickoff_utc"] <= now).any():
        raise ValueError("target game kickoff is at or before as_of_utc")

    # Resolve finality WITHOUT fabricating evidence. In live mode a game contributes only
    # when VERIFIED final before as_of; in historical_replay mode it may instead be included
    # under the documented postgame-availability assumption (labelled, never "verified").
    hg_all = historical_games.copy()
    hg_all["scheduled_kickoff_utc"] = pd.to_datetime(hg_all["scheduled_kickoff_utc"], utc=True, errors="coerce")
    finality = resolve_finality_before_asof(hg_all, play_by_play, as_of_utc=str(now), mode=mode)
    fin_map = finality.set_index(finality["game_id"].astype(str))
    if mode == "historical_replay":
        eligible_mask = fin_map["verified_final_before_asof"] | fin_map["historically_available_before_asof"]
    else:
        eligible_mask = fin_map["verified_final_before_asof"]
    fstat = hg_all["game_id"].astype(str).map(fin_map["finality_status"])
    excluded_in_progress = int((fstat == "IN_PROGRESS_AT_ASOF").sum())
    excluded_unknown = int((fstat == "UNKNOWN_FINALITY").sum())
    excluded_post_asof_games = int((fstat == "POST_ASOF_KICKOFF").sum())
    finality_source_counts = finality["finality_source"].value_counts().to_dict()
    availability_status_counts = finality["availability_status"].replace("", "NONE").value_counts().to_dict()
    finality_exclusion_reason_counts = finality[~eligible_mask.reindex(finality["game_id"].astype(str)).to_numpy()]["exclusion_reason"].value_counts().to_dict()
    eligible_ids = set(fin_map[eligible_mask].index.astype(str))
    hg = hg_all[hg_all["game_id"].astype(str).isin(eligible_ids)].copy()
    excluded_nonfinal = excluded_in_progress + excluded_unknown  # backward-compat aggregate
    max_completion = str(hg["scheduled_kickoff_utc"].max()) if len(hg) else None

    if len(hg) == 0:
        # Fail closed: no eligible prior games -> no EPA/rest features. Market lines still
        # attach; the fundamental scorer will treat the all-NaN EPA rows accordingly.
        import hashlib as _hl
        import json as _js
        lm = tg.assign(game_id=tg["game_id"].astype(str)).set_index("game_id")
        feats = pd.DataFrame({"game_id": tg["game_id"].astype(str).to_numpy()})
        for c in FROZEN_FEATURES:
            feats[c] = np.nan
        feats["home_spread"] = feats["game_id"].map(lm["home_spread"])
        feats["total_line"] = feats["game_id"].map(lm["total_line"])
        feats = feats[["game_id"] + FROZEN_FEATURES]
        fs = feats.sort_values("game_id").reset_index(drop=True)
        payload = {"game_ids": fs["game_id"].tolist(), "columns": FROZEN_FEATURES,
                   "values": [[None if pd.isna(v) else round(float(v), 10) for v in fs[c]] for c in FROZEN_FEATURES]}
        status = {
            "as_of_utc": str(now), "mode": mode, "target_game_count": int(len(tg)),
            "feature_row_count": int(len(feats)), "required_columns": FROZEN_FEATURES,
            "missing_columns": [], "eligible_final_game_count": 0,
            "verified_final_game_count": int(fin_map["verified_final_before_asof"].sum()),
            "historically_available_game_count": int(fin_map["historically_available_before_asof"].sum()),
            "excluded_in_progress_game_count": excluded_in_progress,
            "excluded_unknown_finality_game_count": excluded_unknown,
            "excluded_post_asof_game_count": excluded_post_asof_games,
            "finality_source_counts": finality_source_counts,
            "availability_status_counts": availability_status_counts,
            "feature_snapshot_hash": _hl.sha256(_js.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "no_eligible_completed_games": True,
        }
        return feats, status

    # Play-by-play: only from eligible (final-before-as_of) games; additionally enforce a
    # per-play timestamp cutoff when a play/game date column is available.
    pbp = play_by_play.copy()
    pbp_ids = pbp["game_id"].astype(str)
    excluded_post_asof_plays = int((~pbp_ids.isin(eligible_ids)).sum())
    pbp = pbp[pbp_ids.isin(eligible_ids)].copy()
    ts_col = next((c for c in ("time_of_day", "start_time", "game_date") if c in pbp.columns), None)
    max_play_ts = None
    if ts_col is not None:
        play_ts = pd.to_datetime(pbp[ts_col], utc=True, errors="coerce")
        after = play_ts > now
        excluded_post_asof_plays += int(after.sum())
        pbp = pbp[~after]
        max_play_ts = str(pd.to_datetime(pbp[ts_col], utc=True, errors="coerce").max()) if len(pbp) else None

    team_game = aggregate_advanced_team_game(pbp)
    metric_cols = [c for c in team_game.columns if c not in _ID_COLS]

    # placeholder team-game rows for each target (identifiers only; metrics NaN)
    has_home = "home_team" in team_game.columns
    has_away = "away_team" in team_game.columns
    placeholders = []
    for _, r in tg.iterrows():
        for team, opp in ((r["home_team_id"], r["away_team_id"]), (r["away_team_id"], r["home_team_id"])):
            row = {c: np.nan for c in metric_cols}
            row.update({"game_id": str(r["game_id"]), "team_id": team, "opponent_id": opp,
                        "season": int(r["season"]), "week": r["week"]})
            if has_home:
                row["home_team"] = r["home_team_id"]
            if has_away:
                row["away_team"] = r["away_team_id"]
            placeholders.append(row)
    team_game_all = pd.concat([team_game, pd.DataFrame(placeholders)], ignore_index=True)

    # combined schedule for ordering + rest features
    keep = ["game_id", "season", "week", "home_team_id", "away_team_id", "scheduled_kickoff_utc"]
    all_games = pd.concat([hg[keep], tg[keep]], ignore_index=True)
    all_games["game_id"] = all_games["game_id"].astype(str)
    team_game_all["game_id"] = team_game_all["game_id"].astype(str)

    team_pregame = build_team_pregame_features(team_game_all, all_games, config=cfg)
    pivot = build_game_pregame_matrix(tg.assign(game_id=tg["game_id"].astype(str)), team_pregame)

    diffs, epa_cols = _diff_features(pivot)
    rest, rest_cols = _rest_features(all_games)
    rest = rest.reset_index(drop=True)
    rest["game_id"] = all_games["game_id"].to_numpy()

    # pivot carries game_id; diffs align to pivot rows by position. Drop any market/rest
    # columns the pivot may already carry so the authoritative merges do not collide.
    combined = pd.concat([pivot.reset_index(drop=True), diffs.reset_index(drop=True)], axis=1)
    combined["game_id"] = combined["game_id"].astype(str)
    collide = [c for c in (list(MARKET_FEATURES) + rest_cols) if c in combined.columns]
    combined = combined.drop(columns=collide)

    out = combined.merge(rest[["game_id"] + rest_cols], on="game_id", how="left", validate="one_to_one")
    lines = tg.assign(game_id=tg["game_id"].astype(str))[["game_id", "home_spread", "total_line"]]
    out = out.merge(lines, on="game_id", how="left", validate="one_to_one")

    # exactly one and only one feature row per target game (hard assertion, no dedup)
    out = out.set_index("game_id").reindex(tg["game_id"].astype(str)).reset_index()
    counts = out["game_id"].value_counts()
    if len(out) != len(tg) or (counts != 1).any():
        raise ValueError("live feature assembly did not produce exactly one row per target game")
    missing_columns = [c for c in FROZEN_FEATURES if c not in out.columns]  # captured BEFORE fill
    for col in missing_columns:
        out[col] = np.nan
    features = out[["game_id"] + FROZEN_FEATURES].copy()

    nan_by_feature = {c: int(features[c].isna().sum()) for c in FROZEN_FEATURES}
    # deterministic hash: sort by game_id; include ids, ordered columns, dtypes, values, missingness
    fsorted = features.sort_values("game_id").reset_index(drop=True)
    payload = {
        "game_ids": fsorted["game_id"].astype(str).tolist(),
        "columns": FROZEN_FEATURES,
        "dtypes": [str(fsorted[c].dtype) for c in FROZEN_FEATURES],
        "values": [[None if pd.isna(v) else round(float(v), 10) for v in fsorted[c]] for c in FROZEN_FEATURES],
        "missing": [[bool(m) for m in fsorted[c].isna()] for c in FROZEN_FEATURES],
    }
    snap_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    status = {
        "as_of_utc": str(now),
        "target_game_count": int(len(tg)),
        "feature_row_count": int(len(features)),
        "required_columns": FROZEN_FEATURES,
        "missing_columns": missing_columns,
        "nan_count_by_feature": nan_by_feature,
        "max_eligible_completed_kickoff": str(hg["scheduled_kickoff_utc"].max()) if len(hg) else None,
        "max_eligible_play_timestamp": max_play_ts,
        "mode": mode,
        "eligible_final_game_count": int(len(eligible_ids)),
        "verified_final_game_count": int(fin_map["verified_final_before_asof"].sum()),
        "historically_available_game_count": int(fin_map["historically_available_before_asof"].sum()),
        "excluded_in_progress_game_count": excluded_in_progress,
        "excluded_unknown_finality_game_count": excluded_unknown,
        "excluded_post_asof_game_count": excluded_post_asof_games,
        "excluded_nonfinal_game_count": excluded_nonfinal,
        "excluded_post_asof_play_count": excluded_post_asof_plays,
        "maximum_eligible_completion_timestamp": max_completion,
        "maximum_eligible_play_timestamp": max_play_ts,
        "finality_source_counts": finality_source_counts,
        "availability_status_counts": availability_status_counts,
        "finality_exclusion_reason_counts": finality_exclusion_reason_counts,
        "feature_snapshot_hash": snap_hash,
    }
    return features, status


def build_augmented_feature_matrix(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    rolling_config: PregameRollingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (matrix, feature_manifest) merging market + EPA matchup features."""

    cfg = rolling_config or PregameRollingConfig(windows=(4,))
    team_game = aggregate_advanced_team_game(play_by_play)
    team_pregame = build_team_pregame_features(team_game, games, config=cfg)
    pivot = build_game_pregame_matrix(games, team_pregame)

    diffs, epa_cols = _diff_features(pivot)
    rest, rest_cols = _rest_features(games)
    outcomes = _outcome_columns(games)

    base = outcomes[
        [
            "game_id", "season", "week", "home_team_id", "away_team_id",
            "home_score", "away_score", "home_margin", "total_points",
            "home_win", "home_cover", "over", "home_spread", "total_line",
            "legacy_home_win_probability", "legacy_expected_margin", "legacy_expected_total",
        ]
    ].reset_index(drop=True)

    matrix = pd.concat(
        [base, diffs.reset_index(drop=True), rest.reset_index(drop=True)], axis=1
    )
    # Keep every completed, priced game for continuous margin/total regression. A
    # tie or push only nulls the affected binary target (home_win/home_cover/over
    # stay nullable Int8); it must NOT drop the whole game. Binary consumers filter
    # each target's valid (0/1) rows independently at their own fit/grade sites.
    matrix = matrix.dropna(
        subset=["home_margin", "total_points", "home_spread", "total_line"]
    )

    # explicit integer week ordering (never string-sort)
    week_order = {str(w): w for w in range(1, 19)}
    week_order.update({"Wildcard": 19, "Division": 20, "Conference": 21, "Superbowl": 22})
    matrix["week"] = matrix["week"].astype(str).map(week_order).fillna(23).astype(int)
    matrix = matrix.sort_values(["season", "week"], kind="stable").reset_index(drop=True)
    matrix["game_index"] = np.arange(len(matrix))
    # home_win/home_cover/over remain nullable Int8 (pd.NA marks ties/pushes); do
    # not cast to ordinary int here — that would reintroduce the R1 mislabelling.

    manifest = {
        "market_features": list(MARKET_FEATURES),
        "epa_features": epa_cols,
        "rest_features": rest_cols,
        "all_features": list(MARKET_FEATURES) + epa_cols + rest_cols,
    }
    return matrix, manifest


def leakage_report(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    rolling_config: PregameRollingConfig | None = None,
) -> dict[str, object]:
    """Explicit as-of leakage verification for the EPA pregame features.

    Checks, and returns as a serialisable dict:
    1. first-game NaN: each team's chronologically first game has no rolling
       history (count == 0), proving the current game is excluded;
    2. as-of availability: the most-recent contributing game's postgame
       availability (prev kickoff + 5h) is <= the current game's kickoff, via
       :func:`assert_available_before` (raises on any violation);
    3. future-swap invariance: corrupting each team's *last* game's stats leaves
       every earlier game's pregame features unchanged (no look-ahead).
    """

    cfg = rolling_config or PregameRollingConfig(windows=(4,))
    team_game = aggregate_advanced_team_game(play_by_play)
    team_pregame = build_team_pregame_features(team_game, games, config=cfg)

    # 1. first-game NaN check
    tp = team_pregame.merge(
        games[["game_id", "scheduled_kickoff_utc"]], on="game_id", how="left"
    )
    tp["kickoff"] = pd.to_datetime(tp["scheduled_kickoff_utc"], utc=True, errors="coerce")
    tp = tp.sort_values(["team_id", "kickoff"], kind="stable")
    count_cols = [c for c in tp.columns if c.endswith("__last4_count") or c.endswith("__season_count")]
    first_rows = tp.groupby("team_id", as_index=False).head(1)
    first_game_history = float(
        first_rows[count_cols].fillna(0).to_numpy().sum()
    ) if count_cols else 0.0

    # 2. as-of availability
    tp["prev_kickoff"] = tp.groupby("team_id")["kickoff"].shift(1)
    check = tp.dropna(subset=["prev_kickoff"]).copy()
    check["prev_available_at"] = check["prev_kickoff"] + pd.Timedelta(hours=5.0)
    assert_available_before(
        check,
        available_at_column="prev_available_at",
        prediction_time_column="kickoff",
        allow_equal=False,
    )

    # 3. future-swap invariance
    corrupt = team_game.copy().sort_values(["team_id", "season", "week"], kind="stable")
    metric_cols = [c for c in corrupt.columns if c.startswith(("offense_", "defense_allowed_"))]
    last_idx = corrupt.groupby("team_id").tail(1).index
    corrupt.loc[last_idx, metric_cols] = -999.0
    tp_corrupt = build_team_pregame_features(corrupt, games, config=cfg)
    feature_cols = [c for c in team_pregame.columns if c.endswith(("_mean", "_hl4"))]
    merged = team_pregame.merge(
        tp_corrupt, on=["game_id", "team_id"], suffixes=("_orig", "_corrupt")
    )
    # exclude each team's last game (its own features legitimately unaffected;
    # we only assert earlier games are identical)
    last_games = set(team_game.groupby("team_id").tail(1)["game_id"])
    earlier = merged[~merged["game_id"].isin(last_games)]
    max_abs_diff = 0.0
    for c in feature_cols:
        a, b = earlier[f"{c}_orig"].to_numpy(float), earlier[f"{c}_corrupt"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            max_abs_diff = max(max_abs_diff, float(np.abs(a[both] - b[both]).max()))

    return {
        "first_game_rolling_history_sum": first_game_history,
        "first_game_nan_check_pass": bool(first_game_history == 0.0),
        "as_of_availability_check_pass": True,  # assert_available_before raised otherwise
        "future_swap_max_abs_feature_diff": max_abs_diff,
        "future_swap_invariance_pass": bool(max_abs_diff < 1e-9),
        "n_team_pregame_rows": int(len(team_pregame)),
    }
