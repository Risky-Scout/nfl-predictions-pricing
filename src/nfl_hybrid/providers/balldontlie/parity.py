"""Historical/live feature-family parity matrix (Sections 16, 18, 25).

This is the enforceable artifact Fix 5 exists to produce: for every
important data/feature family, whether the CURRENTLY WIRED historical
pipeline (Fix 2/3's ``build_authoritative_pregame_state`` -- see
``nfl_hybrid.features.pregame_state``) has a live BDL equivalent, and
whether that equivalent has been validated closely enough to be treated as
production-eligible. Nothing here selects a feature set; it only records
what is/isn't safe to select later (Section 25) -- see
:func:`assert_production_eligible`, which any later feature-selection code
should call before treating a BDL-sourced column as a candidate.

Classifications are evidence-based, not guessed. A real 2025 overlap
backtest was actually run 2026-08-22 (``NFL_MODEL_DATA_ROOT`` + a live
``BALLDONTLIE_API_KEY``, 10-game non-cherry-picked stratified sample --
``scripts/bdl_overlap_backtest.py``, results in
``outputs/bdl_overlap_backtest_2025.json``):

- EXACT / VALIDATED_TRANSFORM entries below are grounded in either (a) a
  direct field-for-field mapping to what the wired historical builder
  already consumes (Elo/game result), confirmed by the measured overlap
  run, or (b) an explicit note of what validation would still be required.
- APPROXIMATE_NOT_APPROVED entries are families where the overlap run
  either found a genuine, unresolved numeric disagreement (even a small
  one), or compared the wrong/no historical counterpart for what the
  CURRENTLY WIRED historical feature builder actually reads (Section 25:
  schema similarity alone never justifies a promotion) -- see each entry's
  ``validation_evidence`` for the specific measured finding. Notably: BDL's
  box-score-style counting fields (completions, yards, TDs, INTs, sacks,
  penalties, turnovers, ...) matched nflverse's own published box-score
  tables essentially exactly, but several of nflverse's WIRED PBP-mask-
  derived features (``aggregate_qb_game_efficiency`` /
  ``aggregate_advanced_team_game``) do not themselves agree with the
  official box score on pass-attempt counts -- a genuine gap between two
  nflverse representations of the same game, not a BDL data-quality issue,
  but one that blocks calling BDL a validated substitute for those specific
  wired fields until it is reconciled.
- UNAVAILABLE entries are families where BDL's schema was inspected live
  (2026-08-21) and simply does not contain the required field or its
  provider-native equivalent uses a materially different, non-substitutable
  model (Section 16: never relabel a different metric as nflverse EPA), or
  where no historical counterpart exists to overlap-test against at all
  (e.g. roster_depth).
"""
from __future__ import annotations

from dataclasses import dataclass

EXACT = "EXACT"
VALIDATED_TRANSFORM = "VALIDATED_TRANSFORM"
APPROXIMATE_NOT_APPROVED = "APPROXIMATE_NOT_APPROVED"
UNAVAILABLE = "UNAVAILABLE"

_PRODUCTION_ELIGIBLE_STATUSES = frozenset({EXACT, VALIDATED_TRANSFORM})


@dataclass(frozen=True)
class ParityEntry:
    feature_family: str
    historical_source: str
    historical_definition: str
    live_source: str
    live_definition: str
    parity_status: str
    validation_evidence: str

    @property
    def production_eligible(self) -> bool:
        return self.parity_status in _PRODUCTION_ELIGIBLE_STATUSES


PARITY_MATRIX: tuple[ParityEntry, ...] = (
    ParityEntry(
        feature_family="game_result",
        historical_source="nflverse schedules (backfill.games / backfill.nflverse_games)",
        historical_definition="home_score/away_score as published on the final schedule row",
        live_source="BDL GET /games",
        live_definition="home_team_score/visitor_team_score once status_state == 'final'",
        parity_status=EXACT,
        validation_evidence=(
            "2025 overlap backtest actually run 2026-08-22 (scripts/bdl_overlap_backtest.py, "
            "NFL_MODEL_DATA_ROOT + live BALLDONTLIE_API_KEY, see outputs/bdl_overlap_backtest_2025.json): "
            "10-game non-cherry-picked stratified sample spanning weeks {1,2,4,6,8,10,11,13,15,17} "
            "and 16 distinct teams -- 10/10 games matched by season/week/home/away identity, 10/10 "
            "home_score exact, 10/10 away_score exact, max kickoff-timestamp diff 0.0 minutes. Same "
            "box-score-final definition as the historical source; no transform involved."
        ),
    ),
    ParityEntry(
        feature_family="elo_inputs",
        historical_source="nflverse schedules via build_elo_pregame_state(games) "
        "(nfl_hybrid.features.elo_state) -- game_id, season, week, kickoff, teams, scores only",
        historical_definition="Elo state depends on nothing but game identity + final score",
        live_source="canonical.normalize_games() output (this Fix)",
        live_definition="Identical field set (game_id, season, week, kickoff_utc, home_team, "
        "away_team, home_score, away_score), gated on FINAL status_state",
        parity_status=EXACT,
        validation_evidence="Field-for-field mapping in canonical.py::GAMES_COLUMNS covers every "
        "column build_elo_pregame_state reads; same measured 10-game 2026-08-22 overlap evidence "
        "as game_result (identity + score are the only inputs Elo state depends on).",
    ),
    ParityEntry(
        feature_family="team_box_official",
        historical_source="nflverse official team-game box stats (backfill.team_stats) -- "
        "registered but NOT currently consumed by the wired Fix 2/3 pregame-state pipeline",
        historical_definition="Official league box-score aggregates (yards, first downs, "
        "third/fourth-down efficiency, drives, possession time)",
        live_source="BDL GET /team_stats",
        live_definition="Official BDL box-score aggregates, same category set "
        "(verified live 2026-08-21, game 423945 -- see canonical.py _TEAM_STATS_NUMERIC_FIELDS)",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence=(
            "2025 overlap backtest actually run 2026-08-22 against nflverse's own team_stats table "
            "(the correct historical source for this family -- it is registered but not PBP-derived "
            "in the wired pipeline): 19 team-games compared across the 10-game sample. 15 of 16 "
            "compared fields showed 100% exact agreement (passing_completions, passing_attempts, "
            "rushing_yards, rushing_attempts, sacks, penalties, penalty_yards, interceptions_thrown, "
            "fumbles_lost, first_downs_passing, first_downs_rushing, turnovers[derived], "
            "net_passing_yards[derived], total_yards[derived]) -- net_passing_yards/total_yards only "
            "matched once a REAL sign-convention difference was found and corrected: historical "
            "sack_yards_lost is signed negative, BDL reports the unsigned magnitude (confirmed live, "
            "e.g. 2025_01_ARI_NO ARI: historical=-33, BDL=33; 19/19 exact once corrected). "
            "defensive_touchdowns did NOT validate: 18/19 exact, one unexplained disagreement "
            "(2025_10_ARI_SEA, SEA: historical def_tds=0, BDL defensive_touchdowns=2) with no "
            "identified root cause. Because this one field's disagreement is unresolved and this "
            "family isn't yet wired into production, the family-level status stays conservative "
            "(APPROXIMATE_NOT_APPROVED) despite the other 15 fields validating cleanly -- see "
            "outputs/bdl_overlap_backtest_2025.json 'team_box' for full per-field stats."
        ),
    ),
    ParityEntry(
        feature_family="team_box_pbp_derived (play volume / rates)",
        historical_source="aggregate_advanced_team_game(nflverse PBP) -- "
        "nfl_hybrid.features.pbp_advanced -- scrimmage_plays, dropbacks, pass/rush attempts, "
        "third/fourth-down attempts, explosive rates, completion_rate, air_yards_per_attempt, "
        "sack/turnover counts, drive counts (THIS is the family actually wired into the "
        "'epa' pregame-state family alongside its EPA-dependent metrics)",
        historical_definition="Play-level masks (down/distance/play-type/score-differential) "
        "applied to nflverse PBP rows, then counted/rated per team-game",
        live_source="BDL GET /plays (down/distance/play-type fields exist) -- NO transform built",
        live_definition="n/a -- not implemented in Fix 5",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence="BDL /plays exposes start_down/start_distance/start_yards_to_endzone/"
        "type_slug, structurally similar to nflverse's down/ydstogo/yardline_100/play_type, but "
        "BDL's type_slug vocabulary (verified live: e.g. 'kickoff') has not been mapped to "
        "pbp_advanced.py's play-type masks, and no such recount transform was built or validated "
        "in Fix 5 (Section 17: only adapt the subset with proven parity; Section 25: do not force "
        "an unvalidated family into eligibility). 2025 overlap backtest actually run 2026-08-22 "
        "additionally measured that BDL's canonical play COUNT itself does not agree 1:1 with "
        "nflverse's PBP row count for the same game: across all 10 sampled games BDL consistently "
        "returned MORE canonical plays than nflverse (e.g. 2025_01_ARI_NO: 195 BDL vs. 182 nflverse; "
        "2025_17_DAL_WAS: 198 vs. 187; mean excess ~10 plays/game), even though pagination, dedup, "
        "ordering, and terminal-score reconciliation were all clean (0 anomalies, 10/10 games "
        "reconciled -- see 'pbp' section of outputs/bdl_overlap_backtest_2025.json). Any nflverse "
        "PBP-mask-derived count/rate feature ported onto BDL /plays would need its own row-count "
        "reconciliation before being validated, not just a play-count mapping.",
    ),
    ParityEntry(
        feature_family="qb_player_stats_boxscore",
        historical_source="aggregate_qb_game_efficiency(nflverse PBP) -- attempts, completions, "
        "yards, TDs, INTs, sacks (non-EPA counting fields)",
        historical_definition="Counted directly from PBP rows grouped by passer_player_id",
        live_source="BDL GET /stats (passing_completions, passing_attempts, passing_yards, "
        "passing_touchdowns, passing_interceptions, sacks, ...)",
        live_definition="Official BDL per-player box totals (verified live 2026-08-21)",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence=(
            "2025 overlap backtest actually run 2026-08-22 (10-game sample). Player identity "
            "coverage by normalized name+team+game: 184/192 (95.8%) active offensive skill-position "
            "rows matched, 23/23 (100%) QB rows matched -- no gsis_id/pfr_id crosswalk exists, "
            "matching was by name (see canonical.PLAYER_IDENTITY_PARITY = 'PARTIAL'). completions, "
            "passing_yards, passing_touchdowns, passing_interceptions, and sacks_suffered were "
            "100% exact across BDL vs. nflverse's own player_stats table for all 20-24 comparable "
            "QB rows. HOWEVER: comparing against the ACTUALLY WIRED historical feature builder "
            "(aggregate_qb_game_efficiency, which reads raw PBP directly, not the player_stats "
            "table) revealed 'attempts' does NOT match: the wired PBP 'pass_attempt' mask counted "
            "MORE attempts than the official box score (which BDL matches exactly) in 18 of 19 "
            "spot-checked QB-games (e.g. Kyler Murray 2025_01_ARI_NO: PBP-mask=34, box/BDL=29; "
            "Dak Prescott 2025_17_DAL_WAS: PBP-mask=43, box/BDL=37) -- completions/yards/TDs/INTs/"
            "sacks agreed across all three sources in every case, only 'attempts' diverges. This is "
            "a genuine definitional gap between nflverse's own two representations of the same "
            "game (raw PBP pass_attempt flag vs. published box score), not a BDL data-quality issue "
            "-- but it means BDL cannot be blessed as a drop-in replacement for the wired feature's "
            "'attempts' field without first reconciling that gap, so the family stays "
            "APPROXIMATE_NOT_APPROVED as a whole (Section 25: do not force an unvalidated family "
            "into eligibility merely because most of its fields validated)."
        ),
    ),
    ParityEntry(
        feature_family="epa_success_cpoe (team + QB)",
        historical_source="nflverse PBP columns 'epa', 'success', 'cpoe', 'wpa' -- consumed "
        "directly (not recomputed) by aggregate_advanced_team_game and "
        "aggregate_qb_game_efficiency, and by DEFAULT_QB_METRICS in qb_pregame.py",
        historical_definition="nflverse's own proprietary expected-points/win-probability/"
        "completion-probability models, applied per play",
        live_source="none",
        live_definition="BDL /plays has NO epa, success, cpoe, or wpa field. Verified live "
        "2026-08-21 against the full raw field list for game 423945: id, game, type_slug, "
        "type_abbreviation, type_text, text, short_text, away_score, home_score, scoring_play, "
        "period, clock_display, team, start_yard_line, start_down, start_distance, "
        "start_yards_to_endzone, end_yard_line, end_down, end_distance, end_yards_to_endzone, "
        "end_down_distance_text, end_short_down_distance_text, end_possession_text, "
        "stat_yardage, home_win_probability, wallclock -- none of those is an EPA/success/"
        "CPOE/WPA equivalent. 'home_win_probability' IS present but is a distinct, "
        "provider-native win-probability figure with no validated transform to nflverse's wpa "
        "(Section 16 explicitly forbids relabeling it as one). GOAT-tier "
        "/advanced_stats/passing does expose a differently-modeled "
        "'completion_percentage_above_expectation' (a CPOE-shaped metric from a different "
        "model) -- inspected per Section 2 but deliberately NOT wired into any production path.",
        parity_status=UNAVAILABLE,
        validation_evidence="No EPA-equivalent field exists anywhere in the BDL NFL schema "
        "surfaces checked (plays, advanced_stats/rushing, advanced_stats/passing). This repo "
        "has no independent expected-points model to fall back on, and Fix 5 is explicitly "
        "prohibited from building one (Section 16).",
    ),
    ParityEntry(
        feature_family="success_rate",
        historical_source="nflverse PBP 'success' column (tied to nflverse's own EPA-based "
        "success definition)",
        historical_definition="Per-play boolean success flag from nflverse's model",
        live_source="none",
        live_definition="No success flag or equivalent threshold-based indicator on BDL /plays",
        parity_status=UNAVAILABLE,
        validation_evidence="Same field audit as epa_success_cpoe. A different, "
        "independently-defined down/distance/yards-gained success heuristic could in principle "
        "be built from BDL's start_down/start_distance/stat_yardage fields, but Section 16 "
        "explicitly forbids substituting a differently-defined success metric for nflverse's "
        "own -- that would need to be a separate, later, explicitly-approved definition change, "
        "not a Fix 5 substitution.",
    ),
    ParityEntry(
        feature_family="pass_rush_efficiency (yards/play, completion%, ypc -- non-EPA)",
        historical_source="aggregate_advanced_team_game -- completion_rate, "
        "air_yards_per_attempt, yards_per_play-style rates computed from PBP",
        historical_definition="Simple rate stats over PBP-derived counts",
        live_source="BDL /team_stats (yards_per_play, yards_per_pass, yards_per_rush_attempt) "
        "and /plays (stat_yardage per play)",
        live_definition="Official BDL rate fields (verified live 2026-08-21)",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence=(
            "2025 overlap backtest actually run 2026-08-22. This family's historical source is "
            "aggregate_advanced_team_game's PBP-mask-derived rate fields (offense_completion_rate, "
            "offense_air_yards_per_attempt, yards-per-play-style rates), whose denominators are "
            "built from the same 'pass_attempt' PBP mask found (see qb_player_stats_boxscore "
            "evidence) to disagree with box-score/BDL attempts in 18 of 19 spot-checked team-games "
            "(offense_pass_attempts vs. box-score attempts: only 1/19 exact match). Any BDL-sourced "
            "rate computed with BDL's own (box-score-matching) attempt count would therefore use a "
            "systematically different denominator than the wired feature -- not a compatible "
            "substitute without first reconciling that denominator definition."
        ),
    ),
    ParityEntry(
        feature_family="turnovers",
        historical_source="aggregate_advanced_team_game -- interception/fumble_lost counts "
        "from PBP flags",
        historical_definition="Counted from nflverse's interception/fumble_lost play flags",
        live_source="BDL /team_stats (turnovers, fumbles_lost, interceptions_thrown) and "
        "/stats (fumbles_lost, defensive_interceptions)",
        live_definition="Official BDL box-count fields (verified live 2026-08-21)",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence=(
            "2025 overlap backtest actually run 2026-08-22. BDL turnovers (interceptions_thrown + "
            "fumbles_lost) matched nflverse's own box-score turnovers 19/19 exact. But the ACTUALLY "
            "WIRED historical feature (aggregate_advanced_team_game's offense_turnovers, PBP-mask "
            "derived) matched the box-score figure only 18/19 in the same sample -- one unresolved "
            "disagreement (2025_11_TB_BUF, BUF: offense_turnovers=2 vs. box=3, "
            "offense_fumbles_lost=0 vs. box fumbles_lost_total=1). This is close (94.7% three-way "
            "agreement) but a real, unexplained discrepancy was found in a 19-row sample, and "
            "Section 25 requires a conservative outcome -- staying APPROXIMATE_NOT_APPROVED rather "
            "than rounding a single-digit-percent unexplained gap up to VALIDATED_TRANSFORM."
        ),
    ),
    ParityEntry(
        feature_family="play_volume (drives, pace, total plays)",
        historical_source="aggregate_advanced_team_game._drive_metrics -- drive counts/rates "
        "keyed off nflverse's own 'drive' column",
        historical_definition="nflverse-numbered drive identifiers grouped per team",
        live_source="BDL /team_stats total_drives, total_offensive_plays",
        live_definition="Official BDL per-game aggregate counts (verified live 2026-08-21)",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence="BDL /plays has no explicit drive-number field at all (no 'drive' "
        "equivalent observed in the verified field list), so a play-level drive_metrics parity "
        "would need drive boundaries inferred from possession changes -- an unvalidated "
        "transform, not built in Fix 5. The team_stats-level total_drives aggregate exists but "
        "is, again, unvalidated against the historical estate (2025 overlap backtest run "
        "2026-08-22 did not compare total_drives/total_offensive_plays -- no matching historical "
        "field was available to compare against; see qb_player_stats_boxscore/pass_rush_efficiency "
        "evidence for a related finding that the wired PBP-mask attempt/play counts do not agree "
        "with box-score-style counts in this same sample, which would also affect any play-count "
        "feature in this family).",
    ),
    ParityEntry(
        feature_family="roster_depth",
        historical_source="nflverse weekly rosters/depth charts (backfill.weekly_rosters, "
        "backfill.depth_charts) -- a genuine historical time series",
        historical_definition="Roster/depth state as of each historical week",
        live_source="BDL GET /teams/{id}/roster",
        live_definition="CURRENT depth-chart state only (verified live 2026-08-21: no as-of/"
        "historical parameter beyond 'season'; response is a live snapshot)",
        parity_status=UNAVAILABLE,
        validation_evidence="BDL's roster endpoint is not source-compatible with historical "
        "reconstruction -- it cannot answer 'what was team X's depth chart as of week 3, "
        "2024'. It is a sound LIVE 2026-forward pregame snapshot source (Section 13) but has "
        "no historical-parity counterpart to validate against, so it stays UNAVAILABLE for "
        "parity purposes specifically (not a statement that live roster ingestion itself is "
        "broken).",
    ),
    ParityEntry(
        feature_family="injuries",
        historical_source="nflverse injury reports (backfill.injuries) -- includes "
        "practice-participation status (DNP/Limited/Full) and game-status designation",
        historical_definition="Official weekly injury report fields, richer vocabulary",
        live_source="BDL GET /player_injuries",
        live_definition="{player, status, comment (free text), date} only -- verified live "
        "2026-08-21; no practice-participation field of any kind",
        parity_status=APPROXIMATE_NOT_APPROVED,
        validation_evidence="Same broad concept (player health/availability status) but a "
        "materially narrower field set than the historical estate -- no practice-participation "
        "status is available to map at all (Section 14 explicitly forbids inferring one). "
        "'status' free text (e.g. 'Questionable') is plausibly comparable to nflverse's "
        "game-status designation but no vocabulary-mapping validation was run.",
    ),
)


def by_family(feature_family: str) -> ParityEntry:
    for entry in PARITY_MATRIX:
        if entry.feature_family == feature_family:
            return entry
    raise KeyError(f"No parity entry for feature family: {feature_family!r}")


def production_eligible_families() -> tuple[str, ...]:
    return tuple(e.feature_family for e in PARITY_MATRIX if e.production_eligible)


def non_eligible_families() -> tuple[str, ...]:
    return tuple(e.feature_family for e in PARITY_MATRIX if not e.production_eligible)


class ParityIneligibleError(ValueError):
    """Raised when code tries to treat a non-eligible family as production-ready."""


def assert_production_eligible(feature_family: str) -> ParityEntry:
    """Fail closed: raises unless the family's parity_status is EXACT or
    VALIDATED_TRANSFORM. Any later feature-selection code that wants to draw
    a BDL-sourced column into the production feature set should call this
    first (Section 25) rather than re-deriving its own eligibility opinion."""
    entry = by_family(feature_family)
    if not entry.production_eligible:
        raise ParityIneligibleError(
            f"{feature_family!r} is not production-eligible (parity_status="
            f"{entry.parity_status!r}). See docs/BDL_PARITY_MATRIX.md."
        )
    return entry


def as_records() -> list[dict[str, object]]:
    """Machine-readable form (see scripts/export_bdl_parity_matrix.py)."""
    return [
        {
            "feature_family": e.feature_family,
            "historical_source": e.historical_source,
            "historical_definition": e.historical_definition,
            "live_source": e.live_source,
            "live_definition": e.live_definition,
            "parity_status": e.parity_status,
            "validation_evidence": e.validation_evidence,
            "production_eligible": e.production_eligible,
        }
        for e in PARITY_MATRIX
    ]
