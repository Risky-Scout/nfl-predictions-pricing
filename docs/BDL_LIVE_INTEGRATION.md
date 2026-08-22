# BALLDONTLIE (BDL) 2026 Live Data Integration (Fix 5)

Provider-neutral live-data layer for 2026 NFL games, backed by BALLDONTLIE.
Implements: `BDL raw responses -> provider-specific normalization ->
canonical NFL schemas -> strict finality/completeness gates ->
provider-neutral completed-game state`. Downstream pregame-state builders
(`nfl_hybrid.features.pregame_state`) never need to know whether a
completed game came from the historical nflverse backfill or 2026 BDL --
both paths land on the same canonical shapes.

This document is the provider contract. See `docs/BDL_PARITY_MATRIX.md` /
`docs/BDL_PARITY_MATRIX.json` for which feature families are actually
allowed to be built from that contract.

## Package layout

```
src/nfl_hybrid/providers/balldontlie/
  client.py           authenticated HTTP client, cursor pagination, raw JSON
  team_crosswalk.py   BDL team id/abbreviation -> canonical_team_id (32 franchises)
  canonical.py         raw dicts -> canonical games/team_stats/player_stats/injuries/roster
  plays.py             raw plays -> canonical PBP, dedup/reject rules, completeness proof
  finality.py           status_state finality rule, result_available_at_utc, can_update_* gates
  raw_store.py          NFL_LIVE_DATA_ROOT immutable raw-snapshot provenance
  parity.py             the historical/live parity matrix + production-eligibility guard
```

## Credential

`BALLDONTLIE_API_KEY` (see `.env.example`). Never hardcoded, never
logged/printed (`client.py::_headers` builds the auth header inline and it
is never passed to anything that could print it). All offline tests run
against checked-in fixtures under `tests/fixtures/balldontlie/` and require
no credential. `tests/test_balldontlie_live_smoke.py` runs one small
read-only query ONLY if the env var is set; otherwise it SKIPs cleanly
(`pytest.mark.skipif`), never fails.

## Verified API contract (live, 2026-08-21)

Base URL: `https://api.balldontlie.io/nfl/v1`. Auth header:
`Authorization: <key>` (not a Bearer prefix). Verified live against every
endpoint this integration uses, with a GOAT-tier key: `/games`,
`/games/{id}`, `/team_stats`, `/stats`, `/player_injuries`, `/plays`,
`/teams`, `/teams/{id}/roster`.

### Endpoints used and exact fields consumed

| Endpoint | Client method | Fields consumed |
|---|---|---|
| `GET /games` | `get_games` | `id`, `home_team`/`visitor_team` (`id`, `abbreviation`), `week`, `date`, `season`, `postseason`, `status`, `status_state`, `home_team_score`, `visitor_team_score` |
| `GET /games/{id}` | `get_game` | same as above, single resource |
| `GET /plays` | `get_plays` | `id`, `game.id`, `type_slug`/`type_abbreviation`/`type_text`, `text`/`short_text`, `away_score`/`home_score`, `scoring_play`, `period`, `clock_display`, `team`, `start_down`/`start_distance`/`start_yards_to_endzone`, `end_down`/`end_distance`/`end_yards_to_endzone`, `stat_yardage`, `home_win_probability`, `wallclock` |
| `GET /team_stats` | `get_team_stats` | `game.id`, `team`, `home_away`, and the full box-score field set in `canonical.py::_TEAM_STATS_NUMERIC_FIELDS` |
| `GET /stats` | `get_player_stats` | `game.id`, `player`, `team`, and the field set in `canonical.py::_PLAYER_STATS_NUMERIC_FIELDS` |
| `GET /player_injuries` | `get_player_injuries` | `player` (incl. nested `team`), `status`, `comment`, `date` |
| `GET /teams/{id}/roster` | `get_team_roster` | `player`, `position`, `depth`, `player_name`, `injury_status` |
| `GET /teams` | (team_crosswalk registry source; not called at runtime -- crosswalk is a frozen, hand-verified table) | `id`, `abbreviation`, `full_name` |

Deliberately NOT implemented (Section 4: "do not create API methods for
endpoints Fix 5 does not need"): `/season_stats`, `/team_season_stats`,
`/standings`, `/advanced_stats/*`, `/dfs/*`, `/fantasy/*`, `/odds*`,
`/props*`, `/lineups`, `/box_scores`. `/advanced_stats/rushing` and
`/advanced_stats/passing` were inspected live (Section 2) -- see
`docs/BDL_PARITY_MATRIX.md` "epa_success_cpoe" -- but no client method
calls them.

### Subscription tier

Free tier covers `/teams`, `/players`, `/games`. ALL-STAR adds
`/player_injuries`, `/stats`, `/team_stats`. GOAT adds `/plays`,
`/teams/{id}/roster`, `/advanced_stats/*`. The key used to verify this
integration had GOAT-tier access to every endpoint listed above.

### Verified API discrepancies (Section 2: current API wins over prior assumptions)

1. **`/team_stats` does not honor a `weeks` filter.** A live query with
   `seasons[]=2025&weeks[]=1` returned all 18 weeks of the season
   unfiltered, not just week 1. `BDLClient.get_team_stats` deliberately has
   no `weeks` parameter; filter the returned rows on `row["game"]["week"]`
   or pass `game_ids` instead (which IS honored -- verified separately).
2. **`/games` never says whether a non-postseason game is preseason or
   regular season.** `postseason: true` is unambiguous; for
   `postseason: false` rows, week numbering restarts at 1 for BOTH
   preseason and regular season with no other distinguishing field. The
   distinction exists only in which `season_type` query filter was used to
   fetch the batch. `canonical.normalize_games` requires this as an
   explicit `season_type_hint` parameter and **fails closed**
   (`CanonicalizationError`) when a non-postseason row's hint is missing or
   unrecognized -- it never defaults to `"REG"` or infers season type from
   week number. Proven by
   `test_normalize_games_missing_hint_fails_closed`,
   `test_normalize_games_unrecognized_hint_fails_closed`, and
   `test_normalize_games_preseason_week1_and_regular_week1_not_conflated`
   (the same DAL@PHI week-1 matchup classifies as `PRE` or `REG` purely
   based on which hint is passed, never on week number). Known residual
   limitation: the canonical `game_id` convention
   (`{season}_{week:02d}_{away}_{home}`) does not itself encode
   `season_type`, so a preseason and regular-season game between the same
   two teams in the same numbered week would produce the same `game_id`
   string even though `season_type`/`season_type_basis` correctly differ --
   not exercised in the historical estate (preseason games are not part of
   `backfill.games`), but a caller merging live preseason and regular-season
   BDL batches into one table must key on `(game_id, season_type)`, not
   `game_id` alone.
3. **No `epa`, `wpa`, `success`, or `cpoe` field exists anywhere in the
   `/plays` schema.** See "EPA / advanced feature audit" below.
4. **`/teams/{id}/roster` is a live current-state snapshot with no
   historical as-of parameter** beyond `season` (which appears to select a
   season's roster composition, not a specific week-in-season state).

## Pagination (Section 5)

Cursor-based: response `meta.next_cursor` (absent/`null` when exhausted).
`BDLClient._paginate` follows it to exhaustion, tracks visited cursors and
raises `BDLPaginationError` on a repeated cursor (a provider loop), and
caps at `BDLConfig.max_pages` (default 500) to prevent an unbounded loop on
a broken cursor -- a cap violation also raises, it never silently
truncates. Proven with real two-page fixtures
(`tests/fixtures/balldontlie/games_page1.json` /
`games_page2.json`, `plays_page1.json` / `plays_page2.json`,
`player_injuries_page1.json` / `page2.json`); `test_pagination_exhausts_and_page2_rows_present`
and `test_page2_plays_are_present` assert page-2-only rows are actually
present in the final result, not silently dropped after page 1.

## Team identity crosswalk (Section 6)

`team_crosswalk.py` is a frozen, hand-verified table of all 32 BDL
franchise ids/abbreviations captured from a live `GET /teams` call
(2026-08-21), resolved through the SAME `canonical_team_id` registry Fix
2/Fix 3 already use (`nfl_hybrid.data.team_ids`). The only alias BDL needs
beyond what already existed is `WSH -> WAS`. Fails closed
(`TeamCrosswalkError`) on an unknown provider id, `None`, a duplicate
provider id, or an ambiguous canonical mapping; a completeness test
(`test_crosswalk_completeness_all_32_franchises`) asserts exactly 32
distinct canonical teams.

## Canonical game schema + finality (Sections 7-8)

`canonical.normalize_games` produces the schema Section 7 specifies
(`provider`, `provider_game_id`, `game_id`, `season`, `week`,
`season_type`, `postseason`, `kickoff_utc`, `home_team`, `away_team`,
`home_score`, `away_score`, `status`, `status_state`,
`source_collected_at_utc`, `source_payload_hash`). `game_id` uses the exact
same `{season}_{week:02d}_{away}_{home}` convention the historical
nflverse adapter already uses, so a live game's canonical key lines up
with what a historical row for the same game would use.

**Finality rule** (`finality.is_final`): `status_state == "final"`. Nothing
else. `NON_FINAL_STATUS_STATES` enumerates every other documented value
(`scheduled`, `in_progress`, `postponed`, `canceled`, `delayed`,
`suspended`, `abandoned`, `unknown`); anything not in
`{final} union NON_FINAL_STATUS_STATES` (including a missing value) is
still treated as not-final -- fail closed, never inferred from score
presence, elapsed time, or status text. `tests/test_balldontlie_finality.py`
proves an `in_progress` game with a realistic non-null score
(`games_page2.json`'s LAR@SF fixture, 17-14 in Q3) is still excluded.

## result_available_at_utc (Section 9)

Separate contract from the historical estate's
`compute_result_available_at_utc` in
`nfl_hybrid.evaluation.chronological_oof` (which uses a conservative
kickoff+duration floor snapped to the next Tue/Fri batch boundary -- that
function is untouched by Fix 5 and remains the historical-only rule).

The live rule (`finality.compute_result_available_at_utc`) is: the FIRST
ingestion observation, per game, at which `status_state == "final"`,
computed over an append-only observation log (one row per polling event).
Basis constant: `RESULT_AVAILABILITY_BASIS = "BDL_FIRST_TRUSTED_FINAL_OBSERVATION"`,
persisted alongside the timestamp. This is a genuinely different claim
than "when BDL says the game ended" -- BDL does not publish a final-whistle
timestamp, so this only ever claims "when WE first observed final,"
consistent with Section 9's prohibition on fabricating an exact
final-whistle time. A later poll, even one that reconfirms
`status_state == "final"`, never overwrites the first recorded
observation -- proven by `test_first_final_observation_wins_never_overwritten_by_later_poll`
and `test_idempotent_recomputation_over_growing_log`.

## Family-specific first-usable timestamps (Fix 5 certification, Sections 3-5)

A single game-level `result_available_at_utc` only answers "when did THIS
game first go final" -- it is NOT sufficient to drive historical
Tuesday/Friday replay, because a downstream family's own completeness gate
can pass strictly LATER than plain finality (e.g. a game goes final
Thursday night but PBP completeness doesn't land until Saturday). Using
`result_available_at_utc` as a stand-in for "PBP is usable" would let a
game that finalized Thursday leak PBP-derived state into a Friday cutoff
before PBP was actually complete.

`finality.compute_family_available_at_utc` computes FOUR independent
first-usable timestamps per game from an append-only polling log --
`score_available_at_utc`, `box_available_at_utc`,
`player_stats_available_at_utc`, `pbp_available_at_utc` -- each the first
observation at which that family's own gate (`can_update_score_state` /
`can_update_box_state` / `can_update_player_stats_state` /
`can_update_pbp_state`) passed. Each is independently monotonic and
immutable: a later poll can never move an already-recorded family
timestamp, and one family's gate passing has zero effect on another
family's recorded timestamp.

`finality.family_eligible_as_of(available_at_utc, as_of_utc)` is the shared
Section-4 as-of rule used against any of the four timestamps: eligible iff
the timestamp exists and is `<= as_of_utc`. Identical logic for all four
families -- the family-specific work already happened when the timestamp
was computed.

All five Section-5 scenarios are proven in `tests/test_balldontlie_finality.py`:

- **A** -- game `in_progress` at Friday cutoff -> all four family
  timestamps are `NaT`; `family_eligible_as_of` returns
  `INELIGIBLE_NEVER_OBSERVED_AVAILABLE` for each.
- **B** -- final score complete before Friday -> `score_available_at_utc`
  is eligible Friday.
- **C** -- final score complete before Friday, PBP completes only AFTER
  Friday -> score/Elo eligible Friday, PBP-derived state NOT eligible
  Friday (`INELIGIBLE_NOT_YET_AVAILABLE_AS_OF`).
- **D** -- evaluated as-of the actual PBP-completion timestamp (or later),
  PBP becomes eligible then, not one second earlier
  (`test_scenario_c_and_d_pbp_completing_after_friday_cannot_contaminate_friday_state`
  asserts eligibility one second before the completion timestamp is still
  `False`).
- **E** -- a later poll (even one reconfirming an already-eligible family,
  or newly satisfying a different family) never moves any of the four
  already-recorded family timestamps backward or overwrites them
  (`test_scenario_e_later_poll_never_moves_family_timestamps_backward_or_overwrites`).

## Completeness (Section 10) and gates (Section 23)

Five independent booleans, never collapsed into one:

- `game_final` -- `finality.is_final(status_state)`
- `box_complete` -- `canonical.team_stats_completeness`: exactly one row
  per participating team, both teams matching the canonical game's own
  teams, no duplicates/unexpected teams.
- `player_stats_complete` -- `canonical.player_stats_completeness`: at
  least one player-stat row exists for the game (a floor, not a
  full-roster guarantee -- BDL does not expose a "box score finalized"
  flag independent of the game's own `status_state`).
- `pbp_complete` -- `plays.pbp_completeness`: FINAL + nonempty + unique
  play ids + a clean `PlaysNormalizationReport` (no unresolved
  duplicate-conflict / foreign-game-play anomaly) + terminal-play score
  reconciliation against the canonical game score when reconcilable (a
  `None` reconciliation result -- score not present on any play row -- is
  treated as unavailable-to-check, not a failure, since inventing one
  would be worse than leaving it unknown).
- `roster_snapshot_available` / `injury_snapshot_available` -- simple
  non-emptiness of the respective canonical frame; these are LIVE pregame
  snapshots, never completed-game results (Section 13/14), so they carry
  no finality gate at all.

Gate functions (`finality.can_update_score_state`,
`can_update_box_state`, `can_update_pbp_state`, `can_update_player_stats_state`)
each return a `GateResult(eligible, reason_code)` with the exact reason
codes Section 23 specifies (`ELIGIBLE_FINAL_SCORE`, `ELIGIBLE_BOX_COMPLETE`,
`ELIGIBLE_PBP_COMPLETE`, `ELIGIBLE_PLAYER_STATS_COMPLETE`,
`INELIGIBLE_NOT_FINAL`, `INELIGIBLE_MISSING_SCORE`,
`INELIGIBLE_MISSING_TEAM_STATS`, `INELIGIBLE_PBP_INCOMPLETE`,
`INELIGIBLE_MISSING_PLAYER_STATS`, `INELIGIBLE_UNKNOWN_TEAM`). None of them
fall back to a stale value or another provider on ineligibility -- they
simply report "not yet," and the caller decides what to do with that.

### Tuesday/Friday availability scenarios (Section 24)

All four proven in `tests/test_balldontlie_finality.py`:

- **A** -- Thursday game `in_progress` at Friday cutoff -> `can_update_score_state`
  returns `INELIGIBLE_NOT_FINAL`.
- **B** -- game first observed FINAL with a valid score before Friday
  cutoff -> `ELIGIBLE_FINAL_SCORE`.
- **C** -- game FINAL but PBP incomplete -> score/Elo state IS eligible
  (`can_update_score_state` -> eligible) while PBP/EPA state is NOT
  (`can_update_pbp_state(..., pbp_complete=False)` -> `INELIGIBLE_PBP_INCOMPLETE`).
- **D** -- PBP completing later cannot retroactively alter an
  already-recorded Friday decision: the gate functions are pure/stateless,
  so a Friday-run orchestrator's persisted `GateResult` from that run is
  never mutated by a later call with different inputs (there is nothing in
  this layer that could reach back and change it -- immutability is a
  property of not holding any mutable shared state, not of an explicit
  lock).

## PBP canonicalization (Sections 15, 17, 20)

`plays.normalize_plays` produces the canonical schema in
`CANONICAL_PLAY_COLUMNS`, deduplicating/rejecting per `provider_play_id`:
an id repeated with byte-identical raw content across pages is kept once
(`exact_duplicate_play_ids`); an id repeated with DIFFERENT raw content is
excluded entirely from the canonical output rather than guessing which
version is correct (`conflicting_play_ids`); a play whose own `game.id`
doesn't match the requested game is excluded (`foreign_game_play_ids`).
Final row order is always `(period, wallclock, provider_play_id)` --
deterministic regardless of provider response order -- with a `sequence`
column assigned after sorting. A period-decreasing-while-wallclock-
increasing anomaly is counted (`period_ordering_violations`) but the row
is kept, since dropping one of two disagreeing rows would be a guess.

No provider-neutral PBP adapter layer was needed beyond this canonical
schema itself (Section 17): none of the historical feature builders that
read raw PBP directly (`pbp_advanced.py`) have a validated live-parity
counterpart yet (see the parity matrix), so there is no live-eligible
feature to adapt onto a shared interface today. When one is validated, it
should consume `plays.CANONICAL_PLAY_COLUMNS`, not raw BDL fields.

## EPA / advanced feature audit (Section 16)

**No BDL field is ever relabeled as nflverse EPA.** The full raw `/plays`
field list was captured live (2026-08-21, game 423945) and contains no
`epa`, `wpa`, `success`, or `cpoe` field, nor any equivalent
points-based expected-points model output. `home_win_probability` IS
present but is a distinct, provider-native win-probability figure from a
different (undocumented) model -- there is no validated transform from it
to nflverse's `wpa`, so it is carried through canonically as
`provider_home_win_probability`, never as `wpa`. GOAT-tier
`/advanced_stats/passing` exposes a differently-modeled
`completion_percentage_above_expectation` (a CPOE-*shaped* metric from a
different model) and `/advanced_stats/rushing` exposes an
expected-yards-based rushing efficiency metric -- both were inspected per
Section 2 but deliberately not wired into any canonical output or feature
family. This repo has no independent expected-points model to fall back
on, and Fix 5 built none (explicitly out of scope). Full per-family
classification: `docs/BDL_PARITY_MATRIX.md`.

## Raw snapshot / provenance storage (Sections 21-22)

`raw_store.py` resolves a NEW, separate env var, `NFL_LIVE_DATA_ROOT` --
deliberately distinct from `NFL_MODEL_DATA_ROOT` (the historical, licensed
estate) and `NFL_MODEL_ARTIFACT_ROOT` (derived pipeline output), mirroring
the same external-vs-generated separation Fix 1.5 established. No
hardcoded machine path anywhere. Snapshots are content-addressed
(`sha256(raw_data)` as the filename): writing the identical payload twice
is a no-op on the raw file (idempotence -- proven by
`test_write_snapshot_idempotent_same_payload`), while a genuinely changed
payload is stored as a new, additional immutable file, never overwriting
the prior one (`test_write_snapshot_changed_payload_creates_new_immutable_snapshot`).
Each snapshot has a sibling manifest recording provider, endpoint, query
parameters (secret-like keys redacted -- `test_query_parameters_with_key_like_names_are_redacted`),
requested/received timestamps, response hash, row/page counts, and a
schema version.

## Player identity parity (Section 12)

`canonical.PLAYER_IDENTITY_PARITY = "PARTIAL"`: BDL's `provider_player_id`
is a BDL-native namespace with no crosswalk in this repo today to
nflverse's `gsis_id`/`pfr_id` space. No fuzzy name matching is attempted;
callers needing cross-provider player identity must build an explicit
crosswalk layer first.

## Roster/depth and injuries (Sections 13-14)

Both are LIVE pregame snapshot families, never completed-game results, and
neither is forced into the current production feature set (parity matrix:
`roster_depth` = `UNAVAILABLE` for historical-parity purposes since BDL's
roster endpoint is current-state-only with no as-of history to validate
against; `injuries` = `APPROXIMATE_NOT_APPROVED` since BDL exposes only
`{status, comment, date}` with no practice-participation field at all --
`canonical.normalize_injuries` does not fabricate one). No future roster
or injury snapshot may be used for an earlier forecast; enforcing that
as-of discipline is the caller's forecast-pipeline responsibility (this
layer only stamps `collected_at_utc`, honestly, on every row).

## 2025 overlap backtest (Section 19) -- ACTUALLY RUN 2026-08-22

`scripts/bdl_overlap_backtest.py` was run against `NFL_MODEL_DATA_ROOT` and
a live `BALLDONTLIE_API_KEY`. Sample: 10 non-cherry-picked 2025
regular-season games (deterministic evenly-spaced stride over all 272 REG
games sorted by `(week, game_id)`), spanning weeks
`{1, 2, 4, 6, 8, 10, 11, 13, 15, 17}` and 16 distinct teams, both
home/away orientations. Full report: `outputs/bdl_overlap_backtest_2025.json` (generated output, not
committed as part of this change -- regenerate with the command above).

**Game identity + final score:** 10/10 games matched by season/week/
home/away identity; 10/10 home-score exact, 10/10 away-score exact; max
kickoff-timestamp diff 0.0 minutes.

**Team box** (19 team-games compared against nflverse's own `team_stats`
table): 15 of 16 compared fields were 100% exact
(`passing_completions`, `passing_attempts`, `rushing_yards`,
`rushing_attempts`, `sacks`, `penalties`, `penalty_yards`,
`interceptions_thrown`, `fumbles_lost`, `first_downs_passing`,
`first_downs_rushing`, and the derived `turnovers`/`net_passing_yards`/
`total_yards`). `net_passing_yards`/`total_yards` only matched once a real
sign-convention difference was found and corrected: historical
`sack_yards_lost` is signed negative, BDL reports the unsigned magnitude
(e.g. 2025_01_ARI_NO ARI: historical `-33`, BDL `33`). `defensive_touchdowns`
did NOT validate: 18/19 exact with one unexplained disagreement
(2025_10_ARI_SEA SEA: historical `def_tds=0`, BDL `defensive_touchdowns=2`).

**Player/QB:** player identity coverage (matched by normalized name + team
+ game, since BDL's `provider_player_id` has no crosswalk to nflverse's
`gsis_id`) was 184/192 (95.8%) for active skill-position rows and 23/23
(100%) for QB rows. `completions`, `passing_yards`, `passing_touchdowns`,
`passing_interceptions`, `sacks_suffered`, and every compared
rushing/receiving field were 100% exact across all comparable rows.
**Critically**, comparing against the ACTUALLY WIRED historical feature
builder (`aggregate_qb_game_efficiency`, `aggregate_advanced_team_game` --
which read raw PBP directly, not the box-score-style `team_stats`/
`player_stats` tables) found that nflverse's own PBP `pass_attempt` mask
counts MORE pass attempts than the published box score (which BDL matches)
in 18 of 19 spot-checked team/QB-games -- a genuine gap between two
nflverse representations of the same game, not a BDL data-quality issue,
but one that blocks treating BDL as a validated substitute for those wired
attempt-count-derived fields.

**PBP:** all 10 games paginated across 2 pages each, 0 dedup/foreign-game/
ordering anomalies, 10/10 terminal-score reconciliations succeeded. BDL's
canonical play count was consistently HIGHER than nflverse's PBP row count
for the same game (e.g. 195 vs. 182, 198 vs. 187; mean excess ~10
plays/game) -- ingestion mechanics (pagination, dedup, ordering,
score-reconciliation) are solid, but BDL play count is not 1:1 with
nflverse's PBP row convention.

See `docs/BDL_PARITY_MATRIX.md` for how each finding maps to a
family-level classification -- overall, the measured evidence did NOT
promote any additional family to `EXACT`/`VALIDATED_TRANSFORM` beyond the
two already-approved (`game_result`, `elo_inputs`); several
`APPROXIMATE_NOT_APPROVED` entries now carry concrete measured evidence
(rather than "no overlap run") explaining precisely why they remain
unapproved.

## Fail-closed production gates (Section 23) -- summary

| Question | Answer |
|---|---|
| Can a non-final BDL game alter completed-game score/Elo state? | **NO** -- `can_update_score_state` requires `status_state == "final"` |
| Can a final game with incomplete PBP alter PBP/EPA-derived state? | **NO** -- `can_update_pbp_state(..., pbp_complete=False)` is ineligible even when score state is eligible |
| Is result availability the first trusted final+complete observation, not kickoff/non-null score? | **YES** -- `finality.compute_result_available_at_utc` |
| Is raw BDL PBP ever relabeled as nflverse EPA without a validated transform? | **NO** -- no `epa`/`wpa`/`success`/`cpoe` column is ever produced by `plays.py` |
| Can the integration be tested without a real API credential? | **YES** -- the entire offline suite (`tests/test_balldontlie_*.py` minus the live-smoke test) runs on checked-in fixtures |
| Are provider-specific fields isolated behind a canonical interface? | **YES** -- `canonical.py`/`plays.py` are the only modules that read raw BDL field names |
| Are features lacking live parity prevented from being treated as production-ready? | **YES** -- `parity.assert_production_eligible` raises `ParityIneligibleError` for anything not `EXACT`/`VALIDATED_TRANSFORM` |
| Can a later poll where PBP becomes complete make PBP appear available at an earlier (e.g. Friday) cutoff? | **NO** -- `pbp_available_at_utc` is the FIRST observation where `can_update_pbp_state` passed, independent of `score_available_at_utc`/`result_available_at_utc`; proven by `test_scenario_c_and_d_pbp_completing_after_friday_cannot_contaminate_friday_state` |
| Can a preseason Week 1 game be silently treated as regular-season Week 1? | **NO** -- `canonical.normalize_games` raises `CanonicalizationError` on a non-postseason row with a missing/unrecognized `season_type_hint`; never defaults to `"REG"` |
| Was an actual 2025 BDL-vs-historical overlap measured (not just claimed runnable)? | **YES** -- see "2025 overlap backtest" above, run 2026-08-22 |
