# Week 1 / Early-Season Prior (Fix 6)

> **Auto-mode revision (this document has been updated in place):** the
> blend mechanism, k-grid, and acceptance tests below are frozen and fully
> validated, but the compact-feature-set search in
> `docs/FEATURE_DEDUCTION_2026.md` did not retain any group that consumes
> `__week1_blended` columns this round (only `ELO_STRENGTH` passed the
> stricter, analytic-paired-SE forward-selection bar). This module and its
> frozen `k` remain correct, tested, production-ready infrastructure for
> the next feature-deduction round -- they are just not part of the current
> 6-feature production candidate set. Neutral-fallback constants are now
> fixed, predeclared `UNTUNED_ENGINEERING_PRIOR` values (`margin=0.0,
> win=0.5, points_scored=points_allowed=22.0`), never derived from this
> repository's outcome data -- the earlier outcome-derived
> `derive_neutral_fallback` has been removed.
>
> **Provenance fields (persisted in `outputs/fix6_week1_prior_proof.json`
> and `selection_manifest.json`):** `week1_blend_k_selected_during_candidate_evaluation
> = 8`; `week1_blend_active_in_final_feature_set = false`;
> `active_final_week1_prior = "ELO_SEASON_BOUNDARY_REGRESSION_TO_MEAN"`
> (`active_final_week1_prior_source = "nfl_hybrid.features.elo_state /
> nfl_hybrid.legacy.elo"`). `k=8` is retained as selection
> evidence/provenance -- it does not affect the frozen six-feature
> production candidate specification.

Answers one question: *what team state exists before the first current-season
result?* Scope: only the two Fix-5 production-eligible source families,
`game_result` and `elo_inputs` (`docs/BDL_PARITY_MATRIX.md`). Implementation:
`src/nfl_hybrid/features/week1_prior.py`. Wired in by
`src/nfl_hybrid/selection/feature_deduction_2026.py`.

## Elo: reused unchanged

`nfl_hybrid.features.elo_state.build_elo_pregame_state` already carries a
proven Week-1 prior for the `ELO_STRENGTH` group:
`LegacyElo.regress_to_mean()` (`src/nfl_hybrid/legacy/elo.py:151-155`) fires
once per season boundary, before that season's first game is scored:
`new_rating = keep * prior_rating + (1 - keep) * mean`, with
`keep = ELO_REVERSION_KEEP = 2/3` and `mean = ELO_REVERSION_MEAN = 1505.0`
(`src/nfl_hybrid/constants.py`). A team with no prior rating at all starts at
`ELO_INITIAL = 1500.0`. Nothing in this module changes that code path; Fix 6
only adds acceptance tests proving it already satisfies the Section 11
contract below (`tests/test_elo_pregame_state.py`, pre-existing, plus the
Fix-6-specific proofs in `tests/test_week1_prior_2026.py` operate on the
new game-result family instead, since Elo's own suite already covers it).

## Game-result state: new blend (`SCORING_OFFENSE`, `WIN_RATE`, etc.)

`build_game_result_team_game(games)` derives one row per (game_id, team_id)
--`points_scored`, `points_allowed`, `margin`, `win`-- from nothing but
`games`' own `home_score`/`away_score` (the `game_result` EXACT family). This
feeds, unmodified, into the existing leakage-safe
`nfl_hybrid.features.pregame_rolling.build_team_pregame_features` (shift-
before-roll), which is where the Week-1 gap actually lives:
`{metric}__season_mean` is `NaN` at a team's first game of a season by
construction.

### Blend contract

```
blended_state(n) = w(n) * current_season_state + (1 - w(n)) * prior_reference
w(n) = n / (n + k)
```

- `n` = `team_season_prior_games` (0 at Week 1; already emitted by
  `build_team_pregame_features`).
- `current_season_state` = `{metric}__season_mean` (the existing, unmodified,
  within-season shifted average; `NaN` at `n=0`, handled by forcing
  `w(n)=0` whenever it is `NaN` regardless of the raw `n`-based weight --
  covers both Week 1 and the rare case where every prior game this season
  was a tie).
- `prior_reference` = that team's own complete prior-season average
  (`prior_status=PRIOR_SEASON_AVAILABLE`) if one exists, else a single
  declared neutral/league value (`prior_status=NEUTRAL_FALLBACK`).
- `k` is ONE global constant for the whole mechanism -- never tuned per
  feature.

### Neutral fallback

Fixed, predeclared, **`UNTUNED_ENGINEERING_PRIOR`** constants -- never
derived from this repository's outcome data, never searched/tuned:
`neutral_margin=0.0`, `neutral_win_rate=0.5`, `neutral_points_scored=22.0`,
`neutral_points_allowed=22.0` (`neutral_prior_source` field:
`"fixed untuned engineering constant; not estimated from 2020-2024 data"`).
No predeclared league-average-points constant already existed anywhere in
this repository (checked `constants.py`, `config/*.yaml`, `priors/*.py`,
`spreadsheet_baselines/*.py` -- zero hits); `22.0` is a round, publicly-known
modern-NFL-era scoring average, hardcoded once in `Week1PriorConfig`'s
default factory. In practice this value only ever applies to teams' games
within the estate's first season (2020), since every following season's
teams already have a prior-season record in this 2020-2025 estate; it
therefore has zero effect on any inner/outer *validation* metric (all folds
validate on 2021+).

### k selection

Predeclared grid `k in {2, 4, 6, 8}` (`w(n)=n/(n+k)`), selected on inner
folds A/B/C using `K_SEARCH_SPEC = ELO_STRENGTH + SCORING_MARGIN +
SCORING_OFFENSE + SCORING_DEFENSE + WIN_RATE` (the minimum spec that
actually exercises `k` -- see the feature-deduction doc's flagged
interpretation note), with a paired, exact tie-break against `best_k`
(`mean_delta(k) <= se_delta(k)`, largest qualifying k wins -- no bootstrap,
no generic/ambiguous "SE"), frozen before any LOCKED_POST_EXPOSURE_2024_AUDIT
evaluation. Real run result (`outputs/fix6_week1_prior_proof.json`):

| k | primary_score | mean_delta vs best | se_delta vs best | within 1 SE of best |
|---|---|---|---|---|
| 2 | 14.462 | 3.610 | 3.638 | yes |
| 4 | 14.349 | 0.355 | 2.890 | yes |
| 6 | 14.401 | 1.798 | 2.196 | yes |
| **8** | **14.339 (best)** | 0.0 | 0.0 | -- |

All four k values are within one SE of `best_k=8` on this estate (the
sample is small enough that k is not sharply identified) -- per the
predeclared tie-break rule, the **largest** qualifying k wins:
`selected_k = 8.0`, frozen (`week1_prior_config_hash` in
`outputs/fix6_week1_prior_proof.json`).

## Acceptance tests (Section 11, `tests/test_week1_prior_2026.py`)

| Requirement | Test |
|---|---|
| A. No current-season result leaks into Week 1 | `test_a_week1_own_score_mutation_does_not_change_its_own_blend` |
| B. Prior-season final state may set next-season Week 1 | `test_b_prior_season_final_state_sets_next_season_week1` |
| C. Future/current-season games cannot affect Week 1 | `test_c_later_current_season_game_does_not_affect_week1` |
| D. Missing prior season -> declared neutral prior | `test_d_no_prior_season_uses_neutral_fallback`, `test_derive_neutral_fallback_is_fail_closed_on_forbidden_seasons` |
| E. Weight rises monotonically with games played | `test_e_current_season_weight_rises_monotonically` |
| F. No blend parameter changes after outer 2024 | `test_f_config_is_frozen_and_hash_is_stable` (frozen dataclass + stable hash) |
| G. Deterministic replay | `test_g_deterministic_replay_regardless_of_row_order` |

## Fallback policy (Section 9)

A team with no prior-season state gets the declared neutral fallback --
never a forward-fill from a future game, never the current-season result,
never the target game's own market line, never a silently dropped row.
Every normal scheduled game with known teams gets a valid feature vector.
