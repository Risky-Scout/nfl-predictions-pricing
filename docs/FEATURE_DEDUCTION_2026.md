# Compact Production Feature Deduction (Fix 6, Auto-Mode Revision)

Deduces a compact, interpretable, 2026-live-reproducible feature set,
measured with the FIXED incumbent `JointScoreModel`/`JointScoreConfig`
(`src/nfl_hybrid/modern/joint_score.py`, untouched, unhyperparameter-tuned).
Implementation: `src/nfl_hybrid/selection/feature_deduction_2026.py`.
Orchestrator: `scripts/run_fix6_feature_deduction.py`, executed in the exact
9-phase auto-mode order (source audit -> preregistration -> cache -> inner
k selection -> inner feature selection -> freeze -> LOCKED_POST_EXPOSURE_2024_AUDIT
-> post-freeze diagnostics -> evidence/tests). Every rule below (folds,
groups, primary metric, forward/backward selection rules, tolerance table)
was frozen in `selection_preregistration.json` **before** any inner fold was
evaluated -- nothing here was adjusted after seeing results.

This is **not** a model-family selection. The output is the frozen *input*
candidate feature set for that later, separate stage.

## 1. Development / locked-audit policy

`selection_max_season = 2024`; `selection_forbidden_seasons = [2025, 2026]`.
`enforce_2025_firewall` filters the raw `games` table down to
`season <= 2024` at the single point it is first loaded (the raw parquet
does contain 2025 rows -- the firewall necessarily reads them in order to
remove them, so "0 rows ever seen" would misstate what happened); every
downstream function additionally, independently calls
`assert_selection_season_allowed` (defense in depth). Real run's exact
accounting (`selection_manifest.json` -> `firewall_counts`):

```
raw_rows_season_2025_seen_by_firewall = 285
rows_season_ge_2025_passed_beyond_firewall = 0
rows_season_ge_2025_used_for_feature_building = 0
rows_season_ge_2025_used_for_model_fit = 0
rows_season_ge_2025_used_for_metrics = 0
```

Inner (development) folds -- feature/prior decisions only:

| Fold | Train | Validate |
|---|---|---|
| A | <= 2020 | 2021 |
| B | <= 2021 | 2022 |
| C | <= 2022 | 2023 |

**LOCKED_POST_EXPOSURE_2024_AUDIT** (train <= 2023, validate 2024): run
exactly once, after everything below was frozen -- but 2024 is **not**
independent/untouched evidence for this run. An earlier Fix 6 pass already
evaluated this same fold once and reported its result
(`outer_2024_previously_exposed=true`). It remains a useful locked
stress/audit period only because every rule below was frozen before *this*
run touched 2024 data, and the spec was never revised afterward. 2025
remains completely unused in this stage (reserved for a later post-freeze
audit); 2026 remains the strongest prospective evidence.

The historical estate only spans seasons 2020-2025 (no pre-2020 history),
so Fold A trains on a single season (269 games) -- thin but used as
specified, not replaced with an invented split.

## 2. Production-candidate inventory

Only `game_result` and `elo_inputs` are Fix-5 production-eligible
(`docs/BDL_PARITY_MATRIX.md`). A fresh live-parity/registry audit performed
in this revision additionally excludes two native schedule columns the
first Fix 6 pass had used:

- **`neutral_site`**: not in Fix 5's live-mapped field set
  (`nfl_hybrid.providers.balldontlie.canonical.GAMES_COLUMNS` has no
  `neutral_site` entry; `parity.py`'s `elo_inputs` `live_definition` only
  lists `game_id/season/week/kickoff_utc/home_team/away_team/home_score/away_score`)
  -- `BLOCKED_NO_LIVE_PARITY`.
- **`division_game`**: no static team->division registry exists anywhere in
  this repository (`team_ids.py` has only a name-alias table; a repo-wide
  grep for division/conference registries returns zero hits) -- the native
  column is historical-only with no proven identical live reconstruction --
  `BLOCKED_NO_LIVE_REGISTRY`.

Consequently the `HOME_AWAY_CONTEXT` group from the first Fix 6 pass is
**removed entirely** (no surviving members). `home_rest_days`/
`away_rest_days` (native, historical-only) are also replaced: the `REST`
group now uses `{home,away}_days_since_last_game`, computed by the same
shared `build_team_pregame_features` shift-based transform from each team's
own chronological `kickoff_utc` sequence -- a field Fix 5's `elo_inputs`
entry already proves is live-identical -- `DERIVED_FROM_EXACT_SOURCE`.

**Eligible feature universe: exactly 25 features across 7 groups (1 core +
6 non-core candidates)** -- asserted against the live registry at runtime
(`assert_registry_counts_match_expected`); a mismatch is itself a hard STOP
before the first fit:

| Group | Columns | Kind | Core? |
|---|---|---|---|
| ELO_STRENGTH | `{home,away}_elo_pregame_{rating,win_probability,expected_margin}` (6) | pivoted | **core** |
| SCORING_MARGIN | `{home,away}_margin__{week1_blended,last8_mean}` (4) | pivoted | candidate |
| SCORING_OFFENSE | `{home,away}_points_scored__{week1_blended,last8_mean}` (4) | pivoted | candidate |
| SCORING_DEFENSE | `{home,away}_points_allowed__{week1_blended,last8_mean}` (4) | pivoted | candidate |
| WIN_RATE | `{home,away}_win__{week1_blended,last8_mean}` (4) | pivoted | candidate |
| REST | `{home,away}_days_since_last_game` (2) | pivoted | candidate |
| SEASON_PHASE | `week` (1) | native carrier | candidate |

**Blocked (excluded before evaluation, never entered the tournament):**
`epa`/`opponent_adjusted`/`qb` pregame-state families (PBP/QB-box derived,
Fix-5 blocked) and `market_history` (no proven 2026 live-parity contract for
the T10/closing-line provider) -- full detail in
`NFL_MODEL_ARTIFACT_ROOT/feature-deduction-2026/candidate_feature_inventory.json`
and `outputs/fix6_feature_selection_summary.json` ->
`blocked_high_value_candidates` / `blocked_no_live_registry_columns`.

## 3. Fixed measuring instrument

`JointScoreConfig()` defaults, untouched: `max_iter=180, learning_rate=0.045,
max_leaf_nodes=23, min_samples_leaf=24, l2_regularization=1.0,
random_state=42`. `model_config_hash` persisted in the selection manifest.
No model-family comparison, no hyperparameter search, anywhere in this
module.

## 4. Selection procedure (all rules frozen before Phase 4's first fit)

**Primary metric:** `primary_score = 0.5 * (margin_RMSE + total_RMSE)`
(the Section 13 fallback -- no joint Gaussian NLL/log-score utility exists
anywhere in this repository).

**Paired significance test (analytic, no bootstrap, no RNG, no seed):**
`L_i = 0.5 * (margin_error_i^2 + total_error_i^2)` per game;
`delta_i = L_b_i - L_a_i`; `SE(delta) = sample_std(delta, ddof=1) / sqrt(n)`.

**Forward selection:** start from `CORE_GROUPS = (ELO_STRENGTH,)` --
`k` only affects `*_week1_blended` columns, which don't exist in
`ELO_STRENGTH` alone, so `K_SEARCH_SPEC` (below) is deliberately a larger
spec than `CORE_GROUPS`, documented as an evidence-based deviation. Each
round, every remaining candidate group is evaluated against the current
spec; a group passes only if it wins in >= 2 of the 3 inner folds **and**
`mean(delta) + SE(delta) < 0`; among passers, the lowest pooled
`primary_score` is added (ties: fewer added features, then fixed registry
order). Repeats until no group passes.

**Backward ablation:** paired, signed, no `abs()` -- `mean_delta <=
SE_delta` (delta = reduced-minus-current) removes a group; continuous
updates in fixed registry order; capped at 3 sweeps, with a hard STOP
(`BACKWARD_ABLATION_DID_NOT_CONVERGE`) if the 3rd sweep still removes a
group.

**Fit budget:** 134 fits worst case (`plan_fit_budget()`), well under 200.

## 5. Inner selection results (real 2020-2024 run)

```
baseline (ELO_STRENGTH only): primary_score = 14.5976
```

| Candidate group | Candidate primary_score | Wins of 3 folds | mean_delta | SE_delta | Verdict |
|---|---|---|---|---|---|
| SCORING_MARGIN | 14.6265 | 2/3 | +0.872 | 3.933 | REJECTED_NO_OOS_GAIN |
| SCORING_OFFENSE | 14.5635 | 2/3 | -1.152 | 4.267 | REJECTED_NO_OOS_GAIN |
| SCORING_DEFENSE | 14.5735 | 1/3 | -0.857 | 4.165 | REJECTED_NO_OOS_GAIN |
| WIN_RATE | 14.5682 | 2/3 | -0.934 | 3.385 | REJECTED_NO_OOS_GAIN |
| REST | 14.6319 | 1/3 | +0.914 | 3.009 | REJECTED_NO_OOS_GAIN |
| SEASON_PHASE | 14.6827 | 0/3 | +2.436 | 2.370 | REJECTED_NO_OOS_GAIN |

**No candidate group passed the predeclared retain rule.** `SCORING_OFFENSE`
and `WIN_RATE` were directionally favorable (negative mean_delta, 2/3-fold
wins) but the improvement was not large relative to its standard error
(`mean(delta) + SE(delta)` stayed positive for every group) -- on this
~1,300-row estate, none of the six candidate groups clears the frozen,
one-sided significance bar. This is reported honestly, not adjusted:
per the auto-mode operating contract, a weak result is written to the
manifest as-is, and the frozen rules are never loosened in response to an
observed metric. Backward ablation had nothing to remove (no non-core
group was ever retained) and converged trivially on sweep 1. Full trace
with every fold's per-group numbers:
`NFL_MODEL_ARTIFACT_ROOT/feature-deduction-2026/group_ablation_results.json`.

**Group classification (never mixed -- `group_classification` in
`outputs/fix6_feature_selection_summary.json`):**

- `SELECTED`: `ELO_STRENGTH`.
- `REJECTED_BY_OOS_RULE` (fully eligible, evaluated, Fix-5 production-eligible
  in principle -- excluded solely because the frozen significance rule was
  not cleared): `SCORING_MARGIN`, `SCORING_OFFENSE`, `SCORING_DEFENSE`,
  `WIN_RATE`, `REST`, `SEASON_PHASE`.
- `BLOCKED_BY_LIVE_PARITY_OR_CONTRACT` (never entered the candidate
  tournament at all -- excluded on live-reproducibility/provenance grounds,
  independent of any observed statistical performance): the `epa`/
  `opponent_adjusted`/`qb` pregame-state families, `market_history`,
  `neutral_site`, `division_game` (see Section 2 above).

## 6. Final compact feature set

**Status: `FROZEN_FEATURE_INPUT_FOR_MODEL_FAMILY_SELECTION`** -- this is
the frozen *input* feature specification for the next, separate
model-family-selection stage. It is explicitly **not** a final production
model, not a final production predictor, and not a production-approved
final model: that next stage still has to select/freeze the model family
and regenerate official chronological OOF/calibration evidence
(`nfl_hybrid.evaluation.chronological_oof`) before any production
promotion.

**6 features, 1 group: `ELO_STRENGTH`.**

```
home_elo_pregame_rating, home_elo_pregame_win_probability, home_elo_pregame_expected_margin,
away_elo_pregame_rating, away_elo_pregame_win_probability, away_elo_pregame_expected_margin
```

`feature_manifest_hash`, `week1_prior_config_hash`, `selection_config_hash`,
`model_config_hash`, `candidate_group_registry_hash`, `preregistration_hash`
are all in
`NFL_MODEL_ARTIFACT_ROOT/feature-deduction-2026/selection_manifest.json`
(mirrored in `outputs/fix6_feature_selection_summary.json`). 6 <= 100 --
the compactness gate passes trivially.

**Active vs. selected-during-evaluation Week-1 prior.** `week1_blend_k_selected_during_candidate_evaluation
= 8` was legitimately selected on `K_SEARCH_SPEC` before feature-group
selection began (Section 4), but `week1_blend_active_in_final_feature_set
= false`: no group that consumes `*__week1_blended` columns survived
selection (Section 5). The Week-1 prior blend mechanism
(`docs/WEEK1_PRIOR_2026.md`, `nfl_hybrid.features.week1_prior`) is fully
built, tested, and frozen (`k=8` retained as selection evidence/provenance,
not deleted), but it is not active in the current production candidate
set. The **only** Week-1 prior mechanism actually active in the six
retained features is `active_final_week1_prior =
"ELO_SEASON_BOUNDARY_REGRESSION_TO_MEAN"` (`active_final_week1_prior_source
= "nfl_hybrid.features.elo_state / nfl_hybrid.legacy.elo"`) -- Elo's
pre-existing, unmodified `LegacyElo.regress_to_mean` season-boundary
shrinkage.

**Elo core feature relationship (audited, not pruned -- pruning after
selection would violate the frozen process).** Of the six preregistered
core features, `elo_pregame_win_probability` and `elo_pregame_expected_margin`
(each side) are a **deterministic bijective pair**: `LegacyElo.predict`
(`src/nfl_hybrid/legacy/elo.py:99-100`) computes both as monotonic,
invertible transforms of the identical scalar `adjusted_difference` --
`p_home = 1/(1+10^(-adjusted_difference/scale))` (logistic) and
`margin = adjusted_difference/elo_to_margin` (linear) -- so, for a fixed
game row, either is exactly recoverable from the other. The away-side
values are likewise exact deterministic transforms of the home-side values
(`away_win_probability = 1 - home_win_probability`,
`away_expected_margin = -home_expected_margin`;
`src/nfl_hybrid/features/elo_state.py`). `elo_pregame_rating` (home/away)
is a genuinely different, non-redundant quantity -- the raw pre-context
rating that combines with match-context adjustments not separately exposed
as their own `ELO_STRENGTH` columns (home-field, travel, rest/bye, QB
adjustment, playoff multiplier) to form `adjusted_difference`, so rating
is not recoverable from win_probability/expected_margin alone. Net: **3
non-redundant underlying quantities** (`home_rating`, `away_rating`,
`adjusted_difference`) across the 6 retained columns.
`elo_core_feature_relationship` (persisted verbatim in
`outputs/fix6_feature_selection_summary.json` and
`selection_manifest.json`): *"related deterministic/derived Elo encodings
retained as preregistered core; not independently feature-selected"* --
all six columns are kept exactly as preregistered.

## 7. LOCKED_POST_EXPOSURE_2024_AUDIT

Because the compact spec and the core baseline are literally the same
group set (`ELO_STRENGTH`) this round, every comparison below is exactly
0% relative change by construction -- the audit still ran once, for real,
against the frozen rules, and every rule trivially passes:

| Check | Rule | Result |
|---|---|---|
| Overall | `compact <= core * 1.02` | 0.0% -- PASS |
| Margin | `compact <= core * 1.05` | 0.0% -- PASS |
| Total | `compact <= core * 1.05` | 0.0% -- PASS |
| Week 1 | `compact <= core * 1.10` | 0.0% -- PASS |
| Weeks 2-4 | `compact <= core * 1.10` | 0.0% -- PASS |
| Weeks 5+ | `compact <= core * 1.05` | 0.0% -- PASS |

**Overall: PASS** (all six rules). Margin RMSE 13.788, Total RMSE 13.411,
primary_score 13.600 (272 validate games).

Secondary diagnostics (raw, uncalibrated, canonical T10 ATS/TOTAL Brier/log
loss, attached strictly post-hoc -- never a training feature, never a
selection lever; **not** the native `home_spread_reference`/
`total_line_reference` columns): ATS Brier 0.287 / log loss 0.788 (264
games); TOTAL Brier 0.274 / log loss 0.758 (267 games). Both markets
resolved `status="OK"` -- the new `canonical_market.pregame_ats_t10_2020_2024_confirmation`
/ `..._total_..._2020_2024_confirmation` registry keys (additive, Fix 4
untouched) carry the same `>=3 books, <=5min lag` quality-gate columns as
Fix 4's own 2020-2023 export and cover season 2024.

Full per-game predictions:
`NFL_MODEL_ARTIFACT_ROOT/feature-deduction-2026/outer_2024_predictions.parquet`
(via `selection_manifest.json`'s embedded outer-confirmation record).

## 8. Market-line independence

Hard exclusion list (`FORBIDDEN_MARKET_COLUMNS`) covers every current-market
column on the canonical `games` table, checked at every carrier/feature
assembly point. Poison-pill proof:
`tests/test_feature_deduction_2026.py::test_poison_pill_target_market_mutation_does_not_change_prediction`.

## 9. Interpretability readiness

`HistGradientBoostingRegressor` exposes no `feature_importances_`; used
`sklearn.inspection.permutation_importance` (no new dependency) on the
frozen 2024-fitted margin model instead. Worked 2024 example (`outputs/fix6_feature_selection_summary.json`
-> `interpretability`): game `2024_01_BAL_KC` (Week 1), predicted margin
-5.71 / total 44.38. Local attribution is explicitly labeled
`LOCAL_SENSITIVITY`, `NOT_SHAPLEY_ATTRIBUTION`, `NOT_ADDITIVE_CONTRIBUTION`
-- each of the 6 features perturbed one at a time to a documented reference
value (median computed from the outer model's **training data only**,
`season <= 2023`, never any 2024 validation row) with the resulting
margin/total delta reported. Eventual local (SHAP-style) explanation would
need either a `shap` dependency or a from-scratch occlusion-based method --
neither built here.

## 10. Artifacts

Generated (not committed), `NFL_MODEL_ARTIFACT_ROOT/feature-deduction-2026/`:
`candidate_group_registry.json`, `selection_preregistration.json`,
`candidate_feature_inventory.json`, `group_ablation_results.json`,
`week1_prior_k_grid.json`, `selection_manifest.json`.

Committed evidence: `outputs/fix6_week1_prior_proof.json`,
`outputs/fix6_feature_selection_summary.json`.

Reproduce: `NFL_MODEL_DATA_ROOT=... NFL_MODEL_ARTIFACT_ROOT=... PYTHONPATH=src
python scripts/run_fix6_feature_deduction.py` (single-threaded env vars per
Section 28; runs all 9 phases, including two full focused-pytest passes and
a `git diff --check`).
