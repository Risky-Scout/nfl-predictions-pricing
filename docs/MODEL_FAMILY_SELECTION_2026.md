# Limited Model-Family Selection + Estimator Spec Freeze (Fix 7)

Selects and freezes the predictive estimator family/hyperparameter
specification that consumes the already-frozen Fix 6 six-Elo-feature input
(`outputs/fix6_feature_selection_summary.json`,
`feature_manifest_hash=d523a007f7babfd0e8913f7ad6204f3670120d5d0f36183ecead9e153a8b24bf`).
Implementation: `src/nfl_hybrid/selection/model_family_selection_2026.py`.
Orchestrator: `scripts/run_fix7_model_family_selection.py`, executed in the
exact 9-phase auto-mode order: source/repo/Fix-6 contract audit -> firewall +
build selection matrix + `selection_matrix_hash` -> preregistration freeze +
focused prefit tests -> inner candidate fits -> Ridge alpha selection ->
family selection + final model spec freeze -> `LOCKED_POST_EXPOSURE_2024_AUDIT`
-> post-freeze interpretability (zero refits) -> evidence/tests/git-diff/final
report. Every rule below (candidate registry, primary metric, Ridge/family
selection rules, audit tolerance table) was frozen in
`model_family_selection_preregistration.json` **before** any inner fold was
evaluated.

Fix 7 does **not** reopen feature selection, does **not** calibrate
ATS/TOTAL, and does **not** promote a production model. Its output is a
frozen *estimator specification* for the next, separate stage (official
chronological OOF + ATS/TOTAL calibration,
`nfl_hybrid.evaluation.chronological_oof`).

## 1. Fix-6 feature contract

`load_and_verify_fix6_feature_contract` treats the committed
`outputs/fix6_feature_selection_summary.json` as authoritative and
cross-checks it independently against the live `fd.FEATURE_GROUPS["ELO_STRENGTH"]`
registry and a fresh hash recomputation -- all three must agree (hard STOP on
any divergence, in either direction):

```
home_elo_pregame_rating, home_elo_pregame_win_probability, home_elo_pregame_expected_margin,
away_elo_pregame_rating, away_elo_pregame_win_probability, away_elo_pregame_expected_margin
```

`fix6_feature_manifest_hash = d523a007f7babfd0e8913f7ad6204f3670120d5d0f36183ecead9e153a8b24bf`
(matched the committed value exactly).

## 2. Selection matrix (Phase 2, before preregistration)

`enforce_2025_firewall` -> `build_candidate_matrix` -> six-feature
missingness check -> `compute_selection_matrix_hash`, in that order, with
**no fits and no comparative metrics** in this phase. Real run:

```
selection_matrix_row_count = 1343
selection_matrix_hash = 4034b9a468a8e01859581c338f1123a43544495670e443f68d3906c64d70a79f
raw_rows_season_2025_seen_by_firewall = 285
rows_season_ge_2025_passed_beyond_firewall = 0
rows_season_ge_2025_used_for_feature_building = 0
rows_season_ge_2025_used_for_model_fit = 0
rows_season_ge_2025_used_for_metrics = 0
```

## 3. Preregistration (Phase 3, immutable afterward)

`model_family_selection_preregistration.json` (`NFL_MODEL_ARTIFACT_ROOT/model-family-selection-2026/`)
was written and hashed (`model_family_preregistration_hash =
cafb64d89910c56b312b48cf140f901792c4a77ac7b075c98b903169f8879932`) **before**
Phase 4's first real fit, and re-verified unchanged at the entry to every
subsequent phase. It contains the candidate registry (below), fold
definitions, the primary-metric and paired-SE formulas, the Ridge-alpha and
family-selection rules, the fit budget, and the
`LOCKED_POST_EXPOSURE_2024_AUDIT` tolerance table -- nothing below was
adjusted after seeing a result.

Inner (development) folds, reused unchanged from Fix 6:

| Fold | Train | Validate |
|---|---|---|
| A | <= 2020 | 2021 |
| B | <= 2021 | 2022 |
| C | <= 2022 | 2023 |

`LOCKED_POST_EXPOSURE_2024_AUDIT` fold: train <= 2023, validate 2024. Per the
operator's contract, 2024 is treated as previously-exposed for this stage too
(`outer_2024_previously_exposed=true`) because Fix 6's own outer confirmation
already fit/evaluated a model on this identical six-feature input over this
identical season -- it is never described as an untouched/independent/first
holdout.

## 4. Candidate registry (frozen before first fit)

| Candidate | Family | Complexity | Hyperparameters | `random_state` |
|---|---|---|---|---|
| `RIDGE_ALPHA_0_1` | RIDGE | 0 | `alpha=0.1, solver=svd` | `None` (no such param) |
| `RIDGE_ALPHA_1` | RIDGE | 0 | `alpha=1.0, solver=svd` | `None` |
| `RIDGE_ALPHA_10` | RIDGE | 0 | `alpha=10.0, solver=svd` | `None` |
| `RIDGE_ALPHA_100` | RIDGE | 0 | `alpha=100.0, solver=svd` | `None` |
| `HUBER_FIXED` | HUBER | 1 | `epsilon=1.35, alpha=1e-4, max_iter=1000, tol=1e-5` | `None` |
| `HGBR_INCUMBENT` | HGBR | 2 | full committed `JointScoreConfig` (`max_iter=180, learning_rate=0.045, max_leaf_nodes=23, min_samples_leaf=24, l2_regularization=1.0`) | **`42`** (`fd.MODEL_CONFIG.random_state`, verified equal at Phase 1, hard-gated) |

Ridge/Huber use `StandardScaler(with_mean=True, with_std=True)` fit on
**training-fold rows only**, one independent estimator each for margin and
total (same family/hyperparameters for both -- no mixed hybrids). HGBR uses
the existing `JointScoreModel`/`JointScoreConfig` unmodified, through a
narrow local fit/predict adapter that never calls Fix 6's own
`_fit_predict_fold` (that function requires Fix 6's private
registry-freeze side effect, a Fix-6-specific coupling this module
deliberately does not depend on).

Huber's `INVALID_NUMERICAL` classification is limited to exactly three
triggers (`ConvergenceWarning`, non-finite fitted coefficient/intercept,
non-finite validation prediction); any other exception is a real execution
failure, not a numerical-invalidity classification. In the real run, Huber
was numerically valid on all 3 inner folds x 2 targets (`huber_valid=true`).

## 5. Ridge alpha selection (Phase 5)

`best_alpha = argmin(pooled inner primary_score)`; an alpha qualifies iff
`mean_delta(alpha) <= se_delta(alpha)` vs best (`delta_i = L_alpha_i -
L_best_i`); the **largest** qualifying alpha is selected. Real run:

| Alpha | Pooled primary_score | Mean delta vs best | SE delta vs best | Within 1 SE | Selected |
|---|---|---|---|---|---|
| 0.1 | 13.7485 | 0.9372 | 0.5906 | No | No |
| 1.0 | 13.7329 | 0.5164 | 0.4197 | No | No |
| 10.0 | 13.7211 | 0.2036 | 0.2134 | **Yes** | No |
| **100.0** | **13.7133** | 0.0 (best) | 0.0 | Yes | **Yes** |

`RIDGE_ALPHA_100` was both the pooled-best alpha and the largest qualifying
alpha, so it is `RIDGE_SELECTED` unambiguously.

## 6. Family selection (Phase 6)

Deterministic four-step algorithm (Step A: pooled-best family; Step B: every
family within one pooled SE of best; Step C: lowest-complexity family in
that set is tentative; Step D: tentative wins outright if it is already the
lowest-complexity valid family, otherwise it must reproducibly beat -- >=2 of
3 inner folds -- every lower-complexity family or the promotion falls back to
the lowest-complexity family it failed to beat). Complexity order frozen
before any metric: `RIDGE(0) < HUBER(1) < HGBR(2)`.

Real run: pooled-best family was HUBER (`primary_score=13.6995`); RIDGE
(`13.7133`) was within one SE (`mean_delta=0.4293 <= se_delta=0.5745`), HGBR
(`14.5976`) was not. Tentative = RIDGE (lowest complexity in the one-SE set),
which is already the lowest-complexity valid family, so it wins outright --
`selection_reason="TENTATIVE_IS_LOWEST_COMPLEXITY"`.

**Selected family: RIDGE (`RIDGE_ALPHA_100`).**
`final_model_spec_hash = 418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede`.

## 7. `LOCKED_POST_EXPOSURE_2024_AUDIT` (Phase 7)

Each valid finalist (RIDGE_ALPHA_100, HUBER_FIXED, HGBR_INCUMBENT) fit once
on season<=2023, evaluated once on season==2024. All six predeclared rules
passed for the frozen RIDGE winner (RIDGE was in fact the best-performing
finalist on 2024 in this run, so every ratio-to-best equals exactly 1.0):

| Rule | Selected | Best-of-finalists | Tolerance | Pass |
|---|---|---|---|---|
| Overall | 13.0049 | 13.0049 | <=1.02x | Yes |
| Margin RMSE | 13.0630 | 13.0630 | <=1.05x | Yes |
| Total RMSE | 12.9467 | 12.9467 | <=1.05x | Yes |
| Week 1 | 11.4626 | 11.4626 | <=1.10x | Yes |
| Weeks 2-4 | 13.5402 | 13.1591 (HGBR) | <=1.10x | Yes |
| Weeks 5+ | 12.9816 | 12.9816 | <=1.05x | Yes |

`overall_pass = true`. The frozen winner (RIDGE) was **not** changed after
this audit; no fallback to a 2024-best family, no retuning, no threshold
adjustment.

## 8. Interpretability (Phase 8, zero refits)

RIDGE won, so interpretability is `.coef_`/`.intercept_` off the already-fitted
Phase-7 audit pipeline (standardized and original-scale), disclosing the same
Elo bijective-pair/non-redundancy relationship already documented in Fix 6's
evidence (copied verbatim, not reworded). One `LOCAL_SENSITIVITY` example on
a 2024 row, with every reference value computed from `season<=2023` training
data only, labeled `LOCAL_SENSITIVITY` / `NOT_SHAPLEY_ATTRIBUTION` /
`NOT_ADDITIVE_CONTRIBUTION`. If HGBR had won instead, a predefined no-fit
`sklearn.inspection.permutation_importance` adapter around the fitted
`JointScoreModel` (`_MarginOnlyPredictor`/`_TotalOnlyPredictor`,
`run_hgbr_permutation_importance`) would have been used -- confirmed against
the installed `sklearn==1.9.0` source that this works without any additional
fit. `phase8_additional_real_data_fits = 0`.

## 9. ATS/TOTAL deferred

`ats_total_probability_evaluation = "DEFERRED_UNTIL_FINAL_FAMILY_CHRONOLOGICAL_OOF"`.
No common sigma, no in-sample residual variance, and no market line was used
anywhere in family/hyperparameter selection. The next stage regenerates
official chronological OOF predictions, family-specific residual
uncertainty, and TUE/FRI ATS/TOTAL calibration using this frozen RIDGE
estimator spec.

## 10. Required yes/no answers

A. 2025/2026 outcome passed the firewall? **No.**
B. Six-feature Fix 6 input changed? **No.**
C. Target-game market field entered model fitting? **No.**
D. ROI/CLV/profit used for selection? **No.**
E. Every candidate/hyperparameter preregistered? **Yes.**
F. Ridge alpha chosen only from inner folds? **Yes.**
G. Family selection based only on inner folds? **Yes.**
H. Simpler-family one-SE rule enforced? **Yes** (RIDGE won by default rule).
I. Did 2024 alter the frozen family after the audit? **No.**
J. ATS/TOTAL comparisons deferred until family-specific chronological uncertainty exists? **Yes.**
K. Final estimator spec frozen for official OOF/calibration? **Yes.**
L. 2025 left unused as a model-selection/confirmation period? **Yes.**

## 11. Artifacts

- `outputs/fix7_model_family_selection_summary.json` -- full final report.
- `NFL_MODEL_ARTIFACT_ROOT/model-family-selection-2026/` --
  `model_family_selection_preregistration.json`, `inner_fold_results.json`,
  `ridge_alpha_selection.json`, `family_selection.json`,
  `locked_2024_audit.json`, `interpretability.json`.
- `tests/test_model_family_selection_2026.py` -- focused test suite (39
  tests), plus the existing Fix 6/Elo/pregame/parity/labels/joint-score
  regression tests re-run before and after the real run (121 tests total,
  both passes).

**Status: `FROZEN_ESTIMATOR_SPEC_FOR_OFFICIAL_OOF_CALIBRATION`.** Not a final
production model, not production-approved, not a deployed model. The next
stage (official chronological OOF + ATS/TOTAL calibration) still has to run
before any production promotion.

## 12. Post-freeze evidence replay (additive, non-selecting)

The original Fix 7 selection run did not persist the complete
per-candidate/per-fold regression table or Ridge/Huber 2024 MAEs. A
post-freeze deterministic evidence replay was therefore performed using the
exact frozen matrix, candidate specs, folds, and environment. The replay was
prohibited from changing the selected estimator. All originally persisted
pooled scores/audit quantities were required to reproduce within a fixed
numerical tolerance before reconstructed metrics were accepted.

This is `POSTFREEZE_EVIDENCE_REPLAY`, distinct from `ORIGINAL_SELECTION_EVIDENCE`:
it reconstructs missing evidence about a decision that was already made and
frozen, and has no authority to make or revise a selection decision.
Implementation: `scripts/run_fix7_evidence_replay.py` (reuses the frozen
`src/nfl_hybrid/selection/model_family_selection_2026.py` implementation
unmodified -- no new selection module, no edits to the frozen candidate
registry, matrix-building, Ridge-alpha rule, family-selection algorithm, or
audit-tolerance code). Original artifacts
(`model_family_selection_preregistration.json`, `inner_fold_results.json`,
`ridge_alpha_selection.json`, `family_selection.json`,
`locked_2024_audit.json`, `interpretability.json`) were left untouched; the
replay's own findings live only in new, additive artifacts:
`NFL_MODEL_ARTIFACT_ROOT/model-family-selection-2026/fix7_postfreeze_evidence_replay.json`
(full detail) and `outputs/fix7_postfreeze_evidence_replay_summary.json`
(repo-committed summary).

Real run result: the rebuilt selection matrix reproduced
`selection_matrix_hash=4034b9a468a8e01859581c338f1123a43544495670e443f68d3906c64d70a79f`
(1343 rows) exactly; all 6 candidate `spec_hash`es and the
`candidate_registry_hash` reproduced exactly; there was no pre-existing
prediction cache, so all 18 inner candidate x fold fits and all 3 outer
(2024) finalist fits were genuinely regenerated (`replay_pairs_loaded_from_verified_cache=0`,
`replay_pairs_regenerated=21`), matching the original's implied worst-case
fit budget (21 paired / 42 individual real-data fits total, inner + outer)
exactly. All 6 pooled inner primary_scores, both Ridge-alpha comparison
tables, and the family-selection comparison (`BEST=HUBER`, `ONE_SE_SET=
{RIDGE, HUBER}`, `TENTATIVE=RIDGE`, `TENTATIVE_IS_LOWEST_COMPLEXITY`)
reproduced within `max(1e-10, 1e-12*|original|)` of the originally persisted
values. The 2024 finalist RMSEs/primary/segment scores reproduced exactly;
the previously-missing Ridge/Huber MAEs were reconstructed
(RIDGE margin_MAE=10.041594245583477, total_MAE=10.023943284726913; HUBER
margin_MAE=10.049295621086376, total_MAE=10.01957115256402) and labeled
`POSTFREEZE_REPLAY_RECONSTRUCTED_METRIC`, never conflated with the
originally-persisted RMSE/primary_score fields. All six 2024 audit rules
independently re-passed (`LOCKED_POST_EXPOSURE_2024_AUDIT_REPLAY`,
`overall_pass=true`). `selected_family=RIDGE`, `selected_candidate=
RIDGE_ALPHA_100`, and `final_model_spec_hash=
418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede` were
verified unchanged before and after the replay.
`fix7_postfreeze_evidence_replay_hash=fbc1a3f8cf2ace1985ca9f803bd60c2b1b248b0a367ec4dcb890506f2c014375`
-- kept separate from `final_model_spec_hash`/`model_family_preregistration_hash`/
`candidate_registry_hash`/the Fix-6 feature hash, never folded into any of
them.

### 12.1 Stable scientific hash (provenance hardening)

The hash above superseded an earlier definition that also hashed volatile
execution metadata (the repo's git dirty-file listing at run time), which
changed the hash across the two replay runs (`608ebad9...` then
`d8022796...`) even though every scientific quantity in both runs was
identical. `scripts/finalize_fix7_replay_provenance.py` (no model fits, no
replay fits -- reads only the already-persisted replay evidence) redefined
`fix7_postfreeze_evidence_replay_hash` to cover ONLY a canonical
`SCIENTIFIC_REPLAY_PAYLOAD`: `schema_version`, the five original frozen
hashes, `selection_matrix_hash`/`selection_matrix_row_count`, the Fix-6
feature hash, per-candidate `spec_hash`es, `candidate_registry_hash`, the
18-row inner-fold reconstructed metrics, the pooled/Ridge-paired/family-paired
reproduction checks, the 2024 finalist reconstructed metrics and segment
metrics, the 2024 audit replay checks, firewall counts,
`selected_family_as_frozen`/`selected_candidate_as_frozen`,
`replay_used_for_selection=false`, and the fixed replay environment settings
(`random_state=42`, `single_threaded=true`) -- canonical JSON
(`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`), SHA-256.
Git dirty-file listings, timestamps, file mtimes, output-file existence, and
run sequence number are explicitly excluded from the hash and live instead
under a separate `execution_metadata` object in
`fix7_postfreeze_evidence_replay.json` that carries no weight in the hash.
A focused test (`tests/test_fix7_replay_provenance.py::test_dirty_metadata_excluded_from_scientific_hash`)
proves the same scientific payload with different dirty-file metadata
produces an identical hash.

### 12.2 Replay fit-run provenance

The two real-data replay executions recorded in this branch's history (run 1
producing `fix7_postfreeze_evidence_replay_hash=608ebad9...`, run 2 -- after
documentation-only edits -- producing `d8022796...`) are both real,
successful, real-data fit runs of the already-frozen estimator:
`successful_replay_run_count=2`. Per run: 21 paired / 42 individual real-data
fits (18 inner candidate x fold + 3 outer 2024 finalist). Cumulative across
both runs: 42 paired / 84 individual. `original_real_data_paired_fit_count=21`/
`original_real_data_individual_fit_count=42` (the original Fix 7 selection
run) is tracked separately and was not repeated by either replay.
`all_replay_fits_used_for_selection=false` for every one of these fits, in
both runs. The second replay run was scientifically unnecessary (the
estimator was already frozen before either ran) but harmless: every
reproduction gate in run 2 matched run 1 and the original frozen evidence
exactly, and no replay result at any point had the authority to change the
winner.

**Status: `FIX 7 EVIDENCE REPLAY HARDENED — FROZEN RIDGE SPEC UNCHANGED —
READY FOR COMMIT`.** Not a re-selection, not a new model, not an official
OOF/calibration run.
