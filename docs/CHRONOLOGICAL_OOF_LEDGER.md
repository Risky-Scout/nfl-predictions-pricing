# Chronological OOF Ledger and Uncertainty Methodology (Fix 3)

Implementation: `nfl_hybrid.evaluation.chronological_oof`.
Tests: `tests/test_chronological_oof.py` (unconditional adversarial suite),
`tests/test_chronological_oof_extended.py` (`extended_data`, real 2020-2025
estate).

## Why this exists

Fix 2 (`nfl_hybrid.features.pregame_state.build_authoritative_pregame_state`)
fixed leakage in pregame *state* — no state family, and no state family's
lineage, could reference a game's own outcome or a future game. Fix 3 fixes
the analogous problem one level up the pipeline, for *predictions* and for
*predictive uncertainty*:

- a historical prediction is only real evidence of skill if the model that
  produced it never saw the target game (or any later game) during training
  or preprocessing fitting;
- a predictive uncertainty estimate (margin/total residual SD, residual
  correlation) is only honest if it is never estimated from a model's own
  in-sample (fitted-training) residuals, and never contaminated by the very
  residual it is meant to price.

## Pipeline

```
build_authoritative_pregame_state (Fix 2)   -- the only state source
    -> build_oof_feature_matrix               -- one game-level feature row,
                                                  every numeric home/away
                                                  pivoted column from every
                                                  state family, unmodified
    -> generate_oof_predictions               -- expanding-window OOF point
                                                  predictions (phase 1)
    -> attach_outcomes_and_residuals          -- outcomes attached after the
                                                  prediction is fixed (phase 2)
    -> attach_expanding_oof_uncertainty       -- leave-future-out residual
                                                  SD / correlation, per row
```

`run_chronological_oof(games, play_by_play, ...)` wires all four stages
together; each stage is independently callable and tested.

## The estimator

One fixed, explicitly declared estimator/configuration:
`nfl_hybrid.modern.joint_score.JointScoreModel` (already used by
`nfl_hybrid.evaluation.walkforward.expanding_season_backtest`), with the
existing `JointScoreConfig()` defaults. A fresh, unfitted instance is created
and fit once per target game, on that target's own training set only.
Nothing is tuned, searched, or selected: `ChronologicalOOFConfig` only holds
this one estimator's declared config plus three structural floors
(`min_training_games`, `training_ids_cap`,
`target_cutoff_offset_minutes`) — none of them chosen against held-out
performance.

Point predictions are read via `JointScoreModel.predict_means()`, a public
accessor added for this fix. It deliberately never reads
`margin_sd_` / `total_sd_` / `residual_correlation_` — those three attributes
are fit from the model's *own training residuals* for its own internal
probability math, and are never read anywhere in `chronological_oof.py`
(`tests/test_chronological_oof.py::test_fitted_training_residual_attributes_never_used_for_uncertainty`
asserts this directly from the source).

## Expanding-window training policy

For target game G with `scheduled_kickoff_utc = k`:

- `target_cutoff_utc(G) = k - target_cutoff_offset_minutes` (default 10
  minutes — the same "kickoff minus 10 minutes" convention already declared
  as the official snapshot in `governance/contracts.py`). This is about when
  G itself needs to be forecast; it is unaffected by the correction below.
- The training set is every other game whose **result** was already known --
  `result_available_at_utc < target_cutoff_utc(G)` — evaluated by actual
  timestamp, never by row position and, critically, **never by kickoff order
  alone**. This excludes G itself, every future game, every other game
  sharing G's own kickoff (e.g. the same Sunday slate), and any game whose
  result had not yet resolved by G's cutoff even if that game kicked off
  earlier than G.
- Preprocessing (the imputer/scaler inside `JointScoreModel`) is fit only on
  that same training set, inside the per-target `model.fit(...)` call.
- If `training_game_count < min_training_games`, the row is marked
  `status = "INSUFFICIENT_WARMUP"` with `predicted_margin` /
  `predicted_total` left `NaN` — never silently fit on too little history,
  and never dropped from the ledger (so warmup gating itself stays auditable).

### Correction: result availability, not just kickoff order

The module's first version used `scheduled_kickoff_utc < target_cutoff_utc`
as the training-eligibility rule. That is unsound: kickoff order is not
result-availability order. A game can kick off before another game and still
have its official result confirmed *later* than that other game's own
forecast cutoff (a Monday Night Football game finishing late, a weather
delay, or simply a later production forecast-batch boundary). Relying on
kickoff order alone would let a still-unresolved game quietly enter another
target's training set or uncertainty pool.

`compute_result_available_at_utc(games)` computes a conservative
`result_available_at_utc` per game instead, in two layers, each of which can
only **delay** eligibility, never advance it — never a fabricated exact
final-whistle timestamp:

1. `nfl_hybrid.data.availability.add_postgame_available_at` — this
   repository's own existing, already-documented postgame-availability floor
   (`kickoff + duration_hours`, default 5h; reused unchanged, already used by
   `augmented_matrix.py`'s `historical_replay` mode).
2. Snapped forward to the next production forecast-batch boundary — Tuesday
   or Friday, 12:00 UTC (mid-morning US Eastern, comfortably after any
   Monday/Thursday night game's kickoff-plus-floor). This mirrors the batch
   cadence a live system would actually forecast on: by that boundary, every
   prior game — including Monday Night Football, the latest possible weekly
   kickoff — is unambiguously final. It avoids claiming a precise per-game
   completion time this repository's schema cannot verify (no per-play UTC
   event time, no terminal-game marker; see `augmented_matrix.py`'s own
   `_terminal_play_timestamp`).

`build_oof_feature_matrix` attaches `result_available_at_utc` to every row
automatically. `generate_oof_predictions` **requires** the column and raises
if it is missing, rather than silently falling back to kickoff — the
eligibility rule is an explicit contract, not an inferred convenience.
`assert_available_before` (also reused from `nfl_hybrid.data.availability`)
self-checks, before every `generate_oof_predictions` call returns, that
`max_training_result_available_at_utc < target_cutoff_utc` holds for every
row.

Uncertainty is held to the same standard: `attach_expanding_oof_uncertainty`
and `production_uncertainty` gate residual inclusion on
`result_available_at_utc < cutoff`, not on forecast/processing order — a
residual only enters a target's uncertainty pool once its own game's result
was actually available by that target's cutoff, regardless of which order
the two games were forecast in.

## Ledger schema

### `oof_predictions` (phase 1 — before any outcome is attached)

| column | meaning |
|---|---|
| `game_id` | target game |
| `season`, `week` | target game's season/week |
| `target_cutoff_utc` | this game's own forecast cutoff — when G itself must be predicted |
| `training_as_of_utc` | the as-of timestamp actually used to decide this batch's training-set eligibility. Currently always equal to `target_cutoff_utc` (a batch's training mask is evaluated against its own target cutoff), but a **separate, explicitly-named field** — never silently reused from `target_cutoff_utc` even where the values coincide today |
| `result_available_at_utc` | this game's own conservative result-availability time (`compute_result_available_at_utc`) — a **derived, conservative eligibility timestamp**, never an observed final-whistle time (see `result_availability_basis` below) |
| `max_training_result_available_at_utc` | max `result_available_at_utc` among training games (`NaT` if none) — always `< target_cutoff_utc` for an `"OOF"` row, self-checked via `assert_available_before` before every `generate_oof_predictions` call returns |
| `result_availability_basis` | provenance label for how `result_available_at_utc` was derived; currently always `"HISTORICAL_CONSERVATIVE_BATCH"` (kickoff + `RESULT_AVAILABILITY_DURATION_HOURS`, snapped to the next Tue/Fri batch boundary) — recorded so a reader never mistakes this for a verified/observed finality time |
| `training_game_count` | number of training games |
| `training_game_ids` | sorted tuple of training `game_id`s, capped at `training_ids_cap` (empty + `training_game_ids_truncated=True` above the cap — the count and `max_training_result_available_at_utc` are never truncated, mirroring `state_lineage.py`'s own truncation pattern) |
| `training_game_ids_truncated` | whether the id list above was capped |
| `training_membership_hash` | `sha256` of the sorted, `\|`-joined training `game_id`s — provable membership even when the id list itself is truncated |
| `model_config_hash` | `sha256` of the estimator name/version and every `ChronologicalOOFConfig` field |
| `feature_state_hash` | `sha256` of the feature-column list and every state family's `(transform_name, transform_version)` actually built |
| `status` | `"OOF"` or `"INSUFFICIENT_WARMUP"` |
| `predicted_margin`, `predicted_total` | point predictions (`NaN` if warmup-insufficient) |

### `oof_residual_ledger` (phase 2 — outcomes and residuals attached)

Everything above, plus:

| column | meaning |
|---|---|
| `actual_margin`, `actual_total` | the target game's real outcome |
| `margin_residual`, `total_residual` | `actual - predicted` |
| `margin_residual_sd_oof`, `total_residual_sd_oof` | leave-future-out residual SD (see below) |
| `residual_correlation_oof` | leave-future-out residual correlation |
| `uncertainty_eligible` | whether enough strictly-prior OOF residuals existed to compute the three columns above |

## Uncertainty methodology

`attach_expanding_oof_uncertainty` keeps OOF residuals on their own timeline,
sorted by **`result_available_at_utc`** — not by `target_cutoff_utc` /
forecast-processing order, which is exactly the distinction the result-
availability correction exists to enforce (a game forecast "later" can still
resolve earlier than one forecast "earlier", and vice versa). For row *i*
(ledger rows processed in ascending `target_cutoff_utc` order), a residual
enters the eligible pool only once its own `result_available_at_utc` is
confirmed strictly before row *i*'s `target_cutoff_utc` — via a merge over
two independently-sorted sequences, exact regardless of how kickoff order and
resolution order interleave. A row's own residual can never enter its own
pool (`result_available_at_utc` is always after that same row's own
`target_cutoff_utc` by construction), and a later-resolving residual can
never retroactively affect an earlier row's already-computed uncertainty
(`tests/test_chronological_oof.py::test_target_residual_cannot_change_its_own_uncertainty`,
`::test_future_residual_cannot_change_earlier_uncertainty`,
`::test_residual_excluded_until_available_then_may_enter_later_uncertainty`,
`::test_changing_unresolved_outcome_cannot_affect_current_uncertainty`). Rows
below `min_uncertainty_warmup` eligible prior OOF residuals get `NaN` /
`uncertainty_eligible = False` rather than an estimate from too little data.

`production_uncertainty(ledger, as_of_utc=...)` computes the single scalar
estimate a production/live artifact should actually use: margin/total
residual SD and residual correlation from every `status == "OOF"` row whose
`result_available_at_utc` is strictly before `as_of_utc` — i.e. every
historical OOF residual whose game had actually resolved by that as-of time.
A prediction occurring earlier than `as_of_utc` is not sufficient on its own.

Neither function ever reads `JointScoreModel.margin_sd_` / `.total_sd_` /
`.residual_correlation_`.

## Artifact persistence

`persist_oof_ledger` writes through the existing artifact registry
(`nfl_hybrid.data.external_data`), never a hard-coded path:

| registry key | scope | contents |
|---|---|---|
| `oof_chronological.predictions` | `generated` | phase-1 ledger, parquet |
| `oof_chronological.residual_ledger` | `generated` | phase-2 ledger (with uncertainty columns), parquet |
| `oof_chronological.manifest` | `generated` | feature columns, `feature_state_hash`, `model_config_hash`, `result_availability_basis`, row counts, JSON |

**Deliberately a separate root from the raw historical estate.** These are
`scope="generated"` registry keys: they resolve under
`${NFL_MODEL_ARTIFACT_ROOT}/chronological-oof-2020-2025/`, a distinct
environment variable from `NFL_MODEL_DATA_ROOT` (which `external_data.py`
reserves for the raw/canonical historical estate this pipeline reads
*from* — `backfill-2020-2025/`, `odds-api-history-*/`). A generated,
regenerable pipeline output must never live in the same tree as the raw data
that produced it — mixing the two would make them impossible to distinguish
by location, and would risk a bulk-copy of one silently dragging the other
along. `nfl_hybrid.data.external_data.artifact_root()` resolves this second
root the same way `external_root()` resolves the first: no machine-specific
default, `NFL_MODEL_ARTIFACT_ROOT` (or an explicit `root_override`) is the
only source. Nothing from this fix is committed to the repository itself,
except the small proof artifact below.

## What this fix deliberately does not do

No feature selection (every numeric pregame-state column built is used,
unconditionally — nothing ranked or dropped), no hyperparameter tuning, no
model-family tournament, no probability calibration, and no change to the
predictive targets (margin, total — the same two `JointScoreModel` already
predicts elsewhere). Model fitting happens only as many times as there are
target games actually evaluated, each time on a fresh, unfitted instance of
the one declared estimator.
