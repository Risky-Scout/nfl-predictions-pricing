# Prospective 2026 Strength Promotion Contract

**Status: FROZEN.** This document and
[`outputs/prospective_2026_strength_preregistration.json`](../outputs/prospective_2026_strength_preregistration.json)
freeze -- *before* meaningful 2026 regular-season evidence accumulates --
the exact rules that decide whether the remaining NFL model-strength
statuses can be promoted.

It does **not** change the certified model. No feature selection, no model
selection, no calibration tuning, no betting-threshold tuning, no
retrospective optimisation. The certified production model remains
`ELO_ONLY` / `RIDGE_ALPHA_100` / six certified Elo features / TUE-FRI
card-scoped horizon-as-of state / Fix-8 same-horizon uncertainty / Fix-8
ATS/TOTAL calibration.

| Anchor | Value |
| --- | --- |
| Preregistration schema | `PROSPECTIVE_2026_STRENGTH_V1` |
| Preregistration hash | `a8bfca90d97c54ad42064854d4ed0a1c7115820cae998c5b282a2f9a0dd468e9` |
| Immutable scientific baseline | tag `v2026.1-fix8-certified` = `d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7` |
| Immutable review-ready snapshot | tag `v2026.1-review-ready` = `a26bcb4b1ee6d727df9b6b884ee4fa1933006f62` |
| Implementation base | `cert/prospective-2026-strength-preregistration` @ `64ce7dbf5b12183145e6f96f0efdbca1279f3553` |

The preregistration file is written with the one canonical serialization
`json.dumps(payload, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` (no `default=str`). Its self-hash is computed over the
payload with the hash field removed, then stored back inside the file
(`nfl_hybrid.evaluation.prospective_strength_2026.preregistration_hash`).
Observing 2026 results may never change any frozen value.

## What counts as prospective evidence (Section 2)

**Forecast-time eligibility (item 1).** `target_cutoff_utc` is the
*information as-of* cutoff -- it is **not** a requirement that the
completed forecast file carry `forecast_created_at_utc <=
target_cutoff_utc`. A forecast written seconds or minutes after the cutoff
that used only cutoff-legal inputs stays eligible. What is frozen instead:
every source observation obeys the certified as-of/cutoff rules; market
observations obey their certified timestamp rules relative to
`target_cutoff_utc`; forecast generation corresponds to the certified
TUE/FRI cutoff and occurs inside the certified production due-run
execution window (`nfl_hybrid.production.run_2026.is_within_due_window` /
`current_or_recent_cutoff` -- the existing clock, not a second one);
`forecast_created_at_utc` may be `>= target_cutoff_utc`. A forecast
created outside the permitted execution window, or whose creation maps to
a different card cutoff than it claims, is prospectively ineligible
(`FORECAST_OUTSIDE_EXECUTION_WINDOW`).

Every eligible record must carry: `game_id`, `horizon`, `season`,
`season_type`, `target_cutoff_utc`, `forecast_created_at_utc`,
`git_commit`, certified baseline SHA, operational model spec hash,
feature-state / semantics hash, prediction hash, and -- for any
market-relative metric -- a `market_state_hash` (legacy name
`market_snapshot_hash`). Outcomes are attached later through the immutable
result-attachment path; the original forecast payload is never rewritten.

**Frozen 2026 population (item 2).** Prospective strength evidence is
limited to `season == 2026` and `season_type in {REG, POST}`. **PRESEASON
never enters** sample maturity, calibration scoring, sportsbook
comparisons, model-family stability, profitability, or season-final
completion. `season` + `season_type` are persisted as operational
provenance on every production forecast/evaluation record (no
scientific-model behaviour change); a record that cannot prove its
population membership is excluded, never silently scored.

**Market-state provenance hash (item 3).** `market_state_hash` = SHA-256
over `canonical_json` of the **exact persisted** payload
`{"ATS": <consensus market dict or null>, "TOTAL": <consensus market dict
or null>}` (each market dict = `eligible_books`, `consensus_line`,
`consensus_novig_probability`, `bookmaker_keys`,
`selected_returned_snapshot_timestamps`, `min_observation_age_hours`,
`max_observation_age_hours`, `consensus_method`). Volatile provenance
(`created_at_utc`, `run_id`, `git_commit`), the model's probabilities and
any outcome are excluded by construction. The producer
(`nfl_hybrid.production.run_2026.run_horizon_batch`) persists this
**explicit** hash **at forecast time** in BOTH the forecast-ledger
prediction payload (`prediction.market_state_hash`) and the
evaluation-ledger record (top-level `market_state_hash`); the two must
agree. The reporter reconstructs the payload from the immutable record
alone (it never needs a later mutable raw market file), recomputes the
hash, and requires **exact equality** before using a row for point
forecasting vs sportsbook, sportsbook probability edge, or betting
evidence. A missing, malformed, or mismatching hash makes the row
ineligible for every market-relative metric
(`MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE` / `HASH_MISMATCH`). There is
**no** derive-and-accept fallback.

A row is excluded **only** for a documented integrity / eligibility
reason (`FORECAST_IMMUTABILITY_VIOLATION`, `SCHEMA_DRIFT`,
`IDENTIFIER_FAILURE`, `HASH_MISMATCH`, `MISSING_CONTEMPORANEOUS_MARKET_EVIDENCE`,
`FORECAST_OUTSIDE_EXECUTION_WINDOW`, `OUT_OF_PROSPECTIVE_POPULATION`). A
row is **never** excluded because its prediction was poor.

## Sample-maturity firewall (Section 3)

Unique completed `game_id` values (season 2026, REG+POST only):

| Range | Stage |
| --- | --- |
| 0-63 | `INSUFFICIENT_PROSPECTIVE_SAMPLE` |
| 64-127 | `DESCRIPTIVE_ONLY` |
| 128-199 | `INTERIM_EVIDENCE` |
| >= 200 | `PROMOTION_ELIGIBLE` |

No sportsbook-edge or profitability status may reach `MET_STRONGLY`
before 200 unique completed prospective games.

**`SEASON_FINAL_CONFIRMATORY` (item 9)** is true **only** when the
canonical 2026 REG+POST schedule population is available, every eligible
REG+POST scheduled game has a final result attached, and no remaining
2026 REG+POST game is scheduled / unresolved. Season completion is never
inferred from a calendar date or a sample count alone; if schedule
completeness cannot be proven, `season_final_confirmatory = false`.

## Common inference method (Section 4)

Paired comparisons use a **game-cluster bootstrap**: 10,000 resamples,
seed `20260829`, cluster = `game_id`. Unique games are resampled with
replacement and every associated TUE/FRI and ATS/TOTAL row is carried
together. 95% percentile confidence intervals. The seed, cluster,
resample count and interval method are frozen and may not change after
2026 results are observed.

## Calibration metrics (Section 5)

Streams `ATS_TUE`, `ATS_FRI`, `TOTAL_TUE`, `TOTAL_FRI`, Fix-8 event/push
semantics exactly, scored on non-push conditional-event rows. Metrics: log
loss, Brier, ECE, AUC, sharpness (descriptive). **ECE** uses 10 fixed
equal-width bins `[0.0,0.1) ... [0.9,1.0]`; probability exactly `1.0`
belongs to the final bin; no adaptive bins. Calibration intercept/slope
are reported diagnostically -- no calibration refit for evaluation, only
the exact frozen production calibrators that produced the prospective
probabilities. Rows with `calibration_status != CALIBRATED` never enter
calibrated scoring: `CALIBRATION_IMPROVES_RAW_PROBABILITIES`,
`ABSOLUTE_PROBABILITY_QUALITY` (log loss / Brier / ECE / AUC / slope /
intercept / material-inconsistency) and `DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE`
score the **calibrated** production probability only -- there is **no**
fallback to the raw probability for a not-ready row. Point-forecast
metrics still use the point forecast (not a probability). The
calibrated-metric sample size is reported separately from unique
completed-game maturity, so overall maturity can never manufacture a
calibrated sample that does not exist. No numeric `calibrated_*` value may
exist when calibration is not ready (the merged fail-closed production
behaviour is authoritative, and the reporter re-checks it -- a numeric
`calibrated_*` paired with a non-`CALIBRATED` status is counted as a
`PRODUCTION_READY_PIPELINE` violation, not silently scored).

## Promotion gates

Each gate reaches `MET_STRONGLY` only if **every** listed condition holds
**and** sample maturity is `PROMOTION_ELIGIBLE`. Full numeric thresholds
live in the preregistration JSON under `gates`.

- **`CALIBRATION_IMPROVES_RAW_PROBABILITIES`** (Section 6): calibrated log
  loss / Brier / ECE below raw in all four streams; ALL_FOUR pooled
  calibrated-minus-raw log-loss and Brier 95% CIs entirely below zero; no
  stream with an adverse calibrated-minus-raw log-loss point estimate.
  Also reports `ATS_CALIBRATION_EVIDENCE` / `TOTAL_CALIBRATION_EVIDENCE`
  in `{STRONG, SUGGESTIVE, MIXED, INSUFFICIENT}` (item 5 -- frozen per
  market, over its unique completed eligible games and its TUE/FRI
  streams): **INSUFFICIENT** = fewer than 64 unique completed eligible
  games. **STRONG** = >= 200 unique completed games AND both constituent
  horizon streams improve raw on LL, Brier, and ECE AND pooled
  calibrated-minus-raw LL and Brier 95% CI upper bounds < 0 AND neither
  horizon has an adverse LL point estimate. **SUGGESTIVE** = >= 64 games,
  not STRONG, pooled calibrated-minus-raw LL < 0 and Brier < 0, and
  neither constituent horizon has an adverse LL point estimate. **MIXED**
  = >= 64 games and neither STRONG nor SUGGESTIVE.
- **`ABSOLUTE_PROBABILITY_QUALITY`** (Section 7): ALL_FOUR pooled non-push
  rows -- log loss and its upper 95% CI below `ln 2`
  (`0.6931471805599453`); Brier and its upper 95% CI below `0.250`; ECE
  <= `0.050`; calibration slope in `[0.80, 1.20]`; |intercept| <= `0.05`;
  AUC lower 95% bound > `0.50`; ATS and TOTAL not **materially
  inconsistent** with the ALL_FOUR conclusion. Sharpness is descriptive
  only. If the AUC criterion fails, the row cannot be `MET_STRONGLY` even
  with excellent calibration. **Materially inconsistent (item 6)** is
  frozen: ATS-pooled and TOTAL-pooled are materially inconsistent if
  **either** market has **any** of log loss `>= 0.6931471805599453`,
  Brier `>= 0.250`, AUC `<= 0.50`, or ECE `> 0.075`; both markets must
  avoid all four for ALL_FOUR absolute quality to become `MET_STRONGLY`.
- **`POINT_FORECASTING_VS_SPORTSBOOK`** (Section 8): contemporaneous rows
  only, no closing-line hindsight. Pooled margin and total RMSE each at
  least `0.25` better than the sportsbook consensus; pooled squared-error
  delta 95% CIs entirely below zero; neither TUE nor FRI worse than the
  book by more than `0.10` on either target.
- **`DEMONSTRATED_SPORTSBOOK_PROBABILITY_EDGE`** (Section 9): both ATS
  pooled and TOTAL pooled satisfy `log_loss_delta <= -0.002` with upper
  95% CI < 0, `brier_delta <= -0.001` with upper 95% CI < 0, and no
  constituent TUE/FRI stream adverse (log-loss delta > `+0.002` or Brier
  delta > `+0.001`). ATS and TOTAL are never pooled together to rescue a
  failing market.
- **`MODEL_FAMILY_STABILITY`** (Sections 10-11): shadow-only. Ridge family
  selected in TUE, FRI and pooled; Ridge in the one-SE set in every
  **adequate** prespecified scope; HGBR selected in none; `RIDGE_ALPHA_100`
  the selected Ridge alpha in TUE/FRI/pooled; 10,000 game-cluster
  bootstrap `P(Ridge family) >= 0.80`, `P(Ridge in one-SE set) >= 0.90`,
  `P(HGBR) <= 0.05`. **"Adequate" is frozen (item 4):**
  `ADEQUATE_MODEL_FAMILY_SCOPE_MIN_UNIQUE_GAMES = 64` -- a TUE, FRI,
  pooled, first-third, middle-third, or final-third scope is eligible for
  one-SE / selection conclusions only with at least 64 unique completed
  `game_id` values in that scope; anything smaller is `INSUFFICIENT`. No
  shadow result has production-selection authority during 2026.
- **`PROVEN_PROFITABLE_BETTING_MODEL`** (Sections 12-15): DISABLED for
  profitability until a deterministic executable-book selection policy is
  separately configured and hash-frozen *before* the first eligible live
  wager (`BETTING_RULE_STATUS = NOT_ACTIVATED_FOR_PROFITABILITY`). Once
  active: executable-book policy frozen before the first wager; all
  scored wagers used actual forecast-time executable offers; betting rule
  unchanged; >= 150 settled non-push wagers; >= 200 unique completed
  games; realised ROI >= `+2.0%` with game-cluster bootstrap lower 95% CI
  > 0; net units > 0; median CLV > 0 when valid; ATS and TOTAL each
  positive (or one positive, the other statistically non-adverse); no
  single game > 10% of net profit; no integrity violation.
  **"Statistically non-adverse" is frozen (item 7):** the upper bound of
  that market's game_id-cluster bootstrap 95% ROI CI is `>= 0` **and** its
  realised net units are `>= 0`. A market with negative realised net units
  is never non-adverse.

  **Executable-book policy lock (item 8).** Beyond the config hash, a
  season-long first-wager lock is written exactly once -- when the first
  genuinely eligible executable wager is recorded (first-write-wins) -- to
  `$NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/executable_book_policy_lock.json`.
  It carries the policy hash, the `BETTING_RULE_2026_V1` hash
  (`betting_rule_2026_v1_hash`), the first eligible wager identity, the
  lock timestamp, and the git commit. This preregistration does **not**
  create that file, and does not create `config/executable_books_2026.json`.
  Any later attempt to evaluate/write a wager under a different policy
  hash or betting-rule hash **durably records** a `POLICY_HASH_MISMATCH` /
  `BETTING_RULE_HASH_MISMATCH` event in the append-only integrity-event
  ledger
  `$NFL_MODEL_ARTIFACT_ROOT/production-2026/betting-policy/integrity-events/`
  (schema `BETTING_POLICY_INTEGRITY_EVENT_V1`; each event carries
  `event_type`, `observed_at_utc`, locked + attempted policy/betting-rule
  hashes, the first-wager identity from the lock, the attempted-wager
  identity, `git_commit`, and a deterministic `event_content_hash`;
  append-only / first-write-wins; an identical repeated conflict is
  idempotent) **before** the `BETTING_INTEGRITY_CONFLICT` hard-stop is
  raised. Once **any** valid conflict event has ever been persisted, the
  2026 evidence estate is **permanently contaminated**: the profitability
  reporter scans the ledger on every run and
  `PROVEN_PROFITABLE_BETTING_MODEL` can never become `MET_STRONGLY` -- even
  if `config/executable_books_2026.json` is reverted, the original policy
  hash is restored, or a later call reuses the original betting-rule hash.
  There is **no** automatic clear-contamination path
  (`nfl_hybrid.evaluation.prospective_strength_2026.write_executable_book_policy_lock`
  / `record_betting_integrity_event` / `betting_integrity_estate_status` /
  `verify_executable_book_policy_lock`).
- **Operational gates** (Sections 16-17): `CHRONOLOGY_LEAKAGE_CONTROL`
  and `REPRODUCIBILITY_AUDITABILITY` stay `MET_STRONGLY` only with zero
  future-data / cutoff / outcome-contamination / mutable-forecast
  violations and 100% provenance coverage; `PRODUCTION_READY_PIPELINE`
  stays `MET_STRONGLY` only with >= 99% of scheduled eligible batches
  ending `SUCCESS` or a documented fail-closed status, zero silent
  fallback, zero overwritten forecasts, zero fabricated market states,
  zero numeric `calibrated_*` when not `CALIBRATED`, zero credential
  leakage. Any observed violation demotes the row to
  `DEMOTED_VIOLATION_OBSERVED`.

## Betting rule (`BETTING_RULE_2026_V1`, Sections 12-15)

Markets `ATS`/`TOTAL`; horizons `TUE`/`FRI`; `1.0` unit flat stake;
minimum expected value `+0.025`; at most one ATS and one TOTAL wager per
game/horizon; push = 0 units P/L; if both sides of a market qualify,
`ABSTAIN` and flag `BETTING_INTEGRITY_CONFLICT`. No Kelly, no threshold
optimisation, no chasing, no manual overrides, no best-historical-cutoff,
no strategy switching, no post-cutoff line shopping.

A wager is prospectively evaluable only if the forecast-time ledger
carries an actual executable offer (bookmaker id, market, side, line,
American/decimal odds, quote timestamp, snapshot timestamp). No later
price substitution, no closing-line substitution, no synthetic -110, no
best-book hindsight.

`config/executable_books_2026.json` is **not** created here. Until it
exists and is hash-frozen, profitability status is `NOT_ESTABLISHED`
with reason `EXECUTABLE_BOOK_POLICY_NOT_FROZEN`. When supplied it must be
a fixed ordered list of eligible books or another deterministic
predeclared selection rule; its hash is locked on the first eligible live
wager (`executable_book_policy_lock.json`, see item 8 above) and is
immutable thereafter.

## Sportsbook sign / event orientation (item 10)

The existing certified market convention is unchanged. It is pinned here
as regression contract: a home spread of `-3.5` corresponds to a
sportsbook-implied **home** margin of `+3.5`; the ATS model probability
and the ATS no-vig probability refer to the **same** event/side; the
TOTAL model probability and the TOTAL no-vig probability refer to the
**same** OVER/UNDER orientation; and no sign flip can make the point-market
comparison appear favorable. `tests/test_prospective_strength_2026.py`
carries direct orientation regressions.

## Shadow model-family ledger (Sections 10-11, 20)

A separate evidence path only. The canonical production forecast stays
`RIDGE_ALPHA_100` and never changes, is never selected, blocked, or
overwritten by shadow output. `scripts/run_2026_shadow_model_family.py`
replays the certified card-scoped chronological OOF batching (training
rows = those with `result_available_at_utc < target_cutoff_utc`) for each
of the exact frozen Fix-7 candidates -- `RIDGE_ALPHA_0_1`,
`RIDGE_ALPHA_1`, `RIDGE_ALPHA_10`, `RIDGE_ALPHA_100`, `HUBER_FIXED`,
`HGBR_INCUMBENT` -- using the same six certified Elo features and the same
train-only preprocessing, and writes immutable first-write-wins records
under
`$NFL_MODEL_ARTIFACT_ROOT/production-2026/shadow-model-family-ledger/`
keyed by `(game_id, horizon, candidate, target_cutoff_utc)`. No outcome
column is ever written. A shadow fit that fails is recorded
`SHADOW_FIT_UNAVAILABLE` and cannot affect anything else. Run it against
an already-elapsed games population only.

## Reporter (Sections 18-19)

`scripts/report_2026_strength_scorecard.py` /
`nfl_hybrid.evaluation.prospective_strength_2026.build_scorecard` read
ONLY the immutable forecast ledger, the immutable evaluation ledger +
attached `*.result.json` records, the shadow ledger, the executable-price
ledger (if activated), the run manifests, and
`config/executable_books_2026.json` (presence + hash). They never read a
retrospective 2020-2025 evaluation artifact to score a 2026 gate.
Historical status is displayed as context only (`current_evidence_supported_labels`).

Output:
`$NFL_MODEL_ARTIFACT_ROOT/production-2026/prospective-strength/PROSPECTIVE_2026_STATUS_SCORECARD.{json,md}`.
Every row reports status, sample size, evidence stage, point estimate,
confidence interval where applicable, gate booleans, failure reasons, the
preregistration hash, and the data-through timestamp.

Before enough results exist the reporter does not fail: performance rows
read `INSUFFICIENT_PROSPECTIVE_SAMPLE` / `NOT_DEMONSTRATED` /
`NOT_ESTABLISHED`, and no empty or immature sample can produce `MET` or
`MET_STRONGLY`.

## Current evidence-supported labels (Section 23 -- preserved, context only)

| Row | Label |
| --- | --- |
| Chronology / leakage control | `MET_STRONGLY` |
| Reproducibility / auditability | `MET_STRONGLY` |
| Model-family stability | `MIXED` |
| Calibration improves raw probabilities | `MET_STRONGLY` (historical evidence; prospective confirmation pending) |
| Absolute probability quality | `MODERATE` |
| Point forecasting vs sportsbook | `NOT_MET` |
| Demonstrated sportsbook probability edge | `NOT_DEMONSTRATED` |
| Production-ready pipeline | `MET_STRONGLY` |
| Proven profitable betting model | `NOT_ESTABLISHED` |

No row is promoted merely because this contract and reporter were added.

## Commands

```bash
# Prospective scorecard (safe with an unset / empty artifact root)
PYTHONPATH=src python3 scripts/report_2026_strength_scorecard.py \
  --data-through-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Shadow model-family ledger (already-elapsed games only)
PYTHONPATH=src python3 scripts/run_2026_shadow_model_family.py --horizon ALL

# Book-policy activation: author config/executable_books_2026.json (fixed
# ordered eligible-book list or another deterministic rule); its hash locks
# on the first eligible live wager.

# Result attachment: scripts/run_2026_production_card.py's attach path
# (nfl_hybrid.production.run_2026.attach_result) -- strict
# result_available_at_utc < attachment_run_time, never mutates a forecast.
```
