# Model Strength and Limitations

Reviewer-facing summary of the certified `v2026.1-fix8-certified` football
model (commit `d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7`), the subsequent
FINAL REVIEW CERTIFICATION robustness/calibration audit
(`cert/robustness-production-readiness-2026`, preregistration hash
`5076b0caa55b11d4a44d3d18ab187879ee0fbb98a5286b0fbd1315421703e66a`), and
what is and is not established by the evidence to date.

**This audit performed no feature selection, no re-estimation, and no
recalibration.** RIDGE_ALPHA_100 on six frozen Elo features remains the
certified production model regardless of every finding below.

## Certified model specification

- Feature set: `ELO_ONLY` -- six frozen Elo features (home/away ×
  pregame rating, pregame win probability, pregame expected margin),
  card-scoped horizon-as-of (`HORIZON_CUTOFF_ASOF_ELO_V2_CARD_SCOPED`).
- Estimator: `RIDGE_ALPHA_100` (`alpha=100.0, fit_intercept=True,
  solver="svd"`, `StandardScaler` preprocessing), fit separately for
  margin and total.
- Operational horizons: card-scoped TUE (Monday+1, 12:00 America/New_York)
  and FRI (Monday+4, 12:00 America/New_York), DST-aware.
- Calibration: `three_way.logistic_on_logit_conditional_calibrator`, four
  independent streams (ATS_TUE, ATS_FRI, TOTAL_TUE, TOTAL_FRI), fit
  chronologically with strict `result_available_at_utc < target_cutoff_utc`
  eligibility.
- Certified hashes: `horizon_feature_semantics_hash =
  bf0b136dd9b7c7f3741617b6c088926e539406d60b82563839eb1825de9fc72d`;
  `horizon_membership_ledger_hash =
  b095a2bf9177029eb67e1516fbc79b080ef800cfe18b86989254dd4d0a8dae49`;
  `operational_model_spec_hash =
  3b230bfeee3279c0e3ed9b6a7118931c1a5cf08203155be011f362c66ee8d722`;
  Fix-8 `preregistration_hash =
  0d0a1eefffdbb6878e3246d2b5b0fb8f091bebc0eeb9f3869624620fb9e2847e`.

## 2024 football metrics (LOCKED_POST_EXPOSURE_AUDIT)

Fix 7's own outer confirmation (train ≤2023, validate 2024; this season was
previously exposed via Fix 6's outer confirmation, never described as a
pristine holdout): Ridge `primary_score = 13.0049`, `margin_rmse =
13.0630`, `total_rmse = 12.9467` -- all six predeclared tolerance rules
(`overall`, `margin`, `total`, `week1`, `weeks2_4`, `weeks5_plus`) passed
against the best 2024 finalist. This certification's own diagnostic 2024
re-check reuses this evidence verbatim (no new fit) and confirms
`selected_family = RIDGE`, `overall_pass = True`.

## 2025 post-freeze probability metrics

Fix 8's certified 2025 post-freeze chronological replay (descriptive /
non-selection only), by stream:

| Stream | n scored | Raw log loss | Cal. log loss | Raw Brier | Cal. Brier | Raw ECE | Cal. ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATS_TUE | 172 | 0.7194 | 0.6970 | 0.2628 | 0.2519 | 0.1460 | 0.1059 |
| ATS_FRI | 238 | 0.7189 | 0.6943 | 0.2621 | 0.2506 | 0.1004 | 0.0713 |
| TOTAL_TUE | 175 | 0.7198 | 0.6991 | 0.2627 | 0.2530 | 0.1244 | 0.0875 |
| TOTAL_FRI | 239 | 0.7223 | 0.6953 | 0.2633 | 0.2511 | 0.1242 | 0.0918 |

## Model-family robustness (this certification, Sections 2-6)

**Status: `MODEL_FAMILY_STABILITY = MIXED`.** This is a stability *audit*
of the already-certified model, not a reselection -- it has zero authority
to change the production model, and did not.

Method: the frozen Fix-7 candidate registry (RIDGE_ALPHA_0_1/1/10/100,
HUBER_FIXED, HGBR_INCUMBENT) and the frozen Fix-7 selection procedure
(one-SE rule, complexity-tiebreak, "2-of-3-fold" family-beats-family
check) were applied unchanged to a *new*, TUE/FRI-horizon-scoped,
REG+POST diagnostic population built from the same chronological folds
Fix 7 used (A: train≤2020/val2021, B: train≤2021/val2022, C:
train≤2022/val2023) -- reused directly from
`nfl_hybrid.selection.model_family_selection_2026`, never
reimplemented.

Nine predeclared robustness scopes:

| Scope | Description | Best pooled score | One-SE family set | Selected family | Selected Ridge alpha |
|---|---|---|---|---|---|
| S1 | TUE, all folds | HUBER | RIDGE, HUBER | **RIDGE** | RIDGE_ALPHA_100 |
| S2 | FRI, all folds | HUBER | RIDGE, HUBER | **RIDGE** | RIDGE_ALPHA_100 |
| S3 | TUE+FRI pooled, all folds | HUBER | RIDGE, HUBER | **RIDGE** | RIDGE_ALPHA_100 |
| S4 | folds B+C | RIDGE | RIDGE | **RIDGE** | RIDGE_ALPHA_100 |
| S5 | folds A+C | HUBER | HUBER | HUBER | RIDGE_ALPHA_100 |
| S6 | folds A+B | RIDGE | RIDGE, HUBER | **RIDGE** | RIDGE_ALPHA_100 |
| S7 | validation season 2021 only | HUBER | HUBER | HUBER | RIDGE_ALPHA_100 |
| S8 | validation season 2022 only | RIDGE | RIDGE | **RIDGE** | RIDGE_ALPHA_100 |
| S9 | validation season 2023 only | HUBER | HUBER | HUBER | RIDGE_ALPHA_100 |

Ridge is selected (and stays within one SE of the best candidate) in **6
of 9** scopes, including all three primary scopes (S1/S2/S3). Huber wins
the other 3 -- all single-fold or two-fold subsets, i.e. the smallest,
noisiest slices. **HGBR is never selected and never inside the one-SE set
in any of the 9 scopes.**

10,000-replicate game_id-cluster bootstrap (854 unique game_id clusters,
seed 20260826, applied to scope S3, no refitting inside the bootstrap --
the frozen selection formulas are reapplied to resampled per-game losses):

| Quantity | Value |
|---|---:|
| P(RIDGE family selected) | 68.1% |
| P(HUBER family selected) | 31.9% |
| P(HGBR family selected) | 0.0% |
| P(RIDGE_ALPHA_100 selected \| RIDGE family) | 97.0% of Ridge-family replicates |
| P(RIDGE_ALPHA_100 is the final selected candidate) | 67.0% |
| P(RIDGE inside the one-SE family set) | 67.6% |
| P(HGBR inside the one-SE family set) | 0.0% |

**Why `MIXED` and not `MET_STRONGLY`**: the gate's own predeclared
thresholds (P(Ridge) ≥ 0.80, P(Ridge in one-SE) ≥ 0.90, Ridge inside the
one-SE set in *all* 9 scopes) are not weakened, and three of them fail --
S5/S7/S9 exclude Ridge from the one-SE set, and both bootstrap
probabilities fall short (68.1% and 67.6% vs. the 80%/90% bars). Every
other criterion passes cleanly, and HGBR is uncompetitive by a wide,
unambiguous margin in every scope and every bootstrap replicate. Net
honest read: **Ridge vs. Huber is a genuinely close, somewhat
fold-sensitive contest that the frozen complexity-tiebreak rule resolves
in Ridge's favor; Ridge is never seriously threatened by HGBR.**

2024/2025 diagnostic-only re-check (no selection authority; 2025 uses a
dedicated train≤2024/predict=2025 fit outside the Fix-7 selection
firewall, since 2025 may never enter *selection*): on both TUE and FRI,
Ridge remains competitive with Huber (primary-score deltas within ~10%)
and clearly ahead of HGBR in 2025, consistent with the certified 2024
audit and with the S1-S9 findings above.

## Calibration improves raw probabilities (Sections 7-11)

**Status: `CALIBRATION_IMPROVES_RAW_PROBABILITIES = MET_STRONGLY`.**
Reuses the certified Fix-8 2025 raw and calibrated probability ledgers
verbatim -- no recalibration, no refit.

Paired calibrated-minus-raw deltas (2025 CALIBRATED rows only; negative =
improvement):

| Stream | n | Log-loss Δ | Brier Δ | Raw ECE | Cal. ECE |
|---|---:|---:|---:|---:|---:|
| ATS_TUE | 172 | -0.0223 | -0.0109 | 0.1460 | 0.1059 |
| ATS_FRI | 238 | -0.0246 | -0.0116 | 0.1004 | 0.0713 |
| TOTAL_TUE | 175 | -0.0207 | -0.0098 | 0.1244 | 0.0875 |
| TOTAL_FRI | 239 | -0.0271 | -0.0123 | 0.1242 | 0.0918 |

10,000-replicate game_id-cluster bootstrap (seed 20260826), 95% CI:

| Pooling | Log-loss Δ 95% CI | Brier Δ 95% CI | CI entirely below 0? |
|---|---|---|---|
| ATS (TUE+FRI) | [-0.0394, -0.0088] | [-0.0188, -0.0041] | Yes / Yes |
| TOTAL (TUE+FRI) | [-0.0486, +0.0005] | [-0.0228, +0.0007] | No / No |
| ALL FOUR pooled | [-0.0389, -0.0095] | [-0.0184, -0.0043] | **Yes / Yes** |

All four streams improve on every point-estimate metric (log loss, Brier,
ECE), and the ALL-FOUR pooled 95% CI is entirely below zero for both
log-loss and Brier -- the strong gate's point-estimate and pooled-CI
criteria are all met, and no stream is materially adverse.

- **ATS_CALIBRATION_EVIDENCE: `STRONG`** (both point estimates negative,
  pooled CI entirely below zero).
- **TOTAL_CALIBRATION_EVIDENCE: `SUGGESTIVE`** (both point estimates
  negative, but the TOTAL-only pooled CI includes zero -- the improvement
  is real on average but not yet CI-robust when TOTAL is isolated from
  ATS).

Calibration improvement is a statement about probability quality, not
about sportsbook edge -- see below.

## Absolute probability quality

Computed on the same certified 2025 CALIBRATED rows used above (full
detail: `$NFL_MODEL_ARTIFACT_ROOT/final-review-certification-2026/absolute_quality_and_market_comparison.json`):

| Stream | n | AUC | Sharpness (mean \|p-0.5\|) | Discrimination label |
|---|---:|---:|---:|---|
| ATS_TUE | 172 | 0.406 | 0.009 | INCONCLUSIVE |
| ATS_FRI | 238 | 0.453 | 0.013 | INCONCLUSIVE |
| TOTAL_TUE | 175 | 0.474 | 0.067 | INCONCLUSIVE |
| TOTAL_FRI | 239 | 0.461 | 0.018 | INCONCLUSIVE |

**Said plainly: AUC is at or below 0.5 for all four streams.** This model
shows **no measurable discrimination** on ATS/TOTAL outcomes beyond
calibration -- consistent with these being efficiently-priced,
close-to-coin-flip markets, and consistent with the "Market-relative
scorecard" finding below (no probability edge over the no-vig market
either). This is not spun as a strength: the honest label is
`INCONCLUSIVE`/no-discrimination, not "weak-but-real."

Calibration-slope/intercept regression (logit(p) → y) was also computed
but is **not reported as a reliable number**: sharpness is extremely low
(calibrated probabilities cluster tightly around 0.5, mean absolute
deviation 0.01-0.07), so the slope regression is fit on very little
logit-scale variance and returns numerically unstable coefficients
(magnitude 4-14) that do not have a trustworthy interpretation at this
sample size. Combined with Brier (~0.25) and log-loss (~0.70) sitting
close to a constant-0.5 baseline, the overall absolute-quality read is:

**MODERATE** -- calibration is real and directionally reliable (see
above), but absolute discriminative sharpness is not established; this is
a well-calibrated-but-not-sharp forecaster on these markets, not a strong
discriminator.

## Market-relative scorecard

Point forecast (model vs. sportsbook, 2025, RMSE in points; sign
convention verified numerically: ATS `consensus_line` is the home spread,
`implied_margin = -consensus_line`; TOTAL `consensus_line` is the total
line directly):

| Horizon/Market | n | Model RMSE | Market RMSE | Model − Market |
|---|---:|---:|---:|---:|
| ATS_TUE | 177 | 13.761 | 13.343 | +0.418 |
| TOTAL_TUE | 176 | 13.634 | 13.158 | +0.476 |
| ATS_FRI | 242 | 12.979 | 12.472 | +0.507 |
| TOTAL_FRI | 242 | 13.879 | 13.220 | +0.660 |

The market beats the model on point forecasts in all four combinations,
consistently by roughly 0.4-0.7 RMSE points. **`POINT_FORECASTING_VS_SPORTSBOOK
= NOT_MET`.**

Probability forecast (calibrated model minus no-vig market, paired,
10,000-rep game_id-cluster bootstrap, seed 20260826):

| Stream | n | Log-loss Δ (model − market) | 95% CI |
|---|---:|---:|---|
| ATS_TUE | 172 | +0.0040 | [-0.00003, +0.0081] |
| TOTAL_TUE | 175 | +0.0058 | [-0.0141, +0.0258] |
| ATS_FRI | 238 | +0.0002 | [-0.0040, +0.0042] |
| TOTAL_FRI | 239 | +0.0019 | [-0.0027, +0.0067] |

No stream's CI is entirely below zero (i.e. the model never robustly
beats the market), and no stream's CI is entirely above zero either (the
model is not robustly worse) -- ATS_FRI and both TOTAL streams are
essentially statistical ties with the market; ATS_TUE's point estimate
favors the market but its CI lower bound sits at essentially zero.
**`SPORTSBOOK_PROBABILITY_EDGE = NOT_DEMONSTRATED`.**

This is the predeclared-acceptable, expected outcome: a six-feature
Elo-only Ridge model is not expected to out-forecast an efficient,
high-information betting market on either point forecasts or calibrated
probabilities, and this certification does not manufacture a gate to
claim otherwise. The genuinely notable, positive finding remains upstream
of this comparison: calibration measurably improves the model's *own* raw
probabilities (see "Calibration improves raw probabilities" above) --
that is a real, demonstrated improvement in probability quality, entirely
independent of whether the result also beats the market.

## Model-family robustness result (summary)

See "Model-family robustness" above --
`MODEL_FAMILY_STABILITY = MIXED`, not weakened, fully evidenced.

## Rejected post-freeze research

Three lines of research were tested after the Fix-8 freeze and did **not**
displace the certified ELO_ONLY / RIDGE_ALPHA_100 baseline:

- **vNext (Ridge/Huber blending + market anchoring + alternative
  calibration)** -- `vnext-sprint-2026`. The frozen blend weight selection
  landed at `lambda_margin = lambda_total = 0.0` (i.e. pure Ridge, zero
  weight on any blended/market-anchored component) --
  `A_model_family_stability: RIDGE_REMAINS_PREFERRED`,
  `B_point_forecast_vs_market: NO_INCREMENTAL_POINT_FORECAST_VALUE_DEMONSTRATED`.
  An alternative calibration family (Beta calibration) showed
  `C_calibration: BETA_CALIBRATION_SHOWS_ROBUST_IMPROVEMENT` in this research
  but was not adopted into the certified baseline (Fix 8's logistic-on-logit
  calibrator remains certified).
- **QB lagged-depth features** -- `qb-lagged-depth-feature-selection-2026`.
  `selected_candidate: ELO_ONLY`, `selection_reason:
  NEITHER_QB_CANDIDATE_PASSED_ADOPTION_GATE`.
- **`TEAM_SCORE_STATE_V1.1`** -- `team-score-state-v1-2026` (the
  V0 predecessor was fully invalidated on a separate operator-contract
  mismatch and contributes no evidence here). `selected_candidate:
  ELO_ONLY`, `selection_reason: NEITHER_SCORE_CANDIDATE_PASSED_ADOPTION_GATE`
  -- both score-state candidates improved the pooled point estimate but
  failed the predeclared `bootstrap_ci_entirely_below_zero` robustness
  condition (fold- and horizon-level checks passed; the bootstrap
  robustness check did not).

The common pattern across all three: point-estimate improvements existed
in research, but none passed its own predeclared statistical-robustness
adoption gate. This is exactly the discipline this certification itself
applies to the model-family and calibration questions above.

## Sportsbook edge and profitability -- plain statements

- **Sportsbook probability edge is not demonstrated.** See "Market-relative
  scorecard" above.
- **Profitability is not established.** No valid prospective,
  executable-price betting study exists as of this certification. Nothing
  in this document, the certified 2024/2025 backtests, or the calibration-
  improvement evidence constitutes evidence of profitability -- backtest
  calibration quality and realized betting return are different questions,
  and only the latter can demonstrate profitability, honestly, only
  prospectively.

## Prospective 2026 validation plan

An immutable forecast ledger, append-only run manifest, and a "no future
outcome columns at forecast time" evaluation ledger are live under
`$NFL_MODEL_ARTIFACT_ROOT/production-2026/` (see
[`PRODUCTION_RUNBOOK_2026.md`](PRODUCTION_RUNBOOK_2026.md)). Results may
only be attached once genuinely available
(`result_available_at_utc < attachment_run_time`, strict), and attachment
never mutates the original forecast. This is the only path to a future,
honest claim of sportsbook edge or profitability -- accumulated
forecast-before-result across the 2026 season, reported by
`scripts/report_2026_prospective_performance.py`, which reports
`INSUFFICIENT_PROSPECTIVE_SAMPLE` rather than force a conclusion from too
few games.

## Prospective 2026 strength promotion contract

The exact rules that will decide whether any row above can be *promoted*
on 2026 evidence are now frozen -- before meaningful 2026 regular-season
results exist -- in
[`PROSPECTIVE_VALIDATION_2026.md`](PROSPECTIVE_VALIDATION_2026.md) and
[`outputs/prospective_2026_strength_preregistration.json`](../outputs/prospective_2026_strength_preregistration.json)
(schema `PROSPECTIVE_2026_STRENGTH_V1`, preregistration hash
`a8bfca90d97c54ad42064854d4ed0a1c7115820cae998c5b282a2f9a0dd468e9`). It
freezes the sample-maturity firewall (no edge/profitability status
`MET_STRONGLY` before 200 unique completed prospective games), one
game-cluster bootstrap (10,000 resamples, seed `20260829`, cluster
`game_id`, percentile 95% CI), fixed-bin ECE (10 fixed `[0.0,0.1) ...
[0.9,1.0]` bins), the per-gate `MET_STRONGLY` conditions for calibration
improvement / absolute probability quality / point-forecasting-vs-book /
sportsbook probability edge / model-family stability / profitability, and
the `BETTING_RULE_2026_V1` betting rule (profitability DISABLED until a
deterministic executable-book policy is separately hash-frozen). It does
**not** change the certified model -- no feature selection, model
selection, calibration tuning, betting-threshold tuning, or retrospective
optimisation. The reporter
(`scripts/report_2026_strength_scorecard.py` /
`nfl_hybrid.evaluation.prospective_strength_2026`) reads only the
immutable prospective ledgers, never a retrospective 2020-2025 artifact,
and never turns an empty or immature sample into a `MET` / `MET_STRONGLY`.
The labels in this document are preserved as historical context until
prospective evidence legitimately changes them.

## Production readiness

Canonical entrypoint `scripts/run_2026_production_card.py`. As of this
certification (2026-08-26), a real `--preflight` run in this environment
reports `READY_WAITING_FOR_FIRST_DUE_CUTOFF`: every piece of
infrastructure needed right now is genuinely ready (certified hashes
match, the Fix-8 calibration seed is present, historical data resolves,
ledger directories are writable, the git commit is identified) -- the
only thing not yet available is the live 2026 schedule/market itself,
which is expected pre-season (the historical `backfill.games` source
covers 2020-2025 only; no live market source is registered yet).
`THE_ODDS_API_KEY` and `NFL_LIVE_DATA_ROOT` are also unset in this
environment and will need to be set once the season is live. All
ledger/manifest/immutability/result-attachment mechanics are implemented
and tested (25 focused tests, including a real historical 2024
week-3 TUE/FRI integration run -- see
`tests/test_production_card_2026.py`). Scheduling is
`OPERATOR_SCHEDULE_READY` (a GitHub Actions workflow targeting a
self-hosted runner label, plus a documented launchd/cron fallback), not
yet `AUTOMATED` (no self-hosted runner is currently registered with the
workflow's specific label and running as a persistent service). Full
detail: [`PRODUCTION_RUNBOOK_2026.md`](PRODUCTION_RUNBOOK_2026.md).
