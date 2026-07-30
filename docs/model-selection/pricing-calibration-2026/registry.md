# Pricing-Calibration 2026 — Pre-Registration

**Status:** PRE-REGISTERED. Written and committed **before** any comparison result is
computed. No candidate is added after results are seen; any post-hoc change is a
dated amendment at the bottom, never a silent edit.

This experiment selects the *pricing mathematics* only (margin surface, sigma,
de-vig, consensus). It does not select a model that tries to beat the market — the
frozen market-baseline and live-promotion governance are unchanged.

## Data & cutoffs
- **Margin PMF (empirical candidate):** Spreadspoke games **2002–2024**, signed home
  margin conditioned on the closing home spread. Fit walk-forward (a fold for test
  season Y uses only games with `season < Y`).
- **Sigma & de-vig/consensus:** purchased de-vigged **closing** odds
  (`data/purchased/odds_closing_dev_2022_2024.parquet`) for test seasons **2022,
  2023, 2024**, plus Spreadspoke lines/outcomes for margin/total residuals
  (2020–2024, walk-forward).
- **2025 is untouched** for all fitting and selection. Final 2026 artifact is fit on
  permitted data **through 2024**.

## Walk-forward folds (expanding)
- 2022 predictions use information available through 2021.
- 2023 predictions use information available through 2022.
- 2024 predictions use information available through 2023.

## Outcome definitions
- **Moneyline:** home win = `home_margin > 0`; tie (`==0`) is a push.
- **ATS (home cover at spread s):** `home_margin + s > 0`; push at integer `s` when
  `home_margin + s == 0`.
- **Totals (over at line t):** `total_points > t`; push at integer `t`.
- **Push handling:** pushes are excluded from graded win-probability log-loss/Brier
  (probabilities conditioned on no-push); **push frequency and push calibration are
  reported separately**. A push is never encoded as win, loss, or half.

## Candidates (maximums are hard caps)

### Margin surface
1. **normal_baseline** — current discretized-normal CDF pricing.
2. **empirical_shrinkage** — conditional integer-margin PMF, shrunk toward the
   discretized-normal baseline. At most **three** registered bucket/bandwidth choices:
   `{spread_bucket_width ∈ (1.0, 2.0, 3.0)}`. Normal tail beyond `|margin| > 21`.

### Margin sigma
1. **constant**
2. **linear_in_total** — `sigma = a + b·total`
3. **isotonic** — only with fixed clamped boundary behavior and a minimum-sample rule
   (≥200 games per fitted region).

### Total sigma
Same three-candidate maximum (constant / linear_in_total / isotonic).

### De-vig
1. **proportional**  2. **power**  3. **shin**

### Consensus
1. **equal_mean**  2. **median**  3. **sharpest_book_anchor**

No additional candidates unless an existing repo implementation already requires one.

## Metrics
- **Primary (selection):** log loss (per market).
- **Secondary:** Brier score; equal-mass **ECE** with **10 equal-frequency bins**.
- **Guardrails:** empirical coverage (predicted vs realized) and stability across folds.

## Confidence intervals
- **Paired, game-clustered bootstrap** on per-game log-loss differences.
- **2,000 resamples, seed 20260815.** Games are the cluster/resampling unit.

## Coverage / eligibility rules
- **Minimum bookmaker coverage:** a game-market is eligible only with **≥ 3** eligible
  books at the reference point.
- **Sharpest-book anchor is leakage-safe:** the anchor for test season Y is chosen
  using **only outcomes from seasons < Y** (2022 outcomes → 2023 anchor; 2022–2023 →
  2024 anchor). A book must have **≥ 200 graded quotes across ≥ 2 prior seasons** to be
  anchor-eligible; otherwise fall back to the winning non-anchor consensus method.

## Tie-breaking / complexity preference
When two candidates' primary metrics are **statistically indistinguishable** (bootstrap
CI of the log-loss difference contains 0), select the **simpler and more stable**
method: normal_baseline > empirical; constant > linear > isotonic; proportional >
power > shin; equal_mean > median > sharpest_book_anchor. The empirical/heteroscedastic/
power/Shin/median/anchor options are **not** forced to win.

## Promotion requirements (what gets written into the artifact)
A non-default method replaces the default for a market only if its **out-of-sample
pooled log-loss is lower** and the **paired bootstrap CI of the difference excludes 0**
(and it does not lose on Brier). Otherwise the default (simpler) method is retained.
This selects *pricing math*, not edge; nothing here promotes a market to live staking.

## Amendments
_(none — original pre-registration)_
