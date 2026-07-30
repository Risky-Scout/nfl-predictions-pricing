# Pricing-Calibration 2026 — Report

Selects the *pricing mathematics* (margin surface, sigma, de-vig, consensus) by
leakage-safe walk-forward and freezes one artifact:
`config/pricing_calibration_2026.json` (+ readable PMF export
`config/pricing_calibration_2026_pmf.csv`). Pre-registration:
`docs/model-selection/pricing-calibration-2026/registry.md`. **2025 was untouched.**

## Data & cutoffs
- Empirical margin PMF: Spreadspoke **2002–2024** (6,214 games with closing spread).
- Sigma / de-vig / consensus: purchased de-vigged **closing** odds 2022–2024
  (12,169-credit dataset, `data/purchased/`), Spreadspoke residuals 2020–2024.
- Final artifact fit on permitted data **through 2024**. Holdout **2025** excluded.

## Walk-forward design
Expanding: test season Y trained on `season < Y`, folds 2022/2023/2024. De-vig/
consensus scored on the purchased closing quotes; sharp-book anchor selected using
only prior-season outcomes. Confidence intervals: paired **game-clustered
bootstrap**, 2,000 resamples, seed 20260815.

## Margin surface — candidate table (moneyline, pooled 2022–2024)
| Candidate | Log loss | Brier | ECE (10 eq-mass) |
|---|---|---|---|
| normal_baseline | 0.61169 | 0.21156 | 0.0468 |
| empirical_bw1.0 | 0.60816 | 0.21014 | 0.0328 |
| empirical_bw2.0 | 0.60832 | 0.21033 | 0.0475 |
| **empirical_bw3.0 (selected)** | **0.60614** | **0.20931** | **0.0339** |

`normal − best_empirical` log-loss diff = **+0.0055, 95% CI [−0.0018, +0.0121]**
(contains 0 → statistical tie on the primary metric).

**Push / key-number calibration (the decisive property):** push MAE at key numbers
**empirical 0.066 vs normal 0.088**. P(margin = 3) = **7.7% empirical (matches data)
vs 2.95% normal**. Per registry **Amendment 1**, the empirical surface is selected
because it is *not worse* on the primary metric and *decisively correct* on the
push mass that ATS-push and alternate-line pricing depend on. `normal` is retained
as a configurable fallback.

### Key-number diagnostics — before (normal) vs after (empirical), 3-point game
Home cover + push for mean home margin **+3** (i.e. home −3 favourite):

| Line | Normal cover | Normal push | Empirical cover | Empirical push |
|---|---|---|---|---|
| −2.5 | 0.5156 | 0.0000 | 0.5193 | 0.0000 |
| **−3.0** | 0.4844 | **0.0313** | 0.4426 | **0.0767** |
| −3.5 | 0.4844 | 0.0000 | 0.4426 | 0.0000 |

The normal model under-prices the −3 push by more than 2× (0.031 vs the empirical/
actual 0.077). Half-point lines correctly carry zero push under both.

## Fitted sigma
| Residual | Selected | Value | Note |
|---|---|---|---|
| Margin | **constant** | **12.75** | linear-in-total not materially better; OOS Gaussian NLL 3.9471 (const) vs 3.9495 (linear), CI of difference contains 0 |
| Total | **constant** | **13.05** | linear-in-total not better; NLL 3.9823 vs 3.9824, CI contains 0 |

Complexity was **not** forced: constant sigma is retained for both markets.

### −3 favourite at market totals 37 vs 52
Because total sigma is **constant**, the total-market distribution shape is
level-invariant: over/push/under at the closing total are **0.485 / 0.0295 / 0.485**
at both totals. (Before: default sigma 13.5; after: fitted 13.05 — a negligible
shift, no material change.) A heteroscedastic total sigma would have changed these;
the data did not support it out-of-sample.

## De-vig & consensus (per market)
| Market | Selected de-vig | Selected consensus | Tie-break vs default (default − best) |
|---|---|---|---|
| Moneyline | proportional | equal_mean | +0.00104, CI [−0.00097, +0.00285] (contains 0 → keep default) |
| ATS (spread) | proportional | equal_mean | best *is* default |
| Totals | proportional | equal_mean | +0.00017, CI [−0.00004, +0.00038] (contains 0 → keep default) |

Power, Shin, median, and sharpest-book anchor were all evaluated; **none** beat
`proportional + equal_mean` on the game-clustered bootstrap, so the simplest default
is retained for every market (registry complexity tie-break).

## Runtime (pricing stage, cached inputs)
- Before (normal built-in): ~2 ms/week.
- After (empirical PMF lookup + fitted sigmas): **2.7 ms/week** (16 games × 3
  markets) — well under the 5-second budget and within 2× of the prior stage.
- Deterministic: identical inputs reproduce byte-identical output (verified).

## Cases where the simpler baseline was retained
- **Margin sigma → constant** (linear/isotonic not better).
- **Total sigma → constant.**
- **De-vig → proportional** for all three markets.
- **Consensus → equal_mean** for all three markets.
Only the margin *surface* moved off its simplest default, and only because the
push/key-number correctness it provides is required by the alternate-line math.

## Honest limitations
- The empirical margin PMF is conditioned on the **closing spread bucket** (width
  3.0); very sparse buckets shrink heavily toward the normal baseline, so extreme
  spreads are effectively normal-priced.
- Margin-surface moneyline improvement is a **statistical tie**; the win is in push/
  key-number mass, not in beating the market's win probability.
- De-vig/consensus differences are within noise on this 834-game closing sample; a
  larger multi-season closing sample (self-captured via `capture_lines.py`) could
  eventually separate them.
- Totals keep the normal model (no total PMF was in scope); only the total *variance*
  was fitted (and came out constant).
- Nothing here changes the market-baseline governance: production still prices at the
  de-vigged market and stakes zero until the live gate promotes a market.

## Artifact
`config/pricing_calibration_2026.json` — version `pricing_calibration_2026.v1`,
training cutoff 2024, seed 20260815, source hashes (Spreadspoke + purchased odds),
code commit, selected methods, full candidate grids, empirical PMF export, and
`artifact_sha256`. Rebuild: `PYTHONPATH=src python scripts/build_pricing_calibration.py`.
