# 2026 EPA Candidate Tournament — Pre-Registration

**Status:** PRE-REGISTERED. This document is written and committed *before* any
candidate is scored. Nothing is added after results are seen. Any change after
the first scoring run is recorded as a dated amendment below, never a silent edit.

## Protocol
- **Walk-forward, dev seasons only:** test seasons **2022, 2023, 2024**. Each test
  season trains strictly on prior seasons (`season < test_season`), with the
  immediately-prior season used for per-fold calibration.
- **2025 is never scored by any candidate here.** It is the spent holdout,
  reserved for live 2026 validation only.
- **Market benchmark:** REAL-CLOSING de-vigged closing probabilities purchased
  from The Odds API for 2022–2024 (closing snapshot ≈15 min pre-kickoff,
  consensus across books). Where a dev game lacks a matched closing quote, that
  game is dropped from scoring (not proxied) and the dropped count is reported.
- **Scoring:** `evaluation/market_relative.py`, pooled over 2022–2024, per market
  (moneyline, ATS, totals), with the additional per-season breakdown reported.
- **Features:** the augmented matrix (`features/augmented_matrix.py`) — market
  lines (spread, total) + curated game-level EPA matchup differentials
  (opponent-adjusted off/def EPA, success-rate net, pass/rush EPA splits) at
  rolling-4 and season-to-date, plus rest-days/short-week. All pass the
  `leakage_report` as-of checks.

## Candidates (all consume market + EPA features)
1. **C1 — Market-residual regression.** Ridge regression predicting
   `(actual home margin − market-implied margin)` from EPA + rest features; the
   margin is mapped to ML/ATS/totals probabilities through the joint normal
   score model. Isolates whether EPA explains anything the market missed.
2. **C2 — Gradient boosting per market.** `HistGradientBoostingClassifier` per
   market (ML/ATS/totals), isotonic-calibrated per fold on the prior season only.
3. **C3 — Logistic regression.** Per-market `LogisticRegression` on standardized
   features — the interpretable floor.
4. **C4 — JointScoreModel + EPA mean adjustments.** The existing
   `JointScoreModel` with EPA differentials added to its numeric features
   (market lines remain, EPA informs the mean margin/total).
5. **C5 — Stacked meta-learner.** Logistic meta-model over the out-of-fold
   probabilities of C1–C4 **plus the de-vigged market probability as an explicit
   meta-feature**, per market.

Baseline for all gains = the REAL-CLOSING de-vigged market probability.

## Fixed promotion criteria (pre-registered, ALL required)
A candidate is promoted to **PROVISIONAL_ONLY** for a market only if, pooled over
2022–2024 against the REAL-CLOSING benchmark:
1. **log-loss gain vs market > 0**, AND
2. **Brier gain vs market ≥ 0**, AND
3. **for ATS and totals only:** pick-accuracy **95% bootstrap CI lower bound > 0.524**.

For moneyline, criteria 1 and 2 apply (pick-accuracy vs 0.524 is not required, as
favorites-win accuracy is not a market-beating signal).

- Backtests alone **never** yield `STATISTICALLY_SUPPORTED`. The maximum a
  candidate can earn here is `PROVISIONAL_ONLY`, which stakes **zero** and enters
  live 2026 validation.
- **Every candidate's full numbers are reported** — winners and losers — so the
  attempt count (5 candidates × 3 markets = 15 pre-registered comparisons) and
  multiple-comparison exposure are visible to a skeptical reviewer.
- Ties / failing any single criterion → `RETAIN_BASELINE` for that market.

## Multiple-comparison note
15 candidate×market comparisons are run. No per-comparison correction is applied
to the pre-registered gates (they are directional minimums, not significance
claims); instead, promotion only grants `PROVISIONAL_ONLY`, and the *live* season
gate (≥8 weeks, cumulative log-loss gain > 0, live pick-accuracy CI > 0.524) is
the real evidence bar. This is stated so the reviewer can weight the backtest
accordingly.

## Amendments
_(none — original pre-registration)_
