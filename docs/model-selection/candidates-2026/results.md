# 2026 EPA Tournament — Results

Run per the pre-registered protocol in `registry.md`. Benchmark = **REAL-CLOSING**
de-vigged closing lines purchased for 2022–2024 (834 dev games, 97.7% coverage,
median 19.4 min pre-kickoff, ~15 books). Test seasons 2022–2024; 2025 not scored.

Gains are **model minus market** (positive = model beats the closing market).

| Candidate | Market | n | Model log-loss | Mkt log-loss | LL gain | Brier gain | Pick acc [95% CI] | Promotion |
|---|---|---|---|---|---|---|---|---|
| C1 market-residual | moneyline | 834 | 0.6105 | 0.6074 | −0.0032 | −0.0016 | 0.673 [0.641, 0.705] | RETAIN_BASELINE |
| C1 market-residual | ATS | 834 | 0.6956 | 0.6933 | −0.0023 | −0.0011 | 0.510 [0.474, 0.544] | RETAIN_BASELINE |
| C1 market-residual | total | 834 | 0.6964 | 0.6931 | −0.0033 | −0.0016 | 0.492 [0.457, 0.526] | RETAIN_BASELINE |
| C2 GBM | moneyline | 834 | 0.6964 | 0.6074 | −0.0890 | −0.0291 | 0.606 [0.572, 0.638] | RETAIN_BASELINE |
| C2 GBM | ATS | 834 | 0.7353 | 0.6933 | −0.0419 | −0.0068 | 0.494 [0.460, 0.529] | RETAIN_BASELINE |
| C2 GBM | total | 834 | 0.8082 | 0.6931 | −0.1151 | −0.0088 | 0.525 [0.492, 0.560] | RETAIN_BASELINE |
| C3 logistic | moneyline | 834 | 0.6195 | 0.6074 | −0.0121 | −0.0052 | 0.670 [0.638, 0.704] | RETAIN_BASELINE |
| C3 logistic | ATS | 834 | 0.7090 | 0.6933 | −0.0156 | −0.0070 | 0.512 [0.477, 0.547] | RETAIN_BASELINE |
| C3 logistic | total | 834 | 0.7016 | 0.6931 | −0.0085 | −0.0041 | 0.487 [0.453, 0.522] | RETAIN_BASELINE |
| C4 JointScore+EPA | moneyline | 834 | 0.6598 | 0.6074 | −0.0524 | −0.0237 | 0.619 [0.585, 0.651] | RETAIN_BASELINE |
| C4 JointScore+EPA | ATS | 834 | 0.6947 | 0.6933 | −0.0014 | −0.0007 | 0.494 [0.459, 0.530] | RETAIN_BASELINE |
| C4 JointScore+EPA | total | 834 | 0.6954 | 0.6931 | −0.0023 | −0.0011 | 0.523 [0.489, 0.556] | RETAIN_BASELINE |
| **C5 stacked** | moneyline | 834 | 0.6177 | 0.6074 | −0.0103 | −0.0041 | 0.686 [0.655, 0.718] | RETAIN_BASELINE |
| **C5 stacked** | ATS | 834 | 0.6934 | 0.6933 | **−0.00007** | −0.00003 | 0.511 [0.477, 0.544] | RETAIN_BASELINE |
| **C5 stacked** | total | 834 | 0.6930 | 0.6931 | **+0.00006** | +0.00003 | 0.522 [0.487, 0.556] | RETAIN_BASELINE |

## Verdict: 0 of 15 candidate-markets promoted
**No candidate met the pre-registered promotion criteria on any market.** Game-level
EPA matchup features did not beat the real closing market.

- The best result is the **stacked meta-learner (C5)**, which essentially *ties*
  the closing market on ATS (LL gain −0.00007) and totals (LL gain +0.00006,
  Brier +0.00003). But C5-totals **fails the third gate**: pick-accuracy CI lower
  bound 0.487 < 0.524. Tying the de-vigged market is not beating the vig.
- Every moneyline candidate has high raw pick accuracy (~60–69%) — that is just
  favorites winning, which the market already prices; all have **negative**
  log-loss gain, i.e. worse-calibrated than the close.
- The flexible learners (C2 GBM, C4 JointScore) are the *worst* on moneyline
  (LL gain −0.089, −0.052): with 19 features and ~250–500 training games per fold
  they overfit relative to the market.

## Multiple-comparison exposure
15 pre-registered candidate×market comparisons were run. The single nominally
positive cell (C5 totals, +0.00006 LL) is well within noise and fails its
accuracy gate — consistent with zero true edge, not a missed winner.

Artifacts: `outputs/epa_tournament_scorecard.csv`, `outputs/epa_tournament_summary.json`.
