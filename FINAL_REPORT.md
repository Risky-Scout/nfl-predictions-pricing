# FINAL REPORT — EPA Edge Search + 2026 Season Operations

**Branch:** `overnight-edge-search` · **Run:** Phases D–G, autonomous, committed and
pushed after each phase. **Tests:** 215 pass on a clean Python 3.12 venv.

## Headline verdict — say it plainly
**Game-level EPA matchup features did not beat the real closing market on any
market.** A pre-registered 5-candidate tournament, scored against **REAL-CLOSING**
de-vigged lines purchased for 2022–2024, promoted **0 of 15** candidate-markets.
The 2026 season therefore begins with every market at `RETAIN_BASELINE`, the card
staking **zero**, and the live weekly loop standing up as the search instrument.

This is not a null from a weak setup: the features are rich (opponent-adjusted
off/def EPA, success rate, pass/run splits, rest), leakage-clean (three checks
pass), and judged against the true closing line — the hardest honest benchmark.

---

## Per-market result

| Market | Outcome | Benchmark | Best candidate (pooled 2022–2024) |
|---|---|---|---|
| **Moneyline** | **No candidate met criteria → RETAIN_BASELINE** | REAL-CLOSING | C5 stacked: log-loss gain **−0.0103**, Brier **−0.0041**. All candidates worse-calibrated than the close. |
| **ATS** | **No candidate met criteria → RETAIN_BASELINE** | REAL-CLOSING | C5 stacked: log-loss gain **−0.00007** (ties market), pick acc 51.1% CI [0.477, 0.544] — fails >0.524. |
| **Totals** | **No candidate met criteria → RETAIN_BASELINE** | REAL-CLOSING | C5 stacked: log-loss gain **+0.00006**, Brier **+0.00003** (nominal, within noise), pick acc 52.2% CI [0.487, 0.556] — fails >0.524. |

Full 15-row scorecard: `docs/model-selection/candidates-2026/results.md` and
`outputs/epa_tournament_scorecard.csv`. Pre-registration (fixed before results):
`docs/model-selection/candidates-2026/registry.md`.

**Reading it honestly:** the stacked meta-learner essentially *ties* the closing
market on ATS/totals but never clears the vig (the pick-accuracy CI straddles
52.4%). The flexible learners (GBM, JointScore) are the *worst* on moneyline
(log-loss gain −0.089, −0.052) — with ~19 features and a few hundred training
games per fold they overfit relative to the market. Tying the de-vigged line is
the ceiling EPA reached here; beating the vig it did not.

---

## Leakage-check outputs (Phase E)
From `features/augmented_matrix.leakage_report` on the real 2020–2025 data:

```
first_game_rolling_history_sum     : 0.0     (PASS — a team's first game has no history)
as_of_availability_check_pass      : true    (PASS — prev game available_at = kickoff+5h <= current kickoff)
future_swap_max_abs_feature_diff   : 0.0     (PASS — corrupting future games leaves past features identical)
n_team_pregame_rows                : 3386
```

All three checks pass. The features are built by the existing leakage-safe
builder (`build_team_pregame_features` shifts each metric one game before any
rolling/expanding/EWMA calc); the report verifies that guarantee end to end,
including a future-corruption invariance test (max feature change = 0.0).

---

## Credits spent vs authorization
- **Authorization:** SPEND yes; CLOSING-ONLY for 2022/2023/2024; hard ceiling 13,000.
- **Spent: 12,169 credits** (plan estimate 12,180; balance 4,306,057 → 4,293,888).
  2025 was **not** purchased — it stays the proxy-only holdout, as instructed.
- **Verification call:** one current-odds call (3 credits) to pin the schema fixture
  and confirm the key/balance before the historical spend.
- **Coverage:** 834/854 dev games (97.7%) matched to closing quotes, median 19.4 min
  pre-kickoff, ~15 books. Record: `outputs/odds_closing_dev_summary.json`,
  `outputs/real_closing_benchmark_summary.json`.

**Benchmark note that matters:** the de-vigged closing ATS/totals probabilities are
**0.5001 / 0.5001** — statistically identical to the 0.5 reference proxy. So for
ATS/totals the proxy did *not* overstate beatability of the de-vigged probability;
what the purchase genuinely added is (a) the real closing *lines* for pick-accuracy
grading, (b) a real moneyline de-vig (mean 0.552), and (c) the legitimacy of saying
"vs the actual close." Moneyline is where a proxy would have misled; ATS/totals it
would not have.

---

## Exact weekly commands (2026)
```bash
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"
export THE_ODDS_API_KEY="..."          # optional; current-odds = 3 credits/call

scripts/run_week.sh 2026 <WEEK>        # refresh -> lines -> card(stake 0) -> immutable log
# after games resolve: score vs market, update drift monitor, apply live gate
.venv/bin/python scripts/dryrun_week_loop.py   # rehearsal on 2025 wk1 (no selection)
```
**Live-promotion gate:** a PROVISIONAL market stakes (1/8 Kelly, 5%/15% caps) only
after ≥8 live weeks with cumulative log-loss gain > 0 AND live pick-accuracy CI
lower > 0.524; drift alarm demotes immediately.

---

## What I would flag to a skeptical reviewer
1. **Multiple comparisons.** 15 candidate×market cells were tested. The one
   nominally positive cell (C5 totals, +0.00006 log-loss) is within noise and
   fails its accuracy gate — consistent with zero true edge, not a missed winner.
   No cell would survive any multiple-comparison correction.
2. **Train/test line source differs.** Model *features* use Spreadspoke reference
   lines across all seasons; the *benchmark* for dev test seasons is REAL-CLOSING.
   For ATS/totals the two are ~identical (both de-vig to ~0.5), so this does not
   flatter the model; for moneyline the closing de-vig is the stricter benchmark.
3. **Small per-fold training.** 2022 trains on ~2 seasons (~500 games). The GBM and
   JointScore candidates visibly overfit there. C1/C5 (regularized) are closest to
   the market, which is itself evidence the signal is thin.
4. **EPA availability lag not modelled beyond game-completion.** Features become
   available at kickoff+5h of the prior game; this is enforced. Injuries, weather,
   and line-movement are *not* in these features — their absence is a reason edge
   could exist that this test cannot see, not a reason to trust the null less.
5. **2025 is untouched by selection.** It was used only as an operations rehearsal;
   no candidate was scored on it and no decision used it.

---

## Ranked next data acquisitions (most likely to produce edge)
1. **In-week line movement (opening→closing) for dev seasons.** Buy the 24h/7d
   opening horizons (adds ~2 horizons; see the ceiling plans) to test CLV and
   steam — the one signal a closing-only snapshot cannot reveal. ~24–48k credits.
2. **Injury / inactives timing** with `as_of_utc` — the largest unmodelled pregame
   information source; plausibly where genuine edge over the market lives.
3. **Forecast (not observed) weather** per `docs/UNRESOLVED_SOURCE_GAPS.md §1` —
   totals-relevant, timestamped to the prediction horizon.
4. **Player-level availability / QB starter confirmation** feeding the existing QB
   priors — a QB-out swing is worth several points the closing line already prices,
   but *timing* of that information is the exploitable window.
5. **More seasons of closing lines (2018–2021)** to widen the walk-forward and
   tighten the pick-accuracy CIs that currently straddle 52.4%.

If none of these move the needle either, the live-season log becomes the
instrument of record: it accumulates real out-of-sample evidence week by week, and
the fixed gate — not a backtest — decides if anything ever stakes.

---

## Phase artifacts
- **D:** `scripts/pull_pbp_2020_2025.py`, `scripts/buy_closing_odds_dev.py`,
  `data/odds.require_matched_events`, `tests/test_odds_matching.py`,
  `outputs/odds_closing_dev_summary.json`.
- **E:** `features/augmented_matrix.py` (+ leakage_report),
  `outputs/augmented_feature_matrix.parquet`, `tests/test_augmented_matrix.py`.
- **F:** `docs/model-selection/candidates-2026/{registry,results}.md`,
  `selection/epa_tournament.py`, `scripts/{build_real_closing_benchmark,run_epa_tournament}.py`,
  `outputs/epa_tournament_scorecard.csv`, `tests/test_epa_tournament.py`.
- **G:** `operations/season_loop.py`, `scripts/{run_week.sh,fetch_current_week_lines.py,dryrun_week_loop.py}`,
  README "2026 Season Operations", `outputs/season_2026/rehearsal_2025wk1/`,
  `tests/test_season_loop.py`.
