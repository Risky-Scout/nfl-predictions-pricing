# STATUS — Autonomous Build Run (2026-07-28 → 2026-07-29)

Package `nfl_hybrid` v0.3. This report is deliberately unflattering where the
evidence is unflattering. The headline is simple and true: **as configured, the
model does not beat the market on any market, and the system therefore bets
nothing.** Everything below documents how that conclusion was reached, what was
built, and what remains.

---

## 0. Two things that were NOT as the prompt assumed

1. **The credit ceiling and API key `[BRACKETS]` were left unfilled.** No
   `THE_ODDS_API_KEY` was provided and no numeric ceiling was set. Per the
   prompt's own fallback rule this was treated as key = "none": **all paid odds
   were skipped and reference-line proxies were used, clearly labelled.** Zero
   credits were spent (see §C).
2. **`data/spreadspoke_enhanced.csv` was not at the stated path.** It was found
   in `~/Downloads/NFL Pricing/`, validated against the required counts
   (14,371 rows, seasons 1966–2025, 131 cols ≈ "~125"), and copied into `data/`.
   Validation matched, so the run proceeded (as instructed).

---

## (a) Everything that ran, failed, or was skipped

### Ran successfully
| Step | Result |
|---|---|
| Spreadspoke validation (Task 1) | **PASS** — 14,371 rows; 0 duplicate games on (season,week,home,away); spread signs consistent (14,366 favorites `spread_favorite ≤ 0`, 0 positive, 5 NaN); week column is TEXT (`"1".."18"` + `Wildcard/Division/Conference/Superbowl`) → explicit ordinal ordering built, never string-sorted. |
| nflverse open-source backfill 2020–2025 (Task 1) | **PASS** — real network pull. Canonical `games.parquet` = **1,693 games, seasons 2020–2025, 0 duplicate game_ids**, Spreadspoke crosswalk + reference lines attached. Raw tables: team_stats 3,386 · player_stats 112,447 · weekly_rosters 276,072 · depth_charts 740,289 · snap_counts 157,615 · nflverse_injuries 28,744 (≤2024). SHA-256 manifests written for every dataset. |
| Walk-forward through 2025 (Task 2) | **PASS** — `expanding_season_backtest` run with test seasons **2022–2025**, training strictly on prior seasons (1,139 test games). Outputs in `outputs/walkforward_2025_*.csv`. |
| Market-relative evaluation (Task 3) | **PASS** — new `evaluation/market_relative.py` scores every market vs the de-vigged/reference market. Numbers in §(b). |
| Staking engine + card + CLI (Task 4) | **PASS** — new `staking/kelly.py`, `pricing/betting_card.py`, `predict_week` CLI, `monitoring/calibration_drift.py`; abstention gate wired. Real 2025-wk1 and placeholder-2026-wk1 cards generated. |
| Fundamental 2026 team priors (Task 5) | **PARTIAL PASS** — fit from **real nflverse EPA** (32 teams × 2 metrics), see §(a) skipped for the rest. |
| Fresh-env smoke (Task 6) | **PASS** — clean Python 3.12 venv, `pip install -e ".[data,dev]"`, **196 tests pass**, 2026 card generated end-to-end. |

### Failed then fixed
- **Corrupted `.venv`** (mixed python3.12 + python3.14 dirs) caused
  `ModuleNotFoundError: nfl_hybrid` on console scripts. Fixed by destroying and
  recreating the venv cleanly on Python 3.12.
- **`_apply_weekly_exposure_cap`** used `MultiIndex == tuple` incorrectly →
  fixed to a groupby-indices scaling.
- **CUSUM defaults** were miscalibrated for binary per-game residuals (slack too
  small → alarmed on balanced noise). Retuned (`slack_k=0.25`, `h=4.0`);
  rolling ECE is the primary level signal.

### Skipped (with reason) — never fabricated
- **Paid timestamped odds backfill** — no key/ceiling. `plan-odds` shows the full
  request would cost **138,690 credits** (4,623 unique snapshots), so even the
  template's example 5,000-credit ceiling would buy a negligible slice.
- **Play-by-play + annual rosters** in the backfill — run with
  `--skip-large-tables` for time/memory. Not required for the deliverables;
  needed later for the full tournament feature pipeline and unit-level priors.
- **QB-mixture, roster-continuity, coach priors, and the `market_augmented`
  prior family** — the `data/templates/*.csv` files are **empty header stubs**,
  and no timestamped 2026 snapshot inputs exist. Not fabricated. See §(e).
- **Re-running the frozen 2025 final test** — prohibited by design (hash-guarded,
  one-time, already completed). Its results were read, not regenerated.

### Housekeeping
- 84 macOS sync-conflict duplicate files (`* 2.py`, byte-identical to tracked
  files) were moved out of the tree into the session scratchpad
  (`.../scratchpad/dup_quarantine/`) so they could not corrupt pytest
  collection. They were not deleted.

---

## (b) Market-relative readiness verdict, per market, WITH NUMBERS

There are **two independent lines of evidence**, and they agree.

### Evidence 1 — the frozen 2025 final test (285 games, already completed)
Source: `docs/model-selection/final-2025/final_2025_decisions.csv`. Gains are
model **minus** market (positive = model better).

| Market | Model log-loss | Market log-loss | log-loss gain [95% CI] | Brier gain [95% CI] | Decision |
|---|---|---|---|---|---|
| Moneyline | 0.62857 | 0.62857 | **0.0** [0, 0] | **0.0** [0, 0] | PRODUCTION_MARKET_BASELINE |
| ATS | 0.71808 | 0.71827 | +0.000199 [−0.00034, +0.00075] | **−0.000229 [−0.00031, −0.00015]** | PRODUCTION_MARKET_FALLBACK |
| Total | 0.73410 | 0.73410 | **0.0** [0, 0] | **0.0** [0, 0] | PRODUCTION_MARKET_BASELINE |

The ATS challenger (`ats_push_exact_margin_strength_32`) was the only market that
even attempted to beat the line; its Brier-gain 95% CI is **entirely negative**,
so it failed and fell back to the market. ML and Total production models *are*
the market baseline (gain identically 0).

### Evidence 2 — independent market-anchored walk-forward, 2022–2025 (1,139 games)
Source (produced this run): `outputs/market_relative_*.csv`. Model features are
the reference closing spread/total only (a **REFERENCE-LINE PROXY**, no
timestamped quotes). This measures whether wrapping the market line in the joint
model beats the market. Gains are model minus market.

| Market | Model Brier | Market Brier | Log-loss gain | Pick accuracy [95% CI] | vs 52.4% breakeven | Model ECE |
|---|---|---|---|---|---|---|
| Moneyline | 0.2259 | 0.2105 | **−0.0327** | 61.2% [58.5, 64.1] | n/a (favorites) | 0.024 |
| ATS | 0.2510 | 0.2500 | **−0.0020** | **50.0% [47.1, 52.9]** | **below / CI straddles** | 0.029 |
| Totals | 0.2516 | 0.2500 | **−0.0032** | 51.3% [48.4, 54.3] | **below / CI straddles** | 0.035 |

Margin / CLV proxy (pooled): model MAE **10.20** vs closing-spread MAE **9.54**
(model **worse** by 0.67); the model beats the market margin in only **44.4%** of
games.

### Readiness classification (against the MARKET)
`governance/readiness.classify_market_relative_readiness` →

| Market | Status |
|---|---|
| Moneyline | **RETAIN_BASELINE** (loses on log-loss & Brier vs market) |
| ATS | **RETAIN_BASELINE** (pick accuracy 50.0%, below 52.4%; loses on Brier) |
| Totals | **RETAIN_BASELINE** (pick accuracy 51.3%, below 52.4%; loses on Brier) |

**Plainly: ATS and totals show no edge over the market. Moneyline does not beat
the market either — its 61% pick accuracy is just favorites winning, which the
market already prices, and it loses on both Brier and log-loss.**

---

## (c) Credits spent vs ceiling
- **Credits spent: 0.** No `THE_ODDS_API_KEY` was provided (bracket unfilled),
  so no paid endpoint was ever called.
- **Ceiling: unusable** (bracket unfilled → treated as no-key per the fallback
  rule). The full paid plan would have cost **138,690 credits** regardless.
- All market benchmarks use **de-vigged reference lines from Spreadspoke**,
  labelled `REFERENCE-LINE PROXY` on every row.

---

## (d) What the system WILL and WILL NOT bet, as configured
- **WILL NOT bet anything.** Every market's frozen production family is
  `market_baseline`, i.e. readiness = `RETAIN_BASELINE`. The staking gate
  (`staking/kelly.py`) forces stake = 0 for any `RETAIN_BASELINE` market, and the
  penalized edge is 0 by construction because the production model prices at the
  de-vigged market (`pricing/production.py` enforces `model_probability ==
  market_probability`).
- The card still **prints every price** with the flag
  `no-bet: edge not established` — verified on real 2025-wk1 (48 rows, 0 bets)
  and 2026-wk1 placeholder (12 rows, 0 bets) cards in `outputs/`.
- **It WILL bet only if** a market is re-established as `STATISTICALLY_SUPPORTED`
  / `PROVISIONAL_ONLY` *and* the penalized edge clears 2%. Sizing is then 1/8
  Kelly, capped at 5% per bet and 15% per (season, week). This path is unit-tested
  (`tests/test_staking_kelly.py::test_card_stakes_when_model_beats_market_and_supported`)
  but is **not reachable with the current frozen spec** — which is correct.

---

## (e) Ranked next steps to actually add edge
1. **Timestamped odds coverage (highest value).** The single biggest gap. The
   entire evaluation is anchored to *reference* lines; real closing/opening
   two-way quotes with timestamps are required to (a) measure genuine CLV,
   (b) de-vig properly per book, and (c) find line-shopping edges the closing
   line hides. Requires a funded Odds API key + a real ceiling (full plan ≈
   138,690 credits; buy most-recent seasons first).
2. **Real 2026 snapshot inputs** for priors: rosters, depth charts, starter
   probabilities, injuries — each with `as_of_utc`. Unblocks QB-mixture,
   roster-continuity and coach priors and the `market_augmented` prior family.
   *All `data/templates/*.csv` are currently empty header stubs.*
3. **Pull play-by-play and build the full feature pipeline.** The 2022–2025
   walk-forward here uses only the reference spread/total. Opponent-adjusted EPA,
   QB priors, roster continuity, and pace/situational features (already
   implemented) must feed the joint model before any edge claim is credible.
4. **Forecast weather** per `docs/UNRESOLVED_SOURCE_GAPS.md §1` — retain
   `forecast_issued_utc` / `forecast_valid_utc` / lead time; observed weather is
   only a research proxy.
5. **2020 attendance / crowd-restriction feature** and coordinator/play-caller
   announcement timestamps (source gaps §2, §3).

---

## (f) Exact weekly run commands
Because the setuptools editable import hook is unreliable on this machine (see
note below), run scripts with `PYTHONPATH=src`. `pytest` needs no change (its
config already injects `src`).

```bash
# 0. environment (one time)
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[data,dev]"
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"

# 1. refresh the backfill each week (open sources; skip PBP for speed)
.venv/bin/nfl-hybrid-backfill --output-dir data/backfill_2020_2025 \
  --seasons 2020-2025 --spreadspoke-csv "$SPREADSPOKE_CSV_PATH" \
  --skip-large-tables open-sources

# 2. (optional) price the paid odds plan BEFORE spending — needs a real ceiling
.venv/bin/nfl-hybrid-backfill --output-dir data/backfill_2020_2025 \
  plan-odds --games data/backfill_2020_2025/canonical/games.parquet

# 3. generate the weekly betting card (season/week -> CSV)
.venv/bin/python -m nfl_hybrid.pricing.predict_week \
  --season 2026 --week 1 \
  --games examples/games_2026_week1_placeholder.csv \
  --output outputs/betting_card_2026_wk1.csv
#   (or --backfill-dir data/backfill_2020_2025 to price a completed historical week)

# 4. re-run the market-relative evaluation / readiness
.venv/bin/python examples/walkforward_market_relative_2025.py

# 5. drift check on resolved games -> triggers the calibration-adoption gate
#    (nfl_hybrid.monitoring.calibration_drift.monitor_calibration_drift + build_recalibration_request)

# tests
.venv/bin/python -m pytest -q     # 196 passed
```

**Environment caveat (Task 6):** on **Python 3.14** the default setuptools
editable install's import finder silently fails to add `src` to `sys.path` in
some cwd/sandbox contexts, so `pip install -e` "works" but console scripts raise
`ModuleNotFoundError`. Use **Python 3.11/3.12** (this run uses 3.12) or
`--config-settings editable_mode=compat`, and prefer `PYTHONPATH=src` for
scripts. `pytest` is unaffected.

---

## New / changed code this run
- `src/nfl_hybrid/staking/{__init__,kelly}.py` — penalized-Kelly staking + abstention gate.
- `src/nfl_hybrid/monitoring/{__init__,calibration_drift}.py` — rolling ECE + CUSUM + recalibration trigger.
- `src/nfl_hybrid/evaluation/market_relative.py` — market-relative scorecards + ECE + CLV proxy.
- `src/nfl_hybrid/governance/readiness.py` — added `classify_market_relative_readiness`.
- `src/nfl_hybrid/pricing/{betting_card,predict_week}.py` — card builder + `predict_week` CLI (new pyproject entry point).
- `examples/walkforward_market_relative_2025.py`, `examples/games_2026_week1_placeholder.csv`.
- Tests: `tests/test_{staking_kelly,market_relative,calibration_drift,betting_card,predict_week_cli}.py` (+27 tests → **196 passed**).
- No frozen artifact (final-test, production spec, calibration gate) was modified.

**Leakage discipline:** the walk-forward trains strictly on seasons `< test_season`;
the new layers (staking, pricing, evaluation) consume only pre-game reference
lines and post-hoc outcomes, adding no new predictive feature and therefore no
new leakage surface.
