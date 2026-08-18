# OPERATOR MANUAL — NFL Pricing Engine, 2026 Season

> **Legacy operating path.** This manual documents the current
> legacy/market-baseline weekly operating path (the one the production card
> actually runs today). The 2026 learned-production runner (built on the
> Fix 2 chronological pregame-state families and Fix 3 chronological
> OOF/uncertainty) is still under remediation and is not yet the production
> path — see `STATUS.md` and `REMEDIATION_STATUS_2026.md`.

For a non-author running the season. Read the box first.

> **State of the system (read this):** All three markets — moneyline, ATS, totals —
> open the 2026 season at **`RETAIN_BASELINE`**. The pre-registered EPA tournament
> beat the real closing market on **nothing** (0/15). **The card stakes ZERO.** It
> prices every game at the de-vigged market and prints `no-bet: edge not established`.
> Nothing stakes until the **live record** earns it through the gate in §6. This is
> by design, not a bug. A model that bets unproven markets is a donation machine.

---

## Two paths (offline calibration vs weekly runtime)
- **Offline (rarely, you rebuild the pricing math):**
  `PYTHONPATH=src python scripts/build_pricing_calibration.py` fits/selects the margin
  surface, sigmas, de-vig and consensus by walk-forward and writes the frozen artifact
  `config/pricing_calibration_2026.json` (+ `_pmf.csv`). See `CALIBRATION_REPORT.md`.
  The weekly job never refits — it only loads this artifact.
- **Weekly runtime (every week):** `scripts/run_week.sh 2026 <WEEK>` refreshes data,
  captures odds, loads the frozen artifact, prices, validates, and logs. Pricing is
  deterministic and runs in ~3 ms; identical inputs reproduce identical probabilities.

**Selected pricing math (frozen):** empirical margin PMF (correct key-number pushes,
e.g. P(push −3)=0.077), constant margin sigma 12.75 / total sigma 13.05, and
proportional de-vig + equal-mean consensus for all three markets.

**Audit fields on every card row:** `as_of_utc`, `quote_timestamp_utc`,
`quote_age_minutes`, `bookmakers_used`, `devig_method`, `consensus_method`,
`pricing_surface_version`, `artifact_sha256`, `code_commit`, `market_fair_probability`,
`model_probability`, `push_probability`, `readiness_status`, `qb_review_status` (via
priors), `injury_data_utc`, `data_quality_status`, and `card_status`
(`PRELIMINARY` / `VALID_LIVE_CARD` / `NO_PRICE`). July cards are `PRELIMINARY`.

## Finality / in-progress-game safety (live fundamental features)
Two modes control which prior games feed a target's rolling EPA features:
- **`live` (default, production):** a prior game contributes only when **verified final
  before `as_of_utc`** by real evidence (explicit final status + completion timestamp,
  a verified terminal-play timestamp, a schema completion timestamp, or an official final
  snapshot). Non-null scores or elapsed time are **not** evidence. Absent evidence →
  `UNKNOWN_FINALITY` → excluded (fail closed). With no finality feed configured, live
  fundamental probabilities are honestly **NULL** (`NO_VERIFIED_FINAL_GAMES`).
- **`historical_replay` (backtests/research only):** additionally admits games under the
  repository's documented postgame-availability convention (kickoff + 5 h), clearly
  labelled `HISTORICAL_AVAILABILITY_ASSUMPTION` — this is an availability assumption, **not**
  verified finality, and must never be used in production.

An in-progress or unknown game is excluded and its partial play-by-play is never
aggregated. The run manifest records finality counts and a finality decision hash.
There is no assumed "4 h/5 h verified upper bound" — elapsed duration is not finality
evidence.

## 0. One-time setup
```bash
cd nfl-predictions-pricing
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[data,dev]"
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"
export THE_ODDS_API_KEY="<your key>"      # optional but needed for live lines/capture
.venv/bin/python -m pytest -q               # expect: all green
```
(On Python 3.14 the editable import hook can fail silently — use 3.11/3.12, which
this manual assumes. `PYTHONPATH=src` is the reliable path for scripts.)

### Core vs extended live-feature tests
The live/training feature-parity and leakage guarantees run in **two tiers**:
- **Core (always in CI):** `tests/test_live_features_ci.py` uses only the small
  committed synthetic fixtures under `tests/fixtures/live_features/` and **never
  skips** (26 tests, ~7 s). GitHub CI runs it as its own named step on Python 3.11
  and 3.14 and fails if it reports any `skipped`. Run it directly with
  `PYTHONPATH=src pytest -q tests/test_live_features_ci.py`.
- **Extended (local only):** `tests/test_live_features_extended.py` (marked
  `extended_data`) replays the full private `${NFL_MODEL_DATA_ROOT}/backfill-2020-2025/` parquets and
  **skips** wherever that data is absent. Run it when the backfill is present with
  `PYTHONPATH=src pytest -q tests/test_live_features_extended.py`.

The golden expected outputs (`expected_features.csv`) are committed fixed data. To
intentionally change them, run the manual, `--overwrite`-gated generator and review
the printed old/new hashes and the resulting diff — it never runs in CI:
```bash
PYTHONPATH=src python scripts/regenerate_live_feature_ci_fixture.py --overwrite
```

---

## 1. The weekly sequence
Run **pre-kickoff** each week. One command does steps 1b–4:
```bash
scripts/run_week.sh 2026 <WEEK>
```
It performs, in order:

| Step | Command (what run_week.sh calls) | Cost | Output |
|---|---|---|---|
| 1. Refresh | `nfl-hybrid-backfill … open-sources` | 0 | updated `${NFL_MODEL_DATA_ROOT}/backfill-2020-2025/` |
| 1b. **Capture** | `scripts/capture_lines.py` | **3 credits** | immutable snapshot in `data/line_history/` |
| 2. Lines | `scripts/build_week1_2026_lines.py` | **3 credits** | `outputs/season_2026/lines_wkNN.csv` |
| 3. Price | `predict_week …` | 0 | `outputs/season_2026/card_wkNN.csv` |
| 4. Log | `log_week_predictions` | 0 | `outputs/season_2026/predictions_wkNN.csv` (immutable) |

**After the games resolve** (Monday/Tuesday), run scoring + gate (step 5–6, §5).

### Capture cadence (optional but recommended — builds your edge dataset for ~free)
Run `scripts/capture_lines.py --label <tag>` at four points each week:
**Tue open / Fri / Sun T-60 / Sun T-10**. Each call = **3 credits** for the whole
slate across all 3 markets. That is **12 credits/week**, **~216 credits/season**, for
complete opening-to-closing line movement on every game — the #1 ranked data
acquisition, self-built.

---

## 2. What each output means
- **`lines_wkNN.csv`** — one row per game. `status = PRICED` (posted lines found,
  `n_books` books) or `UNPRICED-AWAITING-LINES` (no lines yet — the game is *not*
  invented; it simply is not priced until lines post).
- **`card_wkNN.csv`** — one row per (game, market). Key columns:
  - `model_spread` / `predicted_home_margin` — the model's view (= market, in production).
  - `calibrated_probability` — win/cover/over probability.
  - `market_fair_probability` — de-vigged market probability.
  - `penalized_edge` — edge after the KL penalty. **0.0 in production by design.**
  - `recommended_stake` / `should_bet` — **0 / False** for every market right now.
  - `readiness_status` — `RETAIN_BASELINE` for all markets at season open.
  - `no_bet_reason` — `no-bet: edge not established`.
  - `calibration_statement` — the standing disclaimer (see §6).
- **`predictions_wkNN.csv`** — the immutable, UTC-timestamped pre-kickoff record.
  **Never edit it.** It is the ground truth for live validation.
- **`data/line_history/*.parquet`** — accumulating book-attributed, de-vigged snapshots.

### Alternate-line quotes (bookmaker surface)
To quote any spread/total ladder for a game, use `pricing.alt_lines`
(`price_spread_ladder` / `price_total_ladder`) — fair probability, fair decimal, and
offered decimal at a configurable hold. See `tests/test_alt_lines.py` for usage.

---

## 3. Credit costs per week
- Card generation: **3 credits** (one current-odds call for lines).
- Recommended capture cadence: **12 credits** (4 calls).
- **Total ≈ 15 credits/week** if you capture; **3/week** if you only price.
- Historical purchases are *not* needed for operations — the season builds its own
  movement dataset via capture.

---

## 4. Human-review points (do these, they are not automated)
1. **August QB battles.** `data/templates/starter_probabilities.csv` rows with
   `season = 2026` and `source = depth_chart_rank` carry `needs_human_review = True`.
   These are rank-based projections from the latest depth chart. Before Week 1,
   confirm each team's actual Week-1 starter and correct the probabilities, then
   re-run `scripts/refit_2026_priors.py`. A wrong starter is worth several points.
2. **Coordinators / play-callers.** Not in the data (external-only). If a team has a
   new OC/DC/play-caller, note it — the coach prior only captures head coaches.
3. **`UNPRICED-AWAITING-LINES` games.** Re-run the line fetch closer to kickoff; do
   not manufacture a number.

---

## 5. After games resolve — score + drift
```bash
# score the logged record against the market and update the cumulative scorecard
.venv/bin/python - <<'PY'
import pandas as pd
from nfl_hybrid.operations import score_resolved_week, cumulative_live_scorecard, live_promotion_decision
# load outputs/season_2026/predictions_wkNN.csv (logged) and a resolved-results frame
# (game_id, home_win, home_cover, over), then:
# weekly = score_resolved_week(logged, resolved)
# cum = cumulative_live_scorecard(all_weekly_scores)
# decision = live_promotion_decision(cum, drift_alarm_markets=<from drift monitor>)
PY
```
Update the **drift monitor** (`monitoring.calibration_drift.monitor_calibration_drift`)
on resolved games. A **drift alarm** on any market → that market is **demoted to
`RETAIN_BASELINE` immediately** (stakes zero), and recalibration is requested through
the calibration-adoption gate. See `scripts/dryrun_week_loop.py` for a full worked
example on 2025 week-1 data.

---

## 6. The fixed promotion / demotion rules (do not change mid-season)
A market is **PROVISIONAL** only if a candidate cleared the backtest gate — none did,
so all markets are `RETAIN_BASELINE` today. Even a PROVISIONAL market **stakes zero**
until the **live gate** passes:

**PROMOTE_TO_LIVE_STAKE** (then stakes 1/8 Kelly, 5% per-bet / 15% weekly caps) requires
**all** of:
- **≥ 8 live weeks** scored, AND
- **cumulative live log-loss gain vs market > 0**, AND
- **live pick-accuracy 95% CI lower bound > 0.524**.

**DEMOTE_TO_RETAIN_BASELINE** — immediate — on any **drift alarm**.

Backtests never grant live staking. Only the immutable live record can. If a season
of live evidence never clears the gate, the honest outcome is: keep pricing at the
market and stake nothing. That is a successful sharp book, not a failure.

---

## 6b. Model acceptance status (as of Stage 3)
- **Step 1 — final-before-as-of / in-progress handling:** PASS.
- **Step 2 — core live/training parity & leakage tests in CI:** PASS.
- **Step 3 — Week 1 shadow reliability evaluation:** Stage 3 evaluation result: COMPLETE (evaluation-only). Repository acceptance requires PR + CI (Python 3.11/3.14) + clean-main verification.
- **Overall model status: NOT READY.**
- **Next pending stage: 4 — make `run_week` fail closed and correct injury timestamp semantics.**

What Step 3 does and does **not** mean:
- Stage 3 **evaluates historical reliability** of the frozen
  `market_augmented_epa_rest_shadow` model. `fundamental_probability` is generated
  by that market-augmented shadow model and is **not independent of the market**.
- It **does not establish betting edge** and **does not promote the shadow model**.
  Production stays `MARKET_BASELINE` (`production_probability == market_probability`).
- The market comparison is **schedule-reference historical reliability with
  unknown quote timestamp** (`line_timestamp_known=false`); it **does not
  establish closing-line, T-60, or T-10 reliability**.
- **Production live shadow availability still depends on a verified finality
  source.** Historical evaluation used the documented
  `HISTORICAL_AVAILABILITY_ASSUMPTION` (replay), never verified finality.
- Pooled Week 1 OOF (n = 64 < 100) → all markets `INCONCLUSIVE_SMALL_SAMPLE`;
  recommendation `RETAIN_MARKET_BASELINE_AND_MONITOR_SHADOW`. See
  `WEEK1_SHADOW_RELIABILITY_REPORT.md` and `reports/week1_shadow_reliability_2026.json`.
- Reproduce locally (needs the private backfill):
  `PYTHONPATH=src python3 scripts/evaluate_week1_shadow_reliability.py`. Evaluator
  correctness is enforced in CI by `tests/test_week1_shadow_reliability.py` (no skips).

---

## 7. If something breaks
- **Card won't generate:** check `lines_wkNN.csv` exists and has `PRICED` rows.
- **`FileExistsError` on logging:** the week's record already exists and is immutable
  by design — do not overwrite it.
- **Odds API error / no key:** operations continue on the CSV template; games show
  `UNPRICED-AWAITING-LINES`. No numbers are invented.
- **Tests red after an update:** stop and fix before trusting any output.
