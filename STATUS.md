# STATUS — 2026-08-18

Package `nfl_hybrid` v0.3. This is a current-state snapshot, not a build-run
log. For the full remediation roadmap with evidence links, see
`REMEDIATION_STATUS_2026.md`. For the historical July 2026 build-run report
this file replaces, see `git log -- STATUS.md`.

**Headline, unchanged since the original build:** as configured, no market's
production model beats the real market on log-loss or Brier, so the weekly
card prices at the de-vigged market and **stakes zero**. See
`OPERATOR_MANUAL.md` and `FINAL_REPORT.md` for the full evidence.

## Component status

| Component | Status | Evidence |
|---|---|---|
| Legacy workbook reconstruction (Elo, QB, scoring, totals, 32-model ensemble) | **MERGED** | `src/nfl_hybrid/legacy/`, `src/nfl_hybrid/spreadsheet_baselines/` |
| Joint margin/total model + market-residual challenger | **MERGED** | `src/nfl_hybrid/modern/joint_score.py` |
| Separate ML/ATS/totals calibration heads, tie/push handling | **MERGED** | `src/nfl_hybrid/calibration/`, `src/nfl_hybrid/labels.py` |
| Centralized external-data resolution (Fix 1.5) | **MERGED** — PR #13 | `src/nfl_hybrid/data/external_data.py`, `tests/test_external_data_resolver.py` |
| Authoritative chronological pregame state + lineage (Fix 2) | **MERGED** — PR #14 | `src/nfl_hybrid/features/pregame_state.py`, `docs/PREGAME_STATE_DICTIONARY.yaml` |
| Chronological OOF + uncertainty (Fix 3) | **IN PROGRESS — not yet merged to `main`** | branch `fix/chronological-oof-uncertainty` |
| Chronological calibration built on Fix 3 | **PENDING** | depends on Fix 3 merging first |
| Staking / abstention gate, calibration-drift monitor | **MERGED** | `src/nfl_hybrid/staking/kelly.py`, `src/nfl_hybrid/monitoring/calibration_drift.py` |
| Market-relative evaluation | **MERGED** | `src/nfl_hybrid/evaluation/market_relative.py` |
| 2026 EPA candidate tournament | **HISTORICAL** — completed, promoted 0/15 candidate-markets | `docs/model-selection/candidates-2026/results.md`, `FINAL_REPORT.md` |
| Frozen pricing-calibration artifact | **HISTORICAL** — frozen, fit through 2024, 2025 held out | `config/pricing_calibration_2026.json`, `CALIBRATION_REPORT.md` |
| Week 1 shadow reliability evaluation | **HISTORICAL** — inconclusive (small sample) | `WEEK1_SHADOW_RELIABILITY_REPORT.md` |
| Learned engine as production weekly engine | **PLANNED** — not started | — |
| BDL live-game/box/PBP integration | **PLANNED** — not started | — |
| Twice-weekly learned model refitting | **PLANNED** — not started | — |

## Remediation roadmap (through the Mike Shackleford audit package)

1. Data forensic audit — **HISTORICAL** (identified the defects Fix 1.5/2/3 address).
2. Fix 1.5 — centralized data resolution — **MERGED**.
3. Fix 2 — chronological historical pregame state — **MERGED**.
4. Fix 3 — chronological OOF + uncertainty — **IN PROGRESS**.
5. Chronological calibration — **PENDING** Fix 3.
6. BDL 2026 final-game/box/PBP integration — **PENDING**.
7. Week 1 preseason prior — **PENDING**.
8. Compact feature deduction — **PENDING**.
9. Model-family / frozen-spec selection (re-run on remediated pipeline) — **PENDING**.
10. Learned-engine production promotion — **PENDING**.
11. Tuesday/Friday orchestration — **PENDING**.
12. Immutable forecast ledger — **PENDING**.
13. Market-relative evaluation (re-validated on remediated pipeline) — **PENDING**.
14. Mike Shackleford audit package — **PENDING**.

Full table with evidence links: `REMEDIATION_STATUS_2026.md`.

## What the system will and will not bet, as configured today

- **WILL NOT bet anything.** Every market's frozen production family is
  `market_baseline` (readiness = `RETAIN_BASELINE`). The staking gate forces
  stake = 0 for any `RETAIN_BASELINE` market.
- The card still prints every price with `no-bet: edge not established`.
- **It WILL bet only if** a market is re-established as
  `STATISTICALLY_SUPPORTED` / `PROVISIONAL_ONLY` and the penalized edge
  clears 2%, gated further by the live-promotion rule in `OPERATOR_MANUAL.md`
  (≥ 8 live weeks, cumulative log-loss gain > 0, live pick-accuracy 95% CI
  lower bound > 0.524).

## Exact current commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,dev]"
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"

# validate the private data estate resolves (requires NFL_MODEL_DATA_ROOT)
python scripts/validate_data_registry.py

# weekly card (pre-kickoff)
scripts/run_week.sh 2026 <WEEK>

# tests
pytest -q
```

See README.md for the full command set (backfill, priors, plan-odds, own-book
line capture) and `REMEDIATION_STATUS_2026.md` for what remains before the
learned engine can be promoted to production.
