> **HISTORICAL AUDIT RECORD — NOT CURRENT SYSTEM STATUS**
> This report records the results and build state of a prior experiment run.
> Numbers below are not rewritten and remain as originally reported. For
> current status, see `STATUS.md` and `REMEDIATION_STATUS_2026.md`.

# Solution B — Step 2 Initial Build Report

**Build date:** July 24, 2026

## Delivered

This repository is a runnable hybrid implementation with:

- corrected adjusted-Elo baseline;
- QB-value and QB-to-Elo functions;
- multiplicative scoring and legacy total calculations;
- American Moneyline conversion and two-way de-vigging;
- all 32 FiveThirtyEight/Yahoo/Vegas ensemble models;
- leakage-safe sequential pregame features;
- a joint margin/total challenger;
- separate Moneyline, ATS, and totals calibration heads;
- explicit tie and push probability;
- joint score simulation;
- an executable provider-agnostic in-play remaining-score class;
- unit tests and a walk-forward example;
- 7,995 workbook-derived games covering 1989–2019.

## Validation design

The benchmark is strictly chronological:

- **2018 test:** train on 1989–2016; calibrate on 2017; test on 2018.
- **2019 test:** train on 1989–2017; calibrate on 2018; test on 2019.

The current workbook dataset has 267 games in each test season. Ties and
market pushes are excluded only from the corresponding binary calibration
metric; their probabilities remain explicit in prediction output.

## Initial results

| Test season | Model/target | Brier | Log loss | MAE | RMSE |
|---|---|---:|---:|---:|---:|
| 2018 | Hybrid Moneyline | 0.2225 | 0.6372 | — | — |
| 2018 | Legacy Elo Moneyline | 0.2233 | 0.6351 | — | — |
| 2019 | Hybrid Moneyline | 0.2134 | 0.6165 | — | — |
| 2019 | Legacy Elo Moneyline | 0.2238 | 0.6402 | — | — |
| 2018 | Hybrid ATS | 0.2502 | 0.6935 | — | — |
| 2019 | Hybrid ATS | 0.2465 | 0.6862 | — | — |
| 2018 | Hybrid Total | 0.2508 | 0.6948 | — | — |
| 2019 | Hybrid Total | 0.2517 | 0.6966 | — | — |
| 2018 | Hybrid Margin | — | — | 10.226 | 13.279 |
| 2018 | Legacy Elo Margin | — | — | 10.294 | 13.426 |
| 2019 | Hybrid Margin | — | — | 10.053 | 13.000 |
| 2019 | Legacy Elo Margin | — | — | 10.499 | 13.478 |
| 2018 | Hybrid Total Points | — | — | 10.896 | 13.832 |
| 2018 | Legacy Total Points | — | — | 11.668 | 15.296 |
| 2019 | Hybrid Total Points | — | — | 11.050 | 13.718 |
| 2019 | Legacy Total Points | — | — | 11.389 | 14.362 |

## Interpretation

The challenger materially improved the 2019 Moneyline result and improved
margin and total-point errors relative to the legacy formulas in both test
seasons. The 2018 Moneyline comparison is mixed: hybrid Brier was slightly
better, while legacy Elo log loss was slightly better.

There is **no established totals edge** in this initial build. Total
probability loss was worse than a 50% baseline in both test seasons. ATS was
approximately coin-flip in 2018 and modestly better in 2019. These are research
results, not evidence of durable betting profitability.

The current benchmark cannot calculate expected betting value or ROI because
the workbook does not contain timestamped two-sided prices for spreads and
totals, and it does not contain robust Moneyline snapshots.

## Production blockers

The in-play model is code-complete as a framework but untrained. The uploaded
workbooks contain no play-level state. A production build still requires:

1. current timestamped schedules, rosters, expected starters, and injuries;
2. forecast weather captured at each prediction horizon;
3. timestamped Moneyline, spread, and total prices;
4. play-by-play snapshots with clock, score, possession, down, distance,
   field position, timeouts, player exits, and revision timestamps;
5. live market snapshots synchronized to each play state.

## Run

```bash
pip install -e ".[dev]"
pytest
mkdir -p outputs
python examples/train_and_predict.py
```
