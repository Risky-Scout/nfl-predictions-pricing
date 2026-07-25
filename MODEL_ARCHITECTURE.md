# Solution B — Hybrid Model Architecture

## 1. Auditable legacy layer

The repository implements the workbook constants and formulas as named,
unit-tested functions:

- adjusted Elo;
- home-field, travel, bye/rest, QB, and playoff adjustments;
- Elo-to-margin conversion;
- margin-of-victory Elo update;
- QB value and QB-to-Elo adjustment;
- multiplicative offense/defensive-weakness scoring;
- 70/30 H2H total blend;
- wind factor;
- American-odds conversion and de-vigging;
- selected FiveThirtyEight/Yahoo/Vegas ensemble formulas.

Known workbook defects are corrected rather than copied.

## 2. Leakage-safe feature layer

`PregameFeatureBuilder` traverses games chronologically. It emits each row's
features before updating Elo, scoring factors, H2H history, or rolling form
with the current final score.

The builder intentionally keeps the legacy scoring factors visible as model
features. This makes the modern model a challenger to—not a replacement for—
the spreadsheets.

## 3. Joint distribution challenger

The production baseline predicts:

- mean home margin;
- mean game total;
- residual standard deviations;
- residual correlation.

The implied team means are:

\[
\mu_H=(\mu_T+\mu_M)/2,\qquad
\mu_A=(\mu_T-\mu_M)/2
\]

Market probabilities are then derived coherently:

\[
P(HomeWin)=P(M>0)
\]

\[
P(HomeCover)=P(M+HomeSpread>0)
\]

\[
P(Over)=P(T-TotalLine>0)
\]

Integer-line push probabilities are retained separately.

## 4. Separate market calibration heads

Moneyline, ATS, and total probabilities receive separate logistic calibration
heads. The heads are fitted on a season later than the model-training set and
earlier than the test season.

## 5. In-play extension

The play-level model predicts remaining home and away points from:

- score, period, and clock;
- possession;
- down, distance, and field position;
- timeouts;
- drive and turnover state;
- current quarterbacks and material player exits;
- live lines available at the prediction timestamp.

The supplied workbooks do not contain these fields, so the in-play class is
implemented but cannot be pretrained from the files alone.

## 6. Validation

Use expanding season/week walk-forward evaluation. Never use a random split.

Primary metrics:

- Moneyline, ATS, totals: Brier score, log loss, calibration;
- margin and total: MAE and RMSE;
- betting layer: expected value at the actual timestamped price, closing-line
  value, ROI with pushes, and bootstrap confidence intervals.

The workbook-derived dataset is only an initial baseline. Production
deployment requires current, licensed, timestamped odds and live event data.
