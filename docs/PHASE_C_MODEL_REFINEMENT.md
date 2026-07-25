# Phase C Model Refinement and 2026 Prior Framework

## Implemented challengers

### 1. Direct joint-score model

Predicts mean margin and mean total from football and market features, then
derives Moneyline, ATS, total, tie, and push probabilities from one residual
distribution.

### 2. Market-residual joint-score model

Uses the exact market snapshot as an anchor:

\[
MarketMargin=-HomeSpread
\]

\[
MarginResidual=ActualMargin-MarketMargin
\]

\[
TotalResidual=ActualTotal-TotalLine
\]

The model predicts only the residuals. Calibration-season multipliers can
shrink residual predictions back toward the market when the estimated edge is
unstable.

The two models remain separate. Model selection is performed by chronological
walk-forward validation rather than by choosing the better test result after
the fact.

## Workbook-history comparison

The uploaded workbook history ends in 2019. The comparison therefore uses:

- train through 2016, calibrate on 2017, test on 2018;
- train through 2017, calibrate on 2018, test on 2019.

| Season | Target | Direct joint | Market residual | Market anchor |
|---|---|---:|---:|---:|
| 2018 | Moneyline Brier | 0.2225 | **0.2135** | unavailable |
| 2019 | Moneyline Brier | **0.2134** | 0.2158 | unavailable |
| 2018 | ATS Brier | 0.2502 | **0.2500** | 0.500 probability baseline |
| 2019 | ATS Brier | **0.2465** | 0.2477 | 0.500 probability baseline |
| 2018 | Total Brier | 0.2508 | **0.2484** | 0.500 probability baseline |
| 2019 | Total Brier | **0.2517** | 0.2534 | 0.500 probability baseline |
| 2018 | Margin MAE | 10.226 | 9.890 | **9.869** |
| 2019 | Margin MAE | **10.053** | 10.170 | 10.170 |
| 2018 | Total-points MAE | 10.896 | **10.669** | 10.732 |
| 2019 | Total-points MAE | 11.050 | 10.892 | **10.798** |

These results are deliberately not presented as a betting edge. They show that
the residual architecture is plausible but unstable across the two available
test seasons. The 2020–2025 source backfill is required for selection.

## Team and unit priors

`EmpiricalBayesTeamPrior` accepts long-form opponent-adjusted metrics with:

```text
entity_id
season
metric
value
sample_size
```

It applies season-recency and effective-sample weights, then shrinks toward the
league mean:

\[
Reliability=rac{N_{eff}}{N_{eff}+K}
\]

\[
PriorMean=LeagueMean+Reliability(RawMean-LeagueMean)
\]

The package tunes season weights and \(K\) through historical season replays.

## Quarterback priors

Quarterback performance is weighted by recent dropbacks, capped at the latest
1,000 dropbacks, and shrunk toward an experience-group mean. Team QB strength
is a mixture over starter candidates:

\[
E[QB]=\sum_q p_q\mu_q
\]

\[
Var(QB)=\sum_q p_q(\sigma_q^2+\mu_q^2)-E[QB]^2
\]

This retains uncertainty from both player performance and the starter decision.

## Roster continuity

The package calculates returning snap shares from prior-season snaps and the
target-season roster, including offense, defense, special teams, and available
position groups. A ridge model learns how continuity and player movement
explain early-season residual performance.

## Coaching priors

Coach and coordinator residual effects are partially pooled by unit. New
coaches receive the unit league mean and the historical between-coach
uncertainty rather than an invented fixed bonus or penalty.

## Opponent adjustment

The play-by-play feature layer aggregates team-game efficiency, then fits:

\[
Metric=Intercept+TeamEffect+OpponentAllowedEffect+\epsilon
\]

with ridge shrinkage. Team effects become opponent-adjusted season metrics used
by the prior builder.

## 2026 freeze rule

Actual 2026 priors must not be frozen until:

1. 2020–2025 play-by-play, rosters, snaps, depth charts, and injuries are
   successfully ingested;
2. timestamped odds are attached at the exact prediction horizons;
3. prior hyperparameters are tuned on historical preseason replays;
4. 2024 is used for selection and 2025 remains untouched final validation;
5. final-cut and Week 1 starter probabilities are timestamped.

The current YAML values are statistically reasonable starting points, but are
explicitly marked provisional.
