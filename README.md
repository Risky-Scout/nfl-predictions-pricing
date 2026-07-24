# NFL Predictions and Pricing

An auditable Python research framework for predicting and pricing NFL
Moneyline, Against the Spread, and game-total markets.

The project combines reconstructed spreadsheet baselines with modern,
leakage-safe modeling techniques, including adjusted Elo, play-by-play
efficiency, quarterback priors, roster continuity, market-residual modeling,
probability calibration, score simulation, and chronological walk-forward
validation.

## Primary objectives

- Predict home and away scoring distributions.
- Estimate Moneyline win probabilities.
- Estimate spread-cover and push probabilities.
- Estimate over, under, and total-push probabilities.
- Compare football-only and market-augmented models.
- Reproduce and audit the logic of the original spreadsheet models.
- Build statistically defensible preseason priors for the 2026 NFL season.
- Extend the framework to live, in-play prediction.
