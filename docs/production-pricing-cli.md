# Production Pricing CSV CLI

The frozen 2025 final test selected the canonical T-10 market
baseline for moneyline, ATS, and total. The pricing CLI therefore
requires `model_probability` to equal `market_probability`.

A wager can still show positive value when a bookmaker's offered
price is better than the consensus-derived fair price.

## Required input columns

- `game_id`
- `market`
- `selection`
- `market_probability`

`model_probability` is optional and defaults to
`market_probability`.

## Supported markets and selections

- `moneyline`: `home`, `away`, `tie`
- `ats`: `home`, `away`
- `total`: `over`, `under`

For ATS and total, provide `push_probability`. Moneyline rows must
use zero.

Provide either `offered_decimal` or `offered_american`.

For an actionable BET/NO_BET result, provide
`probability_lower` and `probability_upper`. Without uncertainty
bounds, the CLI returns `ABSTAIN`.

## Core formulas

For win probability \(p\), push probability \(q\), and offered
decimal odds \(d\):

- fair decimal odds: \((1-q)/p\)
- offered break-even probability: \((1-q)/d\)
- EV per $1: \(p(d-1) - (1-p-q)\)
- ROI: identical to EV per $1 for a $1 stake

A `BET` requires positive EV and positive edge at the supplied
lower probability bound. Otherwise the result is `NO_BET`.
Missing required information or stale quotes produce `ABSTAIN`.
