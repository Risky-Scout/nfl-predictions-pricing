# Source Gaps Remaining After the Locked Stack

The selected stack covers the essential football, result, market, and injury
layers. The following inputs are not automatically solved by that selection.

## 1. Timestamped forecast weather

Spreadspoke and nflverse contain observed or reference weather, not necessarily
the forecast known at each prediction horizon. Before production, select a
forecast/archive provider and retain:

```text
forecast_issued_utc
forecast_valid_utc
lead_time_hours
temperature
wind_speed
wind_gust
precipitation_probability
precipitation_amount
snowfall
visibility
expected_roof_status
```

Observed weather may be used only as a clearly marked research proxy.

## 2. 2020 attendance and crowd restrictions

Limited-attendance games need an explicit home-field feature. This may require
a separate attendance archive if it is not present in the chosen schedule
source.

## 3. Coaching and play-caller announcement times

nflverse schedules preserve coach names, but exact coordinator/play-caller
announcement timestamps may require official team releases or a licensed
provider.

## 4. Official NFL injury export automation

The no-key NFL archive fallback is reproducible only when exported tables retain
team labels. The adapter intentionally refuses to infer teams from page order.
For unattended production, use the Sportradar-compatible adapter.

## 5. Paid historical odds access

The Odds API historical endpoint requires an eligible account and sufficient
credits. The package generates and prices the request plan before any paid
backfill is launched.

## 6. Actual 2026 prior inputs

The prior engine is implemented, but actual 2026 values require the completed
2020–2025 backfill plus timestamped 2026 roster, injury, depth-chart, starter,
and coaching snapshots.
