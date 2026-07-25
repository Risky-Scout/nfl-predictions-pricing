# Historical Odds Backfill Policy

## Standard prediction horizons

The first research pass captures one snapshot per game at:

| Horizon | Minutes before kickoff |
|---|---:|
| 7 days | 10,080 |
| 72 hours | 4,320 |
| 24 hours | 1,440 |
| 6 hours | 360 |
| 1 hour | 60 |
| 15 minutes | 15 |

These horizons support separate pregame models and measure whether a signal
persists as the market incorporates information.

## Why the adapter retains every bookmaker

The raw table keeps every bookmaker/outcome quote. Consensus features are
derived later. This avoids three common errors:

1. choosing a book after observing the result;
2. mixing a spread from one book with a price from another;
3. losing cross-book dispersion, which is itself predictive context.

## Quote identity

A quote is identified by:

```text
provider_event_id
bookmaker_id
market_type
outcome_name
line_value
snapshot_utc
```

The raw provider timestamp and bookmaker/market update timestamp are both
retained.

## De-vigging

The initial implementation uses multiplicative normalization within each
bookmaker, event, market, line, and snapshot:

\[
p_i^{fair}=\frac{p_i^{raw}}{\sum_j p_j^{raw}}
\]

The original price and overround remain available. Later model refinement can
compare multiplicative, power, and Shin-style methods.

## Paid-query guardrail

The historical endpoint can consume substantial plan quota. The backfill:

1. generates a snapshot plan;
2. de-duplicates timestamps;
3. estimates credits;
4. writes the plan and estimate;
5. stops unless the user supplies both `--confirm-cost` and a sufficient
   `--max-credits` ceiling.

## Closing line

“Closing” is not a provider-independent field. It is computed as the latest
active pre-kickoff quote for a specified bookmaker or consensus rule, with a
minimum distance from kickoff to avoid accidentally using in-play quotes.
