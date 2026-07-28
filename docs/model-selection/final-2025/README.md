# Frozen 2025 Final Evaluation

The untouched 2025 final evaluation completed successfully.

| Market | Final production decision | Production model |
|---|---|---|
| Moneyline | Market baseline | `market_t10_canonical` |
| ATS | Market fallback | `market_t10_canonical` |
| Total | Market baseline | `market_t10_canonical` |

The ATS challenger slightly improved log loss but worsened Brier
score, so it failed the predeclared joint scoring gate and reverted
to the canonical market baseline.

No post-test tuning, feature changes, or recalibration are permitted.
