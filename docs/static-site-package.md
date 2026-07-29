# Static NFL Pricing Site Package

This generator creates a self-contained local static package suitable
for editorial review or downstream website integration.

Generated files:

- `index.html`: predictions and pricing board
- `games/*.html`: one page per game
- `performance.html`: untouched 2025 final-test results
- `methodology.html`: formulas, uncertainty, and limitations
- `downloads/pricing.csv`: published pricing data
- final-test CSV downloads
- frozen production specification
- `site_manifest.json`
- local CSS with no external dependency

The package explicitly records that it has not been deployed to
wizardofodds.com.

The frozen production architecture is the canonical T-10 market
baseline for moneyline, ATS, and total. A BET label therefore means
that the available offered price has positive conservative value
relative to the consensus-derived fair price. It does not mean that
an independently trained model defeated the market.
