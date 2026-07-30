# Purchased data (do not delete)

## odds_closing_dev_2022_2024.parquet
De-vigged **closing** NFL odds (moneyline/spread/total) for the development
seasons **2022, 2023, 2024**, purchased from The Odds API historical endpoint.

- **Cost:** 12,169 credits (real money) — this file must not live only on one laptop.
- **Coverage:** 834/854 dev games matched (97.7%), median snapshot ≈19 min pre-kickoff.
- **Rows:** 472,376 book-attributed quote rows (see `.manifest.json`).
- **Integrity:** verify with `shasum -a 256 -c odds_closing_dev_2022_2024.sha256`.
- **2025 is deliberately NOT here** — it is the spent holdout and stays proxy-only.

Derived REAL-CLOSING benchmark (small, in `outputs/`):
`outputs/real_closing_benchmark_2022_2024.parquet`.

### Backup command (off-repo copy)
```bash
# keep a second copy somewhere durable (cloud bucket / external drive)
cp data/purchased/odds_closing_dev_2022_2024.parquet <backup-location>/
shasum -a 256 -c data/purchased/odds_closing_dev_2022_2024.sha256   # integrity check
```

### Regeneration (would re-spend ~12,180 credits — avoid)
`scripts/buy_closing_odds_dev.py` re-purchases the same snapshots. Only run if this
file is lost and no backup exists.
