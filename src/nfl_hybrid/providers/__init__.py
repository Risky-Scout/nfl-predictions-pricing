"""Provider-neutral live/current data integrations (Fix 5).

Distinct from :mod:`nfl_hybrid.data.providers`, which holds the historical
backfill/odds adapters (nflverse, The Odds API, Sportradar injuries,
Spreadspoke). This package holds 2026 LIVE data providers -- currently just
BALLDONTLIE (:mod:`nfl_hybrid.providers.balldontlie`) -- whose job is to
produce the SAME canonical NFL games/plays/stats/roster/injury schemas the
historical estate already uses, behind a strict finality/completeness gate,
so downstream pregame-state builders never need to know whether a completed
game came from historical nflverse backfill or 2026 live ingestion.
"""
