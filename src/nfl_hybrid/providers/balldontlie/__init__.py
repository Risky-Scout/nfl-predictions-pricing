"""BALLDONTLIE (BDL) NFL live-data provider integration (Fix 5).

Pipeline stages, each its own module:

    client.py           -- authenticated HTTP client, pagination, raw JSON
    team_crosswalk.py   -- BDL team id/abbreviation -> canonical_team_id
    canonical.py         -- BDL raw dicts -> canonical games/team_stats/
                             player_stats/injuries/roster DataFrames
    plays.py             -- BDL raw plays -> canonical provider-neutral PBP
    finality.py           -- status_state finality rule, result_available_at_utc,
                             per-family completeness gates (can_update_*)
    raw_store.py          -- NFL_LIVE_DATA_ROOT raw snapshot provenance/idempotence
    parity.py             -- historical/live parity matrix (docs/FIX5_...)

See ``docs/BDL_LIVE_INTEGRATION.md`` for the full provider contract and
``docs/BDL_PARITY_MATRIX.md`` / ``docs/BDL_PARITY_MATRIX.json`` for the
historical/live feature-family parity report.
"""
