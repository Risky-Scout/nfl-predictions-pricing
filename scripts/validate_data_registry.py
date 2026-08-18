"""Report whether every registered external-data asset resolves correctly.

Set NFL_MODEL_DATA_ROOT to point at your local copy of the historical data
estate (see src/nfl_hybrid/data/external_data.py for the full registry).

Run: PYTHONPATH=src python scripts/validate_data_registry.py
Or, once installed: nfl-hybrid-validate-data
"""

from __future__ import annotations

from nfl_hybrid.data.external_data import main

if __name__ == "__main__":
    raise SystemExit(main())
