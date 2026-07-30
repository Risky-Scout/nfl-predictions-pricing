#!/usr/bin/env bash
# 2026 weekly operating loop. Usage: scripts/run_week.sh <SEASON> <WEEK>
# Requires: PYTHONPATH=src (set below), and for live lines THE_ODDS_API_KEY.
set -euo pipefail

SEASON="${1:?usage: run_week.sh SEASON WEEK}"
WEEK="${2:?usage: run_week.sh SEASON WEEK}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
PY=".venv/bin/python"
BACKFILL="data/backfill_2020_2025"
SEASON_DIR="outputs/season_${SEASON}"
mkdir -p "$SEASON_DIR"

echo "== [1/6] incremental data refresh (open sources, skip large tables) =="
if [ -n "${SPREADSPOKE_CSV_PATH:-}" ]; then
  $PY -m nfl_hybrid.data.cli --output-dir "$BACKFILL" --seasons "2020-${SEASON}" \
    --spreadspoke-csv "$SPREADSPOKE_CSV_PATH" --skip-large-tables open-sources || \
    echo "  (refresh failed; continuing on existing backfill)"
else
  echo "  SPREADSPOKE_CSV_PATH unset; skipping refresh, using existing backfill."
fi

echo "== [2/6] fetch current-week lines =="
LINES_CSV="$SEASON_DIR/lines_wk${WEEK}.csv"
if [ -n "${THE_ODDS_API_KEY:-}" ]; then
  echo "  Odds API keyed. current-odds costs regions*markets = 3 credits per call."
  $PY scripts/fetch_current_week_lines.py --season "$SEASON" --week "$WEEK" --output "$LINES_CSV" || \
    echo "  (fetch failed; provide $LINES_CSV manually from the template)"
else
  echo "  No THE_ODDS_API_KEY. Provide $LINES_CSV using examples/games_2026_week1_placeholder.csv as a template."
fi

echo "== [3/6] generate betting card (PROVISIONAL -> stake ZERO until live gate passes) =="
CARD="$SEASON_DIR/card_wk${WEEK}.csv"
if [ -f "$LINES_CSV" ]; then
  $PY -m nfl_hybrid.pricing.predict_week --season "$SEASON" --week "$WEEK" \
    --games "$LINES_CSV" --output "$CARD"
else
  echo "  No lines CSV; card not generated."
fi

echo "== [4/6] log immutable pre-kickoff prediction record =="
$PY -c "
import sys, pandas as pd
from datetime import datetime, timezone
from nfl_hybrid.operations import log_week_predictions
card = pd.read_csv('$CARD')
ts = datetime.now(timezone.utc).isoformat()
p = log_week_predictions(card, output_dir='$SEASON_DIR', season=$SEASON, week=$WEEK, logged_at_utc=ts)
print('  logged', p)
" || echo "  (logging skipped/failed)"

echo "== [5/6] post-resolution scoring + drift monitor (run AFTER games complete) =="
echo "  scripts/score_week.py --season $SEASON --week $WEEK   # scores logged record vs market"

echo "== [6/6] live-promotion gate =="
echo "  A PROVISIONAL market stakes only after >=8 live weeks, cumulative log-loss gain>0,"
echo "  and live pick-accuracy 95% CI lower>0.524. Drift alarm demotes immediately."
echo "done."
