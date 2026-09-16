"""Assemble and write the current-week public feed for this instant.

Resolves the current week's card the same way the snapshot sweep does -- by
the certified TUE card cutoff, so the two can never disagree about which week
it is -- then hands it to the current-week feed exporter, which freezes each
game at its own CLOSE.

This writes the feed into the ARTIFACT tree. It does not touch the nginx path:
``scripts/publish_wizard_nfl_local.py`` performs the atomic publication, and
the orchestrator calls it next. Separating the two keeps assembly failures
from ever reaching the served directory.

Usage:
  python scripts/publish_2026_current_week.py \\
      --artifact-root $NFL_MODEL_ARTIFACT_ROOT \\
      --output $NFL_MODEL_ARTIFACT_ROOT/public/wizardofodds/nfl-pricing/latest.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.production import run_2026 as prod  # noqa: E402

_stage_spec = importlib.util.spec_from_file_location(
    "_run_2026_stage_snapshots", REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py"
)
_stage = importlib.util.module_from_spec(_stage_spec)
_stage_spec.loader.exec_module(_stage)

_feed_spec = importlib.util.spec_from_file_location(
    "_export_current_week_nfl_feed", REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py"
)
_feed = importlib.util.module_from_spec(_feed_spec)
_feed_spec.loader.exec_module(_feed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--market-capture-manifest", default=None)
    args = parser.parse_args(argv)

    artifact_root = Path(args.artifact_root)
    as_of_utc = prod._as_utc(args.as_of) if args.as_of else prod.utc_now()

    games = prod.filter_reg_post(prod.load_games_population_with_provenance()[0])
    card, card_info = _stage.resolve_current_card(games, as_of_utc)
    if card_info["status"] != "OK":
        print(json.dumps({"status": "NO_CURRENT_CARD", "card": card_info}, indent=2))
        return 0

    quotes = None
    if args.market_capture_manifest:
        evidence = prod.evaluate_live_market_source(Path(args.market_capture_manifest))
        if evidence["registered"] and evidence["source"] is not None:
            quotes = evidence["source"].quotes
    observations = _stage.discover_open_observations(card, quotes=quotes)

    try:
        result = _feed.publish_current_week_feed(
            card=card,
            forecast_dir=artifact_root / "production-2026" / "forecast-ledger" / "TUE",
            artifact_root=artifact_root,
            season=card_info["season"],
            week=card_info["week"],
            generated_at_utc=as_of_utc.isoformat(),
            horizon="TUE",
            output_path=Path(args.output),
            open_observations=observations,
        )
    except _feed.WizardExportError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
