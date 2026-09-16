"""Current-week wizard-nfl-pricing-v2 feed, frozen game by game at CLOSE.

WHY THIS EXISTS ALONGSIDE export_wizard_nfl_pricing.py. That exporter
publishes ONE certified card: every game shares one cutoff and one run
manifest. The current-week feed is a different shape -- each game advances
through OPEN, MID and CLOSE on its own clock, so at any moment the live page
carries a mixture: games already past their individual CLOSE, and games still
updating. One run manifest cannot describe that, so the feed is assembled per
game instead.

THE PUBLIC CONTRACT IS UNCHANGED. Output is exactly ``wizard-nfl-pricing-v2``
-- same schema_version, same top-level keys, same eleven game keys in the same
order, same publication gate -- produced by the v2 exporter's OWN translation
and serialization helpers rather than a second implementation of them. Which
stage a game was published from is operational bookkeeping, so it is recorded
in a private sidecar under the artifact root, never added to the public
payload.

ONE-WAY DOOR AT CLOSE. A game's CLOSE projection is its final public word.
Once published it is recorded in the sidecar and, on every later publication,
the frozen bytes are reused and the freshly assembled entry must agree with
them. A later weekly update that would change an already-closed game fails the
whole publication closed rather than rewriting history -- so a game's
published close can never drift, even if its forecast ledger were somehow
re-derived.

FUTURE GAMES STILL MOVE. A game that has not reached its CLOSE publishes from
the best pregame snapshot it has -- MID if there is one, otherwise OPEN -- and
is free to change on the next pass. That is the point: the page shows the
current week, not a frozen Week-1 replay.

NOTHING IS INVENTED. A game with no usable snapshot is simply absent from the
feed; it is never published with a placeholder line, a stale line from another
stage's cutoff, or a projection carried over from another week.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nfl_hybrid.production import snapshot_stages_2026 as st  # noqa: E402


def _load_v2_exporter():
    """Reuse the v2 exporter's translation, serialization and atomic-write
    helpers verbatim. Loading it has no side effect beyond defining them."""
    path = _REPO_ROOT / "scripts" / "export_wizard_nfl_pricing.py"
    spec = importlib.util.spec_from_file_location("_wizard_nfl_pricing_v2_exporter", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the v2 exporter helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = _load_v2_exporter()

WizardExportError = _v2.WizardExportError
_fail = _v2._fail
SCHEMA_VERSION = _v2.SCHEMA_VERSION
TOP_LEVEL_KEY_ORDER = _v2.TOP_LEVEL_KEY_ORDER
GAME_KEY_ORDER = _v2.GAME_KEY_ORDER

# Best-available order: a game publishes from the latest stage it has reached.
PUBLICATION_PREFERENCE = (st.STAGE_CLOSE, st.STAGE_MID, st.STAGE_OPEN)

SIDECAR_NAME = "published_close_state.json"


def sidecar_path(artifact_root: Path, *, season: int, week) -> Path:
    return (
        Path(artifact_root)
        / "production-2026"
        / "published-close-state"
        / f"season={int(season)}"
        / f"week={_week_label(week)}"
        / SIDECAR_NAME
    )


def _week_label(week) -> str:
    try:
        return f"{int(week):02d}"
    except (TypeError, ValueError):
        label = str(week).strip()
        if not label:
            _fail("week is required and is never defaulted")
        return label


def _in_public_key_order(game: dict) -> dict:
    """Restore the frozen public key order.

    The sidecar is stored with ``sort_keys=True`` so it is diffable, which
    loses the v2 key order on the way back in. The ORDER is part of the public
    contract, so it is rebuilt from the frozen values rather than trusting
    whatever order a JSON round-trip happened to produce.
    """
    missing = [key for key in GAME_KEY_ORDER if key not in game]
    extra = [key for key in game if key not in GAME_KEY_ORDER]
    if missing or extra:
        _fail(
            f"frozen published game does not match the {SCHEMA_VERSION} contract "
            f"(missing={missing} unexpected={extra})"
        )
    return {key: game[key] for key in GAME_KEY_ORDER}


def load_close_state(artifact_root: Path, *, season: int, week) -> dict:
    path = sidecar_path(artifact_root, season=season, week=week)
    if not path.is_file():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        _fail(f"published close state at {path} is not an object")
    return state


def save_close_state(artifact_root: Path, *, season: int, week, state: dict) -> Path:
    path = sidecar_path(artifact_root, season=season, week=week)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# selecting each game's publishable snapshot
# ---------------------------------------------------------------------------
def _read_forecast(forecast_dir: Path, *, game_id: str, cutoff) -> dict | None:
    safe = str(cutoff).replace(":", "").replace("+", "_")
    path = Path(forecast_dir) / f"{game_id}__{safe}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def select_publishable_snapshots(
    *, card: pd.DataFrame, forecast_dir: Path, open_observations: dict | None = None
) -> tuple[dict[str, tuple[str, dict]], dict[str, str]]:
    """For each game, the latest stage it has a forecast of record for.

    Returns ``{game_id: (stage, record)}`` plus the games with nothing
    publishable and why.
    """
    observations = open_observations or {}
    earliest = st.card_earliest_kickoff_utc(card["scheduled_kickoff_utc"])
    chosen: dict[str, tuple[str, dict]] = {}
    unpublishable: dict[str, str] = {}

    for row in card.itertuples(index=False):
        game_id = str(row.game_id)
        schedule = st.resolve_game_snapshots(
            game_id=game_id,
            scheduled_kickoff_utc=row.scheduled_kickoff_utc,
            card_earliest_kickoff_utc=earliest,
            open_observed_at_utc=observations.get(game_id),
        )
        for stage in PUBLICATION_PREFERENCE:
            cutoff = schedule.cutoff_for(stage)
            if cutoff is None:
                continue
            record = _read_forecast(forecast_dir, game_id=game_id, cutoff=cutoff)
            if record is not None:
                chosen[game_id] = (stage, record)
                break
        else:
            unpublishable[game_id] = "NO_SNAPSHOT_YET"
    return chosen, unpublishable


# ---------------------------------------------------------------------------
# assembling the feed, honouring frozen closes
# ---------------------------------------------------------------------------
def build_current_week_feed(
    *,
    card: pd.DataFrame,
    forecast_dir: Path,
    artifact_root: Path,
    season: int,
    week,
    generated_at_utc: str,
    horizon: str,
    open_observations: dict | None = None,
) -> tuple[dict, dict, dict]:
    """The feed, the close state it implies, and a per-game stage report."""
    chosen, unpublishable = select_publishable_snapshots(
        card=card, forecast_dir=forecast_dir, open_observations=open_observations
    )
    previous = load_close_state(artifact_root, season=season, week=week)
    state = dict(previous)
    stages: dict[str, str] = {}
    dated: list[tuple] = []

    for game_id, (stage, record) in sorted(chosen.items()):
        kickoff, public_game = _v2._build_public_game(record)

        frozen = previous.get(game_id)
        if frozen is not None:
            frozen_game = _in_public_key_order(frozen["published_game"])
            if public_game != frozen_game:
                _fail(
                    f"{game_id} was already published at CLOSE and a later pass would change it -- "
                    "an already-closed game is never mutated. Refusing to publish the whole feed. "
                    f"frozen={frozen_game} recomputed={public_game}"
                )
            public_game = frozen_game
            stage = st.STAGE_CLOSE
        elif stage == st.STAGE_CLOSE:
            state[game_id] = {
                "published_game": public_game,
                "frozen_at_utc": generated_at_utc,
                "forecast_prediction_hash": record.get("prediction_hash"),
                "snapshot_at_utc": str(record.get("target_cutoff_utc")),
            }

        stages[game_id] = stage
        dated.append((kickoff, public_game))

    # A game that has already been frozen must never silently vanish from the
    # feed because its snapshot file moved: the published close outlives the
    # ledger lookup that produced it.
    for game_id, frozen in sorted(previous.items()):
        if game_id in stages:
            continue
        frozen_game = _in_public_key_order(frozen["published_game"])
        stages[game_id] = st.STAGE_CLOSE
        dated.append((
            _v2._parse_utc_instant(frozen_game["kickoff_utc"], field_name=f"{game_id}: kickoff_utc"),
            frozen_game,
        ))
        unpublishable.pop(game_id, None)

    if not dated:
        _fail("no game in the current week has a publishable snapshot yet -- refusing to publish an empty feed")

    dated.sort(key=lambda pair: (pair[0], pair[1]["game_id"]))
    feed = {
        "schema_version": SCHEMA_VERSION,
        "season": _v2._validate_season(int(season)),
        "week": _v2._parse_week(week),
        "horizon": horizon,
        "generated_at_utc": _v2._format_utc_z(
            _v2._parse_utc_instant(generated_at_utc, field_name="generated_at_utc")
        ),
        "games": [game for _, game in dated],
    }
    assert tuple(feed.keys()) == TOP_LEVEL_KEY_ORDER
    for game in feed["games"]:
        assert tuple(game.keys()) == GAME_KEY_ORDER
    return feed, state, {"stages": stages, "unpublishable": unpublishable}


def publish_current_week_feed(
    *,
    card: pd.DataFrame,
    forecast_dir: Path,
    artifact_root: Path,
    season: int,
    week,
    generated_at_utc: str,
    horizon: str,
    output_path: Path,
    open_observations: dict | None = None,
) -> dict:
    """Assemble, freeze and atomically replace ``latest.json``.

    The close state is written only AFTER the feed is on disk, so a failed
    publication never leaves a game marked frozen at bytes that were never
    served.
    """
    feed, state, report = build_current_week_feed(
        card=card,
        forecast_dir=forecast_dir,
        artifact_root=artifact_root,
        season=season,
        week=week,
        generated_at_utc=generated_at_utc,
        horizon=horizon,
        open_observations=open_observations,
    )
    payload_bytes = _v2.serialize_card(feed)
    latest = _v2.write_latest(Path(output_path), payload_bytes)
    saved = save_close_state(artifact_root, season=season, week=week, state=state)

    return {
        "status": "OK",
        "season": feed["season"],
        "week": feed["week"],
        "game_count": len(feed["games"]),
        "frozen_close_games": sorted(state),
        "stages": report["stages"],
        "unpublishable": report["unpublishable"],
        "latest_path": str(latest),
        "close_state_path": str(saved),
        "published_sha256": sha256(payload_bytes).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--card-json", required=True, help="JSON list of {game_id, scheduled_kickoff_utc}")
    parser.add_argument("--forecast-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True)
    parser.add_argument("--horizon", required=True, choices=list(_v2.ALLOWED_HORIZONS))
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--open-observations-json", default=None)
    args = parser.parse_args(argv)

    card = pd.DataFrame(json.loads(Path(args.card_json).read_text(encoding="utf-8")))
    observations = (
        {}
        if args.open_observations_json is None
        else json.loads(Path(args.open_observations_json).read_text(encoding="utf-8"))
    )
    try:
        result = publish_current_week_feed(
            card=card,
            forecast_dir=Path(args.forecast_dir),
            artifact_root=Path(args.artifact_root),
            season=args.season,
            week=args.week,
            generated_at_utc=args.generated_at_utc,
            horizon=args.horizon,
            output_path=Path(args.output),
            open_observations=observations,
        )
    except WizardExportError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
