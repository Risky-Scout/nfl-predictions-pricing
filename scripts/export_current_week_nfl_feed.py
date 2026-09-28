"""Current-week public pregame board: every game at its latest pregame snapshot.

WHY THIS EXISTS ALONGSIDE export_wizard_nfl_pricing.py. That exporter
publishes ONE certified card: every game shares one cutoff and one run
manifest. The current-week board is a different shape -- each game advances
through OPEN, MID and CLOSE on its own clock, so at any moment the live page
carries a mixture: games already past their individual CLOSE, and games still
updating. One run manifest cannot describe that, so the board is assembled per
game instead.

LATEST AVAILABLE PREGAME SNAPSHOT, PER GAME. The public product is the model's
prediction for the current week BEFORE the games, so a game becomes public as
soon as its OPEN snapshot exists and is then REPRESENTED by the most advanced
snapshot it has reached: CLOSE ahead of MID ahead of OPEN. The board therefore
fills up early in the week and sharpens as the week proceeds, instead of
appearing one game at a time in the final hour before each kickoff.

    PUBLICATION_PREFERENCE = (CLOSE, MID, OPEN)

THIS PRIORITY IS PUBLIC PRESENTATION ONLY. It selects which immutable snapshot
REPRESENTS a game on the board. It does not merge stages, does not rewrite one
stage with another, and does not touch the forecast ledger, the snapshot
performance ledger, the evaluation ledger or the recalibration inputs -- all of
which keep grading OPEN, MID and CLOSE separately and chronologically. Display
preference and evaluation provenance are deliberately separate concerns:
nothing here can let a later stage's information reach an earlier stage's
record, because nothing here writes a stage record at all.

PROVENANCE TRAVELS WITH EACH ROW. A board whose rows come from different
stages would be dishonest without saying so, so each published game carries
the stage it was published from plus the instant the model produced it. That
makes the board's own contract a superset of the certified card's, published
under its own ``schema_version`` (``wizard-nfl-pricing-v3``); the certified
``wizard-nfl-pricing-v2`` card is untouched, and its exporter is not modified.

ONE-WAY DOOR AT CLOSE. A game's CLOSE projection is its final public word.
Once published it is recorded in the sidecar and, on every later publication,
the frozen bytes are reused and the freshly assembled entry must agree with
them. A later weekly update that would change an already-closed game fails the
whole publication closed rather than rewriting history -- so a game's
published close can never drift, even if its forecast ledger were somehow
re-derived. OPEN and MID carry no such door: they are expected to be
superseded, which is the whole point of the preference order above.

A WEEK WITH NOTHING SNAPSHOTTED YET IS STILL PUBLISHED. Between a week's last
kickoff and the next week's first OPEN there is genuinely nothing to show, and
refusing to publish then is what left the page serving a Week-1 card for eight
days -- every attempt exited non-zero, so ``latest.json`` was never replaced
and the site silently advertised a finished week as current. An empty envelope
for the CORRECT week is the honest answer, and publication never falls back to
an older week merely because the current one has no snapshots yet. Equally, a
week where only SOME games have opened publishes exactly those games: one
unopened market never holds the rest of the slate back.

NOTHING IS INVENTED. A game with no usable snapshot is simply absent from the
board; it is never published with a placeholder line, a stale line from another
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
TOP_LEVEL_KEY_ORDER = _v2.TOP_LEVEL_KEY_ORDER

# The certified card's per-game contract, reused verbatim. It is also exactly
# the shape the frozen-CLOSE sidecar stores, so a sidecar written before the
# board carried provenance still validates against it unchanged.
BASE_GAME_KEY_ORDER = _v2.GAME_KEY_ORDER

# Best-available order: a game is REPRESENTED by the latest pregame stage it
# has reached. A game is public from OPEN onwards, and MID then CLOSE replace
# that public representation as each becomes available. The immutable OPEN, MID
# and CLOSE records themselves are untouched by this ordering -- see the module
# docstring.
PUBLICATION_PREFERENCE = (st.STAGE_CLOSE, st.STAGE_MID, st.STAGE_OPEN)

# The board's own public contract: the certified card's keys plus the two
# provenance fields a mixed-stage board cannot be honest without. Published
# under its own schema_version so the certified v2 card stays byte-identical
# and its immutable archives stay valid.
SCHEMA_VERSION = "wizard-nfl-pricing-v3"
STAGE_KEY = "snapshot_stage"
MODEL_GENERATED_KEY = "model_generated_at_utc"
GAME_KEY_ORDER = BASE_GAME_KEY_ORDER + (STAGE_KEY, MODEL_GENERATED_KEY)

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


def _in_base_key_order(game: dict) -> dict:
    """Restore the frozen per-game key order.

    The sidecar is stored with ``sort_keys=True`` so it is diffable, which
    loses the key order on the way back in. The ORDER is part of the public
    contract, so it is rebuilt from the frozen values rather than trusting
    whatever order a JSON round-trip happened to produce.

    The sidecar stores the CERTIFIED-CARD shape, not the board shape: stage and
    model instant are board provenance recorded beside the frozen game rather
    than inside it, which is what lets a sidecar written before the board
    carried provenance keep validating here unchanged.
    """
    missing = [key for key in BASE_GAME_KEY_ORDER if key not in game]
    extra = [key for key in game if key not in BASE_GAME_KEY_ORDER]
    if missing or extra:
        _fail(
            "frozen published game does not match the published per-game contract "
            f"(missing={missing} unexpected={extra})"
        )
    return {key: game[key] for key in BASE_GAME_KEY_ORDER}


def _board_game(base_game: dict, *, stage: str, model_generated_at_utc: str) -> dict:
    """One board row: the certified per-game fields plus their provenance."""
    row = dict(base_game)
    row[STAGE_KEY] = st.validate_stage(stage)
    row[MODEL_GENERATED_KEY] = model_generated_at_utc
    ordered = {key: row[key] for key in GAME_KEY_ORDER}
    assert tuple(ordered.keys()) == GAME_KEY_ORDER
    return ordered


def _model_generated_at_utc(record: dict, *, game_id: str) -> str:
    """When the model PRODUCED this snapshot, read from the immutable row.

    ``run_created_at_utc`` is the same instant the certified exporter publishes
    as its card-wide ``generated_at_utc``; on a mixed-stage board it differs per
    game, so it has to travel with the row. Never the export's own clock, never
    the stage cutoff, and never ``created_at_utc``.
    """
    raw = record.get("run_created_at_utc")
    if not isinstance(raw, str) or not raw:
        _fail(
            f"{game_id}: forecast of record carries no run_created_at_utc -- the instant the model "
            "produced this snapshot is never invented"
        )
    return _v2._format_utc_z(_v2._parse_utc_instant(raw, field_name=f"{game_id}: run_created_at_utc"))


def _frozen_model_generated_at_utc(frozen: dict, *, game_id: str) -> str:
    """The model instant a frozen CLOSE was published with.

    A sidecar written before the board carried provenance has no model instant
    to keep, so the publication instant it DOES carry is used instead. That is a
    real instant recorded by this game's own freeze -- never re-derived from a
    later snapshot, which would be exactly the lookahead this board must not
    introduce.
    """
    for key in (MODEL_GENERATED_KEY, "frozen_at_utc"):
        raw = frozen.get(key)
        if isinstance(raw, str) and raw:
            return _v2._format_utc_z(
                _v2._parse_utc_instant(raw, field_name=f"{game_id}: published close state {key}")
            )
    _fail(f"{game_id}: frozen published close state carries no instant to publish as its model stamp")


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

    Resolved INDEPENDENTLY per game, so partial availability is normal rather
    than exceptional: a game whose market has not opened contributes a
    ``NO_SNAPSHOT_YET`` entry and nothing else, and never delays the games that
    have opened.
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
        kickoff, base_game = _v2._build_public_game(record)
        model_generated = _model_generated_at_utc(record, game_id=game_id)

        frozen = previous.get(game_id)
        if frozen is not None:
            # The one-way door. Reached only for a game that has already
            # published a CLOSE, so a MID or OPEN selected above can never
            # reopen it: the frozen bytes win and must still agree.
            frozen_game = _in_base_key_order(frozen["published_game"])
            if base_game != frozen_game:
                _fail(
                    f"{game_id} was already published at CLOSE and a later pass would change it -- "
                    "an already-closed game is never mutated. Refusing to publish the whole feed. "
                    f"frozen={frozen_game} recomputed={base_game}"
                )
            base_game = frozen_game
            stage = st.STAGE_CLOSE
            model_generated = _frozen_model_generated_at_utc(frozen, game_id=game_id)
        elif stage == st.STAGE_CLOSE:
            state[game_id] = {
                "published_game": base_game,
                "frozen_at_utc": generated_at_utc,
                "forecast_prediction_hash": record.get("prediction_hash"),
                "snapshot_at_utc": str(record.get("target_cutoff_utc")),
                MODEL_GENERATED_KEY: model_generated,
            }

        stages[game_id] = stage
        dated.append((kickoff, _board_game(base_game, stage=stage, model_generated_at_utc=model_generated)))

    # A game that has already been frozen must never silently vanish from the
    # board because its snapshot file moved: the published close outlives the
    # ledger lookup that produced it.
    for game_id, frozen in sorted(previous.items()):
        if game_id in stages:
            continue
        frozen_game = _in_base_key_order(frozen["published_game"])
        stages[game_id] = st.STAGE_CLOSE
        dated.append((
            _v2._parse_utc_instant(frozen_game["kickoff_utc"], field_name=f"{game_id}: kickoff_utc"),
            _board_game(
                frozen_game,
                stage=st.STAGE_CLOSE,
                model_generated_at_utc=_frozen_model_generated_at_utc(frozen, game_id=game_id),
            ),
        ))
        unpublishable.pop(game_id, None)

    # ZERO SNAPSHOTS IS A LEGITIMATE WEEK, NOT A FAILURE.
    #
    # Between a week's last kickoff and the next week's first OPEN there is
    # genuinely nothing to show. Treating that as fail-closed is what
    # left the public page serving the Week-1 card for eight days: every
    # publication attempt exited non-zero, so latest.json was never replaced
    # and the site silently advertised a stale week as current.
    #
    # Publishing an empty envelope for the CORRECT week is the honest answer.
    # It says "this is the current week and nothing has opened yet" instead of
    # "here is a week that finished days ago". Fail-closed is reserved for
    # actual corruption -- a contradicted frozen close, an unidentifiable
    # record, a malformed game -- all of which still raise above.
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
        # Both are successes. AWAITING_FIRST_SNAPSHOT simply names the state so
        # an operator reading a log can tell "no game has opened yet" apart
        # from "something went wrong", without either failing the workflow.
        "status": "OK" if feed["games"] else "OK_AWAITING_FIRST_SNAPSHOT",
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
