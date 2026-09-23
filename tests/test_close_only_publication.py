"""Proofs for CLOSE-only publication and the zero-CLOSE current week.

THE DEFECT. The publisher refused to publish a week with nothing closed yet:

    FAIL_CLOSED: no game in the current week has a publishable snapshot yet
    -- refusing to publish an empty feed              (exit 2 -> workflow 10)

Between a week's last kickoff and the next week's first CLOSE that state is
entirely legitimate, so every publication attempt exited non-zero, latest.json
was never replaced, and the public page served the Week-1 card generated
``2026-09-15T05:14:43Z`` for eight days while advertising it as current.

Hermetic: ``tmp_path`` estates and synthetic forecast records. No network, no
provider, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.production import snapshot_stages_2026 as st

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


feed = _load(REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py", "_feed_close_only")
entry = _load(REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py", "_entry_close_only")

# One card: a Sunday game and a Monday-nighter, the rollover shape.
SUNDAY = pd.Timestamp("2026-09-20T17:00:00Z")
MONDAY = pd.Timestamp("2026-09-22T00:15:00Z")
NEXT_THURSDAY = pd.Timestamp("2026-09-25T00:15:00Z")
OPEN_AT = pd.Timestamp("2026-09-14T14:00:00Z")


def _record(game_id, *, cutoff, kickoff, margin, spread):
    prediction = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "season": 2026, "week": "2", "season_type": "REG",
        "home_team_id": "KC", "away_team_id": "BUF",
        "scheduled_kickoff_utc": str(kickoff).replace("+00:00", "Z").replace(" ", "T"),
        "prediction": {
            "model_status": "OOF", "predicted_margin": margin,
            "predicted_total": 45.0, "model_config_hash": "cfg",
        },
        "markets": {
            name: {
                "status": "OK",
                "market": {
                    "consensus_line": line, "eligible_books": 4,
                    "selected_returned_snapshot_timestamps": ["2026-09-20T15:55:00Z"],
                },
            }
            for name, line in (("ATS", spread), ("TOTAL", 44.5))
        },
    }
    record = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "run_id": "r1", "run_created_at_utc": "2026-09-14T15:30:00Z",
        "created_at_utc": "2026-09-14T15:30:00Z", "prediction": prediction,
    }
    record["prediction_hash"] = feed._v2._sha256_hex(prediction)
    return record


@pytest.fixture
def estate(tmp_path):
    card = pd.DataFrame(
        [
            {"game_id": "SUN", "scheduled_kickoff_utc": SUNDAY},
            {"game_id": "MON", "scheduled_kickoff_utc": MONDAY},
        ]
    )
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    forecast_dir.mkdir(parents=True)
    return {
        "root": tmp_path, "card": card, "forecast_dir": forecast_dir,
        "observations": {"SUN": OPEN_AT, "MON": OPEN_AT},
        "output": tmp_path / "public" / "latest.json",
    }


def _write(estate, game_id, stage, *, margin=6.5, spread=-4.5):
    kickoff = SUNDAY if game_id == "SUN" else MONDAY
    schedule = st.resolve_game_snapshots(
        game_id=game_id, scheduled_kickoff_utc=kickoff,
        card_earliest_kickoff_utc=SUNDAY,
        open_observed_at_utc=estate["observations"][game_id],
    )
    cutoff = schedule.cutoff_for(stage)
    record = _record(game_id, cutoff=cutoff, kickoff=kickoff, margin=margin, spread=spread)
    safe = str(cutoff).replace(":", "").replace("+", "_")
    (estate["forecast_dir"] / f"{game_id}__{safe}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )


def _publish(estate, *, week=2, generated_at="2026-09-22T06:00:00Z"):
    return feed.publish_current_week_feed(
        card=estate["card"], forecast_dir=estate["forecast_dir"],
        artifact_root=estate["root"], season=2026, week=week,
        generated_at_utc=generated_at, horizon="TUE",
        output_path=estate["output"], open_observations=estate["observations"],
    )


def _served(estate) -> dict:
    return json.loads(estate["output"].read_text())


# ===========================================================================
# 3. A week with nothing closed yet still publishes.
# ===========================================================================
def test_a_zero_close_week_publishes_successfully(estate):
    result = _publish(estate)
    assert result["status"] == "OK_AWAITING_FIRST_CLOSE"
    assert result["game_count"] == 0


def test_the_zero_close_envelope_names_the_correct_week(estate):
    _publish(estate)
    served = _served(estate)
    assert served["season"] == 2026
    assert served["week"] == 2
    assert served["games"] == []


def test_the_zero_close_envelope_is_schema_valid(estate):
    _publish(estate)
    served = _served(estate)
    assert served["schema_version"] == "wizard-nfl-pricing-v2"
    assert tuple(served.keys()) == feed.TOP_LEVEL_KEY_ORDER
    assert served["horizon"] == "TUE"
    assert served["generated_at_utc"].endswith("Z")


def test_the_zero_close_publication_is_not_a_failure(estate):
    """It must not raise, so the workflow cannot exit 10."""
    result = _publish(estate)
    assert result["status"].startswith("OK")
    assert estate["output"].is_file()


def test_a_stale_older_week_is_replaced_by_the_empty_current_week(estate):
    """Requirement D: publication never falls back to an older week."""
    estate["output"].parent.mkdir(parents=True, exist_ok=True)
    estate["output"].write_text(
        json.dumps({"schema_version": "wizard-nfl-pricing-v2", "season": 2026,
                    "week": 1, "horizon": "TUE",
                    "generated_at_utc": "2026-09-15T05:14:43Z", "games": [{"stale": True}]})
    )
    _publish(estate)
    served = _served(estate)
    assert served["week"] == 2
    assert served["games"] == []


# ===========================================================================
# 4-5. The feed accumulates CLOSE-backed games.
# ===========================================================================
def test_the_first_close_appears_alone(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    result = _publish(estate)
    assert result["status"] == "OK"
    assert [g["game_id"] for g in _served(estate)["games"]] == ["SUN"]


def test_later_closes_accumulate(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    _write(estate, "MON", st.STAGE_CLOSE, margin=3.5, spread=-2.5)
    result = _publish(estate, generated_at="2026-09-21T23:30:00Z")

    assert result["game_count"] == 2
    assert sorted(g["game_id"] for g in _served(estate)["games"]) == ["MON", "SUN"]
    assert result["frozen_close_games"] == ["MON", "SUN"]


def test_an_already_closed_game_stays_frozen_as_others_join(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    frozen = [g for g in _served(estate)["games"] if g["game_id"] == "SUN"][0]

    _write(estate, "MON", st.STAGE_CLOSE, margin=3.5, spread=-2.5)
    _publish(estate, generated_at="2026-09-21T23:30:00Z")
    assert [g for g in _served(estate)["games"] if g["game_id"] == "SUN"][0] == frozen


# ===========================================================================
# 6. OPEN and MID are NEVER a public fallback.
# ===========================================================================
def test_the_publication_preference_is_close_only():
    assert feed.PUBLICATION_PREFERENCE == (st.STAGE_CLOSE,)


@pytest.mark.parametrize("stage", [st.STAGE_OPEN, st.STAGE_MID])
def test_a_game_with_only_open_or_mid_is_not_published(estate, stage):
    _write(estate, "SUN", stage)
    result = _publish(estate)
    assert result["game_count"] == 0
    assert _served(estate)["games"] == []
    assert result["unpublishable"]["SUN"] == "NO_SNAPSHOT_YET"


def test_open_and_mid_do_not_mask_a_missing_close(estate):
    """Both present for a game, still not published until it closes."""
    _write(estate, "SUN", st.STAGE_OPEN)
    _write(estate, "SUN", st.STAGE_MID)
    assert _publish(estate)["game_count"] == 0

    _write(estate, "SUN", st.STAGE_CLOSE)
    result = _publish(estate, generated_at="2026-09-20T16:30:00Z")
    assert result["stages"] == {"SUN": st.STAGE_CLOSE}


def test_the_public_payload_never_names_a_stage(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    served = _served(estate)
    assert "snapshot_stage" not in served
    for game in served["games"]:
        assert "snapshot_stage" not in game
        assert tuple(game.keys()) == feed.GAME_KEY_ORDER


# ===========================================================================
# 7. Invalidated artifacts cannot reach the page.
# ===========================================================================
def test_an_invalidated_artifact_is_not_reachable_by_the_feed(estate):
    """The feed reads only the explicit forecast dir, so a quarantined row
    under invalidated/ is structurally out of reach."""
    invalidated = estate["root"] / "invalidated" / "stage-open-current-capture-resolver-2026-09-19"
    invalidated.mkdir(parents=True)
    (invalidated / "SUN__OPEN.json").write_text(json.dumps({"snapshot_stage": "OPEN"}))

    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    assert [g["game_id"] for g in _served(estate)["games"]] == ["SUN"]
    assert "invalidated" not in estate["output"].read_text()


# ===========================================================================
# 1-2, 9. The PR #57 publication-card rule is unchanged.
# ===========================================================================
def _rollover_games() -> pd.DataFrame:
    rows = []
    for week, pairs in (
        (2, [(("HOU", "KC"), SUNDAY), (("BUF", "NYJ"), MONDAY)]),
        (3, [(("HOU", "KC"), NEXT_THURSDAY), (("BUF", "NYJ"), pd.Timestamp("2026-09-27T17:00:00Z"))]),
    ):
        for (away, home), kickoff in pairs:
            rows.append(
                {
                    "game_id": f"2026_{week:02d}_{away}_{home}",
                    "season": 2026, "week": week, "season_type": "REG",
                    "home_team_id": home, "away_team_id": away,
                    "scheduled_kickoff_utc": kickoff,
                    "home_score": None, "away_score": None, "neutral_site": False,
                }
            )
    return pd.DataFrame(rows)


def test_the_previous_week_stays_public_before_its_final_kickoff():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-21T05:50:00Z")
    )
    assert info["week"] == "2"


def test_publication_hands_over_after_the_final_kickoff():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-22T01:00:00Z")
    )
    assert info["week"] == "3"


def test_the_handover_still_happens_even_with_no_close_on_the_new_week():
    """Requirement D again, at the card layer: a new week with nothing
    closed must still become the publication card."""
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-23T01:38:00Z")
    )
    assert info["week"] == "3"


# ===========================================================================
# 8. Historical repair can never publish.
# ===========================================================================
def test_the_stage_entrypoint_has_no_publication_path():
    import ast

    tree = ast.parse((REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py").read_text(encoding="utf-8"))
    names = (
        {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    )
    for forbidden in (
        "publish_wizard_nfl_local", "publish_2026_current_week",
        "write_latest", "publish_current_week_feed",
    ):
        assert not any(forbidden in n for n in names), forbidden


def test_the_publisher_accepts_no_historical_week_override():
    """--season/--week exist only on the stage entrypoint, never here."""
    import argparse
    import contextlib
    import io

    parser_src = (REPO_ROOT / "scripts" / "publish_2026_current_week.py").read_text(encoding="utf-8")
    assert '"--season"' not in parser_src
    assert '"--week"' not in parser_src

    buffer = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
        _load(
            REPO_ROOT / "scripts" / "publish_2026_current_week.py", "_pub_help"
        ).main(["--help"])
    assert "--season" not in buffer.getvalue()


# ===========================================================================
# 10-11. Existing non-empty output and consumer compatibility.
# ===========================================================================
def test_a_non_empty_close_feed_is_byte_stable_across_republication(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    first = estate["output"].read_bytes()
    _publish(estate)
    assert estate["output"].read_bytes() == first


def test_the_empty_envelope_is_the_same_contract_a_consumer_already_parses(estate):
    """A page that reads the populated feed can read this one: identical
    top-level keys and types, only the games list is empty."""
    _publish(estate)
    empty = _served(estate)

    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    populated = _served(estate)

    assert tuple(empty.keys()) == tuple(populated.keys()) == feed.TOP_LEVEL_KEY_ORDER
    for key in ("schema_version", "season", "week", "horizon"):
        assert type(empty[key]) is type(populated[key])
    assert isinstance(empty["games"], list)


def test_publication_stays_atomic_with_no_leftover_temp_file(estate):
    _publish(estate)
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    assert not list(estate["output"].parent.glob("*.tmp"))
    assert not list(estate["output"].parent.glob(".*.tmp"))


def test_genuine_corruption_still_fails_closed(estate):
    """Fail-closed is reserved for real inconsistency, not an empty week."""
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    _write(estate, "SUN", st.STAGE_CLOSE, margin=99.0)
    with pytest.raises(feed.WizardExportError, match="already-closed game is never mutated"):
        _publish(estate, generated_at="2026-09-20T16:30:00Z")
