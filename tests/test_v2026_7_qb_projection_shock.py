"""Focused tests for the V2026.7 prospective QB-projection-shock shadow signal.

Scope is deliberately narrow (Section 13 of the brief): primary-QB rule, tie
rule, formula sign, SMOKE exclusion, TUE+FRI both required, manifest hashes
persisted, duplicate ledger row fails closed, one numerical predictor, no
realized outcome needed to create the signal, no target-game data used, warmup
blocks a fitted prediction before 64 games, training uses prior settled rows
only, and append-only ledger behavior.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_v2026_7_qb_projection_shock_shadow.py"
_spec = importlib.util.spec_from_file_location("v2026_7_qb_shock", _MOD_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# --------------------------------------------------------------------------- #
# synthetic capture builders
# --------------------------------------------------------------------------- #
def qb_rec(player_id, team, attempts, yards, *, position="QB"):
    return {
        "player": {"id": player_id},
        "team": {"abbreviation": team},
        "position": position,
        "stats": {"passing_attempts": attempts, "passing_yards": yards},
    }


def game_rec(gid, home, away, *, home_score=None, away_score=None):
    return {
        "id": gid,
        "home_team": {"abbreviation": home},
        "visitor_team": {"abbreviation": away},
        "status_state": "scheduled" if home_score is None else "final",
        "home_team_score": home_score,
        "visitor_team_score": away_score,
    }


def make_capture(
    data_root: Path,
    *,
    season=2026,
    week=1,
    horizon="TUE",
    stamp="20260908T120000Z",
    status="COMPLETE",
    manifest_horizon=None,
    qb_records=None,
    games=None,
    extra_files=None,
) -> Path:
    cdir = (
        data_root
        / "live-observation-log"
        / "balldontlie-2026"
        / f"season={season}"
        / f"week={week:02d}"
        / f"horizon={horizon}"
        / f"capture={stamp}"
    )
    cdir.mkdir(parents=True)
    manifest = {
        "status": status,
        "horizon": manifest_horizon or horizon,
        "manifest_sha256": hashlib.sha256(f"{horizon}:{stamp}".encode()).hexdigest(),
        "season": season,
        "week": week,
    }
    (cdir / "manifest.json").write_text(json.dumps(manifest))
    (cdir / "fantasy_qb_projections.p001.json").write_text(json.dumps({"data": qb_records or []}))
    (cdir / "games.p001.json").write_text(json.dumps({"data": games or []}))
    for name, payload in (extra_files or {}).items():
        (cdir / name).write_text(json.dumps(payload))
    return cdir


def _two_team_records(home, away, *, home_py, away_py, home_backup_py=10.0, away_backup_py=10.0):
    return [
        qb_rec(101, home, 30.0, home_py),
        qb_rec(102, home, 1.3, home_backup_py),
        qb_rec(201, away, 31.0, away_py),
        qb_rec(202, away, 1.4, away_backup_py),
    ]


# --------------------------------------------------------------------------- #
# primary QB rule
# --------------------------------------------------------------------------- #
def test_primary_qb_is_max_projected_passing_attempts():
    records = [qb_rec(1, "AAA", 30.5, 250.0), qb_rec(2, "AAA", 1.2, 9.0), qb_rec(9, "BBB", 40.0, 999.0)]
    res = mod.team_primary_qb(records, "AAA")
    assert res["status"] == "OK"
    assert res["player_id"] == 1
    assert res["passing_yards"] == 250.0
    assert res["passing_attempts"] == 30.5


def test_tie_for_greatest_passing_attempts_is_unavailable():
    records = [qb_rec(1, "AAA", 20.0, 210.0), qb_rec(2, "AAA", 20.0, 150.0)]
    assert mod.team_primary_qb(records, "AAA")["status"] == "TIE"


def test_missing_or_non_numeric_qb_is_unavailable():
    assert mod.team_primary_qb([], "AAA")["status"] == "MISSING"
    assert mod.team_primary_qb([qb_rec(1, "AAA", None, 200.0)], "AAA")["status"] == "MISSING"
    assert mod.team_primary_qb([qb_rec(1, "AAA", "31", "220")], "AAA")["status"] == "MISSING"
    # a non-QB record with huge attempts must never be selected
    rb = qb_rec(5, "AAA", 99.0, 500.0, position="RB")
    assert mod.team_primary_qb([rb, qb_rec(1, "AAA", 22.0, 210.0)], "AAA")["player_id"] == 1


# --------------------------------------------------------------------------- #
# the ONE signal — formula sign
# --------------------------------------------------------------------------- #
def test_team_change_and_shock_formula_sign():
    assert mod.team_qb_projection_change({"passing_yards": 200.0}, {"passing_yards": 232.0}) == 32.0
    # home improves +30, away improves +5  => positive shock (home relative gain)
    assert mod.qb_projection_shock_fri(30.0, 5.0) == 25.0
    # home worsens, away improves => negative shock
    assert mod.qb_projection_shock_fri(-12.0, 4.0) == -16.0


def test_signal_row_sign_end_to_end(tmp_path):
    d = tmp_path / "data"
    tue = make_capture(
        d, horizon="TUE", stamp="20260908T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=200.0, away_py=250.0),
        games=[game_rec(1, "PHI", "WAS")],
    )
    fri = make_capture(
        d, horizon="FRI", stamp="20260911T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=240.0, away_py=245.0),
        games=[game_rec(1, "PHI", "WAS")],
    )
    rows, omitted = mod.build_signal_rows(tue, fri, season=2026, week=1, preregistration_hash="h")
    assert omitted == []
    assert len(rows) == 1
    row = rows[0]
    # home change = 240-200 = +40 ; away change = 245-250 = -5 ; shock = 40 - (-5) = 45
    assert row["home_qb_projection_change"] == 40.0
    assert row["away_qb_projection_change"] == -5.0
    assert row[mod.SIGNAL_NAME] == 45.0


# --------------------------------------------------------------------------- #
# capture eligibility
# --------------------------------------------------------------------------- #
def test_smoke_capture_is_excluded(tmp_path):
    d = tmp_path / "data"
    make_capture(d, horizon="SMOKE", stamp="20260903T053105Z", status="COMPLETE")
    assert mod.resolve_official_capture(d, 2026, 1, "TUE") is None
    assert mod.resolve_official_capture(d, 2026, 1, "FRI") is None


def test_downgraded_tue_request_is_excluded(tmp_path):
    d = tmp_path / "data"
    # directory says TUE but the manifest recorded horizon=SMOKE (off-window downgrade)
    make_capture(d, horizon="TUE", stamp="20260908T120000Z", status="COMPLETE", manifest_horizon="SMOKE")
    assert mod.resolve_official_capture(d, 2026, 1, "TUE") is None


def test_incomplete_capture_is_excluded(tmp_path):
    d = tmp_path / "data"
    make_capture(d, horizon="TUE", stamp="20260908T120000Z", status="INCOMPLETE")
    assert mod.resolve_official_capture(d, 2026, 1, "TUE") is None


def test_multiple_complete_captures_fail_closed(tmp_path):
    d = tmp_path / "data"
    make_capture(d, horizon="TUE", stamp="20260908T120000Z", status="COMPLETE")
    make_capture(d, horizon="TUE", stamp="20260908T180000Z", status="COMPLETE")
    with pytest.raises(mod.FailClosedError):
        mod.resolve_official_capture(d, 2026, 1, "TUE")


def test_tue_and_fri_both_required(tmp_path, capsys):
    d = tmp_path / "data"
    make_capture(d, horizon="TUE", stamp="20260908T120000Z", status="COMPLETE",
                 qb_records=_two_team_records("PHI", "WAS", home_py=200.0, away_py=250.0),
                 games=[game_rec(1, "PHI", "WAS")])
    rc = mod.main(["--season", "2026", "--week", "1", "--data-root", str(d),
                   "--artifact-root", str(tmp_path / "art")])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == mod.STATUS_NOT_READY
    assert payload["tue_capture_available"] is True
    assert payload["fri_capture_available"] is False


def test_main_not_ready_when_only_smoke_exists(tmp_path, capsys):
    d = tmp_path / "data"
    make_capture(d, horizon="SMOKE", stamp="20260903T053105Z", status="COMPLETE")
    rc = mod.main(["--season", "2026", "--week", "1", "--data-root", str(d),
                   "--artifact-root", str(tmp_path / "art")])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == mod.STATUS_NOT_READY


# --------------------------------------------------------------------------- #
# provenance persisted / no outcome required / no target-game data used
# --------------------------------------------------------------------------- #
def _ready_pair(d: Path):
    tue = make_capture(
        d, horizon="TUE", stamp="20260908T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=200.0, away_py=250.0)
        + _two_team_records("KC", "DET", home_py=270.0, away_py=210.0),
        games=[game_rec(1, "PHI", "WAS"), game_rec(2, "KC", "DET")],
    )
    fri = make_capture(
        d, horizon="FRI", stamp="20260911T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=240.0, away_py=245.0)
        + _two_team_records("KC", "DET", home_py=272.0, away_py=205.0),
        games=[game_rec(1, "PHI", "WAS"), game_rec(2, "KC", "DET")],
    )
    return tue, fri


def test_manifest_hashes_persisted(tmp_path):
    d = tmp_path / "data"
    tue, fri = _ready_pair(d)
    rows, _ = mod.build_signal_rows(tue, fri, season=2026, week=1, preregistration_hash="h")
    assert rows
    tue_sha = mod.manifest_sha256(tue)
    fri_sha = mod.manifest_sha256(fri)
    assert len(tue_sha) == 64 and tue_sha != fri_sha
    for row in rows:
        assert row["tue_manifest_sha256"] == tue_sha
        assert row["fri_manifest_sha256"] == fri_sha
        assert row["tue_capture_path"] == str(tue)
        assert row["fri_capture_path"] == str(fri)


def test_signal_creation_needs_no_realized_outcome(tmp_path):
    d = tmp_path / "data"
    tue, fri = _ready_pair(d)  # all games status_state=scheduled, scores None
    rows, omitted = mod.build_signal_rows(tue, fri, season=2026, week=1, preregistration_hash="h")
    assert omitted == []
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row[mod.SIGNAL_NAME], float)
        # nothing resembling an outcome is written into the shadow row
        assert not any("score" in k or "actual" in k or "residual" in k for k in row)


def test_no_target_game_data_used(tmp_path):
    d1 = tmp_path / "clean"
    d2 = tmp_path / "noisy"
    tue1, fri1 = _ready_pair(d1)
    rows_clean, _ = mod.build_signal_rows(tue1, fri1, season=2026, week=1, preregistration_hash="h")

    # identical projections + game ids/teams, but the captures also carry final
    # scores, odds and player-props files that MUST be ignored by the signal
    tue2 = make_capture(
        d2, horizon="TUE", stamp="20260908T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=200.0, away_py=250.0)
        + _two_team_records("KC", "DET", home_py=270.0, away_py=210.0),
        games=[game_rec(1, "PHI", "WAS", home_score=31, away_score=17),
               game_rec(2, "KC", "DET", home_score=24, away_score=27)],
        extra_files={
            "odds_current.p001.json": {"data": [{"game_id": 1, "spread_home": -3.5}]},
            "player_props_game_1.json": {"data": [{"player_id": 101, "line": 268.5}]},
            "injuries.p001.json": {"data": [{"player_id": 101, "status": "Out"}]},
        },
    )
    fri2 = make_capture(
        d2, horizon="FRI", stamp="20260911T120000Z",
        qb_records=_two_team_records("PHI", "WAS", home_py=240.0, away_py=245.0)
        + _two_team_records("KC", "DET", home_py=272.0, away_py=205.0),
        games=[game_rec(1, "PHI", "WAS", home_score=31, away_score=17),
               game_rec(2, "KC", "DET", home_score=24, away_score=27)],
        extra_files={
            "odds_current.p001.json": {"data": [{"game_id": 1, "spread_home": -7.0}]},
            "player_props_game_1.json": {"data": [{"player_id": 101, "line": 300.5}]},
        },
    )
    rows_noisy, _ = mod.build_signal_rows(tue2, fri2, season=2026, week=1, preregistration_hash="h")

    keyed_clean = {r["game_id"]: r[mod.SIGNAL_NAME] for r in rows_clean}
    keyed_noisy = {r["game_id"]: r[mod.SIGNAL_NAME] for r in rows_noisy}
    assert keyed_clean == keyed_noisy


# --------------------------------------------------------------------------- #
# one numerical predictor
# --------------------------------------------------------------------------- #
def test_exactly_one_numerical_predictor():
    doc = mod.load_preregistration()
    assert doc["frozen"]["numerical_predictors"] == ["QB_PROJECTION_SHOCK_FRI"]
    assert doc["frozen"]["number_of_numerical_predictors"] == 1
    assert doc["frozen"]["hard_parsimony_rule"]["features"] == 1


def test_built_row_carries_the_single_predictor(tmp_path):
    d = tmp_path / "data"
    tue, fri = _ready_pair(d)
    rows, _ = mod.build_signal_rows(tue, fri, season=2026, week=1, preregistration_hash="h")
    row = rows[0]
    assert mod.SIGNAL_NAME in row
    forbidden = {
        "qb_identity_change", "injury_status", "depth_chart_status",
        "passing_attempts_change", "fantasy_points", "rushing_projection",
        "player_props", "opening_props",
    }
    assert forbidden.isdisjoint(row.keys())


# --------------------------------------------------------------------------- #
# append-only ledger + duplicate fail-closed
# --------------------------------------------------------------------------- #
def _ledger_row(game_id):
    return {
        "signal_version": mod.SIGNAL_VERSION,
        "signal_name": mod.SIGNAL_NAME,
        "season": 2026,
        "week": 1,
        "game_id": str(game_id),
        mod.SIGNAL_NAME: 1.0,
    }


def test_duplicate_ledger_row_fails_closed(tmp_path):
    art = tmp_path / "art"
    assert mod.append_ledger_rows(art, [_ledger_row(1)]) == 1
    with pytest.raises(mod.FailClosedError):
        mod.append_ledger_rows(art, [_ledger_row(1)])
    # duplicate within a single batch also fails closed
    art2 = tmp_path / "art2"
    with pytest.raises(mod.FailClosedError):
        mod.append_ledger_rows(art2, [_ledger_row(7), _ledger_row(7)])
    assert mod.read_ledger(art2) == []  # nothing written on a failed batch


def test_ledger_is_append_only(tmp_path):
    art = tmp_path / "art"
    mod.append_ledger_rows(art, [_ledger_row(1), _ledger_row(2)])
    before = mod.ledger_path(art).read_text().splitlines()
    mod.append_ledger_rows(art, [_ledger_row(3)])
    after = mod.ledger_path(art).read_text().splitlines()
    assert after[:2] == before[:2]           # existing lines untouched
    assert len(after) == 3
    assert [r["game_id"] for r in mod.read_ledger(art)] == ["1", "2", "3"]


# --------------------------------------------------------------------------- #
# warmup / online model training discipline
# --------------------------------------------------------------------------- #
def _settled_rows(n, *, start=0, residual=True):
    out = []
    for i in range(start, start + n):
        out.append(
            {
                "game_id": str(i),
                mod.SIGNAL_NAME: float((i % 7) - 3),
                "margin_residual": float((i % 5) - 2) if residual else None,
            }
        )
    return out


def test_warmup_blocks_fitted_prediction_before_64():
    assert mod.online_forecast(_settled_rows(63))["status"] == "WARMUP"
    ready = mod.online_forecast(_settled_rows(64))
    assert ready["status"] == "READY"
    assert ready["n_settled"] == 64
    assert "model" in ready


def test_training_uses_prior_settled_rows_only():
    # 63 settled + 10 not-yet-settled rows => still WARMUP (unsettled don't count)
    mixed = _settled_rows(63) + _settled_rows(10, start=1000, residual=False)
    assert mod.online_forecast(mixed)["status"] == "WARMUP"
    # 64 unique settled + unsettled noise => READY, trained on the 64 settled only
    ready = mod.online_forecast(_settled_rows(64) + _settled_rows(5, start=2000, residual=False))
    assert ready["status"] == "READY"
    assert ready["n_settled"] == 64
    # duplicate game_ids do not inflate the unique-settled count past the gate
    dupes = _settled_rows(40) + _settled_rows(40)
    assert mod.online_forecast(dupes)["status"] == "WARMUP"


# --------------------------------------------------------------------------- #
# market residual definition + preregistration hash
# --------------------------------------------------------------------------- #
def test_market_residual_definition():
    assert mod.market_implied_home_margin(-3.5) == 3.5
    # actual home margin +10, home favored by 3.5 (spread -3.5) => residual +6.5
    assert mod.margin_residual(10.0, -3.5) == 6.5
    # MARKET_ONLY predicts residual == 0
    assert mod.margin_residual(3.5, -3.5) == 0.0


def test_preregistration_hash_is_deterministic_and_matches_file():
    path = mod.preregistration_path()
    doc = json.loads(path.read_text())
    recomputed = mod.compute_preregistration_hash(doc["frozen"])
    assert recomputed == doc["preregistration_sha256"]
    assert recomputed == mod.compute_preregistration_hash(doc["frozen"])  # stable
    assert len(recomputed) == 64 and all(c in "0123456789abcdef" for c in recomputed)


def test_no_production_model_or_external_api_touch():
    src = _MOD_PATH.read_text()
    assert "import nfl_hybrid" not in src
    assert "run_2026" not in src
    for banned in ("requests.get", "urllib.request", "http://", "https://api"):
        assert banned not in src
