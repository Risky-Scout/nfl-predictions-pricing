"""Proofs for the 2026 NFL production automation: the composite games
population, the shared preflight/execution population, chronological
retraining eligibility, fail-closed recalibration promotion, the frozen
published contract, and the one authoritative GitHub workflow.

Everything here is HERMETIC and SYNTHETIC. No network call, no real capture,
no real artifact root, no real forecast, no deployment and no SSH. The
synthetic BallDontLie captures are built by the EXISTING helper in
``tests/test_bdl_market_bridge.py`` (``write_capture``), so the captures these
tests validate are hashed exactly the way the real capture script hashes them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data import games_population_2026 as gp26
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.production import recalibration_2026 as rc
from nfl_hybrid.production import run_2026 as prod
from nfl_hybrid.providers.balldontlie import canonical as bdl_canonical
from nfl_hybrid.providers.balldontlie.team_crosswalk import _BDL_TEAMS

from tests.test_bdl_market_bridge import CUTOFF, _game_row, write_capture

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"

# The 32 real BDL franchise ids, paired into the 16 games of a real NFL week.
BDL_TEAM_IDS = sorted(team[0] for team in _BDL_TEAMS)
WEEK1_PAIRS = [(BDL_TEAM_IDS[i], BDL_TEAM_IDS[i + 1]) for i in range(0, 32, 2)]


# ---------------------------------------------------------------------------
# Fixtures: a Week-1-shaped capture and a synthetic certified historical estate
# ---------------------------------------------------------------------------
def week1_games(*, season: int = 2026, week: int = 1, scored: bool = False) -> list[dict]:
    """A full 16-game Week-1-shaped slate over the real 32-franchise
    crosswalk -- the shape of the official Week-1 TUE capture."""
    rows = []
    for index, (home, away) in enumerate(WEEK1_PAIRS):
        row = _game_row(1000 + index, home, away, season=season, week=week)
        if scored:
            row["home_team_score"] = 20 + index
            # BDL's own field name for the away score. Spelling this
            # "away_team_score" silently left the away score absent.
            row["visitor_team_score"] = 17
            row["status"] = "Final"
            row["status_state"] = "final"
        rows.append(row)
    return rows


def _historical_frame() -> pd.DataFrame:
    """A synthetic certified historical estate: seasons 2020-2025, REG only,
    on the population contract. Stands in for ``backfill.games``."""
    rows = []
    for season in range(2020, 2026):
        for week in (1, 2):
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_HOU_KC",
                    "season": season,
                    "week": str(week),
                    "season_type": "REG",
                    "home_team_id": "KC",
                    "away_team_id": "HOU",
                    "scheduled_kickoff_utc": pd.Timestamp(f"{season}-09-{10 + week:02d}T17:00:00Z"),
                    "home_score": 24.0,
                    "away_score": 20.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def historical_estate(monkeypatch, tmp_path) -> pd.DataFrame:
    frame = _historical_frame()
    path = tmp_path / "backfill_games.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(
        gp26, "resolve", lambda key: path if key == "backfill.games" else pytest.fail(f"unexpected key {key}")
    )
    return frame


def ingest(capture_manifest: Path, artifact_root: Path) -> gp26.PopulationUpdate:
    rows, evidence = gp26.games_from_capture(capture_manifest)
    return gp26.update_durable_2026_population(rows, evidence=[evidence], artifact_root_path=artifact_root)


# ===========================================================================
# 1. The EXISTING BallDontLie canonicalizer is reused -- no second normalizer
# ===========================================================================
def test_population_builder_reuses_the_existing_bdl_canonicalizer(monkeypatch, tmp_path):
    calls: list[dict] = []
    original = bdl_canonical.normalize_games

    def spy(raw_games, **kwargs):
        calls.append({"rows": len(raw_games), **kwargs})
        return original(raw_games, **kwargs)

    monkeypatch.setattr(gp26.bdl_canonical, "normalize_games", spy)
    manifest = write_capture(tmp_path / "cap", games=week1_games())
    gp26.games_from_capture(manifest)

    assert len(calls) == 1, "the population builder must canonicalize through exactly one call"
    assert calls[0]["rows"] == 16
    # The capture's OWN recorded season_type is the hint; never a default.
    assert calls[0]["season_type_hint"] == "REG"


def test_population_module_declares_no_team_or_game_id_mapping():
    """A structural guard against a second normalizer creeping in: the module
    must not construct a canonical game_id or map a BDL team id itself."""
    source = (REPO_ROOT / "src/nfl_hybrid/data/games_population_2026.py").read_text()
    for forbidden in ("canonical_team_for_bdl_id", "canonical_team_from_team_object", "_canonical_game_id"):
        assert forbidden not in source


# ===========================================================================
# 2/3. A Week-1-shaped capture yields 16 canonical games; PRE is excluded
# ===========================================================================
def test_official_week1_shaped_capture_produces_sixteen_canonical_games(tmp_path):
    manifest = write_capture(tmp_path / "cap", games=week1_games())
    rows, evidence = gp26.games_from_capture(manifest)

    assert len(rows) == 16
    assert rows["game_id"].nunique() == 16
    assert set(rows["season"]) == {2026}
    assert set(rows["week"]) == {"1"}
    assert set(rows["season_type"]) == {"REG"}
    # 32 distinct franchises, each appearing exactly once.
    assert len(set(rows["home_team_id"]) | set(rows["away_team_id"])) == 32
    # Upcoming games legitimately carry null scores.
    assert rows["home_score"].isna().all()
    assert evidence["canonical_row_count"] == 16
    assert evidence["manifest_sha256"]


def test_a_preseason_capture_is_rejected_outright(tmp_path):
    """PRESEASON is excluded twice over. First: the existing capture validator
    refuses a PRE-labelled capture before a single row is read, so preseason
    evidence can never reach the population at all."""
    manifest = write_capture(tmp_path / "cap", games=week1_games(), season_type="PRE")
    with pytest.raises(gp26.GamesPopulationError, match="season_type 'PRE' is not valid for production"):
        gp26.games_from_capture(manifest)


def test_preseason_rows_are_dropped_by_the_population_projection(tmp_path):
    """And second: even a canonical frame that somehow carries PRE rows loses
    them on projection, so a preseason Week 1 can never collide with a
    regular-season Week 1 canonical game_id."""
    reg = bdl_canonical.normalize_games(week1_games(), season_type_hint="REG")
    pre = bdl_canonical.normalize_games(week1_games(), season_type_hint="PRE")

    assert len(gp26.canonical_games_to_population_rows(reg)) == 16
    assert gp26.canonical_games_to_population_rows(pre).empty

    mixed = gp26.canonical_games_to_population_rows(pd.concat([reg, pre], ignore_index=True))
    assert len(mixed) == 16
    assert set(mixed["season_type"]) == {"REG"}


# ===========================================================================
# 4/5. History preserved; 2026 appended deterministically
# ===========================================================================
def test_historical_2020_2025_games_are_preserved_and_2026_is_appended(tmp_path, historical_estate):
    aroot = tmp_path / "artifacts"
    before = gp26.load_composite_games_population(artifact_root_path=aroot)
    assert before.provenance["historical_max_season"] == 2025
    assert before.schedule_2026_available is False
    assert before.provenance["bdl_2026_row_count"] == 0

    ingest(write_capture(tmp_path / "cap", games=week1_games()), aroot)
    after = gp26.load_composite_games_population(artifact_root_path=aroot)

    # Every historical row survives byte-for-byte.
    historical_ids = set(historical_estate["game_id"])
    assert historical_ids.issubset(set(after.games["game_id"]))
    assert len(after.games) == len(before.games) + 16
    assert after.provenance["historical_row_count"] == len(historical_estate)
    assert after.provenance["max_season"] == 2026
    assert after.schedule_2026_available is True


def test_2026_evidence_merges_deterministically_and_idempotently(tmp_path, historical_estate):
    aroot_a = tmp_path / "a"
    aroot_b = tmp_path / "b"
    first = write_capture(tmp_path / "cap1", games=week1_games())
    second = write_capture(tmp_path / "cap2", games=week1_games(week=2))

    for manifest in (first, second):
        ingest(manifest, aroot_a)
    for manifest in (second, first):  # reversed order
        ingest(manifest, aroot_b)

    hash_a = gp26.population_content_hash(gp26.read_durable_2026_population(aroot_a)[0])
    hash_b = gp26.population_content_hash(gp26.read_durable_2026_population(aroot_b)[0])
    assert hash_a == hash_b, "merge order must not change the population"

    # Re-applying identical evidence rewrites nothing.
    assert ingest(first, aroot_a).status == "IDEMPOTENT_NOOP"


def test_a_null_score_transitions_to_a_final_score_exactly_once(tmp_path):
    aroot = tmp_path / "artifacts"
    scheduled = write_capture(tmp_path / "sched", games=week1_games())
    completed = write_capture(tmp_path / "final", games=week1_games(scored=True))

    ingest(scheduled, aroot)
    assert gp26.read_durable_2026_population(aroot)[0]["home_score"].isna().all()

    ingest(completed, aroot)
    population = gp26.read_durable_2026_population(aroot)[0]
    assert population["home_score"].notna().all()

    # A later, less complete read never un-records a result.
    ingest(write_capture(tmp_path / "sched2", games=week1_games()), aroot)
    assert gp26.read_durable_2026_population(aroot)[0]["home_score"].notna().all()


# ===========================================================================
# 6. Conflicts fail closed
# ===========================================================================
def test_conflicting_canonical_identity_fails_closed(tmp_path):
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap1", games=week1_games()), aroot)

    shifted = week1_games()
    shifted[0]["date"] = "2026-09-14T17:00:00Z"  # same game_id, different kickoff
    conflicting = write_capture(tmp_path / "cap2", games=shifted)

    with pytest.raises(gp26.GamesPopulationConflict, match="conflicting canonical identity"):
        ingest(conflicting, aroot)

    # Nothing was merged: the stored kickoff is untouched.
    stored = gp26.read_durable_2026_population(aroot)[0]
    assert stored["scheduled_kickoff_utc"].dt.hour.eq(17).all()
    assert stored["scheduled_kickoff_utc"].dt.day.eq(13).all()


def test_a_changed_recorded_score_fails_closed(tmp_path):
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap1", games=week1_games(scored=True)), aroot)

    altered = week1_games(scored=True)
    altered[0]["home_team_score"] = 99
    with pytest.raises(gp26.GamesPopulationConflict, match="recorded result"):
        ingest(write_capture(tmp_path / "cap2", games=altered), aroot)


# ===========================================================================
# 6b. An IN-PROGRESS score is never a result
#
# Production regression (Sep 13 2026, game 2026_01_BAL_IND). The 19:55Z daily
# capture caught the game in the fourth quarter at BAL 38 IND 23 and stored
# that scoreline as the result. Because a recorded score is immutable, the
# real final -- BAL 41 IND 23, after Baltimore's last field goal -- was then
# rejected and the whole daily pass failed closed:
#
#   FAIL CLOSED: recorded result for game_id '2026_01_BAL_IND' changed:
#   stored=(23.0, 38.0) incoming=(23.0, 41.0)
#
# Finality comes from the provider's own status_state, never from the mere
# presence of a score or from elapsed kickoff time.
# ===========================================================================
def _bal_at_ind(*, home_score, away_score, status, status_state) -> list[dict]:
    """The real fixture, in BDL's own raw shape: BAL visiting IND. IND is the
    home team, so the population pair is (home=IND, away=BAL)."""
    bal = next(team[0] for team in _BDL_TEAMS if team[1] == "BAL")
    ind = next(team[0] for team in _BDL_TEAMS if team[1] == "IND")
    row = _game_row(7001, ind, bal)
    row["home_team_score"] = home_score
    row["visitor_team_score"] = away_score
    row["status"] = status
    row["status_state"] = status_state
    return [row]


def _scores_of(population: pd.DataFrame, game_id: str) -> tuple[object, object]:
    row = population.loc[population["game_id"] == game_id].iloc[0]
    return row["home_score"], row["away_score"]


def test_an_in_progress_game_with_a_score_is_not_recorded_as_a_result(tmp_path):
    aroot = tmp_path / "artifacts"
    ingest(
        write_capture(
            tmp_path / "q4",
            games=_bal_at_ind(home_score=23, away_score=38, status="Q4 02:11", status_state="in_progress"),
        ),
        aroot,
    )

    population = gp26.read_durable_2026_population(aroot)[0]
    home, away = _scores_of(population, "2026_01_BAL_IND")
    assert pd.isna(home) and pd.isna(away), "a running scoreline is not an outcome"


def test_an_in_progress_score_may_legitimately_advance_in_a_later_capture(tmp_path):
    """The exact production sequence, end to end: the fourth-quarter capture
    must not freeze 38-23, and the real final 41-23 must land cleanly."""
    aroot = tmp_path / "artifacts"
    ingest(
        write_capture(
            tmp_path / "q4",
            games=_bal_at_ind(home_score=23, away_score=38, status="Q4 02:11", status_state="in_progress"),
        ),
        aroot,
    )
    # Baltimore kicks the field goal; a later capture sees a different score.
    ingest(
        write_capture(
            tmp_path / "q4b",
            games=_bal_at_ind(home_score=23, away_score=41, status="Q4 00:32", status_state="in_progress"),
        ),
        aroot,
    )
    ingest(
        write_capture(
            tmp_path / "final",
            games=_bal_at_ind(home_score=23, away_score=41, status="Final", status_state="final"),
        ),
        aroot,
    )

    home, away = _scores_of(gp26.read_durable_2026_population(aroot)[0], "2026_01_BAL_IND")
    assert (home, away) == (23.0, 41.0)


def test_a_final_provider_state_makes_the_result_immutable(tmp_path):
    """Once final, the result is durable: a later, less complete read can
    never un-record it."""
    aroot = tmp_path / "artifacts"
    ingest(
        write_capture(
            tmp_path / "final",
            games=_bal_at_ind(home_score=23, away_score=41, status="Final", status_state="final"),
        ),
        aroot,
    )
    ingest(
        write_capture(
            tmp_path / "blank",
            games=_bal_at_ind(home_score=None, away_score=None, status="Scheduled", status_state="scheduled"),
        ),
        aroot,
    )

    home, away = _scores_of(gp26.read_durable_2026_population(aroot)[0], "2026_01_BAL_IND")
    assert (home, away) == (23.0, 41.0)


def test_a_conflicting_score_for_a_truly_final_game_still_fails_closed(tmp_path):
    """The protection this defect was hiding behind must survive the fix: a
    genuinely finalized result that later changes is still a hard stop."""
    aroot = tmp_path / "artifacts"
    ingest(
        write_capture(
            tmp_path / "final",
            games=_bal_at_ind(home_score=23, away_score=41, status="Final", status_state="final"),
        ),
        aroot,
    )

    with pytest.raises(gp26.GamesPopulationConflict, match="recorded result"):
        ingest(
            write_capture(
                tmp_path / "final2",
                games=_bal_at_ind(home_score=23, away_score=44, status="Final", status_state="final"),
            ),
            aroot,
        )
    home, away = _scores_of(gp26.read_durable_2026_population(aroot)[0], "2026_01_BAL_IND")
    assert (home, away) == (23.0, 41.0), "the stored result must be untouched by a rejected merge"


@pytest.mark.parametrize(
    "status_state",
    ["scheduled", "in_progress", "postponed", "delayed", "suspended", "canceled", "abandoned", "unknown", None],
)
def test_only_a_final_provider_state_can_record_a_score(status_state):
    """Every non-final lifecycle value, and a missing one, withholds the
    score. Fail closed, never inferred from score presence."""
    normalized = bdl_canonical.normalize_games(
        _bal_at_ind(home_score=23, away_score=38, status="whatever", status_state=status_state),
        season_type_hint="REG",
    )
    rows = gp26.canonical_games_to_population_rows(normalized)
    assert pd.isna(rows.loc[0, "home_score"]) and pd.isna(rows.loc[0, "away_score"])


def test_a_final_game_missing_one_score_is_not_a_result():
    """can_update_score_state requires BOTH scores; half a scoreline is not
    an outcome and must not become an immutable result."""
    normalized = bdl_canonical.normalize_games(
        _bal_at_ind(home_score=23, away_score=None, status="Final", status_state="final"),
        season_type_hint="REG",
    )
    rows = gp26.canonical_games_to_population_rows(normalized)
    assert pd.isna(rows.loc[0, "home_score"]) and pd.isna(rows.loc[0, "away_score"])


def test_scheduled_and_future_games_are_unaffected(tmp_path):
    """The fix must change nothing for games that have not started."""
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap", games=week1_games()), aroot)

    population = gp26.read_durable_2026_population(aroot)[0]
    assert len(population) == 16
    assert population["home_score"].isna().all()
    assert population["away_score"].isna().all()


def test_the_population_reuses_the_one_finality_rule_rather_than_restating_it():
    """A structural guard, in the spirit of the module's existing
    no-second-normalizer guard: finality must come from finality.py, so there
    can never be two answers to "is this game over?"."""
    import ast

    source = (REPO_ROOT / "src/nfl_hybrid/data/games_population_2026.py").read_text()
    assert "can_update_score_state" in source
    # Check the CODE, not the prose: the module docstring legitimately
    # describes the rule it defers to.
    docstring = ast.get_docstring(ast.parse(source))
    code = source.replace(docstring, "", 1) if docstring else source
    for forbidden in ('status_state == "final"', "status_state == 'final'", "FINAL_STATUS_STATE ="):
        assert forbidden not in code, "finality must not be re-implemented here"


# ===========================================================================
# 6c. Auditing and repairing a population that was already poisoned
# ===========================================================================
def _load_audit_tool():
    import importlib.util

    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "_population_finality_audit", REPO_ROOT / "scripts/audit_2026_games_population_finality.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _poisoned_estate(tmp_path) -> tuple[Path, Path]:
    """Recreate the real production state: a population that already holds the
    in-progress 38-23 as a result, and evidence carrying the true final 41-23.

    The bad row is written directly, because the fixed code can no longer
    produce it -- which is the point.
    """
    aroot = tmp_path / "artifacts"
    droot = tmp_path / "data"

    frozen = gp26.canonical_games_to_population_rows(
        bdl_canonical.normalize_games(
            _bal_at_ind(home_score=23, away_score=41, status="Final", status_state="final"),
            season_type_hint="REG",
        )
    )
    frozen.loc[0, "away_score"] = 38.0  # the intermediate scoreline, as stored
    gp26.update_durable_2026_population(frozen, evidence=[], artifact_root_path=aroot)

    season_type_dir = droot / "live-observation-log" / gp26_evidence_namespace() / "season=2026" / "season_type=REG"
    write_capture(
        season_type_dir,
        games=_bal_at_ind(home_score=23, away_score=41, status="Final", status_state="final"),
        horizon="GAMES_EVIDENCE",
    )
    return aroot, droot


def gp26_evidence_namespace() -> str:
    return "balldontlie-2026-games-evidence"


def test_the_audit_names_every_prematurely_frozen_row(tmp_path):
    aroot, droot = _poisoned_estate(tmp_path)
    tool = _load_audit_tool()

    population = gp26.read_durable_2026_population(aroot)[0]
    truth = tool._evidence_truth(
        tool.discover_latest_games_evidence(season=2026, data_root=droot), season=2026
    )
    findings = tool.audit(population, truth)

    by_id = {f["game_id"]: f for f in findings}
    assert by_id["2026_01_BAL_IND"]["verdict"] == tool.PREMATURELY_FROZEN_STALE
    assert by_id["2026_01_BAL_IND"]["stored_away_score"] == 38.0
    assert by_id["2026_01_BAL_IND"]["evidence_away_score"] == 41.0


def test_the_audit_reports_every_row_rather_than_stopping_at_the_first(tmp_path):
    """The merge fails closed on the first conflict; the audit must not, or it
    could never establish the blast radius."""
    aroot = tmp_path / "artifacts"
    droot = tmp_path / "data"

    final_games = week1_games(scored=True)
    good = gp26.canonical_games_to_population_rows(
        bdl_canonical.normalize_games(final_games, season_type_hint="REG")
    )
    # Three rows frozen at an earlier, lower scoreline.
    poisoned = good.copy()
    for position in range(3):
        poisoned.loc[position, "away_score"] = 3.0
    gp26.update_durable_2026_population(poisoned, evidence=[], artifact_root_path=aroot)

    write_capture(
        droot / "live-observation-log" / gp26_evidence_namespace() / "season=2026" / "season_type=REG",
        games=final_games,
        horizon="GAMES_EVIDENCE",
    )

    tool = _load_audit_tool()
    findings = tool.audit(
        gp26.read_durable_2026_population(aroot)[0],
        tool._evidence_truth(tool.discover_latest_games_evidence(season=2026, data_root=droot), season=2026),
    )

    stale = [f for f in findings if f["verdict"] == tool.PREMATURELY_FROZEN_STALE]
    assert len(findings) == 16, "every row must be classified"
    assert len(stale) == 3, "all three defects must be reported, not just the first"
    assert len([f for f in findings if f["verdict"] == tool.OK_FINAL]) == 13


def test_the_audit_is_read_only(tmp_path):
    aroot, droot = _poisoned_estate(tmp_path)
    tool = _load_audit_tool()

    before = {p: p.read_bytes() for p in sorted(aroot.rglob("*")) if p.is_file()}
    exit_code = tool.main(["--data-root", str(droot), "--artifact-root", str(aroot)])
    after = {p: p.read_bytes() for p in sorted(aroot.rglob("*")) if p.is_file()}

    assert exit_code == 3, "a defective population must not report success"
    assert before == after, "the audit must not modify anything"


def test_repair_rebuilds_the_population_and_records_the_rejected_value(tmp_path):
    aroot, droot = _poisoned_estate(tmp_path)
    tool = _load_audit_tool()

    assert tool.main(["--repair", "--confirm", "--data-root", str(droot), "--artifact-root", str(aroot)]) == 0

    repaired = gp26.read_durable_2026_population(aroot)[0]
    home, away = _scores_of(repaired, "2026_01_BAL_IND")
    assert (home, away) == (23.0, 41.0)

    provenance = json.loads((gp26.population_dir(aroot) / "canonical_games_2026.repair.json").read_text())
    corrected = {c["game_id"]: c for c in provenance["corrected_cells"]}["2026_01_BAL_IND"]
    assert corrected["rejected_away_score"] == 38.0
    assert corrected["corrected_away_score"] == 41.0
    assert Path(provenance["superseded_population_parquet"]).is_file()

    # And the repaired population is clean on a re-audit.
    assert tool.main(["--data-root", str(droot), "--artifact-root", str(aroot)]) == 0


def test_repair_refuses_without_explicit_confirmation(tmp_path):
    aroot, droot = _poisoned_estate(tmp_path)
    tool = _load_audit_tool()

    before = {p: p.read_bytes() for p in sorted(aroot.rglob("*")) if p.is_file()}
    assert tool.main(["--repair", "--data-root", str(droot), "--artifact-root", str(aroot)]) == 2
    assert {p: p.read_bytes() for p in sorted(aroot.rglob("*")) if p.is_file()} == before


def test_a_clean_population_audits_clean(tmp_path):
    aroot = tmp_path / "artifacts"
    droot = tmp_path / "data"
    final_games = week1_games(scored=True)
    gp26.update_durable_2026_population(
        gp26.canonical_games_to_population_rows(
            bdl_canonical.normalize_games(final_games, season_type_hint="REG")
        ),
        evidence=[],
        artifact_root_path=aroot,
    )
    write_capture(
        droot / "live-observation-log" / gp26_evidence_namespace() / "season=2026" / "season_type=REG",
        games=final_games,
        horizon="GAMES_EVIDENCE",
    )

    tool = _load_audit_tool()
    assert tool.main(["--data-root", str(droot), "--artifact-root", str(aroot)]) == 0


def test_2026_evidence_claiming_a_historical_game_id_fails_closed(tmp_path, monkeypatch):
    aroot = tmp_path / "artifacts"
    # A historical estate that already owns the exact 2026 canonical ids.
    canonical = gp26.canonical_games_to_population_rows(
        bdl_canonical.normalize_games(week1_games(), season_type_hint="REG")
    )
    colliding = canonical.copy()
    colliding["season"] = 2025
    path = tmp_path / "colliding.parquet"
    colliding.to_parquet(path, index=False)
    monkeypatch.setattr(gp26, "resolve", lambda key: path)

    ingest(write_capture(tmp_path / "cap", games=week1_games()), aroot)
    with pytest.raises(gp26.GamesPopulationConflict, match="already present in the certified historical estate"):
        gp26.load_composite_games_population(artifact_root_path=aroot)


def test_historical_estate_containing_2026_fails_closed(tmp_path, monkeypatch):
    frame = _historical_frame()
    frame.loc[0, "season"] = 2026
    path = tmp_path / "bad.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(gp26, "resolve", lambda key: path)

    with pytest.raises(gp26.GamesPopulationConflict, match="unexpectedly contains season"):
        gp26.load_composite_games_population(artifact_root_path=tmp_path / "artifacts")


# ===========================================================================
# 7/8/9/10. Preflight and execution share ONE population
# ===========================================================================
def test_preflight_and_execution_read_the_same_games_population(tmp_path, historical_estate):
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap", games=week1_games()), aroot)

    preflight = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    schedule = preflight["checks"]["schedule_source"]["games_population"]

    direct, provenance = prod.load_games_population_with_provenance(games_population_root=aroot)
    assert schedule["reg_post_content_sha256"] == provenance["reg_post_content_sha256"]
    assert schedule["content_sha256"] == provenance["content_sha256"]
    assert schedule["reg_post_row_count"] == len(direct)


def test_schedule_2026_becomes_available_with_a_valid_bdl_capture(tmp_path, historical_estate):
    aroot = tmp_path / "artifacts"

    without = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    assert without["schedule_2026_available"] is False
    assert without["checks"]["schedule_2026_status"] == "SCHEDULE_UNAVAILABLE"
    assert "schedule_2026_unavailable" in without["blocking_problems"]

    ingest(write_capture(tmp_path / "cap", games=week1_games()), aroot)

    with_capture = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    assert with_capture["schedule_2026_available"] is True
    assert with_capture["checks"]["schedule_2026_status"] == "AVAILABLE"
    assert "schedule_2026_unavailable" not in with_capture["blocking_problems"]
    assert with_capture["checks"]["schedule_source"]["max_season_available"] == 2026


def test_live_market_registration_is_unaffected_by_the_population_change(tmp_path, historical_estate):
    """The market path keeps its own gate: a supplied capture registers, an
    omitted one does not, exactly as before."""
    aroot = tmp_path / "artifacts"
    aroot.mkdir(parents=True, exist_ok=True)
    manifest = write_capture(tmp_path / "cap")

    registered = prod.run_preflight(
        artifact_root_path=aroot,
        games_population_root=aroot,
        market_capture_manifest=manifest,
        expected_season=2026,
        expected_week=1,
        expected_horizon="TUE",
        expected_target_cutoff_utc=CUTOFF,
    )
    assert registered["live_2026_market_source_registered"] is True

    omitted = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    assert omitted["live_2026_market_source_registered"] is False


def test_production_run_ready_is_true_with_a_valid_fixture(tmp_path, monkeypatch, historical_estate):
    """READY requires infra AND both live inputs. Infra readiness is asserted
    through the real summarizer with a synthetic infra-OK check set, so this
    proves the SCHEDULE half without re-deriving the certified hash gate."""
    aroot = tmp_path / "artifacts"
    manifest = write_capture(tmp_path / "cap", games=week1_games())
    ingest(manifest, aroot)

    preflight = prod.run_preflight(
        artifact_root_path=aroot,
        games_population_root=aroot,
        market_capture_manifest=write_capture(tmp_path / "market"),
        expected_season=2026,
        expected_week=1,
        expected_horizon="TUE",
        expected_target_cutoff_utc=CUTOFF,
    )
    assert preflight["schedule_2026_available"] is True
    assert preflight["live_2026_market_source_registered"] is True

    summary = prod.summarize_preflight_readiness(
        infra_blocking=[],
        schedule_2026_available=preflight["schedule_2026_available"],
        live_2026_market_source_registered=preflight["live_2026_market_source_registered"],
    )
    assert summary["production_run_ready"] is True
    assert summary["overall_status"] == "READY"


# ===========================================================================
# 11/12. Chronological retraining eligibility; no backward leakage
# ===========================================================================
def test_completed_2026_games_enter_later_training_eligibility(tmp_path, historical_estate):
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap", games=week1_games(scored=True)), aroot)

    games = prod.load_games_population(games_population_root=aroot)
    available = he.compute_result_available_at_utc(games)
    completed_2026 = games["season"].eq(2026) & games["home_score"].notna()
    assert completed_2026.sum() == 16

    # The EXISTING training mask -- result_available_at_utc < target_cutoff_utc,
    # STRICT -- is what admits them. A cutoff after availability includes them.
    later_cutoff = available[completed_2026].max() + pd.Timedelta(days=1)
    assert (available[completed_2026] < later_cutoff).all()

    # ...and the certified Elo replay enqueues an update event for each,
    # because both scores are present: each completed 2026 game moves the
    # ratings the next eligible fit is built on.
    state = he.build_horizon_elo_state(games, "TUE")
    assert not state.empty


def test_future_results_cannot_leak_backward(tmp_path, historical_estate):
    aroot = tmp_path / "artifacts"
    ingest(write_capture(tmp_path / "cap", games=week1_games(scored=True)), aroot)

    games = prod.load_games_population(games_population_root=aroot)
    available = he.compute_result_available_at_utc(games)
    completed_2026 = games["season"].eq(2026) & games["home_score"].notna()

    # A cutoff BEFORE the results were available admits none of them.
    earlier_cutoff = available[completed_2026].min() - pd.Timedelta(days=1)
    assert not (available[completed_2026] < earlier_cutoff).any()


def test_a_stale_unresolved_game_never_enters_the_population(tmp_path, historical_estate):
    """A game whose result was already chronologically available as of the
    evidence instant, but whose scores are still unknown, is excluded and
    counted -- it can never become a training observation with no outcome."""
    aroot = tmp_path / "artifacts"
    past = week1_games()
    for row in past:
        row["date"] = "2026-01-04T17:00:00Z"  # long before the capture instant
    ingest(write_capture(tmp_path / "cap", games=past), aroot)

    _, manifest = gp26.read_durable_2026_population(aroot)
    assert len(manifest["stale_unresolved_game_ids"]) == 16

    composite = gp26.load_composite_games_population(artifact_root_path=aroot)
    assert composite.provenance["bdl_2026_row_count"] == 0
    assert len(composite.provenance["bdl_2026_stale_unresolved_excluded"]) == 16


# ===========================================================================
# 13. Frozen science is unchanged
# ===========================================================================
def test_frozen_features_and_ridge_alpha_are_unchanged():
    from nfl_hybrid.evaluation import official_horizon_oof as ohf

    assert ohf.ELO_FEATURE_COLUMNS == (
        "home_elo_pregame_rating",
        "home_elo_pregame_win_probability",
        "home_elo_pregame_expected_margin",
        "away_elo_pregame_rating",
        "away_elo_pregame_win_probability",
        "away_elo_pregame_expected_margin",
    )
    assert len(ohf.ELO_FEATURE_COLUMNS) == 6
    assert ohf.RIDGE_HYPERPARAMETERS["alpha"] == 100.0
    assert ohf.MODEL_NAME == "RIDGE_ALPHA_100[margin,total]"
    assert he.HORIZONS == ("TUE", "FRI")
    assert prod.REG_POST_SEASON_TYPES == ("REG", "POST")


def test_the_population_contract_is_exactly_the_certified_requirement():
    """The population's columns are derived FROM the certified Elo
    requirement, so a change there cannot silently diverge here."""
    assert he.REQUIRED_GAME_COLUMNS <= set(gp26.POPULATION_COLUMNS)
    assert gp26.PRODUCTION_SEASON_TYPES == prod.REG_POST_SEASON_TYPES


def test_the_games_evidence_horizon_can_never_be_a_market_capture(tmp_path):
    """A daily games-evidence capture must be rejected by the certified
    market path, which keeps the validator's TUE/FRI default."""
    from nfl_hybrid.data import bdl_market_bridge as bridge

    manifest = write_capture(
        tmp_path / "cap", games=week1_games(), horizon=gp26.GAMES_EVIDENCE_HORIZON,
        requested_horizon=gp26.GAMES_EVIDENCE_HORIZON,
    )
    with pytest.raises(bridge.BdlMarketBridgeError, match="not a production horizon"):
        bridge.validate_capture_manifest(manifest, expected_season=2026)

    # ...while the games-population path accepts it.
    rows, _ = gp26.games_from_capture(manifest)
    assert len(rows) == 16


# ===========================================================================
# 14. The certified calibrator cannot be overwritten outside promotion rules
# ===========================================================================
def test_promotion_is_not_authorized_by_the_frozen_preregistration():
    authorization = rc.promotion_authorization(REPO_ROOT)
    assert authorization["authorized"] is False
    assert authorization["status"] == rc.NOT_AUTHORIZED
    assert "no_scientific_refit" in authorization["reason"] or rc.AUTHORIZATION_KEY in authorization["reason"]


def test_promoting_a_candidate_without_authorization_fails_closed(tmp_path):
    aroot = tmp_path / "artifacts"
    certified = rc.certified_seed_path(aroot)
    certified.parent.mkdir(parents=True, exist_ok=True)
    certified.write_text(json.dumps({"schema_version": "certified", "streams": {}}))
    original = certified.read_bytes()

    with pytest.raises(rc.PromotionRefused):
        rc.promote_candidate(aroot, candidate_identifier="anything", repo_root=REPO_ROOT)
    assert certified.read_bytes() == original, "the certified calibrator must be byte-identical after a refusal"


def test_an_operator_cannot_manufacture_a_promotion_authorization(monkeypatch):
    """Authorization comes only from the frozen preregistration document --
    never from an environment variable, a flag or a filesystem marker."""
    monkeypatch.setenv("CERTIFIED_CALIBRATOR_PROMOTION_AUTHORIZATION", "1")
    monkeypatch.setenv("ALLOW_PROMOTION", "true")
    assert rc.promotion_authorization(REPO_ROOT)["authorized"] is False


def test_a_missing_preregistration_fails_closed(tmp_path):
    assert rc.promotion_authorization(tmp_path)["status"] == rc.AUTHORIZATION_SOURCE_MISSING
    assert rc.promotion_authorization(tmp_path)["authorized"] is False


# ===========================================================================
# 15. The published wizard-nfl-pricing-v2 contract is unchanged
# ===========================================================================
def test_wizard_nfl_pricing_v2_contract_is_unchanged():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    publisher = load("_publisher_contract_probe", REPO_ROOT / "scripts/publish_sportsodds_nfl.py")
    exporter = load("_exporter_contract_probe", REPO_ROOT / "scripts/export_wizard_nfl_pricing.py")

    assert exporter.SCHEMA_VERSION == "wizard-nfl-pricing-v2"
    assert publisher.SCHEMA_VERSION == "wizard-nfl-pricing-v2"
    assert publisher.TOP_LEVEL_KEYS == (
        "schema_version", "season", "week", "horizon", "generated_at_utc", "games",
    )
    assert publisher.GAME_KEYS == (
        "game_id", "kickoff_utc", "away_team", "home_team",
        "predicted_home_margin", "predicted_game_total",
        "market_home_spread", "market_total", "market_as_of_utc",
        "market_ats_book_count", "market_total_book_count",
    )
    assert publisher.MINIMUM_ELIGIBLE_BOOKS == 3
    assert publisher.ALLOWED_HORIZONS == ("TUE", "FRI")


def test_the_ssh_publisher_reuses_the_published_contract_rather_than_restating_it():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ssh_publisher_probe", REPO_ROOT / "scripts/publish_wizard_nfl_local.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.load_and_validate_json is module._contract.load_and_validate_json
    code = python_code_only(REPO_ROOT / "scripts/publish_wizard_nfl_local.py")
    for forbidden in ("ftplib", "socket", "urllib", "requests"):
        assert forbidden not in code, forbidden


# ===========================================================================
# 16-21. The one authoritative workflow
# ===========================================================================
def python_code_only(path: Path) -> str:
    """A Python file's executable source, with every comment and every string
    literal (including docstrings) removed.

    The forbidden-token scans below need this because these modules explain in
    prose exactly what they must never do ("no FTP, no FTPS", "never a
    hard-coded EDT/EST offset"), and a naive substring search would read those
    explanations as the very violations they rule out.
    """
    import io
    import tokenize

    kept = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


def executable_lines(text: str) -> str:
    """The non-comment lines of a shell/YAML/Python file.

    Every forbidden-token scan below runs against THIS, not the raw text: the
    files deliberately explain in prose what they must never do ("there is no
    FTP in this path"), and a naive substring search would read those
    explanations as violations.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


@pytest.fixture(scope="module")
def workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(WORKFLOW_PATH.read_text())
    # YAML 1.1 resolves the bare key `on` to the boolean True; normalise it so
    # the trigger block can be asserted by name.
    if True in parsed:
        parsed["on"] = parsed.pop(True)
    return parsed


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW_PATH.read_text()


def test_the_authoritative_workflow_exists_and_is_the_only_new_production_workflow():
    assert WORKFLOW_PATH.is_file()
    assert WORKFLOW_PATH.name == "nfl_2026_production.yml"


def test_server_touching_jobs_use_the_wizard_production_environment(workflow):
    server_jobs = ("daily-maintenance", "certified-production")
    for job in server_jobs:
        assert workflow["jobs"][job]["environment"] == "wizard-production", job
    # The resolve/report jobs must NOT claim the environment: they never touch
    # the server, and binding them would gate a pure decision on a deployment
    # approval.
    for job in ("resolve", "report"):
        assert "environment" not in workflow["jobs"][job], job


def test_deployment_is_restricted_to_main(workflow):
    for job in ("daily-maintenance", "certified-production"):
        assert "refs/heads/main" in workflow["jobs"][job]["if"], job
    gate = "\n".join(
        step.get("run", "") for step in workflow["jobs"]["resolve"]["steps"]
    )
    assert 'GITHUB_REF}" = "refs/heads/main"' in gate
    assert "deployment_allowed=false" in gate


def test_the_workflow_uses_ssh_and_never_ftp(workflow_text):
    steps = executable_lines(workflow_text)
    assert "ops/wizard/remote.sh" in steps
    for forbidden in ("ftplib", "SPORTSODDS_FTP", "publish_sportsodds_nfl.py", "lftp", "curl -T", "ftp://"):
        assert forbidden not in steps, forbidden

    # And the SSH wrapper itself pins host-key checking rather than relaxing it.
    remote = executable_lines((REPO_ROOT / "ops/wizard/remote.sh").read_text())
    assert "StrictHostKeyChecking=yes" in remote
    assert "StrictHostKeyChecking=no" not in remote
    assert "BatchMode=yes" in remote


def test_daily_maintenance_covers_every_required_daily_task(workflow_text):
    daily = (REPO_ROOT / "ops/wizard/run_daily_maintenance.sh").read_text()
    assert "run_daily_maintenance.sh" in workflow_text
    for required in (
        "refresh_bdl_2026_games_evidence.py",
        "update_2026_games_population.py",
        "attach_2026_results_from_population.py",
        "report_2026_prospective_performance.py",
        "generate_2026_recalibration_candidate.py",
    ):
        assert required in daily, required
    # The daily pass must never publish.
    assert "publish_wizard_nfl_local.py" not in daily
    assert "export_wizard_nfl_pricing.py" not in daily


def test_certified_production_covers_every_required_certified_task(workflow_text):
    certified = (REPO_ROOT / "ops/wizard/run_certified_card.sh").read_text()
    assert "run_certified_card.sh" in workflow_text
    for required in (
        "--preflight",
        "run_2026_production_card.py",
        "export_wizard_nfl_pricing.py",
        "publish_wizard_nfl_local.py",
        "sha256sum",
    ):
        assert required in certified, required
    # READY is mandatory, and the horizon can only ever be TUE or FRI.
    assert 'READY is required' in certified
    assert 'TUE|FRI)' in certified


def test_the_workflow_defines_no_daily_forecast_horizon(workflow, workflow_text):
    options = workflow["on"]["workflow_dispatch"]["inputs"]["horizon"]["options"]
    assert set(options) <= {"", "TUE", "FRI"}
    assert "DAILY" not in {str(o).upper() for o in options if o}


def test_scheduling_uses_america_new_york_logic_not_a_hardcoded_offset(workflow_text):
    assert "resolve_production_schedule.py" in workflow_text
    resolver = (REPO_ROOT / "scripts/resolve_production_schedule.py").read_text()
    assert "is_within_due_window" in resolver
    assert "current_or_recent_cutoff" in resolver
    assert "prod.NY_ZONE" in resolver

    # The executable code carries no offset arithmetic and no second timezone.
    code = python_code_only(REPO_ROOT / "scripts/resolve_production_schedule.py")
    for forbidden in ("EDT", "EST", "utcoffset", "timedelta", "ZoneInfo", "pytz"):
        assert forbidden not in code, forbidden


def test_public_verification_runs_in_both_passes(workflow_text):
    assert workflow_text.count("verify_public_nfl_feed.py") >= 2
    assert "--expect-sha256" in workflow_text


def test_a_not_due_scheduled_invocation_is_a_clean_no_op(workflow):
    for job in ("daily-maintenance", "certified-production"):
        condition = workflow["jobs"][job]["if"]
        assert "run_daily == 'true'" in condition or "run_certified == 'true'" in condition
    report = workflow["jobs"]["report"]
    assert report["if"] == "always()"
    outcome = "\n".join(step.get("run", "") for step in report["steps"])
    assert "clean no-op" in outcome


def test_workflow_dispatch_supports_an_explicit_existing_capture(workflow):
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "market_capture_manifest" in inputs
    assert "market_capture_sha256" in inputs
    certified = (REPO_ROOT / "ops/wizard/run_certified_card.sh").read_text()
    # Capture identity is resolved by the integrity verifier, which always
    # self-verifies the capture's own recorded content hash and fails closed
    # when a declared hash matches neither of its two integrity objects.
    assert "verify_capture_manifest_integrity.py" in certified
    assert "--declared-sha256" in certified
    assert "capture manifest integrity check failed" in certified
    assert "exit 3" in certified


# ===========================================================================
# The server layout never mixes NFL and NCAAF
# ===========================================================================
def test_the_wizard_layout_is_nfl_only_and_assumes_nothing_about_opt_or_srv():
    layout = (REPO_ROOT / "ops/wizard/nfl_production_layout.sh").read_text()
    lowered = layout.lower()
    assert "ncaaf" in lowered  # only as an explicit prohibition
    for line in layout.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert "ncaaf" not in stripped.lower(), line
        assert not stripped.startswith(": \"${WIZARD_NFL_HOME:=/opt"), line
        assert not stripped.startswith(": \"${WIZARD_NFL_HOME:=/srv"), line
    assert '${HOME}/nfl-production-2026' in layout


def test_the_web_directory_is_never_defaulted_to_a_guess():
    layout = (REPO_ROOT / "ops/wizard/nfl_production_layout.sh").read_text()
    assert ': "${WIZARD_NFL_WEB_DIR:=' not in layout
    publisher = (REPO_ROOT / "scripts/publish_wizard_nfl_local.py").read_text()
    assert "Refusing to guess a" in publisher
