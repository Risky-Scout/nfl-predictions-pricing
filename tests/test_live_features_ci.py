"""Core live/training feature-parity and leakage tests — RUN IN NORMAL CI.

Unlike :mod:`tests.test_live_features_extended`, this module uses ONLY the small,
committed, deterministic fixtures under ``tests/fixtures/live_features`` (synthetic
NFL-like teams and games; no private backfill, no purchased odds, no provider
payloads). It therefore never skips when the local 2020-2025 parquets are absent,
so GitHub CI enforces the most important live-feature guarantees on every push:

    all-19-feature train/live parity, identical NaN patterns, golden-output
    agreement, target-score / target-play-by-play / future-game invariance,
    post-as-of play exclusion, in-progress-game exclusion, historical/live mode
    separation, duplicate-target rejection, one-row-per-target, the frozen
    19-feature contract, and deterministic feature/finality hashing.

This module MUST contain no skip markers (a guard test enforces that). It fails
loudly if any committed fixture is missing, malformed, or tampered with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import (
    FROZEN_FEATURES,
    aggregate_advanced_team_game,
    build_augmented_feature_matrix,
    build_live_augmented_features,
    resolve_finality_before_asof,
)

# --- committed fixture locations -------------------------------------------------
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "live_features"
GAMES_CSV = FIXTURE_DIR / "games.csv"
PBP_CSV = FIXTURE_DIR / "pbp.csv"
EXPECTED_CSV = FIXTURE_DIR / "expected_features.csv"
MANIFEST_JSON = FIXTURE_DIR / "fixture_manifest.json"

# The frozen golden-parity targets and the two auxiliary game ids (see the factory).
TARGET_A = "2021_05_BUF_NYJ"   # complete last-four + valid season-to-date
TARGET_B = "2022_01_MIA_BUF"   # early-season / season-boundary missingness
TARGET_C = "2022_03_NE_BUF"    # mid-season, short-week rest flag
GOLDEN_TARGETS = (TARGET_A, TARGET_B, TARGET_C)
TARGET_D = "2022_04_MIA_BUF"   # upcoming: no outcome, no play-by-play
INPROGRESS_GAME = "2021_04_BUF_MIA"  # explicit non-final, excluded in live mode

# Expected home/away mapping per target (game_id encodes SEASON_WEEK_AWAY_HOME).
EXPECTED_MAPPING = {
    TARGET_A: ("NYJ", "BUF"),  # (home, away)
    TARGET_B: ("BUF", "MIA"),
    TARGET_C: ("BUF", "NE"),
    TARGET_D: ("BUF", "MIA"),
}

# Bumped intentionally when tests are added/removed; enforced by the no-skip guard.
EXPECTED_CORE_TEST_COUNT = 26


# --- loaders (fail loudly; NEVER skip) ------------------------------------------
def _require(path: Path) -> Path:
    if not path.exists():
        raise AssertionError(f"required committed fixture is missing: {path}")
    return path


def load_games() -> pd.DataFrame:
    return pd.read_csv(_require(GAMES_CSV))


def load_pbp() -> pd.DataFrame:
    return pd.read_csv(_require(PBP_CSV))


def load_expected() -> pd.DataFrame:
    exp = pd.read_csv(_require(EXPECTED_CSV))
    exp["game_id"] = exp["game_id"].astype(str)
    return exp.set_index("game_id")


def load_manifest() -> dict:
    return json.loads(_require(MANIFEST_JSON).read_text())


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return load_games()


@pytest.fixture(scope="module")
def pbp() -> pd.DataFrame:
    return load_pbp()


@pytest.fixture(scope="module")
def expected() -> pd.DataFrame:
    return load_expected()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest()


@pytest.fixture(scope="module")
def training_matrix(games, pbp) -> pd.DataFrame:
    """The accepted historical training path over the completed games only."""
    completed = games[games["home_score"].notna()].copy()
    matrix, _ = build_augmented_feature_matrix(completed, pbp)
    return matrix


@pytest.fixture(scope="module")
def live_by_target(games, pbp, manifest) -> dict:
    """Live (historical_replay) features + status per golden target, built once."""
    out = {}
    for gid in GOLDEN_TARGETS:
        out[gid] = _live(games, _target_frame(games, gid), pbp, _as_of(manifest, gid))
    return out


# --- helpers --------------------------------------------------------------------
def _target_frame(games: pd.DataFrame, gid: str) -> pd.DataFrame:
    t = games[games["game_id"].astype(str) == gid].copy()
    t["home_spread"] = pd.to_numeric(t["home_spread_reference"], errors="coerce")
    t["total_line"] = pd.to_numeric(t["total_line_reference"], errors="coerce")
    return t


def _as_of(manifest: dict, gid: str) -> str:
    return manifest["target_as_of_utc"][gid]


def _live(games, target, pbp, as_of, mode="historical_replay"):
    return build_live_augmented_features(games, target, pbp, mode=mode, as_of_utc=as_of)


def _assert_matches(actual, golden, tol=1e-10, ctx=""):
    """Exact NaN-pattern + numerical (<= tol) agreement over the 19 frozen features."""
    for c in FROZEN_FEATURES:
        a = actual[c]
        g = golden[c]
        assert pd.isna(a) == pd.isna(g), f"NaN pattern mismatch {ctx} {c}: {a!r} vs {g!r}"
        if pd.notna(a):
            assert abs(float(a) - float(g)) <= tol, f"value mismatch {ctx} {c}: {a!r} vs {g!r}"


def _finality_hash(finality: pd.DataFrame) -> str:
    """Deterministic hash of the verified-finality decision, independent of row order."""
    cols = ["game_id", "verified_final_before_asof", "historically_available_before_asof",
            "finality_status", "availability_status"]
    canon = finality[cols].sort_values("game_id").reset_index(drop=True)
    payload = canon.astype(str).to_dict(orient="records")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ================================================================================
# L. Frozen contract
# ================================================================================
def test_frozen_feature_contract(expected, manifest):
    assert len(FROZEN_FEATURES) == 19
    assert list(expected.reset_index().columns) == ["game_id"] + FROZEN_FEATURES
    # artifact feature contract == FROZEN_FEATURES
    repo_root = Path(__file__).resolve().parents[1]
    artifact_features = json.loads(
        (repo_root / "config" / "shadow_fundamental_2026" / "features.json").read_text()
    )
    assert artifact_features == FROZEN_FEATURES
    # golden fixture + manifest feature contract == FROZEN_FEATURES
    assert manifest["frozen_features"] == FROZEN_FEATURES
    assert manifest["frozen_features_sha256"] == hashlib.sha256(
        json.dumps(FROZEN_FEATURES).encode()
    ).hexdigest()


# ================================================================================
# Fixture integrity — fail loudly if a committed fixture is malformed/tampered.
# ================================================================================
def test_fixture_integrity_hashes(manifest):
    assert manifest["games_csv_sha256"] == hashlib.sha256(GAMES_CSV.read_bytes()).hexdigest()
    assert manifest["pbp_csv_sha256"] == hashlib.sha256(PBP_CSV.read_bytes()).hexdigest()
    assert manifest["expected_features_csv_sha256"] == hashlib.sha256(
        EXPECTED_CSV.read_bytes()
    ).hexdigest()
    assert set(manifest["target_game_ids"]) == set(GOLDEN_TARGETS)


# ================================================================================
# A. Exact live/training parity (all 19 features, order, NaN, home/away mapping)
# ================================================================================
@pytest.mark.parametrize("gid", GOLDEN_TARGETS)
def test_live_training_parity(gid, games, training_matrix, live_by_target):
    target = _target_frame(games, gid)
    feats, status = live_by_target[gid]
    hist = training_matrix[training_matrix["game_id"].astype(str) == gid]

    assert status["feature_row_count"] == 1
    assert len(hist) == 1
    # exact feature order + same game id
    assert list(feats.columns) == ["game_id"] + FROZEN_FEATURES
    assert feats["game_id"].iloc[0] == gid
    assert str(hist["game_id"].iloc[0]) == gid
    # same home and away mapping (game encodes SEASON_WEEK_AWAY_HOME)
    home, away = EXPECTED_MAPPING[gid]
    assert (target["home_team_id"].iloc[0], target["away_team_id"].iloc[0]) == (home, away)
    assert (str(hist["home_team_id"].iloc[0]), str(hist["away_team_id"].iloc[0])) == (home, away)
    # same NaN pattern + |diff| <= 1e-10
    _assert_matches(feats.iloc[0], hist.iloc[0], tol=1e-10, ctx=f"parity[{gid}]")


def test_complete_and_early_season_regimes(live_by_target):
    """At least one target has complete last-four + season-to-date history; at least
    one exercises expected early-season (season-boundary) missingness."""
    season_cols = [c for c in FROZEN_FEATURES if c.endswith("season_mean")]

    a_feats, _ = live_by_target[TARGET_A]
    assert not any(pd.isna(a_feats[c].iloc[0]) for c in FROZEN_FEATURES), (
        "Target A must have a complete last-four + season-to-date feature set"
    )

    b_feats, _ = live_by_target[TARGET_B]
    assert all(pd.isna(b_feats[c].iloc[0]) for c in season_cols), (
        "Target B must show early-season season-to-date missingness"
    )
    # last-four features still present across the season boundary
    assert pd.notna(b_feats["epa_net_edge_last4_mean"].iloc[0])


# ================================================================================
# B. Golden-output agreement — training AND live each independently match golden.
# ================================================================================
@pytest.mark.parametrize("gid", GOLDEN_TARGETS)
def test_training_matches_golden(gid, training_matrix, expected):
    hist = training_matrix[training_matrix["game_id"].astype(str) == gid].iloc[0]
    _assert_matches(hist, expected.loc[gid], tol=1e-10, ctx=f"train-golden[{gid}]")


@pytest.mark.parametrize("gid", GOLDEN_TARGETS)
def test_live_matches_golden(gid, expected, manifest, live_by_target):
    feats, status = live_by_target[gid]
    _assert_matches(feats.iloc[0], expected.loc[gid], tol=1e-10, ctx=f"live-golden[{gid}]")
    # snapshot-hash regression against the manifest
    assert status["feature_snapshot_hash"] == manifest["expected_snapshot_hashes"][gid]


# ================================================================================
# C. Target-score invariance
# ================================================================================
def test_target_score_invariance(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    base, sb = live_by_target[gid]
    g2 = games.copy()
    m = g2["game_id"].astype(str) == gid
    g2.loc[m, "home_score"] = 999
    g2.loc[m, "away_score"] = 0
    alt, sa = _live(g2, _target_frame(g2, gid), pbp, as_of)
    _assert_matches(alt.iloc[0], base.iloc[0], tol=0.0, ctx="score-invariance")
    assert sa["feature_snapshot_hash"] == sb["feature_snapshot_hash"]


# ================================================================================
# D. Target-play-by-play invariance
# ================================================================================
def test_target_pbp_invariance(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    base, sb = live_by_target[gid]
    p2 = pbp.copy()
    pm = p2["game_id"].astype(str) == gid
    p2.loc[pm, "epa"] = 50.0  # corrupt the target's own plays
    alt, sa = _live(games, _target_frame(games, gid), p2, as_of)
    _assert_matches(alt.iloc[0], base.iloc[0], tol=0.0, ctx="pbp-invariance")
    assert sa["feature_snapshot_hash"] == sb["feature_snapshot_hash"]


# ================================================================================
# E. Future-game invariance
# ================================================================================
def test_future_game_invariance(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    base, sb = live_by_target[gid]

    # add a later completed game (2021 wk6) and its play-by-play
    new_game = {
        "game_id": "2021_06_MIA_BUF", "season": 2021, "week": 6,
        "home_team_id": "BUF", "away_team_id": "MIA",
        "scheduled_kickoff_utc": "2021-10-17T17:00:00+00:00",
        "home_spread_reference": -3.0, "total_line_reference": 47.0,
        "home_score": 24, "away_score": 20, "game_status": "FINAL",
        "completion_timestamp_utc": "2021-10-17T20:30:00+00:00",
    }
    g2 = pd.concat([games, pd.DataFrame([new_game])], ignore_index=True)
    template = pbp[pbp["game_id"].astype(str) == INPROGRESS_GAME].copy()
    template["game_id"] = "2021_06_MIA_BUF"
    template["week"] = 6
    template["start_time"] = "2021-10-17T17:05:00+00:00"
    p2 = pd.concat([pbp, template], ignore_index=True)

    alt, sa = _live(g2, _target_frame(g2, gid), p2, as_of)
    _assert_matches(alt.iloc[0], base.iloc[0], tol=0.0, ctx="future-invariance")
    assert sa["feature_snapshot_hash"] == sb["feature_snapshot_hash"]


# ================================================================================
# F. Post-as-of play exclusion
# ================================================================================
def test_post_asof_play_exclusion(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    base, sb = live_by_target[gid]

    prior = "2021_03_NE_BUF"  # eligible prior game for Target A
    extra = pbp[pbp["game_id"].astype(str) == prior].iloc[0].to_dict()
    extra["epa"] = 999.0
    extra["success"] = 1.0
    extra["start_time"] = (pd.to_datetime(as_of, utc=True) + pd.Timedelta(hours=1)).isoformat()
    p2 = pd.concat([pbp, pd.DataFrame([extra])], ignore_index=True)

    alt, sa = _live(games, _target_frame(games, gid), p2, as_of)
    # the post-as-of play is excluded: audit count increments, features unchanged
    assert sa["excluded_post_asof_play_count"] == sb["excluded_post_asof_play_count"] + 1
    _assert_matches(alt.iloc[0], base.iloc[0], tol=0.0, ctx="post-asof-play")
    assert sa["feature_snapshot_hash"] == sb["feature_snapshot_hash"]


# ================================================================================
# G. In-progress / unknown-game exclusion (live mode), then eligibility on evidence
# ================================================================================
def test_in_progress_game_excluded_then_eligible(games, pbp, manifest):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)

    # the in-progress game is not verified/eligible in live mode
    fin = resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="live")
    frow = fin.set_index("game_id").loc[INPROGRESS_GAME]
    assert bool(frow["verified_final_before_asof"]) is False
    assert frow["finality_status"] == "EXPLICIT_NON_FINAL"

    # zero plays reach completed-game aggregation; zero team-game facts produced
    eligible = set(fin[fin["verified_final_before_asof"]]["game_id"].astype(str))
    kept_pbp = pbp[pbp["game_id"].astype(str).isin(eligible)]
    team_game = aggregate_advanced_team_game(kept_pbp)
    assert INPROGRESS_GAME not in set(team_game["game_id"].astype(str))
    assert (kept_pbp["game_id"].astype(str) == INPROGRESS_GAME).sum() == 0

    base, sbase = _live(games, _target_frame(games, gid), pbp, as_of, mode="live")

    # its own play data does not leak while excluded (no target feature impact)
    p2 = pbp.copy()
    p2.loc[p2["game_id"].astype(str) == INPROGRESS_GAME, "epa"] = 77.0
    corrupt, _ = _live(games, _target_frame(games, gid), p2, as_of, mode="live")
    _assert_matches(corrupt.iloc[0], base.iloc[0], tol=0.0, ctx="in-progress-no-impact")

    # add valid finality evidence before as-of -> becomes eligible and changes features
    g2 = games.copy()
    m = g2["game_id"].astype(str) == INPROGRESS_GAME
    ko = pd.to_datetime(g2.loc[m, "scheduled_kickoff_utc"].iloc[0], utc=True)
    g2.loc[m, "game_status"] = "FINAL"
    g2.loc[m, "completion_timestamp_utc"] = (ko + pd.Timedelta(hours=3, minutes=30)).isoformat()
    evid, sev = _live(g2, _target_frame(g2, gid), pbp, as_of, mode="live")

    assert sev["eligible_final_game_count"] == sbase["eligible_final_game_count"] + 1
    assert sev["feature_snapshot_hash"] != sbase["feature_snapshot_hash"]
    changed = [
        c for c in FROZEN_FEATURES
        if (pd.isna(base[c].iloc[0]) != pd.isna(evid[c].iloc[0]))
        or (pd.notna(base[c].iloc[0]) and abs(float(base[c].iloc[0]) - float(evid[c].iloc[0])) > 1e-9)
    ]
    assert changed, "adding finality evidence for the prior game must change target features"


# ================================================================================
# H. Historical/live mode separation
# ================================================================================
def test_live_mode_no_availability_assumption(games, pbp, manifest):
    as_of = _as_of(manifest, TARGET_A)
    live = resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="live")
    replay = resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="historical_replay")

    # live mode never uses HISTORICAL_AVAILABILITY_ASSUMPTION
    assert not live["historically_available_before_asof"].any()
    assert not (live["availability_status"] == "HISTORICAL_AVAILABILITY_ASSUMPTION").any()

    # historical_replay may use it for the in-progress game, and verified stays False
    xr = replay.set_index("game_id").loc[INPROGRESS_GAME]
    assert bool(xr["historically_available_before_asof"]) is True
    assert bool(xr["verified_final_before_asof"]) is False
    assert xr["availability_status"] == "HISTORICAL_AVAILABILITY_ASSUMPTION"
    # and it is unavailable in live mode
    xl = live.set_index("game_id").loc[INPROGRESS_GAME]
    assert bool(xl["historically_available_before_asof"]) is False
    assert bool(xl["verified_final_before_asof"]) is False


# ================================================================================
# I. Kickoff boundary
# ================================================================================
def test_kickoff_at_or_before_asof_fails(games, pbp):
    target = _target_frame(games, TARGET_A)
    at_kickoff = pd.to_datetime(target["scheduled_kickoff_utc"].iloc[0], utc=True).isoformat()
    with pytest.raises(ValueError, match="at or before"):
        _live(games, target, pbp, at_kickoff)


# ================================================================================
# J. Duplicate target rejection
# ================================================================================
def test_duplicate_target_fails(games, pbp, manifest):
    target = _target_frame(games, TARGET_A)
    dup = pd.concat([target, target], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate target"):
        _live(games, dup, pbp, _as_of(manifest, TARGET_A))


# ================================================================================
# K. One-row-per-target invariant (multi-target build incl. no-outcome target D)
# ================================================================================
def test_one_row_per_target(games, pbp, manifest):
    # a single as-of before every target's kickoff (Target A's as-of predates all four)
    as_of = _as_of(manifest, TARGET_A)
    targets = pd.concat(
        [_target_frame(games, g) for g in (TARGET_A, TARGET_B, TARGET_C, TARGET_D)],
        ignore_index=True,
    )
    feats, status = _live(games, targets, pbp, as_of)
    assert status["feature_row_count"] == 4
    assert list(feats["game_id"]) == [TARGET_A, TARGET_B, TARGET_C, TARGET_D]
    assert feats["game_id"].value_counts().max() == 1
    assert list(feats.columns) == ["game_id"] + FROZEN_FEATURES


def test_no_outcome_target_builds(games, pbp, manifest):
    """Target D has no outcome and no play-by-play; live assembly still yields one row
    with market lines attached."""
    as_of = _as_of(manifest, TARGET_A)
    feats, status = _live(games, _target_frame(games, TARGET_D), pbp, as_of)
    assert status["feature_row_count"] == 1
    assert pd.notna(feats["home_spread"].iloc[0])
    assert (pbp["game_id"].astype(str) == TARGET_D).sum() == 0  # genuinely no target pbp


# ================================================================================
# M. Deterministic hashing (values, NaN, snapshot hash, finality hash; row order)
# ================================================================================
def test_deterministic_hashing(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    f1, s1 = live_by_target[gid]
    f2, s2 = _live(games, _target_frame(games, gid), pbp, as_of)
    assert s1["feature_snapshot_hash"] == s2["feature_snapshot_hash"]
    for c in FROZEN_FEATURES:
        assert pd.isna(f1[c].iloc[0]) == pd.isna(f2[c].iloc[0])
        if pd.notna(f1[c].iloc[0]):
            assert float(f1[c].iloc[0]) == float(f2[c].iloc[0])
    # finality decision hash is deterministic too
    h1 = _finality_hash(resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="live"))
    h2 = _finality_hash(resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="live"))
    assert h1 == h2


def test_row_reorder_invariant_hash(games, pbp, manifest, live_by_target):
    gid = TARGET_A
    as_of = _as_of(manifest, gid)
    base, sbase = live_by_target[gid]
    hbase = _finality_hash(resolve_finality_before_asof(games, pbp, as_of_utc=as_of, mode="live"))

    # reverse input row order (canonical sorting must neutralise it)
    g2 = games.iloc[::-1].reset_index(drop=True)
    p2 = pbp.iloc[::-1].reset_index(drop=True)
    alt, salt = _live(g2, _target_frame(g2, gid), p2, as_of)
    # canonical sorting neutralises row order; values agree within tolerance and the
    # rounded snapshot hash is byte-identical (summation order can differ by ~1e-17)
    _assert_matches(alt.iloc[0], base.iloc[0], tol=1e-10, ctx="reorder")
    assert salt["feature_snapshot_hash"] == sbase["feature_snapshot_hash"]
    halt = _finality_hash(resolve_finality_before_asof(g2, p2, as_of_utc=as_of, mode="live"))
    assert halt == hbase


# ================================================================================
# Live-mode fail-closed: no eligible completed games -> all EPA NaN, lines attach.
# ================================================================================
def test_no_eligible_completed_games_live(games, pbp, manifest):
    g2 = games.copy()
    g2["game_status"] = ""
    g2["completion_timestamp_utc"] = ""  # strip all verified-finality evidence
    feats, status = _live(g2, _target_frame(g2, TARGET_A), pbp, _as_of(manifest, TARGET_A), mode="live")
    assert status["verified_final_game_count"] == 0
    assert status.get("no_eligible_completed_games") is True
    assert pd.isna(feats["epa_net_edge_season_mean"].iloc[0])
    assert pd.notna(feats["home_spread"].iloc[0])  # market lines still attach


# ================================================================================
# CI guard (§7): this module must run with ZERO skips and an exact test count.
# ================================================================================
def test_core_suite_runs_with_no_skips(request):
    this_file = Path(__file__).name
    items = [i for i in request.session.items if Path(str(i.fspath)).name == this_file]
    assert len(items) == EXPECTED_CORE_TEST_COUNT, (
        f"expected {EXPECTED_CORE_TEST_COUNT} core tests, collected {len(items)}"
    )
    for item in items:
        marks = {m.name for m in item.iter_markers()}
        assert "skip" not in marks and "skipif" not in marks, (
            f"core CI test must never skip: {item.nodeid}"
        )
    # module carries no module-level skip marker
    import tests.test_live_features_ci as mod
    assert not getattr(mod, "pytestmark", None)
