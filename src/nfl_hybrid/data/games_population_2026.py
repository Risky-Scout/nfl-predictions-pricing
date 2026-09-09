"""The ONE composite 2026 production games/schedule population.

    certified historical backfill.games (2020-2025)
      +
    canonical BallDontLie 2026 REG/POST games
      =
    the single games population used by BOTH production preflight and real
    production execution

WHY THIS MODULE EXISTS
  ``nfl_hybrid.production.run_2026`` originally resolved its games population
  from ``backfill.games`` alone, whose maximum season is 2025. That made
  ``schedule_2026_available`` structurally ``false`` -- production reported
  ``BLOCKED_ON_LIVE_INPUTS`` forever -- even on a host holding a valid,
  verified 2026 BallDontLie capture. The repository already had every piece
  needed to fix that (captured ``/games`` rows,
  :func:`nfl_hybrid.data.bdl_market_bridge.read_games_rows`,
  :func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games` and its
  canonical game-identity/team crosswalk); production simply never consumed
  them.

NO SECOND NORMALIZER
  The 2026 rows here are produced by the EXISTING canonical BDL path,
  verbatim: a capture is validated by
  :func:`nfl_hybrid.data.bdl_market_bridge.validate_capture_manifest`, its
  raw ``/games`` rows are read by
  :func:`nfl_hybrid.data.bdl_market_bridge.read_games_rows`, and those rows
  are canonicalized by
  :func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games`. This
  module only RENAMES that canonical frame's columns onto the games-population
  contract (:data:`nfl_hybrid.features.horizon_elo.REQUIRED_GAME_COLUMNS`
  plus the public identity columns
  :func:`nfl_hybrid.production.run_2026.resolve_forecast_identity` reads).
  There is no team mapping, no game-id construction, no kickoff parsing and
  no season-type inference anywhere in this file.

FROZEN SCIENCE IS UNTOUCHED
  Nothing here changes the six frozen ELO_STRENGTH features, Ridge alpha=100,
  chronological training eligibility, the Fix-8 TUE/FRI horizon semantics,
  ATS/TOTAL target definitions, the >=3-book market rule, the <=48h quote-age
  rule, calibration math, or any certified hash. Appending real 2026 games to
  the population is exactly what makes the ALREADY-CERTIFIED chronological
  rules include completed 2026 games in later eligible fits -- automatically,
  through the existing training mask (``result_available_at_utc <
  target_cutoff_utc``, STRICT), with no new retraining engine.

PRE IS NEVER INCLUDED
  Preseason is excluded here (:data:`PRODUCTION_SEASON_TYPES`) AND again
  downstream by ``run_2026.filter_reg_post``. BDL exposes preseason games and
  restarts preseason week numbering at 1, so a preseason row could otherwise
  collide with a regular-season Week-1 canonical game_id.

UPCOMING GAMES CARRY NULL SCORES
  A scheduled 2026 game has ``home_score``/``away_score`` NULL. That is
  correct and required: the certified Elo replay only enqueues an update event
  for a game with both scores present
  (:func:`nfl_hybrid.features.horizon_elo.build_horizon_elo_state`), and a
  future game's ``result_available_at_utc`` is after every current cutoff, so
  it can never enter a training batch.

FAIL CLOSED
  * A conflicting canonical identity (same ``game_id``, different teams or
    kickoff) -- whether between two 2026 evidence rows, between two captures,
    or between 2026 evidence and the certified historical estate -- raises
    :class:`GamesPopulationConflict`. Nothing is merged "best effort".
  * A recorded final score is immutable: evidence that changes an already
    stored score raises :class:`GamesPopulationConflict`.
  * A game whose result was already chronologically available as of the
    evidence's own as-of instant but whose scores are still unknown is a STALE
    UNRESOLVED row. It is excluded from the population and counted explicitly
    in the manifest, so it can never become a training observation with an
    absent outcome and can never be silently treated as completed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.data.external_data import artifact_root, resolve
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.providers.balldontlie import canonical as bdl_canonical

SCHEMA_VERSION = "games-population-2026-v1"

# The durable 2026 canonical games population lives under the GENERATED
# artifact root -- never NFL_MODEL_DATA_ROOT (which holds the certified
# historical estate) and never the git repository (live data is never
# committed).
POPULATION_NAMESPACE = "games-population-2026"
POPULATION_FILENAME = "canonical_games_2026.parquet"
MANIFEST_FILENAME = "canonical_games_2026.manifest.json"

PRODUCTION_SEASON = 2026
# REG+POST only. Identical to the capture bridge's own
# ``PRODUCTION_SEASON_TYPES`` and to ``run_2026.REG_POST_SEASON_TYPES``;
# PRESEASON is never part of the 2026 production population.
PRODUCTION_SEASON_TYPES: tuple[str, ...] = ("REG", "POST")

# The horizon label a daily games/results-evidence capture records. It is
# deliberately NOT a forecast horizon: it names no cutoff, produces no market,
# and can never satisfy the certified market path (which only ever accepts
# ``bridge.PRODUCTION_HORIZONS``).
GAMES_EVIDENCE_HORIZON = "GAMES_EVIDENCE"

# Horizons whose captures count as valid 2026 schedule/results evidence: the
# daily games-evidence refresh, plus the official TUE/FRI production captures
# (which carry the same verified ``/games`` pages). SMOKE is absent -- schema
# evidence is never production evidence.
GAMES_EVIDENCE_HORIZONS: tuple[str, ...] = (GAMES_EVIDENCE_HORIZON,) + bridge.PRODUCTION_HORIZONS

# The games-population contract: every column the certified horizon/Elo path
# requires (``he.REQUIRED_GAME_COLUMNS``) plus nothing else. Kept as an
# explicit ordered tuple so the composite frame's schema is stable and
# provable, and derived FROM the certified requirement rather than restating
# it independently.
POPULATION_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "season_type",
    "home_team_id",
    "away_team_id",
    "scheduled_kickoff_utc",
    "home_score",
    "away_score",
)

# The canonical identity of a game: the fields two evidence rows must agree on
# before they may be treated as the same game. Scores are deliberately absent
# -- a score legitimately transitions from NULL to final exactly once.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "season_type",
    "home_team_id",
    "away_team_id",
    "scheduled_kickoff_utc",
)

# Column renames from the existing canonical BDL games frame
# (``bdl_canonical.GAMES_COLUMNS``) onto the population contract above. This
# is the ONLY transformation this module performs on canonical BDL output.
_BDL_TO_POPULATION = {
    "home_team": "home_team_id",
    "away_team": "away_team_id",
    "kickoff_utc": "scheduled_kickoff_utc",
}

STATUS_INCLUDED = "INCLUDED"
STATUS_STALE_UNRESOLVED = "STALE_UNRESOLVED"
STATUS_EXCLUDED_SEASON_TYPE = "EXCLUDED_SEASON_TYPE"


class GamesPopulationError(RuntimeError):
    """The composite games population could not be built without fabricating
    or discarding evidence. Never downgraded to a warning."""


class GamesPopulationConflict(GamesPopulationError):
    """Two sources disagree about the same canonical game identity, or an
    immutable recorded score changed. Always fail closed -- production never
    picks a winner between conflicting identities."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256_hex(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _as_utc(value: object, field: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise GamesPopulationError(f"{field} is unparseable: {value!r}") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise GamesPopulationError(f"{field} is missing/unparseable: {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Canonical BDL 2026 games -> population rows. Reuse only; no re-derivation.
# ---------------------------------------------------------------------------
def _nullable_score(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def canonical_games_to_population_rows(normalized: pd.DataFrame) -> pd.DataFrame:
    """Project an EXISTING canonical BDL games frame
    (:func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games`
    output) onto :data:`POPULATION_COLUMNS`.

    Renames only. PRESEASON rows are dropped (never renamed into REG), and a
    row missing any identity field fails closed rather than being filled in.
    """
    required = set(_BDL_TO_POPULATION) | {"game_id", "season", "week", "season_type", "home_score", "away_score"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise GamesPopulationError(
            f"canonical BDL games frame is missing expected column(s) {missing}; "
            "expected the schema produced by balldontlie.canonical.normalize_games"
        )

    frame = normalized.rename(columns=_BDL_TO_POPULATION).copy()
    frame["season_type"] = frame["season_type"].astype(str).str.upper()
    frame = frame.loc[frame["season_type"].isin(PRODUCTION_SEASON_TYPES)].reset_index(drop=True)

    rows: list[dict] = []
    for record in frame.to_dict(orient="records"):
        game_id = str(record["game_id"]).strip()
        if not game_id:
            raise GamesPopulationError("canonical BDL games frame contains a row with an empty game_id")
        home = str(record["home_team_id"]).strip()
        away = str(record["away_team_id"]).strip()
        if not home or not away:
            raise GamesPopulationError(f"{game_id}: canonical BDL row has an empty home/away team id")
        if home == away:
            raise GamesPopulationError(f"{game_id}: canonical BDL row has home_team_id == away_team_id ({home!r})")
        rows.append(
            {
                "game_id": game_id,
                "season": int(record["season"]),
                "week": str(record["week"]),
                "season_type": record["season_type"],
                "home_team_id": home,
                "away_team_id": away,
                "scheduled_kickoff_utc": _as_utc(
                    record["scheduled_kickoff_utc"], f"{game_id}: scheduled_kickoff_utc"
                ),
                "home_score": _nullable_score(record.get("home_score")),
                "away_score": _nullable_score(record.get("away_score")),
            }
        )
    return _normalize_population_frame(pd.DataFrame(rows, columns=list(POPULATION_COLUMNS)))


def games_from_capture(
    manifest_path: str | Path,
    *,
    expected_season: int | None = PRODUCTION_SEASON,
) -> tuple[pd.DataFrame, dict]:
    """Canonical 2026 population rows for ONE explicitly supplied capture.

    The capture is verified by the existing
    :func:`nfl_hybrid.data.bdl_market_bridge.validate_capture_manifest` (schema
    version, COMPLETE status, allowed horizon with no downgrade, recomputed
    manifest hash, per-page response hashes and byte counts), its raw
    ``/games`` rows are read by the existing
    :func:`nfl_hybrid.data.bdl_market_bridge.read_games_rows`, and they are
    canonicalized by the existing
    :func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games` with the
    capture's own recorded ``season_type`` as the query hint. No capture is
    ever auto-selected here.

    Only the ``games`` source is required, and the recognised horizons are
    :data:`GAMES_EVIDENCE_HORIZONS` -- a daily schedule/results refresh
    legitimately carries no odds and no TUE/FRI cutoff, while an official
    TUE/FRI production capture is equally valid schedule evidence. The
    certified MARKET path is unaffected: it calls
    :func:`nfl_hybrid.data.bdl_market_bridge.load_live_market_source`, which
    keeps the validator's own defaults (``games`` + ``odds_current``, TUE/FRI
    only).
    """
    try:
        capture = bridge.validate_capture_manifest(
            manifest_path,
            expected_season=expected_season,
            required_sources=(bridge.GAMES_LOGICAL_NAME,),
            allowed_horizons=GAMES_EVIDENCE_HORIZONS,
        )
    except bridge.BdlMarketBridgeError as exc:
        raise GamesPopulationError(f"capture rejected: {exc}") from exc

    raw_games = bridge.read_games_rows(capture)
    try:
        normalized = bdl_canonical.normalize_games(raw_games, season_type_hint=capture.season_type)
    except bdl_canonical.CanonicalizationError as exc:
        raise GamesPopulationError(f"capture games could not be canonicalized: {exc}") from exc

    rows = canonical_games_to_population_rows(normalized)
    evidence = {
        "provider": bdl_canonical.PROVIDER_NAME,
        "manifest_path": str(capture.manifest_path),
        "manifest_sha256": capture.manifest_sha256,
        "season": capture.season,
        "week": capture.week,
        "horizon": capture.horizon,
        "season_type": capture.season_type,
        "evidence_as_of_utc": _capture_evidence_as_of(capture).isoformat(),
        "raw_games_row_count": len(raw_games),
        "canonical_row_count": int(len(rows)),
    }
    return rows, evidence


def _capture_evidence_as_of(capture: bridge.ValidatedCapture) -> pd.Timestamp:
    """The instant this capture's evidence is current as of: the capture's own
    recorded completion time, falling back to its start time. Never the
    reading machine's clock -- a stored capture's evidence does not become
    more current because it is read later."""
    manifest = capture.manifest
    for field in ("capture_completed_at_utc", "capture_started_at_utc"):
        value = manifest.get(field)
        if value:
            return _as_utc(value, f"capture {field}")
    return capture.nominal_cutoff_utc


# ---------------------------------------------------------------------------
# Frame normalization + deterministic hashing.
# ---------------------------------------------------------------------------
def _normalize_population_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(POPULATION_COLUMNS) - set(frame.columns))
    if missing:
        raise GamesPopulationError(f"games population frame is missing column(s) {missing}")
    out = frame[list(POPULATION_COLUMNS)].copy()
    out["game_id"] = out["game_id"].astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = out["week"].astype(str)
    out["season_type"] = out["season_type"].astype(str).str.upper()
    out["home_team_id"] = out["home_team_id"].astype(str)
    out["away_team_id"] = out["away_team_id"].astype(str)
    out["scheduled_kickoff_utc"] = pd.to_datetime(out["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if out["scheduled_kickoff_utc"].isna().any():
        bad = out.loc[out["scheduled_kickoff_utc"].isna(), "game_id"].tolist()
        raise GamesPopulationError(f"games population rows have an unparseable scheduled_kickoff_utc: {bad}")
    for column in ("home_score", "away_score"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)


def population_content_hash(frame: pd.DataFrame) -> str:
    """Order-independent content hash of a games population.

    Computed over the frame's own values (never parquet bytes, which are not
    byte-stable across writer versions), so "the same population materialized
    twice" is provably the same content, and preflight and execution can prove
    they consumed the SAME population.
    """
    ordered = _normalize_population_frame(frame)
    records = []
    for record in ordered.to_dict(orient="records"):
        records.append(
            [
                record["game_id"],
                int(record["season"]),
                record["week"],
                record["season_type"],
                record["home_team_id"],
                record["away_team_id"],
                pd.Timestamp(record["scheduled_kickoff_utc"]).isoformat(),
                None if pd.isna(record["home_score"]) else repr(float(record["home_score"])),
                None if pd.isna(record["away_score"]) else repr(float(record["away_score"])),
            ]
        )
    records.sort(key=lambda row: row[0])
    return _sha256_hex({"schema_version": SCHEMA_VERSION, "columns": list(POPULATION_COLUMNS), "rows": records})


# ---------------------------------------------------------------------------
# Deterministic merge of new evidence into an existing population.
# ---------------------------------------------------------------------------
def _identity_of(record: dict) -> dict:
    return {
        "season": int(record["season"]),
        "week": str(record["week"]),
        "season_type": str(record["season_type"]),
        "home_team_id": str(record["home_team_id"]),
        "away_team_id": str(record["away_team_id"]),
        "scheduled_kickoff_utc": pd.Timestamp(record["scheduled_kickoff_utc"]).isoformat(),
    }


def _score_pair(record: dict) -> tuple[float | None, float | None]:
    home = record.get("home_score")
    away = record.get("away_score")
    return (
        None if home is None or pd.isna(home) else float(home),
        None if away is None or pd.isna(away) else float(away),
    )


def merge_population_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Deterministically merge ``incoming`` evidence into ``existing``.

    Semantics, in full:

      * a ``game_id`` absent from ``existing`` is APPENDED;
      * a ``game_id`` present in both must carry a byte-identical canonical
        identity (:data:`IDENTITY_COLUMNS`); any difference is a hard
        :class:`GamesPopulationConflict` (the row is never overwritten,
        never merged field-by-field, never resolved by recency);
      * scores may transition NULL -> final exactly once. An already-stored
        final score that the incoming evidence contradicts is a hard
        :class:`GamesPopulationConflict`; final -> NULL is IGNORED (a
        completed result is durable evidence and is never un-recorded by a
        later, less complete read);
      * the result is independent of merge order for any conflict-free
        evidence set.
    """
    existing = _normalize_population_frame(existing) if len(existing) else _empty_population()
    incoming = _normalize_population_frame(incoming) if len(incoming) else _empty_population()

    merged: dict[str, dict] = {str(r["game_id"]): dict(r) for r in existing.to_dict(orient="records")}

    for record in incoming.to_dict(orient="records"):
        game_id = str(record["game_id"])
        current = merged.get(game_id)
        if current is None:
            merged[game_id] = dict(record)
            continue

        current_identity = _identity_of(current)
        incoming_identity = _identity_of(record)
        if current_identity != incoming_identity:
            raise GamesPopulationConflict(
                f"FAIL CLOSED: conflicting canonical identity for game_id {game_id!r}: "
                f"stored={current_identity} incoming={incoming_identity}"
            )

        stored_home, stored_away = _score_pair(current)
        new_home, new_away = _score_pair(record)
        if stored_home is not None or stored_away is not None:
            if (new_home, new_away) != (stored_home, stored_away) and (new_home is not None or new_away is not None):
                raise GamesPopulationConflict(
                    f"FAIL CLOSED: recorded result for game_id {game_id!r} changed: "
                    f"stored=({stored_home}, {stored_away}) incoming=({new_home}, {new_away})"
                )
            continue
        merged[game_id] = dict(record)

    return _normalize_population_frame(pd.DataFrame(list(merged.values()), columns=list(POPULATION_COLUMNS)))


def _empty_population() -> pd.DataFrame:
    return _normalize_population_frame(
        pd.DataFrame(
            {
                "game_id": pd.Series(dtype="object"),
                "season": pd.Series(dtype="int64"),
                "week": pd.Series(dtype="object"),
                "season_type": pd.Series(dtype="object"),
                "home_team_id": pd.Series(dtype="object"),
                "away_team_id": pd.Series(dtype="object"),
                "scheduled_kickoff_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "home_score": pd.Series(dtype="float64"),
                "away_score": pd.Series(dtype="float64"),
            }
        )
    )


def classify_rows(population: pd.DataFrame, *, evidence_as_of_utc: pd.Timestamp | str) -> pd.DataFrame:
    """Label each row ``INCLUDED`` or ``STALE_UNRESOLVED``.

    A row is STALE_UNRESOLVED when its result was already chronologically
    available as of ``evidence_as_of_utc`` -- using the CERTIFIED availability
    policy (:func:`nfl_hybrid.features.horizon_elo.compute_result_available_at_utc`,
    the existing conservative kickoff+5h floor; no second clock, no second
    policy) -- but its scores are still unknown.

    Such a row cannot honestly be a training observation (its outcome is
    absent) and cannot honestly be an upcoming forecast target (its kickoff
    has passed). It is excluded and counted rather than silently carried.
    """
    frame = _normalize_population_frame(population)
    as_of = _as_utc(evidence_as_of_utc, "evidence_as_of_utc")
    if frame.empty:
        frame["result_available_at_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        frame["row_status"] = pd.Series(dtype="object")
        return frame

    frame["result_available_at_utc"] = he.compute_result_available_at_utc(frame)
    scores_known = frame["home_score"].notna() & frame["away_score"].notna()
    resolved_by_now = frame["result_available_at_utc"] <= as_of
    frame["row_status"] = STATUS_INCLUDED
    frame.loc[resolved_by_now & ~scores_known, "row_status"] = STATUS_STALE_UNRESOLVED
    return frame


# ---------------------------------------------------------------------------
# The durable 2026 population store.
# ---------------------------------------------------------------------------
def population_dir(artifact_root_path: Path | None = None) -> Path:
    root = Path(artifact_root_path) if artifact_root_path is not None else artifact_root()
    return root / POPULATION_NAMESPACE


def population_path(artifact_root_path: Path | None = None) -> Path:
    return population_dir(artifact_root_path) / POPULATION_FILENAME


def population_manifest_path(artifact_root_path: Path | None = None) -> Path:
    return population_dir(artifact_root_path) / MANIFEST_FILENAME


def read_durable_2026_population(artifact_root_path: Path | None = None) -> tuple[pd.DataFrame, dict | None]:
    """The durable 2026 canonical population and its manifest, or an empty
    population and ``None`` when no 2026 evidence has been recorded yet.

    Never raises for absence: a host with no 2026 evidence must be able to
    report ``schedule_2026_available=false`` honestly rather than crash.
    """
    try:
        path = population_path(artifact_root_path)
        manifest_file = population_manifest_path(artifact_root_path)
    except Exception:
        return _empty_population(), None
    if not path.is_file():
        return _empty_population(), None
    frame = _normalize_population_frame(pd.read_parquet(path))
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else None
    return frame, manifest


@dataclass(frozen=True)
class PopulationUpdate:
    status: str  # WRITTEN | IDEMPOTENT_NOOP
    path: Path
    manifest: dict
    population: pd.DataFrame


def update_durable_2026_population(
    incoming: pd.DataFrame,
    *,
    evidence: list[dict],
    artifact_root_path: Path | None = None,
) -> PopulationUpdate:
    """Merge new canonical 2026 evidence into the durable population and
    persist it atomically, together with a manifest recording the exact
    contributing capture identities, hashes and row classifications.

    Idempotent: re-applying the same evidence rewrites nothing and reports
    ``IDEMPOTENT_NOOP``. Conflicting identities or a changed recorded score
    fail closed via :func:`merge_population_rows` BEFORE anything is written.
    """
    target_dir = population_dir(artifact_root_path)
    path = target_dir / POPULATION_FILENAME
    manifest_file = target_dir / MANIFEST_FILENAME

    existing, existing_manifest = read_durable_2026_population(artifact_root_path)
    merged = merge_population_rows(existing, incoming)

    off_season = sorted({int(s) for s in merged["season"].tolist()} - {PRODUCTION_SEASON})
    if off_season:
        raise GamesPopulationError(
            f"the durable 2026 population may only contain season {PRODUCTION_SEASON}; got extra season(s) {off_season}"
        )

    prior_evidence = list((existing_manifest or {}).get("evidence", []))
    known = {(e.get("manifest_sha256"), e.get("manifest_path")) for e in prior_evidence}
    combined_evidence = list(prior_evidence)
    for entry in evidence:
        if (entry.get("manifest_sha256"), entry.get("manifest_path")) not in known:
            combined_evidence.append(entry)
            known.add((entry.get("manifest_sha256"), entry.get("manifest_path")))
    combined_evidence.sort(key=lambda e: (str(e.get("evidence_as_of_utc")), str(e.get("manifest_sha256"))))

    evidence_as_of = max(
        (_as_utc(e["evidence_as_of_utc"], "evidence_as_of_utc") for e in combined_evidence if e.get("evidence_as_of_utc")),
        default=None,
    )
    classified = (
        classify_rows(merged, evidence_as_of_utc=evidence_as_of) if evidence_as_of is not None else classify_rows(merged, evidence_as_of_utc=pd.Timestamp("1970-01-01T00:00:00Z"))
    )
    stale = classified.loc[classified["row_status"] == STATUS_STALE_UNRESOLVED, "game_id"].tolist()

    content_hash = population_content_hash(merged)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "season": PRODUCTION_SEASON,
        "season_types": list(PRODUCTION_SEASON_TYPES),
        "row_count": int(len(merged)),
        "completed_row_count": int((merged["home_score"].notna() & merged["away_score"].notna()).sum()),
        "scheduled_row_count": int((merged["home_score"].isna() | merged["away_score"].isna()).sum()),
        "stale_unresolved_game_ids": sorted(stale),
        "content_sha256": content_hash,
        "evidence_as_of_utc": None if evidence_as_of is None else evidence_as_of.isoformat(),
        "evidence": combined_evidence,
    }

    if existing_manifest is not None and existing_manifest.get("content_sha256") == content_hash and path.is_file():
        if existing_manifest.get("evidence") == combined_evidence:
            return PopulationUpdate("IDEMPOTENT_NOOP", path, existing_manifest, merged)

    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet = path.with_name(path.name + ".tmp")
    merged.to_parquet(tmp_parquet, index=False)
    tmp_parquet.replace(path)
    tmp_manifest = manifest_file.with_name(manifest_file.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp_manifest.replace(manifest_file)
    return PopulationUpdate("WRITTEN", path, manifest, merged)


# ---------------------------------------------------------------------------
# The composite population -- the ONE source production reads.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompositePopulation:
    games: pd.DataFrame
    provenance: dict

    @property
    def content_hash(self) -> str:
        return str(self.provenance["content_sha256"])

    @property
    def schedule_2026_available(self) -> bool:
        return bool(self.provenance["schedule_2026_available"])


def _historical_population() -> pd.DataFrame:
    """The certified historical estate, projected onto the population
    contract. ``resolve`` is the repository's single registry entry point --
    no relative-path literal, no fallback dataset."""
    frame = pd.read_parquet(resolve("backfill.games"))
    missing = sorted(set(POPULATION_COLUMNS) - set(frame.columns))
    if missing:
        raise GamesPopulationError(
            f"certified historical backfill.games is missing required column(s) {missing}"
        )
    historical = _normalize_population_frame(frame)
    if (historical["season"] >= PRODUCTION_SEASON).any():
        seasons = sorted({int(s) for s in historical.loc[historical["season"] >= PRODUCTION_SEASON, "season"]})
        raise GamesPopulationConflict(
            f"FAIL CLOSED: certified historical backfill.games unexpectedly contains season(s) {seasons}; "
            f"{PRODUCTION_SEASON} rows must come only from canonical BallDontLie evidence"
        )
    return historical


def load_composite_games_population(
    *,
    artifact_root_path: Path | None = None,
) -> CompositePopulation:
    """The single 2026 production games population:

        certified historical ``backfill.games`` (2020-2025)
          + durable canonical BallDontLie 2026 REG/POST games

    Both production preflight and real production execution call THIS
    function, so they provably consume the same population (compare
    ``provenance["content_sha256"]``).

    The historical estate is required; missing it raises. The 2026 half is
    OPTIONAL by design: with no recorded 2026 evidence the composite is the
    historical population alone and ``schedule_2026_available`` is ``False``
    -- which is exactly what production preflight must report on such a host,
    instead of fabricating 2026 rows.
    """
    historical = _historical_population()
    bdl_2026, bdl_manifest = read_durable_2026_population(artifact_root_path)

    overlap = sorted(set(historical["game_id"]) & set(bdl_2026["game_id"]))
    if overlap:
        raise GamesPopulationConflict(
            "FAIL CLOSED: canonical 2026 BallDontLie evidence claims game_id(s) already present in the "
            f"certified historical estate: {overlap[:10]}{'...' if len(overlap) > 10 else ''}"
        )

    evidence_as_of = (bdl_manifest or {}).get("evidence_as_of_utc")
    if len(bdl_2026) and evidence_as_of:
        classified = classify_rows(bdl_2026, evidence_as_of_utc=evidence_as_of)
        excluded = classified.loc[classified["row_status"] == STATUS_STALE_UNRESOLVED, "game_id"].tolist()
        bdl_2026 = _normalize_population_frame(
            classified.loc[classified["row_status"] == STATUS_INCLUDED, list(POPULATION_COLUMNS)]
        )
    else:
        excluded = []

    composite = _normalize_population_frame(
        pd.concat([historical, bdl_2026], ignore_index=True) if len(bdl_2026) else historical
    )
    if composite["game_id"].duplicated().any():
        duplicates = sorted(composite.loc[composite["game_id"].duplicated(), "game_id"].tolist())
        raise GamesPopulationConflict(
            f"FAIL CLOSED: duplicate game_id(s) in the composite games population: {duplicates[:10]}"
        )

    seasons_2026 = composite.loc[composite["season"] == PRODUCTION_SEASON]
    completed_2026 = int((seasons_2026["home_score"].notna() & seasons_2026["away_score"].notna()).sum())

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "content_sha256": population_content_hash(composite),
        "row_count": int(len(composite)),
        "historical_row_count": int(len(historical)),
        "historical_max_season": int(historical["season"].max()) if len(historical) else None,
        "bdl_2026_row_count": int(len(bdl_2026)),
        "bdl_2026_completed_row_count": completed_2026,
        "bdl_2026_stale_unresolved_excluded": sorted(excluded),
        "bdl_2026_evidence_as_of_utc": evidence_as_of,
        "bdl_2026_population_content_sha256": (bdl_manifest or {}).get("content_sha256"),
        "bdl_2026_evidence": list((bdl_manifest or {}).get("evidence", [])),
        "max_season": int(composite["season"].max()) if len(composite) else None,
        "schedule_2026_available": bool(len(seasons_2026) > 0),
        "season_types": sorted({str(s) for s in composite["season_type"]}),
    }
    return CompositePopulation(games=composite, provenance=provenance)
