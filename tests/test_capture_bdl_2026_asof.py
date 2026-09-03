"""Focused offline tests for ``scripts/capture_bdl_2026_asof.py``.

No network, no credential required: a fake ``requests``-shaped session is
injected and the API key is passed explicitly. Every test either exercises a
pure helper or a fully-faked capture run against a tmp data root.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_bdl_2026_asof.py"


def _load():
    spec = importlib.util.spec_from_file_location("capture_bdl_2026_asof_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # let dataclasses resolve annotations under `from __future__`
    spec.loader.exec_module(module)
    return module


cap = _load()

FAKE_KEY = "SECRET-TEST-KEY-do-not-persist"


class _FakeResponse:
    def __init__(self, status_code: int, body: object, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
        self.text = self.content.decode("utf-8")


class _FakeSession:
    """Routes by URL path suffix. ``routes`` maps a path fragment to either a
    ``_FakeResponse`` or a callable(path, params) -> ``_FakeResponse``."""

    def __init__(self, routes: dict, default: object | None = None):
        self.routes = routes
        self.default = default or _FakeResponse(200, {"data": [], "meta": {"next_cursor": None}})
        self.calls: list[tuple[str, list, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params or [], headers or {}))
        for fragment, handler in self.routes.items():
            if fragment in url:
                return handler(url, params) if callable(handler) else handler
        return self.default(url, params) if callable(self.default) else self.default


def _ok_empty(*_args):
    return _FakeResponse(200, {"data": [], "meta": {"next_cursor": None}})


def _base_routes():
    return {
        "/games": _FakeResponse(
            200,
            {
                "data": [
                    {"id": 111, "date": "2026-09-10T00:20:00Z", "week": 1, "season": 2026,
                     "postseason": False, "status_state": "scheduled"},
                    {"id": 222, "date": "2026-09-13T17:00:00Z", "week": 1, "season": 2026,
                     "postseason": False, "status_state": "scheduled"},
                ],
                "meta": {"next_cursor": None},
            },
        ),
        "/roster": _ok_empty,
        "/player_injuries": _ok_empty,
        "/fantasy/projections": _ok_empty,
        "/odds/opening": _ok_empty,
        "/odds/player_props": _ok_empty,
        "/odds": _ok_empty,
    }


def _request(tmp_path, **overrides):
    kwargs = dict(
        season=2026,
        week=1,
        season_type_label="REG",
        requested_horizon="SMOKE",
        nominal_cutoff_utc=datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc),
        data_root=str(tmp_path),
    )
    kwargs.update(overrides)
    return cap.CaptureRequest(**kwargs)


# --- 1. no credential persistence ------------------------------------------
def test_no_credential_written_anywhere(tmp_path):
    session = _FakeSession(_base_routes())
    manifest = cap.run_capture(
        _request(tmp_path), session=session, api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    capture_dir = Path(manifest["capture_dir"])
    for path in capture_dir.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(encoding="utf-8", errors="ignore"), path
    assert FAKE_KEY not in json.dumps(manifest)
    # the key does travel in the Authorization header on the wire, nowhere else
    assert any(call[2].get("Authorization") == FAKE_KEY for call in session.calls)


def test_redact_params_drops_credential_like_keys():
    redacted = cap.redact_params({"season": 2026, "api_key": "x", "Authorization": "y", "token": "z"})
    assert redacted == {"season": 2026}


# --- 2. append-only path behavior ----------------------------------------
def test_existing_capture_dir_fails_closed(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 9, 8, 15, 59, 0, tzinfo=timezone.utc)
    stamp = fixed_now.strftime("%Y%m%dT%H%M%SZ")
    collide = (
        tmp_path / "live-observation-log" / "balldontlie-2026"
        / "season=2026" / "week=01" / "horizon=SMOKE" / f"capture={stamp}"
    )
    collide.mkdir(parents=True)
    with pytest.raises(cap.FailClosedError):
        cap.run_capture(
            _request(tmp_path), session=_FakeSession(_base_routes()), api_key=FAKE_KEY, now_utc=fixed_now,
        )


# --- 3. deterministic SHA-256 -------------------------------------------
def test_sha256_and_manifest_hash_are_deterministic(tmp_path):
    assert cap.sha256_hex(b"abc") == cap.sha256_hex(b"abc")
    manifest = cap.run_capture(
        _request(tmp_path), session=_FakeSession(_base_routes()), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    stored = manifest["manifest_sha256"]
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert cap.sha256_hex(cap.deterministic_json(body).encode("utf-8")) == stored
    # the manifest file on disk carries the same hash
    on_disk = json.loads((Path(manifest["capture_dir"]) / "manifest.json").read_text())
    assert on_disk["manifest_sha256"] == stored


# --- 4. pagination assembly -------------------------------------------
def test_pagination_follows_cursor_and_keeps_every_page(tmp_path):
    pages = {
        None: _FakeResponse(200, {"data": [{"id": 1}], "meta": {"next_cursor": "c2"}}),
        "c2": _FakeResponse(200, {"data": [{"id": 2}], "meta": {"next_cursor": "c3"}}),
        "c3": _FakeResponse(200, {"data": [{"id": 3}], "meta": {"next_cursor": None}}),
    }

    def injuries_router(_url, params):
        cursor = dict(params or []).get("cursor")
        return pages[cursor]

    routes = _base_routes()
    routes["/player_injuries"] = injuries_router
    manifest = cap.run_capture(
        _request(tmp_path), session=_FakeSession(routes), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    capture_dir = Path(manifest["capture_dir"])
    page_files = sorted(p.name for p in capture_dir.glob("injuries.p*.json"))
    assert page_files == ["injuries.p001.json", "injuries.p002.json", "injuries.p003.json"]
    got_ids = set()
    for name in page_files:
        got_ids.update(row["id"] for row in json.loads((capture_dir / name).read_text())["data"])
    assert got_ids == {1, 2, 3}
    assert manifest["required_source_ok"]["injuries"] is True


def test_pagination_stops_on_repeated_cursor(tmp_path):
    loop = _FakeResponse(200, {"data": [{"id": 1}], "meta": {"next_cursor": "same"}})
    routes = _base_routes()
    routes["/player_injuries"] = lambda *_a: loop
    manifest = cap.run_capture(
        _request(tmp_path), session=_FakeSession(routes), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    assert "injuries" in manifest["pagination_notes"]


# --- 5. REG/POST guard ------------------------------------------------
def test_season_type_guard_rejects_preseason():
    assert cap.resolve_season_type("REG") == ("REG", 2)
    assert cap.resolve_season_type("POST") == ("POST", 3)
    with pytest.raises(ValueError):
        cap.resolve_season_type("PRE")
    with pytest.raises(SystemExit):
        cap.build_arg_parser().parse_args(
            ["--season", "2026", "--week", "1", "--season-type", "PRE", "--horizon", "SMOKE"]
        )


# --- 6. manifest COMPLETE vs INCOMPLETE ------------------------------
def test_manifest_complete_when_all_required_sources_ok(tmp_path):
    manifest = cap.run_capture(
        _request(tmp_path), session=_FakeSession(_base_routes()), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    assert manifest["status"] == "COMPLETE"
    assert all(manifest["required_source_ok"].values())


def test_manifest_incomplete_when_a_required_source_fails_and_no_key_leaks(tmp_path):
    routes = _base_routes()
    routes["/player_injuries"] = lambda *_a: _FakeResponse(500, {"error": "boom"})
    session = _FakeSession(routes)
    manifest = cap.run_capture(
        _request(tmp_path), session=session, api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    assert manifest["status"] == "INCOMPLETE"
    assert manifest["required_source_ok"]["injuries"] is False
    injury_reqs = [r for r in manifest["requests"] if r["logical_name"] == "injuries"]
    assert injury_reqs and all(r["error"] == "HTTP 500" for r in injury_reqs)
    assert injury_reqs[0]["attempts"] == 1 + cap.MAX_RETRIES  # retried, not retried forever
    assert FAKE_KEY not in json.dumps(manifest)


def test_incomplete_when_no_games_skips_player_props(tmp_path):
    routes = _base_routes()
    routes["/games"] = _FakeResponse(200, {"data": [], "meta": {"next_cursor": None}})
    manifest = cap.run_capture(
        _request(tmp_path), session=_FakeSession(routes), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc),
    )
    assert manifest["status"] == "INCOMPLETE"
    props = [r for r in manifest["requests"] if r["logical_name"] == "player_props"]
    assert props and props[0]["error"] == "SKIPPED_NO_TARGET_GAMES"
    # season/week odds do NOT depend on the games list and still run
    for name in ("odds_current", "odds_opening"):
        rec = [r for r in manifest["requests"] if r["logical_name"] == name]
        assert rec and rec[0]["error"] is None


# --- 7. source timestamp eligibility <= cutoff ----------------------
def test_eligibility_counts_split_on_cutoff():
    cutoff = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    rows = [
        {"updated_at": "2026-09-08T15:59:00Z"},   # before
        {"updated_at": "2026-09-08T16:00:00Z"},   # exactly at cutoff -> counts as <=
        {"updated_at": "2026-09-08T18:00:00Z"},   # after
        {"note": "no timestamp here"},             # missing
        {"updated_at": "not-a-date"},              # unparseable -> missing
    ]
    summary = cap.eligibility_counts(rows, ("updated_at",), cutoff)
    assert summary["at_or_before_cutoff"] == 2
    assert summary["after_cutoff"] == 1
    assert summary["missing_or_unparseable_timestamp"] == 2
    assert summary["record_count"] == 5


# --- 8. TUE/FRI timezone + DST conversion --------------------------
def test_tue_fri_cutoff_dst_aware():
    # July: EDT (UTC-4) -> 12:00 local == 16:00Z
    jul_tue = cap.nominal_cutoff_from_card_monday("2026-07-06", "TUE")
    assert (jul_tue.year, jul_tue.month, jul_tue.day, jul_tue.hour) == (2026, 7, 7, 16)
    jul_fri = cap.nominal_cutoff_from_card_monday("2026-07-06", "FRI")
    assert (jul_fri.day, jul_fri.hour) == (10, 16)
    # January: EST (UTC-5) -> 12:00 local == 17:00Z
    jan_tue = cap.nominal_cutoff_from_card_monday("2026-01-05", "TUE")
    assert (jan_tue.month, jan_tue.day, jan_tue.hour) == (1, 6, 17)
    jan_fri = cap.nominal_cutoff_from_card_monday("2026-01-05", "FRI")
    assert (jan_fri.day, jan_fri.hour) == (9, 17)
    # naive --nominal-cutoff string is read in Eastern, not UTC
    assert cap.parse_nominal_cutoff("2026-07-07T12:00:00") == jul_tue


# --- 9. off-window run cannot masquerade as production TUE/FRI ------
def test_off_window_tue_is_refused_without_override():
    cutoff = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    far_after = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(cap.OffWindowError):
        cap.resolve_effective_horizon("TUE", far_after, cutoff, allow_off_window=False)


def test_off_window_tue_with_override_is_downgraded_to_smoke(tmp_path):
    cutoff = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    far_after = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    effective, reason = cap.resolve_effective_horizon("TUE", far_after, cutoff, allow_off_window=True)
    assert effective == "SMOKE" and reason and "forced to SMOKE" in reason

    manifest = cap.run_capture(
        _request(tmp_path, requested_horizon="TUE", nominal_cutoff_utc=cutoff, allow_off_window_smoke=True),
        session=_FakeSession(_base_routes()), api_key=FAKE_KEY, now_utc=far_after,
    )
    assert manifest["horizon"] == "SMOKE"
    assert manifest["requested_horizon"] == "TUE"
    assert "horizon=SMOKE" in manifest["capture_dir"]
    assert "horizon=TUE" not in manifest["capture_dir"]
    assert "TUE" not in Path(manifest["capture_dir"]).name


def test_in_window_tue_is_kept():
    cutoff = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    just_before = datetime(2026, 9, 8, 15, 59, tzinfo=timezone.utc)
    effective, reason = cap.resolve_effective_horizon("TUE", just_before, cutoff, allow_off_window=False)
    assert effective == "TUE" and reason is None


# --- C1 durable append-only evidence rule --------------------------------
def _dir_fingerprint(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_incomplete_capture_is_durable_and_next_run_makes_a_new_directory(tmp_path):
    """An INCOMPLETE capture must survive verbatim; a later run creates a
    different capture directory and never deletes/overwrites the earlier one."""
    routes = _base_routes()
    routes["/player_injuries"] = lambda *_a: _FakeResponse(500, {"error": "boom"})  # force INCOMPLETE

    first = cap.run_capture(
        _request(tmp_path), session=_FakeSession(routes), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 15, 59, 0, tzinfo=timezone.utc),
    )
    first_dir = Path(first["capture_dir"])
    assert first["status"] == "INCOMPLETE"
    assert first_dir.is_dir()
    before = _dir_fingerprint(first_dir)
    assert (first_dir / "manifest.json").exists() and len(before) > 1

    # A second run, later clock -> a distinct capture=<timestamp> directory.
    second = cap.run_capture(
        _request(tmp_path), session=_FakeSession(_base_routes()), api_key=FAKE_KEY,
        now_utc=datetime(2026, 9, 8, 16, 0, 30, tzinfo=timezone.utc),
    )
    second_dir = Path(second["capture_dir"])
    assert second["status"] == "COMPLETE"
    assert second_dir != first_dir

    # The INCOMPLETE evidence is untouched: same directory, same files, same bytes.
    assert first_dir.is_dir()
    assert _dir_fingerprint(first_dir) == before
    assert json.loads((first_dir / "manifest.json").read_text())["status"] == "INCOMPLETE"
    # Both captures coexist under the same week/horizon parent.
    assert first_dir.parent == second_dir.parent
    assert {first_dir.name, second_dir.name} <= {p.name for p in first_dir.parent.iterdir()}

    # Enforcement backstop for requirement 1: the production module imports no
    # deletion/temp module and makes no delete-capable call (AST, so the
    # docstring/comment prose that merely *names* the rule is not matched).
    import ast

    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    banned_modules = {"shutil", "tempfile"}
    banned_call_attrs = {"unlink", "rmtree", "rmdir", "remove", "removedirs", "mkdtemp", "TemporaryDirectory"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not (banned_modules & {a.name.split(".")[0] for a in node.names})
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_modules
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            assert name not in banned_call_attrs, f"delete-capable call {name!r} in capture script"
