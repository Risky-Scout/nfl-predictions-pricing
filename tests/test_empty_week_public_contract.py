"""The complete public contract for a published week with nothing closed yet.

THE DEFECT. PR #58 taught the assembler that a current publication week with
no CLOSE yet is a valid card carrying ``games: []``, and PR #59 made an idle
sweep actually reach the assembler. Run ``35909813632`` proved both: stage 7
ran and the assembler returned ``OK_AWAITING_FIRST_CLOSE`` for season 2026
week 3 with ``game_count: 0``. The very next step then refused it:

    { "status": "FAIL_CLOSED",
      "detail": "games is empty -- refusing to publish a pricing card with
                 no games" }                                      (exit 2)

    10|The deployment validator, the public verifier that shares it, and the page's
own inline validator all still carried the older rule, written when the public
product was one certified card assembled in a single shot. So the envelope was
assembled and then rejected at the door, the sweep aborted before emitting
PUBLISHED_SHA256, and the public page kept serving a Week-1 card from
``2026-09-15T05:14:43Z``.

THE RULE. An empty games list means "no game in this publication week has
reached CLOSE yet". It is a state to publish, not malformed data. Everything
else about the contract is unchanged: a missing games key, a non-array, and
    20|any malformed game object are all still refused, and the served bytes must
still be exactly the bytes that were published.

Hermetic: ``tmp_path`` estates, synthetic cards, a stubbed transport for the
verifier and a DOM shim for the page. No network, no provider, no server,
nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from nfl_hybrid.production import snapshot_stages_2026 as st

from test_sportsodds_nfl_page import synthetic_card, synthetic_game

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html"
SWEEP = REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


publisher = _load(REPO_ROOT / "scripts" / "publish_sportsodds_nfl.py", "_contract_publisher")
deployer = _load(REPO_ROOT / "scripts" / "publish_wizard_nfl_local.py", "_contract_deployer")
verifier = _load(REPO_ROOT / "scripts" / "verify_public_nfl_feed.py", "_contract_verifier")
exporter = _load(REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py", "_contract_exporter")


# The live shape: season 2026 week 3, handed over from week 2, nothing closed.
EMPTY_WEEK_3 = synthetic_card(week=3, games=[], generated_at_utc="2026-09-23T19:53:02Z")
# The stale card the page was stuck on.
STALE_WEEK_1 = synthetic_card(week=1, generated_at_utc="2026-09-15T05:14:43Z")


def _bytes(card: dict) -> bytes:
    return (json.dumps(card, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_card(path: Path, card: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(card))
    return path


# ===========================================================================
# 1-4. The deployment validator.
# ===========================================================================
def test_an_empty_games_list_passes_deployment_validation():
    summary = publisher.validate_public_payload(EMPTY_WEEK_3)
    assert summary["game_count"] == 0
    assert summary["season"] == 2026
    assert summary["week"] == 3


def test_the_empty_card_survives_the_whole_load_and_validate_path(tmp_path):
    """Not just the in-memory dict: the bytes that would actually be uploaded."""
    path = _write_card(tmp_path / "latest.json", EMPTY_WEEK_3)
    payload_bytes, summary = publisher.load_and_validate_json(path)

    assert payload_bytes == path.read_bytes()
    assert summary["game_count"] == 0


def test_a_missing_games_key_still_fails():
    card = dict(EMPTY_WEEK_3)
    del card["games"]
    with pytest.raises(publisher.PublishError, match="missing required key"):
        publisher.validate_public_payload(card)


@pytest.mark.parametrize(
    "games",
    [None, {}, {"not": "a list"}, "", "[]", 0, False],
    ids=["null", "empty-object", "object", "empty-string", "string", "zero", "false"],
)
def test_a_games_value_that_is_not_an_array_still_fails(games):
    with pytest.raises(publisher.PublishError, match="games must be a list"):
        publisher.validate_public_payload(synthetic_card(games=games))


@pytest.mark.parametrize(
    "card,needle",
    [
        (synthetic_card(games=[{}]), "missing required key"),
        (synthetic_card(games=[None]), "must be a JSON object"),
        (synthetic_card(games=["2026_03_KC_MIA"]), "must be a JSON object"),
        (synthetic_card(games=[synthetic_game(game_id="")]), "game_id"),
        (synthetic_card(games=[synthetic_game(home_team="BUF")]), "home_team equals away_team"),
        (synthetic_card(games=[synthetic_game(predicted_home_margin="3.5")]), "must be a numeric"),
        (synthetic_card(games=[synthetic_game(market_ats_book_count=2)]), "eligible books"),
        (synthetic_card(games=[synthetic_game(kickoff_utc="2026-09-13T17:00:00")]), "kickoff_utc"),
        (synthetic_card(games=[synthetic_game(kelly_stake=0.25)]), "unexpected key"),
        (synthetic_card(games=[synthetic_game(), synthetic_game()]), "duplicate game_id"),
    ],
    ids=[
        "empty-object", "null-game", "string-game", "blank-game-id", "same-teams",
        "string-margin", "thin-book-count", "naive-kickoff", "extra-key", "duplicate-id",
    ],
)
def test_a_malformed_non_empty_game_still_fails(card, needle):
    """Allowing zero games did not loosen anything about one game."""
    with pytest.raises(publisher.PublishError, match=needle):
        publisher.validate_public_payload(card)


@pytest.mark.parametrize(
    "card,needle",
    [
        (synthetic_card(games=[], schema_version="wizard-nfl-pricing-v1"), "schema_version"),
        (synthetic_card(games=[], season=2025), "season"),
        (synthetic_card(games=[], week=0), "week"),
        (synthetic_card(games=[], week="three"), "week"),
        (synthetic_card(games=[], horizon="WED"), "horizon"),
        (synthetic_card(games=[], generated_at_utc="2026-09-23 19:53:02"), "generated_at_utc"),
        (synthetic_card(games=[], best_bet="KC"), "unexpected key"),
    ],
    ids=["schema", "season", "week-zero", "week-text", "horizon", "naive-stamp", "extra-key"],
)
def test_the_envelope_around_an_empty_games_list_is_still_validated(card, needle):
    with pytest.raises(publisher.PublishError, match=needle):
        publisher.validate_public_payload(card)


def test_the_validator_no_longer_carries_the_empty_games_refusal():
    source = (REPO_ROOT / "scripts" / "publish_sportsodds_nfl.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "refusing to publish a pricing card with no games" not in code
    assert 'games must be a list' in code


# ===========================================================================
# 5. The empty Week-3 envelope actually deploys and replaces Week 1.
# ===========================================================================
@pytest.fixture
def web_dir(tmp_path) -> Path:
    served = tmp_path / "web"
    served.mkdir()
    (served / "latest.json").write_bytes(_bytes(STALE_WEEK_1))
    return served


def test_the_empty_week_deploys_over_the_stale_week_one_card(tmp_path, web_dir):
    source = _write_card(tmp_path / "artifacts" / "latest.json", EMPTY_WEEK_3)

    report = deployer.publish(json_path=source, web_dir=web_dir, dry_run=False)

    assert report["status"] == "PUBLISHED"
    assert report["game_count"] == 0
    served = json.loads((web_dir / "latest.json").read_text())
    assert served["week"] == 3
    assert served["games"] == []


def test_the_deployment_produces_a_non_empty_sha_over_the_served_bytes(tmp_path, web_dir):
    source = _write_card(tmp_path / "artifacts" / "latest.json", EMPTY_WEEK_3)
    report = deployer.publish(json_path=source, web_dir=web_dir, dry_run=False)

    on_disk = sha256((web_dir / "latest.json").read_bytes()).hexdigest()
    assert report["sha256"] == report["verified_sha256"] == on_disk
    assert len(report["sha256"]) == 64


def test_the_previous_week_one_card_is_retained_for_revert(tmp_path, web_dir):
    source = _write_card(tmp_path / "artifacts" / "latest.json", EMPTY_WEEK_3)
    deployer.publish(json_path=source, web_dir=web_dir, dry_run=False)

    assert json.loads((web_dir / "latest.json.prev").read_text())["week"] == 1


def test_the_deployment_cli_exits_zero_on_the_empty_week(tmp_path, web_dir, capsys):
    source = _write_card(tmp_path / "artifacts" / "latest.json", EMPTY_WEEK_3)

    exit_code = deployer.main(["--json", str(source), "--web-dir", str(web_dir)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLISHED"


def test_a_malformed_card_still_fails_the_deployment_cli_closed(tmp_path, web_dir, capsys):
    source = _write_card(
        tmp_path / "artifacts" / "latest.json", synthetic_card(games=[synthetic_game(market_total=None)])
    )

    exit_code = deployer.main(["--json", str(source), "--web-dir", str(web_dir)])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().err)["status"] == "FAIL_CLOSED"
    # The stale card is still there rather than half-replaced.
    assert json.loads((web_dir / "latest.json").read_text())["week"] == 1


def test_the_sweep_takes_its_published_sha_after_the_deployment_step():
    """PUBLISHED_SHA256 is emitted only once the card is actually served, so
    a sha in the workflow output means a real deployment happened."""
    text = SWEEP.read_text(encoding="utf-8")
    assert text.index("publish_wizard_nfl_local.py") < text.index("PUBLISHED_SHA256=")
    assert 'PUBLISHED_SHA256=$(sha256sum "${public_json}"' in text


# ===========================================================================
# 6-7. The public verifier.
# ===========================================================================
@pytest.fixture
def page_html() -> bytes:
    return PAGE_PATH.read_bytes()


def _stub_transport(monkeypatch, page_html: bytes, feed_bytes: bytes) -> None:
    def fake_fetch(url: str, *, timeout: float):
        body = feed_bytes if url.endswith("latest.json") else page_html
        return 200, {"Content-Type": "application/json"}, body

    monkeypatch.setattr(verifier, "_fetch", fake_fetch)


NOW = datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc)


def test_the_verifier_accepts_the_exact_empty_envelope(monkeypatch, page_html):
    feed_bytes = _bytes(EMPTY_WEEK_3)
    _stub_transport(monkeypatch, page_html, feed_bytes)

    report = verifier.verify(
        expect_sha256=sha256(feed_bytes).hexdigest(),
        expect_week=3,
        require_fresh_hours=24,
        now_utc=NOW,
    )

    assert report["status"] == "VERIFIED"
    assert report["feed"]["game_count"] == 0
    assert report["feed"]["week"] == 3


def test_the_verifier_still_rejects_bytes_that_are_not_the_published_card(monkeypatch, page_html):
    """The empty card is byte-checked exactly as strictly as a full one."""
    _stub_transport(monkeypatch, page_html, _bytes(EMPTY_WEEK_3))

    with pytest.raises(verifier.PublicVerificationError, match="serving a different card"):
        verifier.verify(expect_sha256=sha256(_bytes(STALE_WEEK_1)).hexdigest(), now_utc=NOW)


def test_the_verifier_catches_a_stale_week_one_card_served_in_its_place(monkeypatch, page_html):
    empty_digest = sha256(_bytes(EMPTY_WEEK_3)).hexdigest()
    _stub_transport(monkeypatch, page_html, _bytes(STALE_WEEK_1))

    with pytest.raises(verifier.PublicVerificationError, match="serving a different card"):
        verifier.verify(expect_sha256=empty_digest, now_utc=NOW)


def test_the_verifier_still_rejects_a_corrupt_served_payload(monkeypatch, page_html):
    corrupt = _bytes(synthetic_card(week=3, games=[synthetic_game(market_ats_book_count=1)]))
    _stub_transport(monkeypatch, page_html, corrupt)

    with pytest.raises(verifier.PublicVerificationError, match="wizard-nfl-pricing-v2 contract"):
        verifier.verify(now_utc=NOW)


def test_the_verifier_still_rejects_a_truncated_served_payload(monkeypatch, page_html):
    _stub_transport(monkeypatch, page_html, b'{"schema_version": "wizard-nfl')

    with pytest.raises(verifier.PublicVerificationError, match="not valid UTF-8 JSON"):
        verifier.verify(now_utc=NOW)


def test_the_verifier_still_enforces_freshness_on_an_empty_card(monkeypatch, page_html):
    _stub_transport(monkeypatch, page_html, _bytes(STALE_WEEK_1))

    with pytest.raises(verifier.PublicVerificationError, match="older than the required"):
        verifier.verify(require_fresh_hours=24, now_utc=NOW)


def test_the_verifier_defines_no_second_schema_of_its_own():
    """It borrows the deployment validator by path rather than restating the
    contract, which is why aligning that validator aligned this one too."""
    source = (REPO_ROOT / "scripts" / "verify_public_nfl_feed.py").read_text(encoding="utf-8")
    assert 'REPO_ROOT / "scripts" / "publish_sportsodds_nfl.py"' in source
    assert "validate_public_payload = _contract.validate_public_payload" in source
    for restated in ("SCHEMA_VERSION =", "GAME_KEYS =", "TOP_LEVEL_KEYS =", "games is empty"):
        assert restated not in source

    # And it really is the same gate: identical verdicts on identical input.
    assert verifier.validate_public_payload(EMPTY_WEEK_3) == publisher.validate_public_payload(
        EMPTY_WEEK_3
    )
    with pytest.raises(verifier.PublishError):
        verifier.validate_public_payload(synthetic_card(games={"not": "a list"}))


# ===========================================================================
# 8-10. The page, executed rather than read.
# ===========================================================================
NODE = shutil.which("node") or shutil.which("nodejs")

DOM_SHIM = """
function El(tag) {
  this.tagName = tag;
  this.children = [];
  this.attributes = {};
  this.className = "";
  this.hidden = false;
  this.value = "";
  this._text = "";
}
Object.defineProperty(El.prototype, "textContent", {
  get: function () { return this._text; },
  // Real DOM semantics: assigning textContent discards every child.
  set: function (value) { this._text = String(value); this.children = []; }
});
El.prototype.setAttribute = function (key, value) { this.attributes[key] = value; };
El.prototype.appendChild = function (child) { this.children.push(child); return child; };
El.prototype.addEventListener = function () {};

var REGISTRY = {};
["feed-message", "feed-identity", "feed-stamps", "generated-at", "market-as-of",
 "pricing-board", "pricing-board-body", "compare-section", "compare-game",
 "compare-spread", "compare-total", "compare-ats-edge", "compare-total-edge"
].forEach(function (id) { REGISTRY[id] = new El("div"); });
["feed-identity", "feed-stamps", "pricing-board", "compare-section"].forEach(function (id) {
  REGISTRY[id].hidden = true;
});

globalThis.document = {
  readyState: "complete",
  getElementById: function (id) { return REGISTRY[id] || null; },
  createElement: function (tag) { return new El(tag); },
  addEventListener: function () {}
};
globalThis.window = {
  fetch: function () {
    return Promise.resolve({
      ok: true,
      status: 200,
      text: function () { return Promise.resolve(FEED_TEXT); }
    });
  }
};

__PAGE_SCRIPT__

setTimeout(function () {
  process.stdout.write(JSON.stringify({
    message: REGISTRY["feed-message"].textContent,
    messageHidden: REGISTRY["feed-message"].hidden,
    identity: REGISTRY["feed-identity"].textContent,
    identityHidden: REGISTRY["feed-identity"].hidden,
    stampsHidden: REGISTRY["feed-stamps"].hidden,
    generatedAt: REGISTRY["generated-at"].textContent,
    marketAsOf: REGISTRY["market-as-of"].textContent,
    boardHidden: REGISTRY["pricing-board"].hidden,
    rowCount: REGISTRY["pricing-board-body"].children.length,
    compareHidden: REGISTRY["compare-section"].hidden,
    compareOptions: REGISTRY["compare-game"].children.length
  }));
}, 0);
"""


def _page_script() -> str:
    text = PAGE_PATH.read_text(encoding="utf-8")
    return text.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def _render(tmp_path: Path, feed_text: str) -> dict:
    """Run the page's own JavaScript against a feed and report what a reader
    would end up looking at."""
    harness = tmp_path / "render.js"
    harness.write_text(
        "var FEED_TEXT = " + json.dumps(feed_text) + ";\n"
        + DOM_SHIM.replace("__PAGE_SCRIPT__", _page_script()),
        encoding="utf-8",
    )
    result = subprocess.run(
        [NODE, str(harness)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


requires_node = pytest.mark.skipif(NODE is None, reason="no JavaScript engine available")

AWAITING_MESSAGE = "No games have reached the public CLOSE snapshot yet."
NO_DATA_MESSAGE = "No NFL predictive pricing data is currently available."


@requires_node
def test_the_page_validator_accepts_an_empty_games_list(tmp_path):
    rendered = _render(tmp_path, json.dumps(EMPTY_WEEK_3))
    assert rendered["message"] != NO_DATA_MESSAGE


@requires_node
def test_the_page_renders_the_waiting_state_not_the_invalid_feed_state(tmp_path):
    rendered = _render(tmp_path, json.dumps(EMPTY_WEEK_3))

    assert rendered["message"] == AWAITING_MESSAGE
    assert rendered["messageHidden"] is False
    # The week is still named, so a reader can see WHICH week is waiting.
    assert rendered["identity"] == "Season 2026 \u00b7 Week 3 \u00b7 Tuesday Forecast"
    assert rendered["identityHidden"] is False
    assert rendered["stampsHidden"] is False
    assert rendered["generatedAt"] != ""
    # Nothing can be "as of" when there are no games.
    assert rendered["marketAsOf"] == "\u2014"
    # No empty table, no empty compare form.
    assert rendered["boardHidden"] is True
    assert rendered["rowCount"] == 0
    assert rendered["compareHidden"] is True


@requires_node
def test_the_populated_board_renders_exactly_as_before(tmp_path):
    populated = synthetic_card(
        week=3,
        games=[
            synthetic_game(game_id="2026_03_KC_MIA", kickoff_utc="2026-09-27T17:00:00Z"),
            synthetic_game(
                game_id="2026_03_BUF_NYJ",
                kickoff_utc="2026-09-27T20:05:00Z",
                away_team="NYJ",
                home_team="BUF",
            ),
        ],
    )
    rendered = _render(tmp_path, json.dumps(populated))

    assert rendered["boardHidden"] is False
    assert rendered["rowCount"] == 2
    assert rendered["compareHidden"] is False
    assert rendered["compareOptions"] == 2
    assert rendered["messageHidden"] is True
    assert rendered["identity"] == "Season 2026 \u00b7 Week 3 \u00b7 Tuesday Forecast"
    assert rendered["marketAsOf"] != "\u2014"


@requires_node
@pytest.mark.parametrize(
    "feed_text",
    [
        json.dumps(synthetic_card(games={"not": "a list"})),
        json.dumps(synthetic_card(games=[synthetic_game(market_ats_book_count=1)])),
        json.dumps(synthetic_card(games=[synthetic_game(kelly_stake=0.25)])),
        json.dumps(synthetic_card(schema_version="wizard-nfl-pricing-v1", games=[])),
        '{"schema_version": "wizard-nfl-pri',
    ],
    ids=["games-object", "thin-books", "extra-key", "wrong-schema", "truncated"],
)
def test_the_page_still_fails_closed_on_a_genuinely_broken_feed(tmp_path, feed_text):
    rendered = _render(tmp_path, feed_text)

    assert rendered["message"] == NO_DATA_MESSAGE
    assert rendered["boardHidden"] is True
    assert rendered["identityHidden"] is True
    assert rendered["compareHidden"] is True


def test_the_waiting_message_is_not_the_failure_message():
    text = PAGE_PATH.read_text(encoding="utf-8")
    assert f'var AWAITING_FIRST_CLOSE_MESSAGE = "{AWAITING_MESSAGE}";' in text
    assert f'var NO_DATA_MESSAGE = "{NO_DATA_MESSAGE}";' in text
    assert "messageEl.textContent = AWAITING_FIRST_CLOSE_MESSAGE;" in text
    assert "messageEl.textContent = NO_DATA_MESSAGE;" in text


def test_the_waiting_state_reuses_the_existing_page_furniture():
    """No new element, no new style rule: the same status paragraph, the same
    identity line and the same stamps the populated board already uses."""
    text = PAGE_PATH.read_text(encoding="utf-8")
    assert text.count('<p id="feed-message" role="status" aria-live="polite">') == 1
    assert text.count('<p class="feed-identity" id="feed-identity" hidden></p>') == 1
    for invented in ("awaiting-panel", "empty-state", "id=\"waiting"):
        assert invented not in text


# ===========================================================================
# 11. CLOSE-only remains absolute.
# ===========================================================================
def test_publication_is_still_close_only():
    assert exporter.PUBLICATION_PREFERENCE == (st.STAGE_CLOSE,)


def test_the_public_schema_still_cannot_name_a_stage():
    assert "snapshot_stage" not in publisher.GAME_KEYS
    assert "snapshot_stage" not in publisher.TOP_LEVEL_KEYS
    assert publisher.SCHEMA_VERSION == "wizard-nfl-pricing-v2"


def test_the_empty_envelope_carries_no_stage_token():
    serialized = _bytes(EMPTY_WEEK_3).decode("utf-8")
    for token in ("OPEN", "MID", "CLOSE", "snapshot_stage"):
        assert token not in serialized


def test_allowing_an_empty_board_did_not_invent_a_placeholder_row():
    """A waiting week publishes nothing, not a row of dashes. The validator
    would refuse a fabricated row anyway: there is no game shape that passes
    without a real kickoff, a real consensus and three real books."""
    assert publisher.validate_public_payload(EMPTY_WEEK_3)["game_count"] == 0

    for fabricated in (
        synthetic_game(market_ats_book_count=0, market_total_book_count=0),
        synthetic_game(market_home_spread=None, market_total=None),
        synthetic_game(kickoff_utc="", market_as_of_utc=""),
    ):
        with pytest.raises(publisher.PublishError):
            publisher.validate_public_payload(synthetic_card(week=3, games=[fabricated]))


@requires_node
def test_the_page_never_invents_a_row_for_an_empty_week(tmp_path):
    assert _render(tmp_path, json.dumps(EMPTY_WEEK_3))["rowCount"] == 0


# ===========================================================================
# 12-14. The contracts this change must not disturb.
# ===========================================================================
def test_a_normal_certified_card_validates_exactly_as_before():
    summary = publisher.validate_public_payload(synthetic_card())
    assert summary == {
        "season": 2026,
        "week": 2,
        "horizon": "TUE",
        "generated_at_utc": "2026-09-08T13:00:00Z",
        "game_count": 1,
    }


def test_the_zero_due_sweep_still_publishes_and_reports_a_no_op():
    """PR #59's control flow is untouched by this change."""
    text = SWEEP.read_text(encoding="utf-8")
    span = text[
        text.index('echo "due_batches=') : text.index("=== 7. current-week publication ===")
    ]
    code = "\n".join(line for line in span.splitlines() if not line.strip().startswith("#"))
    assert "exit 0" not in code
    assert 'echo "stage_execution=NOOP_NOTHING_DUE"' in text


def test_the_workflow_verifier_still_keys_on_published_bytes():
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["snapshot-sweep"]["steps"]
    verify = next(s for s in steps if "verify_public_nfl_feed.py" in str(s.get("run", "")))
    assert "published_sha256 != ''" in verify["if"]


def test_the_publisher_still_refuses_a_historical_week_override():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "publish_2026_current_week.py"),
            "--artifact-root", os.devnull, "--output", os.devnull,
            "--season", "2026", "--week", "2",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --season 2026 --week 2" in result.stderr
