"""Focused proofs for the SportsOdds NFL Predictive Pricing publication layer.

Two artifacts are under test and nothing else:

    web/sportsodds/nfl/index.html   the self-contained public page
    scripts/publish_sportsodds_nfl.py   the FTP/FTPS publisher

Standard library only, no environment setup, no network. The page is asserted
as source text (there is no JavaScript engine in this repo's test
dependencies, so the derivation formulas are proved by asserting the exact
canonical expressions the page computes them with), and the publisher is
asserted behaviourally against a fake FTP client -- ``ftplib.FTP`` and
``ftplib.FTP_TLS`` are patched out in every publisher test that reaches the
transport, and the dry-run test additionally poisons ``socket.socket`` so that
any attempt to open a connection would fail loudly.

Run:
    python3 -m unittest discover -s tests -p 'test_sportsodds_nfl_page.py' -q
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html"
PUBLISHER_PATH = REPO_ROOT / "scripts" / "publish_sportsodds_nfl.py"

# The whole intended change set for this task.
EXPECTED_CHANGE_SET = frozenset(
    {
        "web/sportsodds/nfl/index.html",
        "scripts/publish_sportsodds_nfl.py",
        "tests/test_sportsodds_nfl_page.py",
    }
)

# Nothing in the model, capture, calibration or exporter surface may move.
PROTECTED_PREFIXES = (
    "src/",
    "config/",
    "data/",
    "docs/",
    "outputs/",
    "reports/",
    "examples/",
    "scripts/export_wizard_nfl_predictions.py",
    "scripts/export_wizard_nfl_pricing.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-data.txt",
)

EXACT_TITLE = "<title>NFL Predictive Pricing | Wizard of Odds</title>"
EXACT_H1 = "<h1>NFL Predictive Pricing</h1>"
EXACT_SUBTITLE = "Model Fair Lines vs. Market Consensus"
EXACT_INTRO = (
    "The NFL Predictive Pricing Model estimates a fair winning margin and game total for each "
    "matchup. This page compares those model prices with the captured market consensus to show "
    "where the model and market differ."
)
EXACT_EDGE_DISCLAIMER = (
    "Sportsbook Edge and Total Edge are differences in points between model pricing and market "
    "pricing. They are not guarantees of profit or winning wagers."
)
EXACT_ERROR_MESSAGE = "No NFL predictive pricing data is currently available."
METHODOLOGY_URL = "https://github.com/Risky-Scout/nfl-predictions-pricing/blob/main/docs/MIKE_START_HERE.md"

PROHIBITED_HYPE = ("LOCK", "BEST BET", "GUARANTEED", "SURE THING", "PROVEN EDGE", "PROFITABLE SYSTEM")

PRIMARY_COLUMNS = (
    "Matchup",
    "Model Fair Line",
    "Market Consensus",
    "Sportsbook Edge (ATS)",
    "Model Total",
    "Market Total",
    "Total Edge",
    "Winner View",
)

TEST_PASSWORD = "n0t-a-real-password-Ynq3"


def _load_publisher():
    """The publisher is a script, not an importable package module, so it is
    loaded by explicit path exactly the way the sibling exporter tests do."""
    spec = importlib.util.spec_from_file_location("_sportsodds_nfl_publisher_under_test", PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = _load_publisher()
PAGE_TEXT = PAGE_PATH.read_text(encoding="utf-8")
PUBLISHER_TEXT = PUBLISHER_PATH.read_text(encoding="utf-8")


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text)


PAGE_SQUEEZED = squeeze(PAGE_TEXT)


# --------------------------------------------------------------------------- #
# synthetic fixtures -- invented for this test module only. They are never
# written anywhere near a production output tree and no real Week 1 forecast is
# read, generated or copied.
# --------------------------------------------------------------------------- #
def synthetic_game(**overrides) -> dict:
    game = {
        "game_id": "2026-TEST-0001",
        "kickoff_utc": "2026-09-13T17:00:00Z",
        "away_team": "BUF",
        "home_team": "HOU",
        "predicted_home_margin": 5.8,
        "predicted_game_total": 46.1,
        "market_home_spread": -3.5,
        "market_total": 44.0,
        "market_as_of_utc": "2026-09-08T12:30:00Z",
        "market_ats_book_count": 6,
        "market_total_book_count": 5,
    }
    game.update(overrides)
    return game


def synthetic_card(**overrides) -> dict:
    card = {
        "schema_version": "wizard-nfl-pricing-v2",
        "season": 2026,
        "week": 2,
        "horizon": "TUE",
        "generated_at_utc": "2026-09-08T13:00:00Z",
        "games": [synthetic_game()],
    }
    card.update(overrides)
    return card


def synthetic_env(**overrides) -> dict:
    env = {
        "SPORTSODDS_FTP_HOST": "ftp.example-test.invalid",
        "SPORTSODDS_FTP_USER": "sportsodds-test",
        "SPORTSODDS_FTP_PASSWORD": TEST_PASSWORD,
        "SPORTSODDS_FTP_REMOTE_DIR": "/tools/odds-scanner/predictions/NFL",
        "SPORTSODDS_FTP_MODE": "ftp",
    }
    env.update(overrides)
    return {key: value for key, value in env.items() if value is not None}


class FakeFtpClient:
    """Records the FTP conversation. Instances are shared through the class
    attribute ``instances`` so a test can inspect what the publisher did."""

    instances: list["FakeFtpClient"] = []
    is_tls = False
    existing_dirs: set[str] = set()
    refuse_rename_over_existing = False

    def __init__(self, *args, timeout=None, **kwargs):
        self.timeout = timeout
        self.calls: list[tuple] = [("__init__", args, kwargs)]
        self.stored: dict[str, bytes] = {}
        self.cwd_path: list[str] = []
        type(self).instances.append(self)

    def _record(self, name, *args):
        self.calls.append((name,) + args)

    def connect(self, host=None, port=None, timeout=None):
        self._record("connect", host, port, timeout)

    def login(self, user=None, passwd=None):
        self._record("login", user, passwd)

    def prot_p(self):
        self._record("prot_p")

    def cwd(self, path):
        self._record("cwd", path)
        if path == "/":
            self.cwd_path = []
            return
        candidate = "/".join(self.cwd_path + [path])
        if candidate not in type(self).existing_dirs:
            raise publisher.ftplib.error_perm(f"550 {path}: No such directory")
        self.cwd_path.append(path)

    def mkd(self, path):
        self._record("mkd", path)
        type(self).existing_dirs.add("/".join(self.cwd_path + [path]))

    def storbinary(self, command, source):
        payload = source.read()
        self._record("storbinary", command, len(payload))
        self.stored[command.split(" ", 1)[1]] = payload

    def rename(self, from_name, to_name):
        self._record("rename", from_name, to_name)
        if type(self).refuse_rename_over_existing and to_name in self.stored:
            raise publisher.ftplib.error_perm(f"550 {to_name}: rename refused")
        self.stored[to_name] = self.stored.pop(from_name)

    def delete(self, name):
        self._record("delete", name)
        self.stored.pop(name, None)

    def quit(self):
        self._record("quit")

    def close(self):
        self._record("close")

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.existing_dirs = set()
        cls.refuse_rename_over_existing = False

    def method_names(self) -> list[str]:
        return [call[0] for call in self.calls]


class FakeFtpTlsClient(FakeFtpClient):
    is_tls = True
    instances: list["FakeFtpClient"] = []

    @classmethod
    def reset(cls):
        cls.instances = []


class PublisherHarness(unittest.TestCase):
    """Shared publisher plumbing: temp files, patched transport, captured
    output. Never touches the real network or the real environment."""

    def setUp(self):
        FakeFtpClient.reset()
        FakeFtpTlsClient.reset()
        self.tmp_dir = Path(
            __import__("tempfile").mkdtemp(prefix="sportsodds-nfl-publisher-test-"),
        )
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp_dir, ignore_errors=True))
        self.stream = io.StringIO()

    def write_json(self, payload, *, raw: str | None = None) -> Path:
        path = self.tmp_dir / "synthetic-latest.json"
        path.write_text(raw if raw is not None else json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def patched_transport(self):
        return mock.patch.multiple(
            publisher.ftplib,
            FTP=FakeFtpClient,
            FTP_TLS=FakeFtpTlsClient,
        )

    def patched_environ(self, **overrides):
        """Synthetic credentials for the CLI paths, which read os.environ."""
        return mock.patch.dict(os.environ, synthetic_env(**overrides), clear=False)

    def publish(self, *, json_path=None, publish_json=True, publish_html=True, dry_run=False, env=None):
        return publisher.publish(
            json_path=json_path,
            html_path=PAGE_PATH,
            publish_json=publish_json,
            publish_html=publish_html,
            dry_run=dry_run,
            env=synthetic_env() if env is None else env,
            stream=self.stream,
        )

    def output(self) -> str:
        return self.stream.getvalue()


# =========================================================================== #
# 1-5, 15-16: page existence, identity, feed access, self-containment
# =========================================================================== #
class PageIdentityTests(unittest.TestCase):
    def test_01_page_exists(self):
        self.assertTrue(PAGE_PATH.is_file(), f"missing public page: {PAGE_PATH}")
        self.assertGreater(len(PAGE_TEXT.strip()), 0)
        self.assertTrue(PAGE_TEXT.lstrip().lower().startswith("<!doctype html>"))

    def test_02_title_is_exact(self):
        self.assertIn(EXACT_TITLE, PAGE_TEXT)
        titles = re.findall(r"<title>(.*?)</title>", PAGE_TEXT, flags=re.DOTALL)
        self.assertEqual(titles, ["NFL Predictive Pricing | Wizard of Odds"])

    def test_03_h1_is_exact(self):
        self.assertIn(EXACT_H1, PAGE_TEXT)
        h1s = re.findall(r"<h1[^>]*>(.*?)</h1>", PAGE_TEXT, flags=re.DOTALL)
        self.assertEqual([squeeze(text).strip() for text in h1s], ["NFL Predictive Pricing"])

    def test_03b_subtitle_and_intro_copy_exact(self):
        self.assertIn(EXACT_SUBTITLE, PAGE_TEXT)
        self.assertIn(EXACT_INTRO, PAGE_SQUEEZED)
        # The edge disclaimer immediately follows the introductory copy.
        self.assertLess(PAGE_SQUEEZED.index(EXACT_INTRO), PAGE_SQUEEZED.index(EXACT_EDGE_DISCLAIMER))

    def test_03c_no_hype_language(self):
        upper = PAGE_TEXT.upper()
        for phrase in PROHIBITED_HYPE:
            # Whole words only, so CSS "display: block" is not mistaken for LOCK.
            pattern = r"\b" + re.escape(phrase) + r"\b"
            self.assertIsNone(re.search(pattern, upper), f"prohibited marketing language present: {phrase}")

    def test_04_page_fetches_same_origin_latest_json(self):
        self.assertIn('var FEED_URL = "./latest.json";', PAGE_TEXT)
        self.assertIn("window.fetch(FEED_URL,", PAGE_SQUEEZED)
        # The same-origin sibling feed is the only resource the page requests:
        # no other quoted path in the page names a data file.
        self.assertEqual(
            re.findall(r"""['"][^'"\s]*latest\.json[^'"\s]*['"]""", PAGE_TEXT),
            ['"./latest.json"'],
        )
        self.assertEqual(len(re.findall(r"window\.fetch\(", PAGE_TEXT)), 1)

    def test_05_feed_request_is_cache_no_store(self):
        self.assertRegex(
            PAGE_SQUEEZED,
            r'window\.fetch\(FEED_URL, \{ cache: "no-store"',
        )

    def test_06_required_schema_version_enforced(self):
        self.assertIn('var REQUIRED_SCHEMA_VERSION = "wizard-nfl-pricing-v2";', PAGE_TEXT)
        self.assertIn("if (payload.schema_version !== REQUIRED_SCHEMA_VERSION)", PAGE_SQUEEZED)

    def test_06b_required_key_sets_declared_exactly(self):
        top_level = re.search(r"var TOP_LEVEL_KEYS = \[(.*?)\];", PAGE_TEXT, flags=re.DOTALL)
        self.assertIsNotNone(top_level)
        self.assertEqual(
            re.findall(r'"([a-z_]+)"', top_level.group(1)),
            ["schema_version", "season", "week", "horizon", "generated_at_utc", "games"],
        )
        game_keys = re.search(r"var GAME_KEYS = \[(.*?)\];", PAGE_TEXT, flags=re.DOTALL)
        self.assertIsNotNone(game_keys)
        self.assertEqual(
            re.findall(r'"([a-z_]+)"', game_keys.group(1)),
            list(publisher.GAME_KEYS),
        )

    def test_15_no_external_frontend_dependencies(self):
        self.assertIsNone(re.search(r"<script[^>]*\bsrc=", PAGE_TEXT), "page loads external JavaScript")
        self.assertIsNone(re.search(r"<link\b", PAGE_TEXT), "page loads an external stylesheet or resource hint")
        self.assertIsNone(re.search(r"<iframe\b|<img\b|<object\b|<embed\b", PAGE_TEXT))
        self.assertNotIn("@import", PAGE_TEXT)
        self.assertNotIn("integrity=", PAGE_TEXT)
        self.assertNotIn("crossorigin", PAGE_TEXT)
        self.assertNotIn("@font-face", PAGE_TEXT)
        # The methodology link is the only absolute URL on the page.
        self.assertEqual(
            sorted(set(re.findall(r"https?://[^\s\"'<>)]+", PAGE_TEXT))),
            [METHODOLOGY_URL],
        )

    def test_16_no_fake_production_data_embedded(self):
        self.assertIsNone(re.search(r'"schema_version"\s*:', PAGE_TEXT), "page embeds a JSON pricing payload")
        self.assertIsNone(re.search(r'"games"\s*:', PAGE_TEXT))
        self.assertNotIn("application/json", PAGE_TEXT)
        self.assertNotIn("<template", PAGE_TEXT)
        # The published page directory carries the page and nothing else: no
        # sample latest.json, no captured production card.
        self.assertEqual([path.name for path in sorted(PAGE_PATH.parent.iterdir())], ["index.html"])


# =========================================================================== #
# 7-11: pricing derivations, board structure, Compare Your Line
# =========================================================================== #
class PageDerivationTests(unittest.TestCase):
    def test_07_model_fair_spread_formula(self):
        self.assertIn("var modelFairHomeSpread = -game.predicted_home_margin;", PAGE_TEXT)
        # Favourite notation: the negative side is named, zero is PK.
        self.assertIn(
            'return (homeSpread < 0 ? homeTeam : awayTeam) + " -" + formatMagnitude(homeSpread);',
            PAGE_TEXT,
        )
        self.assertIn("if (homeSpread === 0) { return PICK_EM_SHORT; }", PAGE_SQUEEZED)
        self.assertIn("modelFairLine: favoriteNotation(modelFairHomeSpread, game.away_team, game.home_team)", PAGE_TEXT)
        self.assertIn(
            "marketConsensus: favoriteNotation(game.market_home_spread, game.away_team, game.home_team)",
            PAGE_TEXT,
        )
        self.assertIn('var PICK_EM_SHORT = "PK";', PAGE_TEXT)

    def test_08_ats_edge_formula_and_sides(self):
        self.assertIn(
            "var atsHomeEdge = game.predicted_home_margin + game.market_home_spread;",
            PAGE_TEXT,
        )
        self.assertIn("var atsSide = favoriteSide(atsHomeEdge);", PAGE_TEXT)
        self.assertIn(
            'atsEdgeText: atsHomeEdge === 0 ? NO_EDGE : sideTeam(atsSide, game.away_team, game.home_team) '
            '+ " +" + formatMagnitude(atsHomeEdge) + " pts",',
            PAGE_SQUEEZED,
        )
        self.assertIn('var NO_EDGE = "NO EDGE";', PAGE_TEXT)
        # HOME when positive, AWAY when negative, neither at zero.
        self.assertIn(
            'function favoriteSide(impliedHomeMargin) { if (impliedHomeMargin > 0) { return "HOME"; } '
            'if (impliedHomeMargin < 0) { return "AWAY"; } return PICK_EM; }',
            PAGE_SQUEEZED,
        )

    def test_09_total_edge_formula_and_sides(self):
        self.assertIn("var totalEdge = game.predicted_game_total - game.market_total;", PAGE_TEXT)
        self.assertIn(
            'totalEdgeText: totalEdge === 0 ? NO_EDGE : (totalEdge > 0 ? "OVER" : "UNDER") '
            '+ " +" + formatMagnitude(totalEdge) + " pts",',
            PAGE_SQUEEZED,
        )

    def test_10_winner_agreement_logic(self):
        self.assertIn("var marketImpliedHomeMargin = -game.market_home_spread;", PAGE_TEXT)
        self.assertIn("var modelFavorite = favoriteSide(game.predicted_home_margin);", PAGE_TEXT)
        self.assertIn("var marketFavorite = favoriteSide(marketImpliedHomeMargin);", PAGE_TEXT)
        self.assertIn(
            'winnerAgreement: modelFavorite === marketFavorite ? "AGREE" : "DISAGREE",',
            PAGE_TEXT,
        )
        self.assertIn('appendText(winnerView, "Model: " + priced.modelFavoriteTeam, "note");', PAGE_TEXT)
        self.assertIn('appendText(winnerView, "Market: " + priced.marketFavoriteTeam, "note");', PAGE_TEXT)
        # Never inferred from a price the feed does not carry.
        self.assertNotIn("moneyline", PAGE_TEXT.lower())

    def test_10b_primary_columns_in_exact_order(self):
        head = re.search(r"<thead>(.*?)</thead>", PAGE_TEXT, flags=re.DOTALL)
        self.assertIsNotNone(head)
        headers = re.findall(r'<th scope="col">(.*?)</th>', head.group(1), flags=re.DOTALL)
        self.assertEqual([squeeze(text).strip().replace("&amp;", "&") for text in headers], list(PRIMARY_COLUMNS))
        # Every primary group is re-labelled for the mobile card layout.
        labels = set(re.findall(r'cell\("t[hd]", "(.*?)"\)', PAGE_TEXT))
        self.assertEqual(labels, set(PRIMARY_COLUMNS))
        self.assertIn('content: attr(data-label);', PAGE_TEXT)

    def test_10c_sort_is_kickoff_then_game_id(self):
        self.assertIn("games.sort(function (left, right) {", PAGE_TEXT)
        self.assertIn("if (left.kickoffEpochMs !== right.kickoffEpochMs) { return left.kickoffEpochMs - right.kickoffEpochMs; }", PAGE_SQUEEZED)
        self.assertIn("if (left.game_id < right.game_id) { return -1; }", PAGE_SQUEEZED)

    def test_10d_official_times_render_in_eastern_time(self):
        self.assertIn('var ET_TIME_ZONE = "America/New_York";', PAGE_TEXT)
        self.assertIn("var settings = { timeZone: ET_TIME_ZONE };", PAGE_TEXT)
        self.assertIn('new Intl.DateTimeFormat("en-US", settings)', PAGE_TEXT)
        self.assertIn(
            'identityEl.textContent = "Season " + card.season + " \\u00b7 Week " + card.week '
            '+ " \\u00b7 " + ALLOWED_HORIZONS[card.horizon] + " Forecast";',
            PAGE_TEXT,
        )
        self.assertIn('var ALLOWED_HORIZONS = { TUE: "Tuesday", FRI: "Friday" };', PAGE_TEXT)
        self.assertIn('appendText(marketLine, "ATS Books: " + game.market_ats_book_count, "note");', PAGE_TEXT)
        self.assertIn('appendText(marketLine, "Total Books: " + game.market_total_book_count, "note");', PAGE_TEXT)

    def test_11_compare_your_line_section(self):
        self.assertIn("Compare Your Line", PAGE_TEXT)
        self.assertIn('<label for="compare-game">Game</label>', PAGE_TEXT)
        self.assertIn('<label for="compare-spread">Your Home Spread</label>', PAGE_TEXT)
        self.assertIn('<label for="compare-total">Your Game Total</label>', PAGE_TEXT)
        self.assertIn('<select id="compare-game" name="compare-game">', PAGE_TEXT)
        self.assertIn(
            "Enter the HOME team's spread exactly as shown at your sportsbook. Example: -3.5 if the "
            "home team is -3.5; +3.5 if the home team is +3.5.",
            PAGE_SQUEEZED,
        )
        self.assertIn("var userAtsHomeEdge = game.predicted_home_margin + userHomeSpread;", PAGE_TEXT)
        self.assertIn("var userTotalEdge = game.predicted_game_total - userTotal;", PAGE_TEXT)
        self.assertIn("Your ATS Edge", PAGE_TEXT)
        self.assertIn("Your Total Edge", PAGE_TEXT)
        # Nothing is stored or transmitted: no storage APIs, no beacons, no
        # form submission, no analytics.
        for forbidden in ("localStorage", "sessionStorage", "document.cookie", "indexedDB", "sendBeacon",
                          "XMLHttpRequest", "gtag(", "dataLayer", "form.submit", "action="):
            self.assertNotIn(forbidden, PAGE_TEXT, f"Compare Your Line must not use {forbidden}")
        self.assertEqual(len(re.findall(r"window\.fetch\(", PAGE_TEXT)), 1)

    def test_12_sportsbook_edge_disclaimer_present(self):
        self.assertIn(EXACT_EDGE_DISCLAIMER, PAGE_SQUEEZED)
        self.assertIn(
            "Lines move. Compare the model with the price currently available at your sportsbook "
            "before making a decision.",
            PAGE_SQUEEZED,
        )
        self.assertIn(
            "Forecasts are created before results are known and archived by forecast horizon for "
            "prospective evaluation.",
            PAGE_SQUEEZED,
        )
        for glossary_entry in (
            "The point spread implied by the model's predicted scoring margin.",
            "The certified consensus spread reconstructed from eligible market observations available "
            "at the forecast horizon.",
            "The point difference between the model's fair margin and the market-implied margin.",
            "The model's predicted combined final score.",
            "The difference between the model total and market total.",
            "Whether the model and market favor the same team outright.",
        ):
            self.assertIn(glossary_entry, PAGE_SQUEEZED)

    def test_13_methodology_link_exact(self):
        self.assertIn(
            f'<a href="{METHODOLOGY_URL}" target="_blank" rel="noopener noreferrer">',
            PAGE_TEXT,
        )
        self.assertIn("Model Methodology &amp; Review", PAGE_TEXT)

    def test_14_error_message_exact_and_only(self):
        self.assertIn(f'var NO_DATA_MESSAGE = "{EXACT_ERROR_MESSAGE}";', PAGE_TEXT)
        self.assertIn("messageEl.textContent = NO_DATA_MESSAGE;", PAGE_TEXT)
        self.assertIn('["catch"](function () { failClosed(); });', PAGE_SQUEEZED)
        # Fail-closed clears the board rather than leaving stale or partial rows.
        self.assertIn('boardBodyEl.textContent = "";', PAGE_TEXT)
        self.assertIn("compareSectionEl.hidden = true;", PAGE_TEXT)
        # No diagnostics are surfaced to the reader.
        self.assertNotIn("console.log", PAGE_TEXT)
        self.assertNotIn(".stack", PAGE_TEXT)
        for guard in (
            "if (!response.ok)",
            'invalid("games is empty")',
            'invalid("games is not an array")',
            "requireExactKeys(payload, TOP_LEVEL_KEYS,",
            "function requireFiniteNumber(value, label)",
            "function requireInstant(value, label)",
        ):
            self.assertIn(guard, squeeze(PAGE_TEXT) if " " in guard else PAGE_TEXT)


# =========================================================================== #
# 17-18, 33: publisher secrets discipline and required configuration
# =========================================================================== #
class PublisherConfigTests(PublisherHarness):
    def test_17_publisher_contains_no_credentials(self):
        self.assertNotIn("password=\"", PUBLISHER_TEXT)
        for suspicious in ("passwd=\"", "ftp://", "sftp://", "wizardofodds.com\"", "/var/www"):
            self.assertNotIn(suspicious, PUBLISHER_TEXT)
        # Credential names appear only as environment variable names.
        assignments = re.findall(r"^\s*(?:PASSWORD|USER|HOST|SECRET|TOKEN)\s*=\s*[\"'].+", PUBLISHER_TEXT, re.MULTILINE)
        self.assertEqual(assignments, [])
        self.assertEqual(
            publisher.REQUIRED_ENV_VARS,
            (
                "SPORTSODDS_FTP_HOST",
                "SPORTSODDS_FTP_USER",
                "SPORTSODDS_FTP_PASSWORD",
                "SPORTSODDS_FTP_REMOTE_DIR",
                "SPORTSODDS_FTP_MODE",
            ),
        )
        self.assertEqual(publisher.DEFAULT_FTP_PORT, 21)
        # No credential, and no published byte, is ever written to disk here.
        self.assertIsNone(re.search(r"\.write_text\(|\.write_bytes\(|open\([^)]*[\"']w", PUBLISHER_TEXT))

    def test_18_required_env_vars_enforced(self):
        for name in publisher.REQUIRED_ENV_VARS:
            env = synthetic_env()
            del env[name]
            with self.subTest(missing=name):
                with self.assertRaises(publisher.PublishError) as caught:
                    publisher.load_config(env)
                self.assertIn(name, str(caught.exception))

        # Blank is missing, not a default.
        with self.assertRaises(publisher.PublishError):
            publisher.load_config(synthetic_env(SPORTSODDS_FTP_REMOTE_DIR="   "))

        config = publisher.load_config(synthetic_env())
        self.assertEqual(config.port, 21)
        self.assertEqual(config.remote_dir, "/tools/odds-scanner/predictions/NFL")
        self.assertEqual(publisher.load_config(synthetic_env(SPORTSODDS_FTP_PORT="2121")).port, 2121)
        with self.assertRaises(publisher.PublishError):
            publisher.load_config(synthetic_env(SPORTSODDS_FTP_PORT="not-a-port"))

    def test_18b_remote_dir_is_authoritative_and_never_inferred(self):
        # No server filesystem root is prepended, guessed or rewritten.
        for remote_dir in ("/tools/odds-scanner/predictions/NFL", "public_html/tools/odds-scanner/predictions/NFL"):
            config = publisher.load_config(synthetic_env(SPORTSODDS_FTP_REMOTE_DIR=remote_dir))
            self.assertEqual(config.remote_dir, remote_dir)
            self.assertEqual(config.remote_path("latest.json"), f"{remote_dir}/latest.json")
        # No server document root appears anywhere in the publisher.
        self.assertNotIn("/var/www", PUBLISHER_TEXT)
        for bad in ("", "   ", "/", "a/../b", "a\\b", "dir\nSTOR evil"):
            with self.subTest(remote_dir=bad):
                with self.assertRaises(publisher.PublishError):
                    publisher.load_config(synthetic_env(SPORTSODDS_FTP_REMOTE_DIR=bad))

    def test_19_ftp_mode_allowlist_enforced(self):
        self.assertEqual(publisher.ALLOWED_FTP_MODES, ("ftp", "ftps"))
        for mode in ("ftp", "ftps"):
            self.assertEqual(publisher.load_config(synthetic_env(SPORTSODDS_FTP_MODE=mode)).mode, mode)
        for mode in ("FTP", "FTPS", "sftp", "ftpes", "scp", "http", " ftp", "ftp "):
            with self.subTest(mode=mode):
                with self.assertRaises(publisher.PublishError) as caught:
                    publisher.load_config(synthetic_env(SPORTSODDS_FTP_MODE=mode))
                self.assertIn("SPORTSODDS_FTP_MODE", str(caught.exception))

    def test_33_password_never_printed(self):
        config = publisher.load_config(synthetic_env())
        self.assertEqual(config.password, TEST_PASSWORD)
        self.assertNotIn(TEST_PASSWORD, repr(config))
        self.assertNotIn(TEST_PASSWORD, str(config))
        self.assertIn("<redacted>", repr(config))

        json_path = self.write_json(synthetic_card())

        self.assertEqual(self.publish(json_path=json_path, dry_run=True), 0)
        self.assertNotIn(TEST_PASSWORD, self.output())

        self.stream = io.StringIO()
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path), 0)
        self.assertNotIn(TEST_PASSWORD, self.output())
        self.assertIn("password: <redacted>", self.output())

        # The password reaches ftplib and nothing else.
        login_calls = [call for call in FakeFtpClient.instances[0].calls if call[0] == "login"]
        self.assertEqual(login_calls, [("login", "sportsodds-test", TEST_PASSWORD)])

    def test_33b_failed_publication_exits_non_zero(self):
        json_path = self.write_json(synthetic_card(season=2025))
        with self.patched_environ(), mock.patch.object(sys, "stderr", new=io.StringIO()) as stderr:
            status = publisher.main(["--json", str(json_path), "--dry-run"])
        self.assertEqual(status, 2)
        self.assertIn("PUBLISH FAILED", stderr.getvalue())
        self.assertIn("season must be 2026", stderr.getvalue())
        self.assertNotIn(TEST_PASSWORD, stderr.getvalue())


# =========================================================================== #
# 20-26, 21: local validation strictly before any network operation
# =========================================================================== #
class PublisherValidationTests(PublisherHarness):
    def assert_rejected(self, payload, *, needle=None, raw=None):
        json_path = self.write_json(payload, raw=raw)
        with self.assertRaises(publisher.PublishError) as caught:
            self.publish(json_path=json_path, dry_run=True)
        if needle is not None:
            self.assertIn(needle, str(caught.exception))
        self.assertEqual(FakeFtpClient.instances, [])
        self.assertEqual(FakeFtpTlsClient.instances, [])

    def test_20_dry_run_performs_no_network_operation(self):
        json_path = self.write_json(synthetic_card())
        # Any socket creation at all -- by ftplib or anything else -- fails the test.
        with mock.patch.object(socket, "socket", side_effect=AssertionError("dry run opened a socket")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("dry run connected")), \
                self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path, dry_run=True), 0)

        output = self.output()
        self.assertIn("DRY RUN: no network connection will be opened", output)
        self.assertIn("host: ftp.example-test.invalid:21", output)
        self.assertIn("mode: ftp", output)
        self.assertIn("remote target: /tools/odds-scanner/predictions/NFL/latest.json", output)
        self.assertIn("remote target: /tools/odds-scanner/predictions/NFL/index.html", output)
        self.assertIn("validated public JSON: season=2026 week=2 horizon=TUE", output)
        self.assertEqual(FakeFtpClient.instances, [])
        self.assertEqual(FakeFtpTlsClient.instances, [])

    def test_20b_dry_run_validates_config_without_password_contents(self):
        env = synthetic_env()
        del env["SPORTSODDS_FTP_PASSWORD"]
        json_path = self.write_json(synthetic_card())
        self.assertEqual(self.publish(json_path=json_path, dry_run=True, env=env), 0)
        self.assertIn("DRY RUN", self.output())
        # Every other required setting is still enforced in a dry run.
        broken = synthetic_env(SPORTSODDS_FTP_HOST=None)
        with self.assertRaises(publisher.PublishError):
            self.publish(json_path=json_path, dry_run=True, env=broken)

    def test_20c_dry_run_validates_local_html(self):
        with self.assertRaises(publisher.PublishError) as caught:
            publisher.publish(
                json_path=None,
                html_path=self.tmp_dir / "missing-index.html",
                publish_json=False,
                publish_html=True,
                dry_run=True,
                env=synthetic_env(),
                stream=self.stream,
            )
        self.assertIn("page source not found", str(caught.exception))

        decoy = self.tmp_dir / "decoy.html"
        decoy.write_text("<html><title>Something else</title></html>", encoding="utf-8")
        with self.assertRaises(publisher.PublishError):
            publisher.publish(
                json_path=None, html_path=decoy, publish_json=False, publish_html=True,
                dry_run=True, env=synthetic_env(), stream=self.stream,
            )

    def test_21_malformed_schema_rejected_before_network(self):
        self.assert_rejected(synthetic_card(schema_version="wizard-nfl-pricing-v1"), needle="schema_version")
        self.assert_rejected(synthetic_card(season=2025), needle="season")
        self.assert_rejected(synthetic_card(horizon="WED"), needle="horizon")
        self.assert_rejected(synthetic_card(horizon="tue"), needle="horizon")
        self.assert_rejected(synthetic_card(generated_at_utc="2026-09-08 13:00:00"), needle="generated_at_utc")
        self.assert_rejected(synthetic_card(generated_at_utc="not-a-timestamp"), needle="generated_at_utc")
        self.assert_rejected(synthetic_card(week="two"), needle="week")
        self.assert_rejected(None, raw="{not json", needle="not valid JSON")
        self.assert_rejected(None, raw="[]", needle="must be a JSON object")

        missing = synthetic_card()
        del missing["generated_at_utc"]
        self.assert_rejected(missing, needle="missing required key")

        game_missing = synthetic_card(games=[{k: v for k, v in synthetic_game().items() if k != "market_total"}])
        self.assert_rejected(game_missing, needle="missing required key")

        self.assert_rejected(
            synthetic_card(games=[synthetic_game(kickoff_utc="2026-09-13T17:00:00")]),
            needle="kickoff_utc",
        )
        self.assert_rejected(
            synthetic_card(games=[synthetic_game(market_as_of_utc="soon")]),
            needle="market_as_of_utc",
        )
        self.assert_rejected(synthetic_card(games=[synthetic_game(home_team="BUF")]), needle="home_team equals away_team")
        self.assert_rejected(synthetic_card(games={"not": "a list"}), needle="games must be a list")

    def test_22_extra_public_json_keys_rejected(self):
        self.assert_rejected(
            synthetic_card(best_bet="HOU"),
            needle="unexpected key(s) ['best_bet']",
        )
        self.assert_rejected(
            synthetic_card(games=[synthetic_game(kelly_stake=0.25)]),
            needle="unexpected key(s) ['kelly_stake']",
        )

    def test_23_empty_games_rejected(self):
        self.assert_rejected(synthetic_card(games=[]), needle="games is empty")

    def test_24_duplicate_game_id_rejected(self):
        self.assert_rejected(
            synthetic_card(
                games=[
                    synthetic_game(),
                    synthetic_game(kickoff_utc="2026-09-13T20:00:00Z", away_team="NYJ", home_team="MIA"),
                ]
            ),
            needle="duplicate game_id",
        )

    def test_25_book_count_below_three_rejected(self):
        for key in ("market_ats_book_count", "market_total_book_count"):
            with self.subTest(key=key):
                self.assert_rejected(
                    synthetic_card(games=[synthetic_game(**{key: 2})]),
                    needle="below the certified minimum of 3 eligible books",
                )
                self.assert_rejected(synthetic_card(games=[synthetic_game(**{key: 5.0})]), needle="must be an integer")
                self.assert_rejected(synthetic_card(games=[synthetic_game(**{key: True})]), needle="must be an integer")
        self.assertEqual(publisher.MINIMUM_ELIGIBLE_BOOKS, 3)

    def test_26_non_finite_numbers_rejected(self):
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                raw = json.dumps(synthetic_card(), indent=2).replace("5.8", literal)
                self.assert_rejected(None, raw=raw, needle="non-finite literal")

        for key in publisher.NUMERIC_GAME_KEYS:
            with self.subTest(key=key):
                self.assert_rejected(synthetic_card(games=[synthetic_game(**{key: "3.5"})]), needle="must be a numeric")
                self.assert_rejected(synthetic_card(games=[synthetic_game(**{key: None})]), needle="must be a numeric")
                self.assert_rejected(synthetic_card(games=[synthetic_game(**{key: True})]), needle="must be a numeric")

    def test_26b_valid_card_summary(self):
        summary = publisher.validate_public_payload(synthetic_card())
        self.assertEqual(
            summary,
            {
                "season": 2026,
                "week": 2,
                "horizon": "TUE",
                "generated_at_utc": "2026-09-08T13:00:00Z",
                "game_count": 1,
            },
        )


# =========================================================================== #
# 27-32: transport behaviour
# =========================================================================== #
class PublisherTransportTests(PublisherHarness):
    def test_27_ftp_mode_uses_plain_ftp(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path), 0)

        self.assertEqual(len(FakeFtpClient.instances), 1)
        self.assertEqual(FakeFtpTlsClient.instances, [])
        client = FakeFtpClient.instances[0]
        self.assertFalse(client.is_tls)
        self.assertNotIn("prot_p", client.method_names())
        self.assertEqual(client.timeout, publisher.FTP_TIMEOUT_SECONDS)
        self.assertIn(("connect", "ftp.example-test.invalid", 21, publisher.FTP_TIMEOUT_SECONDS), client.calls)
        self.assertIn("quit", client.method_names())

    def test_28_ftps_mode_uses_ftp_tls_then_prot_p(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(
                self.publish(json_path=json_path, env=synthetic_env(SPORTSODDS_FTP_MODE="ftps")),
                0,
            )

        self.assertEqual(FakeFtpClient.instances, [])
        self.assertEqual(len(FakeFtpTlsClient.instances), 1)
        client = FakeFtpTlsClient.instances[0]
        self.assertTrue(client.is_tls)
        names = client.method_names()
        self.assertIn("prot_p", names)
        # Data channel protection is requested only after authentication.
        self.assertLess(names.index("login"), names.index("prot_p"))
        self.assertLess(names.index("prot_p"), names.index("storbinary"))
        self.assertEqual(client.timeout, publisher.FTP_TIMEOUT_SECONDS)

    def test_29_temporary_upload_then_rename(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path), 0)

        client = FakeFtpClient.instances[0]
        stores = [call for call in client.calls if call[0] == "storbinary"]
        renames = [call for call in client.calls if call[0] == "rename"]
        self.assertEqual(len(stores), 2)
        self.assertEqual(len(renames), 2)

        for store_call, rename_call in zip(stores, renames):
            command = store_call[1]
            temporary_name, final_name = rename_call[1], rename_call[2]
            self.assertTrue(command.startswith("STOR "))
            self.assertEqual(command.split(" ", 1)[1], temporary_name)
            self.assertNotEqual(temporary_name, final_name)
            self.assertTrue(temporary_name.endswith(".tmp"))
            self.assertIn(final_name, temporary_name)
            # The store for a file always precedes its rename.
            self.assertLess(client.calls.index(store_call), client.calls.index(rename_call))

        # Temporary names are unique per upload, so a retry cannot collide.
        self.assertEqual(len({call[1] for call in renames}), 2)
        # Only the two fixed public names survive.
        self.assertEqual(sorted(client.stored), ["index.html", "latest.json"])

    def test_29b_rename_refusal_falls_back_to_delete_then_rename(self):
        json_path = self.write_json(synthetic_card())
        FakeFtpClient.refuse_rename_over_existing = True
        with self.patched_transport():
            client = FakeFtpClient()
            client.stored["latest.json"] = b"previous"
            with mock.patch.object(publisher, "connect", return_value=client):
                self.assertEqual(self.publish(json_path=json_path, publish_html=False), 0)
        names = client.method_names()
        self.assertEqual(names.count("rename"), 2)
        self.assertIn("delete", names)
        self.assertLess(names.index("delete"), len(names) - 1)
        self.assertEqual(client.stored["latest.json"], json_path.read_bytes())

    def test_29c_failed_upload_exits_non_zero_and_publishes_nothing_broken(self):
        json_path = self.write_json(synthetic_card())

        class FailingClient(FakeFtpClient):
            def storbinary(self, command, source):
                super().storbinary(command, source)
                raise publisher.ftplib.error_temp("451 disk quota exceeded")

        client = FailingClient()
        with self.patched_environ(), \
                mock.patch.object(publisher, "connect", return_value=client), \
                mock.patch.object(sys, "stdout", new=io.StringIO()), \
                mock.patch.object(sys, "stderr", new=io.StringIO()) as stderr:
            status = publisher.main(["--json", str(json_path), "--json-only"])
        self.assertEqual(status, 2)
        self.assertIn("upload of /tools/odds-scanner/predictions/NFL/latest.json failed", stderr.getvalue())
        self.assertNotIn("latest.json", client.stored)
        # The abandoned temporary upload is cleaned up rather than left behind.
        self.assertIn("delete", client.method_names())
        self.assertEqual(client.stored, {})

    def test_30_remote_names_are_fixed_and_not_user_supplied(self):
        self.assertEqual(publisher.REMOTE_JSON_NAME, "latest.json")
        self.assertEqual(publisher.REMOTE_HTML_NAME, "index.html")

        parser_actions = {
            option
            for action in publisher.build_arg_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            parser_actions,
            {"-h", "--help", "--json", "--html", "--dry-run", "--html-only", "--json-only"},
        )
        # No CLI switch can rename the published files.
        for forbidden in ("--remote-name", "--remote-json", "--remote-html", "--remote-file", "--name"):
            self.assertNotIn(forbidden, parser_actions)

        json_path = self.write_json(synthetic_card())
        differently_named = self.tmp_dir / "week02-TUE.json"
        differently_named.write_bytes(json_path.read_bytes())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=differently_named), 0)
        self.assertEqual(sorted(FakeFtpClient.instances[0].stored), ["index.html", "latest.json"])

    def test_30b_publication_order_is_json_then_html(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path), 0)
        renames = [call[2] for call in FakeFtpClient.instances[0].calls if call[0] == "rename"]
        self.assertEqual(renames, ["latest.json", "index.html"])

    def test_30c_remote_directory_components_created_when_missing(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path), 0)
        client = FakeFtpClient.instances[0]
        self.assertEqual(
            [call for call in client.calls if call[0] in ("cwd", "mkd")],
            [
                ("cwd", "/"),
                ("cwd", "tools"), ("mkd", "tools"), ("cwd", "tools"),
                ("cwd", "odds-scanner"), ("mkd", "odds-scanner"), ("cwd", "odds-scanner"),
                ("cwd", "predictions"), ("mkd", "predictions"), ("cwd", "predictions"),
                ("cwd", "NFL"), ("mkd", "NFL"), ("cwd", "NFL"),
            ],
        )

        # A relative remote dir is walked from the login directory: no leading
        # cwd("/"), because the FTP account may be chrooted.
        FakeFtpClient.reset()
        with self.patched_transport():
            self.assertEqual(
                self.publish(json_path=json_path, env=synthetic_env(SPORTSODDS_FTP_REMOTE_DIR="predictions/NFL")),
                0,
            )
        self.assertNotIn(("cwd", "/"), FakeFtpClient.instances[0].calls)

    def test_31_json_only_supported(self):
        json_path = self.write_json(synthetic_card())
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=json_path, publish_html=False), 0)
        client = FakeFtpClient.instances[0]
        self.assertEqual(sorted(client.stored), ["latest.json"])
        self.assertEqual(client.stored["latest.json"], json_path.read_bytes())
        self.assertIn("page publication skipped (--json-only)", self.output())

    def test_32_html_only_supported(self):
        with self.patched_transport():
            self.assertEqual(self.publish(json_path=None, publish_json=False), 0)
        client = FakeFtpClient.instances[0]
        self.assertEqual(sorted(client.stored), ["index.html"])
        self.assertEqual(client.stored["index.html"], PAGE_PATH.read_bytes())

    def test_32b_cli_selection_flags(self):
        parser = publisher.build_arg_parser()
        self.assertTrue(parser.parse_args(["--json-only"]).json_only)
        self.assertTrue(parser.parse_args(["--html-only"]).html_only)
        self.assertEqual(parser.parse_args([]).html_path, str(publisher.DEFAULT_HTML_SOURCE))
        self.assertEqual(
            Path(parser.parse_args([]).html_path),
            REPO_ROOT / "web" / "sportsodds" / "nfl" / "index.html",
        )
        with mock.patch.object(sys, "stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--json-only", "--html-only"])

    def test_32c_json_publication_requires_a_json_path(self):
        with self.assertRaises(publisher.PublishError) as caught:
            self.publish(json_path=None, dry_run=True)
        self.assertIn("--json <PATH> is required", str(caught.exception))


# =========================================================================== #
# 34-35: nothing outside the intended change set moved
# =========================================================================== #
class RepositoryScopeTests(unittest.TestCase):
    @staticmethod
    def _git(*args) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise unittest.SkipTest(f"git unavailable or not a work tree: {completed.stderr.strip()}")
        return completed.stdout

    def _changed_paths(self) -> list[str]:
        paths: list[str] = []
        for line in self._git("status", "--porcelain", "--untracked-files=all").splitlines():
            entry = line[3:].strip()
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[1]
            paths.append(entry.strip('"'))
        return paths

    def test_34_no_model_files_changed(self):
        changed = self._changed_paths()
        unexpected = sorted(path for path in changed if path not in EXPECTED_CHANGE_SET)
        self.assertEqual(unexpected, [], f"change set escaped its scope: {unexpected}")
        protected = sorted(path for path in changed if path.startswith(PROTECTED_PREFIXES))
        self.assertEqual(protected, [], f"protected model surface modified: {protected}")
        # No tracked file is modified at all: the three files are additions.
        self.assertEqual(self._git("diff", "--name-only", "HEAD").split(), [])

    def test_34b_expected_change_set_present(self):
        for relative in sorted(EXPECTED_CHANGE_SET):
            self.assertTrue((REPO_ROOT / relative).is_file(), f"missing expected file: {relative}")

    def test_35_no_real_week1_output_generated(self):
        changed = self._changed_paths()
        for path in changed:
            self.assertFalse(path.startswith("outputs/"), f"model output written: {path}")
            self.assertFalse(path.startswith("public/"), f"public artifact written: {path}")
            self.assertFalse(path.endswith("latest.json"), f"pricing card written: {path}")
        # No feed file exists anywhere in the published page tree.
        self.assertEqual(list((REPO_ROOT / "web").rglob("*.json")), [])
        self.assertFalse((PAGE_PATH.parent / "latest.json").exists())
        # The publisher reads a card the operator names explicitly; it never
        # discovers, generates or defaults to one.
        self.assertNotIn("season_2026", PUBLISHER_TEXT)
        self.assertNotIn("outputs/", PUBLISHER_TEXT)
        self.assertIsNone(re.search(r"glob\(|rglob\(|iterdir\(|st_mtime", PUBLISHER_TEXT))
        self.assertNotIn("requests", PUBLISHER_TEXT)
        self.assertNotIn("urllib", PUBLISHER_TEXT)
        for module_name in ("balldontlie", "nfl_hybrid", "pandas", "numpy"):
            self.assertNotIn(f"import {module_name}", PUBLISHER_TEXT)


if __name__ == "__main__":
    unittest.main()
