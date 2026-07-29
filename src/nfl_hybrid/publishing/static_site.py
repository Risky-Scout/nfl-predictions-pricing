from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote
import json
import math
import re
import shutil

import pandas as pd

from nfl_hybrid.pricing.production import (
    verify_production_spec,
)


REQUIRED_PRICING_COLUMNS = {
    "game_id",
    "market_normalized",
    "selection_normalized",
    "market_probability",
    "model_probability",
    "push_probability",
    "model_fair_decimal",
    "model_fair_american",
    "offered_decimal_normalized",
    "offered_american_normalized",
    "edge_vs_offer_probability",
    "ev_per_1",
    "roi",
    "conservative_ev_per_1",
    "decision",
    "decision_reason",
    "production_model_name",
    "production_spec_sha256",
}

MARKET_ORDER = {
    "moneyline": 0,
    "ats": 1,
    "total": 2,
}

DECISION_ORDER = {
    "BET": 0,
    "NO_BET": 1,
    "ABSTAIN": 2,
}


@dataclass(frozen=True)
class StaticSiteConfig:
    site_title: str = "NFL Pregame Pricing Lab"
    site_subtitle: str = (
        "Frozen T-10 consensus pricing, offer comparison, "
        "and final-test reporting"
    )
    publisher: str = "Risky Scout"
    deployment_note: str = (
        "Local static integration package. This output is "
        "not deployed to wizardofodds.com."
    )
    show_generation_timestamp: bool = True

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "StaticSiteConfig":
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(**payload)


def _safe_filename(value: Any) -> str:
    text = str(value).strip()
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        text,
    ).strip("-")
    return normalized or "game"


def _fmt_number(
    value: Any,
    digits: int = 3,
) -> str:
    if pd.isna(value):
        return "—"

    number = float(value)
    if not math.isfinite(number):
        return "—"

    return f"{number:.{digits}f}"


def _fmt_probability(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.2f}%"


def _fmt_signed_probability(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{100.0 * float(value):+.2f}%"


def _fmt_decimal(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def _fmt_american(value: Any) -> str:
    if pd.isna(value):
        return "—"

    number = int(round(float(value)))
    return f"+{number}" if number > 0 else str(number)


def _fmt_line(value: Any) -> str:
    if pd.isna(value):
        return "—"

    number = float(value)
    if number > 0:
        return f"+{number:g}"
    return f"{number:g}"


def _fmt_kickoff(value: Any) -> str:
    if pd.isna(value):
        return "—"

    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )
    if pd.isna(timestamp):
        return escape(str(value))

    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _sort_pricing(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "kickoff_utc" in result.columns:
        result["_kickoff_sort"] = pd.to_datetime(
            result["kickoff_utc"],
            utc=True,
            errors="coerce",
        )
    else:
        result["_kickoff_sort"] = pd.NaT

    result["_market_sort"] = (
        result["market_normalized"]
        .map(MARKET_ORDER)
        .fillna(99)
    )
    result["_decision_sort"] = (
        result["decision"]
        .map(DECISION_ORDER)
        .fillna(99)
    )

    return result.sort_values(
        [
            "_kickoff_sort",
            "game_id",
            "_market_sort",
            "_decision_sort",
            "selection_normalized",
        ],
        kind="stable",
        na_position="last",
    ).drop(
        columns=[
            "_kickoff_sort",
            "_market_sort",
            "_decision_sort",
        ]
    ).reset_index(drop=True)


def _validate_pricing(
    frame: pd.DataFrame,
    production_spec: dict[str, Any],
) -> None:
    missing = sorted(
        REQUIRED_PRICING_COLUMNS - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"Pricing CSV is missing required columns: {missing}"
        )

    if frame.empty:
        raise ValueError("Pricing CSV is empty.")

    allowed_markets = {
        "moneyline",
        "ats",
        "total",
    }
    actual_markets = set(
        frame["market_normalized"].astype(str)
    )
    if not actual_markets <= allowed_markets:
        raise ValueError(
            f"Unsupported normalized markets: "
            f"{sorted(actual_markets - allowed_markets)}"
        )

    allowed_decisions = {
        "BET",
        "NO_BET",
        "ABSTAIN",
    }
    actual_decisions = set(
        frame["decision"].astype(str)
    )
    if not actual_decisions <= allowed_decisions:
        raise ValueError(
            f"Unsupported decisions: "
            f"{sorted(actual_decisions - allowed_decisions)}"
        )

    expected_hash = production_spec[
        "production_spec_sha256"
    ]
    found_hashes = set(
        frame["production_spec_sha256"]
        .dropna()
        .astype(str)
    )

    if found_hashes != {expected_hash}:
        raise ValueError(
            "Pricing rows do not all match the frozen "
            "production specification hash."
        )

    if not (
        frame["production_model_name"].astype(str)
        == "market_t10_canonical"
    ).all():
        raise ValueError(
            "Static site input is not using the frozen "
            "market_t10_canonical production model."
        )


def _page_shell(
    *,
    config: StaticSiteConfig,
    title: str,
    body: str,
    relative_prefix: str = "",
    generated_at: str,
) -> str:
    nav = (
        f'<a href="{relative_prefix}index.html">Predictions</a>'
        f'<a href="{relative_prefix}performance.html">Performance</a>'
        f'<a href="{relative_prefix}methodology.html">Methodology</a>'
        f'<a href="{relative_prefix}downloads/pricing.csv">'
        "Download CSV</a>"
    )

    generation = ""
    if config.show_generation_timestamp:
        generation = (
            f"<p>Generated {escape(generated_at)}</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · {escape(config.site_title)}</title>
  <link rel="stylesheet" href="{relative_prefix}assets/site.css">
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <p class="eyebrow">{escape(config.publisher)}</p>
      <h1>{escape(config.site_title)}</h1>
      <p class="subtitle">{escape(config.site_subtitle)}</p>
      <nav>{nav}</nav>
    </div>
  </header>
  <main class="wrap">
    {body}
  </main>
  <footer class="site-footer">
    <div class="wrap">
      <p>{escape(config.deployment_note)}</p>
      {generation}
    </div>
  </footer>
</body>
</html>
"""


def _decision_badge(value: Any) -> str:
    decision = str(value)
    css_class = {
        "BET": "bet",
        "NO_BET": "no-bet",
        "ABSTAIN": "abstain",
    }.get(decision, "unknown")

    return (
        f'<span class="badge {css_class}">'
        f"{escape(decision)}</span>"
    )


def _team_matchup(row: pd.Series) -> str:
    home = str(row.get("home_team", "")).strip()
    away = str(row.get("away_team", "")).strip()

    if home and away:
        return f"{away} at {home}"

    return str(row["game_id"])


def _pricing_table(
    frame: pd.DataFrame,
    *,
    include_game_link: bool,
    game_link_prefix: str = "games/",
) -> str:
    rows: list[str] = []

    for _, row in frame.iterrows():
        game_text = escape(_team_matchup(row))

        if include_game_link:
            filename = (
                _safe_filename(row["game_id"])
                + ".html"
            )
            game_cell = (
                f'<a href="{game_link_prefix}'
                f'{quote(filename)}">{game_text}</a>'
            )
        else:
            game_cell = game_text

        rows.append(
            "<tr>"
            f"<td>{game_cell}</td>"
            f"<td>{escape(_fmt_kickoff(row.get('kickoff_utc')))}</td>"
            f"<td>{escape(str(row['market_normalized']))}</td>"
            f"<td>{escape(str(row['selection_normalized']))}</td>"
            f"<td>{escape(_fmt_line(row.get('line')))}</td>"
            f"<td>{_fmt_probability(row['model_probability'])}</td>"
            f"<td>{_fmt_probability(row['push_probability'])}</td>"
            f"<td>{_fmt_decimal(row['model_fair_decimal'])}</td>"
            f"<td>{_fmt_american(row['model_fair_american'])}</td>"
            f"<td>{_fmt_decimal(row['offered_decimal_normalized'])}</td>"
            f"<td>{_fmt_american(row['offered_american_normalized'])}</td>"
            f"<td>{_fmt_signed_probability(row['edge_vs_offer_probability'])}</td>"
            f"<td>{_fmt_number(row['ev_per_1'], 4)}</td>"
            f"<td>{_fmt_number(row['conservative_ev_per_1'], 4)}</td>"
            f"<td>{_decision_badge(row['decision'])}</td>"
            f"<td>{escape(str(row['decision_reason']))}</td>"
            "</tr>"
        )

    return """
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Game</th>
      <th>Kickoff</th>
      <th>Market</th>
      <th>Selection</th>
      <th>Line</th>
      <th>Fair win</th>
      <th>Push</th>
      <th>Fair decimal</th>
      <th>Fair American</th>
      <th>Offered decimal</th>
      <th>Offered American</th>
      <th>Edge vs offer</th>
      <th>EV/$1</th>
      <th>Conservative EV/$1</th>
      <th>Decision</th>
      <th>Reason</th>
    </tr>
  </thead>
  <tbody>
""" + "\n".join(rows) + """
  </tbody>
</table>
</div>
"""


def _summary_cards(frame: pd.DataFrame) -> str:
    counts = (
        frame["decision"]
        .value_counts()
        .reindex(
            ["BET", "NO_BET", "ABSTAIN"],
            fill_value=0,
        )
    )

    unique_games = frame["game_id"].nunique()
    total_rows = len(frame)

    return f"""
<section class="cards" aria-label="Pricing summary">
  <article class="card">
    <p class="metric">{unique_games}</p>
    <p>games</p>
  </article>
  <article class="card">
    <p class="metric">{total_rows}</p>
    <p>priced selections</p>
  </article>
  <article class="card">
    <p class="metric">{int(counts['BET'])}</p>
    <p>BET</p>
  </article>
  <article class="card">
    <p class="metric">{int(counts['NO_BET'])}</p>
    <p>NO BET</p>
  </article>
  <article class="card">
    <p class="metric">{int(counts['ABSTAIN'])}</p>
    <p>ABSTAIN</p>
  </article>
</section>
"""


def _performance_table(
    decisions: pd.DataFrame,
) -> str:
    rows: list[str] = []

    for _, row in decisions.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['market']))}</td>"
            f"<td>{escape(str(row['production_decision']))}</td>"
            f"<td>{escape(str(row['production_model_name']))}</td>"
            f"<td>{int(row['games'])}</td>"
            f"<td>{_fmt_number(row['candidate_mean_log_loss'], 6)}</td>"
            f"<td>{_fmt_number(row['market_mean_log_loss'], 6)}</td>"
            f"<td>{_fmt_number(row['mean_log_loss_gain'], 6)}</td>"
            f"<td>{_fmt_number(row['candidate_mean_brier'], 6)}</td>"
            f"<td>{_fmt_number(row['market_mean_brier'], 6)}</td>"
            f"<td>{_fmt_number(row['mean_brier_gain'], 6)}</td>"
            "</tr>"
        )

    return """
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Market</th>
      <th>Production decision</th>
      <th>Production model</th>
      <th>Games</th>
      <th>Candidate log loss</th>
      <th>Market log loss</th>
      <th>Log-loss gain</th>
      <th>Candidate Brier</th>
      <th>Market Brier</th>
      <th>Brier gain</th>
    </tr>
  </thead>
  <tbody>
""" + "\n".join(rows) + """
  </tbody>
</table>
</div>
"""


def _write_css(path: Path) -> None:
    path.write_text(
        """
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
  --ink: #172033;
  --muted: #596579;
  --border: #d9dfE8;
  --surface: #f5f7fa;
  --accent: #214f9b;
  --bet: #0b6b3a;
  --no-bet: #805000;
  --abstain: #6b7280;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: white;
}
.wrap {
  width: min(1500px, calc(100% - 32px));
  margin: 0 auto;
}
.site-header {
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  padding: 24px 0 18px;
}
.eyebrow {
  margin: 0;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
h1 {
  margin: 4px 0;
}
.subtitle {
  margin: 0 0 14px;
  color: var(--muted);
}
nav {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}
nav a {
  color: var(--accent);
  font-weight: 700;
  text-decoration: none;
}
main {
  padding: 28px 0 40px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 18px 0 28px;
}
.card {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  background: var(--surface);
}
.card p {
  margin: 0;
}
.metric {
  font-size: 1.7rem;
  font-weight: 800;
}
.notice {
  border-left: 4px solid var(--accent);
  background: var(--surface);
  padding: 12px 14px;
  margin: 18px 0;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid var(--border);
  text-align: right;
  white-space: nowrap;
}
th:first-child, td:first-child,
th:nth-child(2), td:nth-child(2),
th:nth-child(3), td:nth-child(3),
th:nth-child(4), td:nth-child(4),
th:last-child, td:last-child {
  text-align: left;
}
th {
  background: var(--surface);
  position: sticky;
  top: 0;
}
.badge {
  display: inline-block;
  border-radius: 999px;
  color: white;
  font-size: 0.72rem;
  font-weight: 800;
  padding: 3px 8px;
}
.badge.bet { background: var(--bet); }
.badge.no-bet { background: var(--no-bet); }
.badge.abstain { background: var(--abstain); }
.site-footer {
  border-top: 1px solid var(--border);
  color: var(--muted);
  padding: 20px 0 32px;
  font-size: 0.9rem;
}
code {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 4px;
}
a {
  color: var(--accent);
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def build_static_site(
    *,
    pricing_csv_path: str | Path,
    final_decisions_csv_path: str | Path,
    final_scorecard_csv_path: str | Path,
    final_bootstrap_csv_path: str | Path,
    production_spec_path: str | Path,
    site_config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    pricing_csv_path = Path(pricing_csv_path).resolve()
    final_decisions_csv_path = Path(
        final_decisions_csv_path
    ).resolve()
    final_scorecard_csv_path = Path(
        final_scorecard_csv_path
    ).resolve()
    final_bootstrap_csv_path = Path(
        final_bootstrap_csv_path
    ).resolve()
    production_spec_path = Path(
        production_spec_path
    ).resolve()
    site_config_path = Path(
        site_config_path
    ).resolve()
    output_root = Path(output_root).resolve()

    production_spec = verify_production_spec(
        production_spec_path
    )
    config = StaticSiteConfig.from_json(
        site_config_path
    )

    pricing = _sort_pricing(
        pd.read_csv(pricing_csv_path)
    )
    decisions = pd.read_csv(
        final_decisions_csv_path
    )
    scorecard = pd.read_csv(
        final_scorecard_csv_path
    )
    bootstrap = pd.read_csv(
        final_bootstrap_csv_path
    )

    _validate_pricing(pricing, production_spec)

    if len(decisions) != 3:
        raise ValueError(
            "Final decisions CSV must contain exactly "
            "three market rows."
        )

    expected_decisions = {
        "pregame_moneyline":
            "PRODUCTION_MARKET_BASELINE",
        "pregame_ats":
            "PRODUCTION_MARKET_FALLBACK",
        "pregame_total":
            "PRODUCTION_MARKET_BASELINE",
    }
    found_decisions = dict(
        zip(
            decisions["market"].astype(str),
            decisions["production_decision"].astype(str),
        )
    )

    if found_decisions != expected_decisions:
        raise ValueError(
            "Final production decisions do not match "
            "the frozen 2025 evaluation."
        )

    shutil.rmtree(output_root, ignore_errors=True)
    (output_root / "assets").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "games").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "downloads").mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    index_body = f"""
<h2>Pregame pricing board</h2>
<div class="notice">
  <strong>Production model:</strong>
  <code>market_t10_canonical</code>, the canonical T-10 market
  baseline for moneyline, ATS, and total.
  BET/NO BET decisions compare available bookmaker prices with
  the frozen fair consensus and require positive conservative value.
</div>
{_summary_cards(pricing)}
{_pricing_table(pricing, include_game_link=True)}
"""

    (output_root / "index.html").write_text(
        _page_shell(
            config=config,
            title="Predictions",
            body=index_body,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )

    for game_id, game_frame in pricing.groupby(
        "game_id",
        sort=False,
    ):
        first = game_frame.iloc[0]
        matchup = _team_matchup(first)
        body = f"""
<p><a href="../index.html">← Back to predictions</a></p>
<h2>{escape(matchup)}</h2>
<p><strong>Game ID:</strong> {escape(str(game_id))}</p>
<p><strong>Kickoff:</strong> {escape(_fmt_kickoff(first.get('kickoff_utc')))}</p>
{_pricing_table(
    game_frame,
    include_game_link=False,
)}
"""
        filename = _safe_filename(game_id) + ".html"
        (output_root / "games" / filename).write_text(
            _page_shell(
                config=config,
                title=matchup,
                body=body,
                relative_prefix="../",
                generated_at=generated_at,
            ),
            encoding="utf-8",
        )

    ats = decisions[
        decisions["market"].eq("pregame_ats")
    ].iloc[0]

    performance_body = f"""
<h2>Frozen 2025 final-test performance</h2>
<div class="notice">
  The untouched 2025 final test covered 285 games.
  Moneyline and total retained the market baseline.
  ATS reverted to the market baseline because the challenger
  improved log loss by {_fmt_number(ats['mean_log_loss_gain'], 6)}
  while its Brier-score gain was
  {_fmt_number(ats['mean_brier_gain'], 6)}; the negative value
  failed the predeclared joint scoring gate.
</div>
{_performance_table(decisions)}
<h3>Interpretation</h3>
<p>
  These results report predictive scoring against the frozen
  canonical market reference. They do not establish guaranteed
  profitability or superiority over future market prices.
</p>
<p>
  Supporting files:
  <a href="downloads/final_2025_scorecard.csv">scorecard CSV</a>,
  <a href="downloads/final_2025_bootstrap.csv">bootstrap CSV</a>,
  and
  <a href="downloads/final_2025_decisions.csv">decisions CSV</a>.
</p>
"""

    (output_root / "performance.html").write_text(
        _page_shell(
            config=config,
            title="Performance",
            body=performance_body,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )

    methodology_body = """
<h2>Methodology</h2>
<h3>Reference time</h3>
<p>
  Prices are based on the canonical pregame market snapshot
  collected approximately ten minutes before kickoff (T-10).
</p>
<h3>Frozen production architecture</h3>
<p>
  The untouched 2025 final evaluation selected
  <code>market_t10_canonical</code> for moneyline, ATS, and total.
  The published probability is therefore the frozen no-vig
  consensus probability, not a claim of independent market
  outperformance.
</p>
<h3>Fair pricing</h3>
<p>
  With win probability <code>p</code> and push probability
  <code>q</code>, fair decimal odds equal
  <code>(1-q)/p</code>. For markets without pushes,
  this reduces to <code>1/p</code>.
</p>
<h3>Expected value</h3>
<p>
  At offered decimal odds <code>d</code>, EV per $1 equals
  <code>p(d-1) - (1-p-q)</code>. A push contributes zero.
</p>
<h3>Decision policy</h3>
<p>
  BET requires positive edge and positive EV at the lower
  uncertainty bound. NO BET indicates nonpositive conservative
  value. ABSTAIN indicates missing required information or a
  stale quote under the configured policy.
</p>
<h3>Uncertainty and limitations</h3>
<p>
  The final test contains one NFL season and 285 games.
  Results remain subject to sampling variation, market changes,
  quote availability, and implementation error. The system
  supports disciplined comparison and does not guarantee profit.
</p>
"""

    (output_root / "methodology.html").write_text(
        _page_shell(
            config=config,
            title="Methodology",
            body=methodology_body,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )

    _write_css(output_root / "assets" / "site.css")

    pricing.to_csv(
        output_root / "downloads" / "pricing.csv",
        index=False,
    )
    decisions.to_csv(
        output_root
        / "downloads"
        / "final_2025_decisions.csv",
        index=False,
    )
    scorecard.to_csv(
        output_root
        / "downloads"
        / "final_2025_scorecard.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_root
        / "downloads"
        / "final_2025_bootstrap.csv",
        index=False,
    )

    shutil.copy2(
        production_spec_path,
        output_root
        / "downloads"
        / "production_model_spec.json",
    )

    game_pages = sorted(
        path.name
        for path in (output_root / "games").glob("*.html")
    )

    manifest = {
        "status": "STATIC_SITE_BUILT",
        "generated_at": generated_at,
        "games": int(pricing["game_id"].nunique()),
        "pricing_rows": int(len(pricing)),
        "game_pages": game_pages,
        "production_spec_sha256":
            production_spec["production_spec_sha256"],
        "production_model": "market_t10_canonical",
        "markets": sorted(
            pricing["market_normalized"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "decisions": {
            key: int(value)
            for key, value in pricing[
                "decision"
            ].value_counts().to_dict().items()
        },
        "deployed_to_wizard_of_odds": False,
    }

    (output_root / "site_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest
