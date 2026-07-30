from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from nfl_hybrid.publishing.static_site import (
    build_static_site,
)


def _write_production_spec(path: Path) -> str:
    payload = {
        "status": "FINAL_PRODUCTION_SPEC",
        "production_models": [
            {
                "market": "pregame_moneyline",
                "production_model_family":
                    "market_baseline",
                "production_model_name":
                    "market_t10_canonical",
            },
            {
                "market": "pregame_ats",
                "production_model_family":
                    "market_baseline",
                "production_model_name":
                    "market_t10_canonical",
            },
            {
                "market": "pregame_total",
                "production_model_family":
                    "market_baseline",
                "production_model_name":
                    "market_t10_canonical",
            },
        ],
    }

    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    payload["production_spec_sha256"] = digest
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return digest


def _pricing_frame(spec_hash: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g/1", "g/1", "g2"],
            "kickoff_utc": [
                "2026-09-10T00:20:00Z",
                "2026-09-10T00:20:00Z",
                "2026-09-11T00:20:00Z",
            ],
            "home_team": [
                "Home & Co",
                "Home & Co",
                "Home 2",
            ],
            "away_team": [
                "Away <One>",
                "Away <One>",
                "Away 2",
            ],
            "market_normalized": [
                "moneyline",
                "moneyline",
                "ats",
            ],
            "selection_normalized": [
                "home",
                "away",
                "home",
            ],
            "line": [None, None, -3.0],
            "market_probability": [0.55, 0.44, 0.49],
            "model_probability": [0.55, 0.44, 0.49],
            "push_probability": [0.0, 0.0, 0.02],
            "model_fair_decimal": [
                1.8181818,
                2.2727273,
                2.0,
            ],
            "model_fair_american": [
                -122,
                127,
                100,
            ],
            "offered_decimal_normalized": [
                2.0,
                2.3,
                2.05,
            ],
            "offered_american_normalized": [
                100,
                130,
                105,
            ],
            "edge_vs_offer_probability": [
                0.05,
                0.0052174,
                0.0120732,
            ],
            "ev_per_1": [0.10, 0.012, 0.0245],
            "roi": [0.10, 0.012, 0.0245],
            "conservative_ev_per_1": [
                0.06,
                -0.02,
                0.003,
            ],
            "decision": ["BET", "NO_BET", "BET"],
            "decision_reason": [
                "POSITIVE_CONSERVATIVE_VALUE",
                "NONPOSITIVE_CONSERVATIVE_VALUE",
                "POSITIVE_CONSERVATIVE_VALUE",
            ],
            "production_model_name": [
                "market_t10_canonical",
                "market_t10_canonical",
                "market_t10_canonical",
            ],
            "production_spec_sha256": [
                spec_hash,
                spec_hash,
                spec_hash,
            ],
        }
    )


def _decisions_frame() -> pd.DataFrame:
    rows = []

    for market, decision in [
        (
            "pregame_moneyline",
            "PRODUCTION_MARKET_BASELINE",
        ),
        (
            "pregame_ats",
            "PRODUCTION_MARKET_FALLBACK",
        ),
        (
            "pregame_total",
            "PRODUCTION_MARKET_BASELINE",
        ),
    ]:
        rows.append(
            {
                "market": market,
                "production_decision": decision,
                "production_model_name":
                    "market_t10_canonical",
                "games": 285,
                "candidate_mean_log_loss": 0.7,
                "market_mean_log_loss": 0.7,
                "mean_log_loss_gain": 0.0,
                "candidate_mean_brier": 0.5,
                "market_mean_brier": 0.5,
                "mean_brier_gain": 0.0,
            }
        )

    return pd.DataFrame(rows)


def test_build_static_site(tmp_path):
    pricing_path = tmp_path / "pricing.csv"
    decisions_path = tmp_path / "decisions.csv"
    scorecard_path = tmp_path / "scorecard.csv"
    bootstrap_path = tmp_path / "bootstrap.csv"
    spec_path = tmp_path / "production.json"
    config_path = tmp_path / "site.json"
    output_root = tmp_path / "site"

    spec_hash = _write_production_spec(spec_path)
    _pricing_frame(spec_hash).to_csv(
        pricing_path,
        index=False,
    )
    decisions = _decisions_frame()
    decisions.to_csv(decisions_path, index=False)
    decisions.to_csv(scorecard_path, index=False)
    decisions.to_csv(bootstrap_path, index=False)

    config_path.write_text(
        json.dumps(
            {
                "site_title": "Test Site",
                "site_subtitle": "Test subtitle",
                "publisher": "Test Publisher",
                "deployment_note": "Not deployed.",
                "show_generation_timestamp": False,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_static_site(
        pricing_csv_path=pricing_path,
        final_decisions_csv_path=decisions_path,
        final_scorecard_csv_path=scorecard_path,
        final_bootstrap_csv_path=bootstrap_path,
        production_spec_path=spec_path,
        site_config_path=config_path,
        output_root=output_root,
    )

    assert manifest["status"] == "STATIC_SITE_BUILT"
    assert manifest["games"] == 2
    assert manifest["pricing_rows"] == 3
    assert manifest["deployed_to_wizard_of_odds"] is False

    required = [
        "index.html",
        "performance.html",
        "methodology.html",
        "site_manifest.json",
        "assets/site.css",
        "downloads/pricing.csv",
        "downloads/final_2025_decisions.csv",
        "downloads/final_2025_scorecard.csv",
        "downloads/final_2025_bootstrap.csv",
        "downloads/production_model_spec.json",
        "games/g-1.html",
        "games/g2.html",
    ]

    for name in required:
        assert (output_root / name).exists(), name

    index_html = (
        output_root / "index.html"
    ).read_text(encoding="utf-8")

    assert "Away &lt;One&gt; at Home &amp; Co" in index_html
    assert "market_t10_canonical" in index_html


def test_missing_pricing_column_fails(tmp_path):
    spec_path = tmp_path / "production.json"
    spec_hash = _write_production_spec(spec_path)

    pricing = _pricing_frame(spec_hash).drop(
        columns=["ev_per_1"]
    )
    pricing_path = tmp_path / "pricing.csv"
    pricing.to_csv(pricing_path, index=False)

    decisions = _decisions_frame()
    decisions_path = tmp_path / "decisions.csv"
    decisions.to_csv(decisions_path, index=False)

    config_path = tmp_path / "site.json"
    config_path.write_text(
        json.dumps({}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_static_site(
            pricing_csv_path=pricing_path,
            final_decisions_csv_path=decisions_path,
            final_scorecard_csv_path=decisions_path,
            final_bootstrap_csv_path=decisions_path,
            production_spec_path=spec_path,
            site_config_path=config_path,
            output_root=tmp_path / "site",
        )
