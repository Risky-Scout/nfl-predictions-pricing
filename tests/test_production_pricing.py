import hashlib
import json

import pandas as pd
import pytest

from nfl_hybrid.pricing.production import (
    PricingPolicy,
    american_to_decimal,
    decimal_to_american,
    price_csv_frame,
    verify_production_spec,
)


def _spec():
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
    payload["production_spec_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_odds_conversions():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200


def test_push_adjusted_fair_price_and_ev():
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "market": ["ats"],
            "selection": ["home"],
            "market_probability": [0.49],
            "model_probability": [0.49],
            "push_probability": [0.02],
            "offered_decimal": [2.05],
            "probability_lower": [0.48],
            "probability_upper": [0.50],
            "quote_age_minutes": [2.0],
        }
    )

    priced = price_csv_frame(
        frame,
        production_spec=_spec(),
        policy=PricingPolicy(),
    )

    assert priced.loc[0, "model_fair_decimal"] == pytest.approx(
        0.98 / 0.49
    )
    assert priced.loc[0, "ev_per_1"] == pytest.approx(
        0.49 * 1.05 - 0.49
    )


def test_missing_uncertainty_abstains():
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "market": ["moneyline"],
            "selection": ["home"],
            "market_probability": [0.55],
            "offered_decimal": [2.0],
        }
    )

    priced = price_csv_frame(
        frame,
        production_spec=_spec(),
        policy=PricingPolicy(),
    )

    assert priced.loc[0, "decision"] == "ABSTAIN"
    assert (
        "MISSING_UNCERTAINTY_BOUNDS"
        in priced.loc[0, "decision_reason"]
    )


def test_positive_conservative_value_is_bet():
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "market": ["moneyline"],
            "selection": ["home"],
            "market_probability": [0.55],
            "offered_decimal": [2.0],
            "probability_lower": [0.53],
            "probability_upper": [0.57],
            "quote_age_minutes": [3.0],
        }
    )

    priced = price_csv_frame(
        frame,
        production_spec=_spec(),
        policy=PricingPolicy(),
    )

    assert priced.loc[0, "decision"] == "BET"
    assert priced.loc[0, "conservative_ev_per_1"] > 0


def test_nonpositive_conservative_value_is_no_bet():
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "market": ["total"],
            "selection": ["over"],
            "market_probability": [0.50],
            "push_probability": [0.01],
            "offered_decimal": [1.90],
            "probability_lower": [0.48],
            "probability_upper": [0.52],
            "quote_age_minutes": [3.0],
        }
    )

    priced = price_csv_frame(
        frame,
        production_spec=_spec(),
        policy=PricingPolicy(),
    )

    assert priced.loc[0, "decision"] == "NO_BET"


def test_model_probability_must_equal_frozen_market_baseline():
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "market": ["moneyline"],
            "selection": ["home"],
            "market_probability": [0.55],
            "model_probability": [0.56],
            "offered_decimal": [2.0],
            "probability_lower": [0.54],
            "probability_upper": [0.58],
        }
    )

    with pytest.raises(
        ValueError,
        match="market baseline",
    ):
        price_csv_frame(
            frame,
            production_spec=_spec(),
            policy=PricingPolicy(),
        )


def test_verify_production_spec(tmp_path):
    path = tmp_path / "production.json"
    path.write_text(
        json.dumps(_spec()),
        encoding="utf-8",
    )

    verified = verify_production_spec(path)
    assert verified["status"] == "FINAL_PRODUCTION_SPEC"
