from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


_REQUIRED_FILES = (
    "model_spec.yaml",
    "validation_protocol.yaml",
    "uncertainty_contract.yaml",
    "abstention_policy.yaml",
    "data_contracts/markets.yaml",
    "data_contracts/football.yaml",
    "data_contracts/injuries.yaml",
    "data_contracts/weather.yaml",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return payload


def validate_governance_contracts(config_root: str | Path) -> dict[str, Any]:
    root = Path(config_root).expanduser().resolve()
    missing = [name for name in _REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing governance contracts: {missing}")

    documents = {name: _read_yaml(root / name) for name in _REQUIRED_FILES}

    model = documents["model_spec.yaml"]
    for key in (
        "model_name",
        "model_class",
        "unit_of_analysis",
        "prediction_horizon",
        "intended_decision",
        "markets",
    ):
        if key not in model:
            raise ValueError(f"model_spec.yaml: missing {key!r}.")
    if model.get("causal_claims_supported") is not False:
        raise ValueError("The pricing model must explicitly prohibit causal claims.")

    validation = documents["validation_protocol.yaml"]
    if validation.get("chronological_only") is not True:
        raise ValueError("Chronological validation must be mandatory.")
    if validation.get("random_row_split_prohibited") is not True:
        raise ValueError("Random row splitting must be prohibited.")
    if validation.get("untouched_final_test_season") != 2025:
        raise ValueError("The untouched final test must remain 2025.")
    if validation.get("architecture_confirmation_season") != 2024:
        raise ValueError("Architecture confirmation must remain 2024.")

    abstention = documents["abstention_policy.yaml"]
    no_prediction = abstention.get("no_prediction", {})
    if int(no_prediction.get("minimum_eligible_books", 0)) < 3:
        raise ValueError("At least three eligible books are required.")
    if int(no_prediction.get("stale_odds_minutes", 999)) > 15:
        raise ValueError("Stale-odds tolerance cannot exceed 15 minutes.")

    markets = documents["data_contracts/markets.yaml"]
    live = markets.get("live_contract", {})
    if live.get("official_snapshot") != "kickoff_minus_10_minutes":
        raise ValueError("Live market snapshot must be kickoff minus 10 minutes.")

    weather = documents["data_contracts/weather.yaml"]
    if weather.get("observed_final_weather_as_pregame_feature_prohibited") is not True:
        raise ValueError("Observed final weather must be prohibited pregame.")

    injuries = documents["data_contracts/injuries.yaml"]
    if injuries.get("feature_rule", {}).get("simple_injury_counts_prohibited") is not True:
        raise ValueError("Simple injury counts must be prohibited.")

    return {
        "status": "PASS",
        "config_root": str(root),
        "validated_files": list(_REQUIRED_FILES),
    }
