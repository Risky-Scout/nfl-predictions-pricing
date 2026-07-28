from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np
import pandas as pd


EXPECTED_PRODUCTION_MODELS = {
    "pregame_moneyline": (
        "market_baseline",
        "market_t10_canonical",
    ),
    "pregame_ats": (
        "market_baseline",
        "market_t10_canonical",
    ),
    "pregame_total": (
        "market_baseline",
        "market_t10_canonical",
    ),
}

MARKET_ALIASES = {
    "pregame_moneyline": "moneyline",
    "moneyline": "moneyline",
    "h2h": "moneyline",
    "pregame_ats": "ats",
    "ats": "ats",
    "spreads": "ats",
    "pregame_total": "total",
    "total": "total",
    "totals": "total",
}

ALLOWED_SELECTIONS = {
    "moneyline": {"home", "away", "tie"},
    "ats": {"home", "away"},
    "total": {"over", "under"},
}


@dataclass(frozen=True)
class PricingPolicy:
    maximum_quote_age_minutes: float = 15.0
    minimum_conservative_edge_probability: float = 0.0
    minimum_conservative_ev_per_1: float = 0.0
    probability_tolerance: float = 1e-9
    require_quote_age: bool = False
    require_uncertainty_bounds: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "PricingPolicy":
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(**payload)


def american_to_decimal(value: float) -> float:
    american = float(value)

    if not math.isfinite(american) or american == 0:
        raise ValueError(
            "American odds must be finite and nonzero."
        )

    if american > 0:
        return 1.0 + american / 100.0

    return 1.0 + 100.0 / abs(american)


def decimal_to_american(value: float) -> int:
    decimal = float(value)

    if not math.isfinite(decimal) or decimal <= 1.0:
        raise ValueError(
            "Decimal odds must be finite and greater than 1."
        )

    if decimal >= 2.0:
        return int(round(100.0 * (decimal - 1.0)))

    return int(round(-100.0 / (decimal - 1.0)))


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_production_spec(
    path: str | Path,
) -> dict[str, Any]:
    spec_path = Path(path)
    payload = json.loads(
        spec_path.read_text(encoding="utf-8")
    )

    if payload.get("status") != "FINAL_PRODUCTION_SPEC":
        raise ValueError(
            "Production specification is not final."
        )

    recorded_hash = payload.get("production_spec_sha256")
    if not recorded_hash:
        raise ValueError(
            "Production specification hash is missing."
        )

    canonical = dict(payload)
    canonical.pop("production_spec_sha256", None)

    actual_hash = _canonical_json_sha256(canonical)
    if actual_hash != recorded_hash:
        raise ValueError(
            "Production specification content hash mismatch."
        )

    found = {
        row["market"]: (
            row["production_model_family"],
            row["production_model_name"],
        )
        for row in payload["production_models"]
    }

    if found != EXPECTED_PRODUCTION_MODELS:
        raise ValueError(
            "Production models differ from the frozen "
            "2025 final-test decisions."
        )

    return payload


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool,
    default: float | None = None,
) -> pd.Series:
    if column not in frame.columns:
        if required:
            raise ValueError(
                f"Required input column is missing: {column}"
            )

        return pd.Series(
            default,
            index=frame.index,
            dtype=float,
        )

    series = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if required and series.isna().any():
        rows = series[series.isna()].index.tolist()[:10]
        raise ValueError(
            f"{column} contains missing or nonnumeric "
            f"values at rows {rows}."
        )

    return series.astype(float)


def _normalize_market(value: Any) -> str:
    key = str(value).strip().lower()

    try:
        return MARKET_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported market: {value!r}"
        ) from exc


def _normalize_selection(
    market: str,
    value: Any,
) -> str:
    selection = str(value).strip().lower()

    if selection not in ALLOWED_SELECTIONS[market]:
        raise ValueError(
            f"Unsupported selection {value!r} for {market}."
        )

    return selection


def _normalize_offered_decimal(
    frame: pd.DataFrame,
) -> pd.Series:
    decimal = _numeric_series(
        frame,
        "offered_decimal",
        required=False,
    )
    american = _numeric_series(
        frame,
        "offered_american",
        required=False,
    )

    normalized = decimal.copy()

    use_american = normalized.isna() & american.notna()
    normalized.loc[use_american] = american.loc[
        use_american
    ].map(american_to_decimal)

    invalid = normalized.notna() & (
        ~np.isfinite(normalized) | (normalized <= 1.0)
    )
    if invalid.any():
        rows = invalid[invalid].index.tolist()[:10]
        raise ValueError(
            "Offered decimal odds must be greater than 1 "
            f"at rows {rows}."
        )

    both = decimal.notna() & american.notna()
    if both.any():
        converted = american.loc[both].map(
            american_to_decimal
        )
        mismatched = (
            decimal.loc[both] - converted
        ).abs() > 1e-6

        if mismatched.any():
            rows = mismatched[
                mismatched
            ].index.tolist()[:10]
            raise ValueError(
                "offered_decimal and offered_american "
                f"disagree at rows {rows}."
            )

    return normalized


def _fair_decimal(
    win_probability: pd.Series,
    push_probability: pd.Series,
) -> pd.Series:
    return (
        1.0 - push_probability
    ) / win_probability


def _validate_probability(
    series: pd.Series,
    name: str,
) -> None:
    invalid = (
        series.isna()
        | ~np.isfinite(series)
        | (series < 0.0)
        | (series > 1.0)
    )

    if invalid.any():
        rows = invalid[invalid].index.tolist()[:10]
        raise ValueError(
            f"{name} must be between 0 and 1 "
            f"at rows {rows}."
        )


def price_csv_frame(
    frame: pd.DataFrame,
    *,
    production_spec: dict[str, Any],
    policy: PricingPolicy,
) -> pd.DataFrame:
    required_columns = {
        "game_id",
        "market",
        "selection",
        "market_probability",
    }
    missing = sorted(required_columns - set(frame.columns))

    if missing:
        raise ValueError(
            f"Missing required input columns: {missing}"
        )

    output = frame.copy()

    output["market_normalized"] = output[
        "market"
    ].map(_normalize_market)

    output["selection_normalized"] = [
        _normalize_selection(market, selection)
        for market, selection in zip(
            output["market_normalized"],
            output["selection"],
        )
    ]

    market_probability = _numeric_series(
        output,
        "market_probability",
        required=True,
    )

    if "model_probability" in output.columns:
        model_probability = _numeric_series(
            output,
            "model_probability",
            required=True,
        )
    else:
        model_probability = market_probability.copy()

    push_probability = _numeric_series(
        output,
        "push_probability",
        required=False,
        default=0.0,
    ).fillna(0.0)

    _validate_probability(
        market_probability,
        "market_probability",
    )
    _validate_probability(
        model_probability,
        "model_probability",
    )
    _validate_probability(
        push_probability,
        "push_probability",
    )

    if (model_probability <= 0.0).any():
        rows = model_probability[
            model_probability <= 0.0
        ].index.tolist()[:10]
        raise ValueError(
            "model_probability must be greater than zero "
            f"at rows {rows}."
        )

    if (market_probability <= 0.0).any():
        rows = market_probability[
            market_probability <= 0.0
        ].index.tolist()[:10]
        raise ValueError(
            "market_probability must be greater than zero "
            f"at rows {rows}."
        )

    mismatch = (
        model_probability - market_probability
    ).abs() > policy.probability_tolerance

    if mismatch.any():
        rows = mismatch[mismatch].index.tolist()[:10]
        raise ValueError(
            "The frozen production model is the canonical "
            "market baseline, so model_probability must equal "
            f"market_probability at rows {rows}."
        )

    if (
        model_probability + push_probability
        > 1.0 + policy.probability_tolerance
    ).any():
        rows = (
            model_probability + push_probability
            > 1.0 + policy.probability_tolerance
        )
        raise ValueError(
            "model_probability plus push_probability exceeds "
            f"one at rows {rows[rows].index.tolist()[:10]}."
        )

    moneyline = output["market_normalized"].eq(
        "moneyline"
    )
    nonzero_moneyline_push = (
        moneyline
        & (
            push_probability.abs()
            > policy.probability_tolerance
        )
    )

    if nonzero_moneyline_push.any():
        rows = nonzero_moneyline_push[
            nonzero_moneyline_push
        ].index.tolist()[:10]
        raise ValueError(
            "Moneyline rows cannot use push_probability "
            f"at rows {rows}."
        )

    offered_decimal = _normalize_offered_decimal(
        output
    )

    lower = _numeric_series(
        output,
        "probability_lower",
        required=False,
    )
    upper = _numeric_series(
        output,
        "probability_upper",
        required=False,
    )

    supplied_bounds = lower.notna() & upper.notna()

    if supplied_bounds.any():
        _validate_probability(
            lower.loc[supplied_bounds],
            "probability_lower",
        )
        _validate_probability(
            upper.loc[supplied_bounds],
            "probability_upper",
        )

        invalid_order = supplied_bounds & (
            (lower > model_probability)
            | (upper < model_probability)
            | (lower > upper)
        )

        if invalid_order.any():
            rows = invalid_order[
                invalid_order
            ].index.tolist()[:10]
            raise ValueError(
                "Uncertainty bounds must satisfy "
                "lower <= model <= upper "
                f"at rows {rows}."
            )

        invalid_lower_push = supplied_bounds & (
            lower + push_probability
            > 1.0 + policy.probability_tolerance
        )

        if invalid_lower_push.any():
            rows = invalid_lower_push[
                invalid_lower_push
            ].index.tolist()[:10]
            raise ValueError(
                "probability_lower plus push_probability "
                f"exceeds one at rows {rows}."
            )

    quote_age = _numeric_series(
        output,
        "quote_age_minutes",
        required=False,
    )

    market_fair_decimal = _fair_decimal(
        market_probability,
        push_probability,
    )
    model_fair_decimal = _fair_decimal(
        model_probability,
        push_probability,
    )

    output["market_probability"] = market_probability
    output["model_probability"] = model_probability
    output["push_probability"] = push_probability
    output["probability_lower"] = lower
    output["probability_upper"] = upper
    output["uncertainty_width"] = upper - lower

    output["market_fair_decimal"] = (
        market_fair_decimal
    )
    output["market_fair_american"] = (
        market_fair_decimal.map(decimal_to_american)
    )
    output["model_fair_decimal"] = (
        model_fair_decimal
    )
    output["model_fair_american"] = (
        model_fair_decimal.map(decimal_to_american)
    )

    output["offered_decimal_normalized"] = (
        offered_decimal
    )
    output["offered_american_normalized"] = (
        offered_decimal.map(
            lambda value: (
                decimal_to_american(value)
                if pd.notna(value)
                else np.nan
            )
        )
    )

    offered_break_even = (
        1.0 - push_probability
    ) / offered_decimal

    output["offered_break_even_probability"] = (
        offered_break_even
    )
    output["edge_vs_market_probability"] = (
        model_probability - market_probability
    )
    output["edge_vs_offer_probability"] = (
        model_probability - offered_break_even
    )

    loss_probability = (
        1.0 - model_probability - push_probability
    )

    ev_per_1 = (
        model_probability * (offered_decimal - 1.0)
        - loss_probability
    )

    output["ev_per_1"] = ev_per_1
    output["roi"] = ev_per_1

    conservative_loss = (
        1.0 - lower - push_probability
    )
    conservative_ev = (
        lower * (offered_decimal - 1.0)
        - conservative_loss
    )
    conservative_edge = (
        lower - offered_break_even
    )

    output["conservative_ev_per_1"] = (
        conservative_ev
    )
    output["conservative_edge_probability"] = (
        conservative_edge
    )

    decisions: list[str] = []
    reasons: list[str] = []

    for index in output.index:
        abstention_reasons: list[str] = []

        if pd.isna(offered_decimal.loc[index]):
            abstention_reasons.append(
                "MISSING_OFFERED_ODDS"
            )

        bounds_missing = not supplied_bounds.loc[index]
        if (
            policy.require_uncertainty_bounds
            and bounds_missing
        ):
            abstention_reasons.append(
                "MISSING_UNCERTAINTY_BOUNDS"
            )

        age = quote_age.loc[index]
        if policy.require_quote_age and pd.isna(age):
            abstention_reasons.append(
                "MISSING_QUOTE_AGE"
            )
        elif (
            pd.notna(age)
            and age > policy.maximum_quote_age_minutes
        ):
            abstention_reasons.append("STALE_QUOTE")

        if abstention_reasons:
            decisions.append("ABSTAIN")
            reasons.append(
                ";".join(abstention_reasons)
            )
            continue

        if (
            conservative_ev.loc[index]
            > policy.minimum_conservative_ev_per_1
            and conservative_edge.loc[index]
            > policy.minimum_conservative_edge_probability
        ):
            decisions.append("BET")
            reasons.append(
                "POSITIVE_CONSERVATIVE_VALUE"
            )
        else:
            decisions.append("NO_BET")
            reasons.append(
                "NONPOSITIVE_CONSERVATIVE_VALUE"
            )

    output["decision"] = decisions
    output["decision_reason"] = reasons
    output["production_model_family"] = (
        "market_baseline"
    )
    output["production_model_name"] = (
        "market_t10_canonical"
    )
    output["production_spec_sha256"] = (
        production_spec["production_spec_sha256"]
    )

    return output
