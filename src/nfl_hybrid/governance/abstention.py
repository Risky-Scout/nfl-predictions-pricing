from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputState:
    odds_age_minutes: float
    eligible_books: int
    game_mapping_valid: bool = True
    feature_schema_valid: bool = True
    injury_report_current: bool = True
    roof_status_resolved: bool = True
    market_anchor_available: bool = True
    expected_value: float = 0.0
    edge: float = 0.0
    uncertainty: float = 0.0
    quote_still_current: bool = True
    drift_alarm: bool = False
    readiness_status: str = "RETAIN_BASELINE"


def evaluate_abstention(state: InputState) -> tuple[str, tuple[str, ...]]:
    no_prediction: list[str] = []
    if state.odds_age_minutes > 15:
        no_prediction.append("STALE_ODDS")
    if state.eligible_books < 3:
        no_prediction.append("INSUFFICIENT_BOOKS")
    if not state.game_mapping_valid:
        no_prediction.append("GAME_MAPPING_FAILURE")
    if not state.feature_schema_valid:
        no_prediction.append("FEATURE_SCHEMA_MISMATCH")
    if not state.injury_report_current:
        no_prediction.append("INJURY_REPORT_MISSING_OR_STALE")
    if not state.roof_status_resolved:
        no_prediction.append("ROOF_STATUS_UNRESOLVED")
    if not state.market_anchor_available:
        no_prediction.append("MARKET_ANCHOR_MISSING")
    if no_prediction:
        return "NO_PREDICTION", tuple(no_prediction)

    no_bet: list[str] = []
    if state.readiness_status == "RETAIN_BASELINE":
        no_bet.append("MODEL_DID_NOT_QUALIFY")
    if state.expected_value <= 0:
        no_bet.append("NONPOSITIVE_EV")
    if state.edge <= state.uncertainty:
        no_bet.append("EDGE_NOT_GREATER_THAN_UNCERTAINTY")
    if not state.quote_still_current:
        no_bet.append("QUOTE_MOVED")
    if state.drift_alarm:
        no_bet.append("DRIFT_OR_CALIBRATION_ALARM")
    if no_bet:
        return "NO_BET", tuple(no_bet)
    return "BET_ELIGIBLE", ()
