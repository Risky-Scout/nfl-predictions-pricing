import numpy as np
import pandas as pd
from nfl_hybrid.calibration.adoption import (
    apply_calibration_adoption_gate,
    build_calibration_adoption_report,
)

def test_only_supported_calibration_is_adopted():
    frame = pd.DataFrame([
        {"market":"pregame_moneyline","variant":"football_only",
         "three_way_log_loss_gain":0.01,"three_way_log_loss_gain_ci_lower":0.001,
         "three_way_log_loss_gain_ci_upper":0.02,
         "three_way_brier_gain":0.003,"three_way_brier_gain_ci_lower":0.0001,
         "three_way_brier_gain_ci_upper":0.006},
        {"market":"pregame_total","variant":"market_augmented",
         "three_way_log_loss_gain":-0.004,"three_way_log_loss_gain_ci_lower":-0.01,
         "three_way_log_loss_gain_ci_upper":0.002,
         "three_way_brier_gain":-0.001,"three_way_brier_gain_ci_lower":-0.003,
         "three_way_brier_gain_ci_upper":0.001},
    ])
    report = build_calibration_adoption_report(frame)
    assert report.iloc[0]["decision"] == "ADOPT_CALIBRATION"
    assert report.iloc[1]["decision"] == "RETAIN_UNCALIBRATED_PROBABILITIES"

def test_gate_uses_calibrated_only_when_adopted():
    frame = pd.DataFrame([
        {"market":"pregame_moneyline","variant":"football_only",
         "model_lower_probability":0.40,"model_push_probability":0.03,
         "model_upper_probability":0.57,
         "model_conditional_upper_probability":0.58762886598,
         "calibrated_lower_probability":0.45,
         "calibrated_push_probability":0.01,
         "calibrated_upper_probability":0.54,
         "calibrated_conditional_upper_probability":0.54545454545},
        {"market":"pregame_total","variant":"market_augmented",
         "model_lower_probability":0.49,"model_push_probability":0.01,
         "model_upper_probability":0.50,
         "model_conditional_upper_probability":0.50505050505,
         "calibrated_lower_probability":0.50,
         "calibrated_push_probability":0.01,
         "calibrated_upper_probability":0.49,
         "calibrated_conditional_upper_probability":0.49494949495},
    ])
    report = pd.DataFrame([
        {"market":"pregame_moneyline","variant":"football_only",
         "adopt_calibration":True,"decision":"ADOPT_CALIBRATION","decision_reason":"PASS"},
        {"market":"pregame_total","variant":"market_augmented",
         "adopt_calibration":False,
         "decision":"RETAIN_UNCALIBRATED_PROBABILITIES","decision_reason":"FAIL"},
    ])
    result = apply_calibration_adoption_gate(frame, report)
    ml = result[result["market"]=="pregame_moneyline"].iloc[0]
    total = result[result["market"]=="pregame_total"].iloc[0]
    assert np.isclose(ml["production_push_probability"], 0.01)
    assert np.isclose(total["production_upper_probability"], 0.50)
