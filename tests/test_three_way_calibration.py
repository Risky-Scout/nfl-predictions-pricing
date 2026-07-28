import numpy as np
import pandas as pd

from nfl_hybrid.calibration.three_way import (
    CalibrationConfig,
    _fit_push_scales,
    _predict_push,
    _recombine,
    apply_expanding_three_way_calibration,
)


def _frame() -> pd.DataFrame:
    rows = []
    for season in (2021, 2022, 2023):
        for index in range(120):
            actual_push = int(index == 0)
            conditional = 0.45 + 0.10 * ((index % 2) == 0)
            rows.append(
                {
                    "game_id": f"{season}_{(index % 18) + 1:02d}_{index:03d}",
                    "season": season,
                    "market": "pregame_moneyline",
                    "variant": "football_only",
                    "actual_category": 1 if actual_push else int(index % 2) * 2,
                    "actual_push": actual_push,
                    "binary_target": np.nan if actual_push else int(index % 2),
                    "scored_binary": not actual_push,
                    "market_line": 0.0,
                    "model_lower_probability": (1 - 0.03) * (1 - conditional),
                    "model_push_probability": 0.03,
                    "model_upper_probability": (1 - 0.03) * conditional,
                    "model_conditional_upper_probability": conditional,
                    "cluster_id": f"{season}_{(index % 18) + 1:02d}",
                }
            )
    return pd.DataFrame(rows)


def test_recombined_probabilities_sum_to_one():
    lower, push, upper = _recombine(
        np.array([0.4, 0.7]),
        np.array([0.01, 0.05]),
    )
    assert np.allclose(lower + push + upper, 1.0)


def test_half_point_lines_have_zero_push_after_calibration():
    train = pd.DataFrame(
        {
            "model_push_probability": [0.03, 0.03, 0.03],
            "actual_push": [0, 1, 0],
            "market_line": [-3.0, -7.0, -3.0],
        }
    )
    scale, buckets = _fit_push_scales(
        train,
        "pregame_ats",
        CalibrationConfig(),
    )
    test = pd.DataFrame(
        {
            "model_push_probability": [0.02, 0.02],
            "market_line": [-3.5, -7.5],
        }
    )
    result = _predict_push(
        test,
        "pregame_ats",
        scale,
        buckets,
        CalibrationConfig(),
    )
    assert np.allclose(result, 0.0)


def test_rare_event_calibration_reduces_overpredicted_ties():
    frame = _frame()
    calibrated = apply_expanding_three_way_calibration(frame)
    later = calibrated[calibrated["season"] >= 2022]
    assert later["calibrated_push_probability"].mean() < 0.03


def test_first_oof_season_is_not_calibrated_with_future_data():
    frame = _frame()
    calibrated = apply_expanding_three_way_calibration(frame)
    first = calibrated[calibrated["season"] == 2021]
    assert set(first["calibration_method"]) == {"identity_no_prior_oof"}
    assert np.allclose(
        first["calibrated_push_probability"],
        first["model_push_probability"],
    )


def test_future_target_changes_do_not_change_2022_calibration():
    frame = _frame()
    first = apply_expanding_three_way_calibration(frame)
    modified = frame.copy()
    modified.loc[modified["season"] == 2023, "actual_push"] = 1
    modified.loc[modified["season"] == 2023, "actual_category"] = 1
    second = apply_expanding_three_way_calibration(modified)
    first_2022 = first[first["season"] == 2022][
        "calibrated_push_probability"
    ].to_numpy()
    second_2022 = second[second["season"] == 2022][
        "calibrated_push_probability"
    ].to_numpy()
    assert np.allclose(first_2022, second_2022)
