import numpy as np
import pandas as pd

from nfl_hybrid.modern.joint_score import JointScoreModel


def _data(n=250, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    margin = 3 + 7 * x + rng.normal(scale=10, size=n)
    total = 44 + 2 * x + rng.normal(scale=9, size=n)
    frame = pd.DataFrame(
        {
            "x": x,
            "team": np.where(x > 0, "A", "B"),
            "home_spread": -3.0,
            "total_line": 44.5,
            "home_margin": margin,
            "total_points": total,
        }
    )
    frame["home_win"] = (frame["home_margin"] > 0).astype(float)
    frame["home_cover"] = (frame["home_margin"] + frame["home_spread"] > 0).astype(float)
    frame["over"] = (frame["total_points"] > frame["total_line"]).astype(float)
    return frame


def test_joint_model_outputs_probabilities():
    data = _data()
    train = data.iloc[:180]
    calibration = data.iloc[180:220]
    test = data.iloc[220:]
    model = JointScoreModel(["x"], ["team"])
    model.fit(train, calibration=calibration)
    p = model.predict_markets(test)
    for column in ["home_win_probability", "home_cover_probability", "over_probability"]:
        assert p[column].between(0, 1).all()
    assert {"predicted_home_points", "predicted_away_points"}.issubset(p.columns)
