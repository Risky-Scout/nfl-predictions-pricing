import pandas as pd

from nfl_hybrid.features.market_compact import _derive_context


def test_context_fallback_uses_season_type_and_rest_days():
    frame = pd.DataFrame(
        {
            "season_type": ["REG", "REG", "POST", "POST"],
            "home_rest_days": [7, 14, 7, 14],
            "away_rest_days": [7, 7, 14, 14],
        }
    )

    output: dict[str, object] = {}
    _derive_context(frame, output)

    assert output["playoff_flag"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert output["bye_diff"].tolist() == [0.0, 1.0, -1.0, 0.0]
    assert output["rest_days_diff"].tolist() == [0, 7, -7, 0]
