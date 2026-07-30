import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _write_games(path: Path) -> None:
    pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2026, 2026],
            "week": ["1", "1"],
            "home_team": ["NO", "BUF"],
            "away_team": ["ARI", "BAL"],
            "home_spread": [6.0, -1.5],
            "total_line": [44.5, 50.5],
        }
    ).to_csv(path, index=False)


def test_predict_week_cli_generates_card(tmp_path):
    games = tmp_path / "games.csv"
    out = tmp_path / "card.csv"
    _write_games(games)
    env = {"PYTHONPATH": str(REPO / "src")}
    import os

    env = {**os.environ, **env}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nfl_hybrid.pricing.predict_week",
            "--season",
            "2026",
            "--week",
            "1",
            "--games",
            str(games),
            "--production-spec",
            str(REPO / "config" / "production_model_spec.json"),
            "--output",
            str(out),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    card = pd.read_csv(out)
    assert len(card) == 6
    assert int(card["should_bet"].sum()) == 0
    assert "recommended_bets=0" in result.stdout
