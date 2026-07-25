# Workbook-derived historical game data

`workbook_games_1989_2019.csv` is extracted from the `NFL scores` worksheet in
the user-supplied `NFLscoresSTUDENTS` workbook.

Important limitations:

- seasons: 1989 through 2019;
- no timestamped Moneyline prices;
- spread prices and total prices are absent;
- weather is historical observed weather, not a timestamped pregame forecast;
- no injuries, starter-QB status, route/player participation, or play-by-play;
- several workbook dates were locale-converted incorrectly, so `game_index`
  and season/week ordering—not the raw date—are the authoritative sequence.

This file is suitable for reproducing workbook baselines and building an
initial research backtest. It is not sufficient for a production betting or
in-play system.
