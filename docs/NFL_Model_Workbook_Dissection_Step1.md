# NFL Predictive Model Workbook Dissection — Step 1

**Scope:** forensic audit of all 11 uploaded Excel workbooks for pre-game and in-play NFL Moneyline, game total, and against-the-spread modeling.

**Audit date:** July 24, 2026

## 1. Executive conclusion

The uploaded files are **not one finished three-market model**. They are a teaching/model-development suite whose pieces can be assembled into a coherent system:

1. **Adjusted Elo** estimates home win probability and expected margin.
2. **Quarterback value** modifies team strength before kickoff.
3. **Multiplicative offense/defense factors** estimate home and away points.
4. **Historical scoring, head-to-head, and wind adjustments** modify the total.
5. **Market odds** are converted to de-vigged probabilities and blended with external forecasts.
6. **Monte Carlo season simulation** propagates Elo forecasts through schedules and playoffs.
7. A small **minute-level scoring hazard** worksheet is the only in-play-like component, but it models first scoring time—not final Moneyline, ATS, or total.

The workbook family therefore supports a useful **legacy baseline**, but it does not contain a production-ready in-play model and it contains several defects that must not be copied into Python.

### What can be recovered exactly

- Every visible formula, cached result, sheet, defined name, hyperlink, and external-link reference.
- The VBA logic in both macro-enabled workbooks.
- The adjusted-Elo equations, K-factor, margin-of-victory multiplier, season reversion, home-field, rest, playoff, and QB adjustments.
- The multiplicative team scoring equations.
- The odds de-vigging, Brier score, contest scoring, and all 32 FiveThirtyEight/Yahoo/Vegas ensemble formulas.

### What cannot be recovered with absolute certainty

Several files are labeled **STUDENTS** and intentionally leave calculation cells blank. In those cases, the surrounding inputs, worked examples, named Solver cells, cached outputs, and related completed workbooks reveal the intended method, but the missing final formula is an **inference**, not hidden executable logic. This report labels those cases accordingly.

## 2. Audit inventory

| workbook | sheets | nonempty_cells | formula_cells | defined_names | hyperlinks |
| --- | --- | --- | --- | --- | --- |
| ELOprimervideoaccompanyingfile.xlsx | 4 | 120 | 31 | 0 | 0 |
| FiveThirtyEight NFL Predictions Game Modeling by David Glidden.xlsx | 25 | 501312 | 245835 | 9 | 1860 |
| NFLLongTermSTUDENTS.xlsm | 5 | 9749 | 2597 | 7 | 0 |
| NFLMODELSTUDENTS.xlsm | 9 | 21740 | 1242 | 26 | 0 |
| NFLWorksheet.xlsx | 3 | 721 | 61 | 0 | 0 |
| NFLscoresSTUDENTS.xlsx | 4 | 158029 | 381 | 35 | 0 |
| NFLscoresmarketcalibrationSTUDENTS.xlsx | 3 | 17328 | 98 | 28 | 0 |
| QuarterbackadjustmentSTUDENTS.xlsx | 3 | 83 | 17 | 0 | 0 |
| QuarterbackpassingyardsSTUDENTS.xlsx | 5 | 2567 | 45 | 0 | 0 |
| TotalPointsSTUDENTS.xlsx | 6 | 188051 | 644 | 6 | 0 |
| Week1students.xlsx | 3 | 46 | 0 | 0 | 0 |

The audit catalog contains:

- **70 worksheets**
- **553,237 formula cells**
- **1,832 normalized formula patterns**
- **111 defined names**
- extracted VBA from both `.xlsm` files

The companion CSVs and VBA files in the audit bundle provide cell-level evidence.

## 3. Portfolio map

| workbook | primary_role | market_use | completion_status |
| --- | --- | --- | --- |
| FiveThirtyEight NFL Predictions Game Modeling by David Glidden.xlsx | Probability ensemble and contest scoring | Pregame ML probability only | Completed but stale/brittle |
| NFLWorksheet-201127-113441.xlsx | First-scoring-event timing/hazard exercise | Limited in-play auxiliary | Partly completed |
| QuarterbackpassingyardsSTUDENTS-200909-223735.xlsx | QB passing-yards feature/submodel exercise | Pregame player projection | Template/inferred |
| ELOprimervideoaccompanyingfile-200904-092938.xlsx | Generic Elo primer | Foundational | Completed teaching example |
| TotalPointsSTUDENTS-200904-092938.xlsx | Team points and game total model | Pregame totals/team totals | Mostly completed with defects |
| NFLLongTermSTUDENTS-200904-092938.xlsm | Season/playoff Monte Carlo | Long-horizon outcomes | Executable but defective/obsolete |
| NFLscoresmarketcalibrationSTUDENTS-200904-092938.xlsx | Offense/defense score calibration with Solver | Pregame team scores/totals | Student template/incomplete |
| QuarterbackadjustmentSTUDENTS-200904-092938.xlsx | QB value and Elo adjustment | Pregame availability adjustment | Completed example with sign bug |
| NFLMODELSTUDENTS-200904-092938.xlsm | Integrated adjusted-Elo engine | Pregame ML and expected margin | Executable with serious defects |
| Week1students-200904-092938.xlsx | NFL model specification/template | Pregame design scaffold | Blank template |
| NFLscoresSTUDENTS-200904-092938.xlsx | Historical results/market/weather dataset and spread calibration | Raw data / empirical favorite win curve | Completed data workbook |

## 4. Reconstructed end-to-end model

### 4.1 Pregame adjusted Elo

For a home team \(h\) and away team \(a\), the workbook family intends:

\[
D_0 = R_h - R_a
\]

\[
D = D_0 + HFA + Travel + Rest + QB
\]

where:

\[
HFA =
\begin{cases}
55, & \text{non-neutral game}\\
0, & \text{neutral game}
\end{cases}
\]

\[
Travel = 4\left(\frac{d_a-d_h}{1000}\right)
\]

\[
Rest = 25(I_{h,bye}-I_{a,bye})
\]

\[
QB = QB_h-QB_a
\]

and in the workbook implementation:

\[
D_{playoff} = 1.2D
\]

The home win probability is:

\[
P(H)=\frac{1}{1+10^{-D/400}}
\]

The expected home margin, called **supremacy**, is:

\[
\widehat{M}=\frac{D}{25}
\]

The observed game result is coded as 1 for a home win, 0 for an away win, and 0.5 for a tie.

The margin-of-victory multiplier is:

\[
MOV=\ln(|M|+1)\frac{2.2}{0.001D_{winner}+2.2}
\]

The Elo update is:

\[
\Delta R_h=20(y-P(H))MOV
\]

\[
R'_h=R_h+\Delta R_h,\qquad R'_a=R_a-\Delta R_h
\]

The new-season regression is:

\[
R_{new}=\frac{2}{3}R_{old}+\frac{1}{3}(1505)
\]

**Important:** the files use a 55-Elo home-field constant. The simplified official FiveThirtyEight reference code used 65 in one published implementation, so the Python legacy replica must use 55 to match these workbooks rather than silently substituting the external default.

### 4.2 Quarterback value and Elo adjustment

The integrated workbook's intended game-level quarterback value is:

\[
QV=-2.2Att+3.7Comp+\frac{Yds}{5}+11.3TD-14.1INT-8Sack-1.1RushAtt+0.6RushYds+15.9RushTD
\]

Opponent adjustment:

\[
QV_{adj}=QV_{raw}+(\overline{QVAllowed}-QVAllowed_{opponent})
\]

The teaching files show a blend of prior and recent information, for example:

\[
QV_{blend}=0.25QV_{prior}+0.75QV_{recent}
\]

Starter impact on Elo:

\[
QBAdj=3.3(QV_{starter}-QV_{team})
\]

This should be treated as an **availability adjustment**, not as a stand-alone outcome model.

### 4.3 Multiplicative team scoring model

The scoring workbooks estimate attack and defense factors relative to league home/away scoring baselines.

For team \(i\):

\[
Attack_i=
\frac{
PointsScoredAtHome_i/LeagueHomePoints+
PointsScoredAway_i/LeagueAwayPoints
}{2}
\]

The workbook's “defense” value is a **defensive weakness factor**: values above 1 mean the team allows more points than average.

\[
DefenseWeakness_i=
\frac{
PointsAllowedAtHome_i/LeagueAwayPoints+
PointsAllowedAway_i/LeagueHomePoints
}{2}
\]

Expected team points:

\[
\mu_h=LeagueHomePoints\cdot Attack_h\cdot DefenseWeakness_a
\]

\[
\mu_a=LeagueAwayPoints\cdot Attack_a\cdot DefenseWeakness_h
\]

Base total:

\[
T_0=\mu_h+\mu_a
\]

The market-calibration workbook intends Solver to choose 32 attack and 32 defense factors by minimizing squared score error. The supplied student workbook leaves the model/error cells blank, but its named objective and changing-cell ranges, plus its worked formulas, make this intent clear.

### 4.4 Total-points adjustments

The `TotalPoints` workbook applies a fixed head-to-head blend:

\[
T_{blend}=0.70T_0+0.30T_{H2H}
\]

It then applies a linear wind factor:

\[
f(w)=\frac{45.973-0.3787w}{45.973}
\]

\[
\widehat{T}=T_{blend}f(w)
\]

This is the workbook's exact conceptual flow. The Python model should reproduce it only as a legacy baseline because the H2H weight is not sample-size aware, the wind lookup is defective, and the aggregates leak later games into earlier predictions.

### 4.5 Passing-yards/game-script submodel

The quarterback passing-yards workbook calculates:

\[
AttemptsPerGame=\frac{Attempts}{Games}
\]

\[
PassShare=\frac{PassAttempts}{PassAttempts+RushAttempts}
\]

It shrinks team values one-third toward the league mean:

\[
x_{shrunk}=0.667x_{team}+0.333x_{league}
\]

The intended game-script mixture is:

\[
ExpectedPassShare=P(win)PassShare_{winning}+
(1-P(win))PassShare_{losing}
\]

The blank calculation sheet then provides the inputs needed to combine projected volume with QB and opponent yards per attempt:

\[
ProjectedPassYards\approx AdjustedAttempts\times AdjustedYPA
\]

The final combination formula is not present, so that last equation is a reconstruction rather than an exact copied formula.

### 4.6 Market probability conversion

American Moneyline to raw implied probability:

\[
p_{raw}=
\begin{cases}
\frac{100}{odds+100}, & odds>0\\
\frac{odds}{odds-100}, & odds<0
\end{cases}
\]

For a two-way market:

\[
Overround=p_{raw,h}+p_{raw,a}
\]

\[
p_{devig,h}=\frac{p_{raw,h}}{Overround}
\]

The workbook rounds de-vigged probabilities to two decimal places. The Python production model should retain full precision and round only for display.

### 4.7 FiveThirtyEight/Yahoo/Vegas ensemble layer

The large FiveThirtyEight workbook is a probability-ensemble experiment. Its strongest cached in-sample rows were Vegas, Model 29, Model 31, Model 22, Model 32, Model 26, and Model 25. This is **not proof of future profitability** because thresholds and rankings are evaluated in-sample.

The complete formula catalog is supplied as `NFL_538_Ensemble_Model_Catalog.csv`. The most relevant formulas are:

- **Model 19:** Vegas probability.
- **Model 20:** average Yahoo, FiveThirtyEight, and Vegas.
- **Model 22:** average FiveThirtyEight and Vegas.
- **Model 25:** use Vegas when `|538 − Vegas| > 0.10`; otherwise use the three-way average.
- **Model 29:** use the three-way average when `|538 − Vegas| > 0.26`; otherwise average FiveThirtyEight and Vegas.
- **Model 31:** use Vegas when `|538 − Vegas| > 0.20`; otherwise average FiveThirtyEight and Vegas.
- **Model 32:** move Vegas three percentage points toward 50%.

**Model 26 discrepancy:** the written description says the fallback is the three-way average; the actual formula falls back to Vegas. The formula must be treated as source of truth for exact replication.

### 4.8 Scoring and calibration metric

The FiveThirtyEight workbook uses the Brier score:

\[
Brier=(p-y)^2
\]

and a contest-points transformation:

\[
Points=25-100(Brier)
\]

rounded to one decimal and doubled for playoff games in the annual sheets. Only the actual winner's row receives the contest points.

This metric is useful for probability quality, but the contest-points transform should not be the Python training objective. Use proper probabilistic losses—log loss/Brier—and assess calibration separately.

### 4.9 Minute-level scoring hazard worksheet

The in-play-like workbook begins with implied team points and decomposes them into scoring events.

Assumptions shown in the file:

- touchdown-event point share: 0.785
- average touchdown-event value: 6.6
- other-scoring point share: 0.215
- average other-scoring-event value: 2.95
- quarter event shares: 20%, 31%, 21%, 28%

Expected scoring events:

\[
N_{events}=
\frac{0.785T}{6.6}+
\frac{0.215T}{2.95}
\]

Base event probability per minute:

\[
h_0=\frac{N_{events}}{60}
\]

Minute hazard:

\[
h_t=h_0a_t
\]

First-score probability:

\[
P(FirstScore=t)=h_tS_{t-1}
\]

\[
S_t=S_{t-1}-P(FirstScore=t)
\]

The survival process resets at each quarter in the workbook, so it estimates the first scoring minute **within each quarter**. It does not use score, possession, field position, down/distance, timeouts, or player state and cannot serve as a final-outcome live model.

### 4.10 Long-term Monte Carlo simulation

The long-term VBA simulator:

1. updates ratings from completed fixtures;
2. draws the winner from Elo win probability;
3. repeatedly samples a rounded Normal margin until it is positive:
   - home winner: \(N(\widehat{M},11.5)\);
   - away winner: \(N(-\widehat{M},11.5)\);
4. writes the loser score as zero and winner score as the sampled margin;
5. updates Elo;
6. applies standings/tiebreak logic and simulates playoffs.

Because the generated “scores” encode only margin, the simulator cannot price totals and should not be used as the score generator for ATS either.

## 5. Workbook-by-workbook findings

### 5.1 FiveThirtyEight NFL Predictions Game Modeling by David Glidden

**Purpose:** compare and blend FiveThirtyEight, Yahoo pick-em, and market probabilities.

**Data mechanisms found:**

- cached FiveThirtyEight game forecasts;
- Yahoo `IMPORTHTML` pick-em feed;
- OddsPortal American Moneyline scraping;
- team-name normalization table;
- annual and all-season contest scorecards;
- 1,860 hyperlinks, mostly OddsPortal/Yahoo links.

**Outputs:**

- 32 heuristic probability models;
- Brier-derived points;
- ranks by cumulative points;
- weekly/yearly score summaries;
- ancillary pick-em and confidence-interval worksheets.

**Assessment:** useful as an ensemble and calibration lesson. It is not a fundamental score model and its web imports are stale/brittle.

### 5.2 NFLWorksheet

**Purpose:** distribute expected scoring events over minutes/quarters.

**Key input:** pregame implied team points (example total 51.5).

**Assessment:** usable as an auxiliary scoring-time feature or teaching hazard model. It is not a live Moneyline/ATS/total engine.

### 5.3 Quarterbackpassingyards

**Purpose:** project passing volume/efficiency using game script.

**Key features:**

- attempts per game;
- team pass share;
- pass share while winning/losing;
- QB yards per attempt;
- opponent allowed attempts/YPA;
- win probability;
- one-third shrinkage to league mean.

**Assessment:** final calculation is intentionally blank. The ingredients are clear, but exact output logic must be reconstructed.

### 5.4 ELO primer

**Purpose:** demonstrate generic Elo probability and K-factor updates.

\[
P_1=\frac{1}{1+10^{-(R_1-R_2)/400}}
\]

\[
R'_1=R_1+K(Actual-Expected)
\]

**Assessment:** foundational teaching file, not NFL-specific.

### 5.5 TotalPoints

**Purpose:** predict home points, away points, and total.

**Inputs:** historical team scoring by home/away status, league baselines, H2H average, wind, and exploratory year/week/temperature/handicap tables.

**Assessment:** the strongest direct totals logic in the suite, but it requires as-of rolling aggregation, shrinkage, fixed lookup repair, and time-aware validation.

### 5.6 NFLLongTerm

**Purpose:** simulate regular season, standings, playoffs, and championship probabilities.

**Inputs:** Elo ratings, schedule, venues, rest, travel, QB adjustments, playoff flags.

**Outputs:** Super Bowl, conference/division, playoff, and win-count frequencies.

**Assessment:** conceptually useful as a Monte Carlo layer after a valid game model. The supplied implementation has material bugs and an obsolete schedule/bracket.

### 5.7 NFLscoresmarketcalibration

**Purpose:** calibrate multiplicative team attack/defense factors against scores, with market-implied team scores shown alongside.

**Solver setup found:**

- objective: cell R2;
- changing cells: U5:V36;
- minimization;
- exponential game-recency concept with a 360-game half-life.

Recency weight:

\[
w=10\exp\left(\frac{\ln(0.5)}{360}\times AgeInGames\right)
\]

**Assessment:** intended model is identifiable, but the student workbook omits the executable model/error formulas. A Python rebuild should use log-scale team effects with constraints and regularization.

### 5.8 Quarterbackadjustment

**Purpose:** convert box-score performance into QB value, adjust for opponent, blend samples, and turn starter-vs-team difference into Elo points.

**Critical error:** the standalone workbook uses a -14.1 interception coefficient and then subtracts that term, causing interceptions to increase QB value. The integrated workbook has the correct sign structure.

### 5.9 NFLMODEL

**Purpose:** integrated adjusted-Elo game engine.

**Outputs:** pregame home win probability, expected margin, updated ratings.

**Critical VBA findings:**

- travel lookup never successfully marks the stadium column as found;
- travel therefore resolves to zero;
- even after that typo is repaired, the sign is reversed relative to penalizing the traveling team;
- blank future score cells are treated as 0-0 ties and update Elo;
- the bivariate Poisson module contains broken named ranges and is inactive.

**Assessment:** this is the main Moneyline/margin baseline, but its fit and forecast processes must be separated and unit-tested.

### 5.10 Week1students

**Purpose:** blank model-build specification.

**Fields listed:** ratings, home field, distance, rest, QB adjustment, playoff, win probability, supremacy, total, and postgame rating.

**Assessment:** no executable logic; useful only as a schema/design worksheet.

### 5.11 NFLscores

**Purpose:** historical scores, market lines, venue/weather, and empirical favorite-win calibration.

**Observed schema:**

- date, season, week, playoff;
- home/away teams and scores;
- favorite ID and favorite spread;
- total line;
- stadium/neutral;
- temperature, wind, humidity, weather detail.

It derives winner, winner team ID, and whether the favorite won. A line table reports empirical favorite win rate by point spread; for example, the cached sheet shows approximately 56.8% at -3 and 74.2% at -7.

**Assessment:** valuable raw historical data, but the calibration is an unregularized lookup rather than an ATS model.

## 6. Precise target definitions for the Python system

A coherent system should model a joint final-score distribution and then expose multiple market targets.

### Primary continuous targets

\[
HomePoints,\quad AwayPoints
\]

or equivalently:

\[
Margin=HomePoints-AwayPoints
\]

\[
Total=HomePoints+AwayPoints
\]

### Moneyline

\[
Y_{ML}=
\begin{cases}
1,& Margin>0\\
0.5,& Margin=0\\
0,& Margin<0
\end{cases}
\]

The production output should report home win, away win, and tie/OT treatment consistently with sportsbook settlement rules.

### Against the spread

Using a home spread \(s_h\), where a favorite has a negative number:

\[
ATSResult=Margin+s_h
\]

- cover if \(ATSResult>0\);
- push if \(ATSResult=0\);
- no cover if \(ATSResult<0\).

### Total

At total line \(L\):

\[
OUResult=HomePoints+AwayPoints-L
\]

- over if \(OUResult>0\);
- push if \(OUResult=0\);
- under if \(OUResult<0\).

### In-play

At every play snapshot \(t\), preferred labels are:

\[
RemainingHomePoints_t,\quad RemainingAwayPoints_t
\]

or the full final-score distribution conditional on game state. ML, ATS, and total probabilities should then be derived from that same distribution so the three markets remain mathematically consistent.

## 7. Exact data required

The full 31-row field specification is in `NFL_Model_Data_Requirements.csv`.

Minimum pregame data:

1. canonical game/team/venue identifiers and kickoff time;
2. historical scores and completed status;
3. timestamped Moneyline, spread, total, and prices;
4. rolling pregame Elo;
5. rest/bye, neutral site, travel;
6. expected starter QB and QB/team value;
7. timestamped injuries/inactives;
8. rolling offense/defense factors;
9. weather forecasts available at the same lead time as the prediction.

Minimum in-play data:

1. event/play timestamp and ordering;
2. quarter, clock, score, possession;
3. down, distance, yard line;
4. timeouts/overtime state;
5. drive, turnover, penalty, red-zone state;
6. current QB/material player exits;
7. live Moneyline/spread/total quotes with timestamps;
8. source publish, ingestion, and revision timestamps.

## 8. Data-source provenance

### Sources visibly embedded or linked in the workbooks

- FiveThirtyEight game forecasts and archived Elo data.
- Yahoo pick-em pages.
- OddsPortal American Moneylines.
- A PFR-style team/QB statistical table, although the workbook does not preserve a source URL.
- Historical score/line/weather data with a schema matching the Spreadspoke dataset.

Because a matching schema is evidence rather than definitive provenance, the score workbooks should be described as **Spreadspoke-compatible** unless original metadata is recovered.

### Current production replacements

- **Spreadspoke:** historical schedules, scores, point spreads, totals, venue, and weather.
  - https://spreadspoke.com/
- **FiveThirtyEight archived NFL Elo repository/data:** historical Elo and QB-adjusted fields; no longer a current live forecast source.
  - https://github.com/fivethirtyeight/nfl-elo-game
  - https://github.com/fivethirtyeight/data/tree/master/nfl-elo
- **nflverse:** play-by-play, schedules, player/team statistics, rosters, depth charts, and related research data.
  - https://nflverse.nflverse.com/
- **Licensed live data provider:** required for low-latency production play-by-play and live market operation. Sportradar is one example with NFL play-by-play/possession/location feeds.
  - https://developer.sportradar.com/football/reference/nfl-overview
- **Licensed timestamped odds provider:** required for current open, pregame, closing, and live quotes. The workbook's page scraping should not be used in production.
- **Injury/status feed:** nflverse's prior injury source is not available for 2025 onward, so current injury data requires official reports or another provider.

## 9. Defects that must not be ported

| severity | workbook | issue | affected_area |
| --- | --- | --- | --- |
| Critical | NFLMODEL | Travel adjustment disabled | Pregame ML/ATS |
| Critical | NFLMODEL | Travel sign reversed | Pregame ML/ATS |
| Critical | NFLMODEL | Unplayed games treated as 0-0 ties | All forecasts |
| High | Quarterbackadjustment | Interception sign double-negative | All QB-adjusted markets |
| High | TotalPoints | Approximate VLOOKUP on wind table | Totals |
| High | TotalPoints | Future leakage | Totals/ATS |
| High | TotalPoints | H2H fixed 30% weight | Totals |
| High | Market calibration | Incomplete Solver model | Totals/team scores |
| High | Market calibration | Unidentified factors | Totals/team scores |
| High | 538 ensemble | In-sample threshold search | ML |
| High | 538 ensemble | Model 26 description mismatch | ML |
| High | 538 feeds | Stale/brittle web imports | ML/market ensemble |
| High | NFLLongTerm | Margin-only fake scores | Totals/ATS |
| High | NFLLongTerm | Conference winner output bug | Season simulation |
| High | NFLLongTerm | Tie-record bug | Season simulation |
| High | NFLLongTerm | Super Bowl gets home venue/HFA | Season simulation |
| High | NFLLongTerm | Obsolete league structure | Season simulation |
| Medium | NFLscores | Tie/push handling weak | Calibration |
| Medium | TotalPoints | Exact integer wind domain | Totals |
| Medium | NFLMODEL | Bivariate Poisson inactive/broken | All |
| Medium | All student templates | Missing formulas are intentional blanks | All |
| Medium | All | No rigorous temporal validation | All |

## 10. Recommended Python architecture

### Layer A — immutable data and as-of joins

- canonical game/team/player/venue dimensions;
- append-only market snapshots;
- play/event snapshots;
- source publish and ingestion timestamps;
- feature retrieval that forbids data published after the prediction time.

### Layer B — exact legacy replicas

1. adjusted Elo ML/margin;
2. QB adjustment;
3. multiplicative team score model;
4. H2H/wind total baseline;
5. de-vigged market probability;
6. selected FiveThirtyEight/Vegas ensemble formulas;
7. season Monte Carlo rebuilt on valid score distributions.

This layer proves that the spreadsheet logic has been faithfully translated.

### Layer C — modern challenger models

A recommended coherent formulation is a joint home/away score model or a joint margin/total model:

\[
(M,T)\mid X \sim F(\mu_M(X),\mu_T(X),\sigma_M(X),\sigma_T(X),\rho(X))
\]

Possible implementations:

- regularized generalized linear models as transparent baselines;
- gradient-boosted models for conditional mean and quantiles;
- distributional boosting or a probabilistic neural model if data volume and monitoring justify it;
- discrete score simulation with overtime/tie handling.

### Layer D — market-specific calibration heads

- Moneyline calibration: isotonic, Platt/logistic, or beta calibration;
- ATS probability at the quoted spread;
- over probability at the quoted total;
- push probabilities for integer lines;
- optional market ensemble using de-vigged prices as features.

### Layer E — in-play state updater

At each play:

1. start with the pregame joint distribution;
2. condition on score/time/possession/down-distance/field position/timeouts;
3. predict remaining home and away points;
4. simulate or integrate to final score;
5. derive ML, ATS, and total probabilities at every offered line;
6. recalibrate by quarter/game-state segment.

The workbook minute hazard can be retained as one auxiliary feature, not the core state model.

## 11. Validation protocol

Random train/test splits are inappropriate because they leak future team/player information.

Use expanding walk-forward evaluation:

1. train through a cutoff week;
2. predict the next week using only as-of data;
3. ingest final results;
4. update and repeat.

Recommended metrics:

- **ML/ATS/OU probabilities:** log loss, Brier score, calibration curves/error.
- **Margin/total/team scores:** MAE, RMSE, pinball loss/CRPS.
- **Betting evaluation:** closing-line value, expected value at the actual available price, ROI with pushes/limits, and bootstrap confidence intervals.
- **Model comparison:** legacy workbook baseline, pure market baseline, and modern challenger.
- **In-play:** metrics by quarter, clock bucket, possession state, score differential, and quote latency.

Any claim of edge should be based on untouched forward periods and uncertainty intervals, not cumulative in-sample contest points.

## 12. Step 1 deliverables

The audit bundle contains:

- this report;
- workbook/sheet inventory;
- workbook summary;
- all defined names;
- 1,832 normalized formula patterns with example cells;
- complete 32-model ensemble catalog;
- exact data requirements;
- workbook logic map;
- defect register;
- extracted VBA modules and manifest.

## 13. Step 2 build recommendation

The safest build sequence is:

1. implement exact legacy formulas with unit tests;
2. create a leakage-safe feature store and backtest harness;
3. fit a joint score/margin-total challenger;
4. derive and calibrate ML, ATS, and total heads;
5. add play-level in-play updating;
6. compare legacy, market, and challenger models in walk-forward tests.

The preferred design is **one coherent score-distribution engine with separate calibrated market outputs**, while retaining the exact spreadsheet replicas as transparent benchmarks.
