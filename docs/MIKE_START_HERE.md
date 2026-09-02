# NFL Model — Start Here

Mike, this is the short version. It should take about ten minutes.

Here is the bottom line.

This model predicts two numbers for each NFL game: the home team's margin
(home points minus away points) and the total points scored. It produces
those predictions at two fixed times each week, Tuesday and Friday. Once
the sportsbook spread and total are known, the predictions are turned into
probabilities for two bets: the home side against the spread, and the over
against the total.

Calibration helped. The model's raw probabilities were not as well
calibrated. Calibration moved them toward probabilities that matched
observed outcomes more closely.

Two things the model has **not** done. It has not shown that its point
predictions are better than the sportsbook's. It has not shown that its
bet probabilities beat the sportsbook. Nothing here is evidence of
profit.

Because of that, the model is frozen. Rather than keep tuning it on old
seasons until something looks good, we locked it and will test it on the
2026 season as those games happen.

## Scorecard

| Question | Current answer |
| --- | --- |
| Does the model avoid obvious look-ahead bias? | Yes — strong controls |
| Does calibration improve its probabilities? | Yes |
| Does it predict margins better than the sportsbook? | No |
| Does it predict totals better than the sportsbook? | No |
| Has it demonstrated an ATS probability edge? | No |
| Has it demonstrated a totals probability edge? | No |
| Is it proven profitable? | No |
| Is it ready for prospective 2026 evaluation? | Yes |
| Is live 2026 execution fully wired yet? | No — schedule/market plumbing remains |

## What the model actually does

The flow is straightforward:

Past NFL results
→ estimate each team's strength
→ predict the home margin and the game total
→ read the sportsbook spread and total
→ compute against-the-spread and over/under probabilities
→ apply the frozen calibration adjustment

The prediction engine is deliberately plain. Team strength is an Elo-style
rating: each team carries a pregame rating, a pregame win probability, and
a pregame expected margin, for both the home and away side. That is six
numbers per game, and nothing else. No box scores, no play-by-play, no
betting-market data feeds into the strength estimate.

Those six numbers go into a ridge regression, which is ordinary linear
regression with a penalty that keeps the coefficients small and stable.
Margin and total are fit as two separate regressions. The model also
estimates how uncertain each prediction is, which is what lets it produce
a probability later instead of just a point number.

Predictions are generated twice a week. The Tuesday forecast uses only
information that would have been available by the model's Tuesday noon
Eastern cutoff for that week's slate. The Friday forecast uses the same
rule at its Friday noon cutoff.

## One concrete example

This is made up, purely to show the mechanics. It is not a real
prediction and not a bet anyone should place.

The model predicts Home Team by 6 points and 46 total points. The
sportsbook has Home Team -3.5 and a total of 44.5. The model therefore
sees more room on the home side and on the over. The probability layer
then estimates how likely each of those bets is to win. It does not just
treat the 2.5-point gap on the spread, or the 1.5-point gap on the total,
as an automatic bet.

## What I did to keep the backtest honest

The main risk in a project like this is letting the model peek at
information it would not have had in real time. The controls:

A past game can update a team's strength only if its result would
actually have been available before the prediction cutoff. An earlier
game that finished after the cutoff does not count yet.

Tuesday forecasts cannot use anything from Wednesday, Thursday, or Friday
of that week. Friday forecasts cannot use weekend results.

Postponed and rescheduled games were handled explicitly. Each weekly slate
has its own Tuesday and Friday cutoff derived from that slate's earliest
kickoff. A game that kicks off before a given cutoff, such as a Thursday
game relative to the Friday forecast, is simply left out of that
forecast. It is never quietly shifted onto another week's cutoff to make
it fit.

Choosing between candidate models was done on earlier seasons only, using
2021, 2022, and 2023 as the seasons we scored against, with a confirming
check on 2024. The 2025 season was not used to pick the winning model. It
was replayed afterward, under the already-frozen rules, as a check.

Once a rule was frozen, it stayed frozen. We did not go back and loosen
something because a later result was disappointing.

## Why the model is simpler than you might expect

We did try to add more football signal. The list included:

- quarterback and depth-chart information;
- recent team scoring form;
- richer team-efficiency measures such as expected points added;
- alternative regression and model families, including a robust
  regression and a gradient-boosted tree model.

Injuries and weather were also investigated, but I did not have
sufficiently reliable pregame, timestamped data to test them under the
same production-standard rules, so they were not admitted into the
certified model.

Some of these ideas improved the average error a little. None of them
cleared the statistical bar we had set in advance for adding a feature:
the improvement had to hold up under a bootstrap resampling test, not just
in the headline number. So we kept the simpler Elo-based model.

This is a statement about discipline, not a claim that Elo is the best
possible way to model NFL games. It is the version that earned its place
under the rules we committed to.

## The most important result

The sportsbook is still the benchmark to beat, and this model has not
beaten it convincingly.

On the held-out 2025 games, the sportsbook's point forecasts were more
accurate than the model's. For the Friday forecasts, the model's margin
error was about 12.9 points versus the sportsbook's 12.2, and the model's
total error was about 13.8 versus 13.3. The gap runs the same direction,
roughly half a point, across all four combinations of Tuesday/Friday and
spread/total.

On the bet probabilities, the calibrated model came out close to even with
the sportsbook. Pooling the against-the-spread bets, the difference in
log-loss between the calibrated model and the market was near zero, and
its confidence interval crossed zero. In plain terms: the model was not
measurably better than the market, and not measurably worse. The totals
bets looked the same way.

The model's ability to tell winning bets from losing ones was weak. AUC,
which is 0.5 for a coin flip, sat at or just below 0.5 for all four bet
types. That is consistent with these being tight, efficiently priced
markets.

So no betting advantage is claimed.

## What calibration fixed — and what it did not

Calibration made the model's stated probabilities more realistic. If an
uncalibrated model says 70% too often, meaning those predictions win less
than 70% of the time, calibration pulls them toward a level that
historically behaves more like a true 70%. Across all four bet types, the
calibrated probabilities had lower log-loss and lower Brier score than the
raw ones, and the pooled improvement held up under resampling. The
against-the-spread improvement was the more solid of the two; the totals
improvement pointed the same way but was less certain once totals were
looked at on their own.

What calibration cannot do is add football information the model does not
have. It rescales the probabilities the model already produces. It does
not make the underlying margin and total predictions sharper. That is why
better calibration and no market edge can both be true at once.

## What I want you to review

Five questions where your read would actually help:

1. Do you see any remaining look-ahead or selection bias?
2. Is the model-versus-sportsbook comparison fair?
3. Do you agree that the current results do not justify claiming a betting
   edge?
4. Are the probability and calibration methods reasonable for spread and
   totals bets?
5. If you were improving the football signal, what information would you
   prioritize next?

## What happens next

The prediction model is frozen. The rules for evaluating it on 2026 are
also frozen, and were written down before meaningful 2026 results exist.

During the 2026 season, forecasts will be recorded before the games are
played. Results get attached afterward, and the original forecast is never
rewritten. The thresholds that would let us upgrade any of the "No"
answers in the scorecard are fixed and will not be moved because results
come in soft or strong. The point of 2026 is to find out whether the
model adds anything beyond the market.

Two separate readiness questions:

- Scientific review: ready. The repository contains the model code,
  methodology, frozen decisions, tests, results summaries, and audit trail
  needed to review the work. Some large raw-data and generated-artifact
  files live outside the repository, so a completely independent
  byte-for-byte reproduction would require those separately.
- Live production: not ready. The code still needs a live 2026 schedule
  source and a live odds feed wired in. That is software plumbing. It does
  not block review of the research.

## If you want the technical audit trail

- [`MODEL_STRENGTH_AND_LIMITATIONS.md`](MODEL_STRENGTH_AND_LIMITATIONS.md)
  — full metrics, the market comparison, and the rejected research, with
  confidence intervals.
- [`PROSPECTIVE_VALIDATION_2026.md`](PROSPECTIVE_VALIDATION_2026.md) — the
  frozen 2026 promotion rules.
- [`PRODUCTION_RUNBOOK_2026.md`](PRODUCTION_RUNBOOK_2026.md) — how the live
  pipeline is meant to run, and what is still missing.
- [`MIKE_HANDOFF.md`](MIKE_HANDOFF.md) — the original technical handoff.

Technical details, if you want them: ridge regression with alpha 100,
`StandardScaler` preprocessing, margin and total fit separately, one fit
per weekly card rather than per game. Four calibration streams
(spread/total × Tuesday/Friday), each fit in time order and never pooled.

Frozen tags:

- `v2026.1-fix8-certified` — the certified prediction model.
- `v2026.1-review-ready` — the review package.
- `v2026.1-prospective-preregistered` — the frozen 2026 rules.

The 2026 preregistration hash is
`a8bfca90d97c54ad42064854d4ed0a1c7115820cae998c5b282a2f9a0dd468e9`.
