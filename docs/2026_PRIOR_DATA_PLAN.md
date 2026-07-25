# 2026 Prior Data Plan

This document defines the input freeze needed before fitting the 2026 season
priors. The statistical prior engine belongs to Phase C, but Phase A must
capture these inputs exactly.

## Freeze times

Create prior snapshots at:

1. post-draft;
2. start of training camp;
3. after each preseason week;
4. final cutdown;
5. 24 hours before each team's Week 1 game.

Every row needs `as_of_utc`. A later roster move, injury designation, depth
chart, or market price cannot alter an earlier prior snapshot.

## Team efficiency history

For each team and season from 2020–2025, compute opponent-adjusted offense,
defense, and special-teams components from play-by-play:

- EPA/play and success rate;
- dropback and rush splits;
- early-down and late-down efficiency;
- explosive rates;
- sack, interception, and fumble rates;
- red-zone and goal-to-go efficiency;
- pace and drives;
- starting field position;
- field-goal, punt, kickoff, and return EPA.

Store effective sample sizes and posterior uncertainty, not just point
estimates.

## Quarterback history

For every possible 2026 starter:

- latest 1,000 dropbacks with exponential weights;
- career and last-season EPA/dropback;
- CPOE;
- sack and interception rates;
- designed run and scramble value;
- age and experience;
- injury/availability history;
- days since last start;
- historical starter probability.

The game prior uses a mixture over possible starters:

\[
E[QB]=\sum_qP(q\ starts)E[QB_q]
\]

and includes between-starter uncertainty.

## Roster continuity

Using weekly rosters, depth charts, snaps, transactions, and player value:

- returning offensive snap share;
- returning defensive snap share;
- offensive-line continuity;
- returning target share;
- returning rush share;
- pass-rush continuity;
- secondary continuity;
- starter changes;
- additions and losses;
- draft capital;
- age-curve adjustments.

## Coaching

Capture head coach, coordinators, and play callers with exact announcement
dates. Estimate historical coach effects hierarchically after controlling for
quarterback, roster, opponent, venue, and game state.

## Market versions

Produce two prior families:

- `fundamental`: no 2026 game or futures market features;
- `market_augmented`: early game lines and season-win/futures inputs.

This makes the incremental value of football information measurable.

## Required prior output

Each team, quarterback, coach, and unit component must contain:

```text
prior_season
as_of_utc
entity_type
entity_id
component
prior_mean
prior_standard_deviation
effective_sample_size
regression_weight
league_mean
component adjustments
model_version
training_cutoff_utc
source_manifest_hash
```

The prior standard deviation must widen for rookie quarterbacks, unsettled
competitions, major roster turnover, new staffs, and material injury
uncertainty.
