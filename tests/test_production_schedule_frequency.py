"""The scheduled cadence, proved against real kickoff instants.

THE PROBLEM. The sweep polled every fifteen minutes for the whole of Sunday,
Monday and Thursday -- 96 firings a day -- against runs that take 25-45
minutes. The concurrency group serialises production, so the queue never
drained: over a representative twelve-hour window 15 of 40 firings were
cancelled as superseded before they ever ran. Separately, ``daily_due`` is
unconditionally true, so the full daily maintenance pass ran on EVERY one of
those polls, re-doing evidence refresh, population update and recalibration
    10|that the sweep then repeated internally.

WHAT A CLOSE NEEDS FROM THE SCHEDULE. A stage is priced from
``newest_capture_at_or_before(cutoff)``, which splits the requirement in two:

  EXECUTION -- a poll at or after the cutoff makes the stage due. There is no
    deadline; a later poll still prices the cutoff from an observation that
    predates it. Thinning the schedule delays a CLOSE, it cannot lose one.
  PRECISION -- a poll shortly BEFORE the cutoff, because that poll's capture
    is the board the CLOSE is priced from. This is what degrades.

    20|So both are asserted below, separately, against kickoff instants generated
through ``zoneinfo`` so EDT and EST are exercised for real rather than
hand-mapped.

Hermetic: the workflow's own cron lines and its own gate shell. No network,
no server, no provider, nothing triggered.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nfl_2026_production.yml"
NY = ZoneInfo("America/New_York")

DAILY_CRON = "20 11 * * *"
CERTIFIED_CRONS = ("5 16 * * 2,5", "5 17 * * 2,5")

# Every month the sweep crons name, so the hour-coverage and per-day budget
# assertions see the whole active calendar.
SEASON_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
SEASON_END = datetime(2027, 2, 1, tzinfo=timezone.utc)

# The 2026 regular season itself: Week 1 Thursday through the Monday after
# Week 18. Kickoffs are generated only inside this, so the tests are not
# asserting coverage of an August Monday or a February Sunday that has no
# game. It spans the 2026-11-01 DST transition, so every window below is
# exercised in both offsets for real.
FIRST_KICKOFF_DAY = datetime(2026, 9, 3, tzinfo=NY).date()
LAST_KICKOFF_DAY = datetime(2027, 1, 11, tzinfo=NY).date()

CLOSE_LEAD = timedelta(minutes=60)  # CLOSE is kickoff minus 60 minutes.


# --------------------------------------------------------------------------- #
# a cron evaluator, only as much of one as these expressions need
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _field(spec: str, lo: int, hi: int) -> frozenset[int]:
    values: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, raw_step = part.split("/")
            step = int(raw_step)
        if part == "*":
            start, stop = lo, hi
        elif "-" in part:
            start, stop = (int(x) for x in part.split("-"))
        else:
            start = stop = int(part)
        values.update(range(start, stop + 1, step))
    return frozenset(values)


@lru_cache(maxsize=None)
def _compile(cron: str) -> tuple:
    minute, hour, dom, month, dow = cron.split()
    return (
        _field(minute, 0, 59),
        _field(hour, 0, 23),
        _field(dom, 1, 31),
        _field(month, 1, 12),
        _field(dow, 0, 7),
        dom != "*",
        dow != "*",
    )


def _matches(cron: str, moment: datetime) -> bool:
    minutes, hours, doms, months, dows, dom_restricted, dow_restricted = _compile(cron)
    if moment.minute not in minutes or moment.hour not in hours or moment.month not in months:
        return False
    # Cron numbers Sunday 0 (and tolerates 7); Python's isoweekday is Monday 1.
    cron_dow = moment.isoweekday() % 7
    day_ok = moment.day in doms
    week_ok = cron_dow in dows or (cron_dow == 0 and 7 in dows)
    if dom_restricted and dow_restricted:
        return day_ok or week_ok  # POSIX OR-ing; avoided in this workflow
    return day_ok and week_ok


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def crons(workflow) -> list[str]:
    return [entry["cron"] for entry in workflow[True]["schedule"]]


@pytest.fixture(scope="module")
def sweep_crons(crons) -> list[str]:
    return [c for c in crons if c != DAILY_CRON and c not in CERTIFIED_CRONS]


def _firings(cron_list: list[str], start: datetime, end: datetime) -> list[datetime]:
    """Every UTC instant at which any of these crons fires."""
    fired: set[datetime] = set()
    moment = start
    one_minute = timedelta(minutes=1)
    while moment < end:
        for cron in cron_list:
            if _matches(cron, moment):
                fired.add(moment)
                break
        moment += one_minute
    return sorted(fired)


@pytest.fixture(scope="module")
def sweep_firings(sweep_crons) -> list[datetime]:
    return _firings(sweep_crons, SEASON_START, SEASON_END)


# --------------------------------------------------------------------------- #
# real kickoff instants
# --------------------------------------------------------------------------- #
def _kickoffs(weekday: int, local_time: tuple[int, int]) -> list[datetime]:
    """Every kickoff of one weekly slot across the season, as real UTC
    instants -- so the EDT half of the season and the EST half are both
    generated rather than assumed."""
    hour, minute = local_time
    out = []
    day = FIRST_KICKOFF_DAY
    last = LAST_KICKOFF_DAY
    while day <= last:
        if day.isoweekday() == weekday:
            local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)
            out.append(local.astimezone(timezone.utc))
        day += timedelta(days=1)
    return out


# The weekly slate. Monday 1 ... Sunday 7, ET wall clock.
WEEKLY_SLOTS = {
    "THU night 20:15 ET": (4, (20, 15)),
    "SUN international 09:30 ET": (7, (9, 30)),
    "SUN early 13:00 ET": (7, (13, 0)),
    "SUN late 16:05 ET": (7, (16, 5)),
    "SUN late 16:25 ET": (7, (16, 25)),
    "SUN night 20:20 ET": (7, (20, 20)),
    "MON night 20:15 ET": (1, (20, 15)),
}

# Off-pattern slates. These have always been served by the baseline cadence
# rather than a dedicated window, and still are.
OFF_PATTERN_SLOTS = {
    "Thanksgiving THU 12:30 ET": (4, (12, 30)),
    "Thanksgiving THU 16:30 ET": (4, (16, 30)),
    "Black Friday 15:00 ET": (5, (15, 0)),
    "Saturday 13:00 ET": (6, (13, 0)),
    "Saturday 16:30 ET": (6, (16, 30)),
    "Saturday 20:15 ET": (6, (20, 15)),
}


def _execution_lag(firings: list[datetime], cutoff: datetime) -> timedelta | None:
    """How long after the cutoff the stage first becomes executable."""
    later = [f for f in firings if f >= cutoff]
    return (later[0] - cutoff) if later else None


def _precision_lag(firings: list[datetime], cutoff: datetime) -> timedelta | None:
    """How stale the board the CLOSE will be priced from is."""
    earlier = [f for f in firings if f <= cutoff]
    return (cutoff - earlier[-1]) if earlier else None


# ===========================================================================
# 3-5. The weekly CLOSE slates keep fifteen-minute treatment.
# ===========================================================================
@pytest.mark.parametrize("label", sorted(WEEKLY_SLOTS))
def test_every_weekly_close_is_executed_within_fifteen_minutes(label, sweep_firings):
    weekday, local_time = WEEKLY_SLOTS[label]
    kickoffs = _kickoffs(weekday, local_time)
    assert kickoffs, label

    worst = max(
        (_execution_lag(sweep_firings, k - CLOSE_LEAD) for k in kickoffs),
        key=lambda lag: (lag is None, lag or timedelta(0)),
    )
    assert worst is not None, f"{label}: no poll ever follows the CLOSE"
    assert worst <= timedelta(minutes=15), f"{label}: worst execution lag {worst}"


@pytest.mark.parametrize("label", sorted(WEEKLY_SLOTS))
def test_every_weekly_close_is_priced_from_a_board_at_most_fifteen_minutes_old(
    label, sweep_firings
):
    """The precision half. A poll must sit in the run-up to the cutoff, not
    just after it, because its capture is what the CLOSE is priced from."""
    weekday, local_time = WEEKLY_SLOTS[label]
    worst = max(
        (_precision_lag(sweep_firings, k - CLOSE_LEAD) for k in _kickoffs(weekday, local_time)),
        key=lambda lag: (lag is None, lag or timedelta(0)),
    )
    assert worst is not None, f"{label}: no poll ever precedes the CLOSE"
    assert worst <= timedelta(minutes=15), f"{label}: worst capture staleness {worst}"


@pytest.mark.parametrize("label", sorted(WEEKLY_SLOTS))
def test_the_kickoff_itself_is_also_polled_promptly(label, sweep_firings):
    """The window must not stop at the cutoff: the sweep that executes a
    CLOSE may be the one that publishes it."""
    weekday, local_time = WEEKLY_SLOTS[label]
    worst = max(
        (_execution_lag(sweep_firings, k) for k in _kickoffs(weekday, local_time)),
        key=lambda lag: (lag is None, lag or timedelta(0)),
    )
    assert worst is not None and worst <= timedelta(minutes=15), f"{label}: {worst}"


def test_both_daylight_offsets_are_exercised_by_these_slots():
    """Guards the guard: if the season ever stopped spanning the DST change,
    the tests above would silently stop testing EST."""
    offsets = {
        k.astimezone(NY).utcoffset() for k in _kickoffs(*WEEKLY_SLOTS["SUN night 20:20 ET"])
    }
    assert offsets == {timedelta(hours=-4), timedelta(hours=-5)}


# ===========================================================================
# 6-7. Off-pattern kickoffs, OPEN and MID keep the baseline guarantee.
# ===========================================================================
def test_every_hour_of_the_season_contains_a_poll(sweep_firings):
    """The single invariant the whole design rests on. It bounds everything
    the dedicated windows do not name: an off-pattern kickoff, a first board
    appearing at any hour, the Friday MID, the Monday rollover."""
    hours = {f.replace(minute=0, second=0, microsecond=0) for f in sweep_firings}
    expected = set()
    moment = SEASON_START
    while moment < SEASON_END:
        expected.add(moment)
        moment += timedelta(hours=1)
    assert hours == expected


@pytest.mark.parametrize("label", sorted(OFF_PATTERN_SLOTS))
def test_an_off_pattern_close_is_still_executed_within_the_hour(label, sweep_firings):
    weekday, local_time = OFF_PATTERN_SLOTS[label]
    for kickoff in _kickoffs(weekday, local_time):
        cutoff = kickoff - CLOSE_LEAD
        execution = _execution_lag(sweep_firings, cutoff)
        precision = _precision_lag(sweep_firings, cutoff)
        assert execution is not None and execution <= timedelta(minutes=60), f"{label} {cutoff}"
        assert precision is not None and precision <= timedelta(minutes=60), f"{label} {cutoff}"


def test_the_friday_mid_cutoff_is_polled_on_both_sides(sweep_firings):
    """MID is Friday 12:00 America/New_York. It needs the same two things a
    CLOSE needs, at the baseline guarantee."""
    for cutoff in _kickoffs(5, (12, 0)):
        assert _execution_lag(sweep_firings, cutoff) <= timedelta(minutes=60), cutoff
        assert _precision_lag(sweep_firings, cutoff) <= timedelta(minutes=60), cutoff


def test_open_discovery_can_see_a_board_appearing_at_any_hour(sweep_firings):
    """OPEN is the first priceable observation in the accumulated archive, so
    what matters is that no hour of the week goes unobserved."""
    by_weekday_hour = {(f.isoweekday(), f.hour) for f in sweep_firings}
    assert len(by_weekday_hour) == 7 * 24


def test_the_monday_rollover_is_polled_after_the_last_monday_kickoff(sweep_firings):
    """Publication hands over once the week's final kickoff has passed, so a
    poll has to follow it."""
    for kickoff in _kickoffs(1, (20, 15)):
        assert _execution_lag(sweep_firings, kickoff + timedelta(hours=4)) <= timedelta(minutes=60)


# ===========================================================================
# 10. No full-day fifteen-minute polling remains.
# ===========================================================================
def test_no_day_is_polled_every_fifteen_minutes_around_the_clock(sweep_firings):
    per_day: dict = {}
    for firing in sweep_firings:
        per_day.setdefault(firing.date(), []).append(firing)

    for day, firings in per_day.items():
        dense_hours = sum(
            1 for hour in range(24) if sum(1 for f in firings if f.hour == hour) >= 4
        )
        assert dense_hours <= 14, f"{day}: {dense_hours} hours of 15-minute polling"
        assert len(firings) <= 70, f"{day}: {len(firings)} firings"


def test_the_busiest_day_is_materially_lighter_than_before(sweep_firings):
    """96 was the old Sunday/Monday/Thursday figure."""
    per_day: dict = {}
    for firing in sweep_firings:
        per_day.setdefault(firing.date(), 0)
        per_day[firing.date()] += 1
    assert max(per_day.values()) < 96


def test_the_weekly_firing_budget_is_reduced(crons, sweep_crons):
    """A full in-season week, counted from the real expansion rather than
    from arithmetic on the cron strings."""
    # A Sunday-to-Saturday week entirely inside the season and inside EDT.
    week_start = datetime(2026, 9, 20, tzinfo=timezone.utc)
    week_end = week_start + timedelta(days=7)

    old_sweep = _firings(["*/15 * * 9,10,11,12,1 0,1,4", "25 * * 9,10,11,12,1 2,3,5,6"],
                         week_start, week_end)
    new_sweep = _firings(sweep_crons, week_start, week_end)
    all_new = _firings(crons, week_start, week_end)

    assert len(old_sweep) == 384
    assert len(new_sweep) == 243
    assert len(all_new) == 254  # + 7 daily + 4 certified
    assert len(new_sweep) / len(old_sweep) < 0.65


def test_the_baseline_and_the_dense_windows_never_collide(sweep_crons):
    """Two crons matching the same minute would trigger two runs for one
    instant. The baseline sits at :25 and the windows at :00/:15/:30/:45."""
    baseline = [c for c in sweep_crons if c.startswith("25 ")]
    dense = [c for c in sweep_crons if c.startswith("*/15 ")]
    assert baseline and dense
    assert _field("25", 0, 59).isdisjoint(_field("*/15", 0, 59))


# ===========================================================================
# 1-2. Daily and certified instants are untouched.
# ===========================================================================
def test_the_daily_maintenance_cron_is_unchanged(crons):
    assert DAILY_CRON in crons
    assert sum(1 for c in crons if c == DAILY_CRON) == 1


def test_the_certified_tue_fri_crons_are_unchanged(crons):
    for cron in CERTIFIED_CRONS:
        assert cron in crons


def test_the_daily_pass_still_fires_once_every_day_of_the_year():
    firings = _firings([DAILY_CRON], SEASON_START, SEASON_END)
    days = {f.date() for f in firings}
    expected_days = (SEASON_END.date() - SEASON_START.date()).days
    assert len(firings) == len(days) == expected_days


# ===========================================================================
# The daily pass runs once a day, not once a poll.
# ===========================================================================
def _run_gate(workflow: dict, tmp_path: Path, *, triggering_cron: str | None, mode: str = "") -> dict:
    step = next(s for s in workflow["jobs"]["resolve"]["steps"] if s.get("id") == "gate")
    script = tmp_path / "gate.sh"
    script.write_text(step["run"], encoding="utf-8")
    output = tmp_path / "out"
    output.write_text("", encoding="utf-8")

    env = {
        "PATH": os.environ["PATH"],
        "MODE": mode,
        "DAILY_DUE": "true",
        "CERTIFIED_DUE": "false",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_OUTPUT": str(output),
        "DAILY_CRON": DAILY_CRON,
    }
    if triggering_cron is not None:
        env["TRIGGERING_CRON"] = triggering_cron

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )


def test_the_daily_firing_runs_the_daily_pass(workflow, tmp_path):
    assert _run_gate(workflow, tmp_path, triggering_cron=DAILY_CRON)["run_daily"] == "true"


@pytest.mark.parametrize(
    "cron",
    ["25 * * 9,10,11,12,1 *", "*/15 11-21 * 9,10,11,12,1 0", "5 16 * * 2,5"],
)
def test_every_other_firing_does_not_repeat_the_daily_pass(workflow, tmp_path, cron):
    assert _run_gate(workflow, tmp_path, triggering_cron=cron)["run_daily"] == "false"


def test_a_snapshot_poll_still_runs_the_sweep(workflow, tmp_path):
    outputs = _run_gate(workflow, tmp_path, triggering_cron="25 * * 9,10,11,12,1 *")
    assert outputs["run_snapshot_sweep"] == "true"
    assert outputs["publish_mode"] == "publish"


def test_a_manual_dispatch_still_gets_the_daily_pass(workflow, tmp_path):
    """No inputs context means no triggering cron, and an operator asking for
    the daily pass by hand must still get it."""
    assert _run_gate(workflow, tmp_path, triggering_cron=None)["run_daily"] == "true"
    assert _run_gate(workflow, tmp_path, triggering_cron="")["run_daily"] == "true"
    assert _run_gate(workflow, tmp_path, triggering_cron=None, mode="daily_only")["run_daily"] == "true"


def test_the_gate_reads_the_cron_github_actually_supplies(workflow):
    step = next(s for s in workflow["jobs"]["resolve"]["steps"] if s.get("id") == "gate")
    assert step["env"]["TRIGGERING_CRON"] == "${{ github.event.schedule }}"
    assert step["env"]["DAILY_CRON"] == DAILY_CRON


def test_the_declared_daily_cron_is_a_real_schedule_entry(workflow, crons):
    """A typo here would silently stop the daily pass running at all."""
    step = next(s for s in workflow["jobs"]["resolve"]["steps"] if s.get("id") == "gate")
    assert step["env"]["DAILY_CRON"] in crons


# ===========================================================================
# 9, 11. Nothing about the production contract moved.
# ===========================================================================
def test_the_sweep_job_and_its_publication_wiring_are_unchanged(workflow):
    sweep = workflow["jobs"]["snapshot-sweep"]
    assert sweep["env"]["PUBLISH_MODE"] == "${{ needs.resolve.outputs.publish_mode }}"
    verify = next(
        s for s in sweep["steps"] if "verify_public_nfl_feed.py" in str(s.get("run", ""))
    )
    assert "needs.resolve.outputs.publish_mode == 'publish'" in verify["if"]
    assert "published_sha256 != ''" in verify["if"]


def test_the_sweep_still_runs_its_own_evidence_population_and_recalibration():
    """Reducing the daily pass to once a day is only safe because the sweep
    does this work itself on every poll."""
    text = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    for entrypoint in (
        "refresh_bdl_2026_games_evidence.py",
        "update_2026_games_population.py",
        "generate_2026_recalibration_candidate.py --promote-if-eligible",
        "attach_2026_results_from_population.py",
        "publish_2026_current_week.py",
    ):
        assert entrypoint in text, entrypoint


def test_no_stage_definition_or_market_rule_was_touched():
    import pandas as pd

    from nfl_hybrid.data import bdl_market_bridge as bridge
    from nfl_hybrid.production import snapshot_stages_2026 as st

    assert st.CLOSE_LEAD == pd.Timedelta(minutes=60)
    assert CLOSE_LEAD == st.CLOSE_LEAD.to_pytimedelta()
    assert bridge.PRODUCTION_HORIZONS == ("TUE", "FRI")
    assert bridge.STAGE_HORIZONS == (bridge.STAGE_HORIZON,)


def test_the_sweep_control_flow_and_no_op_behaviour_are_unchanged():
    text = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    span = text[
        text.index('echo "due_batches=') : text.index("=== 7. current-week publication ===")
    ]
    code = "\n".join(line for line in span.splitlines() if not line.strip().startswith("#"))
    assert "exit 0" not in code
    assert 'echo "stage_execution=NOOP_NOTHING_DUE"' in text
    assert 'echo "snapshot_action=NOOP_NOTHING_DUE"' in text


def test_the_concurrency_group_still_serialises_production(workflow):
    assert workflow["concurrency"]["group"] == "nfl-2026-production"
    assert workflow["concurrency"]["cancel-in-progress"] is False


# ===========================================================================
# Serialized execution, which cron spacing alone says nothing about.
#
#   EVERYTHING ABOVE MODELS TRIGGER TIMES. A cron that fires every fifteen
#   minutes does not give fifteen-minute execution when production is
#   serialised and a run takes longer than the interval -- the queue simply
#   grows and the CLOSE the window exists to catch is executed late. The
#   assertions in this section are the ones that make the timing guarantees
#   above mean something in wall-clock terms.
#
#   The budgets are measured, not assumed. Across live runs 36053079115,
#   36056606567 and 36082933560 the sweep body took 17.0 / 17.1 / 17.9
#   minutes, of which stage 5 recalibration was 7.0 / 7.1 / 7.3. Job setup
#   (checkout, Python, SSH, bootstrap) and the public-bytes verification add
#   roughly three more.
# ===========================================================================
MEASURED_SWEEP_BODY = timedelta(minutes=18)
MEASURED_RECALIBRATION = timedelta(minutes=7)
JOB_OVERHEAD = timedelta(minutes=3)

IDLE_RUN_BUDGET = MEASURED_SWEEP_BODY - MEASURED_RECALIBRATION + JOB_OVERHEAD  # 14 min
EXECUTING_RUN_BUDGET = MEASURED_SWEEP_BODY + JOB_OVERHEAD  # 21 min

DENSE_INTERVAL = timedelta(minutes=15)


def test_an_idle_sweep_fits_inside_the_dense_interval():
    """The point of the whole amendment. At the old 21-minute idle cost a
    fifteen-minute window could never drain; at 14 it can."""
    assert IDLE_RUN_BUDGET < DENSE_INTERVAL
    assert EXECUTING_RUN_BUDGET > DENSE_INTERVAL  # honest: these still lag


def test_the_busiest_hour_can_actually_drain(sweep_firings):
    """Serialized demand in the densest hour, against the sixty minutes that
    hour has. Computed from the real expansion, not from the cron interval."""
    per_hour: dict = {}
    for firing in sweep_firings:
        key = firing.replace(minute=0, second=0, microsecond=0)
        per_hour[key] = per_hour.get(key, 0) + 1

    busiest = max(per_hour.values())
    assert busiest == 4
    assert busiest * IDLE_RUN_BUDGET < timedelta(hours=1)


def test_the_busiest_day_can_actually_drain(sweep_firings):
    """A day is the horizon that matters for backlog: an hour can borrow from
    the next, a day cannot borrow from the next game day."""
    per_day: dict = {}
    for firing in sweep_firings:
        per_day[firing.date()] = per_day.get(firing.date(), 0) + 1

    busiest = max(per_day.values())
    assert busiest == 63  # Sunday
    assert busiest * IDLE_RUN_BUDGET < timedelta(hours=24)
    # And the old schedule could not, which is why the queue never drained.
    assert 96 * (MEASURED_SWEEP_BODY + JOB_OVERHEAD) > timedelta(hours=24)


def test_the_idle_saving_is_structural_and_not_just_asserted_here():
    """The budget above is only real because the script actually skips the
    expensive step when nothing executed."""
    text = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    stage5 = text[text.index("# --- 5. recalibration") : text.index("# NOTHING NEW TO EXECUTE")]
    code = "\n".join(line for line in stage5.splitlines() if not line.strip().startswith("#"))

    assert 'if [ "${due_batches}" -eq 0 ]; then' in code
    assert "recalibration=SKIPPED_NOTHING_EXECUTED" in code
    assert "generate_2026_recalibration_candidate.py --promote-if-eligible" in code
    # The due count is known before the decision is taken.
    assert text.index('echo "due_batches=') < text.index("# --- 5. recalibration")


def test_the_skip_cannot_reach_the_things_a_later_close_depends_on():
    """Evidence, population and the market capture all sit before the due
    count is even computed, so no gate can accidentally cover them."""
    text = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    decision = text.index('echo "due_batches=')
    for essential in (
        "refresh_bdl_2026_games_evidence.py",
        "update_2026_games_population.py",
        "create_official_capture.sh",
        "run_2026_stage_snapshots.py",
    ):
        assert text.index(essential) < decision, essential


def test_only_recalibration_is_conditional_on_the_due_count():
    """Publication, reporting and retention must stay unconditional -- an
    idle sweep is exactly the sweep that still owes the public feed."""
    text = (REPO_ROOT / "ops" / "wizard" / "run_stage_snapshots.sh").read_text(encoding="utf-8")
    after_stage5 = text[text.index("# --- 6. results"):]
    code = "\n".join(
        line for line in after_stage5.splitlines() if not line.strip().startswith("#")
    )
    # The only due_batches test after stage 5 is the snapshot_action label.
    assert code.count('"${due_batches}" -eq 0') == 1
    assert "snapshot_action=NOOP_NOTHING_DUE" in code
    for unconditional in (
        "attach_2026_results_from_population.py",
        "report_2026_snapshot_performance.py",
        "publish_2026_current_week.py",
        "publish_wizard_nfl_local.py",
        "prune_replaceable_artifacts.py",
    ):
        assert unconditional in code, unconditional
