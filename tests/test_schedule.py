from datetime import datetime

import pytest

from mbench import paths, schedule, store, units

NIGHT = {"start": "03:00", "end": "09:00"}
LATE = {"start": "23:00", "end": "07:00"}


def test_windows_including_ones_across_midnight():
    assert schedule.in_window(datetime(2026, 9, 12, 3, 0), NIGHT)
    assert not schedule.in_window(datetime(2026, 9, 12, 9, 0), NIGHT)
    assert schedule.in_window(datetime(2026, 9, 12, 23, 30), LATE) and schedule.in_window(datetime(2026, 9, 12, 6, 59), LATE)
    assert not schedule.in_window(datetime(2026, 9, 12, 12, 0), LATE)
    assert schedule.in_window(datetime(2026, 9, 12, 12, 0), {"start": "03:00", "end": None})


def test_first_start_is_the_next_window_or_now_inside_one():
    evening = datetime(2026, 9, 11, 18, 40)
    assert schedule.first_start(evening, NIGHT) == datetime(2026, 9, 12, 3, 0)
    assert schedule.first_start(datetime(2026, 9, 12, 4, 0), NIGHT) == datetime(2026, 9, 12, 4, 0)
    assert schedule.first_start(datetime(2026, 9, 12, 4, 0), {"start": "03:00", "end": None}) == datetime(2026, 9, 13, 3, 0)
    assert schedule.first_start(evening, NIGHT, datetime(2026, 9, 14, 3, 0)) == datetime(2026, 9, 14, 3, 0)


def test_parsing_times():
    assert schedule.parse_at("3:00") == (None, "03:00")
    assert schedule.parse_at("2026-09-12 03:30") == (datetime(2026, 9, 12, 3, 30), "03:30")
    with pytest.raises(ValueError, match="not a time"):
        schedule.parse_at("3am")


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "DB", tmp_path / "bench.db")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    actions = []
    monkeypatch.setattr(units, "spawn", lambda run_id, foreground: actions.append(("spawn", run_id)))
    monkeypatch.setattr(units, "stop", lambda run_id: actions.append(("stop", run_id)))
    monkeypatch.setattr(units, "unit_active", lambda run_id: True)
    monkeypatch.setattr(schedule.swap, "unload", lambda: actions.append(("unload",)))
    monkeypatch.setattr(schedule, "remove_timer", lambda: actions.append(("timer off",)))
    return store.connect(), actions


def add(db, run_id, status, not_before, window=NIGHT, created=1.0):
    store.insert_run(db, {"id": run_id, "model": run_id, "suite": "full/v2", "effort": "medium", "status": status,
                          "created": created, "flags": {"not_before": not_before.timestamp(), "window": window}})


def test_due_runs_start_one_at_a_time_in_order(world):
    db, actions = world
    night = datetime(2026, 9, 12, 3, 0)
    add(db, "second", "scheduled", night, created=2.0)
    add(db, "first", "scheduled", night, created=1.0)
    assert schedule.tick(db, datetime(2026, 9, 12, 2, 55)) is None
    assert schedule.tick(db, datetime(2026, 9, 12, 3, 0)) == "first"
    assert store.get_run(db, "first")["status"] == "queued"
    assert schedule.tick(db, datetime(2026, 9, 12, 3, 5)) is None
    store.update_run(db, "first", status="complete")
    assert schedule.tick(db, datetime(2026, 9, 12, 5, 0)) == "second"
    assert actions == [("spawn", "first"), ("spawn", "second"), ("timer off",)]


def test_a_run_still_going_when_the_window_closes_pauses_until_the_next_night(world):
    db, actions = world
    add(db, "long", "running", datetime(2026, 9, 12, 3, 0))
    assert schedule.tick(db, datetime(2026, 9, 12, 9, 0)) is None
    run = store.get_run(db, "long")
    assert run["status"] == "scheduled" and run["flags"]["paused"] == 1
    assert run["flags"]["not_before"] == datetime(2026, 9, 13, 3, 0).timestamp()
    assert actions == [("stop", "long"), ("unload",)]
    assert schedule.tick(db, datetime(2026, 9, 13, 3, 0)) == "long"


def test_a_missed_window_waits_for_the_next_one_instead_of_starting_by_day(world):
    db, actions = world
    add(db, "missed", "scheduled", datetime(2026, 9, 12, 3, 0))
    assert schedule.tick(db, datetime(2026, 9, 12, 14, 0)) is None
    assert store.get_run(db, "missed")["flags"]["not_before"] == datetime(2026, 9, 13, 3, 0).timestamp()


def test_a_run_without_an_end_catches_up_whenever_it_is_due(world):
    db, actions = world
    add(db, "open", "scheduled", datetime(2026, 9, 12, 3, 0), window=None)
    assert schedule.tick(db, datetime(2026, 9, 12, 14, 0)) == "open"


def test_the_timer_unit_runs_this_interpreter():
    files = schedule.unit_files()
    assert "-m mbench.cli tick" in files["mbench-tick.service"]
    assert "OnCalendar=*:0/5" in files["mbench-tick.timer"]
