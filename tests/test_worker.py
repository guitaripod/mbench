from mbench import paths, store, worker


def test_sessions_that_fail_without_gaining_an_answer_are_counted_in_a_row(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "DB", tmp_path / "bench.db")
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "supergpqa.jsonl").write_text("{}\n" * 521)
    (run_dir / "tools.jsonl").write_text("{}\n" * 97)
    (run_dir / "events.jsonl").write_text("{}\n" * 50)
    db = store.connect()
    store.insert_run(db, {"id": "r1", "model": "m", "suite": "full/v1", "status": "running", "flags": {"tasks": ["tools"]}})
    assert worker.answers_kept(run_dir) == 618
    assert [worker.stalled(db, "r1", run_dir) for _ in range(3)] == [1, 2, 3]
    (run_dir / "tools.jsonl").write_text("{}\n" * 98)
    assert worker.stalled(db, "r1", run_dir) == 1
    flags = store.get_run(db, "r1")["flags"]
    assert flags["tasks"] == ["tools"] and flags["stall"] == {"answers": 619, "count": 1}
