import json
import sqlite3
import time

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  name TEXT,
  suite TEXT NOT NULL,
  effort TEXT,
  status TEXT NOT NULL,
  created REAL,
  started REAL,
  finished REAL,
  harness TEXT,
  fingerprint TEXT,
  profile TEXT,
  server TEXT,
  hardware TEXT,
  flags TEXT,
  error TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value REAL,
  unit TEXT,
  n INTEGER,
  lo REAL,
  hi REAL,
  PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS submissions (
  run_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  remote_id TEXT,
  value REAL,
  detail TEXT,
  created REAL
);
"""
JSON_FIELDS = ("profile", "server", "hardware", "flags")


def connect():
    paths.DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(paths.DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def encode(fields):
    return {key: json.dumps(value) if key in JSON_FIELDS and not isinstance(value, str) else value
            for key, value in fields.items()}


def decode(row):
    if row is None:
        return None
    data = dict(row)
    for key in JSON_FIELDS:
        if data.get(key):
            data[key] = json.loads(data[key])
    return data


def insert_run(db, run):
    run = encode({"created": time.time(), **run})
    columns = ", ".join(run)
    db.execute(f"INSERT INTO runs ({columns}) VALUES ({', '.join('?' * len(run))})", tuple(run.values()))
    db.commit()


def update_run(db, run_id, **fields):
    fields = encode(fields)
    assignments = ", ".join(f"{key} = ?" for key in fields)
    db.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id))
    db.commit()


def get_run(db, run_id):
    return decode(db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())


def list_runs(db, model=None, status=None):
    query, args = "SELECT * FROM runs WHERE 1 = 1", []
    if model:
        query += " AND model = ?"
        args.append(model)
    if status:
        query += " AND status = ?"
        args.append(status)
    return [decode(row) for row in db.execute(query + " ORDER BY created DESC", args)]


def delete_runs(db, run_ids):
    """Forgets runs and everything measured in them. The files each run wrote are the caller's to remove; the database
    is the record, so it goes first and together."""
    marks = ",".join("?" for _ in run_ids)
    with db:
        db.execute(f"DELETE FROM metrics WHERE run_id IN ({marks})", list(run_ids))
        db.execute(f"DELETE FROM runs WHERE id IN ({marks})", list(run_ids))


def last_complete(db, model, exclude=None):
    """The model's most recent finished run, the one a new run's stack is compared against."""
    row = db.execute("SELECT * FROM runs WHERE model = ? AND status = 'complete' AND id != ? ORDER BY created DESC LIMIT 1",
                     (model, exclude or "")).fetchone()
    return decode(row)


def set_metrics(db, run_id, metrics):
    db.executemany(
        "INSERT OR REPLACE INTO metrics (run_id, key, value, unit, n, lo, hi) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(run_id, key, entry.get("value"), entry.get("unit"), entry.get("n"), entry.get("lo"), entry.get("hi"))
         for key, entry in metrics.items()],
    )
    db.commit()


def metrics_of(db, run_id):
    rows = db.execute("SELECT key, value, unit, n, lo, hi FROM metrics WHERE run_id = ?", (run_id,))
    return {row["key"]: {"value": row["value"], "unit": row["unit"], "n": row["n"], "lo": row["lo"], "hi": row["hi"]}
            for row in rows}


def add_submission(db, run_id, kind, remote_id, value=None, detail=None):
    db.execute("INSERT INTO submissions (run_id, kind, remote_id, value, detail, created) VALUES (?, ?, ?, ?, ?, ?)",
               (run_id, kind, remote_id, value, json.dumps(detail) if detail is not None else None, time.time()))
    db.commit()


def submissions_of(db, run_id):
    return [dict(row) for row in db.execute("SELECT * FROM submissions WHERE run_id = ? ORDER BY created", (run_id,))]
