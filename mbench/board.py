import json
import time

from . import paths, store, suite

KINDS = ("full", "quick", "legacy")
RANKED_EFFORT = "medium"
TEMPLATE = paths.PACKAGE / "templates" / "leaderboard.html"


def kind_of(run):
    return run["suite"].split("/")[0]


def headline(runs):
    """The newest complete medium-effort full run represents a model; quick or legacy runs stand in until one exists. Other efforts show in history only."""
    best = {}
    for run in runs:
        kind = kind_of(run)
        if kind not in KINDS or (run.get("effort") or RANKED_EFFORT) != RANKED_EFFORT:
            continue
        current = best.get(run["model"])
        if current is None or KINDS.index(kind) < KINDS.index(kind_of(current)):
            best[run["model"]] = run
    return best


def task_entry(metrics, task):
    score = metrics.get(f"{task}.score")
    if not score:
        return None
    return {**score, **{field: (metrics.get(f"{task}.{field}") or {}).get("value")
                        for field in ("tokens", "latency", "truncated")}}


def model_entry(db, run, history):
    metrics = store.metrics_of(db, run["id"])
    profile = run.get("profile") or {}
    flags = run.get("flags") or {}
    lmx_scores = {}
    submissions = []
    for candidate in history:
        candidate_metrics = metrics if candidate["id"] == run["id"] else store.metrics_of(db, candidate["id"])
        for dataset in ("gsm8k", "hellaswag"):
            if dataset not in lmx_scores and f"lmx.{dataset}" in candidate_metrics:
                lmx_scores[dataset] = candidate_metrics[f"lmx.{dataset}"]
        submissions += [{"kind": entry["kind"], "id": entry["remote_id"], "value": entry["value"], "run": candidate["id"]}
                        for entry in store.submissions_of(db, candidate["id"]) if entry["kind"].startswith("speed.")]
    return {
        "id": run["model"],
        "name": run.get("name") or run["model"],
        "engine": profile.get("engine"),
        "quantization": profile.get("quantization"),
        "spec": (profile.get("spec") or {}).get("method"),
        "context": profile.get("context"),
        "fingerprint": run.get("fingerprint"),
        "run": {
            "id": run["id"], "suite": run["suite"], "kind": kind_of(run), "finished": run.get("finished"),
            "effort": run.get("effort"), "harness": run.get("harness"), "contended": flags.get("contended") or [],
            "failedItems": flags.get("failed_items") or 0, "notes": flags.get("notes"),
        },
        "index": metrics.get("index.quality"),
        "tasks": {task: entry for task in suite.INDEX_TASKS if (entry := task_entry(metrics, task))},
        "speed": {key.removeprefix("speed."): entry for key, entry in metrics.items() if key.startswith("speed.")},
        "lmx": lmx_scores,
        "submissions": submissions,
        "server": run.get("server") or {},
        "history": [
            {"id": entry["id"], "suite": entry["suite"], "finished": entry.get("finished"),
             "index": (store.metrics_of(db, entry["id"]).get("index.quality") or {}).get("value"),
             "decode": (store.metrics_of(db, entry["id"]).get("speed.decode") or {}).get("value")}
            for entry in history
        ],
    }


def collect(db):
    runs = store.list_runs(db, status="complete")
    ranked = [run for run in runs if kind_of(run) in KINDS]
    models = []
    for model_id, run in headline(ranked).items():
        history = [entry for entry in ranked if entry["model"] == model_id]
        models.append(model_entry(db, run, history))
    latest = ranked[0] if ranked else None
    return {
        "generated": time.time(),
        "suite": suite.label("full"),
        "hardware": (latest or {}).get("hardware") or {},
        "indexTasks": list(suite.INDEX_TASKS),
        "taskLabels": suite.TASK_LABELS,
        "models": models,
        "database": str(paths.DB),
    }


def build():
    db = store.connect()
    data = collect(db)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    paths.BOARD.parent.mkdir(parents=True, exist_ok=True)
    paths.BOARD.write_text(html)
    return paths.BOARD
