import json
import time

from . import metrics as scores
from . import paths, store, suite

KINDS = ("full", "quick")
EFFORT_ORDER = ("medium", "max", "high", "xhigh", "low", "min", "minimal", "none")
DEFAULT_EFFORT = "medium"
TEMPLATE = paths.PACKAGE / "templates" / "leaderboard.html"
PART_PREFIXES = (".bin.", ".part.")


def kind_of(run):
    return suite.kind_of(run["suite"])


def effort_of(run):
    return run.get("effort") or DEFAULT_EFFORT


def headline(runs, effort=DEFAULT_EFFORT):
    """For one effort, the newest complete full run represents a model; a quick run stands in, marked provisional, until one exists."""
    best = {}
    for run in runs:
        kind = kind_of(run)
        if kind not in KINDS or effort_of(run) != effort:
            continue
        current = best.get(run["model"])
        if current is None or KINDS.index(kind) < KINDS.index(kind_of(current)):
            best[run["model"]] = run
    return best


def task_entry(metrics, task):
    score = metrics.get(f"{task}.score")
    if not score:
        return None
    parts = {key.split(".", 2)[2]: entry["value"] for key, entry in metrics.items()
             if key.startswith(task) and any(key.startswith(task + prefix) for prefix in PART_PREFIXES)}
    return {**score, **{field: (metrics.get(f"{task}.{field}") or {}).get("value")
                        for field in ("tokens", "latency", "truncated", "unreachable", "malformed")},
            "parts": parts}


def warnings_for(tasks, capacity, labels):
    """Server settings that cap scores, said out loud so a low long-context score isn't read as the model's own."""
    out_of_reach = {task: entry["unreachable"] for task, entry in tasks.items() if entry.get("unreachable")}
    if not out_of_reach:
        return []
    limit = capacity.get("context")
    reach = f"{limit // 1024}k tokens per request" if limit else "the server's context window"
    return [f"Capped at {reach}: " + ", ".join(f"{labels.get(task, task)} {share:.0f}% out of reach"
                                              for task, share in out_of_reach.items())]


def model_entry(db, run, history, definition):
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
        for entry in store.submissions_of(db, candidate["id"]):
            if entry["kind"].startswith("speed."):
                detail = json.loads(entry["detail"]) if entry.get("detail") else {}
                submissions.append({"kind": entry["kind"], "id": entry["remote_id"], "value": entry["value"],
                                    "run": candidate["id"], "verified": detail.get("verified")})
    tasks = {task: entry for task in definition["index_tasks"] if (entry := task_entry(metrics, task))}
    capacity = (run.get("server") or {}).get("capacity") or {}
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
            "effort": effort_of(run), "harness": run.get("harness"), "contended": flags.get("contended") or [],
            "failedItems": flags.get("failed_items") or 0, "notes": flags.get("notes"),
            "reused": flags.get("reused"),
        },
        "index": metrics.get("index.quality"),
        "tasks": tasks,
        "capacity": capacity,
        "warnings": warnings_for(tasks, capacity, definition["labels"]),
        "energy": {"perCorrect": metrics.get("energy.per_correct"), "quality": metrics.get("energy.quality")},
        "speed": {key.removeprefix("speed."): entry for key, entry in metrics.items() if key.startswith("speed.")},
        "lmx": lmx_scores,
        "submissions": submissions,
        "server": run.get("server") or {},
        "history": [
            {"id": entry["id"], "suite": entry["suite"], "effort": effort_of(entry), "finished": entry.get("finished"),
             "index": (store.metrics_of(db, entry["id"]).get("index.quality") or {}).get("value"),
             "decode": (store.metrics_of(db, entry["id"]).get("speed.decode") or {}).get("value")}
            for entry in history
        ],
    }


def effort_key(name):
    return (EFFORT_ORDER.index(name) if name in EFFORT_ORDER else len(EFFORT_ORDER), name)


def suite_view(db, runs, version):
    """One suite version's rankings; versions measure different tasks, so their indexes are never ranked together."""
    definition = suite.DEFINITIONS[version]
    rankings = {}
    for effort in sorted({effort_of(run) for run in runs}, key=effort_key):
        chosen = headline(runs, effort)
        if chosen:
            rankings[effort] = [model_entry(db, run, [entry for entry in runs if entry["model"] == model_id], definition)
                                for model_id, run in chosen.items()]
    return {
        "indexTasks": list(definition["index_tasks"]),
        "taskLabels": definition["labels"],
        "taskShort": definition["short"],
        "taskNotes": definition["notes"],
        "groups": {name: list(tasks) for name, tasks in definition["groups"].items()},
        "efforts": list(rankings),
        "rankings": rankings,
        "health": {effort: scores.task_health(models, definition["index_tasks"]) for effort, models in rankings.items()},
    }


def collect(db):
    runs = store.list_runs(db, status="complete")
    ranked = [run for run in runs if kind_of(run) in KINDS and suite.version_of(run["suite"]) in suite.DEFINITIONS]
    versions = sorted({suite.version_of(run["suite"]) for run in ranked} | {suite.VERSION}, key=int, reverse=True)
    suites = {version: suite_view(db, [run for run in ranked if suite.version_of(run["suite"]) == version], version)
              for version in versions}
    latest = ranked[0] if ranked else None
    return {
        "generated": time.time(),
        "suite": suite.label("full"),
        "current": suite.VERSION,
        "hardware": (latest or {}).get("hardware") or {},
        "suites": suites,
        **{key: suites[suite.VERSION][key] for key in ("indexTasks", "taskLabels", "efforts", "rankings")},
        "database": str(paths.DB),
    }


def build():
    db = store.connect()
    data = collect(db)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    paths.BOARD.parent.mkdir(parents=True, exist_ok=True)
    paths.BOARD.write_text(html)
    return paths.BOARD
