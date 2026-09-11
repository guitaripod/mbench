import json

from .paths import ASSETS

VERSION = "1"
CONCURRENCY = 4
QUALITY_TASKS = ("supergpqa", "math", "tools", "mrcr", "graphwalks", "lcb")
INDEX_TASKS = ("supergpqa", "math", "lcb", "mrcr", "graphwalks", "tools")
INDEX_GROUPS = {
    "Knowledge": ("supergpqa",),
    "Math": ("math",),
    "Code": ("lcb",),
    "Long context": ("mrcr", "graphwalks"),
    "Tool use": ("tools",),
}
TASK_LABELS = {
    "supergpqa": "SuperGPQA",
    "math": "AIME + HMMT 2026",
    "lcb": "LiveCodeBench v6",
    "mrcr": "MRCR 8-needle",
    "graphwalks": "Graphwalks",
    "tools": "Tool use",
}
TASK_SHORT = {"supergpqa": "SuperGPQA", "math": "Math", "lcb": "LCB", "mrcr": "MRCR", "graphwalks": "Graphwalks",
              "tools": "Tools"}
TASK_NOTES = {
    "supergpqa": "Graduate-level multiple choice across 13 disciplines, ten options each.",
    "math": "AIME 2026 and HMMT February 2026 final answers, checked symbolically.",
    "lcb": "Problems from January to April 2025, the newest LiveCodeBench has published; "
           "models trained after mid-2025 may have seen them, so read this as an upper bound.",
    "mrcr": "OpenAI's multi-round coreference test: return the n-th of eight near-identical asks in a long chat. "
            "Lengths the model's context can't hold score zero.",
    "graphwalks": "OpenAI's multi-hop graph test: BFS and parent queries over edge lists of 8k to 64k tokens.",
    "tools": "Multi-step episodes against a simulated workspace, graded by the end state rather than the exact calls.",
}
TASK_VERSIONS = {"speed": 1, "supergpqa": 1, "math": 1, "lcb": 1, "mrcr": 1, "graphwalks": 1, "tools": 1}
DEFINITIONS = {
    VERSION: {
        "index_tasks": INDEX_TASKS,
        "groups": INDEX_GROUPS,
        "labels": TASK_LABELS,
        "short": TASK_SHORT,
        "notes": TASK_NOTES,
        "task_versions": TASK_VERSIONS,
    },
}
PROSE_PROMPT = (
    "Write a long-form article for experienced software engineers explaining how a modern out-of-order CPU core "
    "executes instructions. Cover register renaming, the reorder buffer, reservation stations, branch prediction, "
    "memory disambiguation and how all of it interacts with the cache hierarchy. Use concrete examples throughout."
)

SUITES = {
    "full": {
        "speed": {"reps": 3, "max_tokens": 1024, "concurrency": (1, 2, 4), "rounds": 2,
                  "depths": (1000, 8000, 32000, 64000, 120000, 250000), "depth_reps": 2, "depth_max_tokens": 512},
        "supergpqa": {"questions": 520, "max_tokens": 32768},
        "math": {"samples": 2, "max_tokens": 65536},
        "lcb": {"limit": 100, "max_tokens": 65536},
        "mrcr": {"bins": (16384, 32768, 65536, 131072), "per_bin": 15, "max_tokens": 16384},
        "graphwalks": {"bins": (8192, 16384, 32768, 65536), "per_bin": 6, "max_tokens": 32768},
        "tools": {"repeats": 3, "max_tokens": 8192, "max_steps": 10},
    },
    "quick": {
        "speed": {"reps": 1, "max_tokens": 1024, "concurrency": (1, 4), "rounds": 1,
                  "depths": (1000, 32000, 120000), "depth_reps": 1, "depth_max_tokens": 512},
        "supergpqa": {"questions": 156, "max_tokens": 32768},
        "math": {"samples": 1, "max_tokens": 65536},
        "lcb": {"limit": 40, "max_tokens": 65536},
        "mrcr": {"bins": (16384, 32768, 65536, 131072), "per_bin": 5, "max_tokens": 16384},
        "graphwalks": {"bins": (8192, 16384, 32768, 65536), "per_bin": 2, "max_tokens": 32768},
        "tools": {"repeats": 1, "max_tokens": 8192, "max_steps": 10},
    },
    "smoke": {
        "speed": {"reps": 1, "max_tokens": 256, "concurrency": (1,), "rounds": 1,
                  "depths": (1000,), "depth_reps": 1, "depth_max_tokens": 128},
        "supergpqa": {"questions": 13, "max_tokens": 8192, "max_items": 6},
        "math": {"samples": 1, "max_tokens": 16384, "max_items": 2},
        "lcb": {"limit": 100, "max_tokens": 16384, "max_items": 3},
        "mrcr": {"bins": (16384,), "per_bin": 2, "max_tokens": 8192},
        "graphwalks": {"bins": (8192,), "per_bin": 1, "max_tokens": 8192},
        "tools": {"repeats": 1, "max_tokens": 4096, "max_steps": 8, "max_items": 6},
    },
}


def canonical_prompts():
    """localmaxxing's canonical speed prompts (their sha256 is what its verified-run check expects) plus a prose prompt."""
    entries = json.loads((ASSETS / "canonical.json").read_text())["value"]
    prompts = {entry["id"]: entry["text"] for entry in entries}
    prompts["prose-v1"] = PROSE_PROMPT
    return prompts


def label(suite):
    return f"{suite}/v{VERSION}"


def version_of(label_text):
    """The suite definition a run was measured under."""
    return (label_text or "").rsplit("/v", 1)[-1]


def kind_of(label_text):
    return (label_text or "").split("/")[0]


def reusable(task, from_version):
    """A task's answers carry over from an older suite only when that task's items and scoring did not change."""
    source = DEFINITIONS.get(from_version, {}).get("task_versions", {})
    return task in source and source[task] == TASK_VERSIONS.get(task)
