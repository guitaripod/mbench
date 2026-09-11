import json

from .paths import ASSETS

VERSION = "1"
CONCURRENCY = 4
INDEX_TASKS = ("mmlupro", "aime", "lcb", "niah", "tools")
QUALITY_TASKS = ("niah", "mmlupro", "aime", "tools", "lcb")
TASK_LABELS = {
    "mmlupro": "MMLU-Pro",
    "aime": "AIME 2025",
    "lcb": "LiveCodeBench v6",
    "niah": "Needle retrieval",
    "tools": "Tool calls",
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
        "mmlupro": {"per_category": 30, "max_tokens": 32768},
        "aime": {"samples": 4, "max_tokens": 65536},
        "lcb": {"limit": 100, "max_tokens": 65536},
        "niah": {"depths": (16000, 32000, 64000, 120000), "per_depth": 15, "max_tokens": 8192},
        "tools": {"repeats": 3, "max_tokens": 4096},
    },
    "quick": {
        "speed": {"reps": 1, "max_tokens": 1024, "concurrency": (1, 4), "rounds": 1,
                  "depths": (1000, 32000, 120000), "depth_reps": 1, "depth_max_tokens": 512},
        "mmlupro": {"per_category": 10, "max_tokens": 32768},
        "aime": {"samples": 1, "max_tokens": 65536},
        "lcb": {"limit": 40, "max_tokens": 65536},
        "niah": {"depths": (16000, 64000, 120000), "per_depth": 5, "max_tokens": 8192},
        "tools": {"repeats": 1, "max_tokens": 4096},
    },
    "smoke": {
        "speed": {"reps": 1, "max_tokens": 256, "concurrency": (1,), "rounds": 1,
                  "depths": (1000,), "depth_reps": 1, "depth_max_tokens": 128},
        "mmlupro": {"per_category": 1, "max_tokens": 8192, "max_items": 6},
        "aime": {"samples": 1, "max_tokens": 16384, "max_items": 2},
        "lcb": {"limit": 100, "max_tokens": 16384, "max_items": 3},
        "niah": {"depths": (16000,), "per_depth": 2, "max_tokens": 4096},
        "tools": {"repeats": 1, "max_tokens": 2048, "max_items": 6},
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
