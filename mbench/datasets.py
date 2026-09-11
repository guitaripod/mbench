import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import paths
from .tool_cases import CASES, SYSTEM, TOOLS

LETTERS = "ABCDEFGHIJ"
SOURCES = {
    "mmlupro": {
        "repo": "TIGER-Lab/MMLU-Pro", "file": "data/test-00000-of-00001.parquet", "kind": "dataset",
        "sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    },
    "aime": {
        "repo": "MathArena/aime_2025", "file": "data/train-00000-of-00001.parquet", "kind": "dataset",
        "sha256": "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d",
    },
    "lcb": {
        "repo": "livecodebench/code_generation_lite", "file": "test6.jsonl", "kind": "dataset",
        "sha256": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
    },
    "tokenizer": {
        "repo": "openai/gpt-oss-120b", "file": "tokenizer.json", "kind": "model",
        "sha256": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    },
}
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)
LCB_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
LCB_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT."
)
ADJECTIVES = (
    "crimson", "silent", "amber", "frozen", "hollow", "golden", "restless", "velvet", "iron", "misty",
    "scarlet", "quiet", "copper", "bright", "shadow", "ivory", "rapid", "gentle", "stormy", "lunar",
)
NOUNS = (
    "otter", "falcon", "harbor", "lantern", "meadow", "glacier", "compass", "orchard", "beacon", "canyon",
    "sparrow", "quarry", "willow", "summit", "anchor", "thistle", "ember", "reef", "badger", "prism",
)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_path(name):
    return paths.CACHE / "datasets" / name / Path(SOURCES[name]["file"]).name


def fetch(name):
    """Returns the pinned file, downloading it once; a changed upstream file is an error, not an update."""
    source = SOURCES[name]
    target = cached_path(name)
    if target.exists() and sha256_of(target) == source["sha256"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(source["repo"], source["file"], repo_type=source["kind"], cache_dir=paths.CACHE / "hf"))
    actual = sha256_of(downloaded)
    if actual != source["sha256"]:
        raise RuntimeError(f"{name}: upstream file changed (sha256 {actual[:12]}, pinned {source['sha256'][:12]}); "
                           "pinning it again means a new suite version")
    shutil.copyfile(downloaded, target)
    return target


def item(task, item_id, messages, gold, meta, max_tokens, sample=0, tools=None):
    return {"id": item_id, "task": task, "messages": messages, "gold": gold, "meta": meta,
            "max_tokens": max_tokens, "sample": sample, "tools": tools}


def mmlupro(spec, _context=None):
    frame = pd.read_parquet(fetch("mmlupro"))
    rng = random.Random(0)
    items = []
    for category, group in sorted(frame.groupby("category"), key=lambda pair: pair[0]):
        rows = group.to_dict("records")
        rng.shuffle(rows)
        for row in rows[: spec["per_category"]]:
            options = "\n".join(f"({LETTERS[index]}) {text}" for index, text in enumerate(row["options"]))
            prompt = (
                f"The following is a multiple choice question about {category}. Think it through, then finish with "
                "a final line of the form 'Answer: X', where X is the letter of the correct option.\n\n"
                f"Question: {row['question']}\n\nOptions:\n{options}"
            )
            items.append(item("mmlupro", f"mmlupro-{row['question_id']}", [{"role": "user", "content": prompt}],
                              row["answer"], {"category": category}, spec["max_tokens"]))
    return items


def aime(spec, _context=None):
    frame = pd.read_parquet(fetch("aime"))
    items = []
    for row in frame.to_dict("records"):
        prompt = f"{row['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for sample in range(spec["samples"]):
            items.append(item("aime", f"aime-{row['problem_idx']}-s{sample}", [{"role": "user", "content": prompt}],
                              str(int(row["answer"])), {"problem": int(row["problem_idx"])}, spec["max_tokens"], sample))
    return items


def lcb_prompt(row):
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += f"### Format: {LCB_WITH_STARTER}\n```python\n{row['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def lcb(spec, _context=None):
    with fetch("lcb").open() as handle:
        rows = [json.loads(line) for line in handle]
    if spec["limit"] < len(rows):
        rng = random.Random(0)
        by_difficulty = defaultdict(list)
        for row in rows:
            by_difficulty[row["difficulty"]].append(row)
        chosen = []
        for _, group in sorted(by_difficulty.items()):
            rng.shuffle(group)
            chosen += group[: round(spec["limit"] * len(group) / len(rows))]
        rows = chosen
    return [item("lcb", f"lcb-{row['question_id']}",
                 [{"role": "system", "content": LCB_SYSTEM}, {"role": "user", "content": lcb_prompt(row)}],
                 None, {"question_id": row["question_id"], "difficulty": row["difficulty"]}, spec["max_tokens"])
            for row in rows]


def niah(spec, context=None):
    """Ten vault codes hidden in MMLU-Pro question text; the model must return three of them. Depths the model's window can't hold are skipped."""
    from tokenizers import Tokenizer

    depths = [depth for depth in spec["depths"] if context is None or depth + spec["max_tokens"] + 2048 <= context]
    tokenizer = Tokenizer.from_file(str(fetch("tokenizer")))
    paragraphs = [text.strip() for text in pd.read_parquet(fetch("mmlupro"))["question"].tolist() if len(text) > 200]
    lengths = [len(encoding.ids) for encoding in tokenizer.encode_batch(paragraphs)]
    rng = random.Random(0)
    items = []
    for depth in spec["depths"]:
        for sample in range(spec["per_depth"]):
            order = list(range(len(paragraphs)))
            rng.shuffle(order)
            picked, total = [], 0
            for index in order:
                if total + lengths[index] > depth - 300:
                    continue
                picked.append(paragraphs[index])
                total += lengths[index]
                if total > depth - 1200:
                    break
            keys = set()
            while len(keys) < 10:
                keys.add(f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}")
            keys = sorted(keys)
            codes = {key: str(rng.randint(1_000_000, 9_999_999)) for key in keys}
            for key in keys:
                picked.insert(rng.randint(0, len(picked)), f"The access code for the {key} vault is {codes[key]}.")
            asked = rng.sample(keys, 3)
            if depth not in depths:
                continue
            prompt = (
                "Below is a long collection of notes. Somewhere among them are statements giving the access codes "
                "for several vaults.\n\n<notes>\n" + "\n\n".join(picked) + "\n</notes>\n\n"
                f"What are the access codes for the {asked[0]}, {asked[1]} and {asked[2]} vaults? "
                "Reply with exactly three lines in the form '<vault>: <code>'."
            )
            items.append(item("niah", f"niah-{depth}-{sample}", [{"role": "user", "content": prompt}],
                              {key: codes[key] for key in asked}, {"depth": depth}, spec["max_tokens"]))
    return items


def tools(spec, _context=None):
    items = []
    for repeat in range(spec["repeats"]):
        for index, (prompt, expected) in enumerate(CASES):
            items.append(item("tools", f"tools-{index}-r{repeat}",
                              [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              [[name, arguments] for name, arguments in expected], {"case": index},
                              spec["max_tokens"], repeat, TOOLS))
    return items


BUILDERS = {"mmlupro": mmlupro, "aime": aime, "lcb": lcb, "niah": niah, "tools": tools}


def build(task, spec, context=None):
    items = BUILDERS[task](spec, context)
    return items[: spec["max_items"]] if spec.get("max_items") else items
