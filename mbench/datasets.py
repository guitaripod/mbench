import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import paths, tool_cases

SOURCES = {
    "mmlupro": {
        "repo": "TIGER-Lab/MMLU-Pro", "file": "data/test-00000-of-00001.parquet", "kind": "dataset",
        "sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    },
    "supergpqa": {
        "repo": "m-a-p/SuperGPQA", "file": "SuperGPQA-all.jsonl", "kind": "dataset",
        "sha256": "28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910",
    },
    "aime2026": {
        "repo": "MathArena/aime_2026", "file": "data/train-00000-of-00001.parquet", "kind": "dataset",
        "sha256": "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957",
    },
    "hmmt2026": {
        "repo": "MathArena/hmmt_feb_2026", "file": "data/train-00000-of-00001.parquet", "kind": "dataset",
        "sha256": "e5fcff6b1c2262841c0c37bf6d7b42529f284528d5f0d5c45c52e8bf0654a916",
    },
    "lcb": {
        "repo": "livecodebench/code_generation_lite", "file": "test6.jsonl", "kind": "dataset",
        "sha256": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
    },
    "mrcr0": {
        "repo": "openai/mrcr", "file": "8needle/8needle_0.parquet", "kind": "dataset",
        "sha256": "65df601a2e0ae4a3cfb56920a6ef99f26c0de37c6b1018695e8aed684e6a94c1",
    },
    "mrcr1": {
        "repo": "openai/mrcr", "file": "8needle/8needle_1.parquet", "kind": "dataset",
        "sha256": "c80b19573bff1d38e1c157d6a0bdf9cfd1a8ab6372296174c9a7015e164189e3",
    },
    "graphwalks": {
        "repo": "openai/graphwalks", "file": "graphwalks_128k_and_shorter.parquet", "kind": "dataset",
        "sha256": "54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516",
    },
    "tokenizer": {
        "repo": "openai/gpt-oss-120b", "file": "tokenizer.json", "kind": "model",
        "sha256": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    },
}
LETTERS = "ABCDEFGHIJ"
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
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
TEMPLATE_TOKENS = 256
MIN_ANSWER_TOKENS = 2048
GRAPHWALKS_SLACK = 1.05


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


def token_counts(name, texts):
    """gpt-oss token counts for a pinned dataset's texts, cached next to it since long-context sets take seconds to count."""
    key = hashlib.sha256("".join(SOURCES[part]["sha256"] for part in name.split("+")).encode()).hexdigest()[:16]
    cache = paths.CACHE / "datasets" / "token-counts" / f"{name}-{key}.json"
    if cache.exists():
        counts = json.loads(cache.read_text())
        if len(counts) == len(texts):
            return counts
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(fetch("tokenizer")))
    counts = [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(counts))
    return counts


def bin_of(tokens, bins, slack=1.0):
    """The length bin a prompt falls in: (half the bin, the bin] for the first, (previous bin, the bin] after that."""
    lower = bins[0] / 2 * slack
    for size in bins:
        if lower < tokens <= size * slack:
            return size
        lower = size * slack
    return None


def answer_budget(context, prompt_tokens, budget):
    """Tokens left to answer in once the prompt is in the window, by the gpt-oss count; None when too few are left to answer
    at all. Models whose tokenizer needs more tokens are refused by the server, which the runner refits from its reply."""
    if not context:
        return budget
    room = context - prompt_tokens - TEMPLATE_TOKENS
    return min(budget, room) if room >= MIN_ANSWER_TOKENS else None


def item(task, item_id, messages, gold, meta, max_tokens, sample=0, tools=None):
    return {"id": item_id, "task": task, "messages": messages, "gold": gold, "meta": meta,
            "max_tokens": max_tokens, "sample": sample, "tools": tools}


def long_item(task, item_id, messages, gold, meta, tokens, spec, context):
    budget = answer_budget(context, tokens, spec["max_tokens"])
    meta = {**meta, "tokens": tokens, **({} if budget else {"unreachable": True})}
    return item(task, item_id, messages, gold, meta, budget or spec["max_tokens"])


def supergpqa(spec, _context=None):
    """A sample spread over the disciplines in proportion to the full set, so the score estimates the published full-set one."""
    frame = pd.read_json(fetch("supergpqa"), lines=True).sort_values("uuid")
    rng = random.Random(0)
    items = []
    for discipline, group in sorted(frame.groupby("discipline"), key=lambda pair: pair[0]):
        rows = group.to_dict("records")
        rng.shuffle(rows)
        for row in rows[: max(1, round(spec["questions"] * len(rows) / len(frame)))]:
            options = "\n".join(f"({LETTERS[index]}) {text}" for index, text in enumerate(row["options"]))
            prompt = (
                f"The following is a multiple choice question about {discipline} ({row['field']}). Think it through, "
                "then finish with a final line of the form 'Answer: X', where X is the letter of the correct option.\n\n"
                f"Question: {row['question']}\n\nOptions:\n{options}"
            )
            items.append(item("supergpqa", f"supergpqa-{row['uuid']}", [{"role": "user", "content": prompt}],
                              row["answer_letter"], {"discipline": discipline, "difficulty": row["difficulty"]},
                              spec["max_tokens"]))
    return items


def math(spec, _context=None):
    items = []
    for name, competition in (("aime2026", "aime"), ("hmmt2026", "hmmt")):
        for row in pd.read_parquet(fetch(name)).sort_values("problem_idx").to_dict("records"):
            for sample in range(spec["samples"]):
                items.append(item("math", f"{competition}26-{row['problem_idx']}-s{sample}",
                                  [{"role": "user", "content": row["problem"] + MATH_SUFFIX}], str(row["answer"]),
                                  {"competition": competition, "problem": int(row["problem_idx"])},
                                  spec["max_tokens"], sample))
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


def mrcr(spec, context=None):
    """OpenAI MRCR with eight needles, sampled per length bin; lengths past the model's window become zero-score items."""
    frame = pd.concat([pd.read_parquet(fetch(name)) for name in ("mrcr0", "mrcr1")], ignore_index=True)
    frame["tokens"] = token_counts("mrcr0+mrcr1", [
        "".join(message["content"] for message in json.loads(prompt)) + answer
        for prompt, answer in zip(frame["prompt"], frame["answer"])])
    frame["bin"] = [bin_of(tokens, spec["bins"]) for tokens in frame["tokens"]]
    items = []
    for size in spec["bins"]:
        candidates = frame[frame["bin"] == size].sort_values("random_string_to_prepend").to_dict("records")
        for row in random.Random(size).sample(candidates, min(spec["per_bin"], len(candidates))):
            items.append(long_item("mrcr", f"mrcr-{size}-{row['random_string_to_prepend']}", json.loads(row["prompt"]),
                                   {"answer": row["answer"], "prefix": row["random_string_to_prepend"]},
                                   {"bin": size}, row["tokens"], spec, context))
    return items


def graphwalks(spec, context=None):
    """OpenAI Graphwalks BFS and parent queries, the same number of each per length bin; its 32k and 64k prompts run a few hundred tokens over, hence the slack."""
    frame = pd.read_parquet(fetch("graphwalks"))
    frame["tokens"] = token_counts("graphwalks", frame["prompt"].tolist())
    frame["bin"] = [bin_of(tokens, spec["bins"], GRAPHWALKS_SLACK) for tokens in frame["tokens"]]
    frame["key"] = [hashlib.sha256(prompt.encode()).hexdigest()[:12] for prompt in frame["prompt"]]
    items = []
    for size in spec["bins"]:
        for kind in ("bfs", "parents"):
            candidates = frame[(frame["bin"] == size) & (frame["problem_type"] == kind)].sort_values("key").to_dict("records")
            for row in random.Random(f"{size}-{kind}").sample(candidates, min(spec["per_bin"], len(candidates))):
                items.append(long_item("graphwalks", f"graphwalks-{size}-{kind}-{row['key']}",
                                       [{"role": "user", "content": row["prompt"]}], sorted(row["answer_nodes"]),
                                       {"bin": size, "type": kind}, row["tokens"], spec, context))
    return items


def tools(spec, _context=None):
    items = []
    for repeat in range(spec["repeats"]):
        for entry in tool_cases.CASES:
            items.append(item("tools", f"tools-{entry['id']}-r{repeat}", tool_cases.messages_for(entry), entry["id"],
                              {"case": entry["id"], "category": entry["category"], "max_steps": spec["max_steps"]},
                              spec["max_tokens"], repeat, tool_cases.TOOLS))
    return items


BUILDERS = {"supergpqa": supergpqa, "math": math, "lcb": lcb, "mrcr": mrcr, "graphwalks": graphwalks, "tools": tools}


def build(task, spec, context=None):
    items = BUILDERS[task](spec, context)
    return items[: spec["max_items"]] if spec.get("max_items") else items
