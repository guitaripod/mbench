import base64
import json
import pickle
import sys
import zlib

import numpy as np

sys.path.insert(0, "/lcb")
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def private_tests(raw):
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def evaluation_sample(row):
    tests = json.loads(row["public_test_cases"]) + private_tests(row["private_test_cases"])
    metadata = json.loads(row["metadata"])
    return {
        "input_output": json.dumps({
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
            "fn_name": metadata.get("func_name", None),
        })
    }


def extract_code(content):
    """Mirrors lcb_runner's extraction: the block between the last two fence lines."""
    lines = content.split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1:fences[-1]])


def main(results_path, graded_path, dataset_path):
    with open(dataset_path) as handle:
        problems = {row["question_id"]: row for row in map(json.loads, handle)}
    with open(results_path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    samples = [evaluation_sample(problems[record["meta"]["question_id"]]) for record in records]
    generations = [[extract_code(record["content"])] for record in records]
    _, results, _ = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=8, timeout=6)
    with open(graded_path, "w") as out:
        for index, record in enumerate(records):
            passed = bool(np.all(np.array(results[index][0]) > 0))
            out.write(json.dumps({"id": record["id"], "passed": passed}) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
