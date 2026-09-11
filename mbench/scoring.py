import re
from difflib import SequenceMatcher

CHOICE_PATTERNS = (
    r"Answer:\s*\**\s*\(?([A-J])\)?",
    r"answer is\s*\**\s*\(?([A-J])\)?",
    r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
)


def last_boxed(text):
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    return None


def choice(content, gold):
    for pattern in CHOICE_PATTERNS:
        hits = re.findall(pattern, content, flags=re.IGNORECASE)
        if hits:
            prediction = hits[-1].upper()
            return prediction, float(prediction == gold)
    return None, 0.0


def math(content, gold):
    """The last \\boxed{} answer against the gold one, compared symbolically so 7/2, \\frac{7}{2} and 3.5 all count."""
    from math_verify import parse, verify

    boxed = last_boxed(content)
    if boxed is None:
        return None, 0.0
    try:
        return boxed.strip(), float(verify(parse(f"${gold}$"), parse(f"${boxed}$")))
    except Exception:
        return boxed.strip(), float(boxed.strip() == gold.strip())


def mrcr(content, gold):
    """OpenAI's MRCR grade: zero without the requested prefix, otherwise the difflib ratio against the needle."""
    response = content.lstrip()
    if not response.startswith(gold["prefix"]):
        return None, 0.0
    answer = gold["answer"].removeprefix(gold["prefix"])
    return response[:80], SequenceMatcher(None, response.removeprefix(gold["prefix"]), answer).ratio()


def graphwalks(content, gold):
    """F1 of the node set on the response's last line; an unformatted answer scores zero."""
    lines = content.rstrip().split("\n")
    match = re.search(r"Final Answer: ?\[(.*)\]", lines[-1]) if lines else None
    if match is None:
        return None, 0.0
    predicted = {item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()}
    truth = set(gold)
    if not predicted and not truth:
        return [], 1.0
    overlap = len(predicted & truth)
    if overlap == 0:
        return sorted(predicted), 0.0
    precision, recall = overlap / len(predicted), overlap / len(truth)
    return sorted(predicted), 2 * precision * recall / (precision + recall)


TEXT_SCORERS = {"supergpqa": choice, "math": math, "mrcr": mrcr, "graphwalks": graphwalks}
