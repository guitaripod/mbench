import re

HYPHENS = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")


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


def mmlupro(content, gold):
    patterns = (
        r"Answer:\s*\**\s*\(?([A-J])\)?",
        r"answer is\s*\**\s*\(?([A-J])\)?",
        r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
    )
    for pattern in patterns:
        hits = re.findall(pattern, content, flags=re.IGNORECASE)
        if hits:
            prediction = hits[-1].upper()
            return prediction, float(prediction == gold)
    return None, 0.0


def aime(content, gold):
    boxed = last_boxed(content)
    if boxed is None:
        return None, 0.0
    cleaned = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", boxed).rsplit("=", 1)[-1]
    cleaned = cleaned.replace(",", "").replace("$", "").replace(" ", "")
    match = re.search(r"-?\d+", cleaned)
    prediction = str(int(match.group())) if match else cleaned
    return prediction, float(prediction == gold)


def niah(content, gold):
    content = content.translate(HYPHENS)
    found = {}
    for key in gold:
        match = re.search(re.escape(key) + r"[^\n\d]*?(\d{7})", content, flags=re.IGNORECASE)
        found[key] = match.group(1) if match else None
    return found, sum(found[key] == code for key, code in gold.items()) / len(gold)


def normalize_argument(value):
    if isinstance(value, str):
        return value.strip().lower().rstrip("/")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return sorted(normalize_argument(item) for item in value)
    return value


def arguments_match(expected, actual):
    return all(key in actual and normalize_argument(actual[key]) == normalize_argument(value)
               for key, value in expected.items())


def tools(expected, calls):
    """Exact tool name plus every expected argument; extra optional arguments are allowed, extra calls are not."""
    if not expected:
        return not calls
    if len(calls) != len(expected):
        return False
    remaining = list(calls)
    for name, arguments in expected:
        found = next((call for call in remaining if call[0] == name and arguments_match(arguments, call[1])), None)
        if found is None:
            return False
        remaining.remove(found)
    return True


TEXT_SCORERS = {"mmlupro": mmlupro, "aime": aime, "niah": niah}
