import re

from . import datasets

LCB_REPO = "livecodebench/code_generation_lite"
LCB_FILE = re.compile(r"test(\d*)\.jsonl")
MATH_AUTHOR = "MathArena"
MATH_FAMILIES = re.compile(r"^(aime|hmmt_(feb|nov)|smt|brumo|cmimc|apex|arxivmath)[_-]")
FIRST_FRESH_YEAR = 2026


def math_year(name):
    """The year a MathArena set comes from: aime_2026 style names, or arxivmath-0626 style month-year stamps."""
    match = re.search(r"_(20\d\d)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"-(\d\d)(\d\d)$", name)
    return 2000 + int(match.group(2)) if match else None


def pinned_math():
    return {source["repo"] for source in datasets.SOURCES.values() if source["repo"].startswith(MATH_AUTHOR + "/")}


def changed_pins(api):
    """Pinned files whose upstream copy no longer has the pinned sha256 (only LFS files carry one to compare)."""
    findings = []
    for name, source in datasets.SOURCES.items():
        try:
            info = api.get_paths_info(source["repo"], [source["file"]], repo_type=source["kind"])[0]
        except Exception as error:
            findings.append(f"{name}: {source['repo']}/{source['file']} could not be checked ({error.__class__.__name__})")
            continue
        upstream = getattr(getattr(info, "lfs", None), "sha256", None)
        if upstream and upstream != source["sha256"]:
            findings.append(f"{name}: {source['repo']}/{source['file']} changed upstream since it was pinned")
    return findings


def newer_lcb(api):
    match = LCB_FILE.fullmatch(datasets.SOURCES["lcb"]["file"])
    pinned = int(match.group(1) or 1) if match else 1
    releases = [int(match.group(1) or 1) for path in api.list_repo_files(LCB_REPO, repo_type="dataset")
                if (match := LCB_FILE.fullmatch(path))]
    newest = max(releases, default=pinned)
    return [f"LiveCodeBench: {LCB_REPO} has test{newest}.jsonl, newer than the pinned test{pinned}.jsonl"] if newest > pinned else []


def newer_math(api):
    pinned = pinned_math()
    found = []
    for entry in api.list_datasets(author=MATH_AUTHOR, limit=500):
        name = entry.id.split("/", 1)[1]
        if "output" in name or entry.id in pinned or any(entry.id.startswith(repo + "_") for repo in pinned):
            continue
        year = math_year(name)
        if MATH_FAMILIES.match(name) and year and year >= FIRST_FRESH_YEAR:
            kind = "research-level final answers" if name.startswith("arxivmath") else "a final-answer competition"
            found.append(f"MathArena: {entry.id} ({kind}, {year}) isn't in the suite yet")
    return sorted(found)


def check():
    """What changed upstream since the suite was pinned: moved pins, and newer question sets worth a new suite version."""
    from huggingface_hub import HfApi

    api = HfApi()
    return {"changed": changed_pins(api), "newer": newer_lcb(api) + newer_math(api)}
