# mbench

Benchmark every model your [llama-swap](https://github.com/mostlygeek/llama-swap) serves, measured the same way on your own GPU, and rank them on one leaderboard page.

```
mbench run gpt-oss-120b
```

One command per model. It loads the model through llama-swap, measures speed and answer quality, stores everything in SQLite and rebuilds a self-contained leaderboard: a quality-against-speed chart, a sortable table with 95% intervals, and a detail view per model. The run lives in a systemd user unit, so closing the terminal only detaches you from it, and `mbench resume` continues a stopped run from the last saved answer.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/leaderboard-dark.png">
  <img src="docs/leaderboard.png" alt="The leaderboard page: quality index against single-request decode speed, then a table of task scores with 95% intervals">
</picture>

## What a run measures

| Task | Full suite | Scored by |
|---|---|---|
| Speed | 3 prompts × 3 repeats at 1,024 tokens; 1, 2 and 4 requests at once; first-token wait and decode speed from 1k to 250k-token prompts (capped by the context window); GPU power | greedy decoding, streamed timings, `nvidia-smi` |
| MMLU-Pro | 420 questions, 30 per subject | final `Answer: X` line |
| AIME 2025 | 30 problems × 4 samples, 64k-token budget | last `\boxed{}` value |
| LiveCodeBench v6 | 101 problems (Jan–Apr 2025), 64k-token budget | hidden tests, run in a network-less Docker container |
| Needle retrieval | 60 prompts at 16k, 32k, 64k and 120k tokens | three of ten vault codes |
| Tool calls | 20 cases × 3 | exact function and arguments |

The quality index is the mean of the five quality tasks. It only exists once all five have run. Every reasoning effort has its own ranking, so a model's max-effort run sits next to its everyday medium one. `--quick` (30–60 minutes, ranked as provisional) and `--smoke` (a pipeline check, never ranked) run smaller samples.

## Requirements

- Linux with systemd user units and an NVIDIA GPU (`nvidia-smi`)
- llama-swap in front of llama.cpp, SGLang or vLLM
- Docker, for grading LiveCodeBench
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Optional: [lmx](https://github.com/LottoLottoLotto/localmaxxing-cli), logged in, for `--submit`

## Install

```
uv tool install git+https://github.com/guitaripod/mbench
```

## Describe your models

mbench reads each model's launch command from llama-swap's config. What it can't read goes in `~/.config/mbench/models.toml`, keyed by llama-swap model id:

```toml
[qwen3-32b]
thinking = "qwen"
context = 131072

[gpt-oss-120b]
thinking = "openai"
efforts = ["low", "medium", "high"]
context = 131072
hf_id = "openai/gpt-oss-120b"
quantization = "MXFP4"
spec = { method = "EAGLE3", draft = "lmsys/EAGLE3-gpt-oss-120b-bf16", tokens_per_step = 3 }
```

- `thinking` says how the model takes a reasoning level: `openai` sends `reasoning_effort`, `qwen` sends `chat_template_kwargs` with `enable_thinking` and `reasoning_effort`, `none` sends nothing.
- `efforts` lists the levels the model's chat template accepts, lowest first. `--effort max` and `--effort min` pick the last and the first, and any other level has to be on the list. Without `efforts`, `--effort` is sent as written and max/min are refused.
- `context` caps the long-prompt tests. When it is missing, mbench asks the server once the model is loaded.
- `hf_id` and `quantization` are only needed for `--submit`. `spec = { method, draft, tokens_per_step, window }` and `engine = { repository, commit, version }` add detail to those submissions.

If you use [Oh My Pi](https://github.com/can1357/oh-my-pi), mbench also reads `compat.thinkingFormat`, `thinking.efforts` and `contextWindow` from its `models.yml` overrides; `models.toml` wins where both say something. `mbench profile <id>` shows what was resolved and where each value came from.

## Use

```
mbench run <id>                       full suite at medium effort
mbench run <id> --effort max          the model's highest declared effort, ranked separately
mbench run <id> --quick --detach      smaller samples; start and return at once
mbench run <id> --only speed          one task; --skip leaves tasks out
mbench run <id> --submit              also submit to localmaxxing (all, speed or evals)
mbench status                         the run in progress
mbench logs -f                        follow the worker log
mbench cancel / mbench resume         stop a run; continue it later
mbench ls [--effort max]              ranked table in the terminal
mbench board --open                   the leaderboard page
mbench profile <id>                   what mbench knows about a model
```

`mbench -h` and `mbench run -h` cover every option, with examples. One run happens at a time. A run swaps its model in through llama-swap, which unloads whatever else was loaded.

## How comparable the numbers are

Everything mbench controls is fixed. Dataset files are downloaded from Hugging Face once and pinned by sha256, question subsets and needle prompts use seed 0, and prompts, token budgets, reasoning effort and concurrency are the same on every run. Each run records the harness version, the suite version and a fingerprint of the llama-swap command plus the launcher script it calls, so editing a launcher shows up as a new configuration.

Speed runs decode greedily, so the same prompt produces the same tokens and speculative-decoding acceptance is comparable between runs. Quality runs use each model's recommended sampling instead, because reasoning models are tuned for it and loop or degrade at temperature 0. The noise is handled with repeats and 95% intervals next to every score. llama.cpp and vLLM servers also get a per-question seed. SGLang servers do not: its FlashInfer sampler asserts on seeded top-k/top-p requests and would stop the server.

A run waits up to 30 minutes for other GPU work to finish before measuring speed, and flags the result if it had to go ahead anyway. If free RAM drops under 4 GB it unloads the model and stops.

## localmaxxing

`--submit speed` has lmx measure its two canonical prompts, then submits the runs through the [localmaxxing](https://www.localmaxxing.com) API with the prompt hash, output sample and timings that the lmx client leaves out, so they can earn the Verified badge. Speculative-decoding acceptance comes from SGLang's counters. `--submit evals` answers the GSM8K and HellaSwag shards the site doesn't have yet for that model and quantization, at the run's effort. A small local proxy lets lmx's log-likelihood scoring work against SGLang, which rejects the zero-token completion request lmx sends. `--submit` alone does both.

## Files

- `~/.local/share/mbench/bench.db`: runs, metrics and submissions
- `~/.local/share/mbench/runs/<run>/`: every answer as JSONL, speed samples, server settings, the worker log
- `~/.local/share/mbench/leaderboard.html`: rebuilt after every run
- `~/.config/mbench/models.toml`: model facts; `hardware.json` next to it is created by `lmx hardware` on the first `--submit`
- `~/.cache/mbench/`: pinned datasets and the LiveCodeBench harness

Environment overrides: `MBENCH_HOME`, `MBENCH_SWAP_URL` (default `http://127.0.0.1:8081`), `MBENCH_SWAP_CONFIG`, `MBENCH_OMP_MODELS`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`.

## Credits

Questions come from [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MathArena's AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) and [LiveCodeBench](https://livecodebench.github.io), downloaded at run time rather than redistributed. Code grading uses LiveCodeBench's own harness at a pinned commit. The two canonical speed prompts are localmaxxing's.

## Development

```
uv run --group dev pytest
uv tool install --editable .
```

Licensed under GPL-3.0-or-later.
