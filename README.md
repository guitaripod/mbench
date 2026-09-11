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
| SuperGPQA | 521 graduate-level questions, ten options, spread over its 13 disciplines like the full set | final `Answer: X` line |
| AIME + HMMT 2026 | AIME 2026 and HMMT February 2026, 63 problems × 2 samples, 64k-token budget | last `\boxed{}`, compared symbolically with [Math-Verify](https://github.com/huggingface/Math-Verify) |
| LiveCodeBench v6 | 101 problems (Jan–Apr 2025), 64k-token budget | hidden tests, run in a network-less Docker container |
| MRCR, 8 needles | 60 conversations, 15 each at 16k, 32k, 64k and 128k tokens | OpenAI's grade: the requested prefix, then the difflib ratio |
| Graphwalks | 48 graphs, BFS and parent queries at 8k, 16k, 32k and 64k tokens | F1 of the node set on the last line |
| Tool use | 38 episodes × 3 against a simulated workspace: single, parallel and chained calls, error recovery, clarifying questions, irrelevant tools, look-alike tools, refund policy, conditionals and follow-ups | the workspace's end state and the reply, not the exact calls |

The quality index averages five groups equally: knowledge (SuperGPQA), math (AIME + HMMT), code (LiveCodeBench), long context (MRCR and Graphwalks) and tool use. It only exists once every task has run, and its 95% interval bootstraps the questions of every task. Every reasoning effort has its own ranking, so a model's max-effort run sits next to its everyday medium one. `--quick` (45–90 minutes, ranked as provisional) and `--smoke` (a pipeline check, never ranked) run smaller samples.

### Why these tasks

MMLU-Pro, AIME 2025, plain needle-in-a-haystack and single-turn tool calls are the usual picks. Current open models score within a few points of each other on MMLU-Pro and near 100% on the needle and tool tests, so those stopped separating models; AIME 2025 has had a year and a half to leak into training data. SuperGPQA is harder and much larger, the 2026 competitions postdate most of today's models, MRCR and Graphwalks are the long-context tests model makers report against, and the tool episodes follow the [2026 validity audit of tool-calling benchmarks](https://arxiv.org/abs/2607.02577): deterministic checks of what the model did to the world, so a different valid route to the same result still passes.

LiveCodeBench has published nothing newer than April 2025, so models trained after mid-2025 may have seen its problems. It stays because nothing public replaces it for executed code, but read its column as an upper bound.

## Requirements

- Linux with systemd user units and an NVIDIA GPU (`nvidia-smi`)
- llama-swap in front of llama.cpp, SGLang or vLLM
- Docker, for grading LiveCodeBench
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- About 1 GB of disk for the pinned datasets
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
mbench run <id> --reuse               carry over answers that still apply from the last run
mbench run <id> <id> --at 03:00 --until 08:00
                                      several models, one after another, only at night
mbench run <id> --submit              also submit to localmaxxing (all, speed or evals)
mbench status                         the run in progress
mbench logs -f                        follow the worker log
mbench cancel / mbench resume         stop a run; continue it later
mbench ls [--effort max] [--suite 1]  ranked table in the terminal
mbench compare <id> <id>              paired differences, task by task, with 95% intervals
mbench board --open                   the leaderboard page
mbench profile <id>                   what mbench knows about a model
```

`mbench -h` and `mbench run -h` cover every option, with examples. One run happens at a time; more models, or a run started while another is going, wait their turn. A run swaps its model in through llama-swap, which unloads whatever else was loaded.

### Scheduling

`--at 03:00` (or `--at "2026-09-12 03:00"`) starts the runs then instead of now. Add `--until 08:00` and it becomes a daily window: whatever is still running at 08:00 stops, llama-swap unloads the model so the GPU is yours again, and the run continues from its last saved answer at 03:00 the next night, until everything is done. `mbench status` lists what is scheduled and `mbench cancel <run>` takes a run off the schedule; `mbench resume <run> --at 03:00 --until 08:00` puts a stopped one back on.

A user timer, `mbench-tick.timer`, checks every five minutes, starts the next due run when nothing is running, and switches itself off once nothing is scheduled. It keeps working after a reboot or logout if lingering is on (`loginctl enable-linger`).

### Comparing two models

Two scores whose intervals overlap can still differ reliably, because both models answered the same questions. `mbench compare` pairs them question by question and bootstraps the difference, per task and for the quality index, and says which differences are real:

```
  SuperGPQA                70.0   60.0   +10.0  -10.0 to +32.5    40   no clear difference
  Quality index                           +9.7  +0.8 to +18.6     A ahead
```

### Suite versions and --reuse

The questions a run asks are the suite version (`full/v1`). Versions measure different tasks, so the leaderboard ranks each on its own tab and never mixes their indexes, and a run started under an older version can't be resumed by a newer mbench. `--reuse [RUN]` starts a fresh run that copies the answers of an earlier run of the same model, config, effort and suite size for every task whose questions and scoring didn't change, and measures only the rest.

## How comparable the numbers are

Everything mbench controls is fixed. Dataset files are downloaded from Hugging Face once and pinned by sha256, question subsets use fixed seeds, and prompts, token budgets, reasoning effort and concurrency are the same on every run. Each run records the harness version, the suite version and a fingerprint of the llama-swap command plus the launcher script it calls, so editing a launcher shows up as a new configuration.

Speed runs decode greedily, so the same prompt produces the same tokens and speculative-decoding acceptance is comparable between runs. Quality runs use each model's recommended sampling instead, because reasoning models are tuned for it and loop or degrade at temperature 0. The noise is handled with repeats and 95% intervals next to every score. llama.cpp and vLLM servers also get a per-question seed. SGLang servers do not: its FlashInfer sampler asserts on seeded top-k/top-p requests and would stop the server.

Long-context lengths are counted with the gpt-oss (o200k) tokenizer, as OpenAI bins MRCR and Graphwalks; a model whose tokenizer is less efficient reads more tokens for the same prompt. A prompt that can't fit the model's context window with room to answer scores zero rather than being skipped, so a 32k-context model is measured on the same questions as a 256k one and the missing lengths count against it. When the server refuses a request for its length, mbench reads the window and prompt size from the refusal, retries once with an answer budget that fits, and scores zero if even that can't.

A run waits up to 30 minutes for other GPU work to finish before measuring speed, and flags the result if it had to go ahead anyway. If free RAM drops under 4 GB it unloads the model and stops.

## localmaxxing

`--submit speed` has lmx measure its two canonical prompts, then submits the runs through the [localmaxxing](https://www.localmaxxing.com) API with the prompt hash, output sample and timings that the lmx client leaves out, so they can earn the Verified badge. Speculative-decoding acceptance comes from SGLang's counters. `--submit evals` answers the GSM8K and HellaSwag shards the site doesn't have yet for that model and quantization, at the run's effort. A small local proxy lets lmx's log-likelihood scoring work against SGLang, which rejects the zero-token completion request lmx sends. `--submit` alone does both.

## Files

- `~/.local/share/mbench/bench.db`: runs, metrics and submissions
- `~/.local/share/mbench/runs/<run>/`: every answer as JSONL (tool episodes with their calls and results), speed samples, server settings, the worker log
- `~/.local/share/mbench/leaderboard.html`: rebuilt after every run
- `~/.config/mbench/models.toml`: model facts; `hardware.json` next to it is created by `lmx hardware` on the first `--submit`
- `~/.cache/mbench/`: pinned datasets, their token counts and the LiveCodeBench harness

Environment overrides: `MBENCH_HOME`, `MBENCH_SWAP_URL` (default `http://127.0.0.1:8081`), `MBENCH_SWAP_CONFIG`, `MBENCH_OMP_MODELS`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`.

## Credits

Questions come from [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) (ODC-BY), [MathArena's AIME 2026 and HMMT February 2026](https://matharena.ai) (CC BY-NC-SA 4.0), [LiveCodeBench](https://livecodebench.github.io), and OpenAI's [MRCR](https://huggingface.co/datasets/openai/mrcr) and [Graphwalks](https://huggingface.co/datasets/openai/graphwalks) (MIT), all downloaded at run time rather than redistributed. Code grading uses LiveCodeBench's own harness at a pinned commit. The speed test's long prompts are built from MMLU-Pro question text. The two canonical speed prompts are localmaxxing's.

## Development

```
uv run --group dev pytest
uv tool install --editable .
```

Licensed under GPL-3.0-or-later.
