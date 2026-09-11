# mbench

Benchmarks the models llama-swap serves on this box and ranks them on one leaderboard page.

```
mbench run <llama-swap id>          full suite, 1–4 hours, runs in the background
mbench run <id> --quick             30–60 minutes, ranked as provisional
mbench run <id> --smoke             a few items per task to check the pipeline; never ranked
mbench run <id> --submit            also submit localmaxxing speed runs and GSM8K/HellaSwag shards
mbench status                       what is running and how far along it is
mbench logs -f                      follow the worker log
mbench cancel / mbench resume       stop a run and continue it later from where it stopped
mbench ls                           ranked table in the terminal
mbench board --open                 the leaderboard page
mbench profile <id>                 what mbench knows about a model and where each fact came from
```

`mbench run` starts a transient systemd unit and follows its progress. Ctrl-C only detaches; the run carries on.

## What a run measures

| Task | Full suite | Scored by |
|---|---|---|
| Speed | 3 prompts × 3 repeats at 1,024 tokens; 1, 2 and 4 requests at once; prompts from 1k to 250k tokens (capped by the model's window) | greedy decoding, streamed timings, board power |
| MMLU-Pro | 420 questions, 30 per subject | final `Answer: X` line |
| AIME 2025 | 30 problems × 4 samples, 64k-token budget | last `\boxed{}` value |
| LiveCodeBench v6 | 101 problems (Jan–Apr 2025), 64k-token budget | hidden tests in a network-less container |
| Needle retrieval | 60 prompts at 16k, 32k, 64k and 120k tokens | three of ten vault codes |
| Tool calls | 20 cases × 3 | exact function and arguments |

The quality index is the mean of the five quality tasks. It only exists once all five have run. With `--submit`, localmaxxing's GSM8K and HellaSwag shards run too; they appear per model but stay out of the index.

## How deterministic it is

Everything mbench controls is fixed: dataset files are pinned by sha256, question subsets use seed 0, prompts are identical between runs, and so are token budgets, reasoning effort and concurrency. Each run records the harness commit, the suite version and a fingerprint of the llama-swap command plus the launcher script it calls. Editing a launcher therefore shows up as a new configuration.

Speed runs decode greedily, so the same prompt produces the same tokens and drafter acceptance is comparable between runs. Quality runs use each model's recommended sampling instead: reasoning models are tuned for it and loop or degrade at temperature 0. The noise is handled with repeats (AIME × 4, tools × 3) and 95% intervals shown next to every score. llama.cpp servers also get a per-question seed. SGLang servers do not, because their FlashInfer sampler asserts on seeded top-k/top-p requests and would stop the server.

A run waits for other GPU work (ComfyUI, games) to finish before measuring speed, and flags the run if it had to go ahead anyway. If free RAM drops under 4 GB it unloads the model and stops.

## Files

- `~/.local/share/mbench/bench.db`: runs, metrics and submissions (SQLite)
- `~/.local/share/mbench/runs/<run>/`: every answer as JSONL, speed samples, server settings, the worker log
- `~/.local/share/mbench/leaderboard.html`: rebuilt after every run
- `~/.config/mbench/models.toml`: facts mbench can't discover. `hf_id` and `quantization` are needed for `--submit`; `spec`, `engine`, `name`, `thinking` and `context` are optional overrides, keyed by llama-swap id.

mbench reads the llama-swap config for the launch command and Oh My Pi's `models.yml` for how each model takes a thinking level and its context window.

## Development

```
uv run --group dev pytest
uv tool install --editable .
```

Licensed under GPL-3.0-or-later.
