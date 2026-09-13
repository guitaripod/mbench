# mbench

[![tests](https://github.com/guitaripod/mbench/actions/workflows/tests.yml/badge.svg)](https://github.com/guitaripod/mbench/actions/workflows/tests.yml)
[![leaderboard](https://github.com/guitaripod/mbench/actions/workflows/pages.yml/badge.svg)](https://guitaripod.github.io/mbench/)
[![release](https://img.shields.io/github/v/release/guitaripod/mbench?label=release&color=blue)](https://github.com/guitaripod/mbench/releases/latest)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/github/license/guitaripod/mbench?color=blue)](LICENSE)

Rank the models your own GPU runs. mbench measures every model [llama-swap](https://github.com/mostlygeek/llama-swap) serves — speed and answer quality — the same way each time, and builds one leaderboard page out of the results.

```
mbench run gpt-oss-120b
```

One command per model. It loads the model through llama-swap, checks the server, measures, stores everything in SQLite and rebuilds a self-contained page: quality against speed, a sortable table with 95% intervals, a detail view per model. The run lives in a systemd user unit, so closing the terminal only detaches you from it, and `mbench resume` continues from the last saved answer.

It measures the model as your server actually runs it — quantization, chat template, tool-call parser, speculative decoding, context window and all — not the weights in a vacuum.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/leaderboard-dark.png">
  <img src="docs/leaderboard.png" alt="The leaderboard page: quality index against single-request decode speed, then a table of task scores with 95% intervals">
</picture>

## What comes out

<!--LEADERBOARD-->
| Model | Quality | SuperGPQA | Math | LCB | MRCR | Graphwalks | Tools | tok/s | Peak | TTFT 32k | Wh/correct | Run |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen 3.8 27B · SGLang · DFlash2 · 524K | 81.0 | 61.2 | 81.0 | 76.2 | 78.4 | 94.5 | 100.0 | 157 | 400 | 5.0 | 2.67 | 12 Sep |
| GPT-OSS 120B · SGLang · MXFP4 · DFlash · 128K | 71.2 | 51.4 | 87.3 | 83.2 | 11.0 | 64.4 | 96.5 | 206 | 432 | 1.8 | 5.99 | 12 Sep |

_2 models on NVIDIA RTX PRO 6000 Blackwell Workstation Edition, suite full/v1, max effort, newest run 12 Sep 2026._
<!--/LEADERBOARD-->

Quality is the index; then one column per task, single-request decode speed, peak throughput across all requests, first-token wait on a 32k prompt, and board watt-hours per correct answer. The [live page](https://guitaripod.github.io/mbench/) adds 95% intervals, per-length breakdowns and every server setting behind each run. Those are one GPU's numbers — yours will differ with your hardware, quantization and server settings.

## What a run measures

| Task | Full suite | Scored by |
|---|---|---|
| Speed | 3 prompts × 3 repeats at 1,024 tokens; 1 up to 32 requests at once, as many as the server takes; first-token wait and decode speed from 1k to 250k-token prompts; GPU power | greedy decoding, streamed timings, `nvidia-smi` |
| SuperGPQA | 521 graduate-level questions, ten options, spread over its 13 disciplines like the full set | final `Answer: X` line |
| AIME + HMMT 2026 | 63 problems × 2 samples, 64k-token budget | last `\boxed{{}}`, compared symbolically with [Math-Verify](https://github.com/huggingface/Math-Verify) |
| LiveCodeBench v6 | 101 problems (Jan–Apr 2025), 64k-token budget | hidden tests, run in a network-less Docker container |
| MRCR, 8 needles | 60 conversations, 15 each at 16k, 32k, 64k and 128k tokens | OpenAI's grade: the requested prefix, then the difflib ratio |
| Graphwalks | 48 graphs, BFS and parent queries from 8k to 64k tokens | F1 of the node set on the last line |
| Tool use | 38 episodes × 3 against a simulated workspace: chained calls, error recovery, clarifying questions, look-alike tools, refund policy, conditionals, follow-ups | the workspace's end state and the reply, not the exact calls |

The quality index averages five groups equally: knowledge, math, code, long context (MRCR and Graphwalks) and tool use. It appears once every task has run, and its 95% interval bootstraps the questions of every task. Each reasoning effort is ranked on its own, so a model's max-effort run sits beside its everyday medium one, and `--effort none` ranks a model with thinking switched off. `--quick` (45–90 minutes, ranked as provisional) and `--smoke` (a pipeline check, never ranked) run smaller samples.

### Why these tasks

MMLU-Pro, AIME 2025, plain needle-in-a-haystack and single-turn tool calls are the usual picks, and current open models sit within a few points of each other on the first and near 100% on the last two — they no longer separate anything. So: SuperGPQA is harder and much larger, the 2026 competitions postdate most of today's models, MRCR and Graphwalks are the long-context tests the model makers themselves report, and the tool episodes follow the [2026 validity audit of tool-calling benchmarks](https://arxiv.org/abs/2607.02577) — deterministic checks of what the model did to the world, so a different valid route to the same result still passes.

LiveCodeBench has published nothing newer than April 2025, so models trained after mid-2025 may have seen its problems; read that column as an upper bound. `mbench sources` checks Hugging Face for newer releases and for pinned files that moved.

## Numbers you can trust

- **Fixed inputs.** Datasets are downloaded once and pinned by sha256, question subsets use fixed seeds, and every run records the harness version, the suite version and a fingerprint of the llama-swap command plus the launcher script it calls — edit a launcher and it counts as a new configuration.
- **Fixed conditions.** Speed decodes greedily, so the same prompt yields the same tokens and drafter acceptance stays comparable. Quality uses each model's recommended sampling, because reasoning models degrade at temperature 0. A run waits for other GPU work to finish, and says so on the board if it had to go ahead anyway.
- **One stack per speed number.** Every run records the card, its driver and the build of the server that answered — SGLang's version or the commit of the checkout it runs from, llama.cpp's build string. Quality compares across those; decode, peak and watt-hours only compare within one, so the board says when a ranking spans two setups, and `mbench doctor` says when the stack moved under a model since its last run.
- **Intervals, not point scores.** Every task carries a 95% interval (Wilson, or the spread of per-question means), and the index bootstraps questions within each task. `mbench compare` pairs two runs question by question and says which differences survive the noise.
- **Nothing quietly skipped.** A prompt that doesn't fit the model's context scores zero instead of disappearing, and the board says which task that cost. When the server refuses a request for its length, mbench reads the window and prompt size out of the refusal, retries once with a smaller answer budget, then scores zero.
- **Cost as well as speed.** Board energy is integrated across the quality tasks and divided by correct answers, so a fast-but-wasteful model is visible as watt-hours per correct answer.
- **Saturation is flagged.** With three or more models ranked, the board marks any task that isn't separating them — all near the ceiling, all near the floor, or apart by less than the noise.

## Benchmarking a model

You need Linux with systemd user units and an NVIDIA GPU, llama-swap in front of SGLang or llama.cpp, Docker for grading LiveCodeBench, Python 3.12 with [uv](https://docs.astral.sh/uv/), and about 1 GB of disk for the pinned datasets. vLLM speaks the same API but is untested: it doesn't report how many requests it takes at once, so mbench sends 4.

```
uv tool install git+https://github.com/guitaripod/mbench
```

**1. Describe the model.** mbench reads its launch command from llama-swap's config; anything that command doesn't say goes in `~/.config/mbench/models.toml`, keyed by llama-swap model id. `mbench profile <id>` shows what was resolved and where each value came from.

```toml
[gpt-oss-120b]
thinking = "openai"
efforts = ["low", "medium", "high"]
context = 131072
hf_id = "openai/gpt-oss-120b"
quantization = "MXFP4"
spec = { method = "EAGLE3", draft = "lmsys/EAGLE3-gpt-oss-120b-bf16", tokens_per_step = 3 }
```

- `thinking` is how the model takes a reasoning level: `openai` sends `reasoning_effort`, `qwen` sends `chat_template_kwargs` with `enable_thinking` and `reasoning_effort`, `muse` sends `chat_template_kwargs` with `reasoning_strength`, `none` sends nothing.
- `efforts` lists the levels the chat template accepts, lowest first; `--effort max`/`min` pick the ends.
- `context` caps the long-prompt tests. Without it, mbench asks the server once the model is loaded.
- `hf_id`, `quantization`, `spec` and `engine` only matter for `--submit`.
- [Oh My Pi](https://github.com/can1357/oh-my-pi) users: `compat.thinkingFormat`, `thinking.efforts` and `contextWindow` are read from its `models.yml` too; `models.toml` wins.

**2. Check the server.** `mbench doctor <id>` loads the model and verifies tokens per request, reasoning kept apart from the answer, structured tool calls, a conversation carrying a tool result, and a 16k-token prompt.

**3. Run it.** `mbench run <id> --effort max`, or `mbench run <a> <b> --at 02:00 --until 09:00` for a nightly window. Extra models queue and run one at a time. Answers are written as they arrive, so a pause costs nothing, and the local page rebuilds when the run ends.

**4. Read it.** `mbench ls --effort max` in the terminal, `mbench board --open` for the page, `mbench compare <a> <b>` when two models look close and you want to know whether the gap is real.

**5. Publish it.** `scripts/publish.sh max` exports `site/index.html` and `site/board.json`, re-renders both screenshots, and rewrites the table above between its markers. Commit and push: the `pages` workflow redeploys the site on any change under `site/`. The database is the record; the page, the JSON and that table are all projections of it, so nothing is typed by hand.

A model only joins an existing ranking if it ran at the same effort level, and changing the question set means bumping the suite version so old and new runs are ranked apart rather than mixed.

### Every command

```
mbench run <id>                       full suite at medium effort
mbench run <id> --effort max          the model's highest declared effort, ranked separately
mbench run <id> --quick --detach      smaller samples; start and return at once
mbench run <id> --only speed          one task; --skip leaves tasks out
mbench run <id> --reuse               carry over answers that still apply
mbench run <a> <b> --at 03:00 --until 08:00
                                      several models, one after another, only at night
mbench run <id> --submit              also submit to localmaxxing (all, speed or evals)
mbench status                         what is running, and what is scheduled
mbench logs -f                        follow the worker log
mbench cancel / mbench resume         stop a run; continue it later
mbench ls [--effort max] [--markdown] ranked table in the terminal
mbench compare <a> <b>                paired differences, task by task, with 95% intervals
mbench doctor <id>                    check a model's server before spending a night on it
mbench sources                        newer question sets, or pinned files that moved
mbench board --open                   the leaderboard page
mbench export --out site              the page plus board.json, ready to publish
mbench profile <id>                   what mbench knows about a model
```

`mbench -h` and `mbench run -h` cover every option.

## Runs that fit around you

A full run takes hours, so it stays out of the way:

- **`--at 03:00 --until 08:00`** runs models one after another inside a nightly window. Whatever is unfinished at 08:00 stops, llama-swap unloads the model, and the run continues from its saved answers the next night. Near the end of a window a run stops taking questions that couldn't finish in time.
- **Runs give way.** When a game or another GPU job holds the card for a minute, the run unloads the model, pauses and retries every ten minutes, whether it was scheduled or started by hand. `--keep-gpu` turns that off.
- **Every run starts with the doctor checks**, so a misconfigured server fails in the first minute instead of producing a night of zeros.
- **The request count matches the server.** mbench reads how many requests it accepts at once and how much context they share, then sends many short questions in parallel and long conversations one at a time.
- **`--reuse`** starts a fresh run that keeps the answers of an earlier one for every task whose questions and scoring didn't change, and measures only the rest.
- **Notifications.** Finishing, failing or pausing sends a desktop notification; set `notify` in `~/.config/mbench/config.toml` (or `MBENCH_NOTIFY`) to forward it anywhere, e.g. `notify = 'curl -d "$MBENCH_MESSAGE" https://ntfy.sh/your-topic'`.

## localmaxxing

`--submit speed` has lmx measure its two canonical prompts, then submits the runs through the [localmaxxing](https://www.localmaxxing.com) API with the prompt hash, output sample and timings the lmx client leaves out, so they earn the Verified badge. `--submit evals` answers the GSM8K and HellaSwag shards the site doesn't have yet for that model and quantization. `--submit` alone does both. Needs `hf_id` and `quantization` in `models.toml`, and [lmx](https://github.com/LottoLottoLotto/localmaxxing-cli) logged in.

## Files

- `~/.local/share/mbench/bench.db`: runs, metrics and submissions
- `~/.local/share/mbench/runs/<run>/`: every answer as JSONL (tool episodes with their calls and results), speed samples, the doctor report, server settings, the worker log
- `~/.local/share/mbench/leaderboard.html`: rebuilt after every run
- `~/.config/mbench/models.toml`: model facts · `config.toml`: the notify command
- `~/.cache/mbench/`: pinned datasets, their token counts and the LiveCodeBench harness

Environment overrides: `MBENCH_HOME`, `MBENCH_SWAP_URL` (default `http://127.0.0.1:8081`), `MBENCH_SWAP_CONFIG`, `MBENCH_OMP_MODELS`, `MBENCH_NOTIFY`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`.

## Credits

Questions come from [SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) (ODC-BY), [MathArena's AIME 2026 and HMMT February 2026](https://matharena.ai) (CC BY-NC-SA 4.0), [LiveCodeBench](https://livecodebench.github.io), and OpenAI's [MRCR](https://huggingface.co/datasets/openai/mrcr) and [Graphwalks](https://huggingface.co/datasets/openai/graphwalks) (MIT) — downloaded at run time, never redistributed. Code grading uses LiveCodeBench's own harness at a pinned commit. The long speed prompts are built from MMLU-Pro question text; the two canonical speed prompts are localmaxxing's.

## Development

```
uv run --group dev pytest
uv tool install --editable .
scripts/publish.sh max         # site/, docs/leaderboard*.png and the table above
```

Licensed under GPL-3.0-or-later.
