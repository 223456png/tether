# Tether

[![CI](https://github.com/223456png/tether/actions/workflows/ci.yml/badge.svg)](https://github.com/223456png/tether/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A memory-and-context management layer for long-running LLM coding agents — with real, reproducible benchmarks.**

Tether is an experimental agent runtime that answers a practical question: *as a coding agent works for hours, how do you keep its context small, its memory correct, and its progress recoverable — without silently losing information?*

It is built as a pipeline of independently testable components (LLM-driven tool loop → three-layer memory → budget-allocated compression → drift detection → smart recovery → tool interception → MCP ecosystem access), and every claim in this README is backed by a benchmark you can run yourself.

> **Positioning.** Tether is not chasing SOTA compression ratios (LLMLingua-style learned compressors reach 10–20×). It is a *deterministic, zero-model* pipeline whose value proposition is verifiable safety: every compression is validated (ROUGE-L + keyword survival) and rolled back when it would lose information. The benchmarks below measure exactly that trade-off.

## Architecture

```mermaid
flowchart LR
    subgraph Runtime
        SM[State Machine<br/>think → tool → checkpoint] --> CP[Checkpoint Manager<br/>JSONL + workspace fingerprint]
        SM --> TR[Tool Registry<br/>+ duplicate-call interceptor]
    end
    subgraph Memory
        TS[TaskSummary] --> CA
        FS[FileSnapshot] --> CA
        EN[EpisodicNotes] --> CA
    end
    CA[Context Assembler] --> BA[BudgetAllocator<br/>5 trim levels]
    BA --> RV{ROUGE-L + keyword<br/>validation}
    RV -- fail --> RB[Rollback to full context]
    RV -- pass --> LLM([LLM Provider])
    DD[DriftDetector<br/>stat → MD5 → AST] --> SM
    RC[RecoveryManager] --> SM
```

| Phase | Component | What it does |
|-------|-----------|--------------|
| 1 | `runtime/` `checkpoint/` | Async state machine, JSONL checkpoints, tool timeout breaker |
| 2–3 | `memory/` `context/` | Three-layer memory (TaskSummary / FileSnapshot / EpisodicNotes) + budget-allocated context assembly (5 trim levels + per-section caps + head/tail truncation) |
| 4 | `context/validator.py` | ROUGE-L + keyword compression validation with rollback |
| 5 | `filesystem/drift.py` | Three-level file drift detection (stat → MD5 → AST symbols+signatures) |
| 6 | `checkpoint/recovery.py` | Drift-aware recovery planning across 10 interruption scenarios, with content-restore for deleted files |
| 7 | `tools/` | Tool registry with duplicate-call interception (5s window, side-effect-safe) |
| 8 | `benchmarks/` | 7 benchmark experiments with baselines and ablations (incl. agent-level integration eval) |
| 9 | `llm/` | OpenAI-compatible provider layer (DeepSeek tested) + offline mock |
| 10 | `runtime/runtime.py` `llm/` | LLM-driven agent loop: context assembly → OpenAI-style tool calls → done signal (mock fallback kept for offline tests) |
| 11 | `mcp.py` `tools/builtin` | MCP client (stdio JSON-RPC, zero deps): any MCP server's tools join the registry; `update_plan` tool lets the model maintain its own plan |

## Benchmark Results

All numbers below are produced by the code in this repository. Run `python scripts/run_benchmark.py --all` to reproduce the six offline experiments; see [Reproducing the e2e experiment](#reproducing-the-e2e-experiment) for the real-LLM one. Per-experiment Markdown reports are committed under `src/tether/benchmarks/results/`; raw JSON/CSV dumps regenerate deterministically (seeded) via the same command for the six offline experiments. The e2e experiment needs a paid API key, so only its Markdown report is committed (no `results.json`/`results.csv`).

> **What the numbers measure.** The compression/memory/drift/recovery/intercept experiments drive the *components* directly (offline-reproducible, seeded); the **agent experiment (7)** drives the *real* `TetherRuntime` loop end-to-end with a deterministic goal-directed policy, closing that gap. Model intelligence is only measured by the e2e experiment (6).

### 1. Context compression (20 HumanEval tasks, offline validation)

| Strategy | Prompt tokens | Information preserved |
|----------|--------------|----------------------|
| Full context (baseline) | 681 | 100% |
| Last-N truncation | 537 (−21%) | **65%** — drops early tool results |
| **BudgetAllocator (ours)** | **531 (−22%)** | **100%** — ROUGE-L validated, rollback on failure |

### 2. Real-LLM end-to-end, pass@1 (20 HumanEval tasks × 3 arms, DeepSeek, greedy decoding)

Each arm sends a realistic agent context (task summary, file snapshots, tool history, episodic notes) + the task to `deepseek-v4-flash`; the completion is executed against the official HumanEval tests.

| Arm | pass@1 | Avg prompt tokens | Avg total tokens (incl. reasoning) | Avg latency |
|-----|--------|-------------------|-----------------------------------|-------------|
| Full context | 65% (13/20) | 795 | 4548 | 77.2s |
| Last-N | 80% (16/20) | 673 | 2760 | 49.0s |
| **BudgetAllocator (ours)** | **80% (16/20)** | 708 (−11%) | 3609 (**−21%**) | 62.3s |

Takeaway: at n=20 the pass@1 differences are within noise, but the compressed arms *never underperform* the full context while cutting total token cost by 21% — consistent with shorter contexts reducing distraction for reasoning models ("lost in the middle"). Most non-passes are reasoning-budget exhaustion (empty completions at the 8192-token cap) on the lightweight flash model, not wrong code.

### 3. Memory ablation (20 tasks)

| Memory config | Avg disk reads / task | Stale reads | Task success |
|---------------|----------------------|-------------|--------------|
| No memory | 30 | 6 | 100% (slow) |
| Flat cache | 2 | 5.5 | **45%** — stale-data failures |
| **Three-layer (ours)** | **2** | **0** | **100%** |

### 4. File drift detection (10 change types × 10 samples)

**100/100 accuracy** (0 false positives, 0 false negatives), avg **1.3ms** per file. Covers content edits, comment/whitespace changes, appends, deletion, rename, function add/remove/rename, signature changes, and concurrent multi-file edits.

### 5. Recovery (10 interruption scenarios × 5 runs)

**100% overall** recovery success, avg 0.6 steps lost. Every scenario recovers, including `file_deleted`: files the agent touched (read or wrote) get content-backed snapshots, so an external deletion is undone by writing the content back. Files never touched by the agent have no snapshot and remain unrecoverable — the detector still flags them, and recovery fails with an explicit reason.

### 6. Duplicate-call interception (20 simulated tasks)

**100% interception** (60/60 duplicate calls in the 5s window), saving 5100 tokens of redundant tool output.

### 7. Agent-level integration eval (20 scripted tasks × 5 task types, offline)

A deterministic, goal-directed policy stands in for the LLM and drives the **real `TetherRuntime` loop** (context assembly → structured tool calls → memory writes → checkpoints → events) over synthetic file-manipulation tasks; success is verified against the final workspace state.

| Task type | Result |
|-----------|--------|
| create-and-verify | 4/4 |
| edit-existing | 4/4 |
| search-then-fix (closed-loop, reacts to observations) | 4/4 |
| plan-execute (`update_plan` → execute) | 4/4 |
| test-and-report (real pytest) | 4/4 |
| **Overall** | **100% (20/20), avg 3.4 steps/task** |

This closes the gap the other experiments leave: it exercises the harness as one pipeline, offline and deterministically. Its first run caught a real defect (multi-line tool output truncated in the observation channel) — which is exactly what an integration eval is for.

## Quickstart

```bash
git clone https://github.com/223456png/tether.git
cd tether
pip install -e ".[dev]"
python -m pytest tests/ -q          # 133 tests, all offline
```

Or drive an agent task from the command line (no API key needed — it
falls back to the offline mock brain), then turn the event stream into
a one-page report:

```bash
tether run --goal "Create hello.txt containing 'hi' and verify it" --workspace ./ws
tether report --events ./ws/logs/events.jsonl          # markdown summary
```

Plug in any MCP server's tools with `--mcp-cmd` (repeatable):

```bash
tether run --goal "..." --workspace ./ws --mcp-cmd "python path/to/mcp_server.py"
```

Safety rails for real usage: `--require-approval` pauses mutating tools
for console confirmation, and `--max-tokens N` enforces a hard cost
ceiling (the task stops with status STOPPED and a saved checkpoint).

Run the offline benchmarks:

```bash
python scripts/run_benchmark.py --all            # 6 offline experiments
python scripts/run_benchmark.py --experiment compression --num-samples 5
```

### Running the LLM-driven agent loop

```python
import asyncio
from pathlib import Path

from tether.llm.factory import create_provider_from_env
from tether.runtime.runtime import TetherRuntime

# DEEPSEEK_API_KEY (or TETHER_LLM_*) set -> real provider; otherwise the
# offline MockProvider (is_mock=True), so this snippet always runs.
provider, is_mock = create_provider_from_env()

runtime = TetherRuntime(
    goal="Create hello.txt containing 'hi' and verify it reads back",
    workspace_dir=Path("./my-workspace"),
    llm_provider=provider,
    max_steps=30,
)
asyncio.run(runtime.run())
print(runtime.state.status, runtime.state.final_answer)
print(runtime.state.total_tokens, "tokens")
```

Each turn: `ContextAssembler` builds the system context from the three-layer
memory under the token budget → the model returns an OpenAI-style tool call
or a final answer → the tool executes through the registry (with
duplicate-call interception and workspace-boundary checks) → state, memory
and a JSONL event stream (`logs/events.jsonl`) are updated → checkpoint.

Two loop behaviors worth knowing:

- **Errors are observations, not verdicts.** With
  `max_consecutive_failures=N` (the CLI default is 3), a tool exception or
  timeout is fed back into the tool history so the model can retry with
  different arguments; a circuit breaker fails the task only after N
  *consecutive* failures. The default `0` preserves strict fail-fast for
  the mock brain, which cannot adapt.
- **`run_test` is real.** It spawns `python -m pytest <file>` inside the
  workspace and returns the pass/fail summary — on failure the traceback
  tail comes back as the error, so the agent can see *why* and fix it.
- **Human-in-the-loop approval.** With `--require-approval` (or an
  `approval_gate` callable), mutating tools (`write_file`, `run_test`)
  pause for confirmation before executing; a rejection becomes a
  `DENIED` observation the model can adapt to.
- **Hard cost ceiling.** `--max-tokens N` stops the loop (status
  STOPPED, checkpoint saved) once cumulative LLM token usage reaches N.
- **The model plans.** The `update_plan` tool writes the agent's plan
  into the never-pruned TaskSummary layer, so it survives compression
  and re-enters every turn's context.
- **Streaming output.** `--stream` (or an `on_llm_delta` callback)
  emits the model's answer live over SSE — providers without streaming
  support are called unchanged.

Without an API key the loop runs on a scripted/mock brain so every code path stays testable in CI at zero cost.

### Reproducing the e2e experiment

```bash
export DEEPSEEK_API_KEY=sk-...            # any OpenAI-compatible key works via TETHER_LLM_* vars
export TETHER_LLM_MODEL=deepseek-v4-flash
python scripts/run_benchmark.py --experiment e2e --num-samples 20
```

Without an API key the experiment falls back to an offline `MockProvider` and flags every result (`is_mock: true`) — the pipeline stays runnable in CI at zero cost.

## Limitations (read before citing numbers)

- **e2e sample size is small** (20 tasks × 3 arms). Differences of ≤15 percentage points are within noise; we report them as directionally consistent, not significant.
- **Benchmark numbers drive components, not the integrated loop** (see "What the numbers measure" above).
- **Code execution is not sandboxed.** HumanEval completions and test runs execute in plain subprocesses with timeouts (standard practice, but do not point this at untrusted models). Tools restrict file access to the workspace directory, but the loop itself is not a security boundary.
- **`file_deleted` recovery covers only touched files.** Files the agent read or wrote get content-backed snapshots (≤64KB each) and can be restored after external deletion; files never touched by the agent cannot be recovered.
- **The drift test set is self-constructed** (10 change types × 10 samples). 100% accuracy means the ten mutation classes are covered, not production-level generalization.
- **HumanEval subset is bundled offline** (20 problems) because the build environment had no network access to the upstream repo.
- **Compression ratio is ~22% vs learned compressors' 10–20×**; the trade is a hard semantic-preservation guarantee (every compression passes ROUGE-L + keyword validation or rolls back).

## Roadmap

Remaining ideas (design notes in `docs/designs/`):

1. **Sandboxed execution** — containerize test runs to make the
   "not a security boundary" caveat an actual guarantee.
2. **Cross-task episodic memory** — notes are task-scoped today;
   a shared store would let lessons transfer across tasks.
3. **Conversation compaction** — the tool history is windowed today;
   LLM/deterministic summarization would extend long-horizon recall.

Recently shipped (was on this list, now done):

- ✅ **Per-section budget caps** + head+tail truncation (−22% compression
  at 100% validation success).
- ✅ **Checkpoint compaction** — bounded JSONL history.
- ✅ **MCP client** — any MCP server's tools join the registry.
- ✅ **`update_plan`** — the model maintains its own plan in memory.
- ✅ **Approval gate + token budget** — human-in-the-loop and a hard
  cost ceiling.
- ✅ **Streaming** — SSE delta streaming end-to-end (provider → runtime
  callback → CLI `--stream`).
- ✅ **Agent-level integration eval** — 20 scripted tasks through the
  real loop, 100% pass (benchmark 7 above).

## Project Layout

```
src/tether/
├── runtime/        # state machine + LLM/mock agent loop + JSONL event stream
├── checkpoint/     # JSONL checkpoints + smart recovery
├── memory/         # TaskSummary / FileSnapshot / EpisodicNotes
├── context/        # BudgetAllocator (levels + per-section caps) + ROUGE-L validator
├── filesystem/     # DriftDetector (stat → MD5 → AST)
├── tools/          # per-runtime registry + side-effect-safe interceptor + real run_test
├── llm/            # OpenAI-compatible provider (function calling) + offline mock
├── mcp.py          # MCP client (stdio JSON-RPC): external tool servers -> registry
├── reporting.py    # events.jsonl -> one-page markdown run report
├── cli.py          # `tether run` / `tether report` command-line entry points
└── benchmarks/     # 7 experiments, metrics, reports, datasets (+ agent-level eval)
tests/              # 133 tests (all offline, incl. scripted-provider loop + MCP roundtrips)
docs/designs/       # per-phase design documents (HOTL contracts)
```

## License

MIT — see [LICENSE](LICENSE).
