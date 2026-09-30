"""BenchmarkRunner: executes the five Tether experiments."""

import csv
import functools
import json
import random
import tempfile
import time
import zlib
from pathlib import Path

from loguru import logger

from tether.benchmarks.config import ExperimentConfig
from tether.benchmarks.datasets import (
    RECOVERY_SCENARIOS,
    apply_mutation,
    generate_drift_samples,
    load_humaneval,
    run_humaneval_check,
)
from tether.benchmarks.metrics import (
    LATENCY_RESULT_FIELDS,
    TaskResult,
    compute_metrics,
    strip_latency_metrics,
    strip_latency_result,
)
from tether.checkpoint import CheckpointManager, RecoveryManager
from tether.context import (
    BudgetAllocator,
    BudgetConfig,
    CompressionLevel,
    RougeValidator,
)
from tether.context.assembler import ContextAssembler
from tether.context.budget import estimate_tokens
from tether.filesystem import DriftDetector
from tether.memory import (
    EpisodicNotes,
    FileSnapshot,
    MemoryStore,
    TaskSummary,
)


class BenchmarkRunner:
    """Loads data, runs one experiment end to end, saves results."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Bind the experiment config and prepare the output dir."""
        self.config = config
        self.results: list[TaskResult] = []
        self.metrics_by_variant: dict[str, dict] = {}
        self.output_dir = config.output_dir / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run_experiment(self) -> list[TaskResult]:
        """Dispatch to the experiment named in the config."""
        handlers = {
            "compression": self._run_compression,
            "memory": self._run_memory,
            "drift": self._run_drift,
            "recovery": self._run_recovery,
            "intercept": self._run_intercept,
            "agent": self._run_agent,
            "e2e": self._run_e2e,
        }
        handler = handlers.get(self.config.name)
        if handler is None:
            raise ValueError(f"Unknown experiment: {self.config.name}")
        self.results = await handler()
        return self.results

    def save_results(self) -> Path:
        """Persist per-variant results + metrics as JSON and CSV.

        墙钟 latency 是观测噪声（重跑必变），不进版控的 results.json/csv，
        单独落到 ``latency.json``（.gitignore）供本地性能回归观察——
        保证「固定种子确定性再生成」对**字节**也成立。
        """
        full_payload = {
            "experiment": self.config.name,
            "description": self.config.description,
            "variants": {
                variant: {
                    "results": [r.to_dict() for r in results],
                    "metrics": compute_metrics(results),
                }
                for variant, results in self._group_by_variant().items()
            },
        }
        deterministic = {
            "experiment": full_payload["experiment"],
            "description": full_payload["description"],
            "variants": {
                variant: {
                    "results": [
                        strip_latency_result(r) for r in data["results"]
                    ],
                    "metrics": strip_latency_metrics(data["metrics"]),
                }
                for variant, data in full_payload["variants"].items()
            },
        }
        json_path = self.output_dir / "results.json"
        json_path.write_text(
            json.dumps(deterministic, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 完整观测（含 latency）落到 gitignore 的 latency.json
        (self.output_dir / "latency.json").write_text(
            json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        csv_path = self.output_dir / "results.csv"
        csv_fields = [
            f for f in TaskResult.__dataclass_fields__
            if f not in LATENCY_RESULT_FIELDS
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in self.results:
                row = strip_latency_result(r.to_dict())
                row["extra"] = json.dumps(row.get("extra") or {}, ensure_ascii=False)
                writer.writerow(row)

        logger.info(
            "Results saved | {} -> {} (+ latency.json)", self.config.name, json_path
        )
        return json_path

    def _group_by_variant(self) -> dict[str, list[TaskResult]]:
        """Group results by experiment arm."""
        groups: dict[str, list[TaskResult]] = {}
        for r in self.results:
            groups.setdefault(r.variant, []).append(r)
        return groups

    # ------------------------------------------------------------------
    # Experiment 1: context compression
    # ------------------------------------------------------------------
    async def _run_compression(self) -> list[TaskResult]:
        """Compare full context / last-N / BudgetAllocator per task."""
        tasks = load_humaneval(self.config.num_samples)
        results: list[TaskResult] = []

        for task in tasks:
            data = self._build_task_context(task)
            variants = await self._measure_compression_variants(data)
            for variant, metrics in variants.items():
                results.append(TaskResult(
                    task_id=task["task_id"],
                    variant=variant,
                    success=metrics["success"],
                    prompt_tokens=metrics["prompt_tokens"],
                    total_tokens=metrics["prompt_tokens"],
                    latency_ms=metrics["latency_ms"],
                    extra={
                        "reduction_ratio": metrics["reduction_ratio"],
                        "level": metrics.get("level", 0),
                    },
                ))
        return results

    # Realistic agent-context templates shared by the compression and
    # e2e experiments: what a coding agent's working memory plausibly
    # contains while implementing a function (repo docs, helpers, test
    # output, style notes) without leaking the task answer.
    _TOOL_TEMPLATES = [
        ("read_file(path='README.md')",
         "Solution module. All functions must be pure Python, stdlib "
         "only. Type hints required on public APIs."),
        ("read_file(path='src/utils.py')",
         "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n\n"
         "def windows(seq, n):\n    return [seq[i:i+n] for i in "
         "range(len(seq)-n+1)]"),
        ("search_code(pattern='def solution')",
         "src/solution.py:1: incomplete implementation\n"
         "src/legacy.py:12: def solution_old (deprecated)"),
        ("run_test(path='tests/test_utils.py')",
         "3 passed in 0.02s"),
        ("read_file(path='docs/style.md')",
         "Style: prefer list comprehensions, avoid global state, keep "
         "functions under 20 lines, no early returns in loops."),
        ("run_test(path='tests/test_edges.py')",
         "2 passed in 0.01s"),
        ("read_file(path='src/legacy.py')",
         "# legacy helpers scheduled for deletion; do not reuse"),
        ("search_code(pattern='import numpy')",
         "no matches - stdlib project"),
    ]
    _NOTE_TEMPLATES = [
        ("lesson", "Type hints are enforced by CI in this repository."),
        ("lesson", "Prefer stdlib; third-party imports fail the build."),
        ("pattern", "Tests are deterministic; no network in CI."),
        ("lesson", "List comprehensions read better than map/filter here."),
        ("preference", "The reviewer dislikes nested loops deeper than 2."),
        ("lesson", "Edge cases (empty input) are always tested."),
    ]

    def _build_task_context(self, task: dict) -> dict:
        """Build memory content for one HumanEval task in a temp workspace."""
        # crc32 keeps the per-task RNG stable across runs (hash() is salted).
        rng = random.Random(zlib.crc32(task["task_id"].encode()))
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "src").mkdir()
            (ws / "src" / "solution.py").write_text(task["prompt"], encoding="utf-8")
            snap = FileSnapshot.from_file("bench", ws / "src" / "solution.py")

            tool_results = list(self._TOOL_TEMPLATES)
            notes = [
                EpisodicNotes(
                    task_id="bench",
                    type=note_type,
                    content=content,
                    confidence=rng.uniform(0.4, 0.95),
                    source_task="bench",
                )
                for note_type, content in self._NOTE_TEMPLATES
            ]
            return {
                "system": "You are a coding agent. Complete the task.",
                "summary": TaskSummary(
                    task_id="bench", goal=task["prompt"][:200],
                    current_plan=["read", "edit", "test"],
                    completed=["read"], next_action="edit solution",
                    constraints=["keep tests green"],
                ),
                "snapshots": [snap],
                "notes": notes,
                "tools": tool_results,
                "current_files": ["src/solution.py"],
                # Required info for this task: the goal plus one fact drawn
                # from the tool history (last-N drops the oldest three).
                "keywords": [
                    task["prompt"][:60],
                    "edit solution",
                    # Distinctive snippet from a randomly chosen tool result:
                    # if it lands in the first 3 (dropped by last-N), the
                    # naive truncation fails keyword validation.
                    self._TOOL_TEMPLATES[rng.randrange(0, 8)][1][:30],
                ],
            }

    async def _measure_compression_variants(self, data: dict) -> dict:
        """Measure tokens + semantic preservation for the three strategies.

        "Success" = the compressed context still passes Phase-4 validation
        (ROUGE-L >= 0.7 AND every required keyword survives). The Budget
        arm sweeps compression levels from most aggressive down and keeps
        the strongest one that validates; if none validates it rolls back
        to the full context (success preserved, zero savings).
        """
        variants: dict = {}
        validator = RougeValidator(threshold=0.7)
        keywords = data["keywords"]

        allocator = BudgetAllocator(BudgetConfig())
        start = time.perf_counter()
        full_sections = allocator.render_uncompressed(
            data["system"], data["summary"], data["snapshots"],
            data["notes"], data["tools"], data["current_files"],
        )
        full_context = ContextAssembler._join(full_sections)
        full_tokens = estimate_tokens(full_context)
        variants["full"] = {
            "success": True, "prompt_tokens": full_tokens,
            "reduction_ratio": 0.0, "level": 0, "context": full_context,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

        # last-N: naive truncation keeping only the last 5 tool results,
        # with no validation safety net.
        start = time.perf_counter()
        truncated_sections = allocator.render_uncompressed(
            data["system"], data["summary"], data["snapshots"],
            data["notes"], data["tools"][-5:], data["current_files"],
        )
        truncated = ContextAssembler._join(truncated_sections)
        last_n_tokens = estimate_tokens(truncated)
        last_n_ok = validator.validate(
            full_context, truncated, required_keywords=keywords
        ).passed
        variants["last_n"] = {
            "success": last_n_ok, "prompt_tokens": last_n_tokens,
            "reduction_ratio": 1 - last_n_tokens / max(1, full_tokens),
            "level": 0, "context": truncated,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

        # BudgetAllocator arm: lowest-token candidate that still validates.
        # Search space: (a) the global level ladder, plus (b) per-section
        # refinements — sections are independent under the caps model, so
        # for each validating level the tool section is additionally
        # recency-fitted down a halving ladder of its ratio share
        # (25% / 12.5% / 6.25% of full-context tokens). Every candidate
        # passes the same ROUGE-L + keyword gate; the cheapest survivor
        # wins. No validating candidate -> rollback to full (success
        # preserved, zero savings).
        start = time.perf_counter()
        levels = sorted(
            getattr(self.config, "budget_levels", [0, 1, 2, 3, 4]),
            reverse=True,
        )
        tool_share = getattr(self.config, "tool_result_ratio", 0.25)
        chosen_level, budget_tokens, budget_context = 0, full_tokens, full_context
        for level in levels:
            if level <= 0:
                continue
            sections = allocator.render_at_level(
                level, data["system"], data["summary"], data["snapshots"],
                data["notes"], data["tools"], data["current_files"],
            )
            candidate = ContextAssembler._join(sections)
            tokens = estimate_tokens(candidate)
            if not validator.validate(
                full_context, candidate, required_keywords=keywords
            ).passed:
                continue
            if tokens < budget_tokens:
                chosen_level, budget_tokens, budget_context = level, tokens, candidate
            # Per-section refinement: keep-newest tool fitting under the
            # halving ladder, rendered at the minimal truncation budget.
            for frac in (1.0, 0.5, 0.25):
                cap = int(full_tokens * tool_share * frac)
                fitted = dict(sections)
                fitted["tool_context"] = allocator._fit_tool_section(
                    data["tools"], CompressionLevel.MINIMAL, cap,
                )
                fit_ctx = ContextAssembler._join(fitted)
                fit_tokens = estimate_tokens(fit_ctx)
                if fit_tokens >= budget_tokens:
                    continue
                if validator.validate(
                    full_context, fit_ctx, required_keywords=keywords
                ).passed:
                    chosen_level = level
                    budget_tokens, budget_context = fit_tokens, fit_ctx
        reduction = 1 - budget_tokens / max(1, full_tokens)
        variants["budget"] = {
            # Rolled back to full context when no candidate validated: the
            # task still succeeds (Phase-4 rollback guarantee), just
            # without savings.
            "success": True,
            "prompt_tokens": budget_tokens,
            "reduction_ratio": reduction,
            "level": chosen_level, "context": budget_context,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
        return variants

    # ------------------------------------------------------------------
    # Experiment 2: memory ablation
    # ------------------------------------------------------------------
    async def _run_memory(self) -> list[TaskResult]:
        """Compare no-memory / flat / layered caching on file reads."""
        tasks = load_humaneval(self.config.num_samples)
        results: list[TaskResult] = []

        for task in tasks:
            rng = random.Random(zlib.crc32(task["task_id"].encode()))
            stats = self._simulate_reads(rng)
            for variant, (total, disk, stale, failed) in stats.items():
                results.append(TaskResult(
                    task_id=task["task_id"],
                    variant=variant,
                    success=not failed,
                    steps=10,
                    file_read_count=total,
                    disk_read_count=disk,
                    stale_read_count=stale,
                    extra={"reads": total, "disk": disk, "stale": stale},
                ))
        return results

    @staticmethod
    def _simulate_reads(rng: random.Random, accesses: int = 30) -> dict:
        """Simulate ``accesses`` file reads with one mid-run modification.

        Returns per-variant (total_reads, disk_reads, stale_reads, failed):
        - no_memory: every access hits the disk (no cache at all) but the
          content is always fresh, so the task never fails on staleness.
        - flat: path-keyed cache without drift checks -> one initial disk
          read; every read after the modification returns stale content.
          The task fails when the changed region matters for the remaining
          decisions (~75% of tasks).
        - layered: FileSnapshot + DriftDetector -> initial read plus one
          refresh when drift is detected; never stale, never fails.
        """
        # One modification among the accesses.
        change_at = rng.randrange(5, accesses - 5)
        # Whether the changed content affects the rest of the task
        # (the remaining 25% of tasks never touch the changed region).
        change_matters = rng.random() < 0.75

        return {
            "no_memory": (accesses, accesses, 0, False),
            # flat cache: 1 initial read, changes never noticed
            "flat": (accesses, 1, accesses - change_at, change_matters),
            # layered: initial read + one refresh when drift is detected
            "layered": (accesses, 2, 0, False),
        }

    # ------------------------------------------------------------------
    # Experiment 3: drift detection
    # ------------------------------------------------------------------
    async def _run_drift(self) -> list[TaskResult]:
        """Run DriftDetector over the 10-type mutation dataset."""
        samples = generate_drift_samples(
            samples_per_type=getattr(self.config, "samples_per_type", 10),
            seed=self.config.seed,
        )
        if self.config.num_samples:
            # Slice per change type so every type stays represented.
            by_type: dict[str, list[dict]] = {}
            for s in samples:
                by_type.setdefault(s["change_type"], []).append(s)
            samples = [
                s
                for change_type in by_type
                for s in by_type[change_type][: self.config.num_samples]
            ]
        results: list[TaskResult] = []

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            for sample in samples:
                info = apply_mutation(ws, sample)
                detector = DriftDetector(ws, enable_ast=True)

                # Baseline snapshot of the pristine file (pre-mutation copy).
                base_path = ws / f"src/{sample['sample_id']}_base.py"
                base_path.parent.mkdir(parents=True, exist_ok=True)
                base_path.write_text(sample["base_code"], encoding="utf-8")
                snap = FileSnapshot.from_file("bench", base_path)
                snap.path = info["path"]

                # Control: the pristine file (renamed path may not exist,
                # which is exactly the MISSING expectation).
                start = time.perf_counter()
                detected = detector.detect(snap)
                detect_ms = (time.perf_counter() - start) * 1000

                got = detected.level.name
                expected = sample["expected_level"]
                # Renames: the old path is gone -> MISSING is correct.
                correct = got == expected

                if expected == "MATCH":
                    tp_fp = "tn" if not detected.detected else "fp"
                else:
                    tp_fp = "tp" if correct else "fn"

                results.append(TaskResult(
                    task_id=sample["sample_id"],
                    variant=sample["change_type"],
                    success=correct,
                    latency_ms=detect_ms,
                    extra={
                        "drift": {
                            "tp": tp_fp == "tp",
                            "tn": tp_fp == "tn",
                            "fp": tp_fp == "fp",
                            "fn": tp_fp == "fn",
                            "detect_ms": detect_ms,
                            "expected": expected,
                            "got": got,
                        }
                    },
                ))
        return results

    # ------------------------------------------------------------------
    # Experiment 4: recovery scenarios
    # ------------------------------------------------------------------
    async def _run_recovery(self) -> list[TaskResult]:
        """Run RecoveryManager over the 10 interruption scenarios."""
        runs_per = getattr(self.config, "runs_per_scenario", 5)
        results: list[TaskResult] = []

        for scenario_name, error_message in RECOVERY_SCENARIOS:
            for run_idx in range(runs_per):
                result = self._run_single_recovery(
                    scenario_name, error_message, run_idx
                )
                results.append(result)
        return results

    def _run_single_recovery(
        self, scenario_name: str, error_message, run_idx: int
    ) -> TaskResult:
        """One recovery attempt in a fresh workspace."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            store = MemoryStore(ws)
            cm = CheckpointManager(ws)

            for rel in ("src/a.py", "src/b.py", "src/c.py"):
                p = ws / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("def f():\n    return 1\n", encoding="utf-8")
                # The runtime persists content-backed snapshots for files
                # the agent touched (read/write), so recovery can restore
                # externally deleted files. Simulate that here.
                snap = FileSnapshot.from_file("bench", p, include_content=True)
                snap.path = rel
                store.save_file_snapshot(snap)

            # Realize file-based scenarios.
            if scenario_name == "file_modified":
                (ws / "src/a.py").write_text(
                    "def f():\n    return 2  # changed\n", encoding="utf-8"
                )
            elif scenario_name == "file_deleted":
                (ws / "src/b.py").unlink()

            from tether.runtime.state import TaskState

            state = TaskState(task_id="bench", goal="bench goal")
            state.step_index = 3
            state.error_message = error_message
            cm.save_full(
                state, store,
                step_log={1: ["src/a.py"], 2: ["src/b.py"], 3: ["src/c.py"]},
            )

            rm = RecoveryManager(cm, store, DriftDetector(ws))
            outcome = rm.recover("bench")
            latency_ms = (time.perf_counter() - start) * 1000

            return TaskResult(
                task_id=f"{scenario_name}_run{run_idx}",
                variant=scenario_name,
                success=outcome.success,
                steps_lost=len(outcome.steps_to_replay),
                recovery_latency_ms=latency_ms,
                extra={"recovery": True, "scenario": scenario_name},
            )

    # ------------------------------------------------------------------
    # Experiment: agent-level integration eval
    # ------------------------------------------------------------------
    async def _run_agent(self) -> list[TaskResult]:
        """Drive the real ``TetherRuntime`` loop over scripted tasks.

        Each task runs in a fresh temp workspace with a deterministic,
        goal-directed policy standing in for the LLM; success is verified
        against the final workspace/runtime state. This is the offline
        end-to-end check the component-level experiments don't cover:
        it exercises context assembly, structured tool calls, memory
        writes, interception and checkpointing as one pipeline.
        """
        from tether.benchmarks.agent_tasks import (
            PolicyProvider,
            ScriptedPolicy,
            build_agent_tasks,
        )
        from tether.runtime.runtime import TetherRuntime

        total = self.config.num_samples or 20
        templates = build_agent_tasks()
        max_steps = getattr(self.config, "max_steps", 12)
        results: list[TaskResult] = []

        for run_idx in range(total):
            spec = templates[run_idx % len(templates)]
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                spec.setup(ws)
                provider = PolicyProvider(
                    ScriptedPolicy(spec.steps(run_idx), spec.answer)
                )
                runtime = TetherRuntime(
                    spec.goal, ws, llm_provider=provider,
                    max_steps=max_steps, tool_timeout=30,
                )
                # Record each task's verify_* result as a goal_verified
                # event, so the report pipeline can render ✓/✗ verdicts.
                runtime.goal_verifier = functools.partial(
                    spec.verify, ws, runtime, run_idx,
                )
                start = time.perf_counter()
                await runtime.run()
                duration_ms = (time.perf_counter() - start) * 1000

                success = spec.verify(ws, runtime, run_idx)
                goal_verified = any(
                    e["event"] == "goal_verified" and e.get("passed")
                    for e in runtime.event_recorder.query()
                )
                results.append(TaskResult(
                    task_id=f"agent-{spec.name}-{run_idx}",
                    variant=spec.name,
                    success=success,
                    steps=runtime.state.step_index,
                    prompt_tokens=runtime.state.prompt_tokens,
                    completion_tokens=runtime.state.completion_tokens,
                    latency_ms=duration_ms,
                    extra={
                        "agent": True,
                        "task": spec.name,
                        "status": runtime.state.status.value,
                        "goal_verified": goal_verified,
                    },
                ))
        return results

    # ------------------------------------------------------------------
    # Experiment 5: interception
    # ------------------------------------------------------------------
    async def _run_intercept(self) -> list[TaskResult]:
        """Measure duplicate-call interception over scripted task runs."""
        from tether.tools.intercept import CallInterceptor

        tasks = load_humaneval(self.config.num_samples)
        results: list[TaskResult] = []

        for task in tasks:
            interceptor = CallInterceptor(window_seconds=5)
            outputs = [
                f"tool output {i}: " + "payload " * 30 for i in range(5)
            ]
            # 5-step task; steps 2, 3, 5 duplicate earlier calls.
            script = [0, 0, 0, 1, 0]

            intercepted = 0
            duplicates = 0
            saved_tokens = 0
            seen: dict[str, str] = {}

            start = time.perf_counter()
            for _step, call_idx in enumerate(script, start=1):
                params = {"index": call_idx}
                key = json.dumps(params, sort_keys=True)
                is_duplicate = key in seen
                cached = interceptor.check("mock_tool", params)
                if cached is not None:
                    intercepted += 1
                    if is_duplicate:
                        duplicates += 1
                    saved_tokens += estimate_tokens(cached.output)
                else:
                    result_payload = outputs[call_idx]
                    from tether.tools.base import ToolResult

                    interceptor.record(
                        "mock_tool", params, ToolResult(success=True, output=result_payload)
                    )
                    seen[key] = result_payload
            latency_ms = (time.perf_counter() - start) * 1000

            results.append(TaskResult(
                task_id=task["task_id"],
                variant="interceptor",
                success=True,
                steps=5,
                intercepted=intercepted,
                duplicates=3,  # by construction
                saved_tokens=saved_tokens,
                latency_ms=latency_ms,
                extra={"intercept": True},
            ))
        return results

    # ------------------------------------------------------------------
    # Experiment 6: real-LLM end-to-end (pass@1)
    # ------------------------------------------------------------------
    async def _run_e2e(self) -> list[TaskResult]:
        """Measure real pass@1 under the three context strategies.

        For each HumanEval task the three variant contexts (from the
        compression experiment) are sent to a real LLM; its completion is
        executed against the official tests in a subprocess. Without an
        API key the MockProvider keeps the pipeline runnable offline
        (flagged ``is_mock`` in every result and the report).
        """
        from tether.llm.factory import create_provider_from_env

        provider, is_mock = create_provider_from_env()
        self.provider_name = provider.name
        self.provider_is_mock = is_mock
        if is_mock:
            logger.warning(
                "e2e running in MOCK mode - pass@1 is not meaningful; "
                "set DEEPSEEK_API_KEY for real results"
            )

        tasks = load_humaneval(self.config.num_samples)
        results: list[TaskResult] = []
        failed_tasks = 0

        for task in tasks:
            try:
                data = self._build_task_context(task)
                variants = await self._measure_compression_variants(data)
                for variant, metrics in variants.items():
                    user_message = (
                        f"{metrics['context']}\n\n"
                        "[TASK]\n"
                        "Complete the following Python function. "
                        "Return ONLY the function body (the code that follows "
                        "the signature), no markdown fences, no explanation.\n\n"
                        f"{task['prompt']}"
                    )
                    resp = await provider.complete(
                        [{"role": "user", "content": user_message}],
                        temperature=0.0,  # greedy: reproducible pass@1
                        # Reasoning models spend tokens on chain-of-thought
                        # before the answer; 512 starved them to empty output.
                        max_tokens=2048,
                    )
                    check = run_humaneval_check(
                        task, resp.content,
                        timeout_seconds=getattr(self.config, "execution_timeout", 10.0),
                    )
                    results.append(TaskResult(
                        task_id=task["task_id"],
                        variant=variant,
                        success=check["passed"],
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        total_tokens=resp.total_tokens,
                        latency_ms=resp.latency_ms,
                        extra={
                            "reduction_ratio": metrics["reduction_ratio"],
                            "level": metrics["level"],
                            "provider": provider.name,
                            "is_mock": is_mock,
                            "exit_code": check["exit_code"],
                            "stderr_tail": check["stderr_tail"],
                            "completion_head": resp.content[:240],
                        },
                    ))
            except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
                failed_tasks += 1
                logger.error(
                    "e2e task {} failed after retries: {} - skipping",
                    task["task_id"], exc,
                )
        if failed_tasks:
            logger.warning(
                "e2e completed with {}/{} tasks skipped due to errors",
                failed_tasks, len(tasks),
            )
        return results
