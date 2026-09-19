"""TetherRuntime: the main agent loop (think -> tool -> checkpoint).

Two "brain" modes share the same loop:

- ``llm_provider=None`` (default): the legacy deterministic mock thinks in
  random action strings — used by offline tests and demos.
- With an ``LLMProvider``: each turn assembles the context from the
  three-layer memory via ``ContextAssembler``, asks the model for an
  OpenAI-style tool call, executes it, and finishes when the model stops
  calling tools (or when ``max_steps`` is reached).
"""

import asyncio
import inspect
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tether.checkpoint.manager import CheckpointManager
from tether.checkpoint.recovery import RecoveryManager
from tether.context import BudgetAllocator, BudgetConfig, ContextAssembler
from tether.filesystem.drift import DriftDetector, DriftLevel
from tether.llm.base import LLMProvider
from tether.memory.episodic import EpisodicNotes
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.store import MemoryStore
from tether.memory.task_summary import TaskSummary
from tether.runtime.events import EventRecorder
from tether.runtime.state import TaskState, TaskStatus
from tether.tools.base import Tool
from tether.tools.intercept import CallInterceptor
from tether.tools.registry import ToolRegistry

# Matches file-like arguments inside action strings, e.g. read_file('src/a.py').
_FILE_ARG_RE = re.compile(r"'([^']+\.[A-Za-z0-9_]+)'")

# Parses "tool_name(args...)" action strings.
_ACTION_RE = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)

# Key used for positional arguments before schema mapping.
_POSITIONAL_KEY = "__positional__"

# How many recent tool results feed back into the assembled context.
_TOOL_HISTORY_WINDOW = 20

# How many recently touched files are marked "current" for the allocator.
_CURRENT_FILES_WINDOW = 8

# How many completed actions the TaskSummary keeps (older ones drop off).
_SUMMARY_HISTORY_CAP = 20

_SYSTEM_PROMPT_TEMPLATE = """You are Tether, a careful coding agent working inside a workspace directory.

Your goal: {goal}

Working rules:
- Call exactly one tool per turn to inspect or modify the workspace.
- The context above may contain compressed memory; re-read files when they may have changed.
- When the goal is fully achieved, stop calling tools and reply with a concise final summary of what you did."""

_STEP_PROMPT_TEMPLATE = (
    "Step {step}: continue working toward the goal. Call exactly one tool, "
    "or reply with the final summary if the goal is achieved."
)


@dataclass
class AgentDecision:
    """One agent turn: either a tool call or task completion.

    The LLM path produces structured decisions straight from the
    function-calling payload; the mock path parses legacy action strings
    into the same shape.
    """

    tool_name: str = ""
    params: dict[str, object] = field(default_factory=dict)
    finished: bool = False
    final_answer: str = ""
    raw_action: str = ""  # display string for logs / tool history


class TetherRuntime:
    """Async state machine + JSONL checkpoints + tool timeout breaker.

    Phase 5: files are read through ``_read_file_with_drift_check`` so the
    agent never reasons from stale file state.
    Phase 6: checkpoints carry a workspace fingerprint and ``resume()``
    recovers smartly via RecoveryManager (drift-aware replay planning).
    Phase 7: actions run through the ToolRegistry with duplicate-call
    interception (5s window).
    Phase 10: the loop can be driven by a real ``LLMProvider`` with
    structured tool calls instead of the mock thinker.
    """

    def __init__(
        self,
        goal: str,
        workspace_dir: Path,
        tool_timeout: int = 5,
        llm_provider: LLMProvider | None = None,
        max_steps: int = 5,
        max_consecutive_failures: int = 0,
        approval_gate: Callable[[str, dict], Awaitable[bool]] | None = None,
        approval_tools: set[str] | None = None,
        max_total_tokens: int | None = None,
        on_llm_delta: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize runtime state, checkpoint manager and config.

        ``llm_provider`` switches the loop from the offline mock thinker
        to real LLM-driven tool calling. ``max_steps`` is the hard cap on
        executed tool steps (the LLM path usually finishes earlier by
        replying without a tool call).

        ``max_consecutive_failures`` controls error handling: 0 (default)
        fails the task on the first tool exception/timeout — right for
        the mock brain, which cannot adapt. With an LLM, set it to 2-3:
        the error is fed back as an *observation* so the model can retry
        with different arguments, and the circuit breaker stops hopeless
        loops after N consecutive failures.

        Human-in-the-loop: when ``approval_gate`` (an async callable
        ``(tool_name, params) -> bool``) is set, every call to a tool in
        ``approval_tools`` (default: write_file, run_test) is paused for
        approval before execution. A rejection becomes a ``DENIED``
        observation the model can adapt to — the task is not failed.

        ``max_total_tokens`` stops the loop (status STOPPED, checkpoint
        saved) once cumulative LLM token usage reaches the cap: a hard
        cost ceiling. Raise the attribute and ``resume()`` to continue.

        ``on_llm_delta`` (a sync callable receiving each content chunk)
        enables streaming output for providers that support it — used by
        the CLI's ``--stream`` to show the model's answer live.
        """
        self.workspace_dir = Path(workspace_dir)
        self.tool_timeout = tool_timeout
        self.llm_provider = llm_provider
        self.max_steps = max_steps
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self.approval_gate = approval_gate
        self.approval_tools = approval_tools or {"write_file", "run_test"}
        self.max_total_tokens = max_total_tokens
        self.on_llm_delta = on_llm_delta
        self.state = TaskState(goal=goal)
        self.checkpoint_manager = CheckpointManager(self.workspace_dir)
        self.memory_store = MemoryStore(self.workspace_dir)
        self.drift_detector = DriftDetector(self.workspace_dir, enable_ast=True)
        self.recovery_manager = RecoveryManager(
            self.checkpoint_manager, self.memory_store, self.drift_detector
        )
        self.tool_registry = ToolRegistry()
        self.tool_interceptor = CallInterceptor(window_seconds=5)
        self._register_builtin_tools()
        self._stop_requested = False
        self._step_log: dict[int, list[str]] = {}
        self._tool_history: list[tuple[int, str, str]] = []
        self._touched_files: list[str] = []
        self._steps_to_replay: list[int] = []
        self._steps_skipped: list[int] = []
        self._mcp_clients: list = []
        self.event_recorder = EventRecorder(self.workspace_dir, task_id=self.state.task_id)
        self._rebuild_context_pipeline()

    def _rebuild_context_pipeline(self) -> None:
        """(Re)build the allocator/assembler from ``state.context_budget``.

        Rebuilt on recovery too, so a context-overflow recovery that
        lowers the budget actually tightens compression on the next turn.
        """
        self.allocator = BudgetAllocator(
            BudgetConfig(total_budget=self.state.context_budget)
        )
        self.assembler = ContextAssembler(allocator=self.allocator)

    def _register_builtin_tools(self) -> None:
        """Register builtin tools (bound to this workspace) + decorator tools."""
        from tether.tools.builtin import (
            ReadFileTool,
            RunTestTool,
            SearchTool,
            UpdatePlanTool,
            WriteFileTool,
        )
        from tether.tools.registry import get_default_registry

        for tool_cls in (ReadFileTool, WriteFileTool, SearchTool, RunTestTool):
            self.tool_registry.register(tool_cls(self.workspace_dir))
        # The planner writes into the TaskSummary layer (never pruned).
        self.tool_registry.register(
            UpdatePlanTool(
                self.workspace_dir, self.memory_store,
                self.state.task_id, self.state.goal,
            )
        )
        # Decorator-registered tools join the runtime's registry without
        # overwriting the workspace-bound builtins.
        self.tool_registry.merge(get_default_registry(), overwrite=False)

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------
    async def connect_mcp(
        self, command: list[str], server_name: str | None = None
    ) -> list[str]:
        """Connect to an MCP server over stdio and register its tools.

        The server's tools join this runtime's registry as
        ``mcp_{server}_{tool}`` and are immediately callable by the LLM
        through the normal tool-call path. Failures raise ``MCPError`` —
        callers decide whether an unreachable tool server is fatal.
        """
        from tether.mcp import MCPClient, StdioTransport, register_mcp_tools

        client = MCPClient(StdioTransport(command, cwd=self.workspace_dir))
        try:
            client.initialize()
            registered = register_mcp_tools(self.tool_registry, client, server_name)
        except Exception:
            client.close()  # never leak a spawned server process
            raise
        self._mcp_clients.append(client)
        self.event_recorder.record(
            "mcp_connected", server=server_name or "auto", tools=registered,
        )
        return registered

    async def aclose(self) -> None:
        """Close every MCP client and release resources.

        Public counterpart to :meth:`connect_mcp` so callers (the CLI,
        tests) never reach into ``_mcp_clients``. Idempotent.
        """
        for client in self._mcp_clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                logger.warning("Failed to close an MCP client", exc_info=True)
        self._mcp_clients.clear()

    # ------------------------------------------------------------------
    # Thinking (mock or LLM)
    # ------------------------------------------------------------------
    async def _think(self) -> AgentDecision:
        """One agent turn: pick the next tool call, or finish the task."""
        if self.llm_provider is None:
            return await self._mock_think()
        return await self._llm_think()

    async def _mock_think(self) -> AgentDecision:
        """Mock agent thinking: random delay, then a random action string."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        actions = [
            "read_file(path='src/main.py')",
            "search_code(pattern='def auth')",
            "write_file(path='src/notes.md', content='progress note')",
            "run_test(path='tests/test_auth.py')",
        ]
        action = random.choice(actions)
        logger.info("Think | step={} action={}", self.state.step_index, action)
        return self._decision_from_action(action)

    async def _llm_think(self) -> AgentDecision:
        """Assemble context from memory, then ask the LLM for the next move.

        A response with tool calls executes its first call; a response
        without any ends the task (content becomes the final answer).
        """
        assert self.llm_provider is not None
        context = self.assembler.assemble(
            system_prompt=self._system_prompt(),
            task_state=self.state,
            memory_store=self.memory_store,
            tool_results=[
                (display, output)
                for _, display, output in self._tool_history[-_TOOL_HISTORY_WINDOW:]
            ],
            current_files=list(self._touched_files[-_CURRENT_FILES_WINDOW:]),
        )
        # Mark the notes that entered the context (retrieval feedback).
        selected_ids = self.allocator.last_compression_stats.get(
            "selected_episodic_ids"
        ) or []
        self._touch_episodic_notes(list(selected_ids))
        messages = [
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": _STEP_PROMPT_TEMPLATE.format(
                    step=self.state.step_index
                ),
            },
        ]
        logger.info(
            "LLM think | step={} provider={} model={} compression_level={}",
            self.state.step_index, self.llm_provider.name,
            self.llm_provider.model, self.allocator.current_level.name,
        )
        extra_kwargs: dict = {}
        if (
            self.on_llm_delta is not None
            and "on_delta" in inspect.signature(
                self.llm_provider.complete
            ).parameters
        ):
            extra_kwargs["on_delta"] = self.on_llm_delta
        response = await self.llm_provider.complete(
            messages,
            tools=self.tool_registry.get_openai_tools_schema(),
            **extra_kwargs,
        )
        self.state.prompt_tokens += response.prompt_tokens
        self.state.completion_tokens += response.completion_tokens
        self.state.touch()
        self.event_recorder.record(
            "llm_turn",
            step=self.state.step_index,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            compression_level=self.allocator.current_level.name,
            tool_call=response.tool_calls[0].name if response.has_tool_calls else None,
            latency_ms=round(response.latency_ms, 1),
        )

        if response.has_tool_calls:
            call = response.tool_calls[0]
            logger.info(
                "LLM tool call | step={} tool={} params={}",
                self.state.step_index, call.name,
                self._format_call(call.name, call.arguments),
            )
            return AgentDecision(
                tool_name=call.name,
                params=dict(call.arguments),
                raw_action=self._format_call(call.name, call.arguments),
            )
        logger.info(
            "LLM finished | step={} answer={}",
            self.state.step_index, response.content[:200],
        )
        return AgentDecision(finished=True, final_answer=response.content.strip())

    def _system_prompt(self) -> str:
        """Render the system prompt (goal + working rules)."""
        return _SYSTEM_PROMPT_TEMPLATE.format(goal=self.state.goal)

    @staticmethod
    def _format_call(name: str, params: dict[str, object]) -> str:
        """Compact display string for a tool call (long values elided)."""
        inner = ", ".join(
            f"{key}={value!r}" if len(str(value)) <= 40 else f"{key}=<...>"
            for key, value in params.items()
        )
        return f"{name}({inner})"

    def _decision_from_action(self, action: str) -> AgentDecision:
        """Parse a legacy action string into a structured decision."""
        tool_name, params = self._parse_action(action)
        return AgentDecision(
            tool_name=tool_name, params=params, raw_action=action
        )

    # ------------------------------------------------------------------
    # Action parsing (legacy string path)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_files(action: str) -> list[str]:
        """Extract file-like arguments from an action string."""
        return _FILE_ARG_RE.findall(action)

    @staticmethod
    def _files_from_params(params: dict[str, object]) -> list[str]:
        """Extract file paths from structured params by parameter name."""
        return [
            Path(str(value)).as_posix()
            for key, value in params.items()
            if isinstance(value, str)
            and ("path" in key.lower() or "file" in key.lower())
            and value.strip()
        ]

    def _parse_action(self, action: str) -> tuple[str, dict]:
        """Parse an action string into (tool_name, params).

        Supports both ``tool(key='value')`` and positional
        ``tool('value')`` forms; positional values are stored under
        ``__positional__`` and mapped onto the tool's schema afterwards.
        Only used for legacy/mock action strings — the LLM path delivers
        structured arguments directly.
        """
        match = _ACTION_RE.match(action.strip())
        if not match:
            raise ValueError(f"Invalid action format: {action}")

        tool_name = match.group(1)
        params_str = match.group(2).strip()
        params: dict[str, str] = {}

        if params_str:
            if "=" in params_str.split(",")[0]:
                for pair in params_str.split(","):
                    key, value = pair.split("=", 1)
                    params[key.strip()] = value.strip().strip("'\"")
            else:
                # Single positional argument.
                params[_POSITIONAL_KEY] = params_str.strip().strip("'\"")
        return tool_name, params

    def _resolve_params(self, tool: Tool, params: dict) -> dict:
        """Map a positional argument onto the tool's first required param."""
        if _POSITIONAL_KEY not in params:
            return params
        schema = tool.get_parameters_schema()
        required = schema.get("required") or list(
            schema.get("properties", {}).keys()
        )
        if not required:
            raise ValueError(
                f"Tool {tool.name} takes no parameters, got positional arg"
            )
        positional = params.pop(_POSITIONAL_KEY)
        params[required[0]] = positional
        return params

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def _execute_tool(self, candidate: str | AgentDecision) -> str:
        """Execute one tool call and return the text result.

        Accepts a legacy action string (parsed here) or a structured
        ``AgentDecision`` (used verbatim — no string parsing on the LLM
        path). Timeout is enforced by the caller via ``asyncio.wait_for``.
        """
        if isinstance(candidate, AgentDecision):
            tool_name, params = candidate.tool_name, dict(candidate.params)
        else:
            tool_name, params = self._parse_action(candidate)
        return await self._execute_call(tool_name, params)

    async def _execute_call(self, tool_name: str, params: dict[str, object]) -> str:
        """Run a (tool, params) pair through registry + interceptor."""
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        params = self._resolve_params(tool, params)
        self._note_touched_files(params)
        cacheable = getattr(tool, "side_effect_free", True)

        # Human-in-the-loop: mutating tools pause for approval.
        if self.approval_gate is not None and tool_name in self.approval_tools:
            approved = await self.approval_gate(tool_name, params)
            self.event_recorder.record(
                "tool_approval", step=self.state.step_index,
                tool=tool_name, approved=approved,
            )
            if not approved:
                display = self._format_call(tool_name, params)
                logger.warning("Tool denied by approval gate | {}", display)
                self._update_task_summary(
                    display, success=False, error="denied by approval gate"
                )
                return f"DENIED: user rejected {display}"

        # Duplicate read-only call within the window -> cached result.
        cached = self.tool_interceptor.check(tool_name, params, cacheable=cacheable)
        if cached is not None:
            self.event_recorder.record(
                "tool_executed", step=self.state.step_index,
                tool=tool_name, cached=True, success=True,
            )
            return f"[CACHED] {cached.output}"

        result = await tool.execute(**params)
        self.tool_interceptor.record(tool_name, params, result, cacheable=cacheable)
        if not cacheable:
            # The workspace may have changed: no cached read can be trusted.
            self.tool_interceptor.invalidate_all()
        self.event_recorder.record(
            "tool_executed", step=self.state.step_index,
            tool=tool_name, cached=False, success=result.success,
        )
        logger.info(
            "Tool | tool={} params={} success={}",
            tool_name, self._format_call(tool_name, params), result.success,
        )
        display = self._format_call(tool_name, params)
        if result.success:
            if tool_name == "write_file":
                self._snapshot_written_file(params)
            self._update_task_summary(display, success=True)
            return result.output
        error = result.error or "unknown error"
        self._update_task_summary(display, success=False, error=error)
        self._record_mistake_note(display, error)
        return f"ERROR: {error}"

    def _snapshot_written_file(self, params: dict[str, object]) -> None:
        """Persist a content-backed snapshot for a file the agent wrote.

        Content-backed snapshots are what makes external deletions
        restorable (undo/restore); the 64KB per-file cap in FileSnapshot
        bounds the storage cost.
        """
        rel_path = params.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            return
        full_path = self.workspace_dir / rel_path
        if not full_path.exists():
            return
        snap = FileSnapshot.from_file(
            self.state.task_id, full_path, include_content=True
        )
        snap.path = Path(rel_path).as_posix()
        self.memory_store.save_file_snapshot(snap)
        logger.debug("Content-backed snapshot saved | path={}", snap.path)

    def _note_touched_files(self, params: dict[str, object]) -> None:
        """Track files touched by executed tools for context assembly."""
        for key, value in params.items():
            if (
                isinstance(value, str)
                and ("path" in key.lower() or "file" in key.lower())
                and value.strip()
            ):
                rel = Path(value).as_posix()
                if rel not in self._touched_files:
                    self._touched_files.append(rel)

    # ------------------------------------------------------------------
    # Memory writing (three layers are kept alive by the loop itself)
    # ------------------------------------------------------------------
    def _update_task_summary(
        self, action_display: str, success: bool, error: str | None = None
    ) -> None:
        """Refresh the TaskSummary layer after one executed step."""
        summary = self.memory_store.load_task_summary(self.state.task_id)
        if summary is None:
            summary = TaskSummary(task_id=self.state.task_id, goal=self.state.goal)
        if success:
            summary.completed.append(
                f"step {self.state.step_index}: {action_display}"
            )
            summary.completed = summary.completed[-_SUMMARY_HISTORY_CAP:]
            summary.next_action = f"continue after step {self.state.step_index}"
        else:
            summary.next_action = (
                f"step {self.state.step_index} failed ({action_display}): "
                "retry or work around"
            )
        self.memory_store.save_task_summary(summary)

    def _mark_task_finished(self, final_answer: str) -> None:
        """Close out the TaskSummary when the agent finishes the goal."""
        summary = self.memory_store.load_task_summary(self.state.task_id)
        if summary is None:
            summary = TaskSummary(task_id=self.state.task_id, goal=self.state.goal)
        summary.next_action = "task completed"
        if final_answer:
            summary.constraints = list(
                dict.fromkeys(summary.constraints + [f"final: {final_answer[:200]}"])
            )[-5:]
        self.memory_store.save_task_summary(summary)

    def _record_mistake_note(self, action_display: str, error: str) -> None:
        """Persist a 'mistake' episodic note for a failed tool call (deduped)."""
        content = f"{action_display} failed: {error}"
        for note in self.memory_store.load_episodic_notes(self.state.task_id):
            if note.content == content:
                return  # already learned this lesson this task
        self.memory_store.save_episodic_note(
            EpisodicNotes(
                task_id=self.state.task_id,
                type="mistake",
                content=content,
                confidence=0.8,
                source_task=self.state.task_id,
            )
        )
        logger.info("Episodic mistake note recorded | {}", content[:120])

    def _touch_episodic_notes(self, entry_ids: list[str]) -> None:
        """Increment usage stats for notes selected into the last context."""
        if not entry_ids:
            return
        selected = set(entry_ids)
        for note in self.memory_store.load_episodic_notes(self.state.task_id):
            if note.entry_id in selected:
                note.usage_count += 1
                note.last_used = datetime.now(timezone.utc)
                self.memory_store.save_episodic_note(note)

    # ------------------------------------------------------------------
    # Checkpointing / drift
    # ------------------------------------------------------------------
    async def _save_checkpoint(self) -> None:
        """Persist a full checkpoint (state + workspace fingerprint + step log)."""
        self.checkpoint_manager.save_full(
            self.state, self.memory_store, self._step_log
        )

    async def _read_file_with_drift_check(self, file_path: str) -> tuple[str, FileSnapshot]:
        """Read a file through the drift-detection flow.

        1. Load the cached snapshot (if any) and detect drift.
        2. MATCH / METADATA -> reuse the cached snapshot.
        3. Drifted or absent -> invalidate the old snapshot, re-read the
           file, build and store a fresh snapshot.
        """
        rel = Path(file_path).as_posix()
        full_path = self.workspace_dir / rel

        snap, drift = self.memory_store.load_snapshot_with_drift(
            self.state.task_id, rel, self.drift_detector
        )
        if snap is not None and drift.level in (DriftLevel.MATCH, DriftLevel.METADATA):
            logger.debug(
                "File read (cache hit) | path={} drift={}", rel, drift.level.name,
            )
            return full_path.read_text(encoding="utf-8", errors="replace"), snap

        if snap is not None:
            logger.info(
                "File drifted | path={} level={} -> refreshing snapshot",
                rel, drift.level.name,
            )
            self.memory_store.invalidate_file_snapshot(self.state.task_id, rel)

        if not full_path.exists():
            raise FileNotFoundError(f"File missing while reading: {rel}")

        new_snap = FileSnapshot.from_file(
            self.state.task_id, full_path, include_content=True
        )
        new_snap.path = rel  # store paths relative to the workspace
        self.memory_store.save_file_snapshot(new_snap)
        return full_path.read_text(encoding="utf-8", errors="replace"), new_snap

    async def _transition_to(self, new_status: TaskStatus) -> None:
        """Transition to a new status and log the change."""
        old = self.state.status
        self.state.status = new_status
        self.state.touch()
        logger.info("State | {} -> {}", old.value, new_status.value)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the loop to stop at the next safe point (checked per step).

        The task transitions to ``STOPPED`` and a checkpoint is saved, so
        ``resume()`` can continue it later.
        """
        self._stop_requested = True
        logger.info("Stop requested | task_id={}", self.state.task_id)

    async def _handle_tool_error(self, decision: AgentDecision, error_text: str) -> bool:
        """Process a tool exception/timeout. Returns True to fail the task.

        With ``max_consecutive_failures == 0`` every error is fatal. With
        a cap set, the error becomes an *observation*: it lands in the
        tool history (so an LLM can read it and adapt on the next turn),
        a mistake note is recorded, and the loop continues until N
        consecutive failures trip the circuit breaker.
        """
        self._consecutive_failures += 1
        display = decision.raw_action
        self._tool_history.append((self.state.step_index, display, f"ERROR: {error_text}"))
        self._update_task_summary(display, success=False, error=error_text)
        self._record_mistake_note(display, error_text)
        self.event_recorder.record(
            "tool_error", step=self.state.step_index,
            tool=decision.tool_name, consecutive=self._consecutive_failures,
            error=error_text[:200],
        )

        if self.max_consecutive_failures <= 0:
            should_fail = True
        else:
            should_fail = self._consecutive_failures >= self.max_consecutive_failures

        if should_fail:
            self.state.error_message = error_text
            await self._transition_to(TaskStatus.FAILED)
            logger.error("Tool failed | {}", error_text)
            await self._save_checkpoint()
            return True

        logger.warning(
            "Tool error kept as observation ({}/{} consecutive failures) | {}",
            self._consecutive_failures, self.max_consecutive_failures,
            error_text[:120],
        )
        await self._transition_to(TaskStatus.RUNNING)
        return False

    async def run(self) -> None:
        """Main loop: think -> (tool | finish) -> checkpoint until done."""
        await self._transition_to(TaskStatus.RUNNING)
        logger.info(
            "Task started | task_id={} goal={} brain={}",
            self.state.task_id, self.state.goal,
            "llm" if self.llm_provider is not None else "mock",
        )
        self.event_recorder.record(
            "task_started", goal=self.state.goal,
            brain="llm" if self.llm_provider is not None else "mock",
            max_steps=self.max_steps,
        )

        while self.state.status not in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED,
        ):
            if self._stop_requested:
                await self._transition_to(TaskStatus.STOPPED)
                logger.info("Task stopped by request | step={}", self.state.step_index)
                break

            # Hard cost ceiling: stop before burning another LLM turn.
            if (
                self.max_total_tokens is not None
                and self.llm_provider is not None
                and self.state.total_tokens >= self.max_total_tokens
            ):
                self.state.error_message = (
                    f"Token budget exhausted: {self.state.total_tokens} tokens used "
                    f">= cap {self.max_total_tokens}"
                )
                await self._transition_to(TaskStatus.STOPPED)
                logger.warning("Token budget reached | {}", self.state.error_message)
                break

            self.state.step_index += 1
            self.state.touch()

            try:
                decision = await self._think()
            except Exception as exc:
                self.state.error_message = (
                    f"{type(exc).__name__}: {exc} (while thinking at step "
                    f"{self.state.step_index})"
                )
                await self._transition_to(TaskStatus.FAILED)
                logger.error("Think failed | {}", self.state.error_message)
                await self._save_checkpoint()
                break

            # Legacy monkeypatched _think implementations return strings.
            if isinstance(decision, str):
                decision = self._decision_from_action(decision)

            self._step_log[self.state.step_index] = (
                self._files_from_params(decision.params)
                or self._extract_files(decision.raw_action)
            )

            if decision.finished:
                self.state.final_answer = decision.final_answer or None
                self._mark_task_finished(decision.final_answer)
                await self._transition_to(TaskStatus.COMPLETED)
                logger.info(
                    "Task finished by agent | step={} answer={}",
                    self.state.step_index,
                    (decision.final_answer or "")[:200],
                )
                break

            await self._transition_to(TaskStatus.WAITING_TOOL)
            await self._save_checkpoint()

            try:
                logger.info(
                    "Tool | step={} action={} timeout={}s executing",
                    self.state.step_index, decision.raw_action, self.tool_timeout,
                )
                result = await asyncio.wait_for(
                    self._execute_tool(decision), timeout=self.tool_timeout
                )
                self._tool_history.append(
                    (self.state.step_index, decision.raw_action, result)
                )
                self._consecutive_failures = 0
                await self._transition_to(TaskStatus.RUNNING)
                logger.info("Step {} succeeded", self.state.step_index)
            except asyncio.TimeoutError:
                error_text = (
                    f"Tool timeout: '{decision.raw_action}' did not finish within "
                    f"{self.tool_timeout}s at step {self.state.step_index}"
                )
                should_fail = await self._handle_tool_error(decision, error_text)
                if should_fail:
                    break
            except Exception as exc:
                error_text = (
                    f"{type(exc).__name__}: {exc} at step {self.state.step_index}"
                )
                should_fail = await self._handle_tool_error(decision, error_text)
                if should_fail:
                    break

            if self.state.step_index >= self.max_steps:
                logger.warning(
                    "Step cap {} reached; ending task | task_id={}",
                    self.max_steps, self.state.task_id,
                )
                await self._transition_to(TaskStatus.COMPLETED)
                break

        await self._save_checkpoint()
        if self.state.status == TaskStatus.COMPLETED:
            logger.info(
                "Task completed | task_id={} steps={} tokens={}",
                self.state.task_id, self.state.step_index,
                self.state.total_tokens,
            )
            self.event_recorder.record(
                "task_completed", steps=self.state.step_index,
                total_tokens=self.state.total_tokens,
            )
        else:
            logger.error(
                "Task ended | task_id={} status={} error={}",
                self.state.task_id, self.state.status.value,
                self.state.error_message,
            )
            self.event_recorder.record(
                "task_ended", status=self.state.status.value,
                steps=self.state.step_index,
                error=self.state.error_message,
            )

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _clear_affected_tool_results(self, steps: list[int]) -> None:
        """Drop cached tool results for the steps being replayed."""
        if not steps:
            return
        replay = set(steps)
        before = len(self._tool_history)
        self._tool_history = [
            entry for entry in self._tool_history if entry[0] not in replay
        ]
        logger.info(
            "Tool history cleared | removed {} entries for steps {}",
            before - len(self._tool_history), sorted(replay),
        )

    async def resume(self, task_id: str) -> None:
        """Smart recovery entry point (Phase 6).

        Analyzes the latest checkpoint + workspace drift, invalidates stale
        snapshots, clears affected tool results, then continues the loop.
        The context pipeline is rebuilt from the recovered state so a
        lowered ``context_budget`` takes effect immediately.
        """
        logger.info("🔄 Attempting smart recovery for task {}", task_id)

        result = self.recovery_manager.recover(task_id)
        if not result.success:
            logger.error("Recovery analysis failed: {}", result.reason)
            raise RuntimeError(f"Cannot recover: {result.reason}")

        assert result.task_state is not None
        self.state = result.task_state
        self._rebuild_context_pipeline()
        await self._transition_to(TaskStatus.RECOVERING)

        self._clear_affected_tool_results(result.steps_to_replay)
        self._steps_to_replay = result.steps_to_replay
        self._steps_skipped = result.steps_to_skip

        # Continue from the checkpointed step log so future checkpoints
        # keep the full history (and context assembly sees touched files).
        snapshot = self.checkpoint_manager.load_full(task_id)
        if snapshot is not None:
            self._step_log = {int(k): v for k, v in snapshot.step_log.items()}
            self._touched_files = [
                path
                for files in self._step_log.values()
                for path in files
            ][-_CURRENT_FILES_WINDOW:]

        self.state.error_message = None
        await self._transition_to(TaskStatus.RUNNING)
        self.event_recorder.record(
            "recovery", scenario=result.scenario.name if result.scenario else None,
            replay=result.steps_to_replay, skip=result.steps_to_skip,
            backoff_seconds=result.backoff_seconds,
        )
        logger.info(
            "✅ Recovery complete. Replaying steps: {}", result.steps_to_replay,
        )
        await self.run()
