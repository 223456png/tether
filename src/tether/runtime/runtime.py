"""TetherRuntime: the main async loop (think -> tool -> checkpoint)."""

import asyncio
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from tether.checkpoint.manager import CheckpointManager
from tether.checkpoint.recovery import RecoveryManager
from tether.filesystem.drift import DriftDetector, DriftLevel
from tether.memory.file_snapshot import FileSnapshot
from tether.memory.store import MemoryStore
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


class TetherRuntime:
    """Phase-1 runtime: async state machine + JSONL checkpoints + tool timeout breaker.

    Phase 5: files are read through ``_read_file_with_drift_check`` so the
    agent never reasons from stale file state.
    Phase 6: checkpoints carry a workspace fingerprint and ``resume()``
    recovers smartly via RecoveryManager (drift-aware replay planning).
    Phase 7: actions run through the ToolRegistry with duplicate-call
    interception (5s window).
    """

    def __init__(self, goal: str, workspace_dir: Path, tool_timeout: int = 5) -> None:
        """Initialize runtime state, checkpoint manager and config."""
        self.workspace_dir = Path(workspace_dir)
        self.tool_timeout = tool_timeout
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
        self._step_log: Dict[int, List[str]] = {}
        self._tool_history: List[Tuple[int, str, str]] = []
        self._steps_to_replay: List[int] = []
        self._steps_skipped: List[int] = []

    def _register_builtin_tools(self) -> None:
        """Register the builtin tools bound to this workspace."""
        from tether.tools.builtin import (
            ReadFileTool,
            RunTestTool,
            SearchTool,
            WriteFileTool,
        )

        for tool_cls in (ReadFileTool, WriteFileTool, SearchTool, RunTestTool):
            self.tool_registry.register(tool_cls(self.workspace_dir))

    @staticmethod
    def _extract_files(action: str) -> List[str]:
        """Extract file-like arguments from an action string."""
        return _FILE_ARG_RE.findall(action)

    async def _think(self) -> str:
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
        return action

    def _parse_action(self, action: str) -> Tuple[str, dict]:
        """Parse an action string into (tool_name, params).

        Supports both ``tool(key='value')`` and positional
        ``tool('value')`` forms; positional values are stored under
        ``__positional__`` and mapped onto the tool's schema afterwards.
        """
        match = _ACTION_RE.match(action.strip())
        if not match:
            raise ValueError(f"Invalid action format: {action}")

        tool_name = match.group(1)
        params_str = match.group(2).strip()
        params: Dict[str, str] = {}

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

    async def _execute_tool(self, action: str) -> str:
        """Execute a real registered tool with duplicate-call interception.

        Timeout is enforced by the caller via ``asyncio.wait_for``.
        """
        tool_name, params = self._parse_action(action)

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        params = self._resolve_params(tool, params)

        # Duplicate call within the window -> cached result.
        cached = self.tool_interceptor.check(tool_name, params)
        if cached is not None:
            return f"[CACHED] {cached.output}"

        result = await tool.execute(**params)
        self.tool_interceptor.record(tool_name, params, result)
        logger.info(
            "Tool | action={} success={}", action, result.success,
        )
        if result.success:
            return result.output
        return f"ERROR: {result.error}"

    async def _save_checkpoint(self) -> None:
        """Persist a full checkpoint (state + workspace fingerprint + step log)."""
        self.checkpoint_manager.save_full(
            self.state, self.memory_store, self._step_log
        )

    async def _read_file_with_drift_check(self, file_path: str) -> Tuple[str, FileSnapshot]:
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

        new_snap = FileSnapshot.from_file(self.state.task_id, full_path)
        new_snap.path = rel  # store paths relative to the workspace
        self.memory_store.save_file_snapshot(new_snap)
        return full_path.read_text(encoding="utf-8", errors="replace"), new_snap

    async def _transition_to(self, new_status: TaskStatus) -> None:
        """Transition to a new status and log the change."""
        old = self.state.status
        self.state.status = new_status
        self.state.touch()
        logger.info("State | {} -> {}", old.value, new_status.value)

    async def run(self) -> None:
        """Main loop: think -> wait_tool -> execute (with timeout) until done."""
        await self._transition_to(TaskStatus.RUNNING)
        logger.info("Task started | task_id={} goal={}", self.state.task_id, self.state.goal)

        while self.state.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            self.state.step_index += 1
            self.state.touch()

            action = await self._think()
            self._step_log[self.state.step_index] = self._extract_files(action)
            await self._transition_to(TaskStatus.WAITING_TOOL)
            await self._save_checkpoint()

            try:
                logger.info("Tool | step={} action={} timeout={}s executing",
                            self.state.step_index, action, self.tool_timeout)
                result = await asyncio.wait_for(
                    self._execute_tool(action), timeout=self.tool_timeout
                )
                self._tool_history.append((self.state.step_index, action, result))
                await self._transition_to(TaskStatus.RUNNING)
                logger.info("Step {} succeeded", self.state.step_index)
            except asyncio.TimeoutError:
                self.state.error_message = (
                    f"Tool timeout: '{action}' did not finish within "
                    f"{self.tool_timeout}s at step {self.state.step_index}"
                )
                await self._transition_to(TaskStatus.FAILED)
                logger.error("Timeout | {}", self.state.error_message)
                await self._save_checkpoint()
                break
            except Exception as exc:
                self.state.error_message = f"{type(exc).__name__}: {exc}"
                await self._transition_to(TaskStatus.FAILED)
                logger.error("Tool failed | {}", self.state.error_message)
                await self._save_checkpoint()
                break

            if self.state.step_index >= 5:
                await self._transition_to(TaskStatus.COMPLETED)
                break

        await self._save_checkpoint()
        if self.state.status == TaskStatus.COMPLETED:
            logger.info("Task completed | task_id={} steps={}", self.state.task_id, self.state.step_index)
        else:
            logger.error("Task failed | task_id={} error={}", self.state.task_id, self.state.error_message)

    def _clear_affected_tool_results(self, steps: List[int]) -> None:
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
        """
        logger.info("🔄 Attempting smart recovery for task {}", task_id)

        result = self.recovery_manager.recover(task_id)
        if not result.success:
            logger.error("Recovery analysis failed: {}", result.reason)
            raise RuntimeError(f"Cannot recover: {result.reason}")

        assert result.task_state is not None
        self.state = result.task_state
        await self._transition_to(TaskStatus.RECOVERING)

        self._clear_affected_tool_results(result.steps_to_replay)
        self._steps_to_replay = result.steps_to_replay
        self._steps_skipped = result.steps_to_skip

        # Continue from the checkpointed step log so future checkpoints
        # keep the full history.
        snapshot = self.checkpoint_manager.load_full(task_id)
        if snapshot is not None:
            self._step_log = {int(k): v for k, v in snapshot.step_log.items()}

        self.state.error_message = None
        await self._transition_to(TaskStatus.RUNNING)
        logger.info(
            "✅ Recovery complete. Replaying steps: {}", result.steps_to_replay,
        )
        await self.run()
