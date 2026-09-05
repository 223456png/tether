"""Builtin tools: read_file / write_file / search_code / run_test / update_plan."""

import asyncio
import subprocess
import sys
from pathlib import Path

from tether.memory.store import MemoryStore
from tether.tools.base import Tool, ToolResult

# Internal workspace directories never scanned by search_code.
_IGNORED_DIRS = {"checkpoints", "memory", "logs", ".git", "__pycache__",
                 ".pytest_cache", ".libs", "node_modules", ".venv"}


def resolve_in_workspace(workspace: Path, path: str) -> tuple[Path | None, str | None]:
    """Resolve ``path`` under ``workspace``, refusing escapes.

    Returns ``(full_path, None)`` on success or ``(None, error_message)``
    when the resolved target falls outside the workspace (absolute paths,
    ``..`` traversal, drive-relative tricks). Symlinks are resolved before
    the boundary check, so a link pointing outside is rejected too.
    """
    candidate = (workspace / path).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        return None, f"Path escapes workspace: {path}"
    return candidate, None


class ReadFileTool(Tool):
    """Reads a file from the workspace."""

    name = "read_file"
    description = "Read the content of a file in the workspace"
    side_effect_free = True

    def __init__(self, workspace: Path) -> None:
        """Bind the tool to a workspace root."""
        self.workspace = Path(workspace)

    def get_parameters_schema(self) -> dict:
        """Schema: one required string ``path``."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        """Return the file content, or a structured error."""
        full_path, err = resolve_in_workspace(self.workspace, path)
        if err is not None:
            return ToolResult(success=False, error=err)
        try:
            content = full_path.read_text(encoding="utf-8")
            return ToolResult(
                success=True,
                output=content,
                metadata={"path": path, "size": len(content)},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path}")
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))


class WriteFileTool(Tool):
    """Writes a file into the workspace."""

    name = "write_file"
    description = "Write content to a file in the workspace"
    # Mutates the workspace: never cached by the duplicate-call interceptor.
    side_effect_free = False

    def __init__(self, workspace: Path) -> None:
        """Bind the tool to a workspace root."""
        self.workspace = Path(workspace)

    def get_parameters_schema(self) -> dict:
        """Schema: required ``path`` and ``content``."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        """Create parent dirs and write the file atomically (tmp + replace)."""
        full_path, err = resolve_in_workspace(self.workspace, path)
        if err is not None:
            return ToolResult(success=False, error=err)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = full_path.with_suffix(full_path.suffix + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(full_path)
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} chars to {path}",
                metadata={"path": path},
            )
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))


class SearchTool(Tool):
    """Greps the workspace for a pattern."""

    name = "search_code"
    description = "Search the workspace for lines matching a pattern"
    side_effect_free = True

    def __init__(self, workspace: Path) -> None:
        """Bind the tool to a workspace root."""
        self.workspace = Path(workspace)

    def get_parameters_schema(self) -> dict:
        """Schema: one required string ``pattern``."""
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Substring to search for",
                }
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str) -> ToolResult:
        """Scan workspace text files and return up to 20 matching lines."""
        matches: list[str] = []
        if self.workspace.exists():
            for p in sorted(self.workspace.rglob("*")):
                if len(matches) >= 20:
                    break
                if not p.is_file():
                    continue
                rel = p.relative_to(self.workspace)
                if rel.parts and rel.parts[0] in _IGNORED_DIRS:
                    continue
                try:
                    for i, line in enumerate(
                        p.read_text(encoding="utf-8", errors="replace").splitlines(),
                        start=1,
                    ):
                        if pattern in line:
                            matches.append(f"{rel.as_posix()}:{i}: {line.strip()}")
                            if len(matches) >= 20:
                                break
                except OSError:
                    continue
        if not matches:
            return ToolResult(
                success=True,
                output=f"No matches for '{pattern}'",
                metadata={"count": 0},
            )
        return ToolResult(
            success=True,
            output="\n".join(matches),
            metadata={"count": len(matches)},
        )


class RunTestTool(Tool):
    """Runs a pytest test file in a subprocess and summarizes the result."""

    name = "run_test"
    description = (
        "Run a pytest test file (path relative to workspace) and return "
        "the summarized pass/fail output"
    )
    # Result depends on mutable workspace state: not safe to cache blindly.
    side_effect_free = False

    # Output tail kept as the summary (pytest prints its verdict last).
    _OUTPUT_TAIL_LINES = 40

    def __init__(self, workspace: Path, run_timeout: int = 60) -> None:
        """Bind the tool to a workspace root and a subprocess timeout."""
        self.workspace = Path(workspace)
        self.run_timeout = run_timeout

    def get_parameters_schema(self) -> dict:
        """Schema: one required string ``path``."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Test file path relative to workspace",
                }
            },
            "required": ["path"],
        }

    @staticmethod
    def _tail(text: str, lines: int) -> str:
        """Return the last ``lines`` lines of ``text``."""
        stripped = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(stripped[-lines:])

    async def execute(self, path: str) -> ToolResult:
        """Run ``python -m pytest <path>`` inside the workspace.

        The subprocess runs with the workspace as cwd so test-relative
        imports resolve; ``subprocess.run`` kills the child on timeout.
        Success means exit code 0; on failure the output tail is returned
        as the error so the agent can see *why* the tests failed.
        """
        full_path, err = resolve_in_workspace(self.workspace, path)
        if err is not None:
            return ToolResult(success=False, error=err)
        if not full_path.exists():
            return ToolResult(success=False, error=f"Test file not found: {path}")

        cmd = [
            sys.executable, "-m", "pytest", str(full_path),
            "-q", "--no-header", "-p", "no:cacheprovider",
        ]
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                timeout=self.run_timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Test run timed out after {self.run_timeout}s: {path}",
                metadata={"path": path, "timed_out": True},
            )
        except OSError as exc:
            return ToolResult(success=False, error=f"Failed to spawn pytest: {exc}")

        text = completed.stdout.decode("utf-8", errors="replace")
        text += completed.stderr.decode("utf-8", errors="replace")
        summary = self._tail(text, self._OUTPUT_TAIL_LINES)
        exit_code = completed.returncode
        if exit_code == 0:
            return ToolResult(
                success=True,
                output=f"✅ Tests passed: {path}\n{summary}",
                metadata={"path": path, "exit_code": 0},
            )
        return ToolResult(
            success=False,
            output=summary,
            error=f"Tests failed (exit {exit_code}): {path}\n{summary}",
            metadata={"path": path, "exit_code": exit_code},
        )


class UpdatePlanTool(Tool):
    """Lets the agent maintain its own plan in the TaskSummary layer.

    The plan is written to ``TaskSummary.current_plan`` — the one memory
    section the allocator never prunes — so it survives compression and
    re-enters every subsequent turn's context. The model should call this
    when the approach changes or a step completes.
    """

    name = "update_plan"
    description = (
        "Create or update your task plan (ordered steps). Call this when "
        "the approach changes or whenever a plan step completes."
    )
    # Mutates task memory: keep out of the duplicate-call cache.
    side_effect_free = False

    _MAX_STEPS = 20

    def __init__(
        self, workspace: Path, memory_store: MemoryStore, task_id: str, goal: str
    ) -> None:
        """Bind to the runtime's memory store, task id and goal."""
        self.workspace = Path(workspace)  # unused; kept for uniform construction
        self._store = memory_store
        self._task_id = task_id
        self._goal = goal

    def get_parameters_schema(self) -> dict:
        """Schema: one required array of ordered plan steps."""
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered plan steps (replace the whole plan)",
                }
            },
            "required": ["plan"],
        }

    async def execute(self, plan: list) -> ToolResult:
        """Replace ``TaskSummary.current_plan`` with the given steps."""
        steps = [str(step).strip() for step in plan if str(step).strip()][
            : self._MAX_STEPS
        ]
        if not steps:
            return ToolResult(success=False, error="Plan is empty")
        summary = self._store.load_task_summary(self._task_id)
        if summary is None:
            # Agents typically plan as their FIRST action — synthesize the
            # summary instead of failing.
            from tether.memory.task_summary import TaskSummary

            summary = TaskSummary(task_id=self._task_id, goal=self._goal)
        summary.current_plan = steps
        self._store.save_task_summary(summary)
        rendered = "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
        return ToolResult(
            success=True,
            output=f"Plan updated ({len(steps)} steps):\n{rendered}",
            metadata={"steps": len(steps)},
        )
