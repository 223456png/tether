"""Builtin tools: read_file / write_file / search_code / run_test."""

from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from tether.tools.base import Tool, ToolResult

# Internal workspace directories never scanned by search_code.
_IGNORED_DIRS = {"checkpoints", "memory", "logs", ".git", "__pycache__",
                 ".pytest_cache", ".libs", "node_modules", ".venv"}


def resolve_in_workspace(workspace: Path, path: str) -> Tuple[Optional[Path], Optional[str]]:
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
        matches: List[str] = []
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
    """Runs tests (mocked for Phase 7; real runner comes later)."""

    name = "run_test"
    description = "Run a test file (mocked in Phase 7)"
    # Result depends on mutable workspace state: not safe to cache blindly.
    side_effect_free = False

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
                    "description": "Test file path relative to workspace",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        """Mock execution: report success once the file exists."""
        full_path, err = resolve_in_workspace(self.workspace, path)
        if err is not None:
            return ToolResult(success=False, error=err)
        logger.debug("run_test (mock) | {}", path)
        if not full_path.exists():
            return ToolResult(success=False, error=f"Test file not found: {path}")
        return ToolResult(
            success=True,
            output=f"✅ Tests passed: {path} (mock)",
            metadata={"path": path, "mock": True},
        )
