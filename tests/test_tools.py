"""Phase 7 tests: ToolRegistry, decorator, interceptor, runtime integration."""

import asyncio
import time
from pathlib import Path

import pytest

from tether.runtime.runtime import TetherRuntime
from tether.tools import (
    CallInterceptor,
    Tool,
    ToolRegistry,
    ToolResult,
    register_tool,
)
from tether.tools.builtin import ReadFileTool


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

def test_tool_registration(tmp_path: Path) -> None:
    """Register a tool class -> appears in list_tools / get."""
    ToolRegistry.reset()
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))

    assert "read_file" in registry.list_tools()
    assert registry.get("read_file") is not None

    schema = registry.get_tools_schema()
    read_entry = next(s for s in schema if s["name"] == "read_file")
    assert read_entry["parameters"]["required"] == ["path"]
    assert "read_file" in registry.get_tools_prompt()


def test_tool_registry_singleton() -> None:
    """Two lookups return the same singleton instance."""
    assert ToolRegistry() is ToolRegistry()


# ---------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------

def test_tool_execution(tmp_path: Path) -> None:
    """read_file returns the file content via ToolResult."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_bytes(b"print('hello')\n")

    tool = ReadFileTool(tmp_path)
    result = asyncio.run(tool.execute(path="src/app.py"))

    assert result.success is True
    assert "print('hello')" in result.output


def test_tool_not_found(tmp_path: Path) -> None:
    """Unknown tool name -> ValueError from _execute_tool."""
    runtime = TetherRuntime("Unknown tool", tmp_path)
    with pytest.raises(ValueError, match="Unknown tool"):
        asyncio.run(runtime._execute_tool("nonexistent_tool(x='1')"))


# ---------------------------------------------------------------------
# Interceptor
# ---------------------------------------------------------------------

def _make_result(text: str) -> ToolResult:
    return ToolResult(success=True, output=text)


def test_interceptor_cache_hit() -> None:
    """Same tool + same params inside the window -> cached result."""
    interceptor = CallInterceptor(window_seconds=5)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("content"))

    cached = interceptor.check("read_file", {"path": "a.py"})
    assert cached is not None
    assert cached.output == "content"


def test_interceptor_cache_miss_different_params() -> None:
    """Same tool, different params -> no interception."""
    interceptor = CallInterceptor(window_seconds=5)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("content"))

    assert interceptor.check("read_file", {"path": "b.py"}) is None


def test_interceptor_cache_expiry() -> None:
    """After the window elapses, the cache no longer intercepts."""
    interceptor = CallInterceptor(window_seconds=0.3)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("content"))

    time.sleep(0.4)
    assert interceptor.check("read_file", {"path": "a.py"}) is None


def test_interceptor_clear() -> None:
    """clear() wipes records so the next identical call executes."""
    interceptor = CallInterceptor(window_seconds=5)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("content"))
    interceptor.clear("read_file")

    assert interceptor.check("read_file", {"path": "a.py"}) is None


def test_interceptor_stats() -> None:
    """intercept_count accumulates one per intercepted check."""
    interceptor = CallInterceptor(window_seconds=5)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("content"))

    for _ in range(3):
        assert interceptor.check("read_file", {"path": "a.py"}) is not None

    assert interceptor._intercept_count == 3


# ---------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------

def test_runtime_integration(tmp_path: Path) -> None:
    """Runtime steps 1 & 3 call the same tool+params -> step 3 is cached."""
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "src/app.py").write_bytes(b"def alpha():\n    return 1\n")

    runtime = TetherRuntime("Intercept test", tmp_path, tool_timeout=5)

    # Deterministic thinking: the same action every step.
    async def fake_think() -> str:
        return "read_file(path='src/app.py')"

    runtime._think = fake_think  # type: ignore[assignment]
    asyncio.run(runtime.run())

    assert runtime.state.status.value == "completed"
    # Steps 2..5 repeat the step-1 call and must be intercepted.
    assert runtime.tool_interceptor._intercept_count >= 1
    assert any(
        "[CACHED]" in output for _, _, output in runtime._tool_history
    )
    # The first call actually read the file.
    assert runtime._tool_history[0][2].startswith("def alpha")


# ---------------------------------------------------------------------
# Decorator (class + function modes)
# ---------------------------------------------------------------------

def test_register_tool_decorator() -> None:
    """@register_tool supports both Tool classes and async functions."""

    @register_tool(name="dummy_class_tool", description="A dummy class tool")
    class DummyTool(Tool):
        name = "dummy_class_tool"
        description = "A dummy class tool"

        def get_parameters_schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def execute(self) -> ToolResult:
            return ToolResult(success=True, output="class ok")

    @register_tool(name="dummy_func_tool", description="A dummy function tool")
    async def dummy_func(x: str) -> ToolResult:
        return ToolResult(success=True, output=f"func ok: {x}")

    registry = ToolRegistry()
    assert registry.get("dummy_class_tool") is not None
    assert registry.get("dummy_func_tool") is not None

    class_out = asyncio.run(registry.get("dummy_class_tool").execute())
    assert class_out.output == "class ok"

    func_out = asyncio.run(registry.get("dummy_func_tool").execute(x="1"))
    assert func_out.output == "func ok: 1"
