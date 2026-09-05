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
    get_default_registry,
    register_tool,
    reset_default_registry,
)
from tether.tools.builtin import ReadFileTool

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

def test_tool_registration(tmp_path: Path) -> None:
    """Register a tool class -> appears in list_tools / get."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))

    assert "read_file" in registry.list_tools()
    assert registry.get("read_file") is not None

    schema = registry.get_tools_schema()
    read_entry = next(s for s in schema if s["name"] == "read_file")
    assert read_entry["parameters"]["required"] == ["path"]
    assert "read_file" in registry.get_tools_prompt()


def test_registries_are_isolated(tmp_path: Path) -> None:
    """Per-runtime registries: registrations never leak across instances."""
    a = ToolRegistry()
    b = ToolRegistry()
    a.register(ReadFileTool(tmp_path))

    assert a.get("read_file") is not None
    assert b.get("read_file") is None
    assert a is not b


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

def test_register_tool_decorator(tmp_path: Path) -> None:
    """@register_tool supports both Tool classes and async functions."""
    reset_default_registry()

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

    registry = get_default_registry()
    assert registry.get("dummy_class_tool") is not None
    assert registry.get("dummy_func_tool") is not None

    class_out = asyncio.run(registry.get("dummy_class_tool").execute())
    assert class_out.output == "class ok"

    func_out = asyncio.run(registry.get("dummy_func_tool").execute(x="1"))
    assert func_out.output == "func ok: 1"

    # Runtimes merge decorator tools into their own registry.
    runtime = TetherRuntime("Merge check", tmp_path)
    assert runtime.tool_registry.get("dummy_func_tool") is not None
    # Builtins keep priority over same-name decorator registrations.
    assert runtime.tool_registry.get("read_file") is not runtime.tool_registry.get(
        "dummy_class_tool"
    )

    reset_default_registry()


# ---------------------------------------------------------------------
# Workspace boundary (path traversal) + atomic writes
# ---------------------------------------------------------------------

def test_read_file_rejects_traversal(tmp_path: Path) -> None:
    """read_file refuses paths resolving outside the workspace."""
    outside = tmp_path.parent / "tether_outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        tool = ReadFileTool(tmp_path)
        result = asyncio.run(tool.execute(path="../tether_outside_secret.txt"))
        assert result.success is False
        assert "escapes workspace" in result.error
    finally:
        outside.unlink(missing_ok=True)


def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    """Absolute paths pointing outside the workspace are refused."""
    import sys

    tool = ReadFileTool(tmp_path)
    alien = Path(sys.executable)  # definitely not in the workspace
    result = asyncio.run(tool.execute(path=str(alien)))
    assert result.success is False
    assert "escapes workspace" in result.error


def test_write_file_rejects_traversal(tmp_path: Path) -> None:
    """write_file refuses to create files outside the workspace."""
    from tether.tools.builtin import WriteFileTool

    tool = WriteFileTool(tmp_path)
    result = asyncio.run(
        tool.execute(path="../tether_escaped.txt", content="nope")
    )
    assert result.success is False
    assert "escapes workspace" in result.error
    assert not (tmp_path.parent / "tether_escaped.txt").exists()


def test_write_file_is_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    """A successful write leaves no .tmp file next to the target."""
    from tether.tools.builtin import WriteFileTool

    tool = WriteFileTool(tmp_path)
    result = asyncio.run(
        tool.execute(path="src/app.py", content="x = 1\n")
    )
    assert result.success is True
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert list((tmp_path / "src").glob("*.tmp")) == []


# ---------------------------------------------------------------------
# Interceptor safety semantics (side effects, failures, invalidation)
# ---------------------------------------------------------------------

def test_interceptor_never_caches_failed_results() -> None:
    """A failed call is not recorded, so the retry really executes."""
    interceptor = CallInterceptor(window_seconds=5)
    failed = ToolResult(success=False, error="boom")
    interceptor.record("read_file", {"path": "a.py"}, failed)

    assert interceptor.check("read_file", {"path": "a.py"}) is None


def test_interceptor_skips_non_cacheable_tools() -> None:
    """cacheable=False disables both check() and record()."""
    interceptor = CallInterceptor(window_seconds=5)
    ok = ToolResult(success=True, output="wrote")
    interceptor.record("write_file", {"path": "a.py"}, ok, cacheable=False)

    assert interceptor.check("write_file", {"path": "a.py"}, cacheable=False) is None
    # Even if something slipped in, a non-cacheable check must not hit it.
    assert interceptor.check("write_file", {"path": "a.py"}) is None


def test_interceptor_invalidate_all() -> None:
    """invalidate_all() drops cached reads (post-mutation safety)."""
    interceptor = CallInterceptor(window_seconds=5)
    interceptor.record("read_file", {"path": "a.py"}, _make_result("old"))
    interceptor.invalidate_all()

    assert interceptor.check("read_file", {"path": "a.py"}) is None


async def test_read_after_write_is_fresh(tmp_path: Path) -> None:
    """read -> write(same file) -> read returns new content, not [CACHED]."""
    from tests.test_agent_loop import ScriptedProvider, _final_response
    from tether.llm.base import LLMResponse, ToolCall

    def _call(name: str, arguments: dict) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(name=name, arguments=arguments, id="c1")],
        )

    provider = ScriptedProvider([
        _call("read_file", {"path": "note.txt"}),       # old content
        _call("write_file", {"path": "note.txt", "content": "NEW"}),
        _call("read_file", {"path": "note.txt"}),       # must see NEW
        _final_response("done"),
    ])
    (tmp_path / "note.txt").write_text("OLD", encoding="utf-8")
    runtime = TetherRuntime(
        "Rewrite note", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    outputs = [out for _, _, out in runtime._tool_history]
    assert outputs[0] == "OLD"
    assert "NEW" in outputs[2] and not outputs[2].startswith("[CACHED]")


# ---------------------------------------------------------------------
# Real pytest runner (run_test)
# ---------------------------------------------------------------------

def test_run_test_real_pytest_pass(tmp_path: Path) -> None:
    """A passing test file returns success with the pytest summary."""
    from tether.tools.builtin import RunTestTool

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    tool = RunTestTool(tmp_path)
    result = asyncio.run(tool.execute(path="tests/test_ok.py"))

    assert result.success is True
    assert "passed" in result.output
    assert result.metadata["exit_code"] == 0


def test_run_test_real_pytest_fail(tmp_path: Path) -> None:
    """A failing test file returns failure with the traceback tail."""
    from tether.tools.builtin import RunTestTool

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_bad.py").write_text(
        "def test_bad():\n    assert 1 + 1 == 3\n", encoding="utf-8"
    )
    tool = RunTestTool(tmp_path)
    result = asyncio.run(tool.execute(path="tests/test_bad.py"))

    assert result.success is False
    assert "failed" in result.error.lower()
    assert "assert" in result.error
    assert result.metadata["exit_code"] != 0
