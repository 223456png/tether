"""Tests for the MCP client (fake transport + real stdio subprocess)."""

import sys
import time
from pathlib import Path

import pytest

from tether.mcp import (
    MCPClient,
    MCPError,
    MCPTool,
    StdioTransport,
    register_mcp_tools,
    sanitize_server_name,
)
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskStatus


class FakeTransport:
    """Scripted transport double: returns canned results per method."""

    def __init__(self, results: dict, server_info: dict | None = None) -> None:
        self.results = results
        self.server_info_result = server_info or {
            "serverInfo": {"name": "fake", "version": "1.0"}
        }
        self.calls: list[tuple[str, dict]] = []

    def start(self) -> dict:
        return self.server_info_result

    def request(self, method: str, params: dict) -> object:
        self.calls.append((method, params))
        result = self.results.get(method)
        if isinstance(result, MCPError):
            raise result
        return result

    def notify(self, method: str, params: dict) -> None:
        self.calls.append((method, params))

    def close(self) -> None:
        self.calls.append(("close", {}))


def _fake_client(results: dict) -> MCPClient:
    client = MCPClient(FakeTransport(results))  # type: ignore[arg-type]
    client.initialize()
    return client


def test_mcp_client_lists_and_calls_tools() -> None:
    """tools/list parses descriptors; tools/call returns raw content."""
    client = _fake_client({
        "tools/list": {"tools": [{
            "name": "grep",
            "description": "Remote grep",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]},
        "tools/call": {"content": [{"type": "text", "text": "match found"}]},
    })
    tools = client.list_tools()

    assert [t["name"] for t in tools] == ["grep"]
    result = client.call_tool("grep", {"q": "x"})
    assert result["content"][0]["text"] == "match found"


def test_mcp_tool_wraps_remote_call() -> None:
    """MCPTool flattens text content; isError maps to a failed ToolResult."""
    from tether.tools.base import ToolResult

    ok_client = _fake_client({
        "tools/call": {"content": [{"type": "text", "text": "ok"}]}
    })
    ok = MCPTool(ok_client, "do", "Does things", {"type": "object"}, "mcp_fake_do")
    import asyncio

    result = asyncio.run(ok.execute(x="1"))
    assert isinstance(result, ToolResult)
    assert result.success and result.output == "ok"
    assert result.metadata == {"mcp_tool": "do"}

    err_client = _fake_client({
        "tools/call": {
            "content": [{"type": "text", "text": "boom"}],
            "isError": True,
        }
    })
    err = MCPTool(err_client, "boom", "Fails", {"type": "object"}, "mcp_fake_boom")
    failed = asyncio.run(err.execute())
    assert failed.success is False
    assert "boom" in failed.error


def test_sanitize_server_name() -> None:
    """Server names reduce to registry-safe identifiers."""
    assert sanitize_server_name("my-server.2") == "my_server_2"
    assert sanitize_server_name("$$$") == "mcp"


def test_real_stdio_server_roundtrip(tmp_path: Path) -> None:
    """End-to-end against a real MCP subprocess: handshake, list, call."""
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    client = MCPClient(StdioTransport([sys.executable, str(server)], cwd=tmp_path))
    try:
        info = client.initialize()
        assert info["serverInfo"]["name"] == "echo-server"
        tools = client.list_tools()
        assert {t["name"] for t in tools} == {"echo", "fail"}
        result = client.call_tool("echo", {"text": "hello"})
        assert result["content"][0]["text"] == "echo: hello"
    finally:
        client.close()


async def test_runtime_connect_mcp_registers_tools(tmp_path: Path) -> None:
    """connect_mcp registers prefixed tools the LLM can call immediately."""
    from tests.test_agent_loop import ScriptedProvider, _final_response, _tool_response

    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    runtime = TetherRuntime(
        "Use MCP", tmp_path, llm_provider=ScriptedProvider([
            _tool_response("mcp_echo_server_echo", {"text": "ping"}),
            _final_response("done"),
        ]),
        max_steps=10,
    )
    registered = await runtime.connect_mcp([sys.executable, str(server)])
    assert registered == ["mcp_echo_server_echo", "mcp_echo_server_fail"]

    await runtime.run()

    outputs = [out for _, _, out in runtime._tool_history]
    assert outputs[0] == "echo: ping"
    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.event_recorder.query("mcp_connected")

    await runtime.aclose()


def test_stdio_transport_rejects_bad_command() -> None:
    """Spawning a nonexistent binary raises a clean MCPError."""
    transport = StdioTransport(["definitely-not-a-real-binary-xyz"])
    with pytest.raises(MCPError, match="Failed to spawn"):
        transport.start()


def test_stdio_transport_recovers_after_read_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read timeout fails one request but leaves the transport usable.

    Regression test: the previous single-worker ThreadPoolExecutor stayed
    wedged after a timeout (the blocked ``readline`` never released its
    worker), so every later request timed out too.
    """
    import tether.mcp as mcp_module

    server = Path(__file__).parent / "fixtures" / "mcp_slow_server.py"
    client = MCPClient(StdioTransport([sys.executable, str(server)], cwd=tmp_path))
    try:
        client.initialize()
        monkeypatch.setattr(mcp_module, "_REQUEST_TIMEOUT", 0.5)
        with pytest.raises(MCPError):
            client.list_tools()  # first call is delayed past the timeout
        time.sleep(1.2)  # let the delayed response land in the reader queue
        tools = client.list_tools()  # same transport, recovered
        assert {t["name"] for t in tools} == {"echo"}
    finally:
        client.close()


# 2026-09-30 审计修复回归：重名被 registry 保留时不得谎报注册成功。
def test_register_mcp_tools_returns_only_registered() -> None:
    from tether.tools.registry import ToolRegistry

    client = _fake_client({
        "tools/list": {"tools": [
            {"name": "grep", "description": "Remote grep", "inputSchema": {"type": "object"}},
            {"name": "find", "description": "Remote find", "inputSchema": {"type": "object"}},
        ]},
    })
    registry = ToolRegistry()
    first = register_mcp_tools(registry, client)
    assert set(first) == {"mcp_fake_grep", "mcp_fake_find"}

    # Second pass: every name collides with itself (overwrite=False keeps
    # the existing tool) — the return value must not claim success.
    second = register_mcp_tools(registry, client)
    assert second == []
