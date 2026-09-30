"""Minimal MCP (Model Context Protocol) client over stdio JSON-RPC 2.0.

Connects Tether to any MCP server (the standard agent tool ecosystem):
tools exposed by the server are wrapped as regular ``Tool`` objects and
registered into the runtime's registry under ``mcp_{server}_{tool}``.

Zero dependencies: JSON-RPC lines over a spawned subprocess, threaded
I/O wrapped in ``asyncio.to_thread`` so the async runtime never blocks
and never depends on the event loop's subprocess support.
"""

import asyncio
import json
import queue
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from tether import __version__
from tether.tools.base import Tool, ToolResult

_PROTOCOL_VERSION = "2024-11-05"
_REQUEST_TIMEOUT = 30.0


class MCPError(RuntimeError):
    """An MCP-level failure (transport, protocol or tool error)."""


class StdioTransport:
    """Line-delimited JSON-RPC 2.0 over a spawned subprocess's stdio.

    A dedicated daemon reader thread drains the server's stdout into a
    queue; each request pulls from that queue with a timeout. Decoupling
    the read from the request means a slow or stuck server only fails the
    *current* request — the transport stays usable for the next one.
    (A shared worker pool would be permanently wedged by a single timeout,
    because the blocked ``readline`` never releases its worker.)

    Requests are serialized under a lock (MCP servers handle one
    in-flight request fine for our use).
    """

    def __init__(self, command: list[str], cwd: Path | None = None) -> None:
        """Store the server command; no process starts until ``start``."""
        self._command = command
        self._cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._inbox: queue.Queue[object] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_id = 0

    def start(self) -> dict:
        """Spawn the server, start the reader thread, run the handshake."""
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # servers log to stderr
                text=True,
                encoding="utf-8",
                cwd=str(self._cwd) if self._cwd else None,
            )
        except OSError as exc:
            raise MCPError(f"Failed to spawn MCP server {self._command}: {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_loop, name="mcp-reader", daemon=True
        )
        self._reader.start()
        result = self.request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tether", "version": __version__},
            },
        )
        self.notify("notifications/initialized", {})
        return result or {}

    def request(self, method: str, params: dict[str, Any]) -> Any:
        """Send one JSON-RPC request and return its result."""
        if self._proc is None:
            raise MCPError("Transport not started")
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                        "params": params})
            while True:
                message = self._read_message()
                if message.get("id") != request_id:
                    continue  # notification or late/stale response: skip
                if "error" in message:
                    raise MCPError(
                        f"MCP {method} failed: {message['error'].get('message')}"
                    )
                return message.get("result")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no id, no response expected)."""
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise MCPError(f"MCP server write failed: {exc}") from exc

    def _read_loop(self) -> None:
        """Reader thread: forward each parsed message (or error) to the queue."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._inbox.put(MCPError("Transport closed before reading"))
            return
        stdout = proc.stdout
        while True:
            try:
                line = stdout.readline()
            except (OSError, ValueError) as exc:
                self._inbox.put(MCPError(f"MCP server read failed: {exc}"))
                return
            if not line:
                self._inbox.put(MCPError("MCP server closed the connection"))
                return
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                self._inbox.put(
                    MCPError(f"MCP server sent invalid JSON: {line[:120]!r}")
                )

    def _read_message(self) -> dict:
        """Pop the next message; raise on timeout or a transport error."""
        try:
            item = self._inbox.get(timeout=_REQUEST_TIMEOUT)
        except queue.Empty:
            raise MCPError("MCP server read timed out") from None
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, dict)
        return item

    def close(self) -> None:
        """Terminate the server process and stop the reader thread."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        # Unblock any pending/future read so shutdown is immediate.
        self._inbox.put(MCPError("Transport closed"))


class MCPClient:
    """High-level MCP client: initialize, list tools, call tools."""

    def __init__(self, transport: StdioTransport) -> None:
        """Wrap a started-capable transport."""
        self._transport = transport
        self.server_info: dict = {}

    def initialize(self) -> dict:
        """Run the MCP handshake; returns the server's ``serverInfo``."""
        self.server_info = self._transport.start()
        return self.server_info

    def list_tools(self) -> list[dict]:
        """Return the server's tool descriptors (name/description/inputSchema)."""
        result = self._transport.request("tools/list", {}) or {}
        tools = result.get("tools") or []
        return [t for t in tools if isinstance(t, dict) and t.get("name")]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke one remote tool; returns the raw ``content`` result."""
        return self._transport.request(
            "tools/call", {"name": name, "arguments": arguments}
        ) or {}

    def close(self) -> None:
        """Shut the transport down."""
        self._transport.close()


def sanitize_server_name(name: str) -> str:
    """Reduce a server name to registry-safe identifier characters."""
    cleaned = re.sub(r"\W+", "_", name).strip("_")
    return cleaned or "mcp"


class MCPTool(Tool):
    """A remote MCP server tool exposed through the local ToolRegistry."""

    # Semantics are unknown a priori: never cache remote calls.
    side_effect_free = False

    def __init__(
        self,
        client: MCPClient,
        remote_name: str,
        description: str,
        schema: dict,
        qualified_name: str,
    ) -> None:
        """Store the client binding and the registry-qualified name."""
        self._client = client
        self._remote_name = remote_name
        self.description = description
        self._schema = schema or {"type": "object", "properties": {}}
        self.name = qualified_name

    def get_parameters_schema(self) -> dict:
        """The MCP server's JSON ``inputSchema``, passed through."""
        return self._schema

    async def execute(self, **kwargs) -> ToolResult:
        """Call the remote tool and flatten its text content parts."""
        try:
            result = await asyncio.to_thread(
                self._client.call_tool, self._remote_name, kwargs
            )
        except MCPError as exc:
            return ToolResult(success=False, error=str(exc))
        parts = result.get("content") or []
        text = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):
            return ToolResult(
                success=False, error=text or f"MCP tool {self._remote_name} failed"
            )
        return ToolResult(
            success=True, output=text, metadata={"mcp_tool": self._remote_name}
        )


def register_mcp_tools(
    registry, client: MCPClient, server_name: str | None = None
) -> list[str]:
    """Wrap the server's tools and register them; returns qualified names.

    Only tools that were actually registered are listed — a name collision
    with an existing builtin keeps the existing tool, and claiming the MCP
    name anyway would mislead callers (and the model's tool list).
    """
    info = client.server_info.get("serverInfo", {}) if client.server_info else {}
    base = sanitize_server_name(server_name or info.get("name", "mcp"))
    registered: list[str] = []
    skipped: list[str] = []
    for descriptor in client.list_tools():
        qualified = f"mcp_{base}_{descriptor['name']}"
        was_registered = registry.register(
            MCPTool(
                client=client,
                remote_name=descriptor["name"],
                description=descriptor.get("description", ""),
                schema=descriptor.get("inputSchema", {}),
                qualified_name=qualified,
            ),
            overwrite=False,
        )
        if was_registered:
            registered.append(qualified)
        else:
            skipped.append(qualified)
    if skipped:
        logger.warning("MCP tools skipped (name collision) | server={} tools={}", base, skipped)
    logger.info("MCP tools registered | server={} tools={}", base, registered)
    return registered
