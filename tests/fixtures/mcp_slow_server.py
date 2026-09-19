"""A minimal MCP server whose FIRST ``tools/list`` response is delayed.

Used by the transport timeout-recovery test: the first ``tools/list``
exceeds the client's (patched) read timeout, so the client must raise —
and then still work on the *same* transport, which is only possible if
the client decouples reading from requesting.

Line-delimited JSON-RPC 2.0, stdio, same shape as ``mcp_echo_server``.
"""

import json
import sys
import time

_FIRST_LIST_DELAY_SECONDS = 1.0

_TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def main() -> None:
    list_calls = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in message:
            continue  # notification: no response
        method = message.get("method", "")
        request_id = message["id"]
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "slow-server", "version": "1.0.0"},
            }
        elif method == "tools/list":
            list_calls += 1
            if list_calls == 1:
                time.sleep(_FIRST_LIST_DELAY_SECONDS)
            result = {"tools": _TOOLS}
        elif method == "tools/call":
            args = message["params"].get("arguments") or {}
            result = {
                "content": [
                    {"type": "text", "text": f"echo: {args.get('text', '')}"}
                ]
            }
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
