"""A minimal MCP server over stdio used by the MCP client tests.

Implements just enough of the protocol: initialize handshake,
tools/list (echo + fail tools), tools/call. Line-delimited JSON-RPC 2.0.
"""

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "fail",
        "description": "Always fails (for error-path tests)",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def main() -> None:
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
                "serverInfo": {"name": "echo-server", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = message["params"].get("name")
            args = message["params"].get("arguments") or {}
            if name == "echo":
                result = {
                    "content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}]
                }
            else:
                result = {
                    "content": [{"type": "text", "text": "intentional failure"}],
                    "isError": True,
                }
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
