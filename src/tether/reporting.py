"""Run-report generation: aggregate an events.jsonl into markdown.

The event stream records every lifecycle/LLM/tool event; this module
turns one task's events into a single-page report: final status, token
spend, compression levels used, tool call statistics (cache hits,
failures) and recovery actions.
"""

from collections import Counter
from typing import Any


def summarize_events(events: list[dict[str, Any]]) -> str:
    """Render one task's events as a markdown report."""
    if not events:
        return "# Tether run report\n\n(no events recorded)\n"

    started = next((e for e in events if e["event"] == "task_started"), None)
    finished = next(
        (e for e in reversed(events) if e["event"] in ("task_completed", "task_ended")),
        None,
    )
    lines: list[str] = ["# Tether run report", ""]

    lines.append("## Task")
    lines.append("")
    lines.append(f"- task_id: `{events[0].get('task_id', '?')}`")
    lines.append(f"- goal: {started.get('goal', '?') if started else '?'}")
    lines.append(f"- brain: {started.get('brain', '?') if started else '?'}")
    if finished is not None:
        status = "completed" if finished["event"] == "task_completed" else finished.get("status", "?")
        lines.append(f"- status: **{status}**")
        lines.append(f"- steps: {finished.get('steps', '?')}")
        if finished["event"] == "task_completed":
            lines.append(f"- total tokens: {finished.get('total_tokens', '?')}")
        elif finished.get("error"):
            lines.append(f"- error: {finished['error']}")
    lines.append("")

    turns = [e for e in events if e["event"] == "llm_turn"]
    if turns:
        lines.append("## LLM turns")
        lines.append("")
        prompt = sum(t.get("prompt_tokens", 0) for t in turns)
        completion = sum(t.get("completion_tokens", 0) for t in turns)
        latency = sum(t.get("latency_ms", 0) for t in turns)
        levels = Counter(t.get("compression_level", "?") for t in turns)
        level_str = ", ".join(f"{k} x{v}" for k, v in levels.most_common())
        lines.append(f"- turns: {len(turns)}")
        lines.append(f"- prompt tokens: {prompt} | completion tokens: {completion}")
        lines.append(f"- total LLM latency: {latency:.0f}ms")
        lines.append(f"- compression levels: {level_str}")
        lines.append("")

    tools = [e for e in events if e["event"] == "tool_executed"]
    if tools:
        per_tool = Counter(t.get("tool", "?") for t in tools)
        cached = sum(1 for t in tools if t.get("cached"))
        failed = sum(1 for t in tools if not t.get("success"))
        lines.append("## Tool calls")
        lines.append("")
        lines.append(f"- total: {len(tools)} (cached: {cached}, failed: {failed})")
        lines.append(
            "- by tool: " + ", ".join(f"{k} x{v}" for k, v in per_tool.most_common())
        )
        lines.append("")

    errors = [e for e in events if e["event"] == "tool_error"]
    if errors:
        lines.append("## Tool errors (kept as observations)")
        lines.append("")
        for e in errors:
            lines.append(f"- step {e.get('step')}: {str(e.get('error', ''))[:160]}")
        lines.append("")

    recoveries = [e for e in events if e["event"] == "recovery"]
    if recoveries:
        lines.append("## Recoveries")
        lines.append("")
        for e in recoveries:
            lines.append(
                f"- scenario: {e.get('scenario')}, replay: {e.get('replay')}, "
                f"skip: {e.get('skip')}"
            )
        lines.append("")

    mcp = [e for e in events if e["event"] == "mcp_connected"]
    if mcp:
        lines.append("## MCP servers")
        lines.append("")
        for e in mcp:
            lines.append(f"- {e.get('server')}: {', '.join(e.get('tools') or [])}")
        lines.append("")

    return "\n".join(lines) + "\n"


def load_events(path) -> list[dict[str, Any]]:
    """Read a JSONL events file; skips blank/corrupted lines."""
    import json

    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
