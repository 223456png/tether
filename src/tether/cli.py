"""Command-line entry point: run a Tether agent on a workspace.

Examples::

    tether run --goal "Create hello.txt containing 'hi' and verify it"
    tether run --goal "Fix the failing test" --workspace ./repo --max-steps 40
    tether run --goal "demo" --mock          # force the offline mock brain
    tether run --goal "..." --mcp-cmd "python path/to/mcp_server.py"
    tether report --events ./logs/events.jsonl
"""

import argparse
import asyncio
import shlex
import sys
from pathlib import Path

from tether.llm.factory import create_provider_from_env
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskStatus


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (kept separate for testability)."""
    parser = argparse.ArgumentParser(
        prog="tether",
        description="Tether: a runtime for long-running coding agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an agent task to completion")
    run.add_argument("--goal", required=True, help="Task goal for the agent")
    run.add_argument(
        "--workspace", default=".", help="Workspace directory (default: .)"
    )
    run.add_argument(
        "--max-steps", type=int, default=20,
        help="Hard cap on executed tool steps (default: 20)",
    )
    run.add_argument(
        "--tool-timeout", type=int, default=30,
        help="Per-tool-call timeout in seconds (default: 30)",
    )
    run.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="Tool errors kept as observations before the circuit breaker "
             "trips (0 = fail on first error; default: 3)",
    )
    run.add_argument(
        "--mock", action="store_true",
        help="Force the offline mock brain even if an API key is set",
    )
    run.add_argument(
        "--mcp-cmd", action="append", default=None, metavar="COMMAND",
        help="MCP server command to connect before running (repeatable, "
             "shell-quoted, e.g. --mcp-cmd \"python server.py\")",
    )
    run.add_argument(
        "--require-approval", action="store_true",
        help="Pause for console confirmation before write_file/run_test",
    )
    run.add_argument(
        "--max-tokens", type=int, default=None,
        help="Hard cap on cumulative LLM tokens; the task stops (STOPPED) "
             "when reached (default: unlimited)",
    )

    report = sub.add_parser(
        "report", help="Summarize a task's events.jsonl into a markdown report"
    )
    report.add_argument(
        "--events", default="./logs/events.jsonl",
        help="Path to the events.jsonl file (default: ./logs/events.jsonl)",
    )
    report.add_argument(
        "--out", default=None,
        help="Write the report to this file instead of stdout",
    )
    return parser


async def _console_gate(tool: str, params: dict) -> bool:
    """Human-in-the-loop gate: ask on the console before executing."""
    answer = await asyncio.to_thread(
        input, f"[tether approve] run {tool}({params})? [y/N] "
    )
    return answer.strip().lower() in ("y", "yes")


async def _run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand; returns the process exit code."""
    provider, is_mock = create_provider_from_env()
    if args.mock or is_mock:
        # No key / --mock: drop the provider entirely so the loop runs on
        # the offline mock *thinker* (which actually executes tools) —
        # far more useful as a demo than an LLM that never calls any.
        provider = None
        is_mock = True
        print("[tether] no API key configured - running with the offline "
              "mock brain (set DEEPSEEK_API_KEY or TETHER_LLM_API_KEY for a "
              "real model)")

    runtime = TetherRuntime(
        goal=args.goal,
        workspace_dir=Path(args.workspace),
        tool_timeout=args.tool_timeout,
        llm_provider=provider,
        max_steps=args.max_steps,
        max_consecutive_failures=args.max_consecutive_failures,
        approval_gate=_console_gate if args.require_approval else None,
        max_total_tokens=args.max_tokens,
    )

    for mcp_cmd in args.mcp_cmd or []:
        command = shlex.split(mcp_cmd)
        try:
            registered = await runtime.connect_mcp(command)
            print(f"[tether] MCP connected: {', '.join(registered)}")
        except Exception as exc:  # noqa: BLE001 - keep running without the server
            print(f"[tether] WARNING: MCP server {command} failed: {exc}")

    await runtime.run()
    for client in runtime._mcp_clients:
        client.close()

    state = runtime.state
    print("\n=== Tether run ===")
    print(f"status:    {state.status.value}")
    print(f"steps:     {state.step_index}")
    print(f"tokens:    {state.total_tokens}")
    if state.final_answer:
        print(f"answer:    {state.final_answer}")
    if state.error_message:
        print(f"error:     {state.error_message}")
    print(f"events:    {Path(args.workspace) / 'logs' / 'events.jsonl'}")
    return 0 if state.status == TaskStatus.COMPLETED else 1


def _report(args: argparse.Namespace) -> int:
    """Execute the ``report`` subcommand."""
    from tether.reporting import load_events, summarize_events

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"[tether] events file not found: {events_path}", file=sys.stderr)
        return 1
    report = summarize_events(load_events(events_path))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[tether] report written: {out_path}")
    else:
        sys.stdout.write(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "report":
        return _report(args)
    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    sys.exit(main())
