"""Command-line entry point: run a Tether agent on a workspace.

Examples::

    tether run --goal "Create hello.txt containing 'hi' and verify it"
    tether run --goal "Fix the failing test" --workspace ./repo --max-steps 40
    tether run --goal "demo" --mock          # force the offline mock brain
"""

import argparse
import asyncio
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
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand; returns the process exit code."""
    provider, is_mock = create_provider_from_env()
    if args.mock or is_mock:
        provider = _mock_provider()
        is_mock = True
    if is_mock:
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
    )
    await runtime.run()

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


def _mock_provider():
    """Return a fresh MockProvider (import here to keep --mock cheap)."""
    from tether.llm.mock import MockProvider

    return MockProvider()


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    sys.exit(main())
