"""Tests for the agent-level integration eval (policy + task verifiers)."""

import asyncio
from pathlib import Path

from tether.benchmarks.agent_tasks import (
    PolicyProvider,
    ScriptedPolicy,
    Step,
    build_agent_tasks,
)
from tether.benchmarks.config import default_configs
from tether.benchmarks.runner import BenchmarkRunner


def test_policy_provider_feeds_observation_channel() -> None:
    """The policy sees the last tool output exactly as the LLM would."""
    provider = PolicyProvider(ScriptedPolicy([
        Step("write_file", {"path": "multi.txt", "content": "line1\nline2-BUG\n"}),
        Step("read_file", {"path": "multi.txt"}),
    ]))

    # Turn 1: no observation yet.
    resp1 = asyncio.run(provider.complete(
        [{"role": "system", "content": "[RECENT TOOLS]\n(no tool results)"}]
    ))
    assert resp1.tool_calls[0].name == "write_file"

    # Turn 2: the multi-line observation survives intact.
    context = (
        "[RECENT TOOLS]\n"
        "- write_file(path='multi.txt') -> Wrote 13 chars to multi.txt\n"
        "- read_file(path='multi.txt') -> line1\nline2-BUG\n"
    )
    resp2 = asyncio.run(provider.complete(
        [{"role": "system", "content": context}]
    ))
    assert resp2.tool_calls[0].name == "read_file"
    assert PolicyProvider._last_observation(
        [{"role": "system", "content": context}]
    ) == "line1\nline2-BUG"


def test_policy_finishes_after_steps() -> None:
    """Exhausted steps -> final answer without tool calls."""
    policy = ScriptedPolicy([Step("search_code", {"pattern": "x"})], answer="all done")
    policy.next_action("")  # consumes the single step
    resp = policy.next_action("")

    assert resp.tool_calls == []
    assert resp.content == "all done"


async def test_agent_benchmark_end_to_end(tmp_path: Path) -> None:
    """A few tasks run through the real runner and verify green."""
    config = default_configs(3)["agent"]
    config.output_dir = tmp_path
    runner = BenchmarkRunner(config)
    results = await runner._run_agent()

    assert len(results) == 3
    assert all(r.success for r in results), [
        f"{r.task_id}: status={r.extra['status']}" for r in results if not r.success
    ]
    assert all(r.extra["agent"] for r in results)


def test_task_templates_cover_five_types() -> None:
    """The five task types exist and their step builders are valid."""
    tasks = build_agent_tasks()
    names = {t.name for t in tasks}
    assert names == {
        "create_verify", "edit_existing", "search_fix", "plan_execute", "test_report",
    }
    for task in tasks:
        assert task.steps(0), f"{task.name} built no steps"
        assert task.goal and task.verify is not None
