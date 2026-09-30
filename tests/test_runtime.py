"""Core tests for TetherRuntime Phase 1: normal run, timeout, resume."""

import asyncio
from pathlib import Path

import pytest

from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskStatus


async def test_normal_execution(tmp_path: Path) -> None:
    """A normal run completes with 5 steps and at least 2 checkpoint lines."""
    runtime = TetherRuntime("Fix login bug", tmp_path, tool_timeout=5)
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.state.step_index > 0

    ckpt = tmp_path / "checkpoints" / f"{runtime.state.task_id}.jsonl"
    assert ckpt.exists()
    lines = [ln for ln in ckpt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2


async def test_timeout_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow tool triggers timeout and the task fails with a timeout message."""
    runtime = TetherRuntime("Test timeout", tmp_path, tool_timeout=1)

    async def slow_tool(action: str) -> str:
        await asyncio.sleep(5)
        return "slow"

    monkeypatch.setattr(runtime, "_execute_tool", slow_tool)
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert runtime.state.error_message is not None
    assert "timeout" in runtime.state.error_message.lower()


async def test_resume_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A task that times out on the 3rd call can resume from checkpoint and complete."""
    from tether.runtime.runtime import AgentDecision

    runtime = TetherRuntime("Recovery test", tmp_path, tool_timeout=1)
    call_count = {"n": 0}

    async def flaky_tool(action: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 3:
            await asyncio.sleep(5)
            return "slow"
        await asyncio.sleep(0.1)
        return f"✅ Tool '{action}' executed successfully."

    monkeypatch.setattr(runtime, "_execute_tool", flaky_tool)

    # Deterministic thinking: two writes, then a read whose execution
    # (the 3rd call) times out -> FAILED at step 3.
    first_run_actions = iter([
        "write_file(path='a.txt', content='x')",
        "write_file(path='b.txt', content='y')",
        "read_file(path='a.txt')",
    ])

    async def scripted_think() -> AgentDecision:
        action = next(first_run_actions, None)
        if action is None:
            return AgentDecision(finished=True, final_answer="done")
        return runtime._decision_from_action(action)

    monkeypatch.setattr(runtime, "_think", scripted_think)
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert runtime.state.step_index == 3
    task_id = runtime.state.task_id

    recovered = TetherRuntime("Recovery test", tmp_path, tool_timeout=5)

    # After recovery, the brain finishes the task instead of redoing work.
    async def recovered_think() -> AgentDecision:
        return AgentDecision(finished=True, final_answer="recovered and done")

    monkeypatch.setattr(recovered, "_think", recovered_think)
    await recovered.resume(task_id)

    assert recovered.state.task_id == task_id
    assert recovered.state.status == TaskStatus.COMPLETED
    assert recovered.state.step_index > 3
