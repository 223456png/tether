"""Tests for the two runtime safety rails: the approval gate (human-in-
the-loop) and the token budget circuit breaker."""

from pathlib import Path

from tests.test_agent_loop import ScriptedProvider, _final_response, _tool_response
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskStatus


async def test_denied_write_is_an_observation(tmp_path: Path) -> None:
    """A rejected write becomes a DENIED observation; the task continues."""
    calls: list[tuple[str, dict]] = []

    async def gate(tool: str, params: dict) -> bool:
        calls.append((tool, params))
        return False

    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "evil.txt", "content": "nope"}),
        _final_response("user said no, stopping"),
    ])
    runtime = TetherRuntime(
        "Try to write", tmp_path, llm_provider=provider, max_steps=10,
        approval_gate=gate,
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert calls == [("write_file", {"path": "evil.txt", "content": "nope"})]
    # The file was never written.
    assert not (tmp_path / "evil.txt").exists()
    # The denial is an observation in the tool history.
    assert any("DENIED" in out for _, _, out in runtime._tool_history)
    assert runtime.event_recorder.query("tool_approval")[0]["approved"] is False


async def test_approved_write_executes(tmp_path: Path) -> None:
    """An approved mutating tool executes normally."""

    async def gate(tool: str, params: dict) -> bool:
        return True

    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "ok.txt", "content": "yes"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Write ok", tmp_path, llm_provider=provider, max_steps=10,
        approval_gate=gate,
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "yes"


async def test_read_only_tools_bypass_gate(tmp_path: Path) -> None:
    """The default gate only covers mutating tools, not reads."""

    async def gate(tool: str, params: dict) -> bool:  # pragma: no cover
        raise AssertionError("read_file must not be gated")

    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")
    provider = ScriptedProvider([
        _tool_response("read_file", {"path": "seed.txt"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Read only", tmp_path, llm_provider=provider, max_steps=10,
        approval_gate=gate,
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.event_recorder.query("tool_approval") == []


async def test_custom_approval_tool_set(tmp_path: Path) -> None:
    """approval_tools narrows/widens the gated set (search_code gated here)."""

    async def gate(tool: str, params: dict) -> bool:
        return False

    provider = ScriptedProvider([
        _tool_response("search_code", {"pattern": "x"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Gate search", tmp_path, llm_provider=provider, max_steps=10,
        approval_gate=gate, approval_tools={"search_code"},
    )
    await runtime.run()

    assert any("DENIED" in out for _, _, out in runtime._tool_history)


async def test_token_budget_stops_task(tmp_path: Path) -> None:
    """Reaching max_total_tokens transitions to STOPPED with a reason."""
    provider = ScriptedProvider([
        _tool_response("search_code", {"pattern": "a"}),
        _tool_response("search_code", {"pattern": "b"}),
        _tool_response("search_code", {"pattern": "c"}),  # never reached
    ])
    runtime = TetherRuntime(
        "Budgeted", tmp_path, llm_provider=provider, max_steps=10,
        max_total_tokens=150,  # each scripted turn costs 100 prompt tokens
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.STOPPED
    assert runtime.state.total_tokens >= 150
    assert runtime.state.error_message is not None
    assert "budget" in runtime.state.error_message.lower()
    assert len(provider.calls) == 2  # no turn burned past the cap


async def test_no_budget_means_no_stop(tmp_path: Path) -> None:
    """Without a cap the loop is unaffected."""
    provider = ScriptedProvider([
        _tool_response("search_code", {"pattern": "a"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Unbudgeted", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
