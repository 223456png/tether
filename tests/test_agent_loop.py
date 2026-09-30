"""Integration tests for the LLM-driven agent loop (Phase 10).

A scripted fake provider stands in for the real model: it returns
pre-programmed LLMResponses (tool calls / final answers) so the full
think -> tool -> checkpoint pipeline runs deterministically offline.
"""

from pathlib import Path

from tether.checkpoint.manager import CheckpointManager
from tether.checkpoint.recovery import RecoveryManager
from tether.filesystem.drift import DriftDetector
from tether.llm.base import LLMResponse, ToolCall
from tether.memory.episodic import EpisodicNotes
from tether.memory.store import MemoryStore
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskState, TaskStatus
from tether.tools.base import Tool, ToolResult


class ScriptedProvider:
    """LLMProvider double that replays a scripted response sequence."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            return LLMResponse(content="no script left", provider=self.name)
        return self.responses.pop(0)


def _tool_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        prompt_tokens=100,
        completion_tokens=10,
        tool_calls=[ToolCall(name=name, arguments=arguments, id="call_1")],
        provider="scripted",
    )


def _final_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text, prompt_tokens=100, completion_tokens=25, provider="scripted"
    )


class FlakyTool(Tool):
    """Test double: raises on the first call, succeeds afterwards."""

    name = "flaky"
    description = "raises once, then succeeds"
    side_effect_free = False

    def __init__(self) -> None:
        self.calls = 0

    def get_parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("boom")
        return ToolResult(success=True, output="recovered")


async def test_llm_loop_executes_tool_calls_and_finishes(tmp_path: Path) -> None:
    """Tool-call responses execute; a plain response ends the task."""
    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "hello.txt", "content": "line=1, x=2"}),
        _tool_response("read_file", {"path": "hello.txt"}),
        _final_response("Wrote and verified hello.txt. Goal achieved."),
    ])
    runtime = TetherRuntime(
        "Create hello.txt", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.state.final_answer == "Wrote and verified hello.txt. Goal achieved."
    # Content with '=' and ',' survives the structured path intact.
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "line=1, x=2"
    assert len(runtime._tool_history) == 2
    assert runtime.state.step_index == 3  # two tool steps + finishing turn


async def test_llm_loop_sends_tools_schema_and_memory_context(tmp_path: Path) -> None:
    """Each LLM turn carries the OpenAI tool schema and assembled context."""
    (tmp_path / "seed.txt").write_text("seed content", encoding="utf-8")
    provider = ScriptedProvider([
        _tool_response("read_file", {"path": "seed.txt"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Read seed.txt", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    assert provider.calls, "the LLM must have been called"
    first = provider.calls[0]
    tool_names = [t["function"]["name"] for t in first["tools"]]
    assert "read_file" in tool_names and "write_file" in tool_names
    system_text = first["messages"][0]["content"]
    assert "Read seed.txt" in system_text  # goal reaches the system prompt
    assert "[SYSTEM]" in system_text and "[TASK SUMMARY]" in system_text


async def test_llm_loop_accumulates_token_usage(tmp_path: Path) -> None:
    """Usage from every turn accumulates on TaskState."""
    provider = ScriptedProvider([
        _tool_response("search_code", {"pattern": "x"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Search", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    assert runtime.state.prompt_tokens == 200
    assert runtime.state.completion_tokens == 35
    assert runtime.state.total_tokens == 235


async def test_llm_loop_max_steps_cap(tmp_path: Path) -> None:
    """A model that never stops calling tools is capped at max_steps.

    Hitting the cap is a resource-exhaustion stop (STOPPED), matching the
    token-budget semantics — never a fake COMPLETED.
    """
    provider = ScriptedProvider([
        # More responses than the cap; leftovers must never be consumed.
        _tool_response("search_code", {"pattern": "a"}),
        _tool_response("search_code", {"pattern": "b"}),
        _tool_response("search_code", {"pattern": "c"}),
    ])
    runtime = TetherRuntime(
        "Loop forever", tmp_path, llm_provider=provider, max_steps=2
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.STOPPED
    assert runtime.state.error_message is not None
    assert "Step cap reached" in runtime.state.error_message
    assert runtime.state.step_index == 2
    assert len(provider.calls) == 2


async def test_mock_brain_completes_goal_shaped_task(tmp_path: Path) -> None:
    """The offline mock brain actually achieves the quickstart goal.

    "Create <file> containing '<text>'" is executed for real (write ->
    read back) and verified against the workspace, so the report can say
    ✓ — no fake completions.
    """
    runtime = TetherRuntime(
        "Create hello.txt containing 'hi' and verify it", tmp_path,
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hi"
    assert "Created hello.txt" in (runtime.state.final_answer or "")

    verified = runtime.event_recorder.query("goal_verified")
    assert len(verified) == 1
    assert verified[0]["passed"] is True


async def test_mock_brain_is_honest_about_unmatched_goals(tmp_path: Path) -> None:
    """Goals the mock cannot pursue get an honest answer, not a fake win."""
    runtime = TetherRuntime("Refactor the auth module", tmp_path)
    await runtime.run()

    # The loop still runs a scripted demo (write + read back) ...
    tools = runtime.event_recorder.query("tool_executed")
    assert [t["tool"] for t in tools] == ["write_file", "read_file"]
    # ... but the final answer plainly says the goal was NOT pursued.
    assert runtime.state.status == TaskStatus.COMPLETED
    assert "did NOT pursue" in (runtime.state.final_answer or "")


async def test_goal_verifier_is_recorded_in_events(tmp_path: Path) -> None:
    """A caller-supplied goal_verifier lands as a goal_verified event."""
    provider = ScriptedProvider([_final_response("done")])
    runtime = TetherRuntime(
        "Anything", tmp_path, llm_provider=provider,
        goal_verifier=lambda: False,
    )
    await runtime.run()

    verified = runtime.event_recorder.query("goal_verified")
    assert len(verified) == 1
    assert verified[0]["passed"] is False


async def test_llm_loop_provider_failure_fails_task(tmp_path: Path) -> None:
    """A provider exception transitions the task to FAILED with context."""

    class ExplodingProvider(ScriptedProvider):
        async def complete(self, messages, temperature=0.2, max_tokens=1024, tools=None):
            raise ConnectionError("LLM provider failed after 3 attempts")

    runtime = TetherRuntime(
        "Broken provider", tmp_path, llm_provider=ExplodingProvider([]), max_steps=5
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert runtime.state.error_message is not None
    assert "LLM provider failed" in runtime.state.error_message
    # The failure is checkpointed so recovery can pick it up.
    ckpt = tmp_path / "checkpoints" / f"{runtime.state.task_id}.jsonl"
    assert ckpt.exists()


async def test_llm_loop_malformed_tool_arguments_degrade(tmp_path: Path) -> None:
    """Non-dict tool arguments are tolerated without crashing the loop."""

    class WeirdArgsProvider(ScriptedProvider):
        async def complete(self, messages, temperature=0.2, max_tokens=1024, tools=None):
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(name="search_code", arguments={"a": 1})],
            )

    runtime = TetherRuntime(
        "Weird args", tmp_path, llm_provider=WeirdArgsProvider([]), max_steps=5
    )
    await runtime.run()
    # search_code receives unexpected kwarg -> Tool raises TypeError -> FAILED.
    assert runtime.state.status == TaskStatus.FAILED


def test_openai_tools_schema_format(tmp_path: Path) -> None:
    """Registry emits OpenAI function-calling payload format."""
    runtime = TetherRuntime("Schema", tmp_path)
    schema = runtime.tool_registry.get_openai_tools_schema()
    read_entry = next(t for t in schema if t["function"]["name"] == "read_file")
    assert read_entry["type"] == "function"
    assert read_entry["function"]["parameters"]["required"] == ["path"]


async def test_request_stop_transitions_to_stopped(tmp_path: Path) -> None:
    """request_stop() ends the loop with STOPPED and a saved checkpoint."""

    class StopAfterFirstTool(ScriptedProvider):
        async def complete(self, messages, temperature=0.2, max_tokens=1024, tools=None):
            runtime.request_stop()
            return _tool_response("search_code", {"pattern": "x"})

    runtime = TetherRuntime(
        "Stoppable", tmp_path, llm_provider=StopAfterFirstTool([]), max_steps=10
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.STOPPED
    ckpt = tmp_path / "checkpoints" / f"{runtime.state.task_id}.jsonl"
    assert ckpt.exists()


async def test_unknown_tool_via_llm_fails_task(tmp_path: Path) -> None:
    """A tool call for an unregistered tool fails the task with context."""
    provider = ScriptedProvider([
        _tool_response("does_not_exist", {"x": "1"}),
    ])
    runtime = TetherRuntime(
        "Bad tool", tmp_path, llm_provider=provider, max_steps=5
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert "Unknown tool" in (runtime.state.error_message or "")


async def test_mock_path_still_completes(tmp_path: Path) -> None:
    """Without a provider, the legacy mock loop still runs to completion."""
    runtime = TetherRuntime("Mock mode", tmp_path, tool_timeout=5)
    await runtime.run()
    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.state.prompt_tokens == 0  # mock path burns no tokens


# ---------------------------------------------------------------------
# Memory writing (TaskSummary / EpisodicNotes kept alive by the loop)
# ---------------------------------------------------------------------

async def test_task_summary_written_after_steps(tmp_path: Path) -> None:
    """Successful steps append to the TaskSummary's completed list."""
    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "a.txt", "content": "hi"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Write a.txt", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    summary = runtime.memory_store.load_task_summary(runtime.state.task_id)
    assert summary is not None
    assert summary.goal == "Write a.txt"
    assert any("write_file" in entry for entry in summary.completed)
    assert summary.next_action == "task completed"


async def test_mistake_note_recorded_on_tool_failure(tmp_path: Path) -> None:
    """A failing tool call writes a deduplicated 'mistake' episodic note."""
    provider = ScriptedProvider([
        _tool_response("read_file", {"path": "missing.txt"}),  # fails
        _tool_response("read_file", {"path": "missing.txt"}),  # fails again
        _final_response("gave up"),
    ])
    runtime = TetherRuntime(
        "Read missing file", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    notes = runtime.memory_store.load_episodic_notes(runtime.state.task_id)
    mistakes = [n for n in notes if n.type == "mistake"]
    assert len(mistakes) == 1  # two identical failures -> one deduped note
    assert "read_file" in mistakes[0].content
    assert "File not found" in mistakes[0].content


async def test_episodic_usage_count_bumped_when_selected(tmp_path: Path) -> None:
    """Notes selected into the assembled context get usage_count +1."""
    provider = ScriptedProvider([
        _tool_response("search_code", {"pattern": "x"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Search", tmp_path, llm_provider=provider, max_steps=10
    )
    note = EpisodicNotes(
        task_id=runtime.state.task_id,
        type="preference",
        content="prefer small diffs",
        confidence=0.9,
    )
    runtime.memory_store.save_episodic_note(note)
    await runtime.run()

    stored = runtime.memory_store.load_episodic_notes(runtime.state.task_id)
    updated = next(n for n in stored if n.entry_id == note.entry_id)
    # Both LLM turns (tool call + finishing turn) selected the note.
    assert updated.usage_count == 2


async def test_recovery_writes_lesson_note(tmp_path: Path) -> None:
    """A successful recovery records a 'lesson' episodic note."""
    memory_store = MemoryStore(tmp_path)
    checkpoints = CheckpointManager(tmp_path)
    manager = RecoveryManager(
        checkpoints, memory_store, DriftDetector(tmp_path)
    )
    state = TaskState(goal="g")
    state.status = TaskStatus.FAILED
    state.error_message = "Tool timeout: 'read_file' did not finish"
    checkpoints.save_full(state, memory_store, {})

    result = manager.recover(state.task_id)
    assert result.success
    notes = memory_store.load_episodic_notes(state.task_id)
    assert any(n.type == "lesson" and "TIMEOUT" in n.content for n in notes)


# ---------------------------------------------------------------------
# Errors as observations + consecutive-failure circuit breaker
# ---------------------------------------------------------------------

async def test_error_observation_then_recovery(tmp_path: Path) -> None:
    """With a breaker set, a tool exception becomes an observation the
    agent recovers from on the next turn."""
    provider = ScriptedProvider([
        _tool_response("flaky", {}),
        _tool_response("flaky", {}),
        _final_response("recovered after retry"),
    ])
    runtime = TetherRuntime(
        "Use flaky tool", tmp_path, llm_provider=provider,
        max_steps=10, max_consecutive_failures=3,
    )
    runtime.tool_registry.register(FlakyTool())
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    assert runtime.state.error_message is None
    # The exception landed in the tool history as an observation.
    assert any("ERROR" in out for _, _, out in runtime._tool_history)
    assert any("recovered" in out for _, _, out in runtime._tool_history)


async def test_circuit_breaker_trips_after_consecutive_failures(tmp_path: Path) -> None:
    """Repeated failures trip the breaker and fail the task at the cap."""
    provider = ScriptedProvider([
        _tool_response("flaky", {}),
        _tool_response("flaky", {}),
        _tool_response("flaky", {}),  # never reached
    ])
    runtime = TetherRuntime(
        "Always failing", tmp_path, llm_provider=provider,
        max_steps=10, max_consecutive_failures=2,
    )
    runtime.tool_registry.register(FlakyAlwaysTool())
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert runtime.state.step_index == 2
    assert runtime.state.error_message is not None
    assert "ValueError" in runtime.state.error_message
    events = runtime.event_recorder.query("tool_error")
    assert len(events) == 2


async def test_default_fail_fast_behavior_preserved(tmp_path: Path) -> None:
    """Default (cap=0): the first tool exception fails the task (legacy)."""
    provider = ScriptedProvider([
        _tool_response("flaky", {}),
        _final_response("never reached"),
    ])
    runtime = TetherRuntime(
        "Fail fast", tmp_path, llm_provider=provider, max_steps=10
    )
    runtime.tool_registry.register(FlakyAlwaysTool())
    await runtime.run()

    assert runtime.state.status == TaskStatus.FAILED
    assert runtime.state.step_index == 1


class FlakyAlwaysTool(FlakyTool):
    """Test double: raises on every call."""

    async def execute(self) -> ToolResult:
        self.calls += 1
        raise ValueError("boom")


# ---------------------------------------------------------------------
# Observability: JSONL event stream
# ---------------------------------------------------------------------

async def test_event_stream_records_lifecycle_and_tools(tmp_path: Path) -> None:
    """The event log captures lifecycle, LLM turns and tool executions."""
    provider = ScriptedProvider([
        _tool_response("write_file", {"path": "a.txt", "content": "hi"}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Events", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    events = runtime.event_recorder.query()
    names = [e["event"] for e in events]
    assert "task_started" in names
    assert names.count("llm_turn") == 2
    assert names.count("tool_executed") == 1
    assert "task_completed" in names

    tool_event = runtime.event_recorder.query("tool_executed")[0]
    assert tool_event["tool"] == "write_file"
    assert tool_event["cached"] is False

    llm_turn = runtime.event_recorder.query("llm_turn")[0]
    assert llm_turn["compression_level"] == "NONE"
    assert llm_turn["prompt_tokens"] == 100

    # Persisted to disk as JSONL, one JSON object per line.
    log_file = tmp_path / "logs" / "events.jsonl"
    assert log_file.exists()
    lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(events)
    import json as _json
    parsed = [_json.loads(ln) for ln in lines]
    assert parsed[0]["event"] == "task_started"


# ---------------------------------------------------------------------
# Agent self-planning (update_plan tool)
# ---------------------------------------------------------------------

async def test_update_plan_persists_and_enters_context(tmp_path: Path) -> None:
    """The model's plan lands in TaskSummary and re-enters the context."""
    provider = ScriptedProvider([
        _tool_response("update_plan", {"plan": ["read the file", "fix the bug", "run tests"]}),
        _tool_response("read_file", {"path": "seed.txt"}),
        _final_response("done"),
    ])
    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")
    runtime = TetherRuntime(
        "Fix the bug", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    summary = runtime.memory_store.load_task_summary(runtime.state.task_id)
    assert summary is not None
    assert summary.current_plan == ["read the file", "fix the bug", "run tests"]

    # The plan re-enters the context of the NEXT turn (the assembled
    # system prompt of the second LLM call).
    second_context = provider.calls[1]["messages"][0]["content"]
    assert "read the file" in second_context
    assert "run tests" in second_context


async def test_update_plan_rejects_empty(tmp_path: Path) -> None:
    """An empty plan is a structured tool error, not a crash."""
    provider = ScriptedProvider([
        _tool_response("update_plan", {"plan": []}),
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "Empty plan", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    outputs = [out for _, _, out in runtime._tool_history]
    assert outputs[0].startswith("ERROR:")
    assert runtime.state.status == TaskStatus.COMPLETED
