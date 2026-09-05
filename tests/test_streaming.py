"""Tests for SSE streaming (accumulate_sse + runtime on_llm_delta)."""

import io
from pathlib import Path

from tests.test_agent_loop import _final_response, _tool_response
from tether.llm.base import LLMResponse
from tether.llm.openai_compat import OpenAICompatProvider, accumulate_sse
from tether.runtime.runtime import TetherRuntime
from tether.runtime.state import TaskStatus


def test_accumulate_sse_assembles_chunks() -> None:
    """SSE lines fold into an OpenAI-style body; deltas reach the callback."""
    stream = io.BytesIO(b"""data: {"model":"m1","choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}

data: [DONE]

""")
    seen: list[str] = []

    body = accumulate_sse(io.TextIOWrapper(stream, encoding="utf-8"), seen.append)

    assert seen == ["Hel", "lo"]
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2


def test_accumulate_sse_ignores_malformed_lines() -> None:
    """Bad JSON and comment lines are skipped, not fatal."""
    stream = io.BytesIO(
        b": keep-alive\n\ndata: not-json\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
    )
    seen: list[str] = []

    body = accumulate_sse(io.TextIOWrapper(stream, encoding="utf-8"), seen.append)

    assert seen == ["ok"]
    assert body["choices"][0]["message"]["content"] == "ok"


async def test_provider_complete_with_on_delta_streams() -> None:
    """complete(on_delta=...) routes through the streaming POST path."""
    provider = OpenAICompatProvider(api_key="k", max_retries=1)
    payloads: list[dict] = []

    def fake_post_stream(payload: dict, on_delta) -> dict:
        payloads.append(payload)
        for piece in ("He", "llo"):
            on_delta(piece)
        return {
            "model": "m1",
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    provider._post_stream = fake_post_stream  # type: ignore[method-assign]
    seen: list[str] = []

    resp = await provider.complete(
        [{"role": "user", "content": "hi"}], on_delta=seen.append
    )

    assert seen == ["He", "llo"]
    assert resp.content == "Hello"
    assert payloads and payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}


class StreamingScriptedProvider:
    """Scripted provider that accepts (and exercises) on_delta."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        on_delta=None,
    ) -> LLMResponse:
        if on_delta is not None:
            for piece in ("work", "ing"):
                on_delta(piece)
        if not self.responses:
            return _final_response("no script left")
        return self.responses.pop(0)


async def test_runtime_streams_deltas_to_callback(tmp_path: Path) -> None:
    """on_llm_delta reaches providers that declare the on_delta parameter."""
    provider = StreamingScriptedProvider([
        _tool_response("search_code", {"pattern": "x"}),
        _final_response("done"),
    ])
    seen: list[str] = []
    runtime = TetherRuntime(
        "Stream test", tmp_path, llm_provider=provider, max_steps=10,
        on_llm_delta=seen.append,
    )
    await runtime.run()

    assert seen == ["work", "ing"] * 2  # both LLM turns streamed their chunks
    assert runtime.state.status == TaskStatus.COMPLETED


async def test_runtime_without_callback_skips_streaming(tmp_path: Path) -> None:
    """No on_llm_delta -> the provider is called without the kwarg."""
    provider = StreamingScriptedProvider([
        _final_response("done"),
    ])
    runtime = TetherRuntime(
        "No stream", tmp_path, llm_provider=provider, max_steps=10
    )
    await runtime.run()

    assert runtime.state.status == TaskStatus.COMPLETED
    # StreamingScriptedProvider never emitted without on_delta: nothing
    # observed, and the loop finished on the scripted response.
