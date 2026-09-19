"""OpenAI-compatible REST provider (DeepSeek and friends).

Uses the standard library only: ``urllib.request`` wrapped in
``asyncio.to_thread`` so the async runtime stays non-blocking. Retries
with exponential backoff on 429 / 5xx / network errors (max 3 attempts).
"""

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from loguru import logger

from tether.llm.base import LLMResponse, ToolCall

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Upper bound for auto-escalated max_tokens on truncated responses.
_MAX_TOKEN_CEILING = 8192


class OpenAICompatProvider:
    """Async client for any OpenAI-compatible ``/chat/completions`` API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """Store endpoint config; no network happens until ``complete``."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.name = "openai_compat"

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Call the chat completions endpoint with retry + backoff.

        Reasoning models can exhaust ``max_tokens`` on chain-of-thought
        before emitting any content (``finish_reason == "length"`` with
        empty ``content``). When that happens the budget is doubled and
        the call retried, up to ``_MAX_TOKEN_CEILING``. Responses that
        request tool calls are returned as-is (empty content is normal
        there and must not trigger the truncation retry).

        With ``on_delta`` the request uses SSE streaming and the callable
        receives each content chunk as it arrives (the assembled full
        text still comes back on the response).
        """
        start = time.perf_counter()
        last_error: Exception | None = None
        budget = max_tokens

        while True:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": budget,
            }
            if tools:
                payload["tools"] = tools
            if on_delta is not None:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
            response: LLMResponse | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    if on_delta is not None:
                        body = await asyncio.to_thread(
                            self._post_stream, payload, on_delta
                        )
                    else:
                        body = await asyncio.to_thread(
                            self._post, json.dumps(payload).encode("utf-8")
                        )
                    response = self._parse_response(body, start)
                    break
                except urllib.error.HTTPError as exc:
                    # 4xx other than 429 (bad key, wrong model, not found)
                    # will not get better on retry: fail fast so the caller
                    # sees the real cause instead of "failed after N attempts".
                    if exc.code not in _RETRYABLE_STATUS:
                        raise ConnectionError(
                            f"LLM provider returned HTTP {exc.code}: {exc.reason}"
                        ) from exc
                    last_error = exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_error = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt {}/{}): {} - retrying in {}s",
                        attempt, self.max_retries, last_error, wait,
                    )
                    await asyncio.sleep(wait)
            if response is None:
                raise ConnectionError(
                    f"LLM provider failed after {self.max_retries} "
                    f"attempts: {last_error}"
                )

            if response.has_tool_calls:
                return response
            truncated = (
                not response.content.strip()
                or response.extra.get("finish_reason") == "length"
            )
            if truncated and budget < _MAX_TOKEN_CEILING:
                budget = min(budget * 2, _MAX_TOKEN_CEILING)
                logger.info(
                    "Response truncated (finish_reason={}); retrying with "
                    "max_tokens={}",
                    response.extra.get("finish_reason"), budget,
                )
                continue
            return response

    def _post(self, body: bytes) -> dict:
        """Blocking POST to the completions endpoint (runs in a thread)."""
        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, payload: dict, on_delta: Callable[[str], None]) -> dict:
        """Blocking streaming POST (runs in a thread).

        Parses the SSE line protocol, forwards each content delta to
        ``on_delta`` as it arrives, and assembles an OpenAI-style
        response body so the normal parsing path can consume it.
        """
        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
            return accumulate_sse(resp, on_delta)

    def _parse_response(self, body: dict, start: float) -> LLMResponse:
        """Extract content + tool calls + usage from an OpenAI-style body."""
        latency_ms = (time.perf_counter() - start) * 1000
        choices = body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        return LLMResponse(
            content=message.get("content", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            model=body.get("model", self.model),
            provider=self.name,
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            extra={"finish_reason": choices[0].get("finish_reason", "")},
        )

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
        """Normalize the ``message.tool_calls`` payload into ToolCalls.

        ``arguments`` arrives as a JSON *string* on the wire; malformed
        JSON degrades to empty arguments rather than failing the whole
        completion (the runtime surfaces a tool-level error instead).
        """
        calls: list[ToolCall] = []
        for raw in raw_calls or []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            name = function.get("name", "")
            if not name:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    arguments = {"__raw__": arguments}
            except json.JSONDecodeError:
                logger.warning("Malformed tool-call arguments for {}", name)
                arguments = {}
            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    id=str(raw.get("id", "")),
                )
            )
        return calls


def accumulate_sse(line_iter, on_delta: Callable[[str], None]) -> dict:
    """Fold an SSE chunk stream into an OpenAI-style response body.

    ``line_iter`` yields already-decoded lines (a file-like response
    object iterates lines). ``data: [DONE]`` terminates the stream.
    """
    content_parts: list[str] = []
    finish_reason = ""
    usage: dict = {}
    model = ""
    for raw in line_iter:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed SSE chunk: {}", data[:80])
            continue
        model = chunk.get("model", model)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or ""
        if piece:
            content_parts.append(piece)
            on_delta(piece)
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": "".join(content_parts)},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


# Convenience alias: DeepSeek is just a preset endpoint.
class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek chat completions (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat") -> None:
        """Default to the official DeepSeek endpoint."""
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
        )
        self.name = "deepseek"
