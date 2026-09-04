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
from typing import Any, Dict, List, Optional

from loguru import logger

from tether.llm.base import LLMProvider, LLMResponse, ToolCall

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
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: Optional[List[dict]] = None,
    ) -> LLMResponse:
        """Call the chat completions endpoint with retry + backoff.

        Reasoning models can exhaust ``max_tokens`` on chain-of-thought
        before emitting any content (``finish_reason == "length"`` with
        empty ``content``). When that happens the budget is doubled and
        the call retried, up to ``_MAX_TOKEN_CEILING``. Responses that
        request tool calls are returned as-is (empty content is normal
        there and must not trigger the truncation retry).
        """
        start = time.perf_counter()
        last_error: Optional[Exception] = None
        budget = max_tokens

        while True:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": budget,
            }
            if tools:
                payload["tools"] = tools
            response: Optional[LLMResponse] = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    body = await asyncio.to_thread(
                        self._post, json.dumps(payload).encode("utf-8")
                    )
                    response = self._parse_response(body, start)
                    break
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    TimeoutError,
                ) as exc:
                    last_error = exc
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt {}/{}): {} - retrying in {}s",
                        attempt, self.max_retries, exc, wait,
                    )
                    if attempt < self.max_retries:
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
    def _parse_tool_calls(raw_calls: Any) -> List[ToolCall]:
        """Normalize the ``message.tool_calls`` payload into ToolCalls.

        ``arguments`` arrives as a JSON *string* on the wire; malformed
        JSON degrades to empty arguments rather than failing the whole
        completion (the runtime surfaces a tool-level error instead).
        """
        calls: List[ToolCall] = []
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
