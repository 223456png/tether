"""MockProvider: offline fallback that never touches the network.

Returns a canned completion derived from the last user message. Used by
regression tests and by the e2e experiment when no API key is configured,
so every code path stays runnable offline (CI included).
"""

import hashlib
import time

from tether.llm.base import LLMResponse

# Deterministic filler the mock streams back as "completion".
_MOCK_COMPLETION = "def solution(x):\n    return x\n"


class MockProvider:
    """Offline provider with deterministic, usage-reporting responses."""

    def __init__(self, model: str = "mock-1") -> None:
        """Configure the mock model name reported in responses."""
        self.model = model
        self.name = "mock"
        self.call_count = 0

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Return a deterministic completion with estimated token usage.

        ``tools`` is accepted for protocol conformance but ignored — the
        mock never requests tool calls.
        """
        start = time.perf_counter()
        self.call_count += 1
        prompt_text = "".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
        # Rough token estimate (4 chars/token) keeps usage stats realistic.
        prompt_tokens = max(1, len(prompt_text) // 4)
        return LLMResponse(
            content=_MOCK_COMPLETION,
            prompt_tokens=prompt_tokens,
            completion_tokens=max(1, len(_MOCK_COMPLETION) // 4),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
            provider=self.name,
            extra={"prompt_digest": hashlib.md5(prompt_text.encode()).hexdigest()[:8]},
        )
