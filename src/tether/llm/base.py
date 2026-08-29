"""LLMProvider protocol and LLMResponse data type."""

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """One completion result with usage stats for benchmark accounting."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens as reported by the API."""
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal async completion interface all providers implement."""

    name: str
    model: str

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return one completion for the OpenAI-style message list.

        ``messages`` follows the OpenAI chat format:
        ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
        """
        ...
