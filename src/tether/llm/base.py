"""LLMProvider protocol and LLMResponse data type."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """One structured tool invocation requested by the model.

    ``arguments`` holds the already-parsed JSON object from the
    provider's function-calling payload (empty dict on parse failure).
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    """One completion result with usage stats for benchmark accounting."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens as reported by the API."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        """True when the model requested at least one tool invocation."""
        return bool(self.tool_calls)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal async completion interface all providers implement."""

    name: str
    model: str

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Return one completion for the OpenAI-style message list.

        ``messages`` follows the OpenAI chat format:
        ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
        ``tools`` is an optional OpenAI function-calling schema list; when
        given, providers that support it may return ``tool_calls`` on the
        response instead of (or alongside) plain ``content``.
        """
        ...
