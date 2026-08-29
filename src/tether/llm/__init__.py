"""LLM provider abstraction: protocol + response type.

Providers are async and report token usage so benchmark tasks can measure
real costs. Two implementations exist: ``DeepSeekProvider`` (any
OpenAI-compatible REST endpoint) and ``MockProvider`` (offline fallback).
"""

from .base import LLMProvider, LLMResponse
from .factory import create_provider_from_env
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockProvider",
    "OpenAICompatProvider",
    "create_provider_from_env",
]
