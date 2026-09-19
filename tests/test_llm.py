"""Tests for the Phase-9 LLM provider layer."""

import os
import urllib.error
from unittest import mock as unittest_mock

import pytest

from tether.llm.base import LLMProvider, LLMResponse
from tether.llm.factory import create_provider_from_env
from tether.llm.mock import MockProvider
from tether.llm.openai_compat import OpenAICompatProvider


def test_llm_response_token_accounting() -> None:
    """total_tokens = prompt + completion."""
    resp = LLMResponse(content="hi", prompt_tokens=10, completion_tokens=5)
    assert resp.total_tokens == 15


@pytest.mark.asyncio
async def test_mock_provider_reports_usage() -> None:
    """Mock returns deterministic content and estimated token usage."""
    provider = MockProvider()
    resp = await provider.complete(
        [{"role": "user", "content": "a" * 400}], max_tokens=64
    )
    assert resp.provider == "mock"
    assert resp.prompt_tokens == 100  # 400 chars / 4
    assert resp.completion_tokens >= 1
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_openai_compat_parses_response() -> None:
    """Provider parses OpenAI-style JSON into an LLMResponse."""
    provider = OpenAICompatProvider(api_key="test-key")

    def fake_post(body: bytes) -> dict:
        return {
            "model": "deepseek-chat",
            "choices": [
                {"message": {"role": "assistant", "content": "def f(x): return x"}}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }

    with unittest_mock.patch.object(provider, "_post", side_effect=fake_post):
        resp = await provider.complete([{"role": "user", "content": "task"}])
    assert resp.content == "def f(x): return x"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 7
    assert resp.model == "deepseek-chat"


@pytest.mark.asyncio
async def test_openai_compat_retries_then_raises() -> None:
    """Network errors exhaust retries and raise ConnectionError."""
    provider = OpenAICompatProvider(api_key="k", max_retries=2)

    def failing_post(body: bytes) -> dict:
        raise urllib.error.URLError("unreachable")

    async def instant_sleep(_seconds: float) -> None:
        return None

    with unittest_mock.patch.object(
        provider, "_post", side_effect=failing_post
    ), unittest_mock.patch("asyncio.sleep", new=instant_sleep):
        with pytest.raises(ConnectionError):
            await provider.complete([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_openai_compat_fails_fast_on_non_retryable_http_error() -> None:
    """4xx other than 429 must not be retried (no wasted backoff)."""
    provider = OpenAICompatProvider(api_key="k", max_retries=3)
    calls = {"n": 0}

    def unauthorized(body: bytes) -> dict:
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url="https://api.deepseek.com/v1/chat/completions",
            code=401, msg="Unauthorized", hdrs=None, fp=None,
        )

    async def instant_sleep(_seconds: float) -> None:
        return None

    with unittest_mock.patch.object(
        provider, "_post", side_effect=unauthorized
    ), unittest_mock.patch("asyncio.sleep", new=instant_sleep):
        with pytest.raises(ConnectionError):
            await provider.complete([{"role": "user", "content": "x"}])
    assert calls["n"] == 1  # failed fast instead of retrying three times


@pytest.mark.asyncio
async def test_openai_compat_retries_retryable_http_error() -> None:
    """429 is retryable and exhausts max_retries before raising."""
    provider = OpenAICompatProvider(api_key="k", max_retries=2)
    calls = {"n": 0}

    def rate_limited(body: bytes) -> dict:
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url="https://api.deepseek.com/v1/chat/completions",
            code=429, msg="Too Many Requests", hdrs=None, fp=None,
        )

    async def instant_sleep(_seconds: float) -> None:
        return None

    with unittest_mock.patch.object(
        provider, "_post", side_effect=rate_limited
    ), unittest_mock.patch("asyncio.sleep", new=instant_sleep):
        with pytest.raises(ConnectionError):
            await provider.complete([{"role": "user", "content": "x"}])
    assert calls["n"] == 2


def test_factory_env_resolution() -> None:
    """Factory picks DeepSeek / generic / mock based on env vars."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("DEEPSEEK_API_KEY", "TETHER_LLM_API_KEY")}
    with unittest_mock.patch.dict(os.environ, env, clear=True):
        provider, is_mock = create_provider_from_env()
        assert is_mock is True
        assert isinstance(provider, MockProvider)

    with unittest_mock.patch.dict(
        os.environ, {**env, "DEEPSEEK_API_KEY": "sk-test"}
    ):
        provider, is_mock = create_provider_from_env()
        assert is_mock is False
        assert isinstance(provider, OpenAICompatProvider)

    with unittest_mock.patch.dict(
        os.environ,
        {**env, "TETHER_LLM_API_KEY": "sk-x",
         "TETHER_LLM_BASE_URL": "https://example.com/v1",
         "TETHER_LLM_MODEL": "custom-model"},
    ):
        provider, is_mock = create_provider_from_env()
        assert is_mock is False
        assert provider.base_url == "https://example.com/v1"
        assert provider.model == "custom-model"


def test_provider_protocol_conformance() -> None:
    """Both implementations satisfy the runtime-checkable protocol."""
    assert isinstance(MockProvider(), LLMProvider)
    assert isinstance(OpenAICompatProvider(api_key="k"), LLMProvider)
