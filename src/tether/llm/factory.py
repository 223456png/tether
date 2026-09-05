"""Create the configured LLM provider from environment variables.

Resolution order:
1. ``DEEPSEEK_API_KEY`` -> DeepSeekProvider
2. ``TETHER_LLM_API_KEY`` -> generic OpenAI-compatible endpoint
   (``TETHER_LLM_BASE_URL`` defaults to DeepSeek, ``TETHER_LLM_MODEL``
   defaults to ``deepseek-chat``)
3. neither set -> MockProvider (offline fallback, flagged via
   ``is_mock`` on the returned tuple)
"""

import os

from loguru import logger

from tether.llm.base import LLMProvider
from tether.llm.mock import MockProvider
from tether.llm.openai_compat import OpenAICompatProvider

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-chat"


def create_provider_from_env() -> tuple[LLMProvider, bool]:
    """Return ``(provider, is_mock)`` based on the environment."""
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        model = os.environ.get("TETHER_LLM_MODEL", _DEFAULT_MODEL).strip()
        logger.info("LLM provider: DeepSeek (model={})", model)
        return OpenAICompatProvider(
            api_key=deepseek_key,
            base_url=_DEFAULT_BASE_URL,
            model=model,
        ), False

    generic_key = os.environ.get("TETHER_LLM_API_KEY", "").strip()
    if generic_key:
        base_url = os.environ.get(
            "TETHER_LLM_BASE_URL", _DEFAULT_BASE_URL
        ).strip()
        model = os.environ.get("TETHER_LLM_MODEL", _DEFAULT_MODEL).strip()
        logger.info("LLM provider: OpenAI-compatible {} (model={})", base_url, model)
        return OpenAICompatProvider(
            api_key=generic_key, base_url=base_url, model=model
        ), False

    logger.warning("No LLM API key found - falling back to MockProvider")
    return MockProvider(), True
