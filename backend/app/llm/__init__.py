"""LLM provider factory (cached singleton per process).

Adding a provider (Anthropic, Gemini, Ollama, ...) means implementing
:class:`app.llm.base.LLMProvider` and registering it in ``_BUILDERS`` — no
business logic changes.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider

_cached: LLMProvider | None = None


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if settings.llm_provider == "extractive":
        from app.llm.extractive_provider import ExtractiveProvider

        return ExtractiveProvider()
    raise ValueError(f"Unknown LLM provider '{settings.llm_provider}'")


def get_llm_provider() -> LLMProvider:
    global _cached
    if _cached is None:
        _cached = build_llm_provider(get_settings())
    return _cached


def reset_llm_provider() -> None:
    global _cached
    _cached = None
