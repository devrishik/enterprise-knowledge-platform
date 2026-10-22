"""OpenAI Chat Completions provider (default)."""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ProviderUnavailable
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, TokenUsage


class OpenAIProvider(LLMProvider):
    name = "openai"
    is_generative_llm = True

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url or None,
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._client

    def ensure_configured(self) -> None:
        if not self.settings.openai_api_key:
            raise ProviderUnavailable(
                "No OpenAI API key configured. Set EKIP_OPENAI_API_KEY in your environment "
                "or .env file, or switch to the offline provider with EKIP_LLM_PROVIDER=extractive."
            )

    def complete(self, messages: list[LLMMessage], *, json_mode: bool = False) -> LLMResponse:
        self.ensure_configured()
        client = self._get_client()
        kwargs: dict = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderUnavailable(f"OpenAI request failed: {type(exc).__name__}: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        token_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            content=(choice.message.content or "") if choice else "",
            model=self.settings.llm_model,
            usage=token_usage,
            finish_reason=(choice.finish_reason if choice else None),
        )
