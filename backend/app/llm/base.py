"""LLM provider abstraction.

Business logic depends only on :class:`LLMProvider`.  ``openai`` is the
default implementation; ``extractive`` is a fully deterministic, offline
provider (no external calls) that answers strictly by extracting sentences
from the supplied context — used for demos/tests without an API key and as a
reference implementation of the grounding contract.  Additional providers
(Anthropic, Gemini, Ollama) register in ``factory.py`` only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


class LLMProvider(ABC):
    name: str
    # True when the provider is a general-purpose instructed LLM usable for
    # query analysis / rewriting and LLM-as-judge evaluation.  The extractive
    # provider is not: those stages fall back to deterministic heuristics.
    is_generative_llm: bool

    @abstractmethod
    def complete(self, messages: list[LLMMessage], *, json_mode: bool = False) -> LLMResponse:
        """Generate a completion. Raises ProviderUnavailable when misconfigured."""

    def ensure_configured(self) -> None:
        """Raise with a clear setup message if the provider cannot run."""
