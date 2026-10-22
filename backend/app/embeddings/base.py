"""Embedding provider abstraction.

The default provider runs Sentence Transformers locally.  A deterministic
feature-hashing provider exists for fast, dependency-light pipeline tests.
Additional providers (OpenAI, Vertex, ...) only need to implement
:class:`EmbeddingProvider` and register in ``factory.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; vectors must be L2-normalised (cosine space)."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
