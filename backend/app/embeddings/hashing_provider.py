"""Deterministic feature-hashing embedder.

Hashes bag-of-words term frequencies into a fixed-size, L2-normalised vector.
It captures lexical overlap, not semantics — it exists so the full pipeline can
be exercised in fast unit tests without downloading models, and must not be
used where retrieval quality matters.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.embeddings.base import EmbeddingProvider

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 384):
        self.name = "hashing"
        self.dimension = dimension

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN.findall(text.lower())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            counts: dict[str, int] = {}
            for token in self._tokens(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimension
                sign = 1.0 if digest[0] % 2 == 0 else -1.0
                vec[index] += sign * (1.0 + math.log(count))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
