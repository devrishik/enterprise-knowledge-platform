"""Standard information-retrieval metrics (pure functions, unit-tested).

All metrics take an ordered list of retrieved item identifiers and a set of
relevant identifiers (here: filenames of documents expected to support the
answer, from the evaluation dataset).  Identifiers are deduplicated in rank
order first — several chunks of the same document must not count as multiple
relevant hits.
"""

from __future__ import annotations

import math


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant items, what fraction appears in the top-k?"""
    if not relevant:
        return 0.0
    top_k = _unique_in_order(retrieved)[:k]
    return len(set(top_k) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top-k retrieved items, what fraction is relevant?"""
    top_k = _unique_in_order(retrieved)[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none in the list)."""
    for rank, item in enumerate(_unique_in_order(retrieved), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    ranked = _unique_in_order(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked, start=1) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0,100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100 * len(ordered))))
    return round(ordered[rank - 1], 2)
