"""Reciprocal Rank Fusion (RRF).

RRF combines ranked lists without needing comparable score scales:

    score(d) = Σ_lists 1 / (k + rank_i(d)),  rank starts at 1, k default 60

It is robust to the wildly different score distributions of cosine similarity
and ts_rank_cd, which is why it is preferred over score-level mixing here.
Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual
Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

from app.retrieval import ScoredChunk


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists into one ranked list."""
    fused_scores: dict[str, float] = {}
    by_chunk: dict[str, ScoredChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = by_chunk.get(chunk.chunk_id)
            # keep the richest metadata; first occurrence wins for scalar fields
            if existing is None:
                by_chunk[chunk.chunk_id] = chunk

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    max_score = ordered[0][1] if ordered else 0.0
    fused: list[ScoredChunk] = []
    for chunk_id, score in ordered:
        base = by_chunk[chunk_id]
        fused.append(
            ScoredChunk(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                document_filename=base.document_filename,
                text=base.text,
                score=score if max_score == 0 else score / max_score,  # normalised [0,1]
                source="fused",
                page_number=base.page_number,
                section=base.section,
                metadata={**base.metadata, "rrf_score": score},
            )
        )
    return fused
