"""Unit tests: RRF fusion."""

from __future__ import annotations

from app.retrieval import ScoredChunk
from app.retrieval.fusion import reciprocal_rank_fusion


def chunk(cid: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid, document_id="d", document_filename="doc.md",
        text=cid, score=score, source="dense",
    )


def test_rrf_hand_computed():
    # dense: [a, b, c]; lexical: [b, a, d]
    dense = [chunk("a"), chunk("b"), chunk("c")]
    lexical = [chunk("b"), chunk("a"), chunk("d")]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    ids = [c.chunk_id for c in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    a = next(c for c in fused if c.chunk_id == "a")
    b = next(c for c in fused if c.chunk_id == "b")
    c_ = next(c for c in fused if c.chunk_id == "c")
    d = next(c for c in fused if c.chunk_id == "d")

    # rrf(a) = 1/61 + 1/62; rrf(b) = 1/62 + 1/61 -> identical
    assert abs(a.metadata["rrf_score"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(a.score - b.score) < 1e-9
    # appearing in both lists beats appearing in one
    assert a.score > c_.score and a.score > d.score
    # single-list chunks tie (both rank 3)
    assert abs(c_.score - d.score) < 1e-12


def test_rrf_top_k_truncates():
    dense = [chunk(f"d{i}") for i in range(10)]
    lexical = [chunk(f"l{i}") for i in range(10)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_scores_normalised():
    dense = [chunk("a"), chunk("b")]
    lexical = [chunk("a")]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    assert fused[0].score == 1.0  # max normalised
    assert all(0.0 <= c.score <= 1.0 for c in fused)
    assert all(c.source == "fused" for c in fused)


def test_rrf_keeps_richest_metadata():
    a_dense = ScoredChunk("a", "d", "doc.md", "shared text", 0.9, "dense", page_number=3, section="S")
    a_lexical = ScoredChunk("a", "d", "doc.md", "shared text", 0.1, "lexical")
    fused = reciprocal_rank_fusion([[a_dense], [a_lexical]], k=60)
    assert fused[0].page_number == 3
    assert fused[0].section == "S"
