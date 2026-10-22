"""Unit tests: context validation (relevance gate + conflict heuristic)."""

from __future__ import annotations

from app.agents.nodes.context_validator import ContextValidatorNode, detect_conflict_hint
from app.core.config import get_settings


def make_node() -> ContextValidatorNode:
    return ContextValidatorNode(get_settings())


def chunk_dict(cid: str, doc: str, score: float, text: str) -> dict:
    return {
        "chunk_id": cid, "document_id": doc, "document_filename": f"{doc}.md",
        "text": text, "score": score, "source": "reranked",
    }


def test_zero_chunks_is_insufficient():
    state = make_node()({"reranked_results": [], "retrieval_confidence": 0.0})
    assert state["context_validation"]["status"] == "insufficient"
    assert "No chunks" in state["context_validation"]["reason"]


def test_low_confidence_is_insufficient():
    chunks = [chunk_dict("c1", "leave", 0.1, "Some text about leave policy entitlements.")]
    state = make_node()({"reranked_results": chunks, "retrieval_confidence": 0.1})
    assert state["context_validation"]["status"] == "insufficient"


def test_strong_results_are_sufficient():
    chunks = [chunk_dict("c1", "leave", 0.9, "Employees receive 22 days of paid annual leave.")]
    state = make_node()({"reranked_results": chunks, "retrieval_confidence": 0.9})
    assert state["context_validation"]["status"] == "sufficient"
    assert state["context_validation"]["chunk_count"] == 1


def test_conflict_modality_markers_detected():
    a = chunk_dict("c1", "handbook", 0.8,
                   "Employees may work remotely up to two days per week with approval.")
    b = chunk_dict("c2", "remote_policy", 0.75,
                   "Employees may not work remotely during their probation period.")
    assert detect_conflict_hint([a, b]) is True


def test_conflict_disjoint_numbers_detected():
    a = chunk_dict("c1", "handbook", 0.8, "Employees receive 18 days of paid time off per year.")
    b = chunk_dict("c2", "leave_policy", 0.78, "Employees are entitled to 22 days of paid annual leave.")
    assert detect_conflict_hint([a, b]) is True


def test_no_conflict_for_consistent_documents():
    a = chunk_dict("c1", "security", 0.8, "Passwords must rotate every 90 days for production systems.")
    b = chunk_dict("c2", "it", 0.79, "Laptops are refreshed every 3 years through the IT portal.")
    assert detect_conflict_hint([a, b]) is False


def test_no_conflict_within_same_document():
    a = chunk_dict("c1", "leave", 0.8, "Annual leave is 22 days; carryover is 5 days.")
    b = chunk_dict("c2", "leave", 0.78, "Sick leave is 12 days with medical certificate rules.")
    assert detect_conflict_hint([a, b]) is False
