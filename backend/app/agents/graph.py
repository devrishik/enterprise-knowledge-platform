"""LangGraph agentic RAG workflow.

Topology (conditional edges in bold):

START → query_analyzer → **rewrite?** → query_rewriter → retrieve_dense
                                                     ↘ retrieve_lexical
        retrieve_dense ↘                        ↗
        retrieve_lexical → fuse_results → rerank → context_validator
                                  ↑ **insufficient + retries left** (retry_with_alternative)
        context_validator → **sufficient** → generate_answer → citation_validator
        citation_validator → **invalid refs or quotes** → generate_answer (bounded regeneration)
                       └────→ final_response → END
"""

from __future__ import annotations

import re
import time

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.context_validator import ContextValidatorNode
from app.agents.nodes.generation_nodes import (
    CitationValidatorNode,
    FinalResponseNode,
    GenerateAnswerNode,
)
from app.agents.nodes.query_understanding import QueryAnalyzerNode, QueryRewriterNode
from app.agents.nodes.retrieval_nodes import (
    FuseResultsNode,
    RerankNode,
    RetrieveDenseNode,
    RetrieveLexicalNode,
)
from app.agents.state import RAGState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import get_llm_provider
from app.llm.base import LLMMessage
from app.retrieval.rerank import content_words

_ALTERNATIVE_SYSTEM = (
    "Rewrite the question as a single self-contained keyword-rich search query for an "
    "enterprise policy document search engine. Reply with the query text only."
)


class RetryWithAlternativeNode:
    """Weak retrieval: reformulate the query once and retrieve again."""

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        provider = get_llm_provider()
        question = state.get("effective_query") or state["question"]

        if provider.is_generative_llm:
            response = provider.complete(
                [
                    LLMMessage("system", _ALTERNATIVE_SYSTEM),
                    LLMMessage("user", question),
                ]
            )
            alternative = re.sub(r"\s+", " ", response.content).strip().strip('"')[:500]
            method = "llm_reformulation"
        else:
            words = sorted(content_words(question))
            alternative = " ".join(words)[:500]
            method = "keyword_variant"

        if not alternative:
            alternative = question
            method = "unchanged"

        return {
            "effective_query": alternative,
            "retry_count": state.get("retry_count", 0) + 1,
            "dense_results": [],
            "lexical_results": [],
            "stages": [
                {
                    "stage": "retry_with_alternative",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {"method": method, "alternative_query": alternative},
                }
            ],
        }


class RAGWorkflow:
    """Builds and runs the agentic RAG graph.

    ``session`` is used for orchestration-level persistence by the caller; the
    parallel retrieval nodes open their own short-lived sessions (see
    retrieval_nodes.py).
    """

    def __init__(self, session):
        self.settings = get_settings()
        self.session = session
        self.graph = self._build()
        self.log = get_logger("rag.workflow")

    # ------------------------------------------------------------- routing
    def _route_after_analysis(self, state: RAGState) -> str | list[str]:
        analysis = state.get("analysis", {})
        if not analysis.get("answerable_from_documents", True):
            return "final_response"
        if analysis.get("needs_rewrite") or analysis.get("sub_queries"):
            return "query_rewriter"
        return ["retrieve_dense", "retrieve_lexical"]

    def _route_after_validation(self, state: RAGState) -> str:
        validation = state.get("context_validation", {})
        if validation.get("status") == "sufficient":
            return "generate_answer"
        retries = state.get("retry_count", 0)
        # Only retry when retrieval found *something* weak (an alternative
        # formulation may help); retrying on zero results for out-of-corpus
        # questions just burns latency.
        if retries < self.settings.max_query_rewrites and validation.get("chunk_count", 0) > 0:
            return "retry_with_alternative"
        return "final_response"

    def _route_after_citations(self, state: RAGState) -> str:
        return (
            "generate_answer"
            if state.get("citation_validation", {}).get("regenerate")
            else "final_response"
        )

    # ---------------------------------------------------------------- build
    def _build(self):
        builder = StateGraph(RAGState)
        builder.add_node("query_analyzer", QueryAnalyzerNode())
        builder.add_node("query_rewriter", QueryRewriterNode())
        builder.add_node("retrieve_dense", RetrieveDenseNode(self.settings))
        builder.add_node("retrieve_lexical", RetrieveLexicalNode(self.settings))
        builder.add_node("fuse_results", FuseResultsNode(self.settings))
        builder.add_node("rerank", RerankNode(self.settings))
        builder.add_node("context_validator", ContextValidatorNode(self.settings))
        builder.add_node("retry_with_alternative", RetryWithAlternativeNode())
        builder.add_node("generate_answer", GenerateAnswerNode(self.settings))
        builder.add_node("citation_validator", CitationValidatorNode(self.settings))
        builder.add_node("final_response", FinalResponseNode(self.settings))

        builder.add_edge(START, "query_analyzer")
        # The router returns a list for the parallel retrieval fan-out; a
        # path_map cannot express list targets in current LangGraph versions.
        builder.add_conditional_edges("query_analyzer", self._route_after_analysis)
        builder.add_edge("query_rewriter", "retrieve_dense")
        builder.add_edge("query_rewriter", "retrieve_lexical")
        builder.add_edge("retrieve_dense", "fuse_results")
        builder.add_edge("retrieve_lexical", "fuse_results")
        builder.add_edge("fuse_results", "rerank")
        builder.add_edge("rerank", "context_validator")
        builder.add_conditional_edges(
            "context_validator",
            self._route_after_validation,
            {
                "generate_answer": "generate_answer",
                "retry_with_alternative": "retry_with_alternative",
                "final_response": "final_response",
            },
        )
        builder.add_edge("retry_with_alternative", "retrieve_dense")
        builder.add_edge("retry_with_alternative", "retrieve_lexical")
        builder.add_edge("generate_answer", "citation_validator")
        builder.add_conditional_edges(
            "citation_validator",
            self._route_after_citations,
            {"generate_answer": "generate_answer", "final_response": "final_response"},
        )
        builder.add_edge("final_response", END)
        return builder.compile()

    # ------------------------------------------------------------------ run
    def run(self, *, question: str, tenant_id, conversation_history: list[dict] | None = None) -> RAGState:
        initial: RAGState = {
            "question": question,
            "tenant_id": str(tenant_id),
            "conversation_history": conversation_history or [],
            "retry_count": 0,
            "generation_attempts": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        final_state = self.graph.invoke(initial, config={"recursion_limit": 25})
        return final_state
