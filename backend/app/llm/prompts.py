"""Prompts for grounded generation, query analysis and LLM-as-judge evaluation.

The context-block format (``[n] (chunk_id=...)``) is deliberately shared by
the OpenAI provider and the deterministic extractive provider, so both parse
identical prompts.
"""

from __future__ import annotations

from app.retrieval import ScoredChunk

GROUNDED_ANSWER_SYSTEM = """You are an enterprise knowledge assistant. You answer questions strictly
from the supplied CONTEXT blocks.

Rules:
- Base every factual claim on the context. Do NOT use outside knowledge.
- If the context is insufficient, set "insufficient_evidence" to true and say so.
- Never fabricate page numbers, sections or document names.
- Cite with the block number and a verbatim quote copied from that block, for example {"ref": 1, "quote": "exact words copied from block 1"}. The quote must be an exact span of the block text, at least 20 characters, not a paraphrase.
- Do not cite blocks that you did not use.
- If different blocks state conflicting information, set "conflicting_evidence"
  to true, say the sources conflict, and cite every conflicting block.
- Distinguish clearly between what the documents state and what remains
  uncertain.

Respond with ONLY a JSON object:
{
  "answer": "string",
  "citations": [{"ref": 1, "quote": "exact words copied from block 1"}],
  "confidence": 0.0,
  "insufficient_evidence": false,
  "conflicting_evidence": false
}
"confidence" is your calibrated confidence that the answer is fully supported
by the cited context, from 0.0 to 1.0."""

QUERY_ANALYSIS_SYSTEM = """You analyse user questions for a retrieval-augmented system.
Respond with ONLY a JSON object:
{
  "needs_rewrite": false,
  "rewritten_query": "",
  "sub_queries": [],
  "answerable_from_documents": true,
  "reason": "short explanation"
}
Guidance:
- If the question is already self-contained and clear, set needs_rewrite=false
  and leave rewritten_query empty. Do not rewrite clear queries.
- If the question is short, uses pronouns ("it", "this policy") referring to
  the conversation, or is ambiguous, set needs_rewrite=true and produce a
  self-contained rewritten_query.
- If the question combines multiple distinct questions, split them into
  sub_queries (max 3).
- Set answerable_from_documents=false only for chit-chat or clearly
  out-of-scope requests."""

FAITHFULNESS_JUDGE_SYSTEM = """You are a strict evaluator of grounded answers.
Given the QUESTION, the CONTEXT and the ANSWER, decide whether every factual
claim in the ANSWER is directly supported by the CONTEXT.
Respond with ONLY a JSON object:
{"supported_claims": 3, "unsupported_claims": 1, "faithful": false, "explanation": "..."}
"faithful" is true only when unsupported_claims is 0."""

RELEVANCY_JUDGE_SYSTEM = """You are a strict evaluator of answer relevancy.
Given the QUESTION and the ANSWER, judge how well the answer addresses the
question that was actually asked (not whether it is well written).
Respond with ONLY a JSON object:
{"relevancy_score": 4, "explanation": "..."}
relevancy_score is an integer 1-5 (1=unrelated, 5=fully addresses the question)."""


def format_context_block(index: int, chunk: ScoredChunk) -> str:
    header_parts = [f"chunk_id={chunk.chunk_id}", f"document={chunk.document_filename}"]
    if chunk.page_number is not None:
        header_parts.append(f"page={chunk.page_number}")
    if chunk.section:
        header_parts.append(f"section={chunk.section}")
    return f"[{index}] ({', '.join(header_parts)})\n{chunk.text}"


def build_answer_prompt(
    question: str,
    context_chunks: list[ScoredChunk],
    history: list[dict] | None = None,
    conflict_hint: bool = False,
) -> str:
    parts: list[str] = []
    if history:
        recent = history[-4:]
        history_lines = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)
        parts.append(f"[CONVERSATION HISTORY — context only, not a source of facts]\n{history_lines}\n")
    blocks = "\n\n".join(format_context_block(i, c) for i, c in enumerate(context_chunks, start=1))
    parts.append(f"[CONTEXT]\n{blocks}")
    parts.append(f"[QUESTION]\n{question}")
    if conflict_hint:
        parts.append(
            "[NOTE] The retrieval stage flagged possible conflicting evidence between context "
            "blocks. Check carefully and report conflicts explicitly if they are real."
        )
    return "\n\n".join(parts)


def build_analysis_prompt(question: str, history: list[dict] | None = None) -> str:
    if history:
        recent = history[-4:]
        history_lines = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)
        return f"[CONVERSATION HISTORY]\n{history_lines}\n\n[QUESTION TO ANALYSE]\n{question}"
    return f"[QUESTION TO ANALYSE]\n{question}"


def build_judge_prompt(question: str, answer: str, context: str | None = None) -> str:
    parts = [f"[QUESTION]\n{question}"]
    if context is not None:
        parts.append(f"[CONTEXT]\n{context}")
    parts.append(f"[ANSWER]\n{answer}")
    return "\n\n".join(parts)
