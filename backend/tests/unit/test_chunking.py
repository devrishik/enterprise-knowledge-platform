"""Unit tests: structure-aware chunking."""

from __future__ import annotations

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.chunking import FixedSizeChunker, StructureAwareChunker


def make_md_doc() -> ParsedDocument:
    return ParsedDocument(
        filename="policy.md",
        doc_type="md",
        blocks=[
            ParsedBlock("Leave Policy", kind="heading", section="Leave Policy"),
            ParsedBlock("Employees receive 22 days of paid annual leave per year.", section="Leave Policy"),
            ParsedBlock("Unused leave can be carried over up to five days.", section="Leave Policy"),
            ParsedBlock("Sick Leave", kind="heading", section="Sick Leave"),
            ParsedBlock("A medical certificate is required after three days.", section="Sick Leave"),
        ],
    )


def test_sections_preserved_and_heading_prefixed():
    chunker = StructureAwareChunker(chunk_size=1200, chunk_overlap=0)
    chunks = chunker.chunk(make_md_doc())
    assert len(chunks) == 2
    assert chunks[0].section == "Leave Policy"
    assert chunks[0].text.startswith("Leave Policy")  # heading prefix for context
    assert "22 days" in chunks[0].text
    assert chunks[1].section == "Sick Leave"
    assert "medical certificate" in chunks[1].text


def test_small_blocks_merge_within_chunk_size():
    chunker = StructureAwareChunker(chunk_size=5000, chunk_overlap=0)
    chunks = chunker.chunk(make_md_doc())
    # two sections -> two chunks (sections never merge with each other)
    assert len(chunks) == 2
    assert "22 days" in chunks[0].text and "medical certificate" not in chunks[0].text


def test_oversized_blocks_split_with_overlap():
    para = ("Sentence about remote work policy. " * 40).strip()
    parsed = ParsedDocument(
        filename="big.md",
        doc_type="md",
        blocks=[ParsedBlock("Remote Work", kind="heading", section="Remote Work"), ParsedBlock(para, section="Remote Work")],
    )
    chunker = StructureAwareChunker(chunk_size=400, chunk_overlap=100)
    chunks = chunker.chunk(parsed)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 400
        assert chunk.section == "Remote Work"
    # overlap present between consecutive pieces
    assert chunks[0].text[-100:].strip()[:20] in chunks[1].text


def test_pdf_pages_never_merge():
    parsed = ParsedDocument(
        filename="doc.pdf",
        doc_type="pdf",
        blocks=[
            ParsedBlock("Page one content about expenses.", page=1),
            ParsedBlock("Page two content about security.", page=2),
        ],
    )
    chunker = StructureAwareChunker(chunk_size=5000, chunk_overlap=0)
    chunks = chunker.chunk(parsed)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1 and chunks[1].page_number == 2
    assert "expenses" in chunks[0].text and "security" in chunks[1].text


def test_chunk_indices_sequential():
    chunker = StructureAwareChunker(chunk_size=300, chunk_overlap=50)
    parsed = ParsedDocument(
        filename="x.md",
        doc_type="md",
        blocks=[ParsedBlock(f"Heading {i}", kind="heading", section=f"Heading {i}")
                for i in range(5)]
        + [ParsedBlock("Body text. " * 30, section="Heading 4")],
    )
    chunks = chunker.chunk(parsed)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_token_count_approximation():
    chunker = StructureAwareChunker(chunk_size=5000, chunk_overlap=0)
    chunks = chunker.chunk(make_md_doc())
    assert all(c.token_count == len(c.text.split()) for c in chunks)


def test_fixed_strategy_is_naive_split():
    parsed = make_md_doc()
    chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=0)
    chunks = chunker.chunk(parsed)
    assert len(chunks) >= 2  # whole-document linear split, ignores structure
    assert all(c.section is None for c in chunks)
    assert all(c.metadata["strategy"] == "fixed" for c in chunks)


def test_empty_document_yields_no_chunks():
    parsed = ParsedDocument(filename="e.md", doc_type="md", blocks=[])
    chunker = StructureAwareChunker()
    assert chunker.chunk(parsed) == []
