"""Parser utilities: errors, extension mapping, safe filename handling."""

from __future__ import annotations

import re
import unicodedata

from app.core.errors import UnsupportedFileType

_EXTENSION_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
}

# Magic-byte sniffing defends against renamed executables masquerading as docs.
_MAGIC: list[tuple[str, bytes]] = [
    ("pdf", b"%PDF-"),
    ("docx", b"PK\x03\x04"),
]

_FILENAME_STRIP = re.compile(r"[^A-Za-z0-9._\- ]+")


class ParseError(Exception):
    """Raised when a document cannot be parsed (corrupt / empty / wrong type)."""


def detect_doc_type(filename: str, payload: bytes) -> str:
    """Return canonical doc type from extension, validated with magic bytes for PDF/DOCX."""
    name = filename.lower()
    ext = None
    for ext_candidate, doc_type in _EXTENSION_TO_TYPE.items():
        if name.endswith(ext_candidate):
            ext = ext_candidate
            break
    if ext is None:
        raise UnsupportedFileType(
            f"Unsupported file type '{filename}'. Allowed: {', '.join(sorted(set(_EXTENSION_TO_TYPE.values())))}"
        )
    doc_type = _EXTENSION_TO_TYPE[ext]

    head = payload[:512].lstrip()
    for expected_type, magic in _MAGIC:
        if doc_type == expected_type and not head.startswith(magic):
            raise UnsupportedFileType(
                f"File content does not look like a valid {expected_type.upper()} document."
            )
    return doc_type


def safe_filename(filename: str) -> str:
    """Sanitise a client-provided filename: strip paths, control chars, unicode tricks."""
    # Only the final path component is ever used.
    name = filename.replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name)
    name = _FILENAME_STRIP.sub("", name).strip(" .")
    return name[:200] or "uploaded_file"
