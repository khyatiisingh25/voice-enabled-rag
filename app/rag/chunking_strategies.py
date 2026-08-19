import re
from pathlib import Path
from typing import Any


def sentence_chunk(
    text: str,
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Sentence-based chunking.

    Unlike fixed-size chunking, chunks are created using
    sentence boundaries rather than character offsets.
    """

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than 0")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    if overlap_sentences >= sentences_per_chunk:
        raise ValueError(
            "overlap_sentences must be smaller than sentences_per_chunk"
        )

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]

    if not sentences:
        return []

    chunks = []
    step = sentences_per_chunk - overlap_sentences

    for start in range(0, len(sentences), step):
        chunk_sentences = sentences[start:start + sentences_per_chunk]

        if not chunk_sentences:
            break

        chunks.append(" ".join(chunk_sentences))

        if start + sentences_per_chunk >= len(sentences):
            break

    return chunks


def metadata_aware_chunk(
    document: dict[str, Any],
    max_chars: int = 800,
) -> list[dict[str, Any]]:
    """
    Metadata-aware, section-preserving chunking.

    The document is divided into logical sections using blank lines.
    Each chunk preserves document-level metadata and adds section metadata.

    Expected input:
    {
        "text": "...",
        "source": "data/documents/sample.txt",
        "metadata": {
            "title": "...",
            "document_type": "..."
        }
    }
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    text = document.get("text", "").strip()

    if not text:
        return []

    source = document.get("source", "")
    metadata = dict(document.get("metadata", {}))

    sections = [
        section.strip()
        for section in re.split(r"\n\s*\n", text)
        if section.strip()
    ]

    chunks = []

    for section_index, section in enumerate(sections):
        section_metadata = {
            **metadata,
            "source": source,
            "section_index": section_index,
        }

        # Keep logical sections intact whenever possible.
        if len(section) <= max_chars:
            chunks.append(
                {
                    "text": section,
                    "metadata": section_metadata,
                }
            )
            continue

        # If a section is too large, split it by sentences.
        sentence_chunks = sentence_chunk(
            section,
            sentences_per_chunk=3,
            overlap_sentences=0,
        )

        for chunk_index, chunk in enumerate(sentence_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        **section_metadata,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def chunk_with_strategy(
    text: str,
    strategy: str,
    **params: Any,
) -> list[Any]:
    """
    Benchmark-ready interface for the additional strategies.

    Strategies:
    - sentence
    - metadata

    Existing fixed-size chunking is intentionally not modified here.
    """

    if strategy == "sentence":
        return sentence_chunk(text, **params)

    if strategy == "metadata":
        return metadata_aware_chunk(
            {
                "text": text,
                "source": params.pop("source", ""),
                "metadata": params.pop("metadata", {}),
            },
            **params,
        )

    raise ValueError(f"Unknown chunking strategy: {strategy}")