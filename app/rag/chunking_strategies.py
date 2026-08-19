import re
from typing import Any

from app.rag.chunker import chunk_text


FIXED_STRATEGIES = {
    "500/50": {
        "chunk_size": 500,
        "overlap": 50,
    },
    "300/50": {
        "chunk_size": 300,
        "overlap": 50,
    },
    "800/100": {
        "chunk_size": 800,
        "overlap": 100,
    },
}


def sentence_chunk(
    text: str,
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Sentence-boundary-based chunking.

    Chunks are created using sentence boundaries rather than
    fixed character offsets.
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

    Logical sections are preserved whenever possible.
    Each returned chunk contains text and metadata.
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

        if len(section) <= max_chars:
            chunks.append(
                {
                    "text": section,
                    "metadata": section_metadata,
                }
            )
            continue

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


def fixed_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Wrapper around the existing fixed-size chunker.

    The original chunk_text implementation is intentionally
    unchanged.
    """

    return chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _normalize_text_chunks(
    chunks: list[str],
    strategy: str,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert plain text chunks into the common benchmark format.
    """

    base_metadata = dict(metadata or {})
    base_metadata["source"] = source
    base_metadata["strategy"] = strategy

    return [
        {
            "text": chunk,
            "metadata": {
                **base_metadata,
                "chunk_index": index,
            },
        }
        for index, chunk in enumerate(chunks)
    ]


def chunk_with_strategy(
    text: str,
    strategy: str,
    **params: Any,
) -> list[dict[str, Any]]:
    """
    Common benchmark-ready interface for all five strategies.

    Supported strategies:

    - 500/50
    - 300/50
    - 800/100
    - sentence
    - metadata

    Every strategy returns:

    [
        {
            "text": "...",
            "metadata": {...}
        }
    ]

    The existing fixed-size chunk_text() implementation is
    not modified.
    """

    source = params.pop("source", "")
    metadata = params.pop("metadata", {})

    if strategy in FIXED_STRATEGIES:
        config = FIXED_STRATEGIES[strategy]

        chunks = fixed_chunk(
            text,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
        )

        return _normalize_text_chunks(
            chunks,
            strategy=strategy,
            source=source,
            metadata=metadata,
        )

    if strategy == "sentence":
        chunks = sentence_chunk(text, **params)

        return _normalize_text_chunks(
            chunks,
            strategy=strategy,
            source=source,
            metadata=metadata,
        )

    if strategy == "metadata":
        chunks = metadata_aware_chunk(
            {
                "text": text,
                "source": source,
                "metadata": metadata,
            },
            **params,
        )

        # Metadata-aware chunks already contain metadata.
        # Add the strategy name so the common benchmark
        # interface has the same metadata contract.
        for chunk in chunks:
            chunk["metadata"]["strategy"] = strategy

        return chunks

    raise ValueError(f"Unknown chunking strategy: {strategy}")