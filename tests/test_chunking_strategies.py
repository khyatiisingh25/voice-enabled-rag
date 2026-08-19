from app.rag.chunking_strategies import (
    chunk_with_strategy,
    metadata_aware_chunk,
    sentence_chunk,
)


def test_sentence_chunking():
    text = (
        "Sentence one. "
        "Sentence two. "
        "Sentence three. "
        "Sentence four."
    )

    chunks = sentence_chunk(
        text,
        sentences_per_chunk=2,
        overlap_sentences=1,
    )

    assert len(chunks) == 3
    assert chunks[0] == "Sentence one. Sentence two."
    assert chunks[1] == "Sentence two. Sentence three."


def test_metadata_aware_chunking():
    document = {
        "text": (
            "Document Title\n\n"
            "First section contains information.\n\n"
            "Second section contains more information."
        ),
        "source": "data/documents/sample.txt",
        "metadata": {
            "document_type": "text",
        },
    }

    chunks = metadata_aware_chunk(document, max_chars=800)

    assert len(chunks) == 3

    assert chunks[0]["metadata"]["source"] == (
        "data/documents/sample.txt"
    )

    assert chunks[0]["metadata"]["document_type"] == "text"

    assert chunks[0]["metadata"]["section_index"] == 0


def test_benchmark_interface_sentence():
    chunks = chunk_with_strategy(
        "One. Two. Three.",
        strategy="sentence",
        sentences_per_chunk=2,
        overlap_sentences=0,
    )

    assert chunks == ["One. Two.", "Three."]


def test_benchmark_interface_metadata():
    chunks = chunk_with_strategy(
        "Title\n\nBody text.",
        strategy="metadata",
        source="sample.txt",
        metadata={"document_type": "text"},
    )

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["source"] == "sample.txt"