from app.rag.chunking_strategies import (
    chunk_with_strategy,
    metadata_aware_chunk,
    sentence_chunk,
)


def test_fixed_500_50_strategy():
    text = "a" * 1200

    chunks = chunk_with_strategy(
        text,
        strategy="500/50",
    )

    assert len(chunks) > 0
    assert chunks[0]["text"] == "a" * 500
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_fixed_300_50_strategy():
    text = "a" * 1000

    chunks = chunk_with_strategy(
        text,
        strategy="300/50",
    )

    assert len(chunks) > 0
    assert chunks[0]["text"] == "a" * 300
    assert all(isinstance(chunk, dict) for chunk in chunks)


def test_fixed_800_100_strategy():
    text = "a" * 1600

    chunks = chunk_with_strategy(
        text,
        strategy="800/100",
    )

    assert len(chunks) > 0
    assert chunks[0]["text"] == "a" * 800
    assert all(isinstance(chunk, dict) for chunk in chunks)


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

    chunks = metadata_aware_chunk(
        document,
        max_chars=800,
    )

    assert len(chunks) == 3

    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)

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
        source="sample.txt",
    )

    assert chunks == [
        {
            "text": "One. Two.",
            "metadata": {
                "source": "sample.txt",
                "strategy": "sentence",
                "chunk_index": 0,
            },
        },
        {
            "text": "Three.",
            "metadata": {
                "source": "sample.txt",
                "strategy": "sentence",
                "chunk_index": 1,
            },
        },
    ]


def test_benchmark_interface_metadata():
    chunks = chunk_with_strategy(
        "Title\n\nBody text.",
        strategy="metadata",
        source="sample.txt",
        metadata={"document_type": "text"},
    )

    assert len(chunks) == 2
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    assert chunks[0]["metadata"]["source"] == "sample.txt"


def test_all_five_strategies_use_common_interface():
    text = (
        "This is sentence one. "
        "This is sentence two. "
        "This is sentence three. "
        "This is sentence four."
    )

    strategies = [
        ("500/50", {}),
        ("300/50", {}),
        ("800/100", {}),
        (
            "sentence",
            {
                "sentences_per_chunk": 2,
                "overlap_sentences": 0,
            },
        ),
        (
            "metadata",
            {
                "source": "sample.txt",
                "metadata": {"document_type": "text"},
            },
        ),
    ]

    for strategy, params in strategies:
        chunks = chunk_with_strategy(
            text,
            strategy=strategy,
            **params,
        )

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["metadata"], dict)
            assert chunk["metadata"]["strategy"] == strategy