from app.pipeline import RAGPipeline


def test_rag_query_contract():
    pipeline = RAGPipeline()

    result = pipeline.query("What is RAG?")

    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["grounded"], bool)
    assert result["grounded"] is True
    assert len(result["sources"]) > 0


def test_rag_rejects_unsupported_query():
    pipeline = RAGPipeline()

    result = pipeline.query("What is the recipe for chocolate cake?")

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "could not find" in result["answer"].lower()


def test_rag_rejects_unrelated_query():
    pipeline = RAGPipeline()

    result = pipeline.query("Who is the President of France?")

    assert result["grounded"] is False
    assert result["sources"] == []
    assert "could not find" in result["answer"].lower()
