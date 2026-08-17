from app.pipeline import RAGPipeline


def test_rag_query_contract():
    pipeline = RAGPipeline()
    result = pipeline.query("What is RAG?")

    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["grounded"], bool)
    assert result["grounded"] is True
    assert len(result["sources"]) > 0
