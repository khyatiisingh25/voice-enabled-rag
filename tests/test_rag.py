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

def test_gemini_retries_before_success(monkeypatch):
    from app.rag import generator

    attempts = {"count": 0}

    class FakeResponse:
        text = "RAG is Retrieval-Augmented Generation."

    class FakeModels:
        def generate_content(self, model, contents):
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise RuntimeError("temporary Gemini failure")

            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(generator.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(generator.time, "sleep", lambda _: None)

    result = generator.generate_answer(
        "What is RAG?",
        [
            {
                "text": "RAG stands for Retrieval-Augmented Generation.",
                "source": "test-source",
                "score": 0.9,
            }
        ],
    )

    assert attempts["count"] == 3
    assert result["answer"] == "RAG is Retrieval-Augmented Generation."
    assert result["grounded"] is True
