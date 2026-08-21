import time

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import pipeline


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success():
    response = client.post(
        "/query",
        json={"query": "What is machine learning?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "grounded" in data

def test_query_returns_pipeline_contract(monkeypatch):
    expected_response = {
        "answer": "Machine learning is a field of AI.",
        "sources": [
            {
                "title": "test-source",
            }
        ],
        "grounded": True,
    }

    def mock_query(query: str) -> dict:
        return expected_response

    monkeypatch.setattr(pipeline, "query", mock_query)

    response = client.post(
        "/query",
        json={"query": "What is machine learning?"},
    )

    assert response.status_code == 200
    assert response.json() == expected_response


def test_query_empty():
    response = client.post(
        "/query",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_query_missing_field():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


def test_query_too_long():
    response = client.post(
        "/query",
        json={"query": "a" * 2001},
    )

    assert response.status_code == 422


def test_pipeline_timeout(monkeypatch):
    def slow_query(query: str) -> dict:
        time.sleep(6)

        return {
            "answer": "This should not be returned.",
            "sources": [],
            "grounded": False,
        }

    monkeypatch.setattr(pipeline, "query", slow_query)

    response = client.post(
        "/query",
        json={"query": "Test timeout"},
    )

    assert response.status_code == 504
    assert "timeout" in response.json()["detail"].lower()


def test_request_id_header():
    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 36

def test_voice_query_contract(monkeypatch):
    expected_transcript = "What is RAG?"

    def fake_transcribe(audio_path):
        return expected_transcript

    def fake_query(query):
        assert query == expected_transcript
        return {
            "answer": "A Retrieval-Augmented Generation (RAG) system combines document retrieval with a language model.",
            "sources": [{"source": "test-source", "score": 0.9}],
            "grounded": True,
        }

    monkeypatch.setattr("app.main.transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "query", fake_query)

    response = client.post(
        "/voice/query",
        files={
            "audio": (
                "test.wav",
                b"fake-audio-data",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["answer"].startswith("A Retrieval-Augmented")
    assert data["grounded"] is True
    assert len(data["sources"]) == 1
