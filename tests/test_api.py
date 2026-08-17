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