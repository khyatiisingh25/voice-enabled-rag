from fastapi.testclient import TestClient

from app.main import app


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