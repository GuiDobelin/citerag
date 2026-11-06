import pytest
from conftest import LISBON
from fastapi.testclient import TestClient

from citerag.api import create_app, get_rag


@pytest.fixture
def client(rag):
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_query_list_delete(client):
    r = client.post("/documents", files={"file": ("lisbon.txt", LISBON.encode(), "text/plain")})
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]

    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]

    r = client.post("/query", json={"question": "Where is Web Summit held?"})
    assert r.status_code == 200
    assert r.json()["citations"][0]["doc_id"] == doc_id

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_upload_rejects_unsupported_type(client):
    r = client.post("/documents", files={"file": ("photo.png", b"\\x89PNG", "image/png")})
    assert r.status_code == 415


def test_query_validates_input(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_accepts_per_request_mode_and_rejects_invalid(client):
    uploaded = client.post(
        "/documents", files={"file": ("lisbon.txt", LISBON.encode(), "text/plain")}
    )
    assert uploaded.status_code == 201

    response = client.post(
        "/query", json={"question": "Where is Web Summit held?", "mode": "keyword"}
    )
    assert response.status_code == 200
    assert response.json()["citations"][0]["source"] == "lisbon.txt"
    assert client.app.dependency_overrides[get_rag]().retrieval_mode == "hybrid"

    invalid = client.post(
        "/query", json={"question": "Where is Web Summit held?", "mode": "semantic"}
    )
    assert invalid.status_code == 422


def test_query_stream_accepts_per_request_mode_and_rejects_invalid(client):
    uploaded = client.post(
        "/documents", files={"file": ("lisbon.txt", LISBON.encode(), "text/plain")}
    )
    assert uploaded.status_code == 201

    response = client.post(
        "/query/stream",
        json={"question": "Where is Web Summit held?", "mode": "keyword"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
    assert '"source": "lisbon.txt"' in response.text
    assert client.app.dependency_overrides[get_rag]().retrieval_mode == "hybrid"

    invalid = client.post(
        "/query/stream",
        json={"question": "Where is Web Summit held?", "mode": "semantic"},
    )
    assert invalid.status_code == 422
