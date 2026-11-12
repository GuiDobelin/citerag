import httpx
import pytest
from conftest import LISBON, FakeDNS, make_fetcher
from fastapi.testclient import TestClient

from citerag.api import create_app, get_rag


def html(text: str) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text=f"<p>{text}</p>"
    )


@pytest.fixture
def client(rag):
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    return TestClient(app)


def test_ingest_url_returns_document_info_and_is_queryable(client, rag):
    def handler(request):
        if request.url.path == "/lisbon":
            return httpx.Response(301, headers={"location": "/cities/lisbon"})
        return html(LISBON)

    rag.fetcher = make_fetcher(handler)
    r = client.post("/documents/url", json={"url": "https://example.com/lisbon"})

    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "https://example.com/cities/lisbon"
    assert body["chunks"] >= 1
    assert [d["doc_id"] for d in client.get("/documents").json()] == [body["doc_id"]]
    answer = client.post("/query", json={"question": "Where is Web Summit held?"}).json()
    assert answer["citations"][0]["source"] == "https://example.com/cities/lisbon"


@pytest.mark.parametrize(
    ("url", "dns", "status"),
    [
        ("http://localhost:6333/collections", {"localhost": ["127.0.0.1"]}, 400),
        (
            "http://169.254.169.254/latest/meta-data/",
            {"169.254.169.254": ["169.254.169.254"]},
            400,
        ),
        ("ftp://example.com/file.txt", None, 400),
        ("not a url", None, 400),
        ("https://user:pass@example.com/", None, 400),
    ],
)
def test_ingest_url_rejects_blocked_or_invalid_urls(client, rag, url, dns, status):
    rag.fetcher = make_fetcher(lambda r: pytest.fail("no request expected"), FakeDNS(dns))
    r = client.post("/documents/url", json={"url": url})
    assert r.status_code == status
    assert r.json()["detail"]
    assert rag.documents() == []


@pytest.mark.parametrize(
    ("response", "status"),
    [
        (httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"), 415),
        (httpx.Response(404, text="not found"), 502),
        (httpx.Response(503), 502),
        (httpx.Response(200, headers={"content-type": "text/html"}, text="<p> </p>"), 422),
        (
            httpx.Response(
                200, headers={"content-type": "text/plain", "content-length": "99999"}, text="x"
            ),
            413,
        ),
    ],
)
def test_ingest_url_maps_errors_to_status_codes(client, rag, response, status):
    rag.fetcher = make_fetcher(lambda r: response, max_bytes=1000)
    r = client.post("/documents/url", json={"url": "https://example.com/page"})
    assert r.status_code == status
    assert r.json()["detail"]
    assert rag.documents() == []


def test_ingest_url_maps_network_errors_to_502(client, rag):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    rag.fetcher = make_fetcher(handler)
    r = client.post("/documents/url", json={"url": "https://example.com/page"})
    assert r.status_code == 502
    assert "Could not fetch" in r.json()["detail"]


@pytest.mark.parametrize(
    "body", [{}, {"url": ""}, {"url": 42}, {"url": "https://a.io/" + "x" * 2048}]
)
def test_ingest_url_validates_the_body(client, rag, body):
    rag.fetcher = make_fetcher(lambda r: pytest.fail("no request expected"))
    assert client.post("/documents/url", json=body).status_code == 422


def test_openapi_documents_the_endpoint_and_its_errors(client):
    operation = client.get("/openapi.json").json()["paths"]["/documents/url"]["post"]
    assert set(operation["responses"]) == {"201", "400", "413", "415", "422", "502"}
    assert operation["responses"]["201"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DocumentInfo"
    }
    assert "CITERAG_ALLOW_PRIVATE_URLS" in operation["description"]


def test_openapi_schema_references_resolve(client):
    spec = client.get("/openapi.json").json()
    responses = spec["paths"]["/documents/url"]["post"]["responses"]
    refs = [s["$ref"] for s in responses["422"]["content"]["application/json"]["schema"]["anyOf"]]
    refs.append(responses["400"]["content"]["application/json"]["schema"]["$ref"])
    for ref in refs:
        assert ref.removeprefix("#/components/schemas/") in spec["components"]["schemas"]
