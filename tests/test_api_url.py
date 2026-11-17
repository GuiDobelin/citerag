import httpx
import pytest
from conftest import LISBON, FakeDNS, html, make_fetcher, poll


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
    assert set(operation["responses"]) == {"201", "202", "400", "413", "415", "422", "502", "503"}
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


def test_ingest_url_logs_upstream_failures(client, rag, caplog):
    rag.fetcher = make_fetcher(lambda r: httpx.Response(503))
    with caplog.at_level("WARNING", logger="citerag.api"):
        r = client.post("/documents/url", json={"url": "https://example.com/page"})
    assert r.status_code == 502
    assert "https://example.com/page" in caplog.text
    assert "HTTP 503" in caplog.text


def test_async_ingest_url_returns_202_and_the_job_succeeds(client, rag):
    def handler(request):
        if request.url.path == "/lisbon":
            return httpx.Response(302, headers={"location": "/cities/lisbon"})
        return html(LISBON)

    rag.fetcher = make_fetcher(handler)
    r = client.post("/documents/url", json={"url": " https://example.com/lisbon ", "wait": False})

    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "url"
    assert job["source"] == "https://example.com/lisbon"
    assert r.headers["location"] == f"/jobs/{job['id']}"
    done = poll(client, job["id"])
    assert done["status"] == "succeeded"
    assert done["result"]["source"] == "https://example.com/cities/lisbon"
    assert [d.doc_id for d in rag.documents()] == [done["result"]["doc_id"]]


@pytest.mark.parametrize(
    "url", ["ftp://example.com/file.txt", "not a url", "https://user:pass@example.com/"]
)
def test_async_ingest_url_rejects_invalid_urls_synchronously(client, rag, job_manager, url):
    rag.fetcher = make_fetcher(lambda r: pytest.fail("no request expected"))
    sync = client.post("/documents/url", json={"url": url})
    r = client.post("/documents/url", json={"url": url, "wait": False})

    assert r.status_code == 400
    assert r.json()["detail"] == sync.json()["detail"]
    assert job_manager.list() == []


def test_async_ingest_url_validates_the_body(client):
    r = client.post("/documents/url", json={"url": "https://example.com/", "wait": "later"})
    assert r.status_code == 422


def _network_error(request):
    raise httpx.ConnectError("connection refused", request=request)


def _timeout(request):
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize(
    ("handler", "dns", "sync_status"),
    [
        (lambda r: pytest.fail("no request expected"), {"internal.example": ["10.0.0.5"]}, 400),
        (
            lambda r: httpx.Response(
                200, headers={"content-type": "text/plain", "content-length": "99999"}, text="x"
            ),
            None,
            413,
        ),
        (
            lambda r: httpx.Response(200, headers={"content-type": "image/png"}, content=b"x"),
            None,
            415,
        ),
        (lambda r: httpx.Response(200, headers={"content-type": "text/html"}, text=" "), None, 422),
        (lambda r: httpx.Response(404, text="not found"), None, 502),
        (lambda r: httpx.Response(503), None, 502),
        (_network_error, None, 502),
        (_timeout, None, 502),
    ],
    ids=["blocked", "too-large", "unsupported", "no-text", "404", "503", "network", "timeout"],
)
def test_async_ingest_url_failures_match_the_sync_error(client, rag, handler, dns, sync_status):
    """Every error the synchronous endpoint maps to a status ends as a failed job with the
    same message."""
    rag.fetcher = make_fetcher(handler, FakeDNS(dns), max_bytes=1000)
    url = "https://internal.example/page"
    sync = client.post("/documents/url", json={"url": url})
    assert sync.status_code == sync_status

    r = client.post("/documents/url", json={"url": url, "wait": False})

    assert r.status_code == 202
    job = poll(client, r.json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == sync.json()["detail"]
    assert job["result"] is None
    assert rag.documents() == []


def test_async_ingest_url_logs_upstream_failures(client, rag, caplog):
    rag.fetcher = make_fetcher(lambda r: httpx.Response(503))
    with caplog.at_level("WARNING", logger="citerag.api"):
        r = client.post("/documents/url", json={"url": "https://example.com/page", "wait": False})
        poll(client, r.json()["id"])
    assert "https://example.com/page" in caplog.text
    assert "HTTP 503" in caplog.text
