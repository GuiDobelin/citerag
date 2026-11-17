import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import LISBON, HeldExecutor, poll
from fastapi.testclient import TestClient

from citerag.api import close_job_manager, create_app, get_job_manager, get_rag, get_settings
from citerag.config import Settings
from citerag.jobs import JobManager, JobManagerClosedError
from citerag.models import DocumentInfo

DOC = DocumentInfo(doc_id="abc", source="a.txt", chunks=1)
LISBON_BYTES = LISBON.encode()


@pytest.fixture
def executor() -> HeldExecutor:
    return HeldExecutor()


@pytest.fixture
def held(executor) -> JobManager:
    """A job manager whose jobs stay queued until `executor.run_all()`."""
    return JobManager(executor=executor)


def make_client(rag, jobs: JobManager) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_job_manager] = lambda: jobs
    return TestClient(app)


@pytest.fixture
def held_client(rag, held) -> TestClient:
    return make_client(rag, held)


def upload(client, name="lisbon.txt", data=LISBON_BYTES, wait="false"):
    return client.post(
        "/documents", params={"wait": wait}, files={"file": (name, data, "text/plain")}
    )


def test_list_jobs_returns_empty_list(client):
    response = client.get("/jobs")

    assert response.status_code == 200
    assert response.json() == []


def test_list_jobs_returns_newest_first(held_client, held, executor):
    ids = [held.submit("file", f"{n}.txt", lambda: DOC).id for n in range(3)]
    executor.run_all()

    response = held_client.get("/jobs")

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == ids[::-1]


def test_list_jobs_filters_by_status(held_client, held, executor):
    done = held.submit("file", "done.txt", lambda: DOC).id
    executor.run_all()
    queued = [held.submit("file", f"{n}.txt", lambda: DOC).id for n in range(2)]

    response = held_client.get("/jobs", params={"status": "queued"})
    assert [job["id"] for job in response.json()] == queued[::-1]
    response = held_client.get("/jobs", params={"status": "succeeded"})
    assert [job["id"] for job in response.json()] == [done]


def test_list_jobs_rejects_invalid_status(client):
    response = client.get("/jobs", params={"status": "invalid"})

    assert response.status_code == 422


def test_async_upload_returns_202_with_location(held_client, executor):
    r = upload(held_client)

    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "queued"
    assert job["kind"] == "file"
    assert job["source"] == "lisbon.txt"
    assert r.headers["location"] == f"/jobs/{job['id']}"
    assert held_client.get(r.headers["location"]).json()["status"] == "queued"

    executor.run_all()

    done = held_client.get(f"/jobs/{job['id']}").json()
    assert done["status"] == "succeeded"
    assert done["result"]["source"] == "lisbon.txt"
    assert done["result"]["chunks"] >= 1
    assert done["error"] is None


def test_async_upload_is_indexed_in_the_background(client):
    """With real worker threads: poll until done, then the document is queryable."""
    r = upload(client)
    assert r.status_code == 202

    job = poll(client, r.json()["id"])

    assert job["status"] == "succeeded"
    doc_id = job["result"]["doc_id"]
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]
    answer = client.post("/query", json={"question": "Where is Web Summit held?"}).json()
    assert answer["citations"][0]["doc_id"] == doc_id


@pytest.mark.parametrize(
    ("name", "data", "error"),
    [
        ("photo.png", b"\x89PNG", "Unsupported file type"),
        ("empty.txt", b"   ", "No extractable text"),
    ],
)
def test_async_upload_errors_are_reported_in_the_job(client, rag, name, data, error):
    r = upload(client, name, data)
    assert r.status_code == 202

    job = poll(client, r.json()["id"])

    assert job["status"] == "failed"
    assert error in job["error"]
    assert job["result"] is None
    assert rag.documents() == []


def test_async_upload_still_checks_the_size_up_front(client, job_manager):
    client.app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=1)

    r = upload(client, data=b"x" * (1024 * 1024 + 1))

    assert r.status_code == 413
    assert job_manager.list() == []


@pytest.mark.parametrize("params", [{}, {"wait": "true"}])
def test_wait_true_is_the_default_and_stays_synchronous(client, job_manager, params):
    r = client.post(
        "/documents", params=params, files={"file": ("a.txt", b"hello world", "text/plain")}
    )

    assert r.status_code == 201
    assert "location" not in r.headers
    assert job_manager.list() == []


def test_async_upload_returns_503_when_the_queue_is_full(rag, executor):
    client = make_client(rag, JobManager(executor=executor, max_pending=1))

    assert upload(client).status_code == 202
    r = upload(client)

    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert "already queued or running" in r.json()["detail"]


def test_async_upload_returns_503_after_shutdown(held_client, held):
    held.shutdown()

    r = upload(held_client)

    assert r.status_code == 503
    assert r.json()["detail"] == "The server is shutting down."


def test_get_unknown_job_returns_404(client):
    r = client.get("/jobs/does-not-exist")

    assert r.status_code == 404
    assert r.json()["detail"] == "Job not found or expired."


@pytest.fixture
def fresh_manager(monkeypatch):
    """Start with no process-wide job manager and close whatever a test creates."""
    monkeypatch.setattr("citerag.api.get_settings", lambda: Settings(job_workers=1))
    close_job_manager()
    yield
    close_job_manager()


def test_lifespan_shuts_down_the_job_manager(fresh_manager):
    with TestClient(create_app()):
        jobs = get_job_manager()
        assert get_job_manager() is jobs

    with pytest.raises(JobManagerClosedError):
        jobs.submit("file", "a.txt", lambda: DOC)
    assert get_job_manager() is not jobs


def test_lifespan_does_not_create_a_job_manager(fresh_manager, monkeypatch):
    def fail(settings):
        pytest.fail("no job manager should be created")

    monkeypatch.setattr(JobManager, "from_settings", fail)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_concurrent_first_calls_share_one_job_manager(fresh_manager, monkeypatch):
    real = JobManager.from_settings
    created = []

    def slow(settings):
        time.sleep(0.02)  # widen the window in which a racing caller could build another one
        created.append(real(settings))
        return created[-1]

    monkeypatch.setattr(JobManager, "from_settings", slow)
    with ThreadPoolExecutor(8) as pool:
        managers = list(pool.map(lambda _: get_job_manager(), range(8)))

    assert len(created) == 1
    assert all(m is created[0] for m in managers)


def test_location_respects_root_path(rag, held):
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_job_manager] = lambda: held
    r = upload(TestClient(app, root_path="/citerag"))

    assert r.headers["location"] == f"/citerag/jobs/{r.json()['id']}"


def test_openapi_documents_async_responses(client):
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/documents"]["post"]
    assert {"201", "202", "503"} <= set(operation["responses"])
    accepted = operation["responses"]["202"]
    assert accepted["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/Job"}
    assert "Location" in accepted["headers"]
    assert any(p["name"] == "wait" for p in operation["parameters"])
    assert set(spec["paths"]["/jobs/{job_id}"]["get"]["responses"]) >= {"200", "404"}
