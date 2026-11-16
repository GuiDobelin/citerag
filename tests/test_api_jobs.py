from datetime import UTC, datetime

from fastapi.testclient import TestClient

from citerag.api import create_app, get_job_manager
from citerag.jobs import Job, JobManager


def make_client(job_manager: JobManager) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_job_manager] = lambda: job_manager
    return TestClient(app)


def add_job(job_manager: JobManager, job_id: str, status: str) -> None:
    job_manager._jobs[job_id] = Job(
        id=job_id,
        kind="file",
        source=f"{job_id}.txt",
        status=status,
        created_at=datetime.now(UTC),
    )


def test_list_jobs_returns_empty_list(job_manager):
    client = make_client(job_manager)

    response = client.get("/jobs")

    assert response.status_code == 200
    assert response.json() == []


def test_list_jobs_returns_newest_first(job_manager):
    add_job(job_manager, "job-1", "queued")
    add_job(job_manager, "job-2", "running")
    add_job(job_manager, "job-3", "succeeded")

    client = make_client(job_manager)
    response = client.get("/jobs")

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == [
        "job-3",
        "job-2",
        "job-1",
    ]


def test_list_jobs_filters_by_status(job_manager):
    add_job(job_manager, "job-1", "queued")
    add_job(job_manager, "job-2", "running")
    add_job(job_manager, "job-3", "queued")

    client = make_client(job_manager)
    response = client.get("/jobs", params={"status": "queued"})

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == [
        "job-3",
        "job-1",
    ]


def test_list_jobs_rejects_invalid_status(job_manager):
    client = make_client(job_manager)

    response = client.get("/jobs", params={"status": "invalid"})

    assert response.status_code == 422
