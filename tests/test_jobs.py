import threading
from concurrent.futures import Executor, Future
from datetime import UTC, datetime, timedelta

import pytest
from conftest import LISBON

from citerag.config import Settings
from citerag.jobs import INTERNAL_ERROR, SHUTDOWN_ERROR, JobManager, QueueFullError
from citerag.loaders import UnsupportedFileError
from citerag.models import DocumentInfo

DOC = DocumentInfo(doc_id="abc", source="lisbon.txt", chunks=2)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class InlineExecutor(Executor):
    """Runs each job as soon as it is submitted, so tests need no threads."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class HeldExecutor(Executor):
    """Keeps submitted jobs until `run_all`, to observe the `queued` state."""

    def __init__(self) -> None:
        self.pending: list = []

    def submit(self, fn, /, *args, **kwargs):
        self.pending.append((fn, args))
        return Future()

    def run_all(self) -> None:
        while self.pending:
            fn, args = self.pending.pop(0)
            fn(*args)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if cancel_futures:
            self.pending.clear()


def manager(executor=None, clock=None, **kwargs) -> JobManager:
    return JobManager(executor=executor or InlineExecutor(), clock=clock or Clock(), **kwargs)


def test_successful_job_records_result_and_timestamps():
    clock = Clock()
    jobs = manager(clock=clock)
    job = jobs.submit("file", "lisbon.txt", lambda: DOC)

    done = jobs.get(job.id)
    assert done.status == "succeeded"
    assert done.result == DOC
    assert done.error is None
    assert done.kind == "file" and done.source == "lisbon.txt"
    assert done.created_at == done.started_at == done.finished_at == clock.now


def test_submit_returns_a_queued_job_until_it_runs():
    executor = HeldExecutor()
    jobs = manager(executor)
    job = jobs.submit("url", "https://example.com", lambda: DOC)

    assert job.status == "queued"
    assert jobs.get(job.id).status == "queued"
    executor.run_all()
    assert jobs.get(job.id).status == "succeeded"
    # The snapshot returned by submit didn't change under the caller.
    assert job.status == "queued"


def test_snapshots_cannot_modify_the_stored_job():
    jobs = manager()
    job = jobs.submit("file", "lisbon.txt", lambda: DOC)
    snapshot = jobs.get(job.id)
    snapshot.status = "failed"
    snapshot.result.chunks = 99
    assert jobs.get(job.id).status == "succeeded"
    assert jobs.get(job.id).result.chunks == 2


def test_value_errors_become_the_job_error():
    def fail():
        raise UnsupportedFileError("Unsupported file type '.png'.")

    jobs = manager()
    job = jobs.get(jobs.submit("file", "photo.png", fail).id)
    assert job.status == "failed"
    assert job.error == "Unsupported file type '.png'."
    assert job.result is None
    assert job.finished_at is not None


def test_unexpected_errors_are_logged_but_not_exposed(caplog):
    def crash():
        raise RuntimeError("password=hunter2 in a stack trace")

    jobs = manager()
    job = jobs.get(jobs.submit("file", "x.txt", crash).id)
    assert job.status == "failed"
    assert job.error == INTERNAL_ERROR
    assert "hunter2" in caplog.text


def test_running_status_is_visible_while_the_job_runs():
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return DOC

    jobs = JobManager(workers=1)
    try:
        job = jobs.submit("file", "big.pdf", slow)
        assert started.wait(5)
        running = jobs.get(job.id)
        assert running.status == "running"
        assert running.started_at is not None and running.finished_at is None
    finally:
        release.set()
        jobs.shutdown()
    assert jobs.get(job.id).status == "succeeded"


def test_finished_jobs_expire_after_ttl():
    clock = Clock()
    jobs = manager(clock=clock, ttl=60)
    job = jobs.submit("file", "lisbon.txt", lambda: DOC)

    clock.advance(59)
    assert jobs.get(job.id) is not None
    clock.advance(1)
    assert jobs.get(job.id) is None
    assert jobs.list() == []


def test_unfinished_jobs_never_expire():
    clock = Clock()
    executor = HeldExecutor()
    jobs = manager(executor, clock=clock, ttl=60)
    job = jobs.submit("file", "lisbon.txt", lambda: DOC)
    clock.advance(3600)
    assert jobs.get(job.id).status == "queued"


def test_oldest_finished_jobs_are_evicted_beyond_max_jobs():
    clock = Clock()
    jobs = manager(clock=clock, max_jobs=2)
    first = jobs.submit("file", "a.txt", lambda: DOC)
    clock.advance(1)
    second = jobs.submit("file", "b.txt", lambda: DOC)
    clock.advance(1)
    third = jobs.submit("file", "c.txt", lambda: DOC)

    assert jobs.get(first.id) is None
    assert [job.id for job in jobs.list()] == [third.id, second.id]


def test_submit_fails_when_too_many_jobs_are_pending():
    executor = HeldExecutor()
    jobs = manager(executor, max_pending=2)
    jobs.submit("file", "a.txt", lambda: DOC)
    jobs.submit("file", "b.txt", lambda: DOC)
    with pytest.raises(QueueFullError):
        jobs.submit("file", "c.txt", lambda: DOC)
    assert len(jobs.list()) == 2

    executor.run_all()
    jobs.submit("file", "c.txt", lambda: DOC)


def test_submit_fails_when_max_jobs_are_all_unfinished():
    jobs = manager(HeldExecutor(), max_jobs=1, max_pending=5)
    jobs.submit("file", "a.txt", lambda: DOC)
    with pytest.raises(QueueFullError):
        jobs.submit("file", "b.txt", lambda: DOC)


def test_list_is_newest_first_and_filters_by_status():
    executor = HeldExecutor()
    jobs = manager(executor)
    done = jobs.submit("file", "a.txt", lambda: DOC)
    executor.run_all()
    queued = jobs.submit("file", "b.txt", lambda: DOC)

    assert [job.id for job in jobs.list()] == [queued.id, done.id]
    assert [job.id for job in jobs.list("succeeded")] == [done.id]
    assert [job.id for job in jobs.list("queued")] == [queued.id]
    assert jobs.list("failed") == []


def test_shutdown_fails_queued_jobs_and_refuses_new_ones():
    executor = HeldExecutor()
    jobs = manager(executor)
    job = jobs.submit("file", "a.txt", lambda: DOC)

    jobs.shutdown()
    stopped = jobs.get(job.id)
    assert stopped.status == "failed"
    assert stopped.error == SHUTDOWN_ERROR
    with pytest.raises(RuntimeError):
        jobs.submit("file", "b.txt", lambda: DOC)


def test_a_job_failed_by_shutdown_does_not_run_later():
    executor = HeldExecutor()
    jobs = manager(executor)
    calls = []
    job = jobs.submit("file", "a.txt", lambda: calls.append(1) or DOC)
    pending = list(executor.pending)
    jobs.shutdown()
    for fn, args in pending:
        fn(*args)
    assert calls == []
    assert jobs.get(job.id).status == "failed"


def test_executor_errors_on_submit_leave_no_orphan_job():
    class Broken(Executor):
        def submit(self, fn, /, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures")

    jobs = manager(Broken())
    with pytest.raises(RuntimeError):
        jobs.submit("file", "a.txt", lambda: DOC)
    assert jobs.list() == []


def test_jobs_run_real_ingestion(rag):
    jobs = JobManager(workers=1)
    job = jobs.submit("file", "lisbon.txt", lambda: rag.ingest("lisbon.txt", LISBON.encode()))
    jobs.shutdown()
    done = jobs.get(job.id)
    assert done.status == "succeeded"
    assert [d.doc_id for d in rag.documents()] == [done.result.doc_id]


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": 0}, {"ttl": 0}, {"max_jobs": 0}, {"max_pending": 0}],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        JobManager(**kwargs)


def test_from_settings(monkeypatch):
    monkeypatch.setenv("CITERAG_MAX_PENDING_JOBS", "3")
    monkeypatch.setenv("CITERAG_JOB_TTL", "120")
    jobs = JobManager.from_settings(Settings(_env_file=None))
    try:
        assert jobs.max_pending == 3
        assert jobs.ttl == timedelta(seconds=120)
        assert jobs.max_jobs == 1000
    finally:
        jobs.shutdown()
