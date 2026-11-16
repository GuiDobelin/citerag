"""Background ingestion jobs.

Jobs live in the memory of the process that created them: they are lost on restart and each
replica only knows its own jobs. Finished jobs are forgotten after `ttl` seconds, or earlier,
oldest first, once more than `max_jobs` are retained.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from citerag.config import Settings
from citerag.models import DocumentInfo

logger = logging.getLogger(__name__)

JobKind = Literal["file", "url"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
INTERNAL_ERROR = "Internal error while ingesting the document."
SHUTDOWN_ERROR = "The server shut down before the job started."


class Job(BaseModel):
    id: str
    kind: JobKind
    source: str
    status: JobStatus = "queued"
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: DocumentInfo | None = None
    error: str | None = None


class QueueFullError(RuntimeError):
    """Too many jobs are queued or running to accept another one."""


class JobManagerClosedError(RuntimeError):
    """The job manager has been shut down and accepts no new jobs."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobManager:
    """Run ingestion callables on a bounded thread pool and track their status.

    `submit` returns at once with a queued `Job`; `get` and `list` return snapshots, so callers
    never see a job change under them. A callable's `ValueError` message (unsupported file, no
    text, URL errors) becomes the job's `error`, like the synchronous API returns it as a 422;
    any other exception is logged with its traceback and reported as a generic error, so
    internals don't leak to API clients.

    `max_pending` caps queued plus running jobs, because each one holds its input (an upload
    of up to `CITERAG_MAX_UPLOAD_MB`) in memory until it runs.

    Timestamps on jobs come from `clock` (UTC wall time); expiry uses `monotonic`, so a change
    of the system clock doesn't expire jobs early or keep them forever. An `executor` passed
    in is used but not shut down by `shutdown`; its owner does that.
    """

    def __init__(
        self,
        workers: int = 2,
        ttl: float = 3600.0,
        max_jobs: int = 1000,
        max_pending: int = 20,
        *,
        executor: Executor | None = None,
        clock: Callable[[], datetime] = _utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1.")
        if ttl <= 0:
            raise ValueError("ttl must be > 0.")
        if max_jobs < 1 or max_pending < 1:
            raise ValueError("max_jobs and max_pending must be >= 1.")
        self.ttl = ttl
        self.max_jobs = max_jobs
        self.max_pending = max_pending
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="citerag-job"
        )
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        # Insertion order is creation order, which `list` relies on.
        self._jobs: dict[str, Job] = {}
        # Monotonic finish time of each finished job, oldest first, for expiry and eviction.
        self._finished: dict[str, float] = {}
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "JobManager":
        s = settings or Settings()
        return cls(s.job_workers, s.job_ttl, s.max_jobs, s.max_pending_jobs)

    def submit(self, kind: JobKind, source: str, fn: Callable[[], DocumentInfo]) -> Job:
        """Queue `fn` to run in the background and return the new job.

        Raises `QueueFullError` when `max_pending` jobs are unfinished, or when `max_jobs` are
        retained and none of them is finished, and `JobManagerClosedError` after `shutdown`.
        """
        with self._lock:
            if self._closed:
                raise JobManagerClosedError("The job manager has been shut down.")
            self._purge_expired()
            pending = len(self._jobs) - len(self._finished)
            if pending >= self.max_pending:
                raise QueueFullError(
                    f"{pending} ingestion jobs are already queued or running; try again later."
                )
            # Evict the oldest finished jobs to make room; dicts iterate in insertion order.
            for job_id in list(self._finished)[: len(self._jobs) - self.max_jobs + 1]:
                self._forget(job_id)
            if len(self._jobs) >= self.max_jobs:
                raise QueueFullError("Too many ingestion jobs are retained; try again later.")
            job = Job(id=uuid.uuid4().hex, kind=kind, source=source, created_at=self._clock())
            self._jobs[job.id] = job
            snapshot = job.model_copy(deep=True)
        # Submitting outside the lock lets an executor run `fn` inline (as tests do).
        try:
            self._executor.submit(self._run, job.id, fn)
        except BaseException:
            with self._lock:
                self._jobs.pop(job.id, None)
            raise
        return snapshot

    def get(self, job_id: str) -> Job | None:
        """A snapshot of the job, or None if it is unknown or has expired."""
        with self._lock:
            self._purge_expired()
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list(self, status: JobStatus | None = None) -> list[Job]:
        """Snapshots of the retained jobs, newest first, optionally only those in `status`."""
        with self._lock:
            self._purge_expired()
            return [
                job.model_copy(deep=True)
                for job in reversed(self._jobs.values())
                if status is None or job.status == status
            ]

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting jobs and mark the ones that haven't started as failed. With `wait`,
        block until running jobs finish."""
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.status == "queued":
                    self._finish(job.id, error=SHUTDOWN_ERROR)
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, job_id: str, fn: Callable[[], DocumentInfo]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            # Already failed by `shutdown`: don't start it.
            if job is None or job.status != "queued":
                return
            job.status = "running"
            job.started_at = self._clock()
        try:
            result = fn()
        except ValueError as exc:
            self._complete(job_id, error=str(exc) or INTERNAL_ERROR)
        except Exception:
            logger.exception("Ingestion job %s failed", job_id)
            self._complete(job_id, error=INTERNAL_ERROR)
        except BaseException:
            # SystemExit and the like: don't leave the job running forever, then propagate.
            self._complete(job_id, error=INTERNAL_ERROR)
            raise
        else:
            self._complete(job_id, result=result)

    def _complete(
        self, job_id: str, result: DocumentInfo | None = None, error: str | None = None
    ) -> None:
        with self._lock:
            self._finish(job_id, result, error)

    def _finish(
        self, job_id: str, result: DocumentInfo | None = None, error: str | None = None
    ) -> None:
        """Record the outcome of a job. The caller holds the lock."""
        job = self._jobs.get(job_id)
        if job is None or job_id in self._finished:
            return
        job.status = "failed" if error is not None else "succeeded"
        job.result = result
        job.error = error
        job.finished_at = self._clock()
        self._finished[job_id] = self._monotonic()

    def _forget(self, job_id: str) -> None:
        del self._jobs[job_id]
        del self._finished[job_id]

    def _purge_expired(self) -> None:
        cutoff = self._monotonic() - self.ttl
        expired = [job_id for job_id, at in self._finished.items() if at <= cutoff]
        for job_id in expired:
            self._forget(job_id)
