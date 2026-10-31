"""Background job runner. One worker: inference is GPU/unified-memory bound, so
running reconstructions serially is faster and safer than in parallel."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error | cancelled
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    log: list[dict] = field(default_factory=list)
    scene_id: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cancel_requested: bool = False

    def public(self) -> dict:
        d = asdict(self)
        d.pop("cancel_requested")
        return d


class Cancelled(Exception):
    pass


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veronica-job")

    def submit(self, kind: str, fn: Callable[[Callable[[str, float, str], None]], str], scene_id: str | None = None) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, scene_id=scene_id)
        with self._lock:
            self._jobs[job.id] = job

        def progress(stage: str, frac: float, msg: str = "") -> None:
            if job.cancel_requested:
                raise Cancelled()
            job.stage, job.progress, job.message = stage, max(job.progress, frac), msg
            job.log.append({"t": round(time.time() - job.created, 2), "stage": stage, "msg": msg})

        def run() -> None:
            job.status = "running"
            try:
                job.scene_id = fn(progress)
                job.status, job.stage, job.progress = "done", "done", 1.0
            except Cancelled:
                job.status, job.stage = "cancelled", "cancelled"
            except Exception as e:  # surfaced to the client verbatim
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                job.log.append({"t": round(time.time() - job.created, 2), "stage": "error", "msg": traceback.format_exc(limit=4)})
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        j = self._jobs.get(job_id)
        if j and j.status in ("queued", "running"):
            j.cancel_requested = True
            return True
        return False
