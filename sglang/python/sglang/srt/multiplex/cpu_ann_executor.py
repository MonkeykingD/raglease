"""CPU-only Faiss executor for heterogeneous RAG serving baselines."""

from __future__ import annotations

import importlib
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np


def _import_faiss():
    configured = os.environ.get("FAISS_EGG_DIR")
    if configured and configured not in sys.path:
        sys.path.insert(0, configured)
    return importlib.import_module("faiss")


@dataclass(frozen=True)
class CpuAnnSearchResult:
    distances: np.ndarray
    ids: np.ndarray
    queue_wait_ms: float
    search_latency_ms: float
    batch_size: int


@dataclass(frozen=True)
class _PendingSearch:
    queries: np.ndarray
    topk: int
    enqueued_at: float
    future: Future[CpuAnnSearchResult] = field(compare=False)


class CpuAnnExecutor:
    """Run read-only Faiss searches through one coalescing index owner."""

    def __init__(
        self,
        index_path: str,
        *,
        nprobe: int,
        workers: int = 1,
        threads_per_search: int = 8,
        max_batch_queries: int = 1,
    ):
        if nprobe <= 0:
            raise ValueError("nprobe must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if workers != 1:
            raise ValueError(
                "CPU ANN uses one non-reentrant Faiss index owner; set "
                "workers=1 and tune threads_per_search/max_batch_queries"
            )
        if threads_per_search <= 0:
            raise ValueError("threads_per_search must be positive")
        if max_batch_queries <= 0:
            raise ValueError("max_batch_queries must be positive")

        self.index_path = Path(index_path)
        self.nprobe = int(nprobe)
        self.workers = int(workers)
        self.threads_per_search = int(threads_per_search)
        self.max_batch_queries = int(max_batch_queries)
        self._faiss: Optional[Any] = None
        self._index: Optional[Any] = None
        self._queue: Optional[queue.Queue[Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._stop_token = object()
        self._closed = True
        self.index_mode = "cpu_ivfpq"
        self.execution_mode = "coalesced_microbatch"

    @property
    def loaded(self) -> bool:
        return self._index is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.index_path.is_file():
            raise FileNotFoundError(f"CPU ANN index not found: {self.index_path}")

        faiss = _import_faiss()
        index = faiss.read_index(str(self.index_path))
        if hasattr(index, "nprobe"):
            index.nprobe = self.nprobe

        downcast = getattr(faiss, "downcast_index", lambda value: value)
        downcast_index = downcast(index)
        fastscan_type = getattr(faiss, "IndexIVFPQFastScan", None)
        if fastscan_type is not None and isinstance(downcast_index, fastscan_type):
            # Match VectorLiteRAG's optimized CPU-only baseline settings.
            downcast_index.implem = 12
            downcast_index.use_precomputed_table = True
            self.index_mode = "cpu_ivfpq_fastscan"

        self._faiss = faiss
        self._index = index
        self._queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="rag-cpu-retrieval-owner",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        if self._index is None or self._faiss is None:
            return
        self._faiss.omp_set_num_threads(self.threads_per_search)
        backlog: Deque[_PendingSearch] = deque()

        while True:
            if backlog:
                first: Any = backlog.popleft()
            else:
                if self._queue is None:
                    return
                first = self._queue.get()
            if first is self._stop_token:
                return

            pending = [first]
            query_count = len(first.queries)
            stop_after_batch = False
            while query_count < self.max_batch_queries:
                try:
                    if backlog:
                        candidate: Any = backlog.popleft()
                    elif self._queue is not None:
                        candidate = self._queue.get_nowait()
                    else:
                        break
                except queue.Empty:
                    break

                if candidate is self._stop_token:
                    stop_after_batch = True
                    break
                candidate_count = len(candidate.queries)
                if (
                    candidate.topk != first.topk
                    or query_count + candidate_count > self.max_batch_queries
                ):
                    backlog.appendleft(candidate)
                    break
                pending.append(candidate)
                query_count += candidate_count

            self._search_batch(pending)
            if stop_after_batch:
                return

    def _search_batch(self, pending: list[_PendingSearch]) -> None:
        if self._index is None:
            error = RuntimeError("CPU ANN index is closed")
            for item in pending:
                item.future.set_exception(error)
            return

        queries = np.ascontiguousarray(
            np.concatenate([item.queries for item in pending], axis=0),
            dtype=np.float32,
        )
        started_at = time.perf_counter()
        try:
            distances, ids = self._index.search(queries, pending[0].topk)
            distances = np.ascontiguousarray(distances, dtype=np.float32)
            ids = np.ascontiguousarray(ids, dtype=np.int64)
        except BaseException as error:
            for item in pending:
                if not item.future.cancelled():
                    item.future.set_exception(error)
            return
        finished_at = time.perf_counter()

        offset = 0
        for item in pending:
            rows = len(item.queries)
            if not item.future.cancelled():
                item.future.set_result(
                    CpuAnnSearchResult(
                        distances=distances[offset : offset + rows],
                        ids=ids[offset : offset + rows],
                        queue_wait_ms=(started_at - item.enqueued_at) * 1000.0,
                        search_latency_ms=(finished_at - started_at) * 1000.0,
                        batch_size=len(queries),
                    )
                )
            offset += rows

    def submit(
        self,
        queries: np.ndarray,
        topk: int,
        *,
        enqueued_at: Optional[float] = None,
    ) -> Future[CpuAnnSearchResult]:
        if not self.loaded:
            self.load()
        if topk <= 0:
            raise ValueError("topk must be positive")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError(f"ANN queries must be 2-D, got shape={queries.shape}")
        if len(queries) == 0:
            raise ValueError("ANN queries must contain at least one row")
        queued = time.perf_counter() if enqueued_at is None else enqueued_at
        future: Future[CpuAnnSearchResult] = Future()
        pending = _PendingSearch(queries, int(topk), queued, future)
        with self._state_lock:
            if self._closed or self._queue is None:
                raise RuntimeError("CPU ANN executor is closed")
            self._queue.put(pending)
        return future

    def close(self) -> None:
        with self._state_lock:
            if not self._closed and self._queue is not None:
                self._closed = True
                self._queue.put(self._stop_token)
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._queue = None
        self._index = None
        self._faiss = None
        self.index_mode = "cpu_ivfpq"

