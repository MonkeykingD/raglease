"""Faiss ANN executor bound to a specific CUDA (GreenContext) stream.

Key properties:
- Loads ANN indexes before KV cache allocation (memory ordering).
- Full-GPU mode preserves the original asynchronous IVFPQ path.
- VectorLite hot-index mode quantizes once on CPU, searches hot probes on a
  GPU shard and cold probes on the complete CPU FastScan index concurrently,
  then merges their top-k candidates.
- Frontend-GC mode uses dedicated ANN streams in the selected partition, so
  scheduler synchronization of the LLM prefill stream does not wait for ANN.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Sequence

import numpy as np
import torch

from sglang.srt.mem_cache.shared_hbm import (
    DEFAULT_FAISS_TEMP_BYTES,
    SharedHBMOwner,
)

logger = logging.getLogger(__name__)

# The bundled VectorLite Faiss GPU IVF implementation supports at most 4096
# preassigned lists per query. Larger total nprobe values keep the overflow on
# the authoritative CPU index instead of dropping probes.
_FAISS_GPU_MAX_NPROBE = 4096


class _FaissModuleCompat:
    """Use proxy objects while retaining access to low-level SWIG calls.

    Recent Faiss builds expose owning Python proxy classes from ``faiss`` and
    keep pointer-oriented functions such as ``GpuIndex_search`` in the private
    extension module.  Older VectorLite builds exposed both through one module.
    The executor needs both APIs, so resolve public wrappers first and then
    fall back to the private extension.
    """

    def __init__(self, module):
        self._module = module
        self._raw = getattr(module, "_swigfaiss_avx2", None)

    def __getattr__(self, name: str):
        try:
            return getattr(self._module, name)
        except AttributeError:
            if self._raw is not None:
                return getattr(self._raw, name)
            raise


def _faiss_egg_dir() -> Path:
    configured = os.environ.get("FAISS_EGG_DIR")
    if configured:
        return Path(configured).expanduser()

    import importlib.util

    spec = importlib.util.find_spec("faiss")
    if spec is not None and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        return package_dir.parent

    raise RuntimeError(
        "FAISS_EGG_DIR is not set and no installed faiss package was found"
    )


def _import_faiss():
    """Lazy-import Faiss with owning proxies and raw pointer operations."""
    return _import_faiss_python()


def _import_faiss_python():
    """Import the full Faiss package required by hot/cold CPU search."""
    import importlib

    egg_dir = _faiss_egg_dir()
    package_dir = egg_dir / "faiss"
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"Faiss Python package not found: {package_dir}")

    egg_str = str(egg_dir)
    if egg_str not in sys.path:
        sys.path.insert(0, egg_str)

    previous_opt = os.environ.get("FAISS_OPT_LEVEL")
    if previous_opt is None:
        # The custom VectorLite artifact is built with the AVX2 wrapper.
        os.environ["FAISS_OPT_LEVEL"] = "AVX2"
    try:
        faiss = importlib.import_module("faiss")
    finally:
        if previous_opt is None:
            os.environ.pop("FAISS_OPT_LEVEL", None)

    required = (
        "read_index",
        "extract_index_ivf",
        "SearchParametersHNSW",
        "StandardGpuResources",
        "GpuClonerOptions",
        "IndexIVFPQ",
        "INDICES_32_BIT",
    )
    missing = [name for name in required if not hasattr(faiss, name)]
    if missing:
        raise RuntimeError(
            "Faiss Python wrappers required for hot-index mode are missing: "
            + ", ".join(missing)
        )
    return _FaissModuleCompat(faiss)


def _hot_gpu_clone_options(faiss, index):
    """Build the same GPU clone options as VectorLite PartitionedIndex."""
    options = faiss.GpuClonerOptions()
    options.useFloat16 = True
    options.usePrecomputed = True
    options.allowCpuCoarseQuantizer = True
    options.indicesOptions = faiss.INDICES_32_BIT

    index = faiss.downcast_index(index)
    if isinstance(index, faiss.IndexIVFPQ) and index.pq.nbits != 8:
        if not hasattr(options, "interleavedLayout"):
            raise RuntimeError(
                "This Faiss GPU build cannot clone non-8-bit IVFPQ hot "
                "indexes with the default layout. Use the VectorLite Faiss "
                "build whose GpuClonerOptions supports interleavedLayout."
            )
        options.interleavedLayout = True
        logger.info(
            "Enabled Faiss GPU interleavedLayout for hot IndexIVFPQ "
            "pq.nbits=%s",
            index.pq.nbits,
        )
    return options


def _empty_shared_ivfflat_source(faiss, index):
    """Clone IVF metadata without list payloads for shared-HBM startup."""
    index = faiss.downcast_index(index)
    if not hasattr(faiss, "IndexIVFFlat") or not isinstance(
        index, faiss.IndexIVFFlat
    ):
        return None
    source_ivf = faiss.extract_index_ivf(index)
    quantizer = faiss.clone_index(
        faiss.downcast_index(source_ivf.quantizer)
    )
    shell = faiss.IndexIVFFlat(
        quantizer,
        int(index.d),
        int(source_ivf.nlist),
        int(source_ivf.metric_type),
    )
    shell.is_trained = bool(index.is_trained)
    if hasattr(index, "nprobe"):
        shell.nprobe = int(index.nprobe)
    return shell


@dataclass
class AnnResult:
    """Async ANN result: GPU-side, no CPU sync.

    queries_gpu, ids_gpu, and dists_gpu are kept alive until the
    caller consumes the result, preventing PyTorch allocator reuse
    while Faiss kernels are still in-flight on the stream.
    """
    queries_gpu: torch.Tensor  # [batch_size, d] float32, kept alive for Faiss read
    ids_gpu: torch.Tensor  # [batch_size, topk] int64 on GPU
    dists_gpu: torch.Tensor  # [batch_size, topk] float32 on GPU
    start_event: torch.cuda.Event  # recorded before H2D/search work
    event: torch.cuda.Event  # recorded on GC stream AFTER all Faiss kernels
    batch_size: int
    topk: int
    parent_ready_event: Optional[torch.cuda.Event] = None
    keepalive: tuple[Any, ...] = ()

    def to_cpu(self):
        """Block until event, then D2H copy. Call once, AFTER poll()."""
        self.event.synchronize()
        return self.ids_gpu.cpu().numpy(), self.dists_gpu.cpu().numpy()

    def ids_to_cpu(self):
        """Block until event, then D2H copy ids only (skip dists)."""
        self.event.synchronize()
        return self.ids_gpu.cpu().numpy()

    def poll(self) -> bool:
        return self.event.query()

    def gpu_poll(self) -> bool:
        return self.poll()

    def gpu_elapsed_ms(self) -> float:
        """Return stream elapsed time after completion without a new sync."""
        if not self.event.query():
            self.event.synchronize()
        return float(self.start_event.elapsed_time(self.event))

    def metrics_for_span(self, start: int, end: int) -> dict[str, Any]:
        return {
            "index_mode": "full_gpu",
            "hot_probe_ratio": 0.0,
            "hot_probe_count": 0,
            "cold_probe_count": 0,
            "cpu_queue_wait_ms": 0.0,
            "cpu_search_latency_ms": 0.0,
            "coarse_quantize_latency_ms": 0.0,
            "route_latency_ms": 0.0,
            "merge_latency_ms": 0.0,
        }


@dataclass(frozen=True)
class _PreparedColdSearch:
    start: int
    end: int
    future: Future
    enqueued_at: float
    prelaunched: bool = True


@dataclass(frozen=True)
class PreparedHotSearch:
    """CPU-routed probes plus cold work started before GPU lease admission."""

    routed: Any
    coarse_quantize_latency_ms: float
    route_latency_ms: float
    cold_searches: tuple[_PreparedColdSearch, ...] = ()

    # Preserve the historical three-tuple contract for tests and callers that
    # only consume routed probes and route latency.
    def __iter__(self):
        yield self.routed
        yield self.coarse_quantize_latency_ms
        yield self.route_latency_ms

    def __getitem__(self, index):
        return (
            self.routed,
            self.coarse_quantize_latency_ms,
            self.route_latency_ms,
        )[index]

    def __len__(self):
        return 3


@dataclass
class HotAnnResult:
    """One VectorLite-style hot-GPU/cold-CPU hybrid ANN search."""

    gpu_result: Optional[AnnResult]
    cpu_future: Optional[Future]
    batch_size: int
    topk: int
    hot_counts: np.ndarray
    cold_counts: np.ndarray
    coarse_quantize_latency_ms: float
    route_latency_ms: float
    gpu_list_visits: int = 0
    gpu_vectors_scanned: Optional[int] = None
    gpu_max_query_vectors: Optional[int] = None
    gpu_max_list_vectors: Optional[int] = None
    gpu_query_vector_sq_sum: Optional[int] = None
    cpu_searches: tuple[_PreparedColdSearch, ...] = ()
    largest: bool = False
    completion_callback: Optional[Any] = None
    _ids_cpu: Optional[np.ndarray] = None
    _dists_cpu: Optional[np.ndarray] = None
    _cpu_queue_wait_ms: float = 0.0
    _cpu_search_latency_ms: float = 0.0
    _cpu_queue_wait_ms_by_row: Optional[np.ndarray] = None
    _cpu_search_latency_ms_by_row: Optional[np.ndarray] = None
    _merge_latency_ms: float = 0.0
    _completion_reported: bool = False
    _collect_lock: Lock = field(default_factory=Lock)

    def poll(self) -> bool:
        # End-to-end readiness controls the total in-flight request cap. GPU
        # residency is tracked separately by gpu_poll(), so a slow cold search
        # cannot retain ANN SMs or grow an unbounded CPU executor backlog.
        gpu_done = self.gpu_result is None or self.gpu_result.poll()
        if self.cpu_searches:
            cpu_done = all(item.future.done() for item in self.cpu_searches)
        else:
            cpu_done = self.cpu_future is None or self.cpu_future.done()
        return gpu_done and cpu_done

    def gpu_poll(self) -> bool:
        return self.gpu_result is None or self.gpu_result.poll()

    def _collect(self) -> tuple[np.ndarray, np.ndarray]:
        with self._collect_lock:
            if self._ids_cpu is not None and self._dists_cpu is not None:
                return self._ids_cpu, self._dists_cpu

            from sglang.srt.multiplex.hot_index import (
                empty_search_result,
                merge_topk,
            )

            if self.cpu_searches:
                cpu_distances, cpu_ids = empty_search_result(
                    self.batch_size, self.topk, largest=self.largest
                )
                cpu_queue_wait_ms = np.zeros(self.batch_size, dtype=np.float64)
                cpu_search_latency_ms = np.zeros(
                    self.batch_size, dtype=np.float64
                )
                covered = np.zeros(self.batch_size, dtype=np.bool_)
                for search in self.cpu_searches:
                    (
                        search_distances,
                        search_ids,
                        search_latency_ms,
                        queue_wait_ms,
                    ) = search.future.result()
                    if not 0 <= search.start < search.end <= self.batch_size:
                        raise ValueError('CPU result row range outside query batch')
                    rows = slice(search.start, search.end)
                    if covered[rows].any():
                        # A later index trim can spill additional probes for
                        # the SAME query. Retain both disjoint probe results;
                        # assignment would erase the prelaunched CPU branch.
                        search_distances, search_ids = merge_topk(
                            [(cpu_distances[rows], cpu_ids[rows]),
                             (search_distances, search_ids)], self.topk,
                            largest=self.largest)
                    cpu_distances[rows] = search_distances
                    cpu_ids[rows] = search_ids
                    covered[rows] = True
                    # Service is accumulated work on the serial CPU worker;
                    # queue is the largest task wait, not their sum. Neither
                    # value alone is the request's joined wall-clock latency.
                    cpu_queue_wait_ms[rows] = np.maximum(cpu_queue_wait_ms[rows], queue_wait_ms)
                    cpu_search_latency_ms[rows] += search_latency_ms
                self._cpu_queue_wait_ms_by_row = cpu_queue_wait_ms
                self._cpu_search_latency_ms_by_row = cpu_search_latency_ms
                self._cpu_queue_wait_ms = float(cpu_queue_wait_ms.mean())
                self._cpu_search_latency_ms = float(
                    cpu_search_latency_ms.mean()
                )
            elif self.cpu_future is None:
                cpu_distances, cpu_ids = empty_search_result(
                    self.batch_size, self.topk, largest=self.largest
                )
            else:
                (
                    cpu_distances,
                    cpu_ids,
                    self._cpu_search_latency_ms,
                    self._cpu_queue_wait_ms,
                ) = self.cpu_future.result()
                self._cpu_queue_wait_ms_by_row = np.full(
                    self.batch_size,
                    self._cpu_queue_wait_ms,
                    dtype=np.float64,
                )
                self._cpu_search_latency_ms_by_row = np.full(
                    self.batch_size,
                    self._cpu_search_latency_ms,
                    dtype=np.float64,
                )

            if self.gpu_result is None:
                gpu_distances, gpu_ids = empty_search_result(
                    self.batch_size, self.topk, largest=self.largest
                )
            else:
                gpu_ids, gpu_distances = self.gpu_result.to_cpu()

            merge_start = time.perf_counter()
            self._dists_cpu, self._ids_cpu = merge_topk(
                [(cpu_distances, cpu_ids), (gpu_distances, gpu_ids)],
                self.topk,
                largest=self.largest,
            )
            self._merge_latency_ms = (time.perf_counter() - merge_start) * 1000.0
            if (
                self.completion_callback is not None
                and not self._completion_reported
            ):
                try:
                    self.completion_callback(
                        cpu_latency_ms=self._cpu_search_latency_ms,
                        cpu_probes=int(self.cold_counts.sum()),
                        gpu_latency_ms=(
                            self.gpu_result.gpu_elapsed_ms()
                            if self.gpu_result is not None
                            else 0.0
                        ),
                        gpu_probes=int(self.hot_counts.sum()),
                        cpu_queue_ms=self._cpu_queue_wait_ms,
                    )
                    self._completion_reported = True
                except Exception:
                    logger.exception(
                        "Failed to update the shared-HBM cluster value model"
                    )
            return self._ids_cpu, self._dists_cpu

    def to_cpu(self) -> tuple[np.ndarray, np.ndarray]:
        return self._collect()

    def ids_to_cpu(self) -> np.ndarray:
        return self._collect()[0]

    def gpu_elapsed_ms(self) -> float:
        if self.gpu_result is None:
            return 0.0
        return self.gpu_result.gpu_elapsed_ms()

    def metrics_for_span(self, start: int, end: int) -> dict[str, Any]:
        hot_count = int(self.hot_counts[start:end].sum())
        cold_count = int(self.cold_counts[start:end].sum())
        total = hot_count + cold_count
        cpu_queue_wait_ms = self._cpu_queue_wait_ms
        cpu_search_latency_ms = self._cpu_search_latency_ms
        if self._cpu_queue_wait_ms_by_row is not None and end > start:
            cpu_queue_wait_ms = float(
                self._cpu_queue_wait_ms_by_row[start:end].mean()
            )
        if self._cpu_search_latency_ms_by_row is not None and end > start:
            cpu_search_latency_ms = float(
                self._cpu_search_latency_ms_by_row[start:end].mean()
            )
        return {
            "index_mode": "vectorlite_hot_cold",
            "hot_probe_ratio": hot_count / total if total else 0.0,
            "hot_probe_count": hot_count,
            "cold_probe_count": cold_count,
            "cpu_queue_wait_ms": cpu_queue_wait_ms,
            "cpu_search_latency_ms": cpu_search_latency_ms,
            "coarse_quantize_latency_ms": self.coarse_quantize_latency_ms,
            "route_latency_ms": self.route_latency_ms,
            "merge_latency_ms": self._merge_latency_ms,
        }


@dataclass(frozen=True)
class _PendingHotRoute:
    queries: np.ndarray
    future: Future = field(compare=False)


class _HotRouteBatcher:
    """Single-owner coarse-quantizer worker with queue-only microbatching."""

    def __init__(self, owner: Any, *, max_batch_queries: int):
        self.owner = owner
        self.max_batch_queries = max(1, int(max_batch_queries))
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_token = object()
        self._closed = False
        self._state_lock = Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="rag-hot-route",
            daemon=True,
        )
        self._thread.start()

    def submit(self, queries: np.ndarray) -> Future:
        future: Future = Future()
        item = _PendingHotRoute(queries=queries, future=future)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Hot-route batcher is closed")
            self._queue.put(item)
        return future

    def _run(self) -> None:
        backlog: list[Any] = []
        while True:
            first = backlog.pop(0) if backlog else self._queue.get()
            if first is self._stop_token:
                return

            pending = [first]
            query_count = len(first.queries)
            stop_after_batch = False
            while query_count < self.max_batch_queries:
                try:
                    candidate = (
                        backlog.pop(0)
                        if backlog
                        else self._queue.get_nowait()
                    )
                except queue.Empty:
                    break
                if candidate is self._stop_token:
                    stop_after_batch = True
                    break
                candidate_count = len(candidate.queries)
                if query_count + candidate_count > self.max_batch_queries:
                    backlog.insert(0, candidate)
                    break
                pending.append(candidate)
                query_count += candidate_count

            self.owner._quantize_hot_probes_batch(pending)
            if stop_after_batch:
                return

    def close(self) -> None:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._stop_token)
        self._thread.join()


@dataclass(frozen=True)
class _PendingColdSearch:
    queries: np.ndarray
    routed: Any
    enqueued_at: float
    future: Future = field(compare=False)


class _ColdSearchBatcher:
    """Single-owner cold-index worker with queue-only microbatching."""

    def __init__(self, owner: Any, *, max_batch_queries: int):
        self.owner = owner
        self.max_batch_queries = max(1, int(max_batch_queries))
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_token = object()
        self._closed = False
        self._state_lock = Lock()
        self._stats_lock = Lock()
        self._queued_items = 0
        self._queued_rows = 0
        self._peak_queued_items = 0
        self._peak_queued_rows = 0
        self._active_items = 0
        self._active_rows = 0
        # Scalar-only planner observations, guarded by _stats_lock. Future
        # identity is process-local and links PreparedColdSearch to this worker;
        # no query vectors, result buffers, or Future objects escape the snapshot.
        self._planning_queued = {}
        self._planning_active = ()
        self._planning_active_started_at = None
        self._planning_terminal = deque(maxlen=256)
        self._planning_service_samples = deque(maxlen=256)
        self._batches = 0
        self._rows = 0
        self._batch_size_histogram: dict[int, int] = {}
        self._clock_started = time.perf_counter()
        self._clock_last = self._clock_started
        self._worker_active_wall_ms = 0.0
        self._queued_without_active_wall_ms = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="rag-cold-index",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        queries: np.ndarray,
        routed: Any,
        enqueued_at: float,
    ) -> Future:
        future: Future = Future()
        item = _PendingColdSearch(
            queries=queries,
            routed=routed,
            enqueued_at=enqueued_at,
            future=future,
        )
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cold-index batcher is closed")
            with self._stats_lock:
                self._account_wall_time()
                rows = len(queries)
                self._queued_items += 1
                self._queued_rows += rows
                self._planning_queued[id(future)] = (
                    id(future), rows, getattr(routed, "cold_width", None), enqueued_at
                )
                self._peak_queued_items = max(
                    self._peak_queued_items, self._queued_items
                )
                self._peak_queued_rows = max(
                    self._peak_queued_rows, self._queued_rows
                )
            self._queue.put(item)
        return future

    def _account_wall_time(self) -> None:
        """Under stats lock: wall occupancy, NOT CPU utilization or core time."""
        now = time.perf_counter()
        elapsed_ms = max(0.0, (now - self._clock_last) * 1000.0)
        if self._active_items:
            self._worker_active_wall_ms += elapsed_ms
        elif self._queued_rows:
            self._queued_without_active_wall_ms += elapsed_ms
        self._clock_last = now

    def _mark_batch_started(
        self,
        dequeued: Sequence[Any],
        runnable: Sequence[Any],
    ) -> None:
        dequeued_rows = sum(len(item.queries) for item in dequeued)
        runnable_rows = sum(len(item.queries) for item in runnable)
        with self._stats_lock:
            self._account_wall_time()
            self._queued_items -= len(dequeued)
            self._queued_rows -= dequeued_rows
            self._active_items = len(runnable)
            self._active_rows = runnable_rows
            entries = {id(item.future): self._planning_queued.pop(id(item.future))
                       for item in dequeued}
            self._planning_active = tuple(entries[id(item.future)] for item in runnable)
            self._planning_active_started_at = self._clock_last if runnable else None
            running_ids = {id(item.future) for item in runnable}
            for identity, entry in entries.items():
                if identity not in running_ids:
                    self._planning_terminal.append((*entry, self._clock_last, "cancelled"))
            if not runnable:
                return
            self._batches += 1
            self._rows += runnable_rows
            self._batch_size_histogram[runnable_rows] = (
                self._batch_size_histogram.get(runnable_rows, 0) + 1
            )

    def _mark_batch_finished(self, runnable: Sequence[Any]) -> None:
        # Read terminal Future metadata only on the worker thread, outside the
        # stats lock. Never copy result arrays or wait for an unfinished Future.
        statuses = {}
        for item in runnable:
            future = item.future
            status = "unknown"
            if future.cancelled():
                status = "cancelled"
            elif future.done():
                status = "failed" if future.exception(timeout=0) is not None else "succeeded"
            statuses[id(future)] = status
        with self._stats_lock:
            self._account_wall_time()
            if (self._planning_active and self._planning_active_started_at is not None
                    and all(statuses[e[0]] == "succeeded" for e in self._planning_active)):
                shape = tuple((e[1], e[2]) for e in self._planning_active)
                duration_ms = (self._clock_last - self._planning_active_started_at) * 1000.0
                if duration_ms > 0:
                    self._planning_service_samples.append((shape, duration_ms))
            for entry in self._planning_active:
                self._planning_terminal.append((*entry, self._clock_last, statuses[entry[0]]))
            self._active_items = 0
            self._active_rows = 0
            self._planning_active = ()
            self._planning_active_started_at = None

    def planning_snapshot(self) -> dict[str, Any]:
        """Copy published worker state, without querying futures or CUDA.

        Queued includes not-yet-drained cancellations and items being assembled
        into the next batch; they remain queued until _mark_batch_started.
        Active starts before lock/search/host work, NOT exactly at Faiss entry.
        Terminal history retains at most 256 items; missing/expired is unknown,
        not success. Match identity AND enqueue time to avoid Python id reuse.
        Snapshots are observations, not reservation or predicted completion.
        A joint commit must revalidate; callers must not hold this lock while
        planning, waiting for a worker, or accessing the native index.
        """
        with self._stats_lock:
            return {
                "captured_at": time.perf_counter(),
                "max_batch_queries": self.max_batch_queries,
                "queued": tuple(self._planning_queued.values()),
                "active": self._planning_active,
                "active_started_at": self._planning_active_started_at,
                "terminal": tuple(self._planning_terminal),
                # Successful whole-worker wall service, excluding queue wait.
                # Includes native index lock/search/host costs; not pure Faiss.
                "service_samples": tuple(self._planning_service_samples),
                "terminal_fields": ("future_identity", "rows", "cold_width", "enqueued_at",
                                    "observed_finished_at", "status"),
                "entry_fields": ("future_identity", "rows", "cold_width", "enqueued_at"),
            }

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            self._account_wall_time()
            return {
                "enabled": True,
                "observation_wall_ms": (self._clock_last - self._clock_started) * 1000.0,
                "worker_active_wall_ms": self._worker_active_wall_ms,
                "queued_without_active_wall_ms": self._queued_without_active_wall_ms,
                "timing_scope": "cumulative since worker creation; active includes lock/search/host overhead, not CPU core utilization",
                "max_batch_queries": self.max_batch_queries,
                "queued_items": self._queued_items,
                "queued_rows": self._queued_rows,
                "peak_queued_items": self._peak_queued_items,
                "peak_queued_rows": self._peak_queued_rows,
                "active_items": self._active_items,
                "active_rows": self._active_rows,
                "batches": self._batches,
                "rows": self._rows,
                "batch_size_histogram": dict(self._batch_size_histogram),
            }

    def _run(self) -> None:
        backlog: list[Any] = []
        while True:
            first = backlog.pop(0) if backlog else self._queue.get()
            if first is self._stop_token:
                return

            pending = [first]
            query_count = len(first.queries)
            stop_after_batch = False
            while query_count < self.max_batch_queries:
                try:
                    candidate = (
                        backlog.pop(0)
                        if backlog
                        else self._queue.get_nowait()
                    )
                except queue.Empty:
                    break
                if candidate is self._stop_token:
                    stop_after_batch = True
                    break
                candidate_count = len(candidate.queries)
                if query_count + candidate_count > self.max_batch_queries:
                    backlog.insert(0, candidate)
                    break
                pending.append(candidate)
                query_count += candidate_count

            runnable = [
                item
                for item in pending
                if item.future.set_running_or_notify_cancel()
            ]
            self._mark_batch_started(pending, runnable)
            if not runnable:
                if stop_after_batch:
                    return
                continue
            try:
                self.owner._search_cold_preassigned_batch(runnable)
            finally:
                self._mark_batch_finished(runnable)
            if stop_after_batch:
                return

    def close(self) -> None:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._stop_token)
        self._thread.join()


@dataclass
class CpuAnnResult:
    """Pollable CPU fallback with the same result contract as GPU ANN."""

    future: Future
    batch_size: int
    topk: int
    # One worker publishes immutable progress tuples through this shared box.
    # (state, submitted, worker_started, lock_acquired, finished), host seconds.
    planning_observation: Optional[dict] = None
    _ids_cpu: Optional[np.ndarray] = None
    _dists_cpu: Optional[np.ndarray] = None
    _latency_ms: float = 0.0
    _collect_lock: Lock = field(default_factory=Lock)

    def poll(self) -> bool:
        return self.future.done()

    def gpu_poll(self) -> bool:
        return True

    def _collect(self) -> tuple[np.ndarray, np.ndarray]:
        with self._collect_lock:
            if self._ids_cpu is None:
                (
                    self._dists_cpu,
                    self._ids_cpu,
                    self._latency_ms,
                ) = self.future.result()
            return self._ids_cpu, self._dists_cpu

    def to_cpu(self) -> tuple[np.ndarray, np.ndarray]:
        return self._collect()

    def ids_to_cpu(self) -> np.ndarray:
        return self._collect()[0]

    def gpu_elapsed_ms(self) -> float:
        return 0.0

    def metrics_for_span(self, start: int, end: int) -> dict[str, Any]:
        return {
            "index_mode": "cpu_fallback",
            "hot_probe_ratio": 0.0,
            "hot_probe_count": 0,
            "cold_probe_count": self.nprobe_count(start, end),
            "cpu_queue_wait_ms": 0.0,
            "cpu_search_latency_ms": self._latency_ms,
            "coarse_quantize_latency_ms": 0.0,
            "route_latency_ms": 0.0,
            "merge_latency_ms": 0.0,
        }

    def nprobe_count(self, start: int, end: int) -> int:
        return max(0, end - start)


@dataclass(frozen=True)
class SharedHBMReclaimFence:
    """Stream-ordered proof that known Faiss lease users have drained."""

    ready_event: torch.cuda.Event
    source_events: tuple[torch.cuda.Event, ...]
    stream_count: int
    fast_detach_safe: bool = False


def _configure_cold_ivf_parallelism(cpu_ivf: Any, faiss_module: Any) -> None:
    """Enable list parallelism for the shared exact-Flat cold search path.

    Direct search_preassigned bypasses IndexIVF.search's mode-0 outer
    parallel region. Configure before starting workers; retain one owner and
    the existing nprobe/search lock. Other index types and max_codes-limited
    searches keep their existing behavior.
    """
    cpu_ivf = faiss_module.downcast_index(cpu_ivf)
    if isinstance(cpu_ivf, faiss_module.IndexIVFFlat) and int(cpu_ivf.max_codes) == 0:
        cpu_ivf.parallel_mode = 1


class AnnExecutor:
    """Loads an IVFPQ index and runs batch searches on a CUDA stream."""

    def __init__(
        self,
        index_path: str,
        gpu_id: int,
        nprobe: int = 2048,
        topk: int = 25,
        debug: bool = False,
        dedicated_streams: bool = False,
        stream_priority: int = -5,
        hot_index_path: str = "",
        hot_index_map_path: str = "",
        hot_index_shard_id: int = 0,
        cpu_search_threads: int = 8,
        cpu_search_max_batch_queries: int = 8,
        cluster_heat_decay: float = 0.98,
        cluster_miss_boost: float = 4.0,
        cluster_promotion_hysteresis: float = 1.15,
        cluster_min_residency_ms: float = 10.0,
        cpu_cold_max_batch_queries: int = 0,
    ):
        self.index_path = Path(index_path)
        self.gpu_id = gpu_id
        self.nprobe = nprobe
        self.topk = topk
        self.debug = debug
        self.dedicated_streams = dedicated_streams
        self.stream_priority = stream_priority
        self.hot_index_path = Path(hot_index_path) if hot_index_path else None
        self.hot_index_map_path = (
            Path(hot_index_map_path) if hot_index_map_path else None
        )
        self.hot_index_shard_id = hot_index_shard_id
        self.hot_index_mode = bool(self.hot_index_path or self.hot_index_map_path)
        if self.hot_index_mode and not (
            self.hot_index_path and self.hot_index_map_path
        ):
            raise ValueError(
                "Hot-index mode requires both hot_index_path and "
                "hot_index_map_path"
            )
        if hot_index_shard_id < 0:
            raise ValueError("hot_index_shard_id must be non-negative")
        if cpu_search_threads <= 0:
            raise ValueError("cpu_search_threads must be positive")
        self.cpu_search_threads = int(cpu_search_threads)
        if cpu_search_max_batch_queries <= 0:
            raise ValueError(
                "cpu_search_max_batch_queries must be positive"
            )
        self.cpu_search_max_batch_queries = int(
            cpu_search_max_batch_queries
        )
        if cpu_cold_max_batch_queries < 0:
            raise ValueError("cpu_cold_max_batch_queries must be non-negative")
        self.cpu_cold_max_batch_queries = int(
            cpu_cold_max_batch_queries or cpu_search_max_batch_queries
        )
        if not 0.0 < cluster_heat_decay < 1.0:
            raise ValueError("cluster_heat_decay must be in (0, 1)")
        if cluster_miss_boost < 0.0:
            raise ValueError("cluster_miss_boost must be non-negative")
        if cluster_promotion_hysteresis < 1.0:
            raise ValueError(
                "cluster_promotion_hysteresis must be at least 1"
            )
        if cluster_min_residency_ms < 0.0:
            raise ValueError(
                "cluster_min_residency_ms must be non-negative"
            )
        self.cluster_heat_decay = float(cluster_heat_decay)
        self.cluster_miss_boost = float(cluster_miss_boost)
        self.cluster_promotion_hysteresis = float(
            cluster_promotion_hysteresis
        )
        self.cluster_min_residency_ms = float(cluster_min_residency_ms)

        self._swig = None
        self._gpu_res = None
        self._gpu_index = None
        self._cpu_index = None
        self._gpu_source_index = None
        self._quantizer = None
        self._mapping = None
        self._route_batcher: Optional[_HotRouteBatcher] = None
        self._cold_batcher: Optional[_ColdSearchBatcher] = None
        self._cpu_pool: Optional[ThreadPoolExecutor] = None
        self._route_batch_size_histogram: dict[int, int] = {}
        self._route_batches = 0
        self._route_rows = 0
        self._cold_prelaunch_lock = Lock()
        self._cold_prelaunch_submissions = 0
        self._cold_prelaunch_rows = 0
        self._cold_prelaunch_attachments = 0
        self._cold_prelaunch_attached_rows = 0
        self._cold_prelaunch_completed_before_gpu = 0
        self._cold_prelaunch_cancel_attempts = 0
        self._cold_prelaunch_cancelled = 0
        self._cold_prelaunch_lead_ms = deque(maxlen=4096)
        self._cpu_search_lock = Lock()
        self._cpu_search_observation_lock = Lock()
        self._cpu_search_active_observation = None
        self._cpu_search_observation_sequence = 0
        self._cpu_search_hold_samples = deque(maxlen=256)
        self._largest = False
        self._loaded = False
        self._loaded_under_greenctx = False
        self._current_stream: Optional[torch.cuda.Stream] = None
        self._current_stream_ptr: Optional[int] = None
        self._current_stream_idx: Optional[int] = None
        self._bound_ptr_by_stream_idx: dict[int, int] = {}
        self._alternate_streams_by_stream_ptr: dict[
            int, tuple[torch.cuda.Stream, torch.cuda.Stream]
        ] = {}
        self._execution_streams_by_parent_ptr: dict[int, torch.cuda.Stream] = {}
        self._shared_hbm_async_copy_stream: Optional[
            torch.cuda.Stream
        ] = None
        self._shared_hbm_async_copy_stream_resolved = False
        self._warned_no_alternate_streams = False
        self._warned_no_async_copy_stream = False
        self._warned_no_sibling_streams = False
        self._warned_no_dedicated_stream = False
        self._warned_legacy_greenctx_stream_priority = False
        self._warned_non_hnsw_quantizer = False
        self._use_greenctx_guard = True
        self._warned_no_greenctx_context = False
        self._spatial = None
        self._shared_ivf_resident: Optional[np.ndarray] = None
        self._shared_ivf_reload_latency_ms: list[float] = []
        self._shared_ivf_evicted_bytes = 0
        self._shared_ivf_reloaded_bytes = 0
        self._shared_ivf_mapped_bytes = 0
        self._shared_ivf_current_bytes = 0
        self._shared_ivf_cpu_fallback_probes = 0
        self._shared_ivf_ownership_miss_suppressed = 0
        self._shared_ivf_capacity_miss_suppressed = 0
        self._shared_ivf_reference_resident_count = 0
        self._shared_ivf_repack_complete = False
        self._shared_ivf_heat_ewma: Optional[np.ndarray] = None
        self._shared_ivf_list_bytes: Optional[np.ndarray] = None
        self._hot_list_vector_counts: Optional[np.ndarray] = None
        self._shared_ivf_policy = None

    # ------------------------------------------------------------------
    # Load (before KV cache allocation)
    # ------------------------------------------------------------------
    def load(self, initial_stream: torch.cuda.Stream):
        """Load Faiss index, bound to initial_stream. Call BEFORE KV cache alloc."""
        if self._loaded:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        if self.hot_index_mode:
            if not self.hot_index_path.is_file():
                raise FileNotFoundError(
                    f"Hot GPU index not found: {self.hot_index_path}"
                )
            if not self.hot_index_map_path.is_file():
                raise FileNotFoundError(
                    f"Hot-index mapping not found: {self.hot_index_map_path}"
                )

        from sglang.srt.mem_cache.shared_hbm import (
            get_global_shared_hbm_pool,
        )

        shared_hbm_pool = get_global_shared_hbm_pool()
        self._swig = (
            _import_faiss_python()
            if self.hot_index_mode or shared_hbm_pool is not None
            else _import_faiss()
        )
        swig = self._swig
        execution_stream = self._get_or_create_execution_stream(initial_stream)
        self._current_stream = execution_stream

        t0 = time.perf_counter()
        logger.info(
            "Loading ANN index mode=%s cpu=%s hot=%s ...",
            "vectorlite_hot_cold" if self.hot_index_mode else "full_gpu",
            self.index_path,
            self.hot_index_path,
        )

        # 1. Read CPU index
        cpu_index = swig.read_index(str(self.index_path))
        self._cpu_index = cpu_index
        cpu_ivf = swig.extract_index_ivf(cpu_index)
        if self.hot_index_mode:
            _configure_cold_ivf_parallelism(cpu_ivf, swig)
        self._largest = int(cpu_ivf.metric_type) == int(
            swig.METRIC_INNER_PRODUCT
        )
        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = self.nprobe
        if self.debug:
            logger.info("ANN index read: ntotal=%s", swig.Index_ntotal_get(cpu_index))

        gpu_source_index = cpu_index
        if self.hot_index_mode:
            from sglang.srt.multiplex.hot_index import load_vectorlite_mapping

            nlist = int(cpu_ivf.nlist)
            self._mapping = load_vectorlite_mapping(
                self.hot_index_map_path, nlist=nlist
            )
            self._quantizer = swig.downcast_index(cpu_ivf.quantizer)

            # These are the same FastScan settings used by PartitionedIndex.
            if hasattr(cpu_index, "implem"):
                cpu_index.implem = 12
            if hasattr(cpu_index, "use_precomputed_table"):
                cpu_index.use_precomputed_table = True

            if self.hot_index_path.resolve() == self.index_path.resolve():
                # An all-cluster candidate map can use the complete CPU IVF as
                # both the fallback source and the temporary GPU clone source.
                # Reuse the host object so a large index is not read twice.
                gpu_source_index = cpu_index
                logger.info(
                    "Reusing the complete CPU IVF as the shared-HBM "
                    "candidate source"
                )
            else:
                gpu_source_index = swig.read_index(str(self.hot_index_path))
            self._gpu_source_index = gpu_source_index
            hot_ivf = swig.extract_index_ivf(gpu_source_index)
            hot_invlists = hot_ivf.invlists
            self._hot_list_vector_counts = np.asarray(
                [int(hot_invlists.list_size(list_id))
                 for list_id in range(int(hot_ivf.nlist))],
                dtype=np.int64,
            )
            if int(gpu_source_index.d) != int(cpu_index.d):
                raise ValueError(
                    "CPU and hot GPU indexes have different dimensions: "
                    f"{cpu_index.d} != {gpu_source_index.d}"
                )
            if int(hot_ivf.metric_type) != int(cpu_ivf.metric_type):
                raise ValueError("CPU and hot GPU indexes use different metrics")

            selected = self._mapping.shard_lut == self.hot_index_shard_id
            if not np.any(selected):
                raise ValueError(
                    "Mapping has no centroids for hot shard "
                    f"{self.hot_index_shard_id}"
                )
            max_local_cid = int(self._mapping.cid_lut[selected].max())
            if max_local_cid >= int(hot_ivf.nlist):
                raise ValueError(
                    f"Mapping local centroid {max_local_cid} exceeds hot-index "
                    f"nlist={hot_ivf.nlist}"
                )
            other_shards = tuple(
                shard
                for shard in self._mapping.shard_ids
                if shard != self.hot_index_shard_id
            )
            if other_shards:
                logger.warning(
                    "Hot mapping also contains shards %s; this worker loads shard "
                    "%s and routes the other shards through the complete CPU "
                    "fallback",
                    other_shards,
                    self.hot_index_shard_id,
                )
            logger.info(
                "VectorLite hot index configured: entries=%s selected=%s "
                "shard=%s cpu_nlist=%s hot_nlist=%s",
                self._mapping.num_entries,
                int(selected.sum()),
                self.hot_index_shard_id,
                nlist,
                int(hot_ivf.nlist),
            )

        # 2. Create GPU resources and clone to GPU under the target context
        # when the initial stream is a GreenContext stream.
        pushed = self._push_greenctx_context(execution_stream)
        self._loaded_under_greenctx = pushed
        bootstrap_scratch_bound = False
        registered = self._get_spatial().has_greenctx_context_for_stream(
            int(execution_stream.cuda_stream)
        )
        if self.debug:
            logger.info(
                "ANN load parent=0x%x execution=0x%x registered=%s "
                "loaded_under_greenctx=%s",
                initial_stream.cuda_stream,
                execution_stream.cuda_stream,
                registered,
                self._loaded_under_greenctx,
        )
        try:
            self._gpu_res = swig.StandardGpuResources()
            self._bind_stream(execution_stream)
            if shared_hbm_pool is not None:
                shared_hbm_pool.bind_faiss_scratch(self)
                bootstrap_scratch_bound = True

            # 3. Clone to GPU. IVF leasing needs only the coarse quantizer and
            # empty list metadata up front; list payloads are loaded directly
            # into the bounded VMM arena below. This avoids a transient full-
            # index clone before the same payloads are evicted again.
            gpu_clone_source = gpu_source_index
            if (
                shared_hbm_pool is not None
                and shared_hbm_pool.mode == "ivf"
            ):
                empty_source = _empty_shared_ivfflat_source(
                    swig, gpu_source_index
                )
                if empty_source is not None:
                    gpu_clone_source = empty_source
                    logger.info(
                        "Using an empty GPU IVF shell for shared-HBM "
                        "candidate loading"
                    )
            if self.hot_index_mode:
                co = _hot_gpu_clone_options(swig, gpu_clone_source)
            else:
                co = swig.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.usePrecomputed = False
                co.common_ivf_quantizer = True
                co.allowCpuCoarseQuantizer = True
                co.indicesOptions = swig.INDICES_32_BIT

            self._gpu_index = swig.index_cpu_to_gpu(
                self._gpu_res, 0, gpu_clone_source, co
            )
            if gpu_clone_source is not gpu_source_index:
                self._gpu_index.ntotal = int(gpu_source_index.ntotal)
            self._set_nprobe()
            if bootstrap_scratch_bound:
                # IVF shares the pool with a persistent search stack. Keeping
                # that stack bound prevents the first request from evicting
                # the complete index arena merely to recreate the same stack.
                torch.cuda.synchronize(self.gpu_id)
                if shared_hbm_pool.mode != "ivf":
                    shared_hbm_pool.unbind_faiss_scratch(self)
                bootstrap_scratch_bound = False
            if (
                shared_hbm_pool is not None
                and shared_hbm_pool.mode == "ivf"
            ):
                self._install_shared_ivf_arena(
                    shared_hbm_pool,
                    arena_bytes=self._initial_shared_ivf_arena_bytes(
                        shared_hbm_pool
                    ),
                )
        finally:
            if bootstrap_scratch_bound:
                # Do not strand ANN_SCRATCH ownership when index cloning or
                # IVF setup fails part-way through startup.
                torch.cuda.synchronize(self.gpu_id)
                shared_hbm_pool.unbind_faiss_scratch(self)
            self._pop_greenctx_context(pushed)

        if self.hot_index_mode:
            self._route_batcher = _HotRouteBatcher(
                self,
                max_batch_queries=self.cpu_search_max_batch_queries,
            )
            self._cold_batcher = _ColdSearchBatcher(
                self,
                max_batch_queries=self.cpu_cold_max_batch_queries,
            )
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rag-cold-index",
        )
        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "ANN index loaded in %.1fs, mode=%s, parent=0x%x, execution=0x%x",
            elapsed,
            "vectorlite_hot_cold" if self.hot_index_mode else "full_gpu",
            initial_stream.cuda_stream,
            execution_stream.cuda_stream,
        )

    @staticmethod
    def _faiss_id_vector(swig, list_ids):
        vector = swig.Int64Vector()
        for list_id in list_ids:
            vector.push_back(int(list_id))
        return vector

    @staticmethod
    def _initial_shared_ivf_arena_bytes(pool) -> int:
        """Leave enough free shared HBM for the first Faiss search stack."""
        scratch_reserve = min(
            DEFAULT_FAISS_TEMP_BYTES,
            max(0, int(pool.total_bytes) - int(pool.chunk_bytes)),
        )
        return max(
            int(pool.chunk_bytes),
            int(pool.total_bytes) - scratch_reserve,
        )

    def _shared_ivf_method(self, name: str):
        method = getattr(self._gpu_index, name, None)
        if method is None:
            raise RuntimeError(
                f"shared HBM ivf mode requires patched Faiss {name}()"
            )
        return method

    def _evict_shared_ivf_lists(self, list_ids) -> int:
        list_ids = [int(value) for value in list_ids]
        if not list_ids:
            return 0
        method = self._shared_ivf_method("evictInvertedLists")
        released = int(
            method(self._faiss_id_vector(self._swig, list_ids))
        )
        assert self._shared_ivf_resident is not None
        self._shared_ivf_resident[list_ids] = False
        policy = getattr(self, "_shared_ivf_policy", None)
        if policy is not None:
            policy.mark_evicted(list_ids)
        self._shared_ivf_evicted_bytes += released
        return released

    def _install_shared_ivf_arena(
        self,
        pool,
        max_lists: Optional[int] = None,
        arena_bytes: Optional[int] = None,
        planned_candidate_ids: Optional[Sequence[int]] = None,
        strict_arena_bytes: bool = False,
    ) -> int:
        """Move the hot shard's read-only list storage into ANN_INDEX VMM."""
        if not self.hot_index_mode or self._gpu_source_index is None:
            raise RuntimeError(
                "shared HBM ivf mode requires a VectorLite hot index"
            )
        initial_repack = not bool(
            getattr(self, "_shared_ivf_repack_complete", False)
        )
        nlist = int(self._gpu_index.getNumLists())
        if self._shared_ivf_resident is None:
            self._shared_ivf_resident = np.ones(nlist, dtype=np.bool_)
        source_ivf = self._swig.extract_index_ivf(
            self._gpu_source_index
        )
        if self._shared_ivf_list_bytes is None:
            # Local IDs are emitted in offline heat order. Seed the online
            # EWMA with that ranking until live probe observations dominate.
            code_size = int(getattr(source_ivf, "code_size", 0))
            invlists = source_ivf.invlists
            self._shared_ivf_list_bytes = np.asarray(
                [
                    int(invlists.list_size(list_id))
                    * (code_size + 4)
                    for list_id in range(nlist)
                ],
                dtype=np.int64,
            )
        if getattr(self, "_shared_ivf_policy", None) is None:
            from sglang.srt.multiplex.hot_index import ClusterLeasePolicy

            prior = self._shared_ivf_heat_ewma
            mapping = getattr(self, "_mapping", None)
            if prior is None and mapping is not None:
                prior = mapping.offline_prior(
                    self.hot_index_shard_id, nlist
                )
            self._shared_ivf_policy = ClusterLeasePolicy(
                self._shared_ivf_list_bytes,
                heat_decay=getattr(self, "cluster_heat_decay", 0.98),
                miss_boost=getattr(self, "cluster_miss_boost", 4.0),
                promotion_hysteresis=(
                    getattr(self, "cluster_promotion_hysteresis", 1.15)
                ),
                min_residency_ms=getattr(
                    self, "cluster_min_residency_ms", 10.0
                ),
                offline_prior=prior,
            )
            # Keep the historical field as a live compatibility alias for
            # metrics and older tests.
            self._shared_ivf_heat_ewma = (
                self._shared_ivf_policy.heat_ewma
            )
        resident_ids = np.flatnonzero(self._shared_ivf_resident)
        if resident_ids.size:
            self._evict_shared_ivf_lists(resident_ids)

        reload_order = self._shared_ivf_policy.promotion_order(
            np.arange(nlist, dtype=np.int64),
            max_lists=max_lists,
        )
        if planned_candidate_ids is not None:
            reload_order = np.asarray(planned_candidate_ids, dtype=np.int64)
            if (reload_order.ndim != 1
                    or np.any((reload_order < 0) | (reload_order >= nlist))
                    or len(np.unique(reload_order)) != len(reload_order)):
                raise ValueError("Invalid frozen IVF candidate list")
            if max_lists is not None:
                reload_order = reload_order[:max(0, int(max_lists))]
        if reload_order.size == 0:
            return 0
        # Map the bytes needed by the selected lists, rather than claiming
        # every free shared-HBM chunk during the initial full repack.
        # The factor accounts for the GPU code layout being wider than the
        # authoritative CPU payload; per-list slack covers alignment.
        requested_bytes = max(
            pool.chunk_bytes,
            int(self._shared_ivf_list_bytes[reload_order].sum()) * 2
            + int(reload_order.size) * 512,
        )
        if arena_bytes is not None:
            # A bounded refill can reserve a larger arena so later deadline-
            # limited promotions do not immediately exhaust one chunk.
            requested_bytes = (
                int(arena_bytes) if strict_arena_bytes
                else max(requested_bytes, int(arena_bytes))
            )

        address, mapped = pool.map_ann(
            SharedHBMOwner.ANN_INDEX, requested_bytes
        )
        if mapped <= 0:
            return 0
        self._shared_ivf_method("setExternalInvertedListMemory")(
            address, mapped
        )

        started = time.perf_counter()
        # Reloading in value/byte order also packs similarly hot lists next to
        # each other in the external VMM arena.
        for list_id in reload_order:
            list_id = int(list_id)
            list_started = time.perf_counter()
            try:
                self._shared_ivf_method("reloadInvertedList")(
                    source_ivf, list_id
                )
                self._shared_ivf_resident[list_id] = True
                self._shared_ivf_policy.mark_resident([list_id])
                self._shared_ivf_policy.observe_reload(
                    (time.perf_counter() - list_started) * 1000.0
                )
            except Exception as error:
                if "external IVF list arena exhausted" not in str(error):
                    raise
                break
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        resident_count = int(self._shared_ivf_resident.sum())
        if resident_count > 0:
            # The SLO path decides one missing-list reload at a time. Seed its
            # p95 with a per-list cost rather than the one-off full startup
            # population latency.
            self._shared_ivf_reload_latency_ms.append(
                elapsed_ms / resident_count
            )
        available = int(
            self._shared_ivf_method(
                "getExternalInvertedListMemoryAvailable"
            )()
        )
        self._shared_ivf_mapped_bytes = mapped
        self._shared_ivf_current_bytes = max(0, mapped - available)
        self._shared_ivf_reloaded_bytes += self._shared_ivf_current_bytes
        if initial_repack:
            self._shared_ivf_reference_resident_count = resident_count
        self._shared_ivf_repack_complete = True
        logger.info(
            "Shared-HBM IVF arena installed: mapped=%.2f GiB "
            "resident_lists=%s/%s reload_ms=%.3f",
            mapped / (1 << 30),
            int(self._shared_ivf_resident.sum()),
            nlist,
            elapsed_ms,
        )
        return int(self._shared_ivf_resident.sum())

    def preview_shared_ivf_residency(self, retained_bytes: Optional[int] = None):
        """Return detached local-list residency for fixed-lease routing.

        Caller must exclude concurrent reload/reclaim before this read. This
        does not quantize queries, update heat, trim, reload, or wait on future
        arrivals. A smaller prefix uses the SAME native victim enumeration as
        trim_shared_ivf_arena; predicted victims are absent, never preloaded.
        None means this executor has no shared-IVF residency contract.
        The returned (arena_bytes, mask) is not an activation certificate.
        """
        if self._shared_ivf_resident is None:
            return None
        current = int(self._shared_ivf_mapped_bytes)
        target = current if retained_bytes is None else retained_bytes
        if type(target) is not int or not 0 <= target <= current:
            raise ValueError('Residency preview cannot assume future arena growth')
        resident = [bool(value) for value in self._shared_ivf_resident]
        if target < current:
            beyond = self._shared_ivf_method('getExternalInvertedListsBeyond')(target)
            for i in range(beyond.size()):
                list_id = int(beyond.at(i))
                if not 0 <= list_id < len(resident):
                    raise ValueError('Native trim victim outside local index')
                resident[list_id] = False
        return target, tuple(resident)

    def preview_joint_hot_route(self, coarse_distances, coarse_ids, residency):
        """Route already-known probes under one detached residency proposal.

        Reuses the serving router and GPU probe limit; does not quantize again,
        enqueue CPU work or update placement heat. Unknown future probes are
        not an input. Missing lists go to the complete CPU index, preserving
        logical nprobe. Caller supplies this route's cold shape to the predictor,
        rather than reusing the current allocation's CPU/GPU split.
        """
        from sglang.srt.multiplex.hot_index import route_vectorlite_probes

        if residency is None:
            return None
        arena_bytes, mask = residency
        if (type(arena_bytes) is not int or arena_bytes < 0
                or not isinstance(mask, tuple) or any(type(v) is not bool for v in mask)):
            raise ValueError('Detached shared-IVF residency required')
        return route_vectorlite_probes(coarse_distances,coarse_ids,
            self._mapping,self.hot_index_shard_id,resident_local_ids=mask,
            max_hot_probes_per_query=_FAISS_GPU_MAX_NPROBE)

    def trim_shared_ivf_arena(self, pool, retained_bytes: int) -> int:
        """After the aggregate reader fence, retain an unchanged VA prefix."""
        retained_bytes = int(retained_bytes)
        old_bytes = int(self._shared_ivf_mapped_bytes)
        if not 0 <= retained_bytes <= old_bytes:
            raise ValueError("invalid retained IVF prefix")
        if retained_bytes == old_bytes:
            return 0
        # Resolve every capability before mutating pointers. Missing native
        # support is an error, never an unreported whole-arena fallback.
        beyond = self._shared_ivf_method("getExternalInvertedListsBeyond")
        trim = self._shared_ivf_method("trimExternalInvertedListMemory")
        vector = beyond(retained_bytes)
        list_ids = [int(vector.at(i)) for i in range(vector.size())]
        released = self._evict_shared_ivf_lists(list_ids)
        # Eviction publishes null pointers and drains the Faiss stream before
        # the allocator boundary is clipped and physical pages change owner.
        trim(retained_bytes)
        self._shared_ivf_mapped_bytes = retained_bytes
        available = int(self._shared_ivf_method("getExternalInvertedListMemoryAvailable")())
        self._shared_ivf_current_bytes = retained_bytes - available
        pool.trim_ann_index(retained_bytes)
        self._shared_ivf_partial_reclaims = getattr(self, "_shared_ivf_partial_reclaims", 0) + 1
        self._shared_ivf_partial_unmapped_bytes = getattr(self, "_shared_ivf_partial_unmapped_bytes", 0) + old_bytes - retained_bytes
        logger.info("Shared-HBM partial IVF reclaim: before_bytes=%s retained_bytes=%s evicted_lists=%s resident_lists=%s",
                    old_bytes, retained_bytes, len(list_ids), self.shared_ivf_resident_count)
        return released

    def release_shared_ivf_arena(self, pool) -> int:
        if self._shared_ivf_resident is None:
            return 0
        resident_ids = np.flatnonzero(self._shared_ivf_resident)
        released = self._evict_shared_ivf_lists(resident_ids)
        self._shared_ivf_method(
            "clearExternalInvertedListMemory"
        )()
        pool.release_ann(SharedHBMOwner.ANN_INDEX)
        self._shared_ivf_mapped_bytes = 0
        self._shared_ivf_current_bytes = 0
        return released

    def grow_shared_ivf_arena(self, pool, target_bytes: int):
        """Append backing after the aggregate reader fence, retaining all lists."""
        old = int(self._shared_ivf_mapped_bytes)
        target_bytes = int(target_bytes)
        if old <= 0 or target_bytes < old:
            raise ValueError("growth requires a live nonshrinking IVF arena")
        if target_bytes == old:
            return
        grow = self._shared_ivf_method("growExternalInvertedListMemory")
        pool.grow_ann_index(target_bytes)
        try:
            grow(target_bytes)
        except Exception:
            # No new lists have been installed yet; only appended pages revert.
            pool.trim_ann_index(old)
            raise
        self._shared_ivf_mapped_bytes = target_bytes
        self._shared_ivf_partial_growths = getattr(self, "_shared_ivf_partial_growths", 0) + 1

    def predict_shared_ivf_reload(
        self,
        max_lists: int,
        *,
        available_bytes_override: Optional[int] = None,
    ):
        """Return the exact candidate exchange considered for the next refill."""
        if (
            self._shared_ivf_resident is None
            or self._shared_ivf_list_bytes is None
        ):
            return None
        policy = getattr(self, "_shared_ivf_policy", None)
        if policy is None:
            return None
        inactive_ids = np.flatnonzero(~self._shared_ivf_resident)
        resident_ids = np.flatnonzero(self._shared_ivf_resident)
        try:
            available_bytes = int(
                self._shared_ivf_method(
                    "getExternalInvertedListMemoryAvailable"
                )()
            )
        except Exception:
            available_bytes = max(
                0,
                int(self._shared_ivf_mapped_bytes)
                - int(self._shared_ivf_current_bytes),
            )
        if available_bytes_override is not None:
            available_bytes = max(
                available_bytes,
                max(0, int(available_bytes_override)),
            )
        return policy.predict_refill_plan(
            inactive_ids,
            resident_ids,
            available_bytes=available_bytes,
            max_lists=max_lists,
        )

    def reload_shared_ivf_arena(
        self,
        pool,
        max_lists: Optional[int] = None,
        arena_bytes: Optional[int] = None,
        planned_candidate_ids: Optional[Sequence[int]] = None,
        allow_eviction: bool = True,
    ) -> int:
        if self._shared_ivf_resident is None:
            return 0
        reload_started = time.perf_counter()
        previous_arena_bytes = int(self._shared_ivf_mapped_bytes)
        if self._shared_ivf_mapped_bytes > 0 and arena_bytes is not None and int(arena_bytes) > int(self._shared_ivf_mapped_bytes):
            self.grow_shared_ivf_arena(pool, int(arena_bytes))
        arena_growth_ms = (time.perf_counter() - reload_started) * 1000.0
        if self._shared_ivf_mapped_bytes <= 0:
            if allow_eviction:
                return self._install_shared_ivf_arena(
                    pool, max_lists=max_lists, arena_bytes=arena_bytes
                )
            return self._install_shared_ivf_arena(
                pool,
                max_lists=max_lists,
                arena_bytes=arena_bytes,
                planned_candidate_ids=planned_candidate_ids,
                strict_arena_bytes=not allow_eviction,
            )

        inactive_ids = np.flatnonzero(~self._shared_ivf_resident)
        if inactive_ids.size == 0:
            return int(self._shared_ivf_resident.sum())
        if max_lists is not None:
            max_lists = max(0, int(max_lists))
            if max_lists == 0:
                return int(self._shared_ivf_resident.sum())

        assert self._shared_ivf_list_bytes is not None
        source_ivf = self._swig.extract_index_ivf(
            self._gpu_source_index
        )
        policy = getattr(self, "_shared_ivf_policy", None)
        if planned_candidate_ids is not None:
            planned = np.asarray(
                planned_candidate_ids, dtype=np.int64
            ).reshape(-1)
            in_range = (
                (planned >= 0)
                & (planned < self._shared_ivf_resident.size)
            )
            planned = planned[in_range]
            ordered = planned[~self._shared_ivf_resident[planned]]
            if max_lists is not None:
                ordered = ordered[:max_lists]
        elif policy is not None:
            ordered = policy.promotion_order(
                inactive_ids, max_lists=max_lists
            )
        else:
            assert self._shared_ivf_heat_ewma is not None
            benefit_per_byte = self._shared_ivf_heat_ewma / np.maximum(
                1, self._shared_ivf_list_bytes
            )
            ordered = inactive_ids[
                np.argsort(
                    -benefit_per_byte[inactive_ids], kind="stable"
                )
            ]
            if max_lists is not None:
                ordered = ordered[:max_lists]

        available_before = int(
            self._shared_ivf_method(
                "getExternalInvertedListMemoryAvailable"
            )()
        )
        started = time.perf_counter()
        loaded = 0
        loaded_ids = []
        evicted_during_reload = 0
        for list_id in ordered:
            list_id = int(list_id)
            while True:
                list_started = time.perf_counter()
                try:
                    self._shared_ivf_method("reloadInvertedList")(
                        source_ivf, list_id
                    )
                    self._shared_ivf_resident[list_id] = True
                    if policy is not None:
                        policy.mark_resident([list_id])
                        policy.observe_reload(
                            (time.perf_counter() - list_started) * 1000.0
                        )
                    loaded += 1
                    loaded_ids.append(list_id)
                    break
                except Exception as error:
                    if "external IVF list arena exhausted" not in str(error):
                        raise
                    if not allow_eviction:
                        # Prediction can underestimate GPU padding. A free-only
                        # online refill stops, never evicts an existing list.
                        break
                    resident_ids = np.flatnonzero(
                        self._shared_ivf_resident
                    )
                    if policy is not None:
                        victim = policy.replacement_victim(
                            list_id, resident_ids
                        )
                        if victim is None:
                            break
                    else:
                        lower_value = resident_ids[
                            benefit_per_byte[resident_ids]
                            < benefit_per_byte[list_id]
                        ]
                        if lower_value.size == 0:
                            break
                        victim = int(
                            lower_value[
                                np.argmin(benefit_per_byte[lower_value])
                            ]
                        )
                    released = self._evict_shared_ivf_lists([victim])
                    evicted_during_reload += released
                    self._shared_ivf_current_bytes = max(
                        0, self._shared_ivf_current_bytes - released
                    )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if loaded:
            self._shared_ivf_reload_latency_ms.append(
                elapsed_ms / loaded
            )
        available_after = int(
            self._shared_ivf_method(
                "getExternalInvertedListMemoryAvailable"
            )()
        )
        reloaded_bytes = max(
            0,
            available_before
            + evicted_during_reload
            - available_after,
        )
        self._shared_ivf_current_bytes = max(
            0, self._shared_ivf_mapped_bytes - available_after
        )
        self._shared_ivf_reloaded_bytes += reloaded_bytes
        if loaded_ids:
            pending_use = getattr(self, "_shared_ivf_reload_pending_use", set())
            pending_use.update(loaded_ids)
            self._shared_ivf_reload_pending_use = pending_use
        logger.info(
            "Shared-HBM reload executor complete: arena_before=%s arena_after=%s "
            "growth_ms=%.3f list_loop_ms=%.3f requested_lists=%s loaded_lists=%s "
            "loaded_bytes=%s loaded_ids=%s",
            previous_arena_bytes, self._shared_ivf_mapped_bytes,
            arena_growth_ms, elapsed_ms, max_lists, loaded, reloaded_bytes, loaded_ids,
        )
        return int(self._shared_ivf_resident.sum())

    @property
    def shared_ivf_resident_count(self) -> int:
        if self._shared_ivf_resident is None:
            return 0
        return int(self._shared_ivf_resident.sum())

    @property
    def shared_ivf_total_count(self) -> int:
        if self._shared_ivf_resident is None:
            return 0
        return int(self._shared_ivf_resident.size)

    @property
    def shared_ivf_reload_p95_ms(self) -> float:
        if not self._shared_ivf_reload_latency_ms:
            return float("inf")
        return float(
            np.percentile(self._shared_ivf_reload_latency_ms, 95)
        )

    @property
    def shared_ivf_reloaded_bytes(self) -> int:
        return int(self._shared_ivf_reloaded_bytes)

    def shared_ivf_metrics(self) -> dict[str, Any]:
        metrics = {
            "partial_reclaims": getattr(self, "_shared_ivf_partial_reclaims", 0),
            "partial_unmapped_bytes": getattr(self, "_shared_ivf_partial_unmapped_bytes", 0),
            "resident_lists": self.shared_ivf_resident_count,
            "total_lists": (
                int(self._shared_ivf_resident.size)
                if self._shared_ivf_resident is not None
                else 0
            ),
            "mapped_bytes": self._shared_ivf_mapped_bytes,
            "resident_payload_bytes": self._shared_ivf_current_bytes,
            "internal_fragmentation_bytes": max(
                0,
                self._shared_ivf_mapped_bytes
                - self._shared_ivf_current_bytes,
            ),
            "evicted_bytes_total": self._shared_ivf_evicted_bytes,
            "reloaded_bytes_total": self._shared_ivf_reloaded_bytes,
            "reload_p95_ms": (
                self.shared_ivf_reload_p95_ms
                if np.isfinite(self.shared_ivf_reload_p95_ms)
                else None
            ),
            "cpu_fallback_probes": (
                self._shared_ivf_cpu_fallback_probes
            ),
            "ownership_miss_suppressed_probes": (
                self._shared_ivf_ownership_miss_suppressed
            ),
            "capacity_miss_suppressed_probes": (
                self._shared_ivf_capacity_miss_suppressed
            ),
            "reference_resident_lists": (
                self._shared_ivf_reference_resident_count
            ),
            "mean_heat_ewma": (
                float(self._shared_ivf_heat_ewma.mean())
                if self._shared_ivf_heat_ewma is not None
                else 0.0
            ),
        }
        policy = getattr(self, "_shared_ivf_policy", None)
        if policy is not None:
            metrics["cluster_lease"] = policy.metrics()
        return metrics

    def _get_or_create_execution_stream(
        self, parent_stream: torch.cuda.Stream
    ) -> torch.cuda.Stream:
        """Return an ANN-only stream in the parent's resource partition."""
        if not self.dedicated_streams:
            return parent_stream

        parent_ptr = int(parent_stream.cuda_stream)
        cached = self._execution_streams_by_parent_ptr.get(parent_ptr)
        if cached is not None:
            return cached

        spatial = self._get_spatial()
        is_greenctx = spatial.has_greenctx_context_for_stream(parent_ptr)
        if is_greenctx and hasattr(spatial, "create_greenctx_stream_like"):
            execution_stream = self._create_greenctx_stream_like(
                spatial, parent_ptr
            )
        elif is_greenctx:
            # Correctness fallback for an old sgl-kernel. This cannot isolate
            # ANN from scheduler synchronization of the parent GC stream.
            execution_stream = parent_stream
            if not self._warned_no_dedicated_stream:
                logger.warning(
                    "sgl-kernel does not expose create_greenctx_stream_like; "
                    "ANN must reuse the parent GreenContext stream"
                )
                self._warned_no_dedicated_stream = True
        else:
            # Full-SM groups are ordinary, non-GC paths. Keep ANN on a separate
            # stream so it does not contaminate the scheduler's control stream.
            execution_stream = torch.cuda.Stream(
                device=self.gpu_id, priority=self.stream_priority
            )

        self._execution_streams_by_parent_ptr[parent_ptr] = execution_stream
        if self.debug:
            logger.info(
                "ANN execution stream parent=0x%x execution=0x%x greenctx=%s",
                parent_ptr,
                execution_stream.cuda_stream,
                is_greenctx,
            )
        return execution_stream

    def _create_greenctx_stream_like(self, spatial, stream_ptr: int):
        """Create a sibling GC stream across old and new sgl-kernel APIs."""
        try:
            return spatial.create_greenctx_stream_like(
                stream_ptr,
                self.gpu_id,
                priority=self.stream_priority,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'priority'" not in str(exc):
                raise
            if not self._warned_legacy_greenctx_stream_priority:
                logger.warning(
                    "sgl-kernel create_greenctx_stream_like does not accept "
                    "priority; creating ANN sibling streams with its default "
                    "priority"
                )
                self._warned_legacy_greenctx_stream_priority = True
            return spatial.create_greenctx_stream_like(stream_ptr, self.gpu_id)

    def _get_or_create_alternate_streams(
        self, stream: torch.cuda.Stream
    ) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        stream_ptr_int = int(stream.cuda_stream)
        cached = self._alternate_streams_by_stream_ptr.get(stream_ptr_int)
        if cached is not None:
            return cached

        spatial = self._get_spatial()
        if (
            spatial.has_greenctx_context_for_stream(stream_ptr_int)
            and hasattr(spatial, "create_greenctx_stream_like")
        ):
            alt_streams = (
                self._create_greenctx_stream_like(spatial, stream_ptr_int),
                self._create_greenctx_stream_like(spatial, stream_ptr_int),
            )
        elif spatial.has_greenctx_context_for_stream(stream_ptr_int):
            # Keep correctness if the local sgl-kernel has not been rebuilt with
            # create_greenctx_stream_like. This preserves GC binding but falls
            # back to serialized Faiss internal streams.
            if not self._warned_no_sibling_streams:
                logger.warning(
                    "sgl-kernel does not expose create_greenctx_stream_like; "
                    "using the main stream for Faiss alternate streams"
                )
                self._warned_no_sibling_streams = True
            alt_streams = (stream, stream)
        else:
            alt_streams = (
                torch.cuda.Stream(
                    device=self.gpu_id, priority=self.stream_priority
                ),
                torch.cuda.Stream(
                    device=self.gpu_id, priority=self.stream_priority
                ),
            )

        self._alternate_streams_by_stream_ptr[stream_ptr_int] = alt_streams
        return alt_streams

    def _bind_stream(
        self, stream: torch.cuda.Stream, stream_idx: Optional[int] = None
    ):
        """Bind Faiss GPU resources to a specific CUDA stream."""
        stream_ptr_int = int(stream.cuda_stream)
        if self._current_stream_ptr == stream_ptr_int:
            self._current_stream = stream
            self._current_stream_idx = stream_idx
            if stream_idx is not None:
                self._bound_ptr_by_stream_idx[stream_idx] = stream_ptr_int
            return

        swig = self._swig
        stream_ptr = swig.cast_integer_to_cudastream_t(stream.cuda_stream)
        swig.StandardGpuResources_setDefaultStream(self._gpu_res, 0, stream_ptr)
        if hasattr(swig, "StandardGpuResources_setAlternateStreams2"):
            alt_stream0, alt_stream1 = self._get_or_create_alternate_streams(stream)
            alt_stream0_ptr = swig.cast_integer_to_cudastream_t(
                alt_stream0.cuda_stream
            )
            alt_stream1_ptr = swig.cast_integer_to_cudastream_t(
                alt_stream1.cuda_stream
            )
            swig.StandardGpuResources_setAlternateStreams2(
                self._gpu_res, 0, alt_stream0_ptr, alt_stream1_ptr
            )
            if self.debug:
                logger.info(
                    "ANN bind default=0x%x alt0=0x%x alt1=0x%x",
                    stream.cuda_stream,
                    alt_stream0.cuda_stream,
                    alt_stream1.cuda_stream,
                )
        elif not self._warned_no_alternate_streams:
            logger.warning(
                "Faiss does not expose setAlternateStreams2; alternate streams "
                "may not follow GreenContext"
            )
            self._warned_no_alternate_streams = True
        self._current_stream = stream
        self._current_stream_ptr = stream_ptr_int
        self._current_stream_idx = stream_idx
        if stream_idx is not None:
            self._bound_ptr_by_stream_idx[stream_idx] = stream_ptr_int

    def _get_spatial(self):
        if self._spatial is None:
            from sgl_kernel import spatial

            self._spatial = spatial
        return self._spatial

    def _push_greenctx_context(self, stream: torch.cuda.Stream) -> bool:
        if not self._use_greenctx_guard:
            return False

        spatial = self._get_spatial()
        stream_ptr = int(stream.cuda_stream)
        if not spatial.has_greenctx_context_for_stream(stream_ptr):
            if self.debug and not self._warned_no_greenctx_context:
                logger.info(
                    "GreenContext guard skipped: stream is not registered as a "
                    "GreenContext stream ptr=%s",
                    stream_ptr,
                )
                self._warned_no_greenctx_context = True
            return False

        spatial.push_greenctx_context_for_stream(stream_ptr)
        return True

    def _pop_greenctx_context(self, pushed: bool):
        if pushed:
            self._get_spatial().pop_greenctx_context()

    def _set_nprobe(self, nprobe: Optional[int] = None):
        swig = self._swig
        swig.IndexIVFInterface_nprobe_set(
            self._gpu_index, self.nprobe if nprobe is None else nprobe
        )

    # ------------------------------------------------------------------
    # Dynamic stream switching
    # ------------------------------------------------------------------
    def _resolve_shared_hbm_async_copy_stream(
        self,
    ) -> Optional[torch.cuda.Stream]:
        """Wrap Faiss' internally-owned async-copy stream without owning it."""

        cached = getattr(self, "_shared_hbm_async_copy_stream", None)
        if cached is not None:
            return cached
        if getattr(
            self, "_shared_hbm_async_copy_stream_resolved", False
        ):
            return None
        self._shared_hbm_async_copy_stream_resolved = True

        getter = getattr(
            self._swig,
            "StandardGpuResources_getAsyncCopyStreamAsInteger",
            None,
        )
        if self._gpu_res is None or getter is None:
            if not getattr(
                self, "_warned_no_async_copy_stream", False
            ):
                logger.warning(
                    "Faiss does not expose its async-copy stream; shared-HBM "
                    "reclaim will use the synchronizing detach fallback"
                )
                self._warned_no_async_copy_stream = True
            return None

        try:
            pointer = int(getter(self._gpu_res, 0))
            if pointer == 0:
                stream = torch.cuda.default_stream(device=self.gpu_id)
            else:
                stream = torch.cuda.ExternalStream(
                    pointer, device=self.gpu_id
                )
        except Exception:
            if not getattr(
                self, "_warned_no_async_copy_stream", False
            ):
                logger.warning(
                    "Unable to wrap Faiss' async-copy stream; shared-HBM "
                    "reclaim will use the synchronizing detach fallback",
                    exc_info=True,
                )
                self._warned_no_async_copy_stream = True
            return None

        self._shared_hbm_async_copy_stream = stream
        return stream

    def _shared_hbm_stream_inventory(
        self,
    ) -> tuple[tuple[torch.cuda.Stream, ...], bool]:
        """Return every known stream that can still reference ANN lease memory.

        The boolean is the proof required by Faiss' no-sync detach API: it is
        true only when default, alternate, and async-copy stream classes are all
        represented in the returned inventory.
        """
        if self._gpu_res is not None and not hasattr(
            self._swig, "StandardGpuResources_setAlternateStreams2"
        ):
            raise RuntimeError(
                "zero-sync shared HBM requires patched Faiss "
                "StandardGpuResources_setAlternateStreams2"
            )

        candidates = []
        if self._current_stream is not None:
            candidates.append(self._current_stream)
        candidates.extend(self._execution_streams_by_parent_ptr.values())
        for alternate_streams in (
            self._alternate_streams_by_stream_ptr.values()
        ):
            candidates.extend(alternate_streams)
        async_copy_stream = self._resolve_shared_hbm_async_copy_stream()
        if async_copy_stream is not None:
            candidates.append(async_copy_stream)

        streams = []
        seen = set()
        for stream in candidates:
            pointer = int(stream.cuda_stream)
            if pointer in seen:
                continue
            seen.add(pointer)
            streams.append(stream)
        return tuple(streams), async_copy_stream is not None

    def shared_hbm_cuda_streams(self) -> tuple[torch.cuda.Stream, ...]:
        """Return all known default, alternate, and async-copy Faiss streams."""

        return self._shared_hbm_stream_inventory()[0]

    def fence_shared_hbm_users(
        self, fence_stream: torch.cuda.Stream
    ) -> int:
        """Fence ANN lease users without synchronizing the whole device."""
        fence = self.submit_shared_hbm_reclaim_fence(fence_stream)

        # VMM unmap/remap is a host operation, so the compatibility path must
        # observe the aggregate fence before changing ownership. The async
        # shared-HBM controller queries the same event from scheduler safe
        # points instead of blocking here.
        fence.ready_event.synchronize()
        return fence.stream_count

    def submit_shared_hbm_reclaim_fence(
        self, fence_stream: torch.cuda.Stream
    ) -> SharedHBMReclaimFence:
        """Record an aggregate ANN-drain event without blocking the host."""
        streams, fast_detach_safe = self._shared_hbm_stream_inventory()

        source_events = []
        for stream in streams:
            event = torch.cuda.Event()
            event.record(stream)
            source_events.append(event)

        with torch.cuda.stream(fence_stream):
            for event in source_events:
                fence_stream.wait_event(event)
            completed = torch.cuda.Event()
            completed.record(fence_stream)

        return SharedHBMReclaimFence(
            ready_event=completed,
            source_events=tuple(source_events),
            stream_count=len(streams),
            fast_detach_safe=fast_detach_safe,
        )

    def switch_stream(
        self, stream: torch.cuda.Stream, stream_idx: Optional[int] = None
    ):
        """Rebind Faiss to a different stream (e.g., when PD-Mux switches GC group)."""
        execution_stream = self._get_or_create_execution_stream(stream)
        if self._current_stream_ptr == int(execution_stream.cuda_stream):
            self._current_stream = execution_stream
            self._current_stream_idx = stream_idx
            if stream_idx is not None:
                self._bound_ptr_by_stream_idx[stream_idx] = int(
                    execution_stream.cuda_stream
                )
            return
        pushed = self._push_greenctx_context(execution_stream)
        try:
            self._bind_stream(execution_stream, stream_idx=stream_idx)
        finally:
            self._pop_greenctx_context(pushed)

    # ------------------------------------------------------------------
    # VectorLite-compatible hot/cold search
    # ------------------------------------------------------------------
    def _quantize_hot_probes(self, queries: np.ndarray):
        from sglang.srt.multiplex.hot_index import route_vectorlite_probes

        quantize_start = time.perf_counter()
        if hasattr(self._quantizer, "hnsw"):
            params = self._swig.SearchParametersHNSW()
            params.efSearch = int(self.nprobe * 1.1)
            coarse_distances, coarse_ids = self._quantizer.search(
                queries, self.nprobe, params
            )
        else:
            if not self._warned_non_hnsw_quantizer:
                logger.warning(
                    "VectorLite hot-index CPU quantizer is not HNSW; using its "
                    "default search instead of efSearch=1.1*nprobe"
                )
                self._warned_non_hnsw_quantizer = True
            coarse_distances, coarse_ids = self._quantizer.search(
                queries, self.nprobe
            )
        quantize_latency_ms = (time.perf_counter() - quantize_start) * 1000.0

        route_start = time.perf_counter()
        if self._shared_ivf_resident is not None:
            valid = coarse_ids >= 0
            safe_ids = np.where(valid, coarse_ids, 0)
            mapped_shards = self._mapping.shard_lut[safe_ids]
            mapped_ids = self._mapping.cid_lut[safe_ids]
            assigned = valid & (
                mapped_shards == self.hot_index_shard_id
            )
            fallback = assigned.copy()
            fallback[assigned] &= ~self._shared_ivf_resident[
                mapped_ids[assigned]
            ]
            policy = getattr(self, "_shared_ivf_policy", None)
            if policy is not None:
                local_ids = mapped_ids[assigned]
                if self._shared_ivf_mapped_bytes <= 0:
                    # Every list is absent while KV owns the IVF arena. Keep
                    # the access-frequency signal, but do not misclassify this
                    # structural fallback as a query-locality shift.
                    policy.observe_probes(local_ids)
                    self._shared_ivf_ownership_miss_suppressed += int(
                        local_ids.size
                    )
                elif (
                    self.shared_ivf_resident_count
                    < self._shared_ivf_reference_resident_count
                ):
                    # Miss ratios are comparable only at equal index
                    # capacity. During recovery, update demand heat but do not
                    # teach the shift detector that a smaller resident set is
                    # a changed query distribution.
                    policy.observe_probes(local_ids)
                    self._shared_ivf_capacity_miss_suppressed += int(
                        (
                            ~self._shared_ivf_resident[local_ids]
                        ).sum()
                    )
                else:
                    policy.observe_probes(
                        local_ids,
                        ~self._shared_ivf_resident[local_ids],
                    )
            elif self._shared_ivf_heat_ewma is not None:
                self._shared_ivf_heat_ewma *= 0.98
                np.add.at(
                    self._shared_ivf_heat_ewma,
                    mapped_ids[assigned],
                    0.02,
                )
            self._shared_ivf_cpu_fallback_probes += int(
                fallback.sum()
            )
        routed = route_vectorlite_probes(
            coarse_distances,
            coarse_ids,
            self._mapping,
            self.hot_index_shard_id,
            resident_local_ids=self._shared_ivf_resident,
            max_hot_probes_per_query=_FAISS_GPU_MAX_NPROBE,
        )
        route_latency_ms = (time.perf_counter() - route_start) * 1000.0
        return routed, quantize_latency_ms, route_latency_ms

    def _quantize_hot_probes_batch(
        self, pending: list[_PendingHotRoute]
    ) -> None:
        if not pending:
            return

        queries = np.ascontiguousarray(
            np.concatenate([item.queries for item in pending], axis=0),
            dtype=np.float32,
        )
        try:
            routed, quantize_latency_ms, route_latency_ms = (
                self._quantize_hot_probes(queries)
            )
            from sglang.srt.multiplex.hot_index import RoutedProbes

            batch_size = len(queries)
            self._route_batches += 1
            self._route_rows += batch_size
            self._route_batch_size_histogram[batch_size] = (
                self._route_batch_size_histogram.get(batch_size, 0) + 1
            )

            offset = 0
            for item in pending:
                rows = len(item.queries)
                item_routed = RoutedProbes(
                    hot_distances=np.ascontiguousarray(
                        routed.hot_distances[offset : offset + rows]
                    ),
                    hot_ids=np.ascontiguousarray(
                        routed.hot_ids[offset : offset + rows]
                    ),
                    cold_distances=np.ascontiguousarray(
                        routed.cold_distances[offset : offset + rows]
                    ),
                    cold_ids=np.ascontiguousarray(
                        routed.cold_ids[offset : offset + rows]
                    ),
                    hot_counts=np.ascontiguousarray(
                        routed.hot_counts[offset : offset + rows]
                    ),
                    cold_counts=np.ascontiguousarray(
                        routed.cold_counts[offset : offset + rows]
                    ),
                )
                if not item.future.cancelled():
                    cold_searches = self._submit_prepared_cold_search(
                        item.queries,
                        item_routed,
                        prelaunched=True,
                    )
                    item.future.set_result(
                        PreparedHotSearch(
                            routed=item_routed,
                            coarse_quantize_latency_ms=quantize_latency_ms,
                            route_latency_ms=route_latency_ms,
                            cold_searches=cold_searches,
                        )
                    )
                offset += rows
        except BaseException as error:
            for item in pending:
                if not item.future.cancelled():
                    item.future.set_exception(error)

    def hot_route_batching_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._route_batcher is not None,
            "max_batch_queries": self.cpu_search_max_batch_queries,
            "batches": self._route_batches,
            "rows": self._route_rows,
            "batch_size_histogram": dict(self._route_batch_size_histogram),
        }

    def cold_search_batching_stats(self) -> dict[str, Any]:
        batcher = getattr(self, "_cold_batcher", None)
        batcher_stats = (
            batcher.stats()
            if batcher is not None
            else {
                "enabled": False,
                "max_batch_queries": self.cpu_search_max_batch_queries,
            }
        )
        batcher_stats["cpu_search_threads"] = getattr(self, "cpu_search_threads", None)
        batcher_stats["cpu_index_parallel_mode"] = getattr(
            getattr(self, "_cpu_index", None), "parallel_mode", None
        )
        lock = getattr(self, "_cold_prelaunch_lock", None)
        if lock is None:
            return batcher_stats
        with lock:
            lead_ms = list(self._cold_prelaunch_lead_ms)
            batcher_stats.update(
                {
                    "early_submissions": self._cold_prelaunch_submissions,
                    "early_submitted_rows": self._cold_prelaunch_rows,
                    "early_attachments": self._cold_prelaunch_attachments,
                    "early_attached_rows": self._cold_prelaunch_attached_rows,
                    "completed_before_gpu_launch": (
                        self._cold_prelaunch_completed_before_gpu
                    ),
                    "cancel_attempts": self._cold_prelaunch_cancel_attempts,
                    "cancelled_before_service": (
                        self._cold_prelaunch_cancelled
                    ),
                    "prelaunch_lead_ms": {
                        "count": len(lead_ms),
                        "mean": (
                            float(np.mean(lead_ms)) if lead_ms else 0.0
                        ),
                        "p95": (
                            float(np.percentile(lead_ms, 95))
                            if lead_ms
                            else 0.0
                        ),
                        "max": max(lead_ms, default=0.0),
                    },
                }
            )
        return batcher_stats

    def _submit_prepared_cold_search(
        self,
        queries: np.ndarray,
        routed,
        *,
        prelaunched: bool,
    ) -> tuple[_PreparedColdSearch, ...]:
        if routed.cold_width <= 0:
            return ()
        batcher = getattr(self, "_cold_batcher", None)
        if batcher is None:
            # Unit-level route preparation can run before load() constructs the
            # cold worker. Production requests only arrive after load completes.
            return ()
        enqueued_at = time.perf_counter()
        future = batcher.submit(queries, routed, enqueued_at)
        rows = len(queries)
        if prelaunched:
            lock = getattr(self, "_cold_prelaunch_lock", None)
            if lock is not None:
                with lock:
                    self._cold_prelaunch_submissions += 1
                    self._cold_prelaunch_rows += rows
        return (
            _PreparedColdSearch(
                start=0,
                end=rows,
                future=future,
                enqueued_at=enqueued_at,
                prelaunched=prelaunched,
            ),
        )

    def preview_prepared_hot_spill(self, prepared, residency):
        """Predict ONLY not-yet-launched hot probes under target residency.

        Existing cold futures were already submitted and remain unchanged.
        Return residual hot work and additional CPU spill, not the full cold
        branch. Caller uses the latter as new worker arrivals in its forecast.
        This method does not enqueue work, quantize, update heat or read future
        request arrivals. GPU-unlaunched ownership is a caller-side precondition.
        """
        from sglang.srt.multiplex.hot_index import reroute_unsubmitted_hot_probes
        if not isinstance(prepared, PreparedHotSearch) or residency is None:
            raise ValueError('Prepared probes and detached residency required')
        arena_bytes, mask = residency
        if (type(arena_bytes) is not int or arena_bytes < 0 or not isinstance(mask, tuple)
                or any(type(value) is not bool for value in mask)):
            raise ValueError('Invalid detached residency')
        if prepared.routed.cold_width > 0 and not prepared.cold_searches:
            raise ValueError('Original cold branch has no submitted futures')
        return reroute_unsubmitted_hot_probes(prepared.routed, self._mapping,
                                             self.hot_index_shard_id, mask)

    def commit_prepared_hot_spill(self, queries, prepared, residency, *, event_id: int):
        """Commit additional cold probes once for an admitted, unlaunched batch.

        Call under scheduler serialization AFTER joint admission and residency
        validation, never from scoring. Caller must replace its prepared item
        with the returned object before GPU submission or any later decision.
        Existing cold futures are retained; same-event re-entry is idempotent.
        An uncertain submit is recorded and cannot be retried in this event.
        """
        from sglang.srt.multiplex.hot_index import RoutedProbes
        if type(event_id) is not int or event_id < 0 or not isinstance(prepared, PreparedHotSearch):
            raise ValueError('Actual event and prepared CPU futures required')
        if residency is None or len(queries) != len(prepared.routed.hot_counts):
            raise ValueError('Known residency and matching query batch required')
        if prepared.routed.cold_width > 0 and not prepared.cold_searches:
            raise ValueError('Original cold branch has no submitted futures')
        last = getattr(self, '_joint_spill_event', -1)
        if event_id < last:
            raise ValueError('Stale spill decision')
        if event_id > last:
            self._joint_spill_event = event_id
            self._joint_spill_receipts = {}
        key = id(prepared)
        receipts = self._joint_spill_receipts
        if key in receipts:
            original, prior_residency, result = receipts[key]
            if original is not prepared or prior_residency != residency:
                raise ValueError('Cannot replace a committed spill proposal')
            if result is None:
                raise RuntimeError('Spill submission failed or is incomplete; no retry')
            return result
        residual = self.preview_prepared_hot_spill(prepared, residency)
        routed = RoutedProbes(residual.hot_distances,residual.hot_ids,
            np.concatenate((prepared.routed.cold_distances,residual.cold_distances),axis=1),
            np.concatenate((prepared.routed.cold_ids,residual.cold_ids),axis=1),
            residual.hot_counts,prepared.routed.cold_counts+residual.cold_counts)
        receipts[key] = (prepared,residency,None)  # Strong owner prevents id reuse.
        added = self._submit_prepared_cold_search(queries,residual,prelaunched=True)
        if residual.cold_width > 0 and not added:
            raise RuntimeError('Spill CPU worker unavailable')
        result = PreparedHotSearch(routed,prepared.coarse_quantize_latency_ms,
                                   prepared.route_latency_ms,prepared.cold_searches+tuple(added))
        receipts[key] = (prepared,residency,result)
        return result

    def cancel_prepared_cold_searches(
        self, prepared_items: Sequence[Any]
    ) -> int:
        searches = [
            search
            for item in prepared_items
            if isinstance(item, PreparedHotSearch)
            for search in item.cold_searches
            if search.prelaunched
        ]
        if not searches:
            return 0
        cancelled = sum(search.future.cancel() for search in searches)
        lock = getattr(self, "_cold_prelaunch_lock", None)
        if lock is not None:
            with lock:
                self._cold_prelaunch_cancel_attempts += len(searches)
                self._cold_prelaunch_cancelled += cancelled
        return cancelled

    def _record_prelaunched_cold_attachment(
        self,
        searches: Sequence[_PreparedColdSearch],
        gpu_launch_at: float,
    ) -> None:
        early = [search for search in searches if search.prelaunched]
        if not early:
            return
        lock = getattr(self, "_cold_prelaunch_lock", None)
        if lock is None:
            return
        with lock:
            self._cold_prelaunch_attachments += len(early)
            self._cold_prelaunch_attached_rows += sum(
                search.end - search.start for search in early
            )
            self._cold_prelaunch_completed_before_gpu += sum(
                search.future.done() for search in early
            )
            self._cold_prelaunch_lead_ms.extend(
                max(0.0, (gpu_launch_at - search.enqueued_at) * 1000.0)
                for search in early
            )

    @staticmethod
    def combine_prepared_hot_searches(prepared_items: Sequence[Any]):
        """Combine independently routed requests into one ANN microbatch."""
        if not prepared_items:
            raise ValueError("At least one prepared hot search is required")
        if len(prepared_items) == 1:
            return prepared_items[0]

        from sglang.srt.multiplex.hot_index import RoutedProbes

        routed_items = [
            item.routed if isinstance(item, PreparedHotSearch) else item[0]
            for item in prepared_items
        ]
        row_counts = [len(routed.hot_counts) for routed in routed_items]
        total_rows = sum(row_counts)
        if total_rows <= 0:
            raise ValueError("Prepared hot searches must contain query rows")

        hot_width = max(routed.hot_width for routed in routed_items)
        cold_width = max(routed.cold_width for routed in routed_items)

        def pad(array: np.ndarray, width: int, fill_value):
            if array.shape[1] == width:
                return np.ascontiguousarray(array)
            padded = np.full(
                (array.shape[0], width), fill_value, dtype=array.dtype
            )
            padded[:, : array.shape[1]] = array
            return padded

        combined = RoutedProbes(
            hot_distances=np.ascontiguousarray(
                np.concatenate(
                    [
                        pad(routed.hot_distances, hot_width, np.inf)
                        for routed in routed_items
                    ],
                    axis=0,
                )
            ),
            hot_ids=np.ascontiguousarray(
                np.concatenate(
                    [pad(routed.hot_ids, hot_width, -1) for routed in routed_items],
                    axis=0,
                )
            ),
            cold_distances=np.ascontiguousarray(
                np.concatenate(
                    [
                        pad(routed.cold_distances, cold_width, np.inf)
                        for routed in routed_items
                    ],
                    axis=0,
                )
            ),
            cold_ids=np.ascontiguousarray(
                np.concatenate(
                    [
                        pad(routed.cold_ids, cold_width, -1)
                        for routed in routed_items
                    ],
                    axis=0,
                )
            ),
            hot_counts=np.ascontiguousarray(
                np.concatenate(
                    [routed.hot_counts for routed in routed_items], axis=0
                )
            ),
            cold_counts=np.ascontiguousarray(
                np.concatenate(
                    [routed.cold_counts for routed in routed_items], axis=0
                )
            ),
        )
        quantize_latency_ms = sum(
            float(
                item.coarse_quantize_latency_ms
                if isinstance(item, PreparedHotSearch)
                else item[1]
            )
            * rows
            for item, rows in zip(prepared_items, row_counts)
        ) / total_rows
        route_latency_ms = sum(
            float(
                item.route_latency_ms
                if isinstance(item, PreparedHotSearch)
                else item[2]
            )
            * rows
            for item, rows in zip(prepared_items, row_counts)
        ) / total_rows
        cold_searches = []
        offset = 0
        for item, rows in zip(prepared_items, row_counts):
            if isinstance(item, PreparedHotSearch):
                cold_searches.extend(
                    _PreparedColdSearch(
                        start=search.start + offset,
                        end=search.end + offset,
                        future=search.future,
                        enqueued_at=search.enqueued_at,
                        prelaunched=search.prelaunched,
                    )
                    for search in item.cold_searches
                )
            offset += rows
        return PreparedHotSearch(
            routed=combined,
            coarse_quantize_latency_ms=quantize_latency_ms,
            route_latency_ms=route_latency_ms,
            cold_searches=tuple(cold_searches),
        )

    def _observe_hot_search_latency(self, **sample) -> None:
        policy = getattr(self, "_shared_ivf_policy", None)
        if policy is not None:
            policy.observe_search_latency(**sample)

    @contextmanager
    def _observed_cpu_search_lock(self, shape):
        """Keep native mutual exclusion; publish lock-held service separately.

        No FIFO order is inferred for threads waiting on the native mutex.
        Existing whole-worker profiles already include contention: these hold
        samples must not be added to them as another waiting-time charge.
        """
        with self._cpu_search_lock:
            started = time.perf_counter()
            with self._cpu_search_observation_lock:
                self._cpu_search_observation_sequence += 1
                sequence = self._cpu_search_observation_sequence
                self._cpu_search_active_observation = (sequence, shape, started)
            succeeded = False
            try:
                yield
                succeeded = True
            finally:
                finished = time.perf_counter()
                with self._cpu_search_observation_lock:
                    if succeeded and finished > started:
                        self._cpu_search_hold_samples.append((shape, (finished-started)*1000.))
                    self._cpu_search_active_observation = None

    def cpu_search_lock_snapshot(self):
        """Read published observations without acquiring the native search lock."""
        with self._cpu_search_observation_lock:
            return dict(captured_at=time.perf_counter(),
                sequence=self._cpu_search_observation_sequence,
                active=self._cpu_search_active_observation,
                hold_samples=tuple(self._cpu_search_hold_samples))

    def _search_cold_preassigned_batch(
        self, pending: list[_PendingColdSearch]
    ) -> None:
        if not pending:
            return

        queries = np.ascontiguousarray(
            np.concatenate([item.queries for item in pending], axis=0),
            dtype=np.float32,
        )
        max_width = max(item.routed.cold_width for item in pending)
        fill_distance = -np.inf if self._largest else np.inf
        cold_ids = np.full(
            (len(queries), max_width), -1, dtype=np.int64
        )
        cold_distances = np.full(
            (len(queries), max_width),
            fill_distance,
            dtype=np.float32,
        )
        offset = 0
        for item in pending:
            rows = len(item.queries)
            width = item.routed.cold_width
            cold_ids[offset : offset + rows, :width] = (
                item.routed.cold_ids
            )
            cold_distances[offset : offset + rows, :width] = (
                item.routed.cold_distances
            )
            offset += rows

        started = time.perf_counter()
        try:
            with self._observed_cpu_search_lock(('cold_batch',len(queries),max_width,self.topk)):
                self._swig.omp_set_num_threads(self.cpu_search_threads)
                self._cpu_index.nprobe = max_width
                distances, ids = self._cpu_index.search_preassigned(
                    queries,
                    self.topk,
                    cold_ids,
                    cold_distances,
                )
            distances = np.ascontiguousarray(
                distances, dtype=np.float32
            )
            ids = np.ascontiguousarray(ids, dtype=np.int64)
        except BaseException as error:
            for item in pending:
                if not item.future.cancelled():
                    item.future.set_exception(error)
            return
        finished = time.perf_counter()
        latency_ms = (finished - started) * 1000.0

        offset = 0
        for item in pending:
            rows = len(item.queries)
            if not item.future.cancelled():
                item.future.set_result(
                    (
                        distances[offset : offset + rows],
                        ids[offset : offset + rows],
                        latency_ms,
                        max(0.0, (started - item.enqueued_at) * 1000.0),
                    )
                )
            offset += rows

    def _search_cold_preassigned(
        self, queries: np.ndarray, routed, enqueued_at: Optional[float] = None
    ):
        from sglang.srt.multiplex.hot_index import empty_search_result

        if routed.cold_width == 0:
            distances, ids = empty_search_result(
                queries.shape[0], self.topk, largest=self._largest
            )
            return distances, ids, 0.0, 0.0

        started = time.perf_counter()
        queue_wait_ms = (
            max(0.0, (started - enqueued_at) * 1000.0)
            if enqueued_at is not None
            else 0.0
        )
        # Faiss's Python search_preassigned wrapper validates against the
        # index-level nprobe. Serialize the small mutation and read-only search.
        with self._observed_cpu_search_lock(('cold_single',len(queries),routed.cold_width,self.topk)):
            self._swig.omp_set_num_threads(self.cpu_search_threads)
            self._cpu_index.nprobe = routed.cold_width
            distances, ids = self._cpu_index.search_preassigned(
                queries,
                self.topk,
                routed.cold_ids,
                routed.cold_distances,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            np.ascontiguousarray(distances, dtype=np.float32),
            np.ascontiguousarray(ids, dtype=np.int64),
            latency_ms,
            queue_wait_ms,
        )

    def _search_cpu_full(self, queries: np.ndarray, planning_observation=None):
        started = time.perf_counter()
        submitted=planning_observation['progress'][1] if planning_observation is not None else None
        acquired=None
        if planning_observation is not None:
            planning_observation['progress']=('waiting_lock',submitted,started,None,None)
        try:
            with self._observed_cpu_search_lock(('full_cpu',len(queries),self.nprobe,self.topk)):
                acquired=time.perf_counter()
                if planning_observation is not None:
                    planning_observation['progress']=('searching',submitted,started,acquired,None)
                self._swig.omp_set_num_threads(self.cpu_search_threads)
                if hasattr(self._cpu_index, "nprobe"):
                    self._cpu_index.nprobe = self.nprobe
                distances, ids = self._cpu_index.search(queries, self.topk)
        except Exception:
            if planning_observation is not None:
                planning_observation['progress']=('failed',submitted,started,acquired,time.perf_counter())
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        if planning_observation is not None:
            planning_observation['progress']=('search_done',submitted,started,acquired,time.perf_counter())
        return (
            np.ascontiguousarray(distances, dtype=np.float32),
            np.ascontiguousarray(ids, dtype=np.int64),
            latency_ms,
        )

    def search_cpu_async(self, queries: np.ndarray) -> CpuAnnResult:
        """Immediately route an ANN microbatch to the authoritative CPU index."""
        if not self._loaded or self._cpu_pool is None:
            raise RuntimeError("Call load() before search_cpu_async()")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        observation={'progress':('queued',time.perf_counter(),None,None,None)}
        return CpuAnnResult(
            future=self._cpu_pool.submit(self._search_cpu_full, queries, observation),
            batch_size=queries.shape[0],
            topk=self.topk,
            planning_observation=observation,
        )

    def _launch_hot_preassigned(
        self,
        queries: np.ndarray,
        routed,
        stream: torch.cuda.Stream,
        stream_idx: Optional[int],
        wait_for_parent: bool,
    ) -> Optional[AnnResult]:
        if routed.hot_width == 0:
            return None

        swig = self._swig
        batch_size = queries.shape[0]
        execution_stream = self._get_or_create_execution_stream(stream)
        pushed = self._push_greenctx_context(execution_stream)
        try:
            self._bind_stream(execution_stream, stream_idx=stream_idx)
            self._set_nprobe(routed.hot_width)
            with torch.cuda.stream(execution_stream):
                parent_ready_event = None
                if (
                    wait_for_parent
                    and int(execution_stream.cuda_stream) != int(stream.cuda_stream)
                ):
                    parent_ready_event = torch.cuda.Event()
                    parent_ready_event.record(stream)
                    execution_stream.wait_event(parent_ready_event)

                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(execution_stream)
                device = f"cuda:{self.gpu_id}"
                queries_gpu = torch.from_numpy(queries).to(
                    device, non_blocking=True
                )
                hot_ids_gpu = torch.from_numpy(routed.hot_ids).to(
                    device, non_blocking=True
                )
                hot_distances_gpu = torch.from_numpy(
                    routed.hot_distances
                ).to(device, non_blocking=True)
                ids_gpu = torch.empty(
                    (batch_size, self.topk), dtype=torch.int64, device=device
                )
                dists_gpu = torch.empty(
                    (batch_size, self.topk), dtype=torch.float32, device=device
                )

                search_preassigned = getattr(
                    self._gpu_index, "search_preassigned_c", None
                )
                if search_preassigned is None:
                    raise RuntimeError(
                        "Faiss GPU index does not expose search_preassigned_c; "
                        "hot-index mode requires the full Faiss Python wrappers"
                    )
                search_preassigned(
                    batch_size,
                    swig.cast_integer_to_float_ptr(queries_gpu.data_ptr()),
                    self.topk,
                    swig.cast_integer_to_idx_t_ptr(hot_ids_gpu.data_ptr()),
                    swig.cast_integer_to_float_ptr(
                        hot_distances_gpu.data_ptr()
                    ),
                    swig.cast_integer_to_float_ptr(dists_gpu.data_ptr()),
                    swig.cast_integer_to_idx_t_ptr(ids_gpu.data_ptr()),
                    False,
                )
                event = torch.cuda.Event(enable_timing=True)
                event.record(execution_stream)
        finally:
            self._pop_greenctx_context(pushed)

        return AnnResult(
            queries_gpu=queries_gpu,
            ids_gpu=ids_gpu,
            dists_gpu=dists_gpu,
            start_event=start_event,
            event=event,
            batch_size=batch_size,
            topk=self.topk,
            parent_ready_event=parent_ready_event,
            keepalive=(hot_ids_gpu, hot_distances_gpu),
        )

    def _launch_routed_hot_search(
        self,
        queries: np.ndarray,
        routed,
        quantize_latency_ms: float,
        route_latency_ms: float,
        stream: torch.cuda.Stream,
        stream_idx: Optional[int],
        wait_for_parent: bool,
        prelaunched_cpu_searches: Optional[
            Sequence[_PreparedColdSearch]
        ] = None,
    ) -> HotAnnResult:
        if prelaunched_cpu_searches is None:
            cpu_searches = self._submit_prepared_cold_search(
                queries,
                routed,
                prelaunched=False,
            )
            if routed.cold_width > 0 and not cpu_searches:
                raise RuntimeError("Cold-index batcher is not initialized")
        else:
            cpu_searches = tuple(prelaunched_cpu_searches)
        gpu_launch_at = time.perf_counter()
        self._record_prelaunched_cold_attachment(
            cpu_searches, gpu_launch_at
        )
        gpu_result = self._launch_hot_preassigned(
            queries, routed, stream, stream_idx, wait_for_parent
        )
        pending_use = getattr(self, "_shared_ivf_reload_pending_use", None)
        if pending_use:
            used = sorted(pending_use.intersection(
                int(x) for x in routed.hot_ids.reshape(-1) if x >= 0))
            if used:
                logger.info("Shared-HBM reloaded lists submitted to GPU search: ids=%s", used)
                pending_use.difference_update(used)

        if self.debug:
            hot = int(routed.hot_counts.sum())
            cold = int(routed.cold_counts.sum())
            logger.info(
                "VectorLite ANN route batch=%s hot=%s cold=%s hit_rate=%.3f "
                "hot_width=%s cold_width=%s quantize_ms=%.3f route_ms=%.3f",
                queries.shape[0],
                hot,
                cold,
                hot / (hot + cold) if hot + cold else 0.0,
                routed.hot_width,
                routed.cold_width,
                quantize_latency_ms,
                route_latency_ms,
            )

        gpu_work=self.describe_gpu_routed_work(routed)
        gpu_list_visits=0 if gpu_work is None else gpu_work[0]
        gpu_vectors_scanned=None if gpu_work is None else gpu_work[1]
        gpu_max_query_vectors=None if gpu_work is None else gpu_work[2]
        gpu_max_list_vectors=None if gpu_work is None else gpu_work[3]
        gpu_query_vector_sq_sum=None if gpu_work is None else gpu_work[4]
        return HotAnnResult(
            gpu_result=gpu_result,
            cpu_future=None,
            batch_size=queries.shape[0],
            topk=self.topk,
            hot_counts=routed.hot_counts,
            cold_counts=routed.cold_counts,
            coarse_quantize_latency_ms=quantize_latency_ms,
            route_latency_ms=route_latency_ms,
            gpu_list_visits=gpu_list_visits,
            gpu_vectors_scanned=gpu_vectors_scanned,
            gpu_max_query_vectors=gpu_max_query_vectors,
            gpu_max_list_vectors=gpu_max_list_vectors,
            gpu_query_vector_sq_sum=gpu_query_vector_sq_sum,
            cpu_searches=cpu_searches,
            largest=self._largest,
            completion_callback=self._observe_hot_search_latency,
        )

    def describe_gpu_routed_work(self, routed):
        """Return exact totals and imbalance maxima for one frozen route.

        This is read-only and is shared by actual launch instrumentation and
        candidate HBM previews.  A preview therefore prices the lists retained
        by that candidate instead of reusing the current residency's work.
        """
        if routed is None or self._hot_list_vector_counts is None:
            return None
        ids=np.asarray(getattr(routed,'hot_ids',None),dtype=np.int64)
        counts=np.asarray(getattr(routed,'hot_counts',None),dtype=np.int64)
        if ids.ndim!=2 or counts.ndim!=1 or ids.shape[0]!=counts.size:
            raise ValueError('Routed GPU list metadata has inconsistent rows')
        valid=ids>=0
        if not np.array_equal(valid.sum(axis=1),counts):
            raise ValueError('Routed GPU list IDs disagree with per-row counts')
        valid_ids=ids[valid]
        if valid_ids.size==0:
            return (0,0,0,0,0)
        if np.any(valid_ids>=self._hot_list_vector_counts.size):
            raise ValueError('Routed GPU list exceeds captured list-size metadata')
        expanded=np.zeros(ids.shape,dtype=np.int64)
        expanded[valid]=self._hot_list_vector_counts[valid_ids]
        per_query=expanded.sum(axis=1)
        return (int(valid_ids.size),int(expanded.sum()),
                int(per_query.max()),int(expanded.max()),
                int(np.square(per_query,dtype=np.int64).sum()))

    def _search_hot_index_async(
        self,
        queries: np.ndarray,
        stream: torch.cuda.Stream,
        stream_idx: Optional[int],
        wait_for_parent: bool,
        prepared=None,
    ) -> HotAnnResult:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        prelaunched_cpu_searches = None
        if prepared is None:
            prepared = self._quantize_hot_probes(queries)
        if isinstance(prepared, PreparedHotSearch):
            routed = prepared.routed
            quantize_latency_ms = prepared.coarse_quantize_latency_ms
            route_latency_ms = prepared.route_latency_ms
            prelaunched_cpu_searches = prepared.cold_searches
        else:
            routed, quantize_latency_ms, route_latency_ms = prepared
        return self._launch_routed_hot_search(
            queries,
            routed,
            quantize_latency_ms,
            route_latency_ms,
            stream,
            stream_idx,
            wait_for_parent,
            prelaunched_cpu_searches,
        )

    def prepare_hot_search_async(self, queries: np.ndarray) -> Optional[Future]:
        if not self.hot_index_mode:
            return None
        if self._route_batcher is None:
            raise RuntimeError("Hot-index routing batcher is not initialized")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        return self._route_batcher.submit(queries)

    def search_prepared_async(
        self,
        queries: np.ndarray,
        prepared,
        stream: torch.cuda.Stream,
        stream_idx: Optional[int] = None,
        wait_for_parent: bool = False,
    ) -> HotAnnResult:
        if not self._loaded or not self.hot_index_mode:
            raise RuntimeError(
                "Prepared search requires a loaded hot-index executor"
            )
        return self._search_hot_index_async(
            queries,
            stream,
            stream_idx,
            wait_for_parent,
            prepared=prepared,
        )

    # ------------------------------------------------------------------
    # Search entry point
    # ------------------------------------------------------------------
    def search_async(
        self,
        queries: np.ndarray,
        stream: torch.cuda.Stream,
        stream_idx: Optional[int] = None,
        wait_for_parent: bool = False,
    ) -> AnnResult | HotAnnResult:
        """
        Launch a batch ANN search.

        Full-GPU mode returns immediately after queueing kernels on ``stream``.
        Hot-index mode routes probes on a CPU worker, then overlaps the cold
        CPU search with the hot GPU search without blocking the caller.

        Args:
            queries: float32 [batch_size, d] (CPU copied to GPU async)
            stream: Parent frontend stream whose resource partition ANN follows.
            stream_idx: PD-Mux stream group index for stream-level bind caching.

        Returns:
            A pollable full-GPU or hot/cold result. Call ``ids_to_cpu()`` only
            after ``poll()`` reports end-to-end completion.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before search_async()")
        if self.hot_index_mode:
            return self._search_hot_index_async(
                queries, stream, stream_idx, wait_for_parent
            )

        swig = self._swig
        batch_size = queries.shape[0]

        # Force-bind Faiss to the caller's stream each search. Keep the
        # GreenContext pushed across the actual Faiss launch; pushing only for
        # setDefaultStream is not enough to make kernels follow a new GC stream.
        execution_stream = self._get_or_create_execution_stream(stream)
        pushed = self._push_greenctx_context(execution_stream)
        try:
            self._bind_stream(execution_stream, stream_idx=stream_idx)
            registered = self._get_spatial().has_greenctx_context_for_stream(
                int(execution_stream.cuda_stream)
            )
            if self.debug:
                logger.info(
                    "ANN search parent=0x%x execution=0x%x registered=%s "
                    "loaded_under_greenctx=%s pushed_this_call=%s stream_idx=%s",
                    stream.cuda_stream,
                    execution_stream.cuda_stream,
                    registered,
                    self._loaded_under_greenctx,
                    pushed,
                    stream_idx,
                )

            with torch.cuda.stream(execution_stream):
                parent_ready_event = None
                if (
                    wait_for_parent
                    and int(execution_stream.cuda_stream) != int(stream.cuda_stream)
                ):
                    # Order ANN after work already submitted to the frontend
                    # stream without synchronizing either stream on the CPU.
                    parent_ready_event = torch.cuda.Event()
                    parent_ready_event.record(stream)
                    execution_stream.wait_event(parent_ready_event)

                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(execution_stream)

                # GPU input (async copy)
                queries_gpu = torch.from_numpy(
                    np.ascontiguousarray(queries.astype(np.float32))
                ).to(f"cuda:{self.gpu_id}", non_blocking=True)

                # GPU output buffers (stay on GPU, no CPU access until later)
                ids_gpu = torch.empty(
                    (batch_size, self.topk),
                    dtype=torch.int64,
                    device=f"cuda:{self.gpu_id}",
                )
                dists_gpu = torch.empty(
                    (batch_size, self.topk),
                    dtype=torch.float32,
                    device=f"cuda:{self.gpu_id}",
                )

                swig.GpuIndex_search(
                    self._gpu_index,
                    batch_size,
                    swig.cast_integer_to_float_ptr(queries_gpu.data_ptr()),
                    self.topk,
                    swig.cast_integer_to_float_ptr(dists_gpu.data_ptr()),
                    swig.cast_integer_to_idx_t_ptr(ids_gpu.data_ptr()),
                )

                event = torch.cuda.Event(enable_timing=True)
                event.record(execution_stream)
        finally:
            self._pop_greenctx_context(pushed)

        return AnnResult(
            queries_gpu=queries_gpu,
            ids_gpu=ids_gpu,
            dists_gpu=dists_gpu,
            start_event=start_event,
            event=event,
            batch_size=batch_size,
            topk=self.topk,
            parent_ready_event=parent_ready_event,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def free(self):
        if self._route_batcher is not None:
            self._route_batcher.close()
            self._route_batcher = None
        if self._cold_batcher is not None:
            self._cold_batcher.close()
            self._cold_batcher = None
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=True, cancel_futures=False)
            self._cpu_pool = None
        pushed = (
            self._push_greenctx_context(self._current_stream)
            if self._current_stream is not None
            else False
        )
        try:
            from sglang.srt.mem_cache.shared_hbm import (
                get_global_shared_hbm_pool,
            )

            shared_hbm_pool = get_global_shared_hbm_pool()
            if (
                shared_hbm_pool is not None
                and shared_hbm_pool.mode == "ivf"
                and self._shared_ivf_resident is not None
            ):
                self.release_shared_ivf_arena(shared_hbm_pool)
            if shared_hbm_pool is not None and self._gpu_res is not None:
                shared_hbm_pool.unbind_faiss_scratch(self)
            if self._gpu_index is not None:
                # Python proxy objects own their native allocations.  Calling
                # the raw delete function here would double-free them when the
                # proxy is collected.
                self._gpu_index = None
            if self._gpu_res is not None:
                self._gpu_res = None
        finally:
            self._pop_greenctx_context(pushed)
        self._shared_hbm_async_copy_stream = None
        self._shared_hbm_async_copy_stream_resolved = False
        self._cpu_index = None
        self._gpu_source_index = None
        self._quantizer = None
        self._mapping = None
        self._shared_ivf_resident = None
        self._shared_ivf_reference_resident_count = 0
        self._shared_ivf_repack_complete = False
        self._shared_ivf_heat_ewma = None
        self._shared_ivf_list_bytes = None
        self._hot_list_vector_counts = None
        self._shared_ivf_policy = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded
