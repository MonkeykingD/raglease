"""Runtime ownership control for the RAG shared-HBM CUDA VMM pool.

The native allocator owns physical CUDA allocation handles.  This module keeps
policy, SGLang allocator publication, and Faiss binding in Python so the memory
mechanism remains independent from the scheduling policy.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

import torch


logger = logging.getLogger(__name__)

GIB = 1 << 30
MIB = 1 << 20
DEFAULT_FAISS_TEMP_BYTES = int(1.5 * GIB)
FAISS_SCRATCH_ONE_QUERY_BYTES = 160 * MIB
FAISS_SCRATCH_TWO_QUERY_BYTES = 288 * MIB
FAISS_SCRATCH_MAX_MICROBATCH_BYTES = 448 * MIB
# Grow at the allocator's physical extent granularity.  The old eight-extent
# quantum could turn a one-extent demand into a sixteen-extent target (eight
# extents of rounding plus eight of mandatory prefetch), evicting an otherwise
# useful ANN index even when the immediate KV demand fit after a fine trim.
# Prediction-error and demand-rate headroom below can still coalesce genuinely
# bursty growth; this constant must not impose that coalescing unconditionally.
KV_GROWTH_QUANTUM_EXTENTS = 1
KV_GROWTH_MAX_PREFETCH_EXTENTS = 32
KV_ADAPTIVE_LOOKAHEAD_MAX_MS = 500.0
KV_ADAPTIVE_LOOKAHEAD_MIN_SAFETY_MS = 5.0
IVF_IDLE_REFILL_BUDGET_MS = 250.0
IVF_IDLE_REFILL_RETRY_MS = 10.0
IVF_IDLE_REFILL_BACKOFF_MS = 250.0
IVF_IDLE_REFILL_MAX_LISTS = 256


@dataclass(frozen=True)
class KVCapacityPreview:
    """Pure capacity geometry, not a reclaim or admission guarantee.

    Native mapping and allocator publication are separate facts. Index/scratch
    release and live KV ownership must still be fitted by the joint planner.
    No capacity beyond `published_capacity_tokens` is immediately usable.
    """

    required_tokens: int
    required_extents: int
    target_extents: int
    mapped_extents: int
    published_capacity_tokens: int
    native_capacity_tokens: int
    within_pool_capacity: bool
    publication_gap_tokens: int


@dataclass(frozen=True)
class KVCapacityPlan:
    """Uncommitted capacity intent. `grow` is not executable KV backing."""

    geometry: Optional[KVCapacityPreview]
    target_extents: int
    prefetch_extents: int
    reclaim_ann: bool
    status: str
    reason: str
    event_id: Optional[int] = None
    source_identity: tuple = ()
    min_prefetch_extents: int = KV_GROWTH_QUANTUM_EXTENTS


@dataclass(frozen=True)
class ANNWorkspaceFit:
    """Shape fitting only; a planned budget is not proof of mapped scratch."""

    proposed_rows: int
    fitted_rows: int
    required_bytes: int
    budget_bytes: int
    max_inflight: int


@dataclass(frozen=True)
class ANNRetentionPreview:
    """Expected ownership AFTER the selected KV transition, not ready bytes."""
    retained_index_bytes: int
    retained_scratch_bytes: int
    scratch_rebind_required: bool
    event_id: Optional[int]
    source_identity: tuple


@dataclass(frozen=True)
class ANNWorkspacePreview:
    retention: ANNRetentionPreview
    fit: ANNWorkspaceFit
    target_kv_bytes: int
    workspace_budget_bytes: int
    mapping_required: bool


@dataclass(frozen=True)
class JointHBMPlan:
    kv: KVCapacityPlan
    ann: Optional[ANNWorkspacePreview]
    # Planning availability, NOT proof of activation/index-list readiness.
    status: str
    event_id: int


def preview_kv_capacity_geometry(*, required_tokens: int,
                                 static_capacity_tokens: int,
                                 tokens_per_extent: int,
                                 total_extents: int,
                                 mapped_extents: int,
                                 published_capacity_tokens: int) -> KVCapacityPreview:
    """Plan extent geometry without polling, mutation, eviction, or mapping."""
    values = (required_tokens, static_capacity_tokens, tokens_per_extent,
              total_extents, mapped_extents, published_capacity_tokens)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("KV capacities must be nonnegative integers")
    if tokens_per_extent == 0 or mapped_extents > total_extents:
        raise ValueError("Invalid KV extent layout")
    extra = max(0, required_tokens - static_capacity_tokens)
    required_extents = (extra + tokens_per_extent - 1) // tokens_per_extent
    native = static_capacity_tokens + mapped_extents * tokens_per_extent
    return KVCapacityPreview(
        required_tokens, required_extents, min(total_extents, required_extents),
        mapped_extents, published_capacity_tokens, native,
        required_extents <= total_extents, native - published_capacity_tokens)


def resolve_shared_hbm_map_stage_extents(
    configured_extents: int, gpu_name: str
) -> int:
    """Resolve the measured architecture-specific first VMM map stage.

    ``-1`` selects the validated automatic policy. Explicit non-negative
    values remain stable experiment controls: zero disables staged mapping,
    while a positive value is the first-stage extent floor.
    """

    configured_extents = int(configured_extents)
    if configured_extents >= 0:
        return configured_extents
    if configured_extents != -1:
        raise ValueError(
            "shared-HBM map stage extents must be -1 or non-negative"
        )

    normalized_name = str(gpu_name).upper()
    if "A100" in normalized_name:
        return 1
    if "H100" in normalized_name:
        return 2
    raise RuntimeError(
        "automatic shared-HBM map staging is validated only on A100 or H100; "
        f"set --rag-shared-hbm-map-stage-extents explicitly for {gpu_name!r}"
    )


def ensure_shared_hbm_ops_loaded():
    """Load the small VMM dispatcher overlay when it is not in common_ops."""

    import sgl_kernel

    try:
        torch.ops.sgl_kernel.shared_hbm_create
        return
    except AttributeError:
        pass

    candidates = []
    configured = os.getenv("SGLANG_SHARED_HBM_OPS_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    package_dir = Path(sgl_kernel.__file__).resolve().parent
    candidates.append(package_dir / "libshared_hbm_ops.so")

    errors = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            torch.ops.load_library(str(candidate))
            torch.ops.sgl_kernel.shared_hbm_create
            return
        except (AttributeError, OSError, RuntimeError) as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no overlay DSO was found"
    raise RuntimeError(
        "shared-HBM native ops are unavailable; build/install "
        f"libshared_hbm_ops.so ({detail})"
    )


class SharedHBMOwner(IntEnum):
    FREE = 0
    KV_CACHE = 1
    ANN_SCRATCH = 2
    ANN_INDEX = 3
    MOVING = 4


@dataclass(frozen=True)
class SharedHBMKVLayout:
    static_capacity_tokens: int
    tokens_per_extent: int
    total_extents: int
    logical_capacity_tokens: int


@dataclass(frozen=True)
class SharedHBMSnapshot:
    total_bytes: int
    chunk_bytes: int
    free_bytes: int
    kv_bytes: int
    ann_scratch_bytes: int
    ann_index_bytes: int
    moving_bytes: int
    kv_static_capacity_tokens: int
    kv_tokens_per_extent: int
    kv_total_extents: int
    kv_mapped_extents: int
    kv_logical_capacity_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_bytes": self.total_bytes,
            "chunk_bytes": self.chunk_bytes,
            "free_bytes": self.free_bytes,
            "kv_bytes": self.kv_bytes,
            "ann_scratch_bytes": self.ann_scratch_bytes,
            "ann_index_bytes": self.ann_index_bytes,
            "moving_bytes": self.moving_bytes,
            "kv_static_capacity_tokens": self.kv_static_capacity_tokens,
            "kv_tokens_per_extent": self.kv_tokens_per_extent,
            "kv_total_extents": self.kv_total_extents,
            "kv_mapped_extents": self.kv_mapped_extents,
            "kv_logical_capacity_tokens": self.kv_logical_capacity_tokens,
        }


@dataclass(frozen=True)
class SharedHBMDriverStats:
    map_calls: int
    unmap_calls: int
    set_access_calls: int
    set_access_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "map_calls": self.map_calls,
            "unmap_calls": self.unmap_calls,
            "set_access_calls": self.set_access_calls,
            "set_access_bytes": self.set_access_bytes,
        }


class SharedHBMPool:
    """Process-local wrapper around ``libshared_hbm_pool.so``."""

    def __init__(
        self,
        *,
        mode: str,
        pool_gib: float,
        gpu_id: int,
        chunk_bytes: int = 2 << 20,
    ):
        if mode not in ("scratch", "ivf"):
            raise ValueError(f"unsupported shared HBM mode: {mode}")
        self.mode = mode
        self.gpu_id = gpu_id
        properties = torch.cuda.get_device_properties(gpu_id)
        self.gpu_name = properties.name
        self.gpu_total_memory = int(properties.total_memory)
        self.requested_bytes = int(pool_gib * GIB)
        self.chunk_bytes = chunk_bytes
        self._lock = threading.RLock()
        self._kv_layout: Optional[SharedHBMKVLayout] = None
        self._destroyed = False
        self._faiss_bound = False
        # True only after Faiss has dropped its external scratch pointer but
        # before the native ANN_SCRATCH VMM lease has been released.  Keep
        # this distinct from ordinary "unbound": binding while a worker owns
        # the deferred release would otherwise race the unmap.
        self._faiss_scratch_detaching = False
        self._last_faiss_mapping = (0, 0)
        self._last_faiss_requested_bytes = 0
        # Changes only when Faiss is actually attached to, or detached from,
        # a VMM range.  Reusing a sufficiently large existing arena leaves
        # the generation stable, which lets the controller prove that a
        # disjoint native KV map did not invalidate the captured pointer.
        self._faiss_mapping_generation = 0
        # Updated only while ``_lock`` is held, but read without that lock by
        # ANN admission while the native KV-map worker owns it.  Replacing one
        # immutable tuple is atomic under the Python interpreter and avoids
        # turning a disjoint VMM map into a pool-mutex stall.
        self._faiss_mapping_identity = (False, (0, 0), 0)
        self.static_tokens: Optional[int] = None
        self.max_tokens: Optional[int] = None

        # Importing sgl_kernel registers the native dispatcher fragment.
        import sgl_kernel  # noqa: F401

        ensure_shared_hbm_ops_loaded()
        created = torch.ops.sgl_kernel.shared_hbm_create(
            self.requested_bytes, self.chunk_bytes, gpu_id
        )
        self.total_bytes, self.chunk_bytes = map(int, created)
        logger.info(
            "Reserved shared HBM pool: mode=%s total=%.2f GiB chunk=%.2f MiB",
            mode,
            self.total_bytes / GIB,
            self.chunk_bytes / (1 << 20),
        )

    def kv_token_quantum(
        self, token_stride_bytes: int, page_size: int
    ) -> int:
        if token_stride_bytes <= 0:
            raise ValueError("KV token stride must be positive")
        stride_quantum = self.chunk_bytes // math.gcd(
            self.chunk_bytes, token_stride_bytes
        )
        return math.lcm(max(1, page_size), stride_quantum)

    def estimated_borrowable_tokens(
        self,
        cell_size: int,
        page_size: int,
        token_stride_bytes: Optional[int] = None,
    ) -> int:
        if cell_size <= 0:
            return 0
        token_quantum = (
            self.kv_token_quantum(token_stride_bytes, page_size)
            if token_stride_bytes is not None
            else max(1, page_size)
        )
        extent_bytes = token_quantum * cell_size
        return self.total_bytes // extent_bytes * token_quantum

    def configure_kv_bounds(self, static_tokens: int, max_tokens: int):
        if static_tokens <= 0 or max_tokens < static_tokens:
            raise ValueError(
                f"invalid shared-HBM KV bounds: {static_tokens=}, {max_tokens=}"
            )
        self.static_tokens = int(static_tokens)
        self.max_tokens = int(max_tokens)

    def alloc_kv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        static_tokens: int,
        max_tokens: int,
    ) -> torch.Tensor:
        tensor = torch.ops.sgl_kernel.shared_hbm_alloc_kv_tensor(
            list(shape), dtype, static_tokens, max_tokens, self.gpu_id
        )
        # Native KV extents include every registered tensor segment. Tensor
        # metadata gives its exact token stride without a device read/copy.
        self._kv_registered_token_strides = (
            *getattr(self, '_kv_registered_token_strides', ()),
            int(tensor.stride(0)*tensor.element_size()))
        return tensor

    @property
    def kv_extent_bytes(self) -> Optional[int]:
        strides = getattr(self, '_kv_registered_token_strides', ())
        if self._kv_layout is None or not strides:
            return None
        size = int(self._kv_layout.tokens_per_extent)*sum(strides)
        return size if size > 0 and size % self.chunk_bytes == 0 else None

    def finalize_kv(self, page_size: int) -> SharedHBMKVLayout:
        with self._lock:
            values = torch.ops.sgl_kernel.shared_hbm_finalize_kv(page_size)
            self._kv_layout = SharedHBMKVLayout(*map(int, values))
            return self._kv_layout

    @property
    def kv_layout(self) -> SharedHBMKVLayout:
        if self._kv_layout is None:
            raise RuntimeError("shared HBM KV layout is not finalized")
        return self._kv_layout

    def set_kv_extents(self, target_extents: int) -> int:
        with self._lock:
            return int(
                torch.ops.sgl_kernel.shared_hbm_set_kv_extents(target_extents)
            )

    def kv_growable_extents(self) -> int:
        """Return the largest KV prefix that does not overlap an ANN lease."""

        with self._lock:
            return int(
                torch.ops.sgl_kernel.shared_hbm_kv_growable_extents()
            )

    def map_ann(
        self, owner: SharedHBMOwner, requested_bytes: int
    ) -> tuple[int, int]:
        if owner not in (
            SharedHBMOwner.ANN_SCRATCH,
            SharedHBMOwner.ANN_INDEX,
        ):
            raise ValueError(f"invalid ANN owner: {owner}")
        with self._lock:
            address, mapped = torch.ops.sgl_kernel.shared_hbm_map_ann(
                int(owner), int(requested_bytes)
            )
            return int(address), int(mapped)

    def index_prefix_for_kv(self, target_extents: int) -> int:
        with self._lock:
            return int(torch.ops.sgl_kernel.shared_hbm_index_prefix_for_kv(target_extents))

    def trim_ann_index(self, retained_bytes: int):
        with self._lock:
            torch.ops.sgl_kernel.shared_hbm_trim_ann_index(retained_bytes)

    def grow_ann_index(self, target_bytes: int):
        with self._lock:
            torch.ops.sgl_kernel.shared_hbm_grow_ann_index(target_bytes)

    def release_ann(self, owner: SharedHBMOwner):
        with self._lock:
            torch.ops.sgl_kernel.shared_hbm_release_ann(int(owner))
            if owner == SharedHBMOwner.ANN_SCRATCH:
                self._faiss_scratch_detaching = False

    def snapshot(self) -> SharedHBMSnapshot:
        values = torch.ops.sgl_kernel.shared_hbm_snapshot()
        return SharedHBMSnapshot(*map(int, values))

    def driver_stats(self) -> SharedHBMDriverStats:
        values = torch.ops.sgl_kernel.shared_hbm_driver_stats()
        return SharedHBMDriverStats(*map(int, values))

    def faiss_mapping_state(self) -> tuple[bool, tuple[int, int], int]:
        """Return the lock-free immutable identity of the Faiss arena."""

        return self._faiss_mapping_identity

    def faiss_scratch_detaching(self) -> bool:
        """Return whether native scratch release is still outstanding."""

        return bool(self._faiss_scratch_detaching)

    @staticmethod
    def _faiss_method(executor, direct_name: str, swig_name: str):
        direct = getattr(executor._gpu_res, direct_name, None)
        if direct is not None:
            return direct
        return getattr(executor._swig, swig_name, None)

    def bind_faiss_scratch(
        self, executor, requested_bytes: int = DEFAULT_FAISS_TEMP_BYTES
    ) -> int:
        """Map and bind a strict external Faiss temporary stack."""
        with self._lock:
            if self._faiss_scratch_detaching:
                raise RuntimeError(
                    "cannot bind Faiss scratch while native ANN_SCRATCH "
                    "release is pending"
                )
            bind_key = (self.mode,bool(self._faiss_bound),
                        int(self._last_faiss_mapping[1]),min(int(requested_bytes),self.total_bytes))
            bind_started = time.perf_counter()
            if self._faiss_bound:
                if self._last_faiss_mapping[1] >= requested_bytes:
                    self._last_faiss_requested_bytes = min(
                        int(requested_bytes), self.total_bytes
                    )
                    return self._last_faiss_mapping[1]
                self.unbind_faiss_scratch(executor)
            requested_bytes = min(int(requested_bytes), self.total_bytes)
            address, mapped = self.map_ann(
                SharedHBMOwner.ANN_SCRATCH,
                requested_bytes,
            )
            if mapped <= 16:
                return 0
            # If attaching Faiss fails, retain a fail-closed shadow state until
            # the native lease has been released successfully.
            self._faiss_scratch_detaching = True
            method = self._faiss_method(
                executor,
                "setExternalTempMemory",
                "StandardGpuResources_setExternalTempMemory",
            )
            if method is None:
                self.release_ann(SharedHBMOwner.ANN_SCRATCH)
                raise RuntimeError(
                    "shared HBM requires the patched Faiss "
                    "setExternalTempMemory API"
                )
            try:
                if getattr(
                    executor._gpu_res, "setExternalTempMemory", None
                ):
                    method(self.gpu_id, address, mapped, True)
                else:
                    method(
                        executor._gpu_res,
                        self.gpu_id,
                        address,
                        mapped,
                        True,
                    )
            except Exception:
                try:
                    self.release_ann(SharedHBMOwner.ANN_SCRATCH)
                except Exception:
                    logger.exception(
                        "Failed to recover native ANN_SCRATCH after Faiss "
                        "attach failure"
                    )
                raise
            self._faiss_bound = True
            self._faiss_scratch_detaching = False
            self._last_faiss_mapping = (address, mapped)
            self._last_faiss_requested_bytes = requested_bytes
            self._faiss_mapping_generation += 1
            self._faiss_mapping_identity = (
                True,
                (int(address), int(mapped)),
                int(self._faiss_mapping_generation),
            )
            if mapped >= requested_bytes:
                history = getattr(self,'faiss_bind_profile_samples',None)
                if history is None:
                    history = self.faiss_bind_profile_samples = deque(maxlen=1024)
                history.append((bind_key,(time.perf_counter()-bind_started)*1000.0))
            return mapped

    @staticmethod
    def _call_faiss_method(executor, direct_name: str, method, *args):
        if getattr(executor._gpu_res, direct_name, None) is not None:
            method(*args)
        else:
            method(executor._gpu_res, *args)

    def _detach_faiss_scratch_locked(
        self,
        executor,
        *,
        streams_synchronized: bool,
        require_fast: bool = False,
    ) -> bool:
        """Drop Faiss' pointer without releasing the native VMM lease."""

        fast_method = None
        if streams_synchronized:
            fast_method = self._faiss_method(
                executor,
                "clearExternalTempMemoryAfterSync",
                "StandardGpuResources_clearExternalTempMemoryAfterSync",
            )
        if require_fast and fast_method is None:
            return False
        if fast_method is not None:
            self._call_faiss_method(
                executor,
                "clearExternalTempMemoryAfterSync",
                fast_method,
                self.gpu_id,
            )
            used_fast_detach = True
        else:
            method = self._faiss_method(
                executor,
                "clearExternalTempMemory",
                "StandardGpuResources_clearExternalTempMemory",
            )
            if method is None:
                raise RuntimeError(
                    "patched Faiss lost clearExternalTempMemory"
                )
            self._call_faiss_method(
                executor,
                "clearExternalTempMemory",
                method,
                self.gpu_id,
            )
            used_fast_detach = False

        self._faiss_bound = False
        self._faiss_scratch_detaching = True
        self._last_faiss_mapping = (0, 0)
        self._last_faiss_requested_bytes = 0
        self._faiss_mapping_generation += 1
        self._faiss_mapping_identity = (
            False,
            (0, 0),
            int(self._faiss_mapping_generation),
        )
        return used_fast_detach

    def unbind_faiss_scratch(self, executor):
        """Synchronize and detach Faiss before unmapping the VMM range."""

        with self._lock:
            if self._faiss_bound:
                self._detach_faiss_scratch_locked(
                    executor, streams_synchronized=False
                )
            if self._faiss_scratch_detaching:
                self.release_ann(SharedHBMOwner.ANN_SCRATCH)

    def unbind_faiss_scratch_after_sync(self, executor) -> bool:
        """Detach after the caller has fenced every Faiss CUDA stream.

        This is a safety-contract boundary. The caller must prevent new Faiss
        submissions and prove the default, alternate, and async-copy streams
        have completed. Older patched Faiss builds transparently use the safe,
        synchronizing fallback and return ``False``.
        """

        with self._lock:
            used_fast_detach = False
            if self._faiss_bound:
                used_fast_detach = self._detach_faiss_scratch_locked(
                    executor, streams_synchronized=True
                )
            if self._faiss_scratch_detaching:
                self.release_ann(SharedHBMOwner.ANN_SCRATCH)
            return used_fast_detach

    def detach_faiss_scratch_after_sync(self, executor) -> bool:
        """Fast-detach Faiss while retaining native ANN_SCRATCH ownership.

        The caller must already have fenced every Faiss CUDA stream and must
        arrange exactly one later ``release_ann(ANN_SCRATCH)``.  If the
        patched no-sync API is unavailable, this method makes no state change
        and returns ``False`` so the caller can use the synchronous unbind
        fallback.
        """

        with self._lock:
            if not self._faiss_bound:
                return False
            return self._detach_faiss_scratch_locked(
                executor,
                streams_synchronized=True,
                require_fast=True,
            )

    def destroy(self):
        with self._lock:
            if self._destroyed:
                return
            torch.ops.sgl_kernel.shared_hbm_destroy()
            self._destroyed = True


_GLOBAL_POOL: Optional[SharedHBMPool] = None


def get_global_shared_hbm_pool() -> Optional[SharedHBMPool]:
    return _GLOBAL_POOL


def initialize_global_shared_hbm_pool(server_args, gpu_id: int) -> Optional[SharedHBMPool]:
    global _GLOBAL_POOL
    mode = getattr(server_args, "rag_shared_hbm_mode", "off")
    if mode == "off":
        return None
    if _GLOBAL_POOL is not None:
        return _GLOBAL_POOL

    if not torch.cuda.is_available():
        raise RuntimeError("shared HBM requires CUDA")
    properties = torch.cuda.get_device_properties(gpu_id)
    device_name = properties.name
    is_h100 = "H100" in device_name
    # Advertised 80 GB A100 cards report slightly less than 80 GiB through
    # CUDA. Keep the lower-capacity A100 SKU out of the validated profile.
    is_a100_80gb = (
        "A100" in device_name and int(properties.total_memory) >= 75 * GIB
    )
    if not (is_h100 or is_a100_80gb):
        raise RuntimeError(
            "shared HBM v1 is validated only on H100 or A100 80GB; "
            f"found {device_name} with "
            f"{properties.total_memory / GIB:.1f} GiB"
        )
    _GLOBAL_POOL = SharedHBMPool(
        mode=mode,
        pool_gib=float(server_args.rag_shared_hbm_pool_gib),
        gpu_id=gpu_id,
        chunk_bytes=int(
            getattr(server_args, "rag_shared_hbm_chunk_mib", 8)
        )
        * MIB,
    )
    return _GLOBAL_POOL


class SharedHBMController:
    """Coordinates physical ownership with the logical SGLang allocator."""

    def __init__(
        self,
        *,
        pool: SharedHBMPool,
        token_allocator,
        ann_executor,
        lookahead_ms: float,
        min_residency_ms: float,
        kv_reserve_extents: int,
        rerank_prefetch_batches: int = 1,
        kv_publication_stream=None,
        ivf_reload_min_misses: int = 0,
        ivf_reload_max_lists: int = 0,
        kv_consumer_streams=(),
        async_native_map: bool = False,
        map_stage_extents: int = -1,
        kv_map_executor=None,
    ):
        self.pool = pool
        self.token_allocator = token_allocator
        self.kv_cache = token_allocator.get_kvcache()
        self.ann_executor = ann_executor
        self.lookahead_ms = max(1.0, float(lookahead_ms))
        self.min_residency_ms = max(0.0, float(min_residency_ms))
        self.kv_reserve_extents = max(0, int(kv_reserve_extents))
        self.rerank_prefetch_batches = max(
            1, int(rerank_prefetch_batches)
        )
        self.ivf_reload_min_misses = max(0, int(ivf_reload_min_misses))
        self.ivf_reload_max_lists = max(0, int(ivf_reload_max_lists))
        self.last_ivf_reload_observed_misses = 0
        self.ivf_reload_coalesced = 0
        self.ivf_reload_shift_gated = 0
        self.ivf_reload_no_shift_checks = 0
        self.ivf_reload_candidate_predictions = 0
        self.ivf_reload_candidate_gated = 0
        self.ivf_reload_candidate_lists_selected = 0
        self.ivf_reload_candidate_last_benefit_ms = 0.0
        self.ivf_reload_candidate_last_cost_ms = 0.0
        self.ivf_reload_async_deadline_bypasses = 0
        self.ivf_reload_stable_replacement_gated = 0
        self.ivf_reload_attempts = 0
        self.ivf_reload_lists_requested = 0
        self.kv_to_ivf_reclaims = 0
        self.kv_to_ivf_extents_reclaimed = 0
        self.kv_to_ivf_bytes_reclaimed = 0
        self.kv_to_ivf_reclaim_failures = 0
        self.last_kv_to_ivf_target_extents = 0
        self.last_kv_to_ivf_arena_bytes = 0
        self.kv_publication_stream = kv_publication_stream
        self.pending_ivf_reload = None
        online_recovery = os.getenv("RAGLEASE_IVF_ONLINE_RECOVERY", "0")
        if online_recovery not in {"0", "1"}:
            raise ValueError("RAGLEASE_IVF_ONLINE_RECOVERY must be 0 or 1")
        self.ivf_online_recovery_enabled = online_recovery == "1"
        phased_reload_cost = os.getenv("RAGLEASE_IVF_PHASED_RELOAD_COST", "0")
        if phased_reload_cost not in {"0", "1"}:
            raise ValueError("RAGLEASE_IVF_PHASED_RELOAD_COST must be 0 or 1")
        self.ivf_phased_reload_cost_enabled = phased_reload_cost == "1"
        self.ivf_reload_install_latency_ms = deque(maxlen=4096)
        self.ivf_reload_existing_arena_latency_ms = deque(maxlen=4096)
        self.ivf_online_recovery_active = False
        self.ivf_online_recovery_submissions = 0
        self.ivf_online_recovery_completions = 0
        self.ivf_online_recovery_lists = 0
        self.ivf_online_recovery_next_at = 0.0
        self.ivf_reload_async_submissions = 0
        self.ivf_reload_async_completions = 0
        self.ivf_reload_async_errors = 0
        self.ivf_reload_async_cpu_fallbacks = 0
        self.ivf_reload_async_latency_ms = deque(maxlen=4096)
        self.ivf_reload_async_list_counts = deque(maxlen=4096)
        self.kv_consumer_streams = self._deduplicate_streams(
            kv_consumer_streams
        )
        self.async_native_map = bool(async_native_map)
        async_reclaim_fence = os.getenv(
            "SGLANG_SHARED_HBM_ASYNC_RECLAIM_FENCE", "1"
        ).strip()
        if async_reclaim_fence not in {"0", "1"}:
            raise ValueError(
                "SGLANG_SHARED_HBM_ASYNC_RECLAIM_FENCE must be 0 or 1"
            )
        self.async_reclaim_fence_enabled = (
            self.async_native_map and async_reclaim_fence == "1"
        )
        fast_faiss_detach = os.getenv(
            "SGLANG_SHARED_HBM_FAST_FAISS_DETACH", "1"
        ).strip()
        if fast_faiss_detach not in {"0", "1"}:
            raise ValueError(
                "SGLANG_SHARED_HBM_FAST_FAISS_DETACH must be 0 or 1"
            )
        self.fast_faiss_detach_enabled = fast_faiss_detach == "1"
        async_ann_unmap = os.getenv(
            "SGLANG_SHARED_HBM_ASYNC_ANN_UNMAP", "1"
        ).strip()
        if async_ann_unmap not in {"0", "1"}:
            raise ValueError(
                "SGLANG_SHARED_HBM_ASYNC_ANN_UNMAP must be 0 or 1"
            )
        self.async_ann_unmap_configured = async_ann_unmap == "1"
        # Deferred native release is valid only behind the asynchronous
        # all-stream reclaim fence and the patched no-sync Faiss detach.
        self.async_ann_unmap_enabled = (
            self.async_ann_unmap_configured
            and self.async_reclaim_fence_enabled
            and self.fast_faiss_detach_enabled
        )
        ann_preserving_target_cap = os.getenv(
            "SGLANG_SHARED_HBM_ANN_PRESERVING_TARGET_CAP", "1"
        ).strip()
        if ann_preserving_target_cap not in {"0", "1"}:
            raise ValueError(
                "SGLANG_SHARED_HBM_ANN_PRESERVING_TARGET_CAP must be 0 or 1"
            )
        self.ann_preserving_target_cap_enabled = (
            ann_preserving_target_cap == "1"
        )
        partial_index_reclaim = os.getenv("RAGLEASE_PARTIAL_INDEX_RECLAIM", "0").strip()
        if partial_index_reclaim not in {"0", "1"}:
            raise ValueError("RAGLEASE_PARTIAL_INDEX_RECLAIM must be 0 or 1")
        self.partial_index_reclaim_enabled = partial_index_reclaim == "1"
        disjoint_map_ann_overlap = os.getenv(
            "SGLANG_SHARED_HBM_DISJOINT_MAP_ANN_OVERLAP", "1"
        ).strip()
        if disjoint_map_ann_overlap not in {"0", "1"}:
            raise ValueError(
                "SGLANG_SHARED_HBM_DISJOINT_MAP_ANN_OVERLAP must be 0 or 1"
            )
        self.ann_disjoint_map_overlap_enabled = (
            disjoint_map_ann_overlap == "1"
        )
        self.map_stage_extents_configured = int(map_stage_extents)
        self.map_stage_extents = resolve_shared_hbm_map_stage_extents(
            self.map_stage_extents_configured, self.pool.gpu_name
        )
        if self.map_stage_extents_configured == -1:
            logger.info(
                "Resolved automatic shared-HBM KV map stage: gpu=%s extents=%d",
                self.pool.gpu_name,
                self.map_stage_extents,
            )
        self.kv_map_executor = kv_map_executor
        self._owns_kv_map_executor = False
        if self.async_native_map and self.kv_map_executor is None:
            self.kv_map_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="shared-hbm-vmm",
            )
            self._owns_kv_map_executor = True
        self.pending_ann_reclaim = None
        self.pending_kv_native_map = None
        self.kv_native_map_ready = threading.Event()
        self.pending_kv_publications = deque()
        self.kv_growth_async_enqueue_latency_ms = deque(maxlen=4096)
        self.kv_growth_fence_latency_ms = deque(maxlen=4096)
        self.kv_growth_native_map_latency_ms = deque(maxlen=4096)
        self.kv_growth_zero_enqueue_latency_ms = deque(maxlen=4096)
        self.kv_growth_refresh_latency_ms = deque(maxlen=4096)
        self.kv_growth_publication_latency_ms = deque(maxlen=4096)
        self.kv_growth_first_publish_latency_ms = deque(maxlen=4096)
        self.kv_growth_total_latency_ms = deque(maxlen=4096)
        self.kv_growth_async_publications = 0
        self.kv_growth_consumer_stream_waits = 0
        self.kv_growth_reverse_drain_event_waits = 0
        self.last_transition_at = 0.0
        self.handoff_latency_ms = deque(maxlen=4096)
        self.transition_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self.prediction_error_tokens = deque(maxlen=4096)
        self.pending_predictions = deque(maxlen=4096)
        self.last_predicted_tokens = 0
        self.last_prediction_queued_prefill_tokens = 0
        self.last_prediction_prefill_batches = 1
        self.last_prediction_evictable_tokens = 0
        self.last_effective_lookahead_ms = self.lookahead_ms
        self.last_lookahead_observed_map_ms = 0.0
        self.rerank_prediction_calls = 0
        self.rerank_prefetches_with_ann_queued = 0
        self.rerank_all_queue_predictions = 0
        self.kv_backlog_committed = False
        self.kv_backlog_commits = 0
        self.kv_backlog_ann_fallbacks = 0
        self.ann_idle_since: Optional[float] = None
        self.ann_drain_quiescence_ms = max(
            20.0 * self.lookahead_ms, 1000.0
        )
        self.ann_drain_epoch_processed = False
        self.ann_drain_quiescent_predictions = 0
        self.kv_growth_ann_frontier_holds = 0
        self.kv_growth_ann_frontier_hold_tokens = 0
        self.kv_growth_ann_frontier_commits = 0
        self.ivf_idle_maintenance_ticks = 0
        self.ivf_idle_refill_attempts = 0
        self.ivf_idle_refill_no_candidate_holds = 0
        self.ivf_idle_free_capacity_refills = 0
        self.ivf_idle_free_capacity_lists = 0
        self.ivf_request_path_refill_skips = 0
        self.next_ivf_idle_refill_at = 0.0
        self.rerank_prefill_ready_calls = 0
        self.rerank_prefill_ready_tokens_total = 0
        self.last_rerank_extra_prefill_tokens = 0
        self.rerank_extra_prefill_tokens_total = 0
        self.last_kv_growth_at = 0.0
        self.last_kv_growth_required_tokens = 0
        self.last_kv_growth_raw_target_extents = 0
        self.last_kv_growth_target_extents = 0
        self.last_kv_growth_prefetch_extents = 0
        self.last_kv_growth_reason = "none"
        self.last_kv_growth_min_prefetch_extents = 0
        self.kv_growth_prefetched_extents_total = 0
        self.kv_growth_rate_tokens_per_s = 0.0
        self.kv_growth_device_synchronizations = 0
        self.kv_growth_stream_fences = 0
        self.kv_growth_streams_fenced = 0
        self.kv_growth_disjoint_ann_bypasses = 0
        self.kv_growth_disjoint_ann_bytes_preserved = 0
        self.kv_growth_ann_conflicts = 0
        self.kv_growth_ann_preserving_target_caps = 0
        self.kv_growth_ann_preserving_extents_saved = 0
        self.ann_disjoint_map_overlap_admissions = 0
        self.ann_disjoint_map_overlap_rejections = 0
        self.ann_disjoint_map_overlap_rows = 0
        self.kv_growth_growable_query_fallbacks = 0
        self.last_kv_growable_extents = 0
        self._warned_growable_query_fallback = False
        self.kv_growth_async_reclaim_fence_submissions = 0
        self.kv_growth_async_reclaim_fence_polls = 0
        self.kv_growth_async_reclaim_fence_ready = 0
        self.kv_growth_async_reclaim_streams_fenced = 0
        self.kv_growth_async_reclaim_fast_detach_uses = 0
        self.kv_growth_async_reclaim_fallback_detach_uses = 0
        self.kv_growth_async_reclaim_ready_latency_ms = deque(maxlen=4096)
        self.kv_growth_async_reclaim_detach_latency_ms = deque(maxlen=4096)
        self.kv_growth_scheduler_detach_latency_ms = deque(maxlen=4096)
        self.kv_growth_deferred_ann_release_submissions = 0
        self.kv_growth_deferred_ann_release_completions = 0
        self.kv_growth_deferred_ann_release_errors = 0
        self.kv_growth_worker_ann_release_latency_ms = deque(maxlen=4096)
        self.kv_growth_publication_event_waits = 0
        self.kv_growth_native_map_submissions = 0
        self.kv_growth_native_map_completions = 0
        self.kv_growth_native_map_errors = 0
        self.kv_growth_post_map_recovery_attempts = 0
        self.kv_growth_post_map_recovery_successes = 0
        self.kv_growth_post_map_recovery_errors = 0
        self.kv_growth_native_logical_gap_events = 0
        self.kv_growth_native_logical_gap_max_tokens = 0
        self.kv_growth_stages_completed = 0
        self.kv_growth_partial_publications = 0
        self.last_kv_growth_stage_count = 0
        self.kv_growth_completion_notifications = 0
        self.kv_growth_completion_safe_point_calls = 0
        self.kv_growth_completion_ready_hits = 0
        self.kv_growth_completion_reaction_latency_ms = deque(maxlen=4096)
        self.kv_map_worker_prewarm_attempted = False
        self.kv_map_worker_prewarm_succeeded = False
        self.kv_map_worker_prewarm_errors = 0
        self.kv_map_worker_prewarm_latency_ms = 0.0
        self.rerank_growth_requests = 0
        self.cpu_fallbacks = 0
        self.reload_waits = 0
        self.deadline_misses = 0
        self.microbatch_shrinks = 0
        if self.async_native_map and self._owns_kv_map_executor:
            self._prewarm_kv_map_worker()

    def _prewarm_kv_map_worker(self):
        """Create the worker's CUDA state before it enters the hot path."""

        self.kv_map_worker_prewarm_attempted = True
        started = time.perf_counter()
        try:
            future = self.kv_map_executor.submit(
                self.kv_cache.warm_shared_hbm_native_worker
            )
            future.result()
            self.kv_map_worker_prewarm_succeeded = True
        except Exception:
            self.kv_map_worker_prewarm_errors += 1
            logger.warning(
                "Failed to prewarm the shared-HBM VMM worker; the first "
                "mapping may pay CUDA thread initialization cost",
                exc_info=True,
            )
        finally:
            self.kv_map_worker_prewarm_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

    @staticmethod
    def _deduplicate_streams(streams):
        result = []
        seen = set()
        for stream in streams or ():
            pointer = int(stream.cuda_stream)
            if pointer in seen:
                continue
            seen.add(pointer)
            result.append(stream)
        return tuple(result)

    def _observe_pending_kv_publications(self):
        while self.pending_kv_publications:
            publication = self.pending_kv_publications[0]
            if not publication["ready"].query():
                break
            self.pending_kv_publications.popleft()

    def _drain_pending_kv_publications(self):
        while self.pending_kv_publications:
            publication = self.pending_kv_publications.popleft()
            publication["ready"].synchronize()
            self.kv_growth_reverse_drain_event_waits += 1

    def _publish_mapped_capacity(
        self,
        old_capacity: int,
        new_capacity: int,
        zero_ready,
    ):
        logical_capacity = int(
            self.token_allocator.current_mapped_capacity()
        )
        if logical_capacity not in {int(old_capacity), int(new_capacity)}:
            raise RuntimeError(
                "shared-HBM allocator capacity changed while publishing: "
                f"expected={old_capacity} or {new_capacity}, "
                f"actual={logical_capacity}"
            )
        already_published = logical_capacity == int(new_capacity)
        if already_published:
            # A prior attempt committed the allocator but failed while wiring
            # its publication event to every consumer. A device fence is the
            # conservative recovery boundary: it covers zeroing, allocator GPU
            # tensor updates, and any subset of consumer waits already queued.
            torch.cuda.synchronize(self.pool.gpu_id)
            self.kv_growth_publication_event_waits += 1
            return

        publication_stream = self.kv_publication_stream
        try:
            if publication_stream is None:
                # Compatibility path for non-scheduler callers. The validated
                # H100 runtime always supplies a publication stream.
                zero_ready.synchronize()
                self.token_allocator.publish_mapped_capacity(
                    old_capacity, new_capacity
                )
                publication_ready = torch.cuda.Event()
                publication_ready.record(
                    torch.cuda.current_stream(self.pool.gpu_id)
                )
                publication_ready.synchronize()
                self.kv_growth_publication_event_waits += 1
                return

            with torch.cuda.stream(publication_stream):
                publication_stream.wait_event(zero_ready)
                # TokenToKVPoolAllocator builds its free-page suffix with CUDA
                # arange/cat. The publication event must therefore be recorded
                # after allocator publication on this same stream.
                self.token_allocator.publish_mapped_capacity(
                    old_capacity, new_capacity
                )
                publication_ready = torch.cuda.Event()
                publication_ready.record(publication_stream)

            publication_pointer = int(publication_stream.cuda_stream)
            consumer_waits = 0
            for stream in self.kv_consumer_streams:
                if int(stream.cuda_stream) == publication_pointer:
                    continue
                stream.wait_event(publication_ready)
                consumer_waits += 1

            self.pending_kv_publications.append(
                {
                    "ready": publication_ready,
                    "zero_ready": zero_ready,
                    "capacity": int(new_capacity),
                }
            )
            self.kv_growth_async_publications += 1
            self.kv_growth_consumer_stream_waits += consumer_waits
        except Exception:
            committed_capacity = int(
                self.token_allocator.current_mapped_capacity()
            )
            if committed_capacity != int(new_capacity):
                raise
            # Publication committed, but its event fan-out did not complete.
            # Synchronizing here closes the safety gap before the scheduler can
            # allocate any newly visible page. Treat this as an immediately
            # recovered post-map error rather than retrying allocator commit.
            self.kv_growth_post_map_recovery_attempts += 1
            self.kv_growth_post_map_recovery_errors += 1
            torch.cuda.synchronize(self.pool.gpu_id)
            self.kv_growth_publication_event_waits += 1
            self.kv_growth_post_map_recovery_successes += 1
            logger.warning(
                "Recovered shared-HBM allocator publication after a "
                "post-commit CUDA event/wait failure with a device fence",
                exc_info=True,
            )

    def _complete_kv_growth(
        self,
        pending: dict[str, Any],
        new_capacity: int,
        zero_ready,
        native_map_latency_ms: float,
        *,
        final_stage: bool = True,
    ) -> bool:
        post_map = pending.get("post_map")
        if post_map is None:
            post_map = {}
            pending["post_map"] = post_map
        if not bool(post_map.get("latencies_recorded", False)):
            self.kv_growth_native_map_latency_ms.append(
                float(native_map_latency_ms)
            )
            self.kv_growth_zero_enqueue_latency_ms.append(
                float(
                    getattr(
                        self.kv_cache,
                        "shared_hbm_last_zero_enqueue_latency_ms",
                        0.0,
                    )
                )
            )
            self.kv_growth_refresh_latency_ms.append(
                float(
                    getattr(
                        self.kv_cache,
                        "shared_hbm_last_refresh_latency_ms",
                        0.0,
                    )
                )
            )
            post_map["latencies_recorded"] = True
        if not bool(post_map.get("published", False)):
            publication_started = time.perf_counter()
            self._publish_mapped_capacity(
                pending["old_capacity"],
                new_capacity,
                zero_ready,
            )
            self.kv_growth_publication_latency_ms.append(
                (time.perf_counter() - publication_started) * 1000.0
            )
            post_map["published"] = True
            post_map["phase"] = "published"
        if not bool(post_map.get("stage_accounted", False)):
            self.kv_growth_stages_completed += 1
            pending["stages_completed"] = (
                int(pending.get("stages_completed", 0)) + 1
            )
            if pending["stages_completed"] == 1:
                self.kv_growth_first_publish_latency_ms.append(
                    (time.perf_counter() - pending["started"]) * 1000.0
                )
            post_map["stage_accounted"] = True
        if not final_stage:
            if not bool(post_map.get("partial_accounted", False)):
                self.kv_growth_partial_publications += 1
                post_map["partial_accounted"] = True
            logger.info(
                "Shared-HBM ANN->KV stage published: stage=%s "
                "capacity=%s mapped_extents=%s final_target_extents=%s "
                "first_publish_ms=%.3f",
                pending["stages_completed"],
                new_capacity,
                pending["mapped_extents"],
                pending["target"],
                self.kv_growth_first_publish_latency_ms[-1],
            )
            return True

        if not bool(post_map.get("growth_accounted", False)):
            self.last_transition_at = time.perf_counter()
            self.last_kv_growth_at = self.last_transition_at
            self.last_kv_growth_required_tokens = int(
                pending["required_tokens"]
            )
            self.last_kv_growth_raw_target_extents = int(
                pending["raw_target"]
            )
            self.last_kv_growth_target_extents = int(pending["target"])
            self.last_kv_growth_prefetch_extents = int(
                pending["prefetch_extents"]
            )
            self.last_kv_growth_reason = str(pending["reason"])
            self.last_kv_growth_min_prefetch_extents = int(
                pending["min_prefetch_extents"]
            )
            self.kv_growth_prefetched_extents_total += int(
                pending["prefetch_extents"]
            )
            self._record("ann_to_kv")
            total_latency_ms = (
                self.last_transition_at - pending["started"]
            ) * 1000.0
            self.handoff_latency_ms.append(total_latency_ms)
            self.kv_growth_total_latency_ms.append(total_latency_ms)
            profile_key = pending.get('transition_profile_key')
            if profile_key is not None:
                history = getattr(self,'kv_growth_profile_samples',None)
                if history is None:
                    history = self.kv_growth_profile_samples = deque(maxlen=1024)
                history.append((profile_key,float(total_latency_ms)))
            self.last_kv_growth_stage_count = int(
                pending["stages_completed"]
            )
            post_map["total_latency_ms"] = total_latency_ms
            post_map["growth_accounted"] = True
        else:
            total_latency_ms = float(post_map["total_latency_ms"])
        completed = self.pool.snapshot()
        logger.info(
            "Shared-HBM ANN->KV grow complete: capacity=%s "
            "mapped_extents=%s kv_bytes=%s free_bytes=%s "
            "latency_ms=%.3f async_native_map=%s stages=%s",
            new_capacity,
            completed.kv_mapped_extents,
            completed.kv_bytes,
            completed.free_bytes,
            total_latency_ms,
            self.async_native_map,
            self.last_kv_growth_stage_count,
        )
        return True

    def _release_ann_then_map_shared_hbm_extents_native(
        self,
        owner: SharedHBMOwner,
        target_extents: int,
    ):
        """Release one deferred ANN lease, then run the first native map."""

        if owner != SharedHBMOwner.ANN_SCRATCH:
            self.kv_growth_deferred_ann_release_errors += 1
            raise RuntimeError(
                f"unsupported deferred shared-HBM owner: {owner!r}"
            )

        def release_then_map():
            release_started = time.perf_counter()
            try:
                self.pool.release_ann(owner)
                if self._faiss_scratch_release_pending():
                    raise RuntimeError(
                        "native ANN_SCRATCH release returned while the pool "
                        "remained in DETACHING state"
                    )
            except Exception:
                self.kv_growth_deferred_ann_release_errors += 1
                raise
            else:
                self.kv_growth_deferred_ann_release_completions += 1
            finally:
                self.kv_growth_worker_ann_release_latency_ms.append(
                    (time.perf_counter() - release_started) * 1000.0
                )
            return self.kv_cache.map_shared_hbm_extents_native(
                target_extents
            )

        # SharedHBMPool uses an RLock, so its public release/map methods can
        # re-enter it. Holding the same lock across both operations prevents a
        # direct pool caller or teardown from interleaving in the ownership
        # handoff. Test doubles without the lock retain the same serial worker
        # ordering.
        pool_lock = getattr(self.pool, "_lock", None)
        if pool_lock is None:
            return release_then_map()
        with pool_lock:
            return release_then_map()

    def _submit_kv_native_map_stage(self, pending: dict[str, Any]):
        current_extents = int(pending["mapped_extents"])
        final_target = int(pending["target"])
        if (
            self.map_stage_extents <= 0
            or int(pending.get("stages_completed", 0)) > 0
        ):
            stage_target = final_target
        else:
            demand_target = int(
                pending.get("raw_target", current_extents)
            )
            stage_target = min(
                final_target,
                max(
                    demand_target,
                    current_extents + self.map_stage_extents,
                ),
            )
        if stage_target <= current_extents:
            raise RuntimeError(
                "shared-HBM KV map stage made no forward progress: "
                f"current={current_extents}, target={final_target}"
            )
        pending["stage_target"] = stage_target
        self.kv_native_map_ready.clear()
        deferred_release_owner = pending.get("deferred_release_owner")
        if deferred_release_owner is not None:
            if (
                int(pending.get("stages_completed", 0)) != 0
                or bool(pending.get("deferred_release_submitted", False))
            ):
                raise RuntimeError(
                    "deferred ANN release must run exactly once before the "
                    "first shared-HBM KV native map stage"
                )
            try:
                future = self.kv_map_executor.submit(
                    self._release_ann_then_map_shared_hbm_extents_native,
                    SharedHBMOwner(int(deferred_release_owner)),
                    stage_target,
                )
            except Exception:
                self.kv_growth_deferred_ann_release_errors += 1
                raise
            pending["deferred_release_submitted"] = True
            future.shared_hbm_deferred_release_owner = int(
                deferred_release_owner
            )
            self.kv_growth_deferred_ann_release_submissions += 1
        else:
            future = self.kv_map_executor.submit(
                self.kv_cache.map_shared_hbm_extents_native,
                stage_target,
            )
        pending["future"] = future
        future.add_done_callback(self._notify_kv_native_map_done)
        self.kv_growth_native_map_submissions += 1
        logger.info(
            "Shared-HBM ANN->KV native map stage submitted: "
            "stage=%s current_extents=%s stage_target_extents=%s "
            "final_target_extents=%s",
            int(pending.get("stages_completed", 0)) + 1,
            current_extents,
            stage_target,
            final_target,
        )

    def _notify_kv_native_map_done(self, future):
        """Wake scheduler safe points without touching CUDA or the allocator."""

        future.shared_hbm_completed_at = time.perf_counter()
        pending = self.pending_kv_native_map
        if pending is None or pending.get("future") is not future:
            return
        self.kv_growth_completion_notifications += 1
        self.kv_native_map_ready.set()

    def _faiss_scratch_release_pending(self) -> bool:
        state = getattr(self.pool, "faiss_scratch_detaching", None)
        if state is not None:
            return bool(state())
        return bool(
            getattr(self.pool, "_faiss_scratch_detaching", False)
        )

    def _release_ann_leases_for_kv_growth(
        self,
        snapshot: SharedHBMSnapshot,
        *,
        faiss_streams_synchronized: bool = False,
        defer_ann_scratch_release: bool = False,
        target_extents: Optional[int] = None,
    ) -> tuple[str, Optional[SharedHBMOwner]]:
        """Detach ANN mappings after their stream fence is host-visible."""

        faiss_detach_mode = "none"
        deferred_release_owner = None
        scratch_release_pending = self._faiss_scratch_release_pending()
        if snapshot.ann_scratch_bytes > 0 or scratch_release_pending:
            if (
                defer_ann_scratch_release
                and scratch_release_pending
            ):
                # A prior worker/submit failure left Faiss detached but the
                # native owner recoverably leased. Retry the release without
                # treating this as an ordinary unbound arena.
                deferred_release_owner = SharedHBMOwner.ANN_SCRATCH
                faiss_detach_mode = "deferred_retry"
            elif defer_ann_scratch_release:
                detach_after_sync = getattr(
                    self.pool,
                    "detach_faiss_scratch_after_sync",
                    None,
                )
                used_fast_detach = bool(
                    detach_after_sync is not None
                    and detach_after_sync(self.ann_executor)
                )
                if used_fast_detach:
                    deferred_release_owner = SharedHBMOwner.ANN_SCRATCH
                    faiss_detach_mode = "fast_deferred"
                else:
                    # Older patched Faiss builds retain the existing
                    # synchronizing detach+release safety path.
                    used_fast_detach = (
                        self.pool.unbind_faiss_scratch_after_sync(
                            self.ann_executor
                        )
                    )
                    faiss_detach_mode = (
                        "fast" if used_fast_detach else "fallback"
                    )
            elif faiss_streams_synchronized:
                used_fast_detach = (
                    self.pool.unbind_faiss_scratch_after_sync(
                        self.ann_executor
                    )
                )
                faiss_detach_mode = (
                    "fast" if used_fast_detach else "fallback"
                )
            else:
                self.pool.unbind_faiss_scratch(self.ann_executor)
                faiss_detach_mode = "synchronous"
        if (
            self.pool.mode == "ivf"
            and snapshot.ann_index_bytes > 0
        ):
            if getattr(self, "partial_index_reclaim_enabled", False):
                if target_extents is None:
                    raise RuntimeError("partial reclaim requires an explicit KV target")
                retained = self.pool.index_prefix_for_kv(target_extents)
                self.ann_executor.trim_shared_ivf_arena(self.pool, retained)
                self._record("ivf_partial_evict")
            else:
                self.ann_executor.release_shared_ivf_arena(self.pool)
                self._record("ivf_evict")
        return faiss_detach_mode, deferred_release_owner

    def _poll_pending_ann_reclaim(self) -> bool:
        """Advance an event-gated ANN detach from a scheduler safe point."""

        pending = self.pending_ann_reclaim
        if pending is None:
            return False

        self.kv_growth_async_reclaim_fence_polls += 1
        ready_event = pending["fence"].ready_event
        if not ready_event.query():
            return False

        self.kv_growth_async_reclaim_fence_ready += 1
        self.kv_growth_async_reclaim_ready_latency_ms.append(
            max(
                0.0,
                (time.perf_counter() - pending["submitted_at"]) * 1000.0,
            )
        )
        growth = pending["growth"]
        detach_started = time.perf_counter()
        try:
            # query() above is the ownership boundary: before it succeeds,
            # neither Faiss pointers nor their native mappings may change.
            fence = pending["fence"]
            can_fast_detach = (
                self.fast_faiss_detach_enabled
                and bool(getattr(fence, "fast_detach_safe", False))
            )
            (
                detach_mode,
                deferred_release_owner,
            ) = self._release_ann_leases_for_kv_growth(
                growth["snapshot"],
                target_extents=growth["target"],
                faiss_streams_synchronized=can_fast_detach,
                defer_ann_scratch_release=(
                    self.async_ann_unmap_enabled and can_fast_detach
                ),
            )
            if growth["snapshot"].ann_scratch_bytes > 0:
                if detach_mode in {"fast", "fast_deferred"}:
                    self.kv_growth_async_reclaim_fast_detach_uses += 1
                elif detach_mode in {"fallback", "synchronous"}:
                    self.kv_growth_async_reclaim_fallback_detach_uses += 1
            scheduler_detach_latency_ms = (
                time.perf_counter() - detach_started
            ) * 1000.0
            self.kv_growth_async_reclaim_detach_latency_ms.append(
                scheduler_detach_latency_ms
            )
            self.kv_growth_scheduler_detach_latency_ms.append(
                scheduler_detach_latency_ms
            )
            growth["deferred_release_owner"] = deferred_release_owner
            growth["deferred_release_submitted"] = False
            growth["snapshot"] = self.pool.snapshot()
            self.pending_kv_native_map = growth
            self._submit_kv_native_map_stage(growth)
            self.pending_ann_reclaim = None
            logger.info(
                "Shared-HBM async ANN reclaim ready: streams=%s "
                "ready_latency_ms=%.3f detach_latency_ms=%.3f "
                "faiss_detach=%s target_extents=%s",
                pending["fence"].stream_count,
                self.kv_growth_async_reclaim_ready_latency_ms[-1],
                self.kv_growth_async_reclaim_detach_latency_ms[-1],
                detach_mode,
                growth["target"],
            )
            return True
        except Exception:
            self.pending_ann_reclaim = None
            self.pending_kv_native_map = None
            self.kv_native_map_ready.clear()
            self._record_failure("ann_reclaim_async")
            logger.exception(
                "Failed to detach ANN leases after asynchronous stream fence"
            )
            return False

    def poll_kv_growth_completion(self) -> bool:
        """Advance ANN reclaim/native-map work at a scheduler safe point."""

        ivf_reloaded = self._poll_pending_ivf_reload()
        reclaimed = self._poll_pending_ann_reclaim()
        if reclaimed:
            # A newly submitted native stage cannot already have notified this
            # scheduler safe point; consume it on a later loop iteration.
            return True
        if self.pending_kv_native_map is None:
            # A completion callback can race with a legacy direct poll. Drop
            # the resulting stale wakeup without charging an active safe point.
            self.kv_native_map_ready.clear()
            return ivf_reloaded
        self.kv_growth_completion_safe_point_calls += 1
        if not self.kv_native_map_ready.is_set():
            return False
        return self._poll_pending_kv_native_map()

    def _poll_pending_kv_native_map(self) -> bool:
        pending = self.pending_kv_native_map
        if pending is None:
            self.kv_native_map_ready.clear()
            return False
        post_map = pending.get("post_map")
        if post_map is None:
            if not pending["future"].done():
                return False
            self.kv_native_map_ready.clear()
            self.kv_growth_completion_ready_hits += 1
            completed_at = getattr(
                pending["future"], "shared_hbm_completed_at", None
            )
            if completed_at is not None:
                self.kv_growth_completion_reaction_latency_ms.append(
                    max(
                        0.0,
                        (time.perf_counter() - completed_at) * 1000.0,
                    )
                )
            future = pending["future"]
            try:
                new_capacity, native_map_latency_ms = future.result()
            except Exception:
                self.pending_kv_native_map = None
                self.kv_native_map_ready.clear()
                self.kv_growth_native_map_errors += 1
                self._record_failure("ann_to_kv_async")
                logger.exception(
                    "Failed to complete asynchronous shared-HBM KV native map"
                )
                return False
            layout = self.pool.kv_layout
            mapped_extents = max(
                0,
                (
                    int(new_capacity)
                    - layout.static_capacity_tokens
                )
                // layout.tokens_per_extent,
            )
            post_map = {
                "new_capacity": int(new_capacity),
                "native_map_latency_ms": float(native_map_latency_ms),
                "mapped_extents": mapped_extents,
                "final_stage": mapped_extents >= int(pending["target"]),
                "phase": "validate",
                "zero_ready": None,
                "recovery_required": False,
            }
            pending["post_map"] = post_map
            pending["mapped_extents"] = mapped_extents
            self.kv_growth_native_map_completions += 1

        self.kv_native_map_ready.clear()
        recovery_attempt = bool(post_map.get("recovery_required", False))
        if recovery_attempt:
            self.kv_growth_post_map_recovery_attempts += 1
        try:
            future = pending["future"]
            deferred_release_owner = getattr(
                future,
                "shared_hbm_deferred_release_owner",
                None,
            )
            if post_map["phase"] == "validate":
                if deferred_release_owner is not None:
                    if int(pending.get("deferred_release_owner", -1)) != int(
                        deferred_release_owner
                    ):
                        raise RuntimeError(
                            "shared-HBM deferred ANN release owner changed "
                            "while its native stage was in flight"
                        )
                    pending["deferred_release_owner"] = None
                post_map["phase"] = "finalize"
            elif deferred_release_owner is not None and pending.get(
                "deferred_release_owner"
            ) is not None:
                if int(pending.get("deferred_release_owner", -1)) != int(
                    deferred_release_owner
                ):
                    raise RuntimeError(
                        "shared-HBM deferred ANN release owner changed "
                        "while its native stage was in flight"
                    )
                pending["deferred_release_owner"] = None
            if post_map["phase"] == "finalize":
                post_map["zero_ready"] = (
                    self.kv_cache.finalize_shared_hbm_extents_map(
                        pending["old_capacity"],
                        post_map["new_capacity"],
                    )
                )
                post_map["phase"] = "publish"
            self._complete_kv_growth(
                pending,
                post_map["new_capacity"],
                post_map["zero_ready"],
                post_map["native_map_latency_ms"],
                final_stage=post_map["final_stage"],
            )
            if recovery_attempt:
                self.kv_growth_post_map_recovery_successes += 1
            post_map["recovery_required"] = False
            if post_map["final_stage"]:
                self.pending_kv_native_map = None
                return True

            pending["old_capacity"] = int(post_map["new_capacity"])
            # The worker is idle here, so refreshing the cached metrics
            # snapshot cannot contend on the native pool mutex.
            pending["snapshot"] = self.pool.snapshot()
            pending.pop("post_map", None)
            self._submit_kv_native_map_stage(pending)
            return False
        except Exception:
            post_map["recovery_required"] = True
            post_map["last_error_phase"] = str(post_map.get("phase", "unknown"))
            self.kv_growth_post_map_recovery_errors += 1
            self.kv_growth_native_map_errors += 1
            gap_tokens = self._pending_native_logical_gap_tokens()
            if gap_tokens > 0:
                self.kv_growth_native_logical_gap_events += 1
                self.kv_growth_native_logical_gap_max_tokens = max(
                    self.kv_growth_native_logical_gap_max_tokens,
                    gap_tokens,
                )
            # Keep the completed native result live and wake another scheduler
            # safe point. The retry resumes at validate/finalize/publish and
            # never resubmits the native stage or its deferred ANN release.
            self.kv_native_map_ready.set()
            self._record_failure("ann_to_kv_post_map")
            logger.exception(
                "Failed to finalize/publish asynchronous shared-HBM KV "
                "growth; retaining native/logical recovery state"
            )
            return False

    def _pending_native_logical_gap_tokens(self) -> int:
        pending = self.pending_kv_native_map
        if pending is None:
            return 0
        post_map = pending.get("post_map")
        if post_map is None:
            return 0
        try:
            logical_capacity = int(
                self.token_allocator.current_mapped_capacity()
            )
        except Exception:
            return 0
        return max(
            0,
            int(post_map.get("new_capacity", logical_capacity))
            - logical_capacity,
        )

    def _native_map_inflight(self) -> bool:
        self._poll_pending_kv_native_map()
        return self.pending_kv_native_map is not None

    def _record(self, key: str):
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

    def _record_failure(self, key: str):
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

    def _kv_growable_extents(
        self, snapshot: SharedHBMSnapshot, *, record: bool = True
    ) -> int:
        """Return the KV prefix that can grow without reclaiming ANN pages."""
        ann_bytes = int(
            snapshot.ann_scratch_bytes + snapshot.ann_index_bytes
        )
        if ann_bytes <= 0:
            growable_extents = int(snapshot.kv_total_extents)
            if record:
                self.last_kv_growable_extents = growable_extents
            return growable_extents

        query = getattr(self.pool, "kv_growable_extents", None)
        if query is None:
            growable_extents = int(snapshot.kv_mapped_extents)
            if record:
                self.kv_growth_growable_query_fallbacks += 1
        else:
            try:
                growable_extents = int(query())
            except (AttributeError, RuntimeError):
                growable_extents = int(snapshot.kv_mapped_extents)
                if record:
                    self.kv_growth_growable_query_fallbacks += 1
                if record and not self._warned_growable_query_fallback:
                    logger.warning(
                        "Shared-HBM native growable-prefix query is "
                        "unavailable; retaining conservative ANN reclamation"
                    )
                    self._warned_growable_query_fallback = True

        if not (
            snapshot.kv_mapped_extents
            <= growable_extents
            <= snapshot.kv_total_extents
        ):
            raise RuntimeError(
                "invalid shared-HBM growable KV extent count: "
                f"mapped={snapshot.kv_mapped_extents}, "
                f"growable={growable_extents}, "
                f"total={snapshot.kv_total_extents}"
            )
        if record:
            self.last_kv_growable_extents = growable_extents
        return growable_extents

    def _kv_growth_requires_ann_reclaim(
        self,
        snapshot: SharedHBMSnapshot,
        target_extents: int,
        *,
        growable_extents: Optional[int] = None,
    ) -> bool:
        """Check whether the complete KV target intersects live ANN pages."""

        if self._faiss_scratch_release_pending():
            # A failed deferred attempt may already have unmapped every native
            # chunk while leaving the conservative shadow state DETACHING.
            # Retry the idempotent release before any later KV native map.
            self.kv_growth_ann_conflicts += 1
            return True
        ann_bytes = int(
            snapshot.ann_scratch_bytes + snapshot.ann_index_bytes
        )
        if ann_bytes <= 0:
            return False
        if growable_extents is None:
            growable_extents = self._kv_growable_extents(snapshot)
        if int(target_extents) <= growable_extents:
            self.kv_growth_disjoint_ann_bypasses += 1
            self.kv_growth_disjoint_ann_bytes_preserved += ann_bytes
            logger.info(
                "Shared-HBM ANN lease retained during disjoint KV growth: "
                "target_extents=%s growable_extents=%s preserved_bytes=%s",
                target_extents,
                growable_extents,
                ann_bytes,
            )
            return False

        self.kv_growth_ann_conflicts += 1
        return True

    @staticmethod
    def ann_scratch_request_bytes(
        num_queries: int, max_inflight: int = 1
    ) -> int:
        # Measured with the production A100 IVFPQ index at nprobe=2048:
        # 1/2/4-8 rows need at least 128/256/384 MiB. Keep 32-64 MiB of
        # headroom so strict external allocation remains deterministic.
        #
        # Faiss' stack allocator can reuse temporary addresses once a launch
        # returns. That is stream-ordered only for a single in-flight search;
        # multiple ANN streams therefore retain the proven 1.5 GiB arena.
        if int(max_inflight) > 1:
            return DEFAULT_FAISS_TEMP_BYTES
        num_queries = max(1, int(num_queries))
        if num_queries == 1:
            return FAISS_SCRATCH_ONE_QUERY_BYTES
        if num_queries == 2:
            return FAISS_SCRATCH_TWO_QUERY_BYTES
        if num_queries <= 8:
            return FAISS_SCRATCH_MAX_MICROBATCH_BYTES
        extra_groups = math.ceil((num_queries - 8) / 8)
        return min(
            DEFAULT_FAISS_TEMP_BYTES,
            FAISS_SCRATCH_MAX_MICROBATCH_BYTES
            + extra_groups * (128 * MIB),
        )

    @classmethod
    def ann_scratch_capacity_rows(
        cls, available_bytes: int, configured_limit: int
    ) -> int:
        configured_limit = max(1, int(configured_limit))
        for rows in range(configured_limit, 0, -1):
            if cls.ann_scratch_request_bytes(rows) <= available_bytes:
                return rows
        return 0

    @classmethod
    def fit_ann_workspace_budget(cls, proposed_rows: int, budget_bytes: int,
                                 *, max_inflight: int = 1) -> ANNWorkspaceFit:
        """Pure fit for one compute proposal, shared by both admission modes.

        Caller supplies the scratch budget AFTER accounting for the same
        proposal's KV/index obligations. Do not add predicted KV releases or
        pending reload bytes here. Mapping/readiness belongs to safe projection.
        Uses the runtime's existing scratch contract, not a new measured model.
        """
        for value in (proposed_rows, budget_bytes, max_inflight):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("ANN limits must be nonnegative integers")
        if max_inflight == 0:
            raise ValueError("max_inflight must be positive")
        for rows in range(proposed_rows, 0, -1):
            required = cls.ann_scratch_request_bytes(rows, max_inflight)
            if required <= budget_bytes:
                return ANNWorkspaceFit(proposed_rows, rows, required, budget_bytes, max_inflight)
        return ANNWorkspaceFit(proposed_rows, 0, 0, budget_bytes, max_inflight)

    def _faiss_mapping_state(self) -> tuple[bool, tuple[int, int], int]:
        """Read the Faiss pointer identity without querying native VMM state."""

        state = getattr(self.pool, "faiss_mapping_state", None)
        if state is not None:
            return state()
        return (
            bool(getattr(self.pool, "_faiss_bound", False)),
            tuple(getattr(self.pool, "_last_faiss_mapping", (0, 0))),
            int(getattr(self.pool, "_faiss_mapping_generation", 0)),
        )

    def _disjoint_map_bound_scratch_bytes(
        self, requested_bytes: int
    ) -> int:
        """Prove that an in-flight KV map is disjoint from bound scratch.

        This is deliberately a Python-only identity check.  In particular it
        must not call ``snapshot()``, bind or resize Faiss while the native
        worker owns the VMM mutex.
        """

        pending = self.pending_kv_native_map
        if (
            not self.ann_disjoint_map_overlap_enabled
            or self.pool.mode != "scratch"
            or self.pending_ann_reclaim is not None
            or pending is None
            or bool(pending.get("ann_reclaim_required", True))
            or not bool(pending.get("ann_disjoint", False))
            or not bool(pending.get("faiss_bound", False))
        ):
            return 0

        bound, mapping, generation = self._faiss_mapping_state()
        captured_mapping = tuple(
            pending.get("faiss_mapping", (0, 0))
        )
        captured_generation = int(
            pending.get("faiss_mapping_generation", -1)
        )
        if (
            not bound
            or generation != captured_generation
            or mapping != captured_mapping
            or len(mapping) != 2
            or int(mapping[0]) <= 0
            or int(mapping[1]) <= 16
            or int(mapping[1])
            != int(pending.get("faiss_scratch_bytes", -1))
        ):
            return 0

        required_bytes = min(
            max(0, int(requested_bytes)),
            int(self.pool.total_bytes),
        )
        if required_bytes <= 16 or int(mapping[1]) < required_bytes:
            return 0
        return int(mapping[1])

    def limit_ann_microbatch_rows(
        self, scheduler, configured_limit: int
    ) -> int:
        configured_limit = max(0, int(configured_limit))
        if configured_limit == 0:
            # A fitted joint lease may intentionally admit no ANN work. Do
            # not resurrect a one-row launch or touch allocator observations.
            return 0
        if self.pending_ann_reclaim is not None:
            self.ann_disjoint_map_overlap_rejections += 1
            self.microbatch_shrinks += 1
            return 0
        if self._faiss_scratch_release_pending():
            self.ann_disjoint_map_overlap_rejections += 1
            self.microbatch_shrinks += 1
            return 0
        if self._native_map_inflight():
            requested_bytes = self.ann_scratch_request_bytes(1)
            available_bytes = self._disjoint_map_bound_scratch_bytes(
                requested_bytes
            )
            if available_bytes > 0:
                limited = self.ann_scratch_capacity_rows(
                    available_bytes, configured_limit
                )
                if limited > 0:
                    if limited < configured_limit:
                        self.microbatch_shrinks += 1
                    return limited
            self.ann_disjoint_map_overlap_rejections += 1
            self.microbatch_shrinks += 1
            return 0
        snapshot = self.pool.snapshot()
        if getattr(
            scheduler, "_rag_ann_gpu_work_active", lambda: False
        )():
            memory_limit = max(
                1,
                self.ann_scratch_capacity_rows(
                    snapshot.ann_scratch_bytes, configured_limit
                ),
            )
            limited = min(configured_limit, int(memory_limit))
            if limited < configured_limit:
                self.microbatch_shrinks += 1
            return limited
        potential_bytes = (
            snapshot.free_bytes
            + snapshot.ann_scratch_bytes
            + snapshot.ann_index_bytes
        )
        if snapshot.kv_mapped_extents > 0:
            extent_bytes = (
                snapshot.kv_bytes // snapshot.kv_mapped_extents
            )
            predicted = self.predict_kv_tokens(scheduler)
            target_extents = max(
                0,
                math.ceil(
                    max(
                        0,
                        predicted
                        - snapshot.kv_static_capacity_tokens,
                    )
                    / snapshot.kv_tokens_per_extent
                ),
            )
            reclaimable = max(
                0, snapshot.kv_mapped_extents - target_extents
            )
            potential_bytes += reclaimable * extent_bytes
        memory_limit = max(
            1,
            self.ann_scratch_capacity_rows(
                min(DEFAULT_FAISS_TEMP_BYTES, potential_bytes),
                configured_limit,
            ),
        )
        limited = min(configured_limit, int(memory_limit))
        if limited < configured_limit:
            self.microbatch_shrinks += 1
        return limited

    def _residency_guard(self) -> bool:
        return (
            time.perf_counter() - self.last_transition_at
        ) * 1000.0 < self.min_residency_ms

    def _actual_used_tokens(self) -> int:
        current_capacity = max(
            1, self.token_allocator.current_mapped_capacity()
        )
        return max(
            0, current_capacity - self.token_allocator.available_size()
        )

    @staticmethod
    def _queued_prefill_tokens(
        scheduler, admission_batches: int
    ) -> int:
        max_prefill_tokens = max(
            1, int(getattr(scheduler, "max_prefill_tokens", 1))
        )
        token_limit = max_prefill_tokens * max(
            1, int(admission_batches)
        )
        queued_prefill = 0
        for req in getattr(scheduler, "waiting_queue", ()):
            queued_prefill += max(
                0,
                int(
                    getattr(req, "extend_input_len", 0)
                    or len(getattr(req, "origin_input_ids", ()))
                ),
            )
            if queued_prefill >= token_limit:
                break
        return queued_prefill

    @staticmethod
    def _all_queued_prefill_tokens(scheduler) -> int:
        return sum(
            max(
                0,
                int(
                    getattr(req, "extend_input_len", 0)
                    or len(getattr(req, "origin_input_ids", ()))
                ),
            )
            for req in getattr(scheduler, "waiting_queue", ())
        )

    def _effective_kv_lookahead_ms(self, *, record: bool = True) -> float:
        """Cover both demand forecast time and the observed VMM pipeline.

        A fixed 50 ms horizon was shorter than the measured H100 mapping
        barrier. Once samples exist, forecast far enough ahead for the p95
        native-map/clear/publication path, with a small variance margin.
        """

        native_p95_ms = self._percentile(
            self.kv_growth_native_map_latency_ms, 95
        )
        zero_p95_ms = self._percentile(
            self.kv_growth_zero_enqueue_latency_ms, 95
        )
        publication_p95_ms = self._percentile(
            self.kv_growth_publication_latency_ms, 95
        )
        phase_pipeline_ms = (
            native_p95_ms + zero_p95_ms + publication_p95_ms
        )
        total_p95_ms = self._percentile(
            self.kv_growth_total_latency_ms, 95
        )
        first_publish_p95_ms = self._percentile(
            self.kv_growth_first_publish_latency_ms, 95
        )
        if self.map_stage_extents > 0 and first_publish_p95_ms > 0.0:
            # Once staged publication is active, current demand is covered by
            # the first stage. The coalesced prefetch completion may only be
            # observed after a long model forward and must not inflate the
            # demand forecast horizon.
            observed_ms = first_publish_p95_ms
        else:
            observed_ms = max(phase_pipeline_ms, total_p95_ms)
        safety_ms = max(
            KV_ADAPTIVE_LOOKAHEAD_MIN_SAFETY_MS,
            observed_ms * 0.10,
        )
        cap_ms = max(
            self.lookahead_ms, KV_ADAPTIVE_LOOKAHEAD_MAX_MS
        )
        effective_ms = min(
            cap_ms,
            max(
                self.lookahead_ms,
                observed_ms + safety_ms
                if observed_ms > 0.0
                else self.lookahead_ms,
            ),
        )
        if record:
            self.last_lookahead_observed_map_ms = observed_ms
            self.last_effective_lookahead_ms = effective_ms
        return effective_ms

    @staticmethod
    def _evictable_kv_tokens(scheduler) -> int:
        tree_cache = getattr(scheduler, "tree_cache", None)
        if tree_cache is None:
            return 0
        try:
            value = tree_cache.evictable_size()
        except (AttributeError, RuntimeError):
            return 0
        if isinstance(value, tuple):
            value = sum(value)
        return max(0, int(value))

    def predict_kv_tokens(
        self,
        scheduler,
        *,
        prefill_admission_batches: int = 1,
        additional_prefill_tokens: int = 0,
        include_all_queued_prefill: bool = False,
        record: bool = True,
    ) -> int:
        now = time.perf_counter()
        allocated = self._actual_used_tokens()
        evictable = min(
            allocated, self._evictable_kv_tokens(scheduler)
        )
        used = allocated - evictable
        if record:
            self.last_prediction_evictable_tokens = evictable
        if record:
            while (
                self.pending_predictions
                and self.pending_predictions[0][0] <= now
            ):
                _, prediction = self.pending_predictions.popleft()
                self.prediction_error_tokens.append(prediction - used)

        prefill_admission_batches = max(
            1, int(prefill_admission_batches)
        )
        queued_prefill = (
            self._all_queued_prefill_tokens(scheduler)
            if include_all_queued_prefill
            else self._queued_prefill_tokens(
                scheduler, prefill_admission_batches
            )
        )
        queued_prefill += max(0, int(additional_prefill_tokens))
        if record:
            self.last_prediction_queued_prefill_tokens = queued_prefill
            self.last_prediction_prefill_batches = prefill_admission_batches

        running_batch = getattr(scheduler, "running_batch", None)
        decode_bs = len(getattr(running_batch, "reqs", ()))
        predicted_step_ms = None
        predictor = getattr(scheduler, "_rag_ann_decode_prediction_ms", None)
        if predictor is not None:
            predicted_step_ms = predictor()
        if predicted_step_ms is None or predicted_step_ms <= 0:
            predicted_step_ms = max(1.0, self.lookahead_ms)
        effective_lookahead_ms = self._effective_kv_lookahead_ms(
            record=record
        )
        decode_steps = max(
            1, math.ceil(effective_lookahead_ms / predicted_step_ms)
        )
        reserve = (
            self.kv_reserve_extents
            * self.pool.kv_layout.tokens_per_extent
        )
        predicted = used + queued_prefill + decode_bs * decode_steps + reserve
        if record:
            self.last_predicted_tokens = int(predicted)
            self.pending_predictions.append(
                (now + effective_lookahead_ms / 1000.0, int(predicted))
            )
        return int(predicted)

    def _adaptive_kv_growth_target(
        self,
        *,
        required_tokens: int,
        raw_target_extents: int,
        now: float,
        min_prefetch_extents: int = KV_GROWTH_QUANTUM_EXTENTS,
        record: bool = True,
    ) -> tuple[int, int]:
        """Add enough mapped headroom to amortize the next VMM handoff.

        The minimum headroom is one max-prefill admission quantum.  Observed
        prediction shortfalls and the demand rate since the previous growth
        can raise that headroom, while the cap prevents a transient burst
        from eagerly consuming the entire ANN-owned pool.
        """

        layout = self.pool.kv_layout
        underprediction_tokens = [
            max(0.0, -float(error))
            for error in self.prediction_error_tokens
            if error < 0
        ]
        error_headroom_extents = math.ceil(
            self._percentile(underprediction_tokens, 95)
            / layout.tokens_per_extent
        )

        growth_rate_tokens_per_s = 0.0
        if (
            self.last_kv_growth_at > 0.0
            and now > self.last_kv_growth_at
            and required_tokens > self.last_kv_growth_required_tokens
        ):
            growth_rate_tokens_per_s = (
                required_tokens - self.last_kv_growth_required_tokens
            ) / (now - self.last_kv_growth_at)
        if record:
            self.kv_growth_rate_tokens_per_s = growth_rate_tokens_per_s

        # Cover at least the forecast horizon and one observed p95 mapping
        # barrier. During steady bursts this prevents a new scheduler tick
        # from immediately paying another device-wide correctness barrier.
        rate_horizon_ms = self._effective_kv_lookahead_ms(record=record)
        rate_headroom_extents = math.ceil(
            growth_rate_tokens_per_s
            * rate_horizon_ms
            / 1000.0
            / layout.tokens_per_extent
        )

        min_prefetch_extents = min(
            KV_GROWTH_MAX_PREFETCH_EXTENTS,
            max(KV_GROWTH_QUANTUM_EXTENTS, int(min_prefetch_extents)),
        )
        min_prefetch_extents = (
            math.ceil(
                min_prefetch_extents / KV_GROWTH_QUANTUM_EXTENTS
            )
            * KV_GROWTH_QUANTUM_EXTENTS
        )
        prefetch_extents = max(
            min_prefetch_extents,
            error_headroom_extents,
            rate_headroom_extents,
        )
        prefetch_extents = min(
            KV_GROWTH_MAX_PREFETCH_EXTENTS,
            math.ceil(prefetch_extents / KV_GROWTH_QUANTUM_EXTENTS)
            * KV_GROWTH_QUANTUM_EXTENTS,
        )

        rounded_required_extents = (
            math.ceil(raw_target_extents / KV_GROWTH_QUANTUM_EXTENTS)
            * KV_GROWTH_QUANTUM_EXTENTS
        )
        target_extents = min(
            layout.total_extents,
            rounded_required_extents + prefetch_extents,
        )
        effective_prefetch_extents = max(
            0, target_extents - rounded_required_extents
        )
        return target_extents, effective_prefetch_extents

    def _kv_plan_source_identity(self, snapshot) -> tuple:
        """Read under scheduler serialization; no polling or native mutation."""
        return (tuple(sorted(vars(snapshot).items())),
                int(self.token_allocator.current_mapped_capacity()),
                # Equal mapped capacity does not mean equal occupancy: decode
                # allocation/request completion can change the fitted demand
                # between proposal and commit without a VMM mapping change.
                int(self.token_allocator.available_size()),
                self._faiss_mapping_state())

    def make_kv_continuation(self, required_tokens: int, *, event_id: int) -> KVCapacityPlan:
        """Observe current backing only; never choose an elastic target.

        Demand is for work fitted to CURRENT limits, not the rejected proposal.
        Waiting here does not suspend already-backed compatible decode; it
        means no new capacity guarantee is available from this observation.
        """
        if type(event_id) is not int or event_id < 0:
            raise ValueError('Valid continuation event required')
        if any(getattr(self, key, None) is not None for key in
               ('pending_ivf_reload', 'pending_ann_reclaim', 'pending_kv_native_map')):
            return KVCapacityPlan(None, 0, 0, False, 'wait',
                                  'ownership transition pending', event_id)
        if self._faiss_scratch_release_pending():
            return KVCapacityPlan(None, 0, 0, False, 'wait',
                                  'scratch detach pending', event_id)
        snapshot = self.pool.snapshot()
        layout = self.pool.kv_layout
        geometry = preview_kv_capacity_geometry(
            required_tokens=required_tokens,
            static_capacity_tokens=int(layout.static_capacity_tokens),
            tokens_per_extent=int(layout.tokens_per_extent),
            total_extents=int(layout.total_extents),
            mapped_extents=int(snapshot.kv_mapped_extents),
            published_capacity_tokens=int(self.token_allocator.current_mapped_capacity()))
        ready = (geometry.publication_gap_tokens == 0
                 and geometry.within_pool_capacity
                 and required_tokens <= geometry.published_capacity_tokens)
        return KVCapacityPlan(geometry, geometry.mapped_extents, 0, False,
            'ready' if ready else 'wait', 'retain current allocation', event_id,
            self._kv_plan_source_identity(snapshot), 0)

    def plan_kv_capacity(self, required_tokens: int, *,
                         min_prefetch_extents: int = KV_GROWTH_QUANTUM_EXTENTS,
                         now: Optional[float] = None,
                         event_id: Optional[int] = None) -> KVCapacityPlan:
        """Read-only proposal for demand already derived from a compute lease.

        Never poll completion, release Faiss, map pages, publish capacity, or
        record prediction samples here. Caller must snapshot/revalidate native
        ownership before committing; this return value is NOT a release permit.
        Pending transitions are not optimistically counted as available memory.
        """
        # The native worker may hold the VMM mutex. Even a snapshot query can
        # block; do not query it just to decide that an existing transition
        # must finish. A wait plan carries no fresh capacity guarantee.
        if any(getattr(self, key, None) is not None for key in
               ("pending_ivf_reload", "pending_ann_reclaim", "pending_kv_native_map")):
            return KVCapacityPlan(None, 0, 0, False, "wait", "ownership transition pending")
        if self._faiss_scratch_release_pending():
            return KVCapacityPlan(None, 0, 0, False, "wait", "scratch detach pending")
        snapshot = self.pool.snapshot()
        layout = self.pool.kv_layout
        geometry = preview_kv_capacity_geometry(
            required_tokens=required_tokens,
            static_capacity_tokens=int(layout.static_capacity_tokens),
            tokens_per_extent=int(layout.tokens_per_extent),
            total_extents=int(layout.total_extents),
            mapped_extents=int(snapshot.kv_mapped_extents),
            published_capacity_tokens=int(self.token_allocator.current_mapped_capacity()))
        source_identity = self._kv_plan_source_identity(snapshot)
        def result(status, reason, target=None, prefetch=0, reclaim=False):
            return KVCapacityPlan(geometry, geometry.target_extents if target is None else target,
                                  prefetch, reclaim, status, reason, event_id,
                                  source_identity, min_prefetch_extents)
        if not geometry.within_pool_capacity:
            return result("unsupported", "demand exceeds total KV capacity")
        if geometry.publication_gap_tokens != 0:
            return result("wait", "native and published capacities differ")
        if required_tokens <= geometry.published_capacity_tokens:
            return result("ready", "already published", target=geometry.mapped_extents)
        target, prefetch = self._adaptive_kv_growth_target(
            required_tokens=required_tokens, raw_target_extents=geometry.target_extents,
            now=time.perf_counter() if now is None else now,
            min_prefetch_extents=min_prefetch_extents, record=False)
        growable = self._kv_growable_extents(snapshot, record=False)
        if self.ann_preserving_target_cap_enabled and geometry.target_extents <= growable < target:
            target = growable
            prefetch = max(0, target - geometry.target_extents)
        if (getattr(self, "partial_index_reclaim_enabled", False)
                and snapshot.ann_index_bytes > 0 and geometry.target_extents > growable):
            target = geometry.target_extents
            prefetch = 0
        reclaim = snapshot.ann_scratch_bytes + snapshot.ann_index_bytes > 0 and target > growable
        return result("grow", "requires selected transition and publication", target, prefetch, reclaim)

    def preview_ann_retention_for_kv(self, plan: KVCapacityPlan) -> Optional[ANNRetentionPreview]:
        """Mirror the selected KV transition's actual ANN detach/trim policy.

        Do not trim, unbind, poll completion, reload or choose a different KV
        target. The native prefix query is geometry only. Expected retained
        index bytes are not proof its lists are loaded/usable; scratch rebinding
        and request routing still belong to complete joint planning/readiness.
        """
        if plan.status not in ('ready','grow') or plan.geometry is None:
            return None
        if any(getattr(self,key,None) is not None for key in
               ('pending_ivf_reload','pending_ann_reclaim','pending_kv_native_map')):
            return None
        if self._faiss_scratch_release_pending():
            return None
        snapshot = self.pool.snapshot()
        if (getattr(snapshot,'moving_bytes',0) != 0 or not plan.source_identity
                or self._kv_plan_source_identity(snapshot) != plan.source_identity):
            return None
        index = int(snapshot.ann_index_bytes)
        scratch = int(snapshot.ann_scratch_bytes)
        rebind = False
        if plan.status == 'grow' and plan.reclaim_ann:
            rebind = scratch > 0
            scratch = 0
            if index > 0:
                if getattr(self.pool,'mode',None) != 'ivf':
                    return None
                if getattr(self,'partial_index_reclaim_enabled',False):
                    retained = self.pool.index_prefix_for_kv(plan.target_extents)
                    if type(retained) is not int or not 0 <= retained <= index:
                        raise ValueError('Invalid retained IVF prefix geometry')
                    index = retained
                else:
                    index = 0
        return ANNRetentionPreview(index,scratch,rebind,plan.event_id,plan.source_identity)

    def preview_joint_ann_workspace(self, plan: KVCapacityPlan, proposed_rows: int,
                                    *, max_inflight: int = 1,
                                    retain_allocation: bool = False) -> Optional[ANNWorkspacePreview]:
        """Fit scratch AFTER this same plan's KV/index ownership obligations.

        Expected post-transition bytes are not current free space. No binding,
        reclaim, index reload or second KV target search occurs. Physical
        mapping success and list readiness remain activation prerequisites.
        """
        retained = self.preview_ann_retention_for_kv(plan)
        if retained is None:
            return None
        snapshot = self.pool.snapshot()
        if self._kv_plan_source_identity(snapshot) != retained.source_identity:
            return None
        extent = getattr(self.pool, 'kv_extent_bytes', None)
        if type(extent) is not int or extent <= 0:
            return None
        if snapshot.kv_mapped_extents*extent != snapshot.kv_bytes:
            return None
        target_kv = plan.target_extents*extent
        remaining = snapshot.total_bytes-target_kv-retained.retained_index_bytes-snapshot.moving_bytes
        if remaining < 0 or snapshot.chunk_bytes <= 0:
            return None
        budget = remaining//snapshot.chunk_bytes*snapshot.chunk_bytes
        bound, mapping, generation = self._faiss_mapping_state()
        if retain_allocation:
            # Safe continuation may shrink work but cannot acquire new scratch
            # or treat spare pool bytes as an already bound Faiss workspace.
            bound_bytes = (mapping[1] if bound and len(mapping) == 2
                and type(mapping[0]) is int and mapping[0] > 0
                and type(mapping[1]) is int and mapping[1] >= 0 else 0)
            budget = min(budget,retained.retained_scratch_bytes,bound_bytes)
        fit = self.fit_ann_workspace_budget(proposed_rows,budget,max_inflight=max_inflight)
        bound_capacity_valid = (
            bound and len(mapping) == 2
            and type(mapping[0]) is int and mapping[0] > 0
            and type(mapping[1]) is int and mapping[1] >= fit.required_bytes)
        mapping_required = fit.fitted_rows > 0 and (
            not bound_capacity_valid or fit.required_bytes > retained.retained_scratch_bytes)
        return ANNWorkspacePreview(retained,fit,target_kv,budget,mapping_required)

    def plan_joint_hbm(self, required_tokens: int, proposed_ann_rows: int, *,
                       event_id: int, now: float, max_inflight: int = 1) -> JointHBMPlan:
        """One KV proposal plus scratch/index fitting, never recursive search."""
        if type(event_id) is not int or event_id < 0:
            raise ValueError('Actual decision event required')
        kv = self.plan_kv_capacity(required_tokens,event_id=event_id,now=now)
        if kv.status not in ('ready','grow'):
            return JointHBMPlan(kv,None,kv.status,event_id)
        ann = self.preview_joint_ann_workspace(kv,proposed_ann_rows,max_inflight=max_inflight)
        return JointHBMPlan(kv,ann,'available' if ann is not None else 'wait',event_id)

    def current_joint_hbm_continuation(self, required_tokens: int, ann_rows: int, *,
                                       event_id: int, max_inflight: int = 1) -> JointHBMPlan:
        """Retain allocations; do not choose elastic capacity or bind scratch.

        Existing compatible in-flight work still needs its own backing keys in
        the execution projection. A wait here is not a command to stop decode.
        """
        if type(event_id) is not int or event_id < 0:
            raise ValueError('Actual decision event required')
        kv = self.make_kv_continuation(required_tokens,event_id=event_id)
        if kv.status != 'ready':
            return JointHBMPlan(kv,None,kv.status,event_id)
        ann = self.preview_joint_ann_workspace(kv,ann_rows,max_inflight=max_inflight,
                                                retain_allocation=True)
        return JointHBMPlan(kv,ann,'available' if ann is not None else 'wait',event_id)

    def joint_transition_profile_inputs(self, plan: JointHBMPlan):
        """Read exact profile keys for one uncommitted native plan.

        No freezing/querying profiles, mapping or planning here. The caller
        supplies one previously frozen profile bundle for both target and keep.
        None rejects stale/pending state before a possibly blocking snapshot.
        """
        if plan.status != 'available' or plan.ann is None or plan.kv.status not in ('ready','grow'):
            return None
        if any(getattr(self,key,None) is not None for key in
               ('pending_ivf_reload','pending_ann_reclaim','pending_kv_native_map')):
            return None
        if self._faiss_scratch_release_pending():
            return None
        snapshot = self.pool.snapshot()
        if (not plan.kv.source_identity
                or self._kv_plan_source_identity(snapshot) != plan.kv.source_identity):
            return None
        grows = plan.kv.status == 'grow'
        kv_key = self.kv_transition_profile_key(snapshot,plan.kv.target_extents,plan.kv.reclaim_ann) if grows else None
        mapping_needed = plan.ann.fit.fitted_rows > 0 and plan.ann.mapping_required
        workspace_key = None
        if mapping_needed:
            bound,mapping,_ = self._faiss_mapping_state()
            if plan.ann.retention.scratch_rebind_required:
                # Actual detach resets both _faiss_bound and last mapping.
                bound,mapping = False,(0,0)
            if len(mapping)!=2 or type(mapping[1]) is not int or mapping[1]<0:
                return None
            workspace_key = (self.pool.mode,bool(bound),mapping[1],plan.ann.fit.required_bytes)
        return dict(kv_transition_required=grows,kv_key=kv_key,
                    workspace_mapping_required=mapping_needed,workspace_key=workspace_key)

    def joint_allocation_shape(self, plan: JointHBMPlan):
        """Read target ownership geometry, not a physical-ready certificate.

        Used for target/keep allocation equality in admission. Do not include
        event, query count, free-token occupancy or required KV tokens: those
        change without changing allocation. Scratch rounding follows native
        map_free_to_ann; actual mapping/publication remains a commit-time gate.
        """
        if self.joint_transition_profile_inputs(plan) is None:
            return None
        snapshot = self.pool.snapshot()
        ann = plan.ann
        if (self._kv_plan_source_identity(snapshot) != plan.kv.source_identity
                or ann.retention.source_identity != plan.kv.source_identity
                or ann.retention.event_id != plan.event_id
                or plan.kv.event_id != plan.event_id or snapshot.moving_bytes != 0):
            return None
        chunk = snapshot.chunk_bytes
        if type(chunk) is not int or chunk <= 0:
            raise ValueError('Actual shared-pool chunk geometry required')
        scratch = ann.retention.retained_scratch_bytes
        if ann.mapping_required:
            required = ann.fit.required_bytes
            if type(required) is not int or required <= 0 or ann.fit.fitted_rows <= 0:
                raise ValueError('Positive fitted scratch demand required')
            scratch = ((required + chunk - 1) // chunk) * chunk
            if scratch > ann.workspace_budget_bytes:
                return None
        if (ann.target_kv_bytes + ann.retention.retained_index_bytes + scratch
                > snapshot.total_bytes):
            return None
        return (self.pool.mode, snapshot.total_bytes, chunk, plan.kv.target_extents,
                ann.target_kv_bytes, ann.retention.retained_index_bytes, scratch)

    def joint_hbm_backing_ready(self, plan: JointHBMPlan) -> bool:
        """Read actual ownership after transitions; never bind, poll or replan.

        This is only the physical HBM leg of activation. The scheduler must
        additionally validate event/frontier/compute identity and exact index
        list routing. Equal index bytes alone do not establish list readiness.
        Pending native operations are checked BEFORE snapshot to avoid waiting
        on the worker's VMM mutex. Existing compatible decode is not gated here.
        """
        if plan.status != 'available' or plan.ann is None or plan.kv.geometry is None:
            return False
        if any(getattr(self, key, None) is not None for key in
               ('pending_ivf_reload', 'pending_ann_reclaim', 'pending_kv_native_map')):
            return False
        if self._faiss_scratch_release_pending():
            return False
        snapshot = self.pool.snapshot()
        ann = plan.ann
        if (snapshot.moving_bytes != 0
                or snapshot.kv_mapped_extents != plan.kv.target_extents
                or snapshot.kv_bytes != ann.target_kv_bytes
                or snapshot.ann_index_bytes != ann.retention.retained_index_bytes
                or self.token_allocator.current_mapped_capacity() < plan.kv.geometry.required_tokens):
            return False
        if ann.fit.fitted_rows == 0:
            return True
        bound, mapping, _ = self._faiss_mapping_state()
        return bool(bound and len(mapping) == 2
            and type(mapping[0]) is int and mapping[0] > 0
            and type(mapping[1]) is int and mapping[1] >= ann.fit.required_bytes
            and snapshot.ann_scratch_bytes >= ann.fit.required_bytes)

    def commit_joint_ann_workspace(self, plan: JointHBMPlan, scheduler, *,
                                   current_event_id: int, source_identity: tuple) -> str:
        """Bind only the selected workspace after its KV/index transition.

        Caller validates the selected lease and captures a fresh post-transition
        identity under scheduler serialization. No future wait, index eviction,
        KV replanning or fallback binding is performed here. A failed native
        call consumes this event too; exceptions retain their original evidence.
        This does not publish the whole-lease activation marker.
        """
        if (type(current_event_id) is not int or current_event_id < 0
                or plan.event_id != current_event_id or plan.kv.event_id != current_event_id):
            return 'stale'
        last = getattr(self, '_joint_scratch_committed_event_id', -1)
        if current_event_id <= last:
            return 'duplicate' if current_event_id == last else 'stale'
        if plan.status != 'available' or plan.ann is None or plan.kv.geometry is None:
            return 'unsupported'
        if any(getattr(self, key, None) is not None for key in
               ('pending_ivf_reload', 'pending_ann_reclaim', 'pending_kv_native_map')):
            return 'wait'
        if self._faiss_scratch_release_pending():
            return 'wait'
        snapshot = self.pool.snapshot()
        if not source_identity or self._kv_plan_source_identity(snapshot) != source_identity:
            return 'stale'
        ann = plan.ann
        if (snapshot.moving_bytes != 0
                or snapshot.kv_mapped_extents != plan.kv.target_extents
                or snapshot.kv_bytes != ann.target_kv_bytes
                or snapshot.ann_index_bytes != ann.retention.retained_index_bytes
                or self.token_allocator.current_mapped_capacity() < plan.kv.geometry.required_tokens):
            return 'wait'
        if self.joint_hbm_backing_ready(plan):
            return 'ready'
        active = getattr(scheduler, '_rag_ann_gpu_work_active', None)
        if active is None or active():
            return 'wait'  # No pointer replacement while readers hold it.
        required = ann.fit.required_bytes
        if (self.ann_executor is None or ann.fit.fitted_rows <= 0
                or type(required) is not int or required <= 0
                or required > ann.workspace_budget_bytes
                or required > snapshot.free_bytes + snapshot.ann_scratch_bytes):
            return 'unsupported'
        self._joint_scratch_committed_event_id = current_event_id
        mapped = self.pool.bind_faiss_scratch(self.ann_executor, requested_bytes=required)
        if mapped < required:
            return 'failed'
        return 'ready' if self.joint_hbm_backing_ready(plan) else 'failed'

    def observe_committed_kv_capacity_plan(self, plan: KVCapacityPlan, *, current_event_id: int) -> str:
        """Read completion of this exact submitted KV leg without resubmission.

        This certifies published capacity only, not current request occupancy,
        index placement, compute activation, or permission to dispatch work.
        Native polling remains owned by the existing runtime completion path.
        """
        if (type(current_event_id) is not int or current_event_id != plan.event_id
                or current_event_id != getattr(self,'_joint_kv_committed_event_id',None)
                or getattr(self,'_joint_kv_committed_plan',None) is not plan):
            return 'stale'
        if getattr(self,'_joint_kv_submission_status',None) != 'submitted':
            return 'failed'
        if any(getattr(self,key,None) is not None for key in
               ('pending_ivf_reload','pending_ann_reclaim','pending_kv_native_map')):
            return 'wait'
        if self._faiss_scratch_release_pending():
            return 'wait'
        snapshot=self.pool.snapshot()
        layout=self.pool.kv_layout
        geometry=preview_kv_capacity_geometry(
            required_tokens=plan.geometry.required_tokens,
            static_capacity_tokens=int(layout.static_capacity_tokens),
            tokens_per_extent=int(layout.tokens_per_extent),total_extents=int(layout.total_extents),
            mapped_extents=int(snapshot.kv_mapped_extents),
            published_capacity_tokens=int(self.token_allocator.current_mapped_capacity()))
        if geometry.mapped_extents != plan.target_extents:
            return 'stale'
        if (geometry.publication_gap_tokens != 0 or not geometry.within_pool_capacity
                or geometry.required_tokens > geometry.published_capacity_tokens):
            return 'wait'
        return 'ready'

    def commit_kv_capacity_plan(self, plan: KVCapacityPlan, *, current_event_id: int) -> str:
        """Commit an already selected plan under scheduler serialization.

        Returns ready/submitted/stale/duplicate/wait/unsupported/failed. Only `ready`
        means capacity was published. This is not a whole-joint-lease commit:
        the adapter must also revalidate request frontiers and compute safety.
        No retries, proposal selection, or capacity replanning are performed.
        Event IDs must increase monotonically for this controller's lifetime;
        they identify actual scheduler decisions, not retry attempts.
        """
        if (type(current_event_id) is not int or type(plan.event_id) is not int
                or current_event_id < 0 or plan.event_id != current_event_id):
            return "stale"
        # Consume a growth attempt once, even if its executor returns failure
        # before publishing a pending handle. A callback/re-entry must not retry
        # side effects in the same decision. A new real decision may replan.
        last_event = getattr(self, "_joint_kv_committed_event_id", -1)
        if current_event_id <= last_event:
            return "duplicate" if current_event_id == last_event else "stale"
        if plan.status not in ("ready", "grow") or plan.geometry is None:
            return "unsupported" if plan.status == "unsupported" else "wait"
        if any(getattr(self, key, None) is not None for key in
               ("pending_ivf_reload", "pending_ann_reclaim", "pending_kv_native_map")):
            return "stale"
        if self._faiss_scratch_release_pending():
            return "stale"
        snapshot = self.pool.snapshot()
        if not plan.source_identity or self._kv_plan_source_identity(snapshot) != plan.source_identity:
            return "stale"
        g = plan.geometry
        if not g.within_pool_capacity or g.publication_gap_tokens != 0:
            return "unsupported"
        if plan.status == "ready":
            return "ready" if g.required_tokens <= g.published_capacity_tokens else "unsupported"
        if not g.required_extents <= plan.target_extents <= self.pool.kv_layout.total_extents:
            return "unsupported"
        growable = self._kv_growable_extents(snapshot, record=False)
        reclaim = (snapshot.ann_scratch_bytes + snapshot.ann_index_bytes > 0
                   and plan.target_extents > growable)
        if reclaim != plan.reclaim_ann:
            return "stale"
        self._joint_kv_committed_event_id = current_event_id
        self._joint_kv_committed_plan = plan
        self._joint_kv_submission_status = 'failed'
        ok = self._execute_kv_growth_target(
            snapshot=snapshot, required_tokens=g.required_tokens,
            raw_target=g.required_extents, target=plan.target_extents,
            prefetch_extents=plan.prefetch_extents,
            min_prefetch_extents=plan.min_prefetch_extents,
            ann_reclaim_required=plan.reclaim_ann, reason="joint_selected")
        self._joint_kv_submission_status = 'submitted' if ok else 'failed'
        return self._joint_kv_submission_status

    def ensure_kv_capacity(
        self,
        required_tokens: int,
        *,
        min_prefetch_extents: int = KV_GROWTH_QUANTUM_EXTENTS,
        reason: str = "tick",
    ) -> bool:
        self._poll_pending_ivf_reload()
        if getattr(self, "pending_ivf_reload", None) is not None:
            # The refill was admitted only after reserving the predicted KV
            # capacity. Keep ownership stable until its publication finishes.
            return True
        self._poll_pending_ann_reclaim()
        self._poll_pending_kv_native_map()
        self._observe_pending_kv_publications()
        if (
            self.pending_ann_reclaim is not None
            or self.pending_kv_native_map is not None
        ):
            return True
        layout = self.pool.kv_layout
        snapshot = self.pool.snapshot()
        logical_capacity = int(
            self.token_allocator.current_mapped_capacity()
        )
        geometry = preview_kv_capacity_geometry(
            required_tokens=int(required_tokens),
            static_capacity_tokens=int(layout.static_capacity_tokens),
            tokens_per_extent=int(layout.tokens_per_extent),
            total_extents=int(layout.total_extents),
            mapped_extents=int(snapshot.kv_mapped_extents),
            published_capacity_tokens=logical_capacity,
        )
        # Legacy commit behavior is preserved here. The new joint planner must
        # additionally reject an out-of-pool proposal before any commit.
        raw_target = geometry.target_extents
        native_capacity = geometry.native_capacity_tokens
        if native_capacity > logical_capacity:
            gap_tokens = native_capacity - logical_capacity
            self.kv_growth_native_logical_gap_events += 1
            self.kv_growth_native_logical_gap_max_tokens = max(
                self.kv_growth_native_logical_gap_max_tokens,
                gap_tokens,
            )
            self._record_failure("kv_native_logical_gap_untracked")
            logger.error(
                "Refusing shared-HBM KV ensure with an untracked "
                "native/logical capacity gap: native_capacity=%s "
                "logical_capacity=%s gap_tokens=%s",
                native_capacity,
                logical_capacity,
                gap_tokens,
            )
            return False
        if raw_target <= snapshot.kv_mapped_extents:
            return True
        target, prefetch_extents = self._adaptive_kv_growth_target(
            required_tokens=required_tokens,
            raw_target_extents=raw_target,
            now=time.perf_counter(),
            min_prefetch_extents=min_prefetch_extents,
        )
        growable_extents = self._kv_growable_extents(snapshot)
        if (
            self.ann_preserving_target_cap_enabled
            and raw_target <= growable_extents < target
        ):
            saved_extents = target - growable_extents
            target = growable_extents
            prefetch_extents = max(0, target - raw_target)
            self.kv_growth_ann_preserving_target_caps += 1
            self.kv_growth_ann_preserving_extents_saved += saved_extents
            logger.info(
                "Shared-HBM adaptive KV target capped to preserve ANN lease: "
                "raw_target_extents=%s capped_target_extents=%s "
                "saved_extents=%s",
                raw_target,
                target,
                saved_extents,
            )
        if (
            getattr(self, "partial_index_reclaim_enabled", False)
            and snapshot.ann_index_bytes > 0
            and raw_target > growable_extents
        ):
            # On an actual index conflict, reclaim only the extents needed
            # now. Speculative 16-extent prefetch would defeat fine trimming.
            target = raw_target
            prefetch_extents = 0
        ann_reclaim_required = self._kv_growth_requires_ann_reclaim(
            snapshot,
            target,
            growable_extents=growable_extents,
        )

        return self._execute_kv_growth_target(
            snapshot=snapshot, required_tokens=required_tokens,
            raw_target=raw_target, target=target,
            prefetch_extents=prefetch_extents,
            min_prefetch_extents=min_prefetch_extents,
            ann_reclaim_required=ann_reclaim_required, reason=reason)

    def kv_transition_profile_key(self, snapshot, target: int, reclaim_ann: bool):
        """Exact source/target geometry and fixed transition implementation.

        Capture BEFORE staged mapping mutates pending.snapshot/mapped_extents.
        No workload-conditioned precision is claimed; in-flight reader load
        remains a limitation of this geometry-conditioned whole-transition key.
        """
        settings = tuple(getattr(self,name,None) for name in (
            'map_stage_extents','async_native_map','partial_index_reclaim_enabled',
            'async_reclaim_fence_enabled','async_ann_unmap_enabled'))
        mode = getattr(getattr(self,'pool',None),'mode',None)
        if mode not in ('scratch','ivf') or any(value is None for value in settings):
            return None
        return (mode,int(snapshot.kv_mapped_extents),int(target),
                int(snapshot.ann_scratch_bytes),int(snapshot.ann_index_bytes),
                bool(reclaim_ann),*settings)

    def _execute_kv_growth_target(
        self, *, snapshot: SharedHBMSnapshot, required_tokens: int,
        raw_target: int, target: int, prefetch_extents: int,
        min_prefetch_extents: int, ann_reclaim_required: bool, reason: str,
    ) -> bool:
        """Execute an already selected target without another planning pass.

        Internal only: caller must revalidate ownership at the scheduler's
        serialized commit boundary. True means transition submitted/completed,
        NOT permission to use unpublished KV pages. Legacy ensure is currently
        the only caller; joint commit revalidation must precede new callers.
        """
        started = time.perf_counter()
        try:
            logger.info(
                "Shared-HBM ANN->KV grow start: required_tokens=%s "
                "raw_target_extents=%s target_extents=%s "
                "prefetch_extents=%s growth_rate_tokens_per_s=%.1f "
                "reason=%s min_prefetch_extents=%s "
                "mapped_extents=%s free_bytes=%s scratch_bytes=%s "
                "index_bytes=%s",
                required_tokens,
                raw_target,
                target,
                prefetch_extents,
                self.kv_growth_rate_tokens_per_s,
                reason,
                min_prefetch_extents,
                snapshot.kv_mapped_extents,
                snapshot.free_bytes,
                snapshot.ann_scratch_bytes,
                snapshot.ann_index_bytes,
            )
            old_capacity = self.token_allocator.current_mapped_capacity()
            (
                faiss_bound,
                faiss_mapping,
                faiss_mapping_generation,
            ) = self._faiss_mapping_state()
            pending = {
                "started": started,
                "transition_profile_key": (
                    self.kv_transition_profile_key(snapshot,target,ann_reclaim_required)
                    if hasattr(self,'kv_transition_profile_key') else None),
                "old_capacity": old_capacity,
                "mapped_extents": snapshot.kv_mapped_extents,
                "required_tokens": int(required_tokens),
                "raw_target": raw_target,
                "target": target,
                "prefetch_extents": prefetch_extents,
                "reason": str(reason),
                "min_prefetch_extents": int(min_prefetch_extents),
                "snapshot": snapshot,
                "stages_completed": 0,
                "ann_reclaim_required": bool(ann_reclaim_required),
                "ann_disjoint": bool(
                    not ann_reclaim_required
                    and snapshot.ann_scratch_bytes > 0
                ),
                "faiss_bound": faiss_bound,
                "faiss_mapping": faiss_mapping,
                "faiss_mapping_generation": faiss_mapping_generation,
                "faiss_scratch_bytes": int(snapshot.ann_scratch_bytes),
                # Filled only by the async reclaim-ready path after Faiss has
                # entered DETACHING. The single native worker consumes this
                # owner before the first map stage and then clears it.
                "deferred_release_owner": None,
                "deferred_release_submitted": False,
            }
            if self.ann_executor is not None and ann_reclaim_required:
                fence_started = time.perf_counter()
                # Candidate-ready covers the Faiss default stream, while Faiss
                # may still have work on its two alternate streams. Aggregate
                # those known users on the KV maintenance stream instead of
                # synchronizing unrelated work across the whole device.
                maintenance_stream = getattr(
                    self.kv_cache,
                    "shared_hbm_maintenance_stream",
                    None,
                )
                if maintenance_stream is None:
                    raise RuntimeError(
                        "zero-sync shared HBM requires a KV maintenance stream"
                    )
                submit_async_fence = getattr(
                    self.ann_executor,
                    "submit_shared_hbm_reclaim_fence",
                    None,
                )
                if (
                    self.async_reclaim_fence_enabled
                    and submit_async_fence is not None
                ):
                    fence = submit_async_fence(maintenance_stream)
                    self.pending_ann_reclaim = {
                        "growth": pending,
                        "fence": fence,
                        "submitted_at": fence_started,
                    }
                    self.kv_growth_async_reclaim_fence_submissions += 1
                    self.kv_growth_async_reclaim_streams_fenced += int(
                        fence.stream_count
                    )
                    self.kv_growth_async_enqueue_latency_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    logger.info(
                        "Shared-HBM asynchronous ANN reclaim fence "
                        "submitted: streams=%s target_extents=%s "
                        "scheduler_enqueue_ms=%.3f",
                        fence.stream_count,
                        target,
                        self.kv_growth_async_enqueue_latency_ms[-1],
                    )
                    return True

                fenced = self.ann_executor.fence_shared_hbm_users(
                    maintenance_stream
                )
                self.kv_growth_stream_fences += 1
                self.kv_growth_streams_fenced += fenced
                self._release_ann_leases_for_kv_growth(snapshot, target_extents=target)
                pending["snapshot"] = self.pool.snapshot()
                self.kv_growth_fence_latency_ms.append(
                    (time.perf_counter() - fence_started) * 1000.0
                )
            if self.async_native_map:
                self.pending_kv_native_map = pending
                self._submit_kv_native_map_stage(pending)
                self.kv_growth_async_enqueue_latency_ms.append(
                    (time.perf_counter() - started) * 1000.0
                )
                logger.info(
                    "Shared-HBM ANN->KV native map submitted: "
                    "target_extents=%s stage_extents=%s "
                    "scheduler_enqueue_ms=%.3f",
                    target,
                    self.map_stage_extents,
                    self.kv_growth_async_enqueue_latency_ms[-1],
                )
                return True

            new_capacity, zero_ready = (
                self.kv_cache.map_shared_hbm_extents_async(target)
            )
            self.kv_growth_async_enqueue_latency_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            return self._complete_kv_growth(
                pending,
                new_capacity,
                zero_ready,
                float(
                    getattr(
                        self.kv_cache,
                        "shared_hbm_last_native_map_latency_ms",
                        0.0,
                    )
                ),
            )
        except Exception:
            self._record_failure("ann_to_kv")
            logger.exception("Failed to grow shared-HBM KV capacity")
            return False

    def before_ann_launch(
        self,
        scheduler,
        requested_bytes: int = DEFAULT_FAISS_TEMP_BYTES,
        deadline_slack_ms: Optional[float] = None,
        num_queries: int = 1,
    ) -> bool:
        if self.ann_executor is None:
            return False
        frontend = getattr(scheduler, 'rag_frontend_plan', None)
        event = getattr(frontend, 'joint_event_id', None)
        if getattr(scheduler,'rag_joint_event_input_provider',None) is not None and event is None:
            return False  # No first joint admission yet; do not run legacy planning.
        if event is not None:
            # Joint admission owns all allocation changes. Never fall through
            # to legacy scratch binding, index eviction, or KV prediction.
            # The whole-lease activator must publish this exact selected plan;
            # an event marker alone cannot establish physical backing.
            selected = getattr(scheduler, 'rag_joint_active_memory_plan', None)
            if (type(event) is not int
                    or getattr(scheduler, 'rag_joint_active_event_id', None) != event
                    or selected is None or selected.event_id != event
                    or selected.ann is None or not getattr(frontend, 'allow_ann', False)):
                return False
            fit = selected.ann.fit
            if (type(num_queries) is not int or not 0 < num_queries <= fit.fitted_rows
                    or type(requested_bytes) is not int or not 0 < requested_bytes <= fit.required_bytes):
                return False
            return self.joint_hbm_backing_ready(selected)
        self._poll_pending_ivf_reload()
        if getattr(self, "pending_ivf_reload", None) is not None:
            # Existing IVF pointers are being replaced. Route this request to
            # the CPU cold path without stalling the scheduler thread.
            self.ivf_reload_async_cpu_fallbacks = int(
                getattr(self, "ivf_reload_async_cpu_fallbacks", 0)
            ) + max(0, int(num_queries))
            self._record_failure("ivf_reload_inflight")
            return False
        if self._faiss_scratch_release_pending():
            self.ann_disjoint_map_overlap_rejections += 1
            self._record_failure("ann_scratch_detaching")
            return False
        if self.pending_ann_reclaim is not None:
            # Once source events have been recorded, admitting another GPU
            # search could enqueue work behind the fence and race the detach.
            self.ann_disjoint_map_overlap_rejections += 1
            self._record_failure("ann_reclaim_pending")
            return False
        if self._native_map_inflight():
            if (
                self._disjoint_map_bound_scratch_bytes(requested_bytes)
                > 0
            ):
                self.ann_disjoint_map_overlap_admissions += 1
                self.ann_disjoint_map_overlap_rows += max(
                    0, int(num_queries)
                )
                self.ann_idle_since = None
                self.ann_drain_epoch_processed = False
                return True
            self.ann_disjoint_map_overlap_rejections += 1
            self._record_failure("kv_native_map_inflight")
            return False
        if self.kv_backlog_committed:
            # A newly pending ANN request starts a new dependency frontier.
            # Protect live KV and the next executable Prefill batch, not every
            # Prefill that is still blocked on retrieval.
            required_tokens = self.predict_kv_tokens(
                scheduler,
                prefill_admission_batches=1,
                include_all_queued_prefill=False,
            )
            if (
                required_tokens
                > self.pool.kv_layout.static_capacity_tokens
            ):
                self.kv_backlog_ann_fallbacks += 1
                self._record_failure("kv_backlog_committed")
                return False
            self.kv_backlog_committed = False
        self.ann_idle_since = None
        self.ann_drain_epoch_processed = False
        gpu_work_active = getattr(
            scheduler, "_rag_ann_gpu_work_active", lambda: False
        )()
        if gpu_work_active:
            # Existing searches hold references to scratch and IVF list
            # pointers. A larger request falls back to CPU instead of changing
            # either mapping under those readers.
            admitted = (
                self.pool._faiss_bound
                and self.pool._last_faiss_mapping[1]
                >= min(requested_bytes, self.pool.total_bytes)
            )
            if not admitted:
                self._record_failure("ann_extent_inflight")
            return admitted
        started = time.perf_counter()
        snapshot = self.pool.snapshot()
        initial_kv_extents = int(snapshot.kv_mapped_extents)
        if snapshot.ann_scratch_bytes > 0:
            if (
                requested_bytes > snapshot.ann_scratch_bytes
                and self._residency_guard()
            ):
                self._record_failure("min_residency")
                return False
            previous_scratch_bytes = snapshot.ann_scratch_bytes
            evicted_ivf = False
            mapped = self.pool.bind_faiss_scratch(
                self.ann_executor, requested_bytes=requested_bytes
            )
            admitted = mapped >= min(
                requested_bytes, snapshot.total_bytes
            )
            if (
                not admitted
                and self.pool.mode == "ivf"
                and self.pool.snapshot().ann_index_bytes > 0
            ):
                self.ann_executor.release_shared_ivf_arena(self.pool)
                evicted_ivf = True
                self._record("ivf_evict_for_scratch")
                mapped = self.pool.bind_faiss_scratch(
                    self.ann_executor,
                    requested_bytes=requested_bytes,
                )
                admitted = mapped >= min(
                    requested_bytes, snapshot.total_bytes
                )
            if admitted:
                if evicted_ivf or mapped != previous_scratch_bytes:
                    self.last_transition_at = time.perf_counter()
                    self._record("ann_scratch_resize")
                    self.handoff_latency_ms.append(
                        (self.last_transition_at - started) * 1000.0
                    )
                if (
                    self.pool.mode == "ivf"
                    and self.ann_executor.shared_ivf_resident_count
                    < self.ann_executor.shared_ivf_total_count
                ):
                    self.ivf_request_path_refill_skips += 1
                return True

            # Growing an already-bound arena can fail because the remaining
            # physical chunks belong to reclaimable KV extents. Detach a
            # partial mapping and continue through the normal KV-to-ANN path
            # below instead of falling back to CPU prematurely.
            if self.pool._faiss_bound:
                self.pool.unbind_faiss_scratch(self.ann_executor)
            snapshot = self.pool.snapshot()
        if self._residency_guard():
            self._record_failure("min_residency")
            return False

        required_chunks = math.ceil(
            min(requested_bytes, snapshot.total_bytes)
            / snapshot.chunk_bytes
        )
        predicted_kv_tokens = self.predict_kv_tokens(scheduler)
        free_chunks = snapshot.free_bytes // snapshot.chunk_bytes
        if (
            free_chunks < required_chunks
            and self.pool.mode == "ivf"
            and snapshot.ann_index_bytes > 0
        ):
            self.ann_executor.release_shared_ivf_arena(self.pool)
            self._record("ivf_evict_for_scratch")
            snapshot = self.pool.snapshot()
            free_chunks = snapshot.free_bytes // snapshot.chunk_bytes
        target_extents = snapshot.kv_mapped_extents
        extent_chunks = (
            snapshot.kv_bytes // snapshot.chunk_bytes // target_extents
            if target_extents
            else 0
        )
        while free_chunks < required_chunks and target_extents > 0:
            if self.pending_kv_publications:
                self._drain_pending_kv_publications()
            candidate = target_extents - 1
            candidate_capacity = (
                snapshot.kv_static_capacity_tokens
                + candidate * snapshot.kv_tokens_per_extent
            )
            if candidate_capacity < predicted_kv_tokens:
                self._record_failure("kv_prediction_reserve")
                return False
            if not self.token_allocator.freeze_mapped_capacity(
                candidate_capacity
            ):
                tree_cache = getattr(scheduler, "tree_cache", None)
                evictable = (
                    int(tree_cache.evictable_size())
                    if tree_cache is not None
                    else 0
                )
                if evictable > 0:
                    tree_cache.evict(evictable)
                    self._record("radix_cache_evict")
                if not self.token_allocator.freeze_mapped_capacity(
                    candidate_capacity
                ):
                    self._record_failure("live_kv_extent")
                    return False
            previous_capacity = (
                snapshot.kv_static_capacity_tokens
                + target_extents * snapshot.kv_tokens_per_extent
            )
            try:
                self.kv_cache.unmap_shared_hbm_extents(candidate)
            except Exception:
                self.token_allocator.publish_mapped_capacity(
                    candidate_capacity, previous_capacity
                )
                raise
            target_extents = candidate
            free_chunks += extent_chunks

        try:
            mapped = self.pool.bind_faiss_scratch(
                self.ann_executor, requested_bytes=requested_bytes
            )
            if mapped < min(requested_bytes, snapshot.total_bytes):
                if mapped > 16:
                    self.pool.unbind_faiss_scratch(self.ann_executor)
                self._record_failure("no_ann_scratch")
                return False
            self.last_transition_at = time.perf_counter()
            if target_extents < initial_kv_extents:
                self._record("kv_to_ann")
            else:
                self._record("ann_scratch_bind")
            self.handoff_latency_ms.append(
                (self.last_transition_at - started) * 1000.0
            )
            if (
                self.pool.mode == "ivf"
                and self.ann_executor.shared_ivf_resident_count
                < self.ann_executor.shared_ivf_total_count
            ):
                self.ivf_request_path_refill_skips += 1
            return True
        except Exception:
            self._record_failure("kv_to_ann")
            logger.exception("Failed to lease shared HBM to ANN scratch")
            return False
    def _ivf_observed_misses(self) -> int:
        policy = getattr(self.ann_executor, "_shared_ivf_policy", None)
        return int(getattr(policy, "observed_misses", 0))

    def _finish_ivf_reload(
        self,
        pending: dict[str, Any],
        resident: int,
        elapsed_ms: float,
        *,
        asynchronous: bool,
    ) -> bool:
        reloaded_bytes = self.ann_executor.shared_ivf_reloaded_bytes
        if (
            resident <= pending["previous_resident"]
            and reloaded_bytes <= pending["previous_reloaded_bytes"]
        ):
            self._record_failure("ivf_reload_no_space")
            return False
        self.reload_waits += 1
        self.last_transition_at = time.perf_counter()
        self._record("ivf_reload")
        if pending.get("free_only", False):
            self.ivf_online_recovery_completions += 1
            self.ivf_online_recovery_lists += max(
                0, resident - pending["previous_resident"]
            )
            self.ivf_online_recovery_next_at = (
                time.perf_counter() + IVF_IDLE_REFILL_RETRY_MS / 1000.0
            )
        if asynchronous:
            self.ivf_reload_async_completions = int(
                getattr(self, "ivf_reload_async_completions", 0)
            ) + 1
            self.ivf_reload_async_latency_ms.append(elapsed_ms)
            # Keep total transition latency for reporting, but do not charge
            # an already-paid arena installation to every subsequent refill.
            if "arena_install" in pending:
                field = (
                    "ivf_reload_install_latency_ms"
                    if pending["arena_install"]
                    else "ivf_reload_existing_arena_latency_ms"
                )
                history = getattr(self, field, None)
                if history is None:
                    history = deque(maxlen=4096)
                    setattr(self, field, history)
                history.append(elapsed_ms)
            list_counts = getattr(
                self, "ivf_reload_async_list_counts", None
            )
            if list_counts is None:
                list_counts = deque(maxlen=4096)
                self.ivf_reload_async_list_counts = list_counts
            list_counts.append(
                max(1, int(pending.get("planned_count", 1)))
            )
            self._record("ivf_reload_async")
        else:
            self.handoff_latency_ms.append(elapsed_ms)
        return True

    def _poll_pending_ivf_reload(self) -> bool:
        pending = getattr(self, "pending_ivf_reload", None)
        if pending is None or not pending["future"].done():
            return False
        self.pending_ivf_reload = None
        elapsed_ms = (
            time.perf_counter() - float(pending["started_at"])
        ) * 1000.0
        try:
            resident = int(pending["future"].result())
        except Exception:
            self.ivf_reload_async_errors = int(
                getattr(self, "ivf_reload_async_errors", 0)
            ) + 1
            self._record_failure("ivf_reload_async")
            logger.exception("Asynchronous shared-HBM IVF refill failed")
            return False
        return self._finish_ivf_reload(
            pending, resident, elapsed_ms, asynchronous=True
        )

    def _ivf_reload_inflight(self) -> bool:
        self._poll_pending_ivf_reload()
        return getattr(self, "pending_ivf_reload", None) is not None

    def _kv_ownership_transition_inflight(self) -> bool:
        return bool(
            getattr(self, "pending_ann_reclaim", None) is not None
            or getattr(self, "pending_kv_native_map", None) is not None
            or getattr(self, "pending_kv_publications", None)
        )

    def _ivf_scratch_reserve_bytes(
        self, snapshot: SharedHBMSnapshot
    ) -> int:
        """Return additional free bytes needed for a stable Faiss scratch lease."""

        previous_request = int(
            getattr(self.pool, "_last_faiss_requested_bytes", 0)
        )
        desired = previous_request or DEFAULT_FAISS_TEMP_BYTES
        desired = min(max(0, desired), int(snapshot.total_bytes))
        return max(0, desired - int(snapshot.ann_scratch_bytes))

    def _potential_ivf_recovery_arena_bytes(
        self,
        *,
        predicted_kv_tokens: int,
    ) -> Optional[int]:
        """Predict free HBM after contracting KV to its safe target."""

        if self._kv_ownership_transition_inflight():
            return 0
        snapshot_method = getattr(self.pool, "snapshot", None)
        if snapshot_method is None or self.pool.mode != "ivf":
            return None
        snapshot = snapshot_method()
        required_fields = (
            "total_bytes",
            "ann_index_bytes",
            "ann_scratch_bytes",
            "free_bytes",
            "chunk_bytes",
            "kv_bytes",
            "kv_mapped_extents",
            "kv_static_capacity_tokens",
            "kv_tokens_per_extent",
        )
        if any(not hasattr(snapshot, field) for field in required_fields):
            # Keep compatibility with lightweight policy-only test doubles.
            return None
        if int(snapshot.ann_index_bytes) > 0 and not getattr(self, "partial_index_reclaim_enabled", False):
            return None

        mapped_extents = max(0, int(snapshot.kv_mapped_extents))
        tokens_per_extent = max(1, int(snapshot.kv_tokens_per_extent))
        static_tokens = max(0, int(snapshot.kv_static_capacity_tokens))
        target_extents = min(
            mapped_extents,
            max(
                0,
                math.ceil(
                    max(0, int(predicted_kv_tokens) - static_tokens)
                    / tokens_per_extent
                ),
            ),
        )
        extent_bytes = (
            max(0, int(snapshot.kv_bytes)) // mapped_extents
            if mapped_extents > 0
            else 0
        )
        potential_free_bytes = max(0, int(snapshot.free_bytes)) + (
            mapped_extents - target_extents
        ) * extent_bytes
        potential_free_bytes = max(
            0,
            potential_free_bytes
            - self._ivf_scratch_reserve_bytes(snapshot),
        )
        chunk_bytes = max(1, int(snapshot.chunk_bytes))
        return (potential_free_bytes // chunk_bytes) * chunk_bytes

    def _prepare_ivf_recovery_arena(
        self,
        scheduler,
        *,
        predicted_kv_tokens: int,
    ) -> Optional[int]:
        """Return idle KV extents before installing a profitable IVF arena.

        ANN scratch alone is small enough to coexist with a prefetched KV
        lease, so waiting for scratch allocation failure leaves most of the
        hot index stranded on CPU after a Decode burst.  This path runs only
        after the candidate benefit test has accepted a refill.  It contracts
        KV to the current predicted requirement, and the allocator freeze is
        the final live-token safety check.
        """

        if self._kv_ownership_transition_inflight():
            self.kv_to_ivf_reclaim_failures += 1
            self._record_failure("kv_to_ivf_transition_inflight")
            return 0
        snapshot_method = getattr(self.pool, "snapshot", None)
        if snapshot_method is None or self.pool.mode != "ivf":
            return None
        snapshot = snapshot_method()
        required_fields = (
            "total_bytes",
            "ann_index_bytes",
            "ann_scratch_bytes",
            "free_bytes",
            "chunk_bytes",
            "kv_mapped_extents",
            "kv_static_capacity_tokens",
            "kv_tokens_per_extent",
        )
        if any(not hasattr(snapshot, field) for field in required_fields):
            # Keep compatibility with lightweight policy-only test doubles.
            return None
        if int(snapshot.ann_index_bytes) > 0 and not getattr(self, "partial_index_reclaim_enabled", False):
            return None

        mapped_extents = max(0, int(snapshot.kv_mapped_extents))
        tokens_per_extent = max(1, int(snapshot.kv_tokens_per_extent))
        static_tokens = max(0, int(snapshot.kv_static_capacity_tokens))
        target_extents = min(
            mapped_extents,
            max(
                0,
                math.ceil(
                    max(0, int(predicted_kv_tokens) - static_tokens)
                    / tokens_per_extent
                ),
            ),
        )
        if target_extents < mapped_extents:
            if self._residency_guard():
                self.kv_to_ivf_reclaim_failures += 1
                self._record_failure("kv_to_ivf_residency")
                return 0

            target_capacity = (
                static_tokens + target_extents * tokens_per_extent
            )
            previous_capacity = int(
                self.token_allocator.current_mapped_capacity()
            )
            frozen = self.token_allocator.freeze_mapped_capacity(
                target_capacity
            )
            if not frozen:
                tree_cache = getattr(scheduler, "tree_cache", None)
                evictable = self._evictable_kv_tokens(scheduler)
                if tree_cache is not None and evictable > 0:
                    tree_cache.evict(evictable)
                    self._record("radix_cache_evict")
                frozen = self.token_allocator.freeze_mapped_capacity(
                    target_capacity
                )
            if not frozen:
                self.kv_to_ivf_reclaim_failures += 1
                self._record_failure("kv_to_ivf_live_kv")
                return 0

            started = time.perf_counter()
            try:
                self.kv_cache.unmap_shared_hbm_extents(target_extents)
            except Exception:
                try:
                    self.token_allocator.publish_mapped_capacity(
                        target_capacity, previous_capacity
                    )
                except Exception:
                    logger.exception(
                        "Failed to restore logical KV capacity after an "
                        "IVF recovery unmap failure"
                    )
                self.kv_to_ivf_reclaim_failures += 1
                self._record_failure("kv_to_ivf_unmap")
                logger.exception(
                    "Failed to return idle KV extents for IVF recovery"
                )
                return 0

            completed = self.pool.snapshot()
            reclaimed_extents = max(
                0,
                mapped_extents - int(completed.kv_mapped_extents),
            )
            reclaimed_bytes = max(
                0, int(completed.free_bytes) - int(snapshot.free_bytes)
            )
            self.kv_to_ivf_reclaims += 1
            self.kv_to_ivf_extents_reclaimed += reclaimed_extents
            self.kv_to_ivf_bytes_reclaimed += reclaimed_bytes
            self.last_kv_to_ivf_target_extents = int(
                completed.kv_mapped_extents
            )
            self.last_transition_at = time.perf_counter()
            self._record("kv_to_ann")
            self._record("kv_to_ivf")
            self.handoff_latency_ms.append(
                (self.last_transition_at - started) * 1000.0
            )
            snapshot = completed
            logger.info(
                "Shared-HBM KV->IVF recovery: predicted_tokens=%s "
                "target_extents=%s reclaimed_extents=%s "
                "reclaimed_bytes=%s free_bytes=%s",
                predicted_kv_tokens,
                target_extents,
                reclaimed_extents,
                reclaimed_bytes,
                snapshot.free_bytes,
            )

        arena_bytes = (
            int(snapshot.free_bytes) // int(snapshot.chunk_bytes)
        ) * int(snapshot.chunk_bytes)
        arena_bytes = max(
            0,
            arena_bytes - self._ivf_scratch_reserve_bytes(snapshot),
        )
        arena_bytes = (
            arena_bytes // int(snapshot.chunk_bytes)
        ) * int(snapshot.chunk_bytes)
        # Existing prefix remains mapped; return the total target arena size.
        arena_bytes += max(0, int(snapshot.ann_index_bytes))
        self.last_kv_to_ivf_arena_bytes = max(0, arena_bytes)
        return max(0, arena_bytes)

    def _maybe_reload_ivf(
        self,
        scheduler,
        deadline_slack_ms: Optional[float] = None,
        *,
        allow_existing_heat: bool = False,
        fill_free_capacity: bool = False,
        background_max_lists: Optional[int] = None,
        free_only: bool = False,
    ) -> bool:
        self._poll_pending_ivf_reload()
        if getattr(self, "pending_ivf_reload", None) is not None:
            return False
        if free_only and (
            not self.async_native_map
            or self.kv_map_executor is None
            or self._kv_ownership_transition_inflight()
        ):
            return False
        if (
            self.pool.mode != "ivf"
            or self.ann_executor is None
            or (
                self.ann_executor.shared_ivf_resident_count
                >= self.ann_executor.shared_ivf_total_count
            )
        ):
            return True

        policy = getattr(self.ann_executor, "_shared_ivf_policy", None)
        observed_misses = int(getattr(policy, "observed_misses", 0))
        min_misses = int(getattr(self, "ivf_reload_min_misses", 0))
        pending_misses = max(
            0,
            observed_misses
            - int(getattr(self, "last_ivf_reload_observed_misses", 0)),
        )
        shift_detector = getattr(policy, "locality_shift_detected", None)
        locality_shift_active = (
            bool(shift_detector()) if callable(shift_detector) else None
        )
        if (
            locality_shift_active is False
            and float(getattr(policy, "miss_boost", 0.0)) > 0.0
        ):
            # A locality shift is useful urgency evidence, but steady CPU
            # queue debt must still reach the candidate benefit/cost model.
            # Hard-gating here permanently blocks adaptation when the online
            # distribution differs from the offline prior without a step
            # change in miss ratio.
            self.ivf_reload_no_shift_checks = int(
                getattr(self, "ivf_reload_no_shift_checks", 0)
            ) + 1
        if (
            not allow_existing_heat
            and policy is not None
            and min_misses > 0
            and pending_misses < min_misses
        ):
            self.ivf_reload_coalesced = (
                int(getattr(self, "ivf_reload_coalesced", 0)) + 1
            )
            return False

        deadline_ms = max(
            0.0,
            float(
                getattr(scheduler, "rag_ann_deadline_ms", 0.0)
                if deadline_slack_ms is None
                else deadline_slack_ms
            ),
        )
        safety_ms = max(
            0.0,
            float(getattr(scheduler, "rag_ann_safety_margin_ms", 0.0)),
        )
        gpu_history = list(
            getattr(scheduler, "rag_ann_gpu_latency_ms", ())
        )
        gpu_p95_ms = (
            float(torch.tensor(gpu_history).quantile(0.95).item())
            if gpu_history
            else 0.0
        )
        reload_p95_ms = self.ann_executor.shared_ivf_reload_p95_ms
        async_reload_enabled = bool(
            getattr(self, "async_native_map", False)
            and getattr(self, "kv_map_executor", None) is not None
        )
        if reload_p95_ms + gpu_p95_ms + safety_ms > deadline_ms:
            if not async_reload_enabled:
                self._record_failure("ivf_reload_deadline")
                return False
            # While an asynchronous refill is in flight, new ANN work is
            # routed to the authoritative CPU index. Its copy latency is not
            # on the current request's critical path, so the request deadline
            # must not permanently freeze background cluster recovery. The
            # measured cost still bounds max_lists below and remains charged
            # by the candidate benefit/cost admission test.
            self.ivf_reload_async_deadline_bypasses = int(
                getattr(self, "ivf_reload_async_deadline_bypasses", 0)
            ) + 1

        predicted = self.predict_kv_tokens(scheduler)
        if predicted > self.token_allocator.current_mapped_capacity():
            self._record_failure("ivf_reload_kv_reserve")
            return False
        reload_budget_ms = max(
            0.0, deadline_ms - gpu_p95_ms - safety_ms
        )
        max_lists = max(
            1, int(reload_budget_ms // max(reload_p95_ms, 1e-6))
        )
        configured_max_lists = int(
            getattr(self, "ivf_reload_max_lists", 0)
        )
        if configured_max_lists > 0:
            max_lists = min(max_lists, configured_max_lists)
        if background_max_lists is not None:
            max_lists = max(
                max_lists, max(1, int(background_max_lists))
            )
            max_lists = min(
                max_lists,
                self.ann_executor.shared_ivf_total_count,
            )

        planned_candidate_ids = None
        free_arena_bytes = None
        partial_recovery = False
        if free_only:
            snapshot = self.pool.snapshot()
            if int(snapshot.moving_bytes) > 0:
                return False
            # Partial recovery may append capacity, never replace live lists.
            partial_recovery = bool(
                getattr(self, "partial_index_reclaim_enabled", False)
                and snapshot.ann_index_bytes > 0
            )
            if partial_recovery:
                free_arena_bytes = self._potential_ivf_recovery_arena_bytes(
                    predicted_kv_tokens=predicted)
            # The legacy free-only path never resizes a live arena.
            if snapshot.ann_index_bytes == 0:
                available = max(
                    0, snapshot.free_bytes
                    - self._ivf_scratch_reserve_bytes(snapshot)
                )
                free_arena_bytes = (
                    available // snapshot.chunk_bytes
                ) * snapshot.chunk_bytes
                if free_arena_bytes <= 0:
                    return False
        predictor = getattr(
            self.ann_executor, "predict_shared_ivf_reload", None
        )
        if callable(predictor):
            potential_recovery_bytes = free_arena_bytes if free_only else (
                self._potential_ivf_recovery_arena_bytes(
                    predicted_kv_tokens=predicted,
                )
            )
            if potential_recovery_bytes is None:
                plan = predictor(max_lists)
            else:
                plan = predictor(
                    max_lists,
                    available_bytes_override=potential_recovery_bytes,
                )
            self.ivf_reload_candidate_predictions = int(
                getattr(self, "ivf_reload_candidate_predictions", 0)
            ) + 1
            if plan is None or int(getattr(plan, "count", 0)) <= 0:
                self.ivf_reload_candidate_gated = int(
                    getattr(self, "ivf_reload_candidate_gated", 0)
                ) + 1
                self._record_failure("ivf_reload_candidate")
                return False

            free_space_prefix = 0
            for victims in plan.victim_ids:
                if victims:
                    break
                free_space_prefix += 1

            if locality_shift_active is False or free_only:
                # Under a stable distribution, cold credit may consume newly
                # available/borrowed HBM, but it must not evict the profiled
                # hot set. Replacement is reserved for an observed locality
                # shift; otherwise transient CPU queue pressure can turn into
                # repeated full-capacity swaps and self-amplifying fallback.
                if free_space_prefix <= 0:
                    self.ivf_reload_stable_replacement_gated = int(
                        getattr(
                            self,
                            "ivf_reload_stable_replacement_gated",
                            0,
                        )
                    ) + 1
                    self.ivf_reload_candidate_gated = int(
                        getattr(self, "ivf_reload_candidate_gated", 0)
                    ) + 1
                    self._record_failure("ivf_reload_stable_replacement")
                    return False
                plan = plan.prefix(free_space_prefix)

            copy_cost_per_cluster_ms = (
                reload_p95_ms if math.isfinite(reload_p95_ms) else 0.0
            )
            cost_history_field = "ivf_reload_async_latency_ms"
            if getattr(self, "ivf_phased_reload_cost_enabled", False):
                cost_history_field = (
                    "ivf_reload_install_latency_ms"
                    if self.pool.snapshot().ann_index_bytes == 0
                    else "ivf_reload_existing_arena_latency_ms"
                )
            async_latencies = list(getattr(self, cost_history_field, ()))
            async_batch_cost_ms = (
                float(torch.tensor(async_latencies).quantile(0.95).item())
                if async_latencies
                else 0.0
            )
            hysteresis = max(
                1.0, float(getattr(policy, "promotion_hysteresis", 1.0))
            )
            cumulative_benefit_ms = 0.0
            best_count = 0
            best_benefit_ms = 0.0
            best_cost_ms = 0.0
            best_surplus_ms = 0.0
            if fill_free_capacity and free_space_prefix > 0:
                best_count = min(int(plan.count), free_space_prefix)
                best_benefit_ms = float(
                    sum(plan.incremental_benefit_ms[:best_count])
                )
                best_cost_ms = max(
                    async_batch_cost_ms,
                    copy_cost_per_cluster_ms * best_count,
                )
                best_surplus_ms = best_benefit_ms - (
                    hysteresis * best_cost_ms
                )
                self.ivf_idle_free_capacity_refills += 1
                self.ivf_idle_free_capacity_lists += best_count
                cumulative_benefit_ms = best_benefit_ms
            else:
                for count, incremental_ms in enumerate(
                    plan.incremental_benefit_ms, start=1
                ):
                    cumulative_benefit_ms += float(incremental_ms)
                    predicted_cost_ms = max(
                        async_batch_cost_ms,
                        copy_cost_per_cluster_ms * count,
                    )
                    surplus_ms = (
                        cumulative_benefit_ms
                        - hysteresis * predicted_cost_ms
                    )
                    if surplus_ms > best_surplus_ms:
                        best_count = count
                        best_benefit_ms = cumulative_benefit_ms
                        best_cost_ms = predicted_cost_ms
                        best_surplus_ms = surplus_ms
            self.ivf_reload_candidate_last_benefit_ms = (
                cumulative_benefit_ms
            )
            self.ivf_reload_candidate_last_cost_ms = max(
                async_batch_cost_ms,
                copy_cost_per_cluster_ms * int(plan.count),
            )
            if best_count == 0:
                self.ivf_reload_candidate_gated = int(
                    getattr(self, "ivf_reload_candidate_gated", 0)
                ) + 1
                self._record_failure("ivf_reload_candidate")
                return False

            plan = plan.prefix(best_count)
            max_lists = int(plan.count)
            planned_candidate_ids = tuple(plan.candidate_ids)
            self.ivf_reload_candidate_last_benefit_ms = best_benefit_ms
            self.ivf_reload_candidate_last_cost_ms = best_cost_ms
            self.ivf_reload_candidate_lists_selected = int(
                getattr(self, "ivf_reload_candidate_lists_selected", 0)
            ) + max_lists

        if free_only and not partial_recovery:
            if planned_candidate_ids is None:
                return False
            recovery_arena_bytes = free_arena_bytes
        else:
            recovery_arena_bytes = self._prepare_ivf_recovery_arena(
                scheduler,
                predicted_kv_tokens=predicted,
            )
        if recovery_arena_bytes == 0:
            return False

        growing_live_arena = bool(
            recovery_arena_bytes is not None
            and self.pool.snapshot().ann_index_bytes > 0
            and recovery_arena_bytes > self.pool.snapshot().ann_index_bytes
        )

        self.last_ivf_reload_observed_misses = observed_misses
        self.ivf_reload_attempts = (
            int(getattr(self, "ivf_reload_attempts", 0)) + 1
        )
        self.ivf_reload_lists_requested = int(
            getattr(self, "ivf_reload_lists_requested", 0)
        ) + max_lists
        pending = {
            "previous_resident": self.ann_executor.shared_ivf_resident_count,
            "previous_reloaded_bytes": (
                self.ann_executor.shared_ivf_reloaded_bytes
            ),
            "started_at": time.perf_counter(),
            "planned_count": max_lists,
        }
        logger.info(
            "Shared-HBM reload decision: kind=%s predicted_tokens=%s "
            "mapped_capacity=%s arena_target_bytes=%s planned_lists=%s "
            "benefit_ms=%s cost_ms=%s free_only=%s",
            "grow" if growing_live_arena else (
                "install" if getattr(self.ann_executor, "_shared_ivf_mapped_bytes", None) == 0
                else "refill" if getattr(self.ann_executor, "_shared_ivf_mapped_bytes", None) is not None
                else "unknown"),
            predicted, self.token_allocator.current_mapped_capacity(),
            recovery_arena_bytes, max_lists,
            getattr(self, "ivf_reload_candidate_last_benefit_ms", None),
            getattr(self, "ivf_reload_candidate_last_cost_ms", None), free_only,
        )
        if getattr(self, "ivf_phased_reload_cost_enabled", False):
            pending["arena_install"] = self.pool.snapshot().ann_index_bytes == 0
        reload_kwargs = {"max_lists": max_lists}
        if planned_candidate_ids is not None:
            reload_kwargs["planned_candidate_ids"] = planned_candidate_ids
        if recovery_arena_bytes is not None:
            reload_kwargs["arena_bytes"] = recovery_arena_bytes
        if free_only:
            reload_kwargs["allow_eviction"] = False
            pending["free_only"] = True
        if (
            bool(getattr(self, "async_native_map", False))
            and getattr(self, "kv_map_executor", None) is not None
        ):
            try:
                pending["snapshot"] = self.pool.snapshot()
                if free_only or growing_live_arena:
                    # Record known Faiss users on the scheduler thread. The
                    # worker waits before changing pointers; the existing
                    # pending_ivf_reload state blocks all subsequent GPU ANN.
                    fence = self.ann_executor.submit_shared_hbm_reclaim_fence(
                        self.kv_cache.shared_hbm_maintenance_stream
                    )
                    pending["future"] = self.kv_map_executor.submit(
                        self._reload_ivf_after_fence, fence, reload_kwargs
                    )
                    self.ivf_online_recovery_submissions += 1
                else:
                    pending["future"] = self.kv_map_executor.submit(
                        self.ann_executor.reload_shared_ivf_arena,
                        self.pool,
                        **reload_kwargs,
                    )
                self.pending_ivf_reload = pending
                self.ivf_reload_async_submissions = int(
                    getattr(self, "ivf_reload_async_submissions", 0)
                ) + 1
                return False
            except Exception:
                self.ivf_reload_async_errors = int(
                    getattr(self, "ivf_reload_async_errors", 0)
                ) + 1
                self._record_failure("ivf_reload_async_submit")
                logger.exception(
                    "Failed to submit asynchronous shared-HBM IVF refill"
                )
                if free_only or growing_live_arena:
                    # Never fall through to a blocking scheduler-thread refill.
                    return False

        if growing_live_arena:
            fence = self.ann_executor.submit_shared_hbm_reclaim_fence(
                self.kv_cache.shared_hbm_maintenance_stream)
            resident = self._reload_ivf_after_fence(fence, reload_kwargs)
        else:
            resident = self.ann_executor.reload_shared_ivf_arena(
                self.pool, **reload_kwargs)
        elapsed_ms = (time.perf_counter() - pending["started_at"]) * 1000.0
        return self._finish_ivf_reload(
            pending, resident, elapsed_ms, asynchronous=False
        )

    def _reload_ivf_after_fence(self, fence, reload_kwargs):
        started = time.perf_counter()
        fence.ready_event.synchronize()
        fence_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        resident = self.ann_executor.reload_shared_ivf_arena(
            self.pool, **reload_kwargs
        )
        logger.info(
            "Shared-HBM reload worker complete: fence_wait_ms=%.3f "
            "executor_ms=%.3f resident_lists=%s",
            fence_ms, (time.perf_counter() - started) * 1000.0, resident,
        )
        return resident

    def on_ann_online_recovery_tick(self, scheduler) -> bool:
        """Recover a surrendered index using free HBM, without waiting for LLM idle.

        Caller is a scheduler safe point with no inflight ANN result. Preserve
        CPU fallback/publication throughout each bounded refill. This repairs
        post-surrender recovery, not physical IVF shrinking.
        """
        if not getattr(self, "ivf_online_recovery_enabled", False):
            return False
        if (self.pool.mode != "ivf" or self.ann_executor is None
                or not self.async_native_map or self.kv_map_executor is None
                or self._kv_ownership_transition_inflight()
                or self._faiss_scratch_release_pending()
                or getattr(self, "pending_ivf_reload", None) is not None):
            return False
        if (getattr(scheduler, "ann_search_inflight", None)
                or getattr(scheduler, "_rag_ann_gpu_work_active", lambda: True)()):
            return False
        busy = getattr(scheduler, "ann_result_collector_busy", None)
        queue = getattr(scheduler, "ann_result_collect_queue", None)
        if ((busy is not None and busy.is_set())
                or (queue is not None and not queue.empty())):
            return False
        executor = self.ann_executor
        if ((executor.shared_ivf_resident_count == 0
                or (getattr(self, "partial_index_reclaim_enabled", False)
                    and getattr(executor, "_shared_ivf_partial_reclaims", 0) > 0))
                and getattr(executor, "_shared_ivf_evicted_bytes", 0) > 0):
            self.ivf_online_recovery_active = True
        if not self.ivf_online_recovery_active:
            return False
        now = time.perf_counter()
        if now < self.ivf_online_recovery_next_at:
            return False
        # Keep the configured online batch and benefit/cost gate, not the
        #256-list fully-idle bulk-fill exception.
        self.ivf_online_recovery_next_at = now + IVF_IDLE_REFILL_BACKOFF_MS / 1000.0
        self._maybe_reload_ivf(scheduler, allow_existing_heat=True, free_only=True)
        return getattr(self, "pending_ivf_reload", None) is not None

    def _ensure_rerank_capacity(
        self,
        scheduler,
        *,
        additional_prefill_tokens: int = 0,
        prefetch_while_ann_queued: bool = False,
        reason: str,
    ) -> bool:
        """Predict and grow KV while ANN no longer owns the GPU arena."""
        additional_prefill_tokens = max(
            0, int(additional_prefill_tokens)
        )
        pending_ann = bool(
            getattr(scheduler, "ann_search_pending", None)
        )
        effective_prefetch_batches = (
            1
            if pending_ann and not prefetch_while_ann_queued
            else self.rerank_prefetch_batches
        )
        if pending_ann and prefetch_while_ann_queued:
            self.rerank_prefetches_with_ann_queued += 1
        ann_work_active = getattr(
            scheduler, "_rag_ann_gpu_work_active", lambda: False
        )()
        now = time.perf_counter()
        if pending_ann or ann_work_active:
            self.ann_idle_since = None
            self.ann_drain_epoch_processed = False
            ann_drain_quiescent = False
        elif self.ann_idle_since is None:
            self.ann_idle_since = now
            ann_drain_quiescent = False
        else:
            ann_drain_quiescent = (
                now - self.ann_idle_since
            ) * 1000.0 >= self.ann_drain_quiescence_ms
        include_all_queued_prefill = (
            self.rerank_prefetch_batches > 1
            and ann_drain_quiescent
        )
        if include_all_queued_prefill:
            self.rerank_all_queue_predictions += 1
            self.ann_drain_epoch_processed = True
        baseline_prefill = (
            self._queued_prefill_tokens(scheduler, 1)
            + additional_prefill_tokens
        )
        required = self.predict_kv_tokens(
            scheduler,
            prefill_admission_batches=effective_prefetch_batches,
            additional_prefill_tokens=additional_prefill_tokens,
            include_all_queued_prefill=include_all_queued_prefill,
        )
        if (
            include_all_queued_prefill
            and required > self.pool.kv_layout.static_capacity_tokens
            and not self.kv_backlog_committed
        ):
            self.kv_backlog_committed = True
            self.kv_backlog_commits += 1
        extra_prefill = max(
            0,
            self.last_prediction_queued_prefill_tokens
            - baseline_prefill,
        )
        self.rerank_prediction_calls += 1
        self.last_rerank_extra_prefill_tokens = extra_prefill
        self.rerank_extra_prefill_tokens_total += extra_prefill
        current = self.token_allocator.current_mapped_capacity()
        if required <= current:
            return True

        # An IVF arena is one pointer-stable Faiss allocation. Releasing it
        # for the first completed request would route later requests in the
        # same ANN frontier through the CPU cold path. Keep the index until
        # the frontier has remained empty for the configured quiescence
        # window; the periodic quiescent hook commits the accumulated KV
        # demand once that ownership boundary is safe.
        if self.pool.mode == "ivf" and not ann_drain_quiescent:
            snapshot = self.pool.snapshot()
            if snapshot.ann_index_bytes > 0:
                self.kv_growth_ann_frontier_holds += 1
                self.kv_growth_ann_frontier_hold_tokens += max(
                    0, int(required - current)
                )
                self._record("ivf_hold_for_ann_frontier")
                return False

        min_prefetch_extents = min(
            KV_GROWTH_MAX_PREFETCH_EXTENTS,
            effective_prefetch_batches
            * KV_GROWTH_QUANTUM_EXTENTS,
        )
        self.rerank_growth_requests += 1
        return self.ensure_kv_capacity(
            required,
            min_prefetch_extents=min_prefetch_extents,
            reason=reason,
        )

    def on_ann_gpu_idle(self, scheduler):
        """Candidate-ready/rerank hook: reclaim ANN pages for predicted KV."""
        return self._ensure_rerank_capacity(
            scheduler,
            prefetch_while_ann_queued=True,
            reason="rerank",
        )

    def on_ann_quiescent_tick(self, scheduler) -> bool:
        """Commit a drained ANN frontier, then refill only at full idle."""
        pending_ann = bool(
            getattr(scheduler, "ann_search_pending", None)
        )
        ann_work_active = getattr(
            scheduler, "_rag_ann_gpu_work_active", lambda: False
        )()
        running_batch = getattr(scheduler, "running_batch", None)
        llm_work_active = bool(
            getattr(running_batch, "reqs", ()) if running_batch else ()
        )
        waiting_work = bool(getattr(scheduler, "waiting_queue", ()))
        ownership_transition_active = (
            self._kv_ownership_transition_inflight()
        )
        now = time.perf_counter()
        if pending_ann or ann_work_active or ownership_transition_active:
            self.ann_idle_since = None
            self.ann_drain_epoch_processed = False
            self.next_ivf_idle_refill_at = 0.0
            return False
        if self.ann_idle_since is None:
            self.ann_idle_since = now
            return False
        if (
            now - self.ann_idle_since
        ) * 1000.0 < self.ann_drain_quiescence_ms:
            return False

        handled = bool(self.ann_drain_epoch_processed)
        if not self.ann_drain_epoch_processed:
            self.ann_drain_quiescent_predictions += 1
            self._ensure_rerank_capacity(
                scheduler,
                reason="rerank_drain",
            )
            self.ann_drain_epoch_processed = True
            self.kv_growth_ann_frontier_commits += 1
            handled = True

        # A KV handoff may have started above. Index refill, unlike the
        # frontier commit, is legal only when no LLM request can still consume
        # the newly leased KV extents and no Prefill is waiting for them.
        if (
            llm_work_active
            or waiting_work
            or self._kv_ownership_transition_inflight()
        ):
            self.next_ivf_idle_refill_at = 0.0
            return handled

        policy = getattr(self.ann_executor, "_shared_ivf_policy", None)
        observed_misses = int(getattr(policy, "observed_misses", 0))
        resident = int(
            getattr(self.ann_executor, "shared_ivf_resident_count", 0)
        )
        total = int(
            getattr(self.ann_executor, "shared_ivf_total_count", 0)
        )
        if (
            self.pool.mode != "ivf"
            or self.ann_executor is None
            or observed_misses <= 0
            or resident >= total
            or now < self.next_ivf_idle_refill_at
        ):
            return handled

        self.ivf_idle_maintenance_ticks += 1
        attempts_before = self.ivf_reload_attempts
        idle_budget_ms = max(
            IVF_IDLE_REFILL_BUDGET_MS,
            float(getattr(scheduler, "rag_ann_deadline_ms", 0.0)),
        )
        self._maybe_reload_ivf(
            scheduler,
            deadline_slack_ms=idle_budget_ms,
            allow_existing_heat=True,
            fill_free_capacity=True,
            background_max_lists=IVF_IDLE_REFILL_MAX_LISTS,
        )
        new_attempts = max(
            0, self.ivf_reload_attempts - attempts_before
        )
        self.ivf_idle_refill_attempts += new_attempts
        if (
            new_attempts == 0
            and getattr(self, "pending_ivf_reload", None) is None
        ):
            self.ivf_idle_refill_no_candidate_holds += 1
            retry_ms = IVF_IDLE_REFILL_BACKOFF_MS
        else:
            retry_ms = IVF_IDLE_REFILL_RETRY_MS
        self.next_ivf_idle_refill_at = now + retry_ms / 1000.0
        return True

    def on_rag_prefill_ready(
        self, scheduler, actual_prefill_tokens: int, *,
        allow_new_memory_plans: bool = True,
    ) -> bool:
        """Grow KV at the merge-to-prefill boundary using exact demand."""
        if (getattr(scheduler,'rag_joint_event_input_provider',None) is not None
                or getattr(getattr(scheduler, 'rag_frontend_plan', None), 'joint_event_id', None) is not None):
            allow_new_memory_plans = False
        if not allow_new_memory_plans:
            # The queued request supplies demand to the joint snapshot. This
            # arrival callback must not change HBM before joint admission.
            # False is not a KV-ready guarantee; normal admission remains.
            return False
        if self.rerank_prefetch_batches <= 1:
            return True
        actual_prefill_tokens = max(0, int(actual_prefill_tokens))
        self.rerank_prefill_ready_calls += 1
        self.rerank_prefill_ready_tokens_total += actual_prefill_tokens
        return self._ensure_rerank_capacity(
            scheduler,
            additional_prefill_tokens=actual_prefill_tokens,
            reason="rerank_ready",
        )

    def observe_ann_completion(
        self, latency_ms: float, deadline_ms: float
    ):
        if deadline_ms > 0.0 and latency_ms > deadline_ms:
            self.deadline_misses += 1

    @staticmethod
    def _percentile(values, percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1),
        )
        return float(ordered[index])

    def metrics(self) -> dict[str, Any]:
        self._poll_pending_ivf_reload()
        self._poll_pending_kv_native_map()
        self._observe_pending_kv_publications()
        ann_reclaim_pending = self.pending_ann_reclaim is not None
        native_map_pending = self.pending_kv_native_map is not None
        ivf_reload_pending = (
            getattr(self, "pending_ivf_reload", None) is not None
        )
        if native_map_pending or ivf_reload_pending:
            # A native worker may hold the pool mutex while changing mappings.
            # Report its pre-transition snapshot instead of blocking metrics.
            pending = (
                self.pending_kv_native_map
                if native_map_pending
                else self.pending_ivf_reload
            )
            snapshot = pending["snapshot"]
            driver_stats = {}
        else:
            snapshot = self.pool.snapshot()
            driver_stats_method = getattr(
                self.pool, "driver_stats", None
            )
            driver_stats = (
                driver_stats_method().as_dict()
                if driver_stats_method is not None
                else {}
            )
        prediction_errors = [
            float(value) for value in self.prediction_error_tokens
        ]
        ivf_metrics = (
            self.ann_executor.shared_ivf_metrics()
            if self.ann_executor is not None
            and hasattr(self.ann_executor, "shared_ivf_metrics")
            else {}
        )
        scratch_fragmentation = max(
            0,
            snapshot.ann_scratch_bytes
            - self.pool._last_faiss_requested_bytes,
        )
        logical_capacity = int(
            self.token_allocator.current_mapped_capacity()
        )
        snapshot_native_capacity = (
            snapshot.kv_static_capacity_tokens
            + snapshot.kv_mapped_extents * snapshot.kv_tokens_per_extent
        )
        native_logical_gap_tokens = max(
            self._pending_native_logical_gap_tokens(),
            max(0, snapshot_native_capacity - logical_capacity),
        )
        return {
            **snapshot.as_dict(),
            "mode": self.pool.mode,
            "gpu_name": self.pool.gpu_name,
            "gpu_total_memory_bytes": self.pool.gpu_total_memory,
            "driver_calls": driver_stats,
            "owner_chunk_counts": {
                "free": snapshot.free_bytes // snapshot.chunk_bytes,
                "kv_cache": snapshot.kv_bytes // snapshot.chunk_bytes,
                "ann_scratch": (
                    snapshot.ann_scratch_bytes // snapshot.chunk_bytes
                ),
                "ann_index": (
                    snapshot.ann_index_bytes // snapshot.chunk_bytes
                ),
                "moving": snapshot.moving_bytes // snapshot.chunk_bytes,
            },
            "owner_internal_fragmentation_bytes": {
                "free": 0,
                "kv_cache": 0,
                "ann_scratch": scratch_fragmentation,
                "ann_index": int(
                    ivf_metrics.get("internal_fragmentation_bytes", 0)
                ),
                "moving": 0,
            },
            "last_predicted_tokens": self.last_predicted_tokens,
            "last_prediction_queued_prefill_tokens": (
                self.last_prediction_queued_prefill_tokens
            ),
            "last_prediction_evictable_tokens": (
                self.last_prediction_evictable_tokens
            ),
            "last_prediction_prefill_batches": (
                self.last_prediction_prefill_batches
            ),
            "kv_lookahead_configured_ms": self.lookahead_ms,
            "kv_lookahead_effective_ms": (
                self.last_effective_lookahead_ms
            ),
            "kv_lookahead_observed_map_ms": (
                self.last_lookahead_observed_map_ms
            ),
            "rerank_prefetch_batches": self.rerank_prefetch_batches,
            "rerank_prediction_calls": self.rerank_prediction_calls,
            "rerank_growth_requests": self.rerank_growth_requests,
            "rerank_prefetches_with_ann_queued": (
                self.rerank_prefetches_with_ann_queued
            ),
            "rerank_all_queue_predictions": (
                self.rerank_all_queue_predictions
            ),
            "kv_backlog_committed": self.kv_backlog_committed,
            "kv_backlog_commits": self.kv_backlog_commits,
            "kv_backlog_ann_fallbacks": (
                self.kv_backlog_ann_fallbacks
            ),
            "ann_drain_quiescence_ms": self.ann_drain_quiescence_ms,
            "ann_drain_epoch_processed": (
                self.ann_drain_epoch_processed
            ),
            "ann_drain_quiescent_predictions": (
                self.ann_drain_quiescent_predictions
            ),
            "kv_growth_ann_frontier_holds": (
                self.kv_growth_ann_frontier_holds
            ),
            "kv_growth_ann_frontier_hold_tokens": (
                self.kv_growth_ann_frontier_hold_tokens
            ),
            "kv_growth_ann_frontier_commits": (
                self.kv_growth_ann_frontier_commits
            ),
            "ivf_idle_maintenance_ticks": (
                self.ivf_idle_maintenance_ticks
            ),
            "ivf_idle_refill_attempts": self.ivf_idle_refill_attempts,
            "ivf_idle_refill_no_candidate_holds": (
                self.ivf_idle_refill_no_candidate_holds
            ),
            "ivf_idle_free_capacity_refills": (
                self.ivf_idle_free_capacity_refills
            ),
            "ivf_idle_free_capacity_lists": (
                self.ivf_idle_free_capacity_lists
            ),
            "ivf_request_path_refill_skips": (
                self.ivf_request_path_refill_skips
            ),
            "rerank_prefill_ready_calls": (
                self.rerank_prefill_ready_calls
            ),
            "rerank_prefill_ready_tokens_total": (
                self.rerank_prefill_ready_tokens_total
            ),
            "last_rerank_extra_prefill_tokens": (
                self.last_rerank_extra_prefill_tokens
            ),
            "rerank_extra_prefill_tokens_total": (
                self.rerank_extra_prefill_tokens_total
            ),
            "last_kv_growth_raw_target_extents": (
                self.last_kv_growth_raw_target_extents
            ),
            "last_kv_growth_target_extents": (
                self.last_kv_growth_target_extents
            ),
            "last_kv_growth_prefetch_extents": (
                self.last_kv_growth_prefetch_extents
            ),
            "last_kv_growth_reason": self.last_kv_growth_reason,
            "last_kv_growth_min_prefetch_extents": (
                self.last_kv_growth_min_prefetch_extents
            ),
            "kv_growth_prefetched_extents_total": (
                self.kv_growth_prefetched_extents_total
            ),
            "kv_growth_rate_tokens_per_s": (
                self.kv_growth_rate_tokens_per_s
            ),
            "kv_growth_device_synchronizations": (
                self.kv_growth_device_synchronizations
            ),
            "kv_growth_stream_fences": self.kv_growth_stream_fences,
            "kv_growth_streams_fenced": (
                self.kv_growth_streams_fenced
            ),
            "kv_growth_disjoint_ann_bypasses": (
                self.kv_growth_disjoint_ann_bypasses
            ),
            "kv_growth_disjoint_ann_bytes_preserved": (
                self.kv_growth_disjoint_ann_bytes_preserved
            ),
            "kv_growth_ann_conflicts": self.kv_growth_ann_conflicts,
            "kv_growth_ann_preserving_target_cap_enabled": (
                self.ann_preserving_target_cap_enabled
            ),
            "partial_index_reclaim_enabled": getattr(self, "partial_index_reclaim_enabled", False),
            "kv_growth_ann_preserving_target_caps": (
                self.kv_growth_ann_preserving_target_caps
            ),
            "kv_growth_ann_preserving_extents_saved": (
                self.kv_growth_ann_preserving_extents_saved
            ),
            "ann_disjoint_map_overlap_enabled": (
                self.ann_disjoint_map_overlap_enabled
            ),
            "ann_disjoint_map_overlap_admissions": (
                self.ann_disjoint_map_overlap_admissions
            ),
            "ann_disjoint_map_overlap_rejections": (
                self.ann_disjoint_map_overlap_rejections
            ),
            "ann_disjoint_map_overlap_rows": (
                self.ann_disjoint_map_overlap_rows
            ),
            "kv_growth_growable_query_fallbacks": (
                self.kv_growth_growable_query_fallbacks
            ),
            "last_kv_growable_extents": self.last_kv_growable_extents,
            "kv_growth_async_reclaim_fence_enabled": (
                self.async_reclaim_fence_enabled
            ),
            "kv_growth_async_reclaim_pending": ann_reclaim_pending,
            "kv_growth_async_reclaim_fence_submissions": (
                self.kv_growth_async_reclaim_fence_submissions
            ),
            "kv_growth_async_reclaim_fence_polls": (
                self.kv_growth_async_reclaim_fence_polls
            ),
            "kv_growth_async_reclaim_fence_ready": (
                self.kv_growth_async_reclaim_fence_ready
            ),
            "kv_growth_async_reclaim_streams_fenced": (
                self.kv_growth_async_reclaim_streams_fenced
            ),
            "kv_growth_fast_faiss_detach_enabled": (
                self.fast_faiss_detach_enabled
            ),
            "kv_growth_async_ann_unmap_configured": (
                self.async_ann_unmap_configured
            ),
            "kv_growth_async_ann_unmap_enabled": (
                self.async_ann_unmap_enabled
            ),
            "kv_growth_async_reclaim_fast_detach_uses": (
                self.kv_growth_async_reclaim_fast_detach_uses
            ),
            "kv_growth_async_reclaim_fallback_detach_uses": (
                self.kv_growth_async_reclaim_fallback_detach_uses
            ),
            "kv_growth_async_reclaim_ready_latency_p50_ms": self._percentile(
                self.kv_growth_async_reclaim_ready_latency_ms, 50
            ),
            "kv_growth_async_reclaim_ready_latency_p95_ms": self._percentile(
                self.kv_growth_async_reclaim_ready_latency_ms, 95
            ),
            "kv_growth_async_reclaim_ready_latency_p99_ms": self._percentile(
                self.kv_growth_async_reclaim_ready_latency_ms, 99
            ),
            "kv_growth_async_reclaim_detach_latency_p50_ms": self._percentile(
                self.kv_growth_async_reclaim_detach_latency_ms, 50
            ),
            "kv_growth_async_reclaim_detach_latency_p95_ms": self._percentile(
                self.kv_growth_async_reclaim_detach_latency_ms, 95
            ),
            "kv_growth_async_reclaim_detach_latency_p99_ms": self._percentile(
                self.kv_growth_async_reclaim_detach_latency_ms, 99
            ),
            "kv_growth_scheduler_detach_latency_p50_ms": self._percentile(
                self.kv_growth_scheduler_detach_latency_ms, 50
            ),
            "kv_growth_scheduler_detach_latency_p95_ms": self._percentile(
                self.kv_growth_scheduler_detach_latency_ms, 95
            ),
            "kv_growth_scheduler_detach_latency_p99_ms": self._percentile(
                self.kv_growth_scheduler_detach_latency_ms, 99
            ),
            "kv_growth_deferred_ann_release_pending": (
                self._faiss_scratch_release_pending()
            ),
            "kv_growth_deferred_ann_release_submissions": (
                self.kv_growth_deferred_ann_release_submissions
            ),
            "kv_growth_deferred_ann_release_completions": (
                self.kv_growth_deferred_ann_release_completions
            ),
            "kv_growth_deferred_ann_release_errors": (
                self.kv_growth_deferred_ann_release_errors
            ),
            "kv_growth_worker_ann_release_latency_p50_ms": self._percentile(
                self.kv_growth_worker_ann_release_latency_ms, 50
            ),
            "kv_growth_worker_ann_release_latency_p95_ms": self._percentile(
                self.kv_growth_worker_ann_release_latency_ms, 95
            ),
            "kv_growth_worker_ann_release_latency_p99_ms": self._percentile(
                self.kv_growth_worker_ann_release_latency_ms, 99
            ),
            "kv_growth_publication_event_waits": (
                self.kv_growth_publication_event_waits
            ),
            "kv_growth_async_native_map_enabled": (
                self.async_native_map
            ),
            "kv_growth_map_stage_extents_configured": (
                self.map_stage_extents_configured
            ),
            "kv_growth_map_stage_extents_auto": (
                self.map_stage_extents_configured == -1
            ),
            "kv_growth_map_stage_extents": self.map_stage_extents,
            "kv_growth_native_map_pending": native_map_pending,
            "kv_growth_native_map_submissions": (
                self.kv_growth_native_map_submissions
            ),
            "kv_growth_native_map_completions": (
                self.kv_growth_native_map_completions
            ),
            "kv_growth_native_map_errors": (
                self.kv_growth_native_map_errors
            ),
            "kv_growth_post_map_recovery_pending": bool(
                self.pending_kv_native_map is not None
                and self.pending_kv_native_map.get("post_map") is not None
                and self.pending_kv_native_map["post_map"].get(
                    "recovery_required", False
                )
            ),
            "kv_growth_post_map_recovery_attempts": (
                self.kv_growth_post_map_recovery_attempts
            ),
            "kv_growth_post_map_recovery_successes": (
                self.kv_growth_post_map_recovery_successes
            ),
            "kv_growth_post_map_recovery_errors": (
                self.kv_growth_post_map_recovery_errors
            ),
            "kv_growth_native_logical_gap_tokens": (
                native_logical_gap_tokens
            ),
            "kv_growth_native_logical_gap_events": (
                self.kv_growth_native_logical_gap_events
            ),
            "kv_growth_native_logical_gap_max_tokens": (
                self.kv_growth_native_logical_gap_max_tokens
            ),
            "kv_growth_stages_completed": (
                self.kv_growth_stages_completed
            ),
            "kv_growth_partial_publications": (
                self.kv_growth_partial_publications
            ),
            "last_kv_growth_stage_count": (
                self.last_kv_growth_stage_count
            ),
            "kv_growth_completion_notifications": (
                self.kv_growth_completion_notifications
            ),
            "kv_growth_completion_safe_point_calls": (
                self.kv_growth_completion_safe_point_calls
            ),
            "kv_growth_completion_ready_hits": (
                self.kv_growth_completion_ready_hits
            ),
            "kv_growth_completion_reaction_latency_p50_ms": self._percentile(
                self.kv_growth_completion_reaction_latency_ms, 50
            ),
            "kv_growth_completion_reaction_latency_p99_ms": self._percentile(
                self.kv_growth_completion_reaction_latency_ms, 99
            ),
            "kv_map_worker_prewarm_attempted": (
                self.kv_map_worker_prewarm_attempted
            ),
            "kv_map_worker_prewarm_succeeded": (
                self.kv_map_worker_prewarm_succeeded
            ),
            "kv_map_worker_prewarm_errors": (
                self.kv_map_worker_prewarm_errors
            ),
            "kv_map_worker_prewarm_latency_ms": (
                self.kv_map_worker_prewarm_latency_ms
            ),
            "kv_growth_async_publications": (
                self.kv_growth_async_publications
            ),
            "kv_growth_consumer_stream_waits": (
                self.kv_growth_consumer_stream_waits
            ),
            "kv_growth_reverse_drain_event_waits": (
                self.kv_growth_reverse_drain_event_waits
            ),
            "kv_growth_pending_publications": len(
                self.pending_kv_publications
            ),
            "kv_growth_async_enqueue_latency_p50_ms": self._percentile(
                self.kv_growth_async_enqueue_latency_ms, 50
            ),
            "kv_growth_async_enqueue_latency_p95_ms": self._percentile(
                self.kv_growth_async_enqueue_latency_ms, 95
            ),
            "kv_growth_async_enqueue_latency_p99_ms": self._percentile(
                self.kv_growth_async_enqueue_latency_ms, 99
            ),
            "kv_growth_fence_latency_p99_ms": self._percentile(
                self.kv_growth_fence_latency_ms, 99
            ),
            "kv_growth_native_map_latency_p99_ms": self._percentile(
                self.kv_growth_native_map_latency_ms, 99
            ),
            "kv_growth_zero_enqueue_latency_p99_ms": self._percentile(
                self.kv_growth_zero_enqueue_latency_ms, 99
            ),
            "kv_growth_refresh_latency_p99_ms": self._percentile(
                self.kv_growth_refresh_latency_ms, 99
            ),
            "kv_growth_publication_latency_p99_ms": self._percentile(
                self.kv_growth_publication_latency_ms, 99
            ),
            "kv_growth_first_publish_latency_p50_ms": self._percentile(
                self.kv_growth_first_publish_latency_ms, 50
            ),
            "kv_growth_first_publish_latency_p99_ms": self._percentile(
                self.kv_growth_first_publish_latency_ms, 99
            ),
            "kv_growth_total_latency_p99_ms": self._percentile(
                self.kv_growth_total_latency_ms, 99
            ),
            "prediction_error_p50_tokens": self._percentile(
                prediction_errors, 50
            ),
            "prediction_error_p95_tokens": self._percentile(
                prediction_errors, 95
            ),
            "prediction_error_p99_tokens": self._percentile(
                prediction_errors, 99
            ),
            "handoff_count": len(self.handoff_latency_ms),
            "handoff_latency_p50_ms": self._percentile(
                self.handoff_latency_ms, 50
            ),
            "handoff_latency_p95_ms": self._percentile(
                self.handoff_latency_ms, 95
            ),
            "handoff_latency_p99_ms": self._percentile(
                self.handoff_latency_ms, 99
            ),
            "transition_counts": dict(self.transition_counts),
            "failure_counts": dict(self.failure_counts),
            "cpu_fallbacks": self.cpu_fallbacks,
            "reload_waits": self.reload_waits,
            "ivf_reload_min_misses": self.ivf_reload_min_misses,
            "ivf_reload_max_lists": self.ivf_reload_max_lists,
            "ivf_reload_coalesced": self.ivf_reload_coalesced,
            "ivf_reload_shift_gated": int(
                getattr(self, "ivf_reload_shift_gated", 0)
            ),
            "ivf_reload_no_shift_checks": int(
                getattr(self, "ivf_reload_no_shift_checks", 0)
            ),
            "ivf_reload_candidate_predictions": int(
                getattr(self, "ivf_reload_candidate_predictions", 0)
            ),
            "ivf_reload_candidate_gated": int(
                getattr(self, "ivf_reload_candidate_gated", 0)
            ),
            "ivf_reload_candidate_lists_selected": int(
                getattr(self, "ivf_reload_candidate_lists_selected", 0)
            ),
            "ivf_reload_candidate_last_benefit_ms": float(
                getattr(
                    self, "ivf_reload_candidate_last_benefit_ms", 0.0
                )
            ),
            "ivf_reload_candidate_last_cost_ms": float(
                getattr(self, "ivf_reload_candidate_last_cost_ms", 0.0)
            ),
            "ivf_reload_async_deadline_bypasses": int(
                getattr(self, "ivf_reload_async_deadline_bypasses", 0)
            ),
            "ivf_reload_stable_replacement_gated": int(
                getattr(self, "ivf_reload_stable_replacement_gated", 0)
            ),
            "ivf_reload_attempts": self.ivf_reload_attempts,
            "ivf_reload_lists_requested": self.ivf_reload_lists_requested,
            "kv_to_ivf_reclaims": int(
                getattr(self, "kv_to_ivf_reclaims", 0)
            ),
            "kv_to_ivf_extents_reclaimed": int(
                getattr(self, "kv_to_ivf_extents_reclaimed", 0)
            ),
            "kv_to_ivf_bytes_reclaimed": int(
                getattr(self, "kv_to_ivf_bytes_reclaimed", 0)
            ),
            "kv_to_ivf_reclaim_failures": int(
                getattr(self, "kv_to_ivf_reclaim_failures", 0)
            ),
            "last_kv_to_ivf_target_extents": int(
                getattr(self, "last_kv_to_ivf_target_extents", 0)
            ),
            "last_kv_to_ivf_arena_bytes": int(
                getattr(self, "last_kv_to_ivf_arena_bytes", 0)
            ),
            "ivf_reload_pending_misses": max(
                0,
                self._ivf_observed_misses()
                - self.last_ivf_reload_observed_misses,
            ),
            "ivf_reload_async_enabled": bool(
                getattr(self, "async_native_map", False)
                and getattr(self, "kv_map_executor", None) is not None
            ),
            "ivf_reload_async_pending": ivf_reload_pending,
            "ivf_reload_async_submissions": int(
                getattr(self, "ivf_reload_async_submissions", 0)
            ),
            "ivf_reload_async_completions": int(
                getattr(self, "ivf_reload_async_completions", 0)
            ),
            "ivf_online_recovery_enabled": getattr(self, "ivf_online_recovery_enabled", False),
            "ivf_phased_reload_cost_enabled": getattr(self, "ivf_phased_reload_cost_enabled", False),
            "ivf_reload_install_samples": len(getattr(self, "ivf_reload_install_latency_ms", ())),
            "ivf_reload_existing_arena_samples": len(getattr(self, "ivf_reload_existing_arena_latency_ms", ())),
            "ivf_reload_install_p95_ms": self._percentile(
                getattr(self, "ivf_reload_install_latency_ms", ()), 95
            ),
            "ivf_reload_existing_arena_p95_ms": self._percentile(
                getattr(self, "ivf_reload_existing_arena_latency_ms", ()), 95
            ),
            "ivf_online_recovery_submissions": getattr(self, "ivf_online_recovery_submissions", 0),
            "ivf_online_recovery_completions": getattr(self, "ivf_online_recovery_completions", 0),
            "ivf_online_recovery_lists": getattr(self, "ivf_online_recovery_lists", 0),
            "ivf_reload_async_errors": int(
                getattr(self, "ivf_reload_async_errors", 0)
            ),
            "ivf_reload_async_cpu_fallbacks": int(
                getattr(self, "ivf_reload_async_cpu_fallbacks", 0)
            ),
            "ivf_reload_async_latency_p50_ms": self._percentile(
                getattr(self, "ivf_reload_async_latency_ms", ()), 50
            ),
            "ivf_reload_async_latency_p95_ms": self._percentile(
                getattr(self, "ivf_reload_async_latency_ms", ()), 95
            ),
            "ivf_reload_async_latency_p99_ms": self._percentile(
                getattr(self, "ivf_reload_async_latency_ms", ()), 99
            ),
            "deadline_misses": self.deadline_misses,
            "microbatch_shrinks": self.microbatch_shrinks,
            "ivf": ivf_metrics,
        }
