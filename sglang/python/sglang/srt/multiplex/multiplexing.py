"""
Mixin class providing multiplexing scheduling logic
"""

import hashlib
import logging
import math
import os
import threading
import time
from bisect import insort
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from queue import Empty, Queue
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.managers.io_struct import AnnSearchReqOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


# --- decode service-time model v2 (fitted on the v159 calibration) -----------
# Feature definitions and order MUST match fit_solo_models_v2.py, which is where
# these coefficients came from.  Every feature is causal: the rolling window
# holds completed decode intervals, strictly earlier than the one being
# predicted.  A mismatch in either the definitions or the ordering silently
# invalidates the model -- the coefficients do not carry their own feature spec.
RAG_DM2_COEFS = {'intercept': 11.018824, 'batch_size': 0.123191, 'hist10_p50': 0.397823, 'hist10_p90': 0.194182, 'hist10_max': 0.10837, 'hist50_p50': -0.64987, 'hist50_p90': 0.400613, 'hist50_max': -0.012214, 'cont10_frac': 1.991553, 'cont50_frac': 4.639803, 'overlap_class=ann_prefill': -13.406629, 'overlap_class=prefill': -0.789444, 'overlap_class=solo': -1.747344}
RAG_DM2_CONTENDED = ("prefill", "ann_prefill")
RAG_DM2_LEVELS = ("ann", "ann_prefill", "prefill", "solo")   # sorted; first is reference
RAG_DM2_TAU = 0.99


class MetadataHandoff:
    """Reusable cross-stream dependency for one ordered stream pair.

    Driven by the single scheduler thread.  publish() is called on the producer
    stream right after the metadata-producing CUDA work is submitted -- and
    before any later, unrelated long work, so the event cannot accidentally
    cover a whole prefill forward.  acquire() is called on the consumer stream
    before it first reads anything derived from that metadata.

    acquire() only enqueues the wait.  It does not block the CPU and does not
    wait for the GPU.  A CPU-side version counter turns repeated acquire() calls
    for the same published version into an integer comparison, so a consumer
    with several entry points inserts at most one wait per version.

    CUDA permits re-recording an event, and a wait_event() already submitted
    binds the record that existed when it was called, so later re-records do not
    retarget it.  Waiting for a newer version than strictly necessary is safe
    (both records are on the same source stream, so the newer implies the
    older) -- conservative, not incorrect.
    """

    __slots__ = ("source", "target", "event", "_published", "_acquired",
                 "_same_stream")

    def __init__(self, source, target) -> None:
        self.source = source
        self.target = target
        self.event = torch.cuda.Event(enable_timing=False, blocking=False)
        self._published = 0
        self._acquired = 0
        self._same_stream = (
            source.device == target.device
            and source.cuda_stream == target.cuda_stream
        )

    def publish(self) -> None:
        self.event.record(self.source)
        self._published += 1

    def acquire(self) -> None:
        version = self._published
        if version == self._acquired:
            return
        if not self._same_stream:
            self.target.wait_event(self.event)
        self._acquired = version

from sglang.srt.mem_cache.radix_forecast import (
    RadixForecastClock, advance_radix_request_events, radix_evictable_tokens,
    replay_radix_timed_evictions, simulate_radix_eviction,
)
from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.multiplex.stage_predictor import (
    bind_layer_scaled_prefill_query, bind_prefill_compute_profile, bind_decode_compute_profile,
    PressureDecodeComputeProfile, FrozenANNPressureTable,
)
from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, CLIP_MAX_NEW_TOKENS
from sglang.srt.multiplex.lease_profile import (
    FrozenColdWorkerProfile,
    FrozenColdWorkerLinearProfile,
    FrozenJointTransitionProfiles,
    FrozenANNServiceProfile,
    JointTransitionCompletion,
    PrefillReleaseObservation,
    capture_cold_worker_profile,
    capture_decode_decision_profile,
    capture_decode_capacity_profile,
    capture_ann_service_profile,
    capture_joint_transition_profiles,
    capture_ann_release_profile,
    capture_prefill_layer_profile,
    capture_prefill_handoff_profile,
    capture_input_preparation_profile,
    capture_radix_eviction_profile,
)
from sglang.srt.multiplex.lease_frontier import (
    capture_execution_stamp,
    capture_prepared_prefill_contract,
    preview_queued_prefill_shapes,
    capture_inflight_cold_bindings,
    capture_pending_cold_bindings,
    resolve_cold_bindings,
    observe_full_cpu_fallback,
)
from sglang.srt.multiplex.full_lease import (
    FrozenHandoffBatches,
    build_cold_worker_operations,
    build_additional_cold_operations,
    build_prefill_bundle_operations,
    build_prefill_decode_handoff,
    build_observed_input_preparation,
    build_following_input_preparation,
    build_prospective_cache_eviction,
    ProspectiveCacheEvictionShape,
    describe_prospective_prefill,
    derive_prospective_radix_events,
    time_frozen_joint_operations,
    time_frozen_decode_operations,
    bind_decode_stage_model,
    bind_frozen_decode_interference,
    bind_observed_decode_compute_interference,
    time_frozen_prefill_handoff,
    JoiningDecodePlan,
    allocate_decode_kv_prefixes,
    remaining_decode_tokens,
    forecast_active_decode_kv_growth,
    select_reference_profiles,
    freeze_request_urgency,
    CachedPrefillBudget,
    ProspectivePoolObservation,
    make_prospective_capacity_budget,
    derive_prospective_capacity_publications,
    JointMemoryDemand,
    ComputeLease,
    memory_lease_from_hbm_plan,
    apply_joint_activation_readiness,
    derive_joint_memory_demand,
    derive_cached_prefill_memory_demand,
    fit_current_kv_continuation,
    Snapshot,
    Operation,
    Credit,
    retrieval_collection_after_shared_work,
    time_frozen_prefill_operations,
    RuntimeJointPlanner,
    ContinuationUnavailable,
    bind_selected_decode_growth_forecast,
    schedule as schedule_joint_lease,
)
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    set_current_stream_idx,
)
from sglang.srt.multiplex.history_quantile import history_percentile as _history_percentile_value

logger = logging.getLogger(__name__)

_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE"
)
_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT"
)
_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS"
)
_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST"
)
_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN"
)
_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT_ENV = (
    "SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT"
)
_RAG_ANN_PRE_SWITCH_CROSS_CLASS_PREFILL_DEBT_WEIGHT = 2.0
_RAG_ANN_PRE_SWITCH_CROSS_CLASS_MIN_NET_DEBT_MS = 50.0


def _profile_distribution_percentile_ms(
    distribution, percentile: float
) -> Optional[float]:
    if not isinstance(distribution, dict):
        return None
    points = []
    for quantile, key in ((50.0, "p50_ms"), (95.0, "p95_ms"), (99.0, "p99_ms")):
        try:
            value = float(distribution[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0.0:
            points.append((quantile, value))
    if not points:
        try:
            value = float(distribution["mean_ms"])
        except (KeyError, TypeError, ValueError):
            return None
        return value if value > 0.0 else None
    points.sort()
    quantiles, values = zip(*points)
    return float(
        np.interp(
            min(quantiles[-1], max(quantiles[0], float(percentile))),
            quantiles,
            values,
        )
    )


@dataclass(frozen=True)
class FrontendPlan:
    allow_ann: bool
    allow_prefill: bool
    frontend_needed: bool
    target_stream_idx: int
    reason: str
    target_reason: str
    budget_reason: str
    ann_wait_ms: float = 0.0
    prefill_wait_ms: float = 0.0
    predicted_decode_ms: Optional[float] = None
    decode_slack_ms: Optional[float] = None
    forced_ann: bool = False
    controlled_overlap: bool = False
    low_pressure_fast_path: bool = False
    launch_ann_early: bool = False
    immediate_handoff: bool = False
    dependency_prefill_due: bool = False
    prefill_sprint: bool = False
    prefill_exclusive_sprint: bool = False
    candidate_stream_idx: Optional[int] = None
    system_load_low: bool = False
    system_load_reason: str = ""
    decode_batch_p95: Optional[float] = None
    decode_queue_p95_ms: Optional[float] = None
    request_arrival_rps: Optional[float] = None
    prefill_work_equivalent_rps: Optional[float] = None
    dependency_handoff_wait_ms: float = 0.0
    prefill_forward_count: Optional[int] = None
    predicted_prefill_layer_ms: Optional[float] = None
    e2e_slack_ms: Optional[float] = None
    switch_benefit_ms: Optional[float] = None
    round_trip_switch_cost_ms: Optional[float] = None
    # Explicit fitted limits for a joint lease; None keeps legacy semantics.
    ann_batch_cap: Optional[int] = None
    joint_event_id: Optional[int] = None
    prefill_contract: Optional[tuple] = None


def frontend_plan_from_joint_decision(decision, allocation_to_stream):
    """Pure translation of SELECTED executable limits, shared by both modes.

    This is an intent, not activation: the caller must revalidate source state,
    finish required memory/compute transitions and prove backing before making
    it dispatchable. Never translate the rejected target when Full chose keep.
    """
    lease = decision.lease
    if not lease.supported:
        raise ValueError('Cannot translate unsupported joint lease')
    if type(decision.event_id) is not int or decision.event_id < 0:
        raise ValueError('Valid joint decision event required')
    executable = lease.executable
    idx = allocation_to_stream[executable.allocation]
    if type(idx) is not int or idx < 0:
        raise ValueError('Explicit valid compute allocation mapping required')
    rows, layers = executable.ann_batch, executable.prefill_layers
    if any(type(n) is not int or n < 0 for n in (rows,layers)):
        raise ValueError('Nonnegative fitted work limits required')
    return FrontendPlan(
        allow_ann=rows > 0, allow_prefill=layers > 0,
        frontend_needed=rows > 0 or layers > 0, target_stream_idx=idx,
        reason='joint_' + decision.reason, target_reason='joint_selected',
        budget_reason='joint_projected', prefill_forward_count=layers,
        ann_batch_cap=rows, joint_event_id=decision.event_id,
        prefill_contract=getattr(executable, 'prefill_contract', None))


def joint_plan_ready_for_dispatch(scheduler, plan):
    """Intent is not activation; only the serialized commit may set the marker.

    Marker publication must follow whole-lease state/backing revalidation and
    transition readiness. This local gate does not establish those guarantees.
    It expires at the next scheduler tick and requires the selected SM domain.
    """
    event = getattr(plan, 'joint_event_id', None)
    active = getattr(scheduler, 'rag_joint_active_event_id', None)
    return (type(event) is int and event >= 0 and type(active) is int
            and active == event
            and get_current_stream_idx() == plan.target_stream_idx)


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class SchedulerMultiplexMixin:

    def _record_rag_ann_request_arrival(
        self,
        enqueue_time: float,
        query_rows: int,
        *,
        request_id=None,
        is_priming: bool,
    ) -> None:
        """Record real load without letting control priming skew the RPS."""
        query_rows = max(1, int(query_rows))
        if is_priming:
            self.rag_ann_priming_requests = (
                getattr(self, "rag_ann_priming_requests", 0) + 1
            )
            self.rag_ann_priming_rows = (
                getattr(self, "rag_ann_priming_rows", 0) + query_rows
            )
            return
        if request_id is not None:
            request_starts = getattr(self, "rag_request_start_times", None)
            if request_starts is not None:
                request_starts[str(request_id)] = enqueue_time
        for _ in range(query_rows):
            self.rag_request_arrival_times.append(enqueue_time)

    def _rag_ann_pre_switch_p99_downsize_enabled(self) -> bool:
        """Resolve the experiment switch once and reject ambiguous values."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_p99_downsize_enabled",
            None,
        )
        if cached is not None:
            return bool(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_P99_DOWNSIZE_ENV,
            "1",
        ).strip()
        if configured not in {"0", "1"}:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE_ENV} must be 0 or 1"
            )
        enabled = configured == "1"
        self.rag_ann_pre_switch_p99_downsize_enabled = enabled
        return enabled

    def _rag_ann_pre_switch_adaptive_p99_credit_enabled(self) -> bool:
        """Resolve the opt-in local-credit switch exactly once."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_adaptive_p99_credit_enabled",
            None,
        )
        if cached is not None:
            return bool(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT_ENV,
            "0",
        ).strip()
        if configured not in {"0", "1"}:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT_ENV} "
                "must be 0 or 1"
            )
        enabled = configured == "1"
        self.rag_ann_pre_switch_adaptive_p99_credit_enabled = enabled
        return enabled

    def _rag_ann_pre_switch_downsized_min_robust_gain_ms(self) -> float:
        """Resolve the default-off downsized-launch gain threshold once."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_downsized_min_robust_gain_ms",
            None,
        )
        if cached is not None:
            return float(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS_ENV,
            "0",
        ).strip()
        try:
            minimum_ms = float(configured)
        except ValueError as error:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS_ENV} "
                "must be a finite nonnegative number"
            ) from error
        if not np.isfinite(minimum_ms) or minimum_ms < 0.0:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS_ENV} "
                "must be a finite nonnegative number"
            )
        self.rag_ann_pre_switch_downsized_min_robust_gain_ms = minimum_ms
        return minimum_ms

    def _rag_ann_pre_switch_quantile_aligned_switch_cost_enabled(self) -> bool:
        """Use a median switch estimate in the median robust-gain path."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_quantile_aligned_switch_cost_enabled",
            None,
        )
        if cached is not None:
            return bool(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST_ENV,
            "0",
        ).strip()
        if configured not in {"0", "1"}:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST_ENV} "
                "must be 0 or 1"
            )
        enabled = configured == "1"
        self.rag_ann_pre_switch_quantile_aligned_switch_cost_enabled = enabled
        return enabled

    def _rag_ann_pre_switch_batch_net_gain_enabled(self) -> bool:
        """Gate a downsize using all selected and residual request rows."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_batch_net_gain_enabled",
            None,
        )
        if cached is not None:
            return bool(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN_ENV,
            "0",
        ).strip()
        if configured not in {"0", "1"}:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN_ENV} must be 0 or 1"
            )
        enabled = configured == "1"
        self.rag_ann_pre_switch_batch_net_gain_enabled = enabled
        return enabled

    def _rag_ann_pre_switch_cross_class_debt_enabled(self) -> bool:
        """Trade already-unavoidable Prefill debt for larger ANN debt relief."""

        cached = getattr(
            self,
            "rag_ann_pre_switch_cross_class_debt_enabled",
            None,
        )
        if cached is not None:
            return bool(cached)
        configured = os.getenv(
            _RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT_ENV,
            "0",
        ).strip()
        if configured not in {"0", "1"}:
            raise ValueError(
                f"{_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT_ENV} must be 0 or 1"
            )
        enabled = configured == "1"
        self.rag_ann_pre_switch_cross_class_debt_enabled = enabled
        return enabled

    def _rag_ann_uses_partitioned_gc(self) -> bool:
        return getattr(self, "rag_ann_schedule", "decode-gc") in {
            "frontend-gc",
            "decode-gc",
        }

    def _rag_ann_uses_frontend_pool(self) -> bool:
        return getattr(self, "rag_ann_schedule", "decode-gc") == "frontend-gc"

    def _rag_ann_uses_decode_pool(self) -> bool:
        return getattr(self, "rag_ann_schedule", "decode-gc") == "decode-gc"

    def _rag_ann_work_conserving_decode_overlap_enabled(self) -> bool:
        return bool(
            getattr(self, "rag_policy_mode", "full") == "full"
            and self._rag_ann_uses_decode_pool()
            and getattr(
                self,
                "rag_ann_work_conserving_decode_overlap",
                False,
            )
        )

    def _select_rag_ann_epoch_stream(self, stream: torch.cuda.Stream, stream_idx: int):
        epoch_ms = max(0.0, getattr(self, "rag_ann_stream_epoch_ms", 0.0))
        if epoch_ms <= 0.0:
            return stream

        now = time.perf_counter()
        epoch_stream = getattr(self, "rag_ann_epoch_stream", None)
        epoch_idx = getattr(self, "rag_ann_epoch_stream_idx", None)
        epoch_until = getattr(self, "rag_ann_epoch_until", 0.0)
        real_group_num = getattr(self, "real_sm_group_num", 0)

        if epoch_stream is not None and now < epoch_until:
            # Never carry an ANN stream across PD partitions. The decode-only
            # group is a valid stable parent only when ANN shares Decode's pool.
            invalid_decode_only_epoch = bool(
                self._rag_ann_uses_frontend_pool()
                and real_group_num > 0
                and epoch_idx == real_group_num - 1
            )
            if epoch_idx == stream_idx and not invalid_decode_only_epoch:
                return epoch_stream

        self.rag_ann_epoch_stream = stream
        self.rag_ann_epoch_stream_idx = stream_idx
        self.rag_ann_epoch_until = now + epoch_ms / 1000.0
        return stream

    def _extend_rag_ann_frontend_residency(self):
        if not self._rag_ann_uses_frontend_pool():
            return
        epoch_ms = max(0.0, getattr(self, "rag_ann_stream_epoch_ms", 0.0))
        handoff_timeout_ms = max(
            0.0, getattr(self, "rag_ann_handoff_timeout_ms", 0.0)
        )
        dependency_lease_ms = 0.0
        if (
            getattr(self, "rag_policy_mode", "full") == "full"
            and handoff_timeout_ms > 0.0
        ):
            dependency_lease_ms = min(
                handoff_timeout_ms,
                max(
                    0.0,
                    getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0),
                )
                + max(
                    0.0,
                    getattr(self, "rag_lease_ann_service_estimate_ms", 0.0),
                ),
            )
        residency_ms = max(epoch_ms, dependency_lease_ms)
        if residency_ms > 0.0:
            self.rag_ann_frontend_residency_until = max(
                getattr(self, "rag_ann_frontend_residency_until", 0.0),
                time.perf_counter() + residency_ms / 1000.0,
            )

    def _get_rag_ann_stream(
        self,
        fallback_stream: Optional[torch.cuda.Stream] = None,
    ):
        if getattr(self, "rag_ann_schedule", "decode-gc") == "primary":
            ann_stream = getattr(self, "rag_ann_stream", None)
            if ann_stream is not None:
                return ann_stream
            if fallback_stream is not None:
                return fallback_stream
            return torch.cuda.current_stream()

        # A partitioned ANN schedule follows one lane of the current stream
        # group. frontend-gc uses the Prefill lane and therefore cannot launch
        # in Decode-only. decode-gc uses the Decode lane, including the all-SM
        # Decode-only group, so ANN and Decode share the same resource pool.
        if self._rag_ann_uses_partitioned_gc():
            stream_groups = getattr(self, "stream_groups", None)
            real_group_num = getattr(self, "real_sm_group_num", 0)
            stream_idx = get_current_stream_idx()
            if stream_groups and real_group_num > 0:
                if self._rag_ann_uses_frontend_pool() and stream_idx == (
                    real_group_num - 1
                ):
                    return None
                if 0 <= stream_idx < min(real_group_num, len(stream_groups)):
                    lane_idx = 1 if self._rag_ann_uses_decode_pool() else 0
                    return self._select_rag_ann_epoch_stream(
                        stream_groups[stream_idx][lane_idx], stream_idx
                    )

        ann_stream = getattr(self, "rag_ann_stream", None)
        if ann_stream is not None:
            return ann_stream

        if fallback_stream is not None:
            return fallback_stream
        return torch.cuda.current_stream()

    def _ann_task_num_queries(self, task) -> int:
        return int(task["queries"].shape[0])

    def _ann_pending_rows_until_limit(self, pending, limit: int) -> int:
        rows = 0
        for task in pending:
            task_rows = self._ann_task_num_queries(task)
            if rows > 0 and rows + task_rows > limit:
                break
            rows += task_rows
            if rows >= limit:
                break
        return rows

    @staticmethod
    def _increment_histogram(histogram: dict, key, amount: int = 1):
        key = str(key)
        histogram[key] = histogram.get(key, 0) + amount

    def _observe_ms_histogram(self, histogram: dict, value_ms: float):
        for upper_bound in (1, 2, 5, 10, 20, 50, 100):
            if value_ms <= upper_bound:
                self._increment_histogram(histogram, f"le_{upper_bound}_ms")
                return
        self._increment_histogram(histogram, "gt_100_ms")

    @staticmethod
    def _latency_distribution(values) -> dict:
        samples = list(values)
        return {
            "samples": len(samples),
            "mean_ms": float(np.mean(samples)) if samples else 0.0,
            "p50_ms": float(_history_percentile_value(samples, 50)) if samples else 0.0,
            "p95_ms": float(_history_percentile_value(samples, 95)) if samples else 0.0,
            "p99_ms": float(_history_percentile_value(samples, 99)) if samples else 0.0,
        }

    @staticmethod
    def _value_distribution(values) -> dict:
        samples = list(values)
        return {
            "samples": len(samples),
            "mean": float(np.mean(samples)) if samples else 0.0,
            "p50": float(_history_percentile_value(samples, 50)) if samples else 0.0,
            "p95": float(_history_percentile_value(samples, 95)) if samples else 0.0,
            "p99": float(_history_percentile_value(samples, 99)) if samples else 0.0,
        }

    def _capture_rag_handoff_batch_members(self, *, source_execution):
        """Freeze container membership separately from unfinished frontiers.

        Native result processing marks a request finished but does not filter
        running_batch.reqs until update_running_batch. Host merge shapes count
        those still-present members too. Preserve order and completion flags;
        this is not permission to dispatch finished work or reserve its KV.
        No tensors, token IDs, event polling or batch mutation are needed.
        """
        if capture_execution_stamp(self) != source_execution:
            return None
        batches = []
        for name in ('running_batch', 'split_prefill_batch', 'rag_prefill_merge_pending_batch'):
            batch = getattr(self, name, None)
            members = []
            for req in tuple(getattr(batch, 'reqs', ()) or ()):
                prompt = len(req.origin_input_ids)
                if prompt <= 0:
                    return None
                members.append((str(req.rid), prompt, getattr(req, 'finished_reason', None) is not None))
            if len({row[0] for row in members}) != len(members):
                raise ValueError('Duplicate request in observed handoff batch')
            batches.append((name, tuple(members)))
        if capture_execution_stamp(self) != source_execution:
            return None
        return tuple(batches)

    def _capture_rag_prefill_host_stage_shape(self, stage, incoming_batch):
        """Observe host result/merge shape before its mutation; no GPU waits."""
        if stage not in ('result', 'merge', 'deferred_merge'):
            raise ValueError('Explicit prefill host stage required')
        reqs = tuple(getattr(incoming_batch, 'reqs', ()) or ())
        if not reqs:
            return None
        prompts = tuple(len(req.origin_input_ids) for req in reqs)
        running = tuple(getattr(getattr(self, 'running_batch', None), 'reqs', ()) or ())
        if any(n <= 0 for n in prompts):
            return None
        return stage, prompts, len(running)

    def _record_rag_prefill_host_stage(self, shape, started_at, *, result_sequence=None):
        """Record successful host API duration, NOT CUDA completion latency.

        In particular merge_batch may enqueue device work. These observations
        must not be used as complete transition profiles without its event
        completion leg. Shape was captured before result processing can finish
        requests or mutate the running batch. No request IDs are retained.
        """
        if shape is None:
            return
        if result_sequence is not None and (type(result_sequence) is not int
                or result_sequence <= 0 or shape[0] not in ('merge', 'deferred_merge')):
            raise ValueError('Only merge stages may reference a recorded result')
        elapsed = (time.perf_counter() - started_at) * 1000.0
        if not math.isfinite(elapsed) or elapsed <= 0:
            return
        table = getattr(self, 'rag_prefill_host_stage_ms_by_shape', None)
        if table is None:
            table = self.rag_prefill_host_stage_ms_by_shape = {}
        table.setdefault(shape, deque(maxlen=64)).append(elapsed)
        if not getattr(getattr(self, 'server_args', None), 'rag_joint_handoff_profiling', False):
            return
        pending = getattr(self, 'rag_prefill_stage_completion_pending', None)
        if pending is None:
            pending = self.rag_prefill_stage_completion_pending = deque()
        if len(pending) >= 256:
            self.rag_prefill_stage_completion_dropped = getattr(
                self, 'rag_prefill_stage_completion_dropped', 0) + 1
            return
        event = torch.cuda.Event()
        event.record()  # Current result/merge stream; never synchronize here.
        sequence=getattr(self,'rag_prefill_stage_observation_sequence',0)+1
        self.rag_prefill_stage_observation_sequence=sequence
        pending.append((get_current_stream_idx(), shape, started_at, event,
                        sequence, started_at+elapsed/1000., result_sequence))
        return sequence

    def _poll_rag_prefill_stage_completions(self):
        """Nonblocking host-observed completion, including polling delay.

        This is submit-to-observed-release wall time, not GPU kernel duration.
        Result and merge windows can overlap; never sum their sample quantiles
        to infer a full handoff. Native dispatch uses its own completion fences.
        """
        pending = getattr(self, 'rag_prefill_stage_completion_pending', None)
        if not pending:
            return
        # Process one bounded snapshot. Leave an item owned by pending if query
        # raises; observation failure must not silently discard its evidence.
        for _ in range(len(pending)):
            partition, shape, started_at, event, sequence, host_finished_at, result_sequence = pending[0]
            if not event.query():
                pending.rotate(-1)
                continue
            elapsed = (time.perf_counter() - started_at) * 1000.0
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid stage completion clock')
            table = getattr(self, 'rag_prefill_stage_release_ms_by_shape', None)
            if table is None:
                table = self.rag_prefill_stage_release_ms_by_shape = {}
            table.setdefault((partition, *shape), deque(maxlen=64)).append(elapsed)
            ledger=getattr(self,'rag_prefill_stage_observation_ledger',None)
            if ledger is None:
                ledger=self.rag_prefill_stage_observation_ledger=deque(maxlen=4096)
            ledger.append(dict(sequence=sequence,partition=partition,stage=shape[0],
                result_sequence=result_sequence,
                prompt_tokens=shape[1],running_batch_size=shape[2],
                started_at_s=started_at,host_finished_at_s=host_finished_at,
                observed_complete_at_s=started_at+elapsed/1000.))
            pending.popleft()

    def _build_rag_joint_prefill_graph(self, *, executable, memory, frontiers, total_layers,
                                      queued_shapes=(), new_kv_key='joint:kv',
                                      existing_kv_key='active:kv', release_observation=None,
                                      prefix='prefill'):
        """Build selected prefill work from captured shapes, with no timing.

        Preserve submitted layers even with a zero new-work cap. Their KV is
        already owned; only subsequent layers use the new joint activation
        key. Queued members must match the exact selected cached FCFS contract.
        No queue selection, prefix-cache lookup, allocation or live observation.
        """
        if not memory.supported:
            return None
        frontiers = tuple(frontiers)
        by_id = {f.request_id: f for f in frontiers}
        if len(by_id) != len(frontiers):
            raise ValueError('Unique captured request frontiers required')
        cohort = tuple(f for f in frontiers if f.stage == 'prefill_active')
        contract = executable.prefill_contract
        if cohort:
            # Frontiers are ID-sorted; native batch/receipt order is not.
            # Restore the observed execution order without changing membership
            # or detaching any request from its own shape/credit.
            ordered_ids = (release_observation.request_ids if release_observation is not None
                           else tuple(row[0] for row in contract[1]) if contract is not None
                           else tuple(f.request_id for f in cohort))
            if (len(ordered_ids) != len(cohort) or len(set(ordered_ids)) != len(ordered_ids)
                    or set(ordered_ids) != {f.request_id for f in cohort}):
                return None
            cohort = tuple(by_id[rid] for rid in ordered_ids)
            submitted = {f.inflight_prefill_end_layer if f.inflight_prefill_end_layer is not None
                         else f.completed_prefill_layers for f in cohort}
            if len(submitted) != 1:
                return None
            actual = (next(iter(submitted)), tuple((f.request_id, f.prompt_tokens,
                f.prefill_context_tokens, f.prefill_extend_tokens) for f in cohort))
            if executable.prefill_layers and contract != actual:
                return None
        elif executable.prefill_layers:
            if (contract is None and not queued_shapes
                    and not any(f.stage=='prefill_queued' for f in frontiers)):
                # Current layer limit may remain nonzero while all observed
                # requests are upstream. There is no ready prefill to admit;
                # do not confuse that empty cohort with a missing contract.
                return ()
            if contract is None or contract[0] != 0:
                return None
            if tuple(row[0] for row in contract[1]) != tuple(
                    s.request_id for s in queued_shapes[:len(contract[1])]):
                return None
            shapes = {s.request_id: s for s in queued_shapes}
            selected = []
            for rid, prompt, context, extend in contract[1]:
                f = by_id.get(rid); shape = shapes.get(rid)
                if (f is None or shape is None or f.stage != 'prefill_queued'
                        or f.prompt_tokens != prompt
                        or (shape.prompt_tokens, shape.context_tokens, shape.extend_tokens)
                           != (prompt, context, extend)):
                    return None
                selected.append(replace(f, completed_prefill_layers=0,
                    prefill_context_tokens=context, prefill_extend_tokens=extend))
            cohort = tuple(selected)
        else:
            return ()
        inflight = any(f.inflight_prefill_end_layer is not None for f in cohort)
        offset = 0.
        if inflight:
            if (release_observation is None
                    or release_observation.request_ids != tuple(f.request_id for f in cohort)
                    or release_observation.tokens != sum(f.prefill_extend_tokens for f in cohort)
                    or any((f.inflight_prefill_start_layer, f.inflight_prefill_end_layer)
                           != (release_observation.start_layer, release_observation.end_layer)
                           for f in cohort)):
                return None
            offset = release_observation.observation_offset_ms
        keys = dict(memory.ready)
        if ((inflight and existing_kv_key not in keys)
                or (executable.prefill_layers and new_kv_key not in keys)):
            return None
        ops = build_prefill_bundle_operations(cohort, None, total_layers=total_layers,
            fitted_layers=executable.prefill_layers, kv_keys={f.request_id:new_kv_key for f in cohort},
            prefix=prefix, timing=False, inflight_observation_offset_ms=offset)
        # The core builder describes one batch with common KV names. Split
        # existing/new backing here to match the native submission boundary.
        return tuple(replace(op, kv_key=existing_kv_key, memory_keys=(existing_kv_key,))
            if isinstance(op.service_shape, tuple) and op.service_shape[0] == 'prefill_inflight'
            else op for op in ops)

    def _update_rag_running_batch_for_decision(self, batch):
        """Record preparation only after this iteration's real update succeeds."""
        self.rag_joint_prepared_decode = None
        batch = self.update_running_batch(batch)
        if batch is not None and not batch.is_empty():
            self.rag_joint_prepared_decode = (
                batch, batch.out_cache_loc, tuple(req.rid for req in batch.reqs))
        return batch

    def _capture_rag_decode_kv_prefixes(self, *, remaining_iterations,
                                        reserved_tokens, source_execution,
                                        growth_key='joint:kv', prepared_iteration=None):
        """Capture one real allocatable pool, without eviction or reservation.

        Caller supplies post-inflight work and ALL competing reservations from
        the same fitted proposal. The execution stamp must also be checked at
        commit; these prediction guarantees do not reserve allocator slots.
        prepared_iteration is true ONLY immediately after prepare_for_decode
        for this decision (never inferred from a stale out_cache_loc tensor).
        Its allocated but unsubmitted token remains uncredited work, while
        its physical backing must not be charged to the free pool again.
        """
        if source_execution != capture_execution_stamp(self):
            return None
        allocator=getattr(self,'token_to_kv_pool_allocator',None)
        if allocator is None or getattr(allocator,'page_size',None)!=1:
            return None
        if not isinstance(growth_key,str) or not growth_key:
            raise ValueError('Explicit growth backing required')
        # alloc() can use release_pages only through merge_and_sort_free().
        free=len(allocator.free_pages)
        if allocator.need_sort:
            free+=len(allocator.release_pages)
        published=allocator.current_mapped_capacity()
        if type(published) is not int or not 0<=free<=published:
            raise ValueError('Consistent mapped allocator capacity required')
        entries=tuple(remaining_iterations)
        running=tuple(str(req.rid) for req in self.running_batch.reqs)
        if tuple(rid for rid,_ in entries)!=running:
            return None
        if prepared_iteration is None:
            receipt=getattr(self,'rag_joint_prepared_decode',None)
            prepared_iteration=False
            if receipt is not None:
                batch,slots,members=receipt
                if (batch is not self.running_batch
                        or slots is not self.running_batch.out_cache_loc
                        or tuple(str(rid) for rid in members)!=running):
                    return None
                prepared_iteration=True
        if type(prepared_iteration) is not bool:
            raise ValueError('Explicit decode preparation state required')
        prepared=0
        if prepared_iteration:
            slots=getattr(self.running_batch,'out_cache_loc',None)
            mode=getattr(self.running_batch,'forward_mode',None)
            if (slots is None or mode is None or not mode.is_decode()
                    or slots.numel()!=len(running)):
                return None
            if any(type(count) is not int or count<=0 for _,count in entries):
                raise ValueError('Prepared decode members require remaining work')
            prepared=1
        additional=tuple((rid,count-prepared) for rid,count in entries)
        prefixes=allocate_decode_kv_prefixes(additional,free_tokens=free,
            reserved_tokens=reserved_tokens,page_size=allocator.page_size)
        if source_execution != capture_execution_stamp(self):
            return None
        return tuple((rid,(count+prepared,growth_key)) for rid,count in prefixes)

    def _build_rag_joint_generation_work(self, *, executable, memory, frontiers,
                                         reference_profiles, total_layers, query_decode_service,
                                         queued_shapes=(), release_observation=None,
                                         result_dependencies=(), merge_dependencies=(),
                                         existing_kv_key='active:kv', new_kv_key='joint:kv',
                                         observation_offset_ms=0., compute_ready_ms=0.,
                                         kv_growth_guarantees=(), kv_capture_inputs=None,
                                         separate_commit_gate=False):
        """Assemble captured prefill/handoff plus fixed-proposal decode joins.

        No profiling queries or candidate search. Caller supplies the exact
        old-result ordering and frozen decode callback; this does not model
        an unobserved in-flight decode iteration or unknown future requests.
        Ready active decode has no new-prefill join dependency. Held partial
        prefill keeps its submitted work but cannot appear as a decode member.
        """
        if not callable(query_decode_service):
            raise ValueError('Captured decode service callback required')
        frontiers=tuple(frontiers); profiles=dict(reference_profiles)
        prefill=self._build_rag_joint_prefill_graph(executable=executable,memory=memory,
            frontiers=frontiers,total_layers=total_layers,queued_shapes=queued_shapes,
            release_observation=release_observation,existing_kv_key=existing_kv_key,new_kv_key=new_kv_key)
        if prefill is None:
            return None
        operations=list(prefill)
        active=tuple(f for f in frontiers if f.stage=='decode')
        if kv_capture_inputs is not None:
            if kv_growth_guarantees:
                raise ValueError('Use native KV capture or supplied guarantees, not both')
            by_id={f.request_id:f for f in active}
            ordered=tuple(str(req.rid) for req in self.running_batch.reqs)
            if len(by_id)!=len(active) or set(ordered)!=set(by_id):
                return None
            if any(rid not in profiles for rid in ordered):
                return None
            # The normal decision boundary follows prior result processing;
            # a prepared next iteration has not yet emitted any output token.
            remaining=tuple((rid,remaining_decode_tokens(by_id[rid],profiles[rid]))
                            for rid in ordered)
            kv_growth_guarantees=self._capture_rag_decode_kv_prefixes(
                remaining_iterations=remaining,growth_key=new_kv_key,**kv_capture_inputs)
            if kv_growth_guarantees is None:
                return None
        joined=list(active)
        joins=[(f.request_id,()) for f in active]
        emitted=[(f.request_id,0) for f in active]
        kv=[(f.request_id,existing_kv_key) for f in active]
        groups=[]
        if prefill:
            ids=tuple(c.request for c in prefill[-1].credits if c.work=='prefill')
            by_id={f.request_id:f for f in frontiers}
            groups.append((tuple(by_id[rid] for rid in ids),prefill,'prefill-handoff'))
        pending=tuple(f for f in frontiers if f.stage=='prefill_merge_pending')
        if pending:
            groups.append((pending,(),'pending-handoff'))
        for cohort,bundles,prefix in groups:
            if any(f.request_id not in profiles for f in cohort):
                return None
            handoff=build_prefill_decode_handoff(cohort,profiles,prefill_operations=bundles,
                result_dependencies=result_dependencies,merge_dependencies=merge_dependencies,
                existing_kv_key=existing_kv_key,new_kv_key=new_kv_key,prefix=prefix,
                observation_offset_ms=observation_offset_ms,separate_commit_gate=separate_commit_gate)
            if handoff is None:
                continue  # Submitted but unfinished under zero new layers.
            extra,gates,tokens=handoff
            operations.extend(extra);joins.extend(gates);emitted.extend(tokens)
            joined.extend(cohort)
            kv.extend((f.request_id,extra[-1].kv_key) for f in cohort)
        # An idle decode stream is not the end of the request chain. Retain
        # the frozen service callback when already-observed upstream requests
        # may join later in H; prospective expansion needs an empty initial
        # cohort, not an absent generation model. No profile query here.
        prospective=any(f.stage in ('ann_pending','ann_inflight','input_preparation')
                        and f.generation_follows is True for f in frontiers)
        if not joined and not prospective:
            return tuple(operations),None
        if any(f.request_id not in profiles for f in joined):
            return None
        if any(key not in dict(memory.ready) for _,key in kv):
            return None
        growth=tuple(kv_growth_guarantees)
        guarantees=dict(growth)
        if len(guarantees)!=len(growth) or not set(guarantees)<={f.request_id for f in joined}:
            raise ValueError('Captured KV growth guarantee identity mismatch')
        for count,key in guarantees.values():
            if type(count) is not int or count<0 or not isinstance(key,str) or not key:
                raise ValueError('Explicit finite-prefix KV guarantee required')
            if key not in dict(memory.ready):
                return None
        plan=JoiningDecodePlan(tuple(joined),tuple((f.request_id,profiles[f.request_id]) for f in joined),
            tuple(joins),tuple(emitted),tuple(kv),query_decode_service,compute_ready_ms,growth)
        return tuple(operations),plan

    def _time_rag_frozen_prefill(self, operations, *, partition, layer_profile,
                                 release_profile, release_observation=None,
                                 query_bundle_service=None):
        """Price frozen prefill without recapturing clocks, layers or profiles.

        In-flight work belongs to its observed source partition; new bundles
        use the proposed partition. The skeleton must already include the
        observation offset so remaining time is not measured from window zero.
        """
        for op in operations:
            shape = op.service_shape
            if isinstance(shape, tuple) and shape and shape[0] == 'prefill_inflight':
                if release_observation is None:
                    return None
                interval = shape[1]
                observation = release_observation
                if (interval.request_ids != observation.request_ids
                        or interval.start_layer != observation.start_layer
                        or interval.end_layer != observation.end_layer
                        or sum(interval.extend_tokens) != observation.tokens
                        or op.ready_ms < observation.observation_offset_ms):
                    return None
        def new_service(shape):
            # A calibrated native-interval query sees the complete batch and
            # layer boundaries. Missing coverage is unsupported, not permission
            # to silently fall back to uniform per-layer scaling.
            if query_bundle_service is not None:
                return query_bundle_service(partition, shape)
            layer_ms = layer_profile.query_layer_ms(partition, sum(shape.extend_tokens))
            return None if layer_ms is None else layer_ms*(shape.end_layer-shape.start_layer)
        def remaining(shape):
            observation = release_observation
            if observation is None or release_profile is None:
                return None
            return release_profile.query_remaining_ms(
                (observation.partition, observation.tokens,
                 observation.end_layer-observation.start_layer), elapsed_ms=observation.elapsed_ms)
        return time_frozen_prefill_operations(operations, new_service,
                                              query_inflight_remaining=remaining)

    def _capture_rag_prefill_release_observation(self, *, window_started_at, source_execution):
        """Read one in-flight layer interval without polling/mutating its event."""
        if source_execution != capture_execution_stamp(self):
            return None
        meta = getattr(self,'rag_prefill_bundle_inflight_meta',None)
        batch = getattr(self,'split_prefill_batch',None)
        if (not meta or batch is None or meta.get('batch') is not batch
                or getattr(self,'rag_prefill_bundle_inflight_event',None) is None):
            return None
        contract = capture_prepared_prefill_contract(batch, allow_submitted_final=True)
        start = getattr(batch,'rag_completed_prefill_layers',None)
        end = meta.get('end_layer'); layers = meta.get('layers')
        submitted = meta.get('submitted_at'); partition = meta.get('stream_idx')
        if (contract is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end or contract[0] != end or layers != end-start
                or type(partition) is not int or partition < 0
                or type(submitted) not in (int,float) or not math.isfinite(submitted)):
            return None
        tokens = sum(shape[3] for shape in contract[1])
        if tokens != meta.get('tokens'):
            return None
        now = time.perf_counter()
        age = (now-submitted)*1000.; offset = (now-window_started_at)*1000.
        if not math.isfinite(age) or not math.isfinite(offset) or age < 0 or offset < 0:
            return None
        if source_execution != capture_execution_stamp(self):
            return None
        return PrefillReleaseObservation(partition,tokens,start,end,
            tuple(shape[0] for shape in contract[1]),age,offset)

    def _record_rag_prefill_host_release(self, meta):
        """Called only after the matching completion event reports ready.

        Host submit-to-observed-completion includes submission/queue/poll gaps,
        not just CUDA service. Keep it separate for age-conditioned forecasts.
        Exact partition/tokens/layers only; index/workload scope remains external.
        """
        if meta.get('host_release_recorded'):
            return
        started = meta.get('submitted_at')
        shape = (meta.get('stream_idx'), meta.get('tokens'), meta.get('layers'))
        if (type(started) not in (int,float) or not math.isfinite(started)
                or type(shape[0]) is not int or shape[0] < 0
                or any(type(n) is not int or n <= 0 for n in shape[1:])):
            return
        elapsed = (time.perf_counter()-started)*1000.
        if not math.isfinite(elapsed) or elapsed <= 0:
            return
        samples = getattr(self,'rag_prefill_host_release_ms_by_shape',None)
        if samples is None:
            samples = self.rag_prefill_host_release_ms_by_shape = {}
        samples.setdefault(shape,deque(maxlen=128)).append(elapsed)
        meta['host_release_recorded'] = True

    def _record_ann_gpu_service(self, task, gpu_elapsed_ms: float) -> bool:
        """Record one valid completed ANN launch for service prediction.

        A just-completed CUDA event can transiently report a zero elapsed
        value even though ``gpu_poll()`` already returned ready.  Do not
        publish the task's recorded flag until a positive sample is
        available, so the scheduler can retry after the full result poll.
        """
        if task.get("ann_gpu_service_recorded", False):
            return False
        gpu_elapsed_ms = float(gpu_elapsed_ms)
        if not math.isfinite(gpu_elapsed_ms) or gpu_elapsed_ms <= 0.0:
            return False

        task["gpu_elapsed_ms"] = gpu_elapsed_ms
        task["ann_gpu_service_recorded"] = True
        batch_size = max(1, self._ann_task_num_queries(task))
        histories = getattr(self, "rag_ann_gpu_service_ms_by_batch_size", None)
        if histories is None:
            histories = {}
            self.rag_ann_gpu_service_ms_by_batch_size = histories
        histories.setdefault(batch_size, deque(maxlen=64)).append(gpu_elapsed_ms)
        service_samples = getattr(self, "rag_ann_gpu_service_samples", None)
        if service_samples is None:
            service_samples = deque(maxlen=4096)
            self.rag_ann_gpu_service_samples = service_samples
        service_samples.append(
            (task.get("stream_idx"), batch_size, gpu_elapsed_ms, time.perf_counter())
        )
        partition = task.get("stream_idx")
        if type(partition) is int and partition >= 0:
            partition_histories = getattr(self, "rag_ann_gpu_service_ms_by_partition_batch_size", None)
            if partition_histories is None:
                partition_histories = {}
                self.rag_ann_gpu_service_ms_by_partition_batch_size = partition_histories
            partition_histories.setdefault((partition, batch_size), deque(maxlen=64)).append(gpu_elapsed_ms)
            hot_counts = getattr(task.get('result'), 'hot_counts', None)
            if hot_counts is not None:
                counts = tuple(int(value) for value in hot_counts)
                if len(counts) == batch_size and all(n >= 0 for n in counts) and any(counts):
                    # A trimmed index changes GPU work even at the same batch
                    # size. Keep this separate from the legacy row-only table.
                    # Counts are not list-size/content conditioning; callers
                    # must still restrict the index/workload profile domain.
                    shaped = getattr(self, 'rag_ann_gpu_service_ms_by_partition_hot_counts', None)
                    if shaped is None:
                        shaped = self.rag_ann_gpu_service_ms_by_partition_hot_counts = {}
                    shaped.setdefault((partition, counts), deque(maxlen=64)).append(gpu_elapsed_ms)
            result = task.get("result")
            visits = getattr(result, "gpu_list_visits", None)
            vectors = getattr(result, "gpu_vectors_scanned", None)
            max_query = getattr(result, "gpu_max_query_vectors", None)
            max_list = getattr(result, "gpu_max_list_vectors", None)
            query_sq = getattr(result, "gpu_query_vector_sq_sum", None)
            if (
                type(visits) is int
                and visits > 0
                and type(vectors) is int
                and vectors > 0
                and type(max_query) is int
                and max_query > 0
                and type(max_list) is int
                and max_list > 0
                and type(query_sq) is int
                and query_sq > 0
            ):
                work_history = getattr(
                    self, "rag_ann_gpu_service_ms_by_partition_work", None
                )
                if work_history is None:
                    work_history = self.rag_ann_gpu_service_ms_by_partition_work = {}
                key = (partition, batch_size, visits, vectors, max_query, max_list,
                       query_sq)
                work_history.setdefault(key, deque(maxlen=64)).append(gpu_elapsed_ms)
            launched = task.get('launch_time')
            if type(launched) in (int,float) and math.isfinite(launched):
                observed = time.perf_counter()
                wall_ms = (observed-launched)*1000.0
                if math.isfinite(wall_ms) and wall_ms > 0:
                    # Host submission-to-observed-GPU-completion, distinct
                    # from CUDA execution time and full CPU/GPU result join.
                    # Includes host work/stream wait/polling reaction; do not
                    # subtract host age from the GPU-event service profile.
                    release = getattr(self,'rag_ann_gpu_release_ms_by_partition_batch_size',None)
                    if release is None:
                        release = self.rag_ann_gpu_release_ms_by_partition_batch_size = {}
                    release.setdefault((partition,batch_size),deque(maxlen=64)).append(wall_ms)
                    task['ann_gpu_completion_observed_at'] = observed
        return True

    @staticmethod
    def _ann_result_gpu_path(result) -> Optional[str]:
        """Classify a launched result without importing executor result types."""
        if result is None:
            return None
        if hasattr(result, "gpu_result"):
            return (
                "hybrid_gpu"
                if getattr(result, "gpu_result", None) is not None
                else None
            )
        return "full_gpu"

    def _current_ann_gpu_path(self) -> str:
        return (
            "hybrid_gpu"
            if getattr(getattr(self, "ann_executor", None), "hot_index_mode", False)
            else "full_gpu"
        )

    def _record_ann_completion_service(
        self,
        task,
        service_ms_samples,
        gpu_elapsed_ms: float,
    ):
        """Learn one launch-to-output sample for one completed GPU batch.

        ``service_ms_samples`` contains ``latency_ms - queue_wait_ms`` for
        every request merged into the batch. They share one launch, so the
        conservative maximum is recorded once for the batch.
        """
        if task.get("ann_completion_service_recorded", False):
            return
        task["ann_completion_service_recorded"] = True

        if (
            task.get("ann_collection_mode", "worker") != "worker"
            or task.get("launch_context") == "pre_switch"
        ):
            return
        path = task.get("ann_gpu_path")
        if path not in {"full_gpu", "hybrid_gpu"}:
            return
        try:
            services = [float(value) for value in service_ms_samples]
            gpu_elapsed_ms = float(gpu_elapsed_ms)
        except (TypeError, ValueError):
            return
        if (
            not services
            or not np.isfinite(gpu_elapsed_ms)
            or gpu_elapsed_ms < 0.0
            or any(not np.isfinite(value) or value <= 0.0 for value in services)
        ):
            return

        completion_service_ms = max(services)
        post_gpu_tail_ms = completion_service_ms - gpu_elapsed_ms
        if not np.isfinite(post_gpu_tail_ms) or post_gpu_tail_ms < 0.0:
            return

        batch_size = max(1, self._ann_task_num_queries(task))
        completion_histories = getattr(
            self,
            "rag_ann_completion_service_ms_by_path_batch",
            None,
        )
        if completion_histories is None:
            completion_histories = {}
            self.rag_ann_completion_service_ms_by_path_batch = (
                completion_histories
            )
        tail_histories = getattr(
            self,
            "rag_ann_post_gpu_tail_ms_by_path_batch",
            None,
        )
        if tail_histories is None:
            tail_histories = {}
            self.rag_ann_post_gpu_tail_ms_by_path_batch = tail_histories
        completion_histories.setdefault(path, {}).setdefault(
            batch_size, deque(maxlen=64)
        ).append(completion_service_ms)
        tail_histories.setdefault(path, {}).setdefault(
            batch_size, deque(maxlen=64)
        ).append(post_gpu_tail_ms)
        task["ann_completion_service_ms"] = completion_service_ms
        task["ann_post_gpu_tail_ms"] = post_gpu_tail_ms

    def _record_ann_worker_collection_cost(self, task, elapsed_ms: float):
        """Record one normal-path ids_to_cpu/merge cost per completed batch."""
        if task.get("ann_worker_collection_cost_recorded", False):
            return
        task["ann_worker_collection_cost_recorded"] = True
        if (
            task.get("ann_collection_mode", "worker") != "worker"
            or task.get("launch_context") == "pre_switch"
            or task.get("ann_gpu_path") != "full_gpu"
            or not task.get("ann_collection_succeeded", False)
        ):
            return
        try:
            elapsed_ms = float(elapsed_ms)
        except (TypeError, ValueError):
            return
        if not np.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            return
        batch_size = max(1, self._ann_task_num_queries(task))
        histories = getattr(
            self,
            "rag_ann_worker_collection_ms_by_path_batch",
            None,
        )
        if histories is None:
            histories = {}
            self.rag_ann_worker_collection_ms_by_path_batch = histories
        histories.setdefault("full_gpu", {}).setdefault(
            batch_size, deque(maxlen=64)
        ).append(elapsed_ms)

    def _record_ann_worker_send_cost(self, task, send_samples_ms):
        """Record one conservative per-output send sample per normal batch."""
        if task.get("ann_worker_send_cost_recorded", False):
            return
        task["ann_worker_send_cost_recorded"] = True
        if (
            task.get("ann_collection_mode") != "worker"
            or task.get("launch_context") == "pre_switch"
            or task.get("ann_gpu_path") != "full_gpu"
            or not task.get("ann_collection_succeeded", False)
        ):
            return
        try:
            samples = [float(value) for value in send_samples_ms]
        except (TypeError, ValueError):
            return
        if not samples or any(
            not np.isfinite(value) or value < 0.0 for value in samples
        ):
            return
        batch_size = max(1, self._ann_task_num_queries(task))
        histories = getattr(
            self,
            "rag_ann_worker_send_per_output_ms_by_path_batch",
            None,
        )
        if histories is None:
            histories = {}
            self.rag_ann_worker_send_per_output_ms_by_path_batch = histories
        histories.setdefault("full_gpu", {}).setdefault(
            batch_size, deque(maxlen=64)
        ).append(max(samples))

    def _record_ann_inline_collection_tail(
        self, task, collection_ms: float, send_samples_ms
    ):
        """Record pure inline ids_to_cpu/merge plus send wall once per batch."""
        if task.get("ann_inline_collection_tail_recorded", False):
            return
        task["ann_inline_collection_tail_recorded"] = True
        if (
            task.get("ann_collection_mode") != "inline"
            or not task.get("ann_collection_succeeded", False)
            or task.get("ann_gpu_path") != "full_gpu"
        ):
            return
        try:
            collection_ms = float(collection_ms)
            send_samples = [float(value) for value in send_samples_ms]
        except (TypeError, ValueError):
            return
        if (
            not np.isfinite(collection_ms)
            or collection_ms < 0.0
            or not send_samples
            or any(
                not np.isfinite(value) or value < 0.0
                for value in send_samples
            )
        ):
            return
        elapsed_ms = collection_ms + sum(send_samples)
        batch_size = max(1, self._ann_task_num_queries(task))
        histories = getattr(
            self,
            "rag_ann_inline_collection_tail_ms_by_path_batch",
            None,
        )
        if histories is None:
            histories = {}
            self.rag_ann_inline_collection_tail_ms_by_path_batch = histories
        histories.setdefault("full_gpu", {}).setdefault(
            batch_size, deque(maxlen=64)
        ).append(elapsed_ms)
        task["ann_inline_collection_tail_ms"] = elapsed_ms

    def _predict_ann_post_gpu_tail_ms(
        self, batch_size: int, path: str
    ) -> Optional[float]:
        min_samples = max(
            4, int(getattr(self, "rag_ann_service_time_min_samples", 4))
        )
        samples = list(
            getattr(
                self,
                "rag_ann_post_gpu_tail_ms_by_path_batch",
                {},
            )
            .get(path, {})
            .get(max(1, int(batch_size)), ())
        )
        if len(samples) < min_samples:
            return None
        if any(
            not np.isfinite(value) or float(value) < 0.0
            for value in samples
        ):
            return None
        return max(0.0, float(_history_percentile_value(samples, 99)))

    def _predict_ann_inline_collection_tail_ms(
        self, batch_size: int, path: str
    ) -> Optional[float]:
        """Predict switch-blocking full-GPU collection plus result sends."""
        batch_size = max(1, int(batch_size))
        if path != "full_gpu":
            return None

        min_samples = max(
            4, int(getattr(self, "rag_ann_service_time_min_samples", 4))
        )
        samples = list(
            getattr(
                self,
                "rag_ann_inline_collection_tail_ms_by_path_batch",
                {},
            )
            .get(path, {})
            .get(batch_size, ())
        )
        valid_inline = len(samples) >= min_samples and all(
            np.isfinite(value) and float(value) >= 0.0 for value in samples
        )
        if valid_inline:
            predicted_ms = max(0.0, float(_history_percentile_value(samples, 99)))
            source = "inline"
            sample_count = len(samples)
        else:
            worker_collection = list(
                getattr(
                    self,
                    "rag_ann_worker_collection_ms_by_path_batch",
                    {},
                )
                .get("full_gpu", {})
                .get(batch_size, ())
            )
            worker_send = list(
                getattr(
                    self,
                    "rag_ann_worker_send_per_output_ms_by_path_batch",
                    {},
                )
                .get("full_gpu", {})
                .get(batch_size, ())
            )
            valid_worker = (
                len(worker_collection) >= min_samples
                and len(worker_send) >= min_samples
                and all(
                    np.isfinite(value) and float(value) >= 0.0
                    for value in worker_collection
                )
                and all(
                    np.isfinite(value) and float(value) >= 0.0
                    for value in worker_send
                )
            )
            if valid_worker:
                predicted_ms = float(
                    _history_percentile_value(worker_collection, 99)
                ) + batch_size * float(_history_percentile_value(worker_send, 99))
                source = "worker_bootstrap"
                sample_count = min(
                    len(worker_collection), len(worker_send)
                )
            else:
                predicted_ms = None
                source = "unavailable"
                sample_count = min(
                    len(worker_collection), len(worker_send)
                )
        self.rag_ann_inline_tail_prediction_last = {
            "path": path,
            "batch_size": batch_size,
            "source": source,
            "samples": sample_count,
            "predicted_ms": predicted_ms,
        }
        return predicted_ms

    def _predict_ann_completion_service_quantiles_ms(
        self,
        batch_size: int,
        predicted_gpu_ms: float,
        path: Optional[str] = None,
    ) -> Optional[tuple[float, float]]:
        """Return exact-path/exact-batch normal completion p50 and p99.

        The p50 is the empirical normal-path launch-to-output service time
        used by the robust decision envelope. The p99 retains the conservative
        GPU-plus-post-GPU-tail floor used by the tail-risk envelope.
        """
        batch_size = max(1, int(batch_size))
        path = self._current_ann_gpu_path() if path is None else path
        min_samples = max(
            4, int(getattr(self, "rag_ann_service_time_min_samples", 4))
        )
        try:
            predicted_gpu_ms = float(predicted_gpu_ms)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(predicted_gpu_ms) or predicted_gpu_ms < 0.0:
            return None
        samples = list(
            getattr(
                self,
                "rag_ann_completion_service_ms_by_path_batch",
                {},
            )
            .get(path, {})
            .get(batch_size, ())
        )
        tail_p99_ms = self._predict_ann_post_gpu_tail_ms(batch_size, path)
        last_prediction = {
            "path": str(path),
            "batch_size": batch_size,
            "service_samples": len(samples),
            "tail_samples": len(
                getattr(
                    self,
                    "rag_ann_post_gpu_tail_ms_by_path_batch",
                    {},
                )
                .get(path, {})
                .get(batch_size, ())
            ),
            "mature": False,
            "predicted_completion_p50_ms": None,
            "predicted_completion_p99_ms": None,
            "predicted_completion_ms": None,
        }
        self.rag_ann_completion_prediction_last = last_prediction
        if (
            len(samples) < min_samples
            or tail_p99_ms is None
            or any(
                not np.isfinite(value) or float(value) <= 0.0
                for value in samples
            )
        ):
            return None
        service_p50_ms = max(0.001, float(_history_percentile_value(samples, 50)))
        service_p99_ms = max(0.001, float(_history_percentile_value(samples, 99)))
        prediction_p99_ms = max(
            service_p99_ms,
            max(0.001, predicted_gpu_ms) + tail_p99_ms,
        )
        last_prediction["mature"] = True
        last_prediction["predicted_completion_p50_ms"] = service_p50_ms
        last_prediction["predicted_completion_p99_ms"] = prediction_p99_ms
        # Retain the original field as a compatibility alias for p99.
        last_prediction["predicted_completion_ms"] = prediction_p99_ms
        return service_p50_ms, prediction_p99_ms

    def _predict_ann_completion_service_ms(
        self,
        batch_size: int,
        predicted_gpu_ms: float,
        path: Optional[str] = None,
    ) -> Optional[float]:
        """Return an exact-path/exact-batch conservative completion p99."""
        predictions = self._predict_ann_completion_service_quantiles_ms(
            batch_size,
            predicted_gpu_ms,
            path,
        )
        return None if predictions is None else predictions[1]

    def _ann_completion_predictor_metrics(self) -> dict:
        min_samples = max(
            4, int(getattr(self, "rag_ann_service_time_min_samples", 4))
        )
        completion_histories = getattr(
            self,
            "rag_ann_completion_service_ms_by_path_batch",
            {},
        )
        tail_histories = getattr(
            self,
            "rag_ann_post_gpu_tail_ms_by_path_batch",
            {},
        )
        inline_histories = getattr(
            self,
            "rag_ann_inline_collection_tail_ms_by_path_batch",
            {},
        )
        worker_collection_histories = getattr(
            self,
            "rag_ann_worker_collection_ms_by_path_batch",
            {},
        )
        worker_send_histories = getattr(
            self,
            "rag_ann_worker_send_per_output_ms_by_path_batch",
            {},
        )
        paths = (
            set(completion_histories)
            | set(tail_histories)
            | set(inline_histories)
            | set(worker_collection_histories)
            | set(worker_send_histories)
        )
        by_path_batch = {}
        for path in sorted(paths):
            path_metrics = {}
            batch_sizes = (
                set(completion_histories.get(path, {}))
                | set(tail_histories.get(path, {}))
                | set(inline_histories.get(path, {}))
                | set(worker_collection_histories.get(path, {}))
                | set(worker_send_histories.get(path, {}))
            )
            for batch_size in sorted(batch_sizes):
                services = list(
                    completion_histories.get(path, {}).get(batch_size, ())
                )
                tails = list(
                    tail_histories.get(path, {}).get(batch_size, ())
                )
                inline_tails = list(
                    inline_histories.get(path, {}).get(batch_size, ())
                )
                worker_collection = list(
                    worker_collection_histories.get(path, {}).get(
                        batch_size, ()
                    )
                )
                worker_send = list(
                    worker_send_histories.get(path, {}).get(batch_size, ())
                )
                path_metrics[str(batch_size)] = {
                    "service_samples": len(services),
                    "tail_samples": len(tails),
                    "inline_tail_samples": len(inline_tails),
                    "worker_collection_samples": len(worker_collection),
                    "worker_send_samples": len(worker_send),
                    "mature": (
                        len(services) >= min_samples
                        and len(tails) >= min_samples
                    ),
                    "service_p99_ms": (
                        float(_history_percentile_value(services, 99))
                        if services
                        else None
                    ),
                    "service_p50_ms": (
                        float(_history_percentile_value(services, 50))
                        if services
                        else None
                    ),
                    "tail_p99_ms": (
                        float(_history_percentile_value(tails, 99)) if tails else None
                    ),
                    "inline_tail_p99_ms": (
                        float(_history_percentile_value(inline_tails, 99))
                        if inline_tails
                        else None
                    ),
                    "worker_collection_p99_ms": (
                        float(_history_percentile_value(worker_collection, 99))
                        if worker_collection
                        else None
                    ),
                    "worker_send_p99_ms": (
                        float(_history_percentile_value(worker_send, 99))
                        if worker_send
                        else None
                    ),
                }
            by_path_batch[str(path)] = path_metrics
        return {
            "min_samples": min_samples,
            "by_path_batch": by_path_batch,
            "last_prediction": dict(
                getattr(self, "rag_ann_completion_prediction_last", {})
            ),
            "last_inline_tail_prediction": dict(
                getattr(self, "rag_ann_inline_tail_prediction_last", {})
            ),
        }

    def _rag_profile_ann_gpu_service_ms(
        self, batch_size: int
    ) -> Optional[float]:
        batch_size = max(1, int(batch_size))
        profiles = getattr(self, "rag_lease_profile_priors", {}).get(
            "ann_gpu_service_by_batch_size", {}
        )
        if batch_size in profiles:
            return max(0.001, float(profiles[batch_size]))
        available = sorted(
            (max(1, int(size)), max(0.001, float(value)))
            for size, value in profiles.items()
        )
        if not available:
            return None
        larger = [item for item in available if item[0] >= batch_size]
        if larger:
            return larger[0][1]
        observed_size, observed_ms = available[-1]
        return observed_ms * batch_size / observed_size

    def _preview_rag_legacy_ann_work(self, pending, batch_size):
        """Read current-residency routed work without submitting or mutating."""
        if pending is None or batch_size <= 0:
            return None
        stamp=capture_execution_stamp(self)
        controller=getattr(self,'shared_hbm_controller',None)
        executor=getattr(self,'ann_executor',None)
        if controller is None or executor is None:
            return None
        if (getattr(controller,'pending_ann_reclaim',None) is not None
                or getattr(controller,'pending_ivf_reload',None) is not None
                or getattr(controller,'pending_kv_native_map',None) is not None):
            return None
        residency=executor.preview_shared_ivf_residency()
        if residency is None:
            return None
        rows=0
        total=[0,0,0,0,0]
        for task in pending:
            count=self._ann_task_num_queries(task)
            if rows+count>batch_size:
                break
            future=task.get('route_future')
            if (task.get('result') is not None or task.get('launch_time') is not None
                    or future is None or not future.done() or future.cancelled()
                    or future.exception() is not None):
                return None
            prepared=future.result()
            residual=executor.preview_prepared_hot_spill(prepared,residency)
            work=executor.describe_gpu_routed_work(residual)
            if work is None:
                return None
            total[0]+=work[0]
            total[1]+=work[1]
            total[2]=max(total[2],work[2])
            total[3]=max(total[3],work[3])
            total[4]+=work[4]
            rows+=count
            if rows==batch_size:
                break
        if rows!=batch_size or not all(value>0 for value in total):
            return None
        if stamp!=capture_execution_stamp(self):
            return None
        return tuple(total)

    def _record_rag_legacy_calibrated_fallback_reason(self, stage, reason):
        counts=getattr(self,'rag_legacy_calibrated_fallback_reason_counts',{})
        key=f'{stage}:{reason}'
        counts[key]=counts.get(key,0)+1
        self.rag_legacy_calibrated_fallback_reason_counts=counts

    def _predict_rag_legacy_calibrated_ann_ms(
        self, pending, batch_size, *, target_stream_idx=None
    ):
        bundle=getattr(self,'rag_legacy_calibrated_bundle',None)
        if bundle is None:
            return None
        work=self._preview_rag_legacy_ann_work(pending,batch_size)
        partition=(get_current_stream_idx() if target_stream_idx is None
                   else int(target_stream_idx))
        if work is None:
            self._record_rag_legacy_calibrated_fallback_reason(
                'ann','work_preview_unavailable')
            return None
        if (not 0<=partition<len(self.sm_counts)
                or self.sm_counts[partition][1]<=0):
            self._record_rag_legacy_calibrated_fallback_reason(
                'ann','partition_unavailable')
            return None
        model=bundle.stage
        predicted=model.predict_ann_ms(self.sm_counts[partition][1],batch_size,*work,
            expected_identity=bundle.identity)
        if predicted is None:
            predicted=model.predict_ann_nearest_ms(self.sm_counts[partition][1],batch_size,*work,
                expected_identity=bundle.identity)
        if predicted is None:
            self._record_rag_legacy_calibrated_fallback_reason(
                'ann','stage_cell_unavailable')
            return None
        has_prefill=bool(getattr(self,'split_prefill_batch',None)
            and not self.split_prefill_batch.is_empty())
        has_decode=self._decode_batch_active()
        overlap=('prefill_decode' if has_prefill and has_decode else
                 'prefill' if has_prefill else 'decode' if has_decode else None)
        correction_overlap=overlap
        if overlap == 'prefill_decode':
            # No accepted three-way correction cell exists. Compose the two
            # accepted pairwise guards conservatively without multiplying
            # correlated penalties.
            from sglang.srt.multiplex.stage_predictor import ann_correction_features
            features=ann_correction_features(batch_size,work[0],work[1])
            pairwise=[bundle.corrections.query('ann',partition,name,features,
                expected_identity=bundle.identity) for name in ('prefill','decode')]
            if any(factors is None for factors in pairwise):
                self._record_rag_legacy_calibrated_fallback_reason(
                    'ann','three_way_pairwise_correction_unavailable')
                return None
            predicted*=max(float(factors[1]) for factors in pairwise)
            correction_overlap='max(prefill,decode)'
        elif overlap is not None:
            from sglang.srt.multiplex.stage_predictor import ann_correction_features
            factors=bundle.corrections.query('ann',partition,overlap,
                ann_correction_features(batch_size,work[0],work[1]),
                expected_identity=bundle.identity)
            if factors is None:
                self._record_rag_legacy_calibrated_fallback_reason(
                    'ann',f'{overlap}_correction_unavailable')
                return None
            predicted*=factors[1]
        counts=getattr(self,'rag_legacy_calibrated_prediction_counts',{})
        counts['ann']=counts.get('ann',0)+1
        self.rag_legacy_calibrated_prediction_counts=counts
        self.rag_legacy_calibrated_last_ann=dict(partition=partition,batch_size=batch_size,
            work=work,overlap=overlap,correction_overlap=correction_overlap,
            predicted_ms=predicted)
        return predicted

    def _predict_ann_gpu_service_ms(
        self, batch_size: int, *, pending=None, target_stream_idx=None
    ) -> float:
        """Predict p-tail GPU service for a batch using mature nearby buckets."""
        calibrated=self._predict_rag_legacy_calibrated_ann_ms(
            pending, batch_size, target_stream_idx=target_stream_idx
        )
        if calibrated is not None:
            return max(0.001,float(calibrated))
        if getattr(self,'rag_legacy_calibrated_bundle',None) is not None and pending is not None:
            counts=getattr(self,'rag_legacy_calibrated_fallback_counts',{})
            counts['ann']=counts.get('ann',0)+1
            self.rag_legacy_calibrated_fallback_counts=counts
        batch_size = max(1, int(batch_size))
        percentile = min(
            100.0,
            max(
                0.001,
                getattr(self, "rag_ann_service_time_percentile", 95.0),
            ),
        )
        min_samples = max(
            1, getattr(self, "rag_ann_service_time_min_samples", 4)
        )
        histories = getattr(self, "rag_ann_gpu_service_ms_by_batch_size", {})
        exact = list(histories.get(batch_size, ()))
        if len(exact) >= min_samples:
            return max(0.001, float(_history_percentile_value(exact, percentile)))

        mature = []
        for observed_size, history in histories.items():
            samples = list(history)
            if len(samples) < min_samples:
                continue
            mature.append(
                (
                    max(1, int(observed_size)),
                    max(0.001, float(_history_percentile_value(samples, percentile))),
                )
            )

        if mature:
            mature.sort()
            larger = [item for item in mature if item[0] >= batch_size]
            if larger:
                # A measured larger batch is a conservative upper bound.
                predicted_ms = larger[0][1]
            else:
                observed_size, observed_ms = mature[-1]
                predicted_ms = observed_ms * batch_size / observed_size
            if exact:
                predicted_ms = max(
                    predicted_ms, float(_history_percentile_value(exact, percentile))
                )
            return max(0.001, predicted_ms)

        profile_ms = self._rag_profile_ann_gpu_service_ms(batch_size)
        if profile_ms is not None:
            return profile_ms

        # Treat the configured lease estimate as a one-query cold-start cost.
        # Linear scaling is conservative until batching efficiency is learned.
        per_query_ms = max(
            0.001, getattr(self, "rag_lease_ann_service_estimate_ms", 10.0)
        )
        predicted_ms = per_query_ms * batch_size
        if exact:
            predicted_ms = max(
                predicted_ms, float(_history_percentile_value(exact, percentile))
            )
        return max(0.001, predicted_ms)

    def _predict_ann_gpu_critical_path_ms(
        self, batch_size: int
    ) -> Optional[float]:
        """Return exact-batch GPU p99 for deadline counterfactuals."""
        batch_size = max(1, int(batch_size))
        min_samples = max(
            4, int(getattr(self, "rag_ann_service_time_min_samples", 4))
        )
        samples = list(
            getattr(self, "rag_ann_gpu_service_ms_by_batch_size", {}).get(
                batch_size, ()
            )
        )
        if len(samples) < min_samples or any(
            not np.isfinite(value) or float(value) <= 0.0
            for value in samples
        ):
            return self._rag_profile_ann_gpu_service_ms(batch_size)
        return max(0.001, float(_history_percentile_value(samples, 99)))

    def _effective_ann_microbatch_rows(self) -> int:
        """Translate the learned GPU-time cap into a query-row limit."""
        hard_limit = max(1, getattr(self, "rag_ann_max_batch_size", 1))
        time_limit_ms = max(
            0.0, getattr(self, "rag_ann_max_microbatch_gpu_ms", 0.0)
        )
        if time_limit_ms <= 0.0:
            effective_limit = hard_limit
        else:
            effective_limit = 1
            monotonic_prediction_ms = 0.0
            for rows in range(1, hard_limit + 1):
                monotonic_prediction_ms = max(
                    monotonic_prediction_ms,
                    self._predict_ann_gpu_service_ms(rows),
                )
                if rows == 1 or monotonic_prediction_ms <= time_limit_ms:
                    effective_limit = rows
                else:
                    break
        shared_hbm_controller = getattr(
            self, "shared_hbm_controller", None
        )
        if shared_hbm_controller is not None:
            effective_limit = (
                shared_hbm_controller.limit_ann_microbatch_rows(
                    self, effective_limit
                )
            )
        return effective_limit

    def _ann_inflight_predicted_gpu_ms(self, inflight=None) -> float:
        if inflight is None:
            inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]

        queued_ms = 0.0
        for task in inflight:
            result = task.get("result")
            if result is not None and not self._ann_result_gpu_pending(result):
                continue
            predicted_ms = task.get("predicted_gpu_ms")
            if predicted_ms is None:
                predicted_ms = self._predict_ann_gpu_service_ms(
                    self._ann_task_num_queries(task)
                )
            queued_ms += max(
                0.001,
                float(predicted_ms),
            )
        return queued_ms

    def _ann_launch_credit_limit_ms(
        self,
        plan=None,
        *,
        allow_current_epoch: bool = False,
    ) -> float:
        configured_ms = max(
            0.0, getattr(self, "rag_ann_max_queued_gpu_ms", 0.0)
        )
        if configured_ms <= 0.0:
            return float("inf")
        if getattr(self, "rag_ann_partition_switch_deferred", False):
            return 0.0

        plan = getattr(self, "rag_frontend_plan", None) if plan is None else plan
        if plan is not None and self._rag_ann_uses_partitioned_gc():
            target_idx = getattr(plan, "target_stream_idx", None)
            if (
                not allow_current_epoch
                and target_idx is not None
                and target_idx != get_current_stream_idx()
            ):
                # Do not enqueue work into an epoch the planner is retiring.
                return 0.0
        return configured_ms

    def _ann_credit_allows_launch(
        self,
        inflight,
        batch_rows: int,
        *,
        pending=None,
        plan=None,
        allow_atomic_liveness: bool = True,
        record_hold: bool = True,
        allow_current_epoch: bool = False,
    ) -> tuple[bool, float]:
        effective_plan = (
            getattr(self, "rag_frontend_plan", None) if plan is None else plan
        )
        target_stream_idx = getattr(effective_plan, "target_stream_idx", None)
        predicted_ms = self._predict_ann_gpu_service_ms(
            batch_rows,
            pending=pending,
            target_stream_idx=target_stream_idx,
        )
        queued_ms = self._ann_inflight_predicted_gpu_ms(inflight)
        credit_ms = self._ann_launch_credit_limit_ms(
            effective_plan,
            allow_current_epoch=allow_current_epoch,
        )
        self.rag_ann_credit_last_limit_ms = credit_ms
        self.rag_ann_credit_last_queued_ms = queued_ms
        self.rag_ann_credit_last_batch_ms = predicted_ms
        self.rag_ann_credit_last_forced_atomic = False

        if np.isinf(credit_ms) or queued_ms + predicted_ms <= credit_ms:
            return True, predicted_ms

        configured_ms = max(
            0.0, getattr(self, "rag_ann_max_queued_gpu_ms", 0.0)
        )
        if (
            allow_atomic_liveness
            and configured_ms > 0.0
            and credit_ms >= configured_ms - 1e-6
            and queued_ms <= 0.0
        ):
            # A single query is indivisible. Let one atomic launch through if
            # its conservative estimate exceeds the full configured credit;
            # partition-drain credit (zero) is never bypassed.
            if record_hold:
                self.rag_ann_credit_forced_single_launches = (
                    getattr(self, "rag_ann_credit_forced_single_launches", 0)
                    + 1
                )
            self.rag_ann_credit_last_forced_atomic = True
            return True, predicted_ms

        if record_hold:
            self.rag_ann_credit_holds = (
                getattr(self, "rag_ann_credit_holds", 0) + 1
            )
        return False, predicted_ms

    def _select_ann_credit_batch_rows(
        self,
        pending,
        inflight,
        microbatch_limit: int,
        *,
        plan=None,
        allow_atomic_liveness: bool = True,
        record_hold: bool = True,
        allow_current_epoch: bool = False,
        record_downsize: bool = True,
    ) -> tuple[int, float]:
        """Choose the largest pending prefix that fits remaining GPU credit."""
        if microbatch_limit <= 0:
            return 0, 0.0
        prefix_rows = []
        rows = 0
        for task in pending:
            task_rows = self._ann_task_num_queries(task)
            if rows + task_rows > microbatch_limit:
                break
            rows += task_rows
            prefix_rows.append(rows)
            if rows >= microbatch_limit:
                break
        if not prefix_rows:
            return 0, 0.0

        largest_rows = prefix_rows[-1]
        for candidate_rows in reversed(prefix_rows):
            allowed, predicted_ms = self._ann_credit_allows_launch(
                inflight,
                candidate_rows,
                pending=pending,
                plan=plan,
                allow_atomic_liveness=False,
                record_hold=False,
                allow_current_epoch=allow_current_epoch,
            )
            if allowed:
                if candidate_rows < largest_rows and record_downsize:
                    self.rag_ann_credit_downsized_launches = (
                        getattr(self, "rag_ann_credit_downsized_launches", 0)
                        + 1
                    )
                return candidate_rows, predicted_ms

        # Preserve liveness only for the smallest indivisible pending prefix.
        candidate_rows = prefix_rows[0]
        allowed, predicted_ms = self._ann_credit_allows_launch(
            inflight,
            candidate_rows,
            pending=pending,
            plan=plan,
            allow_atomic_liveness=allow_atomic_liveness,
            record_hold=record_hold,
            allow_current_epoch=allow_current_epoch,
        )
        if allowed:
            if candidate_rows < largest_rows and record_downsize:
                self.rag_ann_credit_downsized_launches = (
                    getattr(self, "rag_ann_credit_downsized_launches", 0) + 1
                )
            return candidate_rows, predicted_ms
        return 0, predicted_ms

    def _select_ann_pre_switch_p99_credit_batch_rows(
        self,
        pending,
        batch_rows: int,
        predicted_batch_ms: float,
        queued_gpu_ms: float,
        credit_limit_ms: float,
        predicted_gpu_p99_ms: float,
    ) -> tuple[int, float, float]:
        """Downsize to the largest pending prefix whose exact p99 fits.

        The normal credit selector uses the configured service percentile
        (typically p95). Pre-switch admission is stricter. If the initially
        selected prefix exceeds its non-atomic p99 credit, consider only
        smaller, indivisible pending prefixes with mature exact-batch p99
        observations. Missing buckets are never extrapolated on this path.
        """

        initial_rows = max(1, int(batch_rows))
        initial_p99_ms = max(0.001, float(predicted_gpu_p99_ms))
        selected_rows = initial_rows
        selected_p95_ms = max(0.001, float(predicted_batch_ms))
        selected_p99_ms = initial_p99_ms
        attempted_rows = [initial_rows]
        downsize_enabled = (
            self._rag_ann_pre_switch_p99_downsize_enabled()
        )
        adaptive_enabled = (
            self._rag_ann_pre_switch_adaptive_p99_credit_enabled()
        )
        configured_credit_limit_ms = credit_limit_ms
        adaptive_attempted = False
        adaptive_reason = "disabled"
        adaptive_reference_rows = None
        adaptive_reference_p99_ms = None
        adaptive_lower_ms = None
        adaptive_upper_ms = None
        adaptive_derived_limit_ms = None

        prefix_rows = []
        rows = 0
        for task in pending:
            next_rows = rows + self._ann_task_num_queries(task)
            if next_rows > initial_rows:
                break
            rows = next_rows
            prefix_rows.append(rows)
            if rows >= initial_rows:
                break

        configured_initial_fits = (
            not np.isfinite(configured_credit_limit_ms)
            or queued_gpu_ms + initial_p99_ms
            <= configured_credit_limit_ms + 1e-6
        )
        if adaptive_enabled:
            if initial_rows <= 1:
                adaptive_reason = "atomic_batch"
            elif not np.isfinite(configured_credit_limit_ms):
                adaptive_reason = "unbounded_configured_credit"
            elif not configured_initial_fits:
                adaptive_reason = "configured_credit_already_strict"
            else:
                adaptive_attempted = True
                adaptive_reason = "no_safe_exact_prefix_interval"
                for candidate_rows in reversed(prefix_rows):
                    if candidate_rows >= initial_rows:
                        continue
                    candidate_p99_ms = (
                        self._predict_ann_gpu_critical_path_ms(
                            candidate_rows
                        )
                    )
                    if candidate_p99_ms is None:
                        continue
                    lower_ms = max(
                        selected_p95_ms,
                        float(candidate_p99_ms),
                    )
                    upper_ms = initial_p99_ms
                    if lower_ms + 1e-6 >= upper_ms:
                        continue
                    derived_limit_ms = queued_gpu_ms + (
                        lower_ms + upper_ms
                    ) / 2.0
                    credit_limit_ms = min(
                        configured_credit_limit_ms,
                        derived_limit_ms,
                    )
                    adaptive_reference_rows = candidate_rows
                    adaptive_reference_p99_ms = float(
                        candidate_p99_ms
                    )
                    adaptive_lower_ms = lower_ms
                    adaptive_upper_ms = upper_ms
                    adaptive_derived_limit_ms = credit_limit_ms
                    adaptive_reason = "derived"
                    self.rag_ann_pre_switch_adaptive_p99_credit_derivations = (
                        getattr(
                            self,
                            "rag_ann_pre_switch_adaptive_p99_credit_derivations",
                            0,
                        )
                        + 1
                    )
                    break
                if adaptive_reason != "derived":
                    self.rag_ann_pre_switch_adaptive_p99_credit_unavailable = (
                        getattr(
                            self,
                            "rag_ann_pre_switch_adaptive_p99_credit_unavailable",
                            0,
                        )
                        + 1
                    )

        def fits(candidate_p99_ms: float) -> bool:
            return (
                not np.isfinite(credit_limit_ms)
                or queued_gpu_ms + candidate_p99_ms
                <= credit_limit_ms + 1e-6
            )

        if not fits(initial_p99_ms):
            selected_rows = 0
            if downsize_enabled:
                for candidate_rows in reversed(prefix_rows):
                    if candidate_rows >= initial_rows:
                        continue
                    attempted_rows.append(candidate_rows)
                    candidate_p99_ms = (
                        self._predict_ann_gpu_critical_path_ms(
                            candidate_rows
                        )
                    )
                    if candidate_p99_ms is None or not fits(
                        candidate_p99_ms
                    ):
                        continue
                    selected_rows = candidate_rows
                    selected_p95_ms = self._predict_ann_gpu_service_ms(
                        candidate_rows
                    )
                    selected_p99_ms = candidate_p99_ms
                    break

        rows_reduced = (
            initial_rows - selected_rows if selected_rows > 0 else 0
        )
        downsized = rows_reduced > 0
        if downsized:
            self.rag_ann_pre_switch_gpu_p99_credit_downsizes = (
                getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsizes",
                    0,
                )
                + 1
            )
            self.rag_ann_pre_switch_gpu_p99_credit_rows_reduced = (
                getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_rows_reduced",
                    0,
                )
                + rows_reduced
            )

        self.rag_ann_pre_switch_last_gpu_p99_credit = {
            "downsize_enabled": downsize_enabled,
            "adaptive_enabled": adaptive_enabled,
            "adaptive_attempted": adaptive_attempted,
            "adaptive_reason": adaptive_reason,
            "adaptive_reference_batch_rows": adaptive_reference_rows,
            "adaptive_reference_batch_p99_ms": (
                adaptive_reference_p99_ms
            ),
            "adaptive_lower_ms": adaptive_lower_ms,
            "adaptive_upper_ms": adaptive_upper_ms,
            "adaptive_derived_limit_ms": adaptive_derived_limit_ms,
            "initial_batch_rows": initial_rows,
            "selected_batch_rows": (
                selected_rows if selected_rows > 0 else None
            ),
            "rows_reduced": rows_reduced,
            "attempted_batch_rows": attempted_rows,
            "queued_ms": queued_gpu_ms,
            "initial_batch_p99_ms": initial_p99_ms,
            # Compatibility field: the admitted p99, or the rejected
            # initial p99 when no exact-batch prefix fits.
            "batch_p99_ms": (
                selected_p99_ms if selected_rows > 0 else initial_p99_ms
            ),
            "selected_batch_p99_ms": (
                selected_p99_ms if selected_rows > 0 else None
            ),
            "limit_ms": (
                credit_limit_ms
                if np.isfinite(credit_limit_ms)
                else None
            ),
            "configured_limit_ms": (
                configured_credit_limit_ms
                if np.isfinite(configured_credit_limit_ms)
                else None
            ),
            "fits": selected_rows > 0,
            "downsized": downsized,
            # One overwritten record preserves request-level diagnosis without
            # accumulating request IDs or introducing high-cardinality maps.
            "request_id": None,
            "commit_stage": "selector",
            "final_skip_reason": None,
        }
        return selected_rows, selected_p95_ms, selected_p99_ms

    def _should_wait_for_ann_microbatch(
        self, pending, oldest_wait_ms: float, inflight=None
    ) -> bool:
        """Opportunistically coalesce work that already trails GPU ANN work."""
        max_batch_size = self._effective_ann_microbatch_rows()
        max_batch_wait_ms = max(
            0.0, getattr(self, "rag_ann_max_batch_wait_ms", 0.0)
        )
        if max_batch_size <= 1 or max_batch_wait_ms <= 0.0:
            return False

        pending_rows = self._ann_pending_rows_until_limit(pending, max_batch_size)
        if pending_rows >= max_batch_size:
            return False

        gpu_work_active = self._rag_ann_gpu_work_active(inflight)
        plan = getattr(self, "rag_frontend_plan", None)
        if (
            plan is not None
            and plan.low_pressure_fast_path
            and plan.allow_ann
            and not gpu_work_active
        ):
            return False

        min_pending = max(
            1, getattr(self, "rag_ann_batch_wait_min_pending", 2)
        )
        if pending_rows < min_pending and not gpu_work_active:
            return False

        should_wait = oldest_wait_ms < max_batch_wait_ms
        if should_wait and gpu_work_active:
            self.rag_ann_inflight_coalesce_holds = (
                getattr(self, "rag_ann_inflight_coalesce_holds", 0) + 1
            )
            oldest = pending[0] if pending else None
            if oldest is not None and not oldest.get(
                "inflight_coalesce_recorded", False
            ):
                oldest["inflight_coalesce_recorded"] = True
                self.rag_ann_inflight_coalesce_events = (
                    getattr(self, "rag_ann_inflight_coalesce_events", 0) + 1
                )
        return should_wait

    def _pop_ann_microbatch(self, pending, max_batch_size: Optional[int] = None):
        max_batch_size = max(
            0,
            self._effective_ann_microbatch_rows()
            if max_batch_size is None
            else int(max_batch_size),
        )
        if max_batch_size == 0:
            return None
        tasks = []
        query_arrays = []
        query_spans = []
        rows = 0

        while pending:
            task = pending[0]
            task_rows = self._ann_task_num_queries(task)
            # Requests are indivisible here. Never override the projected
            # row cap just because this is the first request in the batch.
            if rows + task_rows > max_batch_size:
                break

            task = pending.popleft()
            tasks.append(task)
            query_arrays.append(task["queries"])
            query_spans.append((rows, rows + task_rows))
            rows += task_rows
            if rows >= max_batch_size:
                break

        if not tasks:
            return None

        queries = (
            query_arrays[0]
            if len(query_arrays) == 1
            else np.ascontiguousarray(np.concatenate(query_arrays, axis=0))
        )
        return {
            "queries": queries,
            "tasks": tasks,
            "query_spans": query_spans,
            "enqueue_time": min(task["enqueue_time"] for task in tasks),
        }

    def _register_rag_ann_handoff(self, output: AnnSearchReqOutput):
        timeout_ms = getattr(self, "rag_ann_handoff_timeout_ms", 0.0)
        if (
            timeout_ms <= 0.0
            or not output.success
            or output.request_id is None
            or not self._rag_ann_uses_partitioned_gc()
        ):
            return

        now = time.perf_counter()
        request_started_at = getattr(
            self, "rag_request_start_times", {}
        ).pop(str(output.request_id), now)
        self.rag_ann_handoffs[str(output.request_id)] = (
            request_started_at,
            now,
            now + timeout_ms / 1000.0,
        )

    def _complete_rag_ann_handoff(self, request_id):
        if request_id is None:
            return False
        handoffs = getattr(self, "rag_ann_handoffs", None)
        if not handoffs:
            return False
        handoff = handoffs.pop(str(request_id), None)
        if handoff is None:
            return False
        # The global residency timer only bridges ANN completion to request
        # handoff. Once the request reaches the scheduler, concrete Prefill
        # demand must drive partition selection instead of the stale timer.
        if self._rag_ann_uses_frontend_pool():
            self.rag_ann_frontend_residency_until = 0.0
            self.rag_ann_frontend_residency_releases = (
                getattr(self, "rag_ann_frontend_residency_releases", 0) + 1
            )
        if len(handoff) == 3:
            request_started_at, completed_at, deadline = handoff
            completed_timing = (request_started_at, completed_at)
        else:
            completed_at, deadline = handoff
            completed_timing = (completed_at,)
        now = time.perf_counter()
        self.rag_ann_handoff_latency_ms.append(
            (now - completed_at) * 1000.0
        )
        if now >= deadline:
            # Completion can race with lazy pruning when the scheduler is busy.
            # Never treat an already-expired lease as a successful handoff.
            self.rag_ann_handoff_expired += 1
            return False
        self.rag_completed_handoff_timing[str(request_id)] = (
            *completed_timing,
            now,
        )
        self.rag_ann_handoff_completed += 1
        return True

    def _prune_rag_ann_handoffs(self) -> bool:
        handoffs = getattr(self, "rag_ann_handoffs", None)
        if not handoffs:
            return False
        now = time.perf_counter()
        expired = [
            request_id
            for request_id, handoff in handoffs.items()
            if now >= handoff[-1]
        ]
        for request_id in expired:
            handoffs.pop(request_id, None)
        self.rag_ann_handoff_expired += len(expired)
        return bool(handoffs)

    def _record_rag_input_preparation_completion(self, contract, arrived_at):
        """Observe external preparation turnaround on one scheduler clock.

        Includes return transport, passage/prompt work and tokenizer ingress;
        not a CPU kernel service estimate. Buckets describe current-run query
        row counts, not equivalent prompt contents or an exact-shape model.
        """
        if contract is None or contract.get('stage')!='input_preparation':
            return
        started=contract.get('ann_completed_at');rows=contract.get('query_rows')
        if (type(rows) is not int or rows<=0 or type(started) not in (int,float)
                or type(arrived_at) not in (int,float) or not math.isfinite(started)
                or not math.isfinite(arrived_at) or arrived_at<=started):
            return
        table=getattr(self,'rag_input_preparation_turnaround_ms',None)
        if table is None:
            table=self.rag_input_preparation_turnaround_ms={}
        table.setdefault(rows,deque(maxlen=4096)).append((arrived_at-started)*1000.)

    def _send_ann_search_output(self, output: AnnSearchReqOutput):
        preparations = getattr(self, "rag_preparation_contracts", {})
        rid = str(output.request_id)
        if not output.success:
            preparations.pop(rid, None)
        elif rid in preparations:
            # Do not recreate an entry removed by cancellation. This lifetime
            # is independent of the old short resource-handoff timeout.
            preparations[rid]["stage"] = "input_preparation"
            preparations[rid]["ann_completed_at"] = time.perf_counter()
        self._register_rag_ann_handoff(output)
        with _nvtx_range("ann_search_result_send"):
            self.send_to_tokenizer.send_pyobj(output)

    def _collect_ann_search_outputs(self, task):
        task["ann_collection_succeeded"] = False
        result = task["result"]
        ids_np = result.ids_to_cpu()
        gpu_latency_ms = task.get("gpu_elapsed_ms")
        if gpu_latency_ms is None:
            gpu_latency_ms = result.gpu_elapsed_ms()
        self.rag_ann_gpu_latency_ms.append(gpu_latency_ms)
        now = time.perf_counter()
        subtasks = task.get("tasks")
        if subtasks is None:
            subtasks = [task]
            query_spans = [(0, ids_np.shape[0])]
        else:
            query_spans = task["query_spans"]

        outputs = []
        completion_service_ms_samples = []
        for subtask, (start, end) in zip(subtasks, query_spans):
            topk = subtask["topk"] or result.topk
            topk = max(1, min(topk, result.topk))
            queue_wait_ms = float(subtask.get("queue_wait_ms", 0.0))
            latency_ms = (now - subtask["enqueue_time"]) * 1000.0
            completion_service_ms_samples.append(latency_ms - queue_wait_ms)
            shared_hbm_controller = getattr(
                self, "shared_hbm_controller", None
            )
            if shared_hbm_controller is not None:
                shared_hbm_controller.observe_ann_completion(
                    latency_ms, float(self.rag_ann_deadline_ms)
                )
            ann_metrics = result.metrics_for_span(start, end)
            self.rag_ann_cpu_queue_wait_ms.append(
                float(ann_metrics.get("cpu_queue_wait_ms", 0.0))
            )
            self.rag_ann_cpu_search_latency_ms.append(
                ann_metrics["cpu_search_latency_ms"]
            )
            self.rag_ann_coarse_quantize_latency_ms.append(
                ann_metrics["coarse_quantize_latency_ms"]
            )
            self.rag_ann_route_latency_ms.append(
                ann_metrics["route_latency_ms"]
            )
            self.rag_ann_merge_latency_ms.append(
                ann_metrics["merge_latency_ms"]
            )
            self.rag_ann_hot_probe_ratio.append(
                ann_metrics["hot_probe_ratio"]
            )
            outputs.append(
                AnnSearchReqOutput(
                    success=True,
                    ids=ids_np[start:end, :topk].tolist(),
                    distances=None,
                    latency_ms=latency_ms,
                    queue_wait_ms=queue_wait_ms,
                    gpu_latency_ms=gpu_latency_ms,
                    **ann_metrics,
                    batch_size=result.batch_size,
                    request_id=subtask.get("request_id"),
                )
            )
        task["ann_collection_succeeded"] = True
        self._record_ann_completion_service(
            task,
            completion_service_ms_samples,
            gpu_latency_ms,
        )
        return outputs

    def _collect_ann_search_outputs_safe(self, task):
        try:
            return self._collect_ann_search_outputs(task)
        except Exception as e:
            logger.exception("ANN side-channel result collection failed.")
            subtasks = task.get("tasks") or [task]
            return [
                AnnSearchReqOutput(
                    success=False,
                    message=str(e),
                    request_id=subtask.get("request_id"),
                )
                for subtask in subtasks
            ]

    def _record_rag_hybrid_collection_service(self, task, elapsed_ms):
        """Observe post-join collect/output construction, excluding branch wait.

        A readiness marker is published only after result.poll() succeeds.
        Keep inline and worker paths separate and leave old full-GPU tables
        unchanged. Queue wait and send/prepare time are not this measurement.
        """
        if (task.get('rag_hybrid_collection_recorded')
                or not task.get('rag_result_join_observed')
                or not task.get('ann_collection_succeeded')
                or task.get('launch_context') == 'pre_switch'):
            return
        result = task.get('result')
        full_cpu=observe_full_cpu_fallback(result,observed_at=time.perf_counter())
        is_full_cpu=full_cpu is not None and full_cpu.status=='succeeded'
        if not is_full_cpu and (result is None or not hasattr(result, 'gpu_result')):
            return
        mode = task.get('ann_collection_mode')
        if mode not in ('inline', 'worker') or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        rows = self._ann_task_num_queries(task)
        topk = getattr(result, 'topk', None)
        if type(rows) is not int or rows <= 0 or type(topk) is not int or topk <= 0:
            return
        key = (('full_cpu',mode,rows,topk) if is_full_cpu else
               (mode, rows, topk, len(getattr(result, 'cpu_searches', ()) or ()),
                result.gpu_result is not None))
        history = getattr(self, 'rag_hybrid_collection_ms_by_shape', None)
        if history is None:
            history = self.rag_hybrid_collection_ms_by_shape = {}
        history.setdefault(key, deque(maxlen=64)).append(float(elapsed_ms))
        task['rag_hybrid_collection_recorded'] = True

    def _collect_ann_search_outputs_timed(self, task):
        """Measure the whole ids/merge/output construction call."""
        collection_started = time.perf_counter()
        outputs = self._collect_ann_search_outputs_safe(task)
        collection_ms = (
            time.perf_counter() - collection_started
        ) * 1000.0
        task["ann_collection_cpu_ms"] = collection_ms
        self._record_rag_hybrid_collection_service(task, collection_ms)
        return outputs, collection_ms

    def _ensure_ann_result_worker(self):
        if getattr(self, "ann_result_worker", None) is not None:
            return

        self.ann_result_collect_queue = Queue()
        self.ann_result_output_queue = Queue()
        self.ann_result_collector_busy = threading.Event()

        def worker_loop():
            while True:
                task = self.ann_result_collect_queue.get()
                if task is None:
                    return
                self.ann_result_collector_busy.set()
                queued_at=task.get('rag_collection_progress',(None,None))[1]
                started_at=time.perf_counter()
                task['rag_collection_progress']=('active',queued_at,started_at,None)
                try:
                    outputs, collection_cpu_ms = (
                        self._collect_ann_search_outputs_timed(task)
                    )
                    self._record_ann_worker_collection_cost(
                        task, collection_cpu_ms
                    )
                    task['rag_collection_progress']=('completed',queued_at,started_at,time.perf_counter())
                    self.ann_result_output_queue.put((task, outputs))
                except Exception:
                    task['rag_collection_progress']=('failed',queued_at,started_at,time.perf_counter())
                    raise
                finally:
                    self.ann_result_collector_busy.clear()

        self.ann_result_worker = threading.Thread(
            target=worker_loop,
            name="ann-result-collector",
            daemon=True,
        )
        self.ann_result_worker.start()

    def _drain_ann_result_outputs(self):
        output_queue = getattr(self, "ann_result_output_queue", None)
        if output_queue is None:
            return

        while True:
            try:
                item = output_queue.get_nowait()
            except Empty:
                return
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], dict)
            ):
                task, outputs = item
            else:
                task, outputs = None, item
            send_samples_ms = []
            for output in outputs:
                send_started = time.perf_counter()
                self._send_ann_search_output(output)
                send_samples_ms.append(
                    (time.perf_counter() - send_started) * 1000.0
                )
            if task is not None:
                self._record_ann_worker_send_cost(task, send_samples_ms)
                # Scheduler-owned registry spans worker queue, merge and send.
                # Remove only after all outputs have reached the next stage.
                self.rag_ann_collection_pending=[pending for pending in
                    getattr(self,'rag_ann_collection_pending',()) if pending is not task]

    def _poll_ann_search_side_channel(
        self, *, collect_ready_inline: bool = False,
        allow_new_memory_plans: bool = True,
    ) -> int:
        """Collect completions and advance already committed transitions.

        Joint scheduling must pass allow_new_memory_plans=False: observing a
        ready ANN frontier must not pre-commit a different elastic HBM proposal
        before target-vs-keep admission. Legacy callers retain existing hooks.
        This flag does not cancel or freeze a previously submitted conversion.
        """
        inline_completions = 0
        if (getattr(self,'rag_joint_event_input_provider',None) is not None
                or getattr(getattr(self, 'rag_frontend_plan', None), 'joint_event_id', None) is not None):
            # All old polling call sites inherit observation-only semantics
            # while a joint intent is present, including pre-switch harvesting.
            allow_new_memory_plans = False
        self._progress_shared_hbm_kv_map()
        self._drain_ann_result_outputs()

        inflight = getattr(self, "ann_search_inflight", None)
        shared_hbm_controller = getattr(
            self, "shared_hbm_controller", None
        )
        if not inflight:
            if shared_hbm_controller is not None and allow_new_memory_plans:
                online_recovery = getattr(
                    shared_hbm_controller, "on_ann_online_recovery_tick", None
                )
                if online_recovery is not None and online_recovery(self):
                    return inline_completions
                handled = (
                    shared_hbm_controller.on_ann_quiescent_tick(self)
                )
                if not handled:
                    required = (
                        shared_hbm_controller.predict_kv_tokens(self)
                    )
                    shared_hbm_controller.ensure_kv_capacity(required)
            return inline_completions

        if isinstance(inflight, dict):
            inflight = [inflight]

        if not collect_ready_inline:
            self._ensure_ann_result_worker()
        remaining = []
        gpu_became_ready = False
        for task in inflight:
            result = task["result"]
            gpu_started = self._ann_result_gpu_started(result)
            if (
                not task.get("ann_gpu_service_recorded", False)
                and gpu_started
                and not self._ann_result_gpu_pending(result)
            ):
                self._record_ann_gpu_service(task, result.gpu_elapsed_ms())
                gpu_became_ready = True
            if not result.poll():
                remaining.append(task)
                continue
            task['rag_result_join_observed'] = True
            if not task.get("ann_gpu_service_recorded", False):
                # The candidates-ready event and elapsed-time publication can
                # race.  A completed full result is a second safe opportunity
                # to obtain the positive CUDA timing sample before the task
                # leaves the inflight set.
                self._record_ann_gpu_service(task, result.gpu_elapsed_ms())

            with _nvtx_range("ann_search_poll_done"):
                self._extend_rag_ann_frontend_residency()
                if collect_ready_inline:
                    task["ann_collection_mode"] = "inline"
                    outputs, collection_cpu_ms = (
                        self._collect_ann_search_outputs_timed(task)
                    )
                    send_samples_ms = []
                    for output in outputs:
                        send_started = time.perf_counter()
                        self._send_ann_search_output(output)
                        send_samples_ms.append(
                            (time.perf_counter() - send_started) * 1000.0
                        )
                    self._record_ann_inline_collection_tail(
                        task,
                        collection_cpu_ms,
                        send_samples_ms,
                    )
                    inline_completions += 1
                else:
                    task["ann_collection_mode"] = "worker"
                    self.ann_result_collector_busy.set()
                    if not hasattr(self,'rag_ann_collection_pending'):
                        self.rag_ann_collection_pending=[]
                    task['rag_collection_progress']=('queued',time.perf_counter(),None,None)
                    self.rag_ann_collection_pending.append(task)
                    self.ann_result_collect_queue.put(task)

        self.ann_search_inflight = remaining
        if (
            gpu_became_ready
            and allow_new_memory_plans
            and shared_hbm_controller is not None
            and not self._rag_ann_gpu_work_active()
        ):
            # GPU candidates are ready while CPU cold-search/merge/rerank can
            # continue. This is the predictive handoff window.
            shared_hbm_controller.on_ann_gpu_idle(self)
        return inline_completions

    def _progress_shared_hbm_kv_map(self) -> bool:
        """Advance ANN reclaim/VMM stages from the scheduler thread."""

        controller = getattr(self, "shared_hbm_controller", None)
        if (
            controller is None
            or (
                getattr(controller, "pending_ann_reclaim", None) is None
                and getattr(controller, "pending_kv_native_map", None) is None
                and getattr(controller, "pending_ivf_reload", None) is None
            )
        ):
            return False
        return controller.poll_kv_growth_completion()

    def _ann_side_channel_has_work(self) -> bool:
        controller = getattr(self, "shared_hbm_controller", None)
        if controller is not None and (
            getattr(controller, "pending_ann_reclaim", None) is not None
            or getattr(controller, "pending_kv_native_map", None) is not None
            or getattr(controller, "pending_ivf_reload", None) is not None
        ):
            return True
        if getattr(self, "ann_search_pending", None):
            return True
        if getattr(self, "ann_search_inflight", None):
            return True

        collect_queue = getattr(self, "ann_result_collect_queue", None)
        if collect_queue is not None and not collect_queue.empty():
            return True
        collector_busy = getattr(self, "ann_result_collector_busy", None)
        if collector_busy is not None and collector_busy.is_set():
            return True
        output_queue = getattr(self, "ann_result_output_queue", None)
        return output_queue is not None and not output_queue.empty()

    @staticmethod
    def _ann_result_gpu_started(result) -> bool:
        if result is None:
            return False
        gpu_started = getattr(result, "gpu_started", None)
        return True if gpu_started is None else bool(gpu_started())

    @staticmethod
    def _ann_result_gpu_pending(result) -> bool:
        if result is None:
            return True
        gpu_started = getattr(result, "gpu_started", None)
        if gpu_started is not None and not gpu_started():
            return False
        gpu_poll = getattr(result, "gpu_poll", result.poll)
        return not gpu_poll()

    def _rag_ann_gpu_work_active(self, inflight=None) -> bool:
        if inflight is None:
            inflight = getattr(self, "ann_search_inflight", None)
        if not inflight:
            return False
        if isinstance(inflight, dict):
            inflight = [inflight]
        return any(
            self._ann_result_gpu_pending(task.get("result"))
            for task in inflight
        )

    @staticmethod
    def _ann_result_gpu_event(result):
        gpu_result = getattr(result, "gpu_result", result)
        return getattr(gpu_result, "event", None)

    def _maybe_fence_overdue_ann_before_decode(self, decode_stream) -> bool:
        """Bound ANN dispatch debt without synchronizing the host.

        The elapsed-time check alone is too late for an ANN batch launched in
        this scheduler iteration: one long Decode iteration can consume the
        complete credit before the next check. Use the measured Decode p99 to
        fence that batch proactively when the projected debt would cross the
        configured credit. The event insertion is device-side; the caller can
        then harvest the dependency-critical result before enqueuing Decode.
        """

        if not self._rag_ann_uses_decode_pool():
            return False
        credit_ms = max(
            0.0, float(getattr(self, "rag_ann_max_queued_gpu_ms", 0.0))
        )
        if credit_ms <= 0.0:
            return False

        inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]
        # ANN credit bounds outstanding GPU work. Host-side scheduler gaps do
        # not consume that credit and can otherwise trigger premature stream
        # drains even when the next Decode GPU step fits in the remaining
        # window.
        predicted_decode_ms = self._rag_ann_decode_prediction_ms()
        now = time.perf_counter()
        candidates = []
        for task in inflight:
            if task.get("decode_credit_fence_inserted", False):
                continue
            result = task.get("result")
            launched_at = getattr(result, "gpu_launch_time", None)
            if launched_at is None:
                launched_at = task.get("launch_time")
            if (
                launched_at is None
                or not self._ann_result_gpu_pending(result)
            ):
                continue
            age_ms = (now - launched_at) * 1000.0
            remaining_credit_ms = max(0.0, credit_ms - age_ms)
            if age_ms >= credit_ms:
                reason = "elapsed"
            elif (
                predicted_decode_ms is not None
                and predicted_decode_ms >= remaining_credit_ms
            ):
                reason = "predicted"
            else:
                continue
            event = self._ann_result_gpu_event(result)
            if event is not None:
                candidates.append(
                    (
                        launched_at,
                        task,
                        event,
                        age_ms,
                        remaining_credit_ms,
                        reason,
                    )
                )

        if not candidates:
            return False
        (
            _,
            task,
            event,
            age_ms,
            remaining_credit_ms,
            reason,
        ) = min(candidates, key=lambda item: item[0])
        decode_stream.wait_event(event)
        task["decode_credit_fence_inserted"] = True
        task["decode_credit_fence_reason"] = reason
        task["decode_credit_fence_prediction_ms"] = predicted_decode_ms
        self.rag_ann_decode_credit_fences = (
            getattr(self, "rag_ann_decode_credit_fences", 0) + 1
        )
        reason_counts = getattr(
            self, "rag_ann_decode_credit_fence_reason_counts", None
        )
        if reason_counts is None:
            reason_counts = {}
            self.rag_ann_decode_credit_fence_reason_counts = reason_counts
        self._increment_histogram(reason_counts, reason)

        history = getattr(self, "rag_ann_decode_credit_fence_age_ms", None)
        if history is None:
            history = deque(maxlen=4096)
            self.rag_ann_decode_credit_fence_age_ms = history
        history.append(age_ms)

        prediction_history = getattr(
            self, "rag_ann_decode_credit_fence_prediction_ms", None
        )
        if prediction_history is None:
            prediction_history = deque(maxlen=4096)
            self.rag_ann_decode_credit_fence_prediction_ms = (
                prediction_history
            )
        if predicted_decode_ms is not None:
            prediction_history.append(predicted_decode_ms)

        remaining_history = getattr(
            self, "rag_ann_decode_credit_fence_remaining_ms", None
        )
        if remaining_history is None:
            remaining_history = deque(maxlen=4096)
            self.rag_ann_decode_credit_fence_remaining_ms = remaining_history
        remaining_history.append(remaining_credit_ms)
        return True

    def _drain_fenced_ann_before_decode(self, decode_stream) -> int:
        """Publish a credit-fenced ANN handoff before launching Decode."""

        if not self._maybe_fence_overdue_ann_before_decode(decode_stream):
            return 0

        started_at = time.perf_counter()
        decode_stream.synchronize()
        completions = self._poll_ann_search_side_channel(
            collect_ready_inline=True
        )
        wait_ms = (time.perf_counter() - started_at) * 1000.0
        self.rag_ann_decode_credit_sync_drains = (
            getattr(self, "rag_ann_decode_credit_sync_drains", 0) + 1
        )
        self.rag_ann_decode_credit_inline_completions = (
            getattr(
                self, "rag_ann_decode_credit_inline_completions", 0
            )
            + completions
        )
        history = getattr(
            self, "rag_ann_decode_credit_sync_wait_ms", None
        )
        if history is None:
            history = deque(maxlen=4096)
            self.rag_ann_decode_credit_sync_wait_ms = history
        history.append(wait_ms)
        return completions

    @staticmethod
    def _rag_decode_overlap_class(
        *, ann_overlap: bool, prefill_overlap: bool
    ) -> str:
        if ann_overlap and prefill_overlap:
            return "ann_prefill"
        if ann_overlap:
            return "ann"
        if prefill_overlap:
            return "prefill"
        return "solo"

    @staticmethod
    def _rag_intervals_overlap(a0, a1, b0, b1) -> bool:
        """Half-open intersection of two spans read on one common clock."""
        if None in (a0, a1, b0, b1):
            return False
        return a0 < b1 and b0 < a1

    @staticmethod
    def _rag_prefill_interval_features(batch):
        """Per-request prefill shape for one split_prefill_batch.

        extend_lens/prefix_lens are aligned with batch.reqs.  The split is over
        layers, not tokens, so both lists are constant for the whole life of a
        batch and these aggregates vary only ACROSS bundles.
        """
        if batch is None:
            return None
        extend = getattr(batch, "extend_lens", None)
        prefix = getattr(batch, "prefix_lens", None)
        if not extend or not prefix or len(extend) != len(prefix):
            return None
        tokens = [int(x) for x in extend]
        cached = [int(x) for x in prefix]
        if any(x <= 0 for x in tokens) or any(x < 0 for x in cached):
            return None
        return {
            "n_reqs": len(tokens),
            "sum_n": sum(tokens),
            "sum_n2": sum(x * x for x in tokens),
            "sum_nr": sum(a * b for a, b in zip(tokens, cached)),
            "sum_r": sum(cached),
            "max_n": max(tokens),
            "max_r": max(cached),
        }

    def _rag_record_prefill_gpu_interval(self, meta, end_event):
        """Publish one prefill interval on the shared device clock."""
        epoch = getattr(self, "rag_gpu_epoch", None)
        start_event = meta.get("start_event")
        if epoch is None or start_event is None or end_event is None:
            return None
        try:
            start_ms = float(epoch.elapsed_time(start_event))
            end_ms = float(epoch.elapsed_time(end_event))
        except Exception:
            return None
        if not (math.isfinite(start_ms) and math.isfinite(end_ms)):
            return None
        if end_ms < start_ms:
            return None
        intervals = getattr(self, "rag_prefill_gpu_intervals", None)
        if intervals is None:
            intervals = deque(maxlen=256)
            self.rag_prefill_gpu_intervals = intervals
        intervals.append((start_ms, end_ms))
        return (start_ms, end_ms)

    def _rag_coexec_prefill(self, start_ms, end_ms) -> bool:
        """True iff a recently completed prefill interval intersects the span."""
        intervals = getattr(self, "rag_prefill_gpu_intervals", None) or ()
        for other_start, other_end in list(intervals)[-64:]:
            if self._rag_intervals_overlap(start_ms, end_ms, other_start, other_end):
                return True
        return False

    def _record_rag_decode_submission_service(self, launch_context, completed_at):
        """Keep submission-to-completion service separate from inter-token gaps."""
        if launch_context is None:
            return
        started=launch_context.get('submitted_at')
        partition=launch_context.get('stream_idx')
        batch_size=launch_context.get('batch_size')
        overlap=launch_context.get('overlap_class')
        if (type(started) not in (int,float) or not math.isfinite(started)
                or not math.isfinite(completed_at) or completed_at<=started
                or type(partition) is not int or partition<0
                or type(batch_size) is not int or batch_size<=0
                or overlap not in ('solo','ann','prefill','ann_prefill')):
            return
        history=getattr(self,'rag_decode_submission_service_ms',None)
        if history is None:
            history={};self.rag_decode_submission_service_ms=history
        bucket=self._rag_decode_batch_bucket(batch_size)
        history.setdefault(partition,{}).setdefault(bucket,{}).setdefault(
            overlap,deque(maxlen=64)).append((completed_at-started)*1000.)
        pressure_samples = getattr(self, "rag_decode_pressure_samples", None)
        if pressure_samples is None:
            pressure_samples = deque(maxlen=4096)
            self.rag_decode_pressure_samples = pressure_samples
        cached = launch_context.get("cached_tokens")
        pressure_samples.append(dict(
            partition=partition, overlap_class=overlap, batch_size=batch_size,
            cached_total=(sum(cached) if cached else None),
            cached_max=(max(cached) if cached else None),
            cached_list=(tuple(cached) if cached else None),
            coexec_prefill=launch_context.get("coexec_prefill"),
            gpu_t0_ms=launch_context.get("decode_t0_ms"),
            gpu_t1_ms=launch_context.get("decode_t1_ms"),
            submitted_at=started, completed_at=completed_at,
            elapsed_ms=(completed_at-started)*1000.))

    def _record_rag_decode_completion(
        self,
        decode_done: bool,
        decode_gpu_ms: Optional[float] = None,
        launch_context: Optional[dict] = None,
    ):
        if not decode_done:
            self.rag_ann_last_decode_completion_ts = None
            self.rag_ann_last_decode_stream_idx = None
            return

        now = time.perf_counter()
        current_stream_idx = get_current_stream_idx()
        running_batch = getattr(self, "running_batch", None)
        decode_bs = (
            int(launch_context.get("batch_size", 0))
            if launch_context is not None
            else (
                int(running_batch.batch_size())
                if running_batch is not None and not running_batch.is_empty()
                else 0
            )
        )
        if decode_bs > 0:
            self.rag_ann_decode_batch_sizes.append(decode_bs)
            self._update_rag_partition_decode_slo_band(decode_bs, now)
        if decode_gpu_ms is not None and decode_gpu_ms > 0.0:
            decode_gpu_ms = float(decode_gpu_ms)
            self.rag_ann_decode_gpu_ms.append(decode_gpu_ms)
            gpu_history = self.rag_ann_decode_gpu_partition_ms.setdefault(
                current_stream_idx, deque(maxlen=64)
            )
            gpu_history.append(decode_gpu_ms)
            if decode_bs > 0:
                gpu_partition_buckets = (
                    self.rag_ann_decode_gpu_partition_batch_ms.setdefault(
                        current_stream_idx, {}
                    )
                )
                bucket = self._rag_decode_batch_bucket(decode_bs)
                gpu_partition_buckets.setdefault(
                    bucket, deque(maxlen=64)
                ).append(decode_gpu_ms)
                overlap_class = (
                    str(launch_context.get("overlap_class", "solo"))
                    if launch_context is not None
                    else "solo"
                )
                overlap_partition_buckets = getattr(
                    self,
                    "rag_ann_decode_gpu_partition_batch_overlap_ms",
                    None,
                )
                if overlap_partition_buckets is None:
                    overlap_partition_buckets = {}
                    self.rag_ann_decode_gpu_partition_batch_overlap_ms = (
                        overlap_partition_buckets
                    )
                overlap_partition_buckets.setdefault(
                    current_stream_idx, {}
                ).setdefault(bucket, {}).setdefault(
                    overlap_class, deque(maxlen=64)
                ).append(decode_gpu_ms)

            plan = getattr(self, "rag_frontend_plan", None)
            predicted_ms = (
                launch_context.get("predicted_ms")
                if launch_context is not None
                else getattr(plan, "predicted_decode_ms", None)
            )
            prediction_stream_idx = (
                launch_context.get("stream_idx")
                if launch_context is not None
                else getattr(plan, "candidate_stream_idx", None)
            )
            overlap_class = (
                str(launch_context.get("overlap_class", "solo"))
                if launch_context is not None
                else "legacy_plan"
            )
            if (
                predicted_ms is not None
                and prediction_stream_idx == current_stream_idx
            ):
                predicted_ms = float(predicted_ms)
                decode_slo_ms = float(
                    launch_context.get("slo_ms")
                    if launch_context is not None
                    and launch_context.get("slo_ms") is not None
                    else self._rag_ann_decode_slo_ms()
                )
                safety_margin_ms = float(
                    getattr(self, "rag_ann_safety_margin_ms", 0.0)
                )
                predicted_safe = (
                    decode_slo_ms <= 0.0
                    or predicted_ms + safety_margin_ms < decode_slo_ms
                )
                observed_safe = (
                    decode_slo_ms <= 0.0 or decode_gpu_ms < decode_slo_ms
                )
                if predicted_safe and not observed_safe:
                    self.rag_decode_prediction_false_safe += 1
                elif not predicted_safe and observed_safe:
                    self.rag_decode_prediction_false_unsafe += 1
                self.rag_decode_prediction_abs_error_ms.append(
                    abs(predicted_ms - decode_gpu_ms)
                )
                self.rag_decode_prediction_samples += 1
                if getattr(self, "_rag_dm2_window", None) is not None:
                    self._rag_dm2_window.append((float(decode_gpu_ms), overlap_class))
                by_stream = getattr(self, "_rag_dm2_by_stream", None)
                if by_stream is not None:
                    by_stream.setdefault(current_stream_idx, deque(maxlen=50)).append(
                        (float(decode_gpu_ms), overlap_class, int(decode_bs))
                    )
                self.rag_decode_prediction_pairs.append(
                    {
                        "predicted_ms": predicted_ms,
                        "observed_ms": decode_gpu_ms,
                        "stream_idx": current_stream_idx,
                        "prediction_stream_idx": prediction_stream_idx,
                        "batch_size": decode_bs,
                        "slo_ms": decode_slo_ms,
                        "predicted_safe": predicted_safe,
                        "observed_safe": observed_safe,
                        "overlap_class": overlap_class,
                        "coexec_prefill": (
                            launch_context.get("coexec_prefill")
                            if launch_context is not None
                            else None
                        ),
                        "decode_t0_ms": (
                            launch_context.get("decode_t0_ms")
                            if launch_context is not None
                            else None
                        ),
                        "decode_t1_ms": (
                            launch_context.get("decode_t1_ms")
                            if launch_context is not None
                            else None
                        ),
                        "prediction_source": (
                            "decode_launch"
                            if launch_context is not None
                            else "legacy_plan"
                        ),
                        "prediction_components": (
                            launch_context.get("prediction_components")
                            if launch_context is not None
                            else None
                        ),
                    }
                )
            elif predicted_ms is not None:
                self.rag_decode_prediction_partition_mismatches += 1
        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        if (
            current_stream_idx == decode_only_idx
            and getattr(self, "rag_decode_recovery_pending", False)
        ):
            # A recovery window is complete only after a real decode step has
            # run with the full decode partition.
            self.rag_decode_recovery_pending = False
            self.rag_dependency_prefill_burst_chunks = 0
            self.rag_decode_recovery_windows = (
                getattr(self, "rag_decode_recovery_windows", 0) + 1
            )
            self.rag_ann_consecutive_batches = 0
            self.rag_prefill_consecutive_chunks = 0
            # An endpoint sprint is a one-shot acceleration for a dependency
            # handoff. A Decode recovery closes that lease but must not rearm
            # it: unfinished layers continue through bounded mixed-partition
            # bundles. Rearming here makes every small bundle alternate between
            # the endpoint and Decode-only groups, paying a full transition on
            # each round trip.
        if (
            current_stream_idx == decode_only_idx
            and getattr(self, "rag_handoff_decode_release_pending", False)
        ):
            self.rag_handoff_decode_release_pending = False
            self.rag_handoff_decode_release_windows += 1
        previous = getattr(self, "rag_ann_last_decode_completion_ts", None)
        if previous is not None:
            interval_ms = (now - previous) * 1000.0
            self.rag_ann_decode_iteration_ms.append(interval_ms)
            previous_stream_idx = getattr(
                self, "rag_ann_last_decode_stream_idx", None
            )
            if previous_stream_idx == current_stream_idx:
                histories = self.rag_ann_decode_partition_ms
                history = histories.setdefault(
                    current_stream_idx, deque(maxlen=64)
                )
                history.append(interval_ms)
                if decode_bs > 0:
                    partition_buckets = (
                        self.rag_ann_decode_partition_batch_ms.setdefault(
                            current_stream_idx, {}
                        )
                    )
                    bucket = self._rag_decode_batch_bucket(decode_bs)
                    partition_buckets.setdefault(
                        bucket, deque(maxlen=64)
                    ).append(interval_ms)
                    overlap_class = (
                        str(
                            launch_context.get("overlap_class", "solo")
                        )
                        if launch_context is not None
                        else "solo"
                    )
                    overlap_partition_buckets = getattr(
                        self,
                        "rag_ann_decode_partition_batch_overlap_ms",
                        None,
                    )
                    if overlap_partition_buckets is None:
                        overlap_partition_buckets = {}
                        self.rag_ann_decode_partition_batch_overlap_ms = (
                            overlap_partition_buckets
                        )
                    overlap_partition_buckets.setdefault(
                        current_stream_idx, {}
                    ).setdefault(bucket, {}).setdefault(
                        overlap_class, deque(maxlen=64)
                    ).append(interval_ms)
                    if (decode_gpu_ms is not None and decode_gpu_ms > 0.0
                            and getattr(self,'_rag_iteration_overhead_window',None)
                            is not None):
                        self._rag_iteration_overhead_window.append(
                            (float(interval_ms),float(decode_gpu_ms),overlap_class))

                    predicted_iteration_ms = (
                        launch_context.get("predicted_iteration_ms")
                        if launch_context is not None
                        else None
                    )
                    prediction_stream_idx = (
                        launch_context.get("stream_idx")
                        if launch_context is not None
                        else None
                    )
                    if (
                        predicted_iteration_ms is not None
                        and prediction_stream_idx == current_stream_idx
                    ):
                        predicted_iteration_ms = float(
                            predicted_iteration_ms
                        )
                        decode_slo_ms = float(
                            launch_context.get("slo_ms")
                            if launch_context.get("slo_ms") is not None
                            else self._rag_ann_decode_slo_ms()
                        )
                        safety_margin_ms = float(
                            getattr(
                                self, "rag_ann_safety_margin_ms", 0.0
                            )
                        )
                        predicted_safe = (
                            decode_slo_ms <= 0.0
                            or predicted_iteration_ms + safety_margin_ms
                            < decode_slo_ms
                        )
                        observed_safe = (
                            decode_slo_ms <= 0.0
                            or interval_ms < decode_slo_ms
                        )
                        if predicted_safe and not observed_safe:
                            self.rag_decode_iteration_prediction_false_safe += 1
                        elif not predicted_safe and observed_safe:
                            self.rag_decode_iteration_prediction_false_unsafe += 1
                        self.rag_decode_iteration_prediction_abs_error_ms.append(
                            abs(predicted_iteration_ms - interval_ms)
                        )
                        self.rag_decode_iteration_prediction_samples += 1
                        self.rag_decode_iteration_prediction_pairs.append(
                            {
                                "predicted_ms": predicted_iteration_ms,
                                "observed_ms": interval_ms,
                                "stream_idx": current_stream_idx,
                                "batch_size": decode_bs,
                                "slo_ms": decode_slo_ms,
                                "predicted_safe": predicted_safe,
                                "observed_safe": observed_safe,
                                "overlap_class": overlap_class,
                                "prediction_source": "decode_launch",
                            }
                        )
                self._maybe_raise_rag_decode_partition_floor(
                    current_stream_idx
                )
        self.rag_ann_last_decode_completion_ts = now
        self.rag_ann_last_decode_stream_idx = current_stream_idx

    def _record_rag_decode_queue_delay(self, ready_ts: Optional[float]):
        if ready_ts is None:
            return
        self.rag_ann_decode_queue_delay_ms.append(
            (time.perf_counter() - ready_ts) * 1000.0
        )

    def _update_rag_partition_decode_slo_band(
        self, decode_bs: int, now: Optional[float] = None
    ):
        thresholds = getattr(
            self, "rag_partition_decode_slo_batch_thresholds", ()
        )
        partition_slos = getattr(self, "rag_partition_decode_slo_ms", ())
        if not thresholds or len(partition_slos) != len(thresholds) + 1:
            return

        alpha = min(
            1.0,
            max(
                0.001,
                getattr(self, "rag_partition_decode_slo_ewma_alpha", 0.02),
            ),
        )
        previous_ewma = getattr(
            self, "rag_partition_decode_slo_batch_ewma", None
        )
        batch_ewma = (
            float(decode_bs)
            if previous_ewma is None
            else alpha * decode_bs + (1.0 - alpha) * previous_ewma
        )
        self.rag_partition_decode_slo_batch_ewma = batch_ewma

        now = time.perf_counter() if now is None else now
        if not getattr(
            self, "rag_partition_decode_slo_calibration_started", False
        ):
            self.rag_partition_decode_slo_calibration_started = True
            self.rag_partition_decode_slo_next_update_ts = (
                now
                + getattr(
                    self,
                    "rag_partition_decode_slo_initial_calibration_ms",
                    0.0,
                )
                / 1000.0
            )
        next_update_ts = getattr(
            self, "rag_partition_decode_slo_next_update_ts", 0.0
        )
        if now < next_update_ts:
            return

        update_interval_ms = max(
            0.0,
            getattr(self, "rag_partition_decode_slo_min_residency_ms", 0.0),
        )
        self.rag_partition_decode_slo_next_update_ts = (
            now + update_interval_ms / 1000.0
        )
        current_band = max(
            0,
            min(
                len(partition_slos) - 1,
                getattr(self, "rag_partition_decode_slo_band_idx", 0),
            ),
        )
        target_band = sum(
            batch_ewma >= threshold for threshold in thresholds
        )
        hysteresis = max(
            0.0,
            getattr(self, "rag_partition_decode_slo_hysteresis_bs", 0.0),
        )
        control_updates = getattr(
            self, "rag_partition_decode_slo_control_updates", 0
        )
        if control_updates > 0:
            if (
                target_band > current_band
                and batch_ewma < thresholds[current_band] + hysteresis
            ):
                target_band = current_band
            elif (
                target_band < current_band
                and batch_ewma > thresholds[current_band - 1] - hysteresis
            ):
                target_band = current_band
        self.rag_partition_decode_slo_control_updates = control_updates + 1

        if target_band != current_band:
            self.rag_partition_decode_slo_band_idx = target_band
            self.rag_partition_decode_slo_last_change_ts = now
            self.rag_partition_decode_slo_band_changes = (
                getattr(self, "rag_partition_decode_slo_band_changes", 0) + 1
            )

    def _rag_partition_decode_slo_for_stream(
        self, stream_idx: int
    ) -> Optional[float]:
        partition_slos = getattr(self, "rag_partition_decode_slo_ms", ())
        if not partition_slos:
            return None

        load_thresholds = getattr(
            self, "rag_partition_decode_slo_batch_thresholds", ()
        )
        if load_thresholds:
            active_band = max(
                0,
                min(
                    len(partition_slos) - 1,
                    getattr(self, "rag_partition_decode_slo_band_idx", 0),
                ),
            )
            return float(partition_slos[active_band])

        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        if not 1 <= stream_idx <= last_mixed_idx:
            running_batch = getattr(self, "running_batch", None)
            decode_bs = (
                running_batch.batch_size()
                if running_batch is not None and not running_batch.is_empty()
                else 0
            )
            stream_idx = self._adaptive_frontend_stream_idx(decode_bs)
            stream_idx = max(1, min(last_mixed_idx, stream_idx))
        return float(partition_slos[stream_idx - 1])

    def _rag_partition_decode_slo_for_current_state(self) -> Optional[float]:
        partition_slos = getattr(self, "rag_partition_decode_slo_ms", ())
        if not partition_slos:
            return None

        load_thresholds = getattr(
            self, "rag_partition_decode_slo_batch_thresholds", ()
        )
        if load_thresholds:
            active_band = max(
                0,
                min(
                    len(partition_slos) - 1,
                    getattr(self, "rag_partition_decode_slo_band_idx", 0),
                ),
            )
            return float(partition_slos[active_band])

        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        current_stream_idx = get_current_stream_idx()
        if 1 <= current_stream_idx <= last_mixed_idx:
            target_stream_idx = current_stream_idx
        else:
            running_batch = getattr(self, "running_batch", None)
            decode_bs = (
                running_batch.batch_size()
                if running_batch is not None and not running_batch.is_empty()
                else 0
            )
            target_stream_idx = self._adaptive_frontend_stream_idx(decode_bs)
            target_stream_idx = max(1, min(last_mixed_idx, target_stream_idx))

        return float(partition_slos[target_stream_idx - 1])

    def _rag_ann_decode_slo_ms(
        self, target_stream_idx: Optional[int] = None
    ) -> float:
        partition_slo_ms = (
            self._rag_partition_decode_slo_for_current_state()
            if target_stream_idx is None
            else self._rag_partition_decode_slo_for_stream(target_stream_idx)
        )
        if partition_slo_ms is not None:
            return partition_slo_ms
        decode_step_slo_ms = getattr(self, "rag_ann_decode_step_slo_ms", -1.0)
        if decode_step_slo_ms > 0.0:
            return decode_step_slo_ms
        return getattr(self, "rag_ann_itl_slo_ms", -1.0)

    def _rag_decode_batch_bucket(self, decode_bs: int) -> int:
        width = max(1, getattr(self, "rag_ann_decode_batch_bucket_size", 4))
        decode_bs = max(1, int(decode_bs))
        return ((decode_bs - 1) // width) * width + 1

    def _rag_current_decode_batch_size(self) -> int:
        running_batch = getattr(self, "running_batch", None)
        if running_batch is None or running_batch.is_empty():
            return 0
        return int(running_batch.batch_size())

    def _rag_stage_candidate_stream_idx(
        self, *, has_ann: bool, has_prefill: bool
    ) -> int:
        if not self._decode_batch_active():
            return 0

        fixed_idx = int(getattr(self, "rag_e2_fixed_partition_idx", -1))
        if fixed_idx >= 0:
            last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
            if not 1 <= fixed_idx <= last_mixed_idx:
                raise ValueError(
                    "E2 fixed partition must name a configured mixed group"
                )
            return fixed_idx

        decode_bs = self._rag_current_decode_batch_size()
        candidate_idx = self._adaptive_frontend_stream_idx(decode_bs)
        lease_frontend_sm = -1
        if has_prefill and getattr(
            self, "rag_prefill_handoff_active", False
        ):
            lease_frontend_sm = getattr(
                self, "rag_prefill_lease_frontend_sm", -1
            )
        elif has_ann:
            if self._rag_ann_uses_decode_pool():
                return getattr(self, "real_sm_group_num", 1) - 1
            lease_frontend_sm = getattr(
                self, "rag_ann_lease_frontend_sm", -1
            )
        elif has_prefill:
            lease_frontend_sm = getattr(
                self, "rag_prefill_lease_frontend_sm", -1
            )
        lease_idx = self._stream_idx_for_frontend_sm(lease_frontend_sm)
        return lease_idx if lease_idx is not None else candidate_idx


    def _rag_decode_model_v2_ms(self, decode_bs, overlap_class):
        """Fitted decode service-time prediction (quantile tau=0.99, in ms).

        Returns None when the model cannot be applied -- no window yet, or a
        non-finite feature -- and the caller falls back to the incumbent
        predictor.  Cold start therefore behaves exactly as before; the fitted
        model only takes over once there is history to feed it.
        """
        window = getattr(self, "_rag_dm2_window", None)
        if window is None:
            return None
        try:
            rows = list(window)
            vals_all = [float(v) for v, _ in rows if np.isfinite(v)]
            flags_all = [
                1.0 if str(c) in RAG_DM2_CONTENDED else 0.0 for _, c in rows
            ]
            # A SHORT WINDOW IS NOT A REASON TO DECLINE.  The fitter defines a
            # missing history as 0.0 and applies the model to the very first row
            # of every arm.  Returning None here instead made the runtime fall
            # back to the incumbent for the first ~50 decode steps of each run,
            # and that boundary disagreement alone accounted for the live-vs-
            # offline gap (offline false_safe 13, live 50).  Match the fitter:
            # pad with 0.0, and decline only on a non-finite feature.
            feat = {"intercept": 1.0, "batch_size": float(max(1, int(decode_bs)))}
            for w in (10, 50):
                vals = np.asarray(vals_all[-w:], dtype=float)
                flags = np.asarray(flags_all[-w:], dtype=float)
                if vals.size == 0:
                    feat["hist%d_p50" % w] = 0.0
                    feat["hist%d_p90" % w] = 0.0
                    feat["hist%d_max" % w] = 0.0
                    feat["cont%d_frac" % w] = 0.0
                else:
                    feat["hist%d_p50" % w] = float(np.percentile(vals, 50))
                    feat["hist%d_p90" % w] = float(np.percentile(vals, 90))
                    feat["hist%d_max" % w] = float(vals.max())
                    feat["cont%d_frac" % w] = float(flags.mean())
            cls = str(overlap_class if overlap_class is not None else "solo")
            for level in RAG_DM2_LEVELS[1:]:
                feat["overlap_class=%s" % level] = 1.0 if cls == level else 0.0
            total = 0.0
            for name, coef in RAG_DM2_COEFS.items():
                value = feat.get(name)
                if value is None or not np.isfinite(value):
                    return None
                total += coef * value
            if not np.isfinite(total):
                return None
            return max(0.001, float(total))
        except Exception:
            return None
    def _rag_ann_decode_prediction_ms(
        self,
        target_stream_idx: Optional[int] = None,
        decode_bs: Optional[int] = None,
        *,
        include_scheduling: bool = False,
        percentile: Optional[float] = None,
        overlap_class: Optional[str] = None,
        apply_causal_guard: bool = True,
    ) -> Optional[float]:
        diagnostic = (
            os.environ.get("RAGLEASE_DECODE_PRED_DIAG") == "1"
            and not include_scheduling
        )
        current_stream_idx = get_current_stream_idx()
        target_stream_idx = (
            current_stream_idx
            if target_stream_idx is None
            else target_stream_idx
        )
        decode_bs = (
            self._rag_current_decode_batch_size()
            if decode_bs is None
            else int(decode_bs)
        )
        bundle=getattr(self,'rag_legacy_calibrated_bundle',None)
        running=getattr(self,'running_batch',None)
        if (bundle is not None and running is not None and not running.is_empty()
                and 0<=target_stream_idx<len(self.sm_counts)
                and self.sm_counts[target_stream_idx][1]>0):
            cached=tuple(len(req.origin_input_ids)+len(req.output_ids)-1
                         for req in (getattr(running,'reqs',None) or ()))
            if cached and len(cached)==max(1,decode_bs) and all(value>0 for value in cached):
                failure_reason=None
                solo_floor_ms=None
                pressure_ms=None
                causal_floor_ms=None
                predicted=bundle.stage.predict_decode_ms(self.sm_counts[target_stream_idx][1],cached,
                    expected_identity=bundle.identity)
                if predicted is None:
                    predicted=bundle.stage.predict_decode_nearest_ms(
                        self.sm_counts[target_stream_idx][1],cached,expected_identity=bundle.identity)
                stage_ms=predicted
                if predicted is None:
                    failure_reason='stage_cell_unavailable'
                overlap=overlap_class or 'solo'
                history=tuple(getattr(self,'_rag_dm2_window',()) or ())[-50:]
                fullcard_envelope=getattr(bundle,'fullcard_envelope',None)
                fullcard_ms=None
                if (fullcard_envelope is not None
                        # Nominal bundle sizing must retain the pre-existing
                        # predictor. This envelope is an admission-tail bound.
                        and apply_causal_guard
                        and self.sm_counts[target_stream_idx][1]==132
                        and overlap in ('solo','ann')):
                    stream_history=tuple(
                        getattr(self,'_rag_dm2_by_stream',{}).get(target_stream_idx,())
                    )[-50:]
                    fullcard_ms=fullcard_envelope.predict_ms(
                        132,max(1,decode_bs),overlap,stream_history,
                        expected_identity=bundle.identity)
                    if fullcard_ms is not None:
                        predicted=fullcard_ms
                        failure_reason=None
                solo_envelope=getattr(bundle,'solo_envelope',None)
                if (fullcard_ms is None and predicted is not None and overlap=='solo'
                        and solo_envelope is not None):
                    floor=solo_envelope.predict_ms(
                        self.sm_counts[target_stream_idx][1],max(1,decode_bs),history,
                        expected_identity=bundle.identity)
                    if floor is None:
                        failure_reason='solo_envelope_unavailable'
                        predicted=None
                    else:
                        solo_floor_ms=floor
                        predicted=max(predicted,floor)
                if (fullcard_ms is None and predicted is not None
                        and overlap in ('ann','ann_prefill')):
                    ann_ms=self._ann_inflight_predicted_gpu_ms()
                    pending=getattr(self,'ann_search_pending',None) or ()
                    if pending:
                        rows=min(self._effective_ann_microbatch_rows(),
                            sum(self._ann_task_num_queries(task) for task in pending))
                        if rows>0:
                            ann_ms+=self._predict_ann_gpu_service_ms(rows,pending=pending)
                    pressure=bundle.pressure.query_ms(target_stream_idx,'ann',cached,ann_ms,
                        expected_identity=bundle.identity)
                    pressure_ms=pressure
                    predicted=pressure
                    if predicted is None:
                        failure_reason='ann_pressure_cell_unavailable'
                    else:
                        failure_reason=None
                if (fullcard_ms is None and predicted is not None and apply_causal_guard
                        and overlap in ('ann','prefill','ann_prefill')):
                    floor=bundle.decode_guard.predict_ms(max(1,decode_bs),overlap,history,
                        expected_identity=bundle.identity)
                    if floor is None:
                        failure_reason='causal_guard_unavailable'
                        predicted=None
                    else:
                        causal_floor_ms=floor
                        predicted=max(predicted,floor)
                if predicted is not None and include_scheduling:
                    predicted=self.rag_legacy_iteration_guard.predict_total_ms(
                        predicted,tuple(getattr(self,'_rag_iteration_overhead_window',()) or ())[-50:])
                if predicted is not None:
                    if diagnostic:
                        self._rag_last_decode_prediction_components = dict(
                            source=("fullcard_envelope" if fullcard_ms is not None
                                    else "calibrated"), stage_ms=stage_ms,
                            solo_floor_ms=solo_floor_ms,
                            pressure_ms=pressure_ms,
                            causal_floor_ms=causal_floor_ms,
                            fullcard_ms=fullcard_ms,
                            final_ms=predicted,
                        )
                    counts=getattr(self,'rag_legacy_calibrated_prediction_counts',{})
                    counts['decode']=counts.get('decode',0)+1
                    self.rag_legacy_calibrated_prediction_counts=counts
                    self.rag_legacy_calibrated_last_decode=dict(partition=target_stream_idx,
                        cached_tokens=cached,overlap=overlap,include_scheduling=include_scheduling,
                        causal_guard_applied=apply_causal_guard,predicted_ms=predicted)
                    return max(0.001,float(predicted))
                self._record_rag_legacy_calibrated_fallback_reason(
                    'decode',failure_reason or 'prediction_unavailable')
            else:
                self._record_rag_legacy_calibrated_fallback_reason(
                    'decode','shape_unavailable')
            counts=getattr(self,'rag_legacy_calibrated_fallback_counts',{})
            counts['decode']=counts.get('decode',0)+1
            self.rag_legacy_calibrated_fallback_counts=counts
        if os.environ.get("RAGLEASE_DECODE_MODEL") == "v2":
            _v2 = self._rag_decode_model_v2_ms(decode_bs, overlap_class)
            if _v2 is not None:
                if diagnostic:
                    self._rag_last_decode_prediction_components = dict(
                        source="v2_fallback", final_ms=_v2)
                return _v2
        bucket = self._rag_decode_batch_bucket(max(1, decode_bs))
        history_percentile = (
            99.0
            if percentile is None
            else min(100.0, max(0.0, float(percentile)))
        )

        def recent_regime(history):
            if history is None:
                return None
            values = [
                float(value)
                for value in history
                if np.isfinite(value) and float(value) > 0.0
            ]
            if len(values) <= 1:
                return values

            shift_ratio = max(
                1.01,
                float(
                    getattr(
                        self,
                        "rag_decode_prediction_regime_shift_ratio",
                        4.0,
                    )
                ),
            )
            suffix = [values[-1]]
            ordered_suffix = suffix.copy()
            for value in reversed(values[:-1]):
                # The suffix grows by one observation at a time. Maintain its
                # order instead of allocating/partitioning a NumPy array for
                # every prefix and every candidate prediction. Preserve the
                # exact odd/even median and the original regime boundary.
                middle = len(ordered_suffix) // 2
                median = (
                    ordered_suffix[middle]
                    if len(ordered_suffix) % 2
                    else (ordered_suffix[middle - 1] + ordered_suffix[middle]) / 2.0
                )
                reference = max(0.001, median)
                ratio = max(value / reference, reference / value)
                if ratio >= shift_ratio:
                    break
                suffix.append(value)
                insort(ordered_suffix, value)
            suffix.reverse()
            return suffix

        def select_history(
            partition_buckets,
            partition_histories,
            global_history,
            partition_overlap_buckets=None,
        ):
            history = None
            source_stream_idx = target_stream_idx
            if overlap_class is not None and partition_overlap_buckets:
                target_overlap_history = (
                    partition_overlap_buckets.get(target_stream_idx, {})
                    .get(bucket, {})
                    .get(overlap_class)
                )
                if target_overlap_history:
                    return (
                        recent_regime(target_overlap_history),
                        target_stream_idx,
                    )
                if target_stream_idx != current_stream_idx:
                    current_overlap_history = (
                        partition_overlap_buckets.get(current_stream_idx, {})
                        .get(bucket, {})
                        .get(overlap_class)
                    )
                    if current_overlap_history:
                        return (
                            recent_regime(current_overlap_history),
                            current_stream_idx,
                        )
            if history is None or len(history) < 4:
                history = partition_buckets.get(target_stream_idx, {}).get(
                    bucket
                )
                source_stream_idx = target_stream_idx
            if (
                (history is None or len(history) < 4)
                and target_stream_idx != current_stream_idx
            ):
                current_bucket_history = partition_buckets.get(
                    current_stream_idx, {}
                ).get(bucket)
                if (
                    current_bucket_history is not None
                    and len(current_bucket_history) >= 4
                ):
                    history = current_bucket_history
                    source_stream_idx = current_stream_idx
            if history is None or len(history) < 4:
                history = partition_histories.get(target_stream_idx)
                source_stream_idx = target_stream_idx
            if (
                (history is None or len(history) < 4)
                and target_stream_idx != current_stream_idx
            ):
                history = partition_histories.get(current_stream_idx)
                source_stream_idx = current_stream_idx
            if history is None or len(history) < 4:
                history = global_history
                source_stream_idx = current_stream_idx
            return recent_regime(history), source_stream_idx

        def predict_p99(
            partition_buckets,
            partition_histories,
            global_history,
            profile_key,
            profile_distribution_key,
            partition_overlap_buckets=None,
        ) -> Optional[float]:
            history, source_stream_idx = select_history(
                partition_buckets,
                partition_histories,
                global_history,
                partition_overlap_buckets,
            )
            if history is not None and 0 < len(history) < 4:
                predicted_ms = max(float(value) for value in history) * max(
                    1.0,
                    float(
                        getattr(
                            self,
                            "rag_decode_prediction_cold_start_margin",
                            1.25,
                        )
                    ),
                )
                if (
                    source_stream_idx is not None
                    and source_stream_idx != target_stream_idx
                    and source_stream_idx < len(self.sm_counts)
                    and target_stream_idx < len(self.sm_counts)
                ):
                    source_decode_sm = self.sm_counts[source_stream_idx][1]
                    target_decode_sm = self.sm_counts[target_stream_idx][1]
                    if source_decode_sm > 0 and target_decode_sm > 0:
                        predicted_ms *= float(source_decode_sm) / float(
                            target_decode_sm
                        )
                return max(0.001, predicted_ms)
            if history is None or len(history) < 4:
                priors = getattr(self, "rag_lease_profile_priors", {})
                profile_tree = priors.get(profile_key, {})
                profile_distribution_tree = priors.get(
                    profile_distribution_key, {}
                )

                def profile_value(stream_idx: int) -> Optional[float]:
                    if percentile is not None:
                        distributions = profile_distribution_tree.get(
                            stream_idx, {}
                        )
                        if bucket in distributions:
                            value = _profile_distribution_percentile_ms(
                                distributions[bucket], history_percentile
                            )
                            if value is not None:
                                return value
                        values = [
                            value
                            for value in (
                                _profile_distribution_percentile_ms(
                                    distribution, history_percentile
                                )
                                for distribution in distributions.values()
                            )
                            if value is not None
                        ]
                        if values:
                            return max(values)
                    profiles = profile_tree.get(stream_idx, {})
                    if bucket in profiles:
                        return float(profiles[bucket])
                    if profiles:
                        return max(float(value) for value in profiles.values())
                    return None

                target_prior_ms = profile_value(target_stream_idx)
                if target_prior_ms is not None:
                    return max(0.001, target_prior_ms)
                prior_ms = profile_value(current_stream_idx)
                if prior_ms is not None:
                    source_decode_sm = self.sm_counts[current_stream_idx][1]
                    target_decode_sm = self.sm_counts[target_stream_idx][1]
                    if source_decode_sm > 0 and target_decode_sm > 0:
                        prior_ms *= float(source_decode_sm) / float(
                            target_decode_sm
                        )
                    return max(0.001, prior_ms)
                return None

            predicted_p99_ms = float(
                _history_percentile_value(list(history), history_percentile)
            )
            if (
                source_stream_idx is not None
                and source_stream_idx != target_stream_idx
                and source_stream_idx < len(self.sm_counts)
                and target_stream_idx < len(self.sm_counts)
            ):
                source_decode_sm = self.sm_counts[source_stream_idx][1]
                target_decode_sm = self.sm_counts[target_stream_idx][1]
                if source_decode_sm > 0 and target_decode_sm > 0:
                    predicted_p99_ms *= float(source_decode_sm) / float(
                        target_decode_sm
                    )
            return predicted_p99_ms

        gpu_prediction_ms = predict_p99(
            getattr(
                self, "rag_ann_decode_gpu_partition_batch_ms", {}
            ),
            getattr(self, "rag_ann_decode_gpu_partition_ms", {}),
            getattr(self, "rag_ann_decode_gpu_ms", None),
            "decode_gpu_by_partition_batch_bucket",
            "decode_gpu_distribution_by_partition_batch_bucket",
            getattr(
                self,
                "rag_ann_decode_gpu_partition_batch_overlap_ms",
                {},
            ),
        )
        iteration_prediction_ms = None
        if include_scheduling or gpu_prediction_ms is None:
            iteration_prediction_ms = predict_p99(
                getattr(self, "rag_ann_decode_partition_batch_ms", {}),
                getattr(self, "rag_ann_decode_partition_ms", {}),
                getattr(self, "rag_ann_decode_iteration_ms", None),
                "decode_iteration_by_partition_batch_bucket",
                "decode_iteration_distribution_by_partition_batch_bucket",
                getattr(
                    self,
                    "rag_ann_decode_partition_batch_overlap_ms",
                    {},
                ),
            )
        if include_scheduling:
            predictions = [
                value
                for value in (gpu_prediction_ms, iteration_prediction_ms)
                if value is not None
            ]
            return max(predictions) if predictions else None
        result_ms = (
            gpu_prediction_ms
            if gpu_prediction_ms is not None
            else iteration_prediction_ms
        )
        if diagnostic:
            self._rag_last_decode_prediction_components = dict(
                source="legacy_history_or_prior",
                gpu_history_or_prior_ms=gpu_prediction_ms,
                iteration_history_or_prior_ms=iteration_prediction_ms,
                final_ms=result_ms,
            )
        return result_ms

    def _rag_prefill_nominal_decode_prediction_ms(
        self,
        target_stream_idx: Optional[int] = None,
        decode_bs: Optional[int] = None,
        *,
        include_scheduling: bool = False,
        percentile: Optional[float] = None,
        overlap_class: Optional[str] = None,
    ) -> Optional[float]:
        """Return service time for bundle sizing, not the admission floor.

        The calibrated causal Decode guard is deliberately conservative: it
        is a tail-safety floor for admission, not a nominal service-time
        estimate. Feeding that floor back into the legacy Prefill bundle
        heuristic makes every normal co-run look like severe Decode risk and
        fragments Prefill into tiny bundles. Keep the historical predictor
        behavior when no calibrated bundle is installed; with the calibrated
        bundle, bypass only the causal floor. Explicit admission and
        tail-liveness checks continue to call `_rag_ann_decode_prediction_ms`
        directly and therefore retain the guard.
        """
        kwargs = dict(
            include_scheduling=include_scheduling,
            overlap_class=overlap_class,
        )
        if percentile is not None:
            kwargs["percentile"] = percentile
        if getattr(self, "rag_legacy_calibrated_bundle", None) is not None:
            kwargs["apply_causal_guard"] = False
        args = (
            (target_stream_idx,)
            if decode_bs is None
            else (target_stream_idx, decode_bs)
        )
        return self._rag_ann_decode_prediction_ms(*args, **kwargs)

    def _rag_request_arrival_rps(self) -> Optional[float]:
        arrivals = list(getattr(self, "rag_request_arrival_times", ()))
        if len(arrivals) < 4:
            return None
        span_s = arrivals[-1] - arrivals[0]
        if span_s <= 1e-6:
            return None
        return float(len(arrivals) - 1) / float(span_s)

    def _rag_system_load_state(self, candidate_stream_idx: int) -> dict:
        arrival_rps = self._rag_request_arrival_rps()
        if not self._decode_batch_active():
            return {
                "low": True,
                "reason": "no_decode",
                "batch_p95": None,
                "queue_p95_ms": None,
                "predicted_ms": None,
                "slo_ms": None,
                "arrival_rps": arrival_rps,
            }

        batch_history = list(
            getattr(self, "rag_ann_decode_batch_sizes", ())
        )
        queue_history = list(
            getattr(self, "rag_ann_decode_queue_delay_ms", ())
        )
        batch_p95 = (
            float(_history_percentile_value(batch_history, 95))
            if len(batch_history) >= 4
            else None
        )
        queue_p95_ms = (
            float(_history_percentile_value(queue_history, 95))
            if queue_history
            else 0.0
        )
        predicted_ms = self._rag_ann_decode_prediction_ms(
            candidate_stream_idx
        )
        slo_ms = self._rag_ann_decode_slo_ms(candidate_stream_idx)
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        candidate_safe = (
            predicted_ms is not None
            and (
                slo_ms <= 0.0
                or predicted_ms + safety_margin_ms < slo_ms
            )
        )
        batch_low = (
            batch_p95 is not None
            and batch_p95
            <= getattr(
                self,
                "rag_ann_fast_path_decode_batch_p95_threshold",
                16.0,
            )
        )
        queue_low = queue_p95_ms <= getattr(
            self, "rag_ann_fast_path_decode_queue_p95_ms", 2.0
        )
        low = bool(batch_low and queue_low and candidate_safe)
        reasons = []
        if not batch_low:
            reasons.append("batch")
        if not queue_low:
            reasons.append("queue")
        if not candidate_safe:
            reasons.append("candidate")
        return {
            "low": low,
            "reason": "low" if low else "+".join(reasons),
            "batch_p95": batch_p95,
            "queue_p95_ms": queue_p95_ms,
            "predicted_ms": predicted_ms,
            "slo_ms": slo_ms if slo_ms > 0.0 else None,
            "arrival_rps": arrival_rps,
        }

    def _record_rag_ann_slo_defer(self):
        pending = getattr(self, "ann_search_pending", None)
        if pending and not pending[0].get("slo_deferred", False):
            pending[0]["slo_deferred"] = True
            self.rag_ann_slo_deferred_requests += 1

    def _record_rag_ann_budget_reject(self):
        pending = getattr(self, "ann_search_pending", None)
        if not pending:
            return

        task = pending[0]
        self.rag_ann_budget_reject_events += 1
        if "budget_defer_started" not in task:
            task["budget_defer_started"] = time.perf_counter()
        if not task.get("budget_rejected", False):
            task["budget_rejected"] = True
            self.rag_ann_budget_rejected_requests += 1
        self._record_rag_ann_slo_defer()

    def _record_rag_ann_urgency_admission(self, wait_ms: float):
        pending = getattr(self, "ann_search_pending", None)
        if not pending:
            return

        task = pending[0]
        if task.get("urgency_admitted", False):
            return
        task["urgency_admitted"] = True
        self.rag_ann_urgency_admitted_requests += 1
        self.rag_ann_urgency_admission_wait_ms.append(float(wait_ms))

    def _evaluate_rag_ann_decode_budget(
        self, oldest_wait_ms: float, candidate_stream_idx: Optional[int] = None
    ) -> dict:
        if not self._rag_ann_uses_partitioned_gc():
            return {
                "allowed": True,
                "forced": False,
                "reason": "non_gc",
                "predicted_ms": None,
                "slo_ms": None,
            }
        if not self._decode_batch_active():
            return {
                "allowed": True,
                "forced": False,
                "reason": "no_decode",
                "predicted_ms": None,
                "slo_ms": None,
            }

        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        if (
            self._rag_ann_uses_frontend_pool()
            and get_current_stream_idx() == decode_only_idx
        ):
            decode_only_wait_ms = getattr(
                self, "rag_ann_decode_only_wait_ms", 10.0
            )
            if (
                decode_only_wait_ms < 0
                or oldest_wait_ms < decode_only_wait_ms
            ):
                slo_ms = self._rag_ann_decode_slo_ms(candidate_stream_idx)
                return {
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_only_guard",
                    "predicted_ms": self._rag_ann_decode_prediction_ms(
                        candidate_stream_idx,
                        include_scheduling=True,
                        overlap_class="ann",
                    ),
                    "slo_ms": slo_ms if slo_ms > 0 else None,
                }

        slo_ms = self._rag_ann_decode_slo_ms(candidate_stream_idx)
        predicted_ms = self._rag_ann_decode_prediction_ms(
            candidate_stream_idx,
            include_scheduling=True,
            overlap_class="ann",
        )
        if slo_ms <= 0 or predicted_ms is None:
            return {
                "allowed": True,
                "forced": False,
                "reason": "unconstrained",
                "predicted_ms": predicted_ms,
                "slo_ms": slo_ms if slo_ms > 0 else None,
            }

        max_defer_ms = max(
            0.0, getattr(self, "rag_ann_max_defer_ms", 10.0)
        )
        if oldest_wait_ms >= max_defer_ms:
            return {
                "allowed": True,
                "forced": True,
                "reason": "max_defer",
                "predicted_ms": predicted_ms,
                "slo_ms": slo_ms,
                "admission_limit_ms": None,
                "urgency_progress": 1.0,
            }

        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 5.0)
        )
        strict_limit_ms = slo_ms - safety_margin_ms
        urgency_start_ratio = min(
            1.0,
            max(
                0.0,
                getattr(self, "rag_ann_urgency_start_ratio", 1.0),
            ),
        )
        max_slowdown_ratio = max(
            1.0,
            getattr(self, "rag_ann_max_decode_slowdown_ratio", 1.0),
        )
        urgency_progress = 0.0
        admission_limit_ms = strict_limit_ms
        if max_defer_ms > 0.0 and max_slowdown_ratio > 1.0:
            urgency_start_ms = max_defer_ms * urgency_start_ratio
            if oldest_wait_ms > urgency_start_ms:
                urgency_window_ms = max(
                    0.001, max_defer_ms - urgency_start_ms
                )
                urgency_progress = min(
                    1.0,
                    max(
                        0.0,
                        (oldest_wait_ms - urgency_start_ms)
                        / urgency_window_ms,
                    ),
                )
                relaxed_limit_ms = max(
                    strict_limit_ms, slo_ms * max_slowdown_ratio
                )
                admission_limit_ms += urgency_progress * (
                    relaxed_limit_ms - strict_limit_ms
                )

        allowed = predicted_ms < admission_limit_ms
        urgency_allowed = bool(
            allowed
            and urgency_progress > 0.0
            and predicted_ms >= strict_limit_ms
        )
        return {
            "allowed": allowed,
            "forced": False,
            "reason": (
                "decode_urgency"
                if urgency_allowed
                else "decode_slack" if allowed else "decode_budget"
            ),
            "predicted_ms": predicted_ms,
            "slo_ms": slo_ms,
            "admission_limit_ms": admission_limit_ms,
            "urgency_progress": urgency_progress,
        }

    def _oldest_ann_wait_ms(self) -> float:
        pending = getattr(self, "ann_search_pending", None)
        if not pending:
            return 0.0
        return (
            time.perf_counter()
            - pending[0].get("enqueue_time", time.perf_counter())
        ) * 1000.0

    def _oldest_prefill_wait_ms(self) -> float:
        oldest = None
        split_batch = getattr(self, "split_prefill_batch", None)
        request_groups = []
        if split_batch is not None and not split_batch.is_empty():
            request_groups.append(split_batch.reqs)
        waiting_queue = getattr(self, "waiting_queue", None)
        if waiting_queue:
            request_groups.append(waiting_queue)

        for requests in request_groups:
            for req in requests:
                queued_at = getattr(req, "queue_time_start", None)
                if queued_at is not None:
                    oldest = queued_at if oldest is None else min(oldest, queued_at)
        if oldest is None:
            return 0.0
        return (time.perf_counter() - oldest) * 1000.0

    def _oldest_dependency_prefill_wait_ms(self) -> float:
        """Return scheduler wait since ANN handoff for the active Prefill."""
        split_batch = getattr(self, "split_prefill_batch", None)
        if split_batch is None or split_batch.is_empty():
            return 0.0

        oldest = None
        for req in getattr(split_batch, "reqs", ()):
            if not (
                getattr(req, "rag_ann_handoff_ready", False)
                or getattr(req, "rag_prefill_handoff_profiled", False)
            ):
                continue
            ready_at = getattr(req, "rag_scheduler_arrival_at", None)
            if ready_at is None:
                ready_at = getattr(req, "rag_ann_completed_at", None)
            if ready_at is not None:
                oldest = ready_at if oldest is None else min(oldest, ready_at)
        if oldest is None:
            return 0.0
        return max(0.0, (time.perf_counter() - oldest) * 1000.0)

    def _should_wait_for_rag_prefill_cohort(self) -> bool:
        """Coalesce ready RAG handoffs before an exclusive Prefill lease."""
        if (
            not getattr(self, "rag_prefill_exclusive_sprint", False)
            or not getattr(self, "rag_prefill_dynamic_lease", False)
            or getattr(self, "rag_policy_mode", "full") != "full"
            or not self._rag_ann_uses_partitioned_gc()
            or not self._decode_batch_active()
        ):
            return False

        min_work = max(1, getattr(self, "rag_lease_coalesce_min_work", 1))
        max_wait_ms = max(
            0.0, getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0)
        )
        if min_work <= 1 or max_wait_ms <= 0.0:
            return False

        waiting = list(getattr(self, "waiting_queue", None) or ())
        if not waiting:
            return False
        handoffs = [
            req
            for req in waiting
            if getattr(req, "rag_ann_handoff_ready", False)
        ]
        # Do not delay unrelated Prefill work behind a RAG-only cohort policy.
        if not handoffs or len(handoffs) != len(waiting):
            return False
        if len(handoffs) >= min_work:
            return False

        ready_times = []
        for req in handoffs:
            ready_at = getattr(req, "rag_scheduler_arrival_at", None)
            if ready_at is None:
                ready_at = getattr(req, "queue_time_start", None)
            if ready_at is None:
                return False
            ready_times.append(float(ready_at))
        oldest_wait_ms = max(
            0.0, (time.perf_counter() - min(ready_times)) * 1000.0
        )
        if oldest_wait_ms >= max_wait_ms:
            return False

        self.rag_prefill_cohort_holds = (
            getattr(self, "rag_prefill_cohort_holds", 0) + 1
        )
        return True

    def _rag_prefill_work_equivalent_rps(
        self, arrival_rps: Optional[float]
    ) -> Optional[float]:
        """Normalize request rate by average active Prefill tokens."""
        if arrival_rps is None:
            return None
        split_batch = getattr(self, "split_prefill_batch", None)
        if split_batch is None or split_batch.is_empty():
            return float(arrival_rps)

        batch_size = max(1, int(split_batch.batch_size()))
        token_count = max(0, int(getattr(split_batch, "extend_num_tokens", 0)))
        tokens_per_request = token_count / batch_size
        reference_tokens = max(
            1,
            int(
                getattr(
                    self,
                    "rag_prefill_low_load_reference_tokens",
                    1024,
                )
            ),
        )
        work_scale = max(1.0, tokens_per_request / reference_tokens)
        return float(arrival_rps) * work_scale

    def _adaptive_frontend_stream_idx(self, decode_bs: int) -> int:
        manual_divisions = self.pdmux_config.manual_divisions
        if manual_divisions:
            stream_idx = 1
            for i, (_, _, threshold) in enumerate(manual_divisions):
                if decode_bs >= threshold:
                    stream_idx = i + 1
            return stream_idx

        return max(
            1,
            min(
                self.real_sm_group_num - 2,
                decode_bs
                * (self.real_sm_group_num - 2)
                // self.pdmux_config.decode_bs_divisor,
            ),
        )

    def _maybe_raise_rag_decode_partition_floor(self, stream_idx: int):
        if (
            getattr(self, "rag_policy_mode", "full") != "full"
            or not getattr(self, "rag_partition_slo_floor_enabled", True)
            or not self._rag_ann_uses_partitioned_gc()
        ):
            return

        slo_ms = self._rag_ann_decode_slo_ms()
        if slo_ms <= 0:
            return
        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        if not 1 <= stream_idx <= last_mixed_idx:
            return

        history = getattr(self, "rag_ann_decode_partition_ms", {}).get(
            stream_idx
        )
        if history is None or len(history) < 8:
            return

        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        guard_ms = max(0.001, slo_ms - safety_margin_ms)
        interval_p95_ms = float(_history_percentile_value(list(history), 95))
        current_decode_sm = self.sm_counts[stream_idx][1]
        current_floor = getattr(self, "rag_partition_decode_floor_sm", 0)
        if interval_p95_ms < guard_ms:
            if stream_idx == last_mixed_idx:
                self.rag_partition_decode_only_fallback_required = False

            release_guard_ms = max(0.001, guard_ms * 0.9)
            floor_is_observed = (
                current_floor > 0 and current_decode_sm == current_floor
            )
            if floor_is_observed and interval_p95_ms <= release_guard_ms:
                safe_observations = (
                    getattr(
                        self,
                        "rag_partition_floor_safe_observations",
                        0,
                    )
                    + 1
                )
                self.rag_partition_floor_safe_observations = safe_observations
                if safe_observations >= 8:
                    lower_decode_counts = sorted(
                        {
                            self.sm_counts[idx][1]
                            for idx in range(1, last_mixed_idx + 1)
                            if self.sm_counts[idx][1] < current_floor
                        }
                    )
                    self.rag_partition_decode_floor_sm = (
                        lower_decode_counts[-1] if lower_decode_counts else 0
                    )
                    self.rag_partition_slo_demotions = (
                        getattr(self, "rag_partition_slo_demotions", 0) + 1
                    )
                    self.rag_partition_floor_safe_observations = 0
            else:
                self.rag_partition_floor_safe_observations = 0
            return

        self.rag_partition_floor_safe_observations = 0
        safer_decode_counts = sorted(
            {
                self.sm_counts[idx][1]
                for idx in range(1, last_mixed_idx + 1)
                if self.sm_counts[idx][1] > current_decode_sm
            }
        )
        if not safer_decode_counts:
            if not getattr(
                self, "rag_partition_decode_only_fallback_required", False
            ):
                self.rag_partition_decode_only_fallback_activations = (
                    getattr(
                        self,
                        "rag_partition_decode_only_fallback_activations",
                        0,
                    )
                    + 1
                )
            self.rag_partition_decode_only_fallback_required = True
            return

        next_decode_sm = safer_decode_counts[0]
        if next_decode_sm > current_floor:
            self.rag_partition_decode_floor_sm = next_decode_sm
            self.rag_partition_slo_promotions = (
                getattr(self, "rag_partition_slo_promotions", 0) + 1
            )

    def _apply_rag_decode_partition_floor(self, candidate_idx: int) -> int:
        floor_sm = getattr(self, "rag_partition_decode_floor_sm", 0)
        if floor_sm <= 0 or self.sm_counts[candidate_idx][1] >= floor_sm:
            return candidate_idx

        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        candidates = [
            idx
            for idx in range(1, last_mixed_idx + 1)
            if self.sm_counts[idx][1] >= floor_sm
        ]
        if not candidates:
            return candidate_idx
        return min(candidates, key=lambda idx: (self.sm_counts[idx][1], idx))

    def _apply_decode_partition_hysteresis(
        self, candidate_idx: int, decode_bs: int
    ) -> int:
        hysteresis = max(
            0, getattr(self, "rag_partition_hysteresis_bs", 0)
        )
        current_idx = get_current_stream_idx()
        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        manual_divisions = self.pdmux_config.manual_divisions
        if (
            hysteresis <= 0
            or not manual_divisions
            or not 1 <= current_idx <= last_mixed_idx
            or not 1 <= candidate_idx <= last_mixed_idx
            or candidate_idx == current_idx
        ):
            return candidate_idx

        if candidate_idx > current_idx:
            enter_threshold = manual_divisions[candidate_idx - 1][2]
            if decode_bs < enter_threshold + hysteresis:
                self.rag_partition_hysteresis_holds += 1
                return current_idx
        else:
            exit_threshold = manual_divisions[current_idx - 1][2]
            if decode_bs > exit_threshold - hysteresis:
                self.rag_partition_hysteresis_holds += 1
                return current_idx
        return candidate_idx

    def _stream_idx_for_frontend_sm(self, frontend_sm: int) -> Optional[int]:
        if frontend_sm <= 0:
            return None
        mixed = range(1, max(1, getattr(self, "real_sm_group_num", 1) - 1))
        candidates = [idx for idx in mixed if idx < len(self.sm_counts)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda idx: (abs(self.sm_counts[idx][0] - frontend_sm), idx),
        )

    def _apply_partition_residency(self, candidate_idx: int) -> int:
        current_idx = get_current_stream_idx()
        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        min_residency_ms = max(
            0.0, getattr(self, "rag_partition_min_residency_ms", 0.0)
        )
        if (
            min_residency_ms <= 0.0
            or candidate_idx == current_idx
            or not 1 <= current_idx <= last_mixed_idx
            or not 1 <= candidate_idx <= last_mixed_idx
        ):
            return candidate_idx

        elapsed_ms = (
            time.perf_counter()
            - getattr(self, "rag_partition_last_switch_ts", 0.0)
        ) * 1000.0
        if elapsed_ms < min_residency_ms:
            self.rag_partition_residency_holds += 1
            return current_idx
        return candidate_idx

    def _safe_mixed_hold_threshold_ms(self) -> Optional[float]:
        slo_ms = self._rag_ann_decode_slo_ms()
        if slo_ms <= 0.0:
            return None
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        guard_ms = slo_ms - safety_margin_ms
        if guard_ms <= 0.0:
            return None
        ratio = getattr(self, "rag_partition_safe_mixed_hold_ratio", 0.9)
        return guard_ms * min(1.0, max(0.0, ratio))

    def _should_hold_safe_mixed_partition(self, current_idx: int) -> bool:
        hold_enabled = bool(
            getattr(self, "rag_partition_safe_mixed_hold", False)
            or getattr(self, "rag_prefill_dynamic_lease", False)
        )
        if (
            not hold_enabled
            or getattr(self, "rag_policy_mode", "full") != "full"
            or not self._rag_ann_uses_partitioned_gc()
            or getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
            or getattr(self, "rag_decode_recovery_pending", False)
        ):
            return False

        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        if not 1 <= current_idx <= last_mixed_idx:
            return False

        fallback_hold_ms = max(
            0.0, getattr(self, "rag_partition_safe_mixed_max_hold_ms", 5.0)
        )
        frontend_idle_ms = (
            time.perf_counter()
            - getattr(self, "rag_last_frontend_activity_ts", 0.0)
        ) * 1000.0

        threshold_ms = self._safe_mixed_hold_threshold_ms()
        if threshold_ms is None:
            return False
        overlap_class = (
            "ann" if self._rag_ann_gpu_work_active() else "solo"
        )
        service_p95_ms = self._rag_ann_decode_prediction_ms(
            current_idx,
            include_scheduling=False,
            percentile=95.0,
            overlap_class=overlap_class,
        )
        if service_p95_ms is None:
            return False
        if service_p95_ms > threshold_ms:
            return False

        arrival_rps = self._rag_request_arrival_rps()
        if (
            not getattr(self, "rag_prefill_dynamic_lease", False)
            or arrival_rps is None
            or arrival_rps <= 0.0
        ):
            return (
                fallback_hold_ms > 0.0
                and frontend_idle_ms <= fallback_hold_ms
            )

        # Estimate whether keeping the safe mixed partition until the next
        # frontend arrival costs less than a decode-only round trip.
        expected_interval_ms = min(500.0, 1000.0 / arrival_rps)
        hold_horizon_ms = max(fallback_hold_ms, expected_interval_ms)
        if frontend_idle_ms > hold_horizon_ms:
            return False

        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        decode_only_p95_ms = self._rag_ann_decode_prediction_ms(
            decode_only_idx,
            include_scheduling=False,
            percentile=95.0,
            overlap_class="solo",
        )
        if decode_only_p95_ms is None:
            self.rag_partition_safe_mixed_cost_prediction_misses = (
                getattr(
                    self,
                    "rag_partition_safe_mixed_cost_prediction_misses",
                    0,
                )
                + 1
            )
            return (
                fallback_hold_ms > 0.0
                and frontend_idle_ms <= fallback_hold_ms
            )

        remaining_ms = max(0.0, hold_horizon_ms - frontend_idle_ms)
        expected_decode_steps = max(
            1,
            int(
                math.ceil(
                    remaining_ms / max(1.0, decode_only_p95_ms)
                )
            ),
        )
        stay_cost_ms = (
            max(0.0, service_p95_ms - decode_only_p95_ms)
            * expected_decode_steps
        )
        round_trip_cost_ms = (
            self._rag_lease_switch_cost_estimate_ms(
                current_idx, decode_only_idx
            )
            + self._rag_lease_switch_cost_estimate_ms(
                decode_only_idx, current_idx
            )
        )
        self.rag_partition_safe_mixed_last_cost_decision = {
            "arrival_rps": float(arrival_rps),
            "frontend_idle_ms": float(frontend_idle_ms),
            "hold_horizon_ms": float(hold_horizon_ms),
            "mixed_decode_p95_ms": float(service_p95_ms),
            "decode_only_p95_ms": float(decode_only_p95_ms),
            "expected_decode_steps": int(expected_decode_steps),
            "stay_cost_ms": float(stay_cost_ms),
            "round_trip_cost_ms": float(round_trip_cost_ms),
        }
        if stay_cost_ms <= round_trip_cost_ms:
            self.rag_partition_safe_mixed_cost_admissions = (
                getattr(
                    self,
                    "rag_partition_safe_mixed_cost_admissions",
                    0,
                )
                + 1
            )
            return True
        self.rag_partition_safe_mixed_cost_rejections = (
            getattr(
                self,
                "rag_partition_safe_mixed_cost_rejections",
                0,
            )
            + 1
        )
        return False

    def _rag_ann_should_launch_early(self) -> bool:
        launch_site = getattr(self, "rag_ann_launch_site", "early")
        if launch_site == "early":
            return True
        if launch_site != "adaptive":
            return False

        plan = getattr(self, "rag_frontend_plan", None)
        return bool(
            plan is not None
            and plan.allow_ann
            and plan.launch_ann_early
        )

    def _rag_ann_yields_to_prefill_transition(
        self, current_stream_idx: int
    ) -> bool:
        """Do not refill Decode-GC ANN while a dependent Prefill must switch."""
        plan = getattr(self, "rag_frontend_plan", None)
        if (
            plan is None
            or not self._rag_ann_uses_decode_pool()
            or not getattr(plan, "allow_prefill", False)
            or int(getattr(plan, "target_stream_idx", current_stream_idx))
            == current_stream_idx
        ):
            return False
        return bool(
            getattr(plan, "immediate_handoff", False)
            or getattr(plan, "dependency_prefill_due", False)
        )

    def _rag_ann_should_launch_after_prefill(self) -> bool:
        launch_site = getattr(self, "rag_ann_launch_site", "early")
        if launch_site == "after-prefill":
            return True
        if launch_site != "adaptive":
            return False
        return not self._rag_ann_should_launch_early()

    def _rag_low_pressure_fast_path_allowed(
        self,
        *,
        has_ann: bool,
        has_prefill: bool,
        active_ann: bool,
        system_load: dict,
    ) -> bool:
        if (
            getattr(self, "rag_ann_launch_site", "early") != "adaptive"
            or getattr(self, "rag_policy_mode", "full") != "full"
            or not self._rag_ann_uses_partitioned_gc()
            or getattr(self, "rag_ann_frontend_arbitration", "overlap")
            != "fair"
            or active_ann
            or getattr(self, "rag_decode_recovery_pending", False)
            or getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
            or not system_load.get("low", False)
        ):
            return False

        dependency_work = has_ann or (
            has_prefill
            and getattr(self, "rag_prefill_handoff_active", False)
        )
        if not dependency_work:
            return False

        ann_rows, prefill_items = self._rag_lease_work_counts(
            allow_ann=has_ann,
            allow_prefill=has_prefill,
        )
        max_work = max(
            1, getattr(self, "rag_ann_fast_path_max_work", 1)
        )
        return 0 < ann_rows + prefill_items <= max_work

    def _rag_prefill_sprint_eligible(self) -> bool:
        """Return whether the active Prefill fits in one bounded burst."""
        split_batch = getattr(self, "split_prefill_batch", None)
        if split_batch is None or split_batch.is_empty():
            return False

        extend_tokens = int(getattr(split_batch, "extend_num_tokens", 0))
        model_config = getattr(self, "model_config", None)
        num_layers = int(getattr(model_config, "num_hidden_layers", 0))
        split_index = int(getattr(split_batch, "split_index", 0))
        remaining_layers = max(0, num_layers - split_index)
        if extend_tokens <= 0 or remaining_layers <= 0:
            return False

        layer_token_budget = max(
            1, int(getattr(self.pdmux_config, "split_forward_token_budget", 1))
        )
        max_chunks = max(
            1, int(getattr(self, "rag_prefill_max_consecutive_chunks", 1))
        )
        burst_layers = min(
            remaining_layers,
            max(
                1,
                int(getattr(self, "rag_prefill_lease_max_layers", 1)),
            ),
        )
        return extend_tokens * burst_layers <= layer_token_budget * max_chunks

    @staticmethod
    def _rag_prefill_token_bucket(token_count: int) -> int:
        token_count = max(1, int(token_count))
        return 1 << (token_count - 1).bit_length()

    def _rag_prefill_e2e_slack_ms(self) -> Optional[float]:
        slo_ms = float(getattr(self, "rag_e2e_slo_ms", -1.0))
        if slo_ms <= 0.0:
            return None
        split_batch = getattr(self, "split_prefill_batch", None)
        if split_batch is None or split_batch.is_empty():
            return None

        starts = []
        for req in split_batch.reqs:
            started_at = getattr(req, "rag_e2e_started_at", None)
            if started_at is None:
                started_at = getattr(req, "queue_time_start", None)
            if started_at is not None:
                starts.append(float(started_at))
        if not starts:
            return None
        return slo_ms - (time.perf_counter() - min(starts)) * 1000.0

    def _predict_rag_prefill_layer_ms(
        self, stream_idx: int, token_count: int, *, percentile: float = 95.0
    ) -> float:
        token_count = max(1, int(token_count))
        bundle=getattr(self,'rag_legacy_calibrated_bundle',None)
        split=getattr(self,'split_prefill_batch',None)
        if (bundle is not None and split is not None and not split.is_empty()
                and 0<=stream_idx<len(self.sm_counts) and self.sm_counts[stream_idx][0]>0):
            new=tuple(int(value) for value in (getattr(split,'extend_lens',None) or ()))
            cached=tuple(int(value) for value in (getattr(split,'prefix_lens',None) or ()))
            if new and len(new)==len(cached) and sum(new)==token_count:
                predicted=bundle.stage.predict_prefill_ms(self.sm_counts[stream_idx][0],new,cached,
                    expected_identity=bundle.identity)
                if predicted is None:
                    predicted=bundle.stage.predict_prefill_nearest_ms(
                        self.sm_counts[stream_idx][0],new,cached,expected_identity=bundle.identity)
                if predicted is None:
                    self._record_rag_legacy_calibrated_fallback_reason(
                        'prefill','stage_cell_unavailable')
                has_ann=self._rag_ann_gpu_work_active()
                has_decode=self._decode_batch_active()
                overlap=('ann_decode' if has_ann and has_decode else
                         'ann' if has_ann else 'decode' if has_decode else None)
                # Only ANN interference on prefill has an accepted correction.
                # Decode-only overlap retains the partition baseline; a
                # three-way overlap applies the ANN guard only.
                correction_overlap='ann' if overlap in ('ann','ann_decode') else None
                if predicted is not None and correction_overlap is not None:
                    from sglang.srt.multiplex.stage_predictor import prefill_features
                    factors=bundle.corrections.query('prefill',stream_idx,correction_overlap,
                        prefill_features(new,cached),expected_identity=bundle.identity)
                    if factors is None:
                        self._record_rag_legacy_calibrated_fallback_reason(
                            'prefill','ann_correction_unavailable')
                        predicted=None
                    else:
                        predicted*=factors[1]
                if predicted is not None:
                    total_layers=max(1,int(self.model_config.num_hidden_layers))
                    counts=getattr(self,'rag_legacy_calibrated_prediction_counts',{})
                    counts['prefill']=counts.get('prefill',0)+1
                    self.rag_legacy_calibrated_prediction_counts=counts
                    self.rag_legacy_calibrated_last_prefill=dict(partition=stream_idx,
                        new_tokens=new,cached_tokens=cached,overlap=overlap,
                        correction_overlap=correction_overlap,
                        whole_ms=predicted,layer_ms=predicted/total_layers)
                    return max(0.001,predicted/total_layers)
            else:
                self._record_rag_legacy_calibrated_fallback_reason(
                    'prefill','shape_unavailable')
            counts=getattr(self,'rag_legacy_calibrated_fallback_counts',{})
            counts['prefill']=counts.get('prefill',0)+1
            self.rag_legacy_calibrated_fallback_counts=counts
        bucket = self._rag_prefill_token_bucket(token_count)
        percentile = min(100.0, max(0.0, float(percentile)))
        histories = getattr(
            self, "rag_prefill_layer_ms_by_partition_bucket", {}
        )
        direct = list(histories.get((stream_idx, bucket), ()))
        if direct:
            return max(0.001, float(_history_percentile_value(direct, percentile)))

        profile_priors = getattr(self, "rag_lease_profile_priors", {})
        profile_histories = profile_priors.get(
            "prefill_layer_by_partition_bucket", {}
        )
        profile_distributions = profile_priors.get(
            "prefill_layer_distribution_by_partition_bucket", {}
        )
        profile_distribution = profile_distributions.get((stream_idx, bucket))
        profile_distribution_value = _profile_distribution_percentile_ms(
            profile_distribution, percentile
        )
        if profile_distribution_value is not None:
            return max(0.001, profile_distribution_value)
        profile_direct = profile_histories.get((stream_idx, bucket))
        if profile_direct is not None:
            return max(0.001, float(profile_direct))

        candidate_frontend = 0
        sm_counts = getattr(self, "sm_counts", ())
        if 0 <= stream_idx < len(sm_counts):
            candidate_frontend = int(sm_counts[stream_idx][0])
        scaled = []
        if candidate_frontend > 0:
            for (source_idx, source_bucket), history in histories.items():
                if source_bucket != bucket or not history:
                    continue
                source_frontend = int(sm_counts[source_idx][0])
                if source_frontend <= 0:
                    continue
                source_p95 = float(_history_percentile_value(list(history), percentile))
                scaled.append(
                    source_p95 * float(source_frontend) / candidate_frontend
                )
        if scaled:
            return max(0.001, float(np.median(scaled)))

        profile_scaled = []
        if candidate_frontend > 0:
            for (source_idx, source_bucket), value in profile_histories.items():
                if source_bucket != bucket:
                    continue
                source_frontend = int(sm_counts[source_idx][0])
                if source_frontend <= 0:
                    continue
                distribution_value = _profile_distribution_percentile_ms(
                    profile_distributions.get((source_idx, source_bucket)),
                    percentile,
                )
                source_value = (
                    float(value)
                    if distribution_value is None
                    else distribution_value
                )
                profile_scaled.append(
                    source_value * float(source_frontend) / candidate_frontend
                )
        if profile_scaled:
            return max(0.001, float(np.median(profile_scaled)))

        token_budget = max(
            1, int(getattr(self.pdmux_config, "split_forward_token_budget", 1))
        )
        nominal_layers = max(1, token_budget // token_count)
        fallback_bundle_ms = max(
            0.001,
            float(
                getattr(
                    self, "rag_lease_prefill_service_estimate_ms", 10.0
                )
            ),
        )
        return fallback_bundle_ms / nominal_layers

    def _rag_prefill_two_level_layer_counts(self) -> tuple[int, int]:
        num_layers = max(
            1,
            int(
                getattr(
                    getattr(self, "model_config", None),
                    "num_hidden_layers",
                    1,
                )
            ),
        )
        return max(1, (num_layers + 1) // 2), num_layers

    @staticmethod
    def _rag_prefill_request_token_count(
        split_batch, fallback_total_tokens: int
    ) -> tuple[int, bool]:
        """Return the largest per-request Prefill length when available."""
        request_lengths = []
        for req in getattr(split_batch, "reqs", ()) or ():
            value = getattr(req, "extend_input_len", None)
            if value is not None:
                request_lengths.append(max(0, int(value)))
        if not request_lengths:
            extend_lens = getattr(split_batch, "extend_lens", None)
            for value in extend_lens if extend_lens is not None else ():
                request_lengths.append(max(0, int(value)))
        if request_lengths:
            return max(1, max(request_lengths)), True
        return max(1, int(fallback_total_tokens)), False

    def _quantize_rag_prefill_layer_bundle(
        self, requested_layers: int, remaining_layers: int
    ) -> int:
        requested_layers = max(1, int(requested_layers))
        remaining_layers = max(0, int(remaining_layers))
        if remaining_layers <= 0:
            return 0
        if not getattr(self, "rag_prefill_two_level_bundles", False):
            return min(requested_layers, remaining_layers)

        half_layers, full_layers = self._rag_prefill_two_level_layer_counts()
        if requested_layers < half_layers:
            return min(requested_layers, remaining_layers)
        if remaining_layers <= half_layers:
            return remaining_layers
        if remaining_layers >= full_layers and requested_layers >= full_layers:
            return full_layers
        return half_layers

    def _rag_prefill_lease_shape(
        self,
        stream_idx: int,
        predicted_decode_ms: Optional[float] = None,
    ) -> tuple[Optional[int], Optional[float], Optional[float]]:
        split_batch = getattr(self, "split_prefill_batch", None)
        if split_batch is None or split_batch.is_empty():
            return None, None, None

        num_layers = int(
            getattr(getattr(self, "model_config", None), "num_hidden_layers", 0)
        )
        split_index = int(getattr(split_batch, "split_index", 0))
        remaining_layers = max(0, num_layers - split_index)
        token_count = int(getattr(split_batch, "extend_num_tokens", 0))
        request_token_count, has_per_request_lengths = (
            self._rag_prefill_request_token_count(split_batch, token_count)
        )
        e2e_slack_ms = self._rag_prefill_e2e_slack_ms()
        if remaining_layers <= 0:
            return 0, None, e2e_slack_ms

        if not getattr(self, "rag_prefill_dynamic_lease", False):
            if token_count <= 0:
                return remaining_layers, None, e2e_slack_ms
            fixed_count = max(
                1,
                int(
                    getattr(self.pdmux_config, "split_forward_token_budget", 1)
                )
                // token_count,
            )
            return min(remaining_layers, fixed_count), None, e2e_slack_ms

        layer_ms = self._predict_rag_prefill_layer_ms(
            stream_idx, max(1, token_count)
        )
        sm_counts = getattr(self, "sm_counts", ())
        exclusive_endpoint = bool(
            self._decode_batch_active()
            and 0 <= stream_idx < len(sm_counts)
            and int(sm_counts[stream_idx][1]) == 0
        )
        decode_prediction_idx = (
            get_current_stream_idx() if exclusive_endpoint else stream_idx
        )
        candidate_decode_ms = self._rag_prefill_nominal_decode_prediction_ms(
            decode_prediction_idx, overlap_class="prefill"
        )
        if candidate_decode_ms is None:
            candidate_decode_ms = predicted_decode_ms

        configured_min_layers = max(
            1, int(getattr(self, "rag_prefill_lease_min_layers", 1))
        )
        configured_max_layers = max(
            configured_min_layers,
            int(getattr(self, "rag_prefill_lease_max_layers", 8)),
        )
        min_layers = configured_min_layers
        max_layers = configured_max_layers
        if getattr(self, "rag_prefill_two_level_bundles", False):
            half_layers, full_layers = (
                self._rag_prefill_two_level_layer_counts()
            )
            if not self._decode_batch_active():
                min_layers = full_layers
                max_layers = full_layers
            else:
                iteration_decode_ms = self._rag_prefill_nominal_decode_prediction_ms(
                    decode_prediction_idx,
                    include_scheduling=True,
                    overlap_class="prefill",
                )
                guard_decode_ms = (
                    iteration_decode_ms
                    if exclusive_endpoint
                    else candidate_decode_ms
                )
                if exclusive_endpoint:
                    self.rag_prefill_exclusive_iteration_guard_uses += 1
                else:
                    self.rag_prefill_mixed_service_guard_uses += 1
                slo_ms = self._rag_ann_decode_slo_ms(
                    decode_prediction_idx
                )
                safety_margin_ms = max(
                    0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
                )
                decode_risk = bool(
                    guard_decode_ms is not None
                    and slo_ms > 0.0
                    and guard_decode_ms + safety_margin_ms >= slo_ms
                )
                iteration_pressure = bool(
                    not exclusive_endpoint
                    and iteration_decode_ms is not None
                    and slo_ms > 0.0
                    and iteration_decode_ms + safety_margin_ms >= slo_ms
                )
                severe_risk_ratio = max(
                    1.0,
                    float(
                        getattr(
                            self,
                            "rag_prefill_severe_risk_ratio",
                            1.25,
                        )
                    ),
                )
                severe_decode_risk = bool(
                    guard_decode_ms is not None
                    and slo_ms > 0.0
                    and guard_decode_ms + safety_margin_ms
                    >= slo_ms * severe_risk_ratio
                )
                short_token_threshold = max(
                    1,
                    int(
                        getattr(
                            self,
                            "rag_prefill_short_token_threshold",
                            2048,
                        )
                    ),
                )
                long_request = request_token_count > short_token_threshold
                if (
                    has_per_request_lengths
                    and token_count > short_token_threshold
                    and not long_request
                ):
                    self.rag_prefill_aggregate_false_long_avoided += 1
                elif not has_per_request_lengths:
                    self.rag_prefill_request_length_fallbacks += 1
                if severe_decode_risk:
                    min_layers = configured_min_layers
                    max_layers = configured_max_layers
                    self.rag_prefill_risk_bundle_activations += 1
                    self.rag_prefill_severe_risk_bundle_activations += 1
                    if long_request:
                        self.rag_prefill_long_request_bundle_activations += 1
                elif long_request:
                    # Long Prefills need a coarse lease to amortize scheduler
                    # and handoff overhead. Keep at least half of the model in
                    # one bundle while the candidate partition remains within
                    # the Decode guard; only severe risk falls back to the
                    # configured fine-grained bundle above.
                    min_layers = half_layers
                    max_layers = half_layers if decode_risk else full_layers
                    self.rag_prefill_long_request_bundle_activations += 1
                    if decode_risk:
                        self.rag_prefill_risk_bundle_activations += 1
                else:
                    short_layers = min(
                        half_layers,
                        max(
                            1,
                            int(
                                getattr(
                                    self,
                                    "rag_prefill_short_bundle_layers",
                                    16,
                                )
                            ),
                        ),
                    )
                    min_layers = short_layers
                    max_layers = half_layers if decode_risk else full_layers
                    if iteration_pressure and not decode_risk:
                        self.rag_prefill_iteration_pressure_full_bundle_bypasses += 1
                    self.rag_prefill_short_bundle_activations += 1
        if not self._decode_batch_active():
            return (
                self._quantize_rag_prefill_layer_bundle(
                    max_layers, remaining_layers
                ),
                layer_ms,
                e2e_slack_ms,
            )

        if candidate_decode_ms is None:
            aligned_layers = min_layers
        else:
            decode_ratio = max(
                0.001,
                float(getattr(self, "rag_prefill_lease_decode_ratio", 1.0)),
            )
            aligned_layers = int(
                np.ceil(candidate_decode_ms * decode_ratio / layer_ms)
            )

        deadline_layers = min_layers
        if e2e_slack_ms is not None:
            if e2e_slack_ms <= 0.0:
                deadline_layers = max_layers
            elif candidate_decode_ms is not None:
                deadline_layers = int(
                    np.ceil(
                        remaining_layers
                        * max(0.001, candidate_decode_ms)
                        / e2e_slack_ms
                    )
                )
        # An exclusive endpoint pauses Decode instead of overlapping it. Bound
        # that pause to the current Decode iteration time scale; E2E urgency
        # controls how soon another lease is granted, not how long this one can
        # monopolize the GPU.
        forward_count = (
            max(min_layers, aligned_layers)
            if exclusive_endpoint
            else max(min_layers, aligned_layers, deadline_layers)
        )
        return (
            self._quantize_rag_prefill_layer_bundle(
                min(max_layers, forward_count), remaining_layers
            ),
            layer_ms,
            e2e_slack_ms,
        )

    def _rag_tail_liveness_layer_budget(
        self,
        stream_idx: int,
        requested_layers: int,
        layer_ms: float,
    ) -> tuple[int, float]:
        requested_layers = max(1, int(requested_layers))
        layer_ms = max(0.001, float(layer_ms))
        sm_counts = getattr(self, "sm_counts", ())
        mixed_candidate = bool(
            0 <= stream_idx < len(sm_counts)
            and int(sm_counts[stream_idx][1]) > 0
        )
        guard_percentile = min(
            100.0,
            max(
                0.001,
                float(
                    getattr(
                        self, "rag_prefill_decode_guard_percentile", 99.0
                    )
                ),
            ),
        )
        # A mixed GC overlaps Prefill with Decode, so its safety bound is the
        # measured co-run service time. Including scheduler gaps here feeds
        # earlier Prefill deferrals back into the guard. Endpoint leases pause
        # Decode and therefore retain the scheduling-inclusive bound.
        candidate_decode_ms = self._rag_ann_decode_prediction_ms(
            stream_idx,
            include_scheduling=not mixed_candidate,
            percentile=guard_percentile,
            overlap_class=(
                "ann_prefill"
                if self._rag_ann_gpu_work_active()
                else "prefill"
            ),
        )
        if candidate_decode_ms is None:
            return (
                0
                if getattr(self, "rag_prefill_two_level_bundles", False)
                else 1
            ), layer_ms
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        slo_ms = self._rag_ann_decode_slo_ms(stream_idx)
        mixed_decode_safe = bool(
            getattr(self, "rag_prefill_two_level_bundles", False)
            and mixed_candidate
            and (
                slo_ms <= 0.0
                or candidate_decode_ms + safety_margin_ms < slo_ms
            )
        )
        if mixed_decode_safe:
            # The candidate-partition predictor already includes co-run
            # slowdown. On a mixed GC, Prefill does not serially occupy the
            # Decode timeline, so converting residual Decode slack into a
            # second layer cap double-counts the same interference.
            self.rag_prefill_two_level_mixed_budget_bypasses += 1
            return requested_layers, candidate_decode_ms

        decode_slack_ms = (
            max(0.0, slo_ms - candidate_decode_ms - safety_margin_ms)
            if slo_ms > 0.0
            else candidate_decode_ms
        )
        budget_ms = max(
            layer_ms,
            min(candidate_decode_ms, decode_slack_ms),
        )
        budget_layers = max(1, int(np.floor(budget_ms / layer_ms)))
        capped_layers = min(requested_layers, budget_layers)
        if not getattr(self, "rag_prefill_two_level_bundles", False):
            return capped_layers, budget_ms

        split_batch = getattr(self, "split_prefill_batch", None)
        split_index = int(getattr(split_batch, "split_index", 0))
        num_layers = self._rag_prefill_two_level_layer_counts()[1]
        remaining_layers = max(0, num_layers - split_index)
        minimum_layers = min(
            remaining_layers, self._rag_prefill_two_level_layer_counts()[0]
        )
        if minimum_layers <= 0 or capped_layers < minimum_layers:
            configured_min_layers = max(
                1,
                int(
                    getattr(self, "rag_prefill_lease_min_layers", 1)
                ),
            )
            if requested_layers < minimum_layers:
                return (
                    min(
                        requested_layers,
                        remaining_layers,
                        configured_min_layers,
                    ),
                    budget_ms,
                )
            return 0, budget_ms
        return min(requested_layers, minimum_layers), budget_ms

    def _trace_rag_prefill_event(self, kind: str, **fields):
        """Opt-in bounded host timeline; query detection is not GPU finish time."""
        enabled = getattr(self, "_rag_prefill_trace_enabled", None)
        if enabled is None:
            enabled = os.getenv("RAGLEASE_PREFILL_TIMELINE", "0") == "1"
            self._rag_prefill_trace_enabled = enabled
        if not enabled:
            return
        now = time.perf_counter()
        origin = getattr(self, "_rag_prefill_trace_origin", None)
        if origin is None:
            # Trigger from current observable load, never a future trace.
            if self._rag_current_decode_batch_size() < 100:
                return
            origin = now
            self._rag_prefill_trace_origin = now
        count = getattr(self, "_rag_prefill_trace_count", 0)
        if now - origin > 20.0 or count >= 2048:
            return
        self._rag_prefill_trace_count = count + 1
        batch = getattr(self, "split_prefill_batch", None)
        import json
        logger.info("RAGLEASE_PREFILL_TRACE %s", json.dumps(dict(
            seq=count, host_s=now, relative_ms=(now-origin)*1000,
            kind=kind, batch_object=id(batch) if batch is not None else None,
            split_index=getattr(batch, "split_index", None), **fields)))

    def _poll_rag_prefill_bundle_completion(self) -> bool:
        event = getattr(self, "rag_prefill_bundle_inflight_event", None)
        if event is None:
            return False
        if not event.query():
            self._trace_rag_prefill_event("bundle_observed_pending")
            return False

        meta = getattr(self, "rag_prefill_bundle_inflight_meta", None) or {}
        completed_batch = meta.get("batch")
        completed_end = meta.get("end_layer")
        if completed_batch is not None and completed_end is not None:
            # Bind completion to the batch that submitted this event, not a
            # possibly newer self.split_prefill_batch. Updating an endpoint is
            # idempotent; repeated observation never adds the same layers twice.
            completed_batch.rag_completed_prefill_layers = max(
                getattr(completed_batch, "rag_completed_prefill_layers", 0),
                int(completed_end))
        self._record_rag_prefill_host_release(meta)
        start_event = meta.get("start_event")
        layers = max(1, int(meta.get("layers", 1)))
        token_count = max(1, int(meta.get("tokens", 1)))
        stream_idx = int(meta.get("stream_idx", get_current_stream_idx()))
        predicted_bundle_ms = meta.get("predicted_bundle_ms")
        elapsed_ms = None
        interval_ms = None
        if start_event is not None:
            elapsed_ms = max(0.001, float(start_event.elapsed_time(event)))
            layer_ms = elapsed_ms / layers
            interval_ms = self._rag_record_prefill_gpu_interval(meta, event)
            bucket = self._rag_prefill_token_bucket(token_count)
            histories = getattr(
                self, "rag_prefill_layer_ms_by_partition_bucket", None
            )
            if histories is not None:
                histories.setdefault(
                    (stream_idx, bucket), deque(maxlen=128)
                ).append(layer_ms)
            history = getattr(self, "rag_prefill_layer_ms", None)
            if history is not None:
                history.append(layer_ms)
            if predicted_bundle_ms is not None and predicted_bundle_ms > 0.0:
                predicted_bundle_ms = float(predicted_bundle_ms)
                self.rag_prefill_prediction_abs_error_ms.append(
                    abs(predicted_bundle_ms - elapsed_ms)
                )
                self.rag_prefill_prediction_samples += 1
                self.rag_prefill_prediction_pairs.append(
                    {
                        "predicted_ms": predicted_bundle_ms,
                        "observed_ms": elapsed_ms,
                        "stream_idx": stream_idx,
                        "layers": layers,
                        "tokens": token_count,
                        "start_layer": max(
                            0, int(meta.get("end_layer", layers)) - layers),
                        "end_layer": int(meta.get("end_layer", layers)),
                        "gpu_t0_ms": (interval_ms[0] if interval_ms else None),
                        "gpu_t1_ms": (interval_ms[1] if interval_ms else None),
                        "features": self._rag_prefill_interval_features(
                            meta.get("batch")),
                    }
                )
        self._trace_rag_prefill_event("bundle_observed_complete",
            gpu_elapsed_ms=elapsed_ms, layers=layers, partition=stream_idx)
        self._increment_histogram(
            self.rag_prefill_bundle_layer_histogram, layers
        )
        self.rag_prefill_bundle_completions = (
            getattr(self, "rag_prefill_bundle_completions", 0) + 1
        )
        self.rag_prefill_last_bundle_completion_ts = time.perf_counter()
        self.rag_prefill_bundle_inflight_event = None
        self.rag_prefill_bundle_inflight_meta = None
        return True

    def _rag_lease_coalescing_enabled(self) -> bool:
        return bool(
            getattr(self, "rag_policy_mode", "full") == "full"
            and self._rag_ann_uses_partitioned_gc()
            and (
                getattr(self, "rag_prefill_dynamic_lease", False)
                or
                getattr(self, "rag_lease_coalesce_min_work", 1) > 1
                or getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0) > 0.0
                or getattr(self, "rag_lease_switch_cost_ms", 0.0) > 0.0
            )
        )

    def _rag_lease_work_counts(
        self, *, allow_ann: bool, allow_prefill: bool
    ) -> tuple[int, int]:
        ann_rows = 0
        if allow_ann:
            for task in getattr(self, "ann_search_pending", None) or ():
                queries = task.get("queries")
                shape = getattr(queries, "shape", None)
                ann_rows += int(shape[0]) if shape else 1

        prefill_items = 0
        if allow_prefill:
            split_batch = getattr(self, "split_prefill_batch", None)
            if split_batch is not None and not split_batch.is_empty():
                prefill_items = max(1, int(split_batch.batch_size()))
        return ann_rows, prefill_items

    def _should_wait_for_rag_lease_cohort(
        self,
        *,
        allow_ann: bool,
        allow_prefill: bool,
        active_ann: bool,
        forced_ann: bool,
        ann_wait_ms: float,
        prefill_wait_ms: float,
    ) -> bool:
        frontend_selected = bool(
            allow_prefill
            or (allow_ann and self._rag_ann_uses_frontend_pool())
        )
        if (
            not self._rag_lease_coalescing_enabled()
            or not frontend_selected
            or active_ann
            or forced_ann
            or not self._decode_batch_active()
        ):
            return False

        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        if get_current_stream_idx() != decode_only_idx:
            return False

        max_wait_ms = max(
            0.0, getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0)
        )
        min_work = max(1, getattr(self, "rag_lease_coalesce_min_work", 1))
        if max_wait_ms <= 0.0 or min_work <= 1:
            return False

        ann_rows, prefill_items = self._rag_lease_work_counts(
            allow_ann=allow_ann, allow_prefill=allow_prefill
        )
        frontend_ann_rows = (
            ann_rows if self._rag_ann_uses_frontend_pool() else 0
        )
        if frontend_ann_rows + prefill_items >= min_work:
            return False

        selected_wait_ms = max(
            (
                ann_wait_ms
                if allow_ann and self._rag_ann_uses_frontend_pool()
                else 0.0
            ),
            prefill_wait_ms if allow_prefill else 0.0,
        )
        if selected_wait_ms >= max_wait_ms:
            return False

        ann_urgent = allow_ann and ann_wait_ms >= max(
            0.001, getattr(self, "rag_ann_deadline_ms", 50.0)
        )
        prefill_urgent = allow_prefill and prefill_wait_ms >= max(
            0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
        )
        return not ann_urgent and not prefill_urgent

    def _rag_lease_switch_cost_estimate_ms(
        self,
        current_idx: Optional[int] = None,
        candidate_idx: Optional[int] = None,
        *,
        percentile_override: Optional[float] = None,
    ) -> float:
        configured = max(0.0, getattr(self, "rag_lease_switch_cost_ms", 0.0))
        percentile = min(
            100.0,
            max(
                0.001,
                (
                    getattr(self, "rag_lease_switch_cost_percentile", 95.0)
                    if percentile_override is None
                    else float(percentile_override)
                ),
            ),
        )
        transition_observed = []
        if current_idx is not None and candidate_idx is not None:
            transition_observed = list(
                getattr(
                    self,
                    "rag_partition_transition_pair_latency_ms",
                    {},
                ).get((current_idx, candidate_idx), ())
            )
        if not transition_observed:
            transition_observed = list(
                getattr(self, "rag_partition_transition_latency_ms", ())
            )
        if transition_observed:
            return max(
                configured,
                float(_history_percentile_value(transition_observed, percentile)),
            )

        observed = []
        if current_idx is not None and candidate_idx is not None:
            observed = list(
                getattr(
                    self, "rag_partition_switch_pair_latency_ms", {}
                ).get((current_idx, candidate_idx), ())
            )
        if len(observed) < 4:
            profile_ms = getattr(self, "rag_lease_profile_priors", {}).get(
                "switch_latency_by_pair", {}
            ).get((current_idx, candidate_idx))
            if profile_ms is not None:
                return max(configured, float(profile_ms))
            observed = list(getattr(self, "rag_partition_switch_latency_ms", ()))
        if not observed:
            return configured
        return max(configured, float(_history_percentile_value(observed, percentile)))

    def _track_rag_partition_transition_intent(
        self, current_idx: int, candidate_idx: int
    ) -> None:
        intent = getattr(self, "rag_partition_transition_intent", None)
        if (
            intent is not None
            and intent.get("current_idx") == current_idx
            and intent.get("candidate_idx") == candidate_idx
        ):
            return
        if intent is not None:
            self.rag_partition_transition_abandoned = (
                getattr(self, "rag_partition_transition_abandoned", 0) + 1
            )
        self.rag_partition_transition_intent = {
            "current_idx": int(current_idx),
            "candidate_idx": int(candidate_idx),
            "started_at": time.perf_counter(),
        }

    def _cancel_rag_partition_transition_intent(self) -> None:
        if getattr(self, "rag_partition_transition_intent", None) is not None:
            self.rag_partition_transition_abandoned = (
                getattr(self, "rag_partition_transition_abandoned", 0) + 1
            )
        self.rag_partition_transition_intent = None

    def _complete_rag_partition_transition(
        self,
        current_idx: int,
        candidate_idx: int,
        completed_at: float,
        local_switch_latency_ms: float,
    ) -> float:
        intent = getattr(self, "rag_partition_transition_intent", None)
        transition_latency_ms = float(local_switch_latency_ms)
        if (
            intent is not None
            and intent.get("current_idx") == current_idx
            and intent.get("candidate_idx") == candidate_idx
        ):
            transition_latency_ms = max(
                transition_latency_ms,
                (float(completed_at) - float(intent["started_at"]))
                * 1000.0,
            )
        elif intent is not None:
            self.rag_partition_transition_abandoned = (
                getattr(self, "rag_partition_transition_abandoned", 0) + 1
            )

        history = getattr(
            self, "rag_partition_transition_latency_ms", None
        )
        if history is not None:
            history.append(transition_latency_ms)
        pair_histories = getattr(
            self, "rag_partition_transition_pair_latency_ms", None
        )
        if pair_histories is not None:
            pair_histories.setdefault(
                (current_idx, candidate_idx), deque(maxlen=256)
            ).append(transition_latency_ms)
        self.rag_partition_transition_intent = None
        return transition_latency_ms

    def _rag_lease_transition_cost_estimate_ms(
        self, current_idx: int, candidate_idx: int
    ) -> float:
        enter_ms = self._rag_lease_switch_cost_estimate_ms(
            current_idx, candidate_idx
        )
        if not getattr(
            self, "rag_lease_round_trip_switch_cost", False
        ):
            return enter_ms
        leave_ms = self._rag_lease_switch_cost_estimate_ms(
            candidate_idx, current_idx
        )
        return enter_ms + leave_ms

    def _rag_lease_ann_service_ms(self) -> float:
        observed = list(getattr(self, "rag_ann_gpu_latency_ms", ()))
        if observed:
            return max(0.001, float(np.median(observed)))
        profile_values = list(
            getattr(self, "rag_lease_profile_priors", {})
            .get("ann_gpu_service_by_batch_size", {})
            .values()
        )
        if profile_values:
            return max(0.001, float(np.median(profile_values)))
        return max(
            0.001, getattr(self, "rag_lease_ann_service_estimate_ms", 10.0)
        )

    def _estimate_rag_lease_switch_benefit_ms(
        self,
        *,
        current_idx: int,
        candidate_idx: int,
        allow_ann: bool,
        allow_prefill: bool,
        predicted_decode_ms: Optional[float],
        ann_wait_ms: float = 0.0,
        prefill_wait_ms: float = 0.0,
    ) -> float:
        current_frontend, current_decode = self.sm_counts[current_idx]
        candidate_frontend, candidate_decode = self.sm_counts[candidate_idx]
        min_work = max(1, getattr(self, "rag_lease_coalesce_min_work", 1))
        ann_rows, prefill_items = self._rag_lease_work_counts(
            allow_ann=allow_ann, allow_prefill=allow_prefill
        )

        benefit_ms = 0.0
        selected_wait_ms = max(
            (
                ann_wait_ms
                if allow_ann and self._rag_ann_uses_frontend_pool()
                else 0.0
            ),
            prefill_wait_ms if allow_prefill else 0.0,
        )
        coalesce_wait_ms = max(
            0.0, getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0)
        )
        benefit_ms += max(0.0, coalesce_wait_ms - selected_wait_ms)
        if candidate_frontend > 0:
            frontend_ratio_gain = (
                1.0
                if current_frontend <= 0
                else 1.0
                - (float(current_frontend) / float(candidate_frontend))
            )
            if allow_ann and self._rag_ann_uses_frontend_pool():
                ann_scale = max(1, min(ann_rows, min_work))
                benefit_ms += (
                    self._rag_lease_ann_service_ms()
                    * ann_scale
                    * frontend_ratio_gain
                )
            if allow_prefill:
                prefill_scale = max(1, min(prefill_items, min_work))
                if getattr(self, "rag_prefill_dynamic_lease", False):
                    layer_count, layer_ms, e2e_slack_ms = (
                        self._rag_prefill_lease_shape(
                            candidate_idx, predicted_decode_ms
                        )
                    )
                    candidate_prefill_ms = (
                        max(0.001, layer_ms) * max(1, layer_count)
                        if layer_ms is not None and layer_count is not None
                        else max(
                            0.001,
                            getattr(
                                self,
                                "rag_lease_prefill_service_estimate_ms",
                                10.0,
                            ),
                        )
                    )
                    split_batch = getattr(self, "split_prefill_batch", None)
                    split_index = int(getattr(split_batch, "split_index", 0))
                    num_layers = max(
                        1,
                        int(
                            getattr(
                                getattr(self, "model_config", None),
                                "num_hidden_layers",
                                layer_count or 1,
                            )
                        ),
                    )
                    remaining_layers = max(1, num_layers - split_index)
                    horizon_bundles = max(
                        1,
                        int(
                            np.ceil(
                                remaining_layers / max(1, int(layer_count or 1))
                            )
                        ),
                    )
                    horizon_bundles = min(
                        horizon_bundles,
                        max(
                            1,
                            int(
                                getattr(
                                    self,
                                    "rag_prefill_max_consecutive_chunks",
                                    horizon_bundles,
                                )
                            ),
                        ),
                    )
                    if current_frontend > 0:
                        horizon_bundles = 1
                    self.rag_lease_last_horizon_bundles = horizon_bundles
                    self._increment_histogram(
                        self.rag_lease_horizon_bundle_histogram,
                        horizon_bundles,
                    )
                    self._increment_histogram(
                        self.rag_lease_horizon_layer_histogram,
                        min(
                            remaining_layers,
                            horizon_bundles * max(1, int(layer_count or 1)),
                        ),
                    )
                    benefit_ms += (
                        candidate_prefill_ms
                        * prefill_scale
                        * frontend_ratio_gain
                        * horizon_bundles
                    )
                    if e2e_slack_ms is not None:
                        benefit_ms += max(
                            0.0, candidate_prefill_ms - e2e_slack_ms
                        )
                else:
                    prefill_ms = max(
                        0.001,
                        getattr(
                            self,
                            "rag_lease_prefill_service_estimate_ms",
                            10.0,
                        ),
                    )
                    benefit_ms += (
                        prefill_ms * prefill_scale * frontend_ratio_gain
                    )

        current_decode_ms = self._rag_ann_decode_prediction_ms(
            current_idx, include_scheduling=True
        )
        if current_decode_ms is not None and predicted_decode_ms is not None:
            benefit_ms += current_decode_ms - predicted_decode_ms
        elif (
            predicted_decode_ms is not None
            and current_decode > 0
            and candidate_decode > 0
        ):
            current_decode_estimate_ms = (
                predicted_decode_ms
                * float(candidate_decode)
                / float(current_decode)
            )
            benefit_ms += current_decode_estimate_ms - predicted_decode_ms
        return float(benefit_ms)

    def _coalesce_rag_mixed_partition_target(
        self,
        *,
        candidate_idx: int,
        target_reason: str,
        allow_ann: bool,
        allow_prefill: bool,
        forced_ann: bool,
        ann_wait_ms: float,
        prefill_wait_ms: float,
        predicted_decode_ms: Optional[float],
        bypass_work_wait: bool = False,
    ) -> tuple[int, str]:
        current_idx = get_current_stream_idx()
        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        current_is_lease_compatible = (
            current_idx == decode_only_idx
            or 1 <= current_idx <= last_mixed_idx
        )
        if (
            not self._rag_lease_coalescing_enabled()
            or candidate_idx == current_idx
            or (
                forced_ann
                and not getattr(self, "rag_prefill_dynamic_lease", False)
            )
            or not current_is_lease_compatible
            or not 1 <= candidate_idx <= last_mixed_idx
        ):
            return candidate_idx, target_reason

        if (
            getattr(self, "rag_prefill_dynamic_lease", False)
            and self.sm_counts[candidate_idx][1]
            > self.sm_counts[current_idx][1]
        ):
            # Returning Decode capacity is lease reclamation. It should not be
            # vetoed by the gain test that governs borrowing more Frontend SMs.
            self.rag_lease_decode_restore_transitions = (
                getattr(self, "rag_lease_decode_restore_transitions", 0) + 1
            )
            return candidate_idx, f"{target_reason}_decode_restore"

        ann_urgent = allow_ann and ann_wait_ms >= max(
            0.001, getattr(self, "rag_ann_deadline_ms", 50.0)
        )
        prefill_urgent = allow_prefill and prefill_wait_ms >= max(
            0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
        )
        # Deadlines are liveness bounds, including for dynamic Prefill leases.
        # Once dependency work is overdue, candidate selection has already
        # chosen the least-risk mixed partition; the economic cost gate must
        # not keep that work queued indefinitely.
        if ann_urgent or prefill_urgent:
            return candidate_idx, target_reason

        min_work = max(1, getattr(self, "rag_lease_coalesce_min_work", 1))
        max_wait_ms = max(
            0.0, getattr(self, "rag_lease_coalesce_max_wait_ms", 0.0)
        )
        ann_rows, prefill_items = self._rag_lease_work_counts(
            allow_ann=allow_ann, allow_prefill=allow_prefill
        )
        frontend_ann_rows = (
            ann_rows if self._rag_ann_uses_frontend_pool() else 0
        )
        selected_wait_ms = max(
            (
                ann_wait_ms
                if allow_ann and self._rag_ann_uses_frontend_pool()
                else 0.0
            ),
            prefill_wait_ms if allow_prefill else 0.0,
        )
        if (
            not bypass_work_wait
            and min_work > 1
            and frontend_ann_rows + prefill_items < min_work
            and max_wait_ms > 0.0
            and selected_wait_ms < max_wait_ms
        ):
            self.rag_lease_coalesce_work_holds += 1
            return current_idx, f"{target_reason}_coalesced_work"

        # The max-wait bound is also a liveness bound for the switch-cost gate.
        if (
            not bypass_work_wait
            and max_wait_ms > 0.0
            and selected_wait_ms >= max_wait_ms
        ):
            return candidate_idx, target_reason

        switch_cost_ms = self._rag_lease_transition_cost_estimate_ms(
            current_idx, candidate_idx
        )
        if (
            switch_cost_ms <= 0.0
            and not getattr(self, "rag_prefill_dynamic_lease", False)
        ):
            return candidate_idx, target_reason

        benefit_ms = self._estimate_rag_lease_switch_benefit_ms(
            current_idx=current_idx,
            candidate_idx=candidate_idx,
            allow_ann=allow_ann,
            allow_prefill=allow_prefill,
            predicted_decode_ms=predicted_decode_ms,
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
        )
        current_mixed_can_run_prefill = False
        if (
            bypass_work_wait
            and allow_prefill
            and 1 <= current_idx <= last_mixed_idx
            and self.sm_counts[current_idx][0] > 0
        ):
            current_threshold_ms = self._safe_mixed_hold_threshold_ms()
            current_service_ms = self._rag_ann_decode_prediction_ms(
                current_idx,
                include_scheduling=False,
                percentile=95.0,
                overlap_class=(
                    "ann_prefill"
                    if self._rag_ann_gpu_work_active()
                    else "prefill"
                ),
            )
            current_mixed_can_run_prefill = bool(
                current_threshold_ms is not None
                and current_service_ms is not None
                and current_service_ms <= current_threshold_ms
            )
            if current_mixed_can_run_prefill:
                self.rag_prefill_fast_path_resident_cost_checks = (
                    getattr(
                        self,
                        "rag_prefill_fast_path_resident_cost_checks",
                        0,
                    )
                    + 1
                )
        if (
            bypass_work_wait
            and allow_prefill
            and not current_mixed_can_run_prefill
        ):
            # Without the dependency fast path, an unsafe candidate is held
            # until the Prefill deadline. Charge that avoidable critical-path
            # wait as switch benefit while still honoring the measured cost.
            # A service-safe mixed partition can execute Prefill immediately,
            # so moving to another mixed partition receives no liveness bonus.
            benefit_ms += max(
                0.0,
                getattr(self, "rag_prefill_deadline_ms", 500.0)
                - prefill_wait_ms,
            )
        self.rag_lease_switch_benefit_ms.append(benefit_ms)
        required_benefit_ms = switch_cost_ms * max(
            1.0, getattr(self, "rag_lease_min_gain_ratio", 1.0)
        )
        self.rag_lease_last_switch_benefit_ms = benefit_ms
        self.rag_lease_last_transition_cost_ms = switch_cost_ms
        if benefit_ms <= required_benefit_ms:
            self.rag_lease_coalesce_cost_holds += 1
            return current_idx, f"{target_reason}_coalesced_cost"
        return candidate_idx, target_reason

    def _forecast_rag_joint_decode_kv(
        self, *, frontiers, reference_profiles, frozen_decode_profiles,
        partition: int, overlap_class: str, horizon_ms: float,
        initial_wait_ms: float, inflight_iterations: int = 0,
    ):
        """Bind proposal-specific frozen iteration data to active KV forecast.

        Caller supplies one decision snapshot and its frozen profile domain.
        Batch shrink queries those same histories; context is not bucketed in
        this profile version. Missing shapes remain unsupported, not fallback
        to the old global/decode-only estimator. Does not include new prefill
        cohorts or choose a compute lease.
        """
        allocator = getattr(self, 'token_to_kv_pool_allocator', None)
        page_size = getattr(allocator, 'page_size', None)
        query_capacity = getattr(frozen_decode_profiles, 'query_capacity_ms', None)
        if query_capacity is None:
            raise ValueError('Dedicated capacity profile required; latency-tail profile is not a growth bound')
        return forecast_active_decode_kv_growth(
            frontiers, reference_profiles,
            lambda shape: query_capacity(
                partition, overlap_class, len(shape.request_ids)),
            horizon_ms=horizon_ms, initial_wait_ms=initial_wait_ms,
            page_size=page_size, inflight_iterations=inflight_iterations)

    def _plan_rag_joint_kv_demand(self, demand, *, source_execution,
                                source_stream_idx, current_event_id, window_started_at):
        """KV leg of the once-only HBM proposal; never commit/activate here.

        The complete memory planner must additionally fit ANN scratch/index
        residency. A grow/wait plan is not ready backing and must not publish
        rag_joint_active_event_id. Rejected/stale observations do not replan.
        """
        if not isinstance(demand, JointMemoryDemand):
            raise ValueError('Derived joint memory demand required')
        if type(current_event_id) is not int or current_event_id < 0:
            raise ValueError('Actual nonnegative decision event required')
        if not math.isfinite(window_started_at):
            raise ValueError('Finite common decision clock required')
        if (source_stream_idx != get_current_stream_idx()
                or source_execution != capture_execution_stamp(self)):
            return None
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None:
            return None
        return controller.plan_kv_capacity(demand.required_kv_tokens,
            event_id=current_event_id, now=window_started_at)

    def _derive_rag_selected_memory_demand(self, *, proposed_compute, diagnostic,
                                          queued_budget, decode_growth_tokens,
                                          reserve_tokens, source_execution,
                                          conservative_cache_reserve=False):
        """Connect the selected compute cohort to demand without selecting again.

        decode_growth_tokens must be the fixed-proposal forecast for this
        partition/window, not a configured output cap or future-arrival trace.
        Prepared prefill is already allocator-owned; queued input is not.
        This first adapter supports the actual page-size-one serving contract.
        """
        if type(conservative_cache_reserve) is not bool:
            raise ValueError('Explicit conservative cache-reserve policy required')
        if source_execution != capture_execution_stamp(self):
            return None
        allocator = getattr(self, 'token_to_kv_pool_allocator', None)
        if allocator is None or getattr(allocator, 'page_size', None) != 1:
            return None
        published = allocator.current_mapped_capacity()
        free = allocator.available_size()
        if (type(published) is not int or type(free) is not int
                or not 0 <= free <= published):
            raise ValueError('Consistent actual allocator capacity required')
        allocated = published - free
        cohort = diagnostic.get('queued_cohort_demand') if diagnostic else None
        batch = getattr(self, 'split_prefill_batch', None)
        if cohort is not None:
            if batch is not None or queued_budget is None:
                return None
            # A changed allocator observation invalidates this cohort snapshot;
            # do not silently combine old cache locks with new capacity values.
            if (queued_budget.published_capacity_tokens != published
                    or queued_budget.allocated_kv_tokens != allocated):
                return None
            demand = derive_cached_prefill_memory_demand(proposed_compute, queued_budget,
                cohort, decode_growth_tokens=decode_growth_tokens, reserve_tokens=reserve_tokens,
                guaranteed_available_tokens=free if conservative_cache_reserve else None)
        else:
            if proposed_compute.prefill_layers:
                if batch is None or capture_prepared_prefill_contract(batch) != proposed_compute.prefill_contract:
                    return None
            demand = derive_joint_memory_demand(proposed_compute, allocated_kv_tokens=allocated,
                new_prefill_tokens=(), decode_growth_tokens=decode_growth_tokens,
                reserve_tokens=reserve_tokens)
        if source_execution != capture_execution_stamp(self):
            return None
        return demand

    def _plan_rag_joint_hbm_demand(self, demand, *, source_execution,
                                 source_stream_idx, current_event_id, window_started_at,
                                 max_inflight=1, continuation=False):
        """Composite memory bridge shared by Full and Independent-adaptive.

        For continuation, demand must describe current-allocation work, not the
        rejected target. This does not derive it or run compute selection.
        Neither branch commits, polls, activates, or retries on stale state.
        """
        if not isinstance(demand, JointMemoryDemand):
            raise ValueError('Derived joint memory demand required')
        if type(current_event_id) is not int or current_event_id < 0:
            raise ValueError('Actual nonnegative decision event required')
        if type(max_inflight) is not int or max_inflight <= 0 or type(continuation) is not bool:
            raise ValueError('Explicit continuation mode and positive ANN pipeline limit required')
        if not math.isfinite(window_started_at):
            raise ValueError('Finite common decision clock required')
        if (source_stream_idx != get_current_stream_idx()
                or source_execution != capture_execution_stamp(self)):
            return None
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None:
            return None
        if continuation:
            return controller.current_joint_hbm_continuation(demand.required_kv_tokens,
                demand.proposed_ann_rows,event_id=current_event_id,max_inflight=max_inflight)
        return controller.plan_joint_hbm(demand.required_kv_tokens,demand.proposed_ann_rows,
            event_id=current_event_id,now=window_started_at,max_inflight=max_inflight)

    def _build_rag_joint_pending_retrieval_graph(self, *, index_preview=None, cpu_plan,
                                                frozen_cpu_profile, frozen_probe_profile,
                                                frozen_collection_profile, partition,
                                                max_batch_queries, observation_offset_ms,
                                                admit_offset_ms, compute_ready_ms,
                                                scratch_key, index_key, collection_mode,
                                                topk, prefix, timing=True, memory_lease=None):
        """Compose one fitted pending batch; return new ops and its request IDs.

        The shared original CPU operations remain owned by the outer graph.
        No search, submission, profile mutation or future-arrival access occurs.
        timing=False builds the identical dependency/backing graph without
        touching any timing profile. Full can fill durations later under the
        same captured offsets; Independent uses this shared safety skeleton.
        """
        if memory_lease is not None:
            captured = memory_lease.routing_preview
            if captured is None or not memory_lease.supported:
                return None
            if index_preview is not None and index_preview is not captured:
                raise ValueError('Use the selected memory lease routing observation')
            if captured[2] != memory_lease.ann_batch_cap:
                return None
            index_preview = captured
        if index_preview is None or cpu_plan is None:
            return None
        if not index_preview[1]:
            return (), ()
        gpu = self._build_rag_joint_gpu_prediction(index_preview=index_preview,
            frozen_probe_profile=frozen_probe_profile, partition=partition,
            operation_key=f'{prefix}:gpu', scratch_key=scratch_key, index_key=index_key,
            observation_offset_ms=observation_offset_ms, compute_ready_ms=compute_ready_ms, timing=timing)
        if gpu is None:
            return None
        spill_prefix = f'{prefix}:spill'
        spill = self._build_rag_joint_spill_prediction(index_preview=index_preview,
            frozen_cpu_profile=frozen_cpu_profile, preceding_cpu_operations=cpu_plan[0],
            max_batch_queries=max_batch_queries, admit_offset_ms=admit_offset_ms,
            operation_prefix=spill_prefix, timing=timing)
        if spill is None:
            return None
        # New spill is submitted by _commit_rag_pending_ann_spill only after
        # the selected physical HBM plan is ready. Existing CPU searches were
        # submitted earlier and must not acquire these new-lease gates.
        spill = (tuple(replace(op, memory_keys=tuple(dict.fromkeys(
            (*op.memory_keys, scratch_key, index_key)))) for op in spill[0]), spill[1])
        collect_ms = self._predict_rag_joint_collection_ms(index_preview=index_preview,
            frozen_collection_profile=frozen_collection_profile, collection_mode=collection_mode,
            topk=topk) if timing else None
        routes = index_preview[1]
        collection_shape = ('ann_collect', collection_mode, index_preview[2], topk,
            sum(len(prepared.cold_searches) + int(residual.cold_width > 0)
                for _, prepared, residual in routes),
            any(any(int(n) > 0 for n in residual.hot_counts) for _, _, residual in routes))
        return self._join_rag_joint_pending_retrieval(index_preview=index_preview, cpu_plan=cpu_plan,
            gpu_operations=gpu, spill_plan=spill, collection_ms=collect_ms,
            collection_key=f'{prefix}:collect', spill_prefix=spill_prefix, timing=timing,
            collection_shape=collection_shape,
            collection_ready_ms=max(observation_offset_ms, cpu_plan[2], admit_offset_ms),
            collection_memory_keys=(scratch_key, index_key))

    def _predict_rag_joint_collection_ms(self, *, index_preview, frozen_collection_profile,
                                         collection_mode, topk):
        """Shape collect after the actual combined old-cold + spill branches.

        Finished old CPU futures still participate in result merging, so count
        them here even though they have no remaining CPU-search operation.
        """
        if index_preview is None:
            return None
        _, routes, rows = index_preview
        if not routes:
            return None
        searches = 0
        has_gpu = False
        for task, prepared, residual in routes:
            searches += len(prepared.cold_searches)
            if residual.cold_width > 0:
                searches += 1  # _submit_prepared_cold_search submits one future.
            has_gpu = has_gpu or any(int(n) > 0 for n in residual.hot_counts)
        return frozen_collection_profile.query_ms(collection_mode, rows, topk, searches, has_gpu)

    def _join_rag_joint_pending_retrieval(self, *, index_preview, cpu_plan,
                                         gpu_operations, spill_plan, collection_ms,
                                         collection_key, spill_prefix, timing=True, collection_shape=None,
                                         collection_ready_ms=0., collection_memory_keys=()):
        """Join one actually combined pending batch, returning only NEW ops.

        The original CPU graph belongs once to the global prediction. Even an
        empty original dependency requires an observed binding entry: missing
        state is not proof that its CPU branch completed. The supplied collect
        duration excludes branch wait already represented by these edges.
        """
        if index_preview is None or cpu_plan is None or gpu_operations is None or spill_plan is None:
            return None
        _, routes, _ = index_preview
        if not routes:
            return (), ()
        if collection_ms is None and timing:
            return None
        if collection_ms is not None and (not math.isfinite(collection_ms) or collection_ms <= 0):
            raise ValueError('Measured positive result collection service required')
        if not math.isfinite(collection_ready_ms) or collection_ready_ms < 0:
            raise ValueError('Captured collection admission offset required')
        old_ops, dependency_entries, _ = cpu_plan
        dependencies = dict(dependency_entries)
        if len(dependencies) != len(dependency_entries):
            raise ValueError('Duplicate observed CPU batch identities')
        spill_ops, spill_dependencies = spill_plan
        ids = []
        branch_keys = [op.key for op in gpu_operations]
        for position, (task, prepared, residual) in enumerate(routes):
            rid = str(task['request_id'])
            ids.append(rid)
            keys = dependencies.get((rid,))
            if keys is None:
                return None
            branch_keys.extend(keys)
            if residual.cold_width:
                spill_key = spill_dependencies.get(f'{spill_prefix}:task:{position}')
                if spill_key is None:
                    return None
                branch_keys.append(spill_key)
        shared = (*old_ops, *spill_ops, *gpu_operations)
        collect = retrieval_collection_after_shared_work(tuple(ids), branch_keys=branch_keys,
            shared_operations=shared,
            collect=Operation(collection_key, 'collect', collection_ms, ('ann_collect',),
                              service_shape=collection_shape, ready_ms=collection_ready_ms,
                              memory_keys=tuple(collection_memory_keys)))
        return (*spill_ops, *gpu_operations, collect), tuple(ids)

    def _build_rag_joint_gpu_prediction(self, *, index_preview, frozen_probe_profile,
                                       partition, operation_key, scratch_key, index_key,
                                       observation_offset_ms, compute_ready_ms=0., timing=True):
        """One actual combined ANN submission, not one launch per source task.

        Residual ordered hot counts describe the target resident work only.
        CPU spill/original cold completion stays on separate dependencies.
        No logical query credit is assigned before the result collection.
        """
        if index_preview is None:
            return None
        _, routes, admitted_rows = index_preview
        if not routes:
            return ()
        counts = []
        linear_work = [0, 0, 0, 0, 0]
        # Class-level capability detection keeps Independent's timing-free
        # safety projection from touching a profile instance (whose methods
        # are deliberately forbidden there).
        linear_query = getattr(type(frozen_probe_profile), 'query_work_ms', None)
        for task, prepared, residual in routes:
            rows = self._ann_task_num_queries(task)
            if len(residual.hot_counts) != rows:
                raise ValueError('GPU route rows differ from combined submission')
            counts.extend(int(n) for n in residual.hot_counts)
            if callable(linear_query):
                work=self.ann_executor.describe_gpu_routed_work(residual)
                if work is None:
                    return None
                linear_work[0]+=work[0]
                linear_work[1]+=work[1]
                linear_work[2]=max(linear_work[2],work[2])
                linear_work[3]=max(linear_work[3],work[3])
                linear_work[4]+=work[4]
        if len(counts) != admitted_rows:
            raise ValueError('Admitted row total differs from GPU route')
        if not any(counts):
            return ()  # Actual CPU-only search has no GPU operation.
        if callable(linear_query):
            if not all(value>0 for value in linear_work):
                raise ValueError('Nonempty GPU route requires positive ANN work features')
            duration=(linear_query(frozen_probe_profile,partition,admitted_rows,*linear_work)
                      if timing else None)
            service_shape=('ann_gpu_linear',partition,admitted_rows,*linear_work)
        else:
            duration = frozen_probe_profile.query_ms(partition, tuple(counts)) if timing else None
            service_shape=('ann_gpu', partition, tuple(counts))
        if timing and duration is None:
            return None
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError('Unfinished GPU search requires positive profiled service')
        if (not scratch_key or not index_key or scratch_key == index_key
                or not math.isfinite(observation_offset_ms) or observation_offset_ms < 0
                or not math.isfinite(compute_ready_ms) or compute_ready_ms < 0):
            raise ValueError('Distinct backing keys and common-window offsets required')
        return (Operation(operation_key, 'ann', duration, ('ann',),
            ready_ms=observation_offset_ms, compute_ready_ms=compute_ready_ms,
            memory_keys=(scratch_key, index_key),
            service_shape=service_shape),)

    def _build_rag_joint_spill_prediction(self, *, index_preview, frozen_cpu_profile,
                                         preceding_cpu_operations, max_batch_queries,
                                         admit_offset_ms, operation_prefix, timing=True):
        """Bind previewed additional probes to the existing CPU worker model.

        Actual submission passes ALL query rows with a padded cold width, even
        if some rows have no cold probes. Do not compress nonempty rows here.
        Original cold futures are already covered by preceding observations.
        """
        if index_preview is None:
            return None
        _, routes, _ = index_preview
        entries = []
        for position, (task, prepared, residual) in enumerate(routes):
            rows = self._ann_task_num_queries(task)
            if len(residual.cold_counts) != rows:
                raise ValueError('Spill row shape differs from actual submission')
            width = residual.cold_width
            if width:
                entries.append((f'{operation_prefix}:task:{position}', rows, width))
        return build_additional_cold_operations(entries, frozen_cpu_profile.query_remaining_ms if timing else None,
            max_batch_queries=max_batch_queries, admit_offset_ms=admit_offset_ms,
            preceding_operations=preceding_cpu_operations, prefix=operation_prefix, timing=timing)

    def _preview_rag_joint_index_routes(self, *, memory_plan, ann_tasks, source_execution):
        """Plan-specific retained lists and additional cold probes, read-only.

        Return strong prepared owners with residual routes. Existing cold
        futures are modeled separately; do not count residual spill as a full
        retrieval rerun. Pending reload/reclaim is unsupported, not resident.
        This preview is not a physical activation certificate.
        """
        if source_execution != capture_execution_stamp(self):
            return None
        controller = getattr(self, 'shared_hbm_controller', None)
        executor = getattr(self, 'ann_executor', None)
        if controller is None or executor is None or memory_plan.ann is None:
            return None
        # This existing read-only guard checks pending transitions before a
        # native snapshot and verifies this plan's source ownership identity.
        if controller.joint_transition_profile_inputs(memory_plan) is None:
            return None
        residency = executor.preview_shared_ivf_residency(
            memory_plan.ann.retention.retained_index_bytes)
        if residency is None:
            return None
        rows = 0
        routes = []
        seen = set()
        for task in ann_tasks:
            count = self._ann_task_num_queries(task)
            if type(count) is not int or count <= 0:
                raise ValueError('Actual positive pending query count required')
            if rows + count > memory_plan.ann.fit.fitted_rows:
                break
            if id(task) in seen:
                raise ValueError('Duplicate pending ANN task')
            seen.add(id(task))
            if task.get('result') is not None or task.get('launch_time') is not None:
                return None
            future = task.get('route_future')
            if (future is None or not future.done() or future.cancelled()
                    or future.exception() is not None):
                return None
            prepared = future.result()  # terminal route metadata, not CPU search join
            residual = executor.preview_prepared_hot_spill(prepared, residency)
            routes.append((task, prepared, residual))
            rows += count
        if source_execution != capture_execution_stamp(self):
            return None
        return residency, tuple(routes), rows

    def _describe_rag_joint_backing(self, *, memory_plan, ann_tasks, source_execution,
                                    current_event_id, window_started_at, prefill_layers,
                                    existing_ready=(), kv_key='joint:kv',
                                    scratch_key='joint:scratch', index_key='joint:index'):
        """Capture one plan's ownership and retained routing without timing.

        Returns (materializer kwargs, index_preview). The same preview MUST
        feed the operation graph and later selected-lease route validation.
        Base offsets indicate when this observation was available, not that
        grown KV/workspace is executable. Whole-lease activation gates are
        added by the materializer, and native commit checks remain mandatory.
        Already-backed compatible work retains separate caller-captured keys.
        """
        if (source_execution != capture_execution_stamp(self)
                or memory_plan.event_id != current_event_id):
            return None
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None:
            return None
        shape = controller.joint_allocation_shape(memory_plan)
        if shape is None:
            return None
        preview = self._preview_rag_joint_index_routes(memory_plan=memory_plan,
            ann_tasks=ann_tasks, source_execution=source_execution)
        if preview is None:
            return None
        residency, routes, rows = preview
        if (residency[0] != shape[5] or rows != memory_plan.ann.fit.fitted_rows
                or source_execution != capture_execution_stamp(self)):
            return None
        keys = (kv_key, scratch_key, index_key)
        if len(set(keys)) != 3 or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError('Distinct nonempty new-work backing keys required')
        ready = tuple(existing_ready)
        if len(dict(ready)) != len(ready) or set(dict(ready)).intersection(keys):
            raise ValueError('Existing work requires separate unique backing keys')
        if any(not math.isfinite(v) or v < 0 for _, v in ready):
            raise ValueError('Finite captured existing-work readiness required')
        offset = (time.perf_counter() - window_started_at) * 1000.0
        if not math.isfinite(offset) or offset < 0:
            raise ValueError('Common decision window required')
        # Residency matters even when index byte counts match. This identity
        # excludes event/work limits so a safe continuation has no fake switch.
        mask_hash = hashlib.sha256(bytes(residency[1])).hexdigest()
        allocation = 'joint-hbm:' + repr((*shape, mask_hash))
        new_keys = ((kv_key,) if prefill_layers else ())
        if rows:
            new_keys += (scratch_key, index_key)
        backing = dict(allocation=allocation, ready=ready + tuple((k, offset) for k in keys),
            kv_key=kv_key, scratch_key=scratch_key, index_key=index_key,
            new_work_keys=new_keys)
        return backing, preview

    def _plan_rag_target_memory_lease(self, *, proposed_compute, diagnostic, queued_budget,
                                     decode_growth_tokens, reserve_tokens, source_execution,
                                     source_stream_idx, target_stream_idx, current_event_id,
                                     window_started_at, mode, describe_backing=None, backing_inputs=None,
                                     transition_profiles=None, release_profile=None,
                                     max_inflight=1, derived_demand=None):
        """One target memory path, shared by Full and Independent-adaptive.

        describe_backing(plan) is a read-only routing/ownership adapter returning
        allocation, ready, kv_key, scratch_key, index_key, new_work_keys. It MUST
        derive these from the fitted plan and actual list/release observations,
        not mark planned reloads resident or perform another fit/commit.
        None means unsupported/stale, never a request to retry another proposal.
        """
        if mode not in ('full', 'independent_adaptive'):
            raise ValueError('Explicit joint policy required')
        if describe_backing is not None and backing_inputs is not None:
            raise ValueError('Choose one backing adapter, not two observations')
        if describe_backing is None and backing_inputs is None:
            raise ValueError('Plan-specific read-only backing adapter required')
        if describe_backing is not None and not callable(describe_backing):
            raise ValueError('Callable backing adapter required')
        if source_execution != capture_execution_stamp(self):
            return None
        demand = derived_demand
        if demand is None:
            demand = self._derive_rag_selected_memory_demand(proposed_compute=proposed_compute,
                diagnostic=diagnostic, queued_budget=queued_budget,
                decode_growth_tokens=decode_growth_tokens, reserve_tokens=reserve_tokens,
                source_execution=source_execution)
        elif not isinstance(demand, JointMemoryDemand) or demand.compute != proposed_compute:
            raise ValueError('Derived demand must belong to the single selected compute proposal')
        if demand is None:
            return None
        plan = self._plan_rag_joint_hbm_demand(demand, source_execution=source_execution,
            source_stream_idx=source_stream_idx, current_event_id=current_event_id,
            window_started_at=window_started_at, max_inflight=max_inflight, continuation=False)
        if plan is None:
            return None
        # Preserve rejection without invoking routing or timing on an unusable
        # native plan. Caller turns this into unsupported target + safe keep.
        if plan.status != 'available' or plan.ann is None:
            return None
        routing_preview = None
        if backing_inputs is not None:
            described = self._describe_rag_joint_backing(memory_plan=plan,
                source_execution=source_execution, current_event_id=current_event_id,
                window_started_at=window_started_at, prefill_layers=proposed_compute.prefill_layers,
                **backing_inputs)
            if described is None:
                return None
            backing, routing_preview = described
        else:
            backing = describe_backing(plan)
        if backing is None or source_execution != capture_execution_stamp(self):
            return None
        completion = None
        if mode == 'full':
            completion = self._estimate_rag_joint_transition_completion(memory_plan=plan,
                profiles=transition_profiles, target_stream_idx=target_stream_idx,
                window_started_at=window_started_at, source_execution=source_execution,
                release_profile=release_profile)
        memory = self._materialize_rag_joint_memory_lease(proposed_compute=proposed_compute,
            memory_plan=plan, current_event_id=current_event_id,
            source_execution=source_execution, completion=completion, **backing)
        return replace(memory, routing_preview=routing_preview) if memory is not None else None

    def _materialize_rag_joint_memory_lease(self, *, proposed_compute, memory_plan,
                                           allocation, current_event_id, source_execution,
                                           ready, kv_key, scratch_key, index_key,
                                           new_work_keys, completion=None):
        """Translate the single fitted plan; never plan or query a predictor.

        Full supplies its common-window completion estimate. Independent may
        omit it: missing scoring information is not failed safety projection,
        and native activation still checks actual physical backing. The caller
        supplies guaranteed per-work readiness, not merely target budget bytes.
        Existing compatible decode must have a separate, ungated backing key.
        """
        if source_execution != capture_execution_stamp(self):
            return None
        keys = tuple(new_work_keys)
        required = set()
        if proposed_compute.prefill_layers:
            required.add(kv_key)
        if memory_plan.ann is not None and memory_plan.ann.fit.fitted_rows:
            required.update((scratch_key, index_key))
        if not required.issubset(keys):
            raise ValueError('All newly admitted work must wait for joint activation')
        memory = memory_lease_from_hbm_plan(proposed_compute, memory_plan,
            allocation=allocation, event_id=current_event_id, ready=ready,
            kv_key=kv_key, scratch_key=scratch_key, index_key=index_key,
            kv_transition_ms=getattr(completion, 'kv_ms', None),
            scratch_transition_ms=getattr(completion, 'workspace_ms', None))
        memory = apply_joint_activation_readiness(memory, completion, new_work_keys=keys)
        if source_execution != capture_execution_stamp(self):
            return None
        return memory

    def _predict_rag_ann_quiescent_offset(self, *, release_profile, window_started_at,
                                         source_execution):
        """Predict GPU-reader release without waiting for the CPU result join.

        release_profile must be the caller-frozen HOST-clock ANN release data.
        All launched tasks were already submitted, so their age-conditioned
        release offsets combine with max, not an additional serial sum. This
        is a heuristic tail forecast, never a replacement for native fences.
        """
        if source_execution != capture_execution_stamp(self):
            return None
        inflight = getattr(self,'ann_search_inflight',()) or ()
        if isinstance(inflight,dict):
            inflight = (inflight,)
        active = []
        for task in inflight:
            result = task.get('result')
            poll = getattr(result,'gpu_poll',None)
            if poll is None:
                return None
            if poll():
                continue  # CPU-only pending work is not a GPU-reader fence.
            active.append(task)
        now = time.perf_counter()
        offset = (now-window_started_at)*1000.0
        if not math.isfinite(offset) or offset<0:
            raise ValueError('Common window cannot start after ANN observation')
        finishes = [offset]
        for task in active:
            launched = task.get('launch_time'); partition = task.get('stream_idx')
            if (type(launched) not in (int,float) or not math.isfinite(launched)
                    or launched>now or type(partition) is not int or partition<0):
                return None
            rows = self._ann_task_num_queries(task)
            remaining = release_profile.query_remaining_ms((partition,rows),
                                                           elapsed_ms=(now-launched)*1000.0)
            if remaining is None:
                return None
            finishes.append(offset+remaining)
        if source_execution != capture_execution_stamp(self):
            return None
        return max(finishes)

    def _estimate_rag_joint_transition_completion(self, *, memory_plan, profiles,
                                                  target_stream_idx, window_started_at,
                                                  source_execution, ann_quiescent_ms=None,
                                                  release_profile=None):
        """Read-only plan-to-profile bridge; target and keep reuse profiles.

        A live ANN batch needs an explicitly predicted quiescence offset. No
        idle assumption is made when it is active. Native transition already
        pending is unsupported here rather than timed as a fresh transition.
        """
        if not isinstance(profiles,FrozenJointTransitionProfiles):
            raise ValueError('Common frozen transition profiles required')
        if source_execution != capture_execution_stamp(self):
            return None
        controller = getattr(self,'shared_hbm_controller',None)
        if controller is None:
            return None
        inputs = controller.joint_transition_profile_inputs(memory_plan)
        if inputs is None:
            return None
        offset = (time.perf_counter()-window_started_at)*1000.0
        active = self._rag_ann_gpu_work_active()
        if active and ann_quiescent_ms is None and release_profile is not None:
            ann_quiescent_ms = self._predict_rag_ann_quiescent_offset(
                release_profile=release_profile,window_started_at=window_started_at,
                source_execution=source_execution)
        if source_execution != capture_execution_stamp(self):
            return None
        return profiles.query(source_partition=get_current_stream_idx(),
            target_partition=target_stream_idx,observation_offset_ms=offset,
            ann_quiescent_ms=ann_quiescent_ms if active else offset,**inputs)

    def _prepare_rag_joint_kv_transition(
        self, *, decision, kv_plan, source_execution, source_stream_idx: int,
        current_event_id: int,
    ) -> str:
        """Guard the KV leg of an already admitted joint lease.

        Call under scheduler serialization, before any new launch. The adapter
        must supply the KV plan paired with this selected lease, never the
        rejected target's plan after Full selects continuation. This prepares
        memory only: submitted is NOT ready to dispatch or a compute commit.
        Resource identity is checked again by the controller before mutation.
        No proposal generation, utility evaluation or retry occurs here.
        """
        if (type(current_event_id) is not int or current_event_id < 0
                or decision.event_id != current_event_id
                or kv_plan.event_id != current_event_id):
            return 'stale'
        if not decision.lease.supported:
            return 'unsupported'
        if decision.reason != 'target' and kv_plan.status != 'ready':
            # Full rejected the change: never accidentally commit the target's
            # growth while returning its current-allocation continuation.
            return 'unsupported'
        selected_memory = getattr(decision.lease, 'memory', None)
        selected_plan = getattr(selected_memory, 'resource_plan', None)
        if selected_plan is None or getattr(selected_plan, 'kv', None) is not kv_plan:
            return 'unsupported'  # Same event ID does not identify the selected plan.
        if (source_stream_idx != get_current_stream_idx()
                or source_execution != capture_execution_stamp(self)):
            return 'stale'
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None:
            return 'unsupported'
        if getattr(controller,'_joint_kv_committed_event_id',None) == current_event_id:
            return controller.observe_committed_kv_capacity_plan(
                kv_plan,current_event_id=current_event_id)
        return controller.commit_kv_capacity_plan(kv_plan,
                                                   current_event_id=current_event_id)

    def _prepare_rag_joint_workspace_transition(self, *, decision, source_execution,
                                                source_stream_idx, source_memory_identity,
                                                current_event_id):
        """Submit the selected workspace only, never a rejected target's plan.

        The caller provides a validated post-transition execution/memory view.
        This is not an activation: request/index routing and compute commit
        still need their own checks before publishing the active lease.
        """
        if (type(current_event_id) is not int or current_event_id < 0
                or decision.event_id != current_event_id):
            return 'stale'
        memory = getattr(decision.lease, 'memory', None)
        plan = getattr(memory, 'resource_plan', None)
        if (not decision.lease.supported or plan is None or plan.ann is None
                or plan.status != 'available'):
            return 'unsupported'
        if plan.event_id != current_event_id:
            return 'stale'
        if (source_stream_idx != get_current_stream_idx()
                or source_execution != capture_execution_stamp(self)):
            return 'stale'
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None:
            return 'unsupported'
        if decision.reason != 'target':
            # Continuation keeps current physical allocation. Never use a
            # binding callback, even if its old workspace has become invalid.
            if plan.kv.status != 'ready' or plan.ann.mapping_required:
                return 'unsupported'
            return 'ready' if controller.joint_hbm_backing_ready(plan) else 'wait'
        return controller.commit_joint_ann_workspace(plan,self,
            current_event_id=current_event_id,source_identity=source_memory_identity)

    def _commit_rag_pending_ann_spill(self, *, decision, tasks, residency,
                                      source_execution, current_event_id):
        """Replace pending route futures after admitted index transition.

        Caller supplies the exact cohort used in scoring, not a fresh selection.
        No GPU launch occurs here. Returned futures keep original cold futures
        plus committed spill, so later events do not redo already submitted work.
        """
        from concurrent.futures import Future
        if (decision.event_id != current_event_id or not decision.lease.supported
                or source_execution != capture_execution_stamp(self)):
            return 'stale'
        plan = getattr(decision.lease.memory, 'resource_plan', None)
        controller = getattr(self, 'shared_hbm_controller', None)
        if plan is None or controller is None or plan.event_id != current_event_id:
            return 'unsupported'
        if not controller.joint_hbm_backing_ready(plan):
            return 'wait'
        executor = self.ann_executor
        if executor.preview_shared_ivf_residency() != residency:
            return 'stale'  # Same bytes with different lists is not the selected route.
        pending = getattr(self, 'ann_search_pending', ())
        if len({id(task) for task in tasks}) != len(tasks):
            raise ValueError('Duplicate ANN cohort task')
        prepared = []
        for task in tasks:
            if (not any(task is p for p in pending) or 'result' in task
                    or 'launch_time' in task):
                return 'stale'
            future = task.get('route_future')
            if future is None or not future.done() or future.cancelled():
                return 'wait'
            prepared.append(future.result())  # Already terminal; propagate route failure.
        rows = sum(len(task['queries']) for task in tasks)
        if rows > decision.lease.executable.ann_batch:
            return 'unsupported'
        captured = getattr(decision.lease.memory, 'routing_preview', None)
        if captured is not None:
            captured_residency, captured_routes, captured_rows = captured
            if (residency != captured_residency or rows != captured_rows
                    or rows != decision.lease.executable.ann_batch
                    or len(tasks) != len(captured_routes)
                    or any(task is not route[0] or original is not route[1]
                           for task, original, route in zip(tasks, prepared, captured_routes))):
                return 'stale'
        for task, original in zip(tasks, prepared):
            replacement = executor.commit_prepared_hot_spill(task['queries'],original,
                                                             residency,event_id=current_event_id)
            future = Future()
            future.set_result(replacement)
            task['route_future'] = future
        return 'ready'

    def _prepare_rag_selected_joint_lease(self, *, decision, source_execution,
                                          source_stream_idx, allocation_to_stream,
                                          ann_tasks=None, residency=None, current_event_id):
        """One selected-lease preparation pass; never re-enter the optimizer.

        Must run under scheduler serialization. Native async work is advanced
        by existing completion polling, not by spinning/retrying here. Returned
        status identifies the unfinished leg. A new decision must start from
        actual state; this function cannot reuse a rejected target as keep.
        """
        self.rag_joint_active_event_id = None
        self.rag_joint_active_memory_plan = None
        if (decision.event_id != current_event_id or not decision.lease.supported
                or source_execution != capture_execution_stamp(self)
                or source_stream_idx != get_current_stream_idx()):
            return 'stale'
        native = getattr(decision.lease.memory,'resource_plan',None)
        controller = getattr(self,'shared_hbm_controller',None)
        if native is None or controller is None:
            return 'unsupported'
        captured = getattr(decision.lease.memory, 'routing_preview', None)
        if captured is not None:
            captured_residency, routes, rows = captured
            selected_tasks = tuple(route[0] for route in routes)
            if (rows != decision.lease.executable.ann_batch
                    or sum(len(task['queries']) for task in selected_tasks) != rows
                    or (residency is not None and residency != captured_residency)
                    or (ann_tasks is not None and (len(ann_tasks) != len(selected_tasks)
                        or any(a is not b for a, b in zip(ann_tasks, selected_tasks))))):
                return 'stale'
            # Check original prepared owners BEFORE any KV/workspace mutation.
            # Equal request counts do not make a replacement route equivalent.
            pending = getattr(self, 'ann_search_pending', ())
            for task, original, residual in routes:
                future = task.get('route_future')
                if (not any(task is p for p in pending) or task.get('result') is not None
                        or task.get('launch_time') is not None or future is None
                        or not future.done() or future.cancelled()
                        or future.exception() is not None or future.result() is not original):
                    return 'stale'
            ann_tasks, residency = selected_tasks, captured_residency
        elif ann_tasks is None:
            return 'unsupported'
        self.rag_frontend_plan = frontend_plan_from_joint_decision(decision,allocation_to_stream)
        status = self._prepare_rag_joint_kv_transition(decision=decision,kv_plan=native.kv,
            source_execution=source_execution,source_stream_idx=source_stream_idx,
            current_event_id=current_event_id)
        if status != 'ready':
            return 'kv_' + status
        if self.rag_frontend_plan.target_stream_idx != get_current_stream_idx():
            return 'compute_wait'  # Existing safe partition-switch path owns the switch.
        memory_identity = controller._kv_plan_source_identity(controller.pool.snapshot())
        status = self._prepare_rag_joint_workspace_transition(decision=decision,
            source_execution=source_execution,source_stream_idx=source_stream_idx,
            source_memory_identity=memory_identity,current_event_id=current_event_id)
        if status != 'ready':
            return 'workspace_' + status
        if ann_tasks:
            status = self._commit_rag_pending_ann_spill(decision=decision,tasks=ann_tasks,
                residency=residency,source_execution=source_execution,current_event_id=current_event_id)
            if status != 'ready':
                return 'ann_' + status
        # Own successful spill/materialization writes legitimately change the
        # execution identity. Re-capture only at these explicit commit points.
        after_spill = capture_execution_stamp(self)
        status = self._materialize_rag_joint_prefill(decision=decision,
            source_execution=after_spill,current_event_id=current_event_id)
        if status != 'ready':
            return 'prefill_' + status
        after_prefill = capture_execution_stamp(self)
        memory_identity = controller._kv_plan_source_identity(controller.pool.snapshot())
        return self._activate_rag_joint_lease(decision=decision,
            allocation_to_stream=allocation_to_stream,source_execution=after_prefill,
            source_memory_identity=memory_identity,residency=residency,current_event_id=current_event_id)

    def _materialize_rag_joint_prefill(self, *, decision, source_execution, current_event_id):
        """Construct the admitted cohort after backing, before final activation.

        No compute/HBM search. Existing prepared work is retained; a different
        batch cannot be replaced just because a target prefers another cohort.
        Caller must capture a fresh execution stamp after successful mutation.
        """
        if (decision.event_id != current_event_id or not decision.lease.supported
                or source_execution != capture_execution_stamp(self)):
            return 'stale'
        work = decision.lease.executable
        if work.prefill_layers == 0:
            return 'ready'
        contract = getattr(work,'prefill_contract',None)
        if contract is None:
            return 'unsupported'
        existing = getattr(self,'split_prefill_batch',None)
        if existing is not None:
            return 'ready' if capture_prepared_prefill_contract(existing) == contract else 'wait'
        if (getattr(self,'rag_prefill_bundle_inflight_event',None) is not None
                or getattr(self,'rag_prefill_merge_pending_batch',None) is not None):
            return 'wait'
        controller = getattr(self,'shared_hbm_controller',None)
        plan = getattr(decision.lease.memory,'resource_plan',None)
        if controller is None or plan is None or plan.event_id != current_event_id:
            return 'unsupported'
        if not controller.joint_hbm_backing_ready(plan):
            return 'wait'
        batch = self.get_new_batch_prefill(admitted_prefill_contract=contract)
        if batch is None:
            return 'wait'
        # Preserve the actually allocated batch even if a shape bug is found;
        # do not discard live KV or restore a stale queue snapshot.
        self._install_rag_split_prefill_batch(batch)
        if capture_prepared_prefill_contract(batch) != contract:
            raise RuntimeError('Materialized joint prefill contract mismatch')
        return 'ready'

    def _activate_rag_joint_lease(self, *, decision, allocation_to_stream,
                                  source_execution, source_memory_identity,
                                  residency, current_event_id):
        """Publish dispatch intent only after the selected transitions finish.

        Caller owns serialization, admitted-cohort materialization and safe
        compute switching. Capture source_execution AFTER own spill/prefill
        commits, never reuse the pre-transition stamp. No search, allocation,
        prefill construction, waiting, or GPU launch occurs in this method.
        """
        self.rag_joint_active_event_id = None
        self.rag_joint_active_memory_plan = None
        if (type(current_event_id) is not int or current_event_id < 0
                or decision.event_id != current_event_id or not decision.lease.supported
                or source_execution != capture_execution_stamp(self)):
            return 'stale'
        frontend = frontend_plan_from_joint_decision(decision,allocation_to_stream)
        if get_current_stream_idx() != frontend.target_stream_idx:
            return 'wait'
        if getattr(self,'rag_ann_partition_switch_deferred',False):
            return 'wait'
        if frontend.allow_prefill:
            if (frontend.prefill_contract is None
                    or capture_prepared_prefill_contract(
                        getattr(self,'split_prefill_batch',None)) != frontend.prefill_contract):
                return 'wait'
        memory = decision.lease.memory
        captured = getattr(memory, 'routing_preview', None)
        if captured is not None and residency != captured[0]:
            return 'stale'
        plan = getattr(memory,'resource_plan',None)
        controller = getattr(self,'shared_hbm_controller',None)
        if plan is None or controller is None or plan.event_id != current_event_id:
            return 'unsupported'
        if not controller.joint_hbm_backing_ready(plan):
            return 'wait'
        observed = controller.pool.snapshot()
        if controller._kv_plan_source_identity(observed) != source_memory_identity:
            return 'stale'
        if frontend.allow_ann:
            if residency is None or self.ann_executor.preview_shared_ivf_residency() != residency:
                return 'stale'
        # Snapshot reads above must not have changed the execution frontier.
        if source_execution != capture_execution_stamp(self):
            return 'stale'
        self.rag_frontend_plan = frontend
        self.rag_joint_active_memory_plan = plan
        # The joint-branch counters say WHICH check declined a launch but not by
        # how much. plan.ann is non-None here: joint_hbm_backing_ready above the
        # assignment already rejects a plan whose ann is None.
        self.rag_joint_active_ann_fit = (plan.ann.fit.fitted_rows,
            plan.ann.fit.required_bytes, plan.ann.fit.budget_bytes,
            plan.ann.fit.max_inflight)
        self.rag_joint_applied_compute = decision.lease.executable
        self.rag_joint_activation_count = getattr(self,'rag_joint_activation_count',0)+1
        self.rag_joint_active_event_id = current_event_id  # Publish LAST.
        return 'ready'

    def _capture_rag_queued_prefill_shapes(self):
        requests = tuple(getattr(self, 'waiting_queue', ()) or ())
        if not requests:
            return ()
        cache = getattr(self, 'tree_cache', None)
        # Inherited methods on host/hybrid caches do not establish equivalent
        # residency semantics. Add explicit adapters for those implementations.
        if type(cache) is not RadixCache:
            return None
        return preview_queued_prefill_shapes(requests,
            lambda tokens, namespace: cache.peek_prefix_length(
                RadixKey(list(tokens), namespace)))

    def _capture_rag_cached_prefill_budget(self):
        """Detached current-budget preview, not allocation or elastic planning.

        Mirrors only the verified ordinary cached FCFS serving contract. A
        compute proposal may use this observation but must not interpret an
        empty current-budget selection as proof elastic HBM cannot admit work.
        Future target capacity fitting remains a separate, once-only step.
        """
        cache = getattr(self, 'tree_cache', None)
        policy = getattr(self, 'policy', None)
        allocator = getattr(self, 'token_to_kv_pool_allocator', None)
        if (type(cache) is not RadixCache or cache.disable
                or isinstance(allocator, SWATokenToKVPoolAllocator)
                or getattr(self, 'page_size', None) != 1
                or getattr(self, 'pp_size', None) != 1
                or getattr(policy, 'policy', None) != CacheAgnosticPolicy.FCFS
                or getattr(policy, 'enable_priority_scheduling', True)
                or getattr(self, 'chunked_prefill_size', 0) is not None
                or getattr(self, 'chunked_req', None) is not None
                or any(getattr(self, name, True) for name in
                       ('try_preemption', 'enable_lora', 'enable_hierarchical_cache',
                        'enable_hicache_storage', 'is_mixed_chunk'))):
            return None
        requests = tuple(self.waiting_queue)
        keys = []
        def peek(tokens, namespace):
            key = RadixKey(list(tokens), namespace)
            keys.append(key)
            return cache.peek_prefix_length(key)
        shapes = preview_queued_prefill_shapes(requests, peek)
        if shapes is None:
            return None
        limits = tuple((str(req.rid), req.sampling_params.max_new_tokens) for req in requests)
        running = tuple(self.running_batch.reqs)
        ratio = self.new_token_ratio
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError('Invalid observed running reservation ratio')
        reserve = sum(min(req.sampling_params.max_new_tokens-len(req.output_ids),
                          CLIP_MAX_NEW_TOKENS)*ratio for req in running)
        slots = max(0, self.get_num_allocatable_reqs(len(running)))
        published = allocator.current_mapped_capacity()
        free = allocator.available_size()
        if (type(published) is not int or type(free) is not int
                or not 0 <= free <= published):
            raise ValueError('Inconsistent published allocator capacity')
        return CachedPrefillBudget(shapes, limits, CLIP_MAX_NEW_TOKENS,
            free+cache.evictable_size(), reserve,
            self.max_prefill_tokens, slots, cache.peek_prefix_lock_costs(keys),
            bool(self.running_batch.batch_is_full), published, published-free)

    def _capture_rag_inflight_retrieval_work(self, *, source_execution, cpu_plan,
            window_started_at, release_profile, collection_profile, collection_mode, timing=True):
        """Compose existing hybrid CPU/GPU joins with already-queued collection.

        Polls only nonblocking GPU readiness, never Future.result or result.poll.
        Release durations are host-age residuals, not CUDA service plus queue
        time. Logical retrieval receives credit only at its result join.
        """
        if collection_mode not in ('worker','inline') or type(timing) is not bool:
            raise ValueError('Explicit collection path and timing policy required')
        collector=self._capture_rag_collector_work(source_execution=source_execution,
            window_started_at=window_started_at,collection_profile=collection_profile,timing=timing)
        if collector is None:
            return None
        operations=list(collector[0]);ids=list(collector[1])
        queued_tail=(operations[-1].key,) if operations and collection_mode=='worker' else ()
        cpu_dependencies={frozenset(members):tuple(keys) for members,keys in cpu_plan[1]}
        groups=getattr(self,'ann_search_inflight',()) or ()
        if isinstance(groups,dict):groups=(groups,)
        groups=tuple(groups)
        now=time.perf_counter();offset=max((now-window_started_at)*1000.,cpu_plan[2])
        if not math.isfinite(offset) or now<window_started_at:
            raise ValueError('Common retrieval observation window required')
        for index,group in enumerate(groups):
            tasks=tuple(group.get('tasks') or (group,))
            members=tuple(str(task['request_id']) for task in tasks)
            dependencies=cpu_dependencies.get(frozenset(members))
            if dependencies is None:
                return None
            result=group.get('result')
            full_cpu=observe_full_cpu_fallback(result,observed_at=now)
            is_full_cpu=full_cpu is not None
            if is_full_cpu and (full_cpu.status in ('failed','cancelled')
                    or (full_cpu.status!='succeeded' and not dependencies)):
                return None
            if not is_full_cpu and (result is None or not callable(getattr(result,'gpu_poll',None))
                    or not hasattr(result,'cpu_searches') or not hasattr(result,'gpu_result')):
                return None
            rows=sum(int(task['queries'].shape[0]) for task in tasks)
            topk=getattr(result,'topk',None)
            if rows<=0 or type(topk) is not int or topk<=0:
                return None
            branches=list(dependencies)
            if not is_full_cpu and not result.gpu_poll():
                launched=group.get('launch_time');partition=group.get('stream_idx')
                if (type(launched) not in (int,float) or not math.isfinite(launched)
                        or launched>now or type(partition) is not int or partition<0):
                    return None
                age=(now-launched)*1000.
                residual=None
                if timing and release_profile is not None:
                    residual=release_profile.query_remaining_ms((partition,rows),elapsed_ms=age)
                key=f'observed:ann:{index}:gpu_release'
                operations.append(Operation(key,'ann_release',residual,(key,),ready_ms=offset,
                    service_shape=('ann_release',partition,rows,age)))
                branches.append(key)
            shape=((collection_mode,rows,topk) if is_full_cpu else
                (collection_mode,rows,topk,len(result.cpu_searches),result.gpu_result is not None))
            duration=None
            if timing and collection_profile is not None and not is_full_cpu:
                duration=collection_profile.query_ms(*shape)
            elif timing and is_full_cpu and callable(getattr(collection_profile,'query_full_cpu_remaining_ms',None)):
                duration=collection_profile.query_full_cpu_remaining_ms(*shape,elapsed_ms=0.)
            operations.append(Operation(f'observed:ann:{index}:collect','collect',duration,('ann_collect',),
                tuple(Credit(rid,'ann') for rid in members),ready_ms=offset,
                depends_on=tuple(dict.fromkeys((*branches,*queued_tail))),
                service_shape=('ann_collect_full_cpu' if is_full_cpu else 'ann_collect',*shape)))
            ids.extend(members)
        if len(set(ids))!=len(ids):
            raise ValueError('Duplicate request across submitted and collecting ANN')
        if capture_execution_stamp(self)!=source_execution:
            return None
        return tuple(operations),tuple(ids)

    def _capture_rag_collector_work(self, *, source_execution, window_started_at,
                                    collection_profile=None, timing=True):
        """Observe already-joined worker collection without polling/waiting.

        Registry order is the actual single-worker submission order. Completed
        but not yet drained output and failed work require another native
        observation; do not invent zero-cost send or successful completion.
        This covers the collector, not still-running GPU/CPU searches.
        """
        if type(timing) is not bool:
            raise ValueError('Explicit collector timing mode required')
        if capture_execution_stamp(self)!=source_execution:
            return None
        groups=tuple(getattr(self,'rag_ann_collection_pending',()) or ())
        now=time.perf_counter();offset=(now-window_started_at)*1000.
        if not math.isfinite(offset) or offset<0:
            raise ValueError('Common collector observation window required')
        operations=[];ids=[]
        for index,group in enumerate(groups):
            progress=group.get('rag_collection_progress')
            if progress is None or len(progress)!=4 or not group.get('rag_result_join_observed'):
                return None
            state,queued,started,finished=progress
            if state not in ('queued','active') or finished is not None:
                return None
            if (type(queued) not in (int,float) or not math.isfinite(queued) or queued>now
                    or (state=='queued' and started is not None)):
                return None
            if state=='active' and (index!=0 or type(started) not in (int,float)
                    or not math.isfinite(started) or not queued<=started<=now):
                return None
            age=(now-started)*1000. if state=='active' else 0.
            tasks=tuple(group.get('tasks') or (group,))
            members=tuple(str(task['request_id']) for task in tasks)
            rows=sum(int(task['queries'].shape[0]) for task in tasks)
            result=group.get('result')
            topk=getattr(result,'topk',None)
            full_cpu=observe_full_cpu_fallback(result,observed_at=now)
            is_full_cpu=full_cpu is not None
            if is_full_cpu and full_cpu.status!='succeeded':
                return None
            if (rows<=0 or type(topk) is not int or topk<=0
                    or (not is_full_cpu and (result is None or not hasattr(result,'cpu_searches')
                    or not hasattr(result,'gpu_result')))):
                return None
            shape=(('worker',rows,topk) if is_full_cpu else
                ('worker',rows,topk,len(result.cpu_searches),result.gpu_result is not None))
            duration=None
            if timing and collection_profile is not None and not is_full_cpu:
                duration=collection_profile.query_remaining_ms(*shape,elapsed_ms=age)
            elif timing and is_full_cpu and callable(getattr(collection_profile,'query_full_cpu_remaining_ms',None)):
                duration=collection_profile.query_full_cpu_remaining_ms(*shape,elapsed_ms=age)
            key=f'observed:collector:{index}'
            operations.append(Operation(key,'collect',duration,('ann_collect',),
                tuple(Credit(rid,'ann') for rid in members),ready_ms=offset,
                depends_on=(operations[-1].key,) if operations else (),
                service_shape=('ann_collect_full_cpu_residual' if is_full_cpu else 'ann_collect_residual',*shape,age)))
            ids.extend(members)
        if len(set(ids))!=len(ids):
            raise ValueError('Duplicate logical request in collector registry')
        if capture_execution_stamp(self)!=source_execution:
            return None
        return tuple(operations),tuple(ids)

    def _build_rag_joint_observed_work(self, *, work_inputs, executable, memory,
                                       generation_inputs, retrieval_inputs=None):
        """Compose observed-stage adapters without repeating the shared CPU DAG.

        Returns operations, joining-decode plan, and retrieval completion gates.
        The gates are retained for prospective input preparation linkage; this
        is not yet a complete forecast of newly retrieved prompts and cache hits.
        All stages use the same captured request/reference set and fitted lease.
        """
        stamp,references,urgency,cpu_plan,queued_shapes,queued_budget=work_inputs['common']
        if capture_execution_stamp(self)!=stamp:
            return None
        operations=list(cpu_plan[0]);joins=()
        inflight_ids={f.request_id for f in stamp.frontiers if f.stage=='ann_inflight'}
        inflight_work=work_inputs.get('inflight_retrieval_work')
        if inflight_ids and inflight_work is None:
            # CPU dependencies alone do not represent the outstanding GPU
            # search/result join. Never score an incomplete continuation.
            return None
        if inflight_work is not None:
            inflight_operations,completed_ids=inflight_work
            if set(completed_ids)!=inflight_ids or len(set(completed_ids))!=len(completed_ids):
                raise ValueError('Observed in-flight retrieval graph must cover the exact frontier')
            following=build_following_input_preparation(stamp.frontiers,inflight_operations,completed_ids)
            if following is None:
                return None
            new_preparation,joins=following
            operations.extend(inflight_operations)
            operations.extend(new_preparation)
        if retrieval_inputs is not None:
            retrieval=self._build_rag_joint_pending_retrieval_graph(
                cpu_plan=cpu_plan,memory_lease=memory,timing=False,**retrieval_inputs)
            if retrieval is None:
                return None
            new_operations,retrieved_ids=retrieval
            # Same-parent ANN reuses one cached execution stream; changing
            # partitions drains incompatible ANN before switching. Submitted
            # GPU work therefore precedes the new ANN branch in either case.
            # CPU spill can still run independently. Max/dependency gates do
            # not add the same release delay twice to memory readiness.
            submitted_gpu=tuple(op.key for op in (inflight_work[0] if inflight_work else ())
                                if op.stage=='ann_release')
            new_operations=tuple(replace(op,depends_on=tuple(dict.fromkeys(
                (*op.depends_on,*submitted_gpu)))) if op.stage=='ann' else op
                for op in new_operations)
            operations.extend(new_operations)
            following=build_following_input_preparation(stamp.frontiers,new_operations,retrieved_ids)
            if following is None:
                return None
            new_preparation,new_joins=following
            joins=(*joins,*new_joins)
            operations.extend(new_preparation)
        elif executable.ann_batch:
            return None  # Never silently discard admitted ANN work.
        operations.extend(work_inputs['preparation_operations'])
        generation=self._build_rag_joint_generation_work(executable=executable,memory=memory,
            frontiers=stamp.frontiers,reference_profiles=dict(references.profiles),
            queued_shapes=queued_shapes,**generation_inputs)
        if generation is None:
            return None
        generated,decode_plan=generation
        operations.extend(generated)
        if len({op.key for op in operations})!=len(operations):
            raise ValueError('Duplicate shared operation in joint graph')
        if capture_execution_stamp(self)!=stamp:
            return None
        return tuple(operations),decode_plan,joins

    def _bind_rag_joint_decision_callbacks(self, context, *, work_by_allocation,
                                           timing_by_allocation, horizon_ms, switching_margin,
                                           prospective_inputs_builder=None, prospective_by_allocation=None,
                                           engineering_operation_timings=None):
        """Bind captured partition inputs to the actual graph/pricing adapters.

        Profile objects/callbacks must already be frozen by the caller. Copy
        parameter dictionaries but do not deep-copy native task owners. Missing
        partition data remains unsupported; never substitute another partition.
        Optional per-allocation prospective settings use the actual input
        constructor; live provider capture/installation remains separate.
        """
        snapshot=context['snapshot'];work_inputs=context['work_inputs']
        if horizon_ms!=context['horizon_ms']:
            raise ValueError('Joint pricing must use the common capture horizon')
        mode=context['mode']
        if engineering_operation_timings is not None:
            from sglang.srt.multiplex.engineering_profiles import EngineeringOperationTimings
            receipt=getattr(self,'rag_joint_bootstrap_receipt',{})
            if (not isinstance(engineering_operation_timings,EngineeringOperationTimings)
                    or receipt.get('engineering_only') is not True
                    or receipt.get('formal_ready') is not False):
                raise ValueError('Placeholder operation timing requires explicit engineering bootstrap')
        if prospective_inputs_builder is not None and prospective_by_allocation is not None:
            raise ValueError('Choose one prospective input construction path')
        if work_inputs['mode']!=mode:
            raise ValueError('Captured work policy must match admission policy')
        allocations=dict(context['compute_inputs']['allocation_by_stream'])
        def partition_for(allocation):
            matches=[idx for idx,value in allocations.items() if value==allocation]
            if len(matches)!=1:
                raise ValueError('Work parameters require one captured partition')
            return matches[0]
        work={}
        for allocation,args in work_by_allocation.items():
            partition=partition_for(allocation)
            if set(args)-{'generation_inputs','retrieval_inputs'}:
                raise ValueError('Only partition-specific generation/retrieval inputs allowed')
            work[allocation]={key:None if value is None else dict(value)
                              for key,value in args.items()}
            retrieval=work[allocation].get('retrieval_inputs')
            if retrieval is not None and retrieval.get('partition')!=partition:
                raise ValueError('Retrieval profile partition differs from compute allocation')
        timing={}
        prospective_settings={}
        if mode=='full':
            def bind_prefill_models(partition,raw):
                settings=dict(raw);overlap=settings.pop('overlap_class','solo')
                selector=overlap if callable(overlap) else None
                mapping={partition:self.sm_counts[partition][0]}
                if selector is None:
                    return None,{overlap:bind_layer_scaled_prefill_query(
                        partition_to_sm=mapping,overlap_class=overlap,**settings)}
                solo={key:value for key,value in settings.items() if key not in
                    ('corrections','expected_correction_identity')}
                return selector,{
                    'solo':bind_layer_scaled_prefill_query(
                        partition_to_sm=mapping,overlap_class='solo',**solo),
                    'ann':bind_layer_scaled_prefill_query(
                        partition_to_sm=mapping,overlap_class='ann',**settings)}
            for allocation,args in (prospective_by_allocation or {}).items():
                partition=partition_for(allocation)
                prospective_settings[allocation]={**args,'layout':dict(args['layout']) if args.get('layout') is not None else None,
                    'reservations':tuple(args['reservations'])}
                if args.get('physical_allocations') is not None:
                    prospective_settings[allocation]['physical_allocations']=tuple(args['physical_allocations'])
                prefill_model=prospective_settings[allocation].pop('prefill_model_inputs',None)
                if prefill_model is not None:
                    if args.get('query_prefill') is not None:
                        raise ValueError('Choose prospective prefill model or explicit query')
                    selector,queries=bind_prefill_models(partition,prefill_model)
                    queries={category:(lambda shape,q=query,p=partition:q(p,shape))
                             for category,query in queries.items()}
                    if selector is None:
                        prospective_settings[allocation]['query_prefill']=next(iter(queries.values()))
                    else:
                        prospective_settings[allocation]['_prefill_projection_queries']=(selector,queries)
                elif args.get('query_prefill') is None:
                    if 'prefill_profile' in args or ('partition' in args and args['partition']!=partition):
                        raise ValueError('Prospective prefill profile is owned by the common compute capture')
                    prospective_settings[allocation].update(
                        partition=partition,prefill_profile=context['compute_inputs'].get('prefill_profile'))
            for allocation,args in timing_by_allocation.items():
                if 'work_inputs' in args:
                    raise ValueError('Common work observation cannot be overridden')
                timing[allocation]={**args,'prefill_inputs':dict(args['prefill_inputs'])}
                prefill_model = timing[allocation].pop('prefill_model_inputs', None)
                if prefill_model is not None:
                    inputs = timing[allocation]['prefill_inputs']
                    if (inputs.get('query_bundle_service') is not None
                            or inputs.get('bundle_queries_by_grain') is not None):
                        raise ValueError('Choose phase-scaled prefill or interval profile, not both')
                    partition = partition_for(allocation)
                    selector,queries=bind_prefill_models(partition,prefill_model)
                    if selector is None:
                        inputs['query_bundle_service']=next(iter(queries.values()))
                    else:
                        timing[allocation]['_prefill_projection_queries']=(selector,queries)
                bundle_queries=timing[allocation]['prefill_inputs'].get('bundle_queries_by_grain')
                if bundle_queries is not None:
                    if timing[allocation]['prefill_inputs'].get('query_bundle_service') is not None:
                        raise ValueError('Choose fixed bundle query or fitted-grain queries')
                    bundle_queries=dict(bundle_queries)
                    if any(type(grain) is not int or grain<=0 or not callable(query)
                           for grain,query in bundle_queries.items()):
                        raise ValueError('Positive calibrated grains and callable queries required')
                    timing[allocation]['prefill_inputs']['bundle_queries_by_grain']=bundle_queries
                model_inputs=timing[allocation].pop('decode_model_inputs',None)
                if model_inputs is not None:
                    if args.get('decode_timing') is not None:
                        raise ValueError('Choose calibrated decode model or frozen timing callback')
                    partition=partition_for(allocation)
                    settings=dict(model_inputs)
                    overlap=settings.pop('overlap_class')
                    domain=settings.pop('workload_supported',None)
                    residual=settings.pop('query_inflight_remaining',None)
                    overlap_selector=overlap if callable(overlap) else None
                    pressure_model=context['compute_inputs'].get('decode_profile')
                    if isinstance(pressure_model,PressureDecodeComputeProfile):
                        timing[allocation]['_pressure_decode_settings']=(
                            partition,self.sm_counts[partition][1],pressure_model,dict(settings),residual,overlap)
                    classes=('solo','ann','prefill','ann_prefill')
                    if overlap_selector is None and overlap not in classes:
                        raise ValueError('Validated decode overlap class required')
                    clocks={}
                    for category in classes if overlap_selector is not None else (overlap,):
                        interference=None
                        if category!='solo':
                            if isinstance(pressure_model,PressureDecodeComputeProfile):
                                # Rebound below only after fitting determines ANN work.
                                interference=lambda shape,solo:None
                            else:
                                captured=context.get('frozen_compute_profiles') or {}
                                profile=captured.get('decode_service_profile')
                                if profile is None:
                                    raise ValueError('Same-event decode service profile required')
                                interference=bind_frozen_decode_interference(profile,
                                    partition=partition,overlap_class=category,workload_supported=domain)
                        query=bind_decode_stage_model(sm_count=self.sm_counts[partition][1],
                            ann_coexecution=category!='solo',interference_query=interference,**settings)
                        clocks[category]=lambda ops,q=query,r=residual: \
                            time_frozen_decode_operations(ops,q,query_inflight_remaining=r)
                    if overlap_selector is None:
                        timing[allocation]['decode_timing']=clocks[overlap]
                    else:
                        timing[allocation]['_decode_projection_clocks']=(overlap_selector,clocks)
                if timing[allocation]['prefill_inputs'].get('partition')!=partition_for(allocation):
                    raise ValueError('Prefill timing partition differs from compute allocation')
                if args.get('handoff_inputs') is not None:
                    handoff = dict(args['handoff_inputs'])
                    if handoff.get('partition') != partition_for(allocation):
                        raise ValueError('Handoff timing partition differs from compute allocation')
                    handoff['endpoint_shapes'] = {
                        key: tuple((s[0], s[1], tuple(s[2]), s[3]) for s in pair)
                        for key, pair in handoff['endpoint_shapes'].items()}
                    timing[allocation]['handoff_inputs'] = handoff
        def check(s):
            if s is not snapshot:
                raise ValueError('Bound callbacks belong to one joint event')
        def graph(s,executable,memory):
            check(s)
            args=work.get(executable.allocation)
            if args is None:
                return None
            return self._build_rag_joint_observed_work(work_inputs=work_inputs,
                executable=executable,memory=memory,**args)
        def price(s,projection):
            check(s)
            if mode!='full':
                raise ValueError('Independent-adaptive cannot invoke joint pricing')
            if engineering_operation_timings is not None:
                return engineering_operation_timings.price(projection.operations)
            args=timing.get(projection.executable.allocation)
            if args is None:
                return None
            args={**args,'prefill_inputs':dict(args['prefill_inputs'])}
            projected_prefill=args.pop('_prefill_projection_queries',None)
            if projected_prefill is not None:
                selector,queries=projected_prefill;category=selector(projection)
                if category is None:return None
                if category not in queries:
                    raise ValueError('Unsupported projected prefill overlap class')
                args['prefill_inputs']['query_bundle_service']=queries[category]
            projected_clocks=args.pop('_decode_projection_clocks',None)
            if projected_clocks is not None:
                selector,clocks=projected_clocks
                # Caller binds observed in-flight work and validated workload
                # domain; the fitted proposal alone cannot establish overlap.
                category=selector(projection)
                if category is None:return None
                if category not in clocks:
                    raise ValueError('Unsupported projected decode overlap class')
                args['decode_timing']=clocks[category]
            pressure_settings=args.pop('_pressure_decode_settings',None)
            if pressure_settings is not None:
                partition,sm_count,pressure_model,settings,residual,overlap=pressure_settings
                if projected_clocks is None:
                    category=overlap(projection) if callable(overlap) else overlap
                if category is None:return None
                rows=projection.executable.ann_batch
                ann_service=0.
                if rows:
                    # HBM fitting can retain a different set of IVF lists even
                    # when the logical ANN row count is unchanged.  Decode
                    # pressure must therefore use the fitted projection's
                    # actual GPU work, not the compute proposal's batch-only
                    # service estimate.  The latter remains a compatibility
                    # path for exact legacy profiles whose graph has no linear
                    # work shape.
                    linear=tuple(op for op in projection.operations
                        if op.stage=='ann' and isinstance(op.service_shape,tuple)
                        and len(op.service_shape)==7
                        and op.service_shape[0]=='ann_gpu_linear')
                    if linear:
                        if len(linear)!=1:
                            return None
                        shape=linear[0].service_shape
                        if shape[1]!=partition or shape[2]!=rows:
                            return None
                        gpu_profile=args.get('gpu_profile')
                        query=getattr(gpu_profile,'query_work_ms',None)
                        if not callable(query):
                            return None
                        ann_service=query(*shape[1:])
                    else:
                        ann_service=context['compute_inputs']['service_profile'].query_ms(
                            partition,rows)
                outstanding=dict(pressure_model.outstanding_by_partition).get(partition)
                if ann_service is None or outstanding is None:return None
                total=outstanding+ann_service
                if total>0 and category in ('solo','prefill'):
                    return None  # Caller cannot label admitted/in-flight ANN as absent.
                def interference(shape,solo):
                    return pressure_model.pressure.query_ms(partition,category,
                        tuple(n-1 for n in shape.context_tokens),total,
                        expected_identity=pressure_model.expected_pressure_identity)
                query=bind_decode_stage_model(sm_count=sm_count,
                    ann_coexecution=category!='solo',interference_query=interference,**settings)
                args['decode_timing']=lambda ops:time_frozen_decode_operations(
                    ops,query,query_inflight_remaining=residual)
            bundle_queries=args['prefill_inputs'].pop('bundle_queries_by_grain',None)
            if bundle_queries is not None:
                # Price the executable grain AFTER safety fitting. The last
                # short interval retains its traversal grain's calibration.
                query=bundle_queries.get(projection.executable.prefill_layers)
                args['prefill_inputs']['query_bundle_service']=(
                    query if query is not None else lambda partition,shape:None)
            return self._time_rag_joint_work(projection.operations,work_inputs=work_inputs,**args)
        def prospective(s,projection,h):
            check(s)
            if mode!='full' or h!=horizon_ms:
                raise ValueError('Prospective scoring requires Full and the common horizon')
            if prospective_inputs_builder is not None:
                return prospective_inputs_builder(s,projection,h,work_inputs=work_inputs)
            args=prospective_settings.get(projection.executable.allocation)
            if args is None:return None
            args=dict(args);projected=args.pop('_prefill_projection_queries',None)
            if projected is not None:
                selector,queries=projected;category=selector(projection)
                if category is None:return None
                if category not in queries:
                    raise ValueError('Unsupported prospective prefill overlap class')
                args['query_prefill']=queries[category]
            return self._build_rag_prospective_inputs(s,projection,h,work_inputs=work_inputs,**args)
        fields=('snapshot','source_execution','source_stream_idx','mode',
                'compute_inputs','memory_inputs')
        demand_inputs=dict(context['demand_inputs'])
        if prospective_by_allocation is not None:
            policies=[]
            for allocation,args in prospective_settings.items():
                policy=args.get('conservative_cache_reserve',False)
                if type(policy) is not bool:
                    raise ValueError('Explicit per-allocation conservative cache policy required')
                policies.append((allocation,policy))
            demand_inputs['conservative_cache_reserve_by_allocation']=tuple(policies)
        return {**{key:context[key] for key in fields},'demand_inputs':demand_inputs,
                'joint_work_builder':graph,
                'timing_builder':price,'horizon_ms':horizon_ms,'switching_margin':switching_margin,
                'prospective_inputs_builder':prospective if (prospective_inputs_builder is not None
                    or prospective_by_allocation is not None) else None}

    def _build_rag_prospective_inputs(self, snapshot, projection, horizon_ms, *, work_inputs,
            reservations, layout=None, total_layers=None, input_token_budget=None, max_requests=None,
            clip_max_new_tokens=None, kv_key, existing_kv_key, compute_ready_ms,
            query_prefill=None, time_handoff, physical_allocations=None, separate_commit_gate=False,
            prefill_profile=None, partition=None, conservative_cache_reserve=False):
        """Join actual common capture and fitted-lease adapters for Full scoring.

        All scalar settings and service callbacks must already be frozen for
        this partition. No recapture, lookup of future arrivals, planning or
        dispatch. Missing owner/capacity/cache facts stay unavailable.
        """
        if work_inputs['mode']!='full':raise ValueError('Full-only prospective input construction')
        self.rag_joint_prospective_input_failure=None
        def unavailable(reason):
            self.rag_joint_prospective_input_failure=reason
            counts=getattr(self,'rag_joint_prospective_input_failure_counts',None)
            if counts is None:
                counts={};self.rag_joint_prospective_input_failure_counts=counts
            counts[reason]=counts.get(reason,0)+1
            return None
        if type(conservative_cache_reserve) is not bool:
            raise ValueError('Explicit conservative forecast policy required')
        if conservative_cache_reserve:
            receipt=getattr(self,'rag_joint_bootstrap_receipt',{})
            if receipt.get('engineering_only') is not True or receipt.get('formal_ready') is not False:
                raise ValueError('Conservative uncalibrated forecast requires engineering bootstrap')
        stamp,refs=work_inputs['common'][:2]
        if capture_execution_stamp(self)!=stamp:return unavailable('stale_initial_capture')
        if query_prefill is not None and prefill_profile is not None:
            raise ValueError('Choose frozen prefill profile or explicit timing callback')
        captured=work_inputs.get('prospective_settings')
        def setting(name,supplied):
            if captured is None:return supplied
            value=captured[name]
            if supplied is not None and supplied!=value:
                raise ValueError('Prospective setting differs from the captured runtime')
            return value
        layout=setting('layout',layout)
        total_layers=setting('total_layers',total_layers)
        input_token_budget=setting('input_token_budget',input_token_budget)
        clip_max_new_tokens=setting('clip_max_new_tokens',clip_max_new_tokens)
        max_running=None if captured is None else captured['max_running_requests']
        if max_requests is None:max_requests=max_running
        if any(value is None for value in (layout,total_layers,input_token_budget,clip_max_new_tokens,max_requests)):
            return unavailable('missing_prospective_setting')
        if not math.isfinite(horizon_ms) or horizon_ms<=0:raise ValueError('Positive common H required')
        if projection.decode_plan is None:return unavailable('missing_decode_plan')
        shapes=describe_prospective_prefill(stamp.frontiers,refs,
            tuple(op for op in projection.operations if op.stage=='prepare'))
        if shapes is None:return unavailable('unsupported_prospective_shape')
        if query_prefill is None and shapes:
            if prefill_profile is None or partition is None:
                return unavailable('missing_prefill_profile')
            if type(partition) is not int or partition<0:
                raise ValueError('Actual prospective prefill partition required')
            if callable(getattr(prefill_profile, 'query_shape_ms', None)):
                query_prefill=lambda shape:prefill_profile.query_shape_ms(partition, shape)
            else:
                query_prefill=lambda shape:prefill_profile.query_bundle_ms(partition,
                    sum(shape.extend_tokens),shape.start_layer,shape.end_layer)
        elif query_prefill is None:
            # No prospective prefill work exists in this fitted continuation,
            # so a prefill service profile is not an input to its score.
            query_prefill=lambda shape:None
        frontiers={f.request_id:f for f in stamp.frontiers}
        limits={shape.request_id:frontiers[shape.request_id].remaining_output_limit for shape in shapes}
        if any(type(limit) is not int or limit<=0 for limit in limits.values()):
            return unavailable('missing_output_limit')
        budget=self._build_rag_prospective_capacity_budget(snapshot=snapshot,projection=projection,
            work_inputs=work_inputs,reservations=reservations,physical_allocations=physical_allocations,
            layout=layout,kv_key=kv_key)
        if budget is None:return unavailable('missing_capacity_budget')
        rule=work_inputs.get('running_reservation_rule')
        policy=work_inputs.get('completion_policy')
        if rule is None:return unavailable('missing_running_reservation_rule')
        if policy not in ('disabled_unit_page','radix_unit_page_unknown_prefix'):
            return unavailable('unsupported_completion_policy')
        if not conservative_cache_reserve and policy=='radix_unit_page_unknown_prefix':
            clock=work_inputs.get('eviction_clock')
            observed=getattr(clock,'observed_offset_ms',None)
            if (observed is None or not math.isfinite(observed) or observed<0):
                return unavailable('missing_eviction_observation_clock')
            # The frozen radix tree was captured during this decision.  It may
            # constrain work only after that real observation boundary, never
            # retroactively at t=0 of the common scoring window.
            compute_ready_ms=max(compute_ready_ms,observed)
        joins={f.request_id for f in projection.decode_plan.frontiers}
        queue_empty=not any(f.stage in ('prefill_queued','prefill_active','prefill_merge_pending')
            and f.request_id not in joins for f in stamp.frontiers)
        result=dict(frontiers=stamp.frontiers,reference_selection=refs,shapes=shapes,budget=budget,
            max_new_tokens=limits,clip_max_new_tokens=clip_max_new_tokens,input_token_budget=input_token_budget,
            max_requests=max_requests,total_layers=total_layers,kv_key=kv_key,existing_kv_key=existing_kv_key,
            max_running_requests=max_running,
            compute_ready_ms=compute_ready_ms,earlier_queue_empty=queue_empty,query_prefill=query_prefill,
            time_handoff=time_handoff,completion_policy=policy,running_reservation_rule=rule,
            separate_commit_gate=separate_commit_gate,
            conservative_queued_reserve=conservative_cache_reserve,
            cache_backed_admission=policy=='radix_unit_page_unknown_prefix',
            # Zero is a prediction lower bound, NOT a claim about live cache
            # occupancy. Retained cached KV earns no free capacity or eviction
            # credit; real admission/commit still uses the native allocator.
            query_evictable_tokens=(lambda p,t,at:0) if conservative_cache_reserve else
                lambda p,t,at:self._query_rag_prospective_evictable(p,t,at,work_inputs=work_inputs),
            build_cache_eviction=None if conservative_cache_reserve else
                lambda p,t,at,n,key:self._build_rag_predicted_cache_eviction(
                    p,t,at,n,key,work_inputs=work_inputs))
        if capture_execution_stamp(self)!=stamp:return unavailable('stale_final_capture')
        return result

    def _time_rag_joint_handoff(self, operations, *, work_inputs, partition, endpoint_shapes):
        """Bind graph handoff pricing to the event's real frozen observations.

        Endpoint shapes must be provided by the same captured/predicted work
        graph, never inferred from a nearby profile bucket. Pending results
        may belong to an old partition; the not-yet-executed merge must belong
        to this selected partition. No profile recapture, event polling or
        work/lease search is performed here.
        """
        if work_inputs['mode'] != 'full':
            raise ValueError('Independent-adaptive cannot price handoff work')
        stamp = work_inputs['common'][0]
        if capture_execution_stamp(self) != stamp:
            return None
        if type(partition) is not int or partition < 0:
            raise ValueError('Actual selected handoff partition required')
        profile = work_inputs.get('handoff_profile')
        if profile is None:
            return None
        endpoints = dict(endpoint_shapes)
        def query(result, merge):
            shapes = endpoints.get(merge.key)
            if shapes is None:
                return None
            if (not isinstance(shapes, tuple) or len(shapes) != 2
                    or any(not isinstance(s, tuple) or len(s) != 4 for s in shapes)):
                raise ValueError('Exact frozen result/merge endpoint shapes required')
            rs, ms = shapes
            if (rs[1] != 'result' or ms[1] not in ('merge', 'deferred_merge')
                    or ms[0] != partition or (result is not None and rs[0] != partition)):
                raise ValueError('Handoff endpoints differ from selected execution domain')
            if result is not None and rs[2] != tuple(result.service_shape[2]):
                raise ValueError('Handoff endpoint differs from projected prompt shape')
            if merge.service_shape[2] and ms[1] != 'deferred_merge':
                raise ValueError('Pending handoff requires deferred merge observations')
            return profile.query_paired_sample(rs, ms)
        result = time_frozen_prefill_handoff(tuple(operations), query)
        if capture_execution_stamp(self) != stamp:
            return None
        return result

    def _time_rag_joint_work(self, operations, *, work_inputs, cpu_profile,
                             gpu_profile, collection_profile, prefill_inputs,
                             handoff_timing=None, handoff_inputs=None, decode_timing=None,
                             input_preparation_profile=None):
        """Use this event's frozen preparation and fixed-partition stage inputs."""
        if work_inputs['mode']!='full':
            raise ValueError('Independent-adaptive does not price joint work')
        receipt=getattr(self,'rag_joint_bootstrap_receipt',{})
        diagnostic=(receipt.get('engineering_only') is True
                    and receipt.get('formal_ready') is False)
        if diagnostic:
            self.rag_joint_last_timing_failure=()
        if handoff_inputs is not None:
            if handoff_timing is not None:
                raise ValueError('Choose one handoff timing path')
            frozen_handoff_inputs = dict(handoff_inputs)
            handoff_timing = lambda ops: self._time_rag_joint_handoff(
                ops, work_inputs=work_inputs, **frozen_handoff_inputs)
        operations=tuple(operations)
        observed_cpu=tuple(work_inputs['common'][3][0])
        inflight_work=work_inputs.get('inflight_retrieval_work')
        observed=(*observed_cpu,*(inflight_work[0] if inflight_work is not None else ()))
        preserved={op.key:op for op in observed}
        if len(preserved)!=len(observed) or not set(preserved)<={op.key for op in operations}:
            raise ValueError('Common observed retrieval work must be preserved')
        for op in operations:
            if op.key in preserved:
                if op!=preserved[op.key]:
                    raise ValueError('Observed retrieval operation changed after capture')
                if op.duration_ms is None:
                    if diagnostic:
                        self.rag_joint_last_timing_failure=(dict(
                            stage=op.stage,service_shape=repr(op.service_shape)),)
                    return None
        # The common worker snapshot already timed existing searches using
        # their observed ages. Price only new work here; a later profile must
        # not overwrite those residuals differently for target and keep.
        unpriced=tuple(op for op in operations if op.key not in preserved)
        frozen_prefill_inputs=dict(prefill_inputs)
        priced=time_frozen_joint_operations(unpriced,cpu_profile=cpu_profile,
            gpu_profile=gpu_profile,collection_profile=collection_profile,
            input_preparation_profile=(input_preparation_profile
                if input_preparation_profile is not None else work_inputs['preparation_profile']),
            prefill_timing=lambda ops:self._time_rag_frozen_prefill(ops,**frozen_prefill_inputs),
            handoff_timing=handoff_timing,decode_timing=decode_timing)
        if priced is None:
            # Engineering-only failure evidence.  The mixed timing helper is
            # intentionally fail-closed, so its None result otherwise loses
            # the exact frozen shapes that exposed a profile-domain gap.  Do
            # not recapture or retry work here; retain only the already
            # projected immutable shapes for the next acceptance report.
            if diagnostic:
                self.rag_joint_last_timing_failure=tuple(
                    dict(stage=op.stage,service_shape=repr(op.service_shape))
                    for op in unpriced if op.duration_ms is None)[:32]
            return None
        measured={op.key:op for op in priced}
        return tuple(preserved[op.key] if op.key in preserved else measured[op.key]
                     for op in operations)

    def _capture_rag_joint_compute_profiles(self, *, partitions, min_samples,
            service_percentile, decode_tail_percentile, capacity_fast_percentile):
        """Freeze compute/credit/capacity histories at one serialized observation.

        Shared by both joint modes. Empty shape coverage remains unknown in
        the frozen query objects; this capture does not certify deployability
        or fill missing data with another domain's values.
        """
        stamp=capture_execution_stamp(self)
        partitions=tuple(partitions)
        if (not partitions or len(set(partitions))!=len(partitions)
                or any(type(p) is not int or not 0<=p<len(self.sm_counts) for p in partitions)):
            raise ValueError('Unique actual compute partitions required')
        batch=getattr(self,'running_batch',None)
        batch_size=0 if batch is None else len(batch.reqs)
        if bool(batch_size)!=bool(self._decode_batch_active()):
            return None
        ann=capture_ann_service_profile(self,percentile=service_percentile,min_samples=min_samples)
        prefill=capture_prefill_layer_profile(self,percentile=service_percentile,min_samples=min_samples)
        decode=capacity=decode_service=None
        if batch_size:
            decode=capture_decode_decision_profile(self,partitions=partitions,batch_size=batch_size,
                percentile=decode_tail_percentile,min_samples=min_samples)
            capacity=capture_decode_capacity_profile(self,partitions=partitions,batch_size=batch_size,
                fast_percentile=capacity_fast_percentile,min_samples=min_samples)
            # Progress pricing needs submit-to-completion service, not the
            # completion intervals used by the existing guard/capacity paths.
            # Freeze it here as well so callbacks never recapture live history.
            decode_service=capture_decode_decision_profile(self,partitions=partitions,
                batch_size=batch_size,percentile=service_percentile,
                min_samples=min_samples,timing_basis='submission_service')
        if capture_execution_stamp(self)!=stamp:
            return None
        return dict(source_execution=stamp,partitions=partitions,decode_batch_size=batch_size,
                    service_profile=ann,prefill_profile=prefill,decode_profile=decode,
                    capacity_profile=capacity,decode_service_profile=decode_service)

    def _capture_rag_cache_eviction_work(self, *, requested_tokens, key,
            source_execution, window_started_at, frozen_profile, depends_on=()):
        """Bind current-tree read-only reclaim preview to a frozen host cost.

        Call only at a serialized forecast boundary. No new compute/HBM plan,
        cache mutation or promise about future trees; this describes current
        cache state, not a request that has not finished or inserted yet.
        """
        cache=getattr(self,'tree_cache',None)
        if (capture_execution_stamp(self)!=source_execution or type(cache) is not RadixCache
                or cache.disable or getattr(self,'page_size',None)!=1):
            return None
        preview=cache.peek_eviction(requested_tokens)
        if preview is None or not preview['sufficient']:
            return None
        duration=frozen_profile.query_ms(requested_tokens) if frozen_profile is not None else None
        offset=(time.perf_counter()-window_started_at)*1000.
        if not math.isfinite(offset) or offset<0:
            raise ValueError('Common eviction prediction window required')
        operation=build_prospective_cache_eviction(key=key,requested_tokens=requested_tokens,
            predicted_freed_tokens=preview['freed_tokens'],eligible_tokens=preview['freed_tokens'],
            duration_ms=duration,ready_ms=offset,depends_on=depends_on)
        if capture_execution_stamp(self)!=source_execution:
            return None
        return operation

    def _capture_rag_prospective_completion_policy(self):
        """Read verified cache completion semantics without touching tokens/KV."""
        cache=getattr(self,'tree_cache',None)
        spec=getattr(self,'spec_algorithm',None)
        if (type(cache) is not RadixCache or getattr(self,'page_size',None)!=1
                or getattr(self,'pp_size',None)!=1
                or getattr(self,'enable_overlap',True)
                or not callable(getattr(spec,'is_none',None)) or not spec.is_none()
                or type(cache.disable) is not bool):
            return None
        return 'disabled_unit_page' if cache.disable else 'radix_unit_page_unknown_prefix'

    def _build_rag_predicted_cache_eviction(self, projection, prediction, at_ms,
            requested_tokens, key, *, work_inputs):
        """One allocator-triggered forecast call with frozen tree and timing."""
        tree=self._query_rag_prospective_evictable(projection,prediction,at_ms,
            work_inputs=work_inputs,return_model=True)
        profile=work_inputs.get('eviction_profile')
        if tree is None or profile is None:return None
        _,selected,freed=simulate_radix_eviction(tree,requested_tokens)
        duration=profile.query_ms(requested_tokens)
        return build_prospective_cache_eviction(key=key,requested_tokens=requested_tokens,
            predicted_freed_tokens=freed,eligible_tokens=radix_evictable_tokens(tree),
            duration_ms=duration,ready_ms=at_ms)

    def _query_rag_prospective_evictable(self, projection, prediction, at_ms, *, work_inputs,
                                       return_model=False):
        """Replay supported new-request cache lifecycle from the common origin.

        Never substitute the start-of-window evictable count after insertion
        or completion. Explicit eviction selections are checked against the
        evolved tree. Existing-request insertion/collision still requires
        its full transition rather than retaining a stale origin.
        """
        if work_inputs['mode']!='full':
            raise ValueError('Cache forecast belongs to Full prediction')
        if type(return_model) is not bool:raise ValueError('Explicit cache query result required')
        stamp,refs=work_inputs['common'][:2]
        if capture_execution_stamp(self)!=stamp:return None
        policy=work_inputs.get('completion_policy')
        if policy=='disabled_unit_page':return None if return_model else 0
        if policy!='radix_unit_page_unknown_prefix':return None
        model=work_inputs.get('eviction_model');clock=work_inputs.get('eviction_clock')
        if model is None or clock is None or at_ms<clock.observed_offset_ms:return None
        shapes=describe_prospective_prefill(stamp.frontiers,refs,
            tuple(op for op in projection.operations if op.stage=='prepare'))
        if shapes is None:return None
        new_ids={shape.request_id for shape in shapes}
        ends={key:end for key,start,end in prediction.intervals}
        intervals={key:(start,end) for key,start,end in prediction.intervals}
        evictions=[]
        for op in projection.operations:
            if op.stage=='cache_evict' and op.key in intervals:
                shape=op.service_shape
                if not isinstance(shape,ProspectiveCacheEvictionShape):return None
                start,end=intervals[op.key]
                evictions.append((start,end,op.key,shape.requested_tokens,shape.predicted_freed_tokens))
            if (op.stage=='prefill_result' and ends.get(op.key,float('inf'))<=at_ms
                    and any(c.work=='decode' and c.request not in new_ids for c in op.credits)):
                return None
        # Existing decode completions can only release captured request locks
        # and make additional completed tokens evictable. Retaining the
        # frozen locks/tree is therefore a conservative lower bound. Existing
        # prefill results above remain unsupported because insertion/collision
        # can reduce the evictable set and must be modeled explicitly.
        events=derive_prospective_radix_events(shapes,stamp.frontiers,projection.operations,
            prediction,horizon_ms=max(at_ms,clock.observed_offset_ms,1.e-9))
        evolved=replay_radix_timed_evictions(model,clock,events,evictions,through_ms=at_ms)
        if evolved is None:return None
        if capture_execution_stamp(self)!=stamp:return None
        return evolved if return_model else radix_evictable_tokens(evolved)

    def _capture_rag_prospective_settings(self, *, source_execution):
        """Freeze native layout/model/admission limits, not a memory proposal."""
        if capture_execution_stamp(self)!=source_execution:return None
        controller=getattr(self,'shared_hbm_controller',None)
        native=getattr(getattr(controller,'pool',None),'kv_layout',None)
        if native is None:return None
        layout={key:getattr(native,key,None) for key in
                ('static_capacity_tokens','tokens_per_extent','total_extents')}
        if (any(type(n) is not int or n<0 for n in layout.values()) or layout['tokens_per_extent']==0):
            return None
        total=getattr(getattr(self,'model_config',None),'num_hidden_layers',None)
        inputs=getattr(self,'max_prefill_tokens',None)
        reader=getattr(self,'get_num_allocatable_reqs',None)
        if not callable(reader):return None
        running_limit=reader(0) # configured ceiling, not source-only free headroom
        if any(type(n) is not int or n<=0 for n in (total,inputs,running_limit,CLIP_MAX_NEW_TOKENS)):
            return None
        if capture_execution_stamp(self)!=source_execution:return None
        return dict(layout=layout,total_layers=total,input_token_budget=inputs,
                    max_running_requests=running_limit,clip_max_new_tokens=CLIP_MAX_NEW_TOKENS)

    def _capture_rag_running_reservation_rule(self, *, source_execution):
        """Freeze native prefill-admission offsets, not guaranteed decode KV.

        Match PrefillAdder's aggregate floating-point rule without rounding
        every request or subtracting a prepared iteration from emitted output.
        It is an admission constraint separate from physical reservations.
        """
        if capture_execution_stamp(self)!=source_execution:return None
        ratio=getattr(self,'new_token_ratio',None)
        if (type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio<0):return None
        requests=tuple(getattr(getattr(self,'running_batch',None),'reqs',()) or ())
        observed={f.request_id:f for f in source_execution.frontiers if f.stage=='decode'}
        if len(requests)!=len(observed) or {str(req.rid) for req in requests}!=set(observed):return None
        remaining=[]
        for req in requests:
            rid=str(req.rid);limit=getattr(req.sampling_params,'max_new_tokens',None)
            emitted=len(req.output_ids)
            if (type(limit) is not int or limit<emitted
                    or observed[rid].generated_tokens!=emitted
                    or observed[rid].remaining_output_limit!=limit-emitted):return None
            remaining.append((rid,limit-emitted))
        if capture_execution_stamp(self)!=source_execution:return None
        return dict(ratio=ratio,clip_max_new_tokens=CLIP_MAX_NEW_TOKENS,
                    remaining_limits=tuple(remaining),
                    initial_offset=sum(min(left,CLIP_MAX_NEW_TOKENS)*ratio for rid,left in remaining))

    def _capture_rag_decode_kv_coordinates(self, *, source_execution, completion_policy):
        """Capture allocated sequence endpoints without reading GPU tensors.

        These are per-owner coordinates for subsequent allocation DELTAS, not
        disjoint physical byte counts. Shared prefix endpoints must never be
        summed to infer occupancy or returned as free capacity on completion.
        Ordinary non-overlap decoding stores prompt+emitted-1 positions; the
        verified preparation receipt adds the allocated, not yet emitted one.
        """
        if completion_policy not in ('disabled_unit_page','radix_unit_page_unknown_prefix'):
            return None
        if capture_execution_stamp(self)!=source_execution:return None
        batch=getattr(self,'running_batch',None)
        requests=tuple(getattr(batch,'reqs',()) or ())
        observed={f.request_id:f for f in source_execution.frontiers if f.stage=='decode'}
        ids=tuple(str(req.rid) for req in requests)
        if len(set(ids))!=len(ids) or set(ids)!=set(observed):return None
        if not requests:return ()
        receipt=getattr(self,'rag_joint_prepared_decode',None)
        prepared=False
        if receipt is not None:
            owner,slots,members=receipt
            mode=getattr(batch,'forward_mode',None)
            if (owner is not batch or slots is not getattr(batch,'out_cache_loc',None)
                    or tuple(str(rid) for rid in members)!=ids or mode is None
                    or not mode.is_decode() or slots is None or slots.numel()!=len(ids)):
                return None
            prepared=True
        result=[]
        for req,rid in zip(requests,ids):
            f=observed[rid]
            prompt=len(req.origin_input_ids);emitted=len(req.output_ids)
            if (prompt<=0 or emitted<=0 or f.prompt_tokens!=prompt
                    or f.generated_tokens!=emitted):return None
            result.append((rid,prompt+emitted-1+int(prepared)))
        # Native host bookkeeping independently detects a stale or missing
        # preparation receipt. Do not synchronize seq_lens from the device.
        if getattr(batch,'seq_lens_sum',None)!=sum(end for rid,end in result):return None
        if capture_execution_stamp(self)!=source_execution:return None
        return tuple(result)

    def _capture_rag_prefill_kv_coordinates(self, *, source_execution, completion_policy):
        """Read prepared prefill owner endpoints, never infer physical free bytes.

        Native preparation has already charged these allocations to the pool.
        Endpoints (including shared prefix) are for future allocation deltas,
        not additive occupancy or a promise of reclaim on completion.
        """
        if completion_policy not in ('disabled_unit_page','radix_unit_page_unknown_prefix'):
            return None
        if capture_execution_stamp(self)!=source_execution:return None
        observed={f.request_id:f for f in source_execution.frontiers
                  if f.stage in ('prefill_active','prefill_merge_pending')}
        result=[]
        for name,stage in (('split_prefill_batch','prefill_active'),
                           ('rag_prefill_merge_pending_batch','prefill_merge_pending')):
            batch=getattr(self,name,None)
            requests=tuple(getattr(batch,'reqs',()) or ())
            if not requests:continue
            endpoints=[];extend=0
            for req in requests:
                rid=str(req.rid);front=observed.get(rid)
                fill=getattr(req,'fill_ids',None);count=getattr(req,'extend_input_len',None)
                if (front is None or front.stage!=stage or fill is None or not len(fill)
                        or getattr(req,'req_pool_idx',None) is None
                        or type(count) is not int or not 0<=count<=len(fill)
                        or front.prompt_tokens!=len(req.origin_input_ids)
                        or (stage=='prefill_active' and front.prefill_context_tokens!=len(fill))):
                    return None
                endpoints.append((rid,len(fill)));extend+=count
            slots=getattr(batch,'out_cache_loc',None)
            if (slots is None or slots.numel()!=extend
                    or getattr(batch,'seq_lens_sum',None)!=sum(end for rid,end in endpoints)):
                return None
            result.extend(endpoints)
        if len(dict(result))!=len(result) or set(dict(result))!=set(observed):return None
        if capture_execution_stamp(self)!=source_execution:return None
        return tuple(result)

    def _build_rag_prospective_capacity_budget(self, *, snapshot, projection, work_inputs,
            reservations, physical_allocations=None, layout, kv_key):
        """One fitted-lease budget from the same-event pool and owner capture.

        This consumes frozen ownership, not a second live capture or HBM plan.
        A keep lease uses only the published pool. A target uses only its own
        native fitted publication event, never the other proposal's growth.
        """
        if work_inputs['mode']!='full':
            raise ValueError('Prospective joint scoring belongs to Full only')
        stamp=work_inputs['common'][0]
        if capture_execution_stamp(self)!=stamp:return None
        observation=work_inputs.get('physical_pool')
        if observation is None:return None
        reservations=tuple(reservations)
        decode_ids={f.request_id for f in stamp.frontiers if f.stage=='decode'}
        coordinates=work_inputs.get('decode_kv_coordinates')
        if decode_ids and (coordinates is None or {rid for rid,end in coordinates}!=decode_ids):
            return None
        prefill_ids={f.request_id for f in stamp.frontiers
                     if f.stage in ('prefill_active','prefill_merge_pending')}
        prefill_coordinates=work_inputs.get('prefill_kv_coordinates')
        if prefill_ids and (prefill_coordinates is None
                or {rid for rid,end in prefill_coordinates}!=prefill_ids):return None
        coordinates=(*(coordinates or ()),*(prefill_coordinates or ()))
        if physical_allocations is None:
            physical_allocations=coordinates or ()
        physical_allocations=tuple(physical_allocations)
        allocated=dict(physical_allocations)
        if any(allocated.get(rid)!=end for rid,end in (coordinates or ())):
            return None
        if not {rid for rid,total in reservations}<={r.key for r in snapshot.requests}:
            raise ValueError('Only observed snapshot owners can reserve KV')
        if projection.memory is snapshot.memory:
            publications=()
        else:
            publications=derive_prospective_capacity_publications(projection.memory,observation,
                event_id=snapshot.event_id,kv_key=kv_key,**dict(layout))
            if publications is None:return None
        budget=make_prospective_capacity_budget(observation,reservations=reservations,
            physical_allocations=physical_allocations,capacity_publications=publications)
        if capture_execution_stamp(self)!=stamp:return None
        return budget

    def _capture_rag_prospective_pool(self, *, source_execution, completion_policy):
        """Read current published pool only, without evicting or polling maps.

        Keep and target close over this same observation. Target capacity must
        later be derived from its own fitted transition and readiness; this
        method never turns a planned KV growth into free physical tokens.
        """
        if completion_policy not in ('disabled_unit_page','radix_unit_page_unknown_prefix'):
            return None
        allocator=getattr(self,'token_to_kv_pool_allocator',None)
        request_pool=getattr(self,'req_to_token_pool',None)
        cache=getattr(self,'tree_cache',None)
        readers=(getattr(allocator,'current_mapped_capacity',None),
                 getattr(allocator,'available_size',None),
                 getattr(request_pool,'available_size',None))
        if not all(callable(reader) for reader in readers):
            return None
        if capture_execution_stamp(self)!=source_execution:
            return None
        published,free,slots=(reader() for reader in readers)
        evictable=0
        if completion_policy=='radix_unit_page_unknown_prefix':
            reader=getattr(cache,'evictable_size',None)
            if not callable(reader):return None
            evictable=reader()
        observed=ProspectivePoolObservation(published,free,evictable,slots)
        if capture_execution_stamp(self)!=source_execution:
            return None
        return observed

    def _capture_rag_joint_work_inputs(self, *, mode, window_started_at,
                                     inflight_retrieval_inputs=None,
                                     forecast_cache_tree=False, engineering_unpriced_work=False,
                                     **capture_inputs):
        """Capture common frontiers plus external preparation for one decision.

        The same result must be closed over by target and keep builders. This
        does not select compute, allocate HBM, infer a future prompt, or make
        an absent profile valid. Independent constructs the same safety work
        but never captures its Full-only turnaround timing prior.
        """
        if mode not in ('full','independent_adaptive'):
            raise ValueError('Explicit joint policy required')
        if type(forecast_cache_tree) is not bool:
            raise ValueError('Explicit cache forecast flag required')
        if type(engineering_unpriced_work) is not bool:
            raise ValueError('Explicit engineering work-capture flag required')
        if engineering_unpriced_work:
            receipt=getattr(self,'rag_joint_bootstrap_receipt',{})
            if receipt.get('engineering_only') is not True or receipt.get('formal_ready') is not False:
                raise ValueError('Unpriced Full work requires explicit engineering bootstrap')
        timing=mode=='full' and not engineering_unpriced_work
        if 'timing' in capture_inputs:
            raise ValueError('Timing mode is determined by the joint policy')
        common=self._capture_rag_joint_inputs(window_started_at=window_started_at,
            timing=timing,**capture_inputs)
        if common is None:
            return None
        stamp=common[0]
        inflight_work=None
        if inflight_retrieval_inputs is not None:
            inflight_work=self._capture_rag_inflight_retrieval_work(source_execution=stamp,
                cpu_plan=common[3],window_started_at=window_started_at,timing=timing,
                **dict(inflight_retrieval_inputs))
            if inflight_work is None:
                return None
        observed_at=time.perf_counter()
        preparation=build_observed_input_preparation(stamp.frontiers,
            observed_at=observed_at,window_started_at=window_started_at)
        if preparation is None:
            return None
        needs_preparation=bool(preparation) or any(f.stage in ('ann_pending','ann_inflight')
            and f.preparation_follows is True for f in stamp.frontiers)
        profile=(capture_input_preparation_profile(self)
                 if timing and needs_preparation else None)
        handoff_profile=capture_prefill_handoff_profile(self) if timing else None
        handoff_batch_members=self._capture_rag_handoff_batch_members(source_execution=stamp)
        completion_policy=self._capture_rag_prospective_completion_policy()
        physical_pool=self._capture_rag_prospective_pool(
            source_execution=stamp,completion_policy=completion_policy)
        decode_kv_coordinates=self._capture_rag_decode_kv_coordinates(
            source_execution=stamp,completion_policy=completion_policy)
        prefill_kv_coordinates=self._capture_rag_prefill_kv_coordinates(
            source_execution=stamp,completion_policy=completion_policy)
        running_reservation_rule=self._capture_rag_running_reservation_rule(source_execution=stamp)
        prospective_settings=self._capture_rag_prospective_settings(source_execution=stamp)
        model_reader=getattr(getattr(self,'tree_cache',None),'capture_eviction_model',None)
        locked_requests=[];seen_owners=set()
        for name in (('running_batch','split_prefill_batch','rag_prefill_merge_pending_batch')
                     if forecast_cache_tree and mode=='full' else ()):
            for req in (getattr(getattr(self,name,None),'reqs',()) or ()):
                if id(req) not in seen_owners:
                    locked_requests.append(req);seen_owners.add(id(req))
        eviction_model=(model_reader(requests=tuple(locked_requests)) if forecast_cache_tree
                        and mode=='full' and completion_policy=='radix_unit_page_unknown_prefix'
                        and callable(model_reader) else None)
        eviction_clock=None
        if eviction_model is not None:
            perf_before=time.perf_counter()
            monotonic_now=time.monotonic()
            perf_after=time.perf_counter()
            eviction_clock=RadixForecastClock(window_started_at,perf_before,monotonic_now,perf_after)
        eviction_reader=getattr(getattr(self,'tree_cache',None),'eviction_planning_snapshot',None)
        eviction_profile=(capture_radix_eviction_profile(eviction_reader())
            if forecast_cache_tree and mode=='full' and completion_policy=='radix_unit_page_unknown_prefix'
            and callable(eviction_reader) else None)
        if capture_execution_stamp(self)!=stamp:
            return None
        return dict(common=common,preparation_operations=preparation,
                    handoff_profile=handoff_profile,
                    handoff_batch_members=handoff_batch_members,
                    handoff_batches=(FrozenHandoffBatches(handoff_batch_members)
                                     if handoff_batch_members is not None else None),
                    physical_pool=physical_pool,
                    decode_kv_coordinates=decode_kv_coordinates,
                    prefill_kv_coordinates=prefill_kv_coordinates,
                    running_reservation_rule=running_reservation_rule,
                    prospective_settings=prospective_settings,
                    completion_policy=completion_policy,
                    eviction_profile=eviction_profile,
                    eviction_model=eviction_model,
                    eviction_clock=eviction_clock,
                    inflight_retrieval_work=inflight_work,
                    preparation_profile=profile,observed_at=observed_at,
                    window_started_at=window_started_at,mode=mode)

    def _initialize_rag_legacy_calibrated_predictor(self, bundle):
        """Attach accepted phase models while retaining dynamic-only policy."""
        if (getattr(self, "rag_policy_mode", None) not in ("full", "dynamic-only")
                or getattr(self, "rag_legacy_calibrated_bundle", None) is not None
                or getattr(self, "rag_joint_event_input_provider", None) is not None):
            raise ValueError("Legacy calibrated predictor requires a fresh legacy frontend runtime")
        stamp=capture_execution_stamp(self)
        if (stamp.frontiers or any(getattr(self,name,None) for name in
                ("ann_search_pending","ann_search_inflight","waiting_queue"))):
            raise ValueError("Install calibrated predictor before serving work")
        from sglang.srt.multiplex.calibrated_predictor_bundle import CalibratedPredictorBundle
        if not isinstance(bundle,CalibratedPredictorBundle):
            raise ValueError("Accepted calibrated predictor bundle required")
        self.rag_legacy_calibrated_bundle=bundle
        from sglang.srt.multiplex.stage_predictor import CausalIterationOverheadGuard
        self.rag_legacy_iteration_guard=CausalIterationOverheadGuard()
        self.rag_legacy_calibrated_prediction_counts={}
        self.rag_legacy_calibrated_fallback_counts={}
        self.rag_legacy_calibrated_fallback_reason_counts={}

    def _initialize_rag_joint_runtime(self, *, provider, initial_compute,
                                      allocation_by_stream):
        """Install at an idle startup boundary with explicit configured limits.

        This is not hot replacement or profile acceptance. The deployment must
        validate the provider's workload/profile contract before calling here.
        Existing HBM and compute partitions are retained; no work is launched.
        """
        if (getattr(self,'rag_policy_mode',None) not in ('full','independent_adaptive')
                or not callable(provider) or not isinstance(initial_compute,ComputeLease)):
            raise ValueError('Joint policy, provider and explicit initial lease required')
        if (getattr(self,'rag_joint_event_input_provider',None) is not None
                or getattr(self,'rag_joint_applied_compute',None) is not None
                or getattr(self,'rag_joint_pending_preparation',None) is not None):
            raise ValueError('Joint runtime already initialized or transitioning')
        if (any(type(v) is not int or v<0 for v in
                (initial_compute.ann_batch,initial_compute.prefill_layers))
                or initial_compute.prefill_contract is not None):
            raise ValueError('Explicit nonnegative startup limits without request contract required')
        mapping=dict(allocation_by_stream)
        if mapping.get(get_current_stream_idx())!=initial_compute.allocation:
            raise ValueError('Initial lease must retain actual partition')
        stamp=capture_execution_stamp(self)
        if stamp.frontiers or any(getattr(self,name,None) for name in
                ('ann_search_pending','ann_search_inflight','rag_ann_collection_pending',
                 'waiting_queue','rag_prefill_bundle_inflight_event')):
            raise ValueError('Install before requests and in-flight work')
        for name in ('running_batch','split_prefill_batch','rag_prefill_merge_pending_batch'):
            if getattr(getattr(self,name,None),'reqs',()):
                raise ValueError('Install before active batches')
        if stamp!=capture_execution_stamp(self):
            raise ValueError('Execution changed during initialization')
        self.rag_joint_applied_compute=initial_compute
        self.rag_joint_event_input_provider=provider  # Publish gate last.

    def _capture_rag_applied_compute(self, allocation_by_stream):
        """Read last activated fitted limits, never a proposal or closed gate."""
        current=getattr(self,'rag_joint_applied_compute',None)
        if current is None:
            return None  # Initial installation needs explicit observed bootstrap.
        if allocation_by_stream.get(get_current_stream_idx())!=current.allocation:
            return None
        return current

    def _bind_rag_observed_prefill_model(self, *, source_execution, queued_budget,
                                        candidate_streams, predictor, expected_identity,
                                        corrections=None,
                                        expected_correction_identity=(),
                                        overlap_class="solo",
                                        nearest_domain_fallback=False):
        """Bind current prepared/queued cohort without another runtime capture."""
        if capture_execution_stamp(self) != source_execution:
            return None
        batch = getattr(self, 'split_prefill_batch', None)
        if batch is not None:
            contract = capture_prepared_prefill_contract(batch)
            if contract is None:
                return None
            new = tuple(row[3] for row in contract[1])
            cached = tuple(row[2]-row[3] for row in contract[1])
        elif queued_budget is not None:
            selected = queued_budget.propose().selection.request_ids
            by_id = {s.request_id:s for s in queued_budget.shapes}
            if not selected or any(rid not in by_id for rid in selected):
                return None
            new = tuple(by_id[rid].extend_tokens for rid in selected)
            cached = tuple(by_id[rid].context_tokens-by_id[rid].extend_tokens for rid in selected)
        else:
            return None
        result = bind_prefill_compute_profile(predictor, expected_identity=expected_identity,
            partition_to_sm={idx:self.sm_counts[idx][0] for idx in candidate_streams
                             if self.sm_counts[idx][0] > 0},
            total_layers=self.model_config.num_hidden_layers, new_tokens=new,
            cached_tokens=cached, corrections=corrections,
            expected_correction_identity=expected_correction_identity,
            overlap_class=overlap_class,
            nearest_domain_fallback=nearest_domain_fallback)
        return result if capture_execution_stamp(self) == source_execution else None

    def _capture_rag_joint_decision_context(self, *, mode, current_event_id,
            window_started_at, capture_inputs, current_compute=None, allocation_by_stream,
            decode_growth_tokens, reserve_tokens, compute_settings, backing_inputs,
            target_decode_growth_forecast=None, target_decode_growth_inputs=None,
            compute_profile_capture_settings=None,
            prefill_stage_model_inputs=None,
            decode_stage_model_inputs=None,
            ann_pressure_inputs=None,
            transition_profile_capture_settings=None,
            transition_profiles=None, release_profile=None, max_inflight=None,
            busy_until=()):
        """Assemble common observed inputs and current-only keep exactly once.

        Settings contain frozen compute-profile/limit inputs, not a second
        request capture. Actual pending task owners, queued capacity, source
        stamp, window and allocation map are shared by target and keep. This
        prepares contexts only: no target search, elastic plan or dispatch.
        Work/timing callbacks still need to be bound to the returned work input.
        """
        if max_inflight is None:
            # The ANN arena is bound once per lease and must cover every launch
            # the pipeline may overlap. The launch path sizes its request from
            # this same configured depth, so the plan must fit the same number;
            # a fixed plan depth silently disagrees with every launch.
            max_inflight = max(1, int(getattr(self, 'rag_ann_max_inflight', 1)))
        if getattr(self,'rag_joint_pending_preparation',None) is not None:
            return None
        if ann_pressure_inputs is not None and decode_stage_model_inputs is None:
            raise ValueError('ANN pressure requires the common decode stage model')
        horizon=capture_inputs.get('horizon_ms')
        if (isinstance(horizon,bool) or not isinstance(horizon,(int,float))
                or not math.isfinite(horizon) or horizon<=0):
            raise ValueError('Positive finite common prediction horizon required')
        if (target_decode_growth_inputs is not None
                and target_decode_growth_inputs.get('horizon_ms')!=horizon):
            raise ValueError('Decode capacity and urgency must share the same horizon')
        if ((target_decode_growth_forecast is None)==(target_decode_growth_inputs is None)
                or (target_decode_growth_forecast is not None and not callable(target_decode_growth_forecast))):
            raise ValueError('Choose frozen capacity inputs or one selected-compute forecast')
        allocations=dict(allocation_by_stream)
        idx=get_current_stream_idx()
        if current_compute is None:
            current_compute=self._capture_rag_applied_compute(allocations)
            if current_compute is None:
                return None
        if allocations.get(idx)!=current_compute.allocation:
            return None
        reserved={'source_execution','queued_budget','allocation_by_stream','pending'}
        if set(compute_settings)&reserved or 'ann_tasks' in backing_inputs:
            raise ValueError('Runtime-owned inputs cannot be overridden by settings')
        frozen_profiles=None
        compute_settings=dict(compute_settings)
        if compute_profile_capture_settings is not None:
            owned={'service_profile','prefill_profile','decode_profile','decode_batch_size'}
            if (owned&set(compute_settings) or target_decode_growth_inputs is None
                    or 'capacity_profile' in target_decode_growth_inputs):
                raise ValueError('Captured compute/capacity profiles cannot be overridden')
            frozen_profiles=self._capture_rag_joint_compute_profiles(
                partitions=tuple(compute_settings['candidate_streams']),
                **dict(compute_profile_capture_settings))
            if frozen_profiles is None:
                return None
            compute_settings.update({key:frozen_profiles[key] for key in owned})
        work=self._capture_rag_joint_work_inputs(mode=mode,
            window_started_at=window_started_at,**dict(capture_inputs))
        if work is None:
            return None
        stamp,refs,urgency,cpu,queued_shapes,queued_budget=work['common']
        if decode_stage_model_inputs is not None:
            cohort = tuple(f for f in stamp.frontiers if f.stage == 'decode')
            if cohort:
                if any(type(f.prompt_tokens) is not int or f.prompt_tokens <= 0
                       or type(f.generated_tokens) is not int or f.generated_tokens < 1
                       for f in cohort):
                    return None
                model_settings = dict(decode_stage_model_inputs)
                domain = model_settings.pop('workload_supported', None)
                if ann_pressure_inputs is not None:
                    if domain is not None or 'interference_query' in model_settings:
                        raise ValueError('Use pressure-conditioned table or legacy guard, not both')
                    model_settings['interference_query']=lambda *args:None
                elif domain is not None:
                    if 'interference_query' in model_settings:
                        raise ValueError('Choose captured service guard or supplied guard, not both')
                    service = (frozen_profiles or {}).get('decode_service_profile')
                    if service is None:
                        return None
                    model_settings['interference_query'] = bind_observed_decode_compute_interference(
                        service, frontiers=cohort, partitions=compute_settings['candidate_streams'],
                        workload_supported=domain)
                recovery_stream=compute_settings.get('decode_recovery_stream')
                decode_streams=tuple(dict.fromkeys(tuple(compute_settings['candidate_streams'])+
                    (() if recovery_stream is None else (recovery_stream,))))
                if (recovery_stream is not None and
                        (type(recovery_stream) is not int or recovery_stream<0 or
                         recovery_stream not in allocations or
                         self.sm_counts[recovery_stream][1]<=0)):
                    raise ValueError('Decode recovery requires an actual decode allocation')
                compute_settings['decode_profile'] = bind_decode_compute_profile(
                    partition_to_sm={idx:self.sm_counts[idx][1]
                        for idx in decode_streams if self.sm_counts[idx][1]>0},
                    cached_tokens=tuple(f.prompt_tokens+f.generated_tokens-1 for f in cohort),
                    **model_settings)
                if ann_pressure_inputs is not None:
                    table=ann_pressure_inputs['table']
                    identity=ann_pressure_inputs['expected_identity']
                    if not isinstance(table,FrozenANNPressureTable) or table.identity!=identity:
                        raise ValueError('Matching frozen ANN pressure table required')
                    outstanding=self._capture_rag_joint_ann_outstanding_credits(
                        partitions=compute_settings['candidate_streams'],
                        service_profile=compute_settings['service_profile'],source_execution=stamp)
                    if outstanding is None:
                        return None
                    # Decode-only starts only after the common transition path
                    # drains incompatible ANN work.  It grants no new ANN, so
                    # its target-state pressure is exactly zero rather than an
                    # unreadable cross-partition live credit.
                    if recovery_stream is not None:
                        outstanding=tuple(outstanding)+((recovery_stream,0.),)
                    guard=ann_pressure_inputs.get('causal_guard')
                    guard_identity=ann_pressure_inputs.get('expected_guard_identity',())
                    if (guard is None) != (not guard_identity):
                        raise ValueError('Causal guard and identity must be supplied together')
                    iteration_guard=ann_pressure_inputs.get('iteration_guard')
                    solo_envelope=ann_pressure_inputs.get('solo_envelope')
                    solo_identity=ann_pressure_inputs.get('expected_solo_identity',())
                    solo_partition_to_sm=ann_pressure_inputs.get('partition_to_sm',())
                    fullcard_envelope=ann_pressure_inputs.get('fullcard_envelope')
                    fullcard_identity=ann_pressure_inputs.get('expected_fullcard_identity',())
                    fullcard_partition_to_sm=ann_pressure_inputs.get(
                        'fullcard_partition_to_sm',())
                    if (iteration_guard is not None
                            and not callable(getattr(iteration_guard,'predict_total_ms',None))):
                        raise ValueError('Explicit causal iteration-overhead guard required')
                    # Freeze completed history once at the event boundary. Every
                    # candidate at this decision therefore sees the same causal
                    # floor; candidate ANN pressure remains the differentiator.
                    history=tuple(getattr(self,'_rag_dm2_window',())) if guard is not None else ()
                    fullcard_history=()
                    if fullcard_envelope is not None:
                        fullcard_partition=fullcard_partition_to_sm[0][0]
                        fullcard_history=tuple(
                            getattr(self,'_rag_dm2_by_stream',{}).get(fullcard_partition,())
                        )[-50:]
                    iteration_history=tuple(
                        getattr(self,'_rag_iteration_overhead_window',()))
                    if capture_execution_stamp(self)!=stamp:
                        return None
                    compute_settings['decode_profile']=PressureDecodeComputeProfile(
                        compute_settings['decode_profile'],table,identity,
                        tuple(f.prompt_tokens+f.generated_tokens-1 for f in cohort),outstanding,
                        guard,history,guard_identity,iteration_guard,
                        iteration_history,solo_envelope,solo_identity,
                        solo_partition_to_sm,fullcard_envelope,fullcard_identity,
                        fullcard_partition_to_sm,fullcard_history)
        if prefill_stage_model_inputs is not None:
            profile = self._bind_rag_observed_prefill_model(source_execution=stamp,
                queued_budget=queued_budget, candidate_streams=compute_settings['candidate_streams'],
                **dict(prefill_stage_model_inputs))
            has_prefill = (getattr(self, 'split_prefill_batch', None) is not None
                           or bool(getattr(self, 'waiting_queue', ())))
            if profile is None and has_prefill:
                return None
            compute_settings['prefill_profile'] = profile
        if mode=='full' and transition_profile_capture_settings is not None:
            if transition_profiles is not None or release_profile is not None:
                raise ValueError('Use captured transition profiles or supplied profiles, not both')
            settings=dict(transition_profile_capture_settings)
            transition_profiles=capture_joint_transition_profiles(self,**settings)
            if transition_profiles is None:
                return None
            release_profile=capture_ann_release_profile(self,**settings)
            if capture_execution_stamp(self)!=stamp:
                return None
        if frozen_profiles is not None:
            if frozen_profiles['source_execution']!=stamp:
                return None
            target_decode_growth_inputs={**target_decode_growth_inputs,
                'capacity_profile':frozen_profiles['capacity_profile']}
            if frozen_profiles['decode_batch_size']==0:
                if any(f.stage=='decode' for f in stamp.frontiers):
                    return None
                # No existing decode cohort needs incremental slots. New
                # prefill admission still supplies its own KV demand.
                target_decode_growth_forecast=lambda snapshot,proposed,diagnostic:()
                target_decode_growth_inputs=None
        if target_decode_growth_inputs is not None:
            target_decode_growth_forecast=bind_selected_decode_growth_forecast(
                tuple(f for f in stamp.frontiers if f.stage=='decode'),dict(refs.profiles),
                allocation_by_stream=allocations,**dict(target_decode_growth_inputs))
        pending=tuple(getattr(self,'ann_search_pending',()) or ())
        backing=dict(backing_inputs)
        snapshot=self._capture_rag_current_joint_snapshot(
            current_compute=current_compute,allocation_by_stream=allocations,
            queued_budget=queued_budget,pending=pending,
            continuation_settings={**{name:compute_settings[name] for name in (
                'service_profile','prefill_profile','decode_profile','decode_batch_size',
                'microbatch_limit','credit_ms','decode_target_ms','safety_margin_ms','bundle_limit')},
                'decode_recovery_stream':compute_settings.get('decode_recovery_stream')},
            decode_growth_tokens=decode_growth_tokens,reserve_tokens=reserve_tokens,
            source_execution=stamp,current_event_id=current_event_id,
            window_started_at=window_started_at,urgency=urgency,
            backing_inputs=backing,max_inflight=max_inflight,busy_until=tuple(busy_until))
        if (snapshot is None or capture_execution_stamp(self)!=stamp
                or get_current_stream_idx()!=idx):
            return None
        # Keep uses its current-partition bound above. The target bound must
        # wait for choose_compute; sharing keep's scalar would misprice demand
        # whenever the proposed partition changes decode's execution rate.
        demand=dict(queued_budget=queued_budget,reserve_tokens=reserve_tokens,
                    decode_growth_forecast=target_decode_growth_forecast)
        return dict(work_inputs=work,snapshot=snapshot,source_execution=stamp,horizon_ms=horizon,
            frozen_compute_profiles=frozen_profiles,
            source_stream_idx=idx,mode=mode,
            compute_inputs={**dict(compute_settings),'queued_budget':queued_budget,
                            'allocation_by_stream':allocations,'pending':pending},
            demand_inputs=demand,
            memory_inputs={'queued_budget':queued_budget,'reserve_tokens':reserve_tokens,
                'window_started_at':window_started_at,
                'backing_inputs':{**backing,'ann_tasks':pending},'max_inflight':max_inflight,
                'transition_profiles':transition_profiles if mode=='full' else None,
                'release_profile':release_profile if mode=='full' else None})

    def _capture_rag_joint_inputs(self, *, window_started_at: float,
                                  prompt_catalog, estimated_prompt_tokens,
                                  e2e_slo_ms: float, horizon_ms: float,
                                  offline_cpu_profile=None, timing=True):
        """Prepare common target/keep inputs at a serialized decision boundary.

        This is not a reservation or a complete joint snapshot: allocation,
        HBM ownership and worker progress still require commit-time validation.
        Do not retry capture/search inside this decision on an unsupported or
        stale observation. No launches, HBM planning or future arrivals here.
        Runtime admission clocks do not include unseen client-side waiting.
        """
        stamp = capture_execution_stamp(self)
        queued_budget = None
        if getattr(self, 'waiting_queue', ()):
            queued_budget = self._capture_rag_cached_prefill_budget()
            if queued_budget is None:
                return None
        queued_shapes = queued_budget.shapes if queued_budget is not None else ()
        references = select_reference_profiles(
            stamp.frontiers, dict(prompt_catalog),
            estimated_prompt_tokens=estimated_prompt_tokens)
        if references is None:
            return None
        urgency = freeze_request_urgency(
            stamp.frontiers, dict(references.profiles), now_s=window_started_at,
            e2e_slo_ms=e2e_slo_ms, horizon_ms=horizon_ms)
        if urgency is None:
            return None
        cpu_plan = self._capture_rag_inflight_cpu_plan(
            window_started_at=window_started_at,
            offline_profile=offline_cpu_profile, timing=timing)
        if cpu_plan is None or capture_execution_stamp(self) != stamp:
            return None
        return stamp, references, urgency, cpu_plan, queued_shapes, queued_budget

    def _capture_rag_full_cpu_work(self, batches, *, window_started_at):
        """Preserve separate full-CPU Future dependencies without timing guesses."""
        expected={batch.request_ids for batch in batches if batch.reason=='full_cpu_fallback'}
        if not expected:
            return (),(),0.
        groups=getattr(self,'ann_search_inflight',()) or ()
        if isinstance(groups,dict):groups=(groups,)
        now=time.perf_counter();offset=(now-window_started_at)*1000.
        if not math.isfinite(offset) or offset<0:
            raise ValueError('Common full-CPU observation window required')
        operations=[];dependencies=[];seen=set()
        for group in groups:
            tasks=tuple(group.get('tasks') or (group,))
            ids=tuple(str(task['request_id']) for task in tasks)
            if ids not in expected:continue
            if ids in seen:raise ValueError('Duplicate full-CPU batch ownership')
            seen.add(ids)
            observed=observe_full_cpu_fallback(group.get('result'),observed_at=now)
            if observed is None or observed.status in ('failed','cancelled'):
                return None
            keys=()
            if observed.status!='succeeded':
                key=f'observed:full_cpu:{observed.future_identity}'
                rows=sum(int(task['queries'].shape[0]) for task in tasks)
                operations.append(Operation(key,'cpu_retrieval',None,('cpu_retrieval',),
                    ready_ms=offset,service_shape=('full_cpu_fallback',rows,observed.status,observed.progress)))
                keys=(key,)
            dependencies.append((ids,keys))
        if seen!=expected:return None
        return tuple(operations),tuple(dependencies),offset

    def _capture_rag_inflight_cpu_plan(self, *, window_started_at: float,
                                      offline_profile=None, timing=True):
        """One detached worker observation shared by target and continuation.

        Scheduler serialization is required. This captures existing hybrid
        retrieval, observed full-CPU Futures and terminal prepared routes, not
        unfinished routing or future work. Full-CPU timing remains unknown.
        None means unsupported, never zero CPU cost.
        No dispatch, incomplete-future waits, profile updates or search here.
        Returns operations, request-group dependencies, and observation offset
        in the COMMON decision window. Collection must also respect this lower
        bound when a CPU search completed between window start and observation.
        An optional offline profile must already have been restored against the
        current deployment identity. It remains fixed for this decision; live
        samples are not silently merged into a calibrated prior. The same plan
        is reused for target and keep. No mode-specific CPU estimate is allowed.
        """
        if not math.isfinite(window_started_at):
            raise ValueError('Finite common decision clock required')
        if type(timing) is not bool:
            raise ValueError('Explicit CPU timing mode required')
        if (timing and offline_profile is not None
                and not isinstance(offline_profile,
                                   (FrozenColdWorkerProfile, FrozenColdWorkerLinearProfile))):
            raise ValueError('Validated frozen CPU profile required')
        batches = (capture_inflight_cold_bindings(self)
                   + capture_pending_cold_bindings(self))
        if any(not batch.supported and batch.reason!='full_cpu_fallback' for batch in batches):
            return None
        full_cpu=(self._capture_rag_full_cpu_work(batches,window_started_at=window_started_at)
                  if any(batch.reason=='full_cpu_fallback' for batch in batches) else ((),(),0.))
        if full_cpu is None:return None
        batches=tuple(batch for batch in batches if batch.reason!='full_cpu_fallback')
        worker = getattr(getattr(self, 'ann_executor', None), '_cold_batcher', None)
        if worker is None:
            return None if any(b.searches for b in batches) else (full_cpu[0], (*full_cpu[1],*(
                (b.request_ids, ()) for b in batches)), full_cpu[2])
        observed = worker.planning_snapshot()
        offset_ms = (observed['captured_at'] - window_started_at) * 1000.0
        if not math.isfinite(offset_ms) or offset_ms < 0:
            raise ValueError('CPU observation precedes common decision window')
        profile = ((offline_profile if offline_profile is not None
                    else capture_cold_worker_profile(observed)) if timing else None)
        predicted = build_cold_worker_operations(observed, None,
                                                observation_offset_ms=offset_ms, timing=False)
        if predicted is None:
            return None
        operations, completion = predicted
        if timing:
            measured=[]
            for op in operations:
                duration=profile.query_remaining_ms(op.service_shape)
                if duration is not None and (not math.isfinite(duration) or duration<=0):
                    raise ValueError('Positive observed CPU remaining service required')
                measured.append(replace(op,duration_ms=duration))
            operations=tuple(measured)
        dependencies = []
        for batch in batches:
            keys = []
            for search in resolve_cold_bindings(batch, observed):
                if search.status == 'succeeded':
                    continue
                if search.status not in ('queued', 'active'):
                    return None
                key = completion.get((search.binding.future_identity,
                                      search.binding.enqueued_at))
                if key is None:
                    return None
                keys.append(key)
            dependencies.append((batch.request_ids, tuple(dict.fromkeys(keys))))
        return (*operations,*full_cpu[0]), (*dependencies,*full_cpu[1]), max(offset_ms,full_cpu[2])

    def _capture_rag_prefill_compute_proposal(
        self, prefill_cap_idx: int, adaptive_idx: int
    ):
        """Serialized mixed-prefill proposal entry; never applies a lease.

        Read actual prepared shape and freeze all profile queries once. This
        entry is shared by future Full/Independent adapters, not the old policy.
        The caller must already hold the scheduler decision boundary. Unsupported
        diagnostics are not an executable fallback lease.
        """
        batch = getattr(self, "split_prefill_batch", None)
        reqs = tuple(getattr(batch, "reqs", ()) or ())
        total = getattr(batch, "extend_num_tokens", None)
        lengths = tuple(getattr(req, "extend_input_len", None) for req in reqs)
        contexts = tuple(len(req.fill_ids) if getattr(req, "fill_ids", None) is not None
                         else None for req in reqs)
        contract = capture_prepared_prefill_contract(batch)
        valid = (contract is not None and bool(reqs) and type(total) is int and total > 0
                 and all(type(n) is int and type(c) is int and 0 < n <= c
                         for n, c in zip(lengths, contexts))
                 and sum(lengths) == total)
        if not valid:
            return prefill_cap_idx, {"available": False, "reason": "invalid_prepared_batch_shape"}
        decode_bs = self._rag_current_decode_batch_size()
        if decode_bs <= 0:
            return prefill_cap_idx, {"available": False, "reason": "no_active_decode_cohort"}
        partition_count = getattr(self, "real_sm_group_num", 0)
        if not 0 <= adaptive_idx < partition_count:
            raise ValueError("Reference partition outside current partition map")
        guard_percentile = min(100.0, max(0.001,
            float(getattr(self, "rag_prefill_decode_guard_percentile", 99.0))))
        prefill_profile = capture_prefill_layer_profile(self, percentile=95)
        decode_profile = capture_decode_decision_profile(
            self, partitions=range(partition_count), batch_size=decode_bs,
            percentile=guard_percentile,
        )
        selected, diagnostics = self._propose_rag_prefill_mixed_candidate(
            prefill_cap_idx, adaptive_idx, prefill_profile=prefill_profile,
            decode_profile=decode_profile,
        )
        if diagnostics is not None:
            diagnostics = {**diagnostics, "observed_decode_batch": decode_bs,
                           "prefill_contract": contract,
                           "prepared_extend_tokens": total,
                           "prepared_context_tokens": contexts,
                           "profile_source": "frozen_completed_observations"}
        return selected, diagnostics

    def _propose_rag_queued_prefill_compute(self, budget, prefill_cap_idx, adaptive_idx,
                                           *, prefill_profile, decode_profile,
                                           ann_credit_by_partition=None, decode_guard=None):
        """Evaluate one unallocated FCFS cohort, without preparing a batch.

        Profiles must be frozen by the common decision capture. This returns
        a compute proposal and its cohort demand, not an activated lease.
        """
        proposal = budget.propose()
        ids = proposal.selection.request_ids
        if not ids:
            return prefill_cap_idx, {'available': False, 'reason': 'empty_compute_cohort'}
        by_id = {shape.request_id: shape for shape in budget.shapes}
        cohort = tuple(by_id[rid] for rid in ids)
        selected, diagnostic = self._propose_rag_prefill_mixed_candidate(
            prefill_cap_idx, adaptive_idx, prefill_profile=prefill_profile,
            decode_profile=decode_profile, queued_cohort=cohort,
            ann_credit_by_partition=ann_credit_by_partition, decode_guard=decode_guard)
        if diagnostic is not None:
            diagnostic = {**diagnostic, 'queued_cohort_demand': proposal,
                'prefill_contract': (0, tuple((s.request_id,s.prompt_tokens,
                                               s.context_tokens,s.extend_tokens) for s in cohort))}
        return selected, diagnostic

    def _capture_rag_joint_ann_outstanding_credits(self, *, partitions, service_profile,
                                                  source_execution):
        """One conservative GPU ledger observation shared by all partitions.

        Poll each submitted batch once, then price that same observation for
        every candidate. A later GPU completion can only release this reserved
        credit; commit-time safety remains separate. Never cache across calls.
        """
        if not isinstance(service_profile, FrozenANNServiceProfile):
            raise ValueError('Frozen ANN service profile required')
        partitions = tuple(partitions)
        if any(type(p) is not int or p < 0 for p in partitions):
            raise ValueError('Explicit ANN partitions required')
        if source_execution != capture_execution_stamp(self):
            return None
        tasks = getattr(self, 'ann_search_inflight', ()) or ()
        tasks = (tasks,) if isinstance(tasks, dict) else tuple(tasks)
        active = []; seen = set(); supported = True
        for task in tasks:
            if id(task) in seen:
                raise ValueError('Duplicate in-flight ANN task')
            seen.add(id(task))
            poll = getattr(task.get('result'), 'gpu_poll', None)
            if poll is None:
                supported = False
            elif not poll():
                active.append((task.get('stream_idx'), self._ann_task_num_queries(task)))
        result = []
        for partition in partitions:
            value = 0.0 if supported else None
            for owner, rows in active:
                if value is None or owner != partition:
                    value = None
                    break
                service = service_profile.query_ms(partition, rows)
                if service is None:
                    value = None
                    break
                value += service
            result.append((partition, value))
        if source_execution != capture_execution_stamp(self):
            return None
        return tuple(result)


    def _capture_rag_joint_ann_outstanding_credit(self, *, partition, service_profile,
                                                 source_execution):
        """Reserve exact profiled service for GPU-incomplete submitted batches.

        Completion polls never wait for CPU retrieval or synchronize GPU events.
        Until GPU completion, reserve the full measured service: host launch age
        cannot be subtracted from CUDA service. This conservative credit ledger
        is not a remaining-time/quiescence prediction or a statistical bound.
        Other-partition readers require transition handling, not silent omission.
        """
        if not isinstance(service_profile, FrozenANNServiceProfile):
            raise ValueError('Frozen ANN service profile required')
        if type(partition) is not int or partition < 0:
            raise ValueError('Explicit ANN partition required')
        if source_execution != capture_execution_stamp(self):
            return None
        tasks = getattr(self, 'ann_search_inflight', ()) or ()
        if isinstance(tasks, dict):
            tasks = (tasks,)
        outstanding = 0.0
        seen = set()
        for task in tasks:
            if id(task) in seen:
                raise ValueError('Duplicate in-flight ANN task')
            seen.add(id(task))
            result = task.get('result')
            poll = getattr(result, 'gpu_poll', None)
            if poll is None:
                return None
            if poll():
                continue
            if task.get('stream_idx') != partition:
                return None
            rows = self._ann_task_num_queries(task)
            service = service_profile.query_ms(partition, rows)
            if service is None:
                return None
            outstanding += service
        if source_execution != capture_execution_stamp(self):
            return None
        return outstanding

    def _propose_rag_joint_ann_credit(self, *, partition, service_profile,
                                     pending, microbatch_limit, credit_ms, source_execution):
        """Bind the live GPU completion ledger to one read-only credit choice."""
        queued = self._capture_rag_joint_ann_outstanding_credit(
            partition=partition, service_profile=service_profile,
            source_execution=source_execution)
        return self._choose_rag_joint_ann_credit(partition=partition,
            service_profile=service_profile, pending=pending,
            microbatch_limit=microbatch_limit, credit_ms=credit_ms,
            queued_gpu_ms=queued, source_execution=source_execution)

    def _choose_rag_joint_ann_credit(self, *, partition, service_profile,
                                    pending, microbatch_limit, credit_ms,
                                    queued_gpu_ms, source_execution):
        """One compute-side prefix choice; no HBM fitting or legacy mutation.

        queued_gpu_ms is the frozen same-domain outstanding-service estimate,
        NOT total retrieval/CPU latency or host-clock quiescence. Unknown work
        returns unsupported. A credit hold returns zero, not an atomic bypass.
        The proposal's subsequent fitted shape must query this same profile.
        """
        if not isinstance(service_profile, FrozenANNServiceProfile):
            raise ValueError('Frozen ANN service profile required')
        if (type(partition) is not int or partition < 0
                or type(microbatch_limit) is not int or microbatch_limit < 0
                or not math.isfinite(credit_ms) or credit_ms < 0):
            raise ValueError('Explicit partition, row limit and finite GPU credit required')
        if source_execution != capture_execution_stamp(self):
            return None
        if queued_gpu_ms is None:
            return None
        if not math.isfinite(queued_gpu_ms) or queued_gpu_ms < 0:
            raise ValueError('Outstanding GPU service must be nonnegative')
        prefixes = []
        rows = 0
        for task in pending:
            count = self._ann_task_num_queries(task)
            if type(count) is not int or count <= 0:
                raise ValueError('Pending tasks require actual query counts')
            if rows + count > microbatch_limit:
                break
            rows += count
            prefixes.append(rows)
        for rows in reversed(prefixes):
            service = service_profile.query_ms(partition, rows)
            if service is None:
                # A smaller measured prefix is still a compute proposal;
                # no alternative memory plan is being evaluated here.
                continue
            if queued_gpu_ms + service <= credit_ms:
                if source_execution != capture_execution_stamp(self):
                    return None
                return rows, service
        if source_execution != capture_execution_stamp(self):
            return None
        if prefixes and not any(service_profile.query_ms(partition, n) is not None
                                for n in prefixes):
            return None
        return 0, 0.0

    def _choose_rag_joint_nonprefill_compute(self, *, candidate_streams, allocation_by_stream,
                                           service_profile, decode_profile, pending,
                                           microbatch_limit, credit_ms, decode_batch_size,
                                           decode_target_ms, safety_margin_ms, source_execution):
        """Compute-only selection when no prefill work is present.

        With pending ANN, maximize admitted queries per profiled GPU ms subject
        to the decode guard; ties prefer lower decode latency then current SMs.
        Without admitted ANN choose the fastest measured solo decode partition.
        HBM and transition utility are deliberately absent; Full evaluates its
        one resulting fitted joint proposal later. Both policies use this rule.
        """
        if getattr(self, 'split_prefill_batch', None) is not None or getattr(self, 'waiting_queue', ()):
            return None, {'available': False, 'reason': 'prefill_frontier_present'}
        if (type(decode_batch_size) is not int or decode_batch_size < 0
                or not math.isfinite(decode_target_ms) or decode_target_ms <= 0
                or not math.isfinite(safety_margin_ms) or safety_margin_ms < 0):
            raise ValueError('Explicit decode shape and latency guard required')
        streams = tuple(candidate_streams)
        if len(set(streams)) != len(streams) or any(type(i) is not int or i < 0 for i in streams):
            raise ValueError('Unique actual compute partitions required')
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        pending = tuple(pending)
        candidates = []
        active_ann = self._rag_ann_gpu_work_active()
        current = get_current_stream_idx()
        for idx in streams:
            selected = self._propose_rag_joint_ann_credit(partition=idx,
                service_profile=service_profile, pending=pending,
                microbatch_limit=microbatch_limit, credit_ms=credit_ms,
                source_execution=source_execution)
            if selected is None:
                continue
            rows, service = selected
            overlap = 'ann' if rows or active_ann else 'solo'
            decode_ms = ((decode_profile.query_pressure_ms(idx, overlap, decode_batch_size, service)
                          if callable(getattr(decode_profile,'query_pressure_ms',None)) else
                          decode_profile.query_batch_ms(idx, overlap, decode_batch_size))
                         if decode_batch_size else 0.)
            if (decode_ms is None or not math.isfinite(decode_ms) or decode_ms < 0
                    or (decode_batch_size and (decode_ms <= 0
                        or decode_ms + safety_margin_ms > decode_target_ms))):
                continue
            rate = rows / service if rows else 0.
            candidates.append(dict(stream_idx=idx, ann_rows=rows, ann_gpu_ms=service,
                                   decode_ms=decode_ms, ann_queries_per_ms=rate))
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        if not candidates:
            return None, {'available': False, 'reason': 'no_profiled_safe_compute'}
        chosen = max(candidates, key=lambda c: (c['ann_queries_per_ms'], -c['decode_ms'],
                                               c['stream_idx'] == current, -c['stream_idx']))
        idx = chosen['stream_idx']
        return ComputeLease(allocation_by_stream[idx], chosen['ann_rows'], 0), {
            'available': True, 'selected_safe': True, 'selected_stream_idx': idx,
            'candidates': tuple(candidates)}

    def _choose_rag_joint_prefill_without_decode(self, *, queued_budget, candidate_streams,
                                               allocation_by_stream, prefill_profile,
                                               service_profile, pending, microbatch_limit,
                                               credit_ms, bundle_limit, source_execution):
        """One bounded prefill proposal without an active decode batch.

        Use an identical layer bundle across candidates, prefer faster profiled
        prefill, then ANN throughput. Do not invent a decode reference for an
        empty batch. All submitted prefill must first complete; the common
        transition path handles in-flight work instead of resubmitting it.
        """
        if type(bundle_limit) is not int or bundle_limit <= 0:
            raise ValueError('Explicit positive prefill bundle limit required')
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        if self._decode_batch_active():
            return None, {'available': False, 'reason': 'decode_frontier_present'}
        total_layers = getattr(getattr(self, 'model_config', None), 'num_hidden_layers', None)
        if type(total_layers) is not int or total_layers <= 0:
            raise ValueError('Actual model layer count required')
        batch = getattr(self, 'split_prefill_batch', None)
        demand = None
        if batch is not None:
            contract = capture_prepared_prefill_contract(batch)
            completed = getattr(batch, 'rag_completed_prefill_layers', None)
            submitted = getattr(batch, 'split_index', None)
            if (contract is None or type(completed) is not int or type(submitted) is not int
                    or completed != submitted or not 0 <= submitted < total_layers):
                return None, {'available': False, 'reason': 'prefill_not_at_completed_boundary'}
            remaining = total_layers - submitted
        elif queued_budget is not None:
            demand = queued_budget.propose()
            by_id = {s.request_id: s for s in queued_budget.shapes}
            cohort = tuple(by_id[rid] for rid in demand.selection.request_ids)
            contract = (0, tuple((s.request_id, s.prompt_tokens, s.context_tokens,
                                  s.extend_tokens) for s in cohort))
            remaining = total_layers
        else:
            return None, {'available': False, 'reason': 'no_prefill_frontier'}
        if not contract[1]:
            return None, {'available': False, 'reason': 'empty_compute_cohort'}
        tokens = sum(shape[3] for shape in contract[1])
        layers = min(bundle_limit, remaining)
        candidates = []
        streams = tuple(candidate_streams)
        if len(streams) != len(set(streams)) or any(type(i) is not int or i < 0 for i in streams):
            raise ValueError('Unique actual prefill partitions required')
        pending = tuple(pending)
        for idx in streams:
            if self.sm_counts[idx][0] <= 0:
                continue
            layer_ms = prefill_profile.query_layer_ms(idx, tokens)
            if layer_ms is None or not math.isfinite(layer_ms) or layer_ms <= 0:
                continue
            ann = self._propose_rag_joint_ann_credit(partition=idx,
                service_profile=service_profile, pending=pending, microbatch_limit=microbatch_limit,
                credit_ms=credit_ms, source_execution=source_execution)
            if ann is None:
                continue
            rows, service = ann
            candidates.append(dict(stream_idx=idx, bundle_layers=layers, ann_rows=rows,
                predicted_bundle_ms=layers * layer_ms, ann_queries_per_ms=rows / service if rows else 0.))
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        if not candidates:
            return None, {'available': False, 'reason': 'prediction_unavailable'}
        current = get_current_stream_idx()
        chosen = min(candidates, key=lambda c: (c['predicted_bundle_ms'], -c['ann_queries_per_ms'],
                                               c['stream_idx'] != current, c['stream_idx']))
        idx = chosen['stream_idx']
        return ComputeLease(allocation_by_stream[idx], chosen['ann_rows'], layers, contract), {
            'available': True, 'selected_safe': True, 'selected_stream_idx': idx,
            'prefill_contract': contract, 'queued_cohort_demand': demand, 'candidates': tuple(candidates)}

    def _prepare_rag_joint_decision_event(self, *, evaluation_inputs, allocation_to_stream,
                                         ann_tasks=None, residency=None):
        """Evaluate once and prepare only the selected lease at this boundary.

        A non-ready result is not permission to use the rejected target or
        rerun compute selection. Native transition completion must arrive as
        a later event; caller retains compatible-work dispatch safeguards.
        This bridge does not construct missing runtime/profile inputs.
        """
        if getattr(self,'rag_joint_pending_preparation',None) is not None:
            raise ValueError('Resolve pending preparation before a new decision')
        decision=self._evaluate_rag_joint_event(**evaluation_inputs)
        if decision is None:
            return None,'stale'
        snapshot=evaluation_inputs['snapshot']
        if decision.event_id!=snapshot.event_id:
            raise ValueError('Selected decision belongs to another event')
        status=self._prepare_rag_selected_joint_lease(decision=decision,
            source_execution=evaluation_inputs['source_execution'],
            source_stream_idx=evaluation_inputs['source_stream_idx'],
            allocation_to_stream=allocation_to_stream,ann_tasks=ann_tasks,residency=residency,
            current_event_id=snapshot.event_id)
        if status in ('kv_submitted','kv_wait','compute_wait','workspace_submitted','workspace_wait'):
            self.rag_joint_pending_preparation=dict(decision=decision,
                source_execution=evaluation_inputs['source_execution'],
                source_stream_idx=evaluation_inputs['source_stream_idx'],
                allocation_to_stream=dict(allocation_to_stream),
                ann_tasks=None if ann_tasks is None else tuple(ann_tasks),residency=residency,
                current_event_id=snapshot.event_id)
        return decision,status

    def _resume_rag_joint_preparation(self):
        """Advance the same selected intent once, without any optimizer call.

        Execution changes require a real new decision. A compute switch alone
        may change the stream to the already selected target. This path covers
        pre-spill KV/compute/workspace waits, not replay of committed ANN work.
        """
        pending=getattr(self,'rag_joint_pending_preparation',None)
        if pending is None:
            return None,'absent'
        decision=pending['decision']
        current=get_current_stream_idx()
        target=pending['allocation_to_stream'][decision.lease.executable.allocation]
        if (capture_execution_stamp(self)!=pending['source_execution']
                or current not in (pending['source_stream_idx'],target)):
            self.rag_joint_pending_preparation=None
            return decision,'stale'
        status=self._prepare_rag_selected_joint_lease(**{**pending,'source_stream_idx':current})
        if status not in ('kv_submitted','kv_wait','compute_wait','workspace_submitted','workspace_wait'):
            self.rag_joint_pending_preparation=None
        return decision,status

    def _evaluate_rag_joint_event(self, *, snapshot, source_execution, source_stream_idx,
                                  mode, compute_inputs, demand_inputs, memory_inputs,
                                  operations_builder=None, timing_builder=None, decode_builder=None,
                                  joint_work_builder=None,
                                  prospective_inputs_builder=None,
                                  horizon_ms, switching_margin):
        """Runtime admission entry; no activation, dispatch, or stale retry.

        Caller must build snapshot/contexts from one serialized observation.
        A returned decision is still intent: native ownership/transition gates
        in _prepare_rag_selected_joint_lease remain mandatory before dispatch.
        """
        if (source_execution != capture_execution_stamp(self)
                or source_stream_idx != get_current_stream_idx()):
            return None
        compute_inputs = {**compute_inputs, 'source_execution': source_execution}
        demand_inputs = {**demand_inputs, 'source_execution': source_execution}
        memory_inputs = {**memory_inputs, 'source_execution': source_execution,
                         'source_stream_idx': source_stream_idx, 'current_event_id': snapshot.event_id}
        planner = RuntimeJointPlanner(self, snapshot, mode=mode,
            compute_inputs=compute_inputs, demand_inputs=demand_inputs, memory_inputs=memory_inputs,
            operations_builder=operations_builder, timing_builder=timing_builder,
            decode_builder=decode_builder,joint_work_builder=joint_work_builder,
            prospective_inputs_builder=prospective_inputs_builder)
        try:
            decision = schedule_joint_lease(snapshot, planner, mode=mode,
                horizon_ms=horizon_ms, switching_margin=switching_margin)
        except ContinuationUnavailable:
            # Async retrieval/result completion may invalidate capture during
            # read-only planning. Discard this event, never dispatch or retry
            # its proposals; a new real boundary will obtain a fresh snapshot.
            if (source_execution != capture_execution_stamp(self)
                    or source_stream_idx != get_current_stream_idx()):
                self.rag_joint_stale_evaluation_count=getattr(self,'rag_joint_stale_evaluation_count',0)+1
                return None
            raise  # A missing graph on an unchanged state is still a bug.
        self.rag_joint_evaluation_count=getattr(self,'rag_joint_evaluation_count',0)+1
        # Count evaluation outcomes, not activations: a safe unscored keep
        # must not look like a successfully priced joint decision in reports.
        reasons=dict(getattr(self,'rag_joint_decision_reason_counts',{}))
        reasons[decision.reason]=reasons.get(decision.reason,0)+1
        self.rag_joint_decision_reason_counts=reasons
        if (source_execution != capture_execution_stamp(self)
                or source_stream_idx != get_current_stream_idx()):
            self.rag_joint_stale_evaluation_count=getattr(self,'rag_joint_stale_evaluation_count',0)+1
            return None
        return decision

    def _capture_rag_current_joint_snapshot(self, *, current_compute, allocation_by_stream,
                                           memory_allocation=None, queued_budget, pending,
                                           continuation_settings,
                                           decode_growth_tokens, reserve_tokens,
                                           source_execution, current_event_id, window_started_at,
                                           urgency, ready=(), kv_key='joint:kv',
                                           scratch_key='joint:scratch', index_key='joint:index',
                                           busy_until=(), max_inflight=1, backing_inputs=None):
        """Assemble keep with current-only capacity fitting, never elastic search.

        ready/busy_until are caller-captured backing and lane observations from
        this same window. A pending native transition returns None: it does NOT
        authorize stopping existing decode or fabricating zero progress. The
        runtime must keep its compatible-work path until that state is modeled.
        """
        compute, diagnostic = self._capture_rag_current_compute_continuation(
            current_compute=current_compute, allocation_by_stream=allocation_by_stream,
            queued_budget=queued_budget, pending=pending, source_execution=source_execution,
            **continuation_settings)
        if compute is None:
            return None
        demand = self._derive_rag_selected_memory_demand(proposed_compute=compute,
            diagnostic=diagnostic, queued_budget=queued_budget,
            decode_growth_tokens=decode_growth_tokens, reserve_tokens=reserve_tokens,
            source_execution=source_execution)
        if demand is None:
            return None
        idx = get_current_stream_idx()
        allocator = self.token_to_kv_pool_allocator
        published = allocator.current_mapped_capacity()
        free = allocator.available_size()
        demand = fit_current_kv_continuation(demand,
            published_tokens=published, allocated_tokens=published-free)
        compute = demand.compute
        plan = self._plan_rag_joint_hbm_demand(demand, source_execution=source_execution,
            source_stream_idx=idx, current_event_id=current_event_id,
            window_started_at=window_started_at, max_inflight=max_inflight, continuation=True)
        if (plan is None or plan.status != 'available' or plan.kv.status != 'ready'
                or plan.ann is None or plan.ann.mapping_required):
            return None
        controller = getattr(self, 'shared_hbm_controller', None)
        if controller is None or not controller.joint_hbm_backing_ready(plan):
            return None
        routing_preview = None
        if backing_inputs is not None:
            if memory_allocation is not None or ready:
                raise ValueError('Use captured backing or explicit backing, not both')
            described = self._describe_rag_joint_backing(memory_plan=plan, ann_tasks=pending,
                source_execution=source_execution, current_event_id=current_event_id,
                window_started_at=window_started_at, prefill_layers=compute.prefill_layers,
                kv_key=kv_key, scratch_key=scratch_key, index_key=index_key, **backing_inputs)
            if described is None:
                return None
            backing, routing_preview = described
        else:
            keys = ([kv_key] if compute.prefill_layers else [])
            if plan.ann.fit.fitted_rows:
                keys.extend((scratch_key, index_key))
            backing = dict(allocation=memory_allocation, ready=ready, kv_key=kv_key,
                scratch_key=scratch_key, index_key=index_key, new_work_keys=keys)
        offset = (time.perf_counter() - window_started_at) * 1000.0
        if not math.isfinite(offset) or offset < 0:
            raise ValueError('Common observation window required')
        memory = self._materialize_rag_joint_memory_lease(proposed_compute=compute,
            memory_plan=plan, current_event_id=current_event_id,
            source_execution=source_execution, **backing,
            completion=JointTransitionCompletion(offset, offset, offset, offset))
        if memory is None or not memory.supported or source_execution != capture_execution_stamp(self):
            return None
        memory = replace(memory, routing_preview=routing_preview)
        return Snapshot(current_event_id, tuple(urgency.requests), compute, memory, tuple(busy_until))

    def _capture_rag_current_compute_continuation(self, *, current_compute,
                                                 allocation_by_stream, queued_budget,
                                                 pending, source_execution,
                                                 service_profile, prefill_profile,
                                                 decode_profile, decode_batch_size,
                                                 microbatch_limit, credit_ms,
                                                 decode_target_ms, safety_margin_ms,
                                                 bundle_limit,
                                                 decode_recovery_stream=None):
        """Refresh work under the current allocation; never search partitions.

        This constructs the snapshot's keep compute input, not a dispatch
        permission. The allocation is fixed, while ANN credit and the prefill
        bundle are recomputed from this event's frozen observations. In
        particular, an idle zero-work lease is not a permanent zero cap.
        Current HBM fitting and shared projection still follow.
        """
        if source_execution != capture_execution_stamp(self):
            return None, None
        idx = get_current_stream_idx()
        if allocation_by_stream[idx] != current_compute.allocation:
            return None, None
        prefill_sm, ann_decode_sm = self.sm_counts[idx]
        if (type(prefill_sm) is not int or prefill_sm < 0
                or type(ann_decode_sm) is not int or ann_decode_sm < 0):
            raise ValueError('Physical execution-domain SM counts required')
        if (any(type(v) is not int or v < 0 for v in
                (current_compute.ann_batch, current_compute.prefill_layers,
                 decode_batch_size, microbatch_limit, bundle_limit))
                or not math.isfinite(credit_ms) or credit_ms < 0
                or not math.isfinite(decode_target_ms) or decode_target_ms <= 0
                or not math.isfinite(safety_margin_ms) or safety_margin_ms < 0):
            raise ValueError('Explicit continuation limits and decode guard required')
        # The recovery endpoint is deliberately outside the profiled mixed
        # domain.  A continuation may refresh work limits on a mixed allocation,
        # but it must not turn zero-frontend recovery into unprofiled ANN+decode.
        recovery = idx == decode_recovery_stream
        selected = (None if ann_decode_sm == 0 or recovery else
            self._propose_rag_joint_ann_credit(partition=idx,
                service_profile=service_profile,pending=tuple(pending),
                microbatch_limit=microbatch_limit,credit_ms=credit_ms,
                source_execution=source_execution))
        rows, ann_ms = (0, 0.) if selected is None else selected
        layers = 0
        contract = None
        cohort = None
        batch = getattr(self, 'split_prefill_batch', None)
        if bundle_limit and prefill_sm > 0:
            total = getattr(getattr(self, 'model_config', None), 'num_hidden_layers', None)
            if type(total) is not int or total <= 0:
                raise ValueError('Actual model layer count required')
            if batch is not None:
                observed = capture_prepared_prefill_contract(batch)
                completed = getattr(batch, 'rag_completed_prefill_layers', None)
                submitted = getattr(batch, 'split_index', None)
                if (observed is not None and type(completed) is int and type(submitted) is int
                        and completed == submitted and 0 <= submitted < total):
                    contract = observed
                    layers = min(bundle_limit, total-submitted)
            elif queued_budget is not None:
                selection = queued_budget.select()  # CURRENT capacity, not propose().
                if selection.request_ids:
                    cohort = queued_budget.cohort_demand(selection)
                    shapes = queued_budget.shapes[:len(selection.request_ids)]
                    contract = (0, tuple((s.request_id, s.prompt_tokens, s.context_tokens,
                                          s.extend_tokens) for s in shapes))
                    layers = min(bundle_limit, total)
        # No decode can be harmed by making fixed-allocation progress. If this
        # exact shape is outside the utility model, the pricing path can return
        # an unscored continuation; the scheduler must not deadlock. With an
        # active decode batch, unknown/unsafe contention removes new grants.
        if decode_batch_size:
            def predict_decode(ann_rows, prefill_layers, service):
                overlap = ('ann_prefill' if ann_rows and prefill_layers else
                           'ann' if ann_rows else 'prefill' if prefill_layers else 'solo')
                query_pressure=getattr(decode_profile,'query_pressure_ms',None)
                return (query_pressure(idx,overlap,decode_batch_size,service)
                        if callable(query_pressure) else
                        decode_profile.query_batch_ms(idx,overlap,decode_batch_size))

            def decode_safe(value):
                return (value is not None and math.isfinite(value) and value>0
                        and value+safety_margin_ms<=decode_target_ms)

            safe=decode_safe(predict_decode(rows,layers,ann_ms))
            if not safe and rows and layers:
                rows=0;ann_ms=0.
                safe=decode_safe(predict_decode(rows,layers,ann_ms))
            if not safe and layers:
                layers=0;contract=None;cohort=None
                safe=decode_safe(predict_decode(rows,layers,ann_ms))
            if not safe:
                rows=0
        if source_execution != capture_execution_stamp(self):
            return None, None
        return ComputeLease(current_compute.allocation, rows, layers, contract), {
            'queued_cohort_demand': cohort, 'continuation': True}

    def _choose_rag_joint_compute(self, *, queued_budget, candidate_streams,
                                 allocation_by_stream, prefill_profile, decode_profile,
                                 service_profile, pending, microbatch_limit, credit_ms,
                                 decode_batch_size, decode_target_ms, safety_margin_ms,
                                 bundle_limit, prefill_cap_idx, adaptive_idx, source_execution,
                                 decode_recovery_stream=None):
        """Dispatch exactly one compute-selection path from the current frontier.

        No HBM fitting, retry of another compute partition, or joint utility.
        A mixed proposal gets ANN credit for ITS selected partition, followed
        by a guard using the resulting overlap class, not the old solo class.
        """
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        if (type(decode_batch_size) is not int or decode_batch_size < 0
                or bool(decode_batch_size) != bool(self._decode_batch_active())):
            return None, {'available': False, 'reason': 'decode_frontier_mismatch'}
        if (not math.isfinite(decode_target_ms) or decode_target_ms <= 0
                or not math.isfinite(safety_margin_ms) or safety_margin_ms < 0):
            raise ValueError('Explicit finite decode target and margin required')
        streams = tuple(candidate_streams)
        pending = tuple(pending)
        def decode_recovery(reason):
            idx=decode_recovery_stream
            if not decode_batch_size or idx is None:
                return None, {'available':False,'reason':reason}
            if (type(idx) is not int or idx<0 or idx not in allocation_by_stream
                    or idx in streams):
                raise ValueError('Distinct actual decode-recovery partition required')
            query_pressure=getattr(decode_profile,'query_pressure_ms',None)
            value=(query_pressure(idx,'solo',decode_batch_size,0.)
                   if callable(query_pressure) else
                   decode_profile.query_batch_ms(idx,'solo',decode_batch_size))
            if (value is None or not math.isfinite(value) or value<=0
                    or value+safety_margin_ms>decode_target_ms):
                return None, {'available':False,'reason':reason,
                    'decode_recovery_reason':'decode_recovery_not_safe',
                    'decode_recovery_predicted_ms':value}
            return ComputeLease(allocation_by_stream[idx],0,0), {
                'available':True,'selected_safe':True,'selected_stream_idx':idx,
                'selected_overlap':'solo','selected_predicted_decode_ms':value,
                'decode_recovery':True,'decode_recovery_reason':reason}
        has_prefill = (getattr(self, 'split_prefill_batch', None) is not None
                       or bool(getattr(self, 'waiting_queue', ())))
        common = dict(candidate_streams=streams, allocation_by_stream=allocation_by_stream,
            service_profile=service_profile, pending=pending, microbatch_limit=microbatch_limit,
            credit_ms=credit_ms, source_execution=source_execution)
        if not has_prefill:
            proposed, diagnostic=self._choose_rag_joint_nonprefill_compute(
                **common, decode_profile=decode_profile,
                decode_batch_size=decode_batch_size, decode_target_ms=decode_target_ms,
                safety_margin_ms=safety_margin_ms)
            return ((proposed,diagnostic) if proposed is not None else
                    decode_recovery(diagnostic.get('reason','no_profiled_safe_compute')))
        if not decode_batch_size:
            return self._choose_rag_joint_prefill_without_decode(**common, queued_budget=queued_budget,
                prefill_profile=prefill_profile, bundle_limit=bundle_limit)
        ann_candidates = None
        if getattr(prefill_profile, 'muxwise_layer_bundles', False):
            ann_candidates = {idx:self._propose_rag_joint_ann_credit(partition=idx,
                service_profile=service_profile, pending=pending,
                microbatch_limit=microbatch_limit, credit_ms=credit_ms,
                source_execution=source_execution) for idx in streams}
        proposed, diagnostic = self._choose_rag_joint_prefill_compute(queued_budget=queued_budget,
            prefill_cap_idx=prefill_cap_idx, adaptive_idx=adaptive_idx, ann_rows=0,
            allocation_by_stream=allocation_by_stream, prefill_profile=prefill_profile,
            decode_profile=decode_profile, ann_credit_by_partition=ann_candidates,
            decode_guard=(decode_target_ms,safety_margin_ms))
        if proposed is None:
            return decode_recovery(diagnostic.get('reason','no_profiled_safe_compute'))
        matches = [idx for idx in streams if allocation_by_stream[idx] == proposed.allocation]
        if len(matches) != 1:
            return None, {'available': False, 'reason': 'proposal_outside_candidate_domain'}
        idx = matches[0]
        ann = (ann_candidates.get(idx) if ann_candidates is not None else
            self._propose_rag_joint_ann_credit(partition=idx, service_profile=service_profile,
            pending=pending, microbatch_limit=microbatch_limit, credit_ms=credit_ms,
            source_execution=source_execution))
        if ann is None:
            return None, {'available': False, 'reason': 'unknown_selected_ann_credit'}
        rows, service = ann
        overlap = 'ann_prefill' if rows or self._rag_ann_gpu_work_active() else 'prefill'
        decode_ms = (decode_profile.query_pressure_ms(idx, overlap, decode_batch_size, service)
                     if callable(getattr(decode_profile,'query_pressure_ms',None)) else
                     decode_profile.query_batch_ms(idx, overlap, decode_batch_size))
        if (decode_ms is None or not math.isfinite(decode_ms) or decode_ms <= 0
                or decode_ms + safety_margin_ms > decode_target_ms):
            return decode_recovery('selected_overlap_not_decode_safe')
        if source_execution != capture_execution_stamp(self):
            return None, {'available': False, 'reason': 'stale'}
        return ComputeLease(proposed.allocation, rows, proposed.prefill_layers,
                            proposed.prefill_contract), {**diagnostic,
            'selected_stream_idx': idx, 'ann_gpu_ms': service,
            'selected_overlap': overlap, 'selected_predicted_decode_ms': decode_ms}

    def _choose_rag_joint_prefill_compute(self, *, queued_budget, prefill_cap_idx,
                                          adaptive_idx, ann_rows, allocation_by_stream,
                                          prefill_profile, decode_profile,
                                          ann_credit_by_partition=None, decode_guard=None):
        """Translate one mixed-prefill selection to an executable-limit proposal.

        Both stage states consume the SAME caller-frozen profiles. ann_rows is
        the compute-credit proposal, not a workspace-fitted row count. This
        method never asks HBM to fit a candidate or applies utility hysteresis.
        Returns (proposal, diagnostics); unsupported is explicit, not an unsafe
        legacy best-effort partition. ANN-only/decode-only use other frontiers.
        """
        if type(ann_rows) is not int or ann_rows < 0:
            raise ValueError('Nonnegative compute-side ANN credit proposal required')
        batch = getattr(self,'split_prefill_batch',None)
        if batch is not None:
            contract = capture_prepared_prefill_contract(batch)
            if contract is None:
                return None, {'available':False,'reason':'invalid_prepared_contract'}
            selected, diagnostic = self._propose_rag_prefill_mixed_candidate(
                prefill_cap_idx,adaptive_idx,prefill_profile=prefill_profile,
                decode_profile=decode_profile,ann_credit_by_partition=ann_credit_by_partition,
                decode_guard=decode_guard)
        elif queued_budget is not None:
            selected, diagnostic = self._propose_rag_queued_prefill_compute(
                queued_budget,prefill_cap_idx,adaptive_idx,
                prefill_profile=prefill_profile,decode_profile=decode_profile,
                ann_credit_by_partition=ann_credit_by_partition,decode_guard=decode_guard)
            contract = diagnostic.get('prefill_contract') if diagnostic else None
        else:
            return None, {'available':False,'reason':'no_prefill_frontier'}
        if not diagnostic or not diagnostic.get('available'):
            return None, diagnostic
        if not diagnostic.get('selected_safe'):
            return None, {**diagnostic,'available':False,'reason':'no_decode_safe_compute_proposal'}
        chosen = [c for c in diagnostic['candidates'] if c['stream_idx'] == selected]
        if len(chosen) != 1 or contract is None:
            raise ValueError('Selected compute candidate must have one exact work contract')
        layers = chosen[0]['bundle_layers']
        if type(layers) is not int or layers <= 0:
            raise ValueError('Positive selected layer bundle required')
        return ComputeLease(allocation_by_stream[selected],ann_rows,layers,contract), diagnostic

    def _select_rag_prefill_mixed_candidate(
        self, prefill_cap_idx: int, adaptive_idx: int
    ) -> int:
        # Preserve legacy selection and diagnostics. Joint policies must use
        # the proposal path without transition cost, then evaluate G2 once.
        selected, diagnostics = self._propose_rag_prefill_mixed_candidate(
            prefill_cap_idx, adaptive_idx, include_transition_cost=True,
            legacy_submitted_frontier=True,
        )
        if diagnostics is not None:
            self.rag_prefill_partition_selection_last = diagnostics
            if diagnostics["available"]:
                self._increment_histogram(
                    self.rag_prefill_partition_selection_counts, selected
                )
        return selected

    def _propose_rag_prefill_mixed_candidate(
        self, prefill_cap_idx: int, adaptive_idx: int, *,
        include_transition_cost: bool = False,
        legacy_submitted_frontier: bool = False,
        prefill_profile=None,
        decode_profile=None,
        queued_cohort=None,
        ann_credit_by_partition=None,
        decode_guard=None,
    ):
        """Return one compute proposal and detached diagnostics, without commit.

        No memory planning, dispatch, gain admission or diagnostics mutation.
        The joint adapter must use the default: transitions belong to its G2
        evaluation, not this compute ranking. Legacy callers explicitly retain
        their historical transition-inclusive score. New proposals require
        frozen prefill/decode profiles; legacy mode retains its estimators.
        """
        last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
        split_batch = getattr(self, "split_prefill_batch", None)
        if queued_cohort is not None:
            queued_cohort = tuple(queued_cohort)
            if legacy_submitted_frontier or include_transition_cost:
                raise ValueError('Queued proposal cannot use legacy utility selection')
            if split_batch is not None:
                return prefill_cap_idx, {'available': False, 'reason': 'prepared_prefill_exists'}
            if (not queued_cohort
                    or len({s.request_id for s in queued_cohort}) != len(queued_cohort)
                    or any(type(s.extend_tokens) is not int or s.extend_tokens <= 0
                           for s in queued_cohort)):
                raise ValueError('Explicit unique queued cohort shapes required')
        if (
            not getattr(self, "rag_prefill_dynamic_lease", False)
            or (queued_cohort is None and (split_batch is None or split_batch.is_empty()))
            or not self._decode_batch_active()
            or not 1 <= prefill_cap_idx <= last_mixed_idx
        ):
            return prefill_cap_idx, None

        if queued_cohort is not None:
            token_count = sum(s.extend_tokens for s in queued_cohort)
            request_token_count = max(s.extend_tokens for s in queued_cohort)
        else:
            token_count = max(1, int(getattr(split_batch, "extend_num_tokens", 0)))
            request_token_count, _ = self._rag_prefill_request_token_count(split_batch, token_count)
        num_layers = max(
            1,
            int(
                getattr(
                    getattr(self, "model_config", None),
                    "num_hidden_layers",
                    1,
                )
            ),
        )
        if legacy_submitted_frontier:
            remaining_layers = max(
                1, num_layers - int(getattr(split_batch, "split_index", 0))
            )
        elif queued_cohort is not None:
            remaining_layers = num_layers
        else:
            completed = getattr(split_batch, "rag_completed_prefill_layers", None)
            submitted = getattr(split_batch, "split_index", None)
            meta = getattr(self, "rag_prefill_bundle_inflight_meta", None) or {}
            event = getattr(self, "rag_prefill_bundle_inflight_event", None)
            known = (
                type(completed) is int and type(submitted) is int
                and 0 <= completed <= submitted <= num_layers
            )
            if known and submitted > completed:
                count = meta.get("layers")
                known = (
                    event is not None and meta.get("batch") is split_batch
                    and type(count) is int and count == submitted - completed
                    and meta.get("end_layer") == submitted
                )
            elif known and event is not None and meta.get("batch") is split_batch:
                # Completion/event publication must be observed coherently.
                known = False
            if not known:
                return prefill_cap_idx, {
                    "available": False, "reason": "unknown_prefill_frontier",
                    "prefill_cap_idx": prefill_cap_idx, "adaptive_idx": adaptive_idx,
                }
            # Only unsubmitted work can be assigned to a new compute lease.
            # The joint predictor must still include [completed, submitted)
            # as current in-flight work, with its remaining service estimate.
            remaining_layers = num_layers - submitted
            if remaining_layers == 0:
                return prefill_cap_idx, {
                    "available": False, "reason": "no_unsubmitted_prefill",
                    "prefill_cap_idx": prefill_cap_idx, "adaptive_idx": adaptive_idx,
                }
        current_idx = get_current_stream_idx()
        overlap_class = (
            "ann_prefill" if self._rag_ann_gpu_work_active() else "prefill"
        )
        guard_percentile = min(
            100.0,
            max(
                0.001,
                float(
                    getattr(
                        self, "rag_prefill_decode_guard_percentile", 99.0
                    )
                ),
            ),
        )
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        def decode_prediction(partition, overlap, *, guarded):
            if decode_profile is not None:
                if callable(getattr(decode_profile,'query_pressure_ms',None)):
                    selected=(ann_credit_by_partition or {}).get(partition)
                    if selected is None:
                        return None
                    return decode_profile.query_pressure_ms(partition,overlap,
                        decode_profile.batch_size,selected[1])
                return decode_profile.query_ms(partition, overlap)
            if legacy_submitted_frontier:
                predictor = (
                    self._rag_ann_decode_prediction_ms
                    if guarded
                    else self._rag_prefill_nominal_decode_prediction_ms
                )
                return predictor(
                    partition, include_scheduling=False,
                    percentile=guard_percentile, overlap_class=overlap,
                )
            return None

        phase_model_route = getattr(prefill_profile, 'muxwise_layer_bundles', False)
        # Just-enough SM selection uses each candidate's guarded latency, not
        # a slowdown relative to a separate adaptive/solo reference partition.
        if decode_guard is not None:
            target_ms, safety_margin_ms = decode_guard
            if (type(target_ms) not in (int,float) or not math.isfinite(target_ms) or target_ms<=0
                    or type(safety_margin_ms) not in (int,float)
                    or not math.isfinite(safety_margin_ms) or safety_margin_ms<0):
                raise ValueError('Explicit positive decode target and nonnegative margin required')
        reference_decode_ms = (None if phase_model_route else
                               decode_prediction(adaptive_idx, "solo", guarded=False))
        if (reference_decode_ms is None and not legacy_submitted_frontier
                and not phase_model_route):
            return prefill_cap_idx, {
                "available": False, "reason": "missing_decode_reference",
                "prefill_cap_idx": prefill_cap_idx, "adaptive_idx": adaptive_idx,
            }
        candidates = []
        for candidate_idx in range(prefill_cap_idx, last_mixed_idx + 1):
            frontend_sms, decode_sms = self.sm_counts[candidate_idx]
            if frontend_sms <= 0 or decode_sms <= 0:
                continue
            candidate_overlap = overlap_class
            if ann_credit_by_partition is not None:
                credit = ann_credit_by_partition.get(candidate_idx)
                if credit is None:
                    continue
                candidate_overlap = ('ann_prefill' if credit[0] or overlap_class=='ann_prefill'
                                     else 'prefill')
            predicted_decode_ms = decode_prediction(
                candidate_idx, candidate_overlap, guarded=False
            )
            guarded_decode_ms = decode_prediction(
                candidate_idx, candidate_overlap, guarded=True
            )
            if predicted_decode_ms is None or guarded_decode_ms is None:
                continue
            slo_ms = (target_ms if decode_guard is not None
                      else self._rag_ann_decode_slo_ms(candidate_idx))
            safe = bool(
                slo_ms <= 0.0
                or (guarded_decode_ms + safety_margin_ms <= slo_ms
                    if decode_guard is not None else
                    guarded_decode_ms + safety_margin_ms < slo_ms)
            )
            short_threshold = max(
                1,
                int(
                    getattr(
                        self, "rag_prefill_short_token_threshold", 2048
                    )
                ),
            )
            two_level_bundles = getattr(
                self, "rag_prefill_two_level_bundles", False
            )
            severe_risk_ratio = max(
                1.0,
                float(getattr(self, "rag_prefill_severe_risk_ratio", 1.25)),
            )
            severe_decode_risk = bool(
                predicted_decode_ms is not None
                and slo_ms > 0.0
                and predicted_decode_ms + safety_margin_ms
                >= slo_ms * severe_risk_ratio
            )
            if request_token_count <= short_threshold and safe:
                if two_level_bundles:
                    bundle_layers = self._rag_prefill_two_level_layer_counts()[1]
                else:
                    bundle_layers = max(
                        1,
                        int(
                            getattr(
                                self, "rag_prefill_short_bundle_layers", 16
                            )
                        ),
                    )
            elif (
                two_level_bundles
                and request_token_count > short_threshold
                and not severe_decode_risk
            ):
                bundle_layers = self._rag_prefill_two_level_layer_counts()[0]
            else:
                bundle_layers = max(
                    1,
                    int(getattr(self, "rag_prefill_lease_max_layers", 8)),
                )
            bundle_layers = min(remaining_layers, bundle_layers)
            if prefill_profile is not None:
                layer_ms = prefill_profile.query_layer_ms(candidate_idx, token_count)
                if layer_ms is None:
                    continue
            elif legacy_submitted_frontier:
                layer_ms = self._predict_rag_prefill_layer_ms(
                    candidate_idx, token_count
                )
            else:
                # New proposals cannot silently use SM scaling or nominal
                # defaults when their frozen prefill profile is missing.
                continue
            if getattr(prefill_profile, 'muxwise_layer_bundles', False):
                # Match the predicted decode interval using the actual cohort's
                # whole-prefill/layer estimate (MuxWise Eq. for N_PL). Historical
                # token-threshold bundle heuristics remain only on the old path.
                bundle_layers = min(remaining_layers, max(1,
                    math.ceil(predicted_decode_ms / max(0.001, layer_ms))))
            bundle_ms = max(0.001, layer_ms) * bundle_layers
            overlap_steps = max(
                1,
                int(
                    np.ceil(
                        bundle_ms / max(0.001, predicted_decode_ms)
                    )
                ),
            )
            decode_penalty_ms = max(
                0.0,
                predicted_decode_ms
                - (
                    reference_decode_ms
                    if reference_decode_ms is not None
                    else predicted_decode_ms
                ),
            ) * overlap_steps
            transition_ms = (
                self._rag_lease_transition_cost_estimate_ms(
                    current_idx, candidate_idx
                ) if include_transition_cost else 0.0
            )
            candidates.append(
                {
                    "stream_idx": candidate_idx,
                    "safe": safe,
                    "predicted_decode_ms": float(predicted_decode_ms),
                    "guarded_decode_ms": float(guarded_decode_ms),
                    "predicted_bundle_ms": float(bundle_ms),
                    "bundle_layers": int(bundle_layers),
                    "transition_ms": float(transition_ms),
                    "score_ms": float(
                        bundle_ms + decode_penalty_ms + transition_ms
                    ),
                }
            )

        if not candidates:
            diagnostics = {
                "available": False,
                "reason": "prediction_unavailable",
                "prefill_cap_idx": prefill_cap_idx,
                "adaptive_idx": adaptive_idx,
            }
            return prefill_cap_idx, diagnostics

        safe_candidates = [item for item in candidates if item["safe"]]
        pool = safe_candidates or candidates
        if getattr(prefill_profile, 'muxwise_layer_bundles', False):
            if not safe_candidates:
                return prefill_cap_idx, {
                    'available': False, 'reason': 'no_decode_safe_compute_proposal',
                    'selected_safe': False, 'candidates': candidates,
                }
            # Just-enough decode compute, leaving remaining SMs to prefill.
            # Full's later admission owns migration/switch utility; do not
            # reintroduce the old millisecond tradeoff at proposal selection.
            selected = min(safe_candidates, key=lambda item: (
                self.sm_counts[item['stream_idx']][1],
                -self.sm_counts[item['stream_idx']][0],
                item['predicted_decode_ms'], item['stream_idx']))
        else:
            selected = min(
                pool,
                key=lambda item: (
                    item["score_ms"],
                    item["predicted_decode_ms"],
                    item["stream_idx"],
                ),
            )
        adaptive_prediction = next(
            (
                item["predicted_decode_ms"]
                for item in candidates
                if item["stream_idx"] == adaptive_idx
            ),
            reference_decode_ms,
        )
        diagnostics = {
            "available": True,
            "prefill_cap_idx": prefill_cap_idx,
            "adaptive_idx": adaptive_idx,
            "selected_idx": selected["stream_idx"],
            "selected_safe": selected["safe"],
            "selected_predicted_decode_ms": selected["predicted_decode_ms"],
            "adaptive_predicted_decode_ms": adaptive_prediction,
            "used_best_effort": not bool(safe_candidates),
            "candidates": candidates,
        }
        return int(selected["stream_idx"]), diagnostics

    def _rag_prefill_decode_bounds_safe(
        self,
        prefill_idx: int,
        adaptive_idx: int,
    ) -> bool:
        """Decode bounds only: no gain, transition, or urgency admission.

        Kept separate so a joint-policy adapter must not inherit the legacy
        millisecond-benefit gate before applying its own progress admission.
        This retains the legacy profile lookup semantics; it is not a frozen
        joint snapshot or proof that missing-profile fallback is suitable for
        the new predictor.
        """
        # Partition safety models co-run kernel service. Scheduler gaps are
        # controlled independently by bounded bundles and recovery windows.
        guard_percentile = min(
            100.0,
            max(
                0.001,
                float(
                    getattr(
                        self, "rag_prefill_decode_guard_percentile", 99.0
                    )
                ),
            ),
        )
        prefill_overlap = (
            "ann_prefill" if self._rag_ann_gpu_work_active() else "prefill"
        )
        use_causal_floor = os.environ.get(
            "RAGLEASE_PREFILL_CAUSAL_GUARD", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        prefill_predictor = (
            self._rag_ann_decode_prediction_ms
            if use_causal_floor
            else self._rag_prefill_nominal_decode_prediction_ms
        )
        prefill_decode_ms = prefill_predictor(
            prefill_idx,
            include_scheduling=False,
            percentile=guard_percentile,
            overlap_class=prefill_overlap,
        )
        if prefill_decode_ms is None:
            return False

        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        slo_ms = self._rag_ann_decode_slo_ms(prefill_idx)
        if (
            slo_ms > 0.0
            and prefill_decode_ms + safety_margin_ms >= slo_ms
        ):
            return False

        # The causal floor is an absolute admission guard.  Reusing that
        # upper-bound artifact in the relative slowdown test would penalize
        # the same uncertainty twice (against an unguarded solo prediction).
        # Compare like-for-like nominal device service here while retaining
        # the guarded value above for the hard Decode SLO check.
        relative_prefill_decode_ms = (
            self._rag_prefill_nominal_decode_prediction_ms(
                prefill_idx,
                include_scheduling=False,
                percentile=guard_percentile,
                overlap_class=(
                    "ann_prefill"
                    if self._rag_ann_gpu_work_active()
                    else "prefill"
                ),
            )
        )
        adaptive_decode_ms = self._rag_prefill_nominal_decode_prediction_ms(
            adaptive_idx,
            include_scheduling=False,
            percentile=guard_percentile,
            overlap_class="solo",
        )
        if relative_prefill_decode_ms is None:
            relative_prefill_decode_ms = prefill_decode_ms
        max_slowdown = max(
            1.0, getattr(self, "rag_ann_max_decode_slowdown_ratio", 1.0)
        )
        slowdown_safe = bool(
            adaptive_decode_ms is None
            or relative_prefill_decode_ms
            <= adaptive_decode_ms * max_slowdown
        )
        if not slowdown_safe:
            return False

        return True

    def _rag_prefill_lease_decode_safe(
        self,
        prefill_idx: int,
        adaptive_idx: int,
        *,
        prefill_wait_ms: float = 0.0,
        dependency_prefill_due: bool = False,
    ) -> bool:
        """Legacy admission: decode bounds followed by millisecond gain gates.

        Existing policies deliberately retain their old behavior. New Full and
        Independent-adaptive must not call this combined admission method.
        """
        if not self._rag_prefill_decode_bounds_safe(prefill_idx, adaptive_idx):
            return False

        net_gain_ms = self._rag_prefill_candidate_net_gain_ms(
            prefill_idx, adaptive_idx
        )
        if net_gain_ms is None:
            return True
        last = self.rag_prefill_candidate_net_gain_last
        if net_gain_ms > 0.0:
            last["admission_mode"] = "aggregate"
            self.rag_prefill_candidate_net_gain_admissions += 1
            return True

        prefill_deadline_ms = max(
            0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
        )
        tail_due = bool(
            getattr(self, "rag_prefill_tail_aware_admission", True)
            and (
                dependency_prefill_due
                or prefill_wait_ms >= prefill_deadline_ms
            )
        )
        dependency_critical = bool(
            getattr(self, "rag_prefill_tail_aware_admission", True)
            and getattr(self, "rag_prefill_handoff_active", False)
        )
        critical_path_gain_ms = float(
            last.get("critical_path_net_gain_ms", float("-inf"))
        )
        robust_critical_path_gain_ms = float(
            last.get("robust_critical_path_net_gain_ms", float("-inf"))
        )
        round_trip_cost_ms = (
            self._rag_lease_switch_cost_estimate_ms(
                adaptive_idx, prefill_idx
            )
            + self._rag_lease_switch_cost_estimate_ms(
                prefill_idx, adaptive_idx
            )
        )
        required_tail_gain_ms = round_trip_cost_ms * max(
            1.0, getattr(self, "rag_lease_min_gain_ratio", 1.0)
        )
        last.update(
            {
                "tail_due": tail_due,
                "dependency_critical": dependency_critical,
                "prefill_wait_ms": float(prefill_wait_ms),
                "round_trip_switch_cost_ms": round_trip_cost_ms,
                "required_tail_gain_ms": required_tail_gain_ms,
            }
        )
        if tail_due and critical_path_gain_ms > required_tail_gain_ms:
            last["admission_mode"] = "tail_aware"
            self.rag_prefill_candidate_net_gain_admissions += 1
            self.rag_prefill_tail_gain_admissions += 1
            return True
        if tail_due:
            # The gain model arbitrates early leases, but it must not become a
            # second unbounded queue once a dependency Prefill is overdue.
            # Decode SLO and slowdown checks above still bound this lease.
            last["admission_mode"] = "tail_liveness"
            self.rag_prefill_candidate_net_gain_admissions += 1
            self.rag_prefill_tail_liveness_admissions += 1
            return True
        if (
            dependency_critical
            and robust_critical_path_gain_ms > required_tail_gain_ms
        ):
            # A dependency-ready Prefill is on the request's E2E critical path.
            # Admit it early only when conservative phase bounds still cover
            # the Decode slowdown and the round-trip transition cost.
            last["admission_mode"] = "e2e_critical"
            self.rag_prefill_candidate_net_gain_admissions += 1
            self.rag_prefill_e2e_critical_admissions += 1
            return True
        if dependency_critical and critical_path_gain_ms > required_tail_gain_ms:
            self.rag_prefill_e2e_uncertainty_holds += 1
        last["admission_mode"] = "hold"
        self.rag_prefill_candidate_net_gain_holds += 1
        return False

    def _rag_decode_partition_within_slo(self, stream_idx: int) -> bool:
        guard_percentile = min(
            100.0,
            max(
                0.001,
                float(
                    getattr(
                        self, "rag_prefill_decode_guard_percentile", 99.0
                    )
                ),
            ),
        )
        predicted_ms = self._rag_ann_decode_prediction_ms(
            stream_idx,
            include_scheduling=False,
            percentile=guard_percentile,
        )
        if predicted_ms is None:
            return False
        slo_ms = self._rag_ann_decode_slo_ms(stream_idx)
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        return bool(
            slo_ms <= 0.0 or predicted_ms + safety_margin_ms < slo_ms
        )

    def _rag_prefill_candidate_net_gain_ms(
        self, candidate_idx: int, reference_idx: int
    ) -> Optional[float]:
        """Estimate aggregate request-latency gain for an aggressive lease."""
        def unavailable(reason: str) -> None:
            self.rag_prefill_candidate_net_gain_last = {
                "available": False,
                "reason": reason,
                "candidate_stream_idx": candidate_idx,
                "reference_stream_idx": reference_idx,
            }
            return None

        if candidate_idx == reference_idx:
            return unavailable("same_partition")
        sm_counts = getattr(self, "sm_counts", ())
        if not (
            0 <= candidate_idx < len(sm_counts)
            and 0 <= reference_idx < len(sm_counts)
            and int(sm_counts[candidate_idx][0]) > 0
            and int(sm_counts[reference_idx][0]) > 0
        ):
            return unavailable("partition_without_frontend")

        split_batch = getattr(self, "split_prefill_batch", None)
        running_batch = getattr(self, "running_batch", None)
        if (
            split_batch is None
            or split_batch.is_empty()
            or running_batch is None
            or running_batch.is_empty()
        ):
            return unavailable("missing_active_batches")

        token_count = max(1, int(getattr(split_batch, "extend_num_tokens", 0)))
        candidate_decode_ms = self._rag_prefill_nominal_decode_prediction_ms(
            candidate_idx,
            include_scheduling=False,
            overlap_class="prefill",
        )
        reference_decode_ms = self._rag_prefill_nominal_decode_prediction_ms(
            reference_idx,
            include_scheduling=False,
            overlap_class="solo",
        )
        if candidate_decode_ms is None or reference_decode_ms is None:
            return unavailable("decode_prediction_unavailable")

        layer_count, candidate_layer_ms, _ = self._rag_prefill_lease_shape(
            candidate_idx, candidate_decode_ms
        )
        if layer_count is None or layer_count <= 0 or candidate_layer_ms is None:
            return unavailable("prefill_prediction_unavailable")
        reference_layer_ms = self._predict_rag_prefill_layer_ms(
            reference_idx, token_count
        )
        candidate_bundle_ms = max(0.001, candidate_layer_ms) * layer_count
        reference_bundle_ms = max(0.001, reference_layer_ms) * layer_count
        prefill_items = max(1, int(split_batch.batch_size()))
        decode_items = max(1, int(running_batch.batch_size()))
        critical_path_prefill_gain_ms = max(
            0.0, reference_bundle_ms - candidate_bundle_ms
        )
        prefill_gain_ms = critical_path_prefill_gain_ms * prefill_items
        overlap_steps = max(
            1, int(np.ceil(candidate_bundle_ms / candidate_decode_ms))
        )
        critical_path_decode_cost_ms = (
            max(0.0, candidate_decode_ms - reference_decode_ms)
            * overlap_steps
        )
        decode_cost_ms = critical_path_decode_cost_ms * decode_items
        net_gain_ms = prefill_gain_ms - decode_cost_ms
        critical_path_net_gain_ms = (
            critical_path_prefill_gain_ms - critical_path_decode_cost_ms
        )
        reference_layer_lower_ms = self._predict_rag_prefill_layer_ms(
            reference_idx, token_count, percentile=50.0
        )
        candidate_decode_upper_ms = self._rag_ann_decode_prediction_ms(
            candidate_idx,
            include_scheduling=False,
            percentile=99.0,
            overlap_class="prefill",
        )
        reference_decode_lower_ms = self._rag_ann_decode_prediction_ms(
            reference_idx,
            include_scheduling=False,
            percentile=50.0,
            overlap_class="solo",
        )
        if candidate_decode_upper_ms is None:
            candidate_decode_upper_ms = candidate_decode_ms
        if reference_decode_lower_ms is None:
            reference_decode_lower_ms = reference_decode_ms
        robust_reference_bundle_ms = (
            max(0.001, reference_layer_lower_ms) * layer_count
        )
        robust_overlap_steps = max(
            1,
            int(
                np.ceil(
                    candidate_bundle_ms
                    / max(0.001, candidate_decode_upper_ms)
                )
            ),
        )
        robust_critical_path_prefill_gain_ms = max(
            0.0, robust_reference_bundle_ms - candidate_bundle_ms
        )
        robust_critical_path_decode_cost_ms = (
            max(0.0, candidate_decode_upper_ms - reference_decode_lower_ms)
            * robust_overlap_steps
        )
        robust_critical_path_net_gain_ms = (
            robust_critical_path_prefill_gain_ms
            - robust_critical_path_decode_cost_ms
        )
        self.rag_prefill_candidate_net_gain_ms.append(net_gain_ms)
        self.rag_prefill_critical_path_net_gain_ms.append(
            critical_path_net_gain_ms
        )
        self.rag_prefill_robust_critical_path_net_gain_ms.append(
            robust_critical_path_net_gain_ms
        )
        self.rag_prefill_candidate_net_gain_last = {
            "available": True,
            "candidate_stream_idx": candidate_idx,
            "reference_stream_idx": reference_idx,
            "layer_count": layer_count,
            "prefill_items": prefill_items,
            "decode_items": decode_items,
            "candidate_bundle_ms": candidate_bundle_ms,
            "reference_bundle_ms": reference_bundle_ms,
            "overlap_steps": overlap_steps,
            "prefill_gain_ms": prefill_gain_ms,
            "decode_cost_ms": decode_cost_ms,
            "net_gain_ms": net_gain_ms,
            "critical_path_prefill_gain_ms": critical_path_prefill_gain_ms,
            "critical_path_decode_cost_ms": critical_path_decode_cost_ms,
            "critical_path_net_gain_ms": critical_path_net_gain_ms,
            "robust_reference_bundle_ms": robust_reference_bundle_ms,
            "robust_candidate_decode_ms": candidate_decode_upper_ms,
            "robust_reference_decode_ms": reference_decode_lower_ms,
            "robust_overlap_steps": robust_overlap_steps,
            "robust_critical_path_prefill_gain_ms": (
                robust_critical_path_prefill_gain_ms
            ),
            "robust_critical_path_decode_cost_ms": (
                robust_critical_path_decode_cost_ms
            ),
            "robust_critical_path_net_gain_ms": (
                robust_critical_path_net_gain_ms
            ),
        }
        return float(net_gain_ms)

    def _select_frontend_plan_target(
        self,
        *,
        frontend_needed: bool,
        allow_ann: bool,
        allow_prefill: bool,
        active_ann: bool,
        forced_ann: bool = False,
        ann_wait_ms: float = 0.0,
        prefill_wait_ms: float = 0.0,
        predicted_decode_ms: Optional[float] = None,
        dependency_handoff_pending: bool = False,
        dependency_prefill_due: bool = False,
        force_decode_only: bool = False,
        force_decode_only_reason: str = "decode_recovery",
        bypass_work_wait: bool = False,
        bypass_partition_guards: bool = False,
        fast_path_candidate_idx: Optional[int] = None,
    ) -> tuple[int, str]:
        if not self._decode_batch_active():
            return 0, "no_decode"

        current_idx = get_current_stream_idx()
        last_mixed_idx = self.real_sm_group_num - 2
        fixed_idx = int(getattr(self, "rag_e2_fixed_partition_idx", -1))
        if fixed_idx >= 0:
            if not 1 <= fixed_idx <= last_mixed_idx:
                raise ValueError(
                    "E2 fixed partition must name a configured mixed group"
                )
            return fixed_idx, "e2_fixed_partition"
        if force_decode_only:
            return self.real_sm_group_num - 1, force_decode_only_reason
        if (
            dependency_handoff_pending
            and not bypass_partition_guards
            and 1 <= current_idx <= last_mixed_idx
        ):
            self.rag_dependency_lease_holds += 1
            return current_idx, "dependency_handoff_resident"

        if not frontend_needed:
            if self._should_hold_safe_mixed_partition(current_idx):
                self.rag_partition_safe_mixed_holds += 1
                return current_idx, "decode_safe_mixed_hold"
            return self.real_sm_group_num - 1, "decode_only"

        decode_bs = self.running_batch.batch_size()
        raw_adaptive_idx = self._adaptive_frontend_stream_idx(decode_bs)
        if bypass_partition_guards:
            batch_adaptive_idx = raw_adaptive_idx
            adaptive_idx = raw_adaptive_idx
        else:
            batch_adaptive_idx = self._apply_decode_partition_hysteresis(
                raw_adaptive_idx, decode_bs
            )
            adaptive_idx = self._apply_rag_decode_partition_floor(
                batch_adaptive_idx
            )

        if bypass_partition_guards and fast_path_candidate_idx is not None:
            target_idx = fast_path_candidate_idx
            if allow_prefill:
                # Fast-path dependency work bypasses aggregation and residency,
                # not candidate safety. Reuse the partition-aware predictor so
                # a low request rate with a large Decode batch does not blindly
                # borrow the configured maximum Frontend share.
                if 1 <= target_idx <= last_mixed_idx:
                    target_idx = self._select_rag_prefill_mixed_candidate(
                        target_idx, adaptive_idx
                    )
                target_reason = "prefill_fast_path"
            elif allow_ann:
                target_reason = "ann_fast_path"
            else:
                target_reason = "fast_path_candidate"
            return self._coalesce_rag_mixed_partition_target(
                candidate_idx=target_idx,
                target_reason=target_reason,
                allow_ann=allow_ann,
                allow_prefill=allow_prefill,
                forced_ann=forced_ann,
                ann_wait_ms=ann_wait_ms,
                prefill_wait_ms=prefill_wait_ms,
                predicted_decode_ms=predicted_decode_ms,
                bypass_work_wait=bypass_work_wait,
            )
        adaptive_was_floored = adaptive_idx != batch_adaptive_idx
        target_idx = adaptive_idx
        target_reason = (
            "adaptive_slo_floor" if adaptive_was_floored else "adaptive"
        )

        if (
            forced_ann
            and allow_ann
            and not allow_prefill
            and 1 <= current_idx <= last_mixed_idx
            and self.sm_counts[current_idx][1]
            >= self.sm_counts[adaptive_idx][1]
        ):
            # The queueing budget is exhausted. Reuse a decode-safe mixed
            # partition instead of waiting for a stream-group switch, which
            # synchronizes both PD-Mux streams before ANN can launch.
            return current_idx, "ann_max_defer_resident"

        if allow_prefill:
            prefill_idx = self._stream_idx_for_frontend_sm(
                getattr(self, "rag_prefill_lease_frontend_sm", -1)
            )
            if prefill_idx is not None:
                prefill_idx = self._select_rag_prefill_mixed_candidate(
                    prefill_idx, adaptive_idx
                )
                partition_selection = getattr(
                    self, "rag_prefill_partition_selection_last", {}
                )
                candidate_dominates_adaptive = bool(
                    partition_selection.get("available", False)
                    and partition_selection.get("selected_idx") == prefill_idx
                    and partition_selection.get("adaptive_predicted_decode_ms")
                    is not None
                    and partition_selection.get("selected_predicted_decode_ms")
                    < partition_selection.get("adaptive_predicted_decode_ms")
                )
                prefill_deadline_due = prefill_wait_ms >= max(
                    0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
                )
                prefill_decode_safe = self._rag_prefill_lease_decode_safe(
                    prefill_idx,
                    adaptive_idx,
                    prefill_wait_ms=prefill_wait_ms,
                    dependency_prefill_due=dependency_prefill_due,
                )
                if (
                    self.sm_counts[prefill_idx][1]
                    >= self.sm_counts[adaptive_idx][1]
                ):
                    target_idx = prefill_idx
                    target_reason = "prefill_lease"
                elif prefill_decode_safe:
                    target_idx = prefill_idx
                    target_reason = (
                        "prefill_lease_tail_safe"
                        if self.rag_prefill_candidate_net_gain_last.get(
                            "admission_mode"
                        )
                        == "tail_aware"
                        else (
                            "prefill_lease_tail_liveness"
                            if self.rag_prefill_candidate_net_gain_last.get(
                                "admission_mode"
                            )
                            == "tail_liveness"
                            else (
                                "prefill_lease_e2e_critical"
                                if self.rag_prefill_candidate_net_gain_last.get(
                                    "admission_mode"
                                )
                                == "e2e_critical"
                                else "prefill_lease_decode_safe"
                            )
                        )
                    )
                elif candidate_dominates_adaptive:
                    target_idx = prefill_idx
                    target_reason = "prefill_lease_candidate_dominates"
                elif prefill_deadline_due or dependency_prefill_due:
                    # Borrow one partition step for overdue dependency work;
                    # jumping directly to the lease cap can starve Decode.
                    target_idx = max(prefill_idx, adaptive_idx - 1)
                    target_reason = (
                        "prefill_handoff_step"
                        if dependency_prefill_due and not prefill_deadline_due
                        else "prefill_lease_deadline_step"
                    )
                else:
                    target_idx = adaptive_idx
                    target_reason = "prefill_lease_decode_floor"
            else:
                target_reason = (
                    "prefill_adaptive_slo_floor"
                    if adaptive_was_floored
                    else "prefill_adaptive"
                )
        elif allow_ann or active_ann:
            if self._rag_ann_uses_frontend_pool():
                ann_idx = self._stream_idx_for_frontend_sm(
                    getattr(self, "rag_ann_lease_frontend_sm", -1)
                )
                if ann_idx is not None:
                    if (
                        self.sm_counts[ann_idx][1]
                        >= self.sm_counts[adaptive_idx][1]
                    ):
                        target_idx = ann_idx
                        target_reason = "ann_lease"
                    else:
                        target_idx = adaptive_idx
                        target_reason = "ann_lease_decode_floor"
                else:
                    target_reason = (
                        "ann_adaptive_slo_floor"
                        if adaptive_was_floored
                        else "ann_adaptive"
                    )
            else:
                target_reason = (
                    "ann_decode_gc_adaptive_slo_floor"
                    if adaptive_was_floored
                    else "ann_decode_gc_adaptive"
                )

        target_idx, target_reason = self._coalesce_rag_mixed_partition_target(
            candidate_idx=target_idx,
            target_reason=target_reason,
            allow_ann=allow_ann,
            allow_prefill=allow_prefill,
            forced_ann=forced_ann,
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
            predicted_decode_ms=predicted_decode_ms,
            bypass_work_wait=bypass_work_wait,
        )

        if not bypass_partition_guards:
            target_idx = self._apply_partition_residency(target_idx)
            if (
                target_idx == get_current_stream_idx()
                and target_reason != "adaptive"
            ):
                target_reason += "_resident"
        return target_idx, target_reason

    def _rag_joint_service_observations(self):
        """Export detached bounded histories, not new timing or reservations.

        These independently sampled histories are not an atomic decision
        snapshot. Lock-held service is included in worker time, not additive.
        No query vectors, result payloads or Future identities are exported.
        """
        executor=getattr(self,'ann_executor',None)
        worker=getattr(executor,'_cold_batcher',None)
        worker_reader=getattr(worker,'planning_snapshot',None)
        lock_reader=getattr(executor,'cpu_search_lock_snapshot',None)
        cpu=worker_reader() if callable(worker_reader) else None
        lock=lock_reader() if callable(lock_reader) else None
        return dict(schema='joint-service-observations-v1',
            cpu_worker=None if cpu is None else dict(
                clock='worker_wall_including_lock_excluding_queue',
                samples=[dict(shape=[list(row) for row in shape],elapsed_ms=ms)
                         for shape,ms in cpu['service_samples']]),
            cpu_lock=None if lock is None else dict(
                clock='native_lock_held_wall_not_additive_to_worker',
                samples=[dict(shape=list(shape),elapsed_ms=ms)
                         for shape,ms in lock['hold_samples']]),
            decode=dict(clock='host_submit_to_observed_completion',
                samples=[dict(partition=p,batch_bucket=b,overlap_class=o,elapsed_ms=list(values))
                         for p,batches in getattr(self,'rag_decode_submission_service_ms',{}).items()
                         for b,overlaps in batches.items() for o,values in overlaps.items()]),
            pressure=dict(
                clock='decode_submission_to_completion',
                ann_clock='ann_gpu_event_window_ending_at_completion',
                samples=list(getattr(self,'rag_decode_pressure_samples',())),
                ann_samples=[dict(partition=p,batch_size=b,gpu_ms=ms,completed_at=t)
                             for p,b,ms,t in getattr(self,'rag_ann_gpu_service_samples',())]))

    def _rag_joint_runtime_status(self):
        """Report actual path evidence separately from the historical Full label."""
        installed=callable(getattr(self,'rag_joint_event_input_provider',None))
        attempts=getattr(self,'rag_joint_entry_attempts',0)
        decisions=getattr(self,'rag_joint_evaluation_count',0)
        activations=getattr(self,'rag_joint_activation_count',0)
        bootstrap=getattr(self,'rag_joint_bootstrap_receipt',None)
        return dict(schema='joint-runtime-status-v1',
            engineering_bootstrap=None if bootstrap is None else dict(bootstrap),
            legacy_calibrated_predictor_loaded=getattr(self,"rag_legacy_calibrated_bundle",None) is not None,
            legacy_calibrated_prediction_counts=dict(getattr(self,"rag_legacy_calibrated_prediction_counts",{})),
            legacy_calibrated_fallback_counts=dict(getattr(self,"rag_legacy_calibrated_fallback_counts",{})),
            legacy_calibrated_fallback_reason_counts=dict(getattr(self,"rag_legacy_calibrated_fallback_reason_counts",{})),
            legacy_calibrated_last_ann=getattr(self,"rag_legacy_calibrated_last_ann",None),
            legacy_calibrated_last_prefill=getattr(self,"rag_legacy_calibrated_last_prefill",None),
            legacy_calibrated_last_decode=getattr(self,"rag_legacy_calibrated_last_decode",None),
            engineering_missing_input=getattr(self,'rag_joint_engineering_missing_input',None),
            prospective_input_failure_counts=dict(
                getattr(self,'rag_joint_prospective_input_failure_counts',{})),
            forecast_failure_counts=dict(getattr(self,'rag_joint_forecast_failure_counts',{})),
            last_prediction_error=getattr(self,'rag_joint_last_prediction_error',None),
            last_target_projection=getattr(self,'rag_joint_last_target_projection',None),
            forecast_debug=getattr(self,'rag_joint_forecast_debug',None),
            requested_policy=getattr(self,'rag_policy_mode','full'),
            input_provider_installed=installed,
            implementation='single_proposal_joint' if installed else 'legacy_frontend',
            entry_attempts=attempts,evaluated_decisions=decisions,activations=activations,
            decision_reason_counts=dict(getattr(self,'rag_joint_decision_reason_counts',{})),
            stale_evaluations=getattr(self,'rag_joint_stale_evaluation_count',0),
            replan_skips=getattr(self,'rag_joint_replan_skips',0),
            min_replan_ms=getattr(self,'rag_joint_min_replan_ms',0.),
            max_replan_ms=getattr(self,'rag_joint_max_replan_ms',0.),
            pending_preparation=getattr(self,'rag_joint_pending_preparation',None) is not None,
            last_preparation_status=getattr(self,'rag_joint_last_preparation_status',None),
            active_event_id=getattr(self,'rag_joint_active_event_id',None))

    def _rag_joint_replan_signature(self):
        """Cheap state key for real decision events instead of every GPU tick."""
        def reqs(owner):
            return tuple((id(req),len(getattr(req,'output_ids',()) or ())//16)
                         for req in (getattr(owner,'reqs',()) or ()))
        def tasks(name):
            return tuple(id(task) for task in (getattr(self,name,None) or ()))
        split=getattr(self,'split_prefill_batch',None)
        merge=getattr(self,'rag_prefill_merge_pending_batch',None)
        current=getattr(self,'rag_joint_applied_compute',None)
        compute=(None if current is None else
                 (current.allocation,current.ann_batch,current.prefill_layers,
                  current.prefill_contract))
        return (get_current_stream_idx(),compute,
            reqs(getattr(self,'running_batch',None)),
            reqs(split),getattr(split,'split_index',None),reqs(merge),
            tuple(id(req) for req in (getattr(self,'waiting_queue',None) or ())),
            tasks('ann_search_pending'),tasks('ann_search_inflight'),
            tasks('rag_ann_collection_pending'),
            bool(getattr(self,'rag_partition_transition_intent',None)),
            bool(getattr(self,'rag_prefill_bundle_inflight_event',None)))

    def _hold_rag_joint_safe_continuation(self, reason):
        """Keep compatible decode moving without reusing a launch grant."""
        self.rag_joint_active_event_id=None
        self.rag_joint_active_memory_plan=None
        plan=FrontendPlan(False,False,False,get_current_stream_idx(),
            reason,'joint_current_only','joint_no_new_admission',
            ann_batch_cap=0,prefill_forward_count=0,joint_event_id=None)
        self.rag_frontend_plan=plan
        self.rag_prefill_run_allowed=False
        self.rag_joint_last_preparation_status=reason
        self.rag_frontend_last_plan=asdict(plan)

    def _retain_rag_joint_active_lease(self, reason, *, ann_only=False):
        """Retain bounded current-allocation work between real decisions.

        A joint lease is a bounded execution policy, not a one-tick launch token.
        An unchanged observation may retain the whole activated plan. A changed
        request frontier may retain only ANN when the stream/allocation and the
        decode cohort are unchanged: the old request-specific prefill contract
        is removed, while its already committed ANN workspace/index backing and
        fitted row cap remain valid. Partition, decode, or transition changes
        still fail closed and require a fresh decision.
        """
        plan=getattr(self,'rag_frontend_plan',None)
        event=getattr(plan,'joint_event_id',None)
        previous_active=getattr(self,'rag_joint_previous_active_event_id',
            getattr(self,'rag_joint_active_event_id',None))
        if (plan is None or type(event) is not int or event < 0
                or previous_active != event
                or get_current_stream_idx() != plan.target_stream_idx):
            return False
        if ann_only:
            if (not plan.allow_ann or not plan.ann_batch_cap
                    or getattr(self,'rag_partition_transition_intent',None) is not None
                    or getattr(self,'rag_prefill_bundle_inflight_event',None) is not None):
                return False
            plan=replace(plan,allow_prefill=False,frontend_needed=True,
                reason=reason,prefill_forward_count=0,prefill_contract=None)
            self.rag_frontend_plan=plan
            self.rag_prefill_run_allowed=False
        # event_loop_pdmux deliberately closes the preceding tick's launch
        # marker before receiving requests and evaluating this decision. Only
        # republish it after the replan signature/current partition checks
        # above prove that this is the same bounded active lease.
        self.rag_joint_active_event_id=event
        if not joint_plan_ready_for_dispatch(self,plan):
            self.rag_joint_active_event_id=None
            return False
        self.rag_joint_last_preparation_status=reason
        self.rag_frontend_last_plan=asdict(plan)
        return True

    def _plan_rag_bound_joint_event(self):
        """Run the configured same-event input provider through native admission.

        The provider supplies capture/settings and frozen per-partition work
        parameters, not a selected lease. It must not dispatch, optimize, or
        inspect future arrivals. Deployment must install it only after the
        complete profile/forecast contract passes readiness checks. Absence
        preserves legacy operation; an installed but unsupported provider
        never silently changes this boundary back into legacy scheduling.
        """
        provider=getattr(self,'rag_joint_event_input_provider',None)
        if provider is None:
            return False
        if not callable(provider):
            raise ValueError('Explicit joint runtime input provider required')
        mode=getattr(self,'rag_policy_mode',None)
        if mode not in ('full','independent_adaptive'):
            raise ValueError('Joint runtime provider requires a joint policy')
        if getattr(self,'rag_joint_pending_preparation',None) is not None:
            raise ValueError('Pending joint intent must resume before a fresh decision')
        now=time.perf_counter()
        min_replan_ms=max(0.,float(getattr(self,'rag_joint_min_replan_ms',0.)))
        max_replan_ms=max(0.,float(getattr(self,'rag_joint_max_replan_ms',0.)))
        if max_replan_ms and max_replan_ms<min_replan_ms:
            raise ValueError('Joint replan intervals are inconsistent')
        if max_replan_ms:
            signature=self._rag_joint_replan_signature()
            previous_signature=getattr(self,'rag_joint_last_replan_signature',None)
            previous_at=getattr(self,'rag_joint_last_replan_at',None)
            elapsed_ms=(float('inf') if previous_at is None else
                        max(0.,(now-previous_at)*1000.))
            changed=signature!=previous_signature
            due=(previous_at is None or elapsed_ms>=max_replan_ms or
                 (changed and elapsed_ms>=min_replan_ms))
            if not due:
                self.rag_joint_replan_skips=getattr(self,'rag_joint_replan_skips',0)+1
                reason=('joint_replan_min_interval_ann_continuation' if changed
                        else 'joint_replan_unchanged_active')
                ann_only_safe=(changed and previous_signature is not None
                    and signature[:3]==previous_signature[:3]
                    and not signature[10] and not signature[11])
                retained=self._retain_rag_joint_active_lease(reason,
                    ann_only=ann_only_safe) if (not changed or ann_only_safe) else False
                if not retained:
                    self._hold_rag_joint_safe_continuation(
                        'joint_replan_min_interval' if changed
                        else 'joint_replan_unchanged')
                return True
            self.rag_joint_last_replan_signature=signature
            self.rag_joint_last_replan_at=now
        controller=getattr(self,'shared_hbm_controller',None)
        previous=(getattr(self,'rag_joint_decision_sequence',0),
            getattr(controller,'_joint_kv_committed_event_id',-1),
            getattr(controller,'_joint_scratch_committed_event_id',-1))
        if any(type(value) is not int or value < -1 for value in previous):
            raise ValueError('Valid monotonic joint event watermarks required')
        # Existing controller receipts survive an adapter/provider change.
        # Never reuse their IDs or reset their duplicate-submission guards.
        event_id=max(previous)+1
        self.rag_joint_decision_sequence=event_id
        self.rag_joint_entry_attempts=getattr(self,'rag_joint_entry_attempts',0)+1
        started=now
        # Fail closed for frontend work before invoking a possibly unsupported
        # capture. Compatible decode keeps the existing native execution path.
        self.rag_joint_active_event_id=None
        self.rag_joint_active_memory_plan=None
        # Let the read-only provider observe the previous actual limits before
        # replacing the public dispatch intent with this event's closed gate.
        closed_plan=FrontendPlan(False,False,False,get_current_stream_idx(),
            'joint_inputs_unavailable','joint_current_only','joint_no_new_admission',
            ann_batch_cap=0,prefill_forward_count=0,joint_event_id=event_id)
        try:
            inputs=provider(event_id=event_id,window_started_at=started,mode=mode)
        except Exception:
            self.rag_joint_last_preparation_status='input_provider_error'
            self.rag_frontend_last_plan=asdict(closed_plan)
            raise  # Never hide the failure or fall back to legacy decisions.
        finally:
            # The provider may inspect the preceding intent, but failure must
            # not leave that old ANN/prefill grant usable by error recovery.
            self.rag_frontend_plan=closed_plan
            self.rag_prefill_run_allowed=False
        status='provider_inputs_unavailable'
        if inputs is not None:
            callback_builder=inputs.get('callbacks_from_context')
            if callback_builder is not None and ('callbacks' in inputs or not callable(callback_builder)):
                raise ValueError('Choose frozen callbacks or one same-context callback builder')
            context=self._capture_rag_joint_decision_context(mode=mode,current_event_id=event_id,
                window_started_at=started,**inputs['capture'])
            if context is not None:
                # Runtime work/profile assembly depends on the ONE captured
                # context. Do not force providers to recapture it beforehand.
                callback_inputs=(callback_builder(context) if callback_builder is not None
                                 else inputs['callbacks'])
                if callback_inputs is None:
                    self.rag_joint_last_preparation_status='callback_inputs_unavailable'
                    self.rag_frontend_last_plan=asdict(closed_plan)
                    return True
                evaluation=self._bind_rag_joint_decision_callbacks(context,
                    **callback_inputs)
                mapping=context['compute_inputs']['allocation_by_stream']
                reverse={allocation:idx for idx,allocation in mapping.items()}
                if len(reverse)!=len(mapping):
                    raise ValueError('Unique runtime allocation mapping required')
                decision,status=self._prepare_rag_joint_decision_event(
                    evaluation_inputs=evaluation,allocation_to_stream=reverse)
                self.rag_prefill_run_allowed=(status=='ready' and self.rag_frontend_plan.allow_prefill)
            else:
                status='context_inputs_unavailable'
        self.rag_joint_last_preparation_status=status
        self.rag_frontend_last_plan=asdict(self.rag_frontend_plan)
        return True

    def _resume_rag_joint_frontend_preparation(self):
        """Resume owned intent at a native boundary without resetting tick grants."""
        # An in-progress joint intent owns this boundary. In particular, do
        # not overwrite its target with a legacy compute/HBM decision while
        # the selected native conversion is still pending.
        if getattr(self, 'rag_joint_pending_preparation', None) is not None:
            decision, status = self._resume_rag_joint_preparation()
            self.rag_joint_last_preparation_status = status
            if status != 'ready' and status not in (
                    'kv_submitted', 'kv_wait', 'compute_wait',
                    'workspace_submitted', 'workspace_wait'):
                # Stale/failed preparation is not permission to dispatch the
                # old intent. Preserve compatible decode, but grant no new
                # frontend work at this boundary; never search again here.
                self.rag_joint_active_event_id = None
                self.rag_joint_active_memory_plan = None
                self.rag_frontend_plan = FrontendPlan(
                    allow_ann=False, allow_prefill=False, frontend_needed=False,
                    target_stream_idx=get_current_stream_idx(),
                    reason='joint_preparation_' + status,
                    target_reason='joint_continuation_invalidated',
                    budget_reason='joint_no_new_admission',
                    ann_batch_cap=0, prefill_forward_count=0,
                    joint_event_id=decision.event_id)
            self.rag_prefill_run_allowed = (
                status == 'ready' and self.rag_frontend_plan.allow_prefill)
            self.rag_frontend_last_plan = asdict(self.rag_frontend_plan)
            return True
        return False

    def _plan_rag_frontend_actions(self):
        self.rag_frontend_ann_launched = False
        self.rag_frontend_prefill_ran = False
        self.rag_lease_last_switch_benefit_ms = None
        self.rag_lease_last_transition_cost_ms = None
        self.rag_ann_batches_granted_this_tick = 0
        self.rag_ann_tick_grants_recorded = False
        self.rag_ann_tick_limit_held_this_tick = False
        self._prune_rag_ann_handoffs()
        if self._resume_rag_joint_frontend_preparation():
            return
        if self._plan_rag_bound_joint_event():
            return

        arbitration = getattr(self, "rag_ann_frontend_arbitration", "overlap")
        pending = getattr(self, "ann_search_pending", None)
        has_ann = bool(pending)
        active_ann = self._rag_ann_gpu_work_active()
        split_batch = getattr(self, "split_prefill_batch", None)
        has_prefill = split_batch is not None and not split_batch.is_empty()
        split_finished = bool(
            split_batch
            and getattr(split_batch, "split_prefill_finished", False)
        )
        decode_active = self._decode_batch_active()
        policy_mode = getattr(self, "rag_policy_mode", "full")
        work_conserving_decode_overlap = (
            self._rag_ann_work_conserving_decode_overlap_enabled()
        )
        current_stream_idx = get_current_stream_idx()
        idle_stream_idx = (
            self.real_sm_group_num - 1 if decode_active else 0
        )
        if (
            not has_ann
            and not active_ann
            and not has_prefill
            and not split_finished
            and not getattr(self, "rag_prefill_handoff_active", False)
            and not getattr(self, "rag_decode_recovery_pending", False)
            and not getattr(self, "rag_handoff_decode_release_pending", False)
            and getattr(self, "rag_partition_transition_intent", None) is None
            and current_stream_idx == idle_stream_idx
        ):
            # A decode-only fallback is already satisfied at idle_stream_idx.
            # With no frontend work or pending transition, recomputing the full
            # planner cannot grant anything. Keep the fallback flag latched so
            # future ANN/prefill still goes through the ordinary safety gates.
            plan = FrontendPlan(
                allow_ann=False,
                allow_prefill=False,
                frontend_needed=False,
                target_stream_idx=current_stream_idx,
                reason="idle",
                target_reason=(
                    "decode_only_resident"
                    if decode_active
                    else "no_decode_resident"
                ),
                budget_reason="idle",
                system_load_low=True,
                system_load_reason="idle",
            )
            self.rag_frontend_plan = plan
            self.rag_prefill_run_allowed = False
            self.rag_frontend_idle_fast_path_hits = (
                getattr(self, "rag_frontend_idle_fast_path_hits", 0) + 1
            )
            self._increment_histogram(
                self.rag_frontend_plan_reason_counts, plan.reason
            )
            self._increment_histogram(
                self.rag_frontend_target_reason_counts, plan.target_reason
            )
            self.rag_frontend_last_plan = asdict(plan)
            return
        if (
            work_conserving_decode_overlap
            and decode_active
            and (has_ann or active_ann)
            and not has_prefill
            and not split_finished
            and not getattr(self, "rag_prefill_handoff_active", False)
            and not getattr(self, "rag_decode_recovery_pending", False)
            and not getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
            and not getattr(self, "rag_handoff_decode_release_pending", False)
            and getattr(self, "rag_partition_transition_intent", None) is None
            and current_stream_idx == idle_stream_idx
        ):
            # ANN already runs on Decode's sibling stream and the full Decode
            # partition is resident. No candidate, contention, or partition
            # decision can change this tick, so keep the hot path O(1).
            ann_wait_ms = self._oldest_ann_wait_ms() if has_ann else 0.0
            budget = {
                "allowed": True,
                "forced": False,
                "reason": "decode_overlap",
                "predicted_ms": None,
                "slo_ms": None,
            }
            plan = FrontendPlan(
                allow_ann=has_ann,
                allow_prefill=False,
                frontend_needed=False,
                target_stream_idx=current_stream_idx,
                reason=(
                    "ann_decode_resident_fast_path"
                    if has_ann
                    else "ann_decode_inflight_fast_path"
                ),
                target_reason="decode_only_resident",
                budget_reason="decode_overlap",
                ann_wait_ms=ann_wait_ms,
                prefill_wait_ms=0.0,
                launch_ann_early=has_ann,
                candidate_stream_idx=current_stream_idx,
                system_load_low=True,
                system_load_reason="ann_decode_resident",
            )
            self.rag_ann_last_budget = budget
            self.rag_frontend_plan = plan
            self.rag_prefill_run_allowed = False
            self.rag_frontend_ann_decode_fast_path_hits = (
                getattr(self, "rag_frontend_ann_decode_fast_path_hits", 0) + 1
            )
            self._increment_histogram(
                self.rag_frontend_plan_reason_counts, plan.reason
            )
            self._increment_histogram(
                self.rag_frontend_target_reason_counts, plan.target_reason
            )
            self.rag_frontend_last_plan = asdict(plan)
            return
        ann_wait_ms = self._oldest_ann_wait_ms()
        prefill_wait_ms = self._oldest_prefill_wait_ms()
        ann_pressure = ann_wait_ms / max(
            0.001, getattr(self, "rag_ann_deadline_ms", 50.0)
        )
        prefill_pressure = prefill_wait_ms / max(
            0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
        )
        candidate_stream_idx = self._rag_stage_candidate_stream_idx(
            has_ann=has_ann,
            has_prefill=has_prefill,
        )
        system_load = self._rag_system_load_state(candidate_stream_idx)
        if policy_mode == "dynamic-only":
            budget = {
                "allowed": True,
                "forced": False,
                "reason": "disabled_dynamic_only",
                "predicted_ms": None,
                "slo_ms": None,
            }
        else:
            budget = self._evaluate_rag_ann_decode_budget(
                ann_wait_ms, candidate_stream_idx
            )
        if work_conserving_decode_overlap and has_ann:
            # Preserve the counterfactual guarded decision for diagnostics, but
            # do not convert Decode protection into ANN queueing on its lane.
            budget = {
                **budget,
                "guarded_allowed": budget["allowed"],
                "guarded_forced": budget["forced"],
                "guarded_reason": budget["reason"],
                "allowed": True,
                "forced": False,
                "reason": "decode_overlap",
            }
        predicted_ms = budget["predicted_ms"]
        slo_ms = budget["slo_ms"]
        decode_slack_ms = (
            slo_ms - predicted_ms
            if slo_ms is not None and predicted_ms is not None
            else None
        )
        safety_margin_ms = getattr(self, "rag_ann_safety_margin_ms", 5.0)
        decode_safe = (
            decode_slack_ms is None or decode_slack_ms > safety_margin_ms
        )
        prefill_floor_idx = self._adaptive_frontend_stream_idx(
            self._rag_current_decode_batch_size()
        )
        prefill_floor_safe_override = bool(
            has_prefill
            and not decode_safe
            and self._rag_decode_partition_within_slo(prefill_floor_idx)
        )
        frontend_decode_safe = decode_safe or prefill_floor_safe_override
        low_pressure_fast_path = self._rag_low_pressure_fast_path_allowed(
            has_ann=has_ann,
            has_prefill=has_prefill,
            active_ann=active_ann,
            system_load=system_load,
        )
        prefill_batch_p95 = system_load.get("batch_p95")
        prefill_queue_p95_ms = system_load.get("queue_p95_ms")
        prefill_candidate_ms = system_load.get("predicted_ms")
        prefill_candidate_slo_ms = system_load.get("slo_ms")
        prefill_batch_low = bool(
            prefill_batch_p95 is not None
            and prefill_batch_p95
            <= getattr(
                self,
                "rag_ann_fast_path_decode_batch_p95_threshold",
                16.0,
            )
        )
        prefill_queue_low = bool(
            prefill_queue_p95_ms is not None
            and prefill_queue_p95_ms
            <= getattr(
                self, "rag_ann_fast_path_decode_queue_p95_ms", 2.0
            )
        )
        prefill_relaxed_candidate_safe = bool(
            prefill_candidate_ms is not None
            and (
                prefill_candidate_slo_ms is None
                or prefill_candidate_ms + safety_margin_ms
                < prefill_candidate_slo_ms
                * max(
                    1.0,
                    getattr(
                        self, "rag_ann_max_decode_slowdown_ratio", 1.0
                    ),
                )
            )
        )
        prefill_tiny_batch_load = bool(
            prefill_batch_p95 is not None
            and prefill_batch_p95
            <= getattr(
                self,
                "rag_prefill_low_load_batch_p95_threshold",
                4.0,
            )
            and prefill_queue_low
        )
        prefill_arrival_rps = system_load.get("arrival_rps")
        prefill_arrival_threshold = getattr(
            self, "rag_prefill_low_load_arrival_rps_threshold", -1.0
        )
        prefill_work_equivalent_rps = self._rag_prefill_work_equivalent_rps(
            prefill_arrival_rps
        )
        raw_prefill_arrival_low = bool(
            prefill_arrival_threshold > 0.0
            and prefill_arrival_rps is not None
            and prefill_arrival_rps <= prefill_arrival_threshold
        )
        prefill_workload_low = bool(
            prefill_arrival_threshold > 0.0
            and prefill_work_equivalent_rps is not None
            and prefill_work_equivalent_rps <= prefill_arrival_threshold
        )
        prefill_observed_safe_override = bool(
            raw_prefill_arrival_low
            and prefill_batch_low
            and prefill_relaxed_candidate_safe
        )
        prefill_arrival_low = bool(
            prefill_workload_low or prefill_observed_safe_override
        )
        if raw_prefill_arrival_low and not prefill_arrival_low:
            self.rag_prefill_workload_fast_path_rejects = (
                getattr(
                    self,
                    "rag_prefill_workload_fast_path_rejects",
                    0,
                )
                + 1
            )
        low_pressure_prefill_load = bool(
            system_load.get("low", False)
            or prefill_arrival_low
            or prefill_tiny_batch_load
            or (
                prefill_batch_low
                and prefill_queue_low
                and prefill_relaxed_candidate_safe
            )
        )
        low_pressure_prefill_fast_path = bool(
            work_conserving_decode_overlap
            and policy_mode == "full"
            and arbitration == "fair"
            and self._rag_ann_uses_partitioned_gc()
            and has_prefill
            and getattr(self, "rag_prefill_handoff_active", False)
            and low_pressure_prefill_load
            and not getattr(self, "rag_decode_recovery_pending", False)
            and not getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
        )
        no_decode_prefill_sprint = bool(
            low_pressure_prefill_fast_path
            and prefill_arrival_low
            and not active_ann
            and not self._decode_batch_active()
            and self._rag_prefill_sprint_eligible()
        )
        low_pressure_exclusive_prefill_sprint = bool(
            getattr(self, "rag_prefill_exclusive_sprint", False)
            and getattr(self, "rag_prefill_dynamic_lease", False)
            and low_pressure_prefill_fast_path
            and not active_ann
            and self._decode_batch_active()
            and getattr(
                self, "rag_prefill_exclusive_sprint_available", False
            )
            and self._rag_current_decode_batch_size()
            <= max(
                1,
                int(
                    getattr(
                        self,
                        "rag_prefill_low_load_batch_p95_threshold",
                        4,
                    )
                ),
            )
            and self._rag_prefill_sprint_eligible()
        )
        if (
            getattr(self, "rag_prefill_exclusive_sprint", False)
            and getattr(self, "rag_prefill_dynamic_lease", False)
            and low_pressure_prefill_fast_path
            and self._decode_batch_active()
            and self._rag_prefill_sprint_eligible()
        ):
            if active_ann:
                exclusive_block_reason = "active_ann"
            elif not getattr(
                self, "rag_prefill_exclusive_sprint_available", False
            ):
                exclusive_block_reason = "handoff_already_sprinted"
            elif (
                self._rag_current_decode_batch_size()
                > max(
                    1,
                    int(
                        getattr(
                            self,
                            "rag_prefill_low_load_batch_p95_threshold",
                            4,
                        )
                    ),
                )
            ):
                exclusive_block_reason = "decode_batch"
            elif has_ann:
                exclusive_block_reason = "granted_with_pending_ann"
            else:
                exclusive_block_reason = "granted"
            self._increment_histogram(
                self.rag_prefill_exclusive_sprint_block_counts,
                exclusive_block_reason,
            )
        prefill_deadline_due = prefill_pressure >= 1.0
        configured_handoff_ms = getattr(
            self, "rag_ann_handoff_timeout_ms", 0.0
        )
        handoff_wait_limit_ms = max(
            0.001,
            min(
                getattr(self, "rag_prefill_deadline_ms", 500.0),
                configured_handoff_ms
                if configured_handoff_ms > 0.0
                else getattr(self, "rag_prefill_deadline_ms", 500.0),
            ),
        )
        dependency_handoff_wait_ms = (
            self._oldest_dependency_prefill_wait_ms()
        )
        if (
            dependency_handoff_wait_ms <= 0.0
            and getattr(self, "rag_prefill_handoff_active", False)
        ):
            # Compatibility for requests created before handoff timestamps were
            # recorded. New requests use the phase-local scheduler-arrival age.
            dependency_handoff_wait_ms = prefill_wait_ms
        dependency_prefill_due = bool(
            work_conserving_decode_overlap
            and policy_mode == "full"
            and arbitration == "fair"
            and self._rag_ann_uses_partitioned_gc()
            and has_prefill
            and getattr(self, "rag_prefill_handoff_active", False)
            and getattr(
                self, "rag_prefill_exclusive_sprint_available", False
            )
            and not low_pressure_prefill_fast_path
            and dependency_handoff_wait_ms >= handoff_wait_limit_ms
            and not getattr(self, "rag_decode_recovery_pending", False)
            and not getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
        )
        deadline_exclusive_prefill_sprint = bool(
            getattr(self, "rag_prefill_exclusive_sprint", False)
            and getattr(self, "rag_prefill_dynamic_lease", False)
            and dependency_prefill_due
            and not active_ann
            and self._decode_batch_active()
            and self._rag_prefill_sprint_eligible()
        )
        if dependency_prefill_due and getattr(
            self, "rag_prefill_exclusive_sprint", False
        ):
            deadline_sprint_reason = (
                "deadline_granted"
                if deadline_exclusive_prefill_sprint
                else (
                    "deadline_active_ann"
                    if active_ann
                    else "deadline_ineligible"
                )
            )
            self._increment_histogram(
                self.rag_prefill_exclusive_sprint_block_counts,
                deadline_sprint_reason,
            )
        bounded_exclusive_prefill_sprint = bool(
            low_pressure_exclusive_prefill_sprint
            or deadline_exclusive_prefill_sprint
        )
        prefill_sprint = bool(
            no_decode_prefill_sprint or bounded_exclusive_prefill_sprint
        )
        prefill_endpoint_sprint = prefill_sprint
        if work_conserving_decode_overlap:
            # Decode-GC ANN already bypasses admission/coalescing. Keep its
            # unrelated requests on Decode. Low-load handoffs enter immediately;
            # overdue handoffs receive a bounded endpoint bundle below.
            low_pressure_fast_path = low_pressure_prefill_fast_path
        elif budget["forced"]:
            # max_defer is the liveness bound for both aggregation and cost
            # admission, even when the independent load signal remains low.
            low_pressure_fast_path = False
        max_prefill_burst = max(
            1, getattr(self, "rag_prefill_max_consecutive_chunks", 2)
        )
        dependency_recovery_due = bool(
            policy_mode == "full"
            and arbitration == "fair"
            and self._decode_batch_active()
            and getattr(self, "rag_decode_recovery_pending", False)
        )
        prefill_lease_expired = bool(
            policy_mode == "full"
            and arbitration == "fair"
            and self._decode_batch_active()
            and has_prefill
            and not split_finished
            and self.rag_prefill_consecutive_chunks >= max_prefill_burst
        )
        decode_recovery_due = bool(
            dependency_recovery_due or prefill_lease_expired
        )
        if (
            getattr(self, "rag_handoff_decode_release_pending", False)
            and not self._decode_batch_active()
        ):
            self.rag_handoff_decode_release_pending = False
        handoff_release_due = bool(
            policy_mode == "full"
            and not getattr(self, "rag_prefill_dynamic_lease", False)
            and getattr(self, "rag_handoff_decode_release_pending", False)
            and self._decode_batch_active()
        )
        allow_ann = has_ann and (budget["allowed"] or low_pressure_fast_path)
        allow_prefill = has_prefill
        controlled_overlap = False
        immediate_handoff = False
        reason = arbitration

        if policy_mode == "dynamic-only":
            allow_ann = False
            allow_prefill = False
            if active_ann:
                reason = "dynamic_rr_ann_active"
            elif has_ann and has_prefill:
                if getattr(self, "rag_dynamic_rr_next", "ann") == "ann":
                    allow_ann = True
                    reason = "dynamic_rr_ann"
                else:
                    allow_prefill = True
                    reason = "dynamic_rr_prefill"
            elif has_ann:
                allow_ann = True
                reason = "dynamic_rr_ann_only"
            elif has_prefill:
                allow_prefill = True
                reason = "dynamic_rr_prefill_only"
            else:
                reason = "idle"
        elif handoff_release_due:
            allow_ann = bool(has_ann and work_conserving_decode_overlap)
            allow_prefill = False
            controlled_overlap = False
            reason = (
                "ann_decode_overlap_handoff_release"
                if allow_ann
                else "handoff_decode_release"
            )
            self.rag_ann_frontend_residency_until = 0.0
        elif decode_recovery_due:
            # Expire a Prefill lease with one full-SM Decode step. The hard
            # chunk bound applies even while the mixed partition is predicted
            # safe, so sustained Prefill backlog cannot turn a bounded lease
            # into permanent Decode throttling.
            # This branch intentionally precedes deadline/forced arbitration:
            # the recovery is one step and therefore preserves liveness.
            allow_ann = bool(has_ann and work_conserving_decode_overlap)
            allow_prefill = False
            reason = (
                "ann_decode_overlap_recovery" if allow_ann else "decode_recovery"
            )
            if not getattr(self, "rag_decode_recovery_pending", False):
                self.rag_decode_recovery_requests = (
                    getattr(self, "rag_decode_recovery_requests", 0) + 1
                )
            self.rag_decode_recovery_pending = True
            self.rag_ann_frontend_residency_until = 0.0
        elif dependency_prefill_due:
            # Dependency-ready Prefill is on the request critical path. Grant
            # a bounded mixed lease while retaining the partition guards below.
            allow_ann = bool(has_ann and work_conserving_decode_overlap)
            allow_prefill = True
            controlled_overlap = allow_ann
            immediate_handoff = True
            reason = "handoff_bounded"
        elif arbitration == "overlap":
            allow_ann = has_ann
            reason = "overlap"
        elif arbitration == "strict":
            allow_ann = has_ann
            allow_prefill = has_prefill and not active_ann
            reason = "strict_ann" if has_ann or active_ann else "strict_prefill"
        elif (
            arbitration == "fair"
            and work_conserving_decode_overlap
            and has_ann
            and not low_pressure_prefill_fast_path
        ):
            # ANN owns a sibling stream in Decode's GreenContext. Admit it
            # independently of Prefill so ANN+Decode remains work-conserving;
            # when a mixed partition is needed, all three stages may overlap.
            allow_ann = True
            allow_prefill = has_prefill
            controlled_overlap = has_prefill
            immediate_handoff = bool(
                has_prefill
                and getattr(self, "rag_prefill_handoff_active", False)
            )
            reason = "ann_decode_overlap"
        elif (
            arbitration == "fair"
            and has_prefill
            and (
                low_pressure_prefill_fast_path
                or (
                    low_pressure_fast_path
                    and getattr(self, "rag_prefill_handoff_active", False)
                )
            )
        ):
            # A dependency-ready Prefill owns the frontend lease, but ANN in
            # Decode's sibling stream does not consume that frontend pool.
            # Keep that ANN lane work-conserving so the fast handoff does not
            # turn into a global ANN admission barrier.
            allow_ann = bool(has_ann and work_conserving_decode_overlap)
            allow_prefill = True
            controlled_overlap = allow_ann
            immediate_handoff = bool(
                getattr(self, "rag_prefill_handoff_active", False)
            )
            if allow_ann:
                reason = (
                    "ann_decode_overlap_handoff_fast_path"
                    if immediate_handoff
                    else "ann_decode_overlap_prefill_fast_path"
                )
            else:
                reason = (
                    "handoff_fast_path"
                    if immediate_handoff
                    else "prefill_fast_path"
                )
        elif arbitration == "fair" and low_pressure_fast_path and has_ann:
            allow_ann = True
            allow_prefill = False
            reason = "ann_fast_path"
        elif arbitration == "fair" and budget["forced"] and has_ann:
            allow_ann = True
            allow_prefill = False
            reason = "ann_max_defer"
        elif (
            arbitration == "fair"
            and has_ann
            and budget["reason"] == "decode_urgency"
        ):
            max_ann = getattr(self, "rag_ann_max_consecutive_batches", 1)
            if has_prefill and self.rag_ann_consecutive_batches >= max_ann:
                allow_ann = False
                allow_prefill = True
                reason = "prefill_fairness"
            else:
                allow_ann = True
                allow_prefill = False
                reason = "ann_urgency"
        elif (
            arbitration == "fair"
            and has_ann
            and budget["allowed"]
            and ann_pressure >= 1.0
            and (
                not has_prefill
                or self.rag_ann_consecutive_batches
                < getattr(self, "rag_ann_max_consecutive_batches", 1)
            )
        ):
            # A decode-safe ANN that has reached its service deadline should
            # not keep losing to an older Prefill request. The consecutive
            # batch limit still guarantees a Prefill turn under sustained ANN
            # pressure. max_defer remains the later liveness escape hatch for
            # requests that cannot satisfy the decode budget.
            allow_ann = True
            allow_prefill = False
            reason = "ann_deadline"
        elif arbitration == "fair" and not frontend_decode_safe:
            if max(ann_pressure, prefill_pressure) < 1.0:
                allow_ann = False
                allow_prefill = False
                reason = "decode_budget"
            elif (
                has_ann
                and budget["allowed"]
                and ann_pressure >= prefill_pressure
            ):
                allow_ann = True
                allow_prefill = False
                reason = "ann_deadline"
            else:
                allow_ann = False
                # Full must not borrow an inadmissible ANN's overdue age to
                # bypass the decode budget for a not-yet-due prefill batch.
                # Preserve non-full policy behavior. Baseline runtime versions
                # must remain sealed independently when testing this repair.
                allow_prefill = bool(has_prefill and (
                    policy_mode != "full" or prefill_pressure >= 1.0
                ))
                reason = "prefill_deadline" if allow_prefill else "decode_budget"
        elif arbitration == "fair" and has_ann and not budget["allowed"]:
            allow_ann = False
            allow_prefill = has_prefill
            reason = "ann_guard" if has_prefill else budget["reason"]
        elif arbitration == "fair" and has_ann and has_prefill:
            max_ann = getattr(self, "rag_ann_max_consecutive_batches", 1)
            max_prefill = getattr(
                self, "rag_prefill_max_consecutive_chunks", 2
            )
            if self.rag_ann_consecutive_batches >= max_ann:
                allow_ann = False
                reason = "prefill_fairness"
            elif self.rag_prefill_consecutive_chunks >= max_prefill:
                allow_prefill = False
                reason = "ann_fairness"
            else:
                overlap_threshold = getattr(
                    self, "rag_ann_overlap_pressure_threshold", 0.5
                )
                enable_overlap = getattr(
                    self, "rag_ann_enable_controlled_overlap", False
                )
                if enable_overlap and max(
                    ann_pressure, prefill_pressure
                ) >= overlap_threshold:
                    controlled_overlap = True
                    reason = "controlled_overlap"
                else:
                    allow_ann = ann_pressure > prefill_pressure
                    allow_prefill = not allow_ann
                    reason = (
                        "ann_pressure" if allow_ann else "prefill_pressure"
                    )
        elif arbitration == "fair" and has_ann:
            reason = "ann_only"
        elif arbitration == "fair" and has_prefill:
            reason = "prefill_only"
        elif not has_ann and not has_prefill and not active_ann:
            reason = "idle"

        if allow_prefill and prefill_floor_safe_override:
            self.rag_prefill_floor_safe_admissions += 1

        if (
            arbitration == "fair"
            and has_ann
            and not budget["allowed"]
            and not low_pressure_fast_path
        ):
            self._record_rag_ann_budget_reject()

        if not (
            low_pressure_fast_path or dependency_prefill_due
        ) and self._should_wait_for_rag_lease_cohort(
            allow_ann=allow_ann,
            allow_prefill=allow_prefill,
            active_ann=active_ann,
            forced_ann=bool(budget["forced"] and has_ann),
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
        ):
            if work_conserving_decode_overlap and has_ann:
                # Prefill can keep coalescing for a worthwhile mixed-partition
                # lease while ANN proceeds in the current Decode lane.
                allow_ann = True
                allow_prefill = False
                controlled_overlap = False
                immediate_handoff = False
                reason = "ann_decode_overlap_prefill_coalesce"
            else:
                allow_ann = False
                allow_prefill = False
                controlled_overlap = False
                reason = "lease_coalesce_wait"
            self.rag_lease_coalesce_waits += 1

        ann_uses_frontend = self._rag_ann_uses_frontend_pool()
        frontend_activity = bool(
            allow_prefill
            or split_finished
            or (ann_uses_frontend and (allow_ann or active_ann))
        )
        if frontend_activity:
            self.rag_last_frontend_activity_ts = time.perf_counter()

        residency_active = time.perf_counter() < getattr(
            self, "rag_ann_frontend_residency_until", 0.0
        )
        current_stream_idx = get_current_stream_idx()
        dependency_residency_active = bool(
            policy_mode == "full"
            and self._rag_ann_uses_frontend_pool()
            # Logical handoffs may span document assembly and tokenization.
            # Only the short explicit lease may retain a mixed partition.
            and residency_active
            and not low_pressure_fast_path
            and not handoff_release_due
            and not getattr(self, "rag_decode_recovery_pending", False)
            and not getattr(
                self, "rag_partition_decode_only_fallback_required", False
            )
            and not (allow_ann or allow_prefill or active_ann)
            and 1
            <= current_stream_idx
            <= getattr(self, "real_sm_group_num", 1) - 2
        )
        frontend_needed = (
            (ann_uses_frontend and active_ann)
            or (ann_uses_frontend and has_ann and allow_ann)
            or (has_prefill and allow_prefill)
            or split_finished
            or (
                ann_uses_frontend
                and residency_active
                and policy_mode != "full"
            )
            or dependency_residency_active
        )
        forced_ann = bool(
            policy_mode != "dynamic-only"
            and budget["forced"]
            and has_ann
            and not low_pressure_fast_path
        )
        force_decode_only = bool(
            handoff_release_due
            or getattr(self, "rag_decode_recovery_pending", False)
        )
        target_idx, target_reason = self._select_frontend_plan_target(
            frontend_needed=frontend_needed,
            allow_ann=allow_ann,
            allow_prefill=allow_prefill,
            active_ann=active_ann,
            forced_ann=forced_ann,
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
            predicted_decode_ms=predicted_ms,
            dependency_handoff_pending=dependency_residency_active,
            dependency_prefill_due=dependency_prefill_due,
            force_decode_only=force_decode_only,
            force_decode_only_reason=(
                "handoff_decode_release"
                if handoff_release_due
                else "decode_recovery"
            ),
            bypass_work_wait=(
                low_pressure_fast_path or dependency_prefill_due
            ),
            bypass_partition_guards=(
                low_pressure_fast_path or deadline_exclusive_prefill_sprint
            ),
            fast_path_candidate_idx=(
                0 if prefill_endpoint_sprint else candidate_stream_idx
            ),
        )
        if deadline_exclusive_prefill_sprint and target_idx == 0:
            target_reason = "prefill_deadline_endpoint"

        # The endpoint groups use ordinary full-GPU CUDA streams. Their SM
        # tuples express scheduler ownership, not a hardware zero-SM context.
        # Therefore Prefill must never run on the decode-only endpoint while
        # Decode is active: doing so silently degenerates into free GPU
        # contention. An overdue dependency may use an explicitly bounded
        # frontend endpoint lease; otherwise Prefill borrows a legal mixed group.
        decode_only_idx = self.real_sm_group_num - 1
        prefill_on_decode_only = bool(
            allow_prefill
            and self._decode_batch_active()
            and target_idx == decode_only_idx
        )
        if prefill_on_decode_only:
            prefill_deadline_reached = prefill_wait_ms >= max(
                0.001, getattr(self, "rag_prefill_deadline_ms", 500.0)
            )
            last_mixed_idx = decode_only_idx - 1
            fallback_idx = self._apply_rag_decode_partition_floor(
                last_mixed_idx
            )
            can_borrow_mixed = bool(
                not force_decode_only
                and (dependency_prefill_due or prefill_deadline_reached)
                and 1 <= fallback_idx <= last_mixed_idx
            )
            if can_borrow_mixed:
                target_idx = fallback_idx
                target_reason = "prefill_decode_only_mixed_fallback"
                frontend_needed = True
                self.rag_prefill_decode_only_mixed_fallbacks = (
                    getattr(
                        self,
                        "rag_prefill_decode_only_mixed_fallbacks",
                        0,
                    )
                    + 1
                )
            else:
                allow_prefill = False
                immediate_handoff = False
                low_pressure_fast_path = False
                reason = "prefill_decode_only_deferred"
                frontend_needed = bool(
                    (ann_uses_frontend and active_ann) or split_finished
                )
                self.rag_prefill_decode_only_deferrals = getattr(
                    self, "rag_prefill_decode_only_deferrals", 0
                ) + 1

        lease_deferred = bool(
            not active_ann
            and (
                target_reason.endswith("_coalesced_cost")
                or target_reason.endswith("_coalesced_work")
            )
        )
        if lease_deferred:
            if work_conserving_decode_overlap and has_ann:
                allow_ann = True
                allow_prefill = False
                controlled_overlap = False
                immediate_handoff = False
                reason = "ann_decode_overlap_switch_deferred"
                frontend_needed = bool(split_finished)
            else:
                allow_ann = False
                allow_prefill = False
                controlled_overlap = False
                immediate_handoff = False
                low_pressure_fast_path = False
                reason = "lease_switch_deferred"
                frontend_needed = bool(
                    (ann_uses_frontend and active_ann) or split_finished
                )

        if allow_ann and budget["reason"] == "decode_urgency":
            self._record_rag_ann_urgency_admission(ann_wait_ms)
        self.rag_ann_last_budget = dict(budget)

        if forced_ann:
            pending[0].setdefault("forced_release_started", time.perf_counter())

        if allow_prefill:
            prefill_forward_count, prefill_layer_ms, e2e_slack_ms = (
                self._rag_prefill_lease_shape(target_idx, predicted_ms)
            )
            if (
                target_reason.startswith("prefill_lease_tail_liveness")
                and prefill_forward_count is not None
                and prefill_layer_ms is not None
            ):
                requested_layers = prefill_forward_count
                prefill_forward_count, liveness_budget_ms = (
                    self._rag_tail_liveness_layer_budget(
                        target_idx,
                        requested_layers,
                        prefill_layer_ms,
                    )
                )
                self.rag_prefill_tail_liveness_budget_ms.append(
                    liveness_budget_ms
                )
                if prefill_forward_count < requested_layers:
                    self.rag_prefill_tail_liveness_layer_caps += 1
                if prefill_forward_count <= 0:
                    self.rag_prefill_two_level_safety_deferrals += 1
                    allow_prefill = False
                    controlled_overlap = False
                    immediate_handoff = False
                    low_pressure_fast_path = False
                    reason = "prefill_two_level_budget_deferred"
                    frontend_needed = bool(
                        (ann_uses_frontend and active_ann) or split_finished
                    )
                    if not active_ann and not allow_ann:
                        target_idx = get_current_stream_idx()
                        target_reason = (
                            "prefill_two_level_budget_deferred_resident"
                        )
                    prefill_forward_count = None
            if allow_prefill and low_pressure_exclusive_prefill_sprint:
                remaining_prefill_layers = max(
                    1,
                    int(
                        getattr(self.model_config, "num_hidden_layers", 1)
                    )
                    - int(
                        getattr(self.split_prefill_batch, "split_index", 0)
                    ),
                )
                prefill_forward_count = min(
                    remaining_prefill_layers,
                    max(
                        1,
                        int(
                            getattr(
                                self, "rag_prefill_lease_max_layers", 1
                            )
                        ),
                    ),
                )
        else:
            prefill_forward_count = prefill_layer_ms = e2e_slack_ms = None

        plan = FrontendPlan(
            allow_ann=allow_ann,
            allow_prefill=allow_prefill,
            frontend_needed=frontend_needed,
            target_stream_idx=target_idx,
            reason=reason,
            target_reason=target_reason,
            budget_reason=budget["reason"],
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
            predicted_decode_ms=predicted_ms,
            decode_slack_ms=decode_slack_ms,
            forced_ann=forced_ann,
            controlled_overlap=controlled_overlap,
            low_pressure_fast_path=low_pressure_fast_path,
            launch_ann_early=bool(
                allow_ann
                and (
                    low_pressure_fast_path
                    or work_conserving_decode_overlap
                    or (
                        policy_mode == "full"
                        and budget["reason"] in {"decode_urgency", "max_defer"}
                    )
                )
            ),
            immediate_handoff=immediate_handoff,
            dependency_prefill_due=dependency_prefill_due,
            prefill_sprint=prefill_sprint,
            prefill_exclusive_sprint=bounded_exclusive_prefill_sprint,
            candidate_stream_idx=candidate_stream_idx,
            system_load_low=bool(
                system_load["low"] or low_pressure_prefill_fast_path
            ),
            system_load_reason=(
                "prefill_arrival"
                if (
                    low_pressure_prefill_fast_path
                    and prefill_arrival_low
                )
                else (
                    "prefill_tiny_batch"
                    if (
                        low_pressure_prefill_fast_path
                        and prefill_tiny_batch_load
                        and not system_load["low"]
                    )
                    else (
                        "prefill_relaxed"
                        if low_pressure_prefill_fast_path
                        and not system_load["low"]
                        else system_load["reason"]
                    )
                )
            ),
            decode_batch_p95=system_load["batch_p95"],
            decode_queue_p95_ms=system_load["queue_p95_ms"],
            request_arrival_rps=system_load.get("arrival_rps"),
            prefill_work_equivalent_rps=prefill_work_equivalent_rps,
            dependency_handoff_wait_ms=dependency_handoff_wait_ms,
            prefill_forward_count=prefill_forward_count,
            predicted_prefill_layer_ms=prefill_layer_ms,
            e2e_slack_ms=e2e_slack_ms,
            switch_benefit_ms=getattr(
                self, "rag_lease_last_switch_benefit_ms", None
            ),
            round_trip_switch_cost_ms=getattr(
                self, "rag_lease_last_transition_cost_ms", None
            ),
        )
        self.rag_frontend_plan = plan

        self.rag_prefill_run_allowed = allow_prefill
        self._increment_histogram(self.rag_frontend_plan_reason_counts, reason)
        self._increment_histogram(
            self.rag_frontend_target_reason_counts, target_reason
        )
        self.rag_frontend_last_plan = asdict(plan)

    def _finalize_rag_frontend_actions(self):
        self._record_rag_ann_tick_grants()
        ann_launched = getattr(self, "rag_frontend_ann_launched", False)
        prefill_ran = getattr(self, "rag_frontend_prefill_ran", False)
        plan = getattr(self, "rag_frontend_plan", None)
        if prefill_ran and plan is not None and plan.prefill_sprint:
            self.rag_prefill_sprint_chunks = (
                getattr(self, "rag_prefill_sprint_chunks", 0) + 1
            )
            if plan.prefill_exclusive_sprint:
                self.rag_prefill_exclusive_sprint_chunks = (
                    getattr(self, "rag_prefill_exclusive_sprint_chunks", 0)
                    + 1
                )
                # A dependency handoff grants one bounded endpoint bundle.
                # Remaining Prefill layers must return to normal lease planning.
                self.rag_prefill_exclusive_sprint_available = False
        if prefill_ran and plan is not None and plan.dependency_prefill_due:
            # A dependency deadline accelerates only the first Prefill bundle
            # after handoff. Keeping it armed for every remaining layer would
            # manufacture repeated Decode recovery windows and lengthen TTFT.
            self.rag_prefill_exclusive_sprint_available = False
            self.rag_dependency_prefill_admissions = (
                getattr(self, "rag_dependency_prefill_admissions", 0) + 1
            )
            self.rag_dependency_prefill_burst_chunks = (
                getattr(self, "rag_dependency_prefill_burst_chunks", 0) + 1
            )
            max_dependency_burst = max(
                1,
                min(
                    3,
                    getattr(self, "rag_prefill_max_consecutive_chunks", 3),
                ),
            )
            if (
                self._decode_batch_active()
                and self.rag_dependency_prefill_burst_chunks
                >= max_dependency_burst
            ):
                if not getattr(self, "rag_decode_recovery_pending", False):
                    self.rag_decode_recovery_requests = (
                        getattr(self, "rag_decode_recovery_requests", 0) + 1
                    )
                self.rag_decode_recovery_pending = True
                self.rag_ann_frontend_residency_until = 0.0

        decision = None
        if ann_launched and not prefill_ran:
            decision = "ann_only"
            self.rag_ann_consecutive_batches += 1
            self.rag_prefill_consecutive_chunks = 0
            if getattr(self, "rag_policy_mode", "full") == "dynamic-only":
                self.rag_dynamic_rr_next = "prefill"
        elif prefill_ran and not ann_launched:
            decision = "prefill_only"
            self.rag_prefill_consecutive_chunks += 1
            self.rag_ann_consecutive_batches = 0
            if getattr(self, "rag_policy_mode", "full") == "dynamic-only":
                self.rag_dynamic_rr_next = "ann"
        elif ann_launched and prefill_ran:
            decision = "controlled_overlap"
            self.rag_ann_consecutive_batches = 0
            self.rag_prefill_consecutive_chunks = 0
        elif (
            getattr(self, "rag_ann_frontend_arbitration", "overlap") == "fair"
            and (
                bool(getattr(self, "ann_search_pending", None))
                or bool(getattr(self, "split_prefill_batch", None))
            )
        ):
            decision = "decode_only"

        if decision is not None:
            self._increment_histogram(
                self.rag_frontend_decision_counts, decision
            )

    def _record_rag_ann_tick_grants(self):
        if getattr(self, "rag_ann_tick_grants_recorded", False):
            return
        self.rag_ann_tick_grants_recorded = True
        granted_batches = getattr(
            self, "rag_ann_batches_granted_this_tick", 0
        )
        if granted_batches <= 0:
            return

        self.rag_ann_grant_ticks = (
            getattr(self, "rag_ann_grant_ticks", 0) + 1
        )
        self.rag_ann_granted_batches = (
            getattr(self, "rag_ann_granted_batches", 0) + granted_batches
        )
        if granted_batches > 1:
            self.rag_ann_multi_grant_ticks = (
                getattr(self, "rag_ann_multi_grant_ticks", 0) + 1
            )
        histogram = getattr(
            self, "rag_ann_grants_per_tick_histogram", None
        )
        if histogram is None:
            histogram = {}
            self.rag_ann_grants_per_tick_histogram = histogram
        self._increment_histogram(histogram, granted_batches)

    def _rag_ann_frontend_partition_needed(self) -> bool:
        plan = getattr(self, "rag_frontend_plan", None)
        return bool(
            self._rag_ann_uses_frontend_pool()
            and plan is not None
            and plan.frontend_needed
        )

    def _rag_ann_blocks_prefill(self) -> bool:
        plan = getattr(self, "rag_frontend_plan", None)
        arbitration = getattr(self, "rag_ann_frontend_arbitration", "overlap")
        if plan is not None and arbitration == "fair":
            if not plan.allow_prefill:
                return True
            if (
                self._rag_ann_uses_decode_pool()
                and (
                    self._rag_ann_gpu_work_active()
                    or getattr(
                        self,
                        "rag_ann_partition_switch_deferred",
                        False,
                    )
                )
            ):
                current_idx = get_current_stream_idx()
                last_mixed_idx = getattr(self, "real_sm_group_num", 1) - 2
                if (
                    not 1 <= current_idx <= last_mixed_idx
                    and plan.target_stream_idx != current_idx
                ):
                    # An ANN launched in an ordinary all-SM Decode lane must
                    # finish before Prefill can enter a mixed GreenContext.
                    return True
            return False
        if arbitration == "fair" and not getattr(
            self, "rag_prefill_run_allowed", True
        ):
            return True
        return arbitration == "strict" and self._rag_ann_gpu_work_active()

    def _get_rag_ann_runtime_state(self) -> dict:
        history = list(getattr(self, "rag_ann_decode_iteration_ms", []))
        decode_gpu_history = list(
            getattr(self, "rag_ann_decode_gpu_ms", [])
        )
        iteration_overhead_history = [
            max(0.0,float(iteration_ms)-float(gpu_ms))
            for iteration_ms,gpu_ms,_ in getattr(
                self,"_rag_iteration_overhead_window",())
        ]
        batch_history = list(
            getattr(self, "rag_ann_decode_batch_sizes", [])
        )
        queue_history = list(
            getattr(self, "rag_ann_decode_queue_delay_ms", [])
        )
        inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]
        pending = getattr(self, "ann_search_pending", None) or []
        ann_queued_gpu_ms = self._ann_inflight_predicted_gpu_ms(inflight)
        ann_credit_limit_ms = self._ann_launch_credit_limit_ms()
        ann_credit_limit_value = (
            None if np.isinf(ann_credit_limit_ms) else ann_credit_limit_ms
        )
        ann_executor = getattr(self, "ann_executor", None)
        hot_route_batching = (
            ann_executor.hot_route_batching_stats()
            if ann_executor is not None
            and hasattr(ann_executor, "hot_route_batching_stats")
            else {}
        )
        cold_search_batching = (
            ann_executor.cold_search_batching_stats()
            if ann_executor is not None
            and hasattr(ann_executor, "cold_search_batching_stats")
            else {}
        )
        transition_intent = getattr(
            self, "rag_partition_transition_intent", None
        )
        transition_pending = None
        if transition_intent is not None:
            transition_pending = {
                "source_stream_idx": transition_intent["current_idx"],
                "target_stream_idx": transition_intent["candidate_idx"],
                "elapsed_ms": max(
                    0.0,
                    (
                        time.perf_counter()
                        - float(transition_intent["started_at"])
                    )
                    * 1000.0,
                ),
            }
        return {
            "pending_requests": len(pending),
            "inflight_batches": len(inflight),
            "inflight_batches_peak": getattr(self, "rag_ann_inflight_peak", 0),
            "ann_hot_route_batching": hot_route_batching,
            "ann_cold_search_batching": cold_search_batching,
            "hybrid_pipeline_backlog_holds": getattr(
                self,
                "rag_ann_hybrid_pipeline_backlog_holds",
                0,
            ),
            "hybrid_pipeline_second_slot_admissions": getattr(
                self,
                "rag_ann_hybrid_pipeline_second_slot_admissions",
                0,
            ),
            "hybrid_pipeline_window_expansions": getattr(
                self,
                "rag_ann_hybrid_pipeline_window_expansions",
                0,
            ),
            "hybrid_pipeline_window_histogram": dict(
                getattr(
                    self,
                    "rag_ann_hybrid_pipeline_window_histogram",
                    {},
                )
            ),
            "ann_credit": {
                "configured_max_queued_gpu_ms": getattr(
                    self, "rag_ann_max_queued_gpu_ms", 0.0
                ),
                "configured_max_microbatch_gpu_ms": getattr(
                    self, "rag_ann_max_microbatch_gpu_ms", 0.0
                ),
                "service_time_percentile": getattr(
                    self, "rag_ann_service_time_percentile", 95.0
                ),
                "service_time_min_samples": getattr(
                    self, "rag_ann_service_time_min_samples", 4
                ),
                "effective_microbatch_rows": (
                    self._effective_ann_microbatch_rows()
                ),
                "current_limit_ms": ann_credit_limit_value,
                "current_queued_gpu_ms": ann_queued_gpu_ms,
                "current_available_ms": (
                    None
                    if ann_credit_limit_value is None
                    else max(0.0, ann_credit_limit_value - ann_queued_gpu_ms)
                ),
                "peak_queued_gpu_ms": getattr(
                    self, "rag_ann_credit_peak_queued_ms", 0.0
                ),
                "peak_admitted_queued_gpu_ms": getattr(
                    self, "rag_ann_credit_peak_admitted_queued_ms", 0.0
                ),
                "peak_forced_atomic_gpu_ms": getattr(
                    self, "rag_ann_credit_peak_forced_atomic_ms", 0.0
                ),
                "holds": getattr(self, "rag_ann_credit_holds", 0),
                "downsized_launches": getattr(
                    self, "rag_ann_credit_downsized_launches", 0
                ),
                "forced_single_launches": getattr(
                    self, "rag_ann_credit_forced_single_launches", 0
                ),
                "forced_atomic_launches": getattr(
                    self, "rag_ann_credit_forced_single_launches", 0
                ),
                "last_predicted_batch_ms": getattr(
                    self, "rag_ann_credit_last_batch_ms", 0.0
                ),
            },
            "ann_pre_switch_launch": {
                "configured": bool(
                    getattr(self, "rag_ann_pre_switch_launch", False)
                ),
                "enabled": bool(
                    getattr(self, "rag_ann_pre_switch_launch", False)
                    and getattr(self, "rag_policy_mode", "full") == "full"
                    and getattr(
                        self,
                        "rag_ann_frontend_arbitration",
                        "overlap",
                    )
                    == "fair"
                    and getattr(self, "rag_ann_launch_site", "early")
                    == "adaptive"
                    and self._rag_ann_work_conserving_decode_overlap_enabled()
                    and getattr(
                        getattr(
                            getattr(
                                self,
                                "shared_hbm_controller",
                                None,
                            ),
                            "pool",
                            None,
                        ),
                        "mode",
                        "scratch",
                    )
                    != "ivf"
                ),
                "eligible_hbm_mode": getattr(
                    getattr(
                        getattr(
                            self,
                            "shared_hbm_controller",
                            None,
                        ),
                        "pool",
                        None,
                    ),
                    "mode",
                    "scratch",
                )
                != "ivf",
                "checks": getattr(self, "rag_ann_pre_switch_checks", 0),
                "deadline_risk_triggers": getattr(
                    self,
                    "rag_ann_pre_switch_deadline_risk_triggers",
                    0,
                ),
                "intrinsic_debt_recovery_triggers": getattr(
                    self,
                    "rag_ann_pre_switch_intrinsic_debt_recovery_triggers",
                    0,
                ),
                "switch_induced_triggers": getattr(
                    self,
                    "rag_ann_pre_switch_switch_induced_triggers",
                    0,
                ),
                "counterfactual_no_gain_skips": getattr(
                    self,
                    "rag_ann_pre_switch_counterfactual_no_gain_skips",
                    0,
                ),
                "gpu_p99_credit_holds": getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_holds",
                    0,
                ),
                "gpu_p99_credit_downsize_enabled": (
                    self._rag_ann_pre_switch_p99_downsize_enabled()
                ),
                "adaptive_p99_credit_enabled": (
                    self._rag_ann_pre_switch_adaptive_p99_credit_enabled()
                ),
                "quantile_aligned_switch_cost_enabled": (
                    self._rag_ann_pre_switch_quantile_aligned_switch_cost_enabled()
                ),
                "batch_net_gain_enabled": (
                    self._rag_ann_pre_switch_batch_net_gain_enabled()
                ),
                "cross_class_debt_enabled": (
                    self._rag_ann_pre_switch_cross_class_debt_enabled()
                ),
                "downsized_min_robust_gain_ms": (
                    self._rag_ann_pre_switch_downsized_min_robust_gain_ms()
                ),
                "downsized_min_robust_gain_enabled": (
                    self._rag_ann_pre_switch_downsized_min_robust_gain_ms()
                    > 0.0
                ),
                "adaptive_p99_credit_derivations": getattr(
                    self,
                    "rag_ann_pre_switch_adaptive_p99_credit_derivations",
                    0,
                ),
                "adaptive_p99_credit_unavailable": getattr(
                    self,
                    "rag_ann_pre_switch_adaptive_p99_credit_unavailable",
                    0,
                ),
                # Compatibility: these two counters remain selector-level.
                "gpu_p99_credit_downsizes": getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsizes",
                    0,
                ),
                "gpu_p99_credit_rows_reduced": getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_rows_reduced",
                    0,
                ),
                "gpu_p99_credit_downsized_triggers": getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsized_triggers",
                    0,
                ),
                "gpu_p99_credit_downsized_launches": getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsized_launches",
                    0,
                ),
                "robust_gain_threshold_holds": getattr(
                    self,
                    "rag_ann_pre_switch_robust_gain_threshold_holds",
                    0,
                ),
                "robust_gain_threshold_bypasses": getattr(
                    self,
                    "rag_ann_pre_switch_robust_gain_threshold_bypasses",
                    0,
                ),
                "batch_net_gain_unavailable": getattr(
                    self,
                    "rag_ann_pre_switch_batch_net_gain_unavailable",
                    0,
                ),
                "cross_class_debt_candidates": getattr(
                    self,
                    "rag_ann_pre_switch_cross_class_debt_candidates",
                    0,
                ),
                "cross_class_debt_bypasses": getattr(
                    self,
                    "rag_ann_pre_switch_cross_class_debt_bypasses",
                    0,
                ),
                "cross_class_debt_would_admit": getattr(
                    self,
                    "rag_ann_pre_switch_cross_class_debt_would_admit",
                    0,
                ),
                "cross_class_debt_holds": getattr(
                    self,
                    "rag_ann_pre_switch_cross_class_debt_holds",
                    0,
                ),
                "eligible_robust_gain": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_eligible_robust_gain_ms",
                        (),
                    )
                ),
                "eligible_raw_robust_gain": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_eligible_raw_robust_gain_ms",
                        (),
                    )
                ),
                "eligible_batch_net_gain": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_eligible_batch_net_gain_ms",
                        (),
                    )
                ),
                "eligible_cross_class_net_debt": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_eligible_cross_class_net_debt_ms",
                        (),
                    )
                ),
                "last_gpu_p99_credit": dict(
                    getattr(
                        self,
                        "rag_ann_pre_switch_last_gpu_p99_credit",
                        {},
                    )
                ),
                "classification_invariant_violations": getattr(
                    self,
                    "rag_ann_pre_switch_classification_invariant_violations",
                    0,
                ),
                "decision_quantile": 50,
                "trigger_type_counts": dict(
                    getattr(
                        self,
                        "rag_ann_pre_switch_trigger_type_counts",
                        {},
                    )
                ),
                "predicted_gain": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_predicted_gain_ms",
                        (),
                    )
                ),
                "last_counterfactual": dict(
                    getattr(
                        self,
                        "rag_ann_pre_switch_last_counterfactual",
                        {},
                    )
                ),
                "launched_batches": getattr(
                    self, "rag_ann_pre_switch_launch_batches", 0
                ),
                "launched_rows": getattr(
                    self, "rag_ann_pre_switch_launch_rows", 0
                ),
                "switch_deferrals": getattr(
                    self,
                    "rag_ann_pre_switch_switch_deferrals",
                    0,
                ),
                "completion_polls": getattr(
                    self,
                    "rag_ann_pre_switch_completion_polls",
                    0,
                ),
                "inline_completions": getattr(
                    self,
                    "rag_ann_pre_switch_inline_completions",
                    0,
                ),
                "skip_reasons": dict(
                    getattr(
                        self,
                        "rag_ann_pre_switch_no_launch_reason_counts",
                        {},
                    )
                ),
                "trigger_queue_wait": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_trigger_wait_ms",
                        (),
                    )
                ),
                "predicted_switch_latency": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_predicted_switch_ms",
                        (),
                    )
                ),
                "predicted_robust_switch_latency": self._latency_distribution(
                    getattr(
                        self,
                        "rag_ann_pre_switch_predicted_robust_switch_ms",
                        (),
                    )
                ),
                "peak_projected_gpu_ms": getattr(
                    self,
                    "rag_ann_pre_switch_peak_projected_gpu_ms",
                    0.0,
                ),
                "safety_violations": getattr(
                    self,
                    "rag_ann_pre_switch_safety_violations",
                    0,
                ),
                "completion_predictor": (
                    self._ann_completion_predictor_metrics()
                ),
            },
            "ann_tick_grants": {
                "configured_max_batches_per_tick": getattr(
                    self, "rag_ann_max_batches_per_tick", 0
                ),
                "grant_ticks": getattr(self, "rag_ann_grant_ticks", 0),
                "granted_batches": getattr(
                    self, "rag_ann_granted_batches", 0
                ),
                "mean_batches_per_grant_tick": (
                    getattr(self, "rag_ann_granted_batches", 0)
                    / max(1, getattr(self, "rag_ann_grant_ticks", 0))
                ),
                "multi_grant_ticks": getattr(
                    self, "rag_ann_multi_grant_ticks", 0
                ),
                "tick_limit_holds": getattr(
                    self, "rag_ann_tick_limit_holds", 0
                ),
                "batches_per_tick_histogram": getattr(
                    self, "rag_ann_grants_per_tick_histogram", {}
                ),
            },
            "ann_gpu_service_by_batch_size": {
                str(batch_size): self._latency_distribution(samples)
                for batch_size, samples in getattr(
                    self, "rag_ann_gpu_service_ms_by_batch_size", {}
                ).items()
            },
            "ann_gpu_service_by_partition_batch": {
                "%d:%d" % (key[0], key[1]): self._latency_distribution(samples)
                for key, samples in sorted(getattr(
                    self, "rag_ann_gpu_service_ms_by_partition_batch_size", {}
                ).items())
            },
            "decode_iteration_samples": len(history),
            "decode_batch_recent_p95": (
                float(_history_percentile_value(batch_history, 95))
                if batch_history
                else 0.0
            ),
            "request_arrival_rps": self._rag_request_arrival_rps(),
            "decode_iteration_p50_ms": (
                float(_history_percentile_value(history, 50)) if history else 0.0
            ),
            "decode_iteration_p95_ms": (
                float(_history_percentile_value(history, 95)) if history else 0.0
            ),
            "decode_iteration_p99_ms": (
                float(_history_percentile_value(history, 99)) if history else 0.0
            ),
            "decode_iteration_overhead_samples": len(iteration_overhead_history),
            "decode_iteration_overhead_p50_ms": (
                float(_history_percentile_value(iteration_overhead_history,50))
                if iteration_overhead_history else 0.0
            ),
            "decode_iteration_overhead_p99_ms": (
                float(_history_percentile_value(iteration_overhead_history,99))
                if iteration_overhead_history else 0.0
            ),
            "decode_gpu_step_samples": len(decode_gpu_history),
            "decode_gpu_step_p50_ms": (
                float(_history_percentile_value(decode_gpu_history, 50))
                if decode_gpu_history
                else 0.0
            ),
            "decode_gpu_step_p95_ms": (
                float(_history_percentile_value(decode_gpu_history, 95))
                if decode_gpu_history
                else 0.0
            ),
            "decode_gpu_step_p99_ms": (
                float(_history_percentile_value(decode_gpu_history, 99))
                if decode_gpu_history
                else 0.0
            ),
            "decode_queue_delay_p95_ms": (
                float(_history_percentile_value(queue_history, 95))
                if queue_history
                else 0.0
            ),
            "decode_queue_delay_p99_ms": (
                float(_history_percentile_value(queue_history, 99))
                if queue_history
                else 0.0
            ),
            "decode_predicted_gpu_p99_ms": (
                self._rag_ann_decode_prediction_ms() or 0.0
            ),
            "decode_predicted_p99_ms": (
                self._rag_ann_decode_prediction_ms(
                    include_scheduling=True
                )
                or 0.0
            ),
            "decode_iteration_by_partition": {
                str(stream_idx): self._latency_distribution(partition_history)
                for stream_idx, partition_history in getattr(
                    self, "rag_ann_decode_partition_ms", {}
                ).items()
            },
            "decode_iteration_by_partition_batch_bucket": {
                str(stream_idx): {
                    str(bucket): self._latency_distribution(bucket_history)
                    for bucket, bucket_history in bucket_histories.items()
                }
                for stream_idx, bucket_histories in getattr(
                    self, "rag_ann_decode_partition_batch_ms", {}
                ).items()
            },
            "decode_iteration_by_partition_batch_bucket_overlap": {
                str(stream_idx): {
                    str(bucket): {
                        overlap_class: self._latency_distribution(
                            overlap_history
                        )
                        for overlap_class, overlap_history in (
                            overlap_histories.items()
                        )
                    }
                    for bucket, overlap_histories in bucket_histories.items()
                }
                for stream_idx, bucket_histories in getattr(
                    self,
                    "rag_ann_decode_partition_batch_overlap_ms",
                    {},
                ).items()
            },
            "decode_gpu_step_by_partition": {
                str(stream_idx): self._latency_distribution(partition_history)
                for stream_idx, partition_history in getattr(
                    self, "rag_ann_decode_gpu_partition_ms", {}
                ).items()
            },
            "decode_gpu_step_by_partition_batch_bucket": {
                str(stream_idx): {
                    str(bucket): self._latency_distribution(bucket_history)
                    for bucket, bucket_history in bucket_histories.items()
                }
                for stream_idx, bucket_histories in getattr(
                    self, "rag_ann_decode_gpu_partition_batch_ms", {}
                ).items()
            },
            "decode_gpu_step_by_partition_batch_bucket_overlap": {
                str(stream_idx): {
                    str(bucket): {
                        overlap_class: self._latency_distribution(
                            overlap_history
                        )
                        for overlap_class, overlap_history in (
                            overlap_histories.items()
                        )
                    }
                    for bucket, overlap_histories in bucket_histories.items()
                }
                for stream_idx, bucket_histories in getattr(
                    self,
                    "rag_ann_decode_gpu_partition_batch_overlap_ms",
                    {},
                ).items()
            },
            "decode_step_slo_ms": self._rag_ann_decode_slo_ms(),
            "partition_decode_slo_ms": list(
                getattr(self, "rag_partition_decode_slo_ms", ())
            ),
            "partition_decode_slo_batch_thresholds": list(
                getattr(
                    self, "rag_partition_decode_slo_batch_thresholds", ()
                )
            ),
            "partition_decode_slo_band_idx": getattr(
                self, "rag_partition_decode_slo_band_idx", 0
            ),
            "partition_decode_slo_batch_ewma": getattr(
                self, "rag_partition_decode_slo_batch_ewma", None
            ),
            "partition_decode_slo_hysteresis_bs": getattr(
                self, "rag_partition_decode_slo_hysteresis_bs", 0.0
            ),
            "partition_decode_slo_min_residency_ms": getattr(
                self, "rag_partition_decode_slo_min_residency_ms", 0.0
            ),
            "partition_decode_slo_initial_calibration_ms": getattr(
                self,
                "rag_partition_decode_slo_initial_calibration_ms",
                0.0,
            ),
            "partition_decode_slo_next_update_in_ms": max(
                0.0,
                (
                    getattr(
                        self, "rag_partition_decode_slo_next_update_ts", 0.0
                    )
                    - time.perf_counter()
                )
                * 1000.0,
            ),
            "partition_decode_slo_ewma_alpha": getattr(
                self, "rag_partition_decode_slo_ewma_alpha", 0.02
            ),
            "partition_decode_slo_band_changes": getattr(
                self, "rag_partition_decode_slo_band_changes", 0
            ),
            "partition_decode_slo_control_updates": getattr(
                self, "rag_partition_decode_slo_control_updates", 0
            ),
            "partition_slo_floor_enabled": getattr(
                self, "rag_partition_slo_floor_enabled", True
            ),
            "slo_deferred_requests": getattr(
                self, "rag_ann_slo_deferred_requests", 0
            ),
            "ann_stream_group_launch_counts": dict(
                getattr(self, "rag_ann_stream_group_launch_counts", {})
            ),
            "prefill_stream_group_run_counts": dict(
                getattr(self, "rag_prefill_stream_group_run_counts", {})
            ),
            "prefill_dynamic_lease": getattr(
                self, "rag_prefill_dynamic_lease", False
            ),
            "prefill_decode_guard_percentile": float(
                getattr(self, "rag_prefill_decode_guard_percentile", 99.0)
            ),
            "prefill_two_level_bundles_enabled": getattr(
                self, "rag_prefill_two_level_bundles", False
            ),
            "prefill_two_level_mixed_budget_bypasses": getattr(
                self, "rag_prefill_two_level_mixed_budget_bypasses", 0
            ),
            "prefill_two_level_safety_deferrals": getattr(
                self, "rag_prefill_two_level_safety_deferrals", 0
            ),
            "prefill_short_bundle_layers": getattr(
                self, "rag_prefill_short_bundle_layers", 16
            ),
            "prefill_short_token_threshold": getattr(
                self, "rag_prefill_short_token_threshold", 2048
            ),
            "prefill_short_bundle_activations": getattr(
                self, "rag_prefill_short_bundle_activations", 0
            ),
            "prefill_risk_bundle_activations": getattr(
                self, "rag_prefill_risk_bundle_activations", 0
            ),
            "prefill_severe_risk_bundle_activations": getattr(
                self, "rag_prefill_severe_risk_bundle_activations", 0
            ),
            "prefill_long_request_bundle_activations": getattr(
                self, "rag_prefill_long_request_bundle_activations", 0
            ),
            "prefill_aggregate_false_long_avoided": getattr(
                self, "rag_prefill_aggregate_false_long_avoided", 0
            ),
            "prefill_request_length_fallbacks": getattr(
                self, "rag_prefill_request_length_fallbacks", 0
            ),
            "prefill_mixed_service_guard_uses": getattr(
                self, "rag_prefill_mixed_service_guard_uses", 0
            ),
            "prefill_exclusive_iteration_guard_uses": getattr(
                self, "rag_prefill_exclusive_iteration_guard_uses", 0
            ),
            "prefill_iteration_pressure_half_caps": getattr(
                self, "rag_prefill_iteration_pressure_half_caps", 0
            ),
            "prefill_iteration_pressure_full_bundle_bypasses": getattr(
                self,
                "rag_prefill_iteration_pressure_full_bundle_bypasses",
                0,
            ),
            "prefill_bundle_inflight": (
                getattr(self, "rag_prefill_bundle_inflight_event", None)
                is not None
            ),
            "prefill_bundle_completions": getattr(
                self, "rag_prefill_bundle_completions", 0
            ),
            "prefill_bundle_switch_holds": getattr(
                self, "rag_prefill_bundle_switch_holds", 0
            ),
            "prefill_decode_only_deferrals": getattr(
                self, "rag_prefill_decode_only_deferrals", 0
            ),
            "prefill_decode_only_mixed_fallbacks": getattr(
                self, "rag_prefill_decode_only_mixed_fallbacks", 0
            ),
            "prefill_bundle_layer_histogram": dict(
                getattr(self, "rag_prefill_bundle_layer_histogram", {})
            ),
            "prefill_candidate_net_gain": self._latency_distribution(
                getattr(self, "rag_prefill_candidate_net_gain_ms", [])
            ),
            "prefill_critical_path_net_gain": self._latency_distribution(
                getattr(
                    self, "rag_prefill_critical_path_net_gain_ms", []
                )
            ),
            "prefill_robust_critical_path_net_gain": (
                self._latency_distribution(
                    getattr(
                        self,
                        "rag_prefill_robust_critical_path_net_gain_ms",
                        [],
                    )
                )
            ),
            "prefill_candidate_net_gain_admissions": getattr(
                self, "rag_prefill_candidate_net_gain_admissions", 0
            ),
            "prefill_candidate_net_gain_holds": getattr(
                self, "rag_prefill_candidate_net_gain_holds", 0
            ),
            "prefill_e2e_critical_admissions": getattr(
                self, "rag_prefill_e2e_critical_admissions", 0
            ),
            "prefill_e2e_uncertainty_holds": getattr(
                self, "rag_prefill_e2e_uncertainty_holds", 0
            ),
            "prefill_tail_gain_admissions": getattr(
                self, "rag_prefill_tail_gain_admissions", 0
            ),
            "prefill_tail_liveness_admissions": getattr(
                self, "rag_prefill_tail_liveness_admissions", 0
            ),
            "prefill_tail_liveness_layer_caps": getattr(
                self, "rag_prefill_tail_liveness_layer_caps", 0
            ),
            "prefill_tail_liveness_budget": self._latency_distribution(
                getattr(self, "rag_prefill_tail_liveness_budget_ms", [])
            ),
            "prefill_tail_gain_holds": getattr(
                self, "rag_prefill_tail_gain_holds", 0
            ),
            "prefill_candidate_net_gain_last": dict(
                getattr(self, "rag_prefill_candidate_net_gain_last", {})
            ),
            "prefill_floor_safe_admissions": getattr(
                self, "rag_prefill_floor_safe_admissions", 0
            ),
            "prefill_layer_latency_by_partition_bucket": {
                f"{stream_idx}:{token_bucket}": self._latency_distribution(
                    samples
                )
                for (stream_idx, token_bucket), samples in getattr(
                    self,
                    "rag_prefill_layer_ms_by_partition_bucket",
                    {},
                ).items()
            },
            "prefill_prediction_pairs": list(
                getattr(self, "rag_prefill_prediction_pairs", [])
            ),
            "prefill_prediction_abs_error": self._latency_distribution(
                getattr(self, "rag_prefill_prediction_abs_error_ms", [])
            ),
            "prefill_prediction_samples": getattr(
                self, "rag_prefill_prediction_samples", 0
            ),
            "decode_host_submit": self._latency_distribution(
                getattr(self, "rag_decode_host_submit_ms", [])
            ),
            "decode_run_batch_observations": [
                dict(row) for row in getattr(
                    self, "rag_decode_run_batch_observations", []
                )
            ],
            "decode_launch_interval": self._latency_distribution(
                getattr(self, "rag_decode_launch_interval_ms", [])
            ),
            "prefill_host_submit": self._latency_distribution(
                getattr(self, "rag_prefill_host_submit_ms", [])
            ),
            "scheduler_tick_wall": self._latency_distribution(
                getattr(self, "rag_scheduler_tick_wall_ms", [])
            ),
            "frontend_decision_latency": self._latency_distribution(
                getattr(self, "rag_frontend_decision_ms", [])
            ),
            "prefill_submit_first_count": getattr(self, "rag_prefill_submit_first_count", 0),
            "frontend_idle_fast_path_hits": getattr(
                self, "rag_frontend_idle_fast_path_hits", 0
            ),
            "frontend_ann_decode_fast_path_hits": getattr(
                self, "rag_frontend_ann_decode_fast_path_hits", 0
            ),
            "decode_prediction_pairs": list(
                getattr(self, "rag_decode_prediction_pairs", [])
            ),
            "decode_prediction_abs_error": self._latency_distribution(
                getattr(self, "rag_decode_prediction_abs_error_ms", [])
            ),
            "decode_prediction_samples": getattr(
                self, "rag_decode_prediction_samples", 0
            ),
            "decode_prediction_false_safe": getattr(
                self, "rag_decode_prediction_false_safe", 0
            ),
            "decode_prediction_false_unsafe": getattr(
                self, "rag_decode_prediction_false_unsafe", 0
            ),
            "decode_prediction_partition_mismatches": getattr(
                self, "rag_decode_prediction_partition_mismatches", 0
            ),
            "decode_iteration_prediction_pairs": list(
                getattr(self, "rag_decode_iteration_prediction_pairs", [])
            ),
            "decode_iteration_prediction_abs_error": (
                self._latency_distribution(
                    getattr(
                        self,
                        "rag_decode_iteration_prediction_abs_error_ms",
                        [],
                    )
                )
            ),
            "decode_iteration_prediction_samples": getattr(
                self, "rag_decode_iteration_prediction_samples", 0
            ),
            "decode_iteration_prediction_false_safe": getattr(
                self, "rag_decode_iteration_prediction_false_safe", 0
            ),
            "decode_iteration_prediction_false_unsafe": getattr(
                self, "rag_decode_iteration_prediction_false_unsafe", 0
            ),
            "decode_prediction_regime_shift_ratio": getattr(
                self, "rag_decode_prediction_regime_shift_ratio", 4.0
            ),
            "decode_prediction_cold_start_margin": getattr(
                self, "rag_decode_prediction_cold_start_margin", 1.25
            ),
            "prefill_sprint_chunks": getattr(
                self, "rag_prefill_sprint_chunks", 0
            ),
            "prefill_exclusive_sprint_enabled": getattr(
                self, "rag_prefill_exclusive_sprint", False
            ),
            "prefill_exclusive_sprint_available": getattr(
                self, "rag_prefill_exclusive_sprint_available", False
            ),
            "prefill_exclusive_sprint_chunks": getattr(
                self, "rag_prefill_exclusive_sprint_chunks", 0
            ),
            "prefill_exclusive_sprint_block_counts": dict(
                getattr(
                    self, "rag_prefill_exclusive_sprint_block_counts", {}
                )
            ),
            "prefill_exclusive_decode_update_skips": getattr(
                self, "rag_prefill_exclusive_decode_update_skips", 0
            ),
            "prefill_exclusive_deferred_merges": getattr(
                self, "rag_prefill_exclusive_deferred_merges", 0
            ),
            "dependency_prefill_admissions": getattr(
                self, "rag_dependency_prefill_admissions", 0
            ),
            "ann_prefill_transition_holds": getattr(
                self, "rag_ann_prefill_transition_holds", 0
            ),
            "prefill_workload_fast_path_rejects": getattr(
                self, "rag_prefill_workload_fast_path_rejects", 0
            ),
            "prefill_cohort_holds": getattr(
                self, "rag_prefill_cohort_holds", 0
            ),
            "prefill_cohort_releases": getattr(
                self, "rag_prefill_cohort_releases", 0
            ),
            "prefill_cohort_size_histogram": dict(
                getattr(self, "rag_prefill_cohort_size_histogram", {})
            ),
            "prefill_low_load_reference_tokens": getattr(
                self, "rag_prefill_low_load_reference_tokens", 1024
            ),
            "pdmux_stream_group_switches": getattr(
                self, "pdmux_stream_group_switches", 0
            ),
            "partition_switch_deferred": getattr(
                self, "rag_ann_partition_switch_deferred", False
            ),
            "partition_drain_holds": getattr(
                self, "rag_ann_partition_drain_holds", 0
            ),
            "partition_drain_events": getattr(
                self, "rag_ann_partition_drain_events", 0
            ),
            "partition_completion_polls": getattr(
                self, "rag_ann_partition_completion_polls", 0
            ),
            "partition_inline_completions": getattr(
                self, "rag_ann_partition_inline_completions", 0
            ),
            "decode_completion_polls": getattr(
                self, "rag_ann_decode_completion_polls", 0
            ),
            "decode_inline_completions": getattr(
                self, "rag_ann_decode_inline_completions", 0
            ),
            "decode_tick_completion_polls": getattr(
                self, "rag_ann_decode_tick_completion_polls", 0
            ),
            "decode_tick_inline_completions": getattr(
                self, "rag_ann_decode_tick_inline_completions", 0
            ),
            "decode_credit_fences": getattr(
                self, "rag_ann_decode_credit_fences", 0
            ),
            "decode_credit_fence_reasons": dict(
                getattr(
                    self, "rag_ann_decode_credit_fence_reason_counts", {}
                )
            ),
            "decode_credit_fence_age": self._latency_distribution(
                getattr(self, "rag_ann_decode_credit_fence_age_ms", [])
            ),
            "decode_credit_fence_prediction": self._latency_distribution(
                getattr(
                    self, "rag_ann_decode_credit_fence_prediction_ms", []
                )
            ),
            "decode_credit_fence_remaining": self._latency_distribution(
                getattr(
                    self, "rag_ann_decode_credit_fence_remaining_ms", []
                )
            ),
            "decode_credit_sync_drains": getattr(
                self, "rag_ann_decode_credit_sync_drains", 0
            ),
            "decode_credit_inline_completions": getattr(
                self, "rag_ann_decode_credit_inline_completions", 0
            ),
            "decode_credit_sync_wait": self._latency_distribution(
                getattr(self, "rag_ann_decode_credit_sync_wait_ms", [])
            ),
            "partition_drain_predicted": self._latency_distribution(
                getattr(self, "rag_ann_partition_drain_predicted_ms", [])
            ),
            "decode_batch_size_histogram": dict(
                getattr(self, "rag_decode_batch_size_histogram", {})
            ),
            "ann_batch_size_histogram": dict(
                getattr(self, "rag_ann_batch_size_histogram", {})
            ),
            "ann_inflight_coalescing": {
                "events": getattr(
                    self, "rag_ann_inflight_coalesce_events", 0
                ),
                "hold_checks": getattr(
                    self, "rag_ann_inflight_coalesce_holds", 0
                ),
            },
            "ann_priming_requests": int(
                getattr(self, "rag_ann_priming_requests", 0)
            ),
            "ann_priming_rows": int(
                getattr(self, "rag_ann_priming_rows", 0)
            ),
            "frontend_decision_counts": dict(
                getattr(self, "rag_frontend_decision_counts", {})
            ),
            "ann_queue_wait_histogram": dict(
                getattr(self, "rag_ann_queue_wait_histogram", {})
            ),
            "ann_queue_wait": self._latency_distribution(
                getattr(self, "rag_ann_queue_wait_ms", [])
            ),
            "ann_gpu_latency": self._latency_distribution(
                getattr(self, "rag_ann_gpu_latency_ms", [])
            ),
            "ann_cpu_queue_wait": self._latency_distribution(
                getattr(self, "rag_ann_cpu_queue_wait_ms", [])
            ),
            "ann_cpu_search_latency": self._latency_distribution(
                getattr(self, "rag_ann_cpu_search_latency_ms", [])
            ),
            "ann_coarse_quantize_latency": self._latency_distribution(
                getattr(self, "rag_ann_coarse_quantize_latency_ms", [])
            ),
            "ann_route_latency": self._latency_distribution(
                getattr(self, "rag_ann_route_latency_ms", [])
            ),
            "prefill_handoff_completion_observations": {
                "enabled": bool(getattr(getattr(self, 'server_args', None),
                                        'rag_joint_handoff_profiling', False)),
                "clock": "host_submit_to_observed_cuda_completion_including_poll_delay",
                "pending": len(getattr(self, 'rag_prefill_stage_completion_pending', ())),
                "dropped": getattr(self, 'rag_prefill_stage_completion_dropped', 0),
                "event_ledger_schema": "handoff-host-observation-v1",
                "event_sequence_scope": "single_scheduler_process",
                "events": list(getattr(self, 'rag_prefill_stage_observation_ledger', ())),
                "samples": [dict(partition=key[0], stage=key[1], prompt_tokens=list(key[2]),
                    running_batch_size=key[3], elapsed_ms=list(values))
                    for key, values in getattr(self, 'rag_prefill_stage_release_ms_by_shape', {}).items()],
            },
            "input_preparation_turnaround_observations": {
                "clock": "scheduler_ann_complete_to_tokenized_ingress",
                "scope": "current_run_query_rows_not_exact_prompt_shape",
                "samples": [dict(query_rows=rows,elapsed_ms=list(values))
                    for rows,values in getattr(self,'rag_input_preparation_turnaround_ms',{}).items()],
            },
            "ann_merge_latency": self._latency_distribution(
                getattr(self, "rag_ann_merge_latency_ms", [])
            ),
            "ann_hot_probe_ratio": self._value_distribution(
                getattr(self, "rag_ann_hot_probe_ratio", [])
            ),
            "policy_mode": getattr(self, "rag_policy_mode", "full"),
            "e2_fixed_partition_idx": getattr(
                self, "rag_e2_fixed_partition_idx", -1
            ),
            "joint_runtime": self._rag_joint_runtime_status(),
            "joint_service_observations": self._rag_joint_service_observations(),
            "lease_performance_profile": dict(
                getattr(self, "rag_lease_profile_status", {"loaded": False})
            ),
            "ann_work_conserving_decode_overlap": getattr(
                self, "rag_ann_work_conserving_decode_overlap", False
            ),
            "ann_launch_site": getattr(self, "rag_ann_launch_site", "early"),
            "ann_fast_path_pressure_threshold": getattr(
                self, "rag_ann_fast_path_pressure_threshold", 0.5
            ),
            "ann_fast_path_max_work": getattr(
                self, "rag_ann_fast_path_max_work", 1
            ),
            "ann_fast_path_decode_batch_p95_threshold": getattr(
                self,
                "rag_ann_fast_path_decode_batch_p95_threshold",
                16.0,
            ),
            "prefill_low_load_batch_p95_threshold": getattr(
                self,
                "rag_prefill_low_load_batch_p95_threshold",
                4.0,
            ),
            "prefill_low_load_arrival_rps_threshold": getattr(
                self,
                "rag_prefill_low_load_arrival_rps_threshold",
                -1.0,
            ),
            "ann_fast_path_decode_queue_p95_ms": getattr(
                self, "rag_ann_fast_path_decode_queue_p95_ms", 2.0
            ),
            "ann_decode_batch_bucket_size": getattr(
                self, "rag_ann_decode_batch_bucket_size", 4
            ),
            "ann_effective_early": self._rag_ann_should_launch_early(),
            "ann_effective_after_prefill": self._rag_ann_should_launch_after_prefill(),
            "dynamic_rr_next": getattr(self, "rag_dynamic_rr_next", "ann"),
            "frontend_arbitration": getattr(
                self, "rag_ann_frontend_arbitration", "overlap"
            ),
            "frontend_last_plan": getattr(
                self, "rag_frontend_last_plan", "uninitialized"
            ),
            "frontend_plan_reason_counts": dict(
                getattr(self, "rag_frontend_plan_reason_counts", {})
            ),
            "frontend_target_reason_counts": dict(
                getattr(self, "rag_frontend_target_reason_counts", {})
            ),
            "ann_launch_reason_counts": dict(
                getattr(self, "rag_ann_launch_reason_counts", {})
            ),
            "budget_reject_events": getattr(
                self, "rag_ann_budget_reject_events", 0
            ),
            "budget_rejected_requests": getattr(
                self, "rag_ann_budget_rejected_requests", 0
            ),
            "urgency_start_ratio": getattr(
                self, "rag_ann_urgency_start_ratio", 1.0
            ),
            "max_decode_slowdown_ratio": getattr(
                self, "rag_ann_max_decode_slowdown_ratio", 1.0
            ),
            "last_decode_budget": getattr(
                self, "rag_ann_last_budget", None
            ),
            "urgency_admitted_requests": getattr(
                self, "rag_ann_urgency_admitted_requests", 0
            ),
            "urgency_admission_wait": self._latency_distribution(
                getattr(self, "rag_ann_urgency_admission_wait_ms", [])
            ),
            "max_defer_forced_releases": getattr(
                self, "rag_ann_forced_release_events", 0
            ),
            "budget_defer_latency": self._latency_distribution(
                getattr(self, "rag_ann_budget_defer_ms", [])
            ),
            "forced_release_to_launch": self._latency_distribution(
                getattr(self, "rag_ann_forced_release_to_launch_ms", [])
            ),
            "partition_hysteresis_holds": getattr(
                self, "rag_partition_hysteresis_holds", 0
            ),
            "partition_residency_holds": getattr(
                self, "rag_partition_residency_holds", 0
            ),
            "partition_safe_mixed_holds": getattr(
                self, "rag_partition_safe_mixed_holds", 0
            ),
            "partition_safe_mixed_cost_admissions": getattr(
                self, "rag_partition_safe_mixed_cost_admissions", 0
            ),
            "partition_safe_mixed_cost_rejections": getattr(
                self, "rag_partition_safe_mixed_cost_rejections", 0
            ),
            "partition_safe_mixed_cost_prediction_misses": getattr(
                self,
                "rag_partition_safe_mixed_cost_prediction_misses",
                0,
            ),
            "partition_safe_mixed_last_cost_decision": getattr(
                self, "rag_partition_safe_mixed_last_cost_decision", {}
            ),
            "prefill_fast_path_resident_cost_checks": getattr(
                self, "rag_prefill_fast_path_resident_cost_checks", 0
            ),
            "partition_safe_mixed_hold_enabled": getattr(
                self, "rag_partition_safe_mixed_hold", False
            ),
            "partition_stable_dynamic_hold_enabled": getattr(
                self, "rag_prefill_dynamic_lease", False
            ),
            "partition_safe_mixed_hold_threshold_ms": (
                self._safe_mixed_hold_threshold_ms()
            ),
            "partition_safe_mixed_max_hold_ms": getattr(
                self, "rag_partition_safe_mixed_max_hold_ms", 5.0
            ),
            "partition_frontend_idle_ms": max(
                0.0,
                (
                    time.perf_counter()
                    - getattr(self, "rag_last_frontend_activity_ts", 0.0)
                )
                * 1000.0,
            ),
            "partition_decode_floor_sm": getattr(
                self, "rag_partition_decode_floor_sm", 0
            ),
            "partition_slo_promotions": getattr(
                self, "rag_partition_slo_promotions", 0
            ),
            "partition_slo_demotions": getattr(
                self, "rag_partition_slo_demotions", 0
            ),
            "partition_floor_safe_observations": getattr(
                self, "rag_partition_floor_safe_observations", 0
            ),
            "partition_decode_only_fallback_required": getattr(
                self, "rag_partition_decode_only_fallback_required", False
            ),
            "partition_decode_only_fallback_activations": getattr(
                self, "rag_partition_decode_only_fallback_activations", 0
            ),
            "decode_recovery_pending": getattr(
                self, "rag_decode_recovery_pending", False
            ),
            "decode_recovery_requests": getattr(
                self, "rag_decode_recovery_requests", 0
            ),
            "decode_recovery_windows": getattr(
                self, "rag_decode_recovery_windows", 0
            ),
            "partition_switch_latency": self._latency_distribution(
                getattr(self, "rag_partition_switch_latency_ms", [])
            ),
            "partition_switch_latency_by_pair": {
                f"{source_idx}->{target_idx}": self._latency_distribution(samples)
                for (source_idx, target_idx), samples in getattr(
                    self, "rag_partition_switch_pair_latency_ms", {}
                ).items()
            },
            "partition_transition_latency": self._latency_distribution(
                getattr(self, "rag_partition_transition_latency_ms", [])
            ),
            "partition_transition_latency_by_pair": {
                f"{source_idx}->{target_idx}": self._latency_distribution(
                    samples
                )
                for (source_idx, target_idx), samples in getattr(
                    self, "rag_partition_transition_pair_latency_ms", {}
                ).items()
            },
            "partition_transition_pending": transition_pending,
            "partition_transition_abandoned": getattr(
                self, "rag_partition_transition_abandoned", 0
            ),
            "lease_switch_benefit": self._latency_distribution(
                getattr(self, "rag_lease_switch_benefit_ms", [])
            ),
            "lease_coalescing": {
                "min_work": getattr(self, "rag_lease_coalesce_min_work", 1),
                "max_wait_ms": getattr(
                    self, "rag_lease_coalesce_max_wait_ms", 0.0
                ),
                "switch_cost_ms": getattr(
                    self, "rag_lease_switch_cost_ms", 0.0
                ),
                "switch_cost_percentile": getattr(
                    self, "rag_lease_switch_cost_percentile", 95.0
                ),
                "round_trip_cost": getattr(
                    self, "rag_lease_round_trip_switch_cost", False
                ),
                "minimum_gain_ratio": getattr(
                    self, "rag_lease_min_gain_ratio", 1.0
                ),
                "effective_switch_cost_ms": self._rag_lease_switch_cost_estimate_ms(),
                "last_transition_cost_ms": getattr(
                    self, "rag_lease_last_transition_cost_ms", None
                ),
                "last_predicted_benefit_ms": getattr(
                    self, "rag_lease_last_switch_benefit_ms", None
                ),
                "last_horizon_bundles": getattr(
                    self, "rag_lease_last_horizon_bundles", 1
                ),
                "horizon_bundle_histogram": dict(
                    getattr(self, "rag_lease_horizon_bundle_histogram", {})
                ),
                "horizon_layer_histogram": dict(
                    getattr(self, "rag_lease_horizon_layer_histogram", {})
                ),
                "prefill_partition_selection_counts": dict(
                    getattr(self, "rag_prefill_partition_selection_counts", {})
                ),
                "prefill_partition_selection_last": dict(
                    getattr(self, "rag_prefill_partition_selection_last", {})
                ),
                "prefill_min_layers": getattr(
                    self, "rag_prefill_lease_min_layers", 1
                ),
                "prefill_max_layers": getattr(
                    self, "rag_prefill_lease_max_layers", 8
                ),
                "prefill_decode_ratio": getattr(
                    self, "rag_prefill_lease_decode_ratio", 1.0
                ),
                "admission_waits": getattr(self, "rag_lease_coalesce_waits", 0),
                "work_holds": getattr(
                    self, "rag_lease_coalesce_work_holds", 0
                ),
                "cost_holds": getattr(
                    self, "rag_lease_coalesce_cost_holds", 0
                ),
                "decode_restore_transitions": getattr(
                    self, "rag_lease_decode_restore_transitions", 0
                ),
                "dependency_holds": getattr(
                    self, "rag_dependency_lease_holds", 0
                ),
            },
            "partition_switch_reason_counts": dict(
                getattr(self, "rag_partition_switch_reason_counts", {})
            ),
            "ann_consecutive_batches": getattr(
                self, "rag_ann_consecutive_batches", 0
            ),
            "prefill_consecutive_chunks": getattr(
                self, "rag_prefill_consecutive_chunks", 0
            ),
            "handoff_pending": len(getattr(self, "rag_ann_handoffs", {})),
            "handoff_completed": getattr(
                self, "rag_ann_handoff_completed", 0
            ),
            "handoff_expired": getattr(self, "rag_ann_handoff_expired", 0),
            "handoff_latency": self._latency_distribution(
                getattr(self, "rag_ann_handoff_latency_ms", [])
            ),
            "post_ann_handoff_wait": self._latency_distribution(
                getattr(self, "rag_ann_handoff_latency_ms", [])
            ),
            "post_ann_prefill_queue": self._latency_distribution(
                getattr(self, "rag_prefill_queue_wait_ms", [])
            ),
            "post_ann_prefill_gpu": self._latency_distribution(
                getattr(self, "rag_prefill_gpu_latency_ms", [])
            ),
            "post_ann_prefill_wall": self._latency_distribution(
                getattr(self, "rag_prefill_wall_latency_ms", [])
            ),
            "post_ann_first_decode_wait": self._latency_distribution(
                getattr(self, "rag_prefill_to_first_decode_ms", [])
            ),
            "handoff_decode_release_pending": getattr(
                self, "rag_handoff_decode_release_pending", False
            ),
            "handoff_decode_release_windows": getattr(
                self, "rag_handoff_decode_release_windows", 0
            ),
            "handoff_prefill_activations": getattr(
                self, "rag_ann_handoff_prefill_activations", 0
            ),
            "frontend_residency_releases": getattr(
                self, "rag_ann_frontend_residency_releases", 0
            ),
            "controlled_overlap_enabled": getattr(
                self, "rag_ann_enable_controlled_overlap", False
            ),
            "shared_hbm": (
                self.shared_hbm_controller.metrics()
                if getattr(self, "shared_hbm_controller", None) is not None
                else None
            ),
        }

    def _rag_ann_blocks_stream_group_switch(self, target_stream_idx: int) -> bool:
        """Return whether active ANN work is incompatible with the target group.

        frontend-gc may safely release to the ordinary Decode-only group.
        decode-gc keeps Decode in ANN's current resource partition until the
        ANN kernels finish; moving either lane to another GC would violate the
        intended ANN+Decode resource ownership.
        """
        was_blocked = getattr(self, "rag_ann_partition_switch_deferred", False)
        if not self._rag_ann_uses_partitioned_gc():
            self.rag_ann_partition_switch_deferred = False
            return False

        if getattr(self, "rag_prefill_bundle_inflight_event", None) is not None:
            self.rag_prefill_bundle_switch_holds = (
                getattr(self, "rag_prefill_bundle_switch_holds", 0) + 1
            )
            self.rag_ann_partition_switch_deferred = True
            return True

        inflight = getattr(self, "ann_search_inflight", None)
        if not inflight:
            if was_blocked:
                logger.debug("Resuming PD stream-group switch after ANN completion")
            self.rag_ann_partition_switch_deferred = False
            return False
        if isinstance(inflight, dict):
            inflight = [inflight]

        active_stream_idxs = []
        for task in inflight:
            result = task.get("result")
            if self._ann_result_gpu_pending(result):
                active_stream_idxs.append(task.get("stream_idx"))

        decode_only_idx = getattr(self, "real_sm_group_num", 1) - 1
        if self._rag_ann_uses_decode_pool():
            # The ANN execution stream is bound to Decode's GreenContext. Do
            # not move Decode to another partition until those kernels finish.
            compatible = not active_stream_idxs or all(
                idx == target_stream_idx for idx in active_stream_idxs
            )
        else:
            compatible = (
                not active_stream_idxs
                or target_stream_idx == decode_only_idx
                or all(idx == target_stream_idx for idx in active_stream_idxs)
            )
        blocked = not compatible
        if blocked and not was_blocked:
            self.rag_ann_partition_drain_events = (
                getattr(self, "rag_ann_partition_drain_events", 0) + 1
            )
            drain_ms = self._ann_inflight_predicted_gpu_ms(inflight)
            drain_history = getattr(
                self, "rag_ann_partition_drain_predicted_ms", None
            )
            if drain_history is not None:
                drain_history.append(drain_ms)
            logger.debug(
                "Deferring PD stream-group switch target=%s for ANN groups=%s "
                "predicted_drain_ms=%.3f",
                target_stream_idx,
                active_stream_idxs,
                drain_ms,
            )
        elif not blocked and was_blocked:
            logger.debug(
                "Resuming PD stream-group switch target=%s for ANN groups=%s",
                target_stream_idx,
                active_stream_idxs,
            )
        self.rag_ann_partition_switch_deferred = blocked
        return blocked

    def _get_stream_group_idx_for_stream(
        self, stream: torch.cuda.Stream
    ) -> Optional[int]:
        stream_groups = getattr(self, "stream_groups", None)
        if not stream_groups:
            return None

        stream_ptr = int(stream.cuda_stream)
        for idx, group in enumerate(stream_groups):
            for candidate in group:
                if int(candidate.cuda_stream) == stream_ptr:
                    return idx
        return None

    def _record_ann_pre_switch_skip(self, reason: str):
        counts = getattr(
            self, "rag_ann_pre_switch_no_launch_reason_counts", None
        )
        if counts is None:
            counts = {}
            self.rag_ann_pre_switch_no_launch_reason_counts = counts
        self._increment_histogram(counts, reason)

    def _poll_ann_before_partition_switch_if_ready(
        self,
        *,
        was_switch_blocked: bool,
        switch_blocked: bool,
        pre_switch_ann_launched: bool,
    ) -> bool:
        """Harvest an ANN completion before a costly partition transition."""

        if switch_blocked or not (
            was_switch_blocked or pre_switch_ann_launched
        ):
            return False
        inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]
        pre_switch_completion = pre_switch_ann_launched or any(
            task.get("launch_context") == "pre_switch"
            for task in inflight
        )
        inline_completions = self._poll_ann_search_side_channel(
            collect_ready_inline=True
        )
        self.rag_ann_partition_completion_polls = (
            getattr(self, "rag_ann_partition_completion_polls", 0) + 1
        )
        self.rag_ann_partition_inline_completions = (
            getattr(self, "rag_ann_partition_inline_completions", 0)
            + inline_completions
        )
        if pre_switch_completion:
            self.rag_ann_pre_switch_completion_polls = (
                getattr(
                    self,
                    "rag_ann_pre_switch_completion_polls",
                    0,
                )
                + 1
            )
            self.rag_ann_pre_switch_inline_completions = (
                getattr(
                    self,
                    "rag_ann_pre_switch_inline_completions",
                    0,
                )
                + inline_completions
            )
        return True

    def _poll_ann_at_scheduler_tick_start(self) -> int:
        """Poll deferred ANN inline so a following switch cannot hide send."""

        switch_deferred = bool(
            getattr(self, "rag_ann_partition_switch_deferred", False)
        )
        inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]
        pre_switch_completion = switch_deferred and any(
            task.get("launch_context") == "pre_switch"
            for task in inflight
        )
        decode_tick_poll = bool(inflight) and self._rag_ann_uses_decode_pool()
        inline_completions = self._poll_ann_search_side_channel(
            collect_ready_inline=switch_deferred or decode_tick_poll
        )
        if decode_tick_poll:
            self.rag_ann_decode_tick_completion_polls = (
                getattr(self, "rag_ann_decode_tick_completion_polls", 0) + 1
            )
            self.rag_ann_decode_tick_inline_completions = (
                getattr(self, "rag_ann_decode_tick_inline_completions", 0)
                + inline_completions
            )
        if not switch_deferred:
            return inline_completions

        self.rag_ann_partition_completion_polls = (
            getattr(self, "rag_ann_partition_completion_polls", 0) + 1
        )
        self.rag_ann_partition_inline_completions = (
            getattr(self, "rag_ann_partition_inline_completions", 0)
            + inline_completions
        )
        if pre_switch_completion:
            self.rag_ann_pre_switch_completion_polls = (
                getattr(
                    self,
                    "rag_ann_pre_switch_completion_polls",
                    0,
                )
                + 1
            )
            self.rag_ann_pre_switch_inline_completions = (
                getattr(
                    self,
                    "rag_ann_pre_switch_inline_completions",
                    0,
                )
                + inline_completions
            )
        return inline_completions

    def _poll_ann_after_decode_sync(self) -> int:
        """Harvest decode-GC ANN results at the Decode iteration safe point."""

        if (
            not self._rag_ann_uses_decode_pool()
            or not getattr(self, "ann_search_inflight", None)
        ):
            return 0
        inline_completions = self._poll_ann_search_side_channel(
            collect_ready_inline=True
        )
        self.rag_ann_decode_completion_polls = (
            getattr(self, "rag_ann_decode_completion_polls", 0) + 1
        )
        self.rag_ann_decode_inline_completions = (
            getattr(self, "rag_ann_decode_inline_completions", 0)
            + inline_completions
        )
        return inline_completions

    def _ann_pre_switch_hbm_ready(
        self, batch_rows: int
    ) -> tuple[bool, str]:
        """Prove that the fast path will not mutate shared-HBM ownership."""

        controller = getattr(self, "shared_hbm_controller", None)
        if controller is None:
            return True, ""
        if getattr(controller, "pending_ann_reclaim", None) is not None:
            return False, "ann_reclaim_pending"
        if getattr(controller, "pending_kv_native_map", None) is not None:
            return False, "kv_native_map_pending"
        if getattr(controller, "pending_ivf_reload", None) is not None:
            return False, "ivf_reload_pending"
        if getattr(controller, "pending_kv_publications", None):
            return False, "kv_publication_pending"
        if getattr(controller, "kv_backlog_committed", False):
            return False, "kv_backlog_committed"

        pool = getattr(controller, "pool", None)
        if pool is None:
            return False, "missing_hbm_pool"
        if getattr(pool, "mode", "scratch") == "ivf":
            # before_ann_launch() may evict or reload IVF list extents even
            # when the scratch mapping is already large enough. That work is
            # not part of this fast path's latency/credit proof.
            return False, "ivf_residency_mutation"
        if getattr(pool, "faiss_scratch_detaching", lambda: False)():
            return False, "ann_scratch_detaching"
        bound, mapping, _ = pool.faiss_mapping_state()
        if not bound or len(mapping) != 2 or int(mapping[1]) <= 16:
            return False, "ann_scratch_unbound"

        requested_bytes = controller.ann_scratch_request_bytes(
            batch_rows,
            max_inflight=getattr(self, "rag_ann_max_inflight", 1),
        )
        snapshot = pool.snapshot()
        if int(snapshot.moving_bytes) > 0:
            return False, "hbm_moving"
        resident_bytes = min(
            int(mapping[1]), int(snapshot.ann_scratch_bytes)
        )
        if resident_bytes < min(int(requested_bytes), int(snapshot.total_bytes)):
            return False, "ann_scratch_too_small"

        required_tokens = controller.predict_kv_tokens(self, record=False)
        current_capacity = int(
            controller.token_allocator.current_mapped_capacity()
        )
        if required_tokens > current_capacity:
            return False, "kv_growth_predicted"
        return True, ""

    def _maybe_launch_ann_before_partition_switch(
        self, target_stream_idx: int
    ) -> bool:
        """Launch one deadline-risk ANN batch before a blocking GC switch.

        decode-gc can execute ANN in the current Decode lane. If a measured
        partition switch is predicted to consume the oldest ANN request's
        remaining deadline, submit one strictly credit-bounded microbatch in
        the current epoch. The existing active-ANN drain guard then delays the
        switch until that batch's event completes.
        """

        if not getattr(self, "rag_ann_pre_switch_launch", False):
            return False
        self.rag_ann_pre_switch_checks = (
            getattr(self, "rag_ann_pre_switch_checks", 0) + 1
        )

        p99_credit_decision = None

        def skip(reason: str) -> bool:
            self._record_ann_pre_switch_skip(reason)
            if p99_credit_decision is not None:
                # Once the exact-p99 selector has run, every remaining skip
                # reason is a fixed internal gate. Keep only the terminal
                # reason on the single last-decision record.
                p99_credit_decision["final_skip_reason"] = reason
            return False

        plan = getattr(self, "rag_frontend_plan", None)
        pending = getattr(self, "ann_search_pending", None)
        current_stream_idx = get_current_stream_idx()
        if (
            getattr(self, "rag_policy_mode", "full") != "full"
            or getattr(self, "rag_ann_frontend_arbitration", "overlap")
            != "fair"
            or getattr(self, "rag_ann_launch_site", "early") != "adaptive"
            or not self._rag_ann_work_conserving_decode_overlap_enabled()
        ):
            return skip("policy")
        if not pending:
            return skip("no_pending")
        if plan is None:
            return skip("plan:none")
        plan_reason = str(getattr(plan, "reason", ""))
        if not getattr(plan, "allow_ann", False):
            return skip(f"plan:{plan_reason}:ann_disabled")
        if plan_reason != "ann_decode_overlap":
            return skip(f"plan:{plan_reason}:reason")
        if getattr(plan, "forced_ann", False):
            return skip(f"plan:{plan_reason}:forced")
        # An immediate handoff is the normal RAG pipeline state: one request
        # has GPU candidates ready while a later request is waiting for ANN.
        # It is safe to bridge that partition transition only through the
        # explicit Prefill-deadline check below; rejecting the state here made
        # the fast path unreachable under real mixed traffic.
        if target_stream_idx == current_stream_idx:
            return skip("same_partition")
        if (
            getattr(self, "rag_ann_partition_switch_deferred", False)
            or self._rag_ann_gpu_work_active()
        ):
            return skip("ann_active_or_draining")
        if (
            getattr(self, "rag_decode_recovery_pending", False)
            or getattr(self, "rag_handoff_decode_release_pending", False)
        ):
            return skip("decode_or_handoff_recovery")
        if not self._decode_batch_active():
            return skip("no_decode")
        if not (
            0 <= current_stream_idx < len(getattr(self, "sm_counts", ()))
            and self.sm_counts[current_stream_idx][1] > 0
        ):
            return skip("no_decode_sms")

        pair_history = list(
            getattr(self, "rag_partition_switch_pair_latency_ms", {}).get(
                (current_stream_idx, target_stream_idx), ()
            )
        )
        global_history = list(
            getattr(self, "rag_partition_switch_latency_ms", ())
        )
        if len(pair_history) < 4 and len(global_history) < 4:
            return skip("insufficient_switch_samples")

        inflight = getattr(self, "ann_search_inflight", None) or []
        if isinstance(inflight, dict):
            inflight = [inflight]
        max_inflight = max(
            1, int(getattr(self, "rag_ann_max_inflight", 1))
        )
        decode_max_inflight = int(
            getattr(self, "rag_ann_max_inflight_during_decode", 0)
        )
        if decode_max_inflight > 0:
            max_inflight = min(max_inflight, decode_max_inflight)
        if len(inflight) >= max_inflight:
            return skip("inflight_full")
        max_batches_per_tick = max(
            0, int(getattr(self, "rag_ann_max_batches_per_tick", 0))
        )
        if (
            max_batches_per_tick > 0
            and int(
                getattr(self, "rag_ann_batches_granted_this_tick", 0)
            )
            >= max_batches_per_tick
        ):
            return skip("tick_limit")
        microbatch_limit = self._effective_ann_microbatch_rows()
        batch_rows, predicted_batch_ms = self._select_ann_credit_batch_rows(
            pending,
            inflight,
            microbatch_limit,
            plan=plan,
            allow_atomic_liveness=False,
            record_hold=False,
            allow_current_epoch=True,
            record_downsize=False,
        )
        if batch_rows <= 0:
            return skip("credit")

        # A blocked switch does not end this scheduler iteration: one decode
        # step still runs in the current partition, and the ANN completion is
        # not harvested until the following tick. Keep Decode's p99 for the
        # Prefill wall path, but predict ANN's no-prelaunch path from measured
        # launch-to-output service rather than GPU kernel time alone.
        predicted_decode_ms = self._rag_ann_decode_prediction_ms(
            current_stream_idx, include_scheduling=True
        )
        if predicted_decode_ms is None:
            return skip("decode_prediction_unavailable")
        completion_path = self._current_ann_gpu_path()
        if completion_path != "full_gpu":
            return skip("completion_path_unsupported")
        predicted_gpu_critical_ms = self._predict_ann_gpu_critical_path_ms(
            batch_rows
        )
        if predicted_gpu_critical_ms is None:
            return skip("gpu_prediction_unavailable")
        # Keep the ordinary credit-selected batch as the no-prelaunch
        # counterfactual.  The exact-p99 selector below is local to this
        # pre-switch fast path; if it declines to launch, the normal
        # post-switch launcher selects again with the ordinary credit policy
        # and can admit this larger prefix.
        no_launch_batch_rows = batch_rows
        no_launch_predicted_gpu_critical_ms = predicted_gpu_critical_ms
        # Credit selection uses the configured service percentile (normally
        # p95). Pre-switch launch is a stricter tail path: re-check the exact
        # batch p99 against the same non-atomic credit before admission, and
        # carry p99 into the launched task's drain/safety accounting.
        p99_queued_gpu_ms = self._ann_inflight_predicted_gpu_ms(inflight)
        p99_credit_limit_ms = self._ann_launch_credit_limit_ms(
            plan,
            allow_current_epoch=True,
        )
        (
            batch_rows,
            predicted_batch_ms,
            predicted_gpu_critical_ms,
        ) = self._select_ann_pre_switch_p99_credit_batch_rows(
            pending,
            batch_rows,
            predicted_batch_ms,
            p99_queued_gpu_ms,
            p99_credit_limit_ms,
            predicted_gpu_critical_ms,
        )
        p99_credit_decision = self.rag_ann_pre_switch_last_gpu_p99_credit
        p99_credit_decision["request_id"] = pending[0].get("request_id")
        if batch_rows <= 0:
            self.rag_ann_pre_switch_gpu_p99_credit_holds = (
                getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_holds",
                    0,
                )
                + 1
            )
            return skip("gpu_p99_credit")
        strict_predicted_batch_ms = max(
            float(predicted_batch_ms),
            predicted_gpu_critical_ms,
        )
        projected_overlap_ms = max(
            predicted_gpu_critical_ms,
            float(predicted_decode_ms),
        )
        prelaunch_completion_predictions = (
            self._predict_ann_completion_service_quantiles_ms(
                batch_rows,
                predicted_gpu_critical_ms,
                completion_path,
            )
        )
        if prelaunch_completion_predictions is None:
            return skip("completion_prediction_unavailable")
        if no_launch_batch_rows == batch_rows:
            completion_predictions = prelaunch_completion_predictions
        else:
            completion_predictions = (
                self._predict_ann_completion_service_quantiles_ms(
                    no_launch_batch_rows,
                    no_launch_predicted_gpu_critical_ms,
                    completion_path,
                )
            )
            if completion_predictions is None:
                return skip("completion_prediction_unavailable")
        (
            predicted_prelaunch_completion_p50_ms,
            predicted_prelaunch_completion_p99_ms,
        ) = prelaunch_completion_predictions
        (
            predicted_completion_p50_ms,
            predicted_completion_p99_ms,
        ) = completion_predictions
        inline_tail_ms = self._predict_ann_inline_collection_tail_ms(
            batch_rows, completion_path
        )
        if inline_tail_ms is None:
            return skip("inline_tail_prediction_unavailable")
        predicted_switch_ms = self._rag_lease_switch_cost_estimate_ms(
            current_stream_idx,
            target_stream_idx,
        )
        quantile_aligned_switch_cost = (
            self._rag_ann_pre_switch_quantile_aligned_switch_cost_enabled()
        )
        robust_switch_pair_samples = len(
            getattr(
                self,
                "rag_partition_switch_pair_latency_ms",
                {},
            ).get((current_stream_idx, target_stream_idx), ())
        )
        quantile_aligned_switch_cost_applied = bool(
            quantile_aligned_switch_cost
            and robust_switch_pair_samples >= 4
        )
        if quantile_aligned_switch_cost_applied:
            predicted_robust_switch_ms = (
                self._rag_lease_switch_cost_estimate_ms(
                    current_stream_idx,
                    target_stream_idx,
                    percentile_override=50.0,
                )
            )
            quantile_aligned_switch_cost_reason = "pair_p50"
        else:
            # Never substitute a global p50 for an immature transition pair.
            # Its direction mix can be radically cheaper than this pair.
            predicted_robust_switch_ms = predicted_switch_ms
            quantile_aligned_switch_cost_reason = (
                "insufficient_pair_samples"
                if quantile_aligned_switch_cost
                else "disabled"
            )
        robust_switch_history = getattr(
            self,
            "rag_ann_pre_switch_predicted_robust_switch_ms",
            None,
        )
        if robust_switch_history is None:
            robust_switch_history = deque(maxlen=64)
            self.rag_ann_pre_switch_predicted_robust_switch_ms = (
                robust_switch_history
            )
        robust_switch_history.append(predicted_robust_switch_ms)
        safety_margin_ms = max(
            0.0, getattr(self, "rag_ann_safety_margin_ms", 0.0)
        )
        oldest_wait_ms = max(
            0.0, float(getattr(plan, "ann_wait_ms", 0.0))
        )
        deadline_ms = max(
            0.001, float(getattr(self, "rag_ann_deadline_ms", 50.0))
        )
        tail_no_launch_eta_ms = (
            predicted_switch_ms + predicted_completion_p99_ms
        )
        robust_no_launch_eta_ms = (
            predicted_robust_switch_ms + predicted_completion_p50_ms
        )
        prelaunch_p99_eta_ms = projected_overlap_ms + inline_tail_ms
        tail_predicted_gain_ms = tail_no_launch_eta_ms - (
            prelaunch_p99_eta_ms + safety_margin_ms
        )
        robust_predicted_gain_ms = robust_no_launch_eta_ms - (
            prelaunch_p99_eta_ms + safety_margin_ms
        )
        downsized_min_robust_gain_ms = (
            self._rag_ann_pre_switch_downsized_min_robust_gain_ms()
        )
        robust_gain_threshold_candidate = bool(
            p99_credit_decision["downsized"]
            and p99_credit_decision["adaptive_reason"] == "derived"
        )
        tail_deadline_risk = (
            oldest_wait_ms
            + tail_no_launch_eta_ms
            + safety_margin_ms
            >= deadline_ms
        )
        robust_deadline_risk = (
            oldest_wait_ms
            + robust_no_launch_eta_ms
            + safety_margin_ms
            >= deadline_ms
        )
        intrinsic_debt_recovery = (
            oldest_wait_ms
            + predicted_completion_p50_ms
            + safety_margin_ms
            >= deadline_ms
        )
        trigger_type = (
            "intrinsic_debt_recovery"
            if intrinsic_debt_recovery
            else "switch_induced"
        )
        robust_gain_threshold_bypassed = bool(
            robust_gain_threshold_candidate
            and downsized_min_robust_gain_ms > 0.0
            and intrinsic_debt_recovery
        )
        robust_gain_threshold_eligible = bool(
            robust_gain_threshold_candidate
            and not robust_gain_threshold_bypassed
        )
        batch_net_gain_enabled = (
            self._rag_ann_pre_switch_batch_net_gain_enabled()
        )
        residual_rows = max(0, no_launch_batch_rows - batch_rows)
        residual_gpu_critical_ms = None
        residual_completion_p50_ms = None
        residual_completion_p99_ms = None
        residual_eta_ms = None
        residual_gain_ms = None
        batch_net_gain_ms = None
        batch_net_gain_unavailable_reason = None
        if batch_net_gain_enabled and robust_gain_threshold_candidate:
            if residual_rows <= 0:
                batch_net_gain_ms = robust_predicted_gain_ms
            else:
                residual_gpu_critical_ms = (
                    self._predict_ann_gpu_critical_path_ms(residual_rows)
                )
                if residual_gpu_critical_ms is None:
                    batch_net_gain_unavailable_reason = (
                        "residual_gpu_prediction_unavailable"
                    )
                else:
                    residual_completion_predictions = (
                        self._predict_ann_completion_service_quantiles_ms(
                            residual_rows,
                            residual_gpu_critical_ms,
                            completion_path,
                        )
                    )
                    if residual_completion_predictions is None:
                        batch_net_gain_unavailable_reason = (
                            "residual_completion_prediction_unavailable"
                        )
                    else:
                        (
                            residual_completion_p50_ms,
                            residual_completion_p99_ms,
                        ) = residual_completion_predictions
                        residual_eta_ms = (
                            prelaunch_p99_eta_ms
                            + predicted_switch_ms
                            + residual_completion_p99_ms
                        )
                        residual_gain_ms = robust_no_launch_eta_ms - (
                            residual_eta_ms + safety_margin_ms
                        )
                        batch_net_gain_ms = (
                            batch_rows * robust_predicted_gain_ms
                            + residual_rows * residual_gain_ms
                        ) / max(1, no_launch_batch_rows)
        threshold_decision_gain_ms = (
            batch_net_gain_ms
            if batch_net_gain_enabled and batch_net_gain_ms is not None
            else robust_predicted_gain_ms
        )
        cross_class_debt_enabled = (
            self._rag_ann_pre_switch_cross_class_debt_enabled()
        )
        prefill_wait_ms = 0.0
        prefill_deadline_ms = max(
            0.001,
            float(getattr(self, "rag_prefill_deadline_ms", 500.0)),
        )
        prefill_no_launch_ready_ms = None
        prefill_prelaunch_ready_ms = None
        prefill_no_launch_debt_ms = None
        prefill_prelaunch_debt_ms = None
        incremental_prefill_debt_ms = None
        ann_no_launch_deadline_debt_ms = None
        ann_prelaunch_deadline_debt_ms = None
        ann_residual_deadline_debt_ms = None
        ann_deadline_debt_saved_ms = None
        cross_class_net_debt_ms = None
        cross_class_debt_candidate = False
        cross_class_debt_passed = None
        cross_class_debt_bypass_pending = False
        if getattr(plan, "allow_prefill", False):
            prefill_wait_ms = max(
                0.0, float(getattr(plan, "prefill_wait_ms", 0.0))
            )
            # The current Decode step happens in either branch.  Charging the
            # full overlapped prelaunch ETA to Prefill double-counted that
            # unavoidable work.  Compare only the incremental deadline debt.
            prefill_no_launch_ready_ms = (
                prefill_wait_ms
                + predicted_decode_ms
                + predicted_switch_ms
                + safety_margin_ms
            )
            prefill_prelaunch_ready_ms = (
                prefill_wait_ms
                + prelaunch_p99_eta_ms
                + predicted_switch_ms
                + safety_margin_ms
            )
            prefill_no_launch_debt_ms = max(
                0.0, prefill_no_launch_ready_ms - prefill_deadline_ms
            )
            prefill_prelaunch_debt_ms = max(
                0.0, prefill_prelaunch_ready_ms - prefill_deadline_ms
            )
            incremental_prefill_debt_ms = max(
                0.0,
                prefill_prelaunch_debt_ms - prefill_no_launch_debt_ms,
            )
            ann_no_launch_deadline_debt_ms = no_launch_batch_rows * max(
                0.0,
                oldest_wait_ms
                + robust_no_launch_eta_ms
                + safety_margin_ms
                - deadline_ms,
            )
            ann_prelaunch_deadline_debt_ms = batch_rows * max(
                0.0,
                oldest_wait_ms
                + prelaunch_p99_eta_ms
                + safety_margin_ms
                - deadline_ms,
            )
            if residual_rows <= 0:
                ann_residual_deadline_debt_ms = 0.0
            elif residual_eta_ms is not None:
                ann_residual_deadline_debt_ms = residual_rows * max(
                    0.0,
                    oldest_wait_ms
                    + residual_eta_ms
                    + safety_margin_ms
                    - deadline_ms,
                )
            if ann_residual_deadline_debt_ms is not None:
                ann_deadline_debt_saved_ms = (
                    ann_no_launch_deadline_debt_ms
                    - ann_prelaunch_deadline_debt_ms
                    - ann_residual_deadline_debt_ms
                )
                cross_class_net_debt_ms = ann_deadline_debt_saved_ms - (
                    _RAG_ANN_PRE_SWITCH_CROSS_CLASS_PREFILL_DEBT_WEIGHT
                    * incremental_prefill_debt_ms
                )
            cross_class_debt_candidate = bool(
                prefill_prelaunch_ready_ms >= prefill_deadline_ms
                and prefill_no_launch_debt_ms > 0.0
                and intrinsic_debt_recovery
                and batch_net_gain_enabled
                and batch_net_gain_ms is not None
                and ann_deadline_debt_saved_ms is not None
                and incremental_prefill_debt_ms <= deadline_ms
            )
            if cross_class_debt_candidate:
                cross_class_debt_passed = bool(
                    cross_class_net_debt_ms + 1e-6
                    >= _RAG_ANN_PRE_SWITCH_CROSS_CLASS_MIN_NET_DEBT_MS
                )
        counterfactual = {
            "path": completion_path,
            # Compatibility: batch_size remains the actual prelaunch batch.
            "batch_size": batch_rows,
            "prelaunch_batch_size": batch_rows,
            "no_launch_batch_size": no_launch_batch_rows,
            "oldest_wait_ms": oldest_wait_ms,
            "deadline_ms": deadline_ms,
            "safety_margin_ms": safety_margin_ms,
            "predicted_switch_ms": predicted_switch_ms,
            "predicted_tail_switch_ms": predicted_switch_ms,
            "predicted_robust_switch_ms": predicted_robust_switch_ms,
            "tail_switch_percentile": float(
                getattr(self, "rag_lease_switch_cost_percentile", 95.0)
            ),
            "robust_switch_percentile": (
                50.0
                if quantile_aligned_switch_cost_applied
                else float(
                    getattr(
                        self,
                        "rag_lease_switch_cost_percentile",
                        95.0,
                    )
                )
            ),
            "quantile_aligned_switch_cost": quantile_aligned_switch_cost,
            "quantile_aligned_switch_cost_applied": (
                quantile_aligned_switch_cost_applied
            ),
            "quantile_aligned_switch_cost_reason": (
                quantile_aligned_switch_cost_reason
            ),
            "robust_switch_pair_samples": robust_switch_pair_samples,
            # Compatibility fields retain the conservative p99 envelope.
            "predicted_completion_service_ms": predicted_completion_p99_ms,
            "predicted_completion_service_p50_ms": (
                predicted_completion_p50_ms
            ),
            "predicted_completion_service_p99_ms": (
                predicted_completion_p99_ms
            ),
            "predicted_prelaunch_completion_service_p50_ms": (
                predicted_prelaunch_completion_p50_ms
            ),
            "predicted_prelaunch_completion_service_p99_ms": (
                predicted_prelaunch_completion_p99_ms
            ),
            "predicted_gpu_critical_ms": predicted_gpu_critical_ms,
            "no_launch_predicted_gpu_critical_ms": (
                no_launch_predicted_gpu_critical_ms
            ),
            "predicted_overlap_ms": projected_overlap_ms,
            "predicted_inline_collection_send_ms": inline_tail_ms,
            "no_launch_eta_ms": tail_no_launch_eta_ms,
            "prelaunch_eta_ms": prelaunch_p99_eta_ms,
            "predicted_gain_ms": robust_predicted_gain_ms,
            "tail_no_launch_eta_ms": tail_no_launch_eta_ms,
            "robust_no_launch_eta_ms": robust_no_launch_eta_ms,
            "prelaunch_p99_eta_ms": prelaunch_p99_eta_ms,
            "tail_predicted_gain_ms": tail_predicted_gain_ms,
            "robust_predicted_gain_ms": robust_predicted_gain_ms,
            "tail_gain_ms": tail_predicted_gain_ms,
            "robust_gain_ms": robust_predicted_gain_ms,
            "batch_net_gain_enabled": batch_net_gain_enabled,
            "residual_rows": residual_rows,
            "residual_gpu_critical_ms": residual_gpu_critical_ms,
            "residual_completion_service_p50_ms": (
                residual_completion_p50_ms
            ),
            "residual_completion_service_p99_ms": (
                residual_completion_p99_ms
            ),
            "residual_eta_ms": residual_eta_ms,
            "residual_gain_ms": residual_gain_ms,
            "batch_net_gain_ms": batch_net_gain_ms,
            "batch_net_gain_unavailable_reason": (
                batch_net_gain_unavailable_reason
            ),
            "threshold_decision_gain_ms": threshold_decision_gain_ms,
            "cross_class_debt_enabled": cross_class_debt_enabled,
            "prefill_wait_ms": prefill_wait_ms,
            "prefill_deadline_ms": prefill_deadline_ms,
            "prefill_no_launch_ready_ms": prefill_no_launch_ready_ms,
            "prefill_prelaunch_ready_ms": prefill_prelaunch_ready_ms,
            "prefill_no_launch_debt_ms": prefill_no_launch_debt_ms,
            "prefill_prelaunch_debt_ms": prefill_prelaunch_debt_ms,
            "incremental_prefill_debt_ms": incremental_prefill_debt_ms,
            "ann_no_launch_deadline_debt_ms": (
                ann_no_launch_deadline_debt_ms
            ),
            "ann_prelaunch_deadline_debt_ms": (
                ann_prelaunch_deadline_debt_ms
            ),
            "ann_residual_deadline_debt_ms": (
                ann_residual_deadline_debt_ms
            ),
            "ann_deadline_debt_saved_ms": ann_deadline_debt_saved_ms,
            "cross_class_net_debt_ms": cross_class_net_debt_ms,
            "cross_class_prefill_debt_weight": (
                _RAG_ANN_PRE_SWITCH_CROSS_CLASS_PREFILL_DEBT_WEIGHT
            ),
            "cross_class_min_net_debt_ms": (
                _RAG_ANN_PRE_SWITCH_CROSS_CLASS_MIN_NET_DEBT_MS
            ),
            "cross_class_debt_candidate": cross_class_debt_candidate,
            "cross_class_debt_passed": cross_class_debt_passed,
            "cross_class_debt_bypassed": False,
            "cross_class_debt_bypass_counted": False,
            "tail_deadline_risk": tail_deadline_risk,
            "robust_deadline_risk": robust_deadline_risk,
            "decision_quantile": 50,
            "trigger_type": trigger_type,
            "downsized_min_robust_gain_ms": (
                downsized_min_robust_gain_ms
            ),
            "robust_gain_threshold_candidate": (
                robust_gain_threshold_candidate
            ),
            "robust_gain_threshold_eligible": (
                robust_gain_threshold_eligible
            ),
            "robust_gain_threshold_bypassed": (
                robust_gain_threshold_bypassed
            ),
            "robust_gain_threshold_bypass_reason": (
                "intrinsic_debt_recovery"
                if robust_gain_threshold_bypassed
                else None
            ),
            "robust_gain_threshold_bypass_counted": False,
            "robust_gain_threshold_applied": False,
            "robust_gain_threshold_passed": None,
            "robust_gain_headroom_ms": (
                threshold_decision_gain_ms
                - downsized_min_robust_gain_ms
                if robust_gain_threshold_eligible
                else None
            ),
        }
        self.rag_ann_pre_switch_last_counterfactual = counterfactual
        predicted_gains = getattr(
            self, "rag_ann_pre_switch_predicted_gain_ms", None
        )
        if predicted_gains is None:
            predicted_gains = deque(maxlen=64)
            self.rag_ann_pre_switch_predicted_gain_ms = predicted_gains
        predicted_gains.append(robust_predicted_gain_ms)
        if not tail_deadline_risk:
            return skip("deadline_safe")
        if not robust_deadline_risk:
            return skip("deadline_risk_tail_only")
        if (
            prelaunch_p99_eta_ms + safety_margin_ms
            >= robust_no_launch_eta_ms
        ):
            self.rag_ann_pre_switch_counterfactual_no_gain_skips = (
                getattr(
                    self,
                    "rag_ann_pre_switch_counterfactual_no_gain_skips",
                    0,
                )
                + 1
            )
            return skip("counterfactual_no_gain")

        classification_invariant_violation = (
            (intrinsic_debt_recovery and not robust_deadline_risk)
            or (
                not intrinsic_debt_recovery
                and (
                    not robust_deadline_risk
                    or (
                        oldest_wait_ms
                        + predicted_completion_p50_ms
                        + safety_margin_ms
                        >= deadline_ms
                    )
                )
            )
        )
        counterfactual["classification_invariant_violation"] = (
            classification_invariant_violation
        )
        if classification_invariant_violation:
            self.rag_ann_pre_switch_classification_invariant_violations = (
                getattr(
                    self,
                    "rag_ann_pre_switch_classification_invariant_violations",
                    0,
                )
                + 1
            )
            return skip("classification_invariant")

        if (
            prefill_prelaunch_ready_ms is not None
            and prefill_prelaunch_ready_ms >= prefill_deadline_ms
        ):
            if cross_class_debt_candidate:
                self.rag_ann_pre_switch_cross_class_debt_candidates = (
                    getattr(
                        self,
                        "rag_ann_pre_switch_cross_class_debt_candidates",
                        0,
                    )
                    + 1
                )
                cross_class_net_history = getattr(
                    self,
                    "rag_ann_pre_switch_eligible_cross_class_net_debt_ms",
                    None,
                )
                if cross_class_net_history is None:
                    cross_class_net_history = deque(maxlen=64)
                    self.rag_ann_pre_switch_eligible_cross_class_net_debt_ms = (
                        cross_class_net_history
                    )
                cross_class_net_history.append(cross_class_net_debt_ms)
                if cross_class_debt_passed:
                    self.rag_ann_pre_switch_cross_class_debt_would_admit = (
                        getattr(
                            self,
                            "rag_ann_pre_switch_cross_class_debt_would_admit",
                            0,
                        )
                        + 1
                    )
                if (
                    cross_class_debt_enabled
                    and cross_class_debt_passed
                ):
                    cross_class_debt_bypass_pending = True
                    counterfactual["cross_class_debt_bypassed"] = True
                else:
                    if cross_class_debt_enabled:
                        self.rag_ann_pre_switch_cross_class_debt_holds = (
                            getattr(
                                self,
                                "rag_ann_pre_switch_cross_class_debt_holds",
                                0,
                            )
                            + 1
                        )
                    return skip(
                        "cross_class_debt_below_min"
                        if cross_class_debt_enabled
                        else "prefill_deadline"
                    )
            else:
                return skip("prefill_deadline")

        hbm_ready, hbm_reason = self._ann_pre_switch_hbm_ready(batch_rows)
        if not hbm_ready:
            return skip(hbm_reason)
        if self._get_rag_ann_stream() is None:
            return skip("no_stream")

        if cross_class_debt_bypass_pending:
            self.rag_ann_pre_switch_cross_class_debt_bypasses = (
                getattr(
                    self,
                    "rag_ann_pre_switch_cross_class_debt_bypasses",
                    0,
                )
                + 1
            )
            counterfactual["cross_class_debt_bypass_counted"] = True

        if robust_gain_threshold_bypassed:
            self.rag_ann_pre_switch_robust_gain_threshold_bypasses = (
                getattr(
                    self,
                    "rag_ann_pre_switch_robust_gain_threshold_bypasses",
                    0,
                )
                + 1
            )
            counterfactual["robust_gain_threshold_bypass_counted"] = True

        if robust_gain_threshold_eligible:
            if batch_net_gain_enabled and batch_net_gain_ms is None:
                self.rag_ann_pre_switch_batch_net_gain_unavailable = (
                    getattr(
                        self,
                        "rag_ann_pre_switch_batch_net_gain_unavailable",
                        0,
                    )
                    + 1
                )
                return skip(
                    batch_net_gain_unavailable_reason
                    or "batch_net_gain_unavailable"
                )
            counterfactual["robust_gain_threshold_applied"] = True
            eligible_gains = getattr(
                self,
                "rag_ann_pre_switch_eligible_robust_gain_ms",
                None,
            )
            if eligible_gains is None:
                eligible_gains = deque(maxlen=64)
                self.rag_ann_pre_switch_eligible_robust_gain_ms = (
                    eligible_gains
                )
            eligible_gains.append(threshold_decision_gain_ms)
            raw_eligible_gains = getattr(
                self,
                "rag_ann_pre_switch_eligible_raw_robust_gain_ms",
                None,
            )
            if raw_eligible_gains is None:
                raw_eligible_gains = deque(maxlen=64)
                self.rag_ann_pre_switch_eligible_raw_robust_gain_ms = (
                    raw_eligible_gains
                )
            raw_eligible_gains.append(robust_predicted_gain_ms)
            if batch_net_gain_ms is not None:
                batch_net_eligible_gains = getattr(
                    self,
                    "rag_ann_pre_switch_eligible_batch_net_gain_ms",
                    None,
                )
                if batch_net_eligible_gains is None:
                    batch_net_eligible_gains = deque(maxlen=64)
                    self.rag_ann_pre_switch_eligible_batch_net_gain_ms = (
                        batch_net_eligible_gains
                    )
                batch_net_eligible_gains.append(batch_net_gain_ms)
            threshold_passed = (
                threshold_decision_gain_ms + 1e-6
                >= downsized_min_robust_gain_ms
            )
            counterfactual["robust_gain_threshold_passed"] = (
                threshold_passed
            )
            if not threshold_passed:
                self.rag_ann_pre_switch_robust_gain_threshold_holds = (
                    getattr(
                        self,
                        "rag_ann_pre_switch_robust_gain_threshold_holds",
                        0,
                    )
                    + 1
                )
                return skip("robust_gain_below_min")

        p99_credit_decision["commit_stage"] = "trigger"
        if p99_credit_decision["downsized"]:
            self.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers = (
                getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsized_triggers",
                    0,
                )
                + 1
            )
        self.rag_ann_pre_switch_deadline_risk_triggers = (
            getattr(
                self,
                "rag_ann_pre_switch_deadline_risk_triggers",
                0,
            )
            + 1
        )
        if intrinsic_debt_recovery:
            self.rag_ann_pre_switch_intrinsic_debt_recovery_triggers = (
                getattr(
                    self,
                    "rag_ann_pre_switch_intrinsic_debt_recovery_triggers",
                    0,
                )
                + 1
            )
        else:
            self.rag_ann_pre_switch_switch_induced_triggers = (
                getattr(
                    self,
                    "rag_ann_pre_switch_switch_induced_triggers",
                    0,
                )
                + 1
            )
        trigger_counts = getattr(
            self, "rag_ann_pre_switch_trigger_type_counts", None
        )
        if trigger_counts is None:
            trigger_counts = {}
            self.rag_ann_pre_switch_trigger_type_counts = trigger_counts
        self._increment_histogram(trigger_counts, trigger_type)
        self.rag_ann_pre_switch_trigger_wait_ms.append(oldest_wait_ms)
        self.rag_ann_pre_switch_predicted_switch_ms.append(
            predicted_switch_ms
        )
        before_batches = getattr(
            self, "rag_ann_pre_switch_launch_batches", 0
        )
        self._launch_ann_search_side_channel(
            max_new_batches=1,
            allow_current_epoch=True,
            bypass_microbatch_wait=True,
            launch_context="pre_switch",
            first_batch_selection=(batch_rows, strict_predicted_batch_ms),
        )
        launched = (
            getattr(self, "rag_ann_pre_switch_launch_batches", 0)
            > before_batches
        )
        if not launched:
            return skip("launch_failed_or_cpu_fallback")
        p99_credit_decision["commit_stage"] = "launch"
        if p99_credit_decision["downsized"]:
            self.rag_ann_pre_switch_gpu_p99_credit_downsized_launches = (
                getattr(
                    self,
                    "rag_ann_pre_switch_gpu_p99_credit_downsized_launches",
                    0,
                )
                + 1
            )
        return True

    def _hybrid_ann_pipeline_limit(self, pending, inflight, max_inflight):
        """Size the hybrid ANN pipeline from backlog; GPU credit stays binding."""
        max_inflight = max(1, int(max_inflight))
        if (
            max_inflight <= 1
            or getattr(self, "rag_policy_mode", "full") != "full"
            or not getattr(self.ann_executor, "hot_index_mode", False)
        ):
            return max_inflight

        pending_rows = sum(
            self._ann_task_num_queries(task) for task in pending
        )
        microbatch_rows = max(1, self._effective_ann_microbatch_rows())
        pending_batches = max(
            1,
            (pending_rows + microbatch_rows - 1) // microbatch_rows,
        )
        unconstrained_limit = min(
            max_inflight, len(inflight) + pending_batches
        )
        limit = unconstrained_limit
        self._increment_histogram(
            self.rag_ann_hybrid_pipeline_window_histogram, limit
        )
        if limit <= 1:
            self.rag_ann_hybrid_pipeline_backlog_holds = (
                getattr(self, "rag_ann_hybrid_pipeline_backlog_holds", 0) + 1
            )
            return 1

        if len(inflight) < limit:
            # Keep the legacy counter for result compatibility while exposing
            # the complete dynamic-window behavior separately.
            self.rag_ann_hybrid_pipeline_second_slot_admissions = (
                getattr(
                    self,
                    "rag_ann_hybrid_pipeline_second_slot_admissions",
                    0,
                )
                + 1
            )
            self.rag_ann_hybrid_pipeline_window_expansions = (
                getattr(
                    self,
                    "rag_ann_hybrid_pipeline_window_expansions",
                    0,
                )
                + 1
            )
        return max(1, limit)

    def _launch_ann_search_side_channel(
        self,
        *,
        max_new_batches: Optional[int] = None,
        allow_current_epoch: bool = False,
        bypass_microbatch_wait: bool = False,
        launch_context: str = "normal",
        first_batch_selection: Optional[tuple[int, float]] = None,
        fitted_batch_cap: Optional[int] = None,
    ):
        plan = getattr(self, 'rag_frontend_plan', None)
        if (getattr(self,'rag_joint_event_input_provider',None) is not None
                and getattr(plan,'joint_event_id',None) is None):
            # Early launch precedes the first frontend decision in the loop.
            # Installing joint mode must not leave a one-tick legacy bypass.
            return
        if getattr(plan, 'joint_event_id', None) is not None:
            if not joint_plan_ready_for_dispatch(self, plan) or not plan.allow_ann:
                return
            if plan.ann_batch_cap is None:
                raise ValueError('Activated joint lease requires a fitted ANN cap')
            if fitted_batch_cap is not None and fitted_batch_cap != plan.ann_batch_cap:
                raise ValueError('Launch cannot replace the activated joint ANN cap')
            fitted_batch_cap = plan.ann_batch_cap
        if fitted_batch_cap is not None:
            if type(fitted_batch_cap) is not int or fitted_batch_cap < 0:
                raise ValueError('Fitted ANN cap must be a nonnegative integer')
            if fitted_batch_cap == 0:
                return
            if first_batch_selection is not None and first_batch_selection[0] > fitted_batch_cap:
                raise ValueError('Preselected ANN batch exceeds fitted lease')
        pending = getattr(self, "ann_search_pending", None)
        plan = getattr(self, "rag_frontend_plan", None)
        if getattr(self, "rag_ann_partition_switch_deferred", False):
            # A requested cross-GC transition cannot complete while work
            # remains queued on the old ANN execution stream. Stop refilling
            # that stream until the existing batches drain.
            self.rag_ann_partition_drain_holds = (
                getattr(self, "rag_ann_partition_drain_holds", 0) + 1
            )
            return
        if (
            self.ann_executor is None
            or pending is None
            or not pending
            or (
                getattr(self, "rag_ann_frontend_arbitration", "overlap")
                == "fair"
                and plan is not None
                and not plan.allow_ann
            )
        ):
            return

        inflight = getattr(self, "ann_search_inflight", None)
        if inflight is None:
            inflight = []
        elif isinstance(inflight, dict):
            inflight = [inflight]

        max_inflight = self._hybrid_ann_pipeline_limit(
            pending,
            inflight,
            getattr(self, "rag_ann_max_inflight", 1),
        )
        oldest_wait_ms = 0.0
        if pending:
            now = time.perf_counter()
            oldest_wait_ms = (
                now - pending[0].get("enqueue_time", now)
            ) * 1000.0
        if (
            not bypass_microbatch_wait
            and self._should_wait_for_ann_microbatch(
                pending, oldest_wait_ms, inflight
            )
        ):
            return

        decode_active = self._decode_batch_active()
        if decode_active:
            decode_max_inflight = getattr(
                self, "rag_ann_max_inflight_during_decode", 0
            )
            if decode_max_inflight > 0:
                max_inflight = min(max_inflight, decode_max_inflight)

        tick_limit_applied = False
        max_batches_per_tick = max(
            0, getattr(self, "rag_ann_max_batches_per_tick", 0)
        )
        granted_this_tick = max(
            0, getattr(self, "rag_ann_batches_granted_this_tick", 0)
        )
        if (
            getattr(self, "rag_ann_frontend_arbitration", "overlap") == "fair"
            and max_batches_per_tick > 0
        ):
            remaining_tick_grants = max(
                0, max_batches_per_tick - granted_this_tick
            )
            if remaining_tick_grants == 0:
                if (
                    pending
                    and not getattr(
                        self, "rag_ann_tick_limit_held_this_tick", False
                    )
                ):
                    self.rag_ann_tick_limit_holds = (
                        getattr(self, "rag_ann_tick_limit_holds", 0) + 1
                    )
                    self.rag_ann_tick_limit_held_this_tick = True
                return
            tick_max_inflight = min(
                max_inflight, len(inflight) + remaining_tick_grants
            )
            tick_limit_applied = tick_max_inflight < max_inflight
            max_inflight = tick_max_inflight

        ann_stream = self._get_rag_ann_stream()
        if ann_stream is None:
            return
        ann_stream_idx = self._get_stream_group_idx_for_stream(ann_stream)
        microbatch_limit = (
            max(0, int(first_batch_selection[0]))
            if first_batch_selection is not None
            else self._effective_ann_microbatch_rows()
        )
        if fitted_batch_cap is not None:
            microbatch_limit = min(microbatch_limit, fitted_batch_cap)
        if microbatch_limit <= 0:
            return
        granted_batches = 0
        attempted_batches = 0
        max_attempts = None
        if max_new_batches is not None:
            max_attempts = max(0, int(max_new_batches))
            max_inflight = min(
                max_inflight,
                len(inflight) + max_attempts,
            )

        while (
            pending
            and len(inflight) < max_inflight
            and (max_attempts is None or attempted_batches < max_attempts)
        ):
            head_task = pending[0]
            route_future = head_task.get("route_future")
            if route_future is not None and not route_future.done():
                break
            current_microbatch_limit = microbatch_limit
            if route_future is not None:
                # Route preparation returns request-matched probe arrays, but
                # several completed futures can be safely concatenated before
                # GPU/cold-index launch. Stop at the first unfinished future so
                # admission never blocks the scheduler thread.
                prepared_rows = 0
                for candidate in pending:
                    candidate_future = candidate.get("route_future")
                    if candidate_future is None or not candidate_future.done():
                        break
                    candidate_rows = self._ann_task_num_queries(candidate)
                    if prepared_rows + candidate_rows > microbatch_limit:
                        break
                    prepared_rows += candidate_rows
                current_microbatch_limit = min(
                    current_microbatch_limit,
                    max(1, prepared_rows),
                )
            if first_batch_selection is not None:
                next_batch_rows, predicted_gpu_ms = first_batch_selection
                first_batch_selection = None
                if fitted_batch_cap is not None and next_batch_rows > current_microbatch_limit:
                    # Routing readiness changed the preselected shape. Keep
                    # the queue intact; do not attach its old service estimate
                    # to a different batch or replan inside this launch.
                    break
                next_batch_rows = min(
                    next_batch_rows, current_microbatch_limit
                )
            else:
                next_batch_rows, predicted_gpu_ms = (
                    self._select_ann_credit_batch_rows(
                        pending,
                        inflight,
                        current_microbatch_limit,
                        plan=plan,
                        allow_current_epoch=allow_current_epoch,
                    )
                )
            if next_batch_rows <= 0:
                break
            if fitted_batch_cap is not None and next_batch_rows > current_microbatch_limit:
                raise ValueError('ANN credit selection widened fitted lease')
            if fitted_batch_cap is not None:
                actual_prefix_rows = 0
                for candidate in pending:
                    rows = self._ann_task_num_queries(candidate)
                    if actual_prefix_rows + rows > next_batch_rows:
                        break
                    actual_prefix_rows += rows
                    if actual_prefix_rows == next_batch_rows:
                        break
                if actual_prefix_rows != next_batch_rows:
                    break

            task = self._pop_ann_microbatch(pending, next_batch_rows)
            if task is None:
                break
            attempted_batches += 1
            task["predicted_gpu_ms"] = predicted_gpu_ms
            task["credit_forced_atomic"] = getattr(
                self,
                "rag_ann_credit_last_forced_atomic",
                False,
            )
            task["launch_context"] = launch_context
            launch_time = time.perf_counter()
            task["launch_time"] = launch_time
            for subtask in task.get("tasks") or [task]:
                queue_wait_ms = (
                    launch_time - subtask["enqueue_time"]
                ) * 1000.0
                subtask["queue_wait_ms"] = queue_wait_ms
                self.rag_ann_queue_wait_ms.append(queue_wait_ms)
                self._observe_ms_histogram(
                    self.rag_ann_queue_wait_histogram, queue_wait_ms
                )
                defer_started = subtask.pop("budget_defer_started", None)
                if defer_started is not None:
                    self.rag_ann_budget_defer_ms.append(
                        (launch_time - defer_started) * 1000.0
                    )
                forced_started = subtask.pop("forced_release_started", None)
                if forced_started is not None:
                    self.rag_ann_forced_release_to_launch_ms.append(
                        (launch_time - forced_started) * 1000.0
                    )
            self._increment_histogram(
                self.rag_ann_batch_size_histogram,
                self._ann_task_num_queries(task),
            )
            launch_reason = plan.reason if plan is not None else "unplanned"
            self._increment_histogram(
                self.rag_ann_launch_reason_counts, launch_reason
            )
            if plan is not None and plan.forced_ann:
                self.rag_ann_forced_release_events += 1
            try:
                with _nvtx_range("ann_search_launch"):
                    task["stream_idx"] = ann_stream_idx
                    task["parent_stream_ptr"] = int(ann_stream.cuda_stream)
                    shared_hbm_controller = getattr(
                        self, "shared_hbm_controller", None
                    )
                    use_gpu = True
                    task["ann_gpu_path"] = None
                    source_tasks = task.get("tasks") or [task]
                    route_futures = [
                        source_task.get("route_future")
                        for source_task in source_tasks
                    ]
                    prepared_items = None
                    if not all(
                        future is None for future in route_futures
                    ):
                        if any(
                            future is None for future in route_futures
                        ):
                            raise RuntimeError(
                                "Cannot mix prepared and unprepared ANN tasks"
                            )
                        prepared_items = [
                            future.result() for future in route_futures
                        ]
                    if shared_hbm_controller is not None:
                        scratch_bytes = (
                            shared_hbm_controller.ann_scratch_request_bytes(
                                self._ann_task_num_queries(task),
                                max_inflight=getattr(
                                    self, "rag_ann_max_inflight", 1
                                ),
                            )
                        )
                        queue_wait_ms = max(
                            float(subtask.get("queue_wait_ms", 0.0))
                            for subtask in task.get("tasks") or [task]
                        )
                        use_gpu = shared_hbm_controller.before_ann_launch(
                            self,
                            requested_bytes=scratch_bytes,
                            num_queries=self._ann_task_num_queries(task),
                            deadline_slack_ms=max(
                                0.0,
                                float(self.rag_ann_deadline_ms)
                                - queue_wait_ms,
                            ),
                        )
                    if use_gpu:
                        try:
                            wait_for_parent = (
                                getattr(
                                    self,
                                    "rag_ann_frontend_arbitration",
                                    "overlap",
                                )
                                == "strict"
                            )
                            if prepared_items is None:
                                task["result"] = (
                                    self.ann_executor.search_async(
                                        task["queries"],
                                        ann_stream,
                                        stream_idx=ann_stream_idx,
                                        wait_for_parent=wait_for_parent,
                                    )
                                )
                            else:
                                prepared = (
                                    self.ann_executor.combine_prepared_hot_searches(
                                        prepared_items
                                    )
                                )
                                task["result"] = (
                                    self.ann_executor.search_prepared_async(
                                        task["queries"],
                                        prepared,
                                        ann_stream,
                                        stream_idx=ann_stream_idx,
                                        wait_for_parent=wait_for_parent,
                                    )
                                )
                            task["ann_gpu_path"] = self._ann_result_gpu_path(
                                task["result"]
                            )
                        except Exception as error:
                            if (
                                shared_hbm_controller is None
                                or "external temporary memory exhausted"
                                not in str(error)
                            ):
                                raise
                            logger.warning(
                                "Faiss strict shared scratch was exhausted; "
                                "routing microbatch to CPU"
                            )
                            if prepared_items is not None:
                                self.ann_executor.cancel_prepared_cold_searches(
                                    prepared_items
                                )
                            task["result"] = (
                                self.ann_executor.search_cpu_async(
                                    task["queries"]
                                )
                            )
                            task["ann_gpu_path"] = None
                            shared_hbm_controller.cpu_fallbacks += (
                                self._ann_task_num_queries(task)
                            )
                    else:
                        if prepared_items is not None:
                            self.ann_executor.cancel_prepared_cold_searches(
                                prepared_items
                            )
                        task["result"] = self.ann_executor.search_cpu_async(
                            task["queries"]
                        )
                        task["ann_gpu_path"] = None
                        if shared_hbm_controller is not None:
                            shared_hbm_controller.cpu_fallbacks += (
                                self._ann_task_num_queries(task)
                            )
                    inflight.append(task)
                    granted_batches += 1
                    queued_gpu_ms = self._ann_inflight_predicted_gpu_ms(inflight)
                    if (
                        launch_context == "pre_switch"
                        and self._ann_result_gpu_pending(task["result"])
                    ):
                        batch_rows = self._ann_task_num_queries(task)
                        self.rag_ann_pre_switch_launch_batches = (
                            getattr(
                                self,
                                "rag_ann_pre_switch_launch_batches",
                                0,
                            )
                            + 1
                        )
                        self.rag_ann_pre_switch_launch_rows = (
                            getattr(
                                self,
                                "rag_ann_pre_switch_launch_rows",
                                0,
                            )
                            + batch_rows
                        )
                        self.rag_ann_pre_switch_peak_projected_gpu_ms = max(
                            getattr(
                                self,
                                "rag_ann_pre_switch_peak_projected_gpu_ms",
                                0.0,
                            ),
                            queued_gpu_ms,
                        )
                        configured_credit_ms = max(
                            0.0,
                            getattr(self, "rag_ann_max_queued_gpu_ms", 0.0),
                        )
                        if (
                            configured_credit_ms > 0.0
                            and queued_gpu_ms > configured_credit_ms + 1e-6
                        ):
                            self.rag_ann_pre_switch_safety_violations = (
                                getattr(
                                    self,
                                    "rag_ann_pre_switch_safety_violations",
                                    0,
                                )
                                + 1
                            )
                    self.rag_ann_credit_peak_queued_ms = max(
                        getattr(self, "rag_ann_credit_peak_queued_ms", 0.0),
                        queued_gpu_ms,
                    )
                    if task.get("credit_forced_atomic", False):
                        self.rag_ann_credit_peak_forced_atomic_ms = max(
                            getattr(
                                self,
                                "rag_ann_credit_peak_forced_atomic_ms",
                                0.0,
                            ),
                            queued_gpu_ms,
                        )
                    else:
                        self.rag_ann_credit_peak_admitted_queued_ms = max(
                            getattr(
                                self,
                                "rag_ann_credit_peak_admitted_queued_ms",
                                0.0,
                            ),
                            queued_gpu_ms,
                        )
                    self.rag_ann_inflight_peak = max(
                        getattr(self, "rag_ann_inflight_peak", 0),
                        len(inflight),
                    )
                    self.rag_frontend_ann_launched = True
                    group_key = str(
                        ann_stream_idx if ann_stream_idx is not None else "primary"
                    )
                    self.rag_ann_stream_group_launch_counts[group_key] = (
                        self.rag_ann_stream_group_launch_counts.get(group_key, 0) + 1
                    )
            except Exception as e:
                logger.exception("ANN side-channel launch failed.")
                for subtask in task.get("tasks") or [task]:
                    self._send_ann_search_output(
                        AnnSearchReqOutput(
                            success=False,
                            message=str(e),
                            request_id=subtask.get("request_id"),
                        )
                    )

        if granted_batches > 0:
            self.rag_ann_batches_granted_this_tick = (
                granted_this_tick + granted_batches
            )
        if (
            pending
            and tick_limit_applied
            and len(inflight) >= max_inflight
            and not getattr(
                self, "rag_ann_tick_limit_held_this_tick", False
            )
        ):
            self.rag_ann_tick_limit_holds = (
                getattr(self, "rag_ann_tick_limit_holds", 0) + 1
            )
            self.rag_ann_tick_limit_held_this_tick = True

        self.ann_search_inflight = inflight

    def _decode_batch_active(self) -> bool:
        return (
            self.running_batch is not None
            and not self.running_batch.is_empty()
        )

    def _maybe_run_background_synthetic_ann(self, stream_group):
        """Launch QPS-paced synthetic ANN independently of request prefill.

        This is a stress path for co-location experiments. It deliberately
        decouples ANN from the single-request RAG state machine so ANN kernels
        can overlap with an already-running decode batch.
        """
        synthetic_ann = getattr(self, "synthetic_ann", None)
        schedule = getattr(self, "synthetic_ann_schedule", "prefill")
        decode_active = self._decode_batch_active()
        if (
            synthetic_ann is None
            or not synthetic_ann.active
            or schedule not in ("background-primary", "background-frontend")
        ):
            return

        start_delay_s = getattr(self, "synthetic_ann_start_delay_s", 0.0)
        start_time = getattr(self, "synthetic_ann_start_time", 0.0)
        if start_delay_s > 0 and time.monotonic() - start_time < start_delay_s:
            return

        # Keep the stress baseline meaningful: background ANN should be a
        # co-running workload, not an idle-time GPU burner before requests enter
        # decode. Strict overlap is still verified with Nsight.
        if not decode_active:
            return

        if not synthetic_ann.should_run():
            return

        ann_stream = getattr(self, "synthetic_ann_stream", None)
        if ann_stream is None:
            ann_stream = stream_group[0]

        synthetic_ann.run(
            ann_stream,
            decode_overlap_hint=decode_active,
            schedule=schedule,
        )

    def get_target_stream_group_idx(self) -> int:
        plan = getattr(self, "rag_frontend_plan", None)
        if plan is not None:
            return plan.target_stream_idx

        running = self.running_batch is not None and not self.running_batch.is_empty()
        split_batch = self.split_prefill_batch
        split_requires_frontend = bool(split_batch) and (
            getattr(self, "rag_prefill_run_allowed", True)
            or getattr(split_batch, "split_prefill_finished", False)
        )
        frontend_work = (
            split_requires_frontend or self._rag_ann_frontend_partition_needed()
        )
        if running and frontend_work:
            decode_bs = self.running_batch.batch_size()
            return self._adaptive_frontend_stream_idx(decode_bs)
        if running:
            return self.real_sm_group_num - 1
        return 0


    def _rag_prefill_can_submit_first(self, plan, original_idx, stream_idx, waiting):
        enabled = getattr(self, "_rag_prefill_submit_first_enabled", None)
        if enabled is None:
            enabled = os.getenv("RAGLEASE_PREFILL_SUBMIT_FIRST", "0") == "1"
            self._rag_prefill_submit_first_enabled = enabled
        batch = getattr(self, "split_prefill_batch", None)
        return bool(
            enabled
            and getattr(self, "rag_policy_mode", "full") == "full"
            and plan is not None and plan.allow_prefill
            and not getattr(plan, "prefill_sprint", False)
            and original_idx == stream_idx == plan.target_stream_idx
            and 1 <= stream_idx < self.real_sm_group_num - 1
            and getattr(self, "rag_partition_transition_intent", None) is None
            and batch is not None and not batch.is_empty()
            and 0 < batch.split_index < self.model_config.num_hidden_layers
            and not getattr(batch, "split_prefill_finished", False)
            and not waiting
            and getattr(self, "rag_prefill_bundle_inflight_event", None) is None
            and self._decode_batch_active()
            and not self._rag_ann_gpu_work_active()
            and not self._rag_ann_blocks_prefill())

    def _rag_prefill_bundle_forward_count(self, frontend_plan) -> int:
        """Return the planned Prefill lease size without widening a sprint."""
        if getattr(frontend_plan, 'joint_event_id', None) is not None:
            if (not joint_plan_ready_for_dispatch(self, frontend_plan)
                    or not frontend_plan.allow_prefill):
                return 0
            if frontend_plan.prefill_forward_count is None:
                raise ValueError('Activated joint lease requires a fitted prefill limit')
            contract = getattr(frontend_plan, 'prefill_contract', None)
            if (contract is None or contract != capture_prepared_prefill_contract(
                    getattr(self, 'split_prefill_batch', None))):
                return 0  # Preserve the prepared batch for the next real decision.
        planned_layers = (
            getattr(frontend_plan, "prefill_forward_count", None)
            if frontend_plan is not None
            else None
        )
        if planned_layers is not None:
            if type(planned_layers) is not int or planned_layers < 0:
                raise ValueError('Planned prefill layers must be a nonnegative integer')
            return planned_layers

        split_batch = self.split_prefill_batch
        if split_batch.extend_num_tokens > 0:
            return max(
                1,
                self.pdmux_config.split_forward_token_budget
                // split_batch.extend_num_tokens,
            )
        return self.model_config.num_hidden_layers

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self, stream_idx: Optional[int] = None
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if stream_idx is None:
            stream_idx = self.get_target_stream_group_idx()

        set_current_stream_idx(stream_idx)
        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False
        if getattr(self,'rag_joint_event_input_provider',None) is not None:
            # This runs before the first joint intent exists. Queued prefill
            # must remain unallocated until selected-cohort materialization.
            return False
        plan = getattr(self, 'rag_frontend_plan', None)
        if getattr(plan, 'joint_event_id', None) is not None:
            # Joint materialization is explicit and cohort-bound, BEFORE final
            # activation. This legacy path must never choose another batch,
            # even after an activated batch completes within the same tick.
            return False
        if (
            getattr(self, "rag_ann_frontend_arbitration", "overlap") == "strict"
            and self._rag_ann_blocks_prefill()
        ):
            return False

        if self._should_wait_for_rag_prefill_cohort():
            return False

        # add new request
        batch = self.get_new_batch_prefill()
        return self._install_rag_split_prefill_batch(batch)

    def _install_rag_split_prefill_batch(self, batch) -> bool:
        """Shared installation/bookkeeping for legacy and admitted cohorts."""
        if batch and not batch.is_empty():
            # Initialize at the real installation boundary, before any bundle
            # submission. Older ScheduleBatch builds lack this host field.
            # Never infer completion from a nonzero submission cursor.
            if getattr(batch, 'rag_completed_prefill_layers', None) is None:
                if getattr(batch, 'split_index', None) != 0:
                    raise ValueError('Missing completion receipt for started prefill batch')
                batch.rag_completed_prefill_layers = 0
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            handoff_reqs = [
                req
                for req in batch.reqs
                if getattr(req, "rag_ann_handoff_ready", False)
            ]
            self.rag_prefill_handoff_active = bool(handoff_reqs)
            self.rag_prefill_exclusive_sprint_available = (
                self.rag_prefill_handoff_active
            )
            if self.rag_prefill_handoff_active:
                self.rag_prefill_cohort_releases = (
                    getattr(self, "rag_prefill_cohort_releases", 0) + 1
                )
                self._increment_histogram(
                    self.rag_prefill_cohort_size_histogram,
                    len(handoff_reqs),
                )
                batch_started_at = time.perf_counter()
                self.rag_prefill_handoff_batch_started_ts = batch_started_at
                self.rag_prefill_handoff_gpu_events = []
                self.rag_ann_handoff_prefill_activations += 1
                for req in handoff_reqs:
                    scheduler_arrival_at = getattr(
                        req, "rag_scheduler_arrival_at", None
                    )
                    if scheduler_arrival_at is not None:
                        self.rag_prefill_queue_wait_ms.append(
                            (batch_started_at - scheduler_arrival_at) * 1000.0
                        )
                    req.rag_prefill_handoff_profiled = True
                    req.rag_prefill_started_at = batch_started_at
                    req.rag_ann_handoff_ready = False
            self.split_prefill_batch = batch
            return True
        return False

    def _rag_undrained_streams(self, decode_stream):
        """Streams that can still be executing when the CPU reaches a free.

        decode_stream is excluded by identity -- the caller already drains it.
        Returns whatever the fork actually has: the RAG ANN streams, plus every
        stream in the pd-mux stream groups.  Nothing here is assumed to exist.
        """
        out = []
        seen = {id(decode_stream)}
        for name in ("rag_ann_stream", "rag_ann_epoch_stream"):
            stream = getattr(self, name, None)
            if isinstance(stream, torch.cuda.Stream) and id(stream) not in seen:
                seen.add(id(stream))
                out.append((name, stream))
        for idx, group in enumerate(getattr(self, "stream_groups", None) or []):
            for pos, stream in enumerate(group):
                if isinstance(stream, torch.cuda.Stream) and id(stream) not in seen:
                    seen.add(id(stream))
                    out.append(("stream_group_%d_%d" % (idx, pos), stream))
        return out

    def _rag_note_drain(self, name, ms):
        """One timing sample per drain point, plus a periodic summary.

        A barrier that never blocks is not evidence; a barrier that always
        blocks hard is.  This is what tells the two apart in the log.
        """
        if not hasattr(self, "rag_drain_samples"):
            self.rag_drain_samples = {}
            self.rag_drain_counts = {}
            self.rag_drain_reported = 0
        samples = self.rag_drain_samples.setdefault(name, deque(maxlen=20000))
        samples.append(ms)
        self.rag_drain_counts[name] = self.rag_drain_counts.get(name, 0) + 1
        total = sum(self.rag_drain_counts.values())
        if total - self.rag_drain_reported >= 500:
            self.rag_drain_reported = total
            parts = []
            for key in sorted(self.rag_drain_samples):
                ordered = sorted(self.rag_drain_samples[key])
                parts.append(
                    "%s n=%d p50=%.3fms p99=%.3fms max=%.3fms"
                    % (
                        key,
                        len(ordered),
                        ordered[len(ordered) // 2],
                        ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
                        ordered[-1],
                    )
                )
            print("RAGLEASE_DRAIN_STATS " + " | ".join(parts), flush=True)

    @torch.inference_mode()
    def event_loop_pdmux(self):
        """A scheduler loop for pd multiplexing."""
        from sglang.srt.multiplex.joint_bootstrap import bootstrap_engineering_joint_runtime
        bootstrap_engineering_joint_runtime(self)
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        self._rag_dm2_window = deque(maxlen=50)
        self._rag_dm2_by_stream = {}
        self._rag_iteration_overhead_window = deque(maxlen=50)
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        if getattr(self, '_metadata_handoffs', None) is None:
            self._metadata_handoffs = [
                (MetadataHandoff(g[0], g[1]), MetadataHandoff(g[1], g[0]))
                for g in self.stream_groups
            ]
        p_to_d, d_to_p = self._metadata_handoffs[stream_idx]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        # Periodic synthetic ANN stats logging. Use wall-clock time because the
        # scheduler loop can spin very fast while ANN is gated during warmup.
        _ann_stats_interval_s = 10.0
        _ann_last_stats_ts = time.monotonic()
        _ann_last_logged_kernels = -1
        decode_gpu_started = torch.cuda.Event(enable_timing=True)
        decode_gpu_finished = torch.cuda.Event(enable_timing=True)
        # Single device-clock origin so intervals stamped on any stream in this
        # loop can be intersected directly.  Never compared across devices.
        rag_gpu_epoch = torch.cuda.Event(enable_timing=True)
        rag_gpu_epoch.record(decode_stream)
        self.rag_gpu_epoch = rag_gpu_epoch

        while True:
            # DIAGNOSTIC (v66): WAR barrier for schedule-side writes to buffers
            # the previous forward may still be reading -- `req_to_token_pool` in
            # particular.  Mirrors upstream sglang PR #26380.  See the module
            # docstring.  No host synchronisation: these are stream
            # dependencies only, and each covers only work enqueued BEFORE it,
            # so this iteration's own prefill/decode still overlap.
            if os.environ.get('RAGLEASE_WAR_BARRIER') == '1':
                _rag_cur = torch.cuda.current_stream()
                _rag_cur.wait_stream(decode_stream)
                _rag_cur.wait_stream(prefill_stream)
                decode_stream.wait_stream(prefill_stream)
                # v67: v66 was ASYMMETRIC -- prefill never waited for decode, so
                # a schedule-side write issued while the prefill stream was
                # current could still land while decode's kernels were reading
                # the same buffer.  Close that direction too.  Still a WAR
                # barrier: each wait covers only work enqueued BEFORE it.
                prefill_stream.wait_stream(decode_stream)
                if not getattr(self, '_rag_war_announced', False):
                    self._rag_war_announced = True
                    print('RAGLEASE_WAR_BARRIER_ACTIVE', flush=True)
            scheduler_tick_started_at = time.perf_counter()
            self._poll_rag_prefill_stage_completions()
            # A previous tick's joint lease is not launch authorization until
            # this tick has processed inputs and revalidated its signature.
            # Preserve the marker only as proof of the prior activation; the
            # planner may republish it after the current safety checks.
            self.rag_joint_previous_active_event_id = getattr(
                self, "rag_joint_active_event_id", None
            )
            self.rag_joint_active_event_id = None
            if (
                self.synthetic_ann is not None
                and self.synthetic_ann.active
                and time.monotonic() - _ann_last_stats_ts >= _ann_stats_interval_s
            ):
                kernel_count = self.synthetic_ann.kernel_count
                if kernel_count != _ann_last_logged_kernels:
                    self.synthetic_ann.log_stats()
                    _ann_last_logged_kernels = kernel_count
                _ann_last_stats_ts = time.monotonic()

            decode_launch_context = None
            with torch.cuda.stream(decode_stream):
                # DEP 2/3 (P->D): prefill metadata (preparation and commit) must
                # have been submitted before decode reads anything derived from
                # it.  One wait per published version, however many entry points
                # call acquire().
                p_to_d.acquire()
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            # DEP 1 (D->P): decode's recycling -- process_batch_result ->
            # free_group_end -> free -> torch.cat writing the next free_pages --
            # is submitted above, on decode_stream.  Publish it here, before the
            # prefill branch, so prefill cannot read a free_pages whose cat is
            # still in flight.
            d_to_p.publish()
            with torch.cuda.stream(prefill_stream):
                d_to_p.acquire()
                set_pdmux_status(True)

                self._poll_rag_prefill_bundle_completion()
                self._poll_ann_at_scheduler_tick_start()
                if (
                    not adjust_stream_group
                    and getattr(self, "rag_ann_launch_site", "early") == "early"
                    and getattr(
                        self, "rag_ann_frontend_arbitration", "overlap"
                    )
                    != "fair"
                ):
                    self._launch_ann_search_side_channel()

                sm_count = self.sm_counts[stream_idx][0]
                if (
                    not wait_prefill_kernel_done
                    and getattr(self, 'rag_prefill_merge_pending_batch', None)
                    is None
                ):
                    # DEP 2 (P->D).  Publish only if this iteration actually
                    # enters the preparation path.  update_split_prefill_batch()
                    # returns immediately when a split-prefill is already in
                    # flight; publishing unconditionally would place the event
                    # behind that in-flight forward and make decode wait for a
                    # whole prefill.  Keyed on "entered the path", not on the
                    # return value: a preparation can evict or return pages and
                    # still fail to build a batch.
                    _prep_entered = self.split_prefill_batch is None
                    if _prep_entered:
                        d_to_p.acquire()
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )
                    if _prep_entered:
                        p_to_d.publish()

            with torch.cuda.stream(decode_stream):
                # DEP 2/3 (P->D): prefill metadata (preparation and commit) must
                # have been submitted before decode reads anything derived from
                # it.  One wait per published version, however many entry points
                # call acquire().
                p_to_d.acquire()
                set_pdmux_status(False)
                if getattr(
                    self,
                    "rag_decode_update_deferred_for_prefill_sprint",
                    False,
                ):
                    self.rag_joint_prepared_decode = None
                    self.rag_decode_update_deferred_for_prefill_sprint = False
                    self.rag_prefill_exclusive_decode_update_skips += 1
                else:
                    self.running_batch = self._update_rag_running_batch_for_decision(
                        self.running_batch
                    )
                decode_ready_ts = (
                    time.perf_counter() if self._decode_batch_active() else None
                )
                planned_from_stream_idx = stream_idx
                frontend_decision_started_at = time.perf_counter()
                self._plan_rag_frontend_actions()
                trace_plan = getattr(self, "rag_frontend_plan", None)
                self._trace_rag_prefill_event("plan_selected",
                    allow_prefill=getattr(trace_plan, "allow_prefill", None),
                    reason=getattr(trace_plan, "reason", None),
                    target_reason=getattr(trace_plan, "target_reason", None),
                    target_partition=getattr(trace_plan, "target_stream_idx", None),
                    layers=getattr(trace_plan, "prefill_forward_count", None),
                    inflight=getattr(self, "rag_prefill_bundle_inflight_event", None) is not None)
                self.rag_frontend_decision_ms.append(
                    (time.perf_counter() - frontend_decision_started_at)
                    * 1000.0
                )
                target_stream_idx = self.get_target_stream_group_idx()
                adjust_stream_group = (
                    adjust_stream_group or target_stream_idx != stream_idx
                )
                if target_stream_idx != stream_idx:
                    self._track_rag_partition_transition_intent(
                        stream_idx, target_stream_idx
                    )
                else:
                    self._cancel_rag_partition_transition_intent()
                if (
                    self.running_batch.is_empty()
                    and self.split_prefill_batch is None
                    and not self._ann_side_channel_has_work()
                ):
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            pre_switch_ann_launched = False
            if adjust_stream_group:
                pre_switch_ann_launched = (
                    self._maybe_launch_ann_before_partition_switch(
                        self.get_target_stream_group_idx()
                    )
                )

            if adjust_stream_group:
                target_stream_idx = self.get_target_stream_group_idx()
                switch_blocked = False
                if target_stream_idx == stream_idx:
                    self.rag_ann_partition_switch_deferred = False
                    adjust_stream_group = False
                else:
                    was_switch_blocked = getattr(
                        self, "rag_ann_partition_switch_deferred", False
                    )
                    switch_blocked = (
                        self._rag_ann_blocks_stream_group_switch(
                            target_stream_idx
                        )
                    )
                    if pre_switch_ann_launched and switch_blocked:
                        self.rag_ann_pre_switch_switch_deferrals = (
                            getattr(
                                self,
                                "rag_ann_pre_switch_switch_deferrals",
                                0,
                            )
                            + 1
                        )
                    # The ANN event may have become ready after the regular
                    # poll at the start of this tick. Harvest it before the
                    # costly stream synchronizations/partition transition.
                    self._poll_ann_before_partition_switch_if_ready(
                        was_switch_blocked=was_switch_blocked,
                        switch_blocked=switch_blocked,
                        pre_switch_ann_launched=pre_switch_ann_launched,
                    )
                if target_stream_idx != stream_idx and not switch_blocked:
                    previous_stream_idx = stream_idx
                    switch_started = time.perf_counter()
                    with _nvtx_range("pdmux_adjust_stream_group"):
                        prefill_stream.synchronize()
                        decode_stream.synchronize()
                        # This is already a scheduler-owned GPU safe point.
                        # Consume an ANN-drain/native-map completion before
                        # the potentially long stream-group transition hides
                        # it for another loop iteration.
                        self._progress_shared_hbm_kv_map()
                        stream_idx, stream_group = self.adjust_stream_groups(
                            target_stream_idx
                        )
                        switch_finished = time.perf_counter()
                        self.pdmux_stream_group_switches += 1
                        self.rag_partition_last_switch_ts = switch_finished
                        switch_latency_ms = (
                            switch_finished - switch_started
                        ) * 1000.0
                        self.rag_partition_switch_latency_ms.append(switch_latency_ms)
                        pair_history = self.rag_partition_switch_pair_latency_ms.setdefault(
                            (previous_stream_idx, target_stream_idx),
                            deque(maxlen=256),
                        )
                        pair_history.append(switch_latency_ms)
                        self._complete_rag_partition_transition(
                            previous_stream_idx,
                            target_stream_idx,
                            switch_finished,
                            switch_latency_ms,
                        )
                        switch_reason = getattr(
                            getattr(self, "rag_frontend_plan", None),
                            "target_reason",
                            "unplanned",
                        )
                        self._increment_histogram(
                            self.rag_partition_switch_reason_counts,
                            switch_reason,
                        )
                        prefill_stream = stream_group[0]
                        decode_stream = stream_group[1]
                        # Events are context-bound and the pair changed, so
                        # the channels must follow the group.
                        p_to_d, d_to_p = self._metadata_handoffs[stream_idx]
                    adjust_stream_group = False
                    # The selected compute conversion has now completed.
                    # Finish the SAME lease before dispatch/Decode changes
                    # its captured frontier; waiting until the next tick
                    # would unnecessarily invalidate this ready intent.
                    with torch.cuda.stream(decode_stream):
                        set_pdmux_status(False)
                        self._resume_rag_joint_frontend_preparation()
                    if (
                        getattr(self, "rag_ann_launch_site", "early") == "early"
                        and getattr(
                            self, "rag_ann_frontend_arbitration", "overlap"
                        )
                        != "fair"
                    ):
                        # An early launch may have been deferred while the
                        # scheduler was in the decode-only group. Retry after
                        # entering the ANN-compatible mixed partition.
                        self._launch_ann_search_side_channel()
                    logger.debug(
                        "Adjusting stream groups: %s, prefill sm: %s, decode sm: %s",
                        stream_idx,
                        self.sm_counts[stream_idx][0],
                        self.sm_counts[stream_idx][1],
                    )

            frontend_plan = getattr(self, "rag_frontend_plan", None)
            prefill_sprint_active = bool(
                frontend_plan is not None
                and frontend_plan.allow_prefill
                and frontend_plan.prefill_sprint
                and stream_idx == frontend_plan.target_stream_idx
            )
            # A compute lease may keep the prefill-sprint capability after the
            # selected/queued prefill work has disappeared.  Capability alone
            # must not suppress a live decode batch forever: exclusivity is
            # meaningful only while a real prefill batch or its submitted GPU
            # bundle still exists.  The scheduler still preserves the intended
            # exclusive sprint while that work is runnable or in flight.
            prefill_exclusive_can_submit = bool(
                self.split_prefill_batch is not None
                and not self.split_prefill_batch.is_empty()
                and not getattr(
                    self.split_prefill_batch,
                    "split_prefill_finished",
                    False,
                )
                and not wait_prefill_kernel_done
                and getattr(self, "rag_prefill_bundle_inflight_event", None)
                is None
                and not self._rag_ann_blocks_prefill()
                and self._rag_prefill_bundle_forward_count(frontend_plan) > 0
            )
            prefill_exclusive_work = bool(
                wait_prefill_kernel_done
                or getattr(self, "rag_prefill_bundle_inflight_event", None)
                is not None
                or prefill_exclusive_can_submit
            )
            prefill_exclusive_active = bool(
                prefill_sprint_active
                and stream_idx == 0
                and prefill_exclusive_work
            )

            launch_ann_early = bool(
                self._rag_ann_should_launch_early()
                and getattr(self, "rag_ann_frontend_arbitration", "overlap")
                == "fair"
            )
            ann_yields_to_prefill = bool(
                launch_ann_early
                and self._rag_ann_yields_to_prefill_transition(stream_idx)
            )
            if ann_yields_to_prefill:
                if getattr(self, "ann_search_pending", None):
                    self.rag_ann_prefill_transition_holds = (
                        getattr(
                            self,
                            "rag_ann_prefill_transition_holds",
                            0,
                        )
                        + 1
                    )
            elif launch_ann_early:
                self._launch_ann_search_side_channel()

            _prefill_submit_first = self._rag_prefill_can_submit_first(
                frontend_plan, planned_from_stream_idx, stream_idx, wait_prefill_kernel_done)
            if _prefill_submit_first:
                self.rag_prefill_submit_first_count = getattr(self, "rag_prefill_submit_first_count", 0) + 1
            for _submit_prefill in ((True, False) if _prefill_submit_first else (False, True)):
                if _submit_prefill:
                    with torch.cuda.stream(prefill_stream):
                        set_pdmux_status(True)
                        if (
                            self.split_prefill_batch
                            and not self.split_prefill_batch.is_empty()
                            and not wait_prefill_kernel_done
                            and getattr(
                                self, "rag_prefill_bundle_inflight_event", None
                            )
                            is None
                            and not self._rag_ann_blocks_prefill()
                        ):
                            forward_count = self._rag_prefill_bundle_forward_count(
                                frontend_plan
                            )
                            next_split_index = min(
                                self.split_prefill_batch.split_index + forward_count,
                                self.model_config.num_hidden_layers,
                            )
                            forward_count = (
                                next_split_index - self.split_prefill_batch.split_index
                            )
                            if forward_count <= 0:
                                # Zero is a real admission decision. Skip only
                                # this prefill turn; the paired decode turn in
                                # the submission loop must still execute.
                                continue
                            prefill_done = True

                            self.split_prefill_batch.split_forward_count = forward_count
                            self._increment_histogram(
                                self.rag_prefill_stream_group_run_counts, stream_idx
                            )
                            prefill_gpu_event_pair = None
                            dynamic_bundle = getattr(
                                self, "rag_prefill_dynamic_lease", False
                            )
                            if self.rag_prefill_handoff_active or dynamic_bundle:
                                gpu_started = torch.cuda.Event(enable_timing=True)
                                gpu_finished = torch.cuda.Event(enable_timing=True)
                                gpu_started.record(prefill_stream)
                                prefill_gpu_event_pair = (gpu_started, gpu_finished)
                            prefill_submit_started_at = time.perf_counter()
                            self._trace_rag_prefill_event("bundle_submit_begin",
                                layers=forward_count, partition=stream_idx)
                            with _nvtx_range("pdmux_prefill_run_batch"):
                                prefill_result = self.run_batch(self.split_prefill_batch)
                            self._trace_rag_prefill_event("bundle_submit_return",
                                layers=forward_count, partition=stream_idx)
                            self.rag_prefill_host_submit_ms.append(
                                (time.perf_counter() - prefill_submit_started_at) * 1000.0
                            )
                            if prefill_gpu_event_pair is not None:
                                prefill_gpu_event_pair[1].record(prefill_stream)
                                if self.rag_prefill_handoff_active:
                                    self.rag_prefill_handoff_gpu_events.append(
                                        prefill_gpu_event_pair
                                    )
                                if dynamic_bundle:
                                    predicted_layer_ms = (
                                        self._predict_rag_prefill_layer_ms(
                                            stream_idx,
                                            max(
                                                1,
                                                int(
                                                    self.split_prefill_batch.extend_num_tokens
                                                ),
                                            ),
                                        )
                                    )
                                    self.rag_prefill_bundle_inflight_event = (
                                        prefill_gpu_event_pair[1]
                                    )
                                    self.rag_prefill_bundle_inflight_meta = {
                                        "submitted_at": prefill_submit_started_at,
                                        "batch": self.split_prefill_batch,
                                        "end_layer": next_split_index,
                                        "start_event": prefill_gpu_event_pair[0],
                                        "layers": forward_count,
                                        "tokens": max(
                                            1,
                                            int(
                                                self.split_prefill_batch.extend_num_tokens
                                            ),
                                        ),
                                        "stream_idx": stream_idx,
                                        "predicted_bundle_ms": (
                                            predicted_layer_ms * forward_count
                                            if predicted_layer_ms is not None
                                            else None
                                        ),
                                    }
                            self.rag_frontend_prefill_ran = True
                            if next_split_index == self.model_config.num_hidden_layers:
                                self.split_prefill_batch.split_prefill_finished = True
                                prefill_exe_done = prefill_stream.record_event()
                            self.split_prefill_batch.split_index = next_split_index

                            # --- Synthetic ANN injection ---
                            # Run on prefill GC stream when QPS pacing allows.
                            # Uses CUDA events; decode is never globally synchronized.
                            if (
                                self.synthetic_ann is not None
                                and getattr(self, "synthetic_ann_schedule", "prefill")
                                == "prefill"
                                and self.synthetic_ann.should_run()
                            ):
                                self.synthetic_ann.run(
                                    prefill_stream,
                                    decode_overlap_hint=self._decode_batch_active(),
                                    schedule="prefill",
                                )

                        elif wait_prefill_kernel_done:
                            prefill_done = True
                        else:
                            prefill_done = False
                else:
                    with torch.cuda.stream(decode_stream):
                        set_pdmux_status(False)
                        # process decode batch
                        self._maybe_run_background_synthetic_ann(stream_group)
                        if (
                            self.running_batch
                            and not self.running_batch.is_empty()
                            and not prefill_exclusive_active
                        ):
                            self._drain_fenced_ann_before_decode(decode_stream)
                            launch_batch_size = int(self.running_batch.batch_size())
                            ann_overlap = self._rag_ann_gpu_work_active()
                            prefill_overlap = bool(
                                self.split_prefill_batch
                                and not self.split_prefill_batch.is_empty()
                                and not wait_prefill_kernel_done
                                and getattr(
                                    self, "rag_prefill_bundle_inflight_event", None
                                )
                                is None
                                and not self._rag_ann_blocks_prefill()
                            )
                            prefill_overlap = prefill_overlap or (_prefill_submit_first and prefill_done)
                            overlap_class = self._rag_decode_overlap_class(
                                ann_overlap=ann_overlap,
                                prefill_overlap=prefill_overlap,
                            )
                            predicted_decode_gpu_ms = (
                                self._rag_ann_decode_prediction_ms(
                                    stream_idx,
                                    decode_bs=launch_batch_size,
                                    overlap_class=overlap_class,
                                )
                            )
                            decode_prediction_components = (
                                dict(getattr(
                                    self, "_rag_last_decode_prediction_components", {}
                                ))
                                if os.environ.get("RAGLEASE_DECODE_PRED_DIAG") == "1"
                                else None
                            )
                            predicted_decode_iteration_ms = (
                                self._rag_ann_decode_prediction_ms(
                                    stream_idx,
                                    decode_bs=launch_batch_size,
                                    include_scheduling=True,
                                    overlap_class=overlap_class,
                                )
                            )
                            decode_launch_context = {
                                "stream_idx": stream_idx,
                                "batch_size": launch_batch_size,
                                "overlap_class": overlap_class,
                                "predicted_ms": predicted_decode_gpu_ms,
                                "prediction_components": decode_prediction_components,
                                "predicted_iteration_ms": (
                                    predicted_decode_iteration_ms
                                ),
                                "slo_ms": self._rag_ann_decode_slo_ms(stream_idx),
                                "cached_tokens": tuple(
                                    len(req.origin_input_ids)
                                    + len(req.output_ids) - 1
                                    for req in (
                                        getattr(self.running_batch, "reqs", None)
                                        or ()
                                    )
                                ),
                            }
                            self._record_rag_decode_queue_delay(decode_ready_ts)
                            self._increment_histogram(
                                self.rag_decode_batch_size_histogram,
                                self.running_batch.batch_size(),
                            )
                            decode_gpu_started.record(decode_stream)
                            decode_submit_started_at = time.perf_counter()
                            decode_launch_context['submitted_at'] = decode_submit_started_at
                            last_decode_submit_at = getattr(
                                self, "rag_last_decode_submit_ts", None
                            )
                            if last_decode_submit_at is not None:
                                self.rag_decode_launch_interval_ms.append(
                                    (decode_submit_started_at - last_decode_submit_at)
                                    * 1000.0
                                )
                            self.rag_last_decode_submit_ts = decode_submit_started_at
                            # EXPERIMENT A (v54): hold the metadata tensors
                            # alive until AFTER the pre-existing drain.  No new
                            # sync, no device assert, no read of any tensor's
                            # contents -- only the Python reference lifetime
                            # changes.  The point: decode_stream.synchronize()
                            # runs AFTER the enqueue, so it cannot protect T's
                            # storage between the enqueue and the sync.
                            if os.environ.get('RAGLEASE_KEEPALIVE') == '1':
                                _rag_keep = []
                                for _rag_nm in (
                                    'req_pool_indices', 'seq_lens', 'orig_seq_lens',
                                    'input_ids', 'out_cache_loc', 'encoder_lens',
                                    'encoder_out_cache_loc',
                                ):
                                    _rag_t = getattr(self.running_batch, _rag_nm, None)
                                    if isinstance(_rag_t, torch.Tensor):
                                        _rag_keep.append(_rag_t)
                                self._rag_keep = _rag_keep
                                if not getattr(self, '_rag_keep_announced', False):
                                    self._rag_keep_announced = True
                                    print(
                                        'RAGLEASE_KEEPALIVE_ACTIVE n=%d'
                                        % len(_rag_keep),
                                        flush=True,
                                    )
                            with _nvtx_range("pdmux_decode_run_batch"):
                                self.rag_joint_prepared_decode = None
                                if not hasattr(self, "rag_decode_run_batch_observations"):
                                    self.rag_decode_run_batch_observations = deque(maxlen=4096)
                                decode_result = self.run_batch(self.running_batch)
                            self.rag_decode_host_submit_ms.append(
                                (time.perf_counter() - decode_submit_started_at) * 1000.0
                            )
                            decode_gpu_finished.record(decode_stream)
                            decode_done = True
                        else:
                            decode_done = False
                            if (
                                prefill_exclusive_active
                                and self.running_batch
                                and not self.running_batch.is_empty()
                            ):
                                self.rag_prefill_exclusive_skip_diagnostics = (
                                    getattr(
                                        self,
                                        "rag_prefill_exclusive_skip_diagnostics",
                                        0,
                                    )
                                    + 1
                                )
                                diagnostic_count = (
                                    self.rag_prefill_exclusive_skip_diagnostics
                                )
                                if diagnostic_count <= 3 or diagnostic_count % 5000 == 0:
                                    diagnostic_batch = self.split_prefill_batch
                                    print(
                                        "RAGLEASE_PREFILL_EXCLUSIVE_SKIP "
                                        "n=%d wait=%s inflight=%s can_submit=%s "
                                        "batch=%s empty=%s finished=%s split=%s ann_block=%s"
                                        % (
                                            diagnostic_count,
                                            wait_prefill_kernel_done,
                                            getattr(
                                                self,
                                                "rag_prefill_bundle_inflight_event",
                                                None,
                                            )
                                            is not None,
                                            prefill_exclusive_can_submit,
                                            diagnostic_batch is not None,
                                            (
                                                diagnostic_batch.is_empty()
                                                if diagnostic_batch is not None
                                                else None
                                            ),
                                            getattr(
                                                diagnostic_batch,
                                                "split_prefill_finished",
                                                None,
                                            ),
                                            getattr(
                                                diagnostic_batch,
                                                "split_index",
                                                None,
                                            ),
                                            self._rag_ann_blocks_prefill(),
                                        ),
                                        flush=True,
                                    )
                                self.rag_decode_update_deferred_for_prefill_sprint = (
                                    True
                                )

            if (
                not adjust_stream_group
                and self._rag_ann_should_launch_after_prefill()
            ):
                self._launch_ann_search_side_channel()

            diagnostic_prefill = self.split_prefill_batch
            if (
                diagnostic_prefill is not None
                and not diagnostic_prefill.is_empty()
                and not self.rag_frontend_prefill_ran
            ):
                self.rag_prefill_no_submit_diagnostics = (
                    getattr(self, "rag_prefill_no_submit_diagnostics", 0) + 1
                )
                diagnostic_count = self.rag_prefill_no_submit_diagnostics
                if diagnostic_count <= 3 or diagnostic_count % 5000 == 0:
                    print(
                        "RAGLEASE_PREFILL_NO_SUBMIT "
                        "n=%d allow=%s sprint=%s event=%s reason=%s "
                        "stream=%s target=%s wait=%s inflight=%s "
                        "ann_block=%s forward=%s split=%s finished=%s"
                        % (
                            diagnostic_count,
                            getattr(frontend_plan, "allow_prefill", None),
                            getattr(frontend_plan, "prefill_sprint", None),
                            getattr(frontend_plan, "joint_event_id", None),
                            getattr(frontend_plan, "reason", None),
                            stream_idx,
                            getattr(frontend_plan, "target_stream_idx", None),
                            wait_prefill_kernel_done,
                            getattr(
                                self,
                                "rag_prefill_bundle_inflight_event",
                                None,
                            )
                            is not None,
                            self._rag_ann_blocks_prefill(),
                            self._rag_prefill_bundle_forward_count(frontend_plan),
                            getattr(diagnostic_prefill, "split_index", None),
                            getattr(
                                diagnostic_prefill,
                                "split_prefill_finished",
                                None,
                            ),
                        ),
                        flush=True,
                    )

            self._finalize_rag_frontend_actions()

            with torch.cuda.stream(decode_stream):
                # DEP 2/3 (P->D): prefill metadata (preparation and commit) must
                # have been submitted before decode reads anything derived from
                # it.  One wait per published version, however many entry points
                # call acquire().
                p_to_d.acquire()
                set_pdmux_status(False)
                # DIAGNOSTIC (v40): the decode_done -> free/reuse barrier ALREADY
                # EXISTS on this line.  decode_stream is drained before
                # process_batch_result, so the decode stream provably cannot be
                # the producer still writing a slot that is about to be freed.
                # What is added: a stopwatch (is this barrier binding at all?)
                # and RAGLEASE_STREAM_DRAIN, which widens the drain to streams
                # that are NOT covered here.
                _rag_drain_t0 = time.perf_counter()
                decode_stream.synchronize()
                self._rag_note_drain(
                    "decode", (time.perf_counter() - _rag_drain_t0) * 1000.0
                )
                # DIAGNOSTIC (v70): stamp the sequence number at a point we know
                # is good, so the dump can bound the crash window instead of
                # leaving it a guess.  Pure host call, no CUDA.
                try:
                    from sglang.srt import ragtrace as _rag_tr
                    if _rag_tr.A:
                        _rag_tr.checkpoint()
                except Exception:
                    pass
                # EXPERIMENT A (v54): only NOW is it safe to let go.
                if getattr(self, '_rag_keep', None) is not None:
                    self._rag_keep = None
                # DIAGNOSTIC (v44): read the allocator probe.  Legal HERE and
                # nowhere else -- this point is already synchronised, so the
                # .item() calls add no sync of their own.
                # DIAGNOSTIC (v59): arm the profiler only once the drain starts,
                # so the measured window that reaches the fault is unperturbed.
                if os.environ.get('RAGLEASE_PROFILE') == '1':
                    _rag_prof = getattr(self, '_rag_prof', None)
                    if _rag_prof is None and not getattr(
                        self, '_rag_prof_armed', False
                    ):
                        _rag_below = int(
                            os.environ.get('RAGLEASE_PROFILE_ARM_BELOW', '150')
                        )
                        _rag_rb = getattr(self, 'running_batch', None)
                        _rag_nreq = len(_rag_rb.reqs) if _rag_rb is not None else 0
                        # v60: `< _rag_below` alone also matches 0, which is
                        # true at process start.  v59 logged PROFILE_ARMED
                        # n_req=0, i.e. it armed before the workload existed and
                        # likely profiled the whole run -- which would change the
                        # regime that reaches the fault (the v48/v52 trap).
                        # Require a large batch to have been seen FIRST, so that
                        # "small" really means "drained".
                        if _rag_nreq > _rag_below:
                            self._rag_saw_big_batch = True
                        if (_rag_nreq < _rag_below
                                and getattr(self, '_rag_saw_big_batch', False)):
                            self._rag_prof_armed = True
                            try:
                                import torch.profiler as _tp
                                # v61: export DURING the drain, on a schedule.
                                # v59 exported from the crash handler and failed:
                                # after a device-side assert the CUDA context is
                                # dead and export_chrome_trace needs it.  With a
                                # schedule + on_trace_ready the export happens
                                # while the process is healthy, so the last file
                                # lands within one cycle of the crash.
                                _rag_out = os.environ.get('RAGLEASE_PROFILE_OUT')
                                if not _rag_out:
                                    _rag_out = os.path.join(
                                        os.environ.get('RAGLEASE_ROOT', '.'),
                                        'run', 'raglease_trace.json')
                                self._rag_trace_n = 0

                                def _rag_on_trace(_p):
                                    try:
                                        _n = self._rag_trace_n
                                        self._rag_trace_n = _n + 1
                                        _path = _rag_out.replace(
                                            '.json', '_%03d.json' % _n)
                                        _p.export_chrome_trace(_path)
                                        print('RAGLEASE_PROFILE_EXPORTED %s' % _path,
                                              flush=True)
                                    except Exception as _e:
                                        print('RAGLEASE_PROFILE_EXPORT_FAILED %r'
                                              % (_e,), flush=True)

                                _rag_prof = _tp.profile(
                                    activities=[
                                        _tp.ProfilerActivity.CPU,
                                        _tp.ProfilerActivity.CUDA,
                                    ],
                                    with_stack=True,
                                    record_shapes=False,
                                    schedule=_tp.schedule(
                                        wait=0, warmup=2, active=25, repeat=0),
                                    on_trace_ready=_rag_on_trace,
                                )
                                _rag_prof.__enter__()
                                self._rag_prof = _rag_prof
                                print(
                                    'RAGLEASE_PROFILE_ARMED n_req=%d' % _rag_nreq,
                                    flush=True,
                                )
                            except Exception as _rag_e:
                                self._rag_prof_armed = True
                                print('RAGLEASE_PROFILE_ARM_FAILED %r' % (_rag_e,),
                                      flush=True)
                # v63: the step error was swallowed by a bare `except: pass` in
                # v61, which is exactly why RAGLEASE_PROFILE_EXPORTED never
                # appeared -- I could not tell "never fired" from "fired and
                # failed".  Print it, once, and count the steps.
                _rag_prof = getattr(self, '_rag_prof', None)
                if _rag_prof is not None:
                    try:
                        _rag_prof.step()
                        self._rag_steps = getattr(self, '_rag_steps', 0) + 1
                        _rag_n = self._rag_steps
                        if _rag_n in (1, 10, 50) or _rag_n % 100 == 0:
                            print('RAGLEASE_PROFILE_STEPS %d exported=%d'
                                  % (_rag_n, getattr(self, '_rag_trace_n', 0)),
                                  flush=True)
                    except Exception as _rag_se:
                        if not getattr(self, '_rag_step_err', False):
                            self._rag_step_err = True
                            print('RAGLEASE_PROFILE_STEP_FAILED %r' % (_rag_se,),
                                  flush=True)
                _rag_pool = getattr(self, "req_to_token_pool", None)
                _rag_wc = getattr(_rag_pool, "rag_w_calls", None)
                if _rag_wc is not None:
                    # Safe here: this point is ALREADY synchronised, so these
                    # .item() calls add no sync.  v51 also read out here -- its
                    # mistake was throttling on a write COUNT (>= 50), which left
                    # a window the bad write fell into.  The trigger is now BAD
                    # GROWTH, so the reporting granularity is one iteration.
                    _rag_wb = int(_rag_pool.rag_w_bad.item())
                    if _rag_wb > getattr(self, "_rag_w_lastbad", 0):
                        self._rag_w_lastbad = _rag_wb
                        from sglang.srt.mem_cache import memory_pool as _mp_mod
                        print(
                            "RAGLEASE_REQ2TOK_PROBE writes=%d bad=%d bad_min=%d "
                            "bad_max=%d bad_prefix=%d bad_slots=%d cap=%s"
                            % (
                                int(_rag_wc.item()),
                                _rag_wb,
                                int(_rag_pool.rag_w_min.item()),
                                int(_rag_pool.rag_w_max.item()),
                                int(_rag_pool.rag_w_bad_pre.item()),
                                int(_rag_pool.rag_w_bad_slots.item()),
                                getattr(_mp_mod, "RAG_KV_CAP", None),
                            ),
                            flush=True,
                        )
                _rag_alc = getattr(self, "token_to_kv_pool_allocator", None)
                _rag_probe_calls = getattr(_rag_alc, "rag_alloc_calls", None)
                if _rag_probe_calls is not None:
                    _rag_n = int(_rag_probe_calls.item())
                    if _rag_n - getattr(self, "_rag_probe_last", 0) >= 50:
                        self._rag_probe_last = _rag_n
                        print(
                            "RAGLEASE_ALLOC_PROBE calls=%d min=%d max=%d cap=%d bad=%d"
                            % (
                                _rag_n,
                                int(_rag_alc.rag_alloc_min.item()),
                                int(_rag_alc.rag_alloc_max.item()),
                                int(_rag_alc.rag_slot_owned.shape[0]),
                                int(_rag_alc.rag_alloc_bad.item()),
                            ),
                            flush=True,
                        )
                _rag_drain_mode = os.environ.get("RAGLEASE_STREAM_DRAIN", "decode")
                if _rag_drain_mode in ("ann", "all"):
                    for _rag_name, _rag_stream in self._rag_undrained_streams(
                        decode_stream
                    ):
                        _rag_drain_t0 = time.perf_counter()
                        _rag_stream.synchronize()
                        self._rag_note_drain(
                            _rag_name, (time.perf_counter() - _rag_drain_t0) * 1000.0
                        )
                if _rag_drain_mode == "all":
                    _rag_drain_t0 = time.perf_counter()
                    torch.cuda.synchronize()
                    self._rag_note_drain(
                        "device", (time.perf_counter() - _rag_drain_t0) * 1000.0
                    )
                if decode_done:
                    if decode_launch_context is not None:
                        decode_t0_ms = float(
                            rag_gpu_epoch.elapsed_time(decode_gpu_started))
                        decode_t1_ms = float(
                            rag_gpu_epoch.elapsed_time(decode_gpu_finished))
                        decode_launch_context["decode_t0_ms"] = decode_t0_ms
                        decode_launch_context["decode_t1_ms"] = decode_t1_ms
                        decode_launch_context["coexec_prefill"] = (
                            self._rag_coexec_prefill(decode_t0_ms, decode_t1_ms))
                    self._record_rag_decode_submission_service(
                        decode_launch_context, time.perf_counter())
                self._poll_ann_after_decode_sync()
                self._progress_shared_hbm_kv_map()
                decode_gpu_ms = (
                    float(decode_gpu_started.elapsed_time(decode_gpu_finished))
                    if decode_done
                    else None
                )
                if decode_done:
                    first_decode_at = time.perf_counter()
                    for req in self.running_batch.reqs:
                        prefill_completed_at = getattr(
                            req, "rag_prefill_completed_at", None
                        )
                        already_recorded = getattr(
                            req, "rag_prefill_first_decode_recorded", False
                        )
                        if prefill_completed_at is not None and not already_recorded:
                            self.rag_prefill_to_first_decode_ms.append(
                                (first_decode_at - prefill_completed_at) * 1000.0
                            )
                            req.rag_prefill_first_decode_recorded = True
                self._record_rag_decode_completion(
                    decode_done,
                    decode_gpu_ms,
                    launch_context=decode_launch_context,
                )
                if decode_done:
                    with _nvtx_range("pdmux_process_decode_result"):
                        self.process_batch_result(self.running_batch, decode_result)
                    pending_prefill_batch = getattr(
                        self, "rag_prefill_merge_pending_batch", None
                    )
                    if pending_prefill_batch is not None:
                        merge_shape = self._capture_rag_prefill_host_stage_shape(
                            'deferred_merge', pending_prefill_batch)
                        merge_started_at = time.perf_counter()
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(pending_prefill_batch)
                        else:
                            self.running_batch = pending_prefill_batch
                        self.rag_prefill_merge_pending_batch = None
                        self._record_rag_prefill_host_stage(merge_shape, merge_started_at,
                            result_sequence=getattr(self, 'rag_prefill_merge_pending_result_sequence', None))
                        self.rag_prefill_merge_pending_result_sequence = None

            # DEP 1 (D->P): decode's recycling -- process_batch_result ->
            # free_group_end -> free -> torch.cat writing the next free_pages --
            # is submitted above, on decode_stream.  Publish it here, before the
            # prefill branch, so prefill cannot read a free_pages whose cat is
            # still in flight.
            d_to_p.publish()
            with torch.cuda.stream(prefill_stream):
                d_to_p.acquire()
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    # The collective can outlast a native VMM stage. Publish
                    # it before result processing extends the reaction tail.
                    self._progress_shared_hbm_kv_map()
                    if flags.item() == self.tp_size:
                        profiled_reqs = [
                            req
                            for req in self.split_prefill_batch.reqs
                            if getattr(
                                req, "rag_prefill_handoff_profiled", False
                            )
                        ]
                        prefill_completed_at = time.perf_counter()
                        gpu_latency_ms = sum(
                            float(start.elapsed_time(end))
                            for start, end in self.rag_prefill_handoff_gpu_events
                        )
                        batch_started_at = getattr(
                            self, "rag_prefill_handoff_batch_started_ts", None
                        )
                        wall_latency_ms = (
                            (prefill_completed_at - batch_started_at) * 1000.0
                            if batch_started_at is not None
                            else None
                        )
                        for req in profiled_reqs:
                            self.rag_prefill_gpu_latency_ms.append(gpu_latency_ms)
                            if wall_latency_ms is not None:
                                self.rag_prefill_wall_latency_ms.append(
                                    wall_latency_ms
                                )
                            req.rag_prefill_completed_at = prefill_completed_at

                        result_shape = self._capture_rag_prefill_host_stage_shape(
                            'result', self.split_prefill_batch)
                        # DIAGNOSTIC (v40): the prefill result point is gated
                        # by prefill_exe_done.query() + a TP-wide allreduce, so
                        # the prefill stream has already retired through that
                        # event.  RAGLEASE_PREFILL_DRAIN widens that drain for
                        # the same reason as RAGLEASE_STREAM_DRAIN above.
                        _rag_pdrain = os.environ.get("RAGLEASE_PREFILL_DRAIN", "event")
                        if _rag_pdrain in ("stream", "all"):
                            _rag_drain_t0 = time.perf_counter()
                            prefill_stream.synchronize()
                            self._rag_note_drain(
                                "prefill", (time.perf_counter() - _rag_drain_t0) * 1000.0
                            )
                        if _rag_pdrain == "all":
                            _rag_drain_t0 = time.perf_counter()
                            torch.cuda.synchronize()
                            self._rag_note_drain(
                                "device_prefill",
                                (time.perf_counter() - _rag_drain_t0) * 1000.0,
                            )
                        result_started_at = time.perf_counter()
                        with _nvtx_range("pdmux_process_prefill_result"):
                            self.process_batch_result(
                                self.split_prefill_batch, prefill_result
                            )
                        result_sequence = self._record_rag_prefill_host_stage(result_shape, result_started_at)
                        defer_prefill_merge = bool(
                            not decode_done
                            and getattr(
                                self,
                                "rag_decode_update_deferred_for_prefill_sprint",
                                False,
                            )
                            and self.running_batch
                            and not self.running_batch.is_empty()
                        )
                        if defer_prefill_merge:
                            if (
                                getattr(
                                    self, "rag_prefill_merge_pending_batch", None
                                )
                                is not None
                            ):
                                raise RuntimeError(
                                    "Prefill merge already pending after an exclusive sprint"
                                )
                            self.rag_prefill_merge_pending_batch = (
                                self.split_prefill_batch
                            )
                            self.rag_prefill_merge_pending_result_sequence = result_sequence
                            self.rag_prefill_exclusive_deferred_merges += 1
                        else:
                            merge_shape = self._capture_rag_prefill_host_stage_shape(
                                'merge', self.split_prefill_batch)
                            merge_started_at = time.perf_counter()
                            if (
                                self.running_batch
                                and not self.running_batch.is_empty()
                            ):
                                self.running_batch.merge_batch(
                                    self.split_prefill_batch
                                )
                            else:
                                self.running_batch = self.split_prefill_batch
                            self.rag_decode_update_deferred_for_prefill_sprint = (
                                False
                            )
                            self._record_rag_prefill_host_stage(merge_shape, merge_started_at,
                                result_sequence=result_sequence)

                        # DEP 3 (P->D), formerly the #29958 port: same
                        # dependency, but published on the shared channel
                        # instead of allocating a fresh Event per handoff.
                        # Decode acquires it at its branch entry, which orders
                        # everything decode submits afterwards.
                        p_to_d.publish()
                        if profiled_reqs:
                            self.rag_handoff_decode_release_pending = not getattr(
                                self, "rag_prefill_dynamic_lease", False
                            )
                            self.rag_ann_frontend_residency_until = 0.0
                        self.rag_prefill_handoff_batch_started_ts = None
                        self.rag_prefill_handoff_gpu_events = []
                        self.split_prefill_batch = None
                        self.rag_prefill_handoff_active = False
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

            # A large split-prefill can span much longer than the native VMM
            # map. Consume its completion before starting the next scheduler
            # iteration instead of waiting for the ANN poll at the loop head.
            self._progress_shared_hbm_kv_map()
            if decode_done or prefill_done:
                self.rag_scheduler_tick_wall_ms.append(
                    (time.perf_counter() - scheduler_tick_started_at) * 1000.0
                )
