import os
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.shared_hbm import (
    SharedHBMController,
    SharedHBMKVLayout,
    SharedHBMPool,
    SharedHBMSnapshot,
    resolve_shared_hbm_map_stage_extents,
)
from sglang.srt.multiplex.ann_executor import (
    AnnExecutor,
    _FaissModuleCompat,
)
from sglang.srt.multiplex.hot_index import (
    ClusterLeasePolicy,
    ClusterRefillPlan,
    VectorLiteMapping,
    load_vectorlite_mapping,
    route_vectorlite_probes,
)
from sglang.srt.multiplex.multiplexing import SchedulerMultiplexMixin


class _FakeKVCache:
    def __init__(self, mapped_capacity):
        self.mapped_capacity = mapped_capacity

    def get_current_mapped_capacity(self):
        return self.mapped_capacity


class _FakePool:
    mode = "scratch"
    gpu_name = "NVIDIA A100-SXM4-80GB"
    gpu_total_memory = 80 << 30
    _last_faiss_requested_bytes = 0

    def __init__(self):
        self.kv_layout = SharedHBMKVLayout(100, 100, 4, 500)

    def snapshot(self):
        return SharedHBMSnapshot(
            total_bytes=400,
            chunk_bytes=100,
            free_bytes=400,
            kv_bytes=0,
            ann_scratch_bytes=0,
            ann_index_bytes=0,
            moving_bytes=0,
            kv_static_capacity_tokens=100,
            kv_tokens_per_extent=100,
            kv_total_extents=4,
            kv_mapped_extents=0,
            kv_logical_capacity_tokens=500,
        )


class _FakeAllocator:
    def __init__(self, capacity=300, available=260):
        self.capacity = capacity
        self.available = available
        self.cache = object()

    def get_kvcache(self):
        return self.cache

    def current_mapped_capacity(self):
        return self.capacity

    def available_size(self):
        return self.available


class _FakeScheduler:
    waiting_queue = ()
    running_batch = type("Batch", (), {"reqs": [1, 2, 3, 4]})()

    @staticmethod
    def _rag_ann_decode_prediction_ms():
        return 10.0


class _ReclaimingPool:
    mode = "scratch"
    gpu_id = 0
    gpu_name = "NVIDIA A100-SXM4-80GB"
    gpu_total_memory = 80 << 30
    total_bytes = 400
    chunk_bytes = 100

    def __init__(self):
        self.kv_layout = SharedHBMKVLayout(100, 100, 3, 400)
        self.kv_extents = 3
        self.scratch_chunks = 1
        self._faiss_bound = True
        self._faiss_scratch_detaching = False
        self._last_faiss_mapping = (1, 100)
        self._last_faiss_requested_bytes = 100
        self._faiss_mapping_generation = 1
        self.growable_extents = None
        self.synchronous_unbind_calls = 0
        self.fast_unbind_calls = 0
        self.fast_detach_available = True
        self.release_ann_calls = 0
        self.release_ann_error = None
        self.native_map_error = None
        self.operation_order = []

    @property
    def free_chunks(self):
        return 4 - self.kv_extents - self.scratch_chunks

    def snapshot(self):
        return SharedHBMSnapshot(
            total_bytes=self.total_bytes,
            chunk_bytes=self.chunk_bytes,
            free_bytes=self.free_chunks * self.chunk_bytes,
            kv_bytes=self.kv_extents * self.chunk_bytes,
            ann_scratch_bytes=self.scratch_chunks * self.chunk_bytes,
            ann_index_bytes=0,
            moving_bytes=0,
            kv_static_capacity_tokens=(
                self.kv_layout.static_capacity_tokens
            ),
            kv_tokens_per_extent=self.kv_layout.tokens_per_extent,
            kv_total_extents=self.kv_layout.total_extents,
            kv_mapped_extents=self.kv_extents,
            kv_logical_capacity_tokens=(
                self.kv_layout.logical_capacity_tokens
            ),
        )

    def bind_faiss_scratch(self, _executor, requested_bytes):
        if self._faiss_scratch_detaching:
            raise RuntimeError("ANN_SCRATCH release is pending")
        if self._faiss_bound and self._last_faiss_mapping[1] >= requested_bytes:
            return self._last_faiss_mapping[1]
        if self._faiss_bound:
            self.unbind_faiss_scratch(_executor)
        mapped = min(requested_bytes, self.free_chunks * self.chunk_bytes)
        self.scratch_chunks = mapped // self.chunk_bytes
        self._faiss_bound = mapped > 0
        self._faiss_scratch_detaching = False
        self._last_faiss_mapping = (1 if mapped else 0, mapped)
        if mapped > 0:
            self._faiss_mapping_generation += 1
        return mapped

    def unbind_faiss_scratch(self, _executor):
        self.synchronous_unbind_calls += 1
        if self._faiss_bound:
            self.operation_order.append("detach_sync")
            self._faiss_bound = False
            self._faiss_scratch_detaching = True
            self._last_faiss_mapping = (0, 0)
            self._faiss_mapping_generation += 1
        if self._faiss_scratch_detaching:
            self.release_ann(2)

    def unbind_faiss_scratch_after_sync(self, _executor):
        used_fast_detach = False
        if self._faiss_bound and self.fast_detach_available:
            self.fast_unbind_calls += 1
            self.operation_order.append("detach_fast")
            used_fast_detach = True
        elif self._faiss_bound:
            self.synchronous_unbind_calls += 1
            self.operation_order.append("detach_sync")
        if self._faiss_bound:
            self._faiss_bound = False
            self._faiss_scratch_detaching = True
            self._last_faiss_mapping = (0, 0)
            self._faiss_mapping_generation += 1
        if self._faiss_scratch_detaching:
            self.release_ann(2)
        return used_fast_detach

    def detach_faiss_scratch_after_sync(self, _executor):
        if not self._faiss_bound or not self.fast_detach_available:
            return False
        self.fast_unbind_calls += 1
        self.operation_order.append("detach_fast")
        self._faiss_bound = False
        self._faiss_scratch_detaching = True
        self._last_faiss_mapping = (0, 0)
        self._faiss_mapping_generation += 1
        return True

    def faiss_scratch_detaching(self):
        return self._faiss_scratch_detaching

    def release_ann(self, owner):
        self.release_ann_calls += 1
        self.operation_order.append("release")
        if self.release_ann_error is not None:
            raise self.release_ann_error
        if int(owner) != 2:
            raise ValueError(f"unexpected owner: {owner}")
        self.scratch_chunks = 0
        self._faiss_scratch_detaching = False

    def kv_growable_extents(self):
        if self.growable_extents is None:
            return self.kv_extents
        return self.growable_extents


class _ReclaimingKVCache:
    def __init__(self, pool):
        self.pool = pool
        self.shared_hbm_maintenance_stream = object()
        self.finalize_calls = 0
        self.finalize_failures_remaining = 0

    def unmap_shared_hbm_extents(self, target_extents):
        self.pool.kv_extents = target_extents
        return 100 + target_extents * 100

    def map_shared_hbm_extents(self, target_extents):
        self.pool.kv_extents = target_extents
        return 100 + target_extents * 100

    def map_shared_hbm_extents_async(self, target_extents):
        self.pool.kv_extents = target_extents
        self.zero_ready = _FakeCudaEvent()
        return 100 + target_extents * 100, self.zero_ready

    def map_shared_hbm_extents_native(self, target_extents):
        if self.pool._faiss_scratch_detaching:
            raise RuntimeError("KV map raced deferred ANN release")
        self.pool.operation_order.append("map")
        if self.pool.native_map_error is not None:
            raise self.pool.native_map_error
        self.pool.kv_extents = target_extents
        return 100 + target_extents * 100, 12.5

    def finalize_shared_hbm_extents_map(
        self, _old_capacity, _new_capacity
    ):
        self.finalize_calls += 1
        if self.finalize_failures_remaining > 0:
            self.finalize_failures_remaining -= 1
            raise RuntimeError("finalize failed")
        self.zero_ready = _FakeCudaEvent()
        self.shared_hbm_last_zero_enqueue_latency_ms = 0.25
        self.shared_hbm_last_refresh_latency_ms = 0.05
        return self.zero_ready


class _ReclaimingAllocator:
    def __init__(self, pool):
        self.pool = pool
        self.cache = _ReclaimingKVCache(pool)
        self.logical_capacity = 100 + pool.kv_extents * 100

    def get_kvcache(self):
        return self.cache

    def current_mapped_capacity(self):
        return self.logical_capacity

    def available_size(self):
        return self.logical_capacity

    def freeze_mapped_capacity(self, capacity):
        self.logical_capacity = int(capacity)
        return True

    def publish_mapped_capacity(self, old_capacity, new_capacity):
        if int(old_capacity) != self.logical_capacity:
            raise RuntimeError(
                f"unexpected logical capacity: {self.logical_capacity}"
            )
        self.logical_capacity = int(new_capacity)


class _RecoverableAllocator(_ReclaimingAllocator):
    """Model physical/native capacity separately from logical publication."""

    def __init__(self, pool):
        super().__init__(pool)
        self.publish_calls = 0
        self.publish_failures_remaining = 0

    def current_mapped_capacity(self):
        return self.logical_capacity

    def available_size(self):
        return self.logical_capacity

    def publish_mapped_capacity(self, old_capacity, new_capacity):
        self.publish_calls += 1
        if self.publish_failures_remaining > 0:
            self.publish_failures_remaining -= 1
            raise RuntimeError("publish failed")
        if int(old_capacity) != self.logical_capacity:
            raise RuntimeError(
                f"unexpected logical capacity: {self.logical_capacity}"
            )
        self.logical_capacity = int(new_capacity)


class _FenceableAnnExecutor:
    def __init__(
        self,
        fenced_streams=3,
        async_ready=False,
        fast_detach_safe=True,
    ):
        self.fenced_streams = fenced_streams
        self.fast_detach_safe = fast_detach_safe
        self.fence_calls = 0
        self.fence_stream = None
        self.async_fence_calls = 0
        self.async_fence_stream = None
        self.async_fence_event = _FakeCudaEvent(ready=async_ready)

    def fence_shared_hbm_users(self, fence_stream):
        self.fence_calls += 1
        self.fence_stream = fence_stream
        return self.fenced_streams

    def submit_shared_hbm_reclaim_fence(self, fence_stream):
        self.async_fence_calls += 1
        self.async_fence_stream = fence_stream
        return SimpleNamespace(
            ready_event=self.async_fence_event,
            source_events=(),
            stream_count=self.fenced_streams,
            fast_detach_safe=self.fast_detach_safe,
        )


class _FakeCudaStream:
    def __init__(self, pointer):
        self.cuda_stream = pointer
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class _FailingCudaStream(_FakeCudaStream):
    def __init__(self, pointer, failures=1):
        super().__init__(pointer)
        self.failures_remaining = int(failures)

    def wait_event(self, event):
        super().wait_event(event)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("consumer wait failed")


class _FakeCudaEvent:
    def __init__(self, ready=False):
        self.ready = ready
        self.recorded_stream = None
        self.synchronize_calls = 0

    def record(self, stream):
        self.recorded_stream = stream

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronize_calls += 1
        self.ready = True


class _ManualFuture:
    def __init__(self, function, args):
        self.function = function
        self.args = args
        self.value = None
        self.error = None
        self.completed = False
        self.callbacks = []

    def done(self):
        return self.completed

    def result(self):
        if not self.completed:
            raise RuntimeError("manual future is incomplete")
        if self.error is not None:
            raise self.error
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        if self.completed:
            callback(self)

    def run(self):
        try:
            self.value = self.function(*self.args)
        except Exception as error:
            self.error = error
        finally:
            self.completed = True
        for callback in self.callbacks:
            callback(self)


class _ManualExecutor:
    def __init__(self):
        self.future = None

    def submit(self, function, *args):
        self.future = _ManualFuture(function, args)
        return self.future


class _RejectingExecutor:
    def submit(self, _function, *_args):
        raise RuntimeError("worker submission rejected")


class _IdleScheduler:
    waiting_queue = ()
    running_batch = type("Batch", (), {"reqs": []})()
    tree_cache = None

    @staticmethod
    def _rag_ann_gpu_work_active():
        return False

    @staticmethod
    def _rag_ann_decode_prediction_ms():
        return 10.0


class TestSharedHBMMapStageResolution(unittest.TestCase):
    def test_auto_uses_measured_architecture_defaults(self):
        self.assertEqual(
            resolve_shared_hbm_map_stage_extents(
                -1, "NVIDIA A100-SXM4-80GB"
            ),
            1,
        )
        self.assertEqual(
            resolve_shared_hbm_map_stage_extents(
                -1, "NVIDIA H100 80GB HBM3"
            ),
            2,
        )

    def test_explicit_stage_is_architecture_independent(self):
        for configured in (0, 1, 2, 4):
            self.assertEqual(
                resolve_shared_hbm_map_stage_extents(
                    configured, "NVIDIA A100-SXM4-80GB"
                ),
                configured,
            )
            self.assertEqual(
                resolve_shared_hbm_map_stage_extents(
                    configured, "NVIDIA H100 80GB HBM3"
                ),
                configured,
            )

    def test_controller_exposes_configured_and_resolved_auto_values(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )

        self.assertEqual(controller.map_stage_extents_configured, -1)
        self.assertEqual(controller.map_stage_extents, 1)
        metrics = controller.metrics()
        self.assertEqual(
            metrics["kv_growth_map_stage_extents_configured"], -1
        )
        self.assertTrue(metrics["kv_growth_map_stage_extents_auto"])
        self.assertEqual(metrics["kv_growth_map_stage_extents"], 1)

    def test_invalid_auto_target_and_sentinel_fail_closed(self):
        with self.assertRaises(ValueError):
            resolve_shared_hbm_map_stage_extents(
                -2, "NVIDIA A100-SXM4-80GB"
            )
        with self.assertRaises(RuntimeError):
            resolve_shared_hbm_map_stage_extents(-1, "NVIDIA L40S")


class TestLeaseAwareTokenAllocator(unittest.TestCase):
    def test_unmapped_tail_is_never_allocated(self):
        cache = _FakeKVCache(mapped_capacity=4)
        allocator = TokenToKVPoolAllocator(
            size=12,
            dtype=torch.int64,
            device="cpu",
            kvcache=cache,
            need_sort=True,
        )
        self.assertEqual(allocator.free_pages.tolist(), [1, 2, 3, 4])

        allocator.publish_mapped_capacity(4, 8)
        allocated = allocator.alloc(8)
        self.assertEqual(allocated.tolist(), list(range(1, 9)))
        self.assertFalse(allocator.freeze_mapped_capacity(4))

        allocator.free(allocated[4:])
        self.assertTrue(allocator.freeze_mapped_capacity(4))
        allocator.merge_and_sort_free()
        self.assertTrue(bool(torch.all(allocator.free_pages <= 4)))
        self.assertEqual(allocator.current_mapped_capacity(), 4)


class TestSharedHBMForecast(unittest.TestCase):
    def test_faiss_mapping_identity_read_does_not_take_vmm_lock(self):
        pool = SharedHBMPool.__new__(SharedHBMPool)
        pool._faiss_mapping_identity = (True, (1234, 5678), 9)

        class _RejectLock:
            def __enter__(self):
                raise AssertionError("identity read acquired the VMM lock")

            def __exit__(self, *_args):
                return False

        pool._lock = _RejectLock()
        self.assertEqual(
            pool.faiss_mapping_state(), (True, (1234, 5678), 9)
        )

    def test_faiss_bind_rejects_detaching_native_scratch(self):
        pool = SharedHBMPool.__new__(SharedHBMPool)
        pool._lock = nullcontext()
        pool._faiss_scratch_detaching = True

        with self.assertRaisesRegex(RuntimeError, "release is pending"):
            pool.bind_faiss_scratch(object(), requested_bytes=100)

    def test_after_sync_detach_only_keeps_native_owner_until_unbind(self):
        pool = SharedHBMPool.__new__(SharedHBMPool)
        pool._lock = nullcontext()
        pool.gpu_id = 0
        pool._faiss_bound = True
        pool._faiss_scratch_detaching = False
        pool._last_faiss_mapping = (1234, 400)
        pool._last_faiss_requested_bytes = 300
        pool._faiss_mapping_generation = 7
        pool._faiss_mapping_identity = (True, (1234, 400), 7)
        detach_calls = []
        executor = SimpleNamespace(
            _gpu_res=SimpleNamespace(
                clearExternalTempMemoryAfterSync=(
                    lambda device: detach_calls.append(device)
                )
            ),
            _swig=SimpleNamespace(),
        )

        with patch.object(pool, "release_ann") as release_ann:
            self.assertTrue(
                pool.detach_faiss_scratch_after_sync(executor)
            )
            release_ann.assert_not_called()
            self.assertEqual(detach_calls, [0])
            self.assertFalse(pool._faiss_bound)
            self.assertTrue(pool._faiss_scratch_detaching)
            self.assertEqual(
                pool.faiss_mapping_state(), (False, (0, 0), 8)
            )

            release_ann.side_effect = lambda _owner: setattr(
                pool, "_faiss_scratch_detaching", False
            )
            pool.unbind_faiss_scratch(executor)
            release_ann.assert_called_once()
            self.assertFalse(pool._faiss_scratch_detaching)

    @staticmethod
    def _controller_with_a100_pressure_layout():
        pool = _FakePool()
        pool.kv_layout = SharedHBMKVLayout(
            static_capacity_tokens=458751,
            tokens_per_extent=2048,
            total_extents=91,
            logical_capacity_tokens=645119,
        )
        return SharedHBMController(
            pool=pool,
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )

    def test_faiss_scratch_sizing_matches_measured_a100_tiers(self):
        mib = 1 << 20
        self.assertEqual(
            SharedHBMController.ann_scratch_request_bytes(1), 160 * mib
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_request_bytes(2), 288 * mib
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_request_bytes(4), 448 * mib
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_request_bytes(8), 448 * mib
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_request_bytes(
                1, max_inflight=8
            ),
            int(1.5 * (1 << 30)),
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_capacity_rows(287 * mib, 8),
            1,
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_capacity_rows(447 * mib, 8),
            2,
        )
        self.assertEqual(
            SharedHBMController.ann_scratch_capacity_rows(448 * mib, 8),
            8,
        )

    def test_adaptive_growth_prefetches_and_caps_at_pool_end(self):
        controller = self._controller_with_a100_pressure_layout()

        target, prefetched = controller._adaptive_kv_growth_target(
            required_tokens=472229,
            raw_target_extents=7,
            now=1.0,
        )
        self.assertEqual((target, prefetched), (8, 1))

        target, prefetched = controller._adaptive_kv_growth_target(
            required_tokens=639825,
            raw_target_extents=89,
            now=2.0,
        )
        self.assertEqual((target, prefetched), (91, 0))

    def test_rerank_growth_can_request_larger_coalescing_floor(self):
        controller = self._controller_with_a100_pressure_layout()

        target, prefetched = controller._adaptive_kv_growth_target(
            required_tokens=472229,
            raw_target_extents=7,
            now=1.0,
            min_prefetch_extents=32,
        )

        self.assertEqual((target, prefetched), (39, 32))

    def test_adaptive_growth_uses_underprediction_and_demand_rate(self):
        controller = self._controller_with_a100_pressure_layout()
        controller.prediction_error_tokens.append(-50000)

        target, prefetched = controller._adaptive_kv_growth_target(
            required_tokens=480000,
            raw_target_extents=10,
            now=1.0,
        )
        self.assertEqual((target, prefetched), (48, 32))

        controller.prediction_error_tokens.clear()
        controller.last_kv_growth_at = 1.0
        controller.last_kv_growth_required_tokens = 480000
        controller.handoff_latency_ms.append(200.0)
        target, prefetched = controller._adaptive_kv_growth_target(
            required_tokens=680000,
            raw_target_extents=10,
            now=1.1,
        )
        self.assertEqual((target, prefetched), (48, 32))
        self.assertAlmostEqual(
            controller.kv_growth_rate_tokens_per_s, 2000000.0
        )

    def test_adaptive_growth_halves_observed_pressure_handoffs(self):
        controller = self._controller_with_a100_pressure_layout()
        observed_growth_points = [
            (1.0, 7, 110.0),
            (4.0, 17, 203.0),
            (8.0, 34, 65.0),
            (12.0, 52, 120.0),
            (17.0, 71, 123.0),
            (20.0, 81, 158.0),
        ]
        mapped = 0
        transitions = 0
        for now, raw_target, latency_ms in observed_growth_points:
            if raw_target <= mapped:
                continue
            required_tokens = 458751 + raw_target * 2048
            mapped, _ = controller._adaptive_kv_growth_target(
                required_tokens=required_tokens,
                raw_target_extents=raw_target,
                now=now,
            )
            transitions += 1
            controller.last_kv_growth_at = now
            controller.last_kv_growth_required_tokens = required_tokens
            controller.handoff_latency_ms.append(latency_ms)

        self.assertEqual(transitions, 6)
        self.assertEqual(mapped, 91)

    def test_lookahead_prediction_and_delayed_error(self):
        allocator = _FakeAllocator()
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=allocator,
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )
        scheduler = _FakeScheduler()

        with patch(
            "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
            side_effect=[1.0, 1.06],
        ):
            # 40 live + 4 requests * 5 decode steps + 2 safety extents.
            self.assertEqual(controller.predict_kv_tokens(scheduler), 260)
            allocator.available = 100  # 200 actually live at the horizon.
            controller.predict_kv_tokens(scheduler)

        self.assertEqual(list(controller.prediction_error_tokens), [60])

    def test_prediction_excludes_reclaimable_prefix_cache(self):
        allocator = _FakeAllocator()
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=allocator,
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )
        scheduler = SimpleNamespace(
            waiting_queue=(),
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            tree_cache=SimpleNamespace(evictable_size=lambda: 30),
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        # 40 allocated - 30 reclaimable + 20 decode + 200 reserve.
        self.assertEqual(controller.predict_kv_tokens(scheduler), 230)
        self.assertEqual(controller.last_prediction_evictable_tokens, 30)

    def test_unrecorded_forecast_has_no_observability_side_effects(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )
        controller.pending_predictions.append((0.5, 123))
        controller.kv_growth_native_map_latency_ms.append(150.0)
        controller.last_predicted_tokens = -101
        controller.last_effective_lookahead_ms = -102.0
        controller.last_lookahead_observed_map_ms = -103.0
        controller.last_prediction_queued_prefill_tokens = -104
        controller.last_prediction_prefill_batches = -105
        before = (
            list(controller.pending_predictions),
            list(controller.prediction_error_tokens),
            controller.last_predicted_tokens,
            controller.last_effective_lookahead_ms,
            controller.last_lookahead_observed_map_ms,
            controller.last_prediction_queued_prefill_tokens,
            controller.last_prediction_prefill_batches,
        )

        with patch(
            "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
            return_value=1.0,
        ):
            predicted = controller.predict_kv_tokens(
                _FakeScheduler(),
                prefill_admission_batches=3,
                additional_prefill_tokens=7,
                record=False,
            )

        self.assertEqual(predicted, 315)
        self.assertEqual(
            (
                list(controller.pending_predictions),
                list(controller.prediction_error_tokens),
                controller.last_predicted_tokens,
                controller.last_effective_lookahead_ms,
                controller.last_lookahead_observed_map_ms,
                controller.last_prediction_queued_prefill_tokens,
                controller.last_prediction_prefill_batches,
            ),
            before,
        )

    def test_lookahead_adapts_to_observed_vmm_latency(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )
        controller.kv_growth_native_map_latency_ms.extend(
            [80.0, 150.0]
        )

        # p95=150 ms plus 10% safety gives a 165 ms horizon:
        # 40 live + 4 requests * 17 decode steps + 2 safety extents.
        self.assertEqual(
            controller.predict_kv_tokens(_FakeScheduler()), 308
        )
        self.assertEqual(controller.last_effective_lookahead_ms, 165.0)
        self.assertEqual(
            controller.last_lookahead_observed_map_ms, 150.0
        )

    def test_staged_lookahead_uses_first_publish_not_prefetch_poll(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            map_stage_extents=2,
        )
        controller.kv_growth_native_map_latency_ms.append(150.0)
        controller.kv_growth_total_latency_ms.append(2000.0)
        controller.kv_growth_first_publish_latency_ms.append(45.0)

        self.assertEqual(
            controller.predict_kv_tokens(_FakeScheduler()), 260
        )
        self.assertEqual(controller.last_effective_lookahead_ms, 50.0)
        self.assertEqual(
            controller.last_lookahead_observed_map_ms, 45.0
        )

    def test_rerank_window_can_cover_multiple_prefill_admissions(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=4,
        )
        requests = [
            SimpleNamespace(extend_input_len=10)
            for _ in range(4)
        ]
        scheduler = SimpleNamespace(
            waiting_queue=requests,
            max_prefill_tokens=16,
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        # Ordinary ticks retain the one-admission prediction.
        self.assertEqual(controller.predict_kv_tokens(scheduler), 280)
        controller.on_ann_gpu_idle(scheduler)

        self.assertEqual(controller.last_predicted_tokens, 300)
        self.assertEqual(
            controller.last_prediction_queued_prefill_tokens, 40
        )
        self.assertEqual(controller.last_prediction_prefill_batches, 4)
        self.assertEqual(controller.last_rerank_extra_prefill_tokens, 20)
        self.assertEqual(controller.rerank_extra_prefill_tokens_total, 20)

    def test_candidates_ready_prefetches_while_ann_is_only_queued(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=4,
        )
        requests = [
            SimpleNamespace(extend_input_len=10)
            for _ in range(4)
        ]
        scheduler = SimpleNamespace(
            waiting_queue=requests,
            ann_search_pending=[object()],
            max_prefill_tokens=16,
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        controller.on_ann_gpu_idle(scheduler)

        self.assertEqual(controller.last_predicted_tokens, 300)
        self.assertEqual(controller.last_prediction_prefill_batches, 4)
        self.assertEqual(controller.rerank_prefetches_with_ann_queued, 1)

    def test_candidates_ready_covers_full_prefill_backlog_after_ann_drain(
        self,
    ):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(
                capacity=1000, available=960
            ),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=4,
        )
        scheduler = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(extend_input_len=10)
                for _ in range(20)
            ],
            ann_search_pending=(),
            max_prefill_tokens=16,
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            _rag_ann_gpu_work_active=lambda: False,
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        controller.ann_idle_since = 0.0
        with patch(
            "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
            return_value=1.2,
        ):
            controller.on_ann_gpu_idle(scheduler)

        self.assertEqual(controller.last_predicted_tokens, 460)
        self.assertEqual(
            controller.last_prediction_queued_prefill_tokens, 200
        )
        self.assertEqual(controller.rerank_all_queue_predictions, 1)

    def test_ann_drain_requires_quiescence_before_full_backlog(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(
                capacity=1000, available=960
            ),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=4,
        )
        scheduler = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(extend_input_len=10)
                for _ in range(20)
            ],
            ann_search_pending=(),
            max_prefill_tokens=16,
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            _rag_ann_gpu_work_active=lambda: False,
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        with patch(
            "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
            return_value=1.0,
        ):
            controller.on_ann_gpu_idle(scheduler)

        self.assertEqual(controller.last_predicted_tokens, 330)
        self.assertEqual(
            controller.last_prediction_queued_prefill_tokens, 70
        )
        self.assertEqual(controller.rerank_all_queue_predictions, 0)
        self.assertEqual(controller.ann_idle_since, 1.0)

    def test_quiescent_tick_refills_ivf_from_existing_heat(self):
        class _Executor:
            shared_ivf_resident_count = 0
            shared_ivf_total_count = 10

            _shared_ivf_policy = SimpleNamespace(observed_misses=128)

        pool = _FakePool()
        pool.mode = "ivf"
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_FakeAllocator(),
            ann_executor=_Executor(),
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=1,
        )
        controller.ann_idle_since = 0.0
        scheduler = SimpleNamespace(
            ann_search_pending=(),
            _rag_ann_gpu_work_active=lambda: False,
            rag_ann_deadline_ms=50.0,
        )

        def record_reload(
            _scheduler,
            deadline_slack_ms=None,
            *,
            allow_existing_heat=False,
            fill_free_capacity=False,
            background_max_lists=None,
        ):
            self.assertEqual(deadline_slack_ms, 250.0)
            self.assertTrue(allow_existing_heat)
            self.assertTrue(fill_free_capacity)
            self.assertEqual(background_max_lists, 256)
            controller.ivf_reload_attempts += 1
            controller.pending_ivf_reload = object()
            return False

        with (
            patch.object(
                controller,
                "_ensure_rerank_capacity",
                return_value=True,
            ) as ensure,
            patch.object(
                controller,
                "_maybe_reload_ivf",
                side_effect=record_reload,
            ) as reload_ivf,
            patch(
                "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
                return_value=2.0,
            ),
        ):
            self.assertTrue(
                controller.on_ann_quiescent_tick(scheduler)
            )

        ensure.assert_called_once_with(
            scheduler,
            reason="rerank_drain",
        )
        reload_ivf.assert_called_once()
        self.assertTrue(controller.ann_drain_epoch_processed)
        self.assertEqual(controller.ivf_idle_maintenance_ticks, 1)
        self.assertEqual(controller.ivf_idle_refill_attempts, 1)
        self.assertEqual(controller.next_ivf_idle_refill_at, 2.01)

    def test_quiescent_tick_waits_for_llm_and_kv_map_drain(self):
        class _Executor:
            shared_ivf_resident_count = 0
            shared_ivf_total_count = 10

            _shared_ivf_policy = SimpleNamespace(observed_misses=128)

        pool = _FakePool()
        pool.mode = "ivf"
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_FakeAllocator(),
            ann_executor=_Executor(),
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=1,
        )
        controller.ann_idle_since = 0.0
        controller.pending_kv_native_map = object()
        scheduler = SimpleNamespace(
            ann_search_pending=(),
            waiting_queue=(),
            running_batch=SimpleNamespace(reqs=[object()]),
            _rag_ann_gpu_work_active=lambda: False,
            rag_ann_deadline_ms=50.0,
        )

        with (
            patch.object(
                controller,
                "_ensure_rerank_capacity",
                return_value=True,
            ) as ensure,
            patch.object(
                controller,
                "_maybe_reload_ivf",
                return_value=False,
            ) as reload_ivf,
            patch(
                "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
                return_value=2.0,
            ),
        ):
            self.assertFalse(
                controller.on_ann_quiescent_tick(scheduler)
            )

        ensure.assert_not_called()
        reload_ivf.assert_not_called()
        self.assertIsNone(controller.ann_idle_since)

    def test_quiescent_tick_commits_kv_before_full_idle(self):
        class _Executor:
            shared_ivf_resident_count = 0
            shared_ivf_total_count = 10

            _shared_ivf_policy = SimpleNamespace(observed_misses=128)

        pool = _FakePool()
        pool.mode = "ivf"
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_FakeAllocator(),
            ann_executor=_Executor(),
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=1,
        )
        controller.ann_idle_since = 0.0
        scheduler = SimpleNamespace(
            ann_search_pending=(),
            waiting_queue=[SimpleNamespace(extend_input_len=10)],
            running_batch=SimpleNamespace(reqs=[object()]),
            _rag_ann_gpu_work_active=lambda: False,
            rag_ann_deadline_ms=50.0,
        )

        with (
            patch.object(
                controller,
                "_ensure_rerank_capacity",
                return_value=True,
            ) as ensure,
            patch.object(
                controller,
                "_maybe_reload_ivf",
                return_value=False,
            ) as reload_ivf,
            patch(
                "sglang.srt.mem_cache.shared_hbm.time.perf_counter",
                return_value=2.0,
            ),
        ):
            self.assertTrue(
                controller.on_ann_quiescent_tick(scheduler)
            )

        ensure.assert_called_once_with(
            scheduler,
            reason="rerank_drain",
        )
        reload_ivf.assert_not_called()
        self.assertTrue(controller.ann_drain_epoch_processed)
        self.assertEqual(controller.kv_growth_ann_frontier_commits, 1)

    def test_ivf_kv_growth_waits_for_ann_frontier_quiescence(self):
        pool = _FakePool()
        pool.mode = "ivf"
        pool.snapshot = lambda: SimpleNamespace(
            ann_index_bytes=100,
        )
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_FakeAllocator(capacity=100, available=100),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=0,
            rerank_prefetch_batches=2,
        )
        scheduler = SimpleNamespace(
            waiting_queue=[SimpleNamespace(extend_input_len=200)],
            ann_search_pending=[object()],
            max_prefill_tokens=256,
            running_batch=None,
            _rag_ann_gpu_work_active=lambda: False,
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        with patch.object(
            controller, "ensure_kv_capacity", return_value=True
        ) as ensure:
            self.assertFalse(
                controller._ensure_rerank_capacity(
                    scheduler,
                    reason="rerank_ready",
                )
            )

        ensure.assert_not_called()
        self.assertEqual(controller.kv_growth_ann_frontier_holds, 1)
        self.assertGreater(
            controller.kv_growth_ann_frontier_hold_tokens, 0
        )

    def test_prefill_ready_uses_exact_tokens_and_coalesces_growth(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
            rerank_prefetch_batches=4,
        )
        scheduler = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(extend_input_len=10)
                for _ in range(4)
            ],
            ann_search_pending=(),
            max_prefill_tokens=16,
            running_batch=SimpleNamespace(reqs=[1, 2, 3, 4]),
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        with patch.object(
            controller, "ensure_kv_capacity", return_value=True
        ) as ensure:
            self.assertTrue(
                controller.on_rag_prefill_ready(
                    scheduler, actual_prefill_tokens=10
                )
            )

        ensure.assert_called_once_with(
            310,
            min_prefetch_extents=32,
            reason="rerank_ready",
        )
        self.assertEqual(controller.rerank_prefill_ready_calls, 1)
        self.assertEqual(
            controller.rerank_prefill_ready_tokens_total, 10
        )
        self.assertEqual(controller.last_predicted_tokens, 310)
        self.assertEqual(
            controller.last_prediction_queued_prefill_tokens, 50
        )
        self.assertEqual(controller.last_prediction_prefill_batches, 4)
        self.assertEqual(controller.last_rerank_extra_prefill_tokens, 20)
        self.assertEqual(controller.rerank_growth_requests, 1)

    def test_deadline_miss_is_observed_at_completion(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=10,
            kv_reserve_extents=2,
        )
        controller.observe_ann_completion(49.0, 50.0)
        controller.observe_ann_completion(51.0, 50.0)
        self.assertEqual(controller.deadline_misses, 1)

    def test_growing_scratch_reclaims_idle_kv_instead_of_falling_back(self):
        pool = _ReclaimingPool()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=object(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
        )

        admitted = controller.before_ann_launch(
            _IdleScheduler(), requested_bytes=200
        )

        self.assertTrue(admitted)
        self.assertEqual(pool.kv_extents, 2)
        self.assertEqual(pool.scratch_chunks, 2)
        self.assertEqual(controller.transition_counts["kv_to_ann"], 1)

    def test_profitable_ivf_recovery_returns_predicted_excess_kv(self):
        pool = _ReclaimingPool()
        pool.mode = "ivf"
        allocator = _ReclaimingAllocator(pool)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=object(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
        )

        arena_bytes = controller._prepare_ivf_recovery_arena(
            _IdleScheduler(), predicted_kv_tokens=250
        )

        self.assertEqual(pool.kv_extents, 2)
        self.assertEqual(allocator.current_mapped_capacity(), 300)
        self.assertEqual(arena_bytes, 100)
        self.assertEqual(controller.kv_to_ivf_reclaims, 1)
        self.assertEqual(controller.kv_to_ivf_extents_reclaimed, 1)
        self.assertEqual(controller.kv_to_ivf_bytes_reclaimed, 100)
        self.assertEqual(controller.transition_counts["kv_to_ann"], 1)
        self.assertEqual(controller.transition_counts["kv_to_ivf"], 1)

    def test_ivf_recovery_rejects_inflight_kv_native_map(self):
        pool = _ReclaimingPool()
        pool.mode = "ivf"
        allocator = _ReclaimingAllocator(pool)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=object(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
        )
        controller.pending_kv_native_map = object()

        arena_bytes = controller._prepare_ivf_recovery_arena(
            _IdleScheduler(), predicted_kv_tokens=100
        )

        self.assertEqual(arena_bytes, 0)
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(controller.kv_to_ivf_reclaim_failures, 1)
        self.assertEqual(
            controller.failure_counts[
                "kv_to_ivf_transition_inflight"
            ],
            1,
        )

    def test_kv_backlog_commit_sends_new_ann_to_cpu(self):
        pool = _ReclaimingPool()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=object(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            rerank_prefetch_batches=4,
        )
        controller.kv_backlog_committed = True
        scheduler = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(extend_input_len=1000)
            ],
            running_batch=SimpleNamespace(reqs=[]),
            _rag_ann_decode_prediction_ms=lambda: 10.0,
        )

        admitted = controller.before_ann_launch(
            scheduler, requested_bytes=200
        )

        self.assertFalse(admitted)
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertEqual(controller.kv_backlog_ann_fallbacks, 1)
        self.assertEqual(
            controller.failure_counts["kv_backlog_committed"], 1
        )

    def test_new_ann_frontier_does_not_reserve_all_blocked_prefills(self):
        pool = _ReclaimingPool()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=object(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            rerank_prefetch_batches=4,
        )
        controller.kv_backlog_committed = True
        prediction_calls = []
        controller.predict_kv_tokens = lambda _scheduler, **kwargs: (
            prediction_calls.append(kwargs) or 100
        )

        admitted = controller.before_ann_launch(
            _IdleScheduler(), requested_bytes=200
        )

        self.assertTrue(admitted)
        self.assertFalse(controller.kv_backlog_committed)
        self.assertEqual(
            prediction_calls[0]["prefill_admission_batches"], 1
        )
        self.assertFalse(
            prediction_calls[0]["include_all_queued_prefill"]
        )

    def test_disjoint_tail_scratch_is_preserved_during_kv_growth(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 3
        executor = _FenceableAnnExecutor()
        manual_executor = _ManualExecutor()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))

        self.assertEqual(executor.fence_calls, 0)
        self.assertEqual(executor.async_fence_calls, 0)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertTrue(pool._faiss_bound)
        self.assertEqual(controller.kv_growth_disjoint_ann_bypasses, 1)
        self.assertEqual(
            controller.kv_growth_disjoint_ann_bytes_preserved, 100
        )
        self.assertEqual(controller.kv_growth_ann_conflicts, 0)
        self.assertEqual(controller.last_kv_growable_extents, 3)
        self.assertEqual(controller.pending_kv_native_map["target"], 3)

    def test_disjoint_native_map_reuses_exact_bound_scratch(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 3
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=_FenceableAnnExecutor(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            kv_map_executor=_ManualExecutor(),
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        pending = controller.pending_kv_native_map
        self.assertFalse(pending["ann_reclaim_required"])
        self.assertTrue(pending["ann_disjoint"])
        self.assertEqual(pending["faiss_mapping"], (1, 100))
        self.assertEqual(
            pending["faiss_mapping_generation"],
            pool._faiss_mapping_generation,
        )
        with patch.object(
            controller, "ann_scratch_request_bytes", return_value=100
        ), patch.object(
            controller, "ann_scratch_capacity_rows", return_value=4
        ), patch.object(pool, "snapshot") as snapshot:
            self.assertEqual(
                controller.limit_ann_microbatch_rows(_IdleScheduler(), 4),
                4,
            )
            self.assertEqual(
                controller.ann_disjoint_map_overlap_rows, 0
            )
            self.assertTrue(
                controller.before_ann_launch(
                    _IdleScheduler(),
                    requested_bytes=100,
                    num_queries=4,
                )
            )
            snapshot.assert_not_called()

        metrics = controller.metrics()
        self.assertTrue(metrics["ann_disjoint_map_overlap_enabled"])
        self.assertEqual(
            metrics["ann_disjoint_map_overlap_admissions"], 1
        )
        self.assertEqual(metrics["ann_disjoint_map_overlap_rows"], 4)
        self.assertEqual(
            metrics["ann_disjoint_map_overlap_rejections"], 0
        )

    def test_disjoint_native_map_overlap_rejects_unsafe_identity_or_size(self):
        for mutation in ("insufficient", "generation", "ivf"):
            with self.subTest(mutation=mutation):
                pool = _ReclaimingPool()
                pool.kv_extents = 0
                pool.growable_extents = 3
                controller = SharedHBMController(
                    pool=pool,
                    token_allocator=_ReclaimingAllocator(pool),
                    ann_executor=_FenceableAnnExecutor(),
                    lookahead_ms=50,
                    min_residency_ms=0,
                    kv_reserve_extents=0,
                    async_native_map=True,
                    kv_map_executor=_ManualExecutor(),
                )
                self.assertTrue(controller.ensure_kv_capacity(101))
                requested_bytes = 100
                if mutation == "insufficient":
                    requested_bytes = 101
                elif mutation == "generation":
                    pool._faiss_mapping_generation += 1
                else:
                    pool.mode = "ivf"

                with patch.object(pool, "snapshot") as snapshot:
                    self.assertFalse(
                        controller.before_ann_launch(
                            _IdleScheduler(),
                            requested_bytes=requested_bytes,
                        )
                    )
                    snapshot.assert_not_called()
                self.assertEqual(
                    controller.ann_disjoint_map_overlap_rejections, 1
                )

    def test_disjoint_native_map_overlap_ablation_rejects(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 3
        with patch.dict(
            os.environ,
            {"SGLANG_SHARED_HBM_DISJOINT_MAP_ANN_OVERLAP": "0"},
        ):
            controller = SharedHBMController(
                pool=pool,
                token_allocator=_ReclaimingAllocator(pool),
                ann_executor=_FenceableAnnExecutor(),
                lookahead_ms=50,
                min_residency_ms=0,
                kv_reserve_extents=0,
                async_native_map=True,
                kv_map_executor=_ManualExecutor(),
            )
        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertFalse(
            controller.before_ann_launch(
                _IdleScheduler(), requested_bytes=100
            )
        )
        self.assertFalse(controller.ann_disjoint_map_overlap_enabled)
        self.assertEqual(
            controller.ann_disjoint_map_overlap_rejections, 1
        )

    def test_ann_reclaim_pending_never_overlaps_native_map_admission(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 0
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=_FenceableAnnExecutor(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            kv_map_executor=_ManualExecutor(),
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertIsNotNone(controller.pending_ann_reclaim)
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertFalse(
            controller.before_ann_launch(
                _IdleScheduler(), requested_bytes=100
            )
        )
        self.assertEqual(
            controller.limit_ann_microbatch_rows(_IdleScheduler(), 4), 0
        )
        self.assertEqual(
            controller.ann_disjoint_map_overlap_rejections, 2
        )

    def test_adaptive_target_is_capped_before_it_reclaims_ann(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 2
        executor = _FenceableAnnExecutor()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            kv_map_executor=_ManualExecutor(),
        )

        self.assertTrue(controller.ensure_kv_capacity(101))

        self.assertEqual(controller.pending_kv_native_map["target"], 2)
        self.assertIsNone(controller.pending_ann_reclaim)
        self.assertEqual(executor.fence_calls, 0)
        self.assertEqual(executor.async_fence_calls, 0)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertEqual(
            controller.kv_growth_ann_preserving_target_caps, 1
        )
        self.assertEqual(
            controller.kv_growth_ann_preserving_extents_saved, 1
        )
        self.assertEqual(controller.kv_growth_disjoint_ann_bypasses, 1)

    def test_ablation_env_restores_blocking_uncapped_path(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.growable_extents = 2
        executor = _FenceableAnnExecutor()
        with patch.dict(
            os.environ,
            {
                "SGLANG_SHARED_HBM_ASYNC_RECLAIM_FENCE": "0",
                "SGLANG_SHARED_HBM_ANN_PRESERVING_TARGET_CAP": "0",
            },
        ):
            controller = SharedHBMController(
                pool=pool,
                token_allocator=_ReclaimingAllocator(pool),
                ann_executor=executor,
                lookahead_ms=50,
                min_residency_ms=0,
                kv_reserve_extents=0,
                async_native_map=True,
                kv_map_executor=_ManualExecutor(),
            )

        self.assertTrue(controller.ensure_kv_capacity(101))

        self.assertFalse(controller.async_reclaim_fence_enabled)
        self.assertFalse(controller.ann_preserving_target_cap_enabled)
        self.assertEqual(controller.pending_kv_native_map["target"], 3)
        self.assertIsNone(controller.pending_ann_reclaim)
        self.assertEqual(executor.async_fence_calls, 0)
        self.assertEqual(executor.fence_calls, 1)
        self.assertEqual(controller.kv_growth_stream_fences, 1)
        self.assertEqual(pool.scratch_chunks, 0)
        self.assertEqual(
            controller.kv_growth_ann_preserving_target_caps, 0
        )

    def test_kv_growth_fences_only_conflicting_ann_without_device_sync(self):
        for scratch_chunks in (1, 0):
            with self.subTest(scratch_chunks=scratch_chunks):
                pool = _ReclaimingPool()
                pool.kv_extents = 0
                pool.scratch_chunks = scratch_chunks
                pool._faiss_bound = scratch_chunks > 0
                pool._last_faiss_mapping = (
                    (1, 100) if scratch_chunks else (0, 0)
                )
                executor = _FenceableAnnExecutor()
                publication_stream = _FakeCudaStream(11)
                consumer_stream = _FakeCudaStream(12)
                controller = SharedHBMController(
                    pool=pool,
                    token_allocator=_ReclaimingAllocator(pool),
                    ann_executor=executor,
                    lookahead_ms=50,
                    min_residency_ms=0,
                    kv_reserve_extents=0,
                    kv_publication_stream=publication_stream,
                    kv_consumer_streams=(
                        publication_stream,
                        consumer_stream,
                        _FakeCudaStream(12),
                    ),
                )

                with patch(
                    "sglang.srt.mem_cache.shared_hbm.torch.cuda.synchronize"
                ) as device_synchronize, patch(
                    "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event"
                ) as event_type, patch(
                    "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
                    return_value=nullcontext(),
                ):
                    publication_event = _FakeCudaEvent()
                    event_type.return_value = publication_event
                    self.assertTrue(controller.ensure_kv_capacity(101))

                device_synchronize.assert_not_called()
                self.assertIs(
                    publication_event.recorded_stream,
                    publication_stream,
                )
                self.assertEqual(
                    publication_event.synchronize_calls, 0
                )
                self.assertEqual(
                    controller.kv_cache.zero_ready.synchronize_calls, 0
                )
                self.assertEqual(
                    publication_stream.waited_events,
                    [controller.kv_cache.zero_ready],
                )
                self.assertEqual(
                    consumer_stream.waited_events,
                    [publication_event],
                )
                self.assertEqual(
                    controller.kv_growth_device_synchronizations,
                    0,
                )
                expected_fences = 1 if scratch_chunks else 0
                self.assertEqual(
                    controller.kv_growth_stream_fences, expected_fences
                )
                self.assertEqual(
                    controller.kv_growth_streams_fenced,
                    3 * expected_fences,
                )
                self.assertEqual(
                    controller.kv_growth_publication_event_waits, 0
                )
                self.assertEqual(
                    controller.kv_growth_async_publications, 1
                )
                self.assertEqual(
                    controller.kv_growth_consumer_stream_waits, 1
                )
                self.assertEqual(
                    len(controller.pending_kv_publications), 1
                )
                self.assertEqual(executor.fence_calls, expected_fences)
                if scratch_chunks:
                    self.assertIs(
                        executor.fence_stream,
                        controller.kv_cache.shared_hbm_maintenance_stream,
                    )
                else:
                    self.assertIsNone(executor.fence_stream)
                self.assertEqual(pool.kv_extents, 3)

    def test_async_native_map_does_not_publish_before_worker_completion(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        executor = _FenceableAnnExecutor()
        manual_executor = _ManualExecutor()
        publication_stream = _FakeCudaStream(21)
        consumer_stream = _FakeCudaStream(22)
        allocator = _ReclaimingAllocator(pool)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            kv_publication_stream=publication_stream,
            kv_consumer_streams=(
                publication_stream,
                consumer_stream,
            ),
            async_native_map=True,
            map_stage_extents=2,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertIsNotNone(controller.pending_ann_reclaim)
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(pool.kv_extents, 0)
        self.assertEqual(executor.fence_calls, 0)
        self.assertEqual(executor.async_fence_calls, 1)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertEqual(controller.kv_growth_ann_conflicts, 1)
        self.assertEqual(controller.kv_growth_disjoint_ann_bypasses, 0)
        self.assertEqual(
            controller.kv_growth_native_map_submissions, 0
        )
        self.assertEqual(controller.kv_growth_stream_fences, 0)
        self.assertEqual(
            controller.kv_growth_async_reclaim_fence_submissions, 1
        )
        self.assertNotIn("ann_to_kv", controller.transition_counts)
        self.assertFalse(
            controller.before_ann_launch(
                _IdleScheduler(), requested_bytes=100
            )
        )
        self.assertEqual(
            controller.failure_counts["ann_reclaim_pending"], 1
        )

        # Polling an incomplete aggregate event must not clear Faiss pointers,
        # unmap scratch, or submit the native VMM stage.
        self.assertFalse(controller.poll_kv_growth_completion())
        self.assertIsNotNone(controller.pending_ann_reclaim)
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertEqual(pool.fast_unbind_calls, 0)
        self.assertEqual(pool.synchronous_unbind_calls, 0)
        self.assertEqual(
            controller.kv_growth_native_map_submissions, 0
        )

        executor.async_fence_event.ready = True
        self.assertTrue(controller.poll_kv_growth_completion())
        self.assertIsNone(controller.pending_ann_reclaim)
        self.assertIsNotNone(controller.pending_kv_native_map)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertFalse(pool._faiss_bound)
        self.assertTrue(pool._faiss_scratch_detaching)
        self.assertEqual(pool.fast_unbind_calls, 1)
        self.assertEqual(pool.synchronous_unbind_calls, 0)
        self.assertEqual(pool.release_ann_calls, 0)
        self.assertEqual(
            int(
                controller.pending_kv_native_map[
                    "deferred_release_owner"
                ]
            ),
            2,
        )
        self.assertTrue(
            controller.pending_kv_native_map[
                "deferred_release_submitted"
            ]
        )
        self.assertEqual(
            controller.kv_growth_async_reclaim_fast_detach_uses, 1
        )
        self.assertEqual(
            controller.kv_growth_async_reclaim_fallback_detach_uses, 0
        )
        self.assertEqual(
            controller.kv_growth_native_map_submissions, 1
        )
        self.assertEqual(
            controller.kv_growth_async_reclaim_fence_polls, 2
        )
        self.assertEqual(
            controller.kv_growth_async_reclaim_fence_ready, 1
        )
        self.assertEqual(
            executor.async_fence_event.synchronize_calls, 0
        )
        metrics = controller.metrics()
        self.assertTrue(metrics["kv_growth_async_ann_unmap_configured"])
        self.assertTrue(metrics["kv_growth_async_ann_unmap_enabled"])
        self.assertTrue(
            metrics["kv_growth_deferred_ann_release_pending"]
        )
        self.assertEqual(
            metrics["kv_growth_deferred_ann_release_submissions"], 1
        )
        self.assertEqual(
            metrics["kv_growth_deferred_ann_release_completions"], 0
        )

        manual_executor.future.run()
        self.assertEqual(pool.scratch_chunks, 0)
        self.assertFalse(pool._faiss_scratch_detaching)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            pool.operation_order[:3],
            ["detach_fast", "release", "map"],
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_completions, 1
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_errors, 0
        )
        self.assertEqual(
            len(controller.kv_growth_worker_ann_release_latency_ms), 1
        )
        self.assertTrue(controller.kv_native_map_ready.is_set())
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event"
        ) as event_type, patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            publication_event = _FakeCudaEvent()
            event_type.return_value = publication_event
            self.assertFalse(controller.poll_kv_growth_completion())

        self.assertIsNotNone(controller.pending_kv_native_map)
        self.assertFalse(controller.kv_native_map_ready.is_set())
        self.assertEqual(pool.kv_extents, 2)
        self.assertEqual(
            controller.kv_growth_native_map_completions, 1
        )
        self.assertEqual(
            controller.kv_growth_partial_publications, 1
        )
        self.assertEqual(
            len(controller.kv_growth_first_publish_latency_ms), 1
        )
        self.assertNotIn("ann_to_kv", controller.transition_counts)

        manual_executor.future.run()
        self.assertTrue(controller.kv_native_map_ready.is_set())
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event"
        ) as event_type, patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            final_publication_event = _FakeCudaEvent()
            event_type.return_value = final_publication_event
            self.assertTrue(controller.poll_kv_growth_completion())

        self.assertIsNone(controller.pending_kv_native_map)
        self.assertFalse(controller.kv_native_map_ready.is_set())
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(
            controller.kv_growth_native_map_submissions, 2
        )
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            pool.operation_order,
            ["detach_fast", "release", "map", "map"],
        )
        self.assertEqual(
            controller.kv_growth_native_map_completions, 2
        )
        self.assertEqual(controller.kv_growth_native_map_errors, 0)
        self.assertEqual(controller.transition_counts["ann_to_kv"], 1)
        self.assertEqual(
            list(controller.kv_growth_native_map_latency_ms),
            [12.5, 12.5],
        )
        self.assertEqual(controller.last_kv_growth_stage_count, 2)
        self.assertEqual(controller.kv_growth_completion_notifications, 2)
        self.assertEqual(controller.kv_growth_completion_safe_point_calls, 2)
        self.assertEqual(controller.kv_growth_completion_ready_hits, 2)
        self.assertEqual(
            len(controller.kv_growth_completion_reaction_latency_ms), 2
        )
        self.assertEqual(
            publication_stream.waited_events,
            [
                controller.pending_kv_publications[0]["zero_ready"],
                controller.pending_kv_publications[1]["zero_ready"],
            ],
        )
        self.assertEqual(
            consumer_stream.waited_events,
            [publication_event, final_publication_event],
        )

    def test_async_ann_unmap_ablation_releases_on_scheduler(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        with patch.dict(
            os.environ,
            {"SGLANG_SHARED_HBM_ASYNC_ANN_UNMAP": "0"},
        ):
            controller = SharedHBMController(
                pool=pool,
                token_allocator=_ReclaimingAllocator(pool),
                ann_executor=executor,
                lookahead_ms=50,
                min_residency_ms=0,
                kv_reserve_extents=0,
                async_native_map=True,
                kv_map_executor=manual_executor,
            )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())

        self.assertFalse(controller.async_ann_unmap_configured)
        self.assertFalse(controller.async_ann_unmap_enabled)
        self.assertEqual(pool.fast_unbind_calls, 1)
        self.assertEqual(pool.synchronous_unbind_calls, 0)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(pool.scratch_chunks, 0)
        self.assertFalse(pool._faiss_scratch_detaching)
        self.assertEqual(pool.operation_order, ["detach_fast", "release"])
        self.assertIsNone(
            controller.pending_kv_native_map["deferred_release_owner"]
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_submissions, 0
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_completions, 0
        )

        metrics = controller.metrics()
        self.assertFalse(metrics["kv_growth_async_ann_unmap_configured"])
        self.assertFalse(metrics["kv_growth_async_ann_unmap_enabled"])
        self.assertFalse(
            metrics["kv_growth_deferred_ann_release_pending"]
        )
        self.assertEqual(
            len(controller.kv_growth_scheduler_detach_latency_ms), 1
        )

        manual_executor.future.run()
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )

    def test_deferred_release_failure_is_fail_closed_and_retryable(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.release_ann_error = RuntimeError("native release failed")
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        publication_stream = _FakeCudaStream(31)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            kv_publication_stream=publication_stream,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()
        self.assertFalse(controller.poll_kv_growth_completion())

        self.assertIsNone(controller.pending_ann_reclaim)
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertTrue(pool._faiss_scratch_detaching)
        self.assertEqual(pool.scratch_chunks, 1)
        self.assertEqual(pool.kv_extents, 0)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release"]
        )
        self.assertFalse(
            controller.before_ann_launch(
                _IdleScheduler(), requested_bytes=100
            )
        )
        self.assertEqual(
            controller.limit_ann_microbatch_rows(_IdleScheduler(), 4), 0
        )
        self.assertEqual(
            controller.failure_counts["ann_scratch_detaching"], 1
        )
        metrics = controller.metrics()
        self.assertTrue(
            metrics["kv_growth_deferred_ann_release_pending"]
        )
        self.assertEqual(
            metrics["kv_growth_deferred_ann_release_errors"], 1
        )

        pool.release_ann_error = None
        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        self.assertEqual(pool.fast_unbind_calls, 1)
        manual_executor.future.run()
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event"
        ) as event_type, patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            event_type.return_value = _FakeCudaEvent()
            self.assertTrue(controller.poll_kv_growth_completion())

        self.assertFalse(pool._faiss_scratch_detaching)
        self.assertEqual(pool.release_ann_calls, 2)
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(
            pool.operation_order,
            ["detach_fast", "release", "release", "map"],
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_submissions, 2
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_completions, 1
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_errors, 1
        )

    def test_deferred_release_submit_failure_is_retryable(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        executor = _FenceableAnnExecutor(async_ready=True)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=_RejectingExecutor(),
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertFalse(controller.poll_kv_growth_completion())
        self.assertIsNone(controller.pending_ann_reclaim)
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertTrue(pool._faiss_scratch_detaching)
        self.assertEqual(pool.release_ann_calls, 0)
        self.assertEqual(pool.operation_order, ["detach_fast"])
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_submissions, 0
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_errors, 1
        )

        manual_executor = _ManualExecutor()
        controller.kv_map_executor = manual_executor
        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()

        self.assertFalse(pool._faiss_scratch_detaching)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_submissions, 1
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_completions, 1
        )

    def test_map_failure_after_deferred_release_retries_without_release(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.native_map_error = RuntimeError("native map failed")
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()
        self.assertFalse(controller.poll_kv_growth_completion())

        self.assertIsNone(controller.pending_kv_native_map)
        self.assertFalse(pool._faiss_scratch_detaching)
        self.assertEqual(pool.scratch_chunks, 0)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_completions, 1
        )
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_errors, 0
        )
        self.assertEqual(controller.kv_growth_native_map_errors, 1)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )

        pool.native_map_error = None
        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertIsNotNone(controller.pending_kv_native_map)
        self.assertIsNone(controller.pending_ann_reclaim)
        manual_executor.future.run()

        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(pool.kv_extents, 3)
        self.assertEqual(
            pool.operation_order,
            ["detach_fast", "release", "map", "map"],
        )
        self.assertEqual(executor.async_fence_calls, 1)
        self.assertEqual(
            controller.kv_growth_deferred_ann_release_submissions, 1
        )

    def test_finalize_failure_recovers_without_release_or_native_remap(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        allocator = _RecoverableAllocator(pool)
        allocator.cache.finalize_failures_remaining = 1
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        publication_stream = _FakeCudaStream(41)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            kv_publication_stream=publication_stream,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event",
            side_effect=lambda: _FakeCudaEvent(),
        ), patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            self.assertFalse(controller.poll_kv_growth_completion())
            self.assertIsNotNone(controller.pending_kv_native_map)
            self.assertEqual(
                controller.pending_kv_native_map["post_map"]["phase"],
                "finalize",
            )
            self.assertEqual(
                controller._pending_native_logical_gap_tokens(), 300
            )
            # ensure() must resume the retained post-map phase instead of
            # trusting the already-grown physical snapshot and returning early.
            self.assertTrue(controller.ensure_kv_capacity(101))

        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )
        self.assertEqual(allocator.cache.finalize_calls, 2)
        self.assertEqual(allocator.publish_calls, 1)
        self.assertEqual(controller.kv_growth_native_map_submissions, 1)
        self.assertEqual(controller.kv_growth_native_map_completions, 1)
        metrics = controller.metrics()
        self.assertEqual(metrics["kv_growth_post_map_recovery_attempts"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_successes"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_errors"], 1)
        self.assertEqual(metrics["kv_growth_native_logical_gap_tokens"], 0)
        self.assertEqual(
            metrics["kv_growth_native_logical_gap_max_tokens"], 300
        )

    def test_publish_failure_recovers_without_refinalize_or_native_remap(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        allocator = _RecoverableAllocator(pool)
        allocator.publish_failures_remaining = 1
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        publication_stream = _FakeCudaStream(51)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            kv_publication_stream=publication_stream,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event",
            side_effect=lambda: _FakeCudaEvent(),
        ), patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            self.assertFalse(controller.poll_kv_growth_completion())
            self.assertIsNotNone(controller.pending_kv_native_map)
            self.assertEqual(
                controller.pending_kv_native_map["post_map"]["phase"],
                "publish",
            )
            self.assertEqual(
                controller._pending_native_logical_gap_tokens(), 300
            )
            self.assertTrue(controller.ensure_kv_capacity(101))

        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )
        self.assertEqual(allocator.cache.finalize_calls, 1)
        self.assertEqual(allocator.publish_calls, 2)
        self.assertEqual(controller.kv_growth_native_map_submissions, 1)
        self.assertEqual(controller.kv_growth_native_map_completions, 1)
        self.assertEqual(len(controller.kv_growth_native_map_latency_ms), 1)
        self.assertEqual(len(controller.kv_growth_publication_latency_ms), 1)
        self.assertEqual(controller.kv_growth_stages_completed, 1)
        metrics = controller.metrics()
        self.assertEqual(metrics["kv_growth_post_map_recovery_attempts"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_successes"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_errors"], 1)
        self.assertEqual(
            metrics["kv_growth_native_logical_gap_max_tokens"], 300
        )

    def test_consumer_wait_failure_after_publish_uses_sync_recovery(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        allocator = _RecoverableAllocator(pool)
        executor = _FenceableAnnExecutor(async_ready=True)
        manual_executor = _ManualExecutor()
        publication_stream = _FakeCudaStream(61)
        consumer_stream = _FailingCudaStream(62)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            kv_publication_stream=publication_stream,
            kv_consumer_streams=(consumer_stream,),
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=manual_executor,
        )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertTrue(controller.poll_kv_growth_completion())
        manual_executor.future.run()
        synchronized_devices = []
        with patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.Event",
            side_effect=lambda: _FakeCudaEvent(),
        ), patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.stream",
            return_value=nullcontext(),
        ), patch(
            "sglang.srt.mem_cache.shared_hbm.torch.cuda.synchronize",
            side_effect=lambda device: synchronized_devices.append(device),
        ):
            self.assertTrue(controller.poll_kv_growth_completion())

        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(pool.release_ann_calls, 1)
        self.assertEqual(
            pool.operation_order, ["detach_fast", "release", "map"]
        )
        self.assertEqual(allocator.cache.finalize_calls, 1)
        self.assertEqual(allocator.publish_calls, 1)
        self.assertEqual(allocator.logical_capacity, 400)
        self.assertEqual(controller.kv_growth_native_map_submissions, 1)
        self.assertEqual(controller.kv_growth_native_map_completions, 1)
        self.assertEqual(synchronized_devices, [pool.gpu_id])
        metrics = controller.metrics()
        self.assertEqual(metrics["kv_growth_post_map_recovery_attempts"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_successes"], 1)
        self.assertEqual(metrics["kv_growth_post_map_recovery_errors"], 1)
        self.assertEqual(metrics["kv_growth_native_logical_gap_tokens"], 0)

    def test_ensure_fails_closed_on_untracked_native_logical_gap(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        pool.scratch_chunks = 0
        allocator = _RecoverableAllocator(pool)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            map_stage_extents=0,
            kv_map_executor=_ManualExecutor(),
        )
        # Simulate a lost controller pending record after native VMM mapping.
        pool.kv_extents = 1

        self.assertFalse(controller.ensure_kv_capacity(101))
        self.assertIsNone(controller.pending_kv_native_map)
        self.assertEqual(
            controller.failure_counts["kv_native_logical_gap_untracked"], 1
        )
        metrics = controller.metrics()
        self.assertEqual(metrics["kv_growth_native_logical_gap_tokens"], 100)
        self.assertEqual(metrics["kv_growth_native_logical_gap_events"], 1)
        self.assertEqual(
            metrics["kv_growth_native_logical_gap_max_tokens"], 100
        )

    def test_fast_faiss_detach_ablation_uses_safe_fallback(self):
        pool = _ReclaimingPool()
        pool.kv_extents = 0
        executor = _FenceableAnnExecutor(async_ready=True)
        with patch.dict(
            os.environ,
            {"SGLANG_SHARED_HBM_FAST_FAISS_DETACH": "0"},
        ):
            controller = SharedHBMController(
                pool=pool,
                token_allocator=_ReclaimingAllocator(pool),
                ann_executor=executor,
                lookahead_ms=50,
                min_residency_ms=0,
                kv_reserve_extents=0,
                async_native_map=True,
                kv_map_executor=_ManualExecutor(),
            )

        self.assertTrue(controller.ensure_kv_capacity(101))
        self.assertFalse(controller.fast_faiss_detach_enabled)
        self.assertTrue(controller.poll_kv_growth_completion())
        self.assertEqual(pool.fast_unbind_calls, 0)
        self.assertEqual(pool.synchronous_unbind_calls, 1)
        self.assertEqual(
            controller.kv_growth_async_reclaim_fast_detach_uses, 0
        )
        self.assertEqual(
            controller.kv_growth_async_reclaim_fallback_detach_uses, 1
        )
        metrics = controller.metrics()
        self.assertFalse(metrics["kv_growth_fast_faiss_detach_enabled"])
        self.assertEqual(
            metrics["kv_growth_async_reclaim_fallback_detach_uses"], 1
        )

    def test_async_map_covers_demand_then_coalesces_prefetch(self):
        manual_executor = _ManualExecutor()
        pool = _ReclaimingPool()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=_ReclaimingAllocator(pool),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            async_native_map=True,
            map_stage_extents=2,
            kv_map_executor=manual_executor,
        )
        pending = {
            "mapped_extents": 0,
            "raw_target": 7,
            "target": 16,
            "stages_completed": 0,
        }

        controller._submit_kv_native_map_stage(pending)
        self.assertEqual(manual_executor.future.args, (7,))

        pending["mapped_extents"] = 7
        pending["stages_completed"] = 1
        controller._submit_kv_native_map_stage(pending)
        self.assertEqual(manual_executor.future.args, (16,))

    def test_reverse_handoff_drains_async_publication(self):
        controller = SharedHBMController(
            pool=_FakePool(),
            token_allocator=_FakeAllocator(),
            ann_executor=None,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
        )
        pending = _FakeCudaEvent()
        controller.pending_kv_publications.append(
            {"ready": pending, "zero_ready": object(), "capacity": 200}
        )

        controller._drain_pending_kv_publications()

        self.assertEqual(pending.synchronize_calls, 1)
        self.assertEqual(
            controller.kv_growth_reverse_drain_event_waits, 1
        )
        self.assertFalse(controller.pending_kv_publications)


class TestClusterLeasePolicy(unittest.TestCase):
    def test_cpu_queue_pressure_increases_cluster_promotion_value(self):
        low_pressure = ClusterLeasePolicy(
            np.asarray([100], dtype=np.int64),
            offline_prior=np.asarray([1.0], dtype=np.float64),
            promotion_horizon_batches=4.0,
        )
        high_pressure = ClusterLeasePolicy(
            np.asarray([100], dtype=np.int64),
            offline_prior=np.asarray([1.0], dtype=np.float64),
            promotion_horizon_batches=4.0,
        )

        low_pressure.observe_search_latency(
            cpu_latency_ms=10.0,
            cpu_probes=10,
            gpu_latency_ms=1.0,
            gpu_probes=10,
            cpu_queue_ms=0.0,
        )
        high_pressure.observe_search_latency(
            cpu_latency_ms=10.0,
            cpu_probes=10,
            gpu_latency_ms=1.0,
            gpu_probes=10,
            cpu_queue_ms=90.0,
        )

        low_savings = low_pressure.expected_latency_savings_ms([0])[0]
        high_savings = high_pressure.expected_latency_savings_ms([0])[0]
        self.assertGreater(high_savings, low_savings * 2.0)
        self.assertLess(high_savings, low_savings * 3.0)
        metrics = high_pressure.metrics()
        self.assertEqual(metrics["cpu_service_ms_per_probe_ewma"], 1.0)
        self.assertEqual(metrics["cpu_queue_ms_per_probe_ewma"], 9.0)
        self.assertEqual(metrics["cpu_ms_per_probe_ewma"], 10.0)
        self.assertEqual(metrics["promotion_queue_credit_ms_per_probe"], 1.0)
        self.assertEqual(metrics["promotion_saved_ms_per_probe"], 1.9)

    def test_cpu_fallback_pressure_promotes_cold_cluster(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 100, 100], dtype=np.int64),
            heat_decay=0.5,
            miss_boost=4.0,
            offline_prior=np.zeros(3, dtype=np.float64),
        )

        policy.observe_probes(
            np.asarray([2, 2], dtype=np.int64),
            np.asarray([True, True], dtype=np.bool_),
            now_ms=10.0,
        )

        order = policy.promotion_order([0, 1, 2])
        self.assertEqual(order.tolist(), [2])
        self.assertEqual(policy.metrics()["observed_misses"], 2)
        self.assertEqual(policy.miss_ewma.tolist(), [0.0, 0.0, 2.0])

    def test_historical_miss_debt_does_not_count_as_future_savings(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 100], dtype=np.int64),
            heat_decay=0.5,
            miss_boost=8.0,
            offline_prior=np.zeros(2, dtype=np.float64),
            promotion_horizon_batches=4.0,
        )
        policy.observe_probes(
            np.asarray([1], dtype=np.int64),
            np.asarray([True], dtype=np.bool_),
        )
        policy.heat_ewma[1] = 0.0

        self.assertEqual(policy.promotion_order([0, 1]).tolist(), [1])
        self.assertEqual(policy.expected_latency_savings_ms([1])[0], 0.0)

    def test_startup_value_does_not_double_penalize_large_hot_cluster(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 1000], dtype=np.int64),
            offline_prior=np.asarray([1.0, 2.0], dtype=np.float64),
        )

        self.assertEqual(policy.promotion_order([0, 1]).tolist(), [1, 0])

    def test_locality_shift_detector_waits_for_baseline_and_detects_jump(self):
        policy = ClusterLeasePolicy(
            np.asarray([100], dtype=np.int64),
            heat_decay=0.98,
            miss_boost=4.0,
        )
        ids = np.zeros(100, dtype=np.int64)
        baseline_misses = np.zeros(100, dtype=np.bool_)
        baseline_misses[:20] = True
        for _ in range(32):
            policy.observe_probes(ids, baseline_misses)

        self.assertFalse(policy.locality_shift_detected())
        self.assertAlmostEqual(policy.miss_ratio_baseline, 0.2)

        shifted_misses = np.zeros(100, dtype=np.bool_)
        shifted_misses[:70] = True
        for _ in range(5):
            policy.observe_probes(ids, shifted_misses)

        self.assertTrue(policy.locality_shift_detected())
        metrics = policy.metrics()
        self.assertTrue(metrics["miss_ratio_shift_active"])
        self.assertEqual(metrics["miss_ratio_shift_events"], 1)
        self.assertGreater(
            metrics["miss_ratio_fast_ewma"],
            metrics["miss_ratio_shift_threshold"],
        )

    def test_min_residency_and_hysteresis_prevent_churn(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 100], dtype=np.int64),
            heat_decay=0.5,
            miss_boost=4.0,
            promotion_hysteresis=1.5,
            min_residency_ms=100.0,
            offline_prior=np.zeros(2, dtype=np.float64),
        )
        policy.heat_ewma[:] = [1.0, 1.1]
        policy.mark_resident([0], now_ms=0.0)

        self.assertIsNone(
            policy.replacement_victim(1, [0], now_ms=50.0)
        )
        self.assertIsNone(
            policy.replacement_victim(1, [0], now_ms=150.0)
        )

        policy.heat_ewma[1] = 2.0
        self.assertEqual(
            policy.replacement_victim(1, [0], now_ms=150.0), 0
        )
        metrics = policy.metrics()
        self.assertEqual(metrics["residency_rejections"], 1)
        self.assertEqual(metrics["hysteresis_rejections"], 1)

    def test_reload_cost_rejects_one_shot_promotion(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 100], dtype=np.int64),
            offline_prior=np.asarray([0.0, 1.0], dtype=np.float64),
            promotion_horizon_batches=1.0,
        )
        policy.observe_search_latency(
            cpu_latency_ms=10.0,
            cpu_probes=1,
            gpu_latency_ms=1.0,
            gpu_probes=1,
        )
        policy.observe_reload(20.0)

        self.assertEqual(policy.promotion_order([1]).size, 0)

    def test_candidate_plan_accounts_for_victim_value_and_cluster_size(self):
        policy = ClusterLeasePolicy(
            np.asarray([100, 100, 600], dtype=np.int64),
            promotion_hysteresis=1.15,
            promotion_horizon_batches=10.0,
            min_residency_ms=0.0,
            offline_prior=np.zeros(3, dtype=np.float64),
        )
        policy.heat_ewma[:] = [1.0, 2.0, 20.0]
        policy.observe_search_latency(
            cpu_latency_ms=1.0,
            cpu_probes=1,
            gpu_latency_ms=0.0,
            gpu_probes=1,
        )
        policy.mark_resident([0, 1], now_ms=0.0)
        counters_before = (
            policy.hysteresis_rejections,
            policy.residency_rejections,
        )

        plan = policy.predict_refill_plan(
            [2],
            [0, 1],
            available_bytes=0,
            max_lists=1,
            now_ms=100.0,
        )

        self.assertEqual(plan.candidate_ids, (2,))
        self.assertEqual(plan.victim_ids, ((0, 1),))
        self.assertEqual(plan.candidate_bytes, (1112,))
        self.assertEqual(plan.victim_bytes, (1224,))
        self.assertAlmostEqual(plan.expected_benefit_ms, 1170.0)
        self.assertEqual(
            (
                policy.hysteresis_rejections,
                policy.residency_rejections,
            ),
            counters_before,
        )


class TestSharedIVFRouting(unittest.TestCase):
    def test_mapping_order_seeds_local_cluster_prior(self):
        entries = np.asarray(
            [
                [2, 0, 1],
                [0, 0, 2],
                [3, 0, 0],
            ],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ordered.imap")
            entries.tofile(path)
            mapping = load_vectorlite_mapping(path, nlist=4)

        prior = mapping.offline_prior(shard_id=0, local_nlist=3)
        np.testing.assert_allclose(prior, [1.0 / 3.0, 1.0, 0.5])

    def test_evicted_hot_probe_moves_to_cpu_with_original_id(self):
        mapping = VectorLiteMapping(
            cid_lut=np.asarray([0, 1, 2, 0], dtype=np.int64),
            shard_lut=np.asarray([-1, -1, -1, 0], dtype=np.int32),
            num_entries=1,
            shard_ids=(0,),
        )
        distances = np.asarray([[0.1, 0.2]], dtype=np.float32)
        coarse_ids = np.asarray([[3, 1]], dtype=np.int64)

        resident = route_vectorlite_probes(
            distances,
            coarse_ids,
            mapping,
            shard_id=0,
            resident_local_ids=np.asarray([True], dtype=np.bool_),
        )
        self.assertEqual(resident.hot_ids.tolist(), [[0]])
        self.assertEqual(resident.cold_ids.tolist(), [[1]])

        evicted = route_vectorlite_probes(
            distances,
            coarse_ids,
            mapping,
            shard_id=0,
            resident_local_ids=np.asarray([False], dtype=np.bool_),
        )
        self.assertEqual(evicted.hot_width, 0)
        self.assertEqual(evicted.cold_ids.tolist(), [[3, 1]])


    def test_offline_arena_updates_heat_without_locality_miss(self):
        class _Quantizer:
            @staticmethod
            def search(_queries, _nprobe):
                return (
                    np.asarray([[0.1]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                )

        class _Policy:
            def __init__(self):
                self.observations = []

            def observe_probes(self, local_ids, missed=None):
                self.observations.append(
                    (
                        np.asarray(local_ids).copy(),
                        None if missed is None else np.asarray(missed).copy(),
                    )
                )

        executor = AnnExecutor.__new__(AnnExecutor)
        executor._quantizer = _Quantizer()
        executor._warned_non_hnsw_quantizer = True
        executor.nprobe = 1
        executor._mapping = VectorLiteMapping(
            cid_lut=np.asarray([0], dtype=np.int64),
            shard_lut=np.asarray([0], dtype=np.int32),
            num_entries=1,
            shard_ids=(0,),
        )
        executor.hot_index_shard_id = 0
        executor._shared_ivf_resident = np.asarray(
            [False], dtype=np.bool_
        )
        executor._shared_ivf_mapped_bytes = 0
        executor._shared_ivf_reference_resident_count = 1
        executor._shared_ivf_policy = _Policy()
        executor._shared_ivf_heat_ewma = None
        executor._shared_ivf_cpu_fallback_probes = 0
        executor._shared_ivf_ownership_miss_suppressed = 0
        executor._shared_ivf_capacity_miss_suppressed = 0

        executor._quantize_hot_probes(
            np.asarray([[1.0]], dtype=np.float32)
        )

        observed_ids, observed_misses = (
            executor._shared_ivf_policy.observations[0]
        )
        self.assertEqual(observed_ids.tolist(), [0])
        self.assertIsNone(observed_misses)
        self.assertEqual(
            executor._shared_ivf_ownership_miss_suppressed, 1
        )

        executor._shared_ivf_mapped_bytes = 100
        executor._quantize_hot_probes(
            np.asarray([[1.0]], dtype=np.float32)
        )
        self.assertIsNone(
            executor._shared_ivf_policy.observations[1][1]
        )
        self.assertEqual(
            executor._shared_ivf_capacity_miss_suppressed, 1
        )

        executor._shared_ivf_reference_resident_count = 0
        executor._quantize_hot_probes(
            np.asarray([[1.0]], dtype=np.float32)
        )
        self.assertEqual(
            executor._shared_ivf_policy.observations[2][1].tolist(),
            [True],
        )


class TestSharedIVFArenaHeadroom(unittest.TestCase):
    def test_initial_arena_reserves_default_faiss_scratch(self):
        pool = SimpleNamespace(
            total_bytes=5 << 30,
            chunk_bytes=8 << 20,
        )

        arena_bytes = AnnExecutor._initial_shared_ivf_arena_bytes(pool)

        self.assertEqual(arena_bytes, int(3.5 * (1 << 30)))

    def test_reload_prediction_uses_post_reclaim_available_budget(self):
        class _Policy:
            def __init__(self):
                self.available_bytes = []

            def predict_refill_plan(
                self,
                _inactive_ids,
                _resident_ids,
                *,
                available_bytes,
                max_lists,
            ):
                self.available_bytes.append((available_bytes, max_lists))
                return object()

        executor = AnnExecutor.__new__(AnnExecutor)
        executor._shared_ivf_resident = np.asarray(
            [True, False, False], dtype=np.bool_
        )
        executor._shared_ivf_list_bytes = np.asarray(
            [100, 100, 100], dtype=np.int64
        )
        executor._shared_ivf_mapped_bytes = 300
        executor._shared_ivf_current_bytes = 200
        executor._shared_ivf_policy = _Policy()
        executor._shared_ivf_method = lambda _name: lambda: 100

        executor.predict_shared_ivf_reload(
            2, available_bytes_override=250
        )
        executor.predict_shared_ivf_reload(
            1, available_bytes_override=50
        )

        self.assertEqual(
            executor._shared_ivf_policy.available_bytes,
            [(250, 2), (100, 1)],
        )

    def test_bounded_reload_can_reserve_space_for_later_refills(self):
        class _InvertedLists:
            @staticmethod
            def list_size(_list_id):
                return 10

        class _SourceIVF:
            code_size = 1
            invlists = _InvertedLists()

        class _Swig:
            @staticmethod
            def extract_index_ivf(_index):
                return _SourceIVF()

        class _GpuIndex:
            @staticmethod
            def getNumLists():
                return 4

        class _Pool:
            chunk_bytes = 100

            def __init__(self):
                self.requested_bytes = None

            def map_ann(self, _owner, requested_bytes):
                self.requested_bytes = requested_bytes
                return 1, requested_bytes

        pool = _Pool()
        executor = AnnExecutor.__new__(AnnExecutor)
        executor.hot_index_mode = True
        executor._gpu_source_index = object()
        executor._gpu_index = _GpuIndex()
        executor._swig = _Swig()
        executor._shared_ivf_resident = np.zeros(4, dtype=np.bool_)
        executor._shared_ivf_heat_ewma = None
        executor._shared_ivf_list_bytes = None
        executor._shared_ivf_evicted_bytes = 0
        executor._shared_ivf_reload_latency_ms = []
        executor._shared_ivf_reloaded_bytes = 0

        available_bytes = 250
        methods = {
            "setExternalInvertedListMemory": lambda _address, _mapped: None,
            "reloadInvertedList": lambda _source, _list_id: None,
            "getExternalInvertedListMemoryAvailable": lambda: available_bytes,
        }
        executor._shared_ivf_method = methods.__getitem__

        resident = executor._install_shared_ivf_arena(
            pool,
            max_lists=1,
            arena_bytes=1000,
        )

        self.assertEqual(pool.requested_bytes, 1000)
        self.assertEqual(resident, 1)
        self.assertEqual(executor.shared_ivf_resident_count, 1)
        self.assertTrue(executor._shared_ivf_repack_complete)

    def test_rebind_respects_deadline_bounded_refills(self):
        calls = []
        executor = AnnExecutor.__new__(AnnExecutor)
        executor._shared_ivf_mapped_bytes = 0
        executor._shared_ivf_repack_complete = False
        executor._shared_ivf_resident = np.zeros(4, dtype=np.bool_)
        executor._install_shared_ivf_arena = (
            lambda _pool, max_lists, arena_bytes: calls.append(
                (max_lists, arena_bytes)
            )
            or 3
        )

        resident = executor.reload_shared_ivf_arena(
            object(), max_lists=1, arena_bytes=1000
        )

        self.assertEqual(resident, 3)
        self.assertEqual(calls, [(1, 1000)])


    def test_equal_count_replacement_is_a_successful_reload(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 1.0
            shared_ivf_reloaded_bytes = 100

            def reload_shared_ivf_arena(self, _pool, max_lists):
                self.shared_ivf_reloaded_bytes = 200
                return self.shared_ivf_resident_count

        recorded = []
        failures = []
        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = recorded.append
        controller._record_failure = failures.append
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=50.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        reloaded = controller._maybe_reload_ivf(scheduler)

        self.assertTrue(reloaded)
        self.assertEqual(controller.reload_waits, 1)
        self.assertEqual(recorded, ["ivf_reload"])
        self.assertEqual(failures, [])

    def test_ivf_reload_coalesces_miss_debt_and_bounds_batch(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 1.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(observed_misses=63)
                self.max_lists = []

            def reload_shared_ivf_arena(self, _pool, max_lists):
                self.max_lists.append(max_lists)
                self.shared_ivf_reloaded_bytes += 10
                return self.shared_ivf_resident_count

        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = lambda _key: None
        controller._record_failure = lambda _key: None
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 8
        controller.last_ivf_reload_observed_misses = 0
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_attempts = 0
        controller.ivf_reload_lists_requested = 0
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=50.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertFalse(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(controller.ann_executor.max_lists, [])
        self.assertEqual(controller.ivf_reload_coalesced, 1)

        controller.ann_executor._shared_ivf_policy.observed_misses = 128
        self.assertTrue(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(controller.ann_executor.max_lists, [8])
        self.assertEqual(controller.last_ivf_reload_observed_misses, 128)
        self.assertEqual(controller.ivf_reload_attempts, 1)
        self.assertEqual(controller.ivf_reload_lists_requested, 8)

        self.assertFalse(controller._maybe_reload_ivf(scheduler))
        self.assertTrue(
            controller._maybe_reload_ivf(
                scheduler,
                allow_existing_heat=True,
            )
        )
        self.assertEqual(controller.ann_executor.max_lists, [8, 8])

    def test_candidate_prediction_selects_best_profitable_prefix(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 10.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(
                    observed_misses=128,
                    miss_boost=4.0,
                    promotion_hysteresis=1.15,
                    locality_shift_detected=lambda: True,
                )
                self.calls = []
                self.predicted_available_bytes = []

            def predict_shared_ivf_reload(
                self,
                _max_lists,
                *,
                available_bytes_override=None,
            ):
                self.predicted_available_bytes.append(
                    available_bytes_override
                )
                return ClusterRefillPlan(
                    candidate_ids=(7, 8),
                    victim_ids=((1,), (2,)),
                    incremental_benefit_ms=(50.0, 5.0),
                    candidate_bytes=(100, 100),
                    victim_bytes=(100, 100),
                )

            def reload_shared_ivf_arena(
                self, _pool, max_lists, planned_candidate_ids
            ):
                self.calls.append((max_lists, planned_candidate_ids))
                self.shared_ivf_reloaded_bytes += 10
                return self.shared_ivf_resident_count

        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = lambda _key: None
        controller._record_failure = lambda _key: None
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 8
        controller.last_ivf_reload_observed_misses = 0
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_attempts = 0
        controller.ivf_reload_lists_requested = 0
        controller.ivf_reload_async_latency_ms = [10.0]
        controller.ivf_reload_async_list_counts = [8]
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=100.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertTrue(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(controller.ann_executor.calls, [(1, (7,))])
        self.assertEqual(controller.ivf_reload_lists_requested, 1)
        self.assertEqual(controller.ivf_reload_candidate_predictions, 1)
        self.assertEqual(controller.ivf_reload_candidate_lists_selected, 1)
        self.assertAlmostEqual(
            controller.ivf_reload_candidate_last_benefit_ms, 50.0
        )
        self.assertAlmostEqual(
            controller.ivf_reload_candidate_last_cost_ms, 10.0
        )

    def test_idle_refill_fills_free_arena_below_copy_cost(self):
        class _Executor:
            shared_ivf_resident_count = 0
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 10.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(
                    observed_misses=128,
                    miss_boost=4.0,
                    promotion_hysteresis=1.15,
                    locality_shift_detected=lambda: True,
                )
                self.calls = []

            @staticmethod
            def predict_shared_ivf_reload(_max_lists):
                return ClusterRefillPlan(
                    candidate_ids=(7,),
                    victim_ids=((),),
                    incremental_benefit_ms=(1.0,),
                    candidate_bytes=(100,),
                    victim_bytes=(0,),
                )

            def reload_shared_ivf_arena(
                self, _pool, max_lists, planned_candidate_ids
            ):
                self.calls.append((max_lists, planned_candidate_ids))
                self.shared_ivf_reloaded_bytes += 10
                return 1

        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = lambda _key: None
        controller._record_failure = lambda _key: None
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 4
        controller.last_ivf_reload_observed_misses = 128
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_attempts = 0
        controller.ivf_reload_lists_requested = 0
        controller.ivf_reload_async_latency_ms = [10.0]
        controller.ivf_reload_async_list_counts = [1]
        controller.ivf_idle_free_capacity_refills = 0
        controller.ivf_idle_free_capacity_lists = 0
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=250.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertTrue(
            controller._maybe_reload_ivf(
                scheduler,
                allow_existing_heat=True,
                fill_free_capacity=True,
            )
        )
        self.assertEqual(controller.ann_executor.calls, [(1, (7,))])
        self.assertEqual(controller.ivf_idle_free_capacity_refills, 1)
        self.assertEqual(controller.ivf_idle_free_capacity_lists, 1)

    def test_empty_ivf_recovery_reserves_all_safe_reclaimed_space(self):
        class _Executor:
            shared_ivf_resident_count = 0
            shared_ivf_total_count = 4
            shared_ivf_reload_p95_ms = 1.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(
                    observed_misses=128,
                    miss_boost=4.0,
                    promotion_hysteresis=1.0,
                    locality_shift_detected=lambda: True,
                )
                self.calls = []

                self.predicted_available_bytes = []

            def predict_shared_ivf_reload(
                self,
                _max_lists,
                *,
                available_bytes_override=None,
            ):
                self.predicted_available_bytes.append(
                    available_bytes_override
                )
                return ClusterRefillPlan(
                    candidate_ids=(3,),
                    victim_ids=((),),
                    incremental_benefit_ms=(10.0,),
                    candidate_bytes=(100,),
                    victim_bytes=(0,),
                )

            def reload_shared_ivf_arena(
                self,
                _pool,
                max_lists,
                planned_candidate_ids,
                arena_bytes,
            ):
                self.calls.append(
                    (max_lists, planned_candidate_ids, arena_bytes)
                )
                self.shared_ivf_resident_count = 1
                self.shared_ivf_reloaded_bytes += 10
                return 1

        pool = _ReclaimingPool()
        pool.mode = "ivf"
        allocator = _ReclaimingAllocator(pool)
        executor = _Executor()
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=executor,
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
            ivf_reload_min_misses=1,
            ivf_reload_max_lists=4,
        )
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=50.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
            tree_cache=None,
        )

        self.assertTrue(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(pool.kv_extents, 0)
        self.assertEqual(
            executor.calls,
            [(1, (3,), 300)],
        )
        self.assertEqual(executor.predicted_available_bytes, [300])
        self.assertEqual(controller.kv_to_ivf_extents_reclaimed, 3)
        self.assertEqual(controller.last_kv_to_ivf_arena_bytes, 300)

    def test_ivf_recovery_keeps_unbound_scratch_reserve(self):
        pool = _ReclaimingPool()
        pool.mode = "ivf"
        pool.kv_extents = 3
        pool.scratch_chunks = 0
        pool._faiss_bound = False
        pool._last_faiss_mapping = (0, 0)
        pool._last_faiss_requested_bytes = 100
        allocator = _ReclaimingAllocator(pool)
        controller = SharedHBMController(
            pool=pool,
            token_allocator=allocator,
            ann_executor=SimpleNamespace(),
            lookahead_ms=50,
            min_residency_ms=0,
            kv_reserve_extents=0,
        )
        scheduler = SimpleNamespace(tree_cache=None)

        arena_bytes = controller._prepare_ivf_recovery_arena(
            scheduler,
            predicted_kv_tokens=0,
        )

        self.assertEqual(pool.kv_extents, 0)
        self.assertEqual(arena_bytes, 300)
        self.assertEqual(
            controller._ivf_scratch_reserve_bytes(pool.snapshot()),
            100,
        )

    def test_stable_distribution_allows_free_space_but_not_replacement(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 1.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(
                    observed_misses=128,
                    miss_boost=4.0,
                    promotion_hysteresis=1.0,
                    locality_shift_detected=lambda: False,
                )
                self.calls = []

            @staticmethod
            def predict_shared_ivf_reload(_max_lists):
                return ClusterRefillPlan(
                    candidate_ids=(7, 8),
                    victim_ids=((), (2,)),
                    incremental_benefit_ms=(50.0, 50.0),
                    candidate_bytes=(100, 100),
                    victim_bytes=(0, 100),
                )

            def reload_shared_ivf_arena(
                self, _pool, max_lists, planned_candidate_ids
            ):
                self.calls.append((max_lists, planned_candidate_ids))
                self.shared_ivf_resident_count += max_lists
                self.shared_ivf_reloaded_bytes += 10
                return self.shared_ivf_resident_count

        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = lambda _key: None
        controller._record_failure = lambda _key: None
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 8
        controller.last_ivf_reload_observed_misses = 0
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_async_latency_ms = []
        controller.ivf_reload_async_list_counts = []
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=100.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertTrue(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(controller.ann_executor.calls, [(1, (7,))])

    def test_candidate_prediction_rejects_unprofitable_exchange(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 10.0
            shared_ivf_reloaded_bytes = 100
            _shared_ivf_policy = SimpleNamespace(
                observed_misses=128,
                miss_boost=4.0,
                promotion_hysteresis=1.15,
                locality_shift_detected=lambda: True,
            )

            @staticmethod
            def predict_shared_ivf_reload(_max_lists):
                return ClusterRefillPlan(
                    candidate_ids=(7,),
                    victim_ids=((1,),),
                    incremental_benefit_ms=(5.0,),
                    candidate_bytes=(100,),
                    victim_bytes=(100,),
                )

            @staticmethod
            def reload_shared_ivf_arena(*_args, **_kwargs):
                raise AssertionError("unprofitable plan must not execute")

        failures = []
        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(mode="ivf")
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record_failure = failures.append
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 8
        controller.last_ivf_reload_observed_misses = 0
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_async_latency_ms = []
        controller.ivf_reload_async_list_counts = []
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=100.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertFalse(controller._maybe_reload_ivf(scheduler))
        self.assertEqual(failures, ["ivf_reload_candidate"])
        self.assertEqual(controller.last_ivf_reload_observed_misses, 0)
        self.assertEqual(controller.ivf_reload_candidate_gated, 1)

    def test_ivf_reload_runs_async_and_routes_ann_to_cpu_until_complete(self):
        class _Executor:
            shared_ivf_resident_count = 30
            shared_ivf_total_count = 100
            shared_ivf_reload_p95_ms = 100.0
            shared_ivf_reloaded_bytes = 100

            def __init__(self):
                self._shared_ivf_policy = SimpleNamespace(observed_misses=128)
                self.max_lists = []

            def reload_shared_ivf_arena(self, _pool, max_lists):
                self.max_lists.append(max_lists)
                self.shared_ivf_reloaded_bytes += 10
                return self.shared_ivf_resident_count

        class _Worker:
            def __init__(self):
                self.future = None
                self.job = None

            def submit(self, function, *args, **kwargs):
                self.future = Future()
                self.job = (function, args, kwargs)
                return self.future

            def run(self):
                function, args, kwargs = self.job
                self.future.set_result(function(*args, **kwargs))

        recorded = []
        failures = []
        worker = _Worker()
        controller = SharedHBMController.__new__(SharedHBMController)
        controller.pool = SimpleNamespace(
            mode="ivf", snapshot=lambda: SimpleNamespace()
        )
        controller.ann_executor = _Executor()
        controller.token_allocator = SimpleNamespace(
            current_mapped_capacity=lambda: 1000
        )
        controller.predict_kv_tokens = lambda _scheduler: 0
        controller._record = recorded.append
        controller._record_failure = failures.append
        controller.reload_waits = 0
        controller.handoff_latency_ms = []
        controller.ivf_reload_min_misses = 64
        controller.ivf_reload_max_lists = 8
        controller.last_ivf_reload_observed_misses = 0
        controller.ivf_reload_coalesced = 0
        controller.ivf_reload_attempts = 0
        controller.ivf_reload_lists_requested = 0
        controller.pending_ivf_reload = None
        controller.async_native_map = True
        controller.kv_map_executor = worker
        controller.ivf_reload_async_submissions = 0
        controller.ivf_reload_async_completions = 0
        controller.ivf_reload_async_errors = 0
        controller.ivf_reload_async_cpu_fallbacks = 0
        controller.ivf_reload_async_deadline_bypasses = 0
        controller.ivf_reload_async_latency_ms = []
        scheduler = SimpleNamespace(
            rag_ann_deadline_ms=50.0,
            rag_ann_safety_margin_ms=0.0,
            rag_ann_gpu_latency_ms=[],
        )

        self.assertFalse(controller._maybe_reload_ivf(scheduler))
        self.assertIsNotNone(controller.pending_ivf_reload)
        self.assertEqual(controller.ann_executor.max_lists, [])
        self.assertFalse(
            controller.before_ann_launch(scheduler, num_queries=3)
        )
        self.assertEqual(controller.ivf_reload_async_cpu_fallbacks, 3)

        worker.run()
        self.assertTrue(controller._poll_pending_ivf_reload())
        self.assertIsNone(controller.pending_ivf_reload)
        self.assertEqual(controller.ann_executor.max_lists, [1])
        self.assertEqual(controller.ivf_reload_async_deadline_bypasses, 1)
        self.assertEqual(controller.reload_waits, 1)
        self.assertEqual(controller.ivf_reload_async_submissions, 1)
        self.assertEqual(controller.ivf_reload_async_completions, 1)
        self.assertEqual(controller.ivf_reload_async_errors, 0)
        self.assertEqual(controller.handoff_latency_ms, [])
        self.assertIn("ivf_reload", recorded)
        self.assertIn("ivf_reload_async", recorded)
        self.assertEqual(failures, ["ivf_reload_inflight"])

class TestFaissModuleCompat(unittest.TestCase):
    def test_prefers_proxy_api_and_falls_back_to_raw_extension(self):
        raw = SimpleNamespace(
            GpuIndex_search="raw-search",
            shared_name="raw-value",
        )
        proxy = SimpleNamespace(
            _swigfaiss_avx2=raw,
            StandardGpuResources="proxy-resource",
            shared_name="proxy-value",
        )
        compat = _FaissModuleCompat(proxy)

        self.assertEqual(compat.StandardGpuResources, "proxy-resource")
        self.assertEqual(compat.shared_name, "proxy-value")
        self.assertEqual(compat.GpuIndex_search, "raw-search")
        with self.assertRaises(AttributeError):
            _ = compat.missing_name


class TestAnnVectorLiteCpuThreads(unittest.TestCase):
    def test_cold_search_sets_faiss_threads_inside_worker(self):
        thread_counts = []

        class _CpuIndex:
            nprobe = 0

            def search_preassigned(self, queries, topk, ids, distances):
                shape = (queries.shape[0], topk)
                return (
                    np.zeros(shape, dtype=np.float32),
                    np.zeros(shape, dtype=np.int64),
                )

        executor = AnnExecutor.__new__(AnnExecutor)
        executor._swig = SimpleNamespace(
            omp_set_num_threads=lambda count: thread_counts.append(count)
        )
        executor._cpu_index = _CpuIndex()
        executor._cpu_search_lock = nullcontext()
        executor.cpu_search_threads = 8
        executor.topk = 2
        routed = SimpleNamespace(
            cold_width=1,
            cold_ids=np.zeros((1, 1), dtype=np.int64),
            cold_distances=np.zeros((1, 1), dtype=np.float32),
        )

        _, _, _, queue_wait_ms = executor._search_cold_preassigned(
            np.zeros((1, 4), dtype=np.float32), routed, enqueued_at=0.0
        )

        self.assertGreater(queue_wait_ms, 0.0)
        self.assertEqual(thread_counts, [8])
        self.assertEqual(executor._cpu_index.nprobe, 1)


class TestAnnGreenContextCompatibility(unittest.TestCase):
    @staticmethod
    def _executor():
        executor = AnnExecutor.__new__(AnnExecutor)
        executor.gpu_id = 0
        executor.stream_priority = -5
        executor._warned_legacy_greenctx_stream_priority = False
        return executor

    def test_sibling_stream_passes_priority_to_current_api(self):
        calls = []
        expected = object()
        spatial = SimpleNamespace(
            create_greenctx_stream_like=lambda stream, device, **kwargs: (
                calls.append((stream, device, kwargs)) or expected
            )
        )

        result = self._executor()._create_greenctx_stream_like(spatial, 10)

        self.assertIs(result, expected)
        self.assertEqual(calls, [(10, 0, {"priority": -5})])

    def test_sibling_stream_retries_legacy_api_without_priority(self):
        calls = []
        expected = object()

        def create(stream, device, **kwargs):
            calls.append((stream, device, kwargs))
            if kwargs:
                raise TypeError(
                    "create_greenctx_stream_like() got an unexpected keyword "
                    "argument 'priority'"
                )
            return expected

        executor = self._executor()
        result = executor._create_greenctx_stream_like(
            SimpleNamespace(create_greenctx_stream_like=create), 10
        )

        self.assertIs(result, expected)
        self.assertEqual(
            calls,
            [(10, 0, {"priority": -5}), (10, 0, {})],
        )
        self.assertTrue(executor._warned_legacy_greenctx_stream_priority)


class TestAnnSharedHBMStreamInventory(unittest.TestCase):
    @staticmethod
    def _executor():
        executor = AnnExecutor.__new__(AnnExecutor)
        executor._gpu_res = object()
        executor._swig = SimpleNamespace(
            StandardGpuResources_setAlternateStreams2=object()
        )
        executor._current_stream = _FakeCudaStream(1)
        executor._execution_streams_by_parent_ptr = {
            10: executor._current_stream,
            20: _FakeCudaStream(2),
        }
        executor._alternate_streams_by_stream_ptr = {
            1: (_FakeCudaStream(3), _FakeCudaStream(4)),
            2: (_FakeCudaStream(3), _FakeCudaStream(5)),
        }
        executor._shared_hbm_async_copy_stream = _FakeCudaStream(6)
        executor._shared_hbm_async_copy_stream_resolved = True
        return executor

    def test_inventory_includes_async_copy_and_deduplicates_streams(self):
        streams = self._executor().shared_hbm_cuda_streams()
        self.assertEqual(
            [int(stream.cuda_stream) for stream in streams],
            [1, 2, 3, 4, 5, 6],
        )

        executor = self._executor()
        executor._shared_hbm_async_copy_stream = _FakeCudaStream(5)
        streams = executor.shared_hbm_cuda_streams()
        self.assertEqual(
            [int(stream.cuda_stream) for stream in streams],
            [1, 2, 3, 4, 5],
        )

    def test_inventory_wraps_exported_faiss_async_copy_stream(self):
        executor = self._executor()
        executor.gpu_id = 0
        executor._shared_hbm_async_copy_stream = None
        executor._shared_hbm_async_copy_stream_resolved = False
        executor._warned_no_async_copy_stream = False
        calls = []
        executor._swig.StandardGpuResources_getAsyncCopyStreamAsInteger = (
            lambda resources, device: (
                calls.append((resources, device)) or 7
            )
        )

        with patch(
            "sglang.srt.multiplex.ann_executor.torch.cuda.ExternalStream",
            side_effect=lambda pointer, device: _FakeCudaStream(pointer),
        ):
            streams = executor.shared_hbm_cuda_streams()

        self.assertEqual(calls, [(executor._gpu_res, 0)])
        self.assertEqual(
            [int(stream.cuda_stream) for stream in streams],
            [1, 2, 3, 4, 5, 7],
        )

    def test_older_faiss_uses_non_fast_detach_contract(self):
        executor = self._executor()
        executor._shared_hbm_async_copy_stream = None
        executor._shared_hbm_async_copy_stream_resolved = False
        executor._warned_no_async_copy_stream = False
        streams, fast_detach_safe = (
            executor._shared_hbm_stream_inventory()
        )

        self.assertFalse(fast_detach_safe)
        self.assertEqual(
            [int(stream.cuda_stream) for stream in streams],
            [1, 2, 3, 4, 5],
        )

    def test_inventory_rejects_untracked_faiss_alternate_streams(self):
        executor = self._executor()
        executor._swig = SimpleNamespace()
        with self.assertRaisesRegex(
            RuntimeError, "setAlternateStreams2"
        ):
            executor.shared_hbm_cuda_streams()

    def test_async_fence_records_all_sources_without_host_sync(self):
        executor = self._executor()
        maintenance_stream = _FakeCudaStream(99)
        events = [_FakeCudaEvent() for _ in range(7)]
        with patch(
            "sglang.srt.multiplex.ann_executor.torch.cuda.Event",
            side_effect=events,
        ), patch(
            "sglang.srt.multiplex.ann_executor.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            fence = executor.submit_shared_hbm_reclaim_fence(
                maintenance_stream
            )

        self.assertEqual(fence.stream_count, 6)
        self.assertTrue(fence.fast_detach_safe)
        self.assertEqual(
            [
                int(event.recorded_stream.cuda_stream)
                for event in fence.source_events
            ],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            maintenance_stream.waited_events,
            list(fence.source_events),
        )
        self.assertIs(fence.ready_event.recorded_stream, maintenance_stream)
        self.assertEqual(fence.ready_event.synchronize_calls, 0)


class TestSharedHBMSchedulerProgress(unittest.TestCase):
    def test_pending_hbm_work_keeps_side_channel_awake(self):
        pending_states = (
            (object(), None, None),
            (None, object(), None),
            (None, None, object()),
        )
        for reclaim, native, ivf_reload in pending_states:
            with self.subTest(
                reclaim=reclaim is not None,
                native=native is not None,
                ivf_reload=ivf_reload is not None,
            ):
                scheduler = SimpleNamespace(
                    shared_hbm_controller=SimpleNamespace(
                        pending_ann_reclaim=reclaim,
                        pending_kv_native_map=native,
                        pending_ivf_reload=ivf_reload,
                    ),
                    ann_search_pending=(),
                    ann_search_inflight=(),
                    ann_result_collect_queue=None,
                    ann_result_collector_busy=None,
                    ann_result_output_queue=None,
                )
                self.assertTrue(
                    SchedulerMultiplexMixin._ann_side_channel_has_work(
                        scheduler
                    )
                )

    def test_scheduler_safe_point_polls_pending_reclaim(self):
        polls = []
        controller = SimpleNamespace(
            pending_ann_reclaim=object(),
            pending_kv_native_map=None,
            poll_kv_growth_completion=lambda: polls.append(True) or True,
        )
        scheduler = SimpleNamespace(shared_hbm_controller=controller)

        self.assertTrue(
            SchedulerMultiplexMixin._progress_shared_hbm_kv_map(scheduler)
        )
        self.assertEqual(polls, [True])


if __name__ == "__main__":
    unittest.main()
