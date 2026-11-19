"""Benchmark the complete /rag_generate pipeline.

The measured TTFT includes ANN search, document lookup/prompt construction, and
LLM prefill. ITL and TPOT cover the streamed decode phase. After sequential
kernel warmup, target-rate ramp-up requests establish serving concurrency but
are excluded from reported metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

try:
    from .rag_workload import RagWorkload, generate_workload, load_workload
except ImportError:
    from rag_workload import RagWorkload, generate_workload, load_workload


SERVER_INFO_FIELDS = (
    "version",
    "model_path",
    "enable_pdmux",
    "pdmux_config_path",
    "sm_group_num",
    "enable_rag_ann",
    "rag_cpu_retrieval",
    "rag_cpu_retrieval_workers",
    "rag_cpu_retrieval_threads",
    "rag_cpu_retrieval_max_batch_queries",
    "rag_ann_cold_max_batch_queries",
    "rag_ann_schedule",
    "rag_policy_mode",
    "rag_index_nprobe",
    "rag_index_topk",
    "rag_hot_index_path",
    "rag_hot_index_map_path",
    "rag_hot_index_shard_id",
    "rag_shared_hbm_mode",
    "rag_shared_hbm_pool_gib",
    "rag_shared_hbm_chunk_mib",
    "rag_shared_hbm_lookahead_ms",
    "rag_shared_hbm_min_residency_ms",
    "rag_shared_hbm_cluster_heat_decay",
    "rag_shared_hbm_cluster_miss_boost",
    "rag_shared_hbm_cluster_promotion_hysteresis",
    "rag_shared_hbm_kv_reserve_extents",
    "rag_shared_hbm_rerank_prefetch_batches",
    "rag_shared_hbm_async_map",
    "rag_shared_hbm_map_stage_extents",
    "rag_ann_max_inflight",
    "rag_ann_max_batches_per_tick",
    "rag_ann_max_batch_size",
    "rag_ann_max_queued_gpu_ms",
    "rag_ann_max_microbatch_gpu_ms",
    "rag_ann_service_time_percentile",
    "rag_ann_service_time_min_samples",
    "rag_ann_max_batch_wait_ms",
    "rag_ann_batch_wait_min_pending",
    "rag_ann_stream_epoch_ms",
    "rag_ann_launch_site",
    "rag_ann_fast_path_pressure_threshold",
    "rag_ann_fast_path_max_work",
    "rag_ann_fast_path_decode_batch_p95_threshold",
    "rag_prefill_low_load_batch_p95_threshold",
    "rag_prefill_low_load_arrival_rps_threshold",
    "rag_prefill_low_load_reference_tokens",
    "rag_ann_fast_path_decode_queue_p95_ms",
    "rag_ann_decode_batch_bucket_size",
    "rag_ann_decode_only_wait_ms",
    "rag_ann_max_inflight_during_decode",
    "rag_ann_work_conserving_decode_overlap",
    "rag_ann_pre_switch_launch",
    "rag_ann_itl_slo_ms",
    "rag_ann_decode_step_slo_ms",
    "rag_ann_safety_margin_ms",
    "rag_prefill_decode_guard_percentile",
    "rag_ann_urgency_start_ratio",
    "rag_ann_max_decode_slowdown_ratio",
    "rag_ann_max_defer_ms",
    "rag_ann_frontend_arbitration",
    "rag_ann_deadline_ms",
    "rag_prefill_deadline_ms",
    "rag_e2e_slo_ms",
    "rag_ann_overlap_pressure_threshold",
    "rag_ann_max_consecutive_batches",
    "rag_prefill_max_consecutive_chunks",
    "rag_ann_handoff_timeout_ms",
    "rag_ann_lease_frontend_sm",
    "rag_prefill_lease_frontend_sm",
    "rag_partition_hysteresis_bs",
    "rag_partition_min_residency_ms",
    "rag_partition_safe_mixed_hold",
    "rag_partition_safe_mixed_hold_ratio",
    "rag_partition_safe_mixed_min_samples",
    "rag_partition_safe_mixed_max_hold_ms",
    "rag_partition_decode_slo_ms",
    "rag_partition_decode_slo_batch_thresholds",
    "rag_partition_decode_slo_hysteresis_bs",
    "rag_partition_decode_slo_min_residency_ms",
    "rag_partition_decode_slo_initial_calibration_ms",
    "rag_partition_decode_slo_ewma_alpha",
    "rag_partition_slo_floor_enabled",
    "rag_lease_coalesce_min_work",
    "rag_lease_coalesce_max_wait_ms",
    "rag_lease_switch_cost_ms",
    "rag_lease_switch_cost_percentile",
    "rag_lease_round_trip_switch_cost",
    "rag_lease_min_gain_ratio",
    "rag_lease_ann_service_estimate_ms",
    "rag_lease_prefill_service_estimate_ms",
    "rag_prefill_dynamic_lease",
    "rag_prefill_tail_aware_admission",
    "rag_prefill_exclusive_sprint",
    "rag_prefill_lease_min_layers",
    "rag_prefill_lease_max_layers",
    "rag_prefill_lease_decode_ratio",
    "rag_ann_enable_controlled_overlap",
    "rag_ann_serialize_with_prefill",
    "rag_doc_token_cache_size",
    "mem_fraction_static",
    "max_running_requests",
    "max_prefill_tokens",
    "disable_radix_cache",
    "random_seed",
)


@dataclass
class RequestResult:
    request_id: int
    query_id: int
    stage: str = "measure"
    scheduled_arrival_ms: float = 0.0
    dispatch_lag_ms: float = 0.0
    success: bool = False
    error: str = ""
    status: int = 0
    e2e_ms: float = 0.0
    ttft_ms: float = 0.0
    ann_latency_ms: float = 0.0
    ann_queue_wait_ms: float = 0.0
    ann_gpu_latency_ms: float = 0.0
    ann_cpu_queue_wait_ms: float = 0.0
    ann_cpu_search_latency_ms: float = 0.0
    ann_coarse_quantize_latency_ms: float = 0.0
    ann_route_latency_ms: float = 0.0
    ann_merge_latency_ms: float = 0.0
    ann_index_mode: str = "none"
    ann_hot_probe_ratio: float = 0.0
    ann_hot_probe_count: int = 0
    ann_cold_probe_count: int = 0
    ann_batch_size: int = 0
    token_cache_hit: bool = False
    llm_ttft_after_ann_ms: float = 0.0
    tpot_ms: float = 0.0
    itl_ms: List[float] = field(default_factory=list)
    output_tokens: int = 0
    retrieved_ids: List[int] = field(default_factory=list)


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def distribution(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def _linear_slope_ci95(
    x_values: List[float], y_values: List[float]
) -> Dict[str, float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return {"slope": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    centered_x = x - np.mean(x)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return {"slope": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    slope = float(np.dot(centered_x, y - np.mean(y)) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    residuals = y - (intercept + slope * x)
    degrees_of_freedom = len(x_values) - 2
    if degrees_of_freedom <= 0:
        half_width = 0.0
    else:
        residual_variance = float(
            np.dot(residuals, residuals) / degrees_of_freedom
        )
        standard_error = float(np.sqrt(residual_variance / denominator))
        critical = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }.get(degrees_of_freedom, 1.96)
        half_width = critical * standard_error
    return {
        "slope": slope,
        "ci95_low": slope - half_width,
        "ci95_high": slope + half_width,
    }


def summarize_stationarity(
    results: List[RequestResult],
    *,
    offered_request_rate_qps: float,
    duration_s: float,
    normalize_by_observed_trace_rate: bool = False,
    windows: int = 8,
    min_achieved_offered_ratio: float = 0.95,
    max_drain_ratio: float = 0.10,
    max_backlog_growth_fraction: float = 0.05,
) -> Dict[str, Any]:
    measured = sorted(results, key=lambda item: item.scheduled_arrival_ms)
    if not measured:
        return {
            "available": False,
            "passed": False,
            "failures": ["no_measured_requests"],
        }

    first_arrival_ms = float(measured[0].scheduled_arrival_ms)
    last_arrival_ms = float(measured[-1].scheduled_arrival_ms)
    arrival_interval_s = max(
        0.0, (last_arrival_ms - first_arrival_ms) / 1000.0
    )
    observed_trace_qps = (
        (len(measured) - 1) / arrival_interval_s
        if len(measured) > 1 and arrival_interval_s > 0.0
        else float(offered_request_rate_qps)
    )
    offered_qps = max(0.0, float(offered_request_rate_qps))
    successful = [item for item in measured if item.success]
    achieved_qps = len(successful) / max(float(duration_s), 1e-9)
    achieved_nominal_offered_ratio = (
        achieved_qps / offered_qps if offered_qps > 0.0 else 0.0
    )
    normalization_qps = (
        observed_trace_qps
        if normalize_by_observed_trace_rate and observed_trace_qps > 0.0
        else offered_qps
    )
    achieved_offered_ratio = (
        achieved_qps / normalization_qps if normalization_qps > 0.0 else 0.0
    )
    drain_s = max(0.0, float(duration_s) - arrival_interval_s)
    drain_ratio = (
        drain_s / arrival_interval_s if arrival_interval_s > 0.0 else 0.0
    )

    arrival_offsets = [
        float(item.scheduled_arrival_ms) for item in measured
    ]
    completion_offsets = [
        float(item.scheduled_arrival_ms)
        + float(item.dispatch_lag_ms)
        + max(0.0, float(item.e2e_ms))
        for item in measured
    ]
    ann_start_offsets = []
    first_token_offsets = []
    for item in measured:
        dispatched_ms = float(item.scheduled_arrival_ms) + float(
            item.dispatch_lag_ms
        )
        has_ann = (
            item.ann_index_mode != "none"
            or item.ann_latency_ms > 0.0
            or item.ann_queue_wait_ms > 0.0
        )
        ann_start_offsets.append(
            dispatched_ms + max(0.0, float(item.ann_queue_wait_ms))
            if has_ann
            else dispatched_ms
        )
        first_token_offsets.append(
            dispatched_ms + max(0.0, float(item.ttft_ms))
            if item.success and item.ttft_ms > 0.0
            else float("inf")
        )

    window_count = max(4, int(windows))
    if last_arrival_ms > first_arrival_ms:
        boundaries = np.linspace(
            first_arrival_ms,
            last_arrival_ms,
            window_count + 1,
            dtype=np.float64,
        )[1:].tolist()
    else:
        boundaries = [last_arrival_ms] * window_count

    def backlog_series(stage_offsets: List[float]) -> List[int]:
        return [
            sum(
                arrival <= boundary < stage
                for arrival, stage in zip(arrival_offsets, stage_offsets)
            )
            for boundary in boundaries
        ]

    system_backlog = backlog_series(completion_offsets)
    ann_waiting = backlog_series(ann_start_offsets)
    first_token_waiting = backlog_series(first_token_offsets)
    final_start = len(boundaries) // 2
    final_x_s = [
        (boundary - first_arrival_ms) / 1000.0
        for boundary in boundaries[final_start:]
    ]

    profiles = {}
    for name, values in (
        ("system", system_backlog),
        ("ann_waiting", ann_waiting),
        ("first_token_waiting", first_token_waiting),
    ):
        final_values = values[final_start:]
        slope = _linear_slope_ci95(final_x_s, final_values)
        growth = max(0, final_values[-1] - final_values[0])
        profiles[name] = {
            "windows": values,
            "end_backlog": values[-1],
            "final_half_growth": growth,
            "final_half_growth_fraction": growth / len(measured),
            "slope_requests_per_s": slope["slope"],
            "slope_ci95_low": slope["ci95_low"],
            "slope_ci95_high": slope["ci95_high"],
        }

    failures = []
    if achieved_offered_ratio < min_achieved_offered_ratio:
        failures.append("achieved_offered_ratio")
    if drain_ratio > max_drain_ratio:
        failures.append("drain_ratio")
    for name, profile in profiles.items():
        if (
            profile["final_half_growth_fraction"]
            > max_backlog_growth_fraction
            and profile["slope_ci95_low"] > 0.0
        ):
            failures.append(f"{name}_backlog_growth")

    return {
        "available": True,
        "passed": not failures,
        "failures": failures,
        "window_count": window_count,
        "offered_request_rate_qps": offered_qps,
        "observed_trace_rate_qps": observed_trace_qps,
        "achieved_request_rate_qps": achieved_qps,
        "achieved_offered_ratio": achieved_offered_ratio,
        "achieved_nominal_offered_ratio": achieved_nominal_offered_ratio,
        "rate_normalization": (
            "observed_trace"
            if normalize_by_observed_trace_rate
            else "nominal_offered"
        ),
        "arrival_interval_s": arrival_interval_s,
        "drain_s": drain_s,
        "drain_ratio": drain_ratio,
        "profiles": profiles,
        "thresholds": {
            "min_achieved_offered_ratio": min_achieved_offered_ratio,
            "max_drain_ratio": max_drain_ratio,
            "max_backlog_growth_fraction": max_backlog_growth_fraction,
        },
    }


def compute_slo_metrics(
    successful: List[RequestResult],
    duration_s: float,
    *,
    ttft_slo_ms: float,
    itl_slo_ms: float,
    e2e_slo_ms: float,
) -> Dict[str, float]:
    """Compute legacy interactive and end-to-end RAG SLO goodput."""
    latency_slo_compliant = [
        item
        for item in successful
        if (ttft_slo_ms <= 0 or item.ttft_ms <= ttft_slo_ms)
        and (
            itl_slo_ms <= 0
            or (item.itl_ms and percentile(item.itl_ms, 99) <= itl_slo_ms)
        )
    ]
    e2e_slo_compliant = [
        item
        for item in successful
        if e2e_slo_ms <= 0 or item.e2e_ms <= e2e_slo_ms
    ]
    completed = len(successful)
    return {
        "slo_attainment": (
            len(latency_slo_compliant) / completed if completed else 0.0
        ),
        "slo_compliant_qps": len(latency_slo_compliant) / duration_s,
        "e2e_slo_ms": float(e2e_slo_ms),
        "e2e_slo_attainment": (
            len(e2e_slo_compliant) / completed if completed else 0.0
        ),
        "e2e_slo_compliant_qps": len(e2e_slo_compliant) / duration_s,
    }


def load_fbin_vectors(path: Path) -> np.ndarray:
    """Load VectorLiteRAG-style float vectors with a two-int32 header."""
    with path.open("rb") as fin:
        header = np.fromfile(fin, count=2, dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"Invalid fbin header: {path}")
        count, dimension = (int(value) for value in header)
        if count <= 0 or dimension <= 0:
            raise ValueError(
                f"Invalid fbin shape ({count}, {dimension}): {path}"
            )
        values = np.fromfile(fin, dtype=np.float32)
    expected = count * dimension
    if values.size != expected:
        raise ValueError(
            f"Invalid fbin payload: expected {expected} floats, "
            f"found {values.size}: {path}"
        )
    return np.ascontiguousarray(values.reshape(count, dimension))


def exact_batch_sample_counts(
    runtime: Dict[str, Any], batch_size: int
) -> Dict[str, int]:
    """Return auditable sample counts for one exact ANN batch bucket."""
    key = str(batch_size)
    service_history = runtime.get("ann_gpu_service_by_batch_size") or {}
    service = service_history.get(key, service_history.get(batch_size)) or {}
    completion = (
        (runtime.get("ann_pre_switch_launch") or {})
        .get("completion_predictor", {})
        .get("by_path_batch", {})
        .get("full_gpu", {})
    )
    completion_bucket = completion.get(
        key, completion.get(batch_size)
    ) or {}
    return {
        "gpu_service_samples": int(service.get("samples", 0)),
        "completion_service_samples": int(
            completion_bucket.get("service_samples", 0)
        ),
        "completion_tail_samples": int(
            completion_bucket.get("tail_samples", 0)
        ),
        "worker_collection_samples": int(
            completion_bucket.get("worker_collection_samples", 0)
        ),
        "worker_send_samples": int(
            completion_bucket.get("worker_send_samples", 0)
        ),
    }


def validate_ann_priming_snapshot(
    server_metadata: Dict[str, Any],
    batch_sizes: List[int],
    min_samples: int,
    *,
    require_credit_straddle: bool,
) -> Dict[str, Any]:
    """Validate exact-batch ANN history before measured arrivals begin."""
    runtimes = server_metadata.get("rag_ann_runtime") or []
    failures = []
    if len(runtimes) != 1 or not isinstance(runtimes[0], dict):
        failures.append(
            "ANN priming requires exactly one rag_ann_runtime entry "
            f"(found {len(runtimes)})"
        )
        runtime: Dict[str, Any] = {}
    else:
        runtime = runtimes[0]

    raw_history = runtime.get("ann_gpu_service_by_batch_size") or {}
    buckets: Dict[str, Dict[str, Any]] = {}
    for batch_size in batch_sizes:
        key = str(batch_size)
        raw = raw_history.get(key, raw_history.get(batch_size))
        if not isinstance(raw, dict):
            failures.append(f"missing exact batch={batch_size} GPU history")
            continue
        bucket = {
            "samples": int(raw.get("samples", 0)),
            "p95_ms": float(raw.get("p95_ms", 0.0)),
            "p99_ms": float(raw.get("p99_ms", 0.0)),
        }
        buckets[key] = bucket
        if bucket["samples"] < min_samples:
            failures.append(
                f"batch={batch_size} samples={bucket['samples']} "
                f"< {min_samples}"
            )
        if (
            not np.isfinite(bucket["p95_ms"])
            or not np.isfinite(bucket["p99_ms"])
            or bucket["p95_ms"] <= 0.0
            or bucket["p99_ms"] < bucket["p95_ms"]
        ):
            failures.append(f"batch={batch_size} percentiles are invalid")

    completion = (
        (runtime.get("ann_pre_switch_launch") or {})
        .get("completion_predictor", {})
        .get("by_path_batch", {})
        .get("full_gpu", {})
    )
    completion_buckets: Dict[str, Dict[str, Any]] = {}
    for batch_size in batch_sizes:
        key = str(batch_size)
        raw = completion.get(key, completion.get(batch_size))
        if not isinstance(raw, dict):
            failures.append(
                f"missing exact batch={batch_size} full_gpu completion history"
            )
            continue
        bucket = {
            "mature": bool(raw.get("mature", False)),
            "service_samples": int(raw.get("service_samples", 0)),
            "tail_samples": int(raw.get("tail_samples", 0)),
            "worker_collection_samples": int(
                raw.get("worker_collection_samples", 0)
            ),
            "worker_send_samples": int(raw.get("worker_send_samples", 0)),
        }
        completion_buckets[key] = bucket
        if not bucket["mature"]:
            failures.append(
                f"batch={batch_size} completion history is not mature"
            )
        for field in (
            "service_samples",
            "tail_samples",
            "worker_collection_samples",
            "worker_send_samples",
        ):
            if bucket[field] < min_samples:
                failures.append(
                    f"batch={batch_size} {field}={bucket[field]} "
                    f"< {min_samples}"
                )

    ann_credit = runtime.get("ann_credit") or {}
    service_percentile = float(
        ann_credit.get("service_time_percentile", 95.0)
    )
    service_min_samples = int(
        ann_credit.get("service_time_min_samples", 4)
    )
    credit_ms = float(
        ann_credit.get(
            "configured_max_queued_gpu_ms",
            server_metadata.get("rag_ann_max_queued_gpu_ms", 0.0),
        )
    )
    microbatch_ms = float(
        ann_credit.get(
            "configured_max_microbatch_gpu_ms",
            server_metadata.get("rag_ann_max_microbatch_gpu_ms", 0.0),
        )
    )
    if require_credit_straddle and not np.isclose(
        service_percentile, 95.0
    ):
        failures.append(
            "credit straddle validation currently requires the server "
            "ANN service-time percentile to be 95"
        )
    if min_samples < service_min_samples:
        failures.append(
            f"priming min_samples={min_samples} is below the server "
            f"predictor minimum {service_min_samples}"
        )
    valid_interval = None
    if "1" in buckets and "2" in buckets:
        lower_ms = max(buckets["1"]["p99_ms"], buckets["2"]["p95_ms"])
        upper_ms = buckets["2"]["p99_ms"]
        valid_interval = {
            "lower_inclusive_ms": lower_ms,
            "upper_exclusive_ms": upper_ms,
            "nonempty": lower_ms < upper_ms,
        }
        if require_credit_straddle and not (
            lower_ms <= credit_ms < upper_ms
        ):
            failures.append(
                f"credit {credit_ms:.6f} ms is outside the valid "
                f"[{lower_ms:.6f}, {upper_ms:.6f}) ms interval"
            )
        if (
            require_credit_straddle
            and microbatch_ms > 0.0
            and microbatch_ms < buckets["2"]["p95_ms"]
        ):
            failures.append(
                f"microbatch cap {microbatch_ms:.6f} ms is below "
                f"batch2 p95 {buckets['2']['p95_ms']:.6f} ms"
            )
    elif require_credit_straddle:
        failures.append(
            "credit straddle validation requires exact batch=1 and batch=2"
        )

    return {
        "valid": not failures,
        "failures": failures,
        "batch_sizes": batch_sizes,
        "min_samples": min_samples,
        "credit_ms": credit_ms,
        "microbatch_ms": microbatch_ms,
        "service_time_percentile": service_percentile,
        "service_time_min_samples": service_min_samples,
        "require_credit_straddle": require_credit_straddle,
        "buckets": buckets,
        "completion_buckets": completion_buckets,
        "valid_credit_interval": valid_interval,
    }


async def prime_ann_history(
    session: aiohttp.ClientSession,
    base_url: str,
    query_vec_path: Path,
    prime_batch_sizes: List[int],
    validate_batch_sizes: List[int],
    batches_per_size: int,
    min_samples: int,
    topk: int,
    *,
    require_credit_straddle: bool,
) -> Dict[str, Any]:
    """Prime exact ANN batch histories outside the measurement window."""
    vectors = load_fbin_vectors(query_vec_path)
    largest_batch = max(prime_batch_sizes)
    if len(vectors) < largest_batch:
        raise ValueError(
            f"ANN priming needs {largest_batch} query vectors, "
            f"but {query_vec_path} contains {len(vectors)}"
        )

    api_url = base_url.rstrip("/") + "/ann_search"
    before_metadata = await get_server_metadata(session, base_url)
    before_runtimes = before_metadata.get("rag_ann_runtime") or []
    if len(before_runtimes) != 1 or not isinstance(
        before_runtimes[0], dict
    ):
        raise RuntimeError(
            "ANN priming requires exactly one rag_ann_runtime entry "
            f"before priming (found {len(before_runtimes)})"
        )
    before_runtime = before_runtimes[0]
    before_histogram = before_runtime.get("ann_batch_size_histogram") or {}
    response_batch_histogram: Counter[int] = Counter()
    vector_cursor = 0
    for batch_size in prime_batch_sizes:
        for batch_index in range(batches_per_size):
            rows = [
                vectors[(vector_cursor + offset) % len(vectors)].tolist()
                for offset in range(batch_size)
            ]
            vector_cursor = (vector_cursor + batch_size) % len(vectors)
            payload = {
                "queries": rows,
                "topk": topk,
                "request_id": (
                    f"ann-prime-b{batch_size}-{batch_index:04d}"
                ),
                "is_priming": True,
            }
            async with session.post(api_url, json=payload) as response:
                body = await response.json()
                if response.status != 200 or not body.get("success"):
                    raise RuntimeError(
                        f"ANN prime batch={batch_size} failed: "
                        f"status={response.status}, body={body}"
                    )
            observed_batch = int(body.get("batch_size", 0))
            if observed_batch != batch_size:
                raise RuntimeError(
                    f"ANN prime requested batch={batch_size}, "
                    f"server reported batch={observed_batch}"
                )
            if (
                body.get("index_mode") != "full_gpu"
                or float(body.get("gpu_latency_ms", 0.0)) <= 0.0
            ):
                raise RuntimeError(
                    f"ANN prime batch={batch_size} did not use full GPU: "
                    f"index_mode={body.get('index_mode')!r}, "
                    f"gpu_latency_ms={body.get('gpu_latency_ms')!r}"
                )
            response_batch_histogram[observed_batch] += 1

    metadata = await get_server_metadata(session, base_url)
    report = validate_ann_priming_snapshot(
        metadata,
        validate_batch_sizes,
        min_samples,
        require_credit_straddle=require_credit_straddle,
    )
    report.update(
        {
            "enabled": True,
            "query_vec_path": str(query_vec_path),
            "prime_batch_sizes": prime_batch_sizes,
            "validate_batch_sizes": validate_batch_sizes,
            "batches_per_size": batches_per_size,
            "response_batch_histogram": {
                str(key): value
                for key, value in sorted(response_batch_histogram.items())
            },
        }
    )
    runtime = (metadata.get("rag_ann_runtime") or [{}])[0]
    after_histogram = runtime.get("ann_batch_size_histogram") or {}
    runtime_histogram_delta = {
        str(batch_size): int(
            after_histogram.get(
                str(batch_size), after_histogram.get(batch_size, 0)
            )
        )
        - int(
            before_histogram.get(
                str(batch_size), before_histogram.get(batch_size, 0)
            )
        )
        for batch_size in prime_batch_sizes
    }
    report["runtime_batch_histogram_delta"] = runtime_histogram_delta
    expected_histogram = {
        str(batch_size): batches_per_size
        for batch_size in prime_batch_sizes
    }
    before_sample_counts = {
        str(batch_size): exact_batch_sample_counts(
            before_runtime, batch_size
        )
        for batch_size in validate_batch_sizes
    }
    after_sample_counts = {
        str(batch_size): exact_batch_sample_counts(runtime, batch_size)
        for batch_size in validate_batch_sizes
    }
    report["history_sample_counts_before"] = before_sample_counts
    report["history_sample_counts_after"] = after_sample_counts
    report["history_sample_count_delta"] = {
        key: {
            field: after_sample_counts[key][field] - before_counts[field]
            for field in before_counts
        }
        for key, before_counts in before_sample_counts.items()
    }
    # Preserve a pre-measurement counter baseline so offline analysis can
    # distinguish priming/warmup telemetry from ramp and measured activity.
    report["post_priming_runtime"] = runtime
    if report["response_batch_histogram"] != expected_histogram:
        report["valid"] = False
        report["failures"].append(
            "ANN prime response histogram does not match requested batches"
        )
    if runtime_histogram_delta != expected_histogram:
        report["valid"] = False
        report["failures"].append(
            "ANN runtime batch histogram delta does not match requested "
            "batches"
        )
    return report


async def get_server_metadata(
    session: aiohttp.ClientSession, base_url: str
) -> Dict[str, Any]:
    async with session.get(base_url.rstrip("/") + "/get_server_info") as response:
        response.raise_for_status()
        data = await response.json()

    metadata = {key: data.get(key) for key in SERVER_INFO_FIELDS if key in data}
    pdmux_runtime = []
    rag_ann_runtime = []
    for state in data.get("internal_states") or []:
        if not isinstance(state, dict):
            continue
        payload = state.get("internal_state", state)
        if not isinstance(payload, dict):
            continue
        if payload.get("pdmux_runtime") is not None:
            pdmux_runtime.append(payload["pdmux_runtime"])
        if payload.get("rag_ann_runtime") is not None:
            rag_ann_runtime.append(payload["rag_ann_runtime"])
    metadata["pdmux_runtime"] = pdmux_runtime
    metadata["rag_ann_runtime"] = rag_ann_runtime
    return metadata


async def run_request(
    session: aiohttp.ClientSession,
    api_url: str,
    request_id: int,
    query_id: int,
    output_len: int,
    topk: int,
    *,
    stage: str = "measure",
    scheduled_arrival_ms: float = 0.0,
    dispatch_lag_ms: float = 0.0,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    skip_retrieval: bool = False,
    skip_generation: bool = False,
) -> RequestResult:
    result = RequestResult(
        request_id=request_id,
        query_id=query_id,
        stage=stage,
        scheduled_arrival_ms=scheduled_arrival_ms,
        dispatch_lag_ms=dispatch_lag_ms,
    )
    payload = {
        "query_id": query_id,
        "topk": topk,
        "stream": True,
        "include_retrieval": not skip_retrieval,
        "skip_retrieval": skip_retrieval,
        "skip_generation": skip_generation,
        "rid": f"rag-bench-{request_id}",
        "sampling_params": {
            "temperature": temperature,
            "min_new_tokens": output_len,
            "max_new_tokens": output_len,
            "ignore_eos": ignore_eos,
        },
    }

    start = time.perf_counter()
    token_timestamps: List[float] = []
    last_text: Optional[str] = None
    try:
        async with session.post(api_url, json=payload) as response:
            result.status = response.status
            if response.status != 200:
                result.error = await response.text()
                return result

            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    continue

                data = json.loads(body)
                if "error" in data:
                    result.error = str(data["error"])
                    return result

                rag = data.get("rag")
                if rag is not None:
                    result.ann_latency_ms = float(rag.get("ann_latency_ms", 0.0))
                    result.ann_queue_wait_ms = float(
                        rag.get("ann_queue_wait_ms", 0.0)
                    )
                    result.ann_gpu_latency_ms = float(
                        rag.get("ann_gpu_latency_ms", 0.0)
                    )
                    result.ann_cpu_queue_wait_ms = float(
                        rag.get("ann_cpu_queue_wait_ms", 0.0)
                    )
                    result.ann_cpu_search_latency_ms = float(
                        rag.get("ann_cpu_search_latency_ms", 0.0)
                    )
                    result.ann_coarse_quantize_latency_ms = float(
                        rag.get("ann_coarse_quantize_latency_ms", 0.0)
                    )
                    result.ann_route_latency_ms = float(
                        rag.get("ann_route_latency_ms", 0.0)
                    )
                    result.ann_merge_latency_ms = float(
                        rag.get("ann_merge_latency_ms", 0.0)
                    )
                    result.ann_index_mode = str(
                        rag.get("ann_index_mode", "full_gpu")
                    )
                    result.ann_hot_probe_ratio = float(
                        rag.get("ann_hot_probe_ratio", 0.0)
                    )
                    result.ann_hot_probe_count = int(
                        rag.get("ann_hot_probe_count", 0)
                    )
                    result.ann_cold_probe_count = int(
                        rag.get("ann_cold_probe_count", 0)
                    )
                    result.ann_batch_size = int(rag.get("ann_batch_size", 0))
                    result.retrieved_ids = [
                        int(value) for value in (rag.get("ids") or [])
                    ]
                    result.token_cache_hit = bool(
                        rag.get("token_cache_hit", False)
                    )

                meta = data.get("meta_info") or {}
                result.output_tokens = max(
                    result.output_tokens, int(meta.get("completion_tokens", 0) or 0)
                )

                text = data.get("text")
                if text and text != last_text:
                    token_timestamps.append(time.perf_counter())
                    last_text = text

        end = time.perf_counter()
        result.e2e_ms = (end - start) * 1000.0
        if not token_timestamps:
            if skip_generation:
                result.ttft_ms = 0.0
                result.llm_ttft_after_ann_ms = 0.0
                result.tpot_ms = 0.0
                result.success = True
                return result
            result.error = "The stream completed without generated text."
            return result

        result.ttft_ms = (token_timestamps[0] - start) * 1000.0
        result.llm_ttft_after_ann_ms = max(
            0.0, result.ttft_ms - result.ann_latency_ms
        )
        result.itl_ms = [
            (right - left) * 1000.0
            for left, right in zip(token_timestamps, token_timestamps[1:])
        ]
        if result.output_tokens <= 0:
            result.output_tokens = len(token_timestamps)
        if result.output_tokens > 1:
            result.tpot_ms = (result.e2e_ms - result.ttft_ms) / (
                result.output_tokens - 1
            )
        result.success = True
    except Exception as exc:
        result.error = repr(exc)
    return result


async def warmup(
    session: aiohttp.ClientSession,
    api_url: str,
    count: int,
    output_len: int,
    topk: int,
    query_id_start: int,
    temperature: float,
    ignore_eos: bool,
    skip_retrieval: bool = False,
    skip_generation: bool = False,
) -> None:
    for request_id in range(count):
        result = await run_request(
            session,
            api_url,
            -(request_id + 1),
            query_id_start + request_id,
            output_len,
            topk,
            temperature=temperature,
            ignore_eos=ignore_eos,
            skip_retrieval=skip_retrieval,
            skip_generation=skip_generation,
        )
        if not result.success:
            raise RuntimeError(f"Warmup request failed: {result.error}")


async def load_ramp_warmup(
    session: aiohttp.ClientSession,
    api_url: str,
    count: int,
    request_rate: float,
    output_len: int,
    topk: int,
    query_id_start: int,
    temperature: float,
    ignore_eos: bool,
) -> Dict[str, Any]:
    """Train ANN predictors under controlled RAG load before measurement."""
    schedule_start = time.perf_counter()
    tasks = []
    for request_index in range(count):
        target_time = schedule_start + request_index / request_rate
        delay_s = target_time - time.perf_counter()
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        tasks.append(
            asyncio.create_task(
                run_request(
                    session,
                    api_url,
                    -(1_000_000 + request_index),
                    query_id_start + request_index,
                    output_len,
                    topk,
                    stage="ann_prime_load_ramp",
                    scheduled_arrival_ms=(
                        request_index * 1000.0 / request_rate
                    ),
                    dispatch_lag_ms=max(
                        0.0,
                        (time.perf_counter() - target_time) * 1000.0,
                    ),
                    temperature=temperature,
                    ignore_eos=ignore_eos,
                )
            )
        )
    results = await asyncio.gather(*tasks)
    failed = [result for result in results if not result.success]
    if failed:
        raise RuntimeError(
            "ANN priming load ramp failed: "
            + json.dumps(
                [
                    {
                        "request_id": result.request_id,
                        "status": result.status,
                        "error": result.error,
                    }
                    for result in failed[:8]
                ],
                sort_keys=True,
            )
        )
    return {
        "requests": count,
        "request_rate": request_rate,
        "output_len": output_len,
        "completed": len(results),
        "ann_batch_size_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(result.ann_batch_size for result in results).items()
            )
        },
        "ann_latency_ms": distribution(
            [result.ann_latency_ms for result in results]
        ),
        "e2e_ms": distribution([result.e2e_ms for result in results]),
    }


def resolve_workload(args: argparse.Namespace) -> RagWorkload:
    if args.workload_trace is not None:
        return load_workload(Path(args.workload_trace))
    return generate_workload(
        num_requests=args.num_requests,
        ramp_up_requests=args.ramp_up_requests,
        warmup_requests=args.warmup_requests,
        request_rate=args.request_rate,
        output_len=args.output_len,
        topk=args.topk,
        seed=args.seed,
        deterministic_arrivals=args.deterministic_arrivals,
        warmup_output_len=args.warmup_output_len,
        temperature=args.temperature,
    )


def should_normalize_stationarity_by_observed_trace_rate(
    workload: RagWorkload,
    *,
    deterministic_arrivals: bool,
) -> bool:
    """Use the trace's arrival process even when its timestamps are replayed."""
    arrival_process = str(
        workload.metadata.get("arrival_process", "")
    ).strip().lower()
    if arrival_process:
        return arrival_process != "deterministic"
    return not deterministic_arrivals


async def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(
        limit=0,
        force_close=args.disable_http_keepalive,
    )
    api_url = args.base_url.rstrip("/") + "/rag_generate"
    workload = resolve_workload(args)
    workload.validate()
    first_request = workload.requests[0]
    max_query_id = max(request.query_id for request in workload.requests)
    priming_report = None
    priming_path = None
    priming_validate_batch_sizes = None
    if args.skip_retrieval and args.skip_generation:
        raise ValueError(
            "--skip-retrieval and --skip-generation are mutually exclusive"
        )
    if args.skip_retrieval and args.ann_prime_batches_per_size != 0:
        raise ValueError("--skip-retrieval cannot be combined with ANN priming")
    if args.ann_prime_batches_per_size < 0:
        raise ValueError("--ann-prime-batches-per-size must be non-negative")
    if args.ann_prime_load_ramp_requests < 0:
        raise ValueError(
            "--ann-prime-load-ramp-requests must be non-negative"
        )
    if (
        args.ann_prime_load_ramp_requests > 0
        and args.ann_prime_load_ramp_rps <= 0.0
    ):
        raise ValueError(
            "--ann-prime-load-ramp-rps must be positive when the ANN "
            "priming load ramp is enabled"
        )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        server_metadata = await get_server_metadata(session, args.base_url)
        await warmup(
            session,
            api_url,
            workload.warmup_requests,
            workload.warmup_output_len,
            first_request.topk,
            max_query_id + 1,
            workload.temperature,
            workload.ignore_eos,
            args.skip_retrieval,
            args.skip_generation,
        )
        if args.ann_prime_batches_per_size > 0:
            if args.ann_prime_query_vec_path is None:
                raise ValueError(
                    "--ann-prime-query-vec-path is required when ANN "
                    "priming is enabled"
                )
            prime_batch_sizes = [
                int(value)
                for value in args.ann_prime_batch_sizes.split(",")
                if value
            ]
            if (
                not prime_batch_sizes
                or any(value <= 0 for value in prime_batch_sizes)
                or prime_batch_sizes
                != sorted(set(prime_batch_sizes))
            ):
                raise ValueError(
                    "--ann-prime-batch-sizes must be unique positive "
                    "integers in ascending order"
                )
            validate_batch_sizes = [
                int(value)
                for value in args.ann_prime_validate_batch_sizes.split(",")
                if value
            ]
            if (
                not validate_batch_sizes
                or any(value <= 0 for value in validate_batch_sizes)
                or validate_batch_sizes
                != sorted(set(validate_batch_sizes))
            ):
                raise ValueError(
                    "--ann-prime-validate-batch-sizes must be unique "
                    "positive integers in ascending order"
                )
            if not set(prime_batch_sizes).issubset(
                validate_batch_sizes
            ):
                raise ValueError(
                    "Every primed batch size must also be validated"
                )
            priming_validate_batch_sizes = validate_batch_sizes
            if args.ann_prime_min_samples < 4:
                raise ValueError("--ann-prime-min-samples must be at least 4")
            if args.ann_prime_min_samples > 64:
                raise ValueError(
                    "--ann-prime-min-samples cannot exceed the predictor "
                    "history capacity of 64"
                )
            if (
                args.ann_prime_batches_per_size
                < args.ann_prime_min_samples
            ):
                raise ValueError(
                    "--ann-prime-batches-per-size must be at least "
                    "--ann-prime-min-samples"
                )
            defer_credit_straddle = bool(
                args.ann_prime_require_credit_straddle
                and args.ann_prime_load_ramp_requests > 0
            )
            priming_report = await prime_ann_history(
                session,
                args.base_url,
                args.ann_prime_query_vec_path,
                prime_batch_sizes,
                validate_batch_sizes,
                args.ann_prime_batches_per_size,
                args.ann_prime_min_samples,
                first_request.topk,
                require_credit_straddle=(
                    args.ann_prime_require_credit_straddle
                    and not defer_credit_straddle
                ),
            )
            priming_report["initial_require_credit_straddle"] = bool(
                priming_report["require_credit_straddle"]
            )
            priming_report["require_credit_straddle"] = bool(
                args.ann_prime_require_credit_straddle
            )
            if args.ann_prime_load_ramp_requests > 0:
                priming_report["post_ann_only_priming_runtime"] = (
                    priming_report.pop("post_priming_runtime")
                )
                priming_report["load_ramp"] = await load_ramp_warmup(
                    session,
                    api_url,
                    args.ann_prime_load_ramp_requests,
                    args.ann_prime_load_ramp_rps,
                    workload.warmup_output_len,
                    first_request.topk,
                    max_query_id + workload.warmup_requests + 1,
                    workload.temperature,
                    workload.ignore_eos,
                )
                post_load_metadata = await get_server_metadata(
                    session, args.base_url
                )
                post_load_validation = validate_ann_priming_snapshot(
                    post_load_metadata,
                    validate_batch_sizes,
                    args.ann_prime_min_samples,
                    require_credit_straddle=(
                        args.ann_prime_require_credit_straddle
                    ),
                )
                priming_report["post_load_ramp_validation"] = (
                    post_load_validation
                )
                post_load_runtimes = (
                    post_load_metadata.get("rag_ann_runtime") or []
                )
                if (
                    len(post_load_runtimes) == 1
                    and isinstance(post_load_runtimes[0], dict)
                ):
                    priming_report["post_priming_runtime"] = (
                        post_load_runtimes[0]
                    )
                if not post_load_validation["valid"]:
                    priming_report["valid"] = False
                    priming_report["failures"].append(
                        "post-load-ramp ANN validation failed"
                    )
            priming_path = Path(args.output_file).with_suffix(
                ".ann_priming.json"
            )
            priming_path.parent.mkdir(parents=True, exist_ok=True)
            priming_path.write_text(
                json.dumps(priming_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not priming_report["valid"]:
                raise RuntimeError(
                    "ANN priming validation failed: "
                    + json.dumps(priming_report, sort_keys=True)
                )

        ramp_up_tasks = []
        measurement_tasks = []
        measurement_start = None
        measurement_start_unix_s = None
        schedule_start = time.perf_counter()
        for request_spec in workload.requests:
            target_time = schedule_start + request_spec.arrival_offset_ms / 1000.0
            delay_s = target_time - time.perf_counter()
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            dispatch_time = time.perf_counter()
            dispatch_lag_ms = max(0.0, (dispatch_time - target_time) * 1000.0)
            if request_spec.stage == "measure" and measurement_start is None:
                measurement_start = target_time
                measurement_start_unix_s = time.time()

            task = asyncio.create_task(
                run_request(
                    session,
                    api_url,
                    request_spec.request_id,
                    request_spec.query_id,
                    request_spec.output_len,
                    request_spec.topk,
                    stage=request_spec.stage,
                    scheduled_arrival_ms=request_spec.arrival_offset_ms,
                    dispatch_lag_ms=dispatch_lag_ms,
                    temperature=workload.temperature,
                    ignore_eos=workload.ignore_eos,
                    skip_retrieval=args.skip_retrieval,
                    skip_generation=args.skip_generation,
                )
            )
            if request_spec.stage == "ramp_up":
                ramp_up_tasks.append(task)
            else:
                measurement_tasks.append(task)

        results = await asyncio.gather(*measurement_tasks)
        measurement_end = time.perf_counter()
        measurement_end_unix_s = time.time()
        ramp_up_results = await asyncio.gather(*ramp_up_tasks)
        # Refresh runtime counters after the measured workload. Static server
        # configuration is unchanged, while ANN/Decode telemetry now reflects
        # the completed run instead of the pre-warmup state.
        server_metadata = await get_server_metadata(session, args.base_url)
        if (
            priming_report is not None
            and priming_path is not None
            and priming_validate_batch_sizes is not None
        ):
            post_measurement_validation = validate_ann_priming_snapshot(
                server_metadata,
                priming_validate_batch_sizes,
                args.ann_prime_min_samples,
                require_credit_straddle=(
                    args.ann_prime_require_credit_straddle
                ),
            )
            priming_report["post_measurement_validation"] = (
                post_measurement_validation
            )
            priming_path.write_text(
                json.dumps(priming_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not post_measurement_validation["valid"]:
                raise RuntimeError(
                    "Post-measurement ANN priming validation failed: "
                    + json.dumps(
                        post_measurement_validation, sort_keys=True
                    )
                )
        if measurement_start is None:
            raise RuntimeError("Measurement did not start.")
        if measurement_start_unix_s is None:
            raise RuntimeError("Measurement wall-clock timestamp was not recorded.")
        duration_s = measurement_end - measurement_start

    successful = [item for item in results if item.success]
    all_itl = [value for item in successful for value in item.itl_ms]
    per_request_p99_itl = [
        percentile(item.itl_ms, 99) for item in successful if item.itl_ms
    ]
    slo_metrics = compute_slo_metrics(
        successful,
        duration_s,
        ttft_slo_ms=args.ttft_slo_ms,
        itl_slo_ms=args.itl_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
    )
    stationarity = summarize_stationarity(
        results,
        offered_request_rate_qps=args.request_rate,
        duration_s=duration_s,
        normalize_by_observed_trace_rate=(
            should_normalize_stationarity_by_observed_trace_rate(
                workload,
                deterministic_arrivals=args.deterministic_arrivals,
            )
        ),
        windows=args.stationarity_windows,
        min_achieved_offered_ratio=(
            args.stationarity_min_achieved_offered_ratio
        ),
        max_drain_ratio=args.stationarity_max_drain_ratio,
        max_backlog_growth_fraction=(
            args.stationarity_max_backlog_growth_fraction
        ),
    )

    summary = {
        "ramp_up_completed": sum(item.success for item in ramp_up_results),
        "ramp_up_failed": sum(not item.success for item in ramp_up_results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": duration_s,
        "measurement_start_unix_s": measurement_start_unix_s,
        "measurement_end_unix_s": measurement_end_unix_s,
        "request_throughput_qps": len(successful) / duration_s,
        "stationarity": stationarity,
        **slo_metrics,
        "e2e_ms": distribution([item.e2e_ms for item in successful]),
        "rag_ttft_ms": distribution([item.ttft_ms for item in successful]),
        "ann_latency_ms": distribution(
            [item.ann_latency_ms for item in successful]
        ),
        "ann_queue_wait_ms": distribution(
            [item.ann_queue_wait_ms for item in successful]
        ),
        "ann_gpu_latency_ms": distribution(
            [item.ann_gpu_latency_ms for item in successful]
        ),
        "ann_cpu_queue_wait_ms": distribution(
            [item.ann_cpu_queue_wait_ms for item in successful]
        ),
        "ann_cpu_search_latency_ms": distribution(
            [item.ann_cpu_search_latency_ms for item in successful]
        ),
        "ann_coarse_quantize_latency_ms": distribution(
            [item.ann_coarse_quantize_latency_ms for item in successful]
        ),
        "ann_route_latency_ms": distribution(
            [item.ann_route_latency_ms for item in successful]
        ),
        "ann_merge_latency_ms": distribution(
            [item.ann_merge_latency_ms for item in successful]
        ),
        "ann_hot_probe_ratio": distribution(
            [item.ann_hot_probe_ratio for item in successful]
        ),
        "ann_hot_probe_count": sum(
            item.ann_hot_probe_count for item in successful
        ),
        "ann_cold_probe_count": sum(
            item.ann_cold_probe_count for item in successful
        ),
        "ann_index_modes": dict(
            Counter(item.ann_index_mode for item in successful)
        ),
        "ann_batch_size": distribution(
            [float(item.ann_batch_size) for item in successful]
        ),
        "token_cache_hit_rate": (
            sum(item.token_cache_hit for item in successful) / len(successful)
            if successful
            else 0.0
        ),
        "llm_ttft_after_ann_ms": distribution(
            [item.llm_ttft_after_ann_ms for item in successful]
        ),
        "tpot_ms": distribution([item.tpot_ms for item in successful]),
        "itl_ms": distribution(all_itl),
        "per_request_p99_itl_ms": distribution(per_request_p99_itl),
        "dispatch_lag_ms": distribution(
            [item.dispatch_lag_ms for item in successful]
        ),
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "config": config,
        "workload": {
            "path": str(args.workload_trace) if args.workload_trace else None,
            **workload.to_metadata_dict(),
        },
        "server": server_metadata,
        "ann_priming": priming_report,
        "summary": summary,
        "ramp_up_results": [asdict(item) for item in ramp_up_results],
        "requests": [asdict(item) for item in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--workload-trace",
        type=Path,
        help=(
            "Canonical JSONL trace generated by rag_workload.py. When set, "
            "its arrivals, query IDs, output lengths, top-k, and warmup settings "
            "replace the corresponding synthetic workload arguments."
        ),
    )
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--warmup-output-len", type=int, default=32)
    parser.add_argument(
        "--ann-prime-query-vec-path",
        type=Path,
        help=(
            "VectorLiteRAG fbin vectors used for controlled, unmeasured "
            "exact-batch ANN priming."
        ),
    )
    parser.add_argument(
        "--ann-prime-batch-sizes",
        default="1,2",
        help="Comma-separated exact ANN batch sizes to prime.",
    )
    parser.add_argument(
        "--ann-prime-validate-batch-sizes",
        default="1,2",
        help=(
            "Comma-separated exact ANN batch histories required before "
            "measurement. These may include normal warmup buckets that are "
            "not overwritten by ANN-only priming."
        ),
    )
    parser.add_argument(
        "--ann-prime-batches-per-size",
        type=int,
        default=0,
        help="Sequential ANN-only prime requests per exact batch size.",
    )
    parser.add_argument(
        "--ann-prime-min-samples",
        type=int,
        default=4,
        help="Minimum mature exact-batch samples required before measurement.",
    )
    parser.add_argument(
        "--ann-prime-load-ramp-requests",
        type=int,
        default=0,
        help=(
            "Concurrent short-output RAG requests used to train exact-batch "
            "predictors under target load after ANN-only priming. Default "
            "zero keeps this stage disabled."
        ),
    )
    parser.add_argument(
        "--ann-prime-load-ramp-rps",
        type=float,
        default=0.0,
        help="Arrival rate for the optional ANN priming load ramp.",
    )
    parser.add_argument(
        "--ann-prime-require-credit-straddle",
        action="store_true",
        help=(
            "Fail before measurement unless batch1_p99 <= credit and "
            "batch2_p95 <= credit < batch2_p99."
        ),
    )
    parser.add_argument(
        "--ramp-up-requests",
        type=int,
        default=50,
        help=(
            "Requests sent at the target arrival rate before measurement. "
            "They establish steady-state concurrency and are excluded from metrics."
        ),
    )
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--deterministic-arrivals", action="store_true")
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--disable-http-keepalive",
        action="store_true",
        help=(
            "Use a fresh localhost connection for each streaming request. "
            "This avoids reusing connections closed by the server keep-alive "
            "timeout during long open-loop runs."
        ),
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=-1.0)
    parser.add_argument("--itl-slo-ms", type=float, default=-1.0)
    parser.add_argument("--e2e-slo-ms", type=float, default=-1.0)
    parser.add_argument("--stationarity-windows", type=int, default=8)
    parser.add_argument(
        "--stationarity-min-achieved-offered-ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--stationarity-max-drain-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--stationarity-max-backlog-growth-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workload_trace is None:
        if args.request_rate <= 0:
            raise ValueError("--request-rate must be positive")
        if args.num_requests <= 0:
            raise ValueError("--num-requests must be positive")
        if args.ramp_up_requests < 0:
            raise ValueError("--ramp-up-requests must be non-negative")
    result = asyncio.run(benchmark(args))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
