#!/usr/bin/env python3
"""Calibrate request-level CPU Faiss parallelism for a RAG serving baseline.

The calibration loads the index once, sweeps a bounded worker/thread grid,
measures saturation throughput, and then checks queue stability at the target
open-loop request rate. It can emit a CPU-only end-to-end profiler config using
the selected setting.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class SearchRecord:
    sequence: int
    scheduled_at: float
    enqueued_at: float
    started_at: float
    finished_at: float

    @property
    def schedule_lag_ms(self) -> float:
        return max(0.0, self.enqueued_at - self.scheduled_at) * 1000.0

    @property
    def queue_wait_ms(self) -> float:
        return max(0.0, self.started_at - self.enqueued_at) * 1000.0

    @property
    def service_ms(self) -> float:
        return max(0.0, self.finished_at - self.started_at) * 1000.0

    @property
    def latency_ms(self) -> float:
        return max(0.0, self.finished_at - self.enqueued_at) * 1000.0


@dataclass(frozen=True)
class CalibrationResult:
    workers: int
    threads_per_search: int
    cpu_slots: int
    saturation_queries: int
    saturation_wall_s: float
    saturation_qps: float
    saturation_service_mean_ms: float
    saturation_service_p99_ms: float
    saturation_queue_p99_ms: float
    open_loop_rps: float
    open_loop_queries: int
    open_loop_wall_s: Optional[float]
    open_loop_achieved_qps: Optional[float]
    open_loop_schedule_lag_p99_ms: Optional[float]
    open_loop_queue_mean_ms: Optional[float]
    open_loop_queue_p99_ms: Optional[float]
    open_loop_queue_slope_ms_per_request: Optional[float]
    open_loop_queue_tail_growth_ms: Optional[float]
    open_loop_service_mean_ms: Optional[float]
    open_loop_service_p99_ms: Optional[float]
    open_loop_latency_p99_ms: Optional[float]
    capacity_headroom: float
    eligible: bool


def percentile(values: Iterable[float], q: float) -> float:
    materialized = list(values)
    return float(np.percentile(materialized, q)) if materialized else 0.0


def load_fbin_vectors(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, count=2, dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"Invalid fbin header: {path}")
        count, dimension = (int(value) for value in header)
        values = np.fromfile(handle, dtype=np.float32)
    expected = count * dimension
    if count <= 0 or dimension <= 0 or values.size != expected:
        raise ValueError(
            f"Invalid fbin payload in {path}: shape=({count}, {dimension}), "
            f"floats={values.size}, expected={expected}"
        )
    return np.ascontiguousarray(values.reshape(count, dimension))


def import_faiss(faiss_egg_dir: Optional[Path]):
    if faiss_egg_dir is not None:
        resolved = str(faiss_egg_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return importlib.import_module("faiss")


def sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def run_search(
    faiss: Any,
    index: Any,
    query: np.ndarray,
    topk: int,
    threads_per_search: int,
    sequence: int,
    scheduled_at: float,
    enqueued_at: float,
) -> SearchRecord:
    started_at = time.perf_counter()
    faiss.omp_set_num_threads(threads_per_search)
    index.search(query.reshape(1, -1), topk)
    finished_at = time.perf_counter()
    return SearchRecord(
        sequence=sequence,
        scheduled_at=scheduled_at,
        enqueued_at=enqueued_at,
        started_at=started_at,
        finished_at=finished_at,
    )


def collect(futures: Iterable[Future[SearchRecord]]) -> list[SearchRecord]:
    return sorted((future.result() for future in futures), key=lambda row: row.sequence)


def run_burst(
    pool: ThreadPoolExecutor,
    *,
    faiss: Any,
    index: Any,
    vectors: np.ndarray,
    start_row: int,
    query_count: int,
    topk: int,
    threads_per_search: int,
) -> list[SearchRecord]:
    futures: list[Future[SearchRecord]] = []
    for sequence in range(query_count):
        now = time.perf_counter()
        query = vectors[(start_row + sequence) % len(vectors)]
        futures.append(
            pool.submit(
                run_search,
                faiss,
                index,
                query,
                topk,
                threads_per_search,
                sequence,
                now,
                now,
            )
        )
    return collect(futures)


def run_open_loop(
    pool: ThreadPoolExecutor,
    *,
    faiss: Any,
    index: Any,
    vectors: np.ndarray,
    start_row: int,
    query_count: int,
    topk: int,
    threads_per_search: int,
    request_rate: float,
) -> list[SearchRecord]:
    interval_s = 1.0 / request_rate
    first_scheduled_at = time.perf_counter() + 0.25
    futures: list[Future[SearchRecord]] = []
    for sequence in range(query_count):
        scheduled_at = first_scheduled_at + sequence * interval_s
        sleep_until(scheduled_at)
        enqueued_at = time.perf_counter()
        query = vectors[(start_row + sequence) % len(vectors)]
        futures.append(
            pool.submit(
                run_search,
                faiss,
                index,
                query,
                topk,
                threads_per_search,
                sequence,
                scheduled_at,
                enqueued_at,
            )
        )
    return collect(futures)


def wall_seconds(records: list[SearchRecord], *, use_schedule: bool) -> float:
    if not records:
        return 0.0
    started_at = (
        min(row.scheduled_at for row in records)
        if use_schedule
        else min(row.enqueued_at for row in records)
    )
    return max(row.finished_at for row in records) - started_at


def queue_slope(records: list[SearchRecord]) -> float:
    if len(records) < 2:
        return 0.0
    x = np.asarray([row.sequence for row in records], dtype=np.float64)
    y = np.asarray([row.queue_wait_ms for row in records], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def queue_tail_growth(records: list[SearchRecord]) -> float:
    if len(records) < 4:
        return 0.0
    quarter = max(1, len(records) // 4)
    first = statistics.fmean(row.queue_wait_ms for row in records[:quarter])
    last = statistics.fmean(row.queue_wait_ms for row in records[-quarter:])
    return float(last - first)


def calibrate_one(
    *,
    faiss: Any,
    index: Any,
    vectors: np.ndarray,
    workers: int,
    threads_per_search: int,
    warmup_queries: int,
    saturation_queries: int,
    open_loop_queries: int,
    target_rps: float,
    topk: int,
    required_headroom: float,
    max_queue_p99_ms: float,
    max_queue_slope_ms_per_request: float,
) -> CalibrationResult:
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=f"cpu-ann-w{workers}-t{threads_per_search}",
    ) as pool:
        run_burst(
            pool,
            faiss=faiss,
            index=index,
            vectors=vectors,
            start_row=0,
            query_count=warmup_queries,
            topk=topk,
            threads_per_search=threads_per_search,
        )
        burst = run_burst(
            pool,
            faiss=faiss,
            index=index,
            vectors=vectors,
            start_row=warmup_queries,
            query_count=saturation_queries,
            topk=topk,
            threads_per_search=threads_per_search,
        )
        saturation_wall_s = wall_seconds(burst, use_schedule=False)
        saturation_qps = len(burst) / saturation_wall_s
        capacity_headroom = saturation_qps / target_rps

        open_loop: list[SearchRecord] = []
        if capacity_headroom >= 0.95:
            open_loop = run_open_loop(
                pool,
                faiss=faiss,
                index=index,
                vectors=vectors,
                start_row=warmup_queries + saturation_queries,
                query_count=open_loop_queries,
                topk=topk,
                threads_per_search=threads_per_search,
                request_rate=target_rps,
            )

    burst_service = [row.service_ms for row in burst]
    burst_queue = [row.queue_wait_ms for row in burst]
    if open_loop:
        open_loop_wall_s = wall_seconds(open_loop, use_schedule=True)
        open_loop_queue = [row.queue_wait_ms for row in open_loop]
        open_loop_service = [row.service_ms for row in open_loop]
        open_loop_latency = [row.latency_ms for row in open_loop]
        open_loop_slope = queue_slope(open_loop)
        open_loop_growth = queue_tail_growth(open_loop)
        open_loop_queue_p99 = percentile(open_loop_queue, 99)
        eligible = (
            capacity_headroom >= required_headroom
            and open_loop_queue_p99 <= max_queue_p99_ms
            and open_loop_slope <= max_queue_slope_ms_per_request
            and open_loop_growth <= max_queue_p99_ms
        )
        return CalibrationResult(
            workers=workers,
            threads_per_search=threads_per_search,
            cpu_slots=workers * threads_per_search,
            saturation_queries=len(burst),
            saturation_wall_s=saturation_wall_s,
            saturation_qps=saturation_qps,
            saturation_service_mean_ms=statistics.fmean(burst_service),
            saturation_service_p99_ms=percentile(burst_service, 99),
            saturation_queue_p99_ms=percentile(burst_queue, 99),
            open_loop_rps=target_rps,
            open_loop_queries=len(open_loop),
            open_loop_wall_s=open_loop_wall_s,
            open_loop_achieved_qps=len(open_loop) / open_loop_wall_s,
            open_loop_schedule_lag_p99_ms=percentile(
                (row.schedule_lag_ms for row in open_loop), 99
            ),
            open_loop_queue_mean_ms=statistics.fmean(open_loop_queue),
            open_loop_queue_p99_ms=open_loop_queue_p99,
            open_loop_queue_slope_ms_per_request=open_loop_slope,
            open_loop_queue_tail_growth_ms=open_loop_growth,
            open_loop_service_mean_ms=statistics.fmean(open_loop_service),
            open_loop_service_p99_ms=percentile(open_loop_service, 99),
            open_loop_latency_p99_ms=percentile(open_loop_latency, 99),
            capacity_headroom=capacity_headroom,
            eligible=eligible,
        )

    return CalibrationResult(
        workers=workers,
        threads_per_search=threads_per_search,
        cpu_slots=workers * threads_per_search,
        saturation_queries=len(burst),
        saturation_wall_s=saturation_wall_s,
        saturation_qps=saturation_qps,
        saturation_service_mean_ms=statistics.fmean(burst_service),
        saturation_service_p99_ms=percentile(burst_service, 99),
        saturation_queue_p99_ms=percentile(burst_queue, 99),
        open_loop_rps=target_rps,
        open_loop_queries=0,
        open_loop_wall_s=None,
        open_loop_achieved_qps=None,
        open_loop_schedule_lag_p99_ms=None,
        open_loop_queue_mean_ms=None,
        open_loop_queue_p99_ms=None,
        open_loop_queue_slope_ms_per_request=None,
        open_loop_queue_tail_growth_ms=None,
        open_loop_service_mean_ms=None,
        open_loop_service_p99_ms=None,
        open_loop_latency_p99_ms=None,
        capacity_headroom=capacity_headroom,
        eligible=False,
    )


def select_result(results: list[CalibrationResult]) -> Optional[CalibrationResult]:
    eligible = [result for result in results if result.eligible]
    if not eligible:
        return None
    best_latency = min(
        result.open_loop_latency_p99_ms
        for result in eligible
        if result.open_loop_latency_p99_ms is not None
    )
    near_best = [
        result
        for result in eligible
        if result.open_loop_latency_p99_ms is not None
        and result.open_loop_latency_p99_ms <= best_latency * 1.05
    ]
    return min(
        near_best,
        key=lambda result: (
            result.cpu_slots,
            result.open_loop_latency_p99_ms,
            -result.saturation_qps,
        ),
    )


def write_csv(path: Path, results: list[CalibrationResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_profile_config(
    *,
    base_path: Path,
    output_path: Path,
    profile_output_dir: Path,
    selected: CalibrationResult,
    calibration_json: Path,
) -> None:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["output_dir"] = str(profile_output_dir)
    config["runtime"]["cpu_retrieval_workers"] = selected.workers
    config["runtime"]["cpu_retrieval_threads"] = selected.threads_per_search
    config["sweep"]["paired_policies"] = ["cpu-retrieval"]
    config["sweep"]["randomize_case_order"] = False
    config["calibration_provenance"] = {
        "calibration_json": str(calibration_json),
        "selected_workers": selected.workers,
        "selected_threads_per_search": selected.threads_per_search,
        "saturation_qps": selected.saturation_qps,
        "target_rps": selected.open_loop_rps,
        "capacity_headroom": selected.capacity_headroom,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--query-vec-path", type=Path, required=True)
    parser.add_argument("--faiss-egg-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nprobe", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument(
        "--worker-counts",
        type=int,
        nargs="+",
        default=[8, 12, 16, 24, 32, 40, 48],
    )
    parser.add_argument(
        "--threads-per-search",
        type=int,
        nargs="+",
        default=[1],
    )
    parser.add_argument("--warmup-queries", type=int, default=8)
    parser.add_argument("--saturation-queries", type=int, default=96)
    parser.add_argument("--open-loop-queries", type=int, default=96)
    parser.add_argument("--target-rps", type=float, default=6.5)
    parser.add_argument("--required-headroom", type=float, default=1.2)
    parser.add_argument("--max-queue-p99-ms", type=float, default=250.0)
    parser.add_argument(
        "--max-queue-slope-ms-per-request",
        type=float,
        default=2.0,
    )
    parser.add_argument("--max-cpu-slots", type=int)
    parser.add_argument("--base-profile-config", type=Path)
    parser.add_argument("--write-profile-config", type=Path)
    parser.add_argument("--profile-output-dir", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "nprobe",
        "topk",
        "warmup_queries",
        "saturation_queries",
        "open_loop_queries",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.target_rps <= 0 or args.required_headroom < 1.0:
        raise ValueError("target_rps must be positive and required_headroom >= 1")
    profile_args = (
        args.base_profile_config,
        args.write_profile_config,
        args.profile_output_dir,
    )
    if any(value is not None for value in profile_args) and not all(
        value is not None for value in profile_args
    ):
        raise ValueError(
            "--base-profile-config, --write-profile-config, and "
            "--profile-output-dir must be supplied together"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss = import_faiss(args.faiss_egg_dir)
    vectors = load_fbin_vectors(args.query_vec_path.resolve())
    allowed_cpus = sorted(
        os.sched_getaffinity(0)
        if hasattr(os, "sched_getaffinity")
        else range(os.cpu_count() or 1)
    )
    cpu_limit = args.max_cpu_slots or len(allowed_cpus)

    load_started_at = time.perf_counter()
    index = faiss.read_index(str(args.index_path.resolve()))
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    index_mode = "cpu_ivfpq"
    downcast = getattr(faiss, "downcast_index", lambda value: value)(index)
    fastscan_type = getattr(faiss, "IndexIVFPQFastScan", None)
    if fastscan_type is not None and isinstance(downcast, fastscan_type):
        downcast.implem = 12
        downcast.use_precomputed_table = True
        index_mode = "cpu_ivfpq_fastscan"
    load_seconds = time.perf_counter() - load_started_at
    parallel_mode = int(getattr(index, "parallel_mode", 0))
    print(
        f"Loaded index in {load_seconds:.2f}s; vectors={len(vectors)} "
        f"dimension={vectors.shape[1]} parallel_mode={parallel_mode} "
        f"index_mode={index_mode}"
    )

    results: list[CalibrationResult] = []
    for threads_per_search in args.threads_per_search:
        for workers in args.worker_counts:
            if workers <= 0 or threads_per_search <= 0:
                raise ValueError("worker and thread counts must be positive")
            if workers * threads_per_search > cpu_limit:
                print(
                    f"Skipping workers={workers} threads={threads_per_search}: "
                    f"cpu_slots={workers * threads_per_search} > {cpu_limit}"
                )
                continue
            print(
                f"Calibrating workers={workers} "
                f"threads_per_search={threads_per_search}"
            )
            result = calibrate_one(
                faiss=faiss,
                index=index,
                vectors=vectors,
                workers=workers,
                threads_per_search=threads_per_search,
                warmup_queries=args.warmup_queries,
                saturation_queries=args.saturation_queries,
                open_loop_queries=args.open_loop_queries,
                target_rps=args.target_rps,
                topk=args.topk,
                required_headroom=args.required_headroom,
                max_queue_p99_ms=args.max_queue_p99_ms,
                max_queue_slope_ms_per_request=(
                    args.max_queue_slope_ms_per_request
                ),
            )
            results.append(result)
            print(
                f"  saturation={result.saturation_qps:.3f} qps "
                f"headroom={result.capacity_headroom:.2f}x "
                f"open_queue_p99={result.open_loop_queue_p99_ms} "
                f"eligible={result.eligible}"
            )

    if not results:
        raise RuntimeError("No CPU ANN configurations were eligible to run")

    selected = select_result(results)
    status = "selected" if selected is not None else "insufficient_capacity"
    json_path = output_dir / "cpu_ann_calibration.json"
    csv_path = output_dir / "cpu_ann_calibration.csv"
    payload = {
        "status": status,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "allowed_logical_cpus": allowed_cpus,
            "cpu_slot_limit": cpu_limit,
        },
        "inputs": {
            "index_path": str(args.index_path.resolve()),
            "query_vec_path": str(args.query_vec_path.resolve()),
            "nprobe": args.nprobe,
            "topk": args.topk,
            "index_load_seconds": load_seconds,
            "index_parallel_mode": parallel_mode,
            "target_rps": args.target_rps,
            "required_headroom": args.required_headroom,
            "max_queue_p99_ms": args.max_queue_p99_ms,
            "max_queue_slope_ms_per_request": (
                args.max_queue_slope_ms_per_request
            ),
        },
        "selected": asdict(selected) if selected is not None else None,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, results)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if selected is not None and args.base_profile_config is not None:
        write_profile_config(
            base_path=args.base_profile_config.resolve(),
            output_path=args.write_profile_config.resolve(),
            profile_output_dir=args.profile_output_dir.resolve(),
            selected=selected,
            calibration_json=json_path,
        )
        print(f"Wrote {args.write_profile_config.resolve()}")

    if selected is None:
        best = max(results, key=lambda result: result.saturation_qps)
        raise RuntimeError(
            "No configuration met the stability/headroom requirements. "
            f"Best saturation throughput was {best.saturation_qps:.3f} qps "
            f"with workers={best.workers}, "
            f"threads_per_search={best.threads_per_search}."
        )


if __name__ == "__main__":
    main()
