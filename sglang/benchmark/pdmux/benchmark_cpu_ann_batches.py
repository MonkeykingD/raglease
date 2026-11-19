#!/usr/bin/env python3
"""Measure sequential Faiss FastScan batch throughput without index reentrancy."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, count=2, dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"Invalid fbin header: {path}")
        count, dimension = (int(value) for value in header)
        values = np.fromfile(handle, dtype=np.float32)
    if values.size != count * dimension:
        raise ValueError(
            f"Invalid fbin payload: shape=({count}, {dimension}), "
            f"floats={values.size}"
        )
    return np.ascontiguousarray(values.reshape(count, dimension))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--query-vec-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faiss-egg-dir", type=Path)
    parser.add_argument("--nprobe", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32],
    )
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measure-queries", type=int, default=128)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    args = parse_args()
    if args.faiss_egg_dir is not None:
        egg = str(args.faiss_egg_dir.resolve())
        if egg not in sys.path:
            sys.path.insert(0, egg)
    faiss = importlib.import_module("faiss")
    vectors = load_fbin(args.query_vec_path.resolve())

    load_started = time.perf_counter()
    index = faiss.read_index(str(args.index_path.resolve()))
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    downcast = getattr(faiss, "downcast_index", lambda value: value)(index)
    fastscan_type = getattr(faiss, "IndexIVFPQFastScan", None)
    index_mode = "cpu_ivfpq"
    if fastscan_type is not None and isinstance(downcast, fastscan_type):
        downcast.implem = 12
        downcast.use_precomputed_table = True
        index_mode = "cpu_ivfpq_fastscan"
    load_seconds = time.perf_counter() - load_started
    faiss.omp_set_num_threads(args.threads)

    rows = []
    cursor = 0
    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        for _ in range(args.warmup_batches):
            rows_idx = np.arange(cursor, cursor + batch_size) % len(vectors)
            index.search(np.ascontiguousarray(vectors[rows_idx]), args.topk)
            cursor += batch_size

        calls = max(1, (args.measure_queries + batch_size - 1) // batch_size)
        durations_ms: list[float] = []
        measured_queries = 0
        wall_started = time.perf_counter()
        for _ in range(calls):
            current = min(batch_size, args.measure_queries - measured_queries)
            if current <= 0:
                break
            rows_idx = np.arange(cursor, cursor + current) % len(vectors)
            queries = np.ascontiguousarray(vectors[rows_idx])
            started = time.perf_counter()
            index.search(queries, args.topk)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
            measured_queries += current
            cursor += current
        wall_seconds = time.perf_counter() - wall_started
        row = {
            "batch_size": batch_size,
            "calls": len(durations_ms),
            "queries": measured_queries,
            "wall_seconds": wall_seconds,
            "throughput_qps": measured_queries / wall_seconds,
            "batch_latency_mean_ms": float(np.mean(durations_ms)),
            "batch_latency_p50_ms": percentile(durations_ms, 50),
            "batch_latency_p99_ms": percentile(durations_ms, 99),
            "amortized_mean_ms_per_query": (
                sum(durations_ms) / measured_queries
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "index_path": str(args.index_path.resolve()),
        "query_vec_path": str(args.query_vec_path.resolve()),
        "index_mode": index_mode,
        "index_load_seconds": load_seconds,
        "nprobe": args.nprobe,
        "topk": args.topk,
        "threads": args.threads,
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
