"""Offline profiler for the PD-Mux RAG frontend scheduler.

The profiler starts a fresh SGLang server for every candidate, validates the
live server configuration, runs ``bench_rag_serving.py``, and tears down the
whole server process group. It then aggregates repetitions and emits a
deployment profile selected by robust goodput under the configured latency SLO.

Only scheduler/resource parameters are swept here. Retrieval quality settings
such as index type, nprobe, and top-k are inputs supplied by the task.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .rag_workload import generate_workload, load_workload, save_workload
except ImportError:
    from rag_workload import generate_workload, load_workload, save_workload


HYBRID_INDEX_POLICIES = frozenset(
    {
        "vectorliterag",
        "hedrarag-hybrid",
        "naive-hybrid",
        "raglease-hybrid",
    }
)
RAGLEASE_POLICIES = frozenset({"full", "raglease-hybrid"})
UNCOORDINATED_BASELINE_POLICIES = frozenset(
    {
        "ann-only",
        "naive",
        "vectorliterag",
        "hedrarag-hybrid",
        "naive-hybrid",
    }
)
UNCOORDINATED_ANN_DEFAULTS = {
    "ann_max_inflight": 32,
    "ann_max_batches_per_tick": 0,
    "ann_max_inflight_during_decode": 0,
    "ann_max_batch_size": 8,
    "ann_max_queued_gpu_ms": 0.0,
    "ann_max_microbatch_gpu_ms": 0.0,
    "ann_max_batch_wait_ms": 0.0,
    "ann_batch_wait_min_pending": 1,
    "ann_decode_only_wait_ms": 0.0,
}
VECTORLITERAG_ANN_DEFAULTS = {
    **UNCOORDINATED_ANN_DEFAULTS,
    "ann_max_inflight": 1,
    "ann_max_batches_per_tick": 1,
    "ann_max_batch_size": 64,
}
HYBRID_ANN_DEFAULTS = dict(VECTORLITERAG_ANN_DEFAULTS)


def effective_ann_runtime_value(
    runtime: Dict[str, Any], policy: str, key: str, default: Any
) -> Any:
    if policy in HYBRID_INDEX_POLICIES:
        policy_key = f"{policy.replace('-', '_')}_{key}"
        shared_key = f"hybrid_{key}"
        return runtime.get(
            policy_key,
            runtime.get(
                shared_key,
                HYBRID_ANN_DEFAULTS.get(key, runtime.get(key, default)),
            ),
        )
    if policy not in UNCOORDINATED_BASELINE_POLICIES:
        return runtime.get(key, default)
    baseline_key = f"baseline_{key}"
    return runtime.get(
        baseline_key,
        UNCOORDINATED_ANN_DEFAULTS.get(key, runtime.get(key, default)),
    )


def effective_mem_fraction_static(
    runtime: Dict[str, Any], policy: str, default: float = 0.75
) -> float:
    policy_key = f"{policy.replace('-', '_')}_mem_fraction_static"
    return float(runtime.get(policy_key, runtime.get("mem_fraction_static", default)))


def hybrid_cpu_search_threads(
    runtime: Dict[str, Any], policy: str, default: int = 8
) -> int:
    policy_key = f"{policy.replace('-', '_')}_cpu_search_threads"
    return int(
        runtime.get(
            policy_key,
            runtime.get("hybrid_cpu_search_threads", default),
        )
    )


def hybrid_cpu_search_max_batch_queries(
    runtime: Dict[str, Any], policy: str, default: int = 4
) -> int:
    policy_key = f"{policy.replace('-', '_')}_cpu_search_max_batch_queries"
    return int(
        runtime.get(
            policy_key,
            runtime.get("hybrid_cpu_search_max_batch_queries", default),
        )
    )


def resolve_hybrid_index_paths(
    paths: Dict[str, Any], policy: Optional[str] = None
) -> tuple[str, str, str]:
    policy_prefix = (
        "raglease_hybrid"
        if policy == "raglease-hybrid"
        and any(
            paths.get(f"raglease_hybrid_{suffix}")
            for suffix in (
                "cpu_index_path",
                "hot_index_path",
                "hot_index_map_path",
            )
        )
        else "hybrid"
    )
    cpu_index_path = paths.get(
        f"{policy_prefix}_cpu_index_path",
        paths.get("hybrid_cpu_index_path", paths.get("vectorlite_cpu_index_path")),
    )
    hot_index_path = paths.get(
        f"{policy_prefix}_hot_index_path",
        paths.get("hybrid_hot_index_path", paths.get("vectorlite_hot_index_path")),
    )
    hot_index_map_path = paths.get(
        f"{policy_prefix}_hot_index_map_path",
        paths.get(
            "hybrid_hot_index_map_path",
            paths.get("vectorlite_hot_index_map_path"),
        ),
    )
    missing = [
        name
        for name, value in (
            ("hybrid_cpu_index_path", cpu_index_path),
            ("hybrid_hot_index_path", hot_index_path),
            ("hybrid_hot_index_map_path", hot_index_map_path),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Hybrid retrieval requires a CPU index, GPU hot index, and hot "
            "index mapping; configure paths.hybrid_* or the compatible "
            f"paths.vectorlite_* aliases (missing: {', '.join(missing)})"
        )
    return str(cpu_index_path), str(hot_index_path), str(hot_index_map_path)


def cpu_retrieval_runtime_value(
    runtime: Dict[str, Any], policy: str, key: str, default: Any
) -> Any:
    if policy == "hedrarag":
        return runtime.get(f"hedrarag_{key}", runtime.get(key, default))
    return runtime.get(key, default)


def effective_shared_hbm_mode(
    runtime: Dict[str, Any], case: "ProfileCase"
) -> str:
    """Keep hybrid baselines static unless RAGLease explicitly opts in."""
    configured = str(runtime.get("shared_hbm_mode", "off"))
    if not case.is_raglease:
        return "off"
    if case.uses_hybrid_index and not bool(
        runtime.get("raglease_hybrid_shared_hbm_enabled", False)
    ):
        return "off"
    return configured


def allowed_hybrid_result_index_modes(
    runtime: Dict[str, Any], case: "ProfileCase"
) -> set[str]:
    allowed = {"vectorlite_hot_cold"}
    if effective_shared_hbm_mode(runtime, case) != "off":
        allowed.add("cpu_fallback")
    return allowed


@dataclass(frozen=True)
class ProfileCase:
    schedule: str
    request_rate: float
    decode_slo_ms: float
    max_defer_ms: float
    ann_lease_sm: int
    prefill_lease_sm: int
    pdmux_config_path: str
    repetition: int
    policy: str = "full"
    seed: int = 1
    stream_epoch_ms: Optional[float] = None
    ann_max_inflight: Optional[int] = None
    prefill_lease_min_layers: Optional[int] = None
    prefill_lease_max_layers: Optional[int] = None
    prefill_max_consecutive_chunks: Optional[int] = None

    @property
    def policy_mode(self) -> str:
        return "dynamic-only" if self.policy == "dynamic-only" else "full"

    @property
    def is_cpu_retrieval(self) -> bool:
        return self.policy in {"cpu-retrieval", "hedrarag"}

    @property
    def is_hedrarag(self) -> bool:
        return self.policy == "hedrarag"

    @property
    def is_hedrarag_hybrid(self) -> bool:
        return self.policy == "hedrarag-hybrid"

    @property
    def is_llm_only(self) -> bool:
        return self.policy == "llm-only"

    @property
    def is_ann_only(self) -> bool:
        return self.policy == "ann-only"

    @property
    def is_vectorliterag(self) -> bool:
        return self.policy == "vectorliterag"

    @property
    def uses_hybrid_index(self) -> bool:
        return self.policy in HYBRID_INDEX_POLICIES

    @property
    def is_raglease(self) -> bool:
        return self.policy in RAGLEASE_POLICIES

    @property
    def retrieval_backend(self) -> str:
        if self.is_llm_only:
            return "none"
        if self.is_cpu_retrieval:
            return "cpu"
        if self.uses_hybrid_index:
            return "hot_gpu_cold_cpu"
        return "full_gpu"

    @property
    def uses_pdmux(self) -> bool:
        return self.policy not in {
            "cpu-retrieval",
            "hedrarag",
            "hedrarag-hybrid",
            "ann-only",
            "llm-only",
            *UNCOORDINATED_BASELINE_POLICIES,
        }

    @property
    def key(self) -> str:
        epoch = (
            ""
            if self.stream_epoch_ms is None
            else f"_epoch{format_number(self.stream_epoch_ms)}"
        )
        prefill_bundle = (
            ""
            if self.prefill_lease_min_layers is None
            or self.prefill_lease_max_layers is None
            else (
                f"_pl{self.prefill_lease_min_layers}"
                f"-{self.prefill_lease_max_layers}"
            )
        )
        ann_inflight = (
            ""
            if self.ann_max_inflight is None
            else f"_ai{self.ann_max_inflight}"
        )
        prefill_recovery = (
            ""
            if self.prefill_max_consecutive_chunks is None
            else f"_pc{self.prefill_max_consecutive_chunks}"
        )
        return (
            f"{self.policy}_{self.schedule}_rps{format_number(self.request_rate)}_"
            f"slo{format_number(self.decode_slo_ms)}_"
            f"defer{format_number(self.max_defer_ms)}_"
            f"ann{self.ann_lease_sm}_pf{self.prefill_lease_sm}_"
            f"cfg{config_label(self.pdmux_config_path)}{epoch}"
            f"{ann_inflight}{prefill_bundle}{prefill_recovery}"
        )

    @property
    def name(self) -> str:
        return f"{self.key}_seed{self.seed}_rep{self.repetition}"


def format_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def config_label(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem).strip("-")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95_half_width(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
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
    }.get(len(values) - 1, 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def telemetry_distribution(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "samples": len(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0.0,
    }


class GpuTelemetrySampler:
    """Poll nvidia-smi and retain samples from the measured request window."""

    def __init__(self, gpu_index: int, interval_ms: float):
        if interval_ms <= 0:
            raise ValueError("GPU telemetry interval must be positive")
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self._samples: List[Dict[str, float]] = []
        self._errors: List[str] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _query_gpu(self) -> Dict[str, float]:
        command = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(2.0, 4.0 * self.interval_ms / 1000.0),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
        lines = result.stdout.splitlines()
        if not lines:
            raise RuntimeError("nvidia-smi returned no GPU telemetry")
        values = [value.strip() for value in lines[0].split(",")]
        if len(values) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout.strip()}")
        return {
            "timestamp_unix_s": time.time(),
            "gpu_util_pct": float(values[0]),
            "memory_bandwidth_util_pct": float(values[1]),
            "memory_used_mib": float(values[2]),
        }

    def _run(self) -> None:
        interval_s = self.interval_ms / 1000.0
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._samples.append(self._query_gpu())
            except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                if len(self._errors) < 10:
                    self._errors.append(str(error))
            remaining = max(0.0, interval_s - (time.monotonic() - started))
            self._stop_event.wait(remaining)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU telemetry sampler is already running")
        self._thread = threading.Thread(
            target=self._run,
            name="rag-gpu-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 4.0 * self.interval_ms / 1000.0))

    def summarize(
        self, measurement_start_unix_s: float, measurement_end_unix_s: float
    ) -> Dict[str, Any]:
        measured = [
            sample
            for sample in self._samples
            if measurement_start_unix_s
            <= sample["timestamp_unix_s"]
            <= measurement_end_unix_s
        ]
        return {
            "available": bool(measured),
            "gpu_index": self.gpu_index,
            "sampling_interval_ms": self.interval_ms,
            "measurement_start_unix_s": measurement_start_unix_s,
            "measurement_end_unix_s": measurement_end_unix_s,
            "sample_count": len(measured),
            "gpu_util_pct": telemetry_distribution(
                sample["gpu_util_pct"] for sample in measured
            ),
            "memory_bandwidth_util_pct": telemetry_distribution(
                sample["memory_bandwidth_util_pct"] for sample in measured
            ),
            "memory_used_mib": telemetry_distribution(
                sample["memory_used_mib"] for sample in measured
            ),
            "errors": self._errors,
            "metric_definition": (
                "nvidia-smi utilization.memory: percentage of the sample period "
                "during which device memory was being read or written"
            ),
        }


def nested_get(data: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_yaml_compatible(path: Path, data: Dict[str, Any]) -> None:
    """Write YAML when PyYAML is available; JSON is valid YAML otherwise."""
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(data, indent=2)
    path.write_text(text, encoding="utf-8")


def require(config: Dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {}).get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config field: {section}.{key}")
    return value


def validate_experiment_contract(config: Dict[str, Any]) -> None:
    """Reject paired experiments whose policy comparison changes the workload."""
    paths = config.get("paths", {})
    runtime = config.get("runtime", {})
    sweep = config.get("sweep", {})
    contract = config.get("comparison_contract", {})

    if bool(contract.get("require_real_documents", False)):
        if bool(runtime.get("doc_random_fallback", False)):
            raise ValueError(
                "Real-document comparison forbids "
                "runtime.doc_random_fallback=true"
            )
        if not (
            paths.get("doc_db_path") or paths.get("doc_store_path")
        ):
            raise ValueError(
                "Real-document comparison requires paths.doc_db_path or "
                "paths.doc_store_path"
            )

    permutation_path = str(paths.get("query_permutation_path", ""))
    permutation_match = re.search(
        r"(?:^|[_-])nprobe(?P<nprobe>\d+)(?:[_\.-]|$)",
        Path(permutation_path).name,
    )
    if permutation_match and "nprobe" in runtime:
        permutation_nprobe = int(permutation_match.group("nprobe"))
        runtime_nprobe = int(runtime["nprobe"])
        if (
            permutation_nprobe != runtime_nprobe
            and not bool(
                contract.get("allow_query_permutation_nprobe_mismatch", False)
            )
        ):
            raise ValueError(
                "Query permutation nprobe mismatch: "
                f"path encodes nprobe={permutation_nprobe}, "
                f"runtime.nprobe={runtime_nprobe}"
            )

    policies = {str(value) for value in sweep.get("paired_policies", [])}
    hybrid_pair = {"naive-hybrid", "raglease-hybrid"}.issubset(policies)
    raglease_shared_hbm = (
        str(runtime.get("shared_hbm_mode", "off")) != "off"
        and bool(runtime.get("raglease_hybrid_shared_hbm_enabled", False))
    )
    if (
        hybrid_pair
        and raglease_shared_hbm
        and not bool(contract.get("allow_policy_specific_shared_hbm", False))
    ):
        raise ValueError(
            "Naive-Hybrid vs RAGLease-Hybrid changes shared-HBM mode. "
            "Disable raglease_hybrid_shared_hbm_enabled for a matched-compute "
            "comparison, or explicitly set comparison_contract."
            "allow_policy_specific_shared_hbm=true for a memory-system "
            "experiment."
        )


def build_cases(config: Dict[str, Any]) -> List[ProfileCase]:
    validate_experiment_contract(config)
    sweep = config["sweep"]
    managed_schedule = str(
        config.get("runtime", {}).get("ann_schedule", "decode-gc")
    )
    if managed_schedule not in {"primary", "frontend-gc", "decode-gc"}:
        raise ValueError(
            "runtime.ann_schedule must be primary, frontend-gc, or decode-gc, "
            f"got {managed_schedule!r}"
        )
    repetitions = int(sweep.get("repetitions", 3))
    request_rates = [float(value) for value in sweep["request_rates"]]
    decode_slos = [float(value) for value in sweep["decode_slo_ms"]]
    max_defers = [float(value) for value in sweep["max_defer_ms"]]
    ann_leases = [int(value) for value in sweep["ann_lease_sm"]]
    prefill_leases = [int(value) for value in sweep["prefill_lease_sm"]]
    pdmux_configs = [
        str(path)
        for path in sweep.get(
            "pdmux_config_paths", [config["paths"]["pdmux_config_path"]]
        )
    ]
    stream_epochs = (
        [float(value) for value in sweep["ann_stream_epoch_ms"]]
        if "ann_stream_epoch_ms" in sweep
        else [None]
    )
    raw_prefill_layer_bounds = sweep.get("prefill_lease_layer_bounds")
    prefill_layer_bounds: List[tuple[Optional[int], Optional[int]]] = [
        (None, None)
    ]
    if raw_prefill_layer_bounds is not None:
        prefill_layer_bounds = []
        for bounds in raw_prefill_layer_bounds:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "Each sweep.prefill_lease_layer_bounds entry must be "
                    "[min_layers, max_layers]"
                )
            min_layers, max_layers = (int(bounds[0]), int(bounds[1]))
            if min_layers < 1 or max_layers < min_layers:
                raise ValueError(
                    "Prefill lease layer bounds require "
                    "1 <= min_layers <= max_layers"
                )
            prefill_layer_bounds.append((min_layers, max_layers))
        if not prefill_layer_bounds:
            raise ValueError(
                "sweep.prefill_lease_layer_bounds must not be empty"
            )
    raw_prefill_chunk_limits = sweep.get("prefill_max_consecutive_chunks")
    prefill_chunk_limits: List[Optional[int]] = [None]
    if raw_prefill_chunk_limits is not None:
        prefill_chunk_limits = [
            int(value) for value in raw_prefill_chunk_limits
        ]
        if not prefill_chunk_limits or any(
            value < 1 for value in prefill_chunk_limits
        ):
            raise ValueError(
                "sweep.prefill_max_consecutive_chunks must contain positive "
                "integers"
            )
    raw_ann_max_inflight = sweep.get("ann_max_inflight")
    ann_max_inflight_values: List[Optional[int]] = [None]
    if raw_ann_max_inflight is not None:
        ann_max_inflight_values = [int(value) for value in raw_ann_max_inflight]
        if not ann_max_inflight_values or any(
            value < 1 for value in ann_max_inflight_values
        ):
            raise ValueError(
                "sweep.ann_max_inflight must contain positive integers"
            )
        if len(ann_max_inflight_values) != len(set(ann_max_inflight_values)):
            raise ValueError("sweep.ann_max_inflight must not contain duplicates")

    cases: List[ProfileCase] = []
    paired_policies = sweep.get("paired_policies")
    if paired_policies:
        policies = [str(value) for value in paired_policies]
        supported = {
            "cpu-retrieval",
            "hedrarag",
            "hedrarag-hybrid",
            "ann-only",
            "llm-only",
            "naive",
            "naive-hybrid",
            "vectorliterag",
            "dynamic-only",
            "full",
            "raglease-hybrid",
        }
        unknown = set(policies) - supported
        if unknown:
            raise ValueError(f"Unsupported paired policies: {sorted(unknown)}")
        if len(policies) != len(set(policies)):
            raise ValueError("sweep.paired_policies must not contain duplicates")
        primary_raglease_probe = (
            len(policies) == 1 and policies[0] in RAGLEASE_POLICIES
        )
        if managed_schedule == "primary" and not primary_raglease_probe:
            raise ValueError(
                "runtime.ann_schedule=primary is reserved for a single "
                "RAGLease placement probe"
            )
        if len(stream_epochs) > 1 and not primary_raglease_probe:
            raise ValueError(
                "Multi-epoch sweeps require one RAGLease policy to avoid "
                "duplicating policies whose epoch is disabled or irrelevant"
            )
        if len(prefill_layer_bounds) > 1 and not primary_raglease_probe:
            raise ValueError(
                "Multi-value prefill_lease_layer_bounds sweeps require exactly "
                "one RAGLease policy"
            )
        if len(prefill_chunk_limits) > 1 and not primary_raglease_probe:
            raise ValueError(
                "Multi-value prefill_max_consecutive_chunks sweeps require "
                "exactly one RAGLease policy"
            )
        if raw_ann_max_inflight is not None and not primary_raglease_probe:
            raise ValueError(
                "sweep.ann_max_inflight requires exactly one RAGLease policy"
            )
        lease_dimensions = (
            decode_slos,
            max_defers,
            ann_leases,
            prefill_leases,
        )
        if not primary_raglease_probe and any(
            len(values) > 1 for values in lease_dimensions
        ):
            raise ValueError(
                "Multi-value decode_slo_ms, max_defer_ms, ann_lease_sm, or "
                "prefill_lease_sm sweeps require exactly one RAGLease policy"
            )
        lease_settings = (
            list(itertools.product(*lease_dimensions))
            if primary_raglease_probe
            else [
                (
                    decode_slos[0],
                    max_defers[0],
                    ann_leases[0],
                    prefill_leases[0],
                )
            ]
        )

        seeds = [int(value) for value in sweep.get("seeds", [])]
        if not seeds:
            base_seed = int(config.get("workload", {}).get("seed", 1))
            seeds = list(range(base_seed, base_seed + repetitions))
        if len(seeds) != len(set(seeds)):
            raise ValueError("sweep.seeds must not contain duplicates")
        repetitions_per_seed = int(sweep.get("repetitions_per_seed", 1))
        if repetitions_per_seed < 1:
            raise ValueError("sweep.repetitions_per_seed must be positive")

        for (
            request_rate,
            pdmux_config,
            seed,
            stream_epoch_ms,
            ann_max_inflight,
            prefill_layer_bound,
            prefill_chunk_limit,
            repetition_within_seed,
        ) in itertools.product(
            request_rates,
            pdmux_configs,
            seeds,
            stream_epochs,
            ann_max_inflight_values,
            prefill_layer_bounds,
            prefill_chunk_limits,
            range(1, repetitions_per_seed + 1),
        ):
            repetition = (
                seeds.index(seed) * repetitions_per_seed
                + repetition_within_seed
            )
            for slo, max_defer, ann_sm, prefill_sm in lease_settings:
                for policy in policies:
                    is_raglease = policy in RAGLEASE_POLICIES
                    cases.append(
                        ProfileCase(
                            schedule=(
                                "primary"
                                if policy in UNCOORDINATED_BASELINE_POLICIES
                                or policy
                                in {
                                    "cpu-retrieval",
                                    "hedrarag",
                                    "llm-only",
                                }
                                else managed_schedule
                            ),
                            request_rate=request_rate,
                            decode_slo_ms=(slo if is_raglease else -1.0),
                            max_defer_ms=(max_defer if is_raglease else 0.0),
                            ann_lease_sm=(ann_sm if is_raglease else -1),
                            prefill_lease_sm=(
                                prefill_sm if is_raglease else -1
                            ),
                            pdmux_config_path=pdmux_config,
                            repetition=repetition,
                            policy=policy,
                            seed=seed,
                            stream_epoch_ms=(
                                stream_epoch_ms if is_raglease else None
                            ),
                            ann_max_inflight=(
                                ann_max_inflight if is_raglease else None
                            ),
                            prefill_lease_min_layers=(
                                prefill_layer_bound[0] if is_raglease else None
                            ),
                            prefill_lease_max_layers=(
                                prefill_layer_bound[1] if is_raglease else None
                            ),
                            prefill_max_consecutive_chunks=(
                                prefill_chunk_limit if is_raglease else None
                            ),
                        )
                    )
        if sweep.get("randomize_case_order", False):
            random.Random(int(sweep.get("case_order_seed", 1))).shuffle(cases)
        return cases

    if raw_prefill_layer_bounds is not None:
        raise ValueError(
            "sweep.prefill_lease_layer_bounds requires paired_policies mode"
        )
    if raw_prefill_chunk_limits is not None:
        raise ValueError(
            "sweep.prefill_max_consecutive_chunks requires paired_policies mode"
        )

    if managed_schedule == "primary":
        raise ValueError(
            "runtime.ann_schedule=primary requires the paired Full-policy mode"
        )

    if sweep.get("include_primary_baseline", True):
        for request_rate, pdmux_config, repetition in itertools.product(
            request_rates, pdmux_configs, range(1, repetitions + 1)
        ):
            cases.append(
                ProfileCase(
                    schedule="primary",
                    request_rate=request_rate,
                    decode_slo_ms=decode_slos[0],
                    max_defer_ms=max_defers[0],
                    ann_lease_sm=ann_leases[0],
                    prefill_lease_sm=prefill_leases[0],
                    pdmux_config_path=pdmux_config,
                    repetition=repetition,
                    policy="naive",
                    seed=int(config.get("workload", {}).get("seed", 1))
                    + repetition
                    - 1,
                )
            )

    frontend_candidates = itertools.product(
        request_rates,
        decode_slos,
        max_defers,
        ann_leases,
        prefill_leases,
        pdmux_configs,
        stream_epochs,
        range(1, repetitions + 1),
    )
    for (
        request_rate,
        slo,
        max_defer,
        ann_sm,
        prefill_sm,
        pdmux_config,
        stream_epoch_ms,
        repetition,
    ) in frontend_candidates:
        cases.append(
            ProfileCase(
                schedule=managed_schedule,
                request_rate=request_rate,
                decode_slo_ms=slo,
                max_defer_ms=max_defer,
                ann_lease_sm=ann_sm,
                prefill_lease_sm=prefill_sm,
                pdmux_config_path=pdmux_config,
                repetition=repetition,
                policy="full",
                seed=int(config.get("workload", {}).get("seed", 1))
                + repetition
                - 1,
                stream_epoch_ms=stream_epoch_ms,
            )
        )
    if sweep.get("randomize_case_order", True):
        random.Random(int(sweep.get("case_order_seed", 1))).shuffle(cases)
    return cases


class OfflineProfiler:
    def __init__(self, config: Dict[str, Any], output_dir: Path, overwrite: bool):
        self.config = config
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.paths = config["paths"]
        self.runtime = dict(config["runtime"])
        self.workload = config["workload"]
        self.selection = config["selection"]
        self.runtime.setdefault(
            "e2e_slo_ms", self.selection.get("e2e_slo_ms", -1)
        )
        self.project_root = Path(require(config, "paths", "project_root"))
        self.python = str(self.paths.get("python", sys.executable))
        self.host = str(self.runtime.get("host", "127.0.0.1"))
        self.port = int(self.runtime.get("port", 30000))
        self.physical_gpu = int(self.runtime.get("physical_gpu", 0))
        self.base_url = f"http://{self.host}:{self.port}"
        self.server_process: Optional[subprocess.Popen[str]] = None

    def result_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "cases" / f"{case.name}.json"

    def server_log_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "logs" / f"{case.name}.server.log"

    def benchmark_log_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "logs" / f"{case.name}.benchmark.log"

    def server_info_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "cases" / f"{case.name}.server_info.json"

    def server_info_after_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "cases" / f"{case.name}.server_info_after.json"

    def ann_priming_path(self, case: ProfileCase) -> Path:
        return self.output_dir / "cases" / f"{case.name}.ann_priming.json"

    def workload_trace_path(self, case: ProfileCase) -> Path:
        return (
            self.output_dir
            / "workloads"
            / (
                f"rps{format_number(case.request_rate)}_"
                f"seed{case.seed}.jsonl"
            )
        )

    def expected_workload(self, case: ProfileCase):
        return generate_workload(
            num_requests=int(self.workload.get("num_requests", 200)),
            ramp_up_requests=int(self.workload.get("ramp_up_requests", 50)),
            warmup_requests=int(self.workload.get("warmup_requests", 16)),
            request_rate=case.request_rate,
            output_len=int(self.workload.get("output_len", 256)),
            topk=int(self.runtime["topk"]),
            seed=case.seed,
            deterministic_arrivals=bool(
                self.workload.get("deterministic_arrivals", False)
            ),
            query_count=(
                int(self.workload["query_count"])
                if self.workload.get("query_count") is not None
                else None
            ),
            query_id_start=int(self.workload.get("query_id_start", 0)),
            warmup_output_len=int(self.workload.get("warmup_output_len", 32)),
            warmup_interval_ms=float(
                self.workload.get("warmup_interval_ms", 1000.0)
            ),
            burst_size=(
                int(self.workload["burst_size"])
                if self.workload.get("burst_size") is not None
                else None
            ),
            inter_burst_gap_ms=float(
                self.workload.get("inter_burst_gap_ms", 0.0)
            ),
            temperature=float(self.workload.get("temperature", 0.0)),
            ignore_eos=bool(self.workload.get("ignore_eos", True)),
        )

    def prepare_workload_traces(self, cases: List[ProfileCase]) -> None:
        unique_cases = {
            (case.request_rate, case.seed): case for case in cases
        }.values()
        for case in unique_cases:
            path = self.workload_trace_path(case)
            expected = self.expected_workload(case)
            if path.exists() and not self.overwrite:
                actual = load_workload(path)
                if actual.sha256() != expected.sha256():
                    raise RuntimeError(
                        f"Existing workload mismatch: {path}; pass --overwrite "
                        "only if replacing the trace is intentional"
                    )
                continue
            save_workload(path, expected)

    def gpu_pids(self) -> List[int]:
        command = [
            "nvidia-smi",
            "-i",
            str(self.physical_gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    def server_info(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/get_server_info", timeout=timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def assert_clean(self) -> None:
        if self.server_info() is not None:
            raise RuntimeError(f"Port {self.port} already has an SGLang server")
        pids = self.gpu_pids()
        if pids:
            raise RuntimeError(
                f"Physical GPU {self.physical_gpu} is not clean; compute PIDs={pids}"
            )

    def build_server_command(self, case: ProfileCase) -> List[str]:
        ann_launch_site = (
            "early"
            if not case.uses_pdmux
            else str(self.runtime.get("ann_launch_site", "after-prefill"))
        )
        frontend_arbitration = (
            "overlap"
            if not case.uses_pdmux
            else str(self.runtime.get("frontend_arbitration", "fair"))
        )
        handoff_timeout_ms = (
            float(self.runtime.get("ann_handoff_timeout_ms", 30))
            if case.is_raglease
            else 0.0
        )
        pdmux_args = (
            [
                "--enable-pdmux",
                "--pdmux-config-path",
                case.pdmux_config_path,
            ]
            if case.uses_pdmux
            else []
        )
        index_path = str(require(self.config, "paths", "index_path"))
        hybrid_paths = None
        if case.uses_hybrid_index:
            hybrid_paths = resolve_hybrid_index_paths(self.paths, case.policy)
            index_path = hybrid_paths[0]
        elif case.is_hedrarag:
            index_path = str(
                self.paths.get(
                    "hedrarag_cpu_index_path",
                    self.paths.get(
                        "cpu_index_path",
                        self.paths.get("vectorlite_cpu_index_path", index_path),
                    ),
                )
            )
        elif case.is_cpu_retrieval and self.paths.get("cpu_index_path"):
            index_path = str(self.paths["cpu_index_path"])
        ann_max_inflight = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_inflight", 8
        )
        if case.ann_max_inflight is not None:
            ann_max_inflight = case.ann_max_inflight
        ann_max_batches_per_tick = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_batches_per_tick", 0
        )
        ann_max_inflight_during_decode = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_inflight_during_decode", 0
        )
        ann_max_batch_size = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_batch_size", 8
        )
        ann_max_queued_gpu_ms = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_queued_gpu_ms", 30
        )
        ann_max_microbatch_gpu_ms = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_microbatch_gpu_ms", 20
        )
        ann_max_batch_wait_ms = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_max_batch_wait_ms", 2
        )
        ann_batch_wait_min_pending = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_batch_wait_min_pending", 2
        )
        ann_decode_only_wait_ms = effective_ann_runtime_value(
            self.runtime, case.policy, "ann_decode_only_wait_ms", 10
        )
        command = [
            self.python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(require(self.config, "paths", "model_path")),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tp-size",
            str(self.runtime.get("tp_size", 1)),
            "--random-seed",
            str(self.workload.get("seed", 1)),
            "--mem-fraction-static",
            str(effective_mem_fraction_static(self.runtime, case.policy)),
            "--max-prefill-tokens",
            str(self.runtime.get("max_prefill_tokens", 16384)),
            "--chunked-prefill-size",
            str(self.runtime.get("chunked_prefill_size", -1)),
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
            *pdmux_args,
            "--rag-index-path",
            index_path,
            "--rag-index-nprobe",
            str(self.runtime["nprobe"]),
            "--rag-index-topk",
            str(self.runtime["topk"]),
            "--rag-query-vec-path",
            str(require(self.config, "paths", "query_vec_path")),
            "--rag-query-ids-path",
            str(require(self.config, "paths", "query_ids_path")),
            "--rag-doc-token-cache-size",
            str(self.runtime.get("doc_token_cache_size", 1024)),
            "--rag-ann-schedule",
            case.schedule,
            "--rag-policy-mode",
            case.policy_mode,
            "--rag-e2-fixed-partition-idx",
            str(self.runtime.get("e2_fixed_partition_idx", -1)),
            "--rag-ann-launch-site",
            ann_launch_site,
            "--rag-ann-fast-path-pressure-threshold",
            str(self.runtime.get("ann_fast_path_pressure_threshold", 0.5)),
            "--rag-ann-fast-path-max-work",
            str(self.runtime.get("ann_fast_path_max_work", 1)),
            "--rag-ann-fast-path-decode-batch-p95-threshold",
            str(
                self.runtime.get("ann_fast_path_decode_batch_p95_threshold", 16)
            ),
            "--rag-prefill-low-load-batch-p95-threshold",
            str(self.runtime.get("prefill_low_load_batch_p95_threshold", 4)),
            "--rag-prefill-low-load-arrival-rps-threshold",
            str(
                self.runtime.get(
                    "prefill_low_load_arrival_rps_threshold", -1
                )
            ),
            "--rag-prefill-low-load-reference-tokens",
            str(self.runtime.get("prefill_low_load_reference_tokens", 1024)),
            "--rag-ann-fast-path-decode-queue-p95-ms",
            str(self.runtime.get("ann_fast_path_decode_queue_p95_ms", 2)),
            "--rag-ann-decode-batch-bucket-size",
            str(self.runtime.get("ann_decode_batch_bucket_size", 4)),
            "--rag-ann-frontend-arbitration",
            frontend_arbitration,
            "--rag-ann-deadline-ms",
            str(self.runtime.get("ann_deadline_ms", 50)),
            "--rag-prefill-deadline-ms",
            str(self.runtime.get("prefill_deadline_ms", 500)),
            "--rag-e2e-slo-ms",
            str(self.runtime.get("e2e_slo_ms", -1)),
            "--rag-ann-overlap-pressure-threshold",
            str(self.runtime.get("overlap_pressure_threshold", 0.5)),
            "--rag-ann-max-consecutive-batches",
            str(self.runtime.get("max_consecutive_ann_batches", 1)),
            "--rag-prefill-max-consecutive-chunks",
            str(
                self.runtime.get("max_consecutive_prefill_chunks", 2)
                if case.prefill_max_consecutive_chunks is None
                else case.prefill_max_consecutive_chunks
            ),
            "--rag-ann-max-inflight",
            str(ann_max_inflight),
            "--rag-ann-max-batches-per-tick",
            str(ann_max_batches_per_tick),
            "--rag-ann-max-inflight-during-decode",
            str(ann_max_inflight_during_decode),
            "--rag-ann-max-batch-size",
            str(ann_max_batch_size),
            "--rag-ann-max-queued-gpu-ms",
            str(ann_max_queued_gpu_ms),
            "--rag-ann-max-microbatch-gpu-ms",
            str(ann_max_microbatch_gpu_ms),
            "--rag-ann-service-time-percentile",
            str(self.runtime.get("ann_service_time_percentile", 95)),
            "--rag-ann-service-time-min-samples",
            str(self.runtime.get("ann_service_time_min_samples", 4)),
            "--rag-ann-max-batch-wait-ms",
            str(ann_max_batch_wait_ms),
            "--rag-ann-batch-wait-min-pending",
            str(ann_batch_wait_min_pending),
            "--rag-ann-decode-only-wait-ms",
            str(ann_decode_only_wait_ms),
            "--rag-ann-decode-step-slo-ms",
            str(case.decode_slo_ms),
            "--rag-ann-safety-margin-ms",
            str(self.runtime.get("ann_safety_margin_ms", 5)),
            "--rag-prefill-decode-guard-percentile",
            str(self.runtime.get("prefill_decode_guard_percentile", 99)),
            "--rag-ann-urgency-start-ratio",
            str(self.runtime.get("ann_urgency_start_ratio", 1)),
            "--rag-ann-max-decode-slowdown-ratio",
            str(self.runtime.get("ann_max_decode_slowdown_ratio", 1)),
            "--rag-ann-max-defer-ms",
            str(case.max_defer_ms),
            "--rag-ann-stream-epoch-ms",
            str(
                self.runtime.get("ann_stream_epoch_ms", 0)
                if case.stream_epoch_ms is None
                else case.stream_epoch_ms
            ),
            "--rag-ann-handoff-timeout-ms",
            str(handoff_timeout_ms),
            "--rag-ann-lease-frontend-sm",
            str(case.ann_lease_sm),
            "--rag-prefill-lease-frontend-sm",
            str(case.prefill_lease_sm),
            "--rag-partition-hysteresis-bs",
            str(self.runtime.get("partition_hysteresis_bs", 4)),
            "--rag-partition-min-residency-ms",
            str(self.runtime.get("partition_min_residency_ms", 2)),
            "--rag-partition-safe-mixed-hold-ratio",
            str(self.runtime.get("partition_safe_mixed_hold_ratio", 0.9)),
            "--rag-partition-safe-mixed-min-samples",
            str(self.runtime.get("partition_safe_mixed_min_samples", 8)),
            "--rag-partition-safe-mixed-max-hold-ms",
            str(self.runtime.get("partition_safe_mixed_max_hold_ms", 5)),
            "--rag-lease-coalesce-min-work",
            str(self.runtime.get("lease_coalesce_min_work", 1)),
            "--rag-lease-coalesce-max-wait-ms",
            str(self.runtime.get("lease_coalesce_max_wait_ms", 0)),
            "--rag-lease-switch-cost-ms",
            str(self.runtime.get("lease_switch_cost_ms", 0)),
            "--rag-lease-switch-cost-percentile",
            str(self.runtime.get("lease_switch_cost_percentile", 95)),
            "--rag-lease-min-gain-ratio",
            str(self.runtime.get("lease_min_gain_ratio", 1)),
            "--rag-lease-ann-service-estimate-ms",
            str(self.runtime.get("lease_ann_service_estimate_ms", 10)),
            "--rag-lease-prefill-service-estimate-ms",
            str(self.runtime.get("lease_prefill_service_estimate_ms", 10)),
            "--rag-prefill-lease-min-layers",
            str(
                self.runtime.get("prefill_lease_min_layers", 1)
                if case.prefill_lease_min_layers is None
                else case.prefill_lease_min_layers
            ),
            "--rag-prefill-lease-max-layers",
            str(
                self.runtime.get("prefill_lease_max_layers", 8)
                if case.prefill_lease_max_layers is None
                else case.prefill_lease_max_layers
            ),
            "--rag-prefill-lease-decode-ratio",
            str(self.runtime.get("prefill_lease_decode_ratio", 1)),
        ]
        lease_profile_path = self.runtime.get("lease_profile_path")
        if case.is_raglease and lease_profile_path:
            command.extend(
                [
                    "--rag-lease-profile-path",
                    str(lease_profile_path),
                    "--rag-lease-profile-request-rate",
                    str(
                        self.runtime.get(
                            "lease_profile_request_rate", case.request_rate
                        )
                    ),
                ]
            )
        query_permutation_path = self.paths.get("query_permutation_path")
        if query_permutation_path:
            command.extend(
                [
                    "--rag-query-permutation-path",
                    str(query_permutation_path),
                ]
            )
        if self.runtime.get("disable_radix_cache", False):
            command.append("--disable-radix-cache")
        work_conserving = bool(
            case.is_raglease
            and case.schedule == "decode-gc"
            and self.runtime.get("ann_work_conserving_decode_overlap", True)
        )
        if work_conserving:
            command.append("--rag-ann-work-conserving-decode-overlap")
        else:
            command.append(
                "--disable-rag-ann-work-conserving-decode-overlap"
            )
        if case.policy not in UNCOORDINATED_BASELINE_POLICIES and self.runtime.get(
            "lease_round_trip_switch_cost", True
        ):
            command.append("--rag-lease-round-trip-switch-cost")
        else:
            command.append("--disable-rag-lease-round-trip-switch-cost")
        dynamic_prefill_lease = bool(
            case.is_raglease
            and self.runtime.get("prefill_dynamic_lease", True)
        )
        if dynamic_prefill_lease:
            command.append("--rag-prefill-dynamic-lease")
        else:
            command.append("--disable-rag-prefill-dynamic-lease")
        two_level_prefill_bundles = bool(
            case.is_raglease
            and self.runtime.get("prefill_two_level_bundles", False)
        )
        if two_level_prefill_bundles:
            command.append("--rag-prefill-two-level-bundles")
        else:
            command.append("--disable-rag-prefill-two-level-bundles")
        tail_aware_prefill_admission = bool(
            case.is_raglease
            and self.runtime.get("prefill_tail_aware_admission", True)
        )
        if tail_aware_prefill_admission:
            command.append("--rag-prefill-tail-aware-admission")
        else:
            command.append("--disable-rag-prefill-tail-aware-admission")
        exclusive_prefill_sprint = bool(
            case.is_raglease
            and self.runtime.get("prefill_exclusive_sprint", False)
        )
        if exclusive_prefill_sprint:
            command.append("--rag-prefill-exclusive-sprint")
        else:
            command.append("--disable-rag-prefill-exclusive-sprint")
        if (
            case.is_raglease
            and case.schedule == "decode-gc"
            and self.runtime.get("ann_pre_switch_launch", False)
        ):
            command.append("--rag-ann-pre-switch-launch")
        if case.is_raglease and self.runtime.get(
            "enable_controlled_overlap", False
        ):
            command.append("--rag-ann-enable-controlled-overlap")
        if (
            case.is_raglease
            and self.runtime.get("partition_safe_mixed_hold", False)
        ):
            command.append("--rag-partition-safe-mixed-hold")
        partition_decode_slo_ms = self.runtime.get("partition_decode_slo_ms")
        if case.is_raglease and partition_decode_slo_ms:
            command.append("--rag-partition-decode-slo-ms")
            command.extend(str(value) for value in partition_decode_slo_ms)
        decode_slo_batch_thresholds = self.runtime.get(
            "partition_decode_slo_batch_thresholds"
        )
        if case.is_raglease and decode_slo_batch_thresholds:
            command.append("--rag-partition-decode-slo-batch-thresholds")
            command.extend(str(value) for value in decode_slo_batch_thresholds)
            command.extend(
                [
                    "--rag-partition-decode-slo-hysteresis-bs",
                    str(
                        self.runtime.get(
                            "partition_decode_slo_hysteresis_bs", 0
                        )
                    ),
                    "--rag-partition-decode-slo-min-residency-ms",
                    str(
                        self.runtime.get(
                            "partition_decode_slo_min_residency_ms", 0
                        )
                    ),
                    "--rag-partition-decode-slo-initial-calibration-ms",
                    str(
                        self.runtime.get(
                            "partition_decode_slo_initial_calibration_ms", 0
                        )
                    ),
                    "--rag-partition-decode-slo-ewma-alpha",
                    str(
                        self.runtime.get(
                            "partition_decode_slo_ewma_alpha", 0.02
                        )
                    ),
                ]
            )
        floor_enabled = bool(
            case.is_raglease
            and self.runtime.get("partition_slo_floor_enabled", True)
        )
        if not floor_enabled:
            command.append("--disable-rag-partition-slo-floor")
        shared_hbm_mode = effective_shared_hbm_mode(self.runtime, case)
        if case.uses_hybrid_index:
            assert hybrid_paths is not None
            _, hot_index_path, hot_index_map_path = hybrid_paths
        elif case.is_raglease and shared_hbm_mode == "ivf":
            hot_index_path = self.paths.get("hot_index_path")
            hot_index_map_path = self.paths.get("hot_index_map_path")
        else:
            hot_index_path = None
            hot_index_map_path = None
        if bool(hot_index_path) != bool(hot_index_map_path):
            raise ValueError(
                "paths.hot_index_path and paths.hot_index_map_path must be "
                "configured together"
            )
        if hot_index_path:
            command.extend(
                [
                    "--rag-hot-index-path",
                    str(hot_index_path),
                    "--rag-hot-index-map-path",
                    str(hot_index_map_path),
                    "--rag-hot-index-shard-id",
                    str(self.runtime.get("hot_index_shard_id", 0)),
                ]
            )
        command.extend(
            [
                "--rag-shared-hbm-mode",
                shared_hbm_mode,
                "--rag-shared-hbm-pool-gib",
                str(self.runtime.get("shared_hbm_pool_gib", 10)),
                "--rag-shared-hbm-chunk-mib",
                str(self.runtime.get("shared_hbm_chunk_mib", 8)),
                "--rag-shared-hbm-lookahead-ms",
                str(self.runtime.get("shared_hbm_lookahead_ms", 50)),
                "--rag-shared-hbm-min-residency-ms",
                str(
                    self.runtime.get(
                        "shared_hbm_min_residency_ms", 10
                    )
                ),
                "--rag-shared-hbm-cluster-heat-decay",
                str(
                    self.runtime.get(
                        "shared_hbm_cluster_heat_decay", 0.98
                    )
                ),
                "--rag-shared-hbm-cluster-miss-boost",
                str(
                    self.runtime.get(
                        "shared_hbm_cluster_miss_boost", 4.0
                    )
                ),
                "--rag-shared-hbm-cluster-promotion-hysteresis",
                str(
                    self.runtime.get(
                        "shared_hbm_cluster_promotion_hysteresis", 1.15
                    )
                ),
                "--rag-shared-hbm-cluster-reload-min-misses",
                str(
                    self.runtime.get("shared_hbm_cluster_reload_min_misses", 0)
                ),
                "--rag-shared-hbm-cluster-reload-max-lists",
                str(
                    self.runtime.get("shared_hbm_cluster_reload_max_lists", 0)
                ),
                "--rag-shared-hbm-kv-reserve-extents",
                str(
                    self.runtime.get(
                        "shared_hbm_kv_reserve_extents", 2
                    )
                ),
                "--rag-shared-hbm-rerank-prefetch-batches",
                str(
                    self.runtime.get(
                        "shared_hbm_rerank_prefetch_batches", 1
                    )
                ),
                "--rag-shared-hbm-map-stage-extents",
                str(
                    self.runtime.get(
                        "shared_hbm_map_stage_extents", -1
                    )
                ),
            ]
        )
        if self.runtime.get("shared_hbm_async_map", False):
            command.append("--rag-shared-hbm-async-map")
        doc_db_path = self.paths.get("doc_db_path")
        doc_store_path = self.paths.get("doc_store_path")
        if doc_db_path:
            command.extend(["--rag-doc-db-path", str(doc_db_path)])
        elif doc_store_path:
            command.extend(["--rag-doc-store-path", str(doc_store_path)])
        elif self.runtime.get("doc_random_fallback", False):
            command.extend(
                [
                    "--rag-doc-random-fallback",
                    "--rag-doc-random-input-len",
                    str(self.workload.get("document_input_len", 1023)),
                ]
            )
        else:
            raise ValueError(
                "Configure paths.doc_db_path, paths.doc_store_path, or set "
                "runtime.doc_random_fallback=true"
            )
        if case.is_cpu_retrieval:
            cpu_workers = int(
                cpu_retrieval_runtime_value(
                    self.runtime, case.policy, "cpu_retrieval_workers", 1
                )
            )
            cpu_threads = int(
                cpu_retrieval_runtime_value(
                    self.runtime, case.policy, "cpu_retrieval_threads", 8
                )
            )
            cpu_batch_queries = int(
                cpu_retrieval_runtime_value(
                    self.runtime,
                    case.policy,
                    "cpu_retrieval_max_batch_queries",
                    1,
                )
            )
            command.extend(
                [
                    "--rag-cpu-retrieval",
                    "--rag-cpu-retrieval-workers",
                    str(cpu_workers),
                    "--rag-cpu-retrieval-threads",
                    str(cpu_threads),
                    "--rag-cpu-retrieval-max-batch-queries",
                    str(cpu_batch_queries),
                ]
            )
        elif not case.is_llm_only:
            if case.uses_hybrid_index:
                command.extend(
                    [
                        "--rag-cpu-retrieval-threads",
                        str(
                            hybrid_cpu_search_threads(self.runtime, case.policy)
                        ),
                        "--rag-cpu-retrieval-max-batch-queries",
                        str(
                            hybrid_cpu_search_max_batch_queries(
                                self.runtime, case.policy
                            )
                        ),
                    ]
                )
            if case.uses_hybrid_index and "ann_cold_max_batch_queries" in self.runtime:
                command.extend([
                    "--rag-ann-cold-max-batch-queries",
                    str(int(self.runtime["ann_cold_max_batch_queries"])),
                ])
            command.append("--enable-rag-ann")
        return command

    def build_benchmark_command(self, case: ProfileCase) -> List[str]:
        benchmark = self.project_root / "benchmark" / "pdmux" / "bench_rag_serving.py"
        selection = self.selection
        command = [
            self.python,
            str(benchmark),
            "--tag",
            case.key,
            "--base-url",
            self.base_url,
            "--num-requests",
            str(self.workload.get("num_requests", 200)),
            "--warmup-requests",
            str(self.workload.get("warmup_requests", 16)),
            "--warmup-output-len",
            str(self.workload.get("warmup_output_len", 32)),
            "--ramp-up-requests",
            str(self.workload.get("ramp_up_requests", 50)),
            "--request-rate",
            str(case.request_rate),
            "--deterministic-arrivals",
            "--output-len",
            str(self.workload.get("output_len", 256)),
            "--topk",
            str(self.runtime["topk"]),
            "--seed",
            str(case.seed),
            "--workload-trace",
            str(self.workload_trace_path(case)),
            "--ttft-slo-ms",
            str(selection["ttft_slo_ms"]),
            "--itl-slo-ms",
            str(selection["itl_slo_ms"]),
            "--e2e-slo-ms",
            str(selection.get("e2e_slo_ms", -1)),
            "--stationarity-windows",
            str(selection.get("stationarity_windows", 8)),
            "--stationarity-min-achieved-offered-ratio",
            str(
                selection.get(
                    "stationarity_min_achieved_offered_ratio", 0.95
                )
            ),
            "--stationarity-max-drain-ratio",
            str(selection.get("stationarity_max_drain_ratio", 0.10)),
            "--stationarity-max-backlog-growth-fraction",
            str(
                selection.get(
                    "stationarity_max_backlog_growth_fraction", 0.05
                )
            ),
            "--timeout-s",
            str(self.runtime.get("benchmark_request_timeout_s", 300)),
            "--output-file",
            str(self.result_path(case)),
        ]
        if self.runtime.get("benchmark_disable_http_keepalive", False):
            command.append("--disable-http-keepalive")
        prime_batches = int(
            self.runtime.get("ann_prime_batches_per_size", 0)
        )
        if prime_batches > 0 and not (
            case.is_cpu_retrieval or case.is_ann_only or case.is_llm_only
        ):
            batch_sizes = self.runtime.get(
                "ann_prime_batch_sizes", [1, 2]
            )
            if not isinstance(batch_sizes, list):
                raise ValueError("runtime.ann_prime_batch_sizes must be a list")
            validate_batch_sizes = self.runtime.get(
                "ann_prime_validate_batch_sizes", [1, 2]
            )
            if not isinstance(validate_batch_sizes, list):
                raise ValueError(
                    "runtime.ann_prime_validate_batch_sizes must be a list"
                )
            command.extend(
                [
                    "--ann-prime-query-vec-path",
                    str(require(self.config, "paths", "query_vec_path")),
                    "--ann-prime-batch-sizes",
                    ",".join(str(int(value)) for value in batch_sizes),
                    "--ann-prime-validate-batch-sizes",
                    ",".join(
                        str(int(value)) for value in validate_batch_sizes
                    ),
                    "--ann-prime-batches-per-size",
                    str(prime_batches),
                    "--ann-prime-min-samples",
                    str(
                        int(
                            self.runtime.get(
                                "ann_prime_min_samples", 4
                            )
                        )
                    ),
                ]
            )
            if self.runtime.get(
                "ann_prime_require_credit_straddle", False
            ):
                command.append("--ann-prime-require-credit-straddle")
            load_ramp_requests = int(
                self.runtime.get("ann_prime_load_ramp_requests", 0)
            )
            if load_ramp_requests > 0:
                command.extend(
                    [
                        "--ann-prime-load-ramp-requests",
                        str(load_ramp_requests),
                        "--ann-prime-load-ramp-rps",
                        str(
                            float(
                                self.runtime.get(
                                    "ann_prime_load_ramp_rps",
                                    case.request_rate,
                                )
                            )
                        ),
                    ]
                )
        if case.is_llm_only:
            command.append("--skip-retrieval")
        elif case.is_ann_only:
            command.append("--skip-generation")
        return command

    def wait_for_server(self, case: ProfileCase) -> Dict[str, Any]:
        timeout = float(self.runtime.get("startup_timeout_s", 300))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server_process is None or self.server_process.poll() is not None:
                raise RuntimeError("SGLang exited before becoming ready")
            info = self.server_info(timeout=2.0)
            if info is not None:
                self.validate_server_info(case, info, self.runtime, self.paths)
                return info
            time.sleep(1)
        raise TimeoutError(f"SGLang did not become ready within {timeout:g}s")

    @staticmethod
    def validate_server_info(
        case: ProfileCase,
        info: Dict[str, Any],
        expected_runtime: Optional[Dict[str, Any]] = None,
        expected_paths: Optional[Dict[str, Any]] = None,
    ) -> None:
        actual_schedule = info.get("rag_ann_schedule")
        actual_policy_mode = info.get("rag_policy_mode", "full")
        actual_slo = float(info.get("rag_ann_decode_step_slo_ms", -1))
        actual_defer = float(info.get("rag_ann_max_defer_ms", -1))
        if actual_schedule != case.schedule:
            raise RuntimeError(
                f"Schedule mismatch: expected={case.schedule}, actual={actual_schedule}"
            )
        if actual_policy_mode != case.policy_mode:
            raise RuntimeError(
                f"Policy mismatch: expected={case.policy_mode}, "
                f"actual={actual_policy_mode}"
            )
        if not math.isclose(actual_slo, case.decode_slo_ms):
            raise RuntimeError(
                f"Decode SLO mismatch: expected={case.decode_slo_ms}, "
                f"actual={actual_slo}"
            )
        if not math.isclose(actual_defer, case.max_defer_ms):
            raise RuntimeError(
                f"Max-defer mismatch: expected={case.max_defer_ms}, "
                f"actual={actual_defer}"
            )
        mem_fraction_policy_key = (
            f"{case.policy.replace('-', '_')}_mem_fraction_static"
        )
        if expected_runtime is not None and (
            "mem_fraction_static" in expected_runtime
            or mem_fraction_policy_key in expected_runtime
        ):
            expected_mem_fraction = effective_mem_fraction_static(
                expected_runtime, case.policy
            )
            actual_mem_fraction = float(info.get("mem_fraction_static", -1))
            if not math.isclose(actual_mem_fraction, expected_mem_fraction):
                raise RuntimeError(
                    "mem_fraction_static mismatch: "
                    f"expected={expected_mem_fraction}, "
                    f"actual={actual_mem_fraction}"
                )
        expected_launch_site = (
            "early"
            if not case.uses_pdmux
            else str((expected_runtime or {}).get("ann_launch_site", "after-prefill"))
        )
        actual_launch_site = info.get("rag_ann_launch_site")
        if actual_launch_site != expected_launch_site:
            raise RuntimeError(
                f"ANN launch-site mismatch: expected={expected_launch_site}, "
                f"actual={actual_launch_site}"
            )
        expected_arbitration = (
            "overlap"
            if not case.uses_pdmux
            else str(
                (expected_runtime or {}).get("frontend_arbitration", "fair")
            )
        )
        actual_arbitration = info.get("rag_ann_frontend_arbitration")
        if actual_arbitration != expected_arbitration:
            raise RuntimeError(
                f"Arbitration mismatch: expected={expected_arbitration}, "
                f"actual={actual_arbitration}"
            )
        for field, expected in (
            ("rag_ann_lease_frontend_sm", case.ann_lease_sm),
            ("rag_prefill_lease_frontend_sm", case.prefill_lease_sm),
        ):
            actual = int(info.get(field, -999))
            if actual != expected:
                raise RuntimeError(
                    f"{field} mismatch: expected={expected}, actual={actual}"
                )
        if case.prefill_max_consecutive_chunks is not None:
            actual = int(info.get("rag_prefill_max_consecutive_chunks", -999))
            if actual != case.prefill_max_consecutive_chunks:
                raise RuntimeError(
                    "rag_prefill_max_consecutive_chunks mismatch: "
                    f"expected={case.prefill_max_consecutive_chunks}, "
                    f"actual={actual}"
                )
        for field, expected in (
            ("rag_prefill_lease_min_layers", case.prefill_lease_min_layers),
            ("rag_prefill_lease_max_layers", case.prefill_lease_max_layers),
        ):
            if expected is None:
                continue
            actual = int(info.get(field, -999))
            if actual != expected:
                raise RuntimeError(
                    f"{field} mismatch: expected={expected}, actual={actual}"
                )
        if not case.is_raglease and not math.isclose(
            float(info.get("rag_ann_handoff_timeout_ms", -1)), 0.0
        ):
            raise RuntimeError(
                f"{case.policy} unexpectedly enabled ANN-to-Prefill handoff"
            )
        actual_pdmux = bool(info.get("enable_pdmux", False))
        if actual_pdmux != case.uses_pdmux:
            raise RuntimeError(
                f"PD-Mux state mismatch: expected={case.uses_pdmux}, "
                f"actual={actual_pdmux}"
            )
        if case.uses_pdmux:
            actual_config = str(info.get("pdmux_config_path", ""))
            if Path(actual_config).resolve() != Path(
                case.pdmux_config_path
            ).resolve():
                raise RuntimeError(
                    f"PD-Mux config mismatch: expected={case.pdmux_config_path}, "
                    f"actual={actual_config}"
                )
        if case.uses_hybrid_index:
            if not bool(info.get("enable_rag_ann", False)) or bool(
                info.get("rag_cpu_retrieval", False)
            ):
                raise RuntimeError(
                    "Hybrid retrieval must use the GPU ANN executor with its "
                    "internal asynchronous CPU cold path"
                )
            actual_shared_hbm_mode = str(
                info.get("rag_shared_hbm_mode", "off")
            )
            expected_shared_hbm_mode = effective_shared_hbm_mode(
                expected_runtime or {}, case
            )
            if actual_shared_hbm_mode != expected_shared_hbm_mode:
                raise RuntimeError(
                    "Hybrid shared-HBM mode mismatch: "
                    f"expected={expected_shared_hbm_mode}, "
                    f"actual={actual_shared_hbm_mode}"
                )
            if not info.get("rag_hot_index_path") or not info.get(
                "rag_hot_index_map_path"
            ):
                raise RuntimeError(
                    "Hybrid retrieval is missing its GPU hot index or mapping"
                )
            if expected_runtime is not None:
                expected_threads = hybrid_cpu_search_threads(
                    expected_runtime, case.policy
                )
                actual_threads = int(
                    info.get("rag_cpu_retrieval_threads", -1)
                )
                if actual_threads != expected_threads:
                    raise RuntimeError(
                        "Hybrid CPU-search thread mismatch: "
                        f"expected={expected_threads}, actual={actual_threads}"
                    )
            if expected_paths is not None:
                expected_cpu, expected_hot, expected_map = (
                    resolve_hybrid_index_paths(expected_paths, case.policy)
                )
                for field, expected in (
                    ("rag_index_path", expected_cpu),
                    ("rag_hot_index_path", expected_hot),
                    ("rag_hot_index_map_path", expected_map),
                ):
                    actual = str(info.get(field, ""))
                    if Path(actual).resolve() != Path(expected).resolve():
                        raise RuntimeError(
                            f"{field} mismatch: expected={expected}, actual={actual}"
                        )
        if expected_runtime is not None and "disable_radix_cache" in expected_runtime:
            expected_disable_radix_cache = bool(
                expected_runtime["disable_radix_cache"]
            )
            actual_disable_radix_cache = bool(
                info.get("disable_radix_cache", False)
            )
            if actual_disable_radix_cache != expected_disable_radix_cache:
                raise RuntimeError(
                    "disable_radix_cache mismatch: "
                    f"expected={expected_disable_radix_cache}, "
                    f"actual={actual_disable_radix_cache}"
                )
        if case.is_llm_only:
            if bool(info.get("enable_rag_ann", False)) or bool(
                info.get("rag_cpu_retrieval", False)
            ):
                raise RuntimeError(
                    "LLM-only control unexpectedly enabled retrieval"
                )
            if expected_runtime is not None:
                expected_mem_fraction = effective_mem_fraction_static(
                    expected_runtime, case.policy
                )
                actual_mem_fraction = float(
                    info.get("mem_fraction_static", -1)
                )
                if not math.isclose(
                    actual_mem_fraction, expected_mem_fraction
                ):
                    raise RuntimeError(
                        "mem_fraction_static mismatch: "
                        f"expected={expected_mem_fraction}, "
                        f"actual={actual_mem_fraction}"
                    )
            return
        if case.is_ann_only:
            if not bool(info.get("enable_rag_ann", False)):
                raise RuntimeError("ANN-only control did not enable GPU ANN")
            if bool(info.get("rag_cpu_retrieval", False)):
                raise RuntimeError(
                    "ANN-only control unexpectedly enabled CPU retrieval"
                )
        if case.is_cpu_retrieval:
            if bool(info.get("enable_rag_ann", False)):
                raise RuntimeError(
                    "CPU retrieval unexpectedly enabled the GPU ANN executor"
                )
            if not bool(info.get("rag_cpu_retrieval", False)):
                raise RuntimeError("CPU retrieval backend is not enabled")
            if expected_runtime is not None:
                for field, expected in (
                    (
                        "rag_cpu_retrieval_workers",
                        int(
                            cpu_retrieval_runtime_value(
                                expected_runtime,
                                case.policy,
                                "cpu_retrieval_workers",
                                1,
                            )
                        ),
                    ),
                    (
                        "rag_cpu_retrieval_threads",
                        int(
                            cpu_retrieval_runtime_value(
                                expected_runtime,
                                case.policy,
                                "cpu_retrieval_threads",
                                8,
                            )
                        ),
                    ),
                    (
                        "rag_cpu_retrieval_max_batch_queries",
                        int(
                            cpu_retrieval_runtime_value(
                                expected_runtime,
                                case.policy,
                                "cpu_retrieval_max_batch_queries",
                                1,
                            )
                        ),
                    ),
                    ("rag_index_nprobe", int(expected_runtime["nprobe"])),
                    ("rag_index_topk", int(expected_runtime["topk"])),
                    (
                        "rag_doc_token_cache_size",
                        int(expected_runtime.get("doc_token_cache_size", 1024)),
                    ),
                ):
                    actual = int(info.get(field, -999))
                    if actual != expected:
                        raise RuntimeError(
                            f"{field} mismatch: expected={expected}, actual={actual}"
                        )
                expected_mem_fraction = effective_mem_fraction_static(
                    expected_runtime, case.policy
                )
                actual_mem_fraction = float(
                    info.get("mem_fraction_static", -1)
                )
                if not math.isclose(
                    actual_mem_fraction, expected_mem_fraction
                ):
                    raise RuntimeError(
                        "mem_fraction_static mismatch: "
                        f"expected={expected_mem_fraction}, "
                        f"actual={actual_mem_fraction}"
                    )
            return
        if expected_runtime is None:
            return

        if (
            case.policy in UNCOORDINATED_BASELINE_POLICIES
            or case.uses_hybrid_index
        ):
            expected_runtime = dict(expected_runtime)
            for key, default in UNCOORDINATED_ANN_DEFAULTS.items():
                expected_runtime[key] = effective_ann_runtime_value(
                    expected_runtime, case.policy, key, default
                )
        runtime_fields = (
            (
                "rag_ann_fast_path_pressure_threshold",
                "ann_fast_path_pressure_threshold",
                0.5,
                float,
            ),
            (
                "rag_ann_fast_path_max_work",
                "ann_fast_path_max_work",
                1,
                int,
            ),
            (
                "rag_ann_fast_path_decode_batch_p95_threshold",
                "ann_fast_path_decode_batch_p95_threshold",
                16,
                float,
            ),
            (
                "rag_prefill_low_load_batch_p95_threshold",
                "prefill_low_load_batch_p95_threshold",
                4,
                float,
            ),
            (
                "rag_prefill_low_load_arrival_rps_threshold",
                "prefill_low_load_arrival_rps_threshold",
                -1,
                float,
            ),
            (
                "rag_prefill_low_load_reference_tokens",
                "prefill_low_load_reference_tokens",
                1024,
                int,
            ),
            (
                "rag_ann_fast_path_decode_queue_p95_ms",
                "ann_fast_path_decode_queue_p95_ms",
                2,
                float,
            ),
            (
                "rag_ann_decode_batch_bucket_size",
                "ann_decode_batch_bucket_size",
                4,
                int,
            ),
            (
                "rag_ann_urgency_start_ratio",
                "ann_urgency_start_ratio",
                1,
                float,
            ),
            (
                "rag_ann_max_decode_slowdown_ratio",
                "ann_max_decode_slowdown_ratio",
                1,
                float,
            ),
            ("rag_ann_max_inflight", "ann_max_inflight", 8, int),
            (
                "rag_ann_max_batches_per_tick",
                "ann_max_batches_per_tick",
                0,
                int,
            ),
            (
                "rag_ann_max_inflight_during_decode",
                "ann_max_inflight_during_decode",
                0,
                int,
            ),
            ("rag_ann_max_batch_size", "ann_max_batch_size", 8, int),
            (
                "rag_ann_max_queued_gpu_ms",
                "ann_max_queued_gpu_ms",
                30,
                float,
            ),
            (
                "rag_ann_max_microbatch_gpu_ms",
                "ann_max_microbatch_gpu_ms",
                20,
                float,
            ),
            (
                "rag_ann_service_time_percentile",
                "ann_service_time_percentile",
                95,
                float,
            ),
            (
                "rag_ann_service_time_min_samples",
                "ann_service_time_min_samples",
                4,
                int,
            ),
            (
                "rag_prefill_decode_guard_percentile",
                "prefill_decode_guard_percentile",
                99,
                float,
            ),
            ("rag_ann_max_batch_wait_ms", "ann_max_batch_wait_ms", 2, float),
            ("rag_ann_decode_only_wait_ms", "ann_decode_only_wait_ms", 10, float),
            ("rag_partition_hysteresis_bs", "partition_hysteresis_bs", 4, int),
            (
                "rag_partition_min_residency_ms",
                "partition_min_residency_ms",
                2,
                float,
            ),
            (
                "rag_partition_safe_mixed_hold_ratio",
                "partition_safe_mixed_hold_ratio",
                0.9,
                float,
            ),
            (
                "rag_partition_safe_mixed_min_samples",
                "partition_safe_mixed_min_samples",
                8,
                int,
            ),
            (
                "rag_partition_safe_mixed_max_hold_ms",
                "partition_safe_mixed_max_hold_ms",
                5,
                float,
            ),
            (
                "rag_lease_coalesce_min_work",
                "lease_coalesce_min_work",
                1,
                int,
            ),
            (
                "rag_lease_coalesce_max_wait_ms",
                "lease_coalesce_max_wait_ms",
                0,
                float,
            ),
            (
                "rag_lease_switch_cost_ms",
                "lease_switch_cost_ms",
                0,
                float,
            ),
            (
                "rag_lease_switch_cost_percentile",
                "lease_switch_cost_percentile",
                95,
                float,
            ),
            ("rag_e2e_slo_ms", "e2e_slo_ms", -1, float),
            (
                "rag_lease_min_gain_ratio",
                "lease_min_gain_ratio",
                1,
                float,
            ),
            (
                "rag_prefill_lease_min_layers",
                "prefill_lease_min_layers",
                1,
                int,
            ),
            (
                "rag_prefill_lease_max_layers",
                "prefill_lease_max_layers",
                8,
                int,
            ),
            (
                "rag_prefill_lease_decode_ratio",
                "prefill_lease_decode_ratio",
                1,
                float,
            ),
            (
                "rag_prefill_max_consecutive_chunks",
                "max_consecutive_prefill_chunks",
                2,
                int,
            ),
            (
                "rag_lease_ann_service_estimate_ms",
                "lease_ann_service_estimate_ms",
                10,
                float,
            ),
            (
                "rag_lease_prefill_service_estimate_ms",
                "lease_prefill_service_estimate_ms",
                10,
                float,
            ),
        )
        dynamic_only_zero_fields = {
            "rag_partition_hysteresis_bs",
            "rag_partition_min_residency_ms",
        }
        case_runtime_overrides = {
            "rag_ann_max_inflight": case.ann_max_inflight,
            "rag_prefill_lease_min_layers": case.prefill_lease_min_layers,
            "rag_prefill_lease_max_layers": case.prefill_lease_max_layers,
            "rag_prefill_max_consecutive_chunks": (
                case.prefill_max_consecutive_chunks
            ),
        }
        for server_field, runtime_field, default, converter in runtime_fields:
            case_override = case_runtime_overrides.get(server_field)
            expected = converter(
                expected_runtime.get(runtime_field, default)
                if case_override is None
                else case_override
            )
            if (
                case.policy_mode == "dynamic-only"
                and server_field in dynamic_only_zero_fields
            ):
                expected = converter(0)
            actual = converter(info.get(server_field, -999))
            matches = (
                math.isclose(actual, expected)
                if converter is float
                else actual == expected
            )
            if not matches:
                raise RuntimeError(
                    f"{server_field} mismatch: expected={expected}, actual={actual}"
                )
        expected_safe_mixed_hold = bool(
            case.is_raglease
            and expected_runtime.get("partition_safe_mixed_hold", False)
        )
        actual_safe_mixed_hold = bool(
            info.get("rag_partition_safe_mixed_hold", False)
        )
        if actual_safe_mixed_hold != expected_safe_mixed_hold:
            raise RuntimeError(
                "rag_partition_safe_mixed_hold mismatch: "
                f"expected={expected_safe_mixed_hold}, "
                f"actual={actual_safe_mixed_hold}"
            )
        expected_work_conserving_decode_overlap = bool(
            case.is_raglease
            and case.schedule == "decode-gc"
            and expected_runtime.get(
                "ann_work_conserving_decode_overlap", True
            )
        )
        actual_work_conserving_decode_overlap = bool(
            info.get("rag_ann_work_conserving_decode_overlap", False)
        )
        if (
            actual_work_conserving_decode_overlap
            != expected_work_conserving_decode_overlap
        ):
            raise RuntimeError(
                "rag_ann_work_conserving_decode_overlap mismatch: "
                f"expected={expected_work_conserving_decode_overlap}, "
                f"actual={actual_work_conserving_decode_overlap}"
            )
        expected_round_trip_cost = bool(
            case.policy not in UNCOORDINATED_BASELINE_POLICIES
            and expected_runtime.get("lease_round_trip_switch_cost", True)
        )
        actual_round_trip_cost = bool(
            info.get("rag_lease_round_trip_switch_cost", False)
        )
        if actual_round_trip_cost != expected_round_trip_cost:
            raise RuntimeError(
                "rag_lease_round_trip_switch_cost mismatch: "
                f"expected={expected_round_trip_cost}, "
                f"actual={actual_round_trip_cost}"
            )
        expected_dynamic_prefill = bool(
            case.is_raglease
            and expected_runtime.get("prefill_dynamic_lease", True)
        )
        actual_dynamic_prefill = bool(
            info.get("rag_prefill_dynamic_lease", False)
        )
        if actual_dynamic_prefill != expected_dynamic_prefill:
            raise RuntimeError(
                "rag_prefill_dynamic_lease mismatch: "
                f"expected={expected_dynamic_prefill}, "
                f"actual={actual_dynamic_prefill}"
            )
        expected_tail_aware_prefill = bool(
            case.is_raglease
            and expected_runtime.get("prefill_tail_aware_admission", True)
        )
        actual_tail_aware_prefill = bool(
            info.get("rag_prefill_tail_aware_admission", False)
        )
        if actual_tail_aware_prefill != expected_tail_aware_prefill:
            raise RuntimeError(
                "rag_prefill_tail_aware_admission mismatch: "
                f"expected={expected_tail_aware_prefill}, "
                f"actual={actual_tail_aware_prefill}"
            )
        expected_exclusive_prefill = bool(
            case.is_raglease
            and expected_runtime.get("prefill_exclusive_sprint", False)
        )
        actual_exclusive_prefill = bool(
            info.get("rag_prefill_exclusive_sprint", False)
        )
        if actual_exclusive_prefill != expected_exclusive_prefill:
            raise RuntimeError(
                "rag_prefill_exclusive_sprint mismatch: "
                f"expected={expected_exclusive_prefill}, "
                f"actual={actual_exclusive_prefill}"
            )
        expected_pre_switch_launch = bool(
            case.is_raglease
            and case.schedule == "decode-gc"
            and expected_runtime.get("ann_pre_switch_launch", False)
        )
        actual_pre_switch_launch = bool(
            info.get("rag_ann_pre_switch_launch", False)
        )
        if actual_pre_switch_launch != expected_pre_switch_launch:
            raise RuntimeError(
                "rag_ann_pre_switch_launch mismatch: "
                f"expected={expected_pre_switch_launch}, "
                f"actual={actual_pre_switch_launch}"
            )
        expected_partition_slos = (
            [
                float(value)
                for value in expected_runtime.get(
                    "partition_decode_slo_ms", []
                )
            ]
            if case.is_raglease
            else []
        )
        actual_partition_slos = [
            float(value)
            for value in (info.get("rag_partition_decode_slo_ms") or [])
        ]
        if len(actual_partition_slos) != len(expected_partition_slos) or any(
            not math.isclose(actual, expected)
            for actual, expected in zip(
                actual_partition_slos, expected_partition_slos
            )
        ):
            raise RuntimeError(
                "rag_partition_decode_slo_ms mismatch: "
                f"expected={expected_partition_slos}, "
                f"actual={actual_partition_slos}"
            )
        expected_batch_thresholds = (
            [
                float(value)
                for value in expected_runtime.get(
                    "partition_decode_slo_batch_thresholds", []
                )
            ]
            if case.is_raglease
            else []
        )
        actual_batch_thresholds = [
            float(value)
            for value in (
                info.get("rag_partition_decode_slo_batch_thresholds") or []
            )
        ]
        if len(actual_batch_thresholds) != len(expected_batch_thresholds) or any(
            not math.isclose(actual, expected)
            for actual, expected in zip(
                actual_batch_thresholds, expected_batch_thresholds
            )
        ):
            raise RuntimeError(
                "rag_partition_decode_slo_batch_thresholds mismatch: "
                f"expected={expected_batch_thresholds}, "
                f"actual={actual_batch_thresholds}"
            )
        expected_controller_fields = {
            "rag_partition_decode_slo_hysteresis_bs": float(
                expected_runtime.get(
                    "partition_decode_slo_hysteresis_bs", 0
                )
                if expected_batch_thresholds
                else 0
            ),
            "rag_partition_decode_slo_min_residency_ms": float(
                expected_runtime.get(
                    "partition_decode_slo_min_residency_ms", 0
                )
                if expected_batch_thresholds
                else 0
            ),
            "rag_partition_decode_slo_initial_calibration_ms": float(
                expected_runtime.get(
                    "partition_decode_slo_initial_calibration_ms", 0
                )
                if expected_batch_thresholds
                else 0
            ),
            "rag_partition_decode_slo_ewma_alpha": float(
                expected_runtime.get("partition_decode_slo_ewma_alpha", 0.02)
                if expected_batch_thresholds
                else 0.02
            ),
        }
        for field, expected in expected_controller_fields.items():
            actual = float(info.get(field, -999))
            if not math.isclose(actual, expected):
                raise RuntimeError(
                    f"{field} mismatch: expected={expected}, actual={actual}"
                )
        expected_floor_enabled = bool(
            case.is_raglease
            and expected_runtime.get("partition_slo_floor_enabled", True)
        )
        actual_floor_enabled = bool(
            info.get("rag_partition_slo_floor_enabled", False)
        )
        if actual_floor_enabled != expected_floor_enabled:
            raise RuntimeError(
                "rag_partition_slo_floor_enabled mismatch: "
                f"expected={expected_floor_enabled}, "
                f"actual={actual_floor_enabled}"
            )
        expected_epoch = float(
            expected_runtime.get("ann_stream_epoch_ms", 0)
            if case.stream_epoch_ms is None
            else case.stream_epoch_ms
        )
        if case.policy_mode == "dynamic-only":
            expected_epoch = 0.0
        actual_epoch = float(info.get("rag_ann_stream_epoch_ms", -999))
        if not math.isclose(actual_epoch, expected_epoch):
            raise RuntimeError(
                "rag_ann_stream_epoch_ms mismatch: "
                f"expected={expected_epoch}, actual={actual_epoch}"
            )

    def stop_server(self) -> None:
        process = self.server_process
        self.server_process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and self.gpu_pids():
            time.sleep(1)
        remaining = self.gpu_pids()
        if remaining:
            raise RuntimeError(f"Residual GPU worker PIDs after shutdown: {remaining}")
        time.sleep(float(self.runtime.get("cooldown_s", 2)))

    @staticmethod
    def tail(path: Path, lines: int = 80) -> str:
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    def validate_result(self, case: ProfileCase, result: Dict[str, Any]) -> None:
        server = result["server"]
        summary = result["summary"]
        # The per-result server snapshot omits some static path fields. Full
        # path validation already runs against /get_server_info at startup.
        self.validate_server_info(case, server, self.runtime)
        trace = load_workload(self.workload_trace_path(case))
        expected = len(trace.measured)
        actual_sha256 = nested_get(result, "workload", "sha256", default="")
        if actual_sha256 != trace.sha256():
            raise RuntimeError(
                f"Workload fingerprint mismatch: expected={trace.sha256()}, "
                f"actual={actual_sha256}"
            )
        if summary.get("completed") != expected or summary.get("failed") != 0:
            raise RuntimeError(
                f"Incomplete result: completed={summary.get('completed')}, "
                f"failed={summary.get('failed')}"
            )
        max_cache_hit = float(self.selection.get("max_fresh_cache_hit_rate", 0.5))
        if float(summary.get("token_cache_hit_rate", 0)) >= max_cache_hit:
            raise RuntimeError("Token cache suggests that a server was reused")
        if case.is_llm_only:
            successful = [
                request
                for request in result.get("requests", [])
                if request.get("success")
            ]
            if len(successful) != expected:
                raise RuntimeError(
                    "LLM-only result is missing successful request records"
                )
            if server.get("rag_ann_runtime"):
                raise RuntimeError(
                    "LLM-only result unexpectedly contains ANN runtime state"
                )
            ann_fields = (
                "ann_latency_ms",
                "ann_queue_wait_ms",
                "ann_gpu_latency_ms",
                "ann_cpu_queue_wait_ms",
                "ann_cpu_search_latency_ms",
                "ann_coarse_quantize_latency_ms",
                "ann_route_latency_ms",
                "ann_merge_latency_ms",
                "ann_hot_probe_ratio",
                "ann_hot_probe_count",
                "ann_cold_probe_count",
                "ann_batch_size",
            )
            for request in successful:
                if request.get("ann_index_mode") != "none":
                    raise RuntimeError(
                        "LLM-only result contains an ANN index mode"
                    )
                if request.get("retrieved_ids"):
                    raise RuntimeError(
                        "LLM-only result unexpectedly contains retrieved IDs"
                    )
                if any(
                    abs(float(request.get(field, 0))) > 1e-6
                    for field in ann_fields
                ):
                    raise RuntimeError(
                        "LLM-only result reports non-zero ANN work"
                    )
            return
        if case.is_ann_only:
            successful = [
                request
                for request in result.get("requests", [])
                if request.get("success")
            ]
            if len(successful) != expected:
                raise RuntimeError(
                    "ANN-only result is missing successful request records"
                )
            if not bool(result.get("config", {}).get("skip_generation")):
                raise RuntimeError(
                    "ANN-only result did not record skip_generation"
                )
            for request in successful:
                if request.get("ann_index_mode") != "full_gpu":
                    raise RuntimeError(
                        "ANN-only result contains a non-GPU ANN mode"
                    )
                if not request.get("retrieved_ids"):
                    raise RuntimeError(
                        "ANN-only result is missing retrieved IDs"
                    )
                if float(request.get("ann_latency_ms", 0)) <= 0 or float(
                    request.get("ann_gpu_latency_ms", 0)
                ) <= 0:
                    raise RuntimeError(
                        "ANN-only result reports missing ANN work"
                    )
                if int(request.get("output_tokens", 0)) != 0:
                    raise RuntimeError(
                        "ANN-only result unexpectedly generated tokens"
                    )
                if abs(float(request.get("ttft_ms", 0))) > 1e-6 or abs(
                    float(request.get("tpot_ms", 0))
                ) > 1e-6:
                    raise RuntimeError(
                        "ANN-only result reports non-zero LLM latency"
                    )
            return
        if case.is_cpu_retrieval:
            successful = [
                request
                for request in result.get("requests", [])
                if request.get("success")
            ]
            if len(successful) != expected:
                raise RuntimeError(
                    "CPU retrieval result is missing successful request records"
                )
            if any(
                request.get("ann_index_mode")
                not in {"cpu_ivfpq", "cpu_ivfpq_fastscan"}
                for request in successful
            ):
                raise RuntimeError(
                    "CPU retrieval result contains a non-CPU ANN index mode"
                )
            if any(
                abs(float(request.get("ann_gpu_latency_ms", 0))) > 1e-6
                for request in successful
            ):
                raise RuntimeError(
                    "CPU retrieval result reports non-zero GPU ANN latency"
                )
            if any(
                float(request.get("ann_cpu_search_latency_ms", 0)) <= 0
                for request in successful
            ):
                raise RuntimeError(
                    "CPU retrieval result is missing CPU ANN service latency"
                )
            for request in successful:
                accounted_ms = (
                    float(request.get("ann_queue_wait_ms", 0))
                    + float(request.get("ann_cpu_queue_wait_ms", 0))
                    + float(request.get("ann_cpu_search_latency_ms", 0))
                )
                if float(request.get("ann_latency_ms", 0)) + 1e-3 < accounted_ms:                    raise RuntimeError(
                        "CPU ANN latency is smaller than queue plus service time"
                    )
            return

        if case.uses_hybrid_index:
            successful = [
                request
                for request in result.get("requests", [])
                if request.get("success")
            ]
            if len(successful) != expected:
                raise RuntimeError(
                    f"{case.policy} result is missing successful request records"
                )
            allowed_index_modes = allowed_hybrid_result_index_modes(
                self.runtime, case
            )
            if any(
                request.get("ann_index_mode") not in allowed_index_modes
                for request in successful
            ):
                raise RuntimeError(
                    f"{case.policy} result did not use the hot/cold ANN path"
                )
            if not any(
                int(request.get("ann_hot_probe_count", 0)) > 0
                for request in successful
            ):
                raise RuntimeError(
                    f"{case.policy} result contains no GPU hot-index probes"
                )

        runtime_entries = server.get("rag_ann_runtime") or []
        if not runtime_entries:
            raise RuntimeError("GPU ANN result is missing rag_ann_runtime")
        runtime = runtime_entries[-1]
        ann_samples = int(nested_get(runtime, "ann_gpu_latency", "samples", default=0))
        total_sent = len(trace.requests) + trace.warmup_requests
        prime_batches = int(
            self.runtime.get("ann_prime_batches_per_size", 0)
        )
        if prime_batches > 0:
            priming = result.get("ann_priming")
            if not isinstance(priming, dict) or not priming.get("valid"):
                raise RuntimeError(
                    "Enabled ANN priming has no valid result report"
                )
            response_histogram = (
                priming.get("response_batch_histogram") or {}
            )
            total_sent += sum(
                int(value) for value in response_histogram.values()
            )
            load_ramp = priming.get("load_ramp") or {}
            total_sent += int(load_ramp.get("requests", 0))
        if ann_samples <= 0 or ann_samples > total_sent:
            raise RuntimeError(f"Implausible ANN sample count: {ann_samples}")

    def archive_case_artifacts(self, case: ProfileCase, attempt: int) -> Path:
        archive_root = self.output_dir / "failed_attempts" / case.name
        archive_dir = archive_root / f"attempt_{attempt}"
        generation = 1
        while archive_dir.exists():
            generation += 1
            archive_dir = archive_root / f"attempt_{attempt}_{generation}"
        archive_dir.mkdir(parents=True, exist_ok=False)
        for path in (
            self.result_path(case),
            self.ann_priming_path(case),
            self.server_info_path(case),
            self.server_log_path(case),
            self.benchmark_log_path(case),
        ):
            if path.exists():
                shutil.move(str(path), archive_dir / path.name)
        return archive_dir

    def run_case_with_retries(self, case: ProfileCase) -> Dict[str, Any]:
        max_attempts = int(self.runtime.get("case_max_attempts", 1))
        retry_delay_s = float(self.runtime.get("case_retry_delay_s", 5))
        if max_attempts < 1:
            raise ValueError("runtime.case_max_attempts must be positive")
        if retry_delay_s < 0:
            raise ValueError("runtime.case_retry_delay_s must be non-negative")

        for attempt in range(1, max_attempts + 1):
            try:
                return self.run_case(case)
            except Exception as error:
                if attempt == max_attempts:
                    raise
                archive_dir = self.archive_case_artifacts(case, attempt)
                print(
                    f"[retry {attempt}/{max_attempts}] {case.name}: {error}; "
                    f"archived to {archive_dir}",
                    file=sys.stderr,
                    flush=True,
                )
                if retry_delay_s > 0:
                    time.sleep(retry_delay_s)

        raise AssertionError("unreachable")

    def run_case(self, case: ProfileCase) -> Dict[str, Any]:
        result_path = self.result_path(case)
        if result_path.exists() and not self.overwrite:
            result = load_json(result_path)
            self.validate_result(case, result)
            print(f"[skip] {case.name}")
            return result

        self.assert_clean()
        server_log_path = self.server_log_path(case)
        benchmark_log_path = self.benchmark_log_path(case)
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        environment = os.environ.copy()
        cpu_affinity: Optional[List[int]] = None
        if (
            case.is_cpu_retrieval or case.uses_hybrid_index
        ) and hasattr(os, "sched_getaffinity"):
            cpu_affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
            if case.is_cpu_retrieval:
                required_cpus = int(
                    cpu_retrieval_runtime_value(
                        self.runtime, case.policy, "cpu_retrieval_workers", 1
                    )
                ) * int(
                    cpu_retrieval_runtime_value(
                        self.runtime, case.policy, "cpu_retrieval_threads", 8
                    )
                )
                requirement = f"workers*threads={required_cpus}"
            else:
                required_cpus = hybrid_cpu_search_threads(
                    self.runtime, case.policy
                )
                requirement = f"cold_search_threads={required_cpus}"
            if len(cpu_affinity) < required_cpus:
                raise RuntimeError(
                    f"{case.policy} CPU search is oversubscribed: "
                    f"{requirement}, "
                    f"allowed_logical_cpus={len(cpu_affinity)}"
                )
        environment["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu)
        python_path = str(self.project_root / "python")
        runtime_python_root = self.paths.get("runtime_python_root")
        faiss_egg_dir = self.paths.get("faiss_egg_dir")
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in [
                str(faiss_egg_dir) if faiss_egg_dir else "",
                str(runtime_python_root) if runtime_python_root else "",
                python_path,
                environment.get("PYTHONPATH", ""),
            ]
            if value
        )
        ld_preload = self.paths.get("ld_preload")
        if ld_preload:
            environment["LD_PRELOAD"] = str(ld_preload)
        if faiss_egg_dir:
            environment["FAISS_EGG_DIR"] = str(faiss_egg_dir)
            environment["FAISS_OPT_LEVEL"] = "AVX2"
        if case.is_cpu_retrieval:
            omp_threads = int(
                cpu_retrieval_runtime_value(
                    self.runtime,
                    case.policy,
                    "cpu_retrieval_threads",
                    8,
                )
            )
        elif case.uses_hybrid_index:
            omp_threads = hybrid_cpu_search_threads(
                self.runtime, case.policy
            )
        else:
            omp_threads = 0
        if omp_threads > 0:
            environment["OMP_NUM_THREADS"] = str(omp_threads)
            environment["OMP_WAIT_POLICY"] = "PASSIVE"

        print(f"[run] {case.name}")
        gpu_sampler: Optional[GpuTelemetrySampler] = None
        try:
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server_command = self.build_server_command(case)
                debugger = environment.get(
                    "RAGLEASE_SERVER_DEBUGGER", ""
                ).strip()
                if debugger:
                    server_command = [
                        *shlex.split(debugger),
                        "--args",
                        *server_command,
                    ]
                self.server_process = subprocess.Popen(
                    server_command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                info = self.wait_for_server(case)
                self.server_info_path(case).write_text(
                    json.dumps(info, indent=2), encoding="utf-8"
                )
                telemetry_interval_ms = float(
                    self.runtime.get("gpu_telemetry_interval_ms", 0)
                )
                if telemetry_interval_ms > 0:
                    gpu_sampler = GpuTelemetrySampler(
                        self.physical_gpu, telemetry_interval_ms
                    )
                    gpu_sampler.start()

                with benchmark_log_path.open("w", encoding="utf-8") as benchmark_log:
                    benchmark_wall_timeout = self.runtime.get(
                        "benchmark_wall_timeout_s"
                    )
                    benchmark = subprocess.run(
                        self.build_benchmark_command(case),
                        cwd=self.project_root,
                        env=environment,
                        stdout=benchmark_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        timeout=(
                            float(benchmark_wall_timeout)
                            if benchmark_wall_timeout is not None
                            else None
                        ),
                    )
                if benchmark.returncode != 0:
                    raise RuntimeError(
                        f"Benchmark failed with exit code {benchmark.returncode}\n"
                        f"{self.tail(benchmark_log_path)}"
                    )
                info_after = self.server_info(timeout=5.0)
                if info_after is None:
                    raise RuntimeError("SGLang server info unavailable after benchmark")
                self.server_info_after_path(case).write_text(
                    json.dumps(info_after, indent=2), encoding="utf-8"
                )
        except Exception as error:
            server_tail = self.tail(server_log_path)
            raise RuntimeError(f"{case.name} failed: {error}\n{server_tail}") from error
        finally:
            if gpu_sampler is not None:
                gpu_sampler.stop()
            self.stop_server()

        result = load_json(result_path)
        result_changed = False
        if case.is_cpu_retrieval:
            workers = int(
                cpu_retrieval_runtime_value(
                    self.runtime, case.policy, "cpu_retrieval_workers", 1
                )
            )
            threads = int(
                cpu_retrieval_runtime_value(
                    self.runtime, case.policy, "cpu_retrieval_threads", 8
                )
            )
            result["cpu_retrieval_host"] = {
                "workers": workers,
                "threads_per_search": threads,
                "max_batch_queries": int(
                    cpu_retrieval_runtime_value(
                        self.runtime,
                        case.policy,
                        "cpu_retrieval_max_batch_queries",
                        1,
                    )
                ),
                "execution_mode": "coalesced_microbatch",
                "required_logical_cpus": workers * threads,
                "allowed_logical_cpus": cpu_affinity,
            }
            result_changed = True
        if gpu_sampler is not None:
            summary = result["summary"]
            result["gpu_telemetry"] = gpu_sampler.summarize(
                float(summary["measurement_start_unix_s"]),
                float(summary["measurement_end_unix_s"]),
            )
            result_changed = True
        if result_changed:
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.validate_result(case, result)
        return result


def normalized_result_stationarity(result: Dict[str, Any]) -> Dict[str, Any]:
    """Correct legacy Poisson-trace results that used the nominal rate."""
    summary = result.get("summary", {})
    stationarity = dict(summary.get("stationarity") or {})
    arrival_process = str(
        result.get("workload", {}).get("arrival_process", "")
    ).strip().lower()
    if arrival_process in ("", "deterministic"):
        return stationarity

    observed_qps = float(stationarity.get("observed_trace_rate_qps", 0.0))
    if observed_qps <= 0.0:
        return stationarity
    achieved_qps = float(summary.get("request_throughput_qps", 0.0))
    achieved_ratio = achieved_qps / observed_qps
    failures = [
        failure
        for failure in stationarity.get("failures", [])
        if failure != "achieved_offered_ratio"
    ]
    min_ratio = float(
        (stationarity.get("thresholds") or {}).get(
            "min_achieved_offered_ratio", 0.95
        )
    )
    if achieved_ratio < min_ratio:
        failures.append("achieved_offered_ratio")
    stationarity.update(
        {
            "passed": not failures,
            "failures": failures,
            "achieved_offered_ratio": achieved_ratio,
            "rate_normalization": "observed_trace",
        }
    )
    return stationarity


def result_metrics(
    result: Dict[str, Any],
    e2e_slo_ms: Optional[float] = None,
) -> Dict[str, float]:
    summary = result["summary"]
    runtimes = result.get("server", {}).get("rag_ann_runtime") or []
    runtime = runtimes[-1] if runtimes else {}
    gpu_telemetry = result.get("gpu_telemetry", {})
    launches = runtime.get("ann_launch_reason_counts", {})
    switch_latency = runtime.get("partition_switch_latency", {})
    switch_benefit = runtime.get("lease_switch_benefit", {})
    coalescing = runtime.get("lease_coalescing", {})
    stationarity = normalized_result_stationarity(result)
    total_launches = max(1, sum(int(value) for value in launches.values()))

    configured_e2e_slo_ms = (
        float(e2e_slo_ms)
        if e2e_slo_ms is not None
        else float(summary.get("e2e_slo_ms", -1))
    )
    measured_requests = [
        item for item in result.get("requests", []) if item.get("success")
    ]
    if configured_e2e_slo_ms > 0 and measured_requests:
        compliant = [
            item
            for item in measured_requests
            if float(item.get("e2e_ms", 0)) <= configured_e2e_slo_ms
        ]
        e2e_slo_attainment = len(compliant) / len(measured_requests)
        e2e_slo_goodput = len(compliant) / float(summary["duration_s"])
    else:
        e2e_slo_attainment = float(
            summary.get("e2e_slo_attainment", summary["slo_attainment"])
        )
        e2e_slo_goodput = float(
            summary.get(
                "e2e_slo_compliant_qps", summary["slo_compliant_qps"]
            )
        )

    return {
        "completed": float(summary["completed"]),
        "failed": float(summary["failed"]),
        "request_qps": float(summary["request_throughput_qps"]),
        "stationarity_available": float(
            bool(stationarity.get("available", False))
        ),
        "stationarity_pass": float(bool(stationarity.get("passed", False))),
        "achieved_offered_ratio": float(
            stationarity.get("achieved_offered_ratio", 0.0)
        ),
        "drain_ratio": float(stationarity.get("drain_ratio", 0.0)),
        "system_backlog_growth_fraction": float(
            nested_get(
                stationarity,
                "profiles",
                "system",
                "final_half_growth_fraction",
            )
        ),
        "system_backlog_slope_rps": float(
            nested_get(
                stationarity,
                "profiles",
                "system",
                "slope_requests_per_s",
            )
        ),
        "ann_backlog_slope_rps": float(
            nested_get(
                stationarity,
                "profiles",
                "ann_waiting",
                "slope_requests_per_s",
            )
        ),
        "first_token_backlog_slope_rps": float(
            nested_get(
                stationarity,
                "profiles",
                "first_token_waiting",
                "slope_requests_per_s",
            )
        ),
        "slo_attainment": float(summary["slo_attainment"]),
        "slo_goodput": float(summary["slo_compliant_qps"]),
        "e2e_slo_attainment": e2e_slo_attainment,
        "e2e_slo_goodput": e2e_slo_goodput,
        "e2e_p99_ms": float(nested_get(summary, "e2e_ms", "p99")),
        "ttft_p99_ms": float(nested_get(summary, "rag_ttft_ms", "p99")),
        "itl_p99_ms": float(nested_get(summary, "itl_ms", "p99")),
        "tpot_p99_ms": float(nested_get(summary, "tpot_ms", "p99")),
        "ann_p99_ms": float(nested_get(summary, "ann_latency_ms", "p99")),
        "ann_queue_p99_ms": float(
            nested_get(summary, "ann_queue_wait_ms", "p99")
        ),
        "ann_gpu_p99_ms": float(nested_get(summary, "ann_gpu_latency_ms", "p99")),
        "ann_cpu_queue_p99_ms": float(
            nested_get(summary, "ann_cpu_queue_wait_ms", "p99")
        ),
        "ann_cpu_p99_ms": float(
            nested_get(summary, "ann_cpu_search_latency_ms", "p99")
        ),
        "ann_hot_probe_ratio": float(
            nested_get(summary, "ann_hot_probe_ratio", "mean")
        ),
        "decode_step_p99_ms": float(runtime.get("decode_iteration_p99_ms", 0)),
        "memory_bandwidth_mean_pct": float(
            nested_get(gpu_telemetry, "memory_bandwidth_util_pct", "mean")
        ),
        "memory_bandwidth_p95_pct": float(
            nested_get(gpu_telemetry, "memory_bandwidth_util_pct", "p95")
        ),
        "gpu_util_mean_pct": float(
            nested_get(gpu_telemetry, "gpu_util_pct", "mean")
        ),
        "partition_switches": float(runtime.get("pdmux_stream_group_switches", 0)),
        "partition_switch_p50_ms": float(switch_latency.get("p50_ms", 0)),
        "partition_switch_p99_ms": float(switch_latency.get("p99_ms", 0)),
        "lease_switch_benefit_p50_ms": float(switch_benefit.get("p50_ms", 0)),
        "lease_effective_switch_cost_ms": float(
            coalescing.get("effective_switch_cost_ms", 0)
        ),
        "lease_admission_waits": float(coalescing.get("admission_waits", 0)),
        "lease_work_holds": float(coalescing.get("work_holds", 0)),
        "lease_cost_holds": float(coalescing.get("cost_holds", 0)),
        "ann_deadline_fraction": float(launches.get("ann_deadline", 0))
        / total_launches,
        "ann_max_defer_fraction": float(launches.get("ann_max_defer", 0))
        / total_launches,
    }

def aggregate_results(
    cases: List[ProfileCase],
    results: Dict[str, Dict[str, Any]],
    e2e_slo_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[tuple[ProfileCase, Dict[str, Any]]]] = {}
    for case in cases:
        if case.name not in results:
            continue
        grouped.setdefault(case.key, []).append((case, results[case.name]))

    aggregates = []
    for key, entries in grouped.items():
        first_case = entries[0][0]
        rows = [result_metrics(result, e2e_slo_ms) for _, result in entries]
        metrics = {}
        for metric in rows[0]:
            values = [row[metric] for row in rows]
            average = mean(values)
            half_width = ci95_half_width(values)
            metrics[metric] = {
                "mean": average,
                "std": sample_std(values),
                "min": min(values),
                "max": max(values),
                "ci95_low": average - half_width,
                "ci95_high": average + half_width,
            }
        aggregates.append(
            {
                "key": key,
                "policy": first_case.policy,
                "retrieval_backend": first_case.retrieval_backend,
                "schedule": first_case.schedule,
                "request_rate": first_case.request_rate,
                "decode_slo_ms": first_case.decode_slo_ms,
                "max_defer_ms": first_case.max_defer_ms,
                "stream_epoch_ms": first_case.stream_epoch_ms,
                "ann_lease_sm": first_case.ann_lease_sm,
                "prefill_lease_sm": first_case.prefill_lease_sm,
                "prefill_lease_min_layers": (
                    first_case.prefill_lease_min_layers
                ),
                "prefill_lease_max_layers": (
                    first_case.prefill_lease_max_layers
                ),
                "prefill_max_consecutive_chunks": (
                    first_case.prefill_max_consecutive_chunks
                ),
                "pdmux_config_path": first_case.pdmux_config_path,
                "repetitions": len(entries),
                "metrics": metrics,
            }
        )
    return aggregates


def paired_policy_comparisons(
    cases: List[ProfileCase],
    results: Dict[str, Dict[str, Any]],
    e2e_slo_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[float, int, int], Dict[str, Dict[str, Any]]] = {}
    for case in cases:
        result = results.get(case.name)
        if result is None:
            continue
        grouped.setdefault(
            (case.request_rate, case.seed, case.repetition), {}
        )[case.policy] = result

    per_comparison: Dict[tuple[float, str], List[Dict[str, float]]] = {}
    comparisons = (
        ("dynamic-only", "naive"),
        ("vectorliterag", "naive"),
        ("raglease-hybrid", "naive-hybrid"),
        ("raglease-hybrid", "vectorliterag"),
        ("raglease-hybrid", "hedrarag"),
        ("raglease-hybrid", "hedrarag-hybrid"),
        ("full", "naive"),
        ("full", "vectorliterag"),
        ("full", "hedrarag"),
        ("full", "dynamic-only"),
    )
    for (request_rate, seed, repetition), policy_results in grouped.items():
        hashes = {
            nested_get(result, "workload", "sha256", default="")
            for result in policy_results.values()
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"Workload SHA mismatch at rps={request_rate:g}, seed={seed}"
                f", repetition={repetition}"
            )
        for target, baseline in comparisons:
            if target not in policy_results or baseline not in policy_results:
                continue
            target_metrics = result_metrics(policy_results[target], e2e_slo_ms)
            baseline_metrics = result_metrics(policy_results[baseline], e2e_slo_ms)

            def gain(metric: str) -> float:
                base = baseline_metrics[metric]
                return 0.0 if base == 0 else 100.0 * (
                    target_metrics[metric] - base
                ) / base

            def reduction(metric: str) -> float:
                return -gain(metric)

            per_comparison.setdefault(
                (request_rate, f"{target}_vs_{baseline}"), []
            ).append(
                {
                    "e2e_slo_goodput_gain_pct": gain("e2e_slo_goodput"),
                    "e2e_slo_attainment_delta_pp": 100.0
                    * (
                        target_metrics["e2e_slo_attainment"]
                        - baseline_metrics["e2e_slo_attainment"]
                    ),
                    "slo_goodput_gain_pct": gain("slo_goodput"),
                    "slo_attainment_delta_pp": 100.0
                    * (
                        target_metrics["slo_attainment"]
                        - baseline_metrics["slo_attainment"]
                    ),
                    "ttft_p99_reduction_pct": reduction("ttft_p99_ms"),
                    "itl_p99_reduction_pct": reduction("itl_p99_ms"),
                    "e2e_p99_reduction_pct": reduction("e2e_p99_ms"),
                    "ann_p99_change_pct": gain("ann_p99_ms"),
                    "partition_switch_reduction_pct": reduction(
                        "partition_switches"
                    ),
                }
            )

    output = []
    for (request_rate, comparison), rows in sorted(per_comparison.items()):
        metrics = {}
        for metric in rows[0]:
            values = [row[metric] for row in rows]
            average = mean(values)
            half_width = ci95_half_width(values)
            metrics[metric] = {
                "mean": average,
                "std": sample_std(values),
                "ci95_low": average - half_width,
                "ci95_high": average + half_width,
            }
        output.append(
            {
                "request_rate": request_rate,
                "comparison": comparison,
                "paired_seeds": len(rows),
                "metrics": metrics,
            }
        )
    return output


def paired_hybrid_equivalence_audit(
    cases: List[ProfileCase],
    results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Audit the retrieval contract behind each Hybrid scheduler pair."""
    contract = config.get("comparison_contract", {})
    allow_policy_specific_hbm = bool(
        contract.get("allow_policy_specific_shared_hbm", False)
    )
    max_hot_ratio_delta = float(
        contract.get("max_hot_probe_ratio_delta", 0.005)
    )
    grouped: Dict[tuple[float, int, int], Dict[str, Dict[str, Any]]] = {}
    for case in cases:
        result = results.get(case.name)
        if result is None or case.policy not in {
            "naive-hybrid",
            "raglease-hybrid",
        }:
            continue
        grouped.setdefault(
            (case.request_rate, case.seed, case.repetition), {}
        )[case.policy] = result

    audited = []
    server_fields = (
        "rag_index_nprobe",
        "rag_index_topk",
        "rag_hot_index_path",
        "rag_hot_index_map_path",
        "rag_hot_index_shard_id",
        "rag_cpu_retrieval_threads",
        "rag_doc_token_cache_size",
    )
    for (request_rate, seed, repetition), pair in sorted(grouped.items()):
        if not {"naive-hybrid", "raglease-hybrid"}.issubset(pair):
            continue
        naive = pair["naive-hybrid"]
        raglease = pair["raglease-hybrid"]
        naive_server = naive.get("server", {})
        raglease_server = raglease.get("server", {})
        mismatches = [
            field
            for field in server_fields
            if naive_server.get(field) != raglease_server.get(field)
        ]
        if nested_get(naive, "workload", "sha256", default="") != nested_get(
            raglease, "workload", "sha256", default=""
        ):
            mismatches.append("workload.sha256")

        naive_hbm_mode = str(naive_server.get("rag_shared_hbm_mode", "off"))
        raglease_hbm_mode = str(
            raglease_server.get("rag_shared_hbm_mode", "off")
        )
        if naive_hbm_mode != raglease_hbm_mode:
            mismatches.append("rag_shared_hbm_mode")

        naive_hot_ratio = float(
            nested_get(naive, "summary", "ann_hot_probe_ratio", "mean")
        )
        raglease_hot_ratio = float(
            nested_get(raglease, "summary", "ann_hot_probe_ratio", "mean")
        )
        hot_ratio_delta = raglease_hot_ratio - naive_hot_ratio
        if abs(hot_ratio_delta) > max_hot_ratio_delta:
            mismatches.append("summary.ann_hot_probe_ratio.mean")

        strict_mismatches = list(mismatches)
        if allow_policy_specific_hbm:
            strict_mismatches = [
                field
                for field in strict_mismatches
                if field
                not in {
                    "rag_shared_hbm_mode",
                    "summary.ann_hot_probe_ratio.mean",
                }
            ]
        if strict_mismatches:
            raise RuntimeError(
                "Hybrid comparison contract mismatch at "
                f"rps={request_rate:g}, seed={seed}, repetition={repetition}: "
                + ", ".join(strict_mismatches)
            )
        audited.append(
            {
                "request_rate": request_rate,
                "seed": seed,
                "repetition": repetition,
                "naive_shared_hbm_mode": naive_hbm_mode,
                "raglease_shared_hbm_mode": raglease_hbm_mode,
                "naive_hot_probe_ratio": naive_hot_ratio,
                "raglease_hot_probe_ratio": raglease_hot_ratio,
                "hot_probe_ratio_delta": hot_ratio_delta,
                "allowed_policy_specific_shared_hbm": (
                    allow_policy_specific_hbm
                ),
                "observed_mismatches": mismatches,
                "passed": not strict_mismatches,
            }
        )
    return audited


def select_profiles(
    aggregates: List[Dict[str, Any]], selection: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    min_attainment = float(selection.get("min_slo_attainment", 0.99))
    min_repetitions = int(selection.get("min_repetitions", 1))
    e2e_slo = float(selection.get("e2e_slo_ms", -1))
    ttft_slo = float(selection["ttft_slo_ms"])
    itl_slo = float(selection["itl_slo_ms"])
    stability_weight = float(selection.get("stability_weight", 1.0))
    require_stationarity = bool(selection.get("require_stationarity", False))
    use_e2e_slo = e2e_slo > 0
    attainment_metric = (
        "e2e_slo_attainment" if use_e2e_slo else "slo_attainment"
    )
    goodput_metric = "e2e_slo_goodput" if use_e2e_slo else "slo_goodput"

    by_rate: Dict[float, List[Dict[str, Any]]] = {}
    for candidate in aggregates:
        if candidate.get("policy", "full") not in RAGLEASE_POLICIES:
            continue
        by_rate.setdefault(float(candidate["request_rate"]), []).append(candidate)

    selected = {}
    for request_rate, candidates in sorted(by_rate.items()):
        feasible = []
        for candidate in candidates:
            metrics = candidate["metrics"]
            is_feasible = (
                candidate["repetitions"] >= min_repetitions
                and metrics["failed"]["max"] == 0
                and metrics[attainment_metric]["mean"] >= min_attainment
                and (
                    not require_stationarity
                    or (
                        metrics["stationarity_available"]["min"] >= 1.0
                        and metrics["stationarity_pass"]["min"] >= 1.0
                    )
                )
            )
            if not use_e2e_slo:
                is_feasible = (
                    is_feasible
                    and metrics["ttft_p99_ms"]["mean"] <= ttft_slo
                    and metrics["itl_p99_ms"]["mean"] <= itl_slo
                )
            candidate["feasible"] = is_feasible
            candidate["selection_objective"] = (
                "e2e_slo_goodput" if use_e2e_slo else "ttft_itl_slo_goodput"
            )
            if is_feasible:
                robust_goodput = (
                    metrics[goodput_metric]["mean"]
                    - stability_weight * metrics[goodput_metric]["std"]
                )
                candidate["robust_goodput"] = robust_goodput
                feasible.append(candidate)
        if feasible:
            if use_e2e_slo:
                best = min(
                    feasible,
                    key=lambda item: (
                        -item["robust_goodput"],
                        item["metrics"]["e2e_p99_ms"]["mean"],
                        item["metrics"]["ann_p99_ms"]["mean"],
                        item["metrics"]["itl_p99_ms"]["mean"],
                    ),
                )
            else:
                best = min(
                    feasible,
                    key=lambda item: (
                        -item["robust_goodput"],
                        item["metrics"]["itl_p99_ms"]["mean"],
                        item["metrics"]["ttft_p99_ms"]["mean"],
                        item["metrics"]["ann_p99_ms"]["mean"],
                    ),
                )
            selected[format_number(request_rate)] = best
    return selected

def merge_latency_distributions(
    distributions: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    values = [
        value
        for value in distributions
        if isinstance(value, dict) and int(value.get("samples", 0)) > 0
    ]
    if not values:
        return None
    sample_count = sum(int(value.get("samples", 0)) for value in values)
    weighted_mean = sum(
        float(value.get("mean_ms", 0.0)) * int(value.get("samples", 0))
        for value in values
    ) / max(1, sample_count)
    return {
        "samples": float(sample_count),
        "mean_ms": weighted_mean,
        "p50_ms": statistics.median(
            float(value.get("p50_ms", 0.0)) for value in values
        ),
        "p95_ms": max(float(value.get("p95_ms", 0.0)) for value in values),
        "p99_ms": max(float(value.get("p99_ms", 0.0)) for value in values),
    }


def merge_latency_trees(nodes: Iterable[Any]) -> Dict[str, Any]:
    mappings = [node for node in nodes if isinstance(node, dict)]
    keys = sorted({str(key) for node in mappings for key in node})
    merged: Dict[str, Any] = {}
    for key in keys:
        children = [node.get(key) for node in mappings if key in node]
        distributions = [
            child
            for child in children
            if isinstance(child, dict) and "p95_ms" in child
        ]
        if distributions:
            value = merge_latency_distributions(distributions)
        else:
            value = merge_latency_trees(children)
        if value:
            merged[key] = value
    return merged


def build_performance_priors(
    config: Dict[str, Any],
    selected: Dict[str, Dict[str, Any]],
    cases: List[ProfileCase],
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    priors: Dict[str, Dict[str, Any]] = {}
    for rate, candidate in selected.items():
        runtimes = []
        for case in cases:
            if case.key != candidate["key"] or case.name not in results:
                continue
            entries = results[case.name].get("server", {}).get("rag_ann_runtime") or []
            if entries and isinstance(entries[-1], dict):
                runtimes.append(entries[-1])

        ann_profiles = merge_latency_trees(
            runtime.get("ann_gpu_service_by_batch_size", {})
            for runtime in runtimes
        )
        prefill_profiles = merge_latency_trees(
            runtime.get("prefill_layer_latency_by_partition_bucket", {})
            for runtime in runtimes
        )
        decode_iteration_profiles = merge_latency_trees(
            runtime.get("decode_iteration_by_partition_batch_bucket", {})
            for runtime in runtimes
        )
        decode_gpu_profiles = merge_latency_trees(
            runtime.get("decode_gpu_step_by_partition_batch_bucket", {})
            for runtime in runtimes
        )
        switch_profiles = merge_latency_trees(
            runtime.get("partition_switch_latency_by_pair", {})
            for runtime in runtimes
        )

        shared_hbm = [
            runtime.get("shared_hbm")
            for runtime in runtimes
            if isinstance(runtime.get("shared_hbm"), dict)
        ]
        async_reload = merge_latency_distributions(
            {
                "samples": int(value.get("ivf_reload_async_completions", 0)),
                "mean_ms": float(value.get("ivf_reload_async_latency_p50_ms", 0.0)),
                "p50_ms": float(value.get("ivf_reload_async_latency_p50_ms", 0.0)),
                "p95_ms": float(value.get("ivf_reload_async_latency_p95_ms", 0.0)),
                "p99_ms": float(value.get("ivf_reload_async_latency_p99_ms", 0.0)),
            }
            for value in shared_hbm
        )
        reload_per_list = max(
            [
                float(
                    value.get("ivf", {})
                    .get("cluster_lease", {})
                    .get("reload_ms_per_list_ewma", 0.0)
                )
                for value in shared_hbm
            ]
            or [0.0]
        )

        priors[rate] = {
            "phase_latency_profiles": {
                "ann_gpu_service_by_batch_size": ann_profiles,
                "prefill_layer_latency_by_partition_bucket": prefill_profiles,
                "decode_iteration_by_partition_batch_bucket": (
                    decode_iteration_profiles
                ),
                "decode_gpu_step_by_partition_batch_bucket": decode_gpu_profiles,
            },
            "ann_decode_interference_guard": {
                "decode_tail_profiles": decode_gpu_profiles,
                "safety_margin_ms": float(
                    config["runtime"].get("ann_safety_margin_ms", 0.0)
                ),
                "max_decode_slowdown_ratio": float(
                    config["runtime"].get("ann_max_decode_slowdown_ratio", 1.0)
                ),
            },
            "resource_transition_profiles": {
                "green_context_switch_by_pair": switch_profiles,
                "hbm_reload": {
                    "reload_ms_per_list": reload_per_list,
                    "async_reload_latency": async_reload or {},
                },
            },
            "initial_hot_cluster_prior": {
                "source": "vectorlite_offline_probe_frequency_order",
                "hot_index_path": config["paths"].get("hot_index_path"),
                "hot_index_map_path": config["paths"].get("hot_index_map_path"),
            },
        }
    return priors


def build_recommended_profile(
    config: Dict[str, Any],
    selected: Dict[str, Dict[str, Any]],
    performance_priors: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    performance_priors = performance_priors or {}
    profiles = {}
    for rate, candidate in selected.items():
        profiles[rate] = {
            "request_rate": candidate["request_rate"],
            "rag_ann_schedule": candidate["schedule"],
            "rag_ann_work_conserving_decode_overlap": bool(
                candidate["schedule"] == "decode-gc"
                and config["runtime"].get(
                    "ann_work_conserving_decode_overlap", True
                )
            ),
            "rag_ann_decode_step_slo_ms": candidate["decode_slo_ms"],
            "rag_ann_max_defer_ms": candidate["max_defer_ms"],
            "rag_ann_stream_epoch_ms": (
                config["runtime"].get("ann_stream_epoch_ms", 0)
                if candidate["stream_epoch_ms"] is None
                else candidate["stream_epoch_ms"]
            ),
            "rag_ann_lease_frontend_sm": candidate["ann_lease_sm"],
            "rag_prefill_lease_frontend_sm": candidate["prefill_lease_sm"],
            "pdmux_config_path": candidate["pdmux_config_path"],
            "performance_priors": performance_priors.get(rate, {}),
            "measured": {
                key: value["mean"]
                for key, value in candidate["metrics"].items()
                if key
                in {
                    "slo_attainment",
                    "slo_goodput",
                    "stationarity_available",
                    "stationarity_pass",
                    "achieved_offered_ratio",
                    "drain_ratio",
                    "system_backlog_growth_fraction",
                    "system_backlog_slope_rps",
                    "ann_backlog_slope_rps",
                    "first_token_backlog_slope_rps",
                    "e2e_slo_attainment",
                    "e2e_slo_goodput",
                    "e2e_p99_ms",
                    "ttft_p99_ms",
                    "itl_p99_ms",
                    "tpot_p99_ms",
                    "ann_p99_ms",
                    "partition_switches",
                    "partition_switch_p50_ms",
                    "partition_switch_p99_ms",
                    "lease_switch_benefit_p50_ms",
                    "lease_effective_switch_cost_ms",
                    "lease_admission_waits",
                    "lease_work_holds",
                    "lease_cost_holds",
                    "ann_deadline_fraction",
                    "ann_max_defer_fraction",
                }
            },
        }
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_profile_rate": (
            max(
                (
                    float(profile["request_rate"])
                    for profile in profiles.values()
                ),
                default=-1.0,
            )
        ),
        "deployment": {
            "model_path": config["paths"]["model_path"],
            "index_path": config["paths"]["index_path"],
            "hot_index_path": config["paths"].get("hot_index_path"),
            "hot_index_map_path": config["paths"].get("hot_index_map_path"),
            "faiss_egg_dir": config["paths"].get("faiss_egg_dir"),
            "hot_index_shard_id": config["runtime"].get(
                "hot_index_shard_id", 0
            ),
            "physical_gpu": config["runtime"]["physical_gpu"],
            "nprobe": config["runtime"]["nprobe"],
            "topk": config["runtime"]["topk"],
        },
        "fixed_runtime": {
            **{
                key: config["runtime"][key]
                for key in (
                    "ann_deadline_ms",
                    "ann_safety_margin_ms",
                    "partition_hysteresis_bs",
                    "partition_min_residency_ms",
                    "ann_max_inflight",
                    "ann_max_inflight_during_decode",
                    "ann_max_batch_size",
                    "ann_max_queued_gpu_ms",
                    "ann_max_microbatch_gpu_ms",
                    "ann_service_time_percentile",
                    "ann_service_time_min_samples",
                    "ann_max_batch_wait_ms",
                )
            },
            "ann_max_batches_per_tick": config["runtime"].get(
                "ann_max_batches_per_tick", 0
            ),
        },
        "selection_constraints": config["selection"],
        "profiles_by_request_rate": profiles,
    }


def render_report(
    aggregates: List[Dict[str, Any]],
    selected: Dict[str, Dict[str, Any]],
    paired: Optional[List[Dict[str, Any]]] = None,
) -> str:
    selected_keys = {candidate["key"] for candidate in selected.values()}
    lines = [
        "# RAG Scheduler Offline Profiling Report",
        "",
        "| Selected | Policy | Schedule | RPS | Decode SLO | Max defer | Epoch | "
        "ANN/PF SM | Request QPS | Stable | Ach/Off | Drain | "
        "E2E SLO attainment | E2E goodput | P99 E2E | P99 ANN | "
        "Decode-step P99 | Mem BW mean/p95 | P99 TTFT | P99 ITL | Switches | "
        "Deadline % | Forced % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in sorted(
        aggregates,
        key=lambda item: (
            item["request_rate"],
            item["schedule"],
            item["decode_slo_ms"],
            item["max_defer_ms"],
            -1 if item["stream_epoch_ms"] is None else item["stream_epoch_ms"],
        ),
    ):
        metrics = candidate["metrics"]
        lines.append(
            "| {selected} | {policy} | {schedule} | {rps:g} | {slo:g} | {defer:g} | "
            "{epoch} | {ann}/{prefill} | {qps:.3f} | {stable:.0%} | "
            "{achieved_offered:.3f} | {drain:.1%} | {e2e_attainment:.1%} | "
            "{e2e_goodput:.3f} | {e2e:.2f} | {ann_latency:.2f} | "
            "{decode_step:.2f} | {mem_mean:.1f}/{mem_p95:.1f}% | {ttft:.2f} | "
            "{itl:.2f} | {switches:.1f} | {deadline:.1%} | {forced:.1%} |".format(
                selected="yes" if candidate["key"] in selected_keys else "",
                policy=candidate.get("policy", "full"),
                schedule=candidate["schedule"],
                rps=candidate["request_rate"],
                slo=candidate["decode_slo_ms"],
                defer=candidate["max_defer_ms"],
                epoch=(
                    "-"
                    if candidate["stream_epoch_ms"] is None
                    else f"{candidate['stream_epoch_ms']:g}"
                ),
                ann=candidate["ann_lease_sm"],
                prefill=candidate["prefill_lease_sm"],
                qps=metrics["request_qps"]["mean"],
                stable=metrics["stationarity_pass"]["mean"],
                achieved_offered=metrics["achieved_offered_ratio"]["mean"],
                drain=metrics["drain_ratio"]["mean"],
                e2e_attainment=metrics["e2e_slo_attainment"]["mean"],
                e2e_goodput=metrics["e2e_slo_goodput"]["mean"],
                e2e=metrics["e2e_p99_ms"]["mean"],
                ttft=metrics["ttft_p99_ms"]["mean"],
                itl=metrics["itl_p99_ms"]["mean"],
                ann_latency=metrics["ann_p99_ms"]["mean"],
                decode_step=metrics["decode_step_p99_ms"]["mean"],
                mem_mean=metrics["memory_bandwidth_mean_pct"]["mean"],
                mem_p95=metrics["memory_bandwidth_p95_pct"]["mean"],
                switches=metrics["partition_switches"]["mean"],
                deadline=metrics["ann_deadline_fraction"]["mean"],
                forced=metrics["ann_max_defer_fraction"]["mean"],
            )
        )
    if paired:
        lines.extend(
            [
                "",
                "## Paired Policy Comparisons",
                "",
                "| Comparison | RPS | Seeds | E2E goodput gain | "
                "E2E attainment delta | P99 E2E reduction | P99 ANN change | "
                "P99 TTFT reduction | P99 ITL reduction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paired:
            metrics = row["metrics"]
            lines.append(
                "| {comparison} | {rps:g} | {seeds} | {goodput:.2f}% | "
                "{attainment:.2f} pp | {e2e:.2f}% | {ann:.2f}% | "
                "{ttft:.2f}% | {itl:.2f}% |".format(
                    comparison=row["comparison"],
                    rps=row["request_rate"],
                    seeds=row["paired_seeds"],
                    goodput=metrics["e2e_slo_goodput_gain_pct"]["mean"],
                    attainment=metrics["e2e_slo_attainment_delta_pp"]["mean"],
                    e2e=metrics["e2e_p99_reduction_pct"]["mean"],
                    ann=metrics["ann_p99_change_pct"]["mean"],
                    ttft=metrics["ttft_p99_reduction_pct"]["mean"],
                    itl=metrics["itl_p99_reduction_pct"]["mean"],
                )
            )
    lines.extend(
        [
            "",
            "Selection maximizes robust E2E SLO goodput when e2e_slo_ms > 0; "
            "otherwise it preserves the legacy TTFT/ITL SLO objective.",
            "",
        ]
    )
    return "\n".join(lines)

def print_plan(cases: List[ProfileCase]) -> None:
    print(f"Generated {len(cases)} cases")
    for case in cases:
        print(case.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--case-regex", help="Run/analyze only case names matching this regex"
    )
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--allow-large-sweep", action="store_true")
    parser.add_argument("--max-cases", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    cases = build_cases(config)
    if args.case_regex:
        pattern = re.compile(args.case_regex)
        cases = [case for case in cases if pattern.search(case.name)]
    if args.limit_cases is not None:
        if args.limit_cases <= 0:
            raise ValueError("--limit-cases must be positive")
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError("No profiling cases matched the requested filters")
    if len(cases) > args.max_cases and not args.allow_large_sweep:
        raise ValueError(
            f"Sweep has {len(cases)} cases, exceeding --max-cases={args.max_cases}. "
            "Narrow the grid or pass --allow-large-sweep."
        )
    if args.dry_run:
        print_plan(cases)
        return

    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = OfflineProfiler(config, output_dir, args.overwrite)
    profiler.prepare_workload_traces(cases)
    (output_dir / "profiling_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    config_snapshot_dir = output_dir / "pdmux_configs"
    config_snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted({case.pdmux_config_path for case in cases}):
        shutil.copy2(source, config_snapshot_dir / Path(source).name)
    (output_dir / "case_manifest.json").write_text(
        json.dumps(
            [
                asdict(case)
                | {
                    "name": case.name,
                    "workload_trace": str(profiler.workload_trace_path(case)),
                    "workload_sha256": load_workload(
                        profiler.workload_trace_path(case)
                    ).sha256(),
                }
                for case in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    results: Dict[str, Dict[str, Any]] = {}
    if args.analyze_only:
        for case in cases:
            path = profiler.result_path(case)
            if path.exists():
                result = load_json(path)
                profiler.validate_result(case, result)
                results[case.name] = result
    else:
        if os.name != "posix":
            raise RuntimeError("Execution mode requires Linux process groups")
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}]", end=" ")
            results[case.name] = profiler.run_case_with_retries(case)

    e2e_slo_ms = float(config["selection"].get("e2e_slo_ms", -1))
    comparison_audit = paired_hybrid_equivalence_audit(
        cases, results, config
    )
    aggregates = aggregate_results(cases, results, e2e_slo_ms)
    paired = paired_policy_comparisons(cases, results, e2e_slo_ms)
    selected = select_profiles(aggregates, config["selection"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_cases": len(results),
        "expected_cases": len(cases),
        "hybrid_comparison_audit": comparison_audit,
        "aggregates": aggregates,
        "paired_policy_comparisons": paired,
        "selected": selected,
    }
    (output_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    performance_priors = build_performance_priors(
        config, selected, cases, results
    )
    profile = build_recommended_profile(config, selected, performance_priors)
    dump_yaml_compatible(output_dir / "recommended_profile.yaml", profile)
    (output_dir / "profiling_report.md").write_text(
        render_report(aggregates, selected, paired), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'profile_summary.json'}")
    print(f"Wrote {output_dir / 'recommended_profile.yaml'}")
    print(f"Wrote {output_dir / 'profiling_report.md'}")


if __name__ == "__main__":
    main()
