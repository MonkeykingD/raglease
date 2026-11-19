from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from benchmark.pdmux.bench_rag_serving import (
    SERVER_INFO_FIELDS,
    RequestResult,
    should_normalize_stationarity_by_observed_trace_rate,
    summarize_stationarity,
)
from benchmark.pdmux.profile_rag_scheduler import (
    GpuTelemetrySampler,
    OfflineProfiler,
    ProfileCase,
    allowed_hybrid_result_index_modes,
    build_cases,
    build_recommended_profile,
    effective_mem_fraction_static,
    paired_hybrid_equivalence_audit,
    paired_policy_comparisons,
    normalized_result_stationarity,
    resolve_hybrid_index_paths,
    result_metrics,
    select_profiles,
    validate_experiment_contract,
)


def metric(value, std=0.0):
    return {"mean": value, "std": std, "min": value, "max": value}


class TestRagSchedulerProfiler(unittest.TestCase):
    @staticmethod
    def make_profiler(output_dir, **runtime):
        paths = {
            "project_root": str(output_dir),
            "model_path": "/tmp/model",
            "index_path": "/tmp/index",
            "query_vec_path": "/tmp/queries.fbin",
            "query_ids_path": "/tmp/qids.ibin",
        }
        runtime = {
            "nprobe": 1,
            "topk": 1,
            "doc_random_fallback": True,
            **runtime,
        }
        return OfflineProfiler(
            {
                "paths": paths,
                "runtime": runtime,
                "workload": {},
                "selection": {
                    "ttft_slo_ms": -1,
                    "itl_slo_ms": -1,
                    "e2e_slo_ms": -1,
                },
            },
            Path(output_dir),
            overwrite=False,
        )

    @staticmethod
    def make_case():
        return ProfileCase(
            schedule="frontend-gc",
            request_rate=3.5,
            decode_slo_ms=20,
            max_defer_ms=50,
            ann_lease_sm=-1,
            prefill_lease_sm=-1,
            pdmux_config_path="/tmp/t24.yaml",
            repetition=1,
        )

    def test_recommended_profile_uses_numeric_rate_from_encoded_key(self):
        runtime = {
            "physical_gpu": 0,
            "nprobe": 1,
            "topk": 1,
            "ann_deadline_ms": 50,
            "ann_safety_margin_ms": 1,
            "partition_hysteresis_bs": 1,
            "partition_min_residency_ms": 0,
            "ann_max_inflight": 1,
            "ann_max_inflight_during_decode": 0,
            "ann_max_batch_size": 1,
            "ann_max_queued_gpu_ms": 1,
            "ann_max_microbatch_gpu_ms": 1,
            "ann_service_time_percentile": 95,
            "ann_service_time_min_samples": 1,
            "ann_max_batch_wait_ms": 0,
        }
        config = {
            "paths": {"model_path": "/tmp/model", "index_path": "/tmp/index"},
            "runtime": runtime,
            "selection": {},
        }
        candidate = {
            "request_rate": 2.5,
            "schedule": "decode-gc",
            "decode_slo_ms": 30,
            "max_defer_ms": 50,
            "stream_epoch_ms": None,
            "ann_lease_sm": -1,
            "prefill_lease_sm": 72,
            "pdmux_config_path": "/tmp/pdmux.yaml",
            "metrics": {},
        }

        profile = build_recommended_profile(config, {"2p5": candidate})

        self.assertEqual(profile["default_profile_rate"], 2.5)

    def test_shared_hbm_hybrid_allows_controlled_cpu_fallback(self):
        case = replace(self.make_case(), policy="raglease-hybrid")

        self.assertEqual(
            allowed_hybrid_result_index_modes(
                {
                    "shared_hbm_mode": "scratch",
                    "raglease_hybrid_shared_hbm_enabled": True,
                },
                case,
            ),
            {"vectorlite_hot_cold", "cpu_fallback"},
        )
        self.assertEqual(
            allowed_hybrid_result_index_modes(
                {"shared_hbm_mode": "scratch"}, case
            ),
            {"vectorlite_hot_cold"},
        )

    def test_raglease_can_use_policy_specific_static_memory_fraction(self):
        runtime = {
            "mem_fraction_static": 0.39,
            "raglease_hybrid_mem_fraction_static": 0.375,
        }
        self.assertEqual(
            effective_mem_fraction_static(runtime, "vectorliterag"), 0.39
        )
        self.assertEqual(
            effective_mem_fraction_static(runtime, "hedrarag-hybrid"), 0.39
        )
        self.assertEqual(
            effective_mem_fraction_static(runtime, "raglease-hybrid"), 0.375
        )

        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir, **runtime)
            profiler.paths.update(
                {
                    "hybrid_cpu_index_path": "/tmp/ivffs.index",
                    "hybrid_hot_index_path": "/tmp/hot.index",
                    "hybrid_hot_index_map_path": "/tmp/hot.imap",
                }
            )
            raglease = replace(self.make_case(), policy="raglease-hybrid")
            vectorlite = replace(
                self.make_case(),
                policy="vectorliterag",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
            )
            raglease_command = profiler.build_server_command(raglease)
            vectorlite_command = profiler.build_server_command(vectorlite)
            self.assertEqual(
                raglease_command[
                    raglease_command.index("--mem-fraction-static") + 1
                ],
                "0.375",
            )
            self.assertEqual(
                vectorlite_command[
                    vectorlite_command.index("--mem-fraction-static") + 1
                ],
                "0.39",
            )

    def test_raglease_can_use_full_candidate_index_with_static_baselines(self):
        paths = {
            "hybrid_cpu_index_path": "/index/cpu",
            "hybrid_hot_index_path": "/index/static-hot",
            "hybrid_hot_index_map_path": "/index/static.imap",
            "raglease_hybrid_hot_index_path": "/index/full-candidates",
            "raglease_hybrid_hot_index_map_path": "/index/all.imap",
        }

        self.assertEqual(
            resolve_hybrid_index_paths(paths, "vectorliterag"),
            ("/index/cpu", "/index/static-hot", "/index/static.imap"),
        )
        self.assertEqual(
            resolve_hybrid_index_paths(paths, "hedrarag-hybrid"),
            ("/index/cpu", "/index/static-hot", "/index/static.imap"),
        )
        self.assertEqual(
            resolve_hybrid_index_paths(paths, "raglease-hybrid"),
            ("/index/cpu", "/index/full-candidates", "/index/all.imap"),
        )

    def test_case_retry_archives_failed_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, case_max_attempts=2, case_retry_delay_s=0
            )
            case = self.make_case()
            artifacts = (
                profiler.result_path(case),
                profiler.server_info_path(case),
                profiler.server_log_path(case),
                profiler.benchmark_log_path(case),
            )
            for path in artifacts:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("failed", encoding="utf-8")

            expected = {"summary": "ok"}
            profiler.run_case = Mock(
                side_effect=[RuntimeError("transient"), expected]
            )

            self.assertIs(profiler.run_case_with_retries(case), expected)
            self.assertEqual(profiler.run_case.call_count, 2)
            archive_dir = (
                Path(output_dir)
                / "failed_attempts"
                / case.name
                / "attempt_1"
            )
            self.assertEqual(
                {path.name for path in archive_dir.iterdir()},
                {path.name for path in artifacts},
            )
            self.assertTrue(all(not path.exists() for path in artifacts))

    def test_gpu_telemetry_summary_uses_only_measurement_window(self):
        sampler = GpuTelemetrySampler(gpu_index=0, interval_ms=500)
        sampler._samples = [
            {
                "timestamp_unix_s": 99.5,
                "gpu_util_pct": 1.0,
                "memory_bandwidth_util_pct": 2.0,
                "memory_used_mib": 3.0,
            },
            {
                "timestamp_unix_s": 100.0,
                "gpu_util_pct": 40.0,
                "memory_bandwidth_util_pct": 60.0,
                "memory_used_mib": 1000.0,
            },
            {
                "timestamp_unix_s": 101.0,
                "gpu_util_pct": 80.0,
                "memory_bandwidth_util_pct": 100.0,
                "memory_used_mib": 1200.0,
            },
            {
                "timestamp_unix_s": 101.5,
                "gpu_util_pct": 1.0,
                "memory_bandwidth_util_pct": 2.0,
                "memory_used_mib": 3.0,
            },
        ]

        summary = sampler.summarize(100.0, 101.0)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["gpu_util_pct"]["mean"], 60.0)
        self.assertEqual(
            summary["memory_bandwidth_util_pct"]["mean"], 80.0
        )

    def test_stationarity_summary_accepts_bounded_backlog(self):
        requests = [
            RequestResult(
                request_id=index,
                query_id=index,
                scheduled_arrival_ms=index * 1000.0,
                success=True,
                e2e_ms=500.0,
                ttft_ms=200.0,
                ann_latency_ms=50.0,
                ann_queue_wait_ms=10.0,
                ann_index_mode="full_gpu",
            )
            for index in range(10)
        ]

        summary = summarize_stationarity(
            requests,
            offered_request_rate_qps=1.0,
            duration_s=9.5,
        )

        self.assertTrue(summary["passed"])
        self.assertGreater(summary["achieved_offered_ratio"], 1.0)
        self.assertLess(summary["drain_ratio"], 0.1)
        self.assertLessEqual(
            summary["profiles"]["system"]["slope_ci95_low"],
            0.0,
        )

    def test_stationarity_uses_observed_rate_for_poisson_trace(self):
        interval_ms = 1000.0 / 0.94
        requests = [
            RequestResult(
                request_id=index,
                query_id=index,
                scheduled_arrival_ms=index * interval_ms,
                success=True,
                e2e_ms=500.0,
                ttft_ms=200.0,
                ann_latency_ms=50.0,
                ann_queue_wait_ms=10.0,
                ann_index_mode="full_gpu",
            )
            for index in range(100)
        ]
        duration_s = (
            requests[-1].scheduled_arrival_ms
            - requests[0].scheduled_arrival_ms
        ) / 1000.0 + 0.5

        nominal = summarize_stationarity(
            requests,
            offered_request_rate_qps=1.0,
            duration_s=duration_s,
        )
        observed = summarize_stationarity(
            requests,
            offered_request_rate_qps=1.0,
            duration_s=duration_s,
            normalize_by_observed_trace_rate=True,
        )

        self.assertFalse(nominal["passed"])
        self.assertTrue(observed["passed"])
        self.assertEqual(observed["rate_normalization"], "observed_trace")
        self.assertLess(observed["achieved_nominal_offered_ratio"], 0.95)
        self.assertGreater(observed["achieved_offered_ratio"], 0.99)

    def test_poisson_trace_uses_observed_rate_when_replayed_deterministically(self):
        workload = Mock(metadata={"arrival_process": "poisson"})

        self.assertTrue(
            should_normalize_stationarity_by_observed_trace_rate(
                workload,
                deterministic_arrivals=True,
            )
        )

    def test_stationarity_summary_rejects_growing_ann_backlog(self):
        requests = [
            RequestResult(
                request_id=index,
                query_id=index,
                scheduled_arrival_ms=index * 1000.0,
                success=True,
                e2e_ms=(index + 1) * 1000.0,
                ttft_ms=(index + 1) * 800.0,
                ann_latency_ms=(index + 1) * 700.0,
                ann_queue_wait_ms=index * 700.0,
                ann_index_mode="full_gpu",
            )
            for index in range(10)
        ]

        summary = summarize_stationarity(
            requests,
            offered_request_rate_qps=1.0,
            duration_s=19.0,
        )

        self.assertFalse(summary["passed"])
        self.assertIn("achieved_offered_ratio", summary["failures"])
        self.assertIn("drain_ratio", summary["failures"])
        self.assertGreater(
            summary["profiles"]["ann_waiting"]["slope_requests_per_s"],
            0.0,
        )

    def test_result_metrics_accepts_cpu_retrieval_without_gpu_runtime(self):
        distribution = {"mean": 1.0, "p99": 2.0}
        metrics = result_metrics(
            {
                "summary": {
                    "completed": 1,
                    "failed": 0,
                    "request_throughput_qps": 1.0,
                    "slo_attainment": 1.0,
                    "slo_compliant_qps": 1.0,
                    "e2e_ms": distribution,
                    "rag_ttft_ms": distribution,
                    "itl_ms": distribution,
                    "tpot_ms": distribution,
                    "ann_latency_ms": distribution,
                    "ann_queue_wait_ms": distribution,
                    "ann_gpu_latency_ms": distribution,
                    "ann_cpu_queue_wait_ms": distribution,
                    "ann_cpu_search_latency_ms": distribution,
                    "ann_hot_probe_ratio": distribution,
                },
                "server": {"rag_ann_runtime": []},
            }
        )

        self.assertEqual(metrics["ann_cpu_p99_ms"], 2.0)
        self.assertEqual(metrics["decode_step_p99_ms"], 0.0)
        self.assertEqual(metrics["partition_switches"], 0.0)

    def test_profiler_corrects_legacy_poisson_trace_stationarity(self):
        stationarity = normalized_result_stationarity(
            {
                "workload": {"arrival_process": "poisson"},
                "summary": {
                    "request_throughput_qps": 0.93,
                    "stationarity": {
                        "available": True,
                        "passed": False,
                        "failures": ["achieved_offered_ratio"],
                        "observed_trace_rate_qps": 0.94,
                        "achieved_offered_ratio": 0.93,
                        "thresholds": {
                            "min_achieved_offered_ratio": 0.95,
                        },
                    },
                },
            }
        )

        self.assertTrue(stationarity["passed"])
        self.assertEqual(stationarity["rate_normalization"], "observed_trace")
        self.assertGreater(stationarity["achieved_offered_ratio"], 0.98)

    def test_result_metrics_backfills_e2e_slo_from_legacy_requests(self):
        distribution = {"mean": 1.0, "p99": 2.0}
        result = {
            "summary": {
                "completed": 2,
                "failed": 0,
                "duration_s": 2.0,
                "request_throughput_qps": 1.0,
                "slo_attainment": 1.0,
                "slo_compliant_qps": 1.0,
                "e2e_ms": distribution,
                "rag_ttft_ms": distribution,
                "itl_ms": distribution,
                "tpot_ms": distribution,
                "ann_latency_ms": distribution,
                "ann_queue_wait_ms": distribution,
                "ann_gpu_latency_ms": distribution,
                "ann_cpu_queue_wait_ms": distribution,
                "ann_cpu_search_latency_ms": distribution,
                "ann_hot_probe_ratio": distribution,
            },
            "requests": [
                {"success": True, "e2e_ms": 7000},
                {"success": True, "e2e_ms": 9000},
            ],
            "server": {"rag_ann_runtime": []},
        }

        metrics = result_metrics(result, e2e_slo_ms=8000)

        self.assertEqual(metrics["e2e_slo_attainment"], 0.5)
        self.assertEqual(metrics["e2e_slo_goodput"], 0.5)

    def test_paired_policy_comparisons_include_hedrarag(self):
        def result(e2e_p99, ann_p99, goodput):
            distribution = {"mean": 1.0, "p99": 1.0}
            return {
                "workload": {"sha256": "shared-workload"},
                "summary": {
                    "completed": 1,
                    "failed": 0,
                    "request_throughput_qps": goodput,
                    "slo_attainment": 1.0,
                    "slo_compliant_qps": goodput,
                    "e2e_slo_attainment": 1.0,
                    "e2e_slo_compliant_qps": goodput,
                    "e2e_ms": {"mean": e2e_p99, "p99": e2e_p99},
                    "rag_ttft_ms": distribution,
                    "itl_ms": distribution,
                    "tpot_ms": distribution,
                    "ann_latency_ms": {
                        "mean": ann_p99,
                        "p99": ann_p99,
                    },
                    "ann_queue_wait_ms": distribution,
                    "ann_gpu_latency_ms": distribution,
                    "ann_cpu_queue_wait_ms": distribution,
                    "ann_cpu_search_latency_ms": distribution,
                    "ann_hot_probe_ratio": distribution,
                },
                "server": {"rag_ann_runtime": []},
            }

        full = replace(self.make_case(), policy="full", seed=7)
        hedrarag = replace(
            self.make_case(), policy="hedrarag", schedule="primary", seed=7
        )
        comparisons = paired_policy_comparisons(
            [full, hedrarag],
            {
                full.name: result(
                    e2e_p99=4.0, ann_p99=1.0, goodput=2.0
                ),
                hedrarag.name: result(
                    e2e_p99=8.0, ann_p99=2.0, goodput=1.0
                ),
            },
            e2e_slo_ms=10.0,
        )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["comparison"], "full_vs_hedrarag")
        metrics = comparisons[0]["metrics"]
        self.assertEqual(
            metrics["e2e_slo_goodput_gain_pct"]["mean"], 100.0
        )
        self.assertEqual(metrics["e2e_p99_reduction_pct"]["mean"], 50.0)
        self.assertEqual(metrics["ann_p99_change_pct"]["mean"], -50.0)

        naive_hybrid = replace(
            self.make_case(),
            policy="naive-hybrid",
            schedule="primary",
            seed=7,
        )
        raglease_hybrid = replace(
            self.make_case(), policy="raglease-hybrid", seed=7
        )
        comparisons = paired_policy_comparisons(
            [naive_hybrid, raglease_hybrid],
            {
                naive_hybrid.name: result(
                    e2e_p99=8.0, ann_p99=2.0, goodput=1.0
                ),
                raglease_hybrid.name: result(
                    e2e_p99=4.0, ann_p99=1.0, goodput=2.0
                ),
            },
            e2e_slo_ms=10.0,
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(
            comparisons[0]["comparison"],
            "raglease-hybrid_vs_naive-hybrid",
        )

        vectorliterag = replace(
            self.make_case(),
            policy="vectorliterag",
            schedule="primary",
            seed=7,
        )
        comparisons = paired_policy_comparisons(
            [raglease_hybrid, vectorliterag, hedrarag],
            {
                raglease_hybrid.name: result(
                    e2e_p99=4.0, ann_p99=1.0, goodput=2.0
                ),
                vectorliterag.name: result(
                    e2e_p99=8.0, ann_p99=2.0, goodput=1.0
                ),
                hedrarag.name: result(
                    e2e_p99=8.0, ann_p99=2.0, goodput=1.0
                ),
            },
            e2e_slo_ms=10.0,
        )
        self.assertEqual(
            {row["comparison"] for row in comparisons},
            {
                "raglease-hybrid_vs_vectorliterag",
                "raglease-hybrid_vs_hedrarag",
            },
        )
        for comparison in comparisons:
            metrics = comparison["metrics"]
            self.assertEqual(
                metrics["e2e_slo_goodput_gain_pct"]["mean"], 100.0
            )
            self.assertEqual(
                metrics["e2e_p99_reduction_pct"]["mean"], 50.0
            )

    def test_case_retry_keeps_final_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, case_max_attempts=2, case_retry_delay_s=0
            )
            case = self.make_case()
            result_path = profiler.result_path(case)

            def fail_case(*_args):
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text("failed", encoding="utf-8")
                raise RuntimeError("persistent")

            profiler.run_case = Mock(side_effect=fail_case)
            with self.assertRaisesRegex(RuntimeError, "persistent"):
                profiler.run_case_with_retries(case)

            self.assertEqual(profiler.run_case.call_count, 2)
            self.assertTrue(result_path.exists())

    def test_benchmark_command_omits_ann_priming_by_default(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)

            profiler.selection["e2e_slo_ms"] = 8000
            command = profiler.build_benchmark_command(self.make_case())

            self.assertEqual(
                command[command.index("--e2e-slo-ms") + 1],
                "8000",
            )
            self.assertEqual(command[command.index("--timeout-s") + 1], "300")
            self.assertEqual(
                command[command.index("--stationarity-windows") + 1], "8"
            )
            self.assertEqual(
                command[
                    command.index(
                        "--stationarity-min-achieved-offered-ratio"
                    )
                    + 1
                ],
                "0.95",
            )
            self.assertNotIn("--ann-prime-query-vec-path", command)
            self.assertNotIn("--ann-prime-batches-per-size", command)
            self.assertNotIn(
                "--ann-prime-validate-batch-sizes", command
            )
            self.assertNotIn(
                "--ann-prime-load-ramp-requests", command
            )

    def test_benchmark_command_propagates_request_timeout(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, benchmark_request_timeout_s=45
            )

            command = profiler.build_benchmark_command(self.make_case())

            self.assertEqual(command[command.index("--timeout-s") + 1], "45")

    def test_benchmark_command_can_disable_http_keepalive(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, benchmark_disable_http_keepalive=True
            )

            command = profiler.build_benchmark_command(self.make_case())

            self.assertIn("--disable-http-keepalive", command)

    def test_benchmark_command_propagates_ann_priming(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                ann_prime_batch_sizes=[1, 2],
                ann_prime_validate_batch_sizes=[1, 2],
                ann_prime_batches_per_size=32,
                ann_prime_min_samples=16,
                ann_prime_require_credit_straddle=True,
                ann_prime_load_ramp_requests=64,
                ann_prime_load_ramp_rps=100,
            )

            command = profiler.build_benchmark_command(self.make_case())

            self.assertEqual(
                command[
                    command.index("--ann-prime-query-vec-path") + 1
                ],
                "/tmp/queries.fbin",
            )
            self.assertEqual(
                command[command.index("--ann-prime-batch-sizes") + 1],
                "1,2",
            )
            self.assertEqual(
                command[
                    command.index("--ann-prime-validate-batch-sizes") + 1
                ],
                "1,2",
            )
            self.assertEqual(
                command[
                    command.index("--ann-prime-batches-per-size") + 1
                ],
                "32",
            )
            self.assertEqual(
                command[command.index("--ann-prime-min-samples") + 1],
                "16",
            )
            self.assertIn(
                "--ann-prime-require-credit-straddle", command
            )
            self.assertEqual(
                command[
                    command.index("--ann-prime-load-ramp-requests") + 1
                ],
                "64",
            )
            self.assertEqual(
                command[
                    command.index("--ann-prime-load-ramp-rps") + 1
                ],
                "100.0",
            )

    def test_benchmark_preserves_lease_policy_metadata(self):
        expected_fields = {
            "rag_partition_safe_mixed_hold",
            "rag_ann_launch_site",
            "rag_ann_fast_path_pressure_threshold",
            "rag_ann_fast_path_max_work",
            "rag_ann_fast_path_decode_batch_p95_threshold",
            "rag_prefill_low_load_batch_p95_threshold",
            "rag_prefill_low_load_arrival_rps_threshold",
            "rag_ann_fast_path_decode_queue_p95_ms",
            "rag_ann_decode_batch_bucket_size",
            "rag_ann_work_conserving_decode_overlap",
            "rag_ann_pre_switch_launch",
            "rag_ann_max_batches_per_tick",
            "rag_ann_max_queued_gpu_ms",
            "rag_ann_max_microbatch_gpu_ms",
            "rag_ann_service_time_percentile",
            "rag_ann_service_time_min_samples",
            "rag_ann_urgency_start_ratio",
            "rag_ann_max_decode_slowdown_ratio",
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
            "rag_lease_ann_service_estimate_ms",
            "rag_lease_prefill_service_estimate_ms",
            "enable_rag_ann",
            "rag_cpu_retrieval",
            "rag_cpu_retrieval_workers",
            "rag_cpu_retrieval_threads",
            "rag_cpu_retrieval_max_batch_queries",
        }

        self.assertTrue(expected_fields.issubset(SERVER_INFO_FIELDS))

    def test_build_cases_deduplicates_primary_baseline(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "sweep": {
                "include_primary_baseline": True,
                "request_rates": [4, 6],
                "decode_slo_ms": [20, 24],
                "max_defer_ms": [50, 150],
                "ann_lease_sm": [48],
                "prefill_lease_sm": [60],
                "repetitions": 3,
            },
        }
        cases = build_cases(config)
        primary = [case for case in cases if case.schedule == "primary"]
        decode = [case for case in cases if case.schedule == "decode-gc"]

        self.assertEqual(len(primary), 6)
        self.assertEqual(len(decode), 24)
        self.assertEqual(len({case.name for case in cases}), 30)

    def test_build_server_command_disables_slo_floor_for_ablation_policies(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, partition_slo_floor_enabled=True
            )
            full_case = self.make_case()
            self.assertNotIn(
                "--disable-rag-partition-slo-floor",
                profiler.build_server_command(full_case),
            )

            for policy in ("naive", "dynamic-only"):
                command = profiler.build_server_command(
                    replace(full_case, policy=policy)
                )
                self.assertIn("--disable-rag-partition-slo-floor", command)

    def test_build_server_command_scopes_work_conserving_overlap_to_full_decode_gc(
        self,
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, ann_work_conserving_decode_overlap=True
            )
            full_decode = replace(self.make_case(), schedule="decode-gc")
            flag = "--rag-ann-work-conserving-decode-overlap"

            self.assertIn(flag, profiler.build_server_command(full_decode))
            self.assertNotIn(flag, profiler.build_server_command(self.make_case()))
            for policy in ("naive", "dynamic-only"):
                self.assertNotIn(
                    flag,
                    profiler.build_server_command(
                        replace(full_decode, policy=policy)
                    ),
                )

    def test_build_server_command_scopes_pre_switch_launch_to_full_decode_gc(
        self,
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, ann_pre_switch_launch=True
            )
            full_decode = replace(self.make_case(), schedule="decode-gc")
            flag = "--rag-ann-pre-switch-launch"

            self.assertIn(flag, profiler.build_server_command(full_decode))
            self.assertNotIn(
                flag, profiler.build_server_command(self.make_case())
            )
            for policy in ("naive", "dynamic-only"):
                self.assertNotIn(
                    flag,
                    profiler.build_server_command(
                        replace(full_decode, policy=policy)
                    ),
                )

    def test_build_server_command_passes_low_load_arrival_threshold(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                prefill_low_load_arrival_rps_threshold=3,
            )
            command = profiler.build_server_command(self.make_case())
            option_idx = command.index(
                "--rag-prefill-low-load-arrival-rps-threshold"
            )

        self.assertEqual(command[option_idx + 1], "3")

    def test_build_server_command_scopes_lease_profile_to_raglease(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                lease_profile_path="/tmp/lease-profile.yaml",
                lease_profile_request_rate=6.25,
            )
            raglease = profiler.build_server_command(self.make_case())
            naive = profiler.build_server_command(
                replace(self.make_case(), policy="naive")
            )

        path_option = "--rag-lease-profile-path"
        rate_option = "--rag-lease-profile-request-rate"
        self.assertEqual(
            raglease[raglease.index(path_option) + 1],
            "/tmp/lease-profile.yaml",
        )
        self.assertEqual(raglease[raglease.index(rate_option) + 1], "6.25")
        self.assertNotIn(path_option, naive)
        self.assertNotIn(rate_option, naive)

    def test_build_server_command_can_disable_radix_cache(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                doc_token_cache_size=0,
                disable_radix_cache=True,
            )
            command = profiler.build_server_command(self.make_case())

        self.assertIn("--disable-radix-cache", command)
        option_idx = command.index("--rag-doc-token-cache-size")
        self.assertEqual(command[option_idx + 1], "0")

    def test_build_server_command_uses_wide_global_ann_window_without_decode_cap(
        self,
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)
            command = profiler.build_server_command(
                replace(self.make_case(), schedule="decode-gc")
            )
            global_option_idx = command.index("--rag-ann-max-inflight")
            tick_option_idx = command.index(
                "--rag-ann-max-batches-per-tick"
            )
            option_idx = command.index("--rag-ann-max-inflight-during-decode")
            credit_idx = command.index("--rag-ann-max-queued-gpu-ms")
            batch_credit_idx = command.index("--rag-ann-max-microbatch-gpu-ms")
            percentile_idx = command.index("--rag-ann-service-time-percentile")
            min_samples_idx = command.index(
                "--rag-ann-service-time-min-samples"
            )

            self.assertEqual(command[global_option_idx + 1], "8")
            self.assertEqual(command[tick_option_idx + 1], "0")
            self.assertEqual(command[option_idx + 1], "0")
            self.assertEqual(command[credit_idx + 1], "30")
            self.assertEqual(command[batch_credit_idx + 1], "20")
            self.assertEqual(command[percentile_idx + 1], "95")
            self.assertEqual(command[min_samples_idx + 1], "4")

    def test_validate_server_info_rejects_stale_configuration(self):
        case = ProfileCase(
            schedule="frontend-gc",
            request_rate=5.75,
            decode_slo_ms=24,
            max_defer_ms=150,
            ann_lease_sm=48,
            prefill_lease_sm=60,
            pdmux_config_path="/tmp/t24.yaml",
            repetition=1,
        )
        info = {
            "enable_pdmux": True,
            "rag_ann_schedule": "frontend-gc",
            "rag_ann_launch_site": "after-prefill",
            "rag_policy_mode": "full",
            "rag_ann_decode_step_slo_ms": 24,
            "rag_ann_max_defer_ms": 150,
            "rag_ann_frontend_arbitration": "fair",
            "rag_ann_lease_frontend_sm": 48,
            "rag_prefill_lease_frontend_sm": 60,
            "pdmux_config_path": "/tmp/t24.yaml",
        }
        OfflineProfiler.validate_server_info(case, info)

        info["rag_ann_schedule"] = "primary"
        with self.assertRaisesRegex(RuntimeError, "Schedule mismatch"):
            OfflineProfiler.validate_server_info(case, info)


    def test_validate_server_info_rejects_stale_batching_configuration(self):
        case = ProfileCase(
            schedule="frontend-gc",
            request_rate=2,
            decode_slo_ms=20,
            max_defer_ms=50,
            ann_lease_sm=56,
            prefill_lease_sm=72,
            pdmux_config_path="/tmp/t24.yaml",
            repetition=1,
        )
        runtime = {
            "ann_max_inflight": 2,
            "ann_max_batches_per_tick": 0,
            "ann_max_inflight_during_decode": 1,
            "ann_max_batch_size": 8,
            "ann_max_queued_gpu_ms": 30,
            "ann_max_microbatch_gpu_ms": 20,
            "ann_service_time_percentile": 95,
            "ann_service_time_min_samples": 4,
            "ann_max_batch_wait_ms": 2,
            "ann_decode_only_wait_ms": 10,
            "ann_stream_epoch_ms": 0,
            "ann_launch_site": "adaptive",
            "ann_fast_path_pressure_threshold": 0.5,
            "ann_fast_path_max_work": 1,
            "partition_hysteresis_bs": 4,
            "partition_min_residency_ms": 2,
            "partition_safe_mixed_hold": True,
            "partition_safe_mixed_hold_ratio": 0.9,
            "partition_safe_mixed_min_samples": 8,
            "partition_decode_slo_ms": [38, 35, 32],
            "partition_decode_slo_batch_thresholds": [11, 12.5],
            "partition_decode_slo_hysteresis_bs": 0.5,
            "partition_decode_slo_min_residency_ms": 5000,
            "partition_decode_slo_initial_calibration_ms": 10000,
            "partition_decode_slo_ewma_alpha": 0.01,
            "partition_slo_floor_enabled": False,
            "lease_coalesce_min_work": 1,
            "lease_coalesce_max_wait_ms": 0,
            "lease_switch_cost_ms": 0,
            "lease_ann_service_estimate_ms": 10,
            "lease_prefill_service_estimate_ms": 10,
        }
        info = {
            "enable_pdmux": True,
            "rag_ann_schedule": "frontend-gc",
            "rag_ann_launch_site": "adaptive",
            "rag_policy_mode": "full",
            "rag_ann_decode_step_slo_ms": 20,
            "rag_ann_max_defer_ms": 50,
            "rag_ann_frontend_arbitration": "fair",
            "rag_ann_lease_frontend_sm": 56,
            "rag_prefill_lease_frontend_sm": 72,
            "rag_ann_handoff_timeout_ms": 100,
            "pdmux_config_path": "/tmp/t24.yaml",
            "rag_ann_max_inflight": 2,
            "rag_ann_max_batches_per_tick": 0,
            "rag_ann_max_inflight_during_decode": 1,
            "rag_ann_max_batch_size": 8,
            "rag_ann_max_queued_gpu_ms": 30,
            "rag_ann_max_microbatch_gpu_ms": 20,
            "rag_ann_service_time_percentile": 95,
            "rag_ann_service_time_min_samples": 4,
            "rag_prefill_decode_guard_percentile": 99,
            "rag_ann_max_batch_wait_ms": 2,
            "rag_ann_decode_only_wait_ms": 10,
            "rag_ann_stream_epoch_ms": 0,
            "rag_ann_fast_path_pressure_threshold": 0.5,
            "rag_ann_fast_path_max_work": 1,
            "rag_ann_fast_path_decode_batch_p95_threshold": 16,
            "rag_prefill_low_load_batch_p95_threshold": 4,
            "rag_prefill_low_load_arrival_rps_threshold": -1,
            "rag_prefill_low_load_reference_tokens": 1024,
            "rag_ann_fast_path_decode_queue_p95_ms": 2,
            "rag_ann_decode_batch_bucket_size": 4,
            "rag_ann_urgency_start_ratio": 1.0,
            "rag_ann_max_decode_slowdown_ratio": 1.0,
            "rag_partition_hysteresis_bs": 4,
            "rag_partition_min_residency_ms": 2,
            "rag_partition_safe_mixed_hold": True,
            "rag_partition_safe_mixed_hold_ratio": 0.9,
            "rag_partition_safe_mixed_min_samples": 8,
            "rag_partition_safe_mixed_max_hold_ms": 5,
            "rag_partition_decode_slo_ms": [38, 35, 32],
            "rag_partition_decode_slo_batch_thresholds": [11, 12.5],
            "rag_partition_decode_slo_hysteresis_bs": 0.5,
            "rag_partition_decode_slo_min_residency_ms": 5000,
            "rag_partition_decode_slo_initial_calibration_ms": 10000,
            "rag_partition_decode_slo_ewma_alpha": 0.01,
            "rag_partition_slo_floor_enabled": False,
            "rag_lease_coalesce_min_work": 1,
            "rag_lease_coalesce_max_wait_ms": 0,
            "rag_lease_switch_cost_ms": 0,
            "rag_lease_switch_cost_percentile": 95,
            "rag_e2e_slo_ms": -1,
            "rag_lease_round_trip_switch_cost": True,
            "rag_lease_min_gain_ratio": 1,
            "rag_lease_ann_service_estimate_ms": 10,
            "rag_lease_prefill_service_estimate_ms": 10,
            "rag_prefill_dynamic_lease": True,
            "rag_prefill_tail_aware_admission": True,
            "rag_prefill_max_consecutive_chunks": 2,
            "rag_prefill_lease_min_layers": 1,
            "rag_prefill_lease_max_layers": 8,
            "rag_prefill_lease_decode_ratio": 1,
        }
        OfflineProfiler.validate_server_info(case, info, runtime)

        runtime["frontend_arbitration"] = "overlap"
        info["rag_ann_frontend_arbitration"] = "overlap"
        OfflineProfiler.validate_server_info(case, info, runtime)

        info["rag_ann_frontend_arbitration"] = "fair"
        with self.assertRaisesRegex(RuntimeError, "Arbitration mismatch"):
            OfflineProfiler.validate_server_info(case, info, runtime)
        info["rag_ann_frontend_arbitration"] = "overlap"

        bundle_case = replace(
            case,
            prefill_lease_min_layers=4,
            prefill_lease_max_layers=4,
        )
        info["rag_prefill_lease_min_layers"] = 4
        info["rag_prefill_lease_max_layers"] = 4
        OfflineProfiler.validate_server_info(bundle_case, info, runtime)

        info["rag_prefill_lease_min_layers"] = 1
        info["rag_prefill_lease_max_layers"] = 8
        info["rag_ann_max_batch_size"] = 1
        with self.assertRaisesRegex(RuntimeError, "rag_ann_max_batch_size mismatch"):
            OfflineProfiler.validate_server_info(case, info, runtime)

    def test_validate_server_info_accepts_dynamic_only_normalization(self):
        case = ProfileCase(
            schedule="frontend-gc",
            request_rate=6,
            decode_slo_ms=-1,
            max_defer_ms=0,
            ann_lease_sm=-1,
            prefill_lease_sm=-1,
            pdmux_config_path="/tmp/t24.yaml",
            repetition=1,
            policy="dynamic-only",
        )
        runtime = {
            "ann_max_inflight": 2,
            "ann_max_batches_per_tick": 0,
            "ann_max_inflight_during_decode": 1,
            "ann_max_batch_size": 8,
            "ann_max_queued_gpu_ms": 30,
            "ann_max_microbatch_gpu_ms": 20,
            "ann_service_time_percentile": 95,
            "ann_service_time_min_samples": 4,
            "ann_max_batch_wait_ms": 2,
            "ann_decode_only_wait_ms": 10,
            "ann_stream_epoch_ms": 0,
            "ann_launch_site": "after-prefill",
            "ann_fast_path_pressure_threshold": 0.5,
            "ann_fast_path_max_work": 1,
            "partition_hysteresis_bs": 4,
            "partition_min_residency_ms": 2,
            "partition_safe_mixed_hold": True,
            "partition_safe_mixed_hold_ratio": 0.9,
            "partition_safe_mixed_min_samples": 8,
            "lease_coalesce_min_work": 1,
            "lease_coalesce_max_wait_ms": 0,
            "lease_switch_cost_ms": 0,
            "lease_ann_service_estimate_ms": 10,
            "lease_prefill_service_estimate_ms": 10,
        }
        info = {
            "enable_pdmux": True,
            "rag_ann_schedule": "frontend-gc",
            "rag_ann_launch_site": "after-prefill",
            "rag_policy_mode": "dynamic-only",
            "rag_ann_decode_step_slo_ms": -1,
            "rag_ann_max_defer_ms": 0,
            "rag_ann_frontend_arbitration": "fair",
            "rag_ann_lease_frontend_sm": -1,
            "rag_prefill_lease_frontend_sm": -1,
            "rag_ann_handoff_timeout_ms": 0,
            "pdmux_config_path": "/tmp/t24.yaml",
            "rag_ann_max_inflight": 2,
            "rag_ann_max_batches_per_tick": 0,
            "rag_ann_max_inflight_during_decode": 1,
            "rag_ann_max_batch_size": 8,
            "rag_ann_max_queued_gpu_ms": 30,
            "rag_ann_max_microbatch_gpu_ms": 20,
            "rag_ann_service_time_percentile": 95,
            "rag_ann_service_time_min_samples": 4,
            "rag_ann_max_batch_wait_ms": 2,
            "rag_ann_decode_only_wait_ms": 10,
            "rag_ann_stream_epoch_ms": 0,
            "rag_ann_fast_path_pressure_threshold": 0.5,
            "rag_ann_fast_path_max_work": 1,
            "rag_ann_fast_path_decode_batch_p95_threshold": 16,
            "rag_prefill_low_load_batch_p95_threshold": 4,
            "rag_prefill_low_load_arrival_rps_threshold": -1,
            "rag_prefill_low_load_reference_tokens": 1024,
            "rag_ann_fast_path_decode_queue_p95_ms": 2,
            "rag_ann_decode_batch_bucket_size": 4,
            "rag_ann_urgency_start_ratio": 1.0,
            "rag_ann_max_decode_slowdown_ratio": 1.0,
            "rag_partition_hysteresis_bs": 0,
            "rag_partition_min_residency_ms": 0,
            "rag_partition_safe_mixed_hold": False,
            "rag_partition_safe_mixed_hold_ratio": 0.9,
            "rag_partition_safe_mixed_min_samples": 8,
            "rag_partition_safe_mixed_max_hold_ms": 5,
            "rag_partition_decode_slo_hysteresis_bs": 0,
            "rag_partition_decode_slo_min_residency_ms": 0,
            "rag_partition_decode_slo_initial_calibration_ms": 0,
            "rag_partition_decode_slo_ewma_alpha": 0.02,
            "rag_lease_coalesce_min_work": 1,
            "rag_lease_coalesce_max_wait_ms": 0,
            "rag_lease_switch_cost_ms": 0,
            "rag_lease_switch_cost_percentile": 95,
            "rag_e2e_slo_ms": -1,
            "rag_lease_round_trip_switch_cost": True,
            "rag_lease_min_gain_ratio": 1,
            "rag_lease_ann_service_estimate_ms": 10,
            "rag_lease_prefill_service_estimate_ms": 10,
            "rag_prefill_dynamic_lease": False,
            "rag_prefill_max_consecutive_chunks": 2,
            "rag_prefill_lease_min_layers": 1,
            "rag_prefill_lease_max_layers": 8,
            "rag_prefill_lease_decode_ratio": 1,
        }
        OfflineProfiler.validate_server_info(case, info, runtime)

    def test_build_cases_pairs_three_policies_by_seed(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["naive", "dynamic-only", "full"],
                "request_rates": [4, 6],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [56],
                "prefill_lease_sm": [72],
                "seeds": [7, 11],
            },
        }
        cases = build_cases(config)

        self.assertEqual(len(cases), 12)
        for request_rate in (4, 6):
            for seed in (7, 11):
                paired = [
                    case
                    for case in cases
                    if case.request_rate == request_rate and case.seed == seed
                ]
                self.assertEqual(
                    [case.policy for case in paired],
                    ["naive", "dynamic-only", "full"],
                )
                self.assertEqual(paired[0].schedule, "primary")
                self.assertEqual(paired[1].policy_mode, "dynamic-only")
                self.assertEqual(paired[1].ann_lease_sm, -1)
                self.assertEqual(paired[2].ann_lease_sm, 56)

    def test_build_cases_pairs_controlled_baselines_with_full(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "runtime": {"ann_schedule": "decode-gc"},
            "sweep": {
                "paired_policies": [
                "naive",
                "vectorliterag",
                "hedrarag",
                "hedrarag-hybrid",
                "full",
                ],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(
            [(case.policy, case.schedule, case.uses_pdmux) for case in cases],
            [
                ("naive", "primary", False),
                ("vectorliterag", "primary", False),
                ("hedrarag", "primary", False),
                ("hedrarag-hybrid", "primary", False),
                ("full", "decode-gc", True),
            ],
        )

    def test_build_cases_pairs_identical_hybrid_backend_with_distinct_schedulers(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "runtime": {"ann_schedule": "decode-gc"},
            "sweep": {
                "paired_policies": ["naive-hybrid", "raglease-hybrid"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "seeds": [7],
            },
        }

        naive, raglease = build_cases(config)

        self.assertEqual(
            (naive.policy, naive.schedule, naive.uses_pdmux),
            ("naive-hybrid", "primary", False),
        )
        self.assertEqual(
            (raglease.policy, raglease.schedule, raglease.uses_pdmux),
            ("raglease-hybrid", "decode-gc", True),
        )
        self.assertTrue(naive.uses_hybrid_index)
        self.assertTrue(raglease.uses_hybrid_index)
        self.assertEqual(
            naive.retrieval_backend, raglease.retrieval_backend
        )
        self.assertEqual(naive.decode_slo_ms, -1)
        self.assertEqual(naive.max_defer_ms, 0)
        self.assertEqual(naive.ann_lease_sm, -1)
        self.assertEqual(naive.prefill_lease_sm, -1)
        self.assertEqual(raglease.decode_slo_ms, 20)
        self.assertEqual(raglease.max_defer_ms, 50)
        self.assertEqual(raglease.ann_lease_sm, -1)
        self.assertEqual(raglease.prefill_lease_sm, 72)

    def test_hybrid_pair_rejects_policy_specific_shared_hbm_by_default(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {
                "ann_schedule": "decode-gc",
                "shared_hbm_mode": "ivf",
                "raglease_hybrid_shared_hbm_enabled": True,
            },
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["naive-hybrid", "raglease-hybrid"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "seeds": [7],
            },
        }

        with self.assertRaisesRegex(ValueError, "changes shared-HBM mode"):
            build_cases(config)

        config["comparison_contract"] = {
            "allow_policy_specific_shared_hbm": True
        }
        self.assertEqual(len(build_cases(config)), 2)

    def test_query_permutation_must_match_runtime_nprobe(self):
        config = {
            "paths": {
                "pdmux_config_path": "/tmp/t24.yaml",
                "query_permutation_path": (
                    "/tmp/hot_to_cold_nprobe2048_seed27.npy"
                ),
            },
            "runtime": {"nprobe": 4096},
            "sweep": {},
        }

        with self.assertRaisesRegex(ValueError, "nprobe mismatch"):
            validate_experiment_contract(config)

    def test_real_document_contract_rejects_random_fallback(self):
        config = {
            "paths": {"doc_store_path": "/tmp/documents.jsonl"},
            "runtime": {"doc_random_fallback": True},
            "sweep": {},
            "comparison_contract": {"require_real_documents": True},
        }

        with self.assertRaisesRegex(ValueError, "forbids"):
            validate_experiment_contract(config)

    def test_real_document_contract_requires_document_path(self):
        config = {
            "paths": {},
            "runtime": {"doc_random_fallback": False},
            "sweep": {},
            "comparison_contract": {"require_real_documents": True},
        }

        with self.assertRaisesRegex(ValueError, "requires"):
            validate_experiment_contract(config)

    def test_real_document_contract_accepts_document_store(self):
        config = {
            "paths": {"doc_store_path": "/tmp/documents.jsonl"},
            "runtime": {"doc_random_fallback": False},
            "sweep": {},
            "comparison_contract": {"require_real_documents": True},
        }

        validate_experiment_contract(config)

    def test_hybrid_equivalence_audit_rejects_hot_ratio_drift(self):
        naive = replace(
            self.make_case(), policy="naive-hybrid", seed=7
        )
        raglease = replace(
            self.make_case(), policy="raglease-hybrid", seed=7
        )
        common_server = {
            "rag_index_nprobe": 2048,
            "rag_index_topk": 25,
            "rag_hot_index_path": "/tmp/hot.index",
            "rag_hot_index_map_path": "/tmp/hot.imap",
            "rag_hot_index_shard_id": 0,
            "rag_cpu_retrieval_threads": 1,
            "rag_doc_token_cache_size": 1024,
            "rag_shared_hbm_mode": "off",
        }

        def result(hot_ratio):
            return {
                "workload": {"sha256": "same"},
                "server": dict(common_server),
                "summary": {
                    "ann_hot_probe_ratio": {"mean": hot_ratio}
                },
            }

        with self.assertRaisesRegex(RuntimeError, "ann_hot_probe_ratio"):
            paired_hybrid_equivalence_audit(
                [naive, raglease],
                {
                    naive.name: result(0.26),
                    raglease.name: result(0.18),
                },
                {},
            )

    def test_single_raglease_policy_expands_lease_dimensions(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "runtime": {"ann_schedule": "decode-gc"},
            "sweep": {
                "paired_policies": ["raglease-hybrid"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [40, 32],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(len(cases), 2)
        self.assertEqual([case.policy for case in cases], ["raglease-hybrid"] * 2)
        self.assertEqual([case.prefill_lease_sm for case in cases], [40, 32])

    def test_multiple_paired_policies_reject_multi_value_lease_dimensions(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "runtime": {"ann_schedule": "decode-gc"},
            "sweep": {
                "paired_policies": ["naive-hybrid", "raglease-hybrid"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [40, 32],
                "seeds": [7],
            },
        }

        with self.assertRaisesRegex(ValueError, "exactly one RAGLease policy"):
            build_cases(config)

    def test_build_cases_repeats_cpu_retrieval_with_the_same_seed(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 1},
            "sweep": {
                "paired_policies": ["cpu-retrieval"],
                "request_rates": [5.75],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [48],
                "prefill_lease_sm": [60],
                "seeds": [1],
                "repetitions_per_seed": 2,
            },
        }

        cases = build_cases(config)

        self.assertEqual(len(cases), 2)
        self.assertEqual([case.policy for case in cases], ["cpu-retrieval"] * 2)
        self.assertEqual([case.schedule for case in cases], ["primary"] * 2)
        self.assertEqual([case.seed for case in cases], [1, 1])
        self.assertEqual([case.repetition for case in cases], [1, 2])
        self.assertEqual(len({case.name for case in cases}), 2)

    def test_cpu_retrieval_server_command_disables_gpu_ann(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                cpu_retrieval_workers=1,
                cpu_retrieval_threads=8,
                cpu_retrieval_max_batch_queries=12,
            )
            case = replace(
                self.make_case(),
                policy="cpu-retrieval",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            profiler.paths["cpu_index_path"] = "/tmp/ivffs.index"
            command = profiler.build_server_command(case)

            self.assertEqual(
                command[command.index("--rag-index-path") + 1],
                "/tmp/ivffs.index",
            )

            self.assertIn("--rag-cpu-retrieval", command)
            self.assertNotIn("--enable-rag-ann", command)
            self.assertNotIn("--enable-pdmux", command)
            self.assertNotIn("--pdmux-config-path", command)
            self.assertEqual(
                command[command.index("--rag-cpu-retrieval-workers") + 1],
                "1",
            )
            self.assertEqual(
                command[command.index("--rag-cpu-retrieval-threads") + 1],
                "8",
            )

            self.assertEqual(
                command[
                    command.index(
                        "--rag-cpu-retrieval-max-batch-queries"
                    )
                    + 1
                ],
                "12",
            )

    def test_hedrarag_controlled_uses_cpu_gpu_pipeline_overrides(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                cpu_retrieval_workers=8,
                cpu_retrieval_threads=1,
                cpu_retrieval_max_batch_queries=1,
                hedrarag_cpu_retrieval_workers=1,
                hedrarag_cpu_retrieval_threads=12,
                hedrarag_cpu_retrieval_max_batch_queries=8,
            )
            profiler.paths["vectorlite_cpu_index_path"] = "/tmp/ivffs.index"
            profiler.paths["hedrarag_cpu_index_path"] = "/tmp/hedra.index"
            case = replace(
                self.make_case(),
                policy="hedrarag",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            command = profiler.build_server_command(case)

            self.assertTrue(case.is_cpu_retrieval)
            self.assertTrue(case.is_hedrarag)
            self.assertFalse(case.uses_pdmux)
            self.assertEqual(
                command[command.index("--rag-index-path") + 1],
                "/tmp/hedra.index",
            )
            self.assertIn("--rag-cpu-retrieval", command)
            self.assertNotIn("--enable-rag-ann", command)
            self.assertNotIn("--enable-pdmux", command)
            self.assertEqual(
                command[command.index("--rag-cpu-retrieval-workers") + 1],
                "1",
            )
            self.assertEqual(
                command[command.index("--rag-cpu-retrieval-threads") + 1],
                "12",
            )
            self.assertEqual(
                command[
                    command.index("--rag-cpu-retrieval-max-batch-queries") + 1
                ],
                "8",
            )

    def test_llm_only_control_uses_matched_rag_path_without_retrieval(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                ann_prime_batches_per_size=32,
                mem_fraction_static=0.75,
            )
            case = replace(
                self.make_case(),
                policy="llm-only",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            server_command = profiler.build_server_command(case)
            benchmark_command = profiler.build_benchmark_command(case)

            self.assertTrue(case.is_llm_only)
            self.assertFalse(case.uses_pdmux)
            self.assertNotIn("--enable-rag-ann", server_command)
            self.assertNotIn("--rag-cpu-retrieval", server_command)
            self.assertNotIn("--enable-pdmux", server_command)
            self.assertIn("--skip-retrieval", benchmark_command)
            self.assertNotIn(
                "--ann-prime-batches-per-size", benchmark_command
            )

            OfflineProfiler.validate_server_info(
                case,
                {
                    "enable_pdmux": False,
                    "enable_rag_ann": False,
                    "rag_cpu_retrieval": False,
                    "rag_ann_schedule": "primary",
                    "rag_ann_launch_site": "early",
                    "rag_policy_mode": "full",
                    "rag_ann_decode_step_slo_ms": -1,
                    "rag_ann_max_defer_ms": 0,
                    "rag_ann_frontend_arbitration": "overlap",
                    "rag_ann_lease_frontend_sm": -1,
                    "rag_prefill_lease_frontend_sm": -1,
                    "rag_ann_handoff_timeout_ms": 0,
                    "mem_fraction_static": 0.75,
                },
                {"mem_fraction_static": 0.75},
            )
    def test_ann_only_control_uses_matched_rag_path_without_generation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                ann_prime_batches_per_size=32,
            )
            case = replace(
                self.make_case(),
                policy="ann-only",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            server_command = profiler.build_server_command(case)
            benchmark_command = profiler.build_benchmark_command(case)

            self.assertTrue(case.is_ann_only)
            self.assertFalse(case.uses_pdmux)
            self.assertIn("--enable-rag-ann", server_command)
            self.assertNotIn("--rag-cpu-retrieval", server_command)
            self.assertNotIn("--enable-pdmux", server_command)
            for option, expected in (
                ("--rag-ann-max-inflight", "32"),
                ("--rag-ann-max-queued-gpu-ms", "0.0"),
                ("--rag-ann-max-microbatch-gpu-ms", "0.0"),
                ("--rag-ann-max-batch-wait-ms", "0.0"),
                ("--rag-ann-batch-wait-min-pending", "1"),
                ("--rag-ann-decode-only-wait-ms", "0.0"),
            ):
                self.assertEqual(
                    server_command[server_command.index(option) + 1], expected
                )
            self.assertIn("--skip-generation", benchmark_command)
            self.assertNotIn("--skip-retrieval", benchmark_command)
            self.assertNotIn(
                "--ann-prime-batches-per-size", benchmark_command
            )

            OfflineProfiler.validate_server_info(
                case,
                {
                    "enable_pdmux": False,
                    "enable_rag_ann": True,
                    "rag_cpu_retrieval": False,
                    "rag_ann_schedule": "primary",
                    "rag_ann_launch_site": "early",
                    "rag_policy_mode": "full",
                    "rag_ann_decode_step_slo_ms": -1,
                    "rag_ann_max_defer_ms": 0,
                    "rag_ann_frontend_arbitration": "overlap",
                    "rag_ann_lease_frontend_sm": -1,
                    "rag_prefill_lease_frontend_sm": -1,
                    "rag_ann_handoff_timeout_ms": 0,
                    "rag_prefill_dynamic_lease": False,
                },
            )

    def test_validate_cpu_retrieval_server_info(self):
        case = replace(
            self.make_case(),
            policy="cpu-retrieval",
            schedule="primary",
            decode_slo_ms=-1,
            max_defer_ms=0,
            ann_lease_sm=-1,
            prefill_lease_sm=-1,
        )
        runtime = {
            "nprobe": 2048,
            "topk": 25,
            "doc_token_cache_size": 1024,
            "mem_fraction_static": 0.75,
            "cpu_retrieval_workers": 1,
            "cpu_retrieval_threads": 8,
            "cpu_retrieval_max_batch_queries": 12,
        }
        info = {
            "enable_pdmux": False,
            "enable_rag_ann": False,
            "rag_cpu_retrieval": True,
            "rag_cpu_retrieval_workers": 1,
            "rag_cpu_retrieval_threads": 8,
            "rag_cpu_retrieval_max_batch_queries": 12,
            "rag_ann_schedule": "primary",
            "rag_ann_launch_site": "early",
            "rag_policy_mode": "full",
            "rag_ann_decode_step_slo_ms": -1,
            "rag_ann_max_defer_ms": 0,
            "rag_ann_frontend_arbitration": "overlap",
            "rag_ann_lease_frontend_sm": -1,
            "rag_prefill_lease_frontend_sm": -1,
            "rag_ann_handoff_timeout_ms": 0,
            "rag_index_nprobe": 2048,
            "rag_index_topk": 25,
            "rag_doc_token_cache_size": 1024,
            "mem_fraction_static": 0.75,
            "pdmux_config_path": "/tmp/t24.yaml",
        }

        OfflineProfiler.validate_server_info(case, info, runtime)

        info["enable_rag_ann"] = True
        with self.assertRaisesRegex(RuntimeError, "GPU ANN executor"):
            OfflineProfiler.validate_server_info(case, info, runtime)

    def test_naive_server_command_uses_unpartitioned_gpu(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)
            case = replace(
                self.make_case(),
                policy="naive",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            command = profiler.build_server_command(case)

            self.assertNotIn("--enable-pdmux", command)
            self.assertNotIn("--pdmux-config-path", command)
            for option, expected in (
                ("--rag-ann-max-inflight", "32"),
                ("--rag-ann-max-queued-gpu-ms", "0.0"),
                ("--rag-ann-max-microbatch-gpu-ms", "0.0"),
                ("--rag-ann-max-batch-wait-ms", "0.0"),
                ("--rag-ann-batch-wait-min-pending", "1"),
                ("--rag-ann-decode-only-wait-ms", "0.0"),
            ):
                self.assertEqual(command[command.index(option) + 1], expected)
            self.assertEqual(
                command[command.index("--rag-ann-schedule") + 1],
                "primary",
            )
            self.assertEqual(
                command[command.index("--rag-ann-launch-site") + 1],
                "early",
            )
            self.assertEqual(
                command[
                    command.index("--rag-ann-frontend-arbitration") + 1
                ],
                "overlap",
            )
            self.assertEqual(
                command[command.index("--rag-ann-decode-step-slo-ms") + 1],
                "-1",
            )
            self.assertEqual(
                command[command.index("--rag-ann-max-defer-ms") + 1],
                "0",
            )
            self.assertEqual(
                command[command.index("--rag-ann-handoff-timeout-ms") + 1],
                "0.0",
            )

    def test_server_command_forwards_query_permutation_path(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                shared_hbm_cluster_reload_min_misses=128,
                shared_hbm_cluster_reload_max_lists=32,
            )
            profiler.paths["query_permutation_path"] = "/tmp/shift.npy"

            command = profiler.build_server_command(self.make_case())

            self.assertEqual(
                command[
                    command.index("--rag-query-permutation-path") + 1
                ],
                "/tmp/shift.npy",
            )

            self.assertEqual(
                command[
                    command.index("--rag-shared-hbm-cluster-reload-min-misses") + 1
                ],
                "128",
            )
            self.assertEqual(
                command[
                    command.index("--rag-shared-hbm-cluster-reload-max-lists") + 1
                ],
                "32",
            )
            self.assertEqual(
                command[
                    command.index("--rag-shared-hbm-kv-reserve-extents") + 1
                ],
                "2",
            )

    def test_vectorliterag_server_command_uses_only_static_hot_index(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir, shared_hbm_mode="ivf")
            profiler.paths["vectorlite_cpu_index_path"] = "/tmp/ivffs.index"
            profiler.paths["vectorlite_hot_index_path"] = "/tmp/hot.index"
            profiler.paths["vectorlite_hot_index_map_path"] = "/tmp/hot.imap"
            case = replace(
                self.make_case(),
                policy="vectorliterag",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            command = profiler.build_server_command(case)

            self.assertNotIn("--enable-pdmux", command)
            self.assertEqual(
                command[command.index("--rag-index-path") + 1],
                "/tmp/ivffs.index",
            )
            self.assertEqual(
                command[command.index("--rag-hot-index-path") + 1],
                "/tmp/hot.index",
            )
            self.assertEqual(
                command[command.index("--rag-hot-index-map-path") + 1],
                "/tmp/hot.imap",
            )
            self.assertEqual(
                command[command.index("--rag-shared-hbm-mode") + 1], "off"
            )
            self.assertIn("--disable-rag-prefill-dynamic-lease", command)
            self.assertIn("--disable-rag-lease-round-trip-switch-cost", command)
            self.assertEqual(
                command[command.index("--rag-ann-max-inflight") + 1], "1"
            )
            self.assertEqual(
                command[command.index("--rag-ann-max-batches-per-tick") + 1],
                "1",
            )
            self.assertEqual(
                command[command.index("--rag-ann-max-batch-size") + 1], "64"
            )

    def test_hedrarag_hybrid_uses_matched_hot_cold_index_and_batch_cap(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                shared_hbm_mode="ivf",
                hybrid_cpu_search_threads=6,
                hedrarag_hybrid_cpu_search_threads=8,
                hybrid_cpu_search_max_batch_queries=4,
                hedrarag_hybrid_cpu_search_max_batch_queries=3,
                hedrarag_hybrid_ann_max_inflight=1,
                hedrarag_hybrid_ann_max_batches_per_tick=1,
                hedrarag_hybrid_ann_max_batch_size=8,
            )
            profiler.paths["hybrid_cpu_index_path"] = "/tmp/ivf.index"
            profiler.paths["hybrid_hot_index_path"] = "/tmp/hot.index"
            profiler.paths["hybrid_hot_index_map_path"] = "/tmp/hot.imap"
            case = replace(
                self.make_case(),
                policy="hedrarag-hybrid",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )

            command = profiler.build_server_command(case)

            def argument(option):
                return command[command.index(option) + 1]

            self.assertTrue(case.is_hedrarag_hybrid)
            self.assertTrue(case.uses_hybrid_index)
            self.assertFalse(case.is_cpu_retrieval)
            self.assertFalse(case.uses_pdmux)
            self.assertEqual(case.retrieval_backend, "hot_gpu_cold_cpu")
            self.assertIn("--enable-rag-ann", command)
            self.assertNotIn("--rag-cpu-retrieval", command)
            self.assertNotIn("--enable-pdmux", command)
            self.assertEqual(argument("--rag-index-path"), "/tmp/ivf.index")
            self.assertEqual(argument("--rag-hot-index-path"), "/tmp/hot.index")
            self.assertEqual(
                argument("--rag-hot-index-map-path"), "/tmp/hot.imap"
            )
            self.assertEqual(argument("--rag-cpu-retrieval-threads"), "8")
            self.assertEqual(
                argument("--rag-cpu-retrieval-max-batch-queries"), "3"
            )
            self.assertEqual(argument("--rag-ann-max-inflight"), "1")
            self.assertEqual(argument("--rag-ann-max-batches-per-tick"), "1")
            self.assertEqual(argument("--rag-ann-max-batch-size"), "8")
            self.assertEqual(argument("--rag-shared-hbm-mode"), "off")

    def test_hybrid_policies_share_static_index_but_not_coordination(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                shared_hbm_mode="ivf",
                hybrid_cpu_search_threads=6,
                ann_handoff_timeout_ms=125,
            )
            profiler.paths["vectorlite_cpu_index_path"] = "/tmp/ivffs.index"
            profiler.paths["vectorlite_hot_index_path"] = "/tmp/hot.index"
            profiler.paths["vectorlite_hot_index_map_path"] = "/tmp/hot.imap"
            naive = replace(
                self.make_case(),
                policy="naive-hybrid",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )
            raglease = replace(
                self.make_case(),
                policy="raglease-hybrid",
                schedule="decode-gc",
            )

            commands = {
                "naive": profiler.build_server_command(naive),
                "raglease": profiler.build_server_command(raglease),
            }

            def argument(command, option):
                return command[command.index(option) + 1]

            for command in commands.values():
                self.assertEqual(
                    argument(command, "--rag-index-path"),
                    "/tmp/ivffs.index",
                )
                self.assertEqual(
                    argument(command, "--rag-hot-index-path"),
                    "/tmp/hot.index",
                )
                self.assertEqual(
                    argument(command, "--rag-hot-index-map-path"),
                    "/tmp/hot.imap",
                )
                self.assertEqual(
                    argument(command, "--rag-shared-hbm-mode"), "off"
                )
                self.assertEqual(
                    argument(command, "--rag-cpu-retrieval-threads"), "6"
                )

            for command in commands.values():
                self.assertEqual(
                    argument(command, "--rag-ann-max-inflight"), "1"
                )
                self.assertEqual(
                    argument(command, "--rag-ann-max-batches-per-tick"), "1"
                )
                self.assertEqual(
                    argument(command, "--rag-ann-max-batch-size"), "64"
                )

            naive_command = commands["naive"]
            self.assertNotIn("--enable-pdmux", naive_command)
            self.assertEqual(
                argument(naive_command, "--rag-ann-schedule"), "primary"
            )
            self.assertEqual(
                argument(naive_command, "--rag-ann-launch-site"), "early"
            )
            self.assertEqual(
                argument(naive_command, "--rag-ann-handoff-timeout-ms"),
                "0.0",
            )
            self.assertIn(
                "--disable-rag-ann-work-conserving-decode-overlap",
                naive_command,
            )

            raglease_command = commands["raglease"]
            self.assertIn("--enable-pdmux", raglease_command)
            self.assertEqual(
                argument(raglease_command, "--rag-ann-schedule"), "decode-gc"
            )
            self.assertEqual(
                argument(raglease_command, "--rag-ann-handoff-timeout-ms"),
                "125.0",
            )
            self.assertIn(
                "--rag-ann-work-conserving-decode-overlap", raglease_command
            )

    def test_raglease_hybrid_can_opt_into_shared_hbm(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir,
                shared_hbm_mode="ivf",
                raglease_hybrid_shared_hbm_enabled=True,
            )
            profiler.paths["vectorlite_cpu_index_path"] = "/tmp/ivffs.index"
            profiler.paths["vectorlite_hot_index_path"] = "/tmp/hot.index"
            profiler.paths["vectorlite_hot_index_map_path"] = "/tmp/hot.imap"
            naive = replace(
                self.make_case(),
                policy="naive-hybrid",
                schedule="primary",
                decode_slo_ms=-1,
                max_defer_ms=0,
                ann_lease_sm=-1,
                prefill_lease_sm=-1,
            )
            raglease = replace(
                self.make_case(),
                policy="raglease-hybrid",
                schedule="decode-gc",
            )

            naive_command = profiler.build_server_command(naive)
            raglease_command = profiler.build_server_command(raglease)

            self.assertEqual(
                naive_command[
                    naive_command.index("--rag-shared-hbm-mode") + 1
                ],
                "off",
            )
            self.assertEqual(
                raglease_command[
                    raglease_command.index("--rag-shared-hbm-mode") + 1
                ],
                "ivf",
            )


    def test_hybrid_server_info_requires_the_configured_static_backend(self):
        paths = {
            "vectorlite_cpu_index_path": "/tmp/ivffs.index",
            "vectorlite_hot_index_path": "/tmp/hot.index",
            "vectorlite_hot_index_map_path": "/tmp/hot.imap",
        }
        naive = replace(
            self.make_case(),
            policy="naive-hybrid",
            schedule="primary",
            decode_slo_ms=-1,
            max_defer_ms=0,
            ann_lease_sm=-1,
            prefill_lease_sm=-1,
        )
        info = {
            "enable_pdmux": False,
            "pdmux_config_path": "/tmp/t24.yaml",
            "enable_rag_ann": True,
            "rag_cpu_retrieval": False,
            "rag_ann_schedule": "primary",
            "rag_ann_launch_site": "early",
            "rag_policy_mode": "full",
            "rag_ann_decode_step_slo_ms": -1,
            "rag_ann_max_defer_ms": 0,
            "rag_ann_frontend_arbitration": "overlap",
            "rag_ann_lease_frontend_sm": -1,
            "rag_prefill_lease_frontend_sm": -1,
            "rag_ann_handoff_timeout_ms": 0,
            "rag_shared_hbm_mode": "off",
            "rag_index_path": "/tmp/ivffs.index",
            "rag_hot_index_path": "/tmp/hot.index",
            "rag_hot_index_map_path": "/tmp/hot.imap",
        }

        OfflineProfiler.validate_server_info(
            naive, info, expected_paths=paths
        )

        raglease = replace(
            self.make_case(),
            policy="raglease-hybrid",
            schedule="decode-gc",
        )
        info.update(
            {
                "enable_pdmux": True,
                "rag_ann_schedule": "decode-gc",
                "rag_ann_launch_site": "after-prefill",
                "rag_ann_decode_step_slo_ms": 20,
                "rag_ann_max_defer_ms": 50,
                "rag_ann_frontend_arbitration": "fair",
                "rag_ann_handoff_timeout_ms": 30,
            }
        )
        OfflineProfiler.validate_server_info(
            raglease, info, expected_paths=paths
        )

        info["rag_hot_index_path"] = "/tmp/wrong.index"
        with self.assertRaisesRegex(RuntimeError, "rag_hot_index_path mismatch"):
            OfflineProfiler.validate_server_info(
                raglease, info, expected_paths=paths
            )

    def test_validate_naive_rejects_pdmux(self):
        case = replace(
            self.make_case(),
            policy="naive",
            schedule="primary",
            decode_slo_ms=-1,
            max_defer_ms=0,
            ann_lease_sm=-1,
            prefill_lease_sm=-1,
        )
        info = {
            "enable_pdmux": False,
            "rag_ann_schedule": "primary",
            "rag_ann_launch_site": "early",
            "rag_policy_mode": "full",
            "rag_ann_decode_step_slo_ms": -1,
            "rag_ann_max_defer_ms": 0,
            "rag_ann_frontend_arbitration": "overlap",
            "rag_ann_lease_frontend_sm": -1,
            "rag_prefill_lease_frontend_sm": -1,
            "rag_ann_handoff_timeout_ms": 0,
        }

        OfflineProfiler.validate_server_info(case, info)

        info["enable_pdmux"] = True
        with self.assertRaisesRegex(RuntimeError, "PD-Mux state mismatch"):
            OfflineProfiler.validate_server_info(case, info)

    def test_full_server_command_keeps_pdmux(self):
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(
                output_dir, prefill_exclusive_sprint=True
            )

            command = profiler.build_server_command(self.make_case())

            self.assertIn("--enable-pdmux", command)
            self.assertIn("--rag-prefill-exclusive-sprint", command)
            self.assertEqual(
                command[command.index("--pdmux-config-path") + 1],
                "/tmp/t24.yaml",
            )

    def test_build_cases_uses_decode_gc_for_managed_policies(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {"ann_schedule": "decode-gc"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["naive", "dynamic-only", "full"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(
            [(case.policy, case.schedule) for case in cases],
            [
                ("naive", "primary"),
                ("dynamic-only", "decode-gc"),
                ("full", "decode-gc"),
            ],
        )

    def test_build_cases_allows_primary_for_full_placement_probe(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {"ann_schedule": "primary"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["full"],
                "request_rates": [4],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(
            [(case.policy, case.schedule) for case in cases],
            [("full", "primary")],
        )

    def test_build_cases_sweeps_temporal_lease_for_full_policy(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["full"],
                "request_rates": [6],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [56],
                "prefill_lease_sm": [72],
                "ann_stream_epoch_ms": [0, 50, 100, 200],
                "seeds": [7],
            },
        }
        cases = build_cases(config)

        self.assertEqual([case.stream_epoch_ms for case in cases], [0, 50, 100, 200])
        self.assertEqual(len({case.name for case in cases}), 4)
        self.assertTrue(all(f"epoch{int(case.stream_epoch_ms)}" in case.name for case in cases))

    def test_build_cases_sweeps_prefill_layer_bundles_for_raglease(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {"ann_schedule": "decode-gc"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["raglease-hybrid"],
                "request_rates": [6],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [64],
                "prefill_lease_layer_bounds": [[1, 1], [2, 2], [4, 4], [8, 8]],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(
            [
                (case.prefill_lease_min_layers, case.prefill_lease_max_layers)
                for case in cases
            ],
            [(1, 1), (2, 2), (4, 4), (8, 8)],
        )
        self.assertEqual(len({case.name for case in cases}), 4)
        self.assertTrue(all("_pl" in case.name for case in cases))

        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)
            command = profiler.build_server_command(
                replace(cases[2], policy="full")
            )
        self.assertEqual(
            command[command.index("--rag-prefill-lease-min-layers") + 1],
            "4",
        )
        self.assertEqual(
            command[command.index("--rag-prefill-lease-max-layers") + 1],
            "4",
        )

    def test_build_cases_sweeps_ann_max_inflight_for_raglease(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {"ann_schedule": "decode-gc", "ann_max_inflight": 8},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["raglease-hybrid"],
                "request_rates": [12],
                "decode_slo_ms": [35],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [72],
                "ann_max_inflight": [1, 2, 4, 8],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual([case.ann_max_inflight for case in cases], [1, 2, 4, 8])
        self.assertEqual(len({case.name for case in cases}), 4)
        self.assertTrue(
            all(f"_ai{case.ann_max_inflight}" in case.name for case in cases)
        )

        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)
            command = profiler.build_server_command(
                replace(cases[2], policy="full")
            )
        self.assertEqual(
            command[command.index("--rag-ann-max-inflight") + 1],
            "4",
        )

    def test_build_cases_sweeps_prefill_recovery_interval_for_raglease(self):
        config = {
            "paths": {"pdmux_config_path": "/tmp/t24.yaml"},
            "runtime": {"ann_schedule": "decode-gc"},
            "workload": {"seed": 7},
            "sweep": {
                "paired_policies": ["raglease-hybrid"],
                "request_rates": [6],
                "decode_slo_ms": [20],
                "max_defer_ms": [50],
                "ann_lease_sm": [-1],
                "prefill_lease_sm": [64],
                "prefill_max_consecutive_chunks": [1, 2, 4, 8],
                "seeds": [7],
            },
        }

        cases = build_cases(config)

        self.assertEqual(
            [case.prefill_max_consecutive_chunks for case in cases],
            [1, 2, 4, 8],
        )
        self.assertEqual(len({case.name for case in cases}), 4)
        self.assertTrue(all("_pc" in case.name for case in cases))

        with tempfile.TemporaryDirectory() as output_dir:
            profiler = self.make_profiler(output_dir)
            command = profiler.build_server_command(
                replace(cases[1], policy="full")
            )
        self.assertEqual(
            command[
                command.index("--rag-prefill-max-consecutive-chunks") + 1
            ],
            "2",
        )

    def test_selection_uses_slo_constraints_and_robust_goodput(self):
        def candidate(key, goodput, goodput_std, itl):
            metrics = {
                "failed": metric(0),
                "slo_attainment": metric(1.0),
                "slo_goodput": metric(goodput, goodput_std),
                "ttft_p99_ms": metric(700),
                "itl_p99_ms": metric(itl),
                "ann_p99_ms": metric(120),
            }
            return {
                "key": key,
                "schedule": "frontend-gc",
                "request_rate": 6.0,
                "decode_slo_ms": 24,
                "max_defer_ms": 100,
                "ann_lease_sm": 48,
                "prefill_lease_sm": 60,
                "pdmux_config_path": "/tmp/t24.yaml",
                "repetitions": 3,
                "metrics": metrics,
            }

        stable = candidate("stable", 5.7, 0.05, 30)
        noisy = candidate("noisy", 5.8, 0.3, 29)
        violates_itl = candidate("violates", 6.0, 0.0, 40)
        selected = select_profiles(
            [stable, noisy, violates_itl],
            {
                "min_slo_attainment": 0.99,
                "min_repetitions": 3,
                "ttft_slo_ms": 1000,
                "itl_slo_ms": 35,
                "stability_weight": 1.0,
            },
        )

        self.assertEqual(selected["6"]["key"], "stable")

    def test_selection_uses_e2e_slo_goodput_when_configured(self):
        def candidate(key, goodput, goodput_std, attainment, e2e_p99):
            metrics = {
                "failed": metric(0),
                "slo_attainment": metric(0.0),
                "slo_goodput": metric(0.0),
                "e2e_slo_attainment": metric(attainment),
                "e2e_slo_goodput": metric(goodput, goodput_std),
                "e2e_p99_ms": metric(e2e_p99),
                "ttft_p99_ms": metric(5000),
                "itl_p99_ms": metric(100),
                "ann_p99_ms": metric(120),
            }
            return {
                "key": key,
                "policy": "full",
                "schedule": "decode-gc",
                "request_rate": 6.0,
                "decode_slo_ms": 24,
                "max_defer_ms": 100,
                "ann_lease_sm": -1,
                "prefill_lease_sm": 72,
                "pdmux_config_path": "/tmp/t24.yaml",
                "repetitions": 3,
                "metrics": metrics,
            }

        stable = candidate("stable", 5.4, 0.05, 1.0, 7900)
        noisy = candidate("noisy", 5.5, 0.30, 1.0, 7800)
        violates = candidate("violates", 6.0, 0.0, 0.98, 9000)
        selected = select_profiles(
            [stable, noisy, violates],
            {
                "min_slo_attainment": 0.99,
                "min_repetitions": 3,
                "e2e_slo_ms": 8000,
                "ttft_slo_ms": 1000,
                "itl_slo_ms": 35,
                "stability_weight": 1.0,
            },
        )

        self.assertEqual(selected["6"]["key"], "stable")
        self.assertEqual(
            selected["6"]["selection_objective"], "e2e_slo_goodput"
        )

    def test_selection_rejects_unstable_raglease_hybrid_candidate(self):
        def candidate(key, stationarity_pass, goodput):
            return {
                "key": key,
                "policy": "raglease-hybrid",
                "schedule": "decode-gc",
                "request_rate": 6.0,
                "decode_slo_ms": 20,
                "max_defer_ms": 50,
                "ann_lease_sm": -1,
                "prefill_lease_sm": 64,
                "pdmux_config_path": "/tmp/t24.yaml",
                "repetitions": 1,
                "metrics": {
                    "failed": metric(0),
                    "stationarity_available": metric(1),
                    "stationarity_pass": metric(stationarity_pass),
                    "slo_attainment": metric(1.0),
                    "slo_goodput": metric(goodput),
                    "e2e_slo_attainment": metric(1.0),
                    "e2e_slo_goodput": metric(goodput),
                    "e2e_p99_ms": metric(5000),
                    "ttft_p99_ms": metric(1000),
                    "itl_p99_ms": metric(30),
                    "ann_p99_ms": metric(100),
                },
            }

        unstable = candidate("unstable", 0, 5.9)
        stable = candidate("stable", 1, 5.7)
        selected = select_profiles(
            [unstable, stable],
            {
                "min_slo_attainment": 0.99,
                "min_repetitions": 1,
                "e2e_slo_ms": 8000,
                "ttft_slo_ms": 10000,
                "itl_slo_ms": 250,
                "require_stationarity": True,
            },
        )

        self.assertEqual(selected["6"]["key"], "stable")
        self.assertFalse(unstable["feasible"])


if __name__ == "__main__":
    unittest.main()
