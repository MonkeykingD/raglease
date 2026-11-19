import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.pdmux.compare_rag_results import (
    normalize_sglang_result,
    normalize_vectorlite_records,
    summarize_system,
)
from benchmark.pdmux.rag_workload import (
    generate_workload,
    load_workload,
    save_workload,
    unique_value,
)
from benchmark.pdmux.run_vectorlite_trace import build_execution_items


class TestRagWorkload(unittest.TestCase):
    def test_deterministic_trace_has_absolute_arrivals_and_stages(self):
        workload = generate_workload(
            num_requests=3,
            ramp_up_requests=2,
            warmup_requests=1,
            request_rate=4.0,
            output_len=64,
            topk=25,
            seed=7,
            deterministic_arrivals=True,
            query_count=3,
        )

        self.assertEqual(
            [request.request_id for request in workload.requests],
            [-2, -1, 0, 1, 2],
        )
        self.assertEqual(
            [request.stage for request in workload.requests],
            ["ramp_up", "ramp_up", "measure", "measure", "measure"],
        )
        self.assertEqual(
            [request.arrival_offset_ms for request in workload.requests],
            [0.0, 250.0, 500.0, 750.0, 1000.0],
        )
        self.assertEqual(
            [request.query_id for request in workload.requests],
            [0, 1, 2, 0, 1],
        )

    def test_deterministic_trace_inserts_gap_between_measured_bursts(self):
        workload = generate_workload(
            num_requests=4,
            ramp_up_requests=2,
            warmup_requests=0,
            request_rate=4.0,
            output_len=64,
            topk=25,
            seed=7,
            deterministic_arrivals=True,
            burst_size=2,
            inter_burst_gap_ms=1000.0,
        )

        self.assertEqual(
            [
                request.arrival_offset_ms
                for request in workload.measured
            ],
            [500.0, 750.0, 2000.0, 2250.0],
        )
        self.assertEqual(workload.metadata["burst_size"], 2)
        self.assertEqual(
            workload.metadata["inter_burst_gap_ms"], 1000.0
        )

    def test_jsonl_round_trip_preserves_workload_hash(self):
        workload = generate_workload(
            num_requests=4,
            ramp_up_requests=1,
            warmup_requests=2,
            request_rate=3.5,
            output_len=128,
            topk=16,
            seed=11,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            save_workload(path, workload)
            restored = load_workload(path)

        self.assertEqual(restored, workload)
        self.assertEqual(restored.sha256(), workload.sha256())
        self.assertEqual(unique_value(restored.requests, "topk"), 16)

    def test_trace_fingerprint_detects_manual_edits(self):
        workload = generate_workload(
            num_requests=2,
            ramp_up_requests=0,
            warmup_requests=0,
            request_rate=2.0,
            output_len=32,
            topk=4,
            seed=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            save_workload(path, workload)
            text = path.read_text(encoding="utf-8").replace(
                '"query_id": 0', '"query_id": 9', 1
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_workload(path)

    def test_vectorlite_execution_mapping_keeps_trace_query_rows(self):
        workload = generate_workload(
            num_requests=2,
            ramp_up_requests=1,
            warmup_requests=2,
            request_rate=2.0,
            output_len=32,
            topk=4,
            seed=5,
            query_id_start=10,
        )
        execution = build_execution_items(workload)

        self.assertEqual([item.rid for item in execution], list(range(5)))
        self.assertEqual(
            [item.stage for item in execution],
            ["warmup", "warmup", "ramp_up", "measure", "measure"],
        )
        self.assertEqual(
            [item.query_id for item in execution],
            [10, 11, 10, 11, 12],
        )

    def test_vectorlite_fixed_field_validation_rejects_heterogeneity(self):
        workload = generate_workload(
            num_requests=2,
            ramp_up_requests=0,
            warmup_requests=0,
            request_rate=1.0,
            output_len=64,
            topk=8,
            seed=1,
        )
        workload.requests[1] = replace(workload.requests[1], topk=16)
        with self.assertRaisesRegex(ValueError, "one topk"):
            unique_value(workload.requests, "topk")


class TestRagComparison(unittest.TestCase):
    def test_sglang_normalization_keeps_common_millisecond_metrics(self):
        payload = {
            "workload": {"sha256": "same-trace"},
            "summary": {"request_throughput_qps": 7.5},
            "requests": [
                {
                    "request_id": 0,
                    "query_id": 3,
                    "stage": "measure",
                    "success": True,
                    "e2e_ms": 100.0,
                    "ttft_ms": 40.0,
                    "ann_latency_ms": 10.0,
                    "ann_queue_wait_ms": 2.0,
                    "ann_gpu_latency_ms": 8.0,
                    "llm_ttft_after_ann_ms": 30.0,
                    "tpot_ms": 4.0,
                    "output_tokens": 20,
                    "ann_batch_size": 2,
                },
                {"request_id": 1, "query_id": 4, "success": False},
            ],
        }
        rows, metadata = normalize_sglang_result("phase3", payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rag_ttft_ms"], 40.0)
        self.assertEqual(rows[0]["tpot_ms"], 3.0)
        self.assertEqual(metadata["workload_sha256"], "same-trace")
        self.assertEqual(metadata["throughput_qps"], 7.5)

    def test_vectorlite_normalization_filters_warmup_and_converts_seconds(self):
        mapping = {
            "workload": {"sha256": "same-trace"},
            "requests": [
                {
                    "rid": 0,
                    "stage": "warmup",
                    "request_id": None,
                    "query_id": 0,
                },
                {
                    "rid": 1,
                    "stage": "measure",
                    "request_id": 0,
                    "query_id": 7,
                },
            ],
        }
        records = [
            {
                "rid": 0,
                "e2e": 9.0,
                "ttft": 9.0,
            },
            {
                "rid": 1,
                "e2e": 0.100,
                "ttft": 0.040,
                "ann_e2e": 0.010,
                "ann_queue": 0.002,
                "ann_search": 0.008,
                "llm_queue": 0.005,
                "prefill": 0.025,
                "tpot": 0.004,
                "batch_size": 2,
                "ts_arrival": 10.0,
                "ts_completion": 10.1,
            },
        ]
        rows, metadata = normalize_vectorlite_records(
            "vectorlite", records, mapping
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["e2e_ms"], 100.0)
        self.assertEqual(rows[0]["ann_latency_ms"], 10.0)
        self.assertAlmostEqual(rows[0]["llm_ttft_after_ann_ms"], 30.0)
        self.assertAlmostEqual(metadata["throughput_qps"], 10.0)

        summary = summarize_system(rows, metadata)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["metrics"]["tpot_ms"]["p99"], 4.0)


if __name__ == "__main__":
    unittest.main()
