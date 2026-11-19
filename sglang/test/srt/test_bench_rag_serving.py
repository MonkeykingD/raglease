import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.pdmux.bench_rag_serving import (
    RequestResult,
    SERVER_INFO_FIELDS,
    compute_slo_metrics,
    load_fbin_vectors,
    validate_ann_priming_snapshot,
)


def primed_server_metadata(
    *,
    credit_ms=10.0,
    batch1_p95=5.0,
    batch1_p99=6.0,
    batch2_p95=9.5,
    batch2_p99=10.5,
    samples=32,
):
    completion = {
        str(batch_size): {
            "mature": True,
            "service_samples": samples,
            "tail_samples": samples,
            "worker_collection_samples": samples,
            "worker_send_samples": samples,
        }
        for batch_size in (1, 2)
    }
    return {
        "rag_ann_runtime": [
            {
                "ann_credit": {
                    "configured_max_queued_gpu_ms": credit_ms,
                },
                "ann_gpu_service_by_batch_size": {
                    "1": {
                        "samples": samples,
                        "p95_ms": batch1_p95,
                        "p99_ms": batch1_p99,
                    },
                    "2": {
                        "samples": samples,
                        "p95_ms": batch2_p95,
                        "p99_ms": batch2_p99,
                    },
                },
                "ann_pre_switch_launch": {
                    "completion_predictor": {
                        "by_path_batch": {"full_gpu": completion}
                    }
                },
            }
        ]
    }


class TestSloMetrics(unittest.TestCase):
    def test_server_snapshot_preserves_shared_hbm_configuration(self):
        expected_fields = {
            "rag_shared_hbm_mode",
            "rag_shared_hbm_pool_gib",
            "rag_shared_hbm_chunk_mib",
            "rag_shared_hbm_lookahead_ms",
            "rag_shared_hbm_min_residency_ms",
            "rag_shared_hbm_kv_reserve_extents",
            "rag_shared_hbm_rerank_prefetch_batches",
            "rag_shared_hbm_async_map",
            "rag_shared_hbm_map_stage_extents",
        }

        self.assertTrue(expected_fields.issubset(SERVER_INFO_FIELDS))

    def test_e2e_slo_goodput_is_independent_of_ttft_itl_slo(self):
        requests = [
            RequestResult(
                request_id=0,
                query_id=0,
                success=True,
                e2e_ms=7000,
                ttft_ms=500,
                itl_ms=[10, 20],
            ),
            RequestResult(
                request_id=1,
                query_id=1,
                success=True,
                e2e_ms=8000,
                ttft_ms=1200,
                itl_ms=[10],
            ),
            RequestResult(
                request_id=2,
                query_id=2,
                success=True,
                e2e_ms=8001,
                ttft_ms=500,
                itl_ms=[40],
            ),
        ]

        metrics = compute_slo_metrics(
            requests,
            2.0,
            ttft_slo_ms=1000,
            itl_slo_ms=35,
            e2e_slo_ms=8000,
        )

        self.assertAlmostEqual(metrics["slo_attainment"], 1 / 3)
        self.assertEqual(metrics["slo_compliant_qps"], 0.5)
        self.assertAlmostEqual(metrics["e2e_slo_attainment"], 2 / 3)
        self.assertEqual(metrics["e2e_slo_compliant_qps"], 1.0)

    def test_disabled_e2e_slo_counts_all_completed_requests(self):
        requests = [
            RequestResult(
                request_id=0,
                query_id=0,
                success=True,
                e2e_ms=9000,
            )
        ]

        metrics = compute_slo_metrics(
            requests,
            0.5,
            ttft_slo_ms=-1,
            itl_slo_ms=-1,
            e2e_slo_ms=-1,
        )

        self.assertEqual(metrics["e2e_slo_attainment"], 1.0)
        self.assertEqual(metrics["e2e_slo_compliant_qps"], 2.0)


class TestAnnPriming(unittest.TestCase):
    def test_load_fbin_vectors_validates_header_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.fbin"
            with path.open("wb") as output:
                np.asarray([3, 2], dtype=np.int32).tofile(output)
                np.asarray(
                    [[1, 2], [3, 4], [5, 6]], dtype=np.float32
                ).tofile(output)

            vectors = load_fbin_vectors(path)

            self.assertEqual(vectors.shape, (3, 2))
            self.assertEqual(vectors.dtype, np.float32)
            self.assertTrue(vectors.flags.c_contiguous)
            np.testing.assert_array_equal(
                vectors,
                np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            )

    def test_load_fbin_vectors_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.fbin"
            with path.open("wb") as output:
                np.asarray([2, 2], dtype=np.int32).tofile(output)
                np.asarray([1, 2, 3], dtype=np.float32).tofile(output)

            with self.assertRaisesRegex(ValueError, "Invalid fbin payload"):
                load_fbin_vectors(path)

    def test_priming_snapshot_accepts_mature_credit_straddle(self):
        report = validate_ann_priming_snapshot(
            primed_server_metadata(),
            [1, 2],
            16,
            require_credit_straddle=True,
        )

        self.assertTrue(report["valid"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            report["valid_credit_interval"],
            {
                "lower_inclusive_ms": 9.5,
                "upper_exclusive_ms": 10.5,
                "nonempty": True,
            },
        )

    def test_priming_snapshot_rejects_credit_outside_interval(self):
        report = validate_ann_priming_snapshot(
            primed_server_metadata(credit_ms=11.0),
            [1, 2],
            16,
            require_credit_straddle=True,
        )

        self.assertFalse(report["valid"])
        self.assertIn("outside the valid", report["failures"][-1])

    def test_priming_snapshot_rejects_immature_completion_history(self):
        metadata = primed_server_metadata(samples=8)
        completion = metadata["rag_ann_runtime"][0][
            "ann_pre_switch_launch"
        ]["completion_predictor"]["by_path_batch"]["full_gpu"]["2"]
        completion["mature"] = False
        completion["worker_send_samples"] = 2

        report = validate_ann_priming_snapshot(
            metadata,
            [1, 2],
            4,
            require_credit_straddle=True,
        )

        self.assertFalse(report["valid"])
        self.assertIn(
            "batch=2 completion history is not mature",
            report["failures"],
        )
        self.assertIn(
            "batch=2 worker_send_samples=2 < 4",
            report["failures"],
        )

    def test_priming_snapshot_rejects_non_p95_credit_predictor(self):
        metadata = primed_server_metadata()
        metadata["rag_ann_runtime"][0]["ann_credit"][
            "service_time_percentile"
        ] = 99.0

        report = validate_ann_priming_snapshot(
            metadata,
            [1, 2],
            4,
            require_credit_straddle=True,
        )

        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                "service-time percentile to be 95" in failure
                for failure in report["failures"]
            )
        )

    def test_priming_snapshot_rejects_multiple_runtime_entries(self):
        metadata = primed_server_metadata()
        metadata["rag_ann_runtime"].append(
            metadata["rag_ann_runtime"][0]
        )

        report = validate_ann_priming_snapshot(
            metadata,
            [1, 2],
            4,
            require_credit_straddle=False,
        )

        self.assertFalse(report["valid"])
        self.assertIn(
            "exactly one rag_ann_runtime entry",
            report["failures"][0],
        )


if __name__ == "__main__":
    unittest.main()
