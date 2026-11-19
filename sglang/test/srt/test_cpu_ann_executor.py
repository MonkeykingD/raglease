import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "multiplex"
    / "cpu_ann_executor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "cpu_ann_executor_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
CPU_ANN_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CPU_ANN_MODULE
SPEC.loader.exec_module(CPU_ANN_MODULE)
CpuAnnExecutor = CPU_ANN_MODULE.CpuAnnExecutor


class _BlockingIndex:
    def __init__(self):
        self.nprobe = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.rows_per_call = []

    def search(self, queries, topk):
        self.calls += 1
        self.rows_per_call.append(len(queries))
        if self.calls == 1:
            self.started.set()
            self.release.wait(timeout=2)
        rows = len(queries)
        distances = np.zeros((rows, topk), dtype=np.float32)
        ids = np.arange(rows * topk, dtype=np.int64).reshape(rows, topk)
        return distances, ids


class _FastScanIndex(_BlockingIndex):
    def __init__(self):
        super().__init__()
        self.implem = None
        self.use_precomputed_table = False


class _FakeFaiss:
    IndexIVFPQFastScan = _FastScanIndex

    def __init__(self, index):
        self.index = index
        self.thread_counts = []

    def read_index(self, _path):
        return self.index

    def downcast_index(self, index):
        return index

    def omp_set_num_threads(self, count):
        self.thread_counts.append(count)


class TestCpuAnnExecutor(unittest.TestCase):
    def test_rejects_multiple_owners_for_non_reentrant_index(self):
        with self.assertRaisesRegex(ValueError, "non-reentrant"):
            CpuAnnExecutor("unused.index", nprobe=1, workers=2)

    def test_bounded_worker_reports_queue_and_service_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "ivfpq.index"
            index_path.write_bytes(b"stub")
            index = _BlockingIndex()
            faiss = _FakeFaiss(index)
            executor = CpuAnnExecutor(
                str(index_path),
                nprobe=2048,
                workers=1,
                threads_per_search=8,
            )
            try:
                with patch.object(
                    CPU_ANN_MODULE, "_import_faiss", return_value=faiss
                ):
                    executor.load()
                    first = executor.submit(np.zeros((1, 4), np.float32), 2)
                    self.assertTrue(index.started.wait(timeout=1))
                    second = executor.submit(np.ones((1, 4), np.float32), 2)
                    time.sleep(0.02)
                    index.release.set()
                    first_result = first.result(timeout=1)
                    second_result = second.result(timeout=1)
            finally:
                executor.close()

        self.assertEqual(index.nprobe, 2048)
        self.assertEqual(first_result.ids.shape, (1, 2))
        self.assertGreaterEqual(first_result.queue_wait_ms, 0)
        self.assertGreater(second_result.queue_wait_ms, 10)
        self.assertGreaterEqual(first_result.search_latency_ms, 0)
        self.assertEqual(faiss.thread_counts, [8])

    def test_queued_queries_are_coalesced_without_reentrant_search(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "ivfpq.index"
            index_path.write_bytes(b"stub")
            index = _BlockingIndex()
            faiss = _FakeFaiss(index)
            executor = CpuAnnExecutor(
                str(index_path),
                nprobe=2048,
                workers=1,
                threads_per_search=4,
                max_batch_queries=4,
            )
            try:
                with patch.object(
                    CPU_ANN_MODULE, "_import_faiss", return_value=faiss
                ):
                    executor.load()
                    first = executor.submit(np.zeros((1, 4), np.float32), 2)
                    self.assertTrue(index.started.wait(timeout=1))
                    queued = [
                        executor.submit(
                            np.full((1, 4), value, np.float32), 2
                        )
                        for value in (1, 2, 3)
                    ]
                    index.release.set()
                    first.result(timeout=1)
                    results = [future.result(timeout=1) for future in queued]
            finally:
                executor.close()

        self.assertEqual(index.calls, 2)
        self.assertEqual(index.rows_per_call, [1, 3])
        self.assertTrue(all(result.batch_size == 3 for result in results))
        self.assertTrue(
            all(result.ids.shape == (1, 2) for result in results)
        )
        self.assertEqual(faiss.thread_counts, [4])

    def test_fastscan_index_uses_vectorliterag_cpu_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "ivffs.index"
            index_path.write_bytes(b"stub")
            index = _FastScanIndex()
            faiss = _FakeFaiss(index)
            executor = CpuAnnExecutor(str(index_path), nprobe=1024)
            try:
                with patch.object(
                    CPU_ANN_MODULE, "_import_faiss", return_value=faiss
                ):
                    executor.load()
                    self.assertEqual(executor.index_mode, "cpu_ivfpq_fastscan")
                    self.assertEqual(index.implem, 12)
                    self.assertTrue(index.use_precomputed_table)
            finally:
                executor.close()


if __name__ == "__main__":
    unittest.main()
