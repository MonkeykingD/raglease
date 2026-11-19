import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.srt.multiplex.hot_index import (
    load_vectorlite_mapping,
    merge_topk,
    route_vectorlite_probes,
)


class TestVectorLiteHotIndex(unittest.TestCase):
    def test_loads_partitioned_index_triples(self):
        entries = np.array(
            [[0, 0, 1], [2, 1, 0], [4, 0, 0]], dtype=np.int32
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "10_4.imap"
            entries.tofile(path)
            mapping = load_vectorlite_mapping(path, nlist=8)

        np.testing.assert_array_equal(
            mapping.cid_lut, np.array([1, 1, 0, 3, 0, 5, 6, 7])
        )
        np.testing.assert_array_equal(
            mapping.shard_lut, np.array([0, -1, 1, -1, 0, -1, -1, -1])
        )
        self.assertEqual(mapping.num_entries, 3)
        self.assertEqual(mapping.shard_ids, (0, 1))

    def test_routes_selected_hot_shard_and_cpu_fallback(self):
        entries = np.array(
            [[0, 0, 1], [2, 1, 0], [4, 0, 0]], dtype=np.int32
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "10_4.imap"
            entries.tofile(path)
            mapping = load_vectorlite_mapping(path, nlist=8)

        coarse_ids = np.array([[0, 1, 2, 4], [3, 4, 6, -1]])
        coarse_distances = np.array(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, np.inf]],
            dtype=np.float32,
        )
        routed = route_vectorlite_probes(
            coarse_distances, coarse_ids, mapping, shard_id=0
        )

        np.testing.assert_array_equal(routed.hot_counts, [2, 1])
        np.testing.assert_array_equal(routed.cold_counts, [2, 2])
        np.testing.assert_array_equal(routed.hot_ids, [[1, 0], [0, -1]])
        np.testing.assert_array_equal(routed.cold_ids, [[1, 2], [3, 6]])
        np.testing.assert_allclose(
            routed.hot_distances,
            [[0.1, 0.4], [1.1, np.inf]],
        )

    def test_routes_hot_probe_overflow_to_cpu_without_dropping_work(self):
        entries = np.array(
            [[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]],
            dtype=np.int32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hot.imap"
            entries.tofile(path)
            mapping = load_vectorlite_mapping(path, nlist=5)

        coarse_ids = np.array([[0, 1, 4, 2, 3]], dtype=np.int64)
        coarse_distances = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32
        )
        routed = route_vectorlite_probes(
            coarse_distances,
            coarse_ids,
            mapping,
            shard_id=0,
            max_hot_probes_per_query=2,
        )

        np.testing.assert_array_equal(routed.hot_counts, [2])
        np.testing.assert_array_equal(routed.cold_counts, [3])
        np.testing.assert_array_equal(routed.hot_ids, [[0, 1]])
        np.testing.assert_array_equal(routed.cold_ids, [[4, 2, 3]])
        np.testing.assert_allclose(routed.hot_distances, [[0.1, 0.2]])
        np.testing.assert_allclose(
            routed.cold_distances, [[0.3, 0.4, 0.5]]
        )

    def test_merges_cpu_and_gpu_candidates_by_l2_distance(self):
        cpu_distances = np.array([[0.2, 0.8, np.inf]], dtype=np.float32)
        cpu_ids = np.array([[20, 80, -1]], dtype=np.int64)
        gpu_distances = np.array([[0.1, 0.5, 0.7]], dtype=np.float32)
        gpu_ids = np.array([[10, 50, 70]], dtype=np.int64)

        distances, ids = merge_topk(
            [(cpu_distances, cpu_ids), (gpu_distances, gpu_ids)], topk=4
        )

        np.testing.assert_allclose(distances, [[0.1, 0.2, 0.5, 0.7]])
        np.testing.assert_array_equal(ids, [[10, 20, 50, 70]])

    def test_rejects_legacy_non_triple_mapping(self):
        legacy = np.array([2, 10, 0, 11, 1], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.imap"
            legacy.tofile(path)
            with self.assertRaisesRegex(ValueError, "int32 triples"):
                load_vectorlite_mapping(path, nlist=32)


if __name__ == "__main__":
    unittest.main()
