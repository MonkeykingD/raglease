import os
import unittest

import torch


def _is_supported_shared_hbm_gpu():
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return "H100" in properties.name or (
        "A100" in properties.name
        and properties.total_memory >= 75 * (1 << 30)
    )


@unittest.skipUnless(
    _is_supported_shared_hbm_gpu(),
    "CUDA VMM shared-HBM test requires H100 or A100 80GB",
)
class TestSharedHBMVMM(unittest.TestCase):
    def setUp(self):
        import sgl_kernel  # noqa: F401

        if not hasattr(torch.ops.sgl_kernel, "shared_hbm_create"):
            overlay = os.environ.get("SGLANG_SHARED_HBM_OPS_LIBRARY")
            if not overlay:
                self.fail("SGLANG_SHARED_HBM_OPS_LIBRARY is not set")
            torch.ops.load_library(overlay)
        torch.cuda.set_device(0)

    def test_sglang_mha_pool_uses_fixed_vmm_segments(self):
        from sglang.srt.mem_cache import shared_hbm as shared_hbm_module
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = shared_hbm_module.SharedHBMPool(
            mode="scratch",
            pool_gib=0.25,
            gpu_id=0,
        )
        shared_hbm_module._GLOBAL_POOL = pool
        kv_pool = None
        try:
            pool.configure_kv_bounds(1023, 8191)
            kv_pool = MHATokenToKVPool(
                size=8191,
                page_size=1,
                dtype=torch.float16,
                head_num=8,
                head_dim=128,
                layer_num=4,
                device="cuda",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
            self.assertEqual(kv_pool.size, 8191)
            self.assertEqual(kv_pool.get_current_mapped_capacity(), 1023)
            key_ptrs = [tensor.data_ptr() for tensor in kv_pool.k_buffer]
            value_ptrs = [
                tensor.data_ptr() for tensor in kv_pool.v_buffer
            ]

            driver_before = tuple(
                map(
                    int,
                    torch.ops.sgl_kernel.shared_hbm_driver_stats(),
                )
            )
            self.assertEqual(kv_pool.map_shared_hbm_extents(2), 3071)
            driver_after = tuple(
                map(
                    int,
                    torch.ops.sgl_kernel.shared_hbm_driver_stats(),
                )
            )
            self.assertEqual(
                driver_after[2] - driver_before[2],
                len(kv_pool.k_buffer) + len(kv_pool.v_buffer),
            )
            self.assertEqual(
                driver_after[3] - driver_before[3],
                pool.snapshot().kv_bytes,
            )
            for tensor in kv_pool.k_buffer:
                tensor[1024, 0, 0] = 7
            for tensor in kv_pool.v_buffer:
                tensor[2048, 0, 0] = -7
            torch.cuda.synchronize()
            self.assertEqual(kv_pool.unmap_shared_hbm_extents(0), 1023)

            self.assertEqual(
                [tensor.data_ptr() for tensor in kv_pool.k_buffer],
                key_ptrs,
            )
            self.assertEqual(
                [tensor.data_ptr() for tensor in kv_pool.v_buffer],
                value_ptrs,
            )
            snapshot = pool.snapshot()
            self.assertEqual(snapshot.kv_bytes, 0)
            self.assertEqual(snapshot.free_bytes, snapshot.total_bytes)
            self.assertEqual(
                snapshot.free_bytes
                + snapshot.kv_bytes
                + snapshot.ann_scratch_bytes
                + snapshot.ann_index_bytes
                + snapshot.moving_bytes,
                snapshot.total_bytes,
            )
        finally:
            del kv_pool
            torch.cuda.synchronize()
            shared_hbm_module._GLOBAL_POOL = None
            pool.destroy()

    def test_ten_thousand_owner_transitions_keep_fixed_kv_va(self):
        torch.ops.sgl_kernel.shared_hbm_create(128 << 20, 2 << 20, 0)
        try:
            shape = [8192, 1024]
            key = torch.ops.sgl_kernel.shared_hbm_alloc_kv_tensor(
                shape, torch.float16, 1023, 8191, 0
            )
            value = torch.ops.sgl_kernel.shared_hbm_alloc_kv_tensor(
                shape, torch.float16, 1023, 8191, 0
            )
            key_ptr = key.data_ptr()
            value_ptr = value.data_ptr()
            static_tokens, tokens_per_extent, extents, logical = (
                torch.ops.sgl_kernel.shared_hbm_finalize_kv(1)
            )
            self.assertEqual(static_tokens, 1023)
            self.assertEqual(tokens_per_extent, 1024)
            self.assertGreater(extents, 0)
            self.assertGreater(logical, static_tokens)

            driver_before = tuple(
                map(
                    int,
                    torch.ops.sgl_kernel.shared_hbm_driver_stats(),
                )
            )
            for iteration in range(10_000):
                mapped = torch.ops.sgl_kernel.shared_hbm_set_kv_extents(1)
                self.assertEqual(mapped, 2047)
                key[1024, 0] = iteration % 127
                value[1024, 0] = -(iteration % 127)
                torch.cuda.synchronize()
                torch.ops.sgl_kernel.shared_hbm_set_kv_extents(0)

                _, scratch_bytes = (
                    torch.ops.sgl_kernel.shared_hbm_map_ann(
                        2, 4 << 20
                    )
                )
                self.assertEqual(scratch_bytes, 4 << 20)
                torch.ops.sgl_kernel.shared_hbm_release_ann(2)
                _, index_bytes = torch.ops.sgl_kernel.shared_hbm_map_ann(
                    3, 4 << 20
                )
                self.assertEqual(index_bytes, 4 << 20)
                torch.ops.sgl_kernel.shared_hbm_release_ann(3)
            driver_after = tuple(
                map(
                    int,
                    torch.ops.sgl_kernel.shared_hbm_driver_stats(),
                )
            )
            self.assertEqual(
                driver_after[2] - driver_before[2],
                4 * 10_000,
            )

            self.assertEqual(key.data_ptr(), key_ptr)
            self.assertEqual(value.data_ptr(), value_ptr)
            snapshot = torch.ops.sgl_kernel.shared_hbm_snapshot()
            total, chunk, free, kv, scratch, index, moving = snapshot[:7]
            self.assertEqual(free + kv + scratch + index + moving, total)
            self.assertEqual(free, total)
            self.assertGreater(chunk, 0)
            with self.assertRaises(RuntimeError):
                torch.ops.sgl_kernel.shared_hbm_release_ann(0)
        finally:
            torch.ops.sgl_kernel.shared_hbm_destroy()

    def test_tail_scratch_preserves_growable_kv_prefix(self):
        pool_bytes = 128 << 20
        chunk_bytes = 2 << 20
        scratch_bytes = 16 << 20
        torch.ops.sgl_kernel.shared_hbm_create(
            pool_bytes, chunk_bytes, 0
        )
        try:
            # The large logical bound lets KV consume every physical chunk in
            # this small test pool. Each K/V tensor contributes one chunk per
            # extent, so a tail scratch lease leaves a measurable KV prefix.
            shape = [65536, 1024]
            torch.ops.sgl_kernel.shared_hbm_alloc_kv_tensor(
                shape, torch.float16, 1023, 65535, 0
            )
            torch.ops.sgl_kernel.shared_hbm_alloc_kv_tensor(
                shape, torch.float16, 1023, 65535, 0
            )
            _, _, total_extents, _ = (
                torch.ops.sgl_kernel.shared_hbm_finalize_kv(1)
            )
            _, mapped_scratch = torch.ops.sgl_kernel.shared_hbm_map_ann(
                2, scratch_bytes
            )
            self.assertEqual(mapped_scratch, scratch_bytes)

            extent_bytes = pool_bytes // total_extents
            self.assertEqual(pool_bytes % total_extents, 0)
            self.assertEqual(scratch_bytes % extent_bytes, 0)
            expected_boundary = (
                total_extents - scratch_bytes // extent_bytes
            )
            self.assertEqual(
                torch.ops.sgl_kernel.shared_hbm_kv_growable_extents(),
                expected_boundary,
            )

            torch.ops.sgl_kernel.shared_hbm_set_kv_extents(
                expected_boundary
            )
            snapshot = torch.ops.sgl_kernel.shared_hbm_snapshot()
            total, _, free, kv, scratch, index, moving = snapshot[:7]
            self.assertEqual(free, 0)
            self.assertEqual(kv + scratch, total)
            self.assertEqual(scratch, scratch_bytes)
            self.assertEqual(index, 0)
            self.assertEqual(moving, 0)
            with self.assertRaises(RuntimeError):
                torch.ops.sgl_kernel.shared_hbm_set_kv_extents(
                    expected_boundary + 1
                )

            # Releasing the tail lease exposes the rest of the same physical
            # prefix without changing any KV virtual address.
            torch.ops.sgl_kernel.shared_hbm_release_ann(2)
            self.assertEqual(
                torch.ops.sgl_kernel.shared_hbm_kv_growable_extents(),
                total_extents,
            )
            torch.ops.sgl_kernel.shared_hbm_set_kv_extents(total_extents)
            snapshot = torch.ops.sgl_kernel.shared_hbm_snapshot()
            self.assertEqual(snapshot[2], 0)
            self.assertEqual(snapshot[3], snapshot[0])
            self.assertEqual(snapshot[4], 0)
        finally:
            torch.ops.sgl_kernel.shared_hbm_destroy()


if __name__ == "__main__":
    unittest.main()
