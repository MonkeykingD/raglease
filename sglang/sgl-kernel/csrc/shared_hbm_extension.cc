#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "shared_hbm/shared_hbm_pool.h"

namespace {

void check_shbm(int result) {
  if (result != 0) {
    throw std::runtime_error(shbm_last_error());
  }
}

std::vector<int64_t> shared_hbm_create(int64_t total_bytes, int64_t chunk_bytes, int64_t device) {
  check_shbm(shbm_create(device, total_bytes, chunk_bytes));
  SharedHBMSnapshot snapshot{};
  check_shbm(shbm_snapshot(&snapshot));
  return {static_cast<int64_t>(snapshot.total_bytes), static_cast<int64_t>(snapshot.chunk_bytes)};
}

void shared_hbm_destroy() {
  check_shbm(shbm_destroy());
}

at::Tensor shared_hbm_alloc_kv_tensor(
    at::IntArrayRef sizes, c10::ScalarType dtype, int64_t static_tokens, int64_t max_tokens, int64_t device) {
  if (sizes.empty() || sizes[0] != max_tokens + 1) {
    throw std::invalid_argument("shared-HBM KV tensor must contain max_tokens + one dummy slot");
  }
  c10::cuda::CUDAGuard guard(device);
  uint64_t numel = 1;
  for (const int64_t size : sizes) {
    if (size <= 0) {
      throw std::invalid_argument("KV tensor dimensions must be positive");
    }
    numel *= static_cast<uint64_t>(size);
  }
  const uint64_t element_size = c10::elementSize(dtype);
  uint64_t row_elements = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    row_elements *= static_cast<uint64_t>(sizes[i]);
  }
  const uint64_t token_stride = row_elements * element_size;
  const uint64_t tensor_bytes = numel * element_size;
  uint64_t address = 0;
  check_shbm(shbm_alloc_kv_segment(tensor_bytes, token_stride, static_tokens, max_tokens, &address));

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, device);
  return at::from_blob(
      reinterpret_cast<void*>(address), sizes, [](void*) {}, options);
}

std::vector<int64_t> shared_hbm_finalize_kv(int64_t page_size) {
  uint64_t static_capacity = 0;
  uint64_t tokens_per_extent = 0;
  uint64_t total_extents = 0;
  uint64_t logical_capacity = 0;
  check_shbm(
      shbm_finalize_kv_layout(page_size, &static_capacity, &tokens_per_extent, &total_extents, &logical_capacity));
  return {
      static_cast<int64_t>(static_capacity),
      static_cast<int64_t>(tokens_per_extent),
      static_cast<int64_t>(total_extents),
      static_cast<int64_t>(logical_capacity)};
}

int64_t shared_hbm_set_kv_extents(int64_t target_extents) {
  uint64_t capacity = 0;
  check_shbm(shbm_set_kv_extent_count(target_extents, &capacity));
  return static_cast<int64_t>(capacity);
}

int64_t shared_hbm_kv_growable_extents() {
  uint64_t target_extents = 0;
  check_shbm(shbm_kv_growable_extents(&target_extents));
  return static_cast<int64_t>(target_extents);
}

std::vector<int64_t> shared_hbm_map_ann(int64_t owner, int64_t requested_bytes) {
  uint64_t address = 0;
  uint64_t mapped_bytes = 0;
  check_shbm(shbm_map_free_to_ann(owner, requested_bytes, &address, &mapped_bytes));
  return {static_cast<int64_t>(address), static_cast<int64_t>(mapped_bytes)};
}

void shared_hbm_release_ann(int64_t owner) {
  check_shbm(shbm_release_ann(owner));
}

void shared_hbm_grow_ann_index(int64_t target) {
  TORCH_CHECK(target >= 0, "negative ANN growth target");
  check_shbm(shbm_grow_ann_index(target));
}

int64_t shared_hbm_index_prefix_for_kv(int64_t target) {
  if (target < 0) throw std::invalid_argument("negative KV target");
  uint64_t retained = 0;
  check_shbm(shbm_index_prefix_for_kv(target, &retained));
  return static_cast<int64_t>(retained);
}

void shared_hbm_trim_ann_index(int64_t retained) {
  if (retained < 0) throw std::invalid_argument("negative ANN prefix");
  check_shbm(shbm_trim_ann_index(retained));
}

std::vector<int64_t> shared_hbm_snapshot() {
  SharedHBMSnapshot snapshot{};
  check_shbm(shbm_snapshot(&snapshot));
  return {
      static_cast<int64_t>(snapshot.total_bytes),
      static_cast<int64_t>(snapshot.chunk_bytes),
      static_cast<int64_t>(snapshot.free_bytes),
      static_cast<int64_t>(snapshot.kv_bytes),
      static_cast<int64_t>(snapshot.ann_scratch_bytes),
      static_cast<int64_t>(snapshot.ann_index_bytes),
      static_cast<int64_t>(snapshot.moving_bytes),
      static_cast<int64_t>(snapshot.kv_static_capacity_tokens),
      static_cast<int64_t>(snapshot.kv_tokens_per_extent),
      static_cast<int64_t>(snapshot.kv_total_extents),
      static_cast<int64_t>(snapshot.kv_mapped_extents),
      static_cast<int64_t>(snapshot.kv_logical_capacity_tokens)};
}

std::vector<int64_t> shared_hbm_driver_stats() {
  SharedHBMDriverStats stats{};
  check_shbm(shbm_driver_stats(&stats));
  return {
      static_cast<int64_t>(stats.map_calls),
      static_cast<int64_t>(stats.unmap_calls),
      static_cast<int64_t>(stats.set_access_calls),
      static_cast<int64_t>(stats.set_access_bytes)};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("shared_hbm_create(int total_bytes, int chunk_bytes, int device) -> int[]");
  m.impl("shared_hbm_create", &shared_hbm_create);
  m.def("shared_hbm_destroy() -> ()");
  m.impl("shared_hbm_destroy", &shared_hbm_destroy);
  m.def(
      "shared_hbm_alloc_kv_tensor(int[] sizes, ScalarType dtype, "
      "int static_tokens, int max_tokens, int device) -> Tensor");
  m.impl("shared_hbm_alloc_kv_tensor", &shared_hbm_alloc_kv_tensor);
  m.def("shared_hbm_finalize_kv(int page_size) -> int[]");
  m.impl("shared_hbm_finalize_kv", &shared_hbm_finalize_kv);
  m.def("shared_hbm_set_kv_extents(int target_extents) -> int");
  m.impl("shared_hbm_set_kv_extents", &shared_hbm_set_kv_extents);
  m.def("shared_hbm_kv_growable_extents() -> int");
  m.impl("shared_hbm_kv_growable_extents", &shared_hbm_kv_growable_extents);
  m.def("shared_hbm_map_ann(int owner, int requested_bytes) -> int[]");
  m.impl("shared_hbm_map_ann", &shared_hbm_map_ann);
  m.def("shared_hbm_release_ann(int owner) -> ()");
  m.impl("shared_hbm_release_ann", &shared_hbm_release_ann);
  m.def("shared_hbm_index_prefix_for_kv(int target) -> int");
  m.impl("shared_hbm_index_prefix_for_kv", &shared_hbm_index_prefix_for_kv);
  m.def("shared_hbm_trim_ann_index(int retained) -> ()");
  m.def("shared_hbm_grow_ann_index(int target) -> ()");
  m.impl("shared_hbm_grow_ann_index", &shared_hbm_grow_ann_index);
  m.impl("shared_hbm_trim_ann_index", &shared_hbm_trim_ann_index);
  m.def("shared_hbm_snapshot() -> int[]");
  m.impl("shared_hbm_snapshot", &shared_hbm_snapshot);
  m.def("shared_hbm_driver_stats() -> int[]");
  m.impl("shared_hbm_driver_stats", &shared_hbm_driver_stats);
}
