#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define SHBM_EXPORT __declspec(dllexport)
#else
#define SHBM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

enum SharedHBMOwner : int32_t {
  SHBM_OWNER_FREE = 0,
  SHBM_OWNER_KV_CACHE = 1,
  SHBM_OWNER_ANN_SCRATCH = 2,
  SHBM_OWNER_ANN_INDEX = 3,
  SHBM_OWNER_MOVING = 4,
};

struct SharedHBMSnapshot {
  uint64_t total_bytes;
  uint64_t chunk_bytes;
  uint64_t free_bytes;
  uint64_t kv_bytes;
  uint64_t ann_scratch_bytes;
  uint64_t ann_index_bytes;
  uint64_t moving_bytes;
  uint64_t kv_static_capacity_tokens;
  uint64_t kv_tokens_per_extent;
  uint64_t kv_total_extents;
  uint64_t kv_mapped_extents;
  uint64_t kv_logical_capacity_tokens;
};

struct SharedHBMDriverStats {
  uint64_t map_calls;
  uint64_t unmap_calls;
  uint64_t set_access_calls;
  uint64_t set_access_bytes;
};

// The initial implementation intentionally owns one pool per serving process.
// This matches the supported TP=PP=1, non-overlap scheduler configuration.
SHBM_EXPORT int shbm_create(int device, uint64_t requested_bytes, uint64_t requested_chunk_bytes);
SHBM_EXPORT int shbm_destroy();
SHBM_EXPORT const char* shbm_last_error();

// Reserve one fixed-address CUDA tensor segment. Physical memory is attached
// only by shbm_finalize_kv_layout(), after all K/V segments are registered.
SHBM_EXPORT int shbm_alloc_kv_segment(
    uint64_t tensor_bytes,
    uint64_t token_stride_bytes,
    uint64_t static_tokens,
    uint64_t max_tokens,
    uint64_t* address_out);

// Returns static capacity, tokens/extent, total extents and logical capacity.
SHBM_EXPORT int shbm_finalize_kv_layout(
    uint64_t page_size,
    uint64_t* static_capacity_tokens,
    uint64_t* tokens_per_extent,
    uint64_t* total_extents,
    uint64_t* logical_capacity_tokens);

// Grow/shrink the mapped shared KV prefix. The caller must remove shrinking
// token ranges from the logical KV allocator before invoking this function.
SHBM_EXPORT int shbm_set_kv_extent_count(uint64_t target_extents, uint64_t* mapped_capacity_tokens);

// Return the largest KV extent count reachable using only the currently-free
// physical prefix. No ANN lease is reclaimed by this query.
SHBM_EXPORT int shbm_kv_growable_extents(uint64_t* target_extents_out);

// Map currently-free shared chunks contiguously into the ANN scratch or ANN
// index virtual region. Returns the stable region base and mapped byte count.
SHBM_EXPORT int
shbm_map_free_to_ann(int owner, uint64_t requested_bytes, uint64_t* address_out, uint64_t* mapped_bytes_out);
SHBM_EXPORT int shbm_release_ann(int owner);
// Read-only trim plan for a KV physical-prefix target. Scratch must be
// separately detached before KV growth; this query never releases ownership.
SHBM_EXPORT int shbm_index_prefix_for_kv(uint64_t target_extents, uint64_t* retained_bytes_out);
// Caller has drained all index readers, evicted every intersecting list and
// clipped the external allocator. Keep VA and physical handles of the prefix.
SHBM_EXPORT int shbm_trim_ann_index(uint64_t retained_bytes);
// Append free physical chunks without remapping the live index prefix.
SHBM_EXPORT int shbm_grow_ann_index(uint64_t target_bytes);
SHBM_EXPORT int shbm_get_ann_region(int owner, uint64_t* address_out, uint64_t* reserved_bytes_out);

SHBM_EXPORT int shbm_snapshot(SharedHBMSnapshot* snapshot_out);
SHBM_EXPORT int shbm_driver_stats(SharedHBMDriverStats* stats_out);

}  // extern "C"
