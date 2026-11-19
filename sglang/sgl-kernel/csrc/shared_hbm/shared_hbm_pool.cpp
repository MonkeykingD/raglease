#include "shared_hbm_pool.h"

#include <cuda.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

uint64_t checked_lcm(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  const uint64_t divisor = std::gcd(a, b);
  if (a > std::numeric_limits<uint64_t>::max() / (b / divisor)) {
    throw std::overflow_error("shared HBM alignment LCM overflow");
  }
  return a * (b / divisor);
}

std::string cuda_error(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::ostringstream stream;
  stream << operation << " failed";
  if (name != nullptr) {
    stream << " (" << name << ")";
  }
  if (message != nullptr) {
    stream << ": " << message;
  }
  return stream.str();
}

void check_cuda(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(cuda_error(result, operation));
  }
}

class ScopedCudaContext {
 public:
  explicit ScopedCudaContext(CUcontext context) {
    CUcontext current = nullptr;
    check_cuda(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context) {
      check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~ScopedCudaContext() {
    if (pushed_) {
      CUcontext popped = nullptr;
      cuCtxPopCurrent(&popped);
    }
  }

 private:
  bool pushed_{false};
};

struct Chunk {
  CUmemGenericAllocationHandle handle{};
  SharedHBMOwner owner{SHBM_OWNER_FREE};
  CUdeviceptr mapped_address{0};
};

struct PrivateMapping {
  CUmemGenericAllocationHandle handle{};
  CUdeviceptr address{0};
  uint64_t bytes{0};
};

struct KVSegment {
  CUdeviceptr base{0};
  uint64_t reserved_bytes{0};
  uint64_t tensor_bytes{0};
  uint64_t token_stride{0};
  uint64_t static_tokens{0};
  uint64_t max_tokens{0};
  std::vector<PrivateMapping> private_mappings;
};

class SharedHBMPool {
 public:
  SharedHBMPool(int device, uint64_t requested_bytes, uint64_t requested_chunk_bytes) : device_ordinal_(device) {
    const char* ann_allocation_order = std::getenv("SGLANG_SHARED_HBM_ANN_ALLOCATION_ORDER");
    if (ann_allocation_order != nullptr) {
      const std::string order(ann_allocation_order);
      if (order == "tail") {
        ann_alloc_from_tail_ = true;
      } else if (order == "head") {
        ann_alloc_from_tail_ = false;
      } else {
        throw std::invalid_argument(
            "SGLANG_SHARED_HBM_ANN_ALLOCATION_ORDER must be "
            "'head' or 'tail'");
      }
    }

    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, device), "cuDeviceGet");

    CUcontext current = nullptr;
    check_cuda(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current == nullptr) {
      check_cuda(cuDevicePrimaryCtxRetain(&retained_primary_context_, device_), "cuDevicePrimaryCtxRetain");
      check_cuda(cuCtxSetCurrent(retained_primary_context_), "cuCtxSetCurrent");
      retained_primary_ = true;
    }
    check_cuda(cuCtxGetCurrent(&context_), "cuCtxGetCurrent");

    allocation_prop_ = {};
    allocation_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop_.location.id = device;
    // V1 is deliberately process-local (TP=PP=1), so requesting an
    // exportable POSIX handle only narrows driver compatibility without
    // providing any benefit.
    allocation_prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t minimum_granularity = 0;
    check_cuda(
        cuMemGetAllocationGranularity(&minimum_granularity, &allocation_prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");

    const uint64_t default_chunk = 2ULL << 20;
    chunk_bytes_ = align_up(
        std::max<uint64_t>(requested_chunk_bytes == 0 ? default_chunk : requested_chunk_bytes, minimum_granularity),
        minimum_granularity);
    total_bytes_ = requested_bytes / chunk_bytes_ * chunk_bytes_;
    if (total_bytes_ == 0) {
      throw std::invalid_argument("shared HBM pool is smaller than one allocation chunk");
    }

    const uint64_t chunk_count = total_bytes_ / chunk_bytes_;
    try {
      chunks_.reserve(chunk_count);
      for (uint64_t i = 0; i < chunk_count; ++i) {
        Chunk chunk;
        check_cuda(cuMemCreate(&chunk.handle, chunk_bytes_, &allocation_prop_, 0), "cuMemCreate(shared chunk)");
        chunks_.push_back(chunk);
      }

      reserve_region(total_bytes_, &scratch_base_);
      reserve_region(total_bytes_, &index_base_);
    } catch (...) {
      if (scratch_base_ != 0) {
        cuMemAddressFree(scratch_base_, total_bytes_);
      }
      if (index_base_ != 0) {
        cuMemAddressFree(index_base_, total_bytes_);
      }
      for (auto& chunk : chunks_) {
        if (chunk.handle != 0) {
          cuMemRelease(chunk.handle);
        }
      }
      if (retained_primary_) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_);
      }
      throw;
    }
  }

  ~SharedHBMPool() {
    try {
      ScopedCudaContext context(context_);
      release_ann_locked(SHBM_OWNER_ANN_SCRATCH);
      release_ann_locked(SHBM_OWNER_ANN_INDEX);
      set_kv_extent_count_locked(0);

      for (auto& segment : kv_segments_) {
        for (auto& mapping : segment.private_mappings) {
          if (mapping.address != 0) {
            cuMemUnmap(mapping.address, mapping.bytes);
          }
          if (mapping.handle != 0) {
            cuMemRelease(mapping.handle);
          }
        }
        if (segment.base != 0) {
          cuMemAddressFree(segment.base, segment.reserved_bytes);
        }
      }

      if (scratch_base_ != 0) {
        cuMemAddressFree(scratch_base_, total_bytes_);
      }
      if (index_base_ != 0) {
        cuMemAddressFree(index_base_, total_bytes_);
      }
      for (auto& chunk : chunks_) {
        if (chunk.mapped_address != 0) {
          cuMemUnmap(chunk.mapped_address, chunk_bytes_);
        }
        if (chunk.handle != 0) {
          cuMemRelease(chunk.handle);
        }
      }
      if (retained_primary_) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_);
      }
    } catch (...) {
      // Destructors cannot report driver teardown errors safely.
    }
  }

  uint64_t alloc_kv_segment(uint64_t tensor_bytes, uint64_t token_stride, uint64_t static_tokens, uint64_t max_tokens) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    if (kv_finalized_) {
      throw std::logic_error("cannot register KV segments after layout finalization");
    }
    if (token_stride == 0 || tensor_bytes == 0) {
      throw std::invalid_argument("invalid zero-sized KV segment");
    }
    const uint64_t reserved = align_up(tensor_bytes, chunk_bytes_);
    CUdeviceptr address = 0;
    reserve_region(reserved, &address);
    KVSegment segment{address, reserved, tensor_bytes, token_stride, static_tokens, max_tokens, {}};
    try {
      // PyTorch validates the pointer's CUDA memory type while constructing an
      // external tensor.  A merely reserved VMM address is reported as
      // unregistered host memory, so map one chunk before returning it.  The
      // chunk is not temporary: finalize_kv_layout() folds it into the private
      // static KV prefix and maps only the remaining prefix, if any.
      map_private_prefix(segment, chunk_bytes_);
      kv_segments_.push_back(std::move(segment));
    } catch (...) {
      for (auto& mapping : segment.private_mappings) {
        if (mapping.address != 0) {
          cuMemUnmap(mapping.address, mapping.bytes);
        }
        if (mapping.handle != 0) {
          cuMemRelease(mapping.handle);
        }
      }
      cuMemAddressFree(address, reserved);
      throw;
    }
    return address;
  }

  void finalize_kv_layout(
      uint64_t page_size,
      uint64_t* static_capacity,
      uint64_t* tokens_per_extent,
      uint64_t* total_extents,
      uint64_t* logical_capacity) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    if (kv_finalized_) {
      write_layout(static_capacity, tokens_per_extent, total_extents, logical_capacity);
      return;
    }
    if (kv_segments_.empty()) {
      throw std::logic_error("no KV segments were registered");
    }
    page_size = std::max<uint64_t>(1, page_size);

    uint64_t token_quantum = page_size;
    uint64_t configured_static_tokens = kv_segments_.front().static_tokens;
    uint64_t configured_max_tokens = kv_segments_.front().max_tokens;
    for (const auto& segment : kv_segments_) {
      if (segment.static_tokens != configured_static_tokens || segment.max_tokens != configured_max_tokens) {
        throw std::invalid_argument("all shared-HBM KV segments must use identical token bounds");
      }
      const uint64_t stride_quantum = chunk_bytes_ / std::gcd(chunk_bytes_, segment.token_stride);
      token_quantum = checked_lcm(token_quantum, stride_quantum);
    }

    // Entry zero is the padded/dummy KV slot. Align the first shared entry so
    // every per-layer tensor offset is a whole VMM chunk boundary.
    shared_start_entry_ = align_up(configured_static_tokens + 1, token_quantum);
    kv_static_capacity_tokens_ = shared_start_entry_ - 1;
    kv_tokens_per_extent_ = token_quantum;

    chunks_per_kv_extent_ = 0;
    for (auto& segment : kv_segments_) {
      const uint64_t private_bytes = shared_start_entry_ * segment.token_stride;
      if (private_bytes > segment.tensor_bytes) {
        throw std::invalid_argument("static KV prefix exceeds reserved tensor size");
      }
      map_private_prefix(segment, private_bytes);

      const uint64_t per_extent_bytes = kv_tokens_per_extent_ * segment.token_stride;
      if (per_extent_bytes % chunk_bytes_ != 0) {
        throw std::logic_error("KV extent is not aligned to the physical chunk size");
      }
      chunks_per_kv_extent_ += per_extent_bytes / chunk_bytes_;
    }
    if (chunks_per_kv_extent_ == 0) {
      throw std::logic_error("computed zero chunks per KV extent");
    }

    kv_total_extents_ = chunks_.size() / chunks_per_kv_extent_;
    const uint64_t capacity_from_pool = kv_total_extents_ * kv_tokens_per_extent_;
    kv_logical_capacity_tokens_ = std::min(configured_max_tokens, kv_static_capacity_tokens_ + capacity_from_pool);
    kv_total_extents_ = (kv_logical_capacity_tokens_ - kv_static_capacity_tokens_) / kv_tokens_per_extent_;
    kv_finalized_ = true;
    write_layout(static_capacity, tokens_per_extent, total_extents, logical_capacity);
  }

  uint64_t set_kv_extent_count(uint64_t target) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    return set_kv_extent_count_locked(target);
  }

  uint64_t kv_growable_extents() const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!kv_finalized_) {
      throw std::logic_error("KV layout has not been finalized");
    }

    uint64_t target = kv_mapped_extents_;
    for (; target < kv_total_extents_; ++target) {
      const uint64_t chunk_begin = target * chunks_per_kv_extent_;
      const uint64_t chunk_end = chunk_begin + chunks_per_kv_extent_;
      const bool extent_is_free =
          std::all_of(chunks_.begin() + chunk_begin, chunks_.begin() + chunk_end, [](const Chunk& chunk) {
            return chunk.owner == SHBM_OWNER_FREE && chunk.mapped_address == 0;
          });
      if (!extent_is_free) {
        break;
      }
    }
    return target;
  }

  std::pair<uint64_t, uint64_t> map_free_to_ann(SharedHBMOwner owner, uint64_t requested_bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    if (owner != SHBM_OWNER_ANN_SCRATCH && owner != SHBM_OWNER_ANN_INDEX) {
      throw std::invalid_argument("invalid ANN owner");
    }
    release_ann_locked(owner);
    const uint64_t available_chunks = std::count_if(
        chunks_.begin(), chunks_.end(), [](const Chunk& chunk) { return chunk.owner == SHBM_OWNER_FREE; });
    uint64_t requested_chunks =
        requested_bytes == 0
            ? available_chunks
            : std::min<uint64_t>(available_chunks, (requested_bytes + chunk_bytes_ - 1) / chunk_bytes_);
    CUdeviceptr base = owner == SHBM_OWNER_ANN_SCRATCH ? scratch_base_ : index_base_;
    uint64_t mapped = 0;
    std::vector<Chunk*> newly_mapped;
    try {
      const auto map_candidate = [&](Chunk& chunk) {
        if (mapped >= requested_chunks) {
          return;
        }
        if (chunk.owner != SHBM_OWNER_FREE) {
          return;
        }
        const CUdeviceptr target = base + mapped * chunk_bytes_;
        map_chunk_deferred_access(chunk, target);
        newly_mapped.push_back(&chunk);
        ++mapped;
      };
      if (ann_alloc_from_tail_) {
        // Preserve the low physical-chunk prefix for lease-aware KV growth.
        // ANN still receives a contiguous virtual arena; physical handle order
        // is intentionally opaque to Faiss.
        for (auto iterator = chunks_.rbegin(); iterator != chunks_.rend() && mapped < requested_chunks; ++iterator) {
          map_candidate(*iterator);
        }
      } else {
        // Reproduce the original allocation order for controlled ablations.
        for (auto& chunk : chunks_) {
          map_candidate(chunk);
          if (mapped >= requested_chunks) {
            break;
          }
        }
      }
      if (mapped > 0) {
        set_access(base, mapped * chunk_bytes_);
      }
      for (auto* chunk : newly_mapped) {
        chunk->owner = owner;
      }
    } catch (...) {
      for (auto iterator = newly_mapped.rbegin(); iterator != newly_mapped.rend(); ++iterator) {
        try {
          unmap_chunk(**iterator);
        } catch (...) {
        }
      }
      throw;
    }
    return {base, mapped * chunk_bytes_};
  }

  uint64_t index_prefix_for_kv(uint64_t target) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!kv_finalized_ || target < kv_mapped_extents_ || target > kv_total_extents_) {
      throw std::invalid_argument("invalid KV trim target");
    }
    uint64_t retained = 0;
    for (const auto& chunk : chunks_) {
      if (chunk.owner == SHBM_OWNER_ANN_INDEX) retained += chunk_bytes_;
    }
    // KV extents use physical handle indices, not ANN virtual offsets. With
    // tail allocation these conflicting handles map to the ANN virtual tail.
    for (uint64_t i = kv_mapped_extents_ * chunks_per_kv_extent_;
         i < target * chunks_per_kv_extent_; ++i) {
      const auto& chunk = chunks_[i];
      if (chunk.owner == SHBM_OWNER_ANN_INDEX) {
        retained = std::min<uint64_t>(retained, chunk.mapped_address - index_base_);
      } else if (chunk.owner != SHBM_OWNER_FREE && chunk.owner != SHBM_OWNER_ANN_SCRATCH) {
        throw std::logic_error("KV trim target contains an unavailable owner");
      }
    }
    return retained;
  }

  void trim_ann_index(uint64_t retained) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    uint64_t mapped = 0;
    for (const auto& chunk : chunks_) {
      if (chunk.owner == SHBM_OWNER_ANN_INDEX) mapped += chunk_bytes_;
    }
    if (retained > mapped || retained % chunk_bytes_ != 0) {
      throw std::invalid_argument("ANN trim must shrink at a physical chunk boundary");
    }
    // Validate before any unmap; holes indicate a failed transition and must
    // never be silently interpreted as a valid contiguous arena.
    std::vector<Chunk*> by_offset(mapped / chunk_bytes_, nullptr);
    for (auto& chunk : chunks_) {
      if (chunk.owner != SHBM_OWNER_ANN_INDEX) continue;
      const uint64_t offset = chunk.mapped_address - index_base_;
      if (offset >= mapped || offset % chunk_bytes_ != 0 || by_offset[offset / chunk_bytes_] != nullptr) {
        throw std::logic_error("ANN index mapping is not a contiguous prefix");
      }
      by_offset[offset / chunk_bytes_] = &chunk;
    }
    // Descending VA keeps a contiguous surviving prefix even if CUDA fails.
    for (uint64_t count = by_offset.size(); count > retained / chunk_bytes_; --count) {
      unmap_chunk(*by_offset[count - 1]);
    }
  }

  void grow_ann_index(uint64_t target) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    uint64_t mapped = 0;
    for (const auto& c : chunks_) if (c.owner == SHBM_OWNER_ANN_INDEX) mapped += chunk_bytes_;
    if (target < mapped || target > total_bytes_ || target % chunk_bytes_) {
      throw std::invalid_argument("ANN growth requires an aligned nonshrinking target");
    }
    std::vector<bool> occupied(mapped / chunk_bytes_, false);
    for (const auto& c : chunks_) {
      if (c.owner != SHBM_OWNER_ANN_INDEX) continue;
      const uint64_t offset = c.mapped_address - index_base_;
      if (offset >= mapped || offset % chunk_bytes_ || occupied[offset / chunk_bytes_])
        throw std::logic_error("ANN growth requires a contiguous prefix");
      occupied[offset / chunk_bytes_] = true;
    }
    std::vector<Chunk*> candidates;
    for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it)
      if (it->owner == SHBM_OWNER_FREE && it->mapped_address == 0) candidates.push_back(&*it);
    const uint64_t needed = (target - mapped) / chunk_bytes_;
    if (candidates.size() < needed) throw std::runtime_error("Insufficient free HBM for ANN growth");
    std::vector<Chunk*> added;
    try {
      for (uint64_t i = 0; i < needed; ++i) {
        map_chunk_deferred_access(*candidates[i], index_base_ + mapped + i * chunk_bytes_);
        added.push_back(candidates[i]);
      }
      if (target > mapped) set_access(index_base_ + mapped, target - mapped);
      for (auto* c : added) c->owner = SHBM_OWNER_ANN_INDEX;
    } catch (...) {
      for (auto it = added.rbegin(); it != added.rend(); ++it) {
        try { unmap_chunk(**it); } catch (...) {}
      }
      throw;
    }
  }

  void release_ann(SharedHBMOwner owner) {
    std::lock_guard<std::mutex> guard(mutex_);
    ScopedCudaContext context(context_);
    release_ann_locked(owner);
  }

  std::pair<uint64_t, uint64_t> ann_region(SharedHBMOwner owner) const {
    if (owner == SHBM_OWNER_ANN_SCRATCH) {
      return {scratch_base_, total_bytes_};
    }
    if (owner == SHBM_OWNER_ANN_INDEX) {
      return {index_base_, total_bytes_};
    }
    throw std::invalid_argument("invalid ANN owner");
  }

  SharedHBMSnapshot snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    SharedHBMSnapshot result{};
    result.total_bytes = total_bytes_;
    result.chunk_bytes = chunk_bytes_;
    for (const auto& chunk : chunks_) {
      uint64_t* value = nullptr;
      switch (chunk.owner) {
        case SHBM_OWNER_FREE:
          value = &result.free_bytes;
          break;
        case SHBM_OWNER_KV_CACHE:
          value = &result.kv_bytes;
          break;
        case SHBM_OWNER_ANN_SCRATCH:
          value = &result.ann_scratch_bytes;
          break;
        case SHBM_OWNER_ANN_INDEX:
          value = &result.ann_index_bytes;
          break;
        case SHBM_OWNER_MOVING:
          value = &result.moving_bytes;
          break;
      }
      if (value != nullptr) {
        *value += chunk_bytes_;
      }
    }
    result.kv_static_capacity_tokens = kv_static_capacity_tokens_;
    result.kv_tokens_per_extent = kv_tokens_per_extent_;
    result.kv_total_extents = kv_total_extents_;
    result.kv_mapped_extents = kv_mapped_extents_;
    result.kv_logical_capacity_tokens = kv_logical_capacity_tokens_;
    return result;
  }

  SharedHBMDriverStats driver_stats() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return {
        map_calls_,
        unmap_calls_,
        set_access_calls_,
        set_access_bytes_,
    };
  }

 private:
  void reserve_region(uint64_t bytes, CUdeviceptr* address) {
    check_cuda(cuMemAddressReserve(address, bytes, chunk_bytes_, 0, 0), "cuMemAddressReserve");
  }

  void set_access(CUdeviceptr address, uint64_t bytes) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ++set_access_calls_;
    set_access_bytes_ += bytes;
    check_cuda(cuMemSetAccess(address, bytes, &access, 1), "cuMemSetAccess");
  }

  void map_chunk_deferred_access(Chunk& chunk, CUdeviceptr target) {
    if (chunk.owner != SHBM_OWNER_FREE || chunk.mapped_address != 0) {
      throw std::logic_error("attempted to map a non-free shared chunk");
    }
    chunk.owner = SHBM_OWNER_MOVING;
    try {
      ++map_calls_;
      check_cuda(cuMemMap(target, chunk_bytes_, 0, chunk.handle, 0), "cuMemMap");
    } catch (...) {
      chunk.owner = SHBM_OWNER_FREE;
      throw;
    }
    chunk.mapped_address = target;
  }

  void map_chunk(Chunk& chunk, CUdeviceptr target, SharedHBMOwner owner) {
    map_chunk_deferred_access(chunk, target);
    try {
      set_access(target, chunk_bytes_);
    } catch (...) {
      try {
        unmap_chunk(chunk);
      } catch (...) {
      }
      throw;
    }
    chunk.owner = owner;
  }

  void unmap_chunk(Chunk& chunk) {
    if (chunk.mapped_address == 0) {
      chunk.owner = SHBM_OWNER_FREE;
      return;
    }
    const SharedHBMOwner previous_owner = chunk.owner;
    chunk.owner = SHBM_OWNER_MOVING;
    try {
      ++unmap_calls_;
      check_cuda(cuMemUnmap(chunk.mapped_address, chunk_bytes_), "cuMemUnmap");
    } catch (...) {
      chunk.owner = previous_owner;
      throw;
    }
    chunk.mapped_address = 0;
    chunk.owner = SHBM_OWNER_FREE;
  }

  void map_private_prefix(KVSegment& segment, uint64_t bytes) {
    if (bytes == 0) {
      return;
    }
    if (bytes % chunk_bytes_ != 0) {
      throw std::logic_error("private KV prefix must be chunk aligned");
    }
    uint64_t mapped_prefix_bytes = 0;
    for (const auto& mapping : segment.private_mappings) {
      if (mapping.address != segment.base + mapped_prefix_bytes) {
        throw std::logic_error("private KV mappings are not a contiguous prefix");
      }
      mapped_prefix_bytes += mapping.bytes;
    }
    if (mapped_prefix_bytes > bytes) {
      throw std::logic_error("cannot shrink the private KV prefix");
    }
    if (mapped_prefix_bytes == bytes) {
      return;
    }

    const uint64_t remaining_bytes = bytes - mapped_prefix_bytes;
    const CUdeviceptr target = segment.base + mapped_prefix_bytes;
    // A single allocation per K/V layer segment avoids creating tens of
    // thousands of driver handles for a large static KV prefix.  There can be
    // one additional bootstrap allocation created before tensor wrapping.
    CUmemGenericAllocationHandle handle{};
    check_cuda(cuMemCreate(&handle, remaining_bytes, &allocation_prop_, 0), "cuMemCreate(private KV)");
    try {
      ++map_calls_;
      check_cuda(cuMemMap(target, remaining_bytes, 0, handle, 0), "cuMemMap(private KV)");
      try {
        set_access(target, remaining_bytes);
      } catch (...) {
        ++unmap_calls_;
        cuMemUnmap(target, remaining_bytes);
        throw;
      }
    } catch (...) {
      cuMemRelease(handle);
      throw;
    }
    try {
      segment.private_mappings.push_back({handle, target, remaining_bytes});
    } catch (...) {
      cuMemUnmap(target, remaining_bytes);
      cuMemRelease(handle);
      throw;
    }
  }

  void write_layout(
      uint64_t* static_capacity,
      uint64_t* tokens_per_extent,
      uint64_t* total_extents,
      uint64_t* logical_capacity) const {
    *static_capacity = kv_static_capacity_tokens_;
    *tokens_per_extent = kv_tokens_per_extent_;
    *total_extents = kv_total_extents_;
    *logical_capacity = kv_logical_capacity_tokens_;
  }

  uint64_t set_kv_extent_count_locked(uint64_t target) {
    if (!kv_finalized_) {
      throw std::logic_error("KV layout has not been finalized");
    }
    if (target > kv_total_extents_) {
      throw std::out_of_range("requested too many KV extents");
    }
    if (target == kv_mapped_extents_) {
      return kv_static_capacity_tokens_ + kv_mapped_extents_ * kv_tokens_per_extent_;
    }

    if (target < kv_mapped_extents_) {
      for (uint64_t extent = kv_mapped_extents_; extent > target; --extent) {
        const uint64_t chunk_begin = (extent - 1) * chunks_per_kv_extent_;
        const uint64_t chunk_end = chunk_begin + chunks_per_kv_extent_;
        for (uint64_t chunk_id = chunk_begin; chunk_id < chunk_end; ++chunk_id) {
          auto& chunk = chunks_[chunk_id];
          if (chunk.owner != SHBM_OWNER_KV_CACHE) {
            throw std::logic_error("KV extent contains a chunk not owned by KV");
          }
          unmap_chunk(chunk);
        }
      }
      kv_mapped_extents_ = target;
    } else {
      const uint64_t previous_mapped_extents = kv_mapped_extents_;
      for (uint64_t extent = previous_mapped_extents; extent < target; ++extent) {
        const uint64_t chunk_begin = extent * chunks_per_kv_extent_;
        const uint64_t chunk_end = chunk_begin + chunks_per_kv_extent_;
        for (uint64_t chunk_id = chunk_begin; chunk_id < chunk_end; ++chunk_id) {
          if (chunks_[chunk_id].owner != SHBM_OWNER_FREE ||
              chunks_[chunk_id].mapped_address != 0) {
            throw std::runtime_error("KV growth requires chunks currently leased to ANN");
          }
        }
      }

      // cuMemMap still operates on individual physical chunks, but access
      // permissions apply to virtual ranges.  Map the complete contiguous
      // range for each K/V segment first, then use one cuMemSetAccess call for
      // all newly added extents in that segment.
      std::vector<Chunk*> newly_mapped;
      newly_mapped.reserve(
          static_cast<size_t>((target - previous_mapped_extents) * chunks_per_kv_extent_));
      uint64_t segment_chunk_offset = 0;
      try {
        for (auto& segment : kv_segments_) {
          const uint64_t bytes_per_extent = kv_tokens_per_extent_ * segment.token_stride;
          const uint64_t segment_chunks = bytes_per_extent / chunk_bytes_;
          for (uint64_t extent = previous_mapped_extents; extent < target; ++extent) {
            const uint64_t chunk_begin =
                extent * chunks_per_kv_extent_ + segment_chunk_offset;
            const CUdeviceptr segment_target =
                segment.base + (shared_start_entry_ + extent * kv_tokens_per_extent_) * segment.token_stride;
            for (uint64_t i = 0; i < segment_chunks; ++i) {
              auto& chunk = chunks_[chunk_begin + i];
              map_chunk_deferred_access(chunk, segment_target + i * chunk_bytes_);
              newly_mapped.push_back(&chunk);
            }
          }
          const CUdeviceptr access_begin =
              segment.base +
              (shared_start_entry_ + previous_mapped_extents * kv_tokens_per_extent_) *
                  segment.token_stride;
          set_access(
              access_begin,
              (target - previous_mapped_extents) * bytes_per_extent);
          segment_chunk_offset += segment_chunks;
        }
        if (segment_chunk_offset != chunks_per_kv_extent_) {
          throw std::logic_error("KV extent chunk accounting mismatch");
        }
        for (auto* chunk : newly_mapped) {
          chunk->owner = SHBM_OWNER_KV_CACHE;
        }
      } catch (...) {
        for (auto iterator = newly_mapped.rbegin(); iterator != newly_mapped.rend(); ++iterator) {
          try {
            unmap_chunk(**iterator);
          } catch (...) {
          }
        }
        throw;
      }
      kv_mapped_extents_ = target;
    }

    return kv_static_capacity_tokens_ + kv_mapped_extents_ * kv_tokens_per_extent_;
  }

  void release_ann_locked(SharedHBMOwner owner) {
    if (owner != SHBM_OWNER_ANN_SCRATCH && owner != SHBM_OWNER_ANN_INDEX) {
      throw std::invalid_argument("invalid ANN owner");
    }
    for (auto& chunk : chunks_) {
      if (chunk.owner == owner) {
        unmap_chunk(chunk);
      }
    }
  }

  int device_ordinal_{0};
  CUdevice device_{};
  CUcontext context_{nullptr};
  CUcontext retained_primary_context_{nullptr};
  bool retained_primary_{false};
  CUmemAllocationProp allocation_prop_{};
  uint64_t total_bytes_{0};
  uint64_t chunk_bytes_{0};
  CUdeviceptr scratch_base_{0};
  CUdeviceptr index_base_{0};
  std::vector<Chunk> chunks_;
  std::vector<KVSegment> kv_segments_;

  bool kv_finalized_{false};
  uint64_t shared_start_entry_{0};
  uint64_t kv_static_capacity_tokens_{0};
  uint64_t kv_tokens_per_extent_{0};
  uint64_t chunks_per_kv_extent_{0};
  uint64_t kv_total_extents_{0};
  uint64_t kv_mapped_extents_{0};
  uint64_t kv_logical_capacity_tokens_{0};
  uint64_t map_calls_{0};
  uint64_t unmap_calls_{0};
  uint64_t set_access_calls_{0};
  uint64_t set_access_bytes_{0};
  bool ann_alloc_from_tail_{true};
  mutable std::mutex mutex_;
};

std::mutex g_pool_mutex;
std::unique_ptr<SharedHBMPool> g_pool;

template <typename Fn>
int guarded(Fn&& fn) {
  try {
    fn();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown shared HBM failure";
    return -1;
  }
}

SharedHBMPool& pool() {
  if (!g_pool) {
    throw std::logic_error("shared HBM pool has not been created");
  }
  return *g_pool;
}

SharedHBMOwner parse_owner(int owner) {
  if (owner == SHBM_OWNER_ANN_SCRATCH) {
    return SHBM_OWNER_ANN_SCRATCH;
  }
  if (owner == SHBM_OWNER_ANN_INDEX) {
    return SHBM_OWNER_ANN_INDEX;
  }
  throw std::invalid_argument("owner is not an ANN memory class");
}

}  // namespace

extern "C" {

int shbm_create(int device, uint64_t requested_bytes, uint64_t requested_chunk_bytes) {
  std::lock_guard<std::mutex> guard(g_pool_mutex);
  return guarded([&]() {
    if (g_pool) {
      throw std::logic_error("shared HBM pool already exists");
    }
    g_pool = std::make_unique<SharedHBMPool>(device, requested_bytes, requested_chunk_bytes);
  });
}

int shbm_destroy() {
  std::lock_guard<std::mutex> guard(g_pool_mutex);
  return guarded([&]() { g_pool.reset(); });
}

const char* shbm_last_error() {
  return g_last_error.c_str();
}

int shbm_alloc_kv_segment(
    uint64_t tensor_bytes,
    uint64_t token_stride_bytes,
    uint64_t static_tokens,
    uint64_t max_tokens,
    uint64_t* address_out) {
  return guarded([&]() {
    if (address_out == nullptr) {
      throw std::invalid_argument("address_out is null");
    }
    *address_out = pool().alloc_kv_segment(tensor_bytes, token_stride_bytes, static_tokens, max_tokens);
  });
}

int shbm_finalize_kv_layout(
    uint64_t page_size,
    uint64_t* static_capacity_tokens,
    uint64_t* tokens_per_extent,
    uint64_t* total_extents,
    uint64_t* logical_capacity_tokens) {
  return guarded([&]() {
    if (static_capacity_tokens == nullptr || tokens_per_extent == nullptr || total_extents == nullptr ||
        logical_capacity_tokens == nullptr) {
      throw std::invalid_argument("KV layout output pointer is null");
    }
    pool().finalize_kv_layout(
        page_size, static_capacity_tokens, tokens_per_extent, total_extents, logical_capacity_tokens);
  });
}

int shbm_set_kv_extent_count(uint64_t target_extents, uint64_t* mapped_capacity_tokens) {
  return guarded([&]() {
    if (mapped_capacity_tokens == nullptr) {
      throw std::invalid_argument("mapped_capacity_tokens is null");
    }
    *mapped_capacity_tokens = pool().set_kv_extent_count(target_extents);
  });
}

int shbm_kv_growable_extents(uint64_t* target_extents_out) {
  return guarded([&]() {
    if (target_extents_out == nullptr) {
      throw std::invalid_argument("target_extents_out is null");
    }
    *target_extents_out = pool().kv_growable_extents();
  });
}

int shbm_map_free_to_ann(int owner, uint64_t requested_bytes, uint64_t* address_out, uint64_t* mapped_bytes_out) {
  return guarded([&]() {
    if (address_out == nullptr || mapped_bytes_out == nullptr) {
      throw std::invalid_argument("ANN mapping output pointer is null");
    }
    const auto mapping = pool().map_free_to_ann(parse_owner(owner), requested_bytes);
    *address_out = mapping.first;
    *mapped_bytes_out = mapping.second;
  });
}

int shbm_release_ann(int owner) {
  return guarded([&]() { pool().release_ann(parse_owner(owner)); });
}

int shbm_index_prefix_for_kv(uint64_t target, uint64_t* retained_bytes_out) {
  return guarded([&]() {
    if (!retained_bytes_out) throw std::invalid_argument("null trim output");
    *retained_bytes_out = pool().index_prefix_for_kv(target);
  });
}

int shbm_trim_ann_index(uint64_t retained_bytes) {
  return guarded([&]() { pool().trim_ann_index(retained_bytes); });
}

int shbm_grow_ann_index(uint64_t target_bytes) {
  return guarded([&]() { pool().grow_ann_index(target_bytes); });
}

int shbm_get_ann_region(int owner, uint64_t* address_out, uint64_t* reserved_bytes_out) {
  return guarded([&]() {
    if (address_out == nullptr || reserved_bytes_out == nullptr) {
      throw std::invalid_argument("ANN region output pointer is null");
    }
    const auto region = pool().ann_region(parse_owner(owner));
    *address_out = region.first;
    *reserved_bytes_out = region.second;
  });
}

int shbm_snapshot(SharedHBMSnapshot* snapshot_out) {
  return guarded([&]() {
    if (snapshot_out == nullptr) {
      throw std::invalid_argument("snapshot_out is null");
    }
    *snapshot_out = pool().snapshot();
  });
}

int shbm_driver_stats(SharedHBMDriverStats* stats_out) {
  return guarded([&]() {
    if (stats_out == nullptr) {
      throw std::invalid_argument("stats_out is null");
    }
    *stats_out = pool().driver_stats();
  });
}

}  // extern "C"
