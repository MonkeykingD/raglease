// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "cuda_utils.h"
#include "greenctx_stream.h"

static int CUDA_DRIVER_VERSION;
static std::mutex GREENCTX_STREAM_MAP_MUTEX;
static std::unordered_map<int64_t, CUcontext> GREENCTX_STREAM_TO_CONTEXT;

static CUstream int64_to_custream(int64_t stream) {
  return reinterpret_cast<CUstream>(static_cast<uintptr_t>(stream));
}

static int64_t ptr_to_int64(const void* ptr) {
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
}

using PFN_cuGreenCtxStreamCreate = CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int);

auto probe_cuGreenCtxStreamCreate() -> PFN_cuGreenCtxStreamCreate {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  const CUresult status =
      cuGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), CUDA_DRIVER_VERSION, 0, nullptr);
  if (status == CUDA_ERROR_NOT_FOUND) {
    return nullptr;
  }
  CUDA_DRV(status);
  return pfn;
}

static std::vector<int64_t> create_greenctx_stream_fallback(CUgreenCtx gctx[2]) {
  CUstream streamA, streamB;
  CUcontext ctx;

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamA, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreate(&streamB, CU_STREAM_NON_BLOCKING));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

inline void destroy_green_context(CUgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(cuGreenCtxDestroy(gctx));
}

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(CUgreenCtx gctx[2]) {
  // This symbol is introduced in CUDA 12.5
  const static auto pfn = probe_cuGreenCtxStreamCreate();
  if (!pfn) {
    TORCH_WARN("cuGreenCtxStreamCreate(cuda>=12.5) is not available, using fallback");
    return create_greenctx_stream_fallback(gctx);
  }

  CUstream streamA, streamB;
  CUDA_DRV(pfn(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(pfn(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  return {(int64_t)streamA, (int64_t)streamB};
}

static void register_greenctx_stream_context(CUstream stream, CUgreenCtx gctx) {
  CUcontext ctx;
  CUDA_DRV(cuCtxFromGreenCtx(&ctx, gctx));
  std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
  GREENCTX_STREAM_TO_CONTEXT[ptr_to_int64(stream)] = ctx;
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUDA_DRV(cuDriverGetVersion(&CUDA_DRIVER_VERSION));

  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];

  TORCH_CHECK(smA > 0 && smB > 0, "SM counts must be positive");

  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  const int smCountA = resources[0].sm.smCount;
  const int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> streams = create_greenctx_stream_direct_dynamic(gctx);
  register_greenctx_stream_context(int64_to_custream(streams[0]), gctx[0]);
  register_greenctx_stream_context(int64_to_custream(streams[1]), gctx[1]);

  destroy_green_context(gctx[2]);

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}

int64_t create_greenctx_stream_like(int64_t stream, int64_t priority) {
  CUcontext ctx;
  {
    std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
    auto it = GREENCTX_STREAM_TO_CONTEXT.find(stream);
    TORCH_CHECK(
        it != GREENCTX_STREAM_TO_CONTEXT.end(),
        "No GreenContext CUDA context registered for stream ",
        stream);
    ctx = it->second;
  }

  CUstream new_stream;
  CUDA_DRV(cuCtxPushCurrent(ctx));
  CUDA_DRV(cuStreamCreateWithPriority(
      &new_stream, CU_STREAM_NON_BLOCKING, static_cast<int>(priority)));
  CUDA_DRV(cuCtxPopCurrent(nullptr));

  {
    std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
    GREENCTX_STREAM_TO_CONTEXT[ptr_to_int64(new_stream)] = ctx;
  }

  return ptr_to_int64(new_stream);
}

bool has_greenctx_context_for_stream(int64_t stream) {
  std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
  return GREENCTX_STREAM_TO_CONTEXT.find(stream) != GREENCTX_STREAM_TO_CONTEXT.end();
}

int64_t get_greenctx_context_for_stream(int64_t stream) {
  std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
  auto it = GREENCTX_STREAM_TO_CONTEXT.find(stream);
  TORCH_CHECK(
      it != GREENCTX_STREAM_TO_CONTEXT.end(),
      "No GreenContext CUDA context registered for stream ",
      stream);
  return ptr_to_int64(it->second);
}

void push_greenctx_context_for_stream(int64_t stream) {
  CUcontext ctx;
  {
    std::lock_guard<std::mutex> lock(GREENCTX_STREAM_MAP_MUTEX);
    auto it = GREENCTX_STREAM_TO_CONTEXT.find(stream);
    TORCH_CHECK(
        it != GREENCTX_STREAM_TO_CONTEXT.end(),
        "No GreenContext CUDA context registered for stream ",
        stream);
    ctx = it->second;
  }
  CUDA_DRV(cuCtxPushCurrent(ctx));
}

void pop_greenctx_context() {
  CUcontext ctx = nullptr;
  CUDA_DRV(cuCtxPopCurrent(&ctx));
}
