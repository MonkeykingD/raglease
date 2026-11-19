# Shared HBM for KV cache and VectorLiteRAG

This experimental path reserves one process-local CUDA VMM pool before Faiss
loading and KV-cache profiling. Physical chunks are exclusively owned by
`FREE`, `KV_CACHE`, `ANN_SCRATCH`, or `ANN_INDEX`; ownership changes remap the
same CUDA allocation handles into stable KV, scratch, or IVF virtual ranges.

The first supported configurations are H100 and A100 80GB, one process,
TP=PP=1, MHA with page size 1, overlap scheduling disabled, and CUDA Graph
disabled. The A100 40GB SKU is deliberately rejected because the initial
10 GiB pool and SLO profile have not been validated at that capacity. Startup
fails if these constraints are not met. PD-Mux/GreenContext builds require a
CUDA toolkit and driver exposing the CUDA 12.4 Green Context APIs; the
`primary` ANN schedule does not partition SMs.

```bash
python -m sglang.launch_server \
  ... \
  --enable-rag-ann \
  --rag-ann-schedule primary \
  --rag-shared-hbm-mode scratch \
  --rag-shared-hbm-pool-gib 10 \
  --rag-shared-hbm-chunk-mib 8 \
  --rag-shared-hbm-lookahead-ms 50 \
  --rag-shared-hbm-min-residency-ms 10 \
  --rag-shared-hbm-cluster-heat-decay 0.98 \
  --rag-shared-hbm-cluster-miss-boost 4 \
  --rag-shared-hbm-cluster-promotion-hysteresis 1.15 \
  --rag-shared-hbm-kv-reserve-extents 2 \
  --rag-shared-hbm-async-map \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --page-size 1
```

`scratch` leases the pool between mapped KV extents and a strict external
Faiss temporary stack. A temporary-stack overflow cannot fall back to
`cudaMalloc`; admission uses a smaller scratch lease where possible and sends
the microbatch to the authoritative CPU index if it cannot be admitted.
Physical ANN pages are selected from the end of the pool, opposite the
low-to-high KV prefix. Before a KV handoff the native allocator reports the
largest prefix that remains entirely `FREE`. If the complete KV growth target
fits below that boundary, SGLang keeps the live Faiss arena mapped and skips
its stream fence and remap. The original fence-and-release path remains
mandatory when the two ownership ranges would overlap.

For controlled allocator ablations only,
`SGLANG_SHARED_HBM_ANN_ALLOCATION_ORDER=head` restores the original
low-to-high ANN physical-page order; `tail` is the production default. This
environment variable changes placement, not the contiguous ANN virtual
address exposed to Faiss.

Build the patched Faiss artifact against the allocator DSO installed by
`sgl-kernel`, so both sides retain one process-global ownership table:

```bash
cmake -S . -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_SHARED_HBM_POOL_LIBRARY=/path/to/sgl_kernel/libshared_hbm_pool.so
```

Keep the directory containing that DSO on `LD_LIBRARY_PATH` when importing
the patched Faiss package. In shared mode SGLang imports `sgl_kernel` first,
so the Faiss dependency resolves to the already-loaded allocator instance.

`ivf` additionally requires the VectorLite hot index and mapping options. GPU
codes and IDs are reloaded into the `ANN_INDEX` VMM range at IVF-cluster
granularity. The offline hot-cluster order seeds the policy; live probe
frequency and CPU-fallback misses then update an EWMA demand estimate. A
candidate's value per byte combines that demand with measured CPU-versus-GPU
probe latency and subtracts its measured reload cost. Minimum residency and a
candidate-to-victim hysteresis ratio prevent cluster churn. Evicted probes are
changed back to their original CPU centroid IDs and merged with resident GPU
results. Reload waits are allowed only when the current ANN deadline slack
covers reload p95, GPU search p95, and the configured safety margin.

At the candidates-ready boundary, the controller forecasts 50 ms of KV demand.
It first detaches idle Faiss scratch, then drains read-only IVF residency while
CPU cold search/merge can continue, maps the required KV prefix, zeros the new
range on a maintenance stream, and publishes those token slots only after the
event completes. Active KV is never lent out. SGLang's existing
`get_cpu_copy`/`load_cpu_copy` and layer-transfer synchronization remain
available for a future opt-in active-KV offload policy, but are deliberately
not used by the default policy.

The asynchronous mapper uses an architecture-aware first-stage default:
one extent on A100 80GB and two extents on H100. These values are selected
with `--rag-shared-hbm-map-stage-extents -1` (the default). The first-stage
value is a floor: immediate KV demand can raise it. In the current H100
mixed-load trace, both configured values 1 and 2 therefore mapped two extents;
their endpoint difference must not be interpreted as a stage-size effect.
Pass `0` for a single monolithic map or a positive extent count to pin an
experiment to an explicit staging policy. Both the configured and resolved
values are exposed in the shared-HBM snapshot.

`--rag-shared-hbm-async-map` moves the blocking CUDA VMM
`cuMemMap`/`cuMemSetAccess` work to one dedicated host thread. The scheduler
continues serving existing decode work while mapping is in flight, but ANN
admission is held until the worker completes and the newly mapped KV range has
been zeroed and published. The option is off by default pending a full H100
repeat; it is validated as an opt-in path on A100 80GB. Its state and
submission/completion/error counts are included in the shared-HBM snapshot,
along with separate scheduler-enqueue, native-map, zero-enqueue, and
publication latency histograms.

The default physical ownership chunk is 8 MiB. On the validated 10 GiB H100
pool this reduced driver map calls by about 74% versus 2 MiB while retaining
22 independently reclaimable KV extents. Larger 16/32 MiB chunks remain
available for explicit experiments, but trade away adjustment granularity and
showed less stable native-map tail latency under mixed load.

`/get_internal_state` exposes the `rag_ann_runtime.shared_hbm` snapshot,
including owner bytes/chunks, transition and failure counts, handoff
percentiles, prediction error, CPU fallbacks, and IVF residency statistics.

For comparable H100 or A100 80GB runs:

```bash
python benchmark/pdmux/run_shared_hbm_h100.py \
  --config benchmark/pdmux/rag_scheduler_profile.example.json \
  --modes off scratch ivf
```

Use `run_shared_hbm_a100.py` for the same sweep on A100 80GB. Keep the
PD-Mux SM divisions valid for compute capability 8.x; the example 48/60 split
fits the A100's 108 SMs.
