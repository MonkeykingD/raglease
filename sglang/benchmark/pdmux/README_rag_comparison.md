# VectorLiteRAG-Compatible RAG Evaluation

This directory contains an end-to-end RAG serving path and a paired evaluation
harness for comparing the modified SGLang runtime with VectorLiteRAG.

The comparison deliberately reuses VectorLiteRAG data artifacts and workload
semantics without reusing its runtime queues. The modified SGLang server owns
ANN admission, Faiss execution, document lookup, prompt construction, prefill,
and decode. The original VectorLiteRAG checkout remains an unchanged baseline.

## System boundary

```text
Canonical workload trace
          |
          +--> HTTP client --> SGLang /rag_generate
          |                    ANN scheduler -> Faiss -> docs -> prefill/decode
          |
          +--> trace adapter --> original VectorLiteRAG runtime
                                ANNSEngine -> DocRetriever -> LLMEngine
```

The query encoder is outside the measured path. This matches VectorLiteRAG's
artifact evaluation, which uses pre-encoded query vectors. Add an online
embedding service only as a separate experiment; otherwise the comparison is
not measuring the same pipeline.

## Shared data

The SGLang server can consume the existing VectorLiteRAG layout directly:

```text
$VLITE/database/
  wikiall/
    ivfpq.index
    ivffs.index
    test_qvec.fbin
    test_qids.ibin
    llama8b/1gpus/
      150_0_2048.index
      150_2048.imap
  query_database/
  text_database/
```

Use the same model, index artifacts, query vectors, `nprobe`, `topk`, prompt
format, output length, and GPU count for both systems. The default
`/rag_generate` prompt is `{context}`, matching VectorLiteRAG's document-only
prompt.

The hot-index data path follows VectorLiteRAG `PartitionedIndex`: its
offline profiler selects hot coarse centroids, `partition_ivf` copies those
inverted lists into a GPU shard, and the `.imap` stores int32 triples of
`(original centroid, shard id, shard-local centroid)`. At serving time SGLang
uses the complete `ivffs.index` HNSW quantizer once, routes hot probes to the
GPU shard and cold probes to the CPU FastScan index in parallel, then merges
the candidates by distance. This is a static offline partition, not an LRU or
online migration cache.

### Controlled hybrid scheduling policies

The offline profiler exposes the same static hot/cold retrieval backend under
two scheduler policies:

- `naive-hybrid` runs hot-index GPU ANN in the primary CUDA context without
  PD-Mux, Green Context partitioning, leases, handoff admission, or ANN credit
  limits. ANN and LLM streams compete freely for the GPU.
- `raglease-hybrid` runs the identical hot/cold search through the normal
  RAGLease policy, including the configured Green Context placement,
  dependency handoff, ANN credit, and Prefill lease controller.

Both policies load the same complete CPU FastScan index, GPU hot shard, and
`.imap`. Configure them with either the generic controlled-comparison names:

```json
{
  "paths": {
    "hybrid_cpu_index_path": "/path/to/ivffs.index",
    "hybrid_hot_index_path": "/path/to/hot.index",
    "hybrid_hot_index_map_path": "/path/to/hot.imap"
  }
}
```

or the existing `vectorlite_cpu_index_path`, `vectorlite_hot_index_path`, and
`vectorlite_hot_index_map_path` aliases. `runtime.hybrid_cpu_search_threads`
sets the common OpenMP capacity of the cold search. Static hybrid comparisons
force `rag_shared_hbm_mode=off` for both policies so online index migration
does not alter the residency decision. Use the separate shared-HBM experiment
when evaluating dynamic ownership between index pages and KV cache.

Generate missing hot artifacts with the original VectorLiteRAG profiler so
both systems consume the same partition decision:

```bash
cd /path/to/vector-lite-rag
python main.py \
  --model llama8b \
  --index wikiall \
  --gpu_type H100 \
  --num_gpus 1 \
  --search_slo 150 \
  --search_nprobe 2048 \
  --is_profiling
```

## 1. Generate one canonical workload

Run this once per repetition/seed and reuse the resulting file for every
scheduler configuration:

```bash
python benchmark/pdmux/rag_workload.py \
  --output logs/rag_compare/seed1/workload.jsonl \
  --num-requests 500 \
  --ramp-up-requests 100 \
  --warmup-requests 16 \
  --request-rate 8 \
  --output-len 256 \
  --topk 25 \
  --query-count 10000 \
  --seed 1
```

The JSONL trace fixes absolute arrival offsets, request IDs, query rows,
ramp-up membership, output length, and top-k. A SHA-256 fingerprint is carried
into both result formats and checked by the comparison tool.

## 2. Start the modified SGLang RAG server

The exact model and PD-Mux configuration are machine-specific. A typical
frontend-GC launch is:

```bash
VLITE=/path/to/vector-lite-rag
DATA=$VLITE/database
DATASET=wikiall
VLITE_MODEL=llama8b
SEARCH_SLO=150
NPROBE=2048
SHARD_DIR="$DATA/$DATASET/$VLITE_MODEL/1gpus"

export FAISS_EGG_DIR=/path/to/vectorlite/faiss-9.9.9-py3.10.egg
# Set LD_PRELOAD to the matching libmkl_rt.so when the custom Faiss build
# requires it in your environment.
# VectorLite's default index uses 4-bit PQ. Use the same Faiss build: its
# GpuClonerOptions must expose interleavedLayout for the hot shard clone.

PYTHONPATH=python python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-pdmux \
  --pdmux-config-path benchmark/pdmux/pdmux_adaptive_fair_decode60_t24.yaml \
  --sm-group-num 3 \
  --enable-rag-ann \
  --rag-index-path "$DATA/$DATASET/ivffs.index" \
  --rag-hot-index-path "$SHARD_DIR/${SEARCH_SLO}_0_${NPROBE}.index" \
  --rag-hot-index-map-path "$SHARD_DIR/${SEARCH_SLO}_${NPROBE}.imap" \
  --rag-hot-index-shard-id 0 \
  --rag-ann-schedule decode-gc \
  --rag-ann-max-batch-size 1 \
  --rag-index-nprobe "$NPROBE" \
  --rag-index-topk 25 \
  --rag-query-vec-path "$DATA/$DATASET/test_qvec.fbin" \
  --rag-query-ids-path "$DATA/$DATASET/test_qids.ibin" \
  --rag-query-db-path "$DATA/query_database" \
  --rag-doc-db-path "$DATA/text_database" \
  --rag-doc-token-cache-size 0
```

For a strict one-GPU comparison, generate the VectorLite shard with
`--num_gpus 1`; then every mapped hot centroid has shard id 0. If a mapping
created for several GPUs is supplied to one SGLang worker, only the selected
shard is loaded and centroids owned by the other shards safely fall back to
the complete CPU index. Omitting both `--rag-hot-index-*` paths keeps the old
full-index-on-GPU mode.

Keep `--rag-ann-max-batch-size 1` for the strict baseline. It preserves
VectorLite's per-query GPU-only fast return: a query with no cold probes is
returned as soon as its hot search finishes. Larger SGLang microbatches are a
separate optimization and currently release results at batch granularity.

Each response reports `ann_index_mode`, hot/cold probe counts and hit ratio,
coarse quantization, CPU search, GPU search, route, and merge latency. The
benchmark aggregates the same fields, so a run can be checked for the intended
hot/cold behavior before comparing end-to-end SLOs.

Keep `--rag-doc-token-cache-size 0` for the strict VectorLiteRAG comparison.
Caching can be evaluated later as a separate optimization.

Run the trace:

```bash
python benchmark/pdmux/bench_rag_serving.py \
  --base-url http://127.0.0.1:30000 \
  --workload-trace logs/rag_compare/seed1/workload.jsonl \
  --output-file logs/rag_compare/seed1/sglang_phase3.json
```

Restart the server between policies. For example, run the same trace with
`--rag-ann-schedule primary` for a naive colocated baseline and with
`decode-gc` for the proposed ANN+Decode resource-sharing scheduler.
`frontend-gc` remains available for the previous ANN+Prefill placement
ablation. Do not reuse a warmed server across systems when reporting paired
results.

## 3. Run the original VectorLiteRAG baseline

Stop the SGLang server first so both systems receive the same GPUs. Run the
adapter in the environment used by VectorLiteRAG:

```bash
python benchmark/pdmux/run_vectorlite_trace.py \
  --vectorlite-repo ../vector-lite-rag \
  --workload-trace logs/rag_compare/seed1/workload.jsonl \
  --output-dir logs/rag_compare/seed1/vectorlite \
  --run-name vectorlite \
  --document-source leveldb \
  -- \
  --model llama8b \
  --index wikiall \
  --gpu_type H100 \
  --num_gpus 1 \
  --llm_backend sglang \
  --search_mode vlite \
  --search_nprobe 2048 \
  --search_topk 25
```

Arguments after `--` are parsed by the baseline checkout. The adapter does not
edit VectorLiteRAG. It injects the trace request source, reorders the loaded
query vectors so VectorLiteRAG row `rid` selects exactly the trace's `query_id`,
and selects the existing LevelDB document path. Warmup and ramp-up rows remain
in the raw baseline output but are removed by the common normalizer.

`--document-source leveldb` makes both systems retrieve and concatenate the
same real document records. To reproduce the artifact's synthetic document
path instead, use `--document-source vectorlite-random`; start SGLang without
`--rag-doc-db-path` and add `--rag-doc-random-fallback` with matching pool and
input lengths. The LevelDB mode is recommended for the complete RAG result.

VectorLiteRAG supports one output length and one top-k per process. The adapter
rejects heterogeneous traces instead of silently changing the workload.

## 4. Produce the common report

```bash
python benchmark/pdmux/compare_rag_results.py \
  --baseline vectorlite \
  --vectorlite-result vectorlite \
    logs/rag_compare/seed1/vectorlite/raw/vectorlite_raw.parquet \
    logs/rag_compare/seed1/vectorlite/vectorlite_trace_mapping.json \
  --sglang-result phase3 \
    logs/rag_compare/seed1/sglang_phase3.json \
  --output-dir logs/rag_compare/seed1/comparison
```

Multiple `--sglang-result LABEL JSON` and
`--vectorlite-result LABEL RAW_PARQUET TRACE_MAPPING` entries are supported.

Generated files:

- `normalized_requests.csv`: one common millisecond-scale row per measured request
- `summary.csv`: mean, p50, p95, and p99 for each common metric
- `pairwise.csv`: candidate latency deltas relative to the selected baseline
- `comparison.json`: machine-readable report with workload fingerprint
- `comparison_cdf.png`: E2E, TTFT, ANN, and TPOT CDFs

## Metric correspondence

| Common metric | Modified SGLang | VectorLiteRAG |
| --- | --- | --- |
| RAG TTFT | client arrival to first token | ANN queue + search + doc/LLM queue + prefill |
| ANN E2E | `ann_latency_ms` | `ann_e2e` |
| ANN queue | `ann_queue_wait_ms` | `ann_queue` |
| LLM TTFT after ANN | TTFT minus ANN latency | LLM queue + prefill |
| TPOT | decode span / output tokens | generation duration / output tokens |
| E2E | client arrival to stream completion | request arrival to completion |

VectorLiteRAG's `ann_search` is wall-clock search time, while SGLang also
reports CUDA-event GPU time. The report maps both to `ann_gpu_latency_ms` for
convenience, but strict cross-system claims should use ANN E2E and ANN queue.
The normalizer also recomputes SGLang TPOT with VectorLiteRAG's denominator so
the common report does not mix `N`-token and `N-1`-interval conventions.

## Experimental discipline

- Use at least five paired seeds and report confidence intervals.
- Restart the serving process before every scheduler/policy run.
- Keep clocks, MPS state, GPU power limits, model, and memory fractions fixed.
- Use the workload hash check; do not use `--allow-workload-mismatch` in paper results.
- Report retrieval quality separately if index partitioning changes recall.
