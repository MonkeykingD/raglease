# RAGLease Architecture Overview

## Figure contract

The architecture figure follows the four design subsections:

1. **Profile-Guided Lease Estimation** takes the hardware/deployment profile
   and representative workloads and SLOs, then emits a hardware-specific lease
   envelope for online coordination.
2. **Dependency-Guided Compute Leasing** proposes a bounded compute grant: an
   ANN credit microbatch, a Prefill layer bundle, or a Decode-only turn under a
   selected Green Context split.
3. **Hotness-Aware Elastic HBM Leasing** predicts candidate and victim IVF
   clusters and changes ownership of CUDA VMM extents among hot-index data,
   ANN scratch, and KV cache.
4. **Joint Lease Coordination** reconciles the two lease intents at scheduler
   safe points. It delays incompatible ownership or partition transitions and
   preserves overlap when no transition is required.

The fourth component is deliberately described as coordination rather than a
global optimizer. The implementation performs launch feasibility checks and
orders Green Context and VMM transitions; it does not solve an atomic
mixed-integer compute-memory allocation problem every iteration.

## Implementation trace

| Figure element | Implemented behavior | Primary code evidence | Claim boundary |
| --- | --- | --- | --- |
| Hardware-specific lease envelope | Enumerates candidates, rejects failed/non-stationary candidates, selects robust SLO goodput, and writes a deployment profile | `sglang/benchmark/pdmux/profile_rag_scheduler.py:526`, `:3116`, `:3196` | The generated profile currently materializes compute-policy fields. Online HBM observations refine memory decisions. |
| Request-scoped `ANN_i -> Prefill_i` handoff | Registers, expires, and consumes the matching handoff token | `sglang/python/sglang/srt/multiplex/multiplexing.py:1500`, `:1520` | The handoff transfers scheduling urgency, not persistent SM or HBM ownership. |
| ANN credit microbatch | Chooses the largest pending prefix that fits measured millisecond credit | `sglang/python/sglang/srt/multiplex/multiplexing.py:1163` | Credit bounds admitted ANN work; it is not a hard kernel preemption guarantee. |
| Bounded Prefill layer bundle | Predicts and chooses the number of layers in the current Prefill lease | `sglang/python/sglang/srt/multiplex/multiplexing.py:3134` | The decision is reevaluated at bundle boundaries, not at arbitrary instruction boundaries. |
| Dependency-guided compute plan | Produces and finalizes the per-round ANN/Prefill/Decode action and target stream group | `sglang/python/sglang/srt/multiplex/multiplexing.py:3835`, `:4640` | The figure uses a compact logical lease tuple; the implementation stores the corresponding plan fields and counters. |
| Work-conserving ANN--Decode overlap | Places ANN in the Decode pool when `decode-gc` and admits it alongside active Decode work | `sglang/python/sglang/srt/multiplex/multiplexing.py:272`, `:3864` | Green Context partitions SM placement but not HBM bandwidth or cache. |
| VMM-backed HBM ownership | Manages physical extents owned by KV, ANN scratch, or ANN index | `sglang/python/sglang/srt/mem_cache/shared_hbm.py:103`, `:167`, `:577` | The figure shows only the managed elastic pool; model weights and fixed allocations are omitted. |
| Hot-cluster candidate/victim prediction | Combines offline hot order with online heat, miss debt, measured CPU/GPU search time, and reload cost | `sglang/python/sglang/srt/multiplex/hot_index.py:79`, `:172`, `:316`, `:334` | Candidate prediction controls cluster refill; it does not change retrieval quality parameters such as `nprobe` or `top-k`. |
| Compute-memory compatibility gate | Limits ANN rows under HBM pressure and validates scratch ownership before ANN launch | `sglang/python/sglang/srt/mem_cache/shared_hbm.py:1715`, `:2297` | This is a launch gate and fallback path, not an all-resource global optimum. |
| Safe-point transition ordering | Advances asynchronous reclaim/map stages at scheduler-owned completion and switch boundaries | `sglang/python/sglang/srt/multiplex/multiplexing.py:1819`, `:5938`, `:7598`, `:7816` | Compatible work continues without a global device synchronization; incompatible transitions still require fencing. |

## Caption

**RAGLease architecture.** At deployment time, a profile-guided lease estimator
uses the hardware and deployment profile together with representative
workloads and SLOs to construct a hardware-specific lease envelope. At runtime,
dependency-guided compute leasing and hotness-aware elastic HBM leasing propose
bounded compute and memory actions from request progress and resource state.
Joint lease coordination reconciles these actions at scheduler safe points and
applies them to dynamic Green Context partitions and a CUDA VMM-backed HBM
pool. Execution feedback updates subsequent decisions. ANN executes
work-conservingly in the Decode Green Context, while Prefill advances through
bounded layer bundles.

## LaTeX inclusion

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/raglease_architecture_overview.pdf}
    \caption{RAGLease architecture. At deployment time, a profile-guided lease
    estimator uses the hardware and deployment profile together with
    representative workloads and SLOs to construct a hardware-specific lease
    envelope. At runtime, dependency-guided compute leasing and hotness-aware
    elastic HBM leasing propose bounded compute and memory actions from request
    progress and resource state. Joint lease coordination reconciles these
    actions at scheduler safe points and applies them to dynamic Green Context
    partitions and a CUDA VMM-backed HBM pool. Execution feedback updates
    subsequent decisions.}
    \label{fig:raglease-architecture}
\end{figure*}
```

## Artifacts

- Generator: `paper/figures/raglease_architecture_overview.py`
- Vector PDF: `paper/figures/raglease_architecture_overview.pdf`
- Editable SVG: `paper/figures/raglease_architecture_overview.svg`
- Review PNG: `paper/figures/raglease_architecture_overview.png`
