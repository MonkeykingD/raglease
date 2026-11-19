# RAGLease

RAGLease coordinates ANN retrieval, LLM Prefill, and Decode on a shared GPU.
It combines dependency-guided compute leases with elastic HBM ownership for
hot IVF clusters, ANN scratch, and KV cache.

- [Browse RAGLease source](sglang/)
- [Design and code map](docs/RAGLEASE_DESIGN.md)
- [Detailed architecture trace](docs/raglease_architecture_overview.md)
- [Source release and reconstruction instructions](release/README.md)

The `sglang/` directory contains the RAGLease runtime changes, CUDA kernels,
benchmarks, example configurations, and tests at their original SGLang paths.
To build or run the system, apply these files over the pinned SGLang checkout
using the reconstruction instructions above. Cluster allocation wrappers are in
`cluster/paratera/`.

Model weights, datasets, run outputs, and credentials are not included.
