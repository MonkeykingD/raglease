# RAGLease

RAGLease coordinates ANN retrieval, LLM Prefill, and Decode on a shared GPU.
It combines dependency-guided compute leases with elastic HBM ownership for
hot IVF clusters, ANN scratch, and KV cache.

- [Design and code map](docs/RAGLEASE_DESIGN.md)
- [Detailed architecture trace](docs/raglease_architecture_overview.md)
- [Source release and reconstruction instructions](release/README.md)

The SGLang implementation is distributed as an exact source overlay over the
pinned SGLang base commit listed in `release/README.md`. The archive includes
only reviewed implementation, integration, benchmark, configuration, and test
files. Cluster allocation wrappers are in `cluster/paratera/`.

Model weights, datasets, run outputs, and credentials are not included.
