# RAGLease local source release

Browse the RAGLease changes in [sglang/](../sglang/). The 75 source, config,
and test files match the archive byte for byte. The source tree also includes
the SGLang and SGL Kernel license files and third-party notices from the base.

- Base SGLang commit: `a29ea71014f00da6edd346f1a3c8a6d39f179be6` from the nested
  `sglang/` checkout.
- `sglang-raglease-overlay.tar.gz`: exact copies of 75 changed or new source,
  integration, benchmark, test, and example configuration files.
- `sglang-raglease-files.txt`: explicit path list and tracked/new distinction.
- `sglang-raglease-sha256.txt`: per-file integrity checksums.

Verify the expanded files from the root of this repository:

```bash
(cd sglang && sha256sum --strict -c ../release/sglang-raglease-sha256.txt)
```

To reconstruct a full source tree, prepare a separate SGLang checkout at the
base commit above, then apply the archive from this repository's root:

```bash
tar -xzf release/sglang-raglease-overlay.tar.gz -C /path/to/clean-sglang
```

The published checks cover file integrity and Python syntax. The
[design overview](../docs/RAGLEASE_DESIGN.md) links to the runtime entry points.
