# RAGLease local source release

This directory packages the RAGLease SGLang changes without experiments, models,
indexes, logs, checkpoints, credentials, or the unrelated modified files in the
nested SGLang worktree.

- Base SGLang commit: `a29ea71014f00da6edd346f1a3c8a6d39f179be6` from the nested
  `sglang/` checkout.
- `sglang-raglease-overlay.tar.gz`: exact copies of 75 changed or new source,
  integration, benchmark, test, and example configuration files.
- `sglang-raglease-files.txt`: explicit path list and tracked/new distinction.
- `sglang-raglease-sha256.txt`: per-file integrity checksums.

To reconstruct the source in a separate checkout of that exact SGLang commit:

```bash
mkdir -p /path/to/clean-sglang
# Populate /path/to/clean-sglang at the base commit first.
tar -xzf release/sglang-raglease-overlay.tar.gz -C /path/to/clean-sglang
cd /path/to/clean-sglang
sha256sum -c /path/to/release/sglang-raglease-sha256.txt
```

The overlay has been extracted over a clean archive of the base commit, and all
75 checksums passed. All 52 packaged Python files passed syntax compilation.
This verifies source integrity, not GPU behavior or reproduction of formal
experiment results. See `docs/RAGLEASE_DESIGN.md` for architecture and the
versioned `docs/experiments/` protocols for experiment-specific settings.
