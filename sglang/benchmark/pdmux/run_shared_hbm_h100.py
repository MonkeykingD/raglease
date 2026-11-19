"""Run reproducible off/scratch/ivf shared-HBM H100/A100-80GB groups.

The input is the normal ``rag_scheduler_profile`` JSON. This wrapper keeps the
workload and policy sweep identical while changing only shared-HBM ownership
mode and writing each group to an independent output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


MODES = ("off", "scratch", "ivf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument("--pool-gib", type=float, default=10.0)
    parser.add_argument(
        "--scratch-pool-gib",
        type=float,
        default=None,
        help="Optional scratch-mode pool size; defaults to --pool-gib.",
    )
    parser.add_argument(
        "--ivf-pool-gib",
        type=float,
        default=None,
        help="Optional ivf-mode pool size; defaults to --pool-gib.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        choices=(2, 4, 8, 16, 32, 64),
        default=None,
    )
    parser.add_argument("--lookahead-ms", type=float, default=50.0)
    parser.add_argument("--min-residency-ms", type=float, default=10.0)
    parser.add_argument("--cluster-heat-decay", type=float, default=0.98)
    parser.add_argument("--cluster-miss-boost", type=float, default=4.0)
    parser.add_argument(
        "--cluster-promotion-hysteresis", type=float, default=1.15
    )
    parser.add_argument("--kv-reserve-extents", type=int, default=2)
    parser.add_argument(
        "--rerank-prefetch-batches", type=int, default=1
    )
    parser.add_argument("--async-map", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_output = Path(config["output_dir"]).expanduser()
    profiler = Path(__file__).with_name("profile_rag_scheduler.py")

    if "ivf" in args.modes:
        paths = config.get("paths", {})
        if not (
            paths.get("hot_index_path")
            and paths.get("hot_index_map_path")
        ):
            parser.error(
                "ivf mode requires paths.hot_index_path and "
                "paths.hot_index_map_path"
            )

    with tempfile.TemporaryDirectory(prefix="sglang-shared-hbm-") as tmp:
        tmpdir = Path(tmp)
        for mode in args.modes:
            case = json.loads(json.dumps(config))
            runtime = case.setdefault("runtime", {})
            mode_pool_gib = {
                "scratch": args.scratch_pool_gib,
                "ivf": args.ivf_pool_gib,
            }.get(mode)
            if mode_pool_gib is None:
                mode_pool_gib = args.pool_gib
            runtime.update(
                {
                    "shared_hbm_mode": mode,
                    "shared_hbm_pool_gib": mode_pool_gib,
                    "shared_hbm_chunk_mib": (
                        args.chunk_mib
                        if args.chunk_mib is not None
                        else int(runtime.get("shared_hbm_chunk_mib", 8))
                    ),
                    "shared_hbm_lookahead_ms": args.lookahead_ms,
                    "shared_hbm_min_residency_ms": (
                        args.min_residency_ms
                    ),
                    "shared_hbm_cluster_heat_decay": (
                        args.cluster_heat_decay
                    ),
                    "shared_hbm_cluster_miss_boost": (
                        args.cluster_miss_boost
                    ),
                    "shared_hbm_cluster_promotion_hysteresis": (
                        args.cluster_promotion_hysteresis
                    ),
                    "shared_hbm_kv_reserve_extents": (
                        args.kv_reserve_extents
                    ),
                    "shared_hbm_rerank_prefetch_batches": (
                        args.rerank_prefetch_batches
                    ),
                    "shared_hbm_async_map": args.async_map,
                }
            )
            case["output_dir"] = str(base_output / mode)
            generated = tmpdir / f"{mode}.json"
            generated.write_text(
                json.dumps(case, indent=2) + "\n", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(profiler),
                "--config",
                str(generated),
            ]
            if args.dry_run:
                command.append("--dry-run")
            if args.overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
