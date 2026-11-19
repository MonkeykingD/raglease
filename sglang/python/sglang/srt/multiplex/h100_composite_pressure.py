"""Load the explicitly nonformal H100 pressure composite for integration."""
import hashlib
import json
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import ANNPressureCell, FrozenANNPressureTable


def _key(row):
    return (row["partition"], row["overlap"], tuple(row["batch_range"]),
            tuple(row["context_total_range"]), tuple(row["context_max_range"]),
            tuple(row["ann_gpu_ms_range"]))


def load_h100_composite_pressure(path, *, expected_sha256, expected_identity):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("H100 pressure composite hash mismatch")
    artifact = json.loads(payload)
    schema = artifact.get("schema")
    if (schema not in ("raglease.h100-composite-pressure.v1",
                       "raglease.h100-composite-pressure.v2")
            or artifact.get("formal_ready") is not False
            or artifact.get("engineering_ready") is not True
            or tuple(artifact.get("identity", ())) != expected_identity):
        raise ValueError("H100 pressure composite provenance mismatch")
    if schema.endswith(".v1") and artifact.get("sources") != {
                "refined_sha256": "7047c586aa044e3f49ee797ffdcf08e24a60cf683d01cfbdf44f277d4b12abe0",
                "fresh_sha256": "ccb63040d22ef8f11964bc29100a59b4eb58351f8379d9b0378b019cf39804e8",
            }:
        raise ValueError("H100 pressure composite provenance mismatch")
    if schema.endswith(".v2") and artifact.get("sources") != {
                "long_sha256": "67ad48a9cf0dfd8eaeb482bce89e5453e1e6dfd093eb35323c376611ed60f32f",
                "short_sha256": "4b419b3763e6e82816fb11348c5862077a2a4e7fcca8bca450fb7fce7137b543",
            }:
        raise ValueError("H100 pressure composite provenance mismatch")
    rows = artifact.get("cells", ())
    lineage = artifact.get("lineage", ())
    keys = [_key(row) for row in rows]
    original = rows[:12] if schema.endswith(".v2") else rows
    if (len(keys) != (20 if schema.endswith(".v2") else 12)
            or len(set(keys)) != len(keys)
            or {(row["partition"], tuple(row["batch_range"])) for row in original}
               != {(partition, bounds) for partition in range(4)
                   for bounds in ((1, 9), (9, 33), (33, 65))}
            or any(row.get("overlap") != "ann" or row.get("passed") is not True
                   or row.get("overlap_verified") is not True
                   or row.get("heldout_samples", 0) <= 0 for row in rows)
            or len(lineage) != (0 if schema.endswith(".v2") else 12)
            or len({(item["partition"], tuple(item["batch_range"]))
                    for item in lineage}) != (0 if schema.endswith(".v2") else 12)
            or sum(item.get("source") == "fresh" for item in lineage) != (0 if schema.endswith(".v2") else 4)
            or any(item.get("source_artifact_sha256") != artifact["sources"][
                       "fresh_sha256" if item.get("source") == "fresh" else "refined_sha256"]
                   for item in lineage)):
        raise ValueError("H100 pressure composite lacks complete per-cell evidence")
    if schema.endswith(".v2"):
        if ({(row["partition"], tuple(row["batch_range"])) for row in rows[12:]}
                != {(partition, bounds) for partition in range(4)
                    for bounds in ((9, 33), (33, 65))}
                or any(tuple(row["context_max_range"]) != (0, 1024)
                       or row.get("training_samples") != 80
                       or row.get("heldout_samples") != 40
                       for row in rows[12:])):
            raise ValueError("H100 short-context cells differ from accepted grid")
    cells = tuple(ANNPressureCell(
        row["partition"], row["overlap"], tuple(row["batch_range"]),
        tuple(row["context_total_range"]), tuple(row["context_max_range"]),
        tuple(row["ann_gpu_ms_range"]), row["service_ms"]) for row in rows)
    return FrozenANNPressureTable(expected_identity, cells)
