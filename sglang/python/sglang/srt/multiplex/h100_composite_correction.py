"""Explicitly nonformal H100 correction composite with per-cell provenance."""
import hashlib
import json
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import (
    CoexecutionCorrectionCell, FrozenCoexecutionCorrections,
)


def _key(cell):
    return (cell["stage"], cell["partition"], cell["overlap"],
            tuple(tuple(bounds) for bounds in cell["feature_bounds"]))


def load_h100_composite_correction(path, *, expected_sha256, expected_identity):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("H100 correction composite hash mismatch")
    artifact = json.loads(payload)
    if (artifact.get("schema") != "raglease.h100-composite-correction.v1"
            or artifact.get("formal_ready") is not False
            or artifact.get("engineering_ready") is not True
            or artifact.get("cells") != 28
            or artifact.get("sources") != {
                "refined_sha256": "bde7bc4acb76760aaf51ee25b02210f4c7127c232384f132295fdb0f05aca4c5",
                "fresh_sha256": "e3b9cb181ff5c05a52772ef8db546d7be5c7db6266866ff2e81107db08447eea",
            }):
        raise ValueError("H100 composite is not the pinned per-cell engineering policy")
    table = artifact.get("table", {})
    if tuple(table.get("identity", ())) != expected_identity:
        raise ValueError("H100 correction composite belongs to another deployment")
    raw_cells = table.get("cells", ())
    lineage = artifact.get("lineage", ())
    cell_keys = [_key(cell) for cell in raw_cells]
    lineage_keys = [_key(item) for item in lineage]
    if (len(cell_keys) != 28 or len(set(cell_keys)) != 28
            or set(cell_keys) != set(lineage_keys)
            or sum(item.get("source") == "fresh" for item in lineage) != 10
            or any(item.get("heldout_assessment", {}).get("passed") is not True
                   or item.get("source") not in ("refined", "fresh")
                   or item.get("source_artifact_sha256") != artifact["sources"][
                       "fresh_sha256" if item.get("source") == "fresh" else "refined_sha256"]
                   for item in lineage)):
        raise ValueError("H100 correction composite lacks full per-cell validation")
    cells = tuple(CoexecutionCorrectionCell(
        cell["stage"], cell["partition"], cell["overlap"],
        tuple(tuple(bounds) for bounds in cell["feature_bounds"]),
        cell["nominal_factor"], cell["guard_factor"], cell["samples"])
        for cell in raw_cells)
    return FrozenCoexecutionCorrections(expected_identity, cells)
