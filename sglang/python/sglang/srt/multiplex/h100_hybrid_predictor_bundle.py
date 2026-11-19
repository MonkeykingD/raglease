"""Nonformal H100 E4b integration bundle with truthful component gates.

This loader reuses the two A100 predictor-build schemas and the H100 measured
components. It deliberately does not relabel the failed static Decode stage
or the post-selected correction composite as a formal accepted bundle.
"""
import hashlib
import json
from pathlib import Path

from sglang.srt.multiplex.calibrated_predictor_bundle import (
    CalibratedPredictorBundle, _member, _remap_partitioned_profile,
)
from sglang.srt.multiplex.h100_composite_correction import (
    load_h100_composite_correction,
)
from sglang.srt.multiplex.h100_composite_pressure import (
    load_h100_composite_pressure,
)
from sglang.srt.multiplex.fullcard_decode_envelope import (
    load_accepted_fullcard_decode_envelope,
)
from sglang.srt.multiplex.solo_decode_envelope import (
    load_accepted_solo_decode_envelope,
)
from sglang.srt.multiplex.online_prefill_stage import (
    load_engineering_online_prefill_stage,
)
from sglang.srt.multiplex.stage_predictor import (
    StageFit, StagePredictor,
    load_accepted_causal_decode_guard,
)


MEMBERS = frozenset(("ann", "stage", "corrections", "pressure",
                     "decode_guard", "solo_envelope"))
EDGE_MEMBERS = MEMBERS | {"edge_prefill"}
FULLCARD_MEMBERS = EDGE_MEMBERS | {
    "fullcard_decode_model", "fullcard_decode_assessment"}
ONLINE_PREFILL_MEMBERS = FULLCARD_MEMBERS | {"prefill_online"}
EDGE_PREFILL_PLAN_SHA = "5517cd5eb5347eb1f71340441ec05b6472d2e66a9fc332f82e6a705b0f7f4f33"
RUNTIME_MANIFEST_SHA = "07d394e41a8abb05d3a31eb8cc4b50803daee0e18057f98ecaac797cc3d3e1ee"


def _load_stage_component(path, digest, identity, *, ann):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("H100 phase component artifact hash mismatch")
    artifact = json.loads(payload)
    predictor = artifact.get("predictor", {})
    if tuple(predictor.get("identity", ())) != identity:
        raise ValueError("H100 phase component identity mismatch")
    if ann:
        if (artifact.get("schema") != "raglease.h100-ann-structural-assessment.v2"
                or artifact.get("accepted") is not True
                or artifact.get("formal_ready") is not True
                or set(artifact.get("per_sm", {})) != {"24", "40", "56", "72"}
                or any(row.get("passed") is not True or row.get("coverage") != 1.
                       for row in artifact["per_sm"].values())):
            raise ValueError("H100 ANN component lacks structural held-out acceptance")
        wanted = "ann"
        expected_sm = {24, 40, 56, 72}
    else:
        cells = artifact.get("assessment", {}).get("cells", ())
        if (artifact.get("schema") != "raglease.stage-calibration-assessment.v1"
                or not cells
                or any(c.get("passed") is not True for c in cells
                       if c.get("stage") == "prefill")
                or {c.get("sm") for c in cells if c.get("stage") == "prefill"}
                != {56, 72, 88, 104}):
            raise ValueError("H100 Prefill component lacks held-out acceptance")
        # Decode static cells are retained only as shape-dependent nominal
        # values. Their assessment failed, and solo admission is guarded by
        # the separate independently accepted envelope.
        wanted = None
        expected_sm = {24, 40, 56, 72, 88, 104}
    fits = []
    for row in predictor.get("fits", ()):
        if wanted is not None and row["stage"] != wanted:
            raise ValueError("Unexpected stage in H100 ANN artifact")
        fits.append(StageFit(row["stage"], row["sm_count"],
                             tuple(row["coefficients"]),
                             tuple(tuple(bounds) for bounds in row["feature_bounds"])))
    if {f.sm_count for f in fits} != expected_sm:
        raise ValueError("H100 phase component has incomplete SM coverage")
    return fits


def _load_edge_prefill(path, digest, identity):
    """Admit only the independently passing 132-SM Prefill component.

    The same artifact's large-batch Decode cell failed its frozen gate and
    must not enter the stage predictor through this path.
    """
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("H100 full-card stage artifact hash mismatch")
    artifact = json.loads(payload)
    cells = artifact.get("assessment", {}).get("cells", ())
    prefill = [row for row in cells if row.get("stage") == "prefill"]
    fits = artifact.get("predictor", {}).get("fits", ())
    selected = [row for row in fits if row.get("stage") == "prefill"]
    if (artifact.get("schema") != "raglease.stage-calibration-assessment.v1"
            or artifact.get("plan_sha256") != EDGE_PREFILL_PLAN_SHA
            or artifact.get("runtime_verification", {}).get(
                "runtime_manifest_sha256") != RUNTIME_MANIFEST_SHA
            or tuple(artifact.get("predictor", {}).get("identity", ())) != identity
            or len(prefill) != 3
            or any(row.get("sm") != 132 or row.get("coverage") != 1.
                   or row.get("passed") is not True for row in prefill)
            or len(selected) != 1 or selected[0].get("sm_count") != 132):
        raise ValueError("H100 132-SM Prefill component lacks held-out acceptance")
    row = selected[0]
    return StageFit("prefill", 132, tuple(row["coefficients"]),
                    tuple(tuple(bounds) for bounds in row["feature_bounds"]))


def load_h100_hybrid_predictor_bundle(manifest_path, *, expected_sha256,
                                      artifact_root, expected_identity,
                                      expected_partition_to_sm):
    payload = Path(manifest_path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("H100 hybrid manifest hash mismatch")
    data = json.loads(payload)
    identity = tuple(data.get("identity", ()))
    mapping = tuple((int(p), int(sm)) for p, sm in data.get("partition_to_sm", ()))
    profile_mapping = tuple((int(p), int(q)) for p, q in
                            data.get("profile_partition_by_runtime", ()))
    schema = data.get("schema")
    if (schema not in ("raglease.h100-hybrid-predictor-deployment.v1",
                       "raglease.h100-hybrid-predictor-deployment.v2",
                       "raglease.h100-hybrid-predictor-deployment.v3",
                       "raglease.h100-hybrid-predictor-deployment.v4",
                       "raglease.h100-hybrid-predictor-deployment.v5",
                       "raglease.h100-hybrid-predictor-deployment.v6",
                       "raglease.h100-hybrid-predictor-deployment.v7")
            or data.get("engineering_only") is not True
            or data.get("formal_ready") is not False
            or identity != expected_identity
            or mapping != tuple(expected_partition_to_sm)
            or {sm for _, sm in mapping} != {24, 40, 56, 72}
            or {p for p, _ in mapping} != {p for p, _ in profile_mapping}
            or {q for _, q in profile_mapping} != {0, 1, 2, 3}
            or len(dict(mapping)) != 4 or len(dict(profile_mapping)) != 4):
        raise ValueError("H100 hybrid deployment identity or SM grid mismatch")
    root = Path(artifact_root).resolve(strict=True)
    members = data.get("artifacts", {})
    required = (ONLINE_PREFILL_MEMBERS if schema.endswith(".v7") else
                FULLCARD_MEMBERS if schema.endswith((".v4", ".v5", ".v6")) else
                EDGE_MEMBERS if schema.endswith((".v2", ".v3")) else MEMBERS)
    if not root.is_dir() or set(members) != required:
        raise ValueError("H100 component artifact set differs from schema")
    paths = {name: _member(root, entry) for name, entry in members.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("H100 component artifact paths must be distinct")
    fits = (_load_stage_component(paths["ann"], members["ann"]["sha256"],
                                  identity, ann=True)
            + _load_stage_component(paths["stage"], members["stage"]["sha256"],
                                    identity, ann=False))
    if schema.endswith((".v2", ".v3", ".v4", ".v5", ".v6", ".v7")):
        fits.append(_load_edge_prefill(
            paths["edge_prefill"], members["edge_prefill"]["sha256"], identity))
    keys = [(fit.stage, fit.sm_count) for fit in fits]
    expected_fits = (len(EDGE_MEMBERS) + 6 if schema.endswith((".v4", ".v5", ".v6", ".v7"))
                     else len(required) + 6)
    if len(keys) != expected_fits or len(set(keys)) != len(keys):
        raise ValueError("H100 ANN/Prefill/Decode component fits differ from grid")
    stage = StagePredictor(identity, tuple(fits))
    if schema.endswith(".v7"):
        stage = load_engineering_online_prefill_stage(
            stage, paths["prefill_online"],
            expected_sha256=members["prefill_online"]["sha256"],
            expected_identity=identity)
    corrections = load_h100_composite_correction(
        paths["corrections"], expected_sha256=members["corrections"]["sha256"],
        expected_identity=identity)
    corrections = _remap_partitioned_profile(
        corrections, profile_mapping, "H100 correction")
    pressure = load_h100_composite_pressure(
        paths["pressure"], expected_sha256=members["pressure"]["sha256"],
        expected_identity=identity)
    if len(pressure.cells) != (20 if schema.endswith((".v3", ".v4", ".v5", ".v6", ".v7")) else 12):
        raise ValueError("H100 pressure cell count differs from deployment schema")
    pressure = _remap_partitioned_profile(
        pressure, profile_mapping, "H100 ANN pressure")
    guard = load_accepted_causal_decode_guard(
        paths["decode_guard"], expected_sha256=members["decode_guard"]["sha256"],
        expected_identity=identity)
    solo = load_accepted_solo_decode_envelope(
        paths["solo_envelope"], expected_sha256=members["solo_envelope"]["sha256"],
        expected_identity=identity)
    fullcard = None
    if schema.endswith((".v4", ".v5", ".v6", ".v7")):
        fullcard = load_accepted_fullcard_decode_envelope(
            paths["fullcard_decode_model"], paths["fullcard_decode_assessment"],
            expected_model_sha256=members["fullcard_decode_model"]["sha256"],
            expected_assessment_sha256=members["fullcard_decode_assessment"]["sha256"],
            expected_identity=identity, accepted_base=solo.base,
            expected_model_schema=(
                "raglease.h100-fullcard-decode-serving-envelope.v5"
                if schema.endswith((".v6", ".v7")) else
                "raglease.h100-fullcard-decode-serving-envelope.v4"
                if schema.endswith(".v5") else
                "raglease.h100-fullcard-decode-serving-envelope.v3"))
    return CalibratedPredictorBundle(
        identity, mapping, profile_mapping, stage, corrections, pressure, guard,
        digest, solo,
        "H100 E4b nonformal hybrid engineering core: accepted ANN/Prefill, "
        "co-run guard and solo envelope; post-selected correction and pressure "
        "composites plus static Decode fit and online Prefill residual remain nonformal", False,
        fullcard_envelope=fullcard)
