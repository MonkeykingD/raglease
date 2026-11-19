"""Strict deployment bundle for the four accepted predictor artifacts."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import (
    load_accepted_ann_pressure_table,
    load_accepted_causal_decode_guard,
    load_accepted_corrections,
    load_accepted_predictor,
)


@dataclass(frozen=True)
class CalibratedPredictorBundle:
    identity: tuple
    partition_to_sm: tuple
    profile_partition_by_runtime: tuple
    stage: object
    corrections: object
    pressure: object
    decode_guard: object
    manifest_sha256: str
    # Optional separately assessed solo-Decode component. Legacy v1 bundles
    # keep their original behavior; H100 hybrid bundles must pin this member.
    solo_envelope: object = None
    prediction_source: str = ''
    formal_ready: bool = True
    # H100 132-SM Decode uses a separately assessed serving-history component.
    # Keep this optional and last so all existing positional bundle calls retain
    # their meaning.
    fullcard_envelope: object = None


def _remap_partitioned_profile(profile, mapping, name):
    """Translate offline cell coordinates to the runtime partition ids.

    Profiling uses a dense 0..N-1 grid while the runtime also contains edge
    partitions.  The mapping is deployment data, not an inferred +1 offset.
    Validate complete coverage before constructing a new immutable profile.
    """
    inverse={profile_partition:runtime_partition
             for runtime_partition,profile_partition in mapping}
    cells=tuple(profile.cells)
    used={cell.partition for cell in cells}
    if not cells or used!=set(inverse):
        raise ValueError(name+' profile partitions differ from deployment mapping')
    return type(profile)(profile.identity,tuple(
        replace(cell,partition=inverse[cell.partition]) for cell in cells))


def _member(root,entry):
    rel=entry.get('path')
    if (not isinstance(rel,str) or not rel or Path(rel).is_absolute()
            or '..' in Path(rel).parts):
        raise ValueError('Artifact paths must be normalized and relative')
    raw=root/rel
    resolved=raw.resolve(strict=True)
    if resolved!=raw or root not in resolved.parents:
        raise ValueError('Artifact member must be a regular in-root file')
    if not resolved.is_file():
        raise ValueError('Artifact member must be a file')
    return resolved


def load_calibrated_predictor_bundle(manifest_path, *, expected_sha256,
                                     artifact_root, expected_identity,
                                     expected_partition_to_sm):
    manifest_path=Path(manifest_path)
    payload=manifest_path.read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=expected_sha256:
        raise ValueError('Predictor bundle manifest hash mismatch')
    data=json.loads(payload)
    identity=tuple(data.get('identity',()))
    mapping=tuple((int(p),int(sm)) for p,sm in data.get('partition_to_sm',()))
    profile_mapping=tuple((int(runtime),int(profile)) for runtime,profile in
                          data.get('profile_partition_by_runtime',()))
    if (data.get('schema')!='raglease.predictor-deployment.v1'
            or data.get('accepted') is not True
            or data.get('formal_ready') is not True
            or identity!=expected_identity
            or mapping!=tuple(expected_partition_to_sm)
            or {runtime for runtime,sm in mapping}!={runtime for runtime,profile in profile_mapping}
            or len(dict(profile_mapping))!=len(profile_mapping)
            or len({profile for runtime,profile in profile_mapping})!=len(profile_mapping)
            or any(runtime<0 or profile<0 for runtime,profile in profile_mapping)):
        raise ValueError('Predictor deployment manifest mismatch')
    if (len(dict(mapping))!=len(mapping)
            or any(p<0 or sm<=0 for p,sm in mapping)):
        raise ValueError('Unique positive partition-to-SM mapping required')
    root=Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Predictor artifact root must be a directory')
    members=data.get('artifacts',{})
    if set(members)!=set(('stage','corrections','pressure','decode_guard')):
        raise ValueError('Exactly four accepted predictor artifacts required')
    paths={name:_member(root,entry) for name,entry in members.items()}
    if len(set(paths.values()))!=len(paths):
        raise ValueError('Predictor artifacts must be distinct files')
    stage_entry=members['stage']
    stage=load_accepted_predictor(paths['stage'],
        expected_sha256=stage_entry['sha256'],
        expected_plan_sha256=stage_entry['plan_sha256'],
        expected_supplement_plan_sha256=stage_entry.get('supplement_plan_sha256'),
        expected_identity=identity)
    correction_entry=members['corrections']
    corrections=load_accepted_corrections(paths['corrections'],
        expected_sha256=correction_entry['sha256'],
        expected_plan_sha256=correction_entry['plan_sha256'],
        expected_identity=identity)
    corrections=_remap_partitioned_profile(corrections,profile_mapping,'Correction')
    pressure_entry=members['pressure']
    pressure=load_accepted_ann_pressure_table(paths['pressure'],
        expected_sha256=pressure_entry['sha256'],
        expected_plan_sha256=pressure_entry['plan_sha256'],
        expected_identity=identity)
    pressure=_remap_partitioned_profile(pressure,profile_mapping,'ANN pressure')
    guard_entry=members['decode_guard']
    guard=load_accepted_causal_decode_guard(paths['decode_guard'],
        expected_sha256=guard_entry['sha256'],expected_identity=identity)
    return CalibratedPredictorBundle(identity,mapping,profile_mapping,stage,corrections,
        pressure,guard,digest)
