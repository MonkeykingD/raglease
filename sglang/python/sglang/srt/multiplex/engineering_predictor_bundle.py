"""Strict two-artifact bundle for explicitly nonformal predictor wiring."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from sglang.srt.multiplex.stage_predictor import (
    load_accepted_predictor,
    load_engineering_causal_decode_guard,
)


@dataclass(frozen=True)
class EngineeringPredictorBundle:
    identity: tuple
    partition_to_sm: tuple
    stage: object
    decode_guard: object
    manifest_sha256: str
    prediction_source: str


def _member(root, entry):
    rel=entry.get('path')
    if (not isinstance(rel,str) or not rel or Path(rel).is_absolute()
            or '..' in Path(rel).parts):
        raise ValueError('Artifact paths must be normalized and relative')
    raw=root/rel
    resolved=raw.resolve(strict=True)
    if resolved!=raw or root not in resolved.parents or not resolved.is_file():
        raise ValueError('Artifact member must be a regular in-root file')
    return resolved


def load_engineering_predictor_bundle(manifest_path, *, expected_sha256,
        artifact_root, expected_identity, expected_partition_to_sm):
    payload=Path(manifest_path).read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=expected_sha256:
        raise ValueError('Engineering predictor manifest hash mismatch')
    data=json.loads(payload)
    identity=tuple(data.get('identity',()))
    mapping=tuple((int(p),int(sm)) for p,sm in data.get('partition_to_sm',()))
    source=data.get('prediction_source')
    if (data.get('schema')!='raglease.predictor-engineering-deployment.v1'
            or data.get('engineering_only') is not True
            or data.get('formal_ready') is not False
            or data.get('engineering_ready') is not True
            or identity!=expected_identity
            or mapping!=tuple(expected_partition_to_sm)
            or not isinstance(source,str) or not source.strip()):
        raise ValueError('Engineering predictor deployment manifest mismatch')
    if len(dict(mapping))!=len(mapping) or any(p<0 or sm<=0 for p,sm in mapping):
        raise ValueError('Unique positive partition-to-SM mapping required')
    root=Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Predictor artifact root must be a directory')
    members=data.get('artifacts',{})
    if set(members)!=set(('stage','decode_guard')):
        raise ValueError('Exactly stage and engineering guard artifacts required')
    paths={name:_member(root,entry) for name,entry in members.items()}
    if len(set(paths.values()))!=2:
        raise ValueError('Predictor artifacts must be distinct files')
    stage_entry=members['stage']
    stage=load_accepted_predictor(paths['stage'],
        expected_sha256=stage_entry['sha256'],
        expected_plan_sha256=stage_entry['plan_sha256'],
        expected_supplement_plan_sha256=stage_entry.get('supplement_plan_sha256'),
        expected_identity=identity)
    guard_entry=members['decode_guard']
    guard=load_engineering_causal_decode_guard(paths['decode_guard'],
        expected_sha256=guard_entry['sha256'],expected_identity=identity)
    return EngineeringPredictorBundle(identity,mapping,stage,guard,digest,source)
