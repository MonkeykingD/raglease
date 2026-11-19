"""Explicit placeholder PERFORMANCE inputs for wiring, never capacity facts.

No queue, allocator, residency, readiness, or transition ownership is exposed
here. These values must not be loaded through the calibrated-artifact loader.
"""
from dataclasses import dataclass, replace
import math

from sglang.srt.multiplex.stage_predictor import (
    StageFit, StagePredictor, ANNPressureCell, FrozenANNPressureTable,
)
from sglang.srt.multiplex.full_lease import RequestReferenceProfile


@dataclass(frozen=True)
class EngineeringPerformanceInputs:
    stage_model: StagePredictor
    pressure_table: FrozenANNPressureTable
    reference_catalog: tuple
    formal_ready: bool = False
    prediction_source: str = 'engineering-placeholder-v1; not measured'


@dataclass(frozen=True)
class EngineeringOperationTimings:
    """Only fill duration_ms; never manufacture readiness, work or dependencies."""
    stage_ms: tuple

    def __post_init__(self):
        if (not isinstance(self.stage_ms,tuple) or len(dict(self.stage_ms))!=len(self.stage_ms)
                or any(not isinstance(stage,str) or not stage or type(ms) not in (int,float)
                       or not math.isfinite(ms) or ms<0 for stage,ms in self.stage_ms)):
            raise ValueError('Unique stages with finite nonnegative engineering durations required')

    def price(self, operations):
        lookup=dict(self.stage_ms);result=[]
        for op in operations:
            if op.duration_ms is not None:
                result.append(op)  # Existing observed/predicted residual stays intact.
            elif op.stage not in lookup:
                return None
            else:
                result.append(replace(op,duration_ms=lookup[op.stage]))
        return tuple(result)


def make_engineering_performance_inputs(*, sm_counts, total_layers, max_prompt_tokens,
        max_context_tokens, max_batch, declared_output_tokens,
        ann_ms=2., prepare_ms=1., prefill_ms=28., decode_ms=10.):
    """Build bounded placeholder models, with identical inputs for both modes.

    Output length is an explicit engineering contract, not a natural-EOS oracle.
    SM-independent solo constants deliberately make no scaling/benefit claim.
    ANN pressure contributes its estimated time to the mock shared duration;
    this is an interface exercise, NOT a physically justified contention model.
    """
    dimensions=(total_layers,max_prompt_tokens,max_context_tokens,max_batch,declared_output_tokens)
    if any(type(v) is not int or v<=0 for v in dimensions):
        raise ValueError('Explicit positive engineering shape/output bounds required')
    if max_prompt_tokens>max_context_tokens:
        raise ValueError('Prompt bound exceeds context bound')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
           for v in (ann_ms,prepare_ms,prefill_ms,decode_ms)):
        raise ValueError('Positive finite placeholder times required')
    sms=tuple(tuple(pair) for pair in sm_counts)
    if not sms or any(len(pair)!=2 or any(type(n) is not int or n<0 for n in pair)
                      or sum(pair)<=0 for pair in sms):
        raise ValueError('Actual nonnegative SM partition pairs required')
    identity=('engineering-placeholder-v1',str(sms),str(total_layers))
    fits=[]
    for sm in sorted({pair[0] for pair in sms if pair[0]>0}):
        fits.append(StageFit('prefill',sm,(0.,0.,0.,float(prefill_ms)),
            ((0.,max_batch*(max_context_tokens/1024.)**2),
             (0.,max_batch*(max_context_tokens/1024.)**2),
             (0.,max_batch*max_context_tokens/1024.))))
    for sm in sorted({pair[1] for pair in sms if pair[1]>0}):
        fits.append(StageFit('decode',sm,(0.,0.,float(decode_ms)),
            ((0.,max_batch*max_context_tokens/1024.),(1.,float(max_batch)))))
    cells=[]
    for partition,(_,decode_sm) in enumerate(sms):
        if not decode_sm:continue
        for overlap in ('ann','prefill','ann_prefill'):
            for low,high in ((0.,ann_ms),(ann_ms,2*ann_ms),(2*ann_ms,4*ann_ms)):
                cells.append(ANNPressureCell(partition,overlap,(1,max_batch+1),
                    (0,max_batch*max_context_tokens+1),(0,max_context_tokens+1),
                    (low,high),float(decode_ms+(high if overlap!='prefill' else 0.))))
    catalog=tuple((n,RequestReferenceProfile('engineering-fixed-reference-v1',n,total_layers,
        ann_ms,prepare_ms,prefill_ms/total_layers,decode_ms,declared_output_tokens))
        for n in range(1,max_prompt_tokens+1))
    return EngineeringPerformanceInputs(StagePredictor(identity,tuple(fits)),
        FrozenANNPressureTable(identity,tuple(cells)),catalog)
