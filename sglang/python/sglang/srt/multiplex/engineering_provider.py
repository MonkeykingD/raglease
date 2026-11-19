"""Live-state provider assembly for explicitly nonformal wiring runs.

Performance values are placeholders. Native capture, HBM planning and commit
remain authoritative. Unsupported native observations stay unsupported.
"""
from dataclasses import dataclass, replace
import importlib
import json
import math
import os

from sglang.srt.multiplex.joint_bootstrap import (
    CalibratedLegacyPredictorBootstrap,
    EngineeringJointBootstrap,
)
from sglang.srt.multiplex.engineering_profiles import (
    make_engineering_performance_inputs, EngineeringOperationTimings,
)
from sglang.srt.multiplex.full_lease import ComputeLease
from sglang.srt.multiplex.lease_profile import FrozenANNServiceProfile, FrozenJointTransitionProfiles
from sglang.srt.multiplex.lease_frontier import capture_execution_stamp
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx


@dataclass(frozen=True)
class _ANNEntry:
    partition: int
    batch_size: int
    gpu_ms: float


@dataclass(frozen=True)
class _EngineeringRemainingTime:
    """Explicit placeholder RESIDUAL time; never a physical readiness result."""
    ms: float = 2.
    directional: bool = False

    def query_remaining_ms(self, *key, elapsed_ms=0.):
        if not math.isfinite(elapsed_ms) or elapsed_ms<0:
            raise ValueError('Nonnegative observed age required')
        return 0. if self.directional and len(key)==2 and key[0]==key[1] else self.ms


@dataclass(frozen=True)
class _BootstrapCollectionProfile:
    """Small collection-only prior while the live run accumulates samples.

    This is deliberately not a substitute for ANN, prefill, or decode service.
    Calibrated-core runs expose this remaining nonformal boundary in their
    prediction source and never route through EngineeringOperationTimings.
    """
    ms: float = 1.

    def query_ms(self, mode, rows, topk, searches, has_gpu):
        if (mode!='worker' or any(type(value) is not int or value<0
                for value in (rows,topk,searches)) or rows<=0 or topk<=0
                or type(has_gpu) is not bool):
            raise ValueError('Valid frozen collection shape required')
        return self.ms


def _profiled_mixed_partitions(sm_counts):
    """Map only partitions covered by the mixed prefill/decode profiles."""
    mapping = tuple((index, decode_sm)
                    for index, (prefill_sm, decode_sm) in enumerate(sm_counts)
                    if prefill_sm > 0 and decode_sm > 0)
    if not mapping:
        raise ValueError('At least one mixed prefill/decode partition is required')
    return mapping


class EngineeringLiveProvider:
    def __init__(self, scheduler, *, output_tokens, horizon_ms=50., ann_limit=4,
                 predictor_bundle=None, calibrated_bundle=None, cpu_ann_profile=None):
        self.scheduler=scheduler
        self.horizon_ms=float(horizon_ms)
        if not math.isfinite(self.horizon_ms) or self.horizon_ms<=0:
            raise ValueError('Positive engineering horizon required')
        self.mapping={i:f'partition:{i}' for i in range(len(scheduler.sm_counts))}
        self.ann_limit=ann_limit
        self.output_tokens=output_tokens
        base=make_engineering_performance_inputs(sm_counts=scheduler.sm_counts,
            total_layers=scheduler.model_config.num_hidden_layers,max_prompt_tokens=1024,
            max_context_tokens=1024+output_tokens,max_batch=scheduler.get_num_allocatable_reqs(0),
            declared_output_tokens=output_tokens)
        if predictor_bundle is not None and calibrated_bundle is not None:
            raise ValueError('Choose engineering or calibrated predictor bundle')
        if (calibrated_bundle is None)!=(cpu_ann_profile is None):
            raise ValueError('Calibrated bundle and accepted CPU ANN profile are required together')
        self.calibrated=calibrated_bundle is not None
        self.decode_guard=None
        self.solo_envelope=None
        self.fullcard_envelope=None
        self.iteration_guard=None
        self.corrections=None
        self.cpu_ann_profile=cpu_ann_profile
        self.candidate_partitions=tuple(self.mapping)
        self.decode_recovery_partition=len(scheduler.sm_counts)-1
        if predictor_bundle is not None:
            mapping=_profiled_mixed_partitions(scheduler.sm_counts)
            if (predictor_bundle.partition_to_sm!=mapping
                    or predictor_bundle.stage.identity!=predictor_bundle.identity
                    or predictor_bundle.decode_guard.identity!=predictor_bundle.identity):
                raise ValueError('Engineering predictor bundle differs from runtime partitions')
            base=replace(base,stage_model=predictor_bundle.stage,
                         prediction_source=predictor_bundle.prediction_source)
            self.decode_guard=predictor_bundle.decode_guard
        if calibrated_bundle is not None:
            mapping=_profiled_mixed_partitions(scheduler.sm_counts)
            if (calibrated_bundle.partition_to_sm!=mapping
                    or calibrated_bundle.stage.identity!=calibrated_bundle.identity
                    or calibrated_bundle.corrections.identity!=calibrated_bundle.identity
                    or calibrated_bundle.pressure.identity!=calibrated_bundle.identity
                    or calibrated_bundle.decode_guard.identity!=calibrated_bundle.identity):
                raise ValueError('Calibrated predictor bundle differs from runtime partitions')
            base=replace(base,stage_model=calibrated_bundle.stage,
                pressure_table=calibrated_bundle.pressure,formal_ready=False,
                prediction_source=(calibrated_bundle.prediction_source or
                    'accepted stage/correction/ANN-pressure/decode-guard/CPU-ANN core; '
                    'bootstrap overheads and bounded prefill/ANN/decode/CPU-tail fallbacks remain nonformal'))
            self.decode_guard=calibrated_bundle.decode_guard
            self.solo_envelope=calibrated_bundle.solo_envelope
            self.fullcard_envelope=calibrated_bundle.fullcard_envelope
            guard_module=importlib.import_module(type(self.decode_guard).__module__)
            self.iteration_guard=guard_module.CausalIterationOverheadGuard()
            self.corrections=calibrated_bundle.corrections
            self.candidate_partitions=tuple(partition for partition,sm in mapping)
            self.cpu_ann_profile=replace(
                cpu_ann_profile,sparse_survivor_guard_fallback=True)
        self.performance=base
        self.ann_profile=FrozenANNServiceProfile(tuple(_ANNEntry(p,n,float(n))
            for p in self.mapping for n in range(1,ann_limit+1)))
        self.release_profile=_EngineeringRemainingTime()
        self.transitions=FrozenJointTransitionProfiles(_EngineeringRemainingTime(directional=True),
            _EngineeringRemainingTime(),_EngineeringRemainingTime())
        self.collection_profile=_BootstrapCollectionProfile()
        self.timings=EngineeringOperationTimings(tuple((stage,ms) for stage,ms in
            (('ann',2.),('ann_release',2.),('cpu_retrieval',2.),('collect',1.),
             ('prepare',1.),('prefill',1.),('prefill_result',1.),('prefill_merge',1.),
             ('prefill_commit',1.),
             ('decode',10.))))

    def __call__(self, *, event_id, window_started_at, mode):
        s=self.scheduler
        s.rag_joint_engineering_missing_input=None
        def unavailable(reason):
            s.rag_joint_engineering_missing_input=reason
            return None
        if mode!=s.rag_policy_mode:
            raise ValueError('Provider cannot change admission policy')
        stamp=capture_execution_stamp(s)
        # This explicit wiring contract may not be silently used for EOS or
        # the earlier variable-output E1b trajectory.
        if any(f.remaining_output_limit is not None and
               f.remaining_output_limit>self.output_tokens for f in stamp.frontiers):
            return unavailable('declared_output_limit')
        running=len(getattr(getattr(s,'running_batch',None),'reqs',()) or ())
        model=self.performance.stage_model
        if any(f.stage == 'decode' and (type(f.remaining_output_limit) is not int
                                       or f.remaining_output_limit < 0)
               for f in stamp.frontiers):
            return unavailable('decode_remaining_output_limit')
        # The memory-demand API consumes per-request growth, not an aggregate.
        # Keep observed identities and cap each placeholder forecast by that
        # request's explicit remaining work; this never claims allocated KV.
        growth=tuple((f.request_id, min(f.remaining_output_limit,
                                      math.ceil(self.horizon_ms/10.)))
                     for f in stamp.frontiers if f.stage == 'decode'
                     and type(f.remaining_output_limit) is int)
        pressure_inputs=dict(table=self.performance.pressure_table,
                             expected_identity=self.performance.pressure_table.identity)
        if self.decode_guard is not None:
            pressure_inputs.update(causal_guard=self.decode_guard,
                                   expected_guard_identity=self.decode_guard.identity,
                                   iteration_guard=self.iteration_guard)
        if self.solo_envelope is not None:
            pressure_inputs.update(solo_envelope=self.solo_envelope,
                expected_solo_identity=self.solo_envelope.identity,
                partition_to_sm=tuple((partition,s.sm_counts[partition][1])
                    for partition in self.candidate_partitions))
        if self.fullcard_envelope is not None:
            recovery_sm=s.sm_counts[self.decode_recovery_partition][1]
            if recovery_sm != 132:
                raise ValueError('H100 full-card Decode component requires 132 SM')
            pressure_inputs.update(
                fullcard_envelope=self.fullcard_envelope,
                expected_fullcard_identity=self.fullcard_envelope.identity,
                fullcard_partition_to_sm=((self.decode_recovery_partition,recovery_sm),))
        has_ann=any(f.stage in ('ann_pending','ann_inflight') for f in stamp.frontiers)
        prefill_inputs=dict(predictor=model,expected_identity=model.identity,
                            nearest_domain_fallback=self.calibrated)
        if self.calibrated and has_ann:
            prefill_inputs.update(corrections=self.corrections,
                expected_correction_identity=self.corrections.identity,overlap_class='ann')
        capture=dict(capture_inputs=dict(
                prompt_catalog=dict(self.performance.reference_catalog),estimated_prompt_tokens=1024,
                e2e_slo_ms=10000.,horizon_ms=self.horizon_ms,
                engineering_unpriced_work=not self.calibrated,
                forecast_cache_tree=True,
                offline_cpu_profile=self.cpu_ann_profile,
                inflight_retrieval_inputs=dict(release_profile=self.release_profile,
                                               collection_profile=self.collection_profile,
                                               collection_mode='worker')),
            allocation_by_stream=self.mapping,decode_growth_tokens=growth,reserve_tokens=0,
            target_decode_growth_forecast=lambda *args:growth,
            compute_settings=dict(candidate_streams=self.candidate_partitions,service_profile=self.ann_profile,
                prefill_profile=None,decode_profile=None,decode_batch_size=running,
                microbatch_limit=self.ann_limit,credit_ms=4.,decode_target_ms=35.,safety_margin_ms=1.,
                bundle_limit=8,prefill_cap_idx=1,adaptive_idx=len(self.mapping)-1,
                decode_recovery_stream=self.decode_recovery_partition),
            prefill_stage_model_inputs=prefill_inputs,
            decode_stage_model_inputs=dict(predictor=model,expected_identity=model.identity,
                                           nearest_domain_fallback=self.calibrated),
            ann_pressure_inputs=pressure_inputs,
            transition_profiles=self.transitions,release_profile=self.release_profile,
            backing_inputs={})
        # Existing backing is not guessed here. The following native capture
        # must validate owner coordinates/pool before publishing active:kv.
        policy=s._capture_rag_prospective_completion_policy()
        coords=s._capture_rag_decode_kv_coordinates(source_execution=stamp,completion_policy=policy)
        pool=s._capture_rag_prospective_pool(source_execution=stamp,completion_policy=policy)
        if capture_execution_stamp(s)!=stamp:
            return unavailable('stale_provider_capture')
        if coords is None:
            return unavailable('native_decode_kv_coordinates')
        if pool is None:
            return unavailable('native_published_pool')
        capture['backing_inputs']['existing_ready']=(('active:kv',0.),)
        release=None
        if any(getattr(f,'inflight_prefill_end_layer',None) is not None for f in stamp.frontiers):
            release=s._capture_rag_prefill_release_observation(
                window_started_at=window_started_at,source_execution=stamp)
            if capture_execution_stamp(s)!=stamp:
                return unavailable('stale_prefill_release_capture')
            if release is None:
                return unavailable('native_prefill_release_observation')
        return dict(capture=capture,callbacks_from_context=lambda context:
            self.callbacks(context,release_observation=release))

    def callbacks(self, context, *, release_observation=None):
        """Bind actual captured work to generation/retrieval/prospective adapters."""
        s=self.scheduler
        work=context.get('work_inputs')
        if work is None or work.get('physical_pool') is None or work.get('prospective_settings') is None:
            s.rag_joint_engineering_missing_input='native_pool_or_admission_settings'
            return None
        stamp,refs,urgency,cpu,shapes,budget=work['common']
        if capture_execution_stamp(s)!=stamp:
            s.rag_joint_engineering_missing_input='stale_context'
            return None
        executor=getattr(s,'ann_executor',None)
        worker=getattr(executor,'_cold_batcher',None)
        max_cpu_batch=getattr(worker,'max_batch_queries',None)
        topk=getattr(executor,'topk',None)
        if (type(max_cpu_batch) is not int or max_cpu_batch<=0 or type(topk) is not int or topk<=0):
            s.rag_joint_engineering_missing_input='native_retrieval_configuration'
            return None
        # Conservative engineering reservation from observed queued shapes.
        # It is not a claim that these slots are physically allocated/free.
        reserved=sum(shape.context_tokens for shape in shapes)
        reservations=[]
        for f in stamp.frontiers:
            if f.stage in ('decode','prefill_active','prefill_merge_pending'):
                if type(f.prompt_tokens) is not int or type(f.remaining_output_limit) is not int:
                    s.rag_joint_engineering_missing_input='native_request_limits'
                    return None
                generated=f.generated_tokens or 0
                if type(generated) is not int or generated<0:return None
                reservations.append((f.request_id,f.prompt_tokens+generated+f.remaining_output_limit))
        work_by_allocation={};prospective={};timing={}
        layers=s.model_config.num_hidden_layers
        # Safe continuation must exist for the current allocation, including
        # unprofiled pure-prefill/decode edge partitions.  Only profiled mixed
        # partitions receive calibrated timing and can be ranked as switches.
        partitions=self.mapping.items()
        prefill_profile=(context.get('compute_inputs') or {}).get('prefill_profile')
        has_decode=any(f.stage=='decode' for f in stamp.frontiers)
        history=tuple(getattr(s,'_rag_dm2_window',()))
        iteration_history=tuple(getattr(s,'_rag_iteration_overhead_window',()))
        gpu_profile=None
        if self.calibrated:
            from sglang.srt.multiplex.lease_profile import bind_linear_ann_work_profile
            gpu_profile=bind_linear_ann_work_profile(self.performance.stage_model,
                expected_identity=self.performance.stage_model.identity,
                partition_to_sm={partition:s.sm_counts[partition][1]
                                 for partition in self.candidate_partitions},
                corrections=self.corrections if has_decode else None,
                expected_correction_identity=(self.corrections.identity if has_decode else ()),
                overlap_class='decode' if has_decode else 'solo',
                nearest_domain_fallback=True,
                correction_guard_fallback=has_decode)
        for partition,allocation in partitions:
            profiled=partition in self.candidate_partitions
            prefill_sm,decode_sm=s.sm_counts[partition]
            prefill_overlap=lambda projection:('ann'
                if projection.executable.ann_batch else 'solo')
            prefill_model=(None if not (self.calibrated and prefill_sm>0) else dict(
                predictor=self.performance.stage_model,
                expected_identity=self.performance.stage_model.identity,
                total_layers=layers,zero_cached_fallback=True,
                nearest_domain_fallback=True,
                **(dict(corrections=self.corrections,
                    expected_correction_identity=self.corrections.identity,
                    overlap_class=prefill_overlap) if profiled else
                   dict(overlap_class='solo'))))
            query_decode=lambda shape:10.
            if self.calibrated and decode_sm>0:
                from sglang.srt.multiplex.full_lease import bind_decode_stage_model
                interference=None
                if profiled:
                    guard=self.decode_guard
                    iteration_guard=self.iteration_guard
                    def interference(shape,solo,g=guard,h=history,ig=iteration_guard,
                                     ih=iteration_history):
                        floor=g.predict_ms(len(shape.request_ids),'ann_prefill',h,
                                           expected_identity=g.identity)
                        service=None if floor is None else max(solo,floor)
                        return (service if service is None or ig is None else
                                ig.predict_total_ms(service,ih))
                query_decode=bind_decode_stage_model(self.performance.stage_model,
                    sm_count=decode_sm,
                    expected_identity=self.performance.stage_model.identity,
                    ann_coexecution=profiled,interference_query=interference,
                    nearest_domain_fallback=True)
            generation=dict(total_layers=layers,query_decode_service=query_decode,
                release_observation=release_observation,
                existing_kv_key='active:kv',new_kv_key='joint:kv',
                kv_capture_inputs=dict(reserved_tokens=reserved,source_execution=stamp),
                observation_offset_ms=cpu[2],separate_commit_gate=True)
            retrieval=dict(frozen_cpu_profile=self.cpu_ann_profile,
                frozen_probe_profile=gpu_profile,
                frozen_collection_profile=self.collection_profile,
                partition=partition,max_batch_queries=max_cpu_batch,
                observation_offset_ms=cpu[2],admit_offset_ms=cpu[2],compute_ready_ms=0.,
                scratch_key='joint:scratch',index_key='joint:index',collection_mode='worker',
                topk=topk,prefix='engineering:retrieval')
            work_by_allocation[allocation]=dict(generation_inputs=generation,retrieval_inputs=retrieval)
            if self.calibrated:
                def overlap(projection):
                    ann=bool(projection.executable.ann_batch)
                    prefill=bool(projection.executable.prefill_layers)
                    return ('ann_prefill' if ann and prefill else 'ann' if ann else
                            'prefill' if prefill else 'solo')
                timing[allocation]=dict(cpu_profile=self.cpu_ann_profile,
                    gpu_profile=gpu_profile,collection_profile=self.collection_profile,
                    input_preparation_profile=self.release_profile,
                    prefill_inputs=dict(partition=partition,layer_profile=prefill_profile,
                        release_profile=self.release_profile,
                        release_observation=release_observation),
                    prefill_model_inputs=prefill_model,
                    handoff_timing=self.timings.price)
                if has_decode and decode_sm>0:
                    timing[allocation]['decode_model_inputs']=dict(
                        predictor=self.performance.stage_model,
                        expected_identity=self.performance.stage_model.identity,
                        nearest_domain_fallback=True,
                        overlap_class=overlap if profiled else 'solo')
            query_prefill=(None if prefill_profile is None else
                lambda shape,p=partition,profile=prefill_profile:
                    profile.query_guard_shape_ms(p,shape))
            prospective[allocation]=dict(reservations=tuple(reservations),
                conservative_cache_reserve=False,
                kv_key='joint:kv',existing_kv_key='active:kv',compute_ready_ms=0.,
                query_prefill=(None if prefill_model is not None else query_prefill if self.calibrated else
                    lambda shape:float(shape.end_layer-shape.start_layer)),
                time_handoff=self.timings.price,separate_commit_gate=True)
            if prefill_model is not None:
                prospective[allocation]['prefill_model_inputs']=prefill_model
        s.rag_joint_engineering_missing_input=None
        return dict(work_by_allocation=work_by_allocation,timing_by_allocation=timing,
            horizon_ms=self.horizon_ms,switching_margin=0.,
            engineering_operation_timings=None if self.calibrated else self.timings,
            prospective_by_allocation=prospective if context['mode']=='full' else None)


def _optional_predictor_bundle(scheduler):
    names=('RAGLEASE_ENGINEERING_PREDICTOR_MANIFEST',
           'RAGLEASE_ENGINEERING_PREDICTOR_MANIFEST_SHA256',
           'RAGLEASE_ENGINEERING_PREDICTOR_ROOT',
           'RAGLEASE_ENGINEERING_PREDICTOR_IDENTITY_JSON')
    values={name:os.environ.get(name,'') for name in names}
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise ValueError('All engineering predictor bundle pins are required together')
    try:
        identity=tuple(json.loads(values[names[3]]))
    except (TypeError,ValueError) as error:
        raise ValueError('Valid engineering predictor identity JSON required') from error
    if not identity or any(not isinstance(value,str) or not value for value in identity):
        raise ValueError('Nonempty string engineering predictor identity required')
    mapping=_profiled_mixed_partitions(scheduler.sm_counts)
    from sglang.srt.multiplex.engineering_predictor_bundle import (
        load_engineering_predictor_bundle,
    )
    return load_engineering_predictor_bundle(values[names[0]],
        expected_sha256=values[names[1]],artifact_root=values[names[2]],
        expected_identity=identity,expected_partition_to_sm=mapping)


def _optional_calibrated_bundle(scheduler):
    h100_names=('RAGLEASE_H100_HYBRID_PREDICTOR_MANIFEST',
                'RAGLEASE_H100_HYBRID_PREDICTOR_MANIFEST_SHA256',
                'RAGLEASE_H100_HYBRID_PREDICTOR_ROOT',
                'RAGLEASE_H100_HYBRID_PREDICTOR_IDENTITY_JSON')
    h100_values={name:os.environ.get(name,'') for name in h100_names}
    names=('RAGLEASE_CALIBRATED_PREDICTOR_MANIFEST',
           'RAGLEASE_CALIBRATED_PREDICTOR_MANIFEST_SHA256',
           'RAGLEASE_CALIBRATED_PREDICTOR_ROOT',
           'RAGLEASE_CALIBRATED_PREDICTOR_IDENTITY_JSON')
    values={name:os.environ.get(name,'') for name in names}
    if any(h100_values.values()):
        if any(values.values()) or not all(h100_values.values()):
            raise ValueError('Choose one complete calibrated or H100 hybrid predictor bundle')
        try:identity=tuple(json.loads(h100_values[h100_names[3]]))
        except (TypeError,ValueError) as error:
            raise ValueError('Valid H100 hybrid predictor identity JSON required') from error
        mapping=_profiled_mixed_partitions(scheduler.sm_counts)
        from sglang.srt.multiplex.h100_hybrid_predictor_bundle import (
            load_h100_hybrid_predictor_bundle,
        )
        return load_h100_hybrid_predictor_bundle(
            h100_values[h100_names[0]],expected_sha256=h100_values[h100_names[1]],
            artifact_root=h100_values[h100_names[2]],expected_identity=identity,
            expected_partition_to_sm=mapping)
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise ValueError('All calibrated predictor bundle pins are required together')
    try:identity=tuple(json.loads(values[names[3]]))
    except (TypeError,ValueError) as error:
        raise ValueError('Valid calibrated predictor identity JSON required') from error
    mapping=_profiled_mixed_partitions(scheduler.sm_counts)
    from sglang.srt.multiplex.calibrated_predictor_bundle import (
        load_calibrated_predictor_bundle,
    )
    return load_calibrated_predictor_bundle(values[names[0]],
        expected_sha256=values[names[1]],artifact_root=values[names[2]],
        expected_identity=identity,expected_partition_to_sm=mapping)


def _required_cpu_ann_profile():
    names=('RAGLEASE_CPU_ANN_PROFILE','RAGLEASE_CPU_ANN_PROFILE_SHA256',
           'RAGLEASE_CPU_ANN_IDENTITY_JSON')
    values={name:os.environ.get(name,'') for name in names}
    if not all(values.values()):
        raise ValueError('Accepted CPU ANN profile pins are required in calibrated mode')
    try:identity=tuple(json.loads(values[names[2]]))
    except (TypeError,ValueError) as error:
        raise ValueError('Valid CPU ANN identity JSON required') from error
    from sglang.srt.multiplex.lease_profile import load_accepted_cpu_ann_profile
    return load_accepted_cpu_ann_profile(values[names[0]],
        expected_sha256=values[names[1]],expected_identity=identity)


def build(scheduler):
    raw=os.environ.get('RAGLEASE_ENGINEERING_OUTPUT_TOKENS','')
    if not raw.isdecimal() or int(raw)<=0:
        raise ValueError('Explicit RAGLEASE_ENGINEERING_OUTPUT_TOKENS required')
    calibrated=_optional_calibrated_bundle(scheduler)
    bundle=None if calibrated is not None else _optional_predictor_bundle(scheduler)
    cpu_profile=_required_cpu_ann_profile() if calibrated is not None else None
    provider=EngineeringLiveProvider(scheduler,output_tokens=int(raw),
                                     predictor_bundle=bundle,
                                     calibrated_bundle=calibrated,
                                     cpu_ann_profile=cpu_profile)
    current=get_current_stream_idx()
    prefill_sm,ann_decode_sm=scheduler.sm_counts[current]
    initial_compute=ComputeLease(
        provider.mapping[current],
        min(provider.ann_limit,4) if current in provider.candidate_partitions else 0,
        min(8,scheduler.model_config.num_hidden_layers) if prefill_sm>0 else 0)
    return EngineeringJointBootstrap(provider,initial_compute,
        tuple(provider.mapping.items()),provider.performance.prediction_source)


def build_legacy_predictor(scheduler):
    """Load the accepted bundle without replacing legacy scheduling policy."""
    calibrated=_optional_calibrated_bundle(scheduler)
    if calibrated is None:
        raise ValueError('Calibrated predictor pins are required for legacy adapter')
    formal_ready=bool(calibrated.formal_ready)
    source=(calibrated.prediction_source if not formal_ready else
        'accepted ANN/Prefill/Decode models and coexecution guards; '
        'legacy frontend scheduling and admission policy retained')
    return CalibratedLegacyPredictorBootstrap(calibrated,source,formal_ready)
