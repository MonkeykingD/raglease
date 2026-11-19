"""Explicit engineering bootstrap at the real PDMux idle startup boundary.

This is NOT a calibrated deployment or a fallback to fabricated queue/KV state.
An in-package factory assembles prediction settings and a live-input provider;
the existing native initializer and per-event safety/activation gates remain.
"""
from dataclasses import dataclass
import importlib
import os
import re


def _finite_env_ms(name, default):
    raw=os.environ.get(name,str(default))
    try:
        value=float(raw)
    except (TypeError,ValueError):
        raise ValueError('%s must be finite milliseconds' % name)
    if not value>=0. or value==float('inf'):
        raise ValueError('%s must be finite nonnegative milliseconds' % name)
    return value


@dataclass(frozen=True)
class EngineeringJointBootstrap:
    provider: object
    initial_compute: object
    allocation_by_stream: tuple
    prediction_source: str


@dataclass(frozen=True)
class CalibratedLegacyPredictorBootstrap:
    """Pinned phase models attached to the legacy/dynamic-only scheduler.

    ``formal_ready`` records whole-bundle acceptance; component acceptance
    alone must not promote an engineering integration to a formal runtime.
    """
    bundle: object
    prediction_source: str
    formal_ready: bool = True


def bootstrap_engineering_joint_runtime(scheduler, *, factory_spec=None):
    spec = (os.environ.get('RAGLEASE_ENGINEERING_JOINT_FACTORY','')
            if factory_spec is None else factory_spec)
    if not spec:
        if (getattr(scheduler, 'rag_policy_mode', None) == 'independent_adaptive'
                and not callable(getattr(scheduler, 'rag_joint_event_input_provider', None))):
            raise ValueError('Independent-adaptive requires an installed joint provider')
        return False  # Explicit opt-in; preserve historical/controlled runtimes.
    mode=getattr(scheduler,'rag_policy_mode',None)
    if (mode not in ('full','independent_adaptive','dynamic-only')
            or getattr(scheduler,'rag_joint_event_input_provider',None) is not None
            or getattr(scheduler,'rag_joint_bootstrap_receipt',None) is not None):
        raise ValueError('Predictor bootstrap requires an uninitialized supported policy')
    if not isinstance(spec,str) or not re.fullmatch(
            r'sglang\.srt\.multiplex\.[A-Za-z_]\w*:[A-Za-z_]\w*',spec):
        raise ValueError('Explicit in-package engineering factory module:function required')
    module, name = spec.split(':')
    factory = getattr(importlib.import_module(module),name)
    if not callable(factory):
        raise ValueError('Callable engineering provider factory required')
    config = factory(scheduler)
    if isinstance(config,CalibratedLegacyPredictorBootstrap):
        if mode not in ('full','dynamic-only'):
            raise ValueError('Legacy predictor bootstrap requires full legacy or dynamic-only policy')
        if (not isinstance(config.prediction_source,str)
                or not config.prediction_source.strip()):
            raise ValueError('Explicit calibrated predictor provenance required')
        scheduler._initialize_rag_legacy_calibrated_predictor(config.bundle)
        scheduler.rag_joint_bootstrap_receipt=dict(
            schema='raglease.legacy-calibrated-predictor-bootstrap.v1',
            factory=spec,prediction_source=config.prediction_source,
            engineering_only=not config.formal_ready,
            formal_ready=config.formal_ready,performance_evidence=False)
        return True
    if mode=='dynamic-only':
        raise ValueError('Dynamic-only factory must return a legacy predictor bootstrap')
    if (not isinstance(config,EngineeringJointBootstrap) or not callable(config.provider)
            or not isinstance(config.prediction_source,str) or not config.prediction_source.strip()
            or not isinstance(config.allocation_by_stream,tuple)):
        raise ValueError('Explicit engineering provider, limits and prediction provenance required')
    mapping=dict(config.allocation_by_stream)
    if len(mapping)!=len(config.allocation_by_stream):
        raise ValueError('Duplicate startup stream mapping')
    min_replan_ms=_finite_env_ms('RAGLEASE_JOINT_MIN_REPLAN_MS',25.)
    max_replan_ms=_finite_env_ms('RAGLEASE_JOINT_MAX_REPLAN_MS',250.)
    if max_replan_ms<min_replan_ms or max_replan_ms<=0.:
        raise ValueError('Joint max replan interval must cover the positive minimum')
    # Validate every bootstrap setting before the initializer publishes the
    # provider.  The initializer checks actual partition and idle state and
    # never allocates HBM or dispatches requests.
    scheduler._initialize_rag_joint_runtime(provider=config.provider,
        initial_compute=config.initial_compute,allocation_by_stream=mapping)
    scheduler.rag_joint_min_replan_ms=min_replan_ms
    scheduler.rag_joint_max_replan_ms=max_replan_ms
    scheduler.rag_joint_bootstrap_receipt=dict(schema='raglease.engineering-bootstrap.v1',
        factory=spec,prediction_source=config.prediction_source,
        min_replan_ms=min_replan_ms,max_replan_ms=max_replan_ms,
        engineering_only=True,formal_ready=False,performance_evidence=False)
    return True
