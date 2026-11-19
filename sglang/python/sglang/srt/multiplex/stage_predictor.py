"""Small calibrated phase predictor inspired by MuxWise, without CUDA state.

Provenance: ykcombat/sglang, slo_scheduler commit
a068b3e484b09325d926c920643d59ca20084ca0, multiplex/predictor.py.
Reuse its normalized polynomial form, NOT its model-specific coefficients,
SM rounding, minimum-token clamp, or hardcoded SLOs. Prefill uses per-request
feature sums as in the paper rather than squaring the batch's token sum.
Outputs are solo phase milliseconds, not P99 guarantees or end-to-end utility.
"""
from dataclasses import dataclass
import math
import hashlib
import json
import statistics
from pathlib import Path


def _linear_percentile(values, q):
    """NumPy-default compatible percentile for a small frozen history."""
    ordered = sorted(values)
    if not ordered:
        return 0.
    position = (len(ordered)-1)*q
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


@dataclass(frozen=True)
class CausalDecodeGuard:
    """Frozen causal P99 floor; it is not a candidate-performance model.

    The feature contract matches the accepted v162 boundary behaviour: the
    first prediction uses zero history, later predictions use only completed
    decode iterations, and windows contain at most 50 observations. ANN batch,
    partition and proposed ANN GPU work are intentionally absent; those belong
    to FrozenANNPressureTable. Joint prediction composes the two as a pointwise
    maximum, never as an additive latency correction.
    """
    identity: tuple
    coefficients: tuple
    tau: float = .99
    max_history: int = 50

    feature_order = ('intercept','batch_size','hist10_p50','hist10_p90',
        'hist10_max','hist50_p50','hist50_p90','hist50_max','cont10_frac',
        'cont50_frac','overlap_class=ann_prefill','overlap_class=prefill',
        'overlap_class=solo')
    overlap_levels = ('ann','ann_prefill','prefill','solo')
    contended_levels = ('prefill','ann_prefill')

    def __post_init__(self):
        if (not isinstance(self.identity,tuple) or not self.identity
                or any(not isinstance(v,str) or not v for v in self.identity)
                or not isinstance(self.coefficients,tuple)
                or len(self.coefficients)!=len(self.feature_order)
                or any(type(v) not in (int,float) or not math.isfinite(v)
                       for v in self.coefficients)
                or type(self.tau) not in (int,float) or self.tau!=.99
                or type(self.max_history) is not int or self.max_history!=50):
            raise ValueError('Pinned v162-compatible causal decode guard required')

    def predict_ms(self, batch_size, overlap, history, *, expected_identity):
        if expected_identity!=self.identity:
            raise ValueError('Causal decode guard belongs to another deployment')
        if type(batch_size) is not int or batch_size<=0 or overlap not in self.overlap_levels:
            raise ValueError('Positive decode batch and known overlap class required')
        rows=tuple(history)
        if len(rows)>self.max_history:
            raise ValueError('Causal history exceeds frozen window')
        for row in rows:
            if (not isinstance(row,tuple) or len(row)!=2
                    or type(row[0]) not in (int,float) or not math.isfinite(row[0])
                    or row[0]<=0 or row[1] not in self.overlap_levels):
                raise ValueError('Finite completed decode observations required')
        features={'intercept':1.,'batch_size':float(batch_size)}
        for window in (10,50):
            recent=rows[-window:]
            values=[float(ms) for ms,_ in recent]
            features[f'hist{window}_p50']=_linear_percentile(values,.5)
            features[f'hist{window}_p90']=_linear_percentile(values,.9)
            features[f'hist{window}_max']=max(values) if values else 0.
            features[f'cont{window}_frac']=(
                sum(cls in self.contended_levels for _,cls in recent)/len(recent)
                if recent else 0.)
        for level in self.overlap_levels[1:]:
            features[f'overlap_class={level}']=float(overlap==level)
        value=sum(coef*features[name] for name,coef in
                  zip(self.feature_order,self.coefficients))
        return max(.001,value) if math.isfinite(value) else None


@dataclass(frozen=True)
class CausalIterationOverheadGuard:
    """Add observed host/scheduler overhead to candidate GPU service.

    Offline decode fits and ANN-pressure cells in the calibrated bundle use
    device milliseconds.  The serving SLO, however, applies to a complete
    decode iteration.  This guard keeps those two clocks explicit: every
    history row contains one *completed* same-partition iteration and its GPU
    service, and a candidate's own device-time estimate receives the causal
    tail of ``max(0, iteration_ms - gpu_ms)``.  It never relabels the accepted
    device-time artifact and never reads an unfinished or future iteration.
    """
    tau: float = .99
    max_history: int = 50
    cold_start_margin: float = 1.25

    def __post_init__(self):
        if (type(self.tau) not in (int,float) or not 0.<self.tau<=1.
                or type(self.max_history) is not int or self.max_history<=0
                or type(self.cold_start_margin) not in (int,float)
                or not math.isfinite(self.cold_start_margin)
                or self.cold_start_margin<1.):
            raise ValueError('Explicit causal iteration-overhead policy required')

    def predict_total_ms(self, service_ms, history):
        if (type(service_ms) not in (int,float) or not math.isfinite(service_ms)
                or service_ms<=0):
            raise ValueError('Positive finite candidate GPU service required')
        rows=tuple(history)
        if len(rows)>self.max_history:
            raise ValueError('Causal iteration history exceeds frozen window')
        overhead=[]
        for row in rows:
            if (not isinstance(row,tuple) or len(row)!=3
                    or any(type(v) not in (int,float) or not math.isfinite(v)
                           or v<=0 for v in row[:2])
                    or row[2] not in CausalDecodeGuard.overlap_levels):
                raise ValueError('Completed iteration/GPU observations required')
            overhead.append(max(0.,float(row[0])-float(row[1])))
        if not overhead:
            return float(service_ms)
        tail=(_linear_percentile(overhead,self.tau) if len(overhead)>=4
              else max(overhead)*self.cold_start_margin)
        value=float(service_ms)+tail
        return value if math.isfinite(value) else None


def _load_causal_decode_guard(path, *, expected_sha256, expected_identity,
                              engineering):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Causal decode guard artifact hash mismatch')
    artifact=json.loads(payload)
    assessment=artifact.get('assessment',{})
    readiness = ((artifact.get('formal_ready') is False
                  and artifact.get('engineering_ready') is True
                  and artifact.get('intended_use')=='engineering_bootstrap')
                 if engineering else
                 (artifact.get('formal_ready') is True
                  and artifact.get('intended_use','formal')=='formal'))
    if (artifact.get('schema')!='raglease.causal-decode-guard.v1'
            or not readiness or assessment.get('accepted') is not True
            or assessment.get('false_safe_count') is None
            or assessment.get('evaluated_rows',0)<=0):
        raise ValueError('Causal decode guard did not pass held-out assessment')
    model=artifact.get('model',{})
    if tuple(model.get('feature_order',()))!=CausalDecodeGuard.feature_order:
        raise ValueError('Causal decode guard feature order mismatch')
    identity=tuple(model.get('identity',()))
    if identity!=expected_identity:
        raise ValueError('Causal decode guard belongs to another deployment')
    return CausalDecodeGuard(identity,tuple(model.get('coefficients',())),
                             model.get('tau'),model.get('max_history'))


def load_accepted_causal_decode_guard(path, *, expected_sha256,
                                      expected_identity):
    """Load a pinned formal held-out-assessed causal guard."""
    return _load_causal_decode_guard(path,expected_sha256=expected_sha256,
        expected_identity=expected_identity,engineering=False)


def load_engineering_causal_decode_guard(path, *, expected_sha256,
                                         expected_identity):
    """Load an explicit reduced-sample guard without granting formal status."""
    return _load_causal_decode_guard(path,expected_sha256=expected_sha256,
        expected_identity=expected_identity,engineering=True)


def load_native_bundle_query(path, *, expected_sha256, expected_runtime_id,
                             expected_dtype, expected_backend, expected_model,
                             partition_to_sm, bundle_layers):
    """Bind measured solo interval means; no extrapolation or P99 guarantee.

    A single measured shape is deliberately not a validated general predictor.
    Partition IDs are mapped explicitly to actual SM counts, never rounded.
    Select one traversal grain to avoid mixing differently measured chains.
    """
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Native bundle artifact hash mismatch')
    data = json.loads(payload)
    args = data['args']
    if (data.get('schema') != 'raglease.native-prefill-bundles.v1'
            or data.get('execution') != 'solo_native_split_prefill_zero_prefix'
            or data.get('logits_agreement') is not True
            or args['runtime_id'] != expected_runtime_id
            or args['model'] != expected_model
            or data['dtype'] != expected_dtype
            or data['attention_backend'] != expected_backend):
        raise ValueError('Native bundle execution provenance mismatch')
    total, count, warmup = data['total_layers'], args['samples'], args['warmup']
    if (any(type(n) is not int or n <= 0 for n in
            (total, count, bundle_layers, args['batch'], args['length'], data['actual_sm']))
            or type(warmup) is not int or warmup < 0
            or data['actual_sm'] != args['sm']):
        raise ValueError('Invalid native measurement dimensions')
    bounds = tuple((a, min(a+bundle_layers, total)) for a in range(0,total,bundle_layers))
    rows = [row for row in data['rows'] if row['bundle_layers'] == bundle_layers]
    if (len(rows) != count or {row['repetition'] for row in rows} != set(range(warmup,warmup+count))):
        raise ValueError('Incomplete native bundle repetitions')
    values = {bound: [] for bound in bounds}
    for row in rows:
        intervals = row['intervals']
        if tuple((i['start_layer'],i['end_layer']) for i in intervals) != bounds:
            raise ValueError('Incomplete native layer traversal')
        for i, bound in zip(intervals,bounds):
            value = i['gpu_ms']
            if (type(value) not in (int,float) or not math.isfinite(value) or value <= 0
                    or i['includes_embedding'] is not (bound[0] == 0)
                    or i['includes_logits_and_sample'] is not (bound[1] == total)):
                raise ValueError('Invalid native interval timing')
            values[bound].append(value)
    means = {bound: statistics.mean(samples) for bound,samples in values.items()}
    mapping = dict(partition_to_sm)
    if any(type(p) is not int or p < 0 or type(sm) is not int or sm <= 0
           for p,sm in mapping.items()):
        raise ValueError('Explicit partition to actual SM mapping required')
    expected_tokens = (args['length'],)*args['batch']
    def query(partition, shape):
        if (mapping.get(partition) != data['actual_sm']
                or tuple(shape.extend_tokens) != expected_tokens
                or tuple(shape.context_tokens) != expected_tokens
                or len(shape.request_ids) != args['batch']
                or len(set(shape.request_ids)) != len(shape.request_ids)):
            return None
        return means.get((shape.start_layer,shape.end_layer))
    return query


def _tokens(values, name, *, positive=False):
    values = tuple(values)
    if not values or any(type(v) is not int or v < int(positive) for v in values):
        raise ValueError('Invalid ' + name)
    return values


def prefill_features(new_tokens, cached_tokens):
    new = _tokens(new_tokens, 'new prefill tokens', positive=True)
    cached = _tokens(cached_tokens, 'cached prefill tokens')
    if len(new) != len(cached):
        raise ValueError('One cached length per prefill request required')
    return (sum((n/1024.)**2 for n in new),
            sum((n/1024.)*(r/1024.) for n, r in zip(new, cached)),
            sum(new)/1024., 1.)


def ann_features(batch_size, list_visits, vectors_scanned,
                 max_query_vectors, max_list_vectors,
                 query_vector_sq_sum=None):
    """Paper ANN features for the admitted GPU execution unit.

    L_A and V_A describe routed resident work, not the complete index.
    Keep their native count units in the public contract; the offline fitter
    scales columns internally for conditioning and exports native coefficients.
    """
    values=(batch_size,list_visits,vectors_scanned,
            max_query_vectors,max_list_vectors)
    if any(type(v) is not int or v<=0 for v in values):
        raise ValueError('Positive ANN batch, routed work and imbalance features required')
    if max_query_vectors>vectors_scanned or max_list_vectors>max_query_vectors:
        raise ValueError('ANN imbalance features exceed total routed work')
    base=(float(batch_size),float(list_visits),float(vectors_scanned),
          float(max_query_vectors),float(max_list_vectors))
    if query_vector_sq_sum is None:
        return base+(1.,)
    if (type(query_vector_sq_sum) is not int or query_vector_sq_sum<=0
            or query_vector_sq_sum<max_query_vectors*max_query_vectors
            or query_vector_sq_sum>max_query_vectors*vectors_scanned):
        raise ValueError('Invalid ANN per-query vector square sum')
    return base+(float(query_vector_sq_sum),1.)


def ann_correction_features(batch_size, list_visits, vectors_scanned):
    """Coarse prospective bucket coordinates for ANN slowdown factors.

    The allocation-specific solo model uses the richer imbalance-aware feature
    vector above.  Existing slowdown cells remain prospectively bucketed by
    total admitted work so adding the solo features does not post-hoc redefine
    the coexecution experiment grid.
    """
    values=(batch_size,list_visits,vectors_scanned)
    if any(type(v) is not int or v<=0 for v in values):
        raise ValueError('Positive ANN correction-bucket features required')
    return float(batch_size),float(list_visits),float(vectors_scanned),1.


def decode_features(cached_tokens):
    cached = _tokens(cached_tokens, 'decode context tokens')
    return sum(cached)/1024., float(len(cached)), 1.


@dataclass(frozen=True)
class CoexecutionCorrectionCell:
    stage: str
    partition: int
    overlap: str
    feature_bounds: tuple
    nominal_factor: float
    guard_factor: float
    samples: int

    def __post_init__(self):
        allowed={'ann':('prefill','decode','prefill_decode'),
                 'prefill':('ann','decode','ann_decode')}
        if (self.stage not in allowed or type(self.partition) is not int
                or self.partition<0 or self.overlap not in allowed[self.stage]
                or not isinstance(self.feature_bounds,tuple)
                or len(self.feature_bounds)!=3 or type(self.samples) is not int
                or self.samples<=0):
            raise ValueError('Invalid coexecution correction cell coordinates')
        for bounds in self.feature_bounds:
            if (not isinstance(bounds,tuple) or len(bounds)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v)
                           for v in bounds) or not 0<=bounds[0]<bounds[1]):
                raise ValueError('Finite half-open correction ranges required')
        if (type(self.nominal_factor) not in (int,float)
                or type(self.guard_factor) not in (int,float)
                or not math.isfinite(self.nominal_factor)
                or not math.isfinite(self.guard_factor)
                or not 0<self.nominal_factor<=self.guard_factor):
            raise ValueError('Positive ordered slowdown factors required')


@dataclass(frozen=True)
class FrozenCoexecutionCorrections:
    """Measured/solo ratios for matching execution units and allocations."""
    identity: tuple
    cells: tuple

    def __post_init__(self):
        if (not isinstance(self.identity,tuple) or not self.identity
                or any(not isinstance(v,str) or not v for v in self.identity)
                or not isinstance(self.cells,tuple)
                or any(not isinstance(c,CoexecutionCorrectionCell) for c in self.cells)):
            raise ValueError('Pinned immutable correction table required')

    def query(self, stage, partition, overlap, features, *, expected_identity):
        if expected_identity!=self.identity:
            raise ValueError('Coexecution corrections belong to another deployment')
        values=tuple(features)
        if len(values)!=4 or values[-1]!=1. or any(
                type(v) not in (int,float) or not math.isfinite(v) or v<0
                for v in values[:-1]):
            raise ValueError('Three finite workload features and intercept required')
        matches=[c for c in self.cells if c.stage==stage and c.partition==partition
            and c.overlap==overlap and all(lo<=v<hi for v,(lo,hi) in
                zip(values[:-1],c.feature_bounds))]
        if len(matches)>1:
            raise ValueError('Ambiguous coexecution correction cells')
        if not matches:
            return None
        cell=matches[0]
        return cell.nominal_factor,cell.guard_factor


def load_accepted_corrections(path, *, expected_sha256, expected_plan_sha256,
                              expected_identity):
    """Load only a held-out accepted slowdown table with pinned provenance."""
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Coexecution correction artifact hash mismatch')
    artifact=json.loads(payload)
    assessment=artifact.get('assessment',{})
    accepted_schemas={
        'raglease.coexecution-correction-assessment.v1',
        'raglease.coexecution-correction-refinement-assessment.v1',
    }
    if (artifact.get('schema') not in accepted_schemas
            or artifact.get('plan_sha256')!=expected_plan_sha256
            or assessment.get('accepted') is not True
            or not assessment.get('cells')
            or any(cell.get('passed') is not True for cell in assessment['cells'])):
        raise ValueError('Coexecution corrections did not pass held-out assessment')
    data=artifact.get('table',{})
    identity=tuple(data.get('identity',()))
    if identity!=expected_identity:
        raise ValueError('Coexecution corrections belong to another deployment')
    cells=tuple(CoexecutionCorrectionCell(c['stage'],c['partition'],c['overlap'],
        tuple(tuple(bounds) for bounds in c['feature_bounds']),c['nominal_factor'],
        c['guard_factor'],c['samples']) for c in data.get('cells',()))
    assessed={(c['stage'],c['partition'],c['overlap'],
               tuple(tuple(b) for b in c['feature_bounds']))
              for c in assessment['cells'] if c.get('passed') is True}
    actual={(c.stage,c.partition,c.overlap,c.feature_bounds) for c in cells}
    if not cells or actual!=assessed:
        raise ValueError('Correction cells and held-out assessment disagree')
    return FrozenCoexecutionCorrections(identity,cells)


def bind_layer_scaled_prefill_query(predictor, *, expected_identity,
                                    partition_to_sm, total_layers,
                                    corrections=None,
                                    expected_correction_identity=(),
                                    overlap_class='solo',
                                    zero_cached_fallback=False,
                                    nearest_domain_fallback=False):
    """MuxWise-style whole-phase estimate scaled to the executable layer span.

    This is a scheduling approximation, not a measured interval or P99 bound.
    Current known cached lengths enter the phase model; no cache-state simulator
    or future hit prediction is involved. Calibration coefficients/coverage stay
    unchanged, including unsupported returns from the underlying phase model.
    """
    if type(total_layers) is not int or total_layers <= 0:
        raise ValueError('Positive total layer count required')
    if predictor.identity != expected_identity:
        raise ValueError('Stage predictor calibration belongs to another deployment')
    corrected=corrections is not None
    if corrected != bool(expected_correction_identity):
        raise ValueError('Correction table and identity must be configured together')
    if corrected and overlap_class=='solo':
        raise ValueError('Coexecution corrections require a non-solo overlap')
    if (type(zero_cached_fallback) is not bool
            or type(nearest_domain_fallback) is not bool):
        raise ValueError('Explicit cached-prefix fallback policy required')
    mapping = dict(partition_to_sm)
    if any(type(p) is not int or p < 0 or type(sm) is not int or sm <= 0
           for p, sm in mapping.items()):
        raise ValueError('Explicit partition to actual SM mapping required')

    def query(partition, shape):
        sm = mapping.get(partition)
        start, end = shape.start_layer, shape.end_layer
        if (sm is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= total_layers):
            return None
        new, context = tuple(shape.extend_tokens), tuple(shape.context_tokens)
        ids = tuple(shape.request_ids)
        if (not new or len(new) != len(context) or len(ids) != len(new)
                or len(set(ids)) != len(ids)
                or any(type(n) is not int or type(c) is not int or not 0 < n <= c
                       for n, c in zip(new, context))):
            return None
        cached=tuple(c-n for n,c in zip(new,context))
        whole = predictor.predict_prefill_ms(sm, new,
            cached, expected_identity=expected_identity)
        if whole is None and zero_cached_fallback and any(cached):
            # MuxWise-level approximation: do not simulate future cache hits.
            # Reuse the accepted no-prefix model as a conservative proxy for
            # the same newly computed tokens; unsupported new-token shapes
            # remain unsupported.
            whole=predictor.predict_prefill_ms(sm,new,(0,)*len(cached),
                expected_identity=expected_identity)
        if whole is None and nearest_domain_fallback:
            whole=predictor.predict_prefill_nearest_ms(
                sm,new,cached,expected_identity=expected_identity)
        if whole is None:return None
        if corrected:
            factors=corrections.query('prefill',partition,overlap_class,
                prefill_features(new,cached),expected_identity=expected_correction_identity)
            if factors is None and zero_cached_fallback and any(cached):
                factors=corrections.query('prefill',partition,overlap_class,
                    prefill_features(new,(0,)*len(cached)),
                    expected_identity=expected_correction_identity)
            if factors is None and nearest_domain_fallback:
                guards=[cell.guard_factor for cell in corrections.cells
                        if cell.stage=='prefill' and cell.partition==partition
                        and cell.overlap==overlap_class]
                factors=None if not guards else (max(guards),max(guards))
            if factors is None:return None
            whole*=factors[0]
        return whole * (end-start) / total_layers

    return query


@dataclass(frozen=True)
class PhasePrefillComputeProfile:
    """Frozen actual cohort adapted to the compute chooser's layer interface.

    Token totals alone cannot encode prefix reuse or heterogeneous requests.
    Bind the same cohort used by the proposal, rather than interpreting a total
    as one request. The marker selects MuxWise's decode-matching layer count.
    """
    layer_ms_by_partition: tuple
    new_token_total: int
    shape_query: object
    muxwise_layer_bundles: bool = True
    guard_layer_ms_by_partition: tuple = ()
    guard_shape_query: object = None
    overlap_class: str = 'solo'

    def query_layer_ms(self, partition, token_count):
        if token_count != self.new_token_total:
            return None
        guarded=dict(self.guard_layer_ms_by_partition)
        return guarded.get(partition,dict(self.layer_ms_by_partition).get(partition))

    def query_nominal_layer_ms(self, partition, token_count):
        if token_count != self.new_token_total:
            return None
        return dict(self.layer_ms_by_partition).get(partition)

    def query_shape_ms(self, partition, shape):
        # Prospective cohorts may differ from the current compute proposal.
        # Evaluate their own new/cached lengths, never just the token total.
        return self.shape_query(partition, shape)

    def query_guard_shape_ms(self, partition, shape):
        query=self.guard_shape_query or self.shape_query
        return query(partition,shape)


def bind_prefill_compute_profile(predictor, *, expected_identity,
                                 partition_to_sm, total_layers,
                                 new_tokens, cached_tokens, corrections=None,
                                 expected_correction_identity=(), overlap_class='solo',
                                 nearest_domain_fallback=False):
    new = _tokens(new_tokens, 'new prefill tokens', positive=True)
    cached = _tokens(cached_tokens, 'cached prefill tokens')
    prefill_features(new, cached)  # validate matching per-request shapes
    if type(total_layers) is not int or total_layers <= 0:
        raise ValueError('Positive total layer count required')
    corrected=corrections is not None
    if corrected != bool(expected_correction_identity):
        raise ValueError('Correction table and identity must be configured together')
    if corrected and overlap_class=='solo':
        raise ValueError('Coexecution corrections require a non-solo overlap')
    values = [];guards=[]
    for partition, sm in dict(partition_to_sm).items():
        if type(partition) is not int or partition < 0:
            raise ValueError('Explicit partition required')
        whole = predictor.predict_prefill_ms(sm, new, cached,
                                            expected_identity=expected_identity)
        if whole is None and nearest_domain_fallback:
            whole = predictor.predict_prefill_nearest_ms(
                sm,new,cached,expected_identity=expected_identity)
        if whole is None:
            continue
        factors=((1.,1.) if not corrected else corrections.query('prefill',partition,
            overlap_class,prefill_features(new,cached),
            expected_identity=expected_correction_identity))
        if factors is None and corrected and nearest_domain_fallback:
            tail=[cell.guard_factor for cell in corrections.cells
                  if cell.stage=='prefill' and cell.partition==partition
                  and cell.overlap==overlap_class]
            factors=None if not tail else (max(tail),max(tail))
        if factors is not None:
            values.append((partition,whole*factors[0]/total_layers))
            guards.append((partition,whole*factors[1]/total_layers))
    solo_query = bind_layer_scaled_prefill_query(predictor, expected_identity=expected_identity,
        partition_to_sm=partition_to_sm, total_layers=total_layers,
        nearest_domain_fallback=nearest_domain_fallback)
    def corrected_query(partition,shape,guard):
        solo=solo_query(partition,shape)
        if solo is None or not corrected:
            return solo
        features=prefill_features(tuple(shape.extend_tokens),
            tuple(c-n for n,c in zip(shape.extend_tokens,shape.context_tokens)))
        factors=corrections.query('prefill',partition,overlap_class,features,
                                  expected_identity=expected_correction_identity)
        if factors is None and nearest_domain_fallback:
            tail=[cell.guard_factor for cell in corrections.cells
                  if cell.stage=='prefill' and cell.partition==partition
                  and cell.overlap==overlap_class]
            factors=None if not tail else (max(tail),max(tail))
        return None if factors is None else solo*factors[int(guard)]
    nominal=lambda partition,shape:corrected_query(partition,shape,False)
    guard=lambda partition,shape:corrected_query(partition,shape,True)
    return PhasePrefillComputeProfile(tuple(values),sum(new),nominal,True,
        tuple(guards),guard,overlap_class)


@dataclass(frozen=True)
class PhaseDecodeComputeProfile:
    batch_size: int
    values: tuple

    def query_ms(self, partition, overlap_class):
        return self.query_batch_ms(partition, overlap_class, self.batch_size)

    def query_batch_ms(self, partition, overlap_class, batch_size):
        if batch_size != self.batch_size:
            return None  # this profile is for the one observed cohort
        return dict(self.values).get((partition, overlap_class))


def bind_decode_compute_profile(predictor, *, expected_identity, partition_to_sm,
                                cached_tokens, interference_query,
                                nearest_domain_fallback=False):
    """Solo polynomial plus caller-frozen measured guard, not a guessed factor.

    Guard signature: (partition, overlap, cached_tokens, solo_ms) -> service_ms.
    It must include the observed ANN/prefill domain. Missing guard cells stay
    unsupported. Freeze values at binding so selection does not query live data.
    """
    cached = _tokens(cached_tokens, 'decode context tokens')
    if not callable(interference_query) or type(nearest_domain_fallback) is not bool:
        raise ValueError('Explicit frozen coexecution query required')
    values = []
    for partition, sm in dict(partition_to_sm).items():
        if type(partition) is not int or partition < 0:
            raise ValueError('Explicit partition required')
        solo = predictor.predict_decode_ms(sm, cached, expected_identity=expected_identity)
        if solo is None and nearest_domain_fallback:
            solo = predictor.predict_decode_nearest_ms(
                sm,cached,expected_identity=expected_identity)
        if solo is None:
            continue
        values.append(((partition, 'solo'), solo))
        for overlap in ('ann', 'prefill', 'ann_prefill'):
            guarded = interference_query(partition, overlap, cached, solo)
            if guarded is not None:
                if (type(guarded) not in (int,float) or not math.isfinite(guarded)
                        or guarded <= 0):
                    raise ValueError('Positive finite guarded service required')
                values.append(((partition, overlap), guarded))
    return PhaseDecodeComputeProfile(len(cached), tuple(values))


@dataclass(frozen=True)
class ANNPressureCell:
    partition: int
    overlap: str
    batch_range: tuple
    context_total_range: tuple
    context_max_range: tuple
    ann_gpu_ms_range: tuple
    service_ms: float


@dataclass(frozen=True)
class FrozenANNPressureTable:
    """Measured TOTAL decode service, not an additive solo correction.

    identity pins model/runtime/index/nprobe/topk/submission protocol externally.
    Half-open buckets avoid boundary ambiguity. No interpolation/extrapolation.
    Construction does not certify calibration acceptance or deployment readiness.
    """
    identity: tuple
    cells: tuple

    def __post_init__(self):
        if not self.identity or not isinstance(self.cells,tuple):
            raise ValueError('Pinned pressure identity and immutable cells required')
        for c in self.cells:
            if (type(c.partition) is not int or c.partition<0
                    or c.overlap not in ('ann','prefill','ann_prefill')
                    or type(c.service_ms) not in (int,float)
                    or not math.isfinite(c.service_ms) or c.service_ms<=0):
                raise ValueError('Invalid measured pressure cell')
            for bounds in (c.batch_range,c.context_total_range,c.context_max_range,c.ann_gpu_ms_range):
                if (not isinstance(bounds,tuple) or len(bounds)!=2 or
                    any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds)
                        or not 0<=bounds[0]<bounds[1]):
                    raise ValueError('Finite half-open pressure ranges required')

    def query_ms(self, partition, overlap, cached_tokens, ann_gpu_ms, *, expected_identity):
        if expected_identity!=self.identity:
            raise ValueError('Pressure profile workload identity mismatch')
        cached=_tokens(cached_tokens,'decode cached context')
        if not cached or type(ann_gpu_ms) not in (int,float) or not math.isfinite(ann_gpu_ms) or ann_gpu_ms<0:
            raise ValueError('Observed context and nonnegative ANN pressure required')
        features=(len(cached),sum(cached),max(cached),ann_gpu_ms)
        matches=[c for c in self.cells if c.partition==partition and c.overlap==overlap
            and all(lo<=v<hi for v,(lo,hi) in zip(features,
                (c.batch_range,c.context_total_range,c.context_max_range,c.ann_gpu_ms_range)))]
        if len(matches)>1:
            raise ValueError('Ambiguous pressure cells')
        return matches[0].service_ms if matches else None


def load_accepted_ann_pressure_table(path, *, expected_sha256,
                                     expected_plan_sha256,
                                     expected_identity):
    """Load held-out accepted TOTAL decode-P99 cells.

    The candidate builder is deliberately nondeployable.  Only the separate
    assessment schema is accepted here, and every installed cell must have
    prospective overlap evidence plus a passing held-out coverage check.
    """
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('ANN pressure assessment hash mismatch')
    data=json.loads(payload)
    assessment=data.get('assessment',{})
    identity=tuple(data.get('identity',()))
    if (data.get('schema')!='raglease.ann-pressure-assessment.v1'
            or data.get('plan_sha256')!=expected_plan_sha256
            or identity!=expected_identity
            or assessment.get('accepted') is not True):
        raise ValueError('ANN pressure assessment provenance mismatch')
    rows=assessment.get('cells')
    if (not rows or any(row.get('passed') is not True
                        or row.get('overlap_verified') is not True
                        or row.get('heldout_samples',0)<=0 for row in rows)):
        raise ValueError('Every ANN pressure cell must pass held-out overlap validation')
    cells=[]
    for row in rows:
        cells.append(ANNPressureCell(
            row['partition'],row['overlap'],tuple(row['batch_range']),
            tuple(row['context_total_range']),tuple(row['context_max_range']),
            tuple(row['ann_gpu_ms_range']),row['service_ms']))
    keys=[(c.partition,c.overlap,c.batch_range,c.context_total_range,
           c.context_max_range,c.ann_gpu_ms_range) for c in cells]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate accepted ANN pressure cell')
    return FrozenANNPressureTable(identity,tuple(cells))


@dataclass(frozen=True)
class PressureDecodeComputeProfile:
    """One captured cohort; proposals supply their OWN ANN service estimate."""
    solo: PhaseDecodeComputeProfile
    pressure: FrozenANNPressureTable
    expected_pressure_identity: tuple
    cached_tokens: tuple
    outstanding_by_partition: tuple
    causal_guard: object = None
    causal_history: tuple = ()
    expected_guard_identity: tuple = ()
    iteration_guard: object = None
    iteration_history: tuple = ()
    solo_envelope: object = None
    expected_solo_identity: tuple = ()
    partition_to_sm: tuple = ()
    fullcard_envelope: object = None
    expected_fullcard_identity: tuple = ()
    fullcard_partition_to_sm: tuple = ()
    fullcard_history: tuple = ()

    def __post_init__(self):
        configured=self.causal_guard is not None
        if configured != bool(self.expected_guard_identity):
            raise ValueError('Causal guard and identity must be configured together')
        if configured:
            self.causal_guard.predict_ms(max(1,len(self.cached_tokens)),'solo',
                self.causal_history,expected_identity=self.expected_guard_identity)
        if self.iteration_guard is not None:
            if not isinstance(self.iteration_guard,CausalIterationOverheadGuard):
                raise ValueError('Explicit causal iteration-overhead guard required')
            self.iteration_guard.predict_total_ms(1.,self.iteration_history)
        if self.solo_envelope is not None:
            if (getattr(self.solo_envelope,'identity',None)!=self.expected_solo_identity
                    or not callable(getattr(self.solo_envelope,'predict_ms',None))
                    or not self.partition_to_sm):
                raise ValueError('Matching partition-aware solo Decode envelope required')
            mapping=dict(self.partition_to_sm)
            if len(mapping)!=len(self.partition_to_sm) or any(
                    type(partition) is not int or partition<0 or
                    type(sm) is not int or sm<=0 for partition,sm in mapping.items()):
                raise ValueError('Unique partition-to-SM mapping required for solo envelope')
        elif self.expected_solo_identity or self.partition_to_sm:
            raise ValueError('Solo envelope metadata without accepted component')
        if self.fullcard_envelope is not None:
            if (getattr(self.fullcard_envelope,'identity',None)!=self.expected_fullcard_identity
                    or not callable(getattr(self.fullcard_envelope,'predict_ms',None))
                    or len(self.fullcard_partition_to_sm)!=1
                    or type(self.fullcard_partition_to_sm[0][0]) is not int
                    or self.fullcard_partition_to_sm[0][0]<0
                    or self.fullcard_partition_to_sm[0][1]!=132
                    or len(self.fullcard_history)>50):
                raise ValueError('Matching 132-SM Decode envelope and history required')
            self.fullcard_envelope.predict_ms(
                132,max(1,len(self.cached_tokens)),'solo',self.fullcard_history,
                expected_identity=self.expected_fullcard_identity)
        elif (self.expected_fullcard_identity or self.fullcard_partition_to_sm
              or self.fullcard_history):
            raise ValueError('Full-card envelope metadata without accepted component')

    @property
    def batch_size(self):
        return len(self.cached_tokens)

    def query_pressure_ms(self, partition, overlap, batch_size, proposed_ann_gpu_ms):
        if batch_size!=len(self.cached_tokens):
            return None
        outstanding=dict(self.outstanding_by_partition).get(partition)
        if outstanding is None or proposed_ann_gpu_ms is None:
            return None
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0
               for v in (outstanding,proposed_ann_gpu_ms)):
            raise ValueError('Frozen outstanding and proposed ANN service required')
        total=outstanding+proposed_ann_gpu_ms
        if overlap=='solo':
            base=self.solo.query_ms(partition,'solo') if total==0 else None
            if total==0 and self.solo_envelope is not None:
                sm=dict(self.partition_to_sm).get(partition)
                floor=(None if sm is None else self.solo_envelope.predict_ms(
                    sm,batch_size,self.causal_history,
                    expected_identity=self.expected_solo_identity))
                base=(None if floor is None else max(base or 0.,floor))
            if total==0 and self.fullcard_envelope is not None:
                sm=dict(self.fullcard_partition_to_sm).get(partition)
                if sm is not None:
                    fullcard=self.fullcard_envelope.predict_ms(
                        sm,batch_size,'solo',self.fullcard_history,
                        expected_identity=self.expected_fullcard_identity)
                    base=(None if fullcard is None else max(base or 0.,fullcard))
            return self._with_iteration(base)
        if overlap=='prefill':
            base=self.solo.query_ms(partition,'solo') if total==0 else None
            return self._guarded(base,batch_size,overlap,require_guard=True)
        if overlap not in ('ann','ann_prefill') or total==0:
            return None
        # The candidate-conditioned table isolates ANN pressure. Ambient
        # prefill contention is supplied by the separately validated causal
        # guard; do not require a combinatorial pressure table for both labels.
        base=self.pressure.query_ms(partition,'ann',self.cached_tokens,total,
            expected_identity=self.expected_pressure_identity)
        # Candidate-conditioned pressure remains the differentiator, while the
        # causal model supplies the same-event tail floor for ANN-only as well
        # as ANN+prefill.  The two predictions are maxima, never additive.
        return self._guarded(base,batch_size,overlap,
                             require_guard=(overlap=='ann_prefill'))

    def _guarded(self, base_ms, batch_size, overlap, require_guard=False):
        if base_ms is None:
            return base_ms
        if self.causal_guard is None:
            service=None if require_guard else base_ms
        else:
            floor=self.causal_guard.predict_ms(batch_size,overlap,self.causal_history,
                expected_identity=self.expected_guard_identity)
            service=None if floor is None else max(base_ms,floor)
        return self._with_iteration(service)

    def _with_iteration(self, service_ms):
        if service_ms is None or self.iteration_guard is None:
            return service_ms
        return self.iteration_guard.predict_total_ms(service_ms,self.iteration_history)

    def query_batch_ms(self, partition, overlap, batch_size):
        # Coexecution must never silently omit a proposal's ANN pressure.
        return self.query_pressure_ms(partition,overlap,batch_size,0.) if overlap=='solo' else None

    def query_ms(self, partition, overlap):
        return self.query_batch_ms(partition,overlap,len(self.cached_tokens))


@dataclass(frozen=True)
class StageFit:
    stage: str
    sm_count: int
    coefficients: tuple
    # Bounds on nonconstant features actually covered by calibration.
    feature_bounds: tuple

    def __post_init__(self):
        sizes = {'ann':(6,7), 'prefill': (4,), 'decode': (3,)}.get(self.stage)
        if sizes is None or type(self.sm_count) is not int or self.sm_count <= 0:
            raise ValueError('Known stage and actual positive SM count required')
        if (not isinstance(self.coefficients, tuple) or len(self.coefficients) not in sizes
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.coefficients)):
            raise ValueError('Finite calibrated stage coefficients required')
        if (not isinstance(self.feature_bounds, tuple)
                or len(self.feature_bounds) != len(self.coefficients)-1):
            raise ValueError('Explicit calibration coverage required')
        for bound in self.feature_bounds:
            if (not isinstance(bound, tuple) or len(bound) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bound)
                    or not 0 <= bound[0] <= bound[1]):
                raise ValueError('Invalid calibration feature interval')

    def covers(self, features):
        return all(lo <= value <= hi for value, (lo, hi) in zip(features[:-1], self.feature_bounds))


@dataclass(frozen=True)
class StagePredictor:
    # Explicit hardware/model/execution identity, supplied by calibrated artifact.
    identity: tuple
    fits: tuple[StageFit, ...]

    def __post_init__(self):
        if (not isinstance(self.identity, tuple) or not self.identity
                or any(not isinstance(v, str) or not v for v in self.identity)
                or not isinstance(self.fits, tuple) or any(not isinstance(f, StageFit) for f in self.fits)):
            raise ValueError('Frozen deployment identity and fitted models required')

    def _query(self, stage, sm_count, features, expected_identity):
        if expected_identity != self.identity:
            raise ValueError('Stage predictor calibration belongs to another deployment')
        if type(sm_count) is not int or sm_count <= 0:
            raise ValueError('Actual SM count required; no implicit rounding')
        matching = [fit for fit in self.fits if fit.stage == stage and fit.sm_count == sm_count
                    and fit.covers(features)]
        if not matching:
            return None
        if len(matching) != 1:
            raise ValueError('Ambiguous overlapping stage calibration fits')
        value = sum(a*b for a, b in zip(matching[0].coefficients, features))
        return value if math.isfinite(value) and value > 0 else None

    def predict_prefill_ms(self, sm_count, new_tokens, cached_tokens, *, expected_identity):
        """Full prefill phase for the supplied batch, not a per-layer duration."""
        return self._query('prefill', sm_count, prefill_features(new_tokens, cached_tokens), expected_identity)

    def predict_prefill_nearest_ms(self, sm_count, new_tokens, cached_tokens, *, expected_identity):
        """Explicit bounded fallback at the nearest assessed prefill boundary."""
        if expected_identity!=self.identity:
            raise ValueError('Stage predictor calibration belongs to another deployment')
        features=prefill_features(new_tokens,cached_tokens)
        matches=[fit for fit in self.fits if fit.stage=='prefill' and fit.sm_count==sm_count]
        if len(matches)!=1:return None
        fit=matches[0]
        clipped=tuple(min(max(value,lo),hi) for value,(lo,hi) in
                      zip(features[:-1],fit.feature_bounds))+(features[-1],)
        value=sum(a*b for a,b in zip(fit.coefficients,clipped))
        return value if math.isfinite(value) and value>0 else None

    def _ann_features(self, sm_count, batch_size, list_visits, vectors_scanned,
                      max_query_vectors, max_list_vectors, query_vector_sq_sum):
        widths={len(fit.coefficients) for fit in self.fits
                if fit.stage=='ann' and fit.sm_count==sm_count}
        if not widths:
            return None
        if len(widths)!=1:
            raise ValueError('Ambiguous ANN feature schemas for one SM allocation')
        width=next(iter(widths))
        if width==7 and query_vector_sq_sum is None:
            return None
        return ann_features(batch_size,list_visits,vectors_scanned,
                            max_query_vectors,max_list_vectors,
                            query_vector_sq_sum if width==7 else None)

    def predict_ann_ms(self, sm_count, batch_size, list_visits, vectors_scanned,
                       max_query_vectors, max_list_vectors,
                       query_vector_sq_sum=None, *, expected_identity):
        """One admitted ANN GPU microbatch, excluding CPU search and merge."""
        features=self._ann_features(sm_count,batch_size,list_visits,vectors_scanned,
            max_query_vectors,max_list_vectors,query_vector_sq_sum)
        return None if features is None else self._query(
            'ann',sm_count,features,expected_identity)

    def predict_ann_nearest_ms(self, sm_count, batch_size, list_visits, vectors_scanned,
                               max_query_vectors, max_list_vectors,
                               query_vector_sq_sum=None, *, expected_identity):
        """Explicit bounded fallback at the nearest assessed feature boundary."""
        if expected_identity!=self.identity:
            raise ValueError('Stage predictor calibration belongs to another deployment')
        features=self._ann_features(sm_count,batch_size,list_visits,vectors_scanned,
            max_query_vectors,max_list_vectors,query_vector_sq_sum)
        if features is None:return None
        matches=[fit for fit in self.fits if fit.stage=='ann' and fit.sm_count==sm_count]
        if len(matches)!=1:return None
        fit=matches[0]
        clipped=tuple(min(max(value,lo),hi) for value,(lo,hi) in
                      zip(features[:-1],fit.feature_bounds))+(features[-1],)
        value=sum(a*b for a,b in zip(fit.coefficients,clipped))
        return value if math.isfinite(value) and value>0 else None

    def predict_decode_ms(self, sm_count, cached_tokens, *, expected_identity):
        """One complete decode iteration, not total remaining generation."""
        return self._query('decode', sm_count, decode_features(cached_tokens), expected_identity)

    def predict_decode_nearest_ms(self, sm_count, cached_tokens, *, expected_identity):
        """Explicit bounded fallback at the nearest assessed decode boundary."""
        if expected_identity!=self.identity:
            raise ValueError('Stage predictor calibration belongs to another deployment')
        features=decode_features(cached_tokens)
        matches=[fit for fit in self.fits if fit.stage=='decode' and fit.sm_count==sm_count]
        if len(matches)!=1:return None
        fit=matches[0]
        clipped=tuple(min(max(value,lo),hi) for value,(lo,hi) in
                      zip(features[:-1],fit.feature_bounds))+(features[-1],)
        value=sum(a*b for a,b in zip(fit.coefficients,clipped))
        return value if math.isfinite(value) and value>0 else None


def load_accepted_predictor(path, *, expected_sha256, expected_plan_sha256, expected_identity,
                            expected_supplement_plan_sha256=None):
    """Load a pinned, assessed phase model, never an unreviewed fitter output.

    Deployment must explicitly pin the result hash and identity. A successful
    solo assessment does not provide a contention guard or joint scheduler gate.
    """
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Stage model artifact hash mismatch')
    artifact=json.loads(payload)
    assessment=artifact['assessment']
    if expected_supplement_plan_sha256 is None:
        if (artifact.get('schema')!='raglease.stage-calibration-assessment.v1' or
                artifact.get('plan_sha256')!=expected_plan_sha256):
            raise ValueError('Unexpected stage calibration provenance')
        cells=assessment.get('cells',[])
    else:
        if (artifact.get('schema')!='raglease.cached-stage-assessment.v1' or
                artifact.get('base_plan_sha256')!=expected_plan_sha256 or
                artifact.get('supplement_plan_sha256')!=expected_supplement_plan_sha256):
            raise ValueError('Unexpected combined stage calibration provenance')
        groups=[assessment.get(name,{}) for name in ('zero_prefix_and_decode','cached_prefill')]
        if any(g.get('phase_model_accepted') is not True or not g.get('cells') for g in groups):
            raise ValueError('Both old and cached held-out shapes must pass')
        cells=[cell for group in groups for cell in group['cells']]
    if (assessment.get('phase_model_accepted') is not True or not cells or
            any(c.get('passed') is not True for c in cells)):
        raise ValueError('Stage calibration did not pass held-out assessment')
    data=artifact['predictor']
    identity=tuple(data['identity'])
    if identity!=expected_identity:
        raise ValueError('Stage calibration belongs to another deployment')
    fits=tuple(StageFit(f['stage'],f['sm_count'],tuple(f['coefficients']),
        tuple(tuple(b) for b in f['feature_bounds'])) for f in data['fits'])
    keys=[(f.stage,f.sm_count) for f in fits]
    if not fits or len(set(keys))!=len(keys):
        raise ValueError('Expected one calibrated fit per stage and SM count')
    assessed={(c['stage'],c['sm']) for c in cells}
    if set(keys)!=assessed:
        raise ValueError('Stage fits and assessed partitions disagree')
    return StagePredictor(identity,fits)


def load_accepted_bundle_model(path, *, expected_sha256, expected_plan_sha256,
                               expected_identity, partition_to_sm, bundle_layers):
    """Bind held-out-accepted interval models to frozen prefill timing.

    Only zero-prefix feature coverage is supported. The selected traversal
    grain stays fixed for this query; no inner scheduling or alternate search.
    """
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Bundle model artifact hash mismatch')
    data=json.loads(payload)
    if (data.get('schema')!='raglease.native-bundle-assessment.v1'
            or data.get('plan_sha256')!=expected_plan_sha256
            or tuple(data.get('identity',()))!=expected_identity
            or data.get('accepted') is not True or not data.get('fits')):
        raise ValueError('Bundle model provenance or assessment mismatch')
    fits={}
    for fit in data['fits']:
        if (fit.get('accepted') is not True or not fit.get('cells')
                or any(c.get('passed') is not True for c in fit['cells'])):
            raise ValueError('Bundle model has failed held-out cells')
        key=(fit['sm'],fit['bundle_layers'],fit['start_layer'],fit['end_layer'])
        if key in fits:raise ValueError('Duplicate interval model')
        coefficients=tuple(fit['coefficients']);bounds=tuple(tuple(b) for b in fit['feature_bounds'])
        if (len(coefficients)!=3 or any(type(c) not in (int,float) or not math.isfinite(c) for c in coefficients)
                or len(bounds)!=2 or any(len(b)!=2 or not all(math.isfinite(v) for v in b)
                                              or not 0<=b[0]<=b[1] for b in bounds)):
            raise ValueError('Invalid interval model coefficients or coverage')
        fits[key]=(coefficients,bounds)
    mapping=dict(partition_to_sm)
    if type(bundle_layers) is not int or bundle_layers<=0 or any(
            type(p) is not int or p<0 or type(sm) is not int or sm<=0 for p,sm in mapping.items()):
        raise ValueError('Explicit grain and actual partition mapping required')
    def query(partition,shape):
        new=_tokens(shape.extend_tokens,'new prefill tokens',positive=True)
        context=_tokens(shape.context_tokens,'prefill context tokens',positive=True)
        if (context!=new or len(shape.request_ids)!=len(new)
                or len(set(shape.request_ids))!=len(new)):
            return None
        fit=fits.get((mapping.get(partition),bundle_layers,shape.start_layer,shape.end_layer))
        if fit is None:return None
        coefficients,bounds=fit
        features=(sum((n/1024.)**2 for n in new),sum(new)/1024.,1.)
        if not all(lo<=v<=hi for v,(lo,hi) in zip(features,bounds)):return None
        result=sum(c*f for c,f in zip(coefficients,features))
        return result if math.isfinite(result) and result>0 else None
    return query
