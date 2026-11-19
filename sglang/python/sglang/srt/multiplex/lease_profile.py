"""Cold-start priors produced by the offline RAGLease profiler."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from bisect import bisect_right
from types import MappingProxyType
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenMemoryTransitionProfile:
    percentile: float
    min_samples: int
    samples: tuple

    def query_remaining_ms(self, key, *, elapsed_ms=0.0):
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError('Nonnegative finite observed transition age required')
        if key is None:
            return None
        values = next((v for k,v in self.samples if k==key),())
        remaining = tuple(v-elapsed_ms for v in values if elapsed_ms==0 or v>elapsed_ms)
        if len(remaining)<self.min_samples:
            return None
        pos=(len(remaining)-1)*self.percentile/100
        lo=int(pos);hi=min(lo+1,len(remaining)-1)
        return remaining[lo]+(remaining[hi]-remaining[lo])*(pos-lo)


@dataclass(frozen=True)
class JointTransitionCompletion:
    kv_ms: float | None
    compute_ms: float | None
    workspace_ms: float | None
    activation_ms: float | None


def capture_input_preparation_profile(scheduler, *, percentile=95., min_samples=4):
    """Current-run external turnaround prior, not exact CPU service or an SLO.

    No reuse across datasets/templates/client concurrency without validation.
    Completed observations are a heuristic prior, not a censored-data model;
    conditional survivor support may be insufficient, in which case query
    returns None. Freeze once for the common target/keep decision window.
    """
    if (type(min_samples) is not int or min_samples<=0
            or not math.isfinite(percentile) or not 0<=percentile<=100):
        raise ValueError('Invalid preparation profile selection')
    samples=[]
    for rows,history in getattr(scheduler,'rag_input_preparation_turnaround_ms',{}).items():
        values=tuple(history)
        if (type(rows) is not int or rows<=0 or any(type(v) not in (int,float)
                or not math.isfinite(v) or v<=0 for v in values)):
            raise ValueError('Invalid observed preparation turnaround')
        samples.append((rows,tuple(sorted(values))))
    return FrozenMemoryTransitionProfile(percentile,min_samples,tuple(samples))


@dataclass(frozen=True)
class PrefillReleaseObservation:
    partition: int
    tokens: int
    start_layer: int
    end_layer: int
    request_ids: tuple[str, ...]
    elapsed_ms: float
    observation_offset_ms: float


@dataclass(frozen=True)
class PrefillHandoffSample:
    result_sequence: int
    merge_sequence: int
    # Both endpoints retain their own partition, prompts and running count.
    shape: tuple
    result_host_ms: float
    inter_stage_ms: float
    merge_host_ms: float
    observed_tail_ms: float

    @property
    def total_ms(self):
        return self.result_host_ms + self.inter_stage_ms + self.merge_host_ms + self.observed_tail_ms


def capture_prefill_handoff_samples(scheduler):
    """Pair only explicitly linked completed observations, never nearby rows.

    The four components partition one wall-clock envelope. Inter-stage time
    includes scheduler/profiler overhead and possibly deferred decode waiting;
    it is not a pure decode measurement. The tail includes observation delay,
    not just GPU work. Do not carry this historical wait into a new dependency
    graph which already accounts for the old decode's completion.
    """
    rows = tuple(dict(row) for row in getattr(scheduler, 'rag_prefill_stage_observation_ledger', ()))
    by_sequence = {}
    for row in rows:
        seq = row.get('sequence')
        if type(seq) is not int or seq <= 0 or seq in by_sequence:
            raise ValueError('Unique positive handoff observation sequence required')
        by_sequence[seq] = row
    pairs = []; consumed = set()
    def endpoint(row):
        partition = row['partition']
        if type(partition) is not int or partition < 0:
            raise ValueError('Invalid handoff observation partition')
        shape = (partition, *_prefill_handoff_shape(row['stage'], row['prompt_tokens'], row['running_batch_size']))
        times = tuple(row[k] for k in ('started_at_s', 'host_finished_at_s', 'observed_complete_at_s'))
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
                or not times[0] <= times[1] <= times[2]):
            raise ValueError('Invalid handoff observation clock order')
        return shape, times
    for merge in rows:
        ref = merge.get('result_sequence')
        if ref is None:
            continue  # Old ledgers, dropped results and unpaired results stay unpaired.
        if (type(ref) is not int or ref <= 0 or ref >= merge['sequence']
                or merge['stage'] not in ('merge', 'deferred_merge') or ref in consumed):
            raise ValueError('Invalid or reused result-to-merge association')
        consumed.add(ref)
        result = by_sequence.get(ref)
        if result is None:
            continue  # Bounded ledger eviction or completion not observed yet.
        if result['stage'] != 'result' or result.get('result_sequence') is not None:
            raise ValueError('Merge must reference an actual result observation')
        result_shape, (rs, rh, rc) = endpoint(result)
        merge_shape, (ms, mh, mc) = endpoint(merge)
        if rh > ms:
            raise ValueError('Native host result must finish before merge starts')
        pairs.append(PrefillHandoffSample(ref, merge['sequence'], (result_shape, merge_shape),
            (rh-rs)*1000., (ms-rh)*1000., (mh-ms)*1000., (max(rc, mc)-mh)*1000.))
    return tuple(sorted(pairs, key=lambda p: p.merge_sequence))


@dataclass(frozen=True)
class FrozenPrefillHandoffProfile:
    """Separate host API and submit-to-observed-release priors.

    Host histories are not partition-conditioned in the native recorder.
    Release histories are. Neither is a joint result+merge distribution;
    overlapping stage quantiles must not be summed into a handoff quantile.
    """
    host: FrozenMemoryTransitionProfile
    release: FrozenMemoryTransitionProfile
    paired_samples: tuple[PrefillHandoffSample, ...] = ()

    def query_paired_sample(self, result_shape, merge_shape):
        """Nearest-rank total-envelope quantile, retaining one coherent row.

        Components are NOT their individual quantiles. Exact endpoint shapes
        only; this historical envelope is not an automatically usable service
        time for a different concurrency/dependency situation.
        """
        shape = (result_shape, merge_shape)
        rows = sorted((s for s in self.paired_samples if s.shape == shape),
                      key=lambda s: (s.total_ms, s.merge_sequence))
        if len(rows) < self.release.min_samples:
            return None
        rank = max(0, math.ceil(len(rows)*self.release.percentile/100.)-1)
        return rows[rank]

    def query_host_ms(self, stage, prompt_tokens, running_batch_size):
        shape = _prefill_handoff_shape(stage, prompt_tokens, running_batch_size)
        return self.host.query_remaining_ms(shape)

    def query_release_ms(self, partition, stage, prompt_tokens,
                         running_batch_size, *, elapsed_ms=0.):
        if type(partition) is not int or partition < 0:
            raise ValueError('Actual prefill handoff partition required')
        shape = _prefill_handoff_shape(stage, prompt_tokens, running_batch_size)
        return self.release.query_remaining_ms((partition, *shape), elapsed_ms=elapsed_ms)


def _prefill_handoff_shape(stage, prompt_tokens, running_batch_size):
    prompts = tuple(prompt_tokens)
    if (stage not in ('result', 'merge', 'deferred_merge') or not prompts
            or any(type(n) is not int or n <= 0 for n in prompts)
            or type(running_batch_size) is not int or running_batch_size < 0):
        raise ValueError('Exact observed prefill host stage shape required')
    return stage, prompts, running_batch_size


def capture_prefill_handoff_profile(scheduler, *, percentile=95., min_samples=4):
    """Freeze existing successful observations without polling CUDA events.

    Missing release observations (including disabled instrumentation) remain
    unknown. Do not substitute the host API duration for device completion.
    Profiles apply only to the current run's workload/hardware envelope.
    """
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid handoff profile selection')
    def freeze(name, partitioned):
        samples = []
        for key, history in getattr(scheduler, name, {}).items():
            if not isinstance(key, tuple) or len(key) != (4 if partitioned else 3):
                raise ValueError('Invalid observed handoff key')
            if partitioned:
                if type(key[0]) is not int or key[0] < 0:
                    raise ValueError('Invalid observed handoff partition')
                shape = (key[0], *_prefill_handoff_shape(*key[1:]))
            else:
                shape = _prefill_handoff_shape(*key)
            values = tuple(history)
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in values):
                raise ValueError('Invalid observed handoff duration')
            samples.append((shape, tuple(sorted(values))))
        return FrozenMemoryTransitionProfile(percentile, min_samples, tuple(samples))
    return FrozenPrefillHandoffProfile(
        freeze('rag_prefill_host_stage_ms_by_shape', False),
        freeze('rag_prefill_stage_release_ms_by_shape', True),
        capture_prefill_handoff_samples(scheduler))


def capture_prefill_release_profile(scheduler, *, percentile=95., min_samples=4):
    """Freeze host release observations, never CUDA per-layer service samples."""
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid prefill release profile selection')
    samples = []
    for key, history in getattr(scheduler,'rag_prefill_host_release_ms_by_shape',{}).items():
        if (not isinstance(key,tuple) or len(key)!=3 or type(key[0]) is not int or key[0]<0
                or any(type(n) is not int or n<=0 for n in key[1:])):
            continue
        values=tuple(history)
        if values and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values):
            samples.append((key,tuple(sorted(values))))
    return FrozenMemoryTransitionProfile(percentile,min_samples,tuple(samples))


@dataclass(frozen=True)
class FrozenJointTransitionProfiles:
    compute: Any
    kv: Any
    workspace: Any

    def query(self, *, source_partition, target_partition, observation_offset_ms,
              kv_transition_required, kv_key, workspace_mapping_required,
              workspace_key, ann_quiescent_ms):
        """Query one not-yet-started proposal using common frozen measurements.

        Offset is the real observation time relative to the common origin.
        Ongoing transitions must use their observed age/actual dependency state
        separately; this method must not restart their clocks at zero.
        """
        if (not math.isfinite(observation_offset_ms) or observation_offset_ms<0
                or type(kv_transition_required) is not bool
                or type(workspace_mapping_required) is not bool):
            raise ValueError('Valid common offset and explicit transition requirements needed')
        compute = self.compute.query_remaining_ms(source_partition,target_partition)
        kv = self.kv.query_remaining_ms(kv_key) if kv_transition_required else 0.
        workspace = self.workspace.query_remaining_ms(workspace_key) if workspace_mapping_required else 0.
        return compose_joint_transition_completion(
            compute_ready_ms=None if compute is None else observation_offset_ms+compute,
            kv_ready_ms=None if kv is None else observation_offset_ms+kv,
            ann_quiescent_ms=ann_quiescent_ms,workspace_bind_ms=workspace,
            workspace_mapping_required=workspace_mapping_required)


def capture_joint_transition_profiles(scheduler, *, percentile=95., min_samples=4):
    """Capture once per serialized decision, reuse for target AND continuation."""
    controller = getattr(scheduler,'shared_hbm_controller',None)
    if controller is None:
        return None
    return FrozenJointTransitionProfiles(
        capture_compute_transition_profile(scheduler,percentile=percentile,min_samples=min_samples),
        capture_kv_transition_profile(controller,percentile=percentile,min_samples=min_samples),
        capture_workspace_transition_profile(controller.pool,percentile=percentile,min_samples=min_samples))


def compose_joint_transition_completion(*, kv_ready_ms, compute_ready_ms,
                                        ann_quiescent_ms, workspace_bind_ms,
                                        workspace_mapping_required):
    """Compose the selected preparation DAG, not independent latency sums.

    All *_ready/quiescent values are absolute offsets from the shared window
    origin. workspace_bind_ms alone is a duration, beginning after the KV,
    compute and ANN-reader gates. None is unknown, not a zero-cost transition.
    Existing compatible decode is NOT gated by the returned activation time.
    Diagnostic no-transition views must compose their own gates, not reuse G2.
    """
    if type(workspace_mapping_required) is not bool:
        raise ValueError('Explicit workspace transition requirement needed')
    for value in (kv_ready_ms,compute_ready_ms,ann_quiescent_ms,workspace_bind_ms):
        if value is not None and (not math.isfinite(value) or value<0):
            raise ValueError('Finite nonnegative observed/predicted time required')
    if workspace_mapping_required:
        gates=(kv_ready_ms,compute_ready_ms,ann_quiescent_ms,workspace_bind_ms)
        workspace=(max(gates[:3])+workspace_bind_ms if all(v is not None for v in gates) else None)
    else:
        workspace=0.0  # Existing verified binding; readers need not be drained.
    gates=(kv_ready_ms,compute_ready_ms,workspace)
    activation=max(gates) if all(v is not None for v in gates) else None
    return JointTransitionCompletion(kv_ready_ms,compute_ready_ms,workspace,activation)


def capture_ann_release_profile(scheduler, *, percentile=95., min_samples=4):
    """Host-clock submission-to-observed-GPU-release, not CUDA service.

    Exact partition/batch keys; contains polling reaction and is not explicitly
    conditioned on concurrent workload. Use host age with this profile only.
    """
    if (not math.isfinite(percentile) or not 0<=percentile<=100
            or type(min_samples) is not int or min_samples<=0):
        raise ValueError('Invalid ANN release profile selection')
    grouped=[]
    for key,history in getattr(scheduler,'rag_ann_gpu_release_ms_by_partition_batch_size',{}).items():
        if (not isinstance(key,tuple) or len(key)!=2 or type(key[0]) is not int or key[0]<0
                or type(key[1]) is not int or key[1]<=0):
            raise ValueError('Invalid ANN release shape')
        values=tuple(sorted(float(v) for v in history))
        if any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('Invalid observed ANN release time')
        grouped.append((key,values))
    return FrozenMemoryTransitionProfile(percentile,min_samples,tuple(sorted(grouped)))


def capture_workspace_transition_profile(pool, *, percentile=95., min_samples=4):
    """Successful bind wall time under exact mode/old binding/requested bytes.

    Includes unbind when replacing a smaller stack, not external ANN drain or
    preceding KV migration. Fragmentation/other pool ownership is not bucketed.
    """
    if (not math.isfinite(percentile) or not 0<=percentile<=100
            or type(min_samples) is not int or min_samples<=0):
        raise ValueError('Invalid workspace transition profile selection')
    grouped={}
    for key,ms in tuple(getattr(pool,'faiss_bind_profile_samples',())):
        if (not isinstance(key,tuple) or len(key)!=4 or key[0] not in ('scratch','ivf')
                or type(key[1]) is not bool or any(type(v) is not int or v<0 for v in key[2:])
                or key[3]<=16 or not math.isfinite(ms) or ms<0):
            raise ValueError('Malformed successful workspace binding observation')
        grouped.setdefault(key,[]).append(float(ms))
    return FrozenMemoryTransitionProfile(percentile,min_samples,
        tuple(sorted((key,tuple(sorted(values))) for key,values in grouped.items())))


def capture_kv_transition_profile(controller, *, percentile=95., min_samples=4):
    """Freeze geometry-conditioned COMPLETE KV transitions only.

    Missing exact shapes never borrow a smaller growth or unrelated reclaim.
    Query returns total/residual duration; caller converts to a common-window
    completion offset and does not add constituent fence/map costs again.
    """
    if (not math.isfinite(percentile) or not 0<=percentile<=100
            or type(min_samples) is not int or min_samples<=0):
        raise ValueError('Invalid memory transition profile selection')
    grouped={}
    for key,ms in tuple(getattr(controller,'kv_growth_profile_samples',())):
        if (not isinstance(key,tuple) or len(key)!=11 or key[0] not in ('ivf','scratch')
                or any(type(v) is not int or v<0 for v in key[1:5])
                or type(key[5]) is not bool or type(key[6]) is not int or key[6]<0
                or any(type(v) is not bool for v in key[7:])
                or not math.isfinite(ms) or ms<0):
            raise ValueError('Malformed complete KV transition observation')
        grouped.setdefault(key,[]).append(float(ms))
    return FrozenMemoryTransitionProfile(percentile,min_samples,
        tuple(sorted((key,tuple(sorted(values))) for key,values in grouped.items())))


@dataclass(frozen=True)
class FrozenComputeTransitionProfile:
    """Exact directional whole-transition observations, including drain wait.

    No unrelated-pair/global switch fallback or unobserved round trip. For an
    ongoing transition, condition on durations exceeding its observed age.
    These samples are not conditioned on the active workload; callers must
    retain that limitation and must not add another drain cost to this value.
    """
    percentile: float
    min_samples: int
    samples: tuple

    def query_remaining_ms(self, source: int, target: int, *, elapsed_ms=0.0):
        if (any(type(v) is not int or v < 0 for v in (source,target))
                or not math.isfinite(elapsed_ms) or elapsed_ms < 0):
            raise ValueError('Valid partition pair and observed transition age required')
        if source == target:
            return 0.0
        values = next((ms for pair,ms in self.samples if pair==(source,target)),())
        eligible = tuple(v-elapsed_ms for v in values if elapsed_ms==0 or v>elapsed_ms)
        if len(eligible) < self.min_samples:
            return None
        pos = (len(eligible)-1)*self.percentile/100
        lo = int(pos); hi = min(lo+1,len(eligible)-1)
        return eligible[lo]+(eligible[hi]-eligible[lo])*(pos-lo)


def capture_compute_transition_profile(scheduler, *, percentile=95., min_samples=4):
    if (not math.isfinite(percentile) or not 0 <= percentile <= 100
            or type(min_samples) is not int or min_samples <= 0):
        raise ValueError('Valid transition profile selection required')
    samples = []
    for pair,history in getattr(scheduler,'rag_partition_transition_pair_latency_ms',{}).items():
        if len(pair)!=2 or any(type(v) is not int or v<0 for v in pair):
            raise ValueError('Malformed observed partition pair')
        values = tuple(sorted(float(ms) for ms in history))
        if any(not math.isfinite(ms) or ms<0 for ms in values):
            raise ValueError('Invalid completed transition observation')
        samples.append((tuple(pair),values))
    return FrozenComputeTransitionProfile(percentile,min_samples,tuple(sorted(samples)))


@dataclass(frozen=True)
class FrozenColdWorkerProfile:
    percentile: float
    min_samples: int
    samples: tuple
    _by_shape: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (type(self.min_samples) is not int or self.min_samples <= 0
                or not math.isfinite(self.percentile) or not 0 <= self.percentile <= 100):
            raise ValueError('Invalid CPU profile selection')
        grouped = {}
        frozen = []
        for raw_key, ms in self.samples:
            key = tuple(tuple(pair) for pair in raw_key)
            if (not key or any(len(pair) != 2 or any(type(x) is not int or x <= 0
                                                    for x in pair) for pair in key)
                    or type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0):
                raise ValueError('Invalid CPU service observation')
            frozen.append((key, float(ms)))
            grouped.setdefault(key, []).append(float(ms))
        object.__setattr__(self, 'samples', tuple(frozen))
        object.__setattr__(self, '_by_shape', MappingProxyType(
            {key: tuple(sorted(values)) for key, values in grouped.items()}))

    def query_remaining_ms(self, shape) -> float | None:
        """Exact observed rows/list-width shape; conditional active residual.

        Active samples must have survived the observed elapsed wall time. This
        is an empirical residual estimate, not max(0, P95(total)-elapsed).
        Sparse surviving history is unsupported. CPU thread count, index and
        concurrent workload must be held within the caller's calibrated scope.
        """
        key = tuple((e[1], e[2]) for e in shape.entries)
        elapsed = shape.elapsed_ms if shape.active else 0.0
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Nonnegative finite CPU elapsed time required')
        values = self._by_shape.get(key, ())
        first = bisect_right(values, elapsed)
        count = len(values) - first
        if count < self.min_samples:
            return None
        position = (count - 1) * self.percentile / 100
        low = int(position); high = min(low + 1, count - 1)
        # Subtract before interpolation, preserving the former arithmetic order.
        lower, upper = values[first+low]-elapsed, values[first+high]-elapsed
        return lower + (upper-lower) * (position-low)


@dataclass(frozen=True)
class FrozenColdWorkerLinearProfile:
    """Deployment-frozen CPU ANN service model with survivor residuals.

    The native worker executes a concatenated batch at the largest cold width,
    so the model features are total rows, maximum cold width, their product,
    and number of queued request fragments. Tail ratios are measured
    total-service/base-prediction ratios from the training split. For active
    work the query conditions that empirical distribution on samples which
    survived the observed age before taking the frozen percentile.
    """
    identity: tuple
    coefficients: tuple
    feature_bounds: tuple
    tail_ratios: tuple
    percentile: float = 95.0
    min_survivors: int = 16
    sparse_survivor_guard_fallback: bool = False

    def __post_init__(self):
        if (not isinstance(self.identity, tuple) or not self.identity
                or any(not isinstance(v, str) or not v for v in self.identity)
                or not isinstance(self.coefficients, tuple)
                or len(self.coefficients) != 5
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in self.coefficients)
                or not isinstance(self.feature_bounds, tuple)
                or len(self.feature_bounds) != 4
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       or any(type(v) not in (int, float) or not math.isfinite(v)
                              for v in pair)
                       or not 0 <= pair[0] < pair[1]
                       for pair in self.feature_bounds)
                or not isinstance(self.tail_ratios, tuple)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in self.tail_ratios)
                or type(self.min_survivors) is not int or self.min_survivors <= 0
                or type(self.sparse_survivor_guard_fallback) is not bool
                or len(self.tail_ratios) < self.min_survivors
                or type(self.percentile) not in (int, float)
                or not math.isfinite(self.percentile)
                or not 0 <= self.percentile <= 100):
            raise ValueError('Invalid calibrated CPU ANN model')
        object.__setattr__(self, 'tail_ratios', tuple(sorted(float(v)
                                                            for v in self.tail_ratios)))

    @staticmethod
    def features(shape):
        entries = tuple(shape.entries)
        if (not entries or any(len(entry) < 3 or type(entry[1]) is not int
                or entry[1] <= 0 or type(entry[2]) is not int or entry[2] <= 0
                for entry in entries)):
            raise ValueError('Positive CPU rows and cold widths required')
        rows = sum(entry[1] for entry in entries)
        width = max(entry[2] for entry in entries)
        return float(rows), float(width), float(rows * width), float(len(entries))

    def query_remaining_ms(self, shape, *, expected_identity=None):
        if expected_identity is not None and expected_identity != self.identity:
            raise ValueError('CPU ANN model belongs to another deployment')
        features = self.features(shape)
        if not all(lo <= value < hi for value, (lo, hi) in
                   zip(features, self.feature_bounds)):
            return None
        base = sum(coef * value for coef, value in
                   zip(self.coefficients, (*features, 1.0)))
        if not math.isfinite(base) or base <= 0:
            return None
        elapsed = shape.elapsed_ms if shape.active else 0.0
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Nonnegative finite CPU elapsed time required')
        first = bisect_right(self.tail_ratios, elapsed / base)
        survivors = self.tail_ratios[first:]
        if len(survivors) < self.min_survivors:
            if not self.sparse_survivor_guard_fallback:
                return None
            # The accepted conditional estimator remains unchanged.  A
            # preliminary deployment may explicitly guard its sparse far
            # tail with the largest observed total-service multiplier.  Use
            # that whole duration as ADDITIONAL residual rather than
            # subtracting age: this is deliberately conservative and avoids
            # treating an empirically old but still active task as complete.
            guarded = base * self.tail_ratios[-1]
            return guarded if math.isfinite(guarded) and guarded > 0 else None
        position = (len(survivors) - 1) * self.percentile / 100.0
        low = int(position); high = min(low + 1, len(survivors) - 1)
        ratio = survivors[low] + (survivors[high] - survivors[low]) * (position - low)
        remaining = base * ratio - elapsed
        return remaining if math.isfinite(remaining) and remaining > 0 else None


def load_accepted_cpu_ann_profile(path, *, expected_sha256, expected_identity):
    """Load a held-out-assessed CPU ANN profile with pinned deployment identity."""
    import hashlib
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('CPU ANN profile artifact hash mismatch')
    artifact = json.loads(payload)
    assessment = artifact.get('assessment', {})
    model = artifact.get('model', {})
    identity = tuple(model.get('identity', ()))
    if (artifact.get('schema') != 'raglease.cpu-ann-linear-assessment.v1'
            or artifact.get('formal_ready') is not True
            or assessment.get('accepted') is not True
            or assessment.get('heldout_rows', 0) <= 0
            or assessment.get('coverage') is None
            or identity != expected_identity):
        raise ValueError('CPU ANN profile did not pass held-out assessment')
    return FrozenColdWorkerLinearProfile(
        identity=identity,
        coefficients=tuple(model.get('coefficients', ())),
        feature_bounds=tuple(tuple(pair) for pair in model.get('feature_bounds', ())),
        tail_ratios=tuple(model.get('tail_ratios', ())),
        percentile=model.get('percentile'),
        min_survivors=model.get('min_survivors'),
    )


def capture_cold_worker_profile(worker_snapshot, *, percentile=95, min_samples=4):
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid CPU profile selection')
    samples = []
    for key, duration in worker_snapshot.get('service_samples', ()):
        key = tuple(tuple(pair) for pair in key)
        if (not key or any(len(pair) != 2 or any(type(x) is not int or x <= 0
                                                for x in pair) for pair in key)
                or type(duration) not in (int, float)
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError('Invalid CPU service observation')
        samples.append((key, float(duration)))
    return FrozenColdWorkerProfile(percentile, min_samples, tuple(samples))


COLD_IDENTITY_FIELDS = frozenset({
    'hardware', 'index_sha256', 'nprobe', 'topk', 'cpu_threads',
    'max_batch_queries', 'workload_envelope_sha256',
})


def restore_cold_worker_samples(artifact, *, expected_identity, percentile=95,
                                min_samples=4):
    """Load deduplicated offline samples into the same exact-shape query API.

    The workload envelope identifies the measured placement/coexecution range,
    not a falsely constant residency. This is an empirical unconditioned prior,
    not a certified bound. The caller must validate the current deployment
    against this complete identity; no stage/SM interpolation is performed.
    Raw observation identities prevent repeated telemetry snapshots from being
    interpreted as independent samples. Missing shapes still return None.
    """
    if (not isinstance(expected_identity, dict)
            or set(expected_identity) != COLD_IDENTITY_FIELDS):
        raise ValueError('Complete CPU measurement identity required')
    expected = json.loads(json.dumps(expected_identity, allow_nan=False))
    for name in ('nprobe', 'topk', 'cpu_threads', 'max_batch_queries'):
        if type(expected[name]) is not int or expected[name] <= 0:
            raise ValueError('Positive integer CPU workload dimensions required')
    for name in ('index_sha256', 'workload_envelope_sha256'):
        value = expected[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Exact measurement digests required')
    if not isinstance(expected['hardware'], str) or not expected['hardware'].strip():
        raise ValueError('Explicit CPU/GPU hardware identity required')
    if (artifact.get('schema') != 'raglease.cold-worker.v1'
            or artifact.get('identity') != expected
            or artifact.get('context_conditioned') is not False):
        raise ValueError('CPU profile schema/domain mismatch')
    rows = artifact.get('observations')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty deduplicated CPU observations required')
    previous = None
    samples = []
    for row in rows:
        start, end, ms = (row.get(k) for k in
            ('started_monotonic_s', 'finished_monotonic_s', 'elapsed_ms'))
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end,ms))
                or start < 0 or end <= start or ms <= 0
                or not math.isclose(ms,(end-start)*1000,rel_tol=1e-6,abs_tol=1e-6)
                or (previous is not None and start < previous)):
            raise ValueError('Invalid/repeated single-worker observation')
        previous = end
        samples.append((row.get('shape', ()), ms))
    return capture_cold_worker_profile({'service_samples': samples},
                                      percentile=percentile, min_samples=min_samples)


@dataclass(frozen=True)
class ExactANNProfile:
    partition: int
    batch_size: int
    samples: int
    percentile: float
    gpu_ms: float
    source: str = 'live_partition_batch'


@dataclass(frozen=True)
class FrozenPrefillLayerProfile:
    percentile: float
    # (partition, power-of-two token bucket, per-layer ms, sample count)
    entries: tuple[tuple[int, int, float, int], ...]

    def query_layer_ms(self, partition: int, token_count: int) -> float | None:
        if (type(partition) is not int or partition < 0
                or type(token_count) is not int or token_count <= 0):
            raise ValueError('Valid partition and positive prefill tokens required')
        bucket = 1 << (token_count - 1).bit_length()
        return next((ms for p, b, ms, _ in self.entries
                     if p == partition and b == bucket), None)

    def query_bundle_ms(self, partition, token_count, start_layer, end_layer):
        """Linear normalized-bundle estimate, not a measured bundle quantile.

        Keep exact partition/token-bucket coverage. No layer-interval or
        batch-shape conditioning is claimed by this existing profile format.
        """
        if (type(start_layer) is not int or type(end_layer) is not int
                or start_layer < 0 or end_layer <= start_layer):
            raise ValueError('Positive actual prefill layer interval required')
        layer_ms = self.query_layer_ms(partition, token_count)
        return None if layer_ms is None else layer_ms*(end_layer-start_layer)


def capture_prefill_layer_profile(scheduler, *, percentile: float = 95,
                                  min_samples: int = 4):
    """Freeze completed CUDA bundle elapsed/layers observations.

    Exact partition and existing power-of-two TOKEN bucket only. No SM scaling,
    nominal fallback, cold prior or another bucket. These are normalized bundle
    observations, NOT individual layer measurements or complete host service;
    multiplying by a new bundle length is a linear estimate, not its measured
    P95. Layer interval, request batch and overlap are not conditioned here.
    The caller must constrain its workload envelope and validate that estimate.
    """
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid prefill profile selection')
    entries = []
    histories = getattr(scheduler, 'rag_prefill_layer_ms_by_partition_bucket', {})
    for key, samples in histories.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        partition, bucket = key
        if (type(partition) is not int or partition < 0 or type(bucket) is not int
                or bucket <= 0 or bucket & (bucket - 1)):
            continue
        values = tuple(samples)
        if len(values) < min_samples or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in values):
            continue
        ordered = sorted(values)
        at = (len(ordered)-1)*percentile/100
        lo, hi = math.floor(at), math.ceil(at)
        ms = ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
        entries.append((partition, bucket, ms, len(values)))
    return FrozenPrefillLayerProfile(percentile, tuple(sorted(entries)))


@dataclass(frozen=True)
class FrozenDecodeBucketProfile:
    partition: int
    overlap_class: str
    bucket_width: int
    percentile: float
    # (bucket lower bound, measured iteration ms, sample count)
    entries: tuple[tuple[int, float, int], ...]
    timing_basis: str = 'completion_interval'

    def query_ms(self, batch_size: int) -> float | None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Positive decode batch required')
        bucket = ((batch_size-1)//self.bucket_width)*self.bucket_width+1
        return next((ms for key, ms, _ in self.entries if key == bucket), None)


def capture_decode_iteration_profile(scheduler, *, partition: int, overlap_class: str,
                                     percentile: float = 95, min_samples: int = 4,
                                     timing_basis: str = 'completion_interval'):
    """Freeze one existing live partition/batch/overlap table per decision.

    Uses COMPLETE iteration samples, not GPU-only time. Bucket boundaries are
    exactly the runtime's existing rule. No other-partition/solo/global fallback,
    no rescaling and no mutation. These observations are NOT context-length
    conditioned: the adapter must restrict use to its declared workload domain.
    """
    width = getattr(scheduler, 'rag_ann_decode_batch_bucket_size', 4)
    if (type(partition) is not int or partition < 0 or type(width) is not int or width <= 0
            or overlap_class not in ('solo','ann','prefill','ann_prefill')
            or type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid decode profile selection')
    tables={'completion_interval':'rag_ann_decode_partition_batch_overlap_ms',
            'submission_service':'rag_decode_submission_service_ms'}
    if timing_basis not in tables:
        raise ValueError('Explicit decode timing basis required')
    buckets = getattr(scheduler, tables[timing_basis], {}).get(partition, {})
    entries = []
    for bucket, overlaps in buckets.items():
        if type(bucket) is not int or bucket <= 0 or (bucket-1) % width:
            continue
        values = tuple(overlaps.get(overlap_class, ()))
        if len(values) < min_samples or any(not math.isfinite(v) or v <= 0 for v in values):
            continue
        values = sorted(values)
        at = (len(values)-1)*percentile/100
        lo, hi = math.floor(at), math.ceil(at)
        ms = values[lo] + (values[hi]-values[lo])*(at-lo)
        entries.append((bucket, ms, len(values)))
    return FrozenDecodeBucketProfile(partition, overlap_class, width, percentile,
                                     tuple(sorted(entries)),timing_basis)


@dataclass(frozen=True)
class FrozenDecodeDecisionProfile:
    batch_size: int
    profiles: tuple[FrozenDecodeBucketProfile, ...]

    def query_ms(self, partition: int, overlap_class: str) -> float | None:
        return self.query_batch_ms(partition, overlap_class, self.batch_size)

    def query_batch_ms(self, partition: int, overlap_class: str,
                       batch_size: int) -> float | None:
        """Query another forecast cohort from the SAME frozen histories.

        Retains measured batch buckets, not context-length conditioning. No
        live read, global fallback or extrapolation to a missing bucket.
        """
        profile = next((p for p in self.profiles if p.partition == partition
                        and p.overlap_class == overlap_class), None)
        return None if profile is None else profile.query_ms(batch_size)


def capture_decode_decision_profile(scheduler, *, partitions, batch_size: int,
                                    percentile: float = 99, min_samples: int = 4,
                                    timing_basis: str = 'completion_interval'):
    """All candidate/overlap queries share one observed batch and frozen data.

    Capture under scheduler serialization. Complete iteration durations retain
    the limitations of capture_decode_iteration_profile (no context buckets).
    The candidate selector must not substitute solo when co-run data is absent.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive observed decode batch required')
    partitions = tuple(partitions)
    if any(type(p) is not int or p < 0 for p in partitions) or len(set(partitions)) != len(partitions):
        raise ValueError('Unique nonnegative partitions required')
    profiles = tuple(capture_decode_iteration_profile(
        scheduler, partition=p, overlap_class=overlap,
        percentile=percentile, min_samples=min_samples,timing_basis=timing_basis)
        for p in partitions for overlap in ('solo','ann','prefill','ann_prefill'))
    return FrozenDecodeDecisionProfile(batch_size, profiles)


@dataclass(frozen=True)
class FrozenDecodeCapacityProfile:
    fast_percentile: float
    iterations: FrozenDecodeDecisionProfile

    def query_capacity_ms(self, partition: int, overlap_class: str,
                          batch_size: int) -> float | None:
        return self.iterations.query_batch_ms(partition, overlap_class, batch_size)


def capture_decode_capacity_profile(scheduler, *, partitions, batch_size: int,
                                    fast_percentile: float, min_samples: int = 4):
    """Separate fast-service forecast for capacity, NOT a latency SLO table.

    Explicit lower-half quantile anticipates faster KV consumption. It is not
    a hard lower bound on iteration duration or an OOM safety proof. Actual
    allocation checks and reserves remain mandatory; workload-domain and
    forecast error must be checked before freezing formal configurations.
    """
    if not math.isfinite(fast_percentile) or not 0 <= fast_percentile <= 50:
        raise ValueError('Explicit lower-half capacity quantile required, not latency P99')
    frozen = capture_decode_decision_profile(scheduler, partitions=partitions,
        batch_size=batch_size, percentile=fast_percentile, min_samples=min_samples)
    return FrozenDecodeCapacityProfile(fast_percentile, frozen)


ANN_IDENTITY_FIELDS = frozenset({
    'gpu', 'partition_sm_counts', 'index_sha256', 'nprobe', 'topk',
    'gpu_index_residency', 'coexecution_bucket',
})


def export_exact_ann_samples(scheduler, *, identity: dict) -> dict:
    """Explicit workload envelope: callers must capture it with the samples.

    Raw samples preserve percentile semantics; a saved P95 cannot answer P99.
    This format is separate from legacy shape-aggregated cold-start priors.
    Use only a calibration window whose identity stayed fixed. The ordinary
    dynamic runtime history does NOT establish a fixed residency/overlap bucket;
    attaching its final-state identity would mislabel earlier samples. This
    function validates representation, not experimental provenance.
    """
    if set(identity) != ANN_IDENTITY_FIELDS or any(v is None for v in identity.values()):
        raise ValueError('Complete ANN measurement identity required')
    # JSON round-trip detaches nested identity values from mutable runtime data.
    frozen_identity = json.loads(json.dumps(identity, allow_nan=False))
    histories = getattr(scheduler, 'rag_ann_gpu_service_ms_by_partition_batch_size', {})
    samples = {f'{partition}:{batch}': list(values)
               for (partition, batch), values in histories.items()}
    artifact = dict(schema='raglease.exact-ann.v1', identity=frozen_identity, samples=samples)
    # The same validator is used on both sides of persistence.
    restore_exact_ann_samples(artifact, expected_identity=frozen_identity)
    return artifact


def restore_exact_ann_samples(artifact: dict, *, expected_identity: dict):
    """Validate into a detached prior map; never overwrite live measurements.

    A mismatch rejects the entire artifact. In particular, no nprobe-only
    compatibility shortcut and no inferred index residency/overlap identity.
    Runtime integration must supply a measured stable workload bucket.
    """
    if (set(expected_identity) != ANN_IDENTITY_FIELDS
            or any(v is None for v in expected_identity.values())):
        raise ValueError('Complete expected ANN identity required')
    expected = json.loads(json.dumps(expected_identity, allow_nan=False))
    if artifact.get('schema') != 'raglease.exact-ann.v1' or artifact.get('identity') != expected:
        raise ValueError('Incompatible exact ANN profile')
    entries = artifact.get('samples')
    if not isinstance(entries, dict):
        raise ValueError('Missing exact ANN samples')
    restored = {}
    for key, values in entries.items():
        try:
            partition, batch = map(int, key.split(':'))
        except (AttributeError, TypeError, ValueError):
            raise ValueError('Invalid exact ANN shape') from None
        if partition < 0 or batch <= 0 or key != f'{partition}:{batch}':
            raise ValueError('Invalid exact ANN shape')
        if not isinstance(values, (list, tuple)) or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Invalid exact ANN sample')
        restored[(partition, batch)] = tuple(values)
    return restored


@dataclass(frozen=True)
class FrozenANNServiceProfile:
    entries: tuple

    def query_ms(self, partition: int, batch_size: int):
        if type(partition) is not int or partition < 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Exact positive ANN shape required')
        return next((entry.gpu_ms for entry in self.entries
                     if entry.partition == partition and entry.batch_size == batch_size), None)


@dataclass(frozen=True)
class PredictedANNProfile:
    partition: int
    batch_size: int
    gpu_ms: float
    list_visits: int
    vectors_scanned: int
    max_query_vectors: int
    max_list_vectors: int
    query_vector_sq_sum: int = 0
    source: str = 'linear_ann_nominal_coexecution'


@dataclass(frozen=True)
class FrozenLinearANNWorkProfile:
    """Candidate-route ANN pricing with an optional nominal slowdown cell."""
    predictor: object
    expected_identity: tuple
    partition_to_sm: tuple
    corrections: object = None
    expected_correction_identity: tuple = ()
    overlap_class: str = 'solo'
    nearest_domain_fallback: bool = False
    correction_guard_fallback: bool = False

    def __post_init__(self):
        mapping=dict(self.partition_to_sm)
        if (len(mapping)!=len(self.partition_to_sm)
                or any(type(p) is not int or p<0 or type(sm) is not int or sm<=0
                       for p,sm in self.partition_to_sm)):
            raise ValueError('Explicit ANN partition-to-SM mapping required')
        corrected=self.corrections is not None
        if corrected != bool(self.expected_correction_identity):
            raise ValueError('Correction table and identity must be configured together')
        if corrected and self.overlap_class=='solo':
            raise ValueError('ANN correction requires a non-solo overlap')
        if (type(self.nearest_domain_fallback) is not bool
                or type(self.correction_guard_fallback) is not bool
                or self.correction_guard_fallback and not corrected):
            raise ValueError('Explicit compatible ANN fallback policies required')

    def query_work_ms(self,partition,batch_size,list_visits,vectors_scanned,
                      max_query_vectors,max_list_vectors,query_vector_sq_sum=None):
        sm=dict(self.partition_to_sm).get(partition)
        if sm is None:
            return None
        solo=self.predictor.predict_ann_ms(sm,batch_size,list_visits,vectors_scanned,
            max_query_vectors,max_list_vectors,query_vector_sq_sum,
            expected_identity=self.expected_identity)
        if solo is None and self.nearest_domain_fallback:
            solo=self.predictor.predict_ann_nearest_ms(sm,batch_size,list_visits,
                vectors_scanned,max_query_vectors,max_list_vectors,
                query_vector_sq_sum,
                expected_identity=self.expected_identity)
        if solo is None or self.corrections is None:
            return solo
        factors=self.corrections.query('ann',partition,self.overlap_class,
            (float(batch_size),float(list_visits),float(vectors_scanned),1.),
            expected_identity=self.expected_correction_identity)
        if factors is None and self.correction_guard_fallback:
            guards=[cell.guard_factor for cell in self.corrections.cells
                    if cell.stage=='ann' and cell.partition==partition
                    and cell.overlap==self.overlap_class]
            factors=None if not guards else (max(guards),max(guards))
        return None if factors is None else solo*factors[0]


def bind_linear_ann_work_profile(predictor, *, expected_identity,
                                 partition_to_sm, corrections=None,
                                 expected_correction_identity=(),
                                 overlap_class='solo',nearest_domain_fallback=False,
                                 correction_guard_fallback=False):
    """Bind the model without collapsing candidate-specific L_A and V_A."""
    return FrozenLinearANNWorkProfile(predictor,expected_identity,
        tuple(sorted(dict(partition_to_sm).items())),corrections,
        tuple(expected_correction_identity),overlap_class,
        nearest_domain_fallback,correction_guard_fallback)


def bind_linear_ann_service_profile(predictor, *, expected_identity,
                                    partition_to_sm, feature_query, max_batch_size,
                                    corrections=None, expected_correction_identity=(),
                                    overlap_class='solo'):
    """Materialize the ANN linear model behind the native exact-query interface.

    feature_query(partition, batch) returns admitted GPU list visits, scanned
    vectors, maximum per-query scanned vectors, and maximum single-list size
    for the evaluated routing/residency. Unsupported routing shapes remain
    absent. Coexecuted profiles use the median measured/solo ratio; ANN has no
    separate guard factor in the paper contract.
    """
    if not callable(feature_query) or type(max_batch_size) is not int or max_batch_size<=0:
        raise ValueError('Explicit ANN feature query and positive batch limit required')
    corrected=corrections is not None
    if corrected != bool(expected_correction_identity):
        raise ValueError('Correction table and identity must be configured together')
    if corrected and overlap_class=='solo':
        raise ValueError('ANN correction requires a non-solo overlap')
    entries=[]
    for partition,sm in sorted(dict(partition_to_sm).items()):
        if type(partition) is not int or partition<0 or type(sm) is not int or sm<=0:
            raise ValueError('Explicit ANN partition-to-SM mapping required')
        for batch in range(1,max_batch_size+1):
            work=feature_query(partition,batch)
            if work is None:
                continue
            if not isinstance(work,(tuple,list)) or len(work) not in (4,5):
                raise ValueError('ANN feature query must return totals and imbalance maxima')
            visits,vectors,max_query,max_list,*extra=work
            query_sq=extra[0] if extra else None
            solo=predictor.predict_ann_ms(sm,batch,visits,vectors,max_query,max_list,query_sq,
                                          expected_identity=expected_identity)
            if solo is None:
                continue
            factor=1.
            if corrected:
                pair=corrections.query('ann',partition,overlap_class,
                    (float(batch),float(visits),float(vectors),1.),
                    expected_identity=expected_correction_identity)
                if pair is None:
                    continue
                factor=pair[0]
            entries.append(PredictedANNProfile(partition,batch,solo*factor,
                                               visits,vectors,max_query,max_list,
                                               0 if query_sq is None else query_sq))
    return FrozenANNServiceProfile(tuple(entries))


@dataclass(frozen=True)
class FrozenANNProbeServiceProfile:
    table: FrozenMemoryTransitionProfile

    def query_ms(self, partition, hot_counts):
        counts = tuple(hot_counts)
        if (type(partition) is not int or partition < 0 or not counts
                or any(type(n) is not int or n < 0 for n in counts)):
            raise ValueError('Exact ordered hot probe counts required')
        if not any(counts):
            return 0.0  # No GPU search, not a missing-profile fallback.
        return self.table.query_remaining_ms((partition, counts), elapsed_ms=0.)


@dataclass(frozen=True)
class FrozenRadixEvictionProfile:
    """Empirical host cost, conditional on a sufficient native eviction.

    Capacity/lock eligibility must be checked separately. Tree shape varies;
    this is neither a hard latency bound nor evidence of physical readiness.
    """
    table: FrozenMemoryTransitionProfile

    def query_ms(self, requested_tokens):
        if type(requested_tokens) is not int or requested_tokens<=0:
            raise ValueError('Positive exact eviction request required')
        return self.table.query_remaining_ms((requested_tokens,),elapsed_ms=0.)


def capture_radix_eviction_profile(observations, *, percentile=95., min_samples=4):
    if (type(min_samples) is not int or min_samples<=0
            or not math.isfinite(percentile) or not 0<=percentile<=100):
        raise ValueError('Invalid eviction profile selection')
    grouped={}
    for sample in tuple(observations):
        if not isinstance(sample,(tuple,list)) or len(sample)!=4:
            continue
        requested,freed,nodes,duration=sample
        if (any(type(n) is not int or n<=0 for n in (requested,freed,nodes))
                or freed<requested or type(duration) not in (int,float)
                or not math.isfinite(duration) or duration<=0):
            continue
        grouped.setdefault((requested,),[]).append(duration)
    return FrozenRadixEvictionProfile(FrozenMemoryTransitionProfile(percentile,min_samples,
        tuple((key,tuple(sorted(values))) for key,values in sorted(grouped.items()))))


@dataclass(frozen=True)
class FrozenHybridCollectionProfile:
    table: FrozenMemoryTransitionProfile

    def query_full_cpu_remaining_ms(self, mode, rows, topk, *, elapsed_ms=0.):
        """Full-CPU post-join collection; never borrow hybrid samples."""
        if (mode not in ('worker','inline') or type(rows) is not int or rows<=0
                or type(topk) is not int or topk<=0):
            raise ValueError('Exact full-CPU collection shape required')
        return self.table.query_remaining_ms(('full_cpu',mode,rows,topk),elapsed_ms=elapsed_ms)

    def query_ms(self, mode, rows, topk, cpu_search_count, has_gpu):
        return self.query_remaining_ms(mode,rows,topk,cpu_search_count,has_gpu,elapsed_ms=0.)

    def query_remaining_ms(self, mode, rows, topk, cpu_search_count, has_gpu, *, elapsed_ms):
        """Conditional post-join service residual, not queue or retrieval age.

        The caller supplies age since collection actually started. Queued
        work uses zero age and is serialized behind the active collector.
        This empirical residual is not a hard bound or a censoring correction.
        """
        if (mode not in ('worker', 'inline') or type(rows) is not int or rows <= 0
                or type(topk) is not int or topk <= 0
                or type(cpu_search_count) is not int or cpu_search_count < 0
                or type(has_gpu) is not bool):
            raise ValueError('Exact hybrid collection shape required')
        return self.table.query_remaining_ms((mode, rows, topk, cpu_search_count, has_gpu),
                                             elapsed_ms=elapsed_ms)


def capture_hybrid_collection_profile(scheduler, *, percentile=95., min_samples=4):
    """Freeze post-join service only; never substitute launch-to-result time."""
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid collection profile selection')
    entries = []
    for key, history in getattr(scheduler, 'rag_hybrid_collection_ms_by_shape', {}).items():
        full_cpu=(isinstance(key,tuple) and len(key)==4 and key[0]=='full_cpu'
                  and key[1] in ('worker','inline')
                  and all(type(n) is int and n>0 for n in key[2:]))
        if not full_cpu and (not isinstance(key, tuple) or len(key) != 5 or key[0] not in ('worker', 'inline')
                or any(type(n) is not int or n <= 0 for n in key[1:3])
                or type(key[3]) is not int or key[3] < 0 or type(key[4]) is not bool):
            continue
        values = tuple(history)
        if values and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in values):
            entries.append((key, tuple(sorted(values))))
    return FrozenHybridCollectionProfile(FrozenMemoryTransitionProfile(percentile, min_samples, tuple(entries)))


def capture_ann_probe_service_profile(scheduler, *, percentile=95., min_samples=4):
    """Freeze exact partition/ordered hot-count samples, without row fallback.

    Uses the common keyed quantile representation at elapsed zero ONLY; these
    are CUDA service measurements, never host-clock residual/release samples.
    Index/list-size/workload scope still requires external identity validation.
    """
    if (type(min_samples) is not int or min_samples <= 0 or not math.isfinite(percentile)
            or not 0 <= percentile <= 100):
        raise ValueError('Invalid ANN probe profile selection')
    history = getattr(scheduler, 'rag_ann_gpu_service_ms_by_partition_hot_counts', {})
    samples = []
    for key, values in history.items():
        if (not isinstance(key, tuple) or len(key) != 2 or type(key[0]) is not int
                or key[0] < 0 or not isinstance(key[1], tuple) or not key[1]
                or any(type(n) is not int or n < 0 for n in key[1]) or not any(key[1])):
            continue
        values = tuple(values)
        if values and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            samples.append((key, tuple(sorted(values))))
    return FrozenANNProbeServiceProfile(FrozenMemoryTransitionProfile(percentile, min_samples, tuple(samples)))


def capture_ann_service_profile(scheduler, *, percentile=95., min_samples=4):
    """Freeze exact measured shapes once for compute credit and fitted scoring.

    No unknown-shape extrapolation. This is service, not host-clock release;
    the live-reader drain estimate must keep using the separate release table.
    Call under scheduler serialization, within the declared workload domain.
    """
    if (type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid ANN profile capture settings')
    history = getattr(scheduler, 'rag_ann_gpu_service_ms_by_partition_batch_size', {})
    entries = []
    for key in tuple(history):
        if (not isinstance(key, tuple) or len(key) != 2
                or type(key[0]) is not int or key[0] < 0
                or type(key[1]) is not int or key[1] <= 0):
            continue
        entry = query_exact_ann_profile(scheduler, partition=key[0], batch_size=key[1],
                                        percentile=percentile, min_samples=min_samples)
        if entry is not None:
            entries.append(entry)
    return FrozenANNServiceProfile(tuple(entries))


def query_exact_ann_profile(scheduler, *, partition: int, batch_size: int,
                            percentile: float = 95.0, min_samples: int = 4):
    """Read the fitted shape only; never extrapolate or use another partition.

    This is GPU service, NOT CPU retrieval, merge or total ANN completion time.
    Co-execution/workload conditioning remains the adapter's responsibility.
    Missing or invalid measurements return None, not a zero-cost operation.
    """
    if (type(partition) is not int or partition < 0 or type(batch_size) is not int
            or batch_size <= 0 or type(min_samples) is not int or min_samples <= 0
            or not math.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError('Invalid exact ANN profile query')
    history = getattr(scheduler, 'rag_ann_gpu_service_ms_by_partition_batch_size', {})
    values = tuple(history.get((partition, batch_size), ()))
    if len(values) < min_samples:
        return None
    if any(not math.isfinite(v) or v <= 0 for v in values):
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * percentile / 100
    low, high = math.floor(at), math.ceil(at)
    value = ordered[low] + (ordered[high] - ordered[low]) * (at - low)
    return ExactANNProfile(partition, batch_size, len(values), percentile, value)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Lease profile must contain a mapping: {path}")
    return data


def load_lease_performance_profile(
    path: str | Path, request_rate: float = -1.0
) -> dict[str, Any]:
    """Load the nearest request-rate profile from an offline profiler output."""

    profile_path = Path(path).expanduser().resolve()
    data = _load_mapping(profile_path)
    profiles = data.get("profiles_by_request_rate")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            f"Lease profile has no profiles_by_request_rate entries: {profile_path}"
        )

    available: list[tuple[float, str, dict[str, Any]]] = []
    for key, value in profiles.items():
        if not isinstance(value, dict) or not isinstance(
            value.get("performance_priors"), dict
        ):
            continue
        try:
            rate = float(value.get("request_rate", key))
        except (TypeError, ValueError):
            continue
        available.append((rate, str(key), value))
    if not available:
        raise ValueError(
            f"Lease profile contains no usable performance_priors: {profile_path}"
        )

    target_rate = float(request_rate)
    if target_rate <= 0.0:
        try:
            target_rate = float(data.get("default_profile_rate", -1.0))
        except (TypeError, ValueError):
            target_rate = -1.0
    if target_rate <= 0.0:
        target_rate = max(rate for rate, _, _ in available)

    selected_rate, selected_key, selected = min(
        available, key=lambda item: (abs(item[0] - target_rate), -item[0])
    )
    deployment = data.get("deployment", {})
    if not isinstance(deployment, dict):
        deployment = {}
    fixed_workload = data.get("fixed_workload", {})
    if not isinstance(fixed_workload, dict):
        fixed_workload = {}
    else:
        fixed_workload = dict(fixed_workload)
    if "nprobe" not in fixed_workload and deployment.get("nprobe") is not None:
        fixed_workload["nprobe"] = deployment["nprobe"]

    return {
        "path": str(profile_path),
        "schema_version": int(data.get("schema_version", 1)),
        "deployment": deployment,
        "fixed_workload": fixed_workload,
        "selected_key": selected_key,
        "selected_request_rate": selected_rate,
        "performance_priors": selected["performance_priors"],
    }


def _p95_ms(distribution: Any) -> float | None:
    if not isinstance(distribution, dict):
        return None
    try:
        value = float(distribution.get("p95_ms", 0.0))
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _latency_distribution(distribution: Any) -> dict[str, float]:
    if not isinstance(distribution, dict):
        return {}
    loaded: dict[str, float] = {}
    for key in ("samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        try:
            value = float(distribution[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value >= 0.0:
            loaded[key] = value
    return loaded


def _load_partition_bucket_tree(tree: Any) -> dict[int, dict[int, float]]:
    loaded: dict[int, dict[int, float]] = {}
    if not isinstance(tree, dict):
        return loaded
    for stream_key, buckets in tree.items():
        if not isinstance(buckets, dict):
            continue
        try:
            stream_idx = int(stream_key)
        except (TypeError, ValueError):
            continue
        target_buckets = loaded.setdefault(stream_idx, {})
        for bucket_key, distribution in buckets.items():
            try:
                bucket = int(bucket_key)
            except (TypeError, ValueError):
                continue
            value = _p95_ms(distribution)
            if value is not None:
                target_buckets[bucket] = value
    return loaded


def _load_partition_bucket_distribution_tree(
    tree: Any,
) -> dict[int, dict[int, dict[str, float]]]:
    loaded: dict[int, dict[int, dict[str, float]]] = {}
    if not isinstance(tree, dict):
        return loaded
    for stream_key, buckets in tree.items():
        if not isinstance(buckets, dict):
            continue
        try:
            stream_idx = int(stream_key)
        except (TypeError, ValueError):
            continue
        target_buckets = loaded.setdefault(stream_idx, {})
        for bucket_key, distribution in buckets.items():
            try:
                bucket = int(bucket_key)
            except (TypeError, ValueError):
                continue
            loaded_distribution = _latency_distribution(distribution)
            if loaded_distribution:
                target_buckets[bucket] = loaded_distribution
    return loaded


def apply_lease_performance_profile(
    scheduler, selection: dict[str, Any]
) -> dict[str, Any]:
    """Install cold-start priors without mixing them into live telemetry."""

    source = selection.get("performance_priors", {})
    phase = source.get("phase_latency_profiles", {})
    transitions = source.get("resource_transition_profiles", {})
    priors: dict[str, Any] = {
        "ann_gpu_service_by_batch_size": {},
        "prefill_layer_by_partition_bucket": {},
        "prefill_layer_distribution_by_partition_bucket": {},
        "decode_iteration_by_partition_batch_bucket": {},
        "decode_iteration_distribution_by_partition_batch_bucket": {},
        "decode_gpu_by_partition_batch_bucket": {},
        "decode_gpu_distribution_by_partition_batch_bucket": {},
        "switch_latency_by_pair": {},
    }
    seeded = {
        "ann_batch_buckets": 0,
        "prefill_layer_buckets": 0,
        "decode_buckets": 0,
        "switch_pairs": 0,
        "hbm_reload": 0,
    }
    fixed_workload = selection.get("fixed_workload", {})
    profile_nprobe = fixed_workload.get("nprobe")
    runtime_nprobe = getattr(
        getattr(scheduler, "ann_executor", None), "nprobe", None
    )
    try:
        profile_nprobe = int(profile_nprobe)
    except (TypeError, ValueError):
        profile_nprobe = None
    try:
        runtime_nprobe = int(runtime_nprobe)
    except (TypeError, ValueError):
        runtime_nprobe = None
    if runtime_nprobe is None:
        ann_profile_compatible = True
        ann_prior_skip_reason = None
    elif profile_nprobe is None:
        ann_profile_compatible = False
        ann_prior_skip_reason = "missing_profile_nprobe"
    else:
        ann_profile_compatible = profile_nprobe == runtime_nprobe
        ann_prior_skip_reason = (
            None if ann_profile_compatible else "nprobe_mismatch"
        )

    ann_profiles = phase.get("ann_gpu_service_by_batch_size", {})
    if ann_profile_compatible and isinstance(ann_profiles, dict):
        for batch_key, distribution in ann_profiles.items():
            try:
                batch_size = int(batch_key)
            except (TypeError, ValueError):
                continue
            value = _p95_ms(distribution)
            if value is None:
                continue
            priors["ann_gpu_service_by_batch_size"][batch_size] = value
            seeded["ann_batch_buckets"] += 1

    prefill_profiles = phase.get(
        "prefill_layer_latency_by_partition_bucket", {}
    )
    if isinstance(prefill_profiles, dict):
        for key, distribution in prefill_profiles.items():
            try:
                stream_text, bucket_text = str(key).split(":", 1)
                profile_key = (int(stream_text), int(bucket_text))
            except (TypeError, ValueError):
                continue
            value = _p95_ms(distribution)
            if value is None:
                continue
            priors["prefill_layer_by_partition_bucket"][profile_key] = value
            loaded_distribution = _latency_distribution(distribution)
            if loaded_distribution:
                priors[
                    "prefill_layer_distribution_by_partition_bucket"
                ][profile_key] = loaded_distribution
            seeded["prefill_layer_buckets"] += 1

    decode_iteration = _load_partition_bucket_tree(
        phase.get("decode_iteration_by_partition_batch_bucket", {})
    )
    decode_gpu = _load_partition_bucket_tree(
        phase.get("decode_gpu_step_by_partition_batch_bucket", {})
    )
    decode_iteration_distributions = _load_partition_bucket_distribution_tree(
        phase.get("decode_iteration_by_partition_batch_bucket", {})
    )
    decode_gpu_distributions = _load_partition_bucket_distribution_tree(
        phase.get("decode_gpu_step_by_partition_batch_bucket", {})
    )
    priors["decode_iteration_by_partition_batch_bucket"] = decode_iteration
    priors[
        "decode_iteration_distribution_by_partition_batch_bucket"
    ] = decode_iteration_distributions
    priors["decode_gpu_by_partition_batch_bucket"] = decode_gpu
    priors[
        "decode_gpu_distribution_by_partition_batch_bucket"
    ] = decode_gpu_distributions
    seeded["decode_buckets"] = sum(
        len(buckets) for buckets in decode_iteration.values()
    ) + sum(len(buckets) for buckets in decode_gpu.values())

    switch_profiles = transitions.get("green_context_switch_by_pair", {})
    if isinstance(switch_profiles, dict):
        for pair_key, distribution in switch_profiles.items():
            try:
                source_text, target_text = str(pair_key).split("->", 1)
                pair = (int(source_text), int(target_text))
            except (TypeError, ValueError):
                continue
            value = _p95_ms(distribution)
            if value is None:
                continue
            priors["switch_latency_by_pair"][pair] = value
            seeded["switch_pairs"] += 1

    hbm_profile = transitions.get("hbm_reload", {})
    if isinstance(hbm_profile, dict):
        try:
            reload_per_list = float(hbm_profile.get("reload_ms_per_list", 0.0))
        except (TypeError, ValueError):
            reload_per_list = 0.0
        policy = getattr(
            getattr(scheduler, "ann_executor", None), "_shared_ivf_policy", None
        )
        if policy is not None and reload_per_list > 0.0:
            policy.reload_ms_per_list = reload_per_list
            priors["hbm_reload_ms_per_list"] = reload_per_list
            seeded["hbm_reload"] += 1
        async_p95 = _p95_ms(hbm_profile.get("async_reload_latency"))
        if async_p95 is not None:
            priors["hbm_async_reload_p95_ms"] = async_p95
            seeded["hbm_reload"] += 1

    scheduler.rag_lease_profile_priors = priors
    status = {
        "loaded": True,
        "path": selection.get("path"),
        "schema_version": selection.get("schema_version"),
        "selected_request_rate": selection.get("selected_request_rate"),
        "seeded": seeded,
        "storage": "separate_prior_table",
        "compatibility": {
            "profile_nprobe": profile_nprobe,
            "runtime_nprobe": runtime_nprobe,
            "ann_profile_compatible": ann_profile_compatible,
            "ann_prior_skip_reason": ann_prior_skip_reason,
        },
    }
    scheduler.rag_lease_profile_status = status
    return status
