"""Pure single-proposal Full admission and fixed-plan event prediction.

This module never launches work or mutates the HBM allocator. The runtime must
provide *pure* planning callbacks and revalidate the snapshot at commit time.
It is deliberately not wired into the legacy side-effecting scheduler yet.
All times are offsets from one snapshot, not from transition completion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
import os
from typing import Protocol


def nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def engineering_failure_trace_enabled() -> bool:
    """Keep whole-interpreter failure tracing out of performance runs.

    ``sys.setprofile`` observes every Python call/return in the scheduler
    thread.  It is useful for a bounded engineering diagnostic, but enabling
    it implicitly for every nonformal decision turns an unsupported forecast
    into seconds of frontend blocking.  The trace never affects safety or
    admission and is therefore an explicit debugging facility.
    """
    return os.environ.get('RAGLEASE_ENGINEERING_FAILURE_TRACE') == '1'


@dataclass(frozen=True)
class FrozenHandoffBatches:
    """Host container metadata, not runnable work or KV ownership.

    A forecast driver must invoke these transitions at actual modeled host
    events. Token completion only marks members; filtering and transfer are
    distinct. This helper does not infer event times or future arrivals.
    """
    batches: tuple

    def __post_init__(self):
        names = ('running_batch', 'split_prefill_batch', 'rag_prefill_merge_pending_batch')
        if not isinstance(self.batches, tuple) or tuple(name for name, _ in self.batches) != names:
            raise ValueError('All three ordered handoff containers required')
        for name, members in self.batches:
            if (not isinstance(members, tuple) or any(not isinstance(r, tuple) or len(r)!=3 for r in members)
                    or len({r[0] for r in members}) != len(members)):
                raise ValueError('Frozen unique handoff batch members required')
            for rid, prompt, finished in members:
                if (not isinstance(rid, str) or not rid or type(prompt) is not int or prompt <= 0
                        or type(finished) is not bool):
                    raise ValueError('Invalid handoff member metadata')

    def _replace_batch(self, name, members):
        if name not in dict(self.batches):
            raise ValueError('Unknown handoff container')
        return FrozenHandoffBatches(tuple((n, tuple(members) if n == name else rows)
                                         for n, rows in self.batches))

    def mark_finished(self, request_ids):
        ids = tuple(request_ids)
        known = {r[0] for _, rows in self.batches for r in rows}
        if len(set(ids)) != len(ids) or not set(ids) <= known:
            raise ValueError('Completion must identify observed container members')
        return FrozenHandoffBatches(tuple((n, tuple((rid, prompt, done or rid in ids)
            for rid, prompt, done in rows)) for n, rows in self.batches))

    def filter_batch(self, name, *, excluded_request_ids=()):
        rows = dict(self.batches).get(name)
        if rows is None:
            raise ValueError('Unknown handoff container')
        excluded = tuple(excluded_request_ids)
        if len(set(excluded)) != len(excluded) or not set(excluded) <= {r[0] for r in rows}:
            raise ValueError('Filter exclusions must be explicit present members')
        return self._replace_batch(name, (r for r in rows if not r[2] and r[0] not in excluded))

    def transfer_batch(self, source, target, *, append):
        """Model caller's merge/move AND subsequent source-container clearing."""
        batches = dict(self.batches)
        if (source == target or source not in batches or target not in batches or type(append) is not bool):
            raise ValueError('Explicit distinct handoff transfer endpoints required')
        if not append and batches[target]:
            raise ValueError('Cannot overwrite occupied handoff container')
        merged = (*batches[target], *batches[source]) if append else batches[source]
        return self._replace_batch(target, merged)._replace_batch(source, ())

    def host_shape(self, partition, stage, incoming):
        if (type(partition) is not int or partition < 0
                or stage not in ('result', 'merge', 'deferred_merge')):
            raise ValueError('Explicit host observation partition/stage required')
        batches = dict(self.batches)
        if incoming not in batches:
            raise ValueError('Unknown incoming handoff container')
        rows = batches[incoming]
        if not rows:
            return None
        return partition, stage, tuple(r[1] for r in rows), len(batches['running_batch'])


@dataclass(frozen=True)
class ReferenceWork:
    key: str
    remaining: float


@dataclass(frozen=True)
class Request:
    key: str
    weight: float
    work: tuple[ReferenceWork, ...]


@dataclass(frozen=True)
class RequestReferenceProfile:
    """Fixed-reference measurements for one actual/estimated prompt shape.

    Values are reference-work units, not service under a candidate partition.
    Unknown prompt/output shapes need an explicit profiled estimate supplied by
    the adapter. A natural-generation token limit is not such an estimate.
    """
    reference_id: str
    prompt_tokens: int
    total_layers: int
    ann_query_ms: float
    prepare_ms: float
    prefill_layer_ms: float
    decode_token_ms: float
    estimated_total_output_tokens: int | None = None


@dataclass(frozen=True)
class ReferenceSelection:
    profiles: tuple[tuple[str, RequestReferenceProfile], ...]
    estimated_prompt_requests: tuple[str, ...]


def select_reference_profiles(frontiers, prompt_catalog, *, estimated_prompt_tokens: int | None):
    """Select common reference units before candidate construction.

    An ANN/preparation frontier has no retrieved prompt yet. Use only an
    explicitly configured profiled estimate and retain its provenance; never
    look ahead into evaluation passages or pretend downstream work is absent.
    Known prompt lengths require their exact catalog entry, not a different
    measured shape or unimplemented interpolation. None means missing data.
    """
    frontiers=tuple(frontiers)
    if len({f.request_id for f in frontiers}) != len(frontiers):
        raise ValueError('Unique reference request set required')
    if estimated_prompt_tokens is not None and (
            type(estimated_prompt_tokens) is not int or estimated_prompt_tokens <= 0):
        raise ValueError('Explicit positive prompt estimate required')
    if len({p.reference_id for p in prompt_catalog.values()}) > 1:
        raise ValueError('Reference catalog mixes configurations')
    for length,p in prompt_catalog.items():
        if type(length) is not int or length <= 0 or length != p.prompt_tokens:
            raise ValueError('Reference catalog shape mismatch')
    selected=[]; estimated=[]
    for f in frontiers:
        shape=f.prompt_tokens
        if shape is None:
            if f.stage not in ('ann_pending','ann_inflight','input_preparation'):
                return None
            shape=estimated_prompt_tokens
            estimated.append(f.request_id)
        profile=prompt_catalog.get(shape)
        if profile is None:
            return None
        selected.append((f.request_id,profile))
    return ReferenceSelection(tuple(selected),tuple(estimated))


def remaining_decode_tokens(frontier, profile: RequestReferenceProfile) -> int:
    """Shared output-length interpretation for denominator and event builder."""
    remaining = frontier.remaining_fixed_output if frontier.generation_follows else 0
    if remaining is None:
        estimate = profile.estimated_total_output_tokens
        if type(estimate) is not int or estimate < 0:
            raise ValueError('Natural/unknown output requires an explicit estimate')
        remaining = max(0, estimate - frontier.generated_tokens)
        if frontier.remaining_output_limit is not None:
            remaining = min(remaining, frontier.remaining_output_limit)
    if type(remaining) is not int or remaining < 0:
        raise ValueError('Invalid remaining output work')
    return remaining


def build_reference_requests(frontiers, profiles: dict[str, RequestReferenceProfile],
                             weights: dict[str, float]) -> tuple[Request, ...]:
    """Freeze common denominators ONCE before compute/memory proposals.

    Stage keys are logical work: CPU and GPU implementations share `ann`.
    For successive decode/layer operations, credit fractions must be cumulative
    completed units / units remaining in this snapshot (predict uses max).
    No execution batch, candidate partition or transition enters this function.
    """
    frontiers = tuple(frontiers)
    ids = [f.request_id for f in frontiers]
    if len(ids) != len(set(ids)) or set(ids) != set(profiles) or set(ids) != set(weights):
        raise ValueError('Reference inputs must cover the same unique request set')
    if len({p.reference_id for p in profiles.values()}) > 1:
        raise ValueError('One common reference configuration required')
    result = []
    stages = ('ann_pending', 'ann_inflight', 'input_preparation', 'prefill_queued',
              'prefill_active', 'prefill_merge_pending', 'decode')
    for f in frontiers:
        p = profiles[f.request_id]
        if f.stage not in stages or not p.reference_id:
            raise ValueError('Unknown frontier or reference configuration')
        if type(f.generated_tokens) is not int or f.generated_tokens < 0:
            raise ValueError('Invalid observed generated-token count')
        if type(f.preparation_follows) is not bool or type(f.generation_follows) is not bool:
            raise ValueError('Explicit downstream stage contract required')
        if f.generation_follows and not f.preparation_follows:
            raise ValueError('Generation requires an input preparation contract')
        if f.stage in stages[3:] and not f.generation_follows:
            raise ValueError('Generation frontier contradicts request contract')
        for count in (p.prompt_tokens, p.total_layers):
            if type(count) is not int or count <= 0:
                raise ValueError('Explicit positive reference shape required')
        if f.prompt_tokens is not None and f.prompt_tokens != p.prompt_tokens:
            raise ValueError('Reference profile does not match observed prompt')
        for value in (p.ann_query_ms, p.prepare_ms, p.prefill_layer_ms, p.decode_token_ms):
            nonnegative(value, 'reference service')
            if value == 0:
                raise ValueError('Reference work needs a positive measured service')
        nonnegative(weights[f.request_id], 'urgency weight')
        work = []
        if f.stage in ('ann_pending', 'ann_inflight'):
            if type(f.query_rows) is not int or f.query_rows <= 0:
                raise ValueError('Missing logical ANN query count')
            work.append(ReferenceWork('ann', f.query_rows * p.ann_query_ms))
        if f.preparation_follows and f.stage in stages[:3]:
            work.append(ReferenceWork('prepare', p.prepare_ms))
        if f.generation_follows and f.stage in stages[:5]:
            completed = f.completed_prefill_layers if f.stage == 'prefill_active' else 0
            if type(completed) is not int or not 0 <= completed <= p.total_layers:
                raise ValueError('Observed completed prefill endpoint required')
            if completed < p.total_layers:
                work.append(ReferenceWork('prefill', (p.total_layers-completed)*p.prefill_layer_ms))
        remaining = remaining_decode_tokens(f, p)
        if remaining:
            work.append(ReferenceWork('decode', remaining*p.decode_token_ms))
        result.append(Request(f.request_id, weights[f.request_id], tuple(work)))
    return tuple(result)


@dataclass(frozen=True)
class UrgentRequests:
    requests: tuple[Request, ...]
    time_to_deadline_ms: tuple[tuple[str, float], ...]


def freeze_request_urgency(frontiers, profiles, *, now_s: float,
                           e2e_slo_ms: float, horizon_ms: float):
    """Freeze one request set, reference denominator and bounded urgency.

    w=H/max(H, deadline-now). This is deadline proximity, NOT predicted
    completion laxity or SLO success probability. Late requests retain weight1
    rather than disappearing. Reference-work totals are never substituted for
    concurrent execution latency. All proposals and keep share this result.
    started_at must use the same monotonic runtime clock as now_s. When planned
    client arrival is unavailable, the adapter must report its admission-clock
    limitation rather than infer unseen client waiting or future arrivals.
    Missing clocks return None; invalid clocks/contracts are errors.
    """
    for value,name in ((now_s,'current clock'),(e2e_slo_ms,'E2E target'),
                       (horizon_ms,'prediction horizon')):
        nonnegative(value,name)
    if e2e_slo_ms == 0 or horizon_ms == 0:
        raise ValueError('Positive E2E target and horizon required')
    frontiers=tuple(frontiers)
    if any(getattr(f,'started_at',None) is None for f in frontiers):
        return None
    weights={}; deadlines=[]
    for f in frontiers:
        nonnegative(f.started_at,'request start clock')
        if f.started_at > now_s:
            raise ValueError('Request clock is in the future')
        remaining=e2e_slo_ms-(now_s-f.started_at)*1000.0
        weights[f.request_id]=horizon_ms/max(horizon_ms,remaining)
        deadlines.append((f.request_id,remaining))
    requests=build_reference_requests(frontiers,profiles,weights)
    return UrgentRequests(requests,tuple(deadlines))


@dataclass(frozen=True)
class ComputeLease:
    allocation: str
    ann_batch: int
    prefill_layers: int
    # Exact prepared cohort/shape; runtime refuses joint prefill without it.
    prefill_contract: tuple | None = None


@dataclass(frozen=True)
class PrefillCohortSelection:
    request_ids: tuple[str, ...]
    extend_tokens: int
    newly_protected_tokens: int
    stop_reason: str


@dataclass(frozen=True)
class CachedPrefillBudget:
    shapes: tuple
    max_new_tokens: tuple
    clip_max_new_tokens: int
    available_and_evictable_tokens: int
    running_token_reserve: float
    input_token_budget: int
    max_requests: int
    cumulative_lock_tokens: tuple
    batch_is_full: bool = False
    published_capacity_tokens: int | None = None
    allocated_kv_tokens: int | None = None

    def select(self):
        return select_cached_prefill_cohort(self.shapes,
            max_new_tokens=dict(self.max_new_tokens),
            clip_max_new_tokens=self.clip_max_new_tokens,
            available_and_evictable_tokens=self.available_and_evictable_tokens,
            running_token_reserve=self.running_token_reserve,
            input_token_budget=self.input_token_budget,
            max_requests=0 if self.batch_is_full else self.max_requests,
            cumulative_lock_tokens=self.cumulative_lock_tokens)

    def propose(self):
        """One compute-side cohort, before choosing elastic HBM.

        Respect FCFS, actual request slots and input limits but not the current
        memory shortage/full latch. Compute a sufficient budget analytically,
        then apply the SAME selector once; do not search memory allocations.
        The result is demand, never permission to dispatch or clear the latch.
        """
        limits = dict(self.max_new_tokens)
        if len(limits) != len(self.max_new_tokens):
            raise ValueError('Duplicate queued output contract')
        work = {s.request_id:s.extend_tokens+min(limits[s.request_id],self.clip_max_new_tokens)
                for s in self.shapes}
        sufficient = math.floor(self.running_token_reserve + sum(work.values())
                               + max(self.cumulative_lock_tokens,default=0)) + 1
        planned = replace(self, available_and_evictable_tokens=max(0,sufficient),
                          batch_is_full=False).select()
        return self.cohort_demand(planned)

    def cohort_demand(self, planned):
        """Account an already selected cohort without selecting or expanding it."""
        ids = planned.request_ids
        if tuple(s.request_id for s in self.shapes[:len(ids)]) != ids:
            raise ValueError('Demand must describe an actual FCFS prefix')
        limits = dict(self.max_new_tokens)
        work = {s.request_id: s.extend_tokens + min(limits[s.request_id], self.clip_max_new_tokens)
                for s in self.shapes}
        minimum = (math.floor(self.running_token_reserve
                              + sum(work[rid] for rid in planned.request_ids)
                              + planned.newly_protected_tokens)+1
                   if planned.request_ids else 0)
        return PrefillComputeCohort(planned, minimum,
            max(0,minimum-self.available_and_evictable_tokens))


@dataclass(frozen=True)
class PrefillComputeCohort:
    selection: PrefillCohortSelection
    # Free + evictable tokens BEFORE locking this proposed cohort. Includes
    # the existing adder's output reserve, not just immediate input slots.
    minimum_available_and_evictable_tokens: int
    additional_capacity_tokens: int


def select_cached_prefill_cohort(shapes, *, max_new_tokens, clip_max_new_tokens,
                                 available_and_evictable_tokens, running_token_reserve,
                                 input_token_budget, max_requests, cumulative_lock_tokens,
                                 page_size=1):
    """Pure counterpart of the ordinary cached, non-chunked FCFS adder.

    Current E1 uses unit pages, no host loading/LoRA/preemption/chunking. This
    function does not generalize that contract silently. Caller supplies the
    real clipped-output constant and observed running reserve, not future EOS.
    Lock costs describe successive queue prefixes from the SAME radix snapshot.
    A rejected head stops selection; the runtime's first-request input-budget
    exception and strict total-capacity boundary are preserved, not retuned.
    Returned input slots are new allocation; protected cache slots are not.
    """
    shapes = tuple(shapes); locks = tuple(cumulative_lock_tokens)
    ids = tuple(s.request_id for s in shapes)
    if len(set(ids)) != len(ids) or set(ids) != set(max_new_tokens) or len(locks) != len(ids):
        raise ValueError('Consistent unique ordered cohort inputs required')
    if type(page_size) is not int or page_size != 1:
        raise ValueError('Cached FCFS preview currently requires unit pages')
    for value in (clip_max_new_tokens, available_and_evictable_tokens,
                  input_token_budget, max_requests, *locks, *max_new_tokens.values()):
        if type(value) is not int or value < 0:
            raise ValueError('Nonnegative integer admission budgets required')
    nonnegative(running_token_reserve, 'observed running reserve')
    if any(b < a for a,b in zip(locks,locks[1:])):
        raise ValueError('Cumulative prefix protection cannot decrease')
    selected = []; inputs = 0; protected = 0
    total_offset = running_token_reserve
    reason = 'queue_exhausted'
    for shape, locked in zip(shapes, locks):
        if type(shape.extend_tokens) is not int or shape.extend_tokens <= 0:
            raise ValueError('Positive exact uncached input length required')
        if len(selected) >= max_requests:
            reason = 'request_slots'; break
        output = min(max_new_tokens[shape.request_id], clip_max_new_tokens)
        required = shape.extend_tokens + output
        if required >= available_and_evictable_tokens-protected-total_offset:
            reason = 'total_tokens'; break
        if shape.extend_tokens >= input_token_budget-inputs and selected:
            reason = 'input_budget'; break
        if required >= available_and_evictable_tokens-locked-total_offset:
            reason = 'prefix_lock_capacity'; break
        selected.append(shape.request_id)
        protected = locked
        inputs += shape.extend_tokens
        total_offset += required
        if (available_and_evictable_tokens-protected-total_offset <= 0
                or available_and_evictable_tokens-protected-inputs <= 0):
            reason = 'total_tokens'; break
        if input_token_budget-inputs <= 0:
            reason = 'input_budget'; break
    return PrefillCohortSelection(tuple(selected), inputs, protected, reason)


@dataclass(frozen=True)
class MemoryLease:
    allocation: str
    ann_batch_cap: int
    prefill_layers_cap: int
    # Keys identify actual backing/guaranteed release, not desired capacity.
    # Infinity is allowed: the backing will not become usable in this window.
    ready: tuple[tuple[str, float], ...] = ()
    supported: bool = True
    # Extra migration completion gates; base KV/release readiness above is
    # never removed by diagnostic counterfactuals.
    transition_ready: tuple[tuple[str, float], ...] = ()
    # Opaque immutable runtime intent belonging to THIS projected lease.
    # Never substitute another same-event target during physical preparation.
    resource_plan: object | None = None
    # Plan-specific retained routing captured by the runtime backing adapter.
    # Builders/activation must reuse this observation, not reroute a new view.
    routing_preview: object | None = None
    # Safety/physical feasibility is independent from timing-profile coverage.
    # An adapter must set this when a required transition estimate is missing;
    # do not encode an unknown delay as zero or infinity (a known nonarrival).
    prediction_unavailable: str | None = None


@dataclass(frozen=True)
class JointMemoryDemand:
    compute: ComputeLease
    required_kv_tokens: int
    proposed_ann_rows: int
    prefill_tokens: int
    decode_tokens: int
    admission_capacity_floor: int = 0


def fit_current_kv_continuation(demand, *, published_tokens, allocated_tokens):
    """Fit the keep branch to observed backing, without a new allocation.

    A target's forecast is not a prerequisite for retaining current ownership.
    Drop unbacked NEW prefill rather than claiming eviction has happened. Bound
    the keep growth budget to physically free slots; per-request future progress
    must still obey the prospective pool and native allocator checks. Prepared
    prefill already owns its KV and can retain its current layer limit.
    This must never be applied to the target demand.
    """
    if not isinstance(demand, JointMemoryDemand):
        raise ValueError('Explicit current demand required')
    if (type(published_tokens) is not int or type(allocated_tokens) is not int
            or not 0 <= allocated_tokens <= published_tokens):
        raise ValueError('Consistent published physical capacity required')
    if demand.required_kv_tokens <= published_tokens:
        return demand
    compute = demand.compute
    if demand.prefill_tokens:
        compute = replace(compute, prefill_layers=0, prefill_contract=None)
    return JointMemoryDemand(compute, published_tokens, demand.proposed_ann_rows,
        0, min(demand.decode_tokens, published_tokens-allocated_tokens), 0)


def memory_lease_from_hbm_plan(proposed, plan, *, allocation, event_id,
                               ready, kv_key, scratch_key, index_key,
                               kv_transition_ms=None, scratch_transition_ms=None):
    """Translate one native plan, without replanning or asserting activation.

    `ready` describes guaranteed base backing/release, including actual index
    list readiness supplied by the routing adapter. Transition estimates are
    absolute completion offsets from the COMMON window origin, not durations
    to add to one another. Existing decode must use its own backing key.
    Missing required estimates invalidate scoring, not physical feasibility;
    Independent still must await the real native commit/activation fences.
    Allocation identity must describe all target ownership, not only KV size.
    """
    if not isinstance(allocation, str) or not allocation:
        raise ValueError('Explicit joint allocation identity required')
    if plan.event_id != event_id or plan.kv.event_id != event_id:
        raise ValueError('HBM plan belongs to a different decision event')
    if plan.status != 'available' or plan.ann is None or plan.kv.status not in ('ready', 'grow'):
        return MemoryLease(allocation, 0, 0, supported=False, resource_plan=plan)
    fit = plan.ann.fit
    if (fit.proposed_rows != proposed.ann_batch or type(fit.fitted_rows) is not int
            or not 0 <= fit.fitted_rows <= proposed.ann_batch):
        raise ValueError('Workspace fit must belong to the single compute proposal')
    backing = dict(ready)
    if len(backing) != len(ready):
        raise ValueError('Duplicate backing keys')
    required = [kv_key] + ([scratch_key, index_key] if fit.fitted_rows else [])
    if any(not isinstance(key, str) or not key or key not in backing for key in required):
        return MemoryLease(allocation, 0, 0, supported=False, resource_plan=plan)
    if len(set(required)) != len(required):
        raise ValueError('KV, scratch and index require distinct readiness gates')
    for value in backing.values():
        nonnegative(value, 'base backing readiness')
    missing = []; transitions = []
    for needed, key, estimate in (
            (plan.kv.status == 'grow', kv_key, kv_transition_ms),
            (fit.fitted_rows > 0 and plan.ann.mapping_required, scratch_key, scratch_transition_ms)):
        if not needed:
            continue
        if estimate is None:
            missing.append(key)
        else:
            nonnegative(estimate, 'absolute transition completion')
            transitions.append((key, estimate))
    return MemoryLease(allocation, fit.fitted_rows, proposed.prefill_layers,
        ready=tuple(ready), transition_ready=tuple(transitions), resource_plan=plan,
        prediction_unavailable=('Missing transition profiles: ' + ', '.join(missing)) if missing else None)


def apply_joint_activation_readiness(memory, completion, *, new_work_keys):
    """New work waits for whole-lease activation, existing work does not.

    Keys must name only newly admitted work's backing. Preserve separate keys
    for compatible existing decode/in-flight work. Diagnostic views must pass
    completion recomposed for that view instead of treating G2 as universal.
    """
    if not memory.supported:
        return memory
    keys=tuple(new_work_keys)
    if len(keys)!=len(set(keys)) or any(key not in dict(memory.ready) for key in keys):
        raise ValueError('Unique declared new-work backing keys required')
    activation=getattr(completion,'activation_ms',None)
    if activation is None:
        return replace(memory,prediction_unavailable=memory.prediction_unavailable or
                       'Joint activation completion unavailable')
    nonnegative(activation,'joint activation completion')
    transitions=dict(memory.transition_ready)
    for key in keys:
        transitions[key]=max(transitions.get(key,0.),activation)
    return replace(memory,transition_ready=tuple(transitions.items()))


def derive_joint_memory_demand(proposed: ComputeLease, *, allocated_kv_tokens: int,
                               new_prefill_tokens: tuple[tuple[str, int], ...],
                               decode_growth_tokens: tuple[tuple[str, int], ...],
                               reserve_tokens: int = 0,
                               admission_capacity_floor: int = 0) -> JointMemoryDemand:
    """Account demand for ONE proposal from already-selected current work.

    new_prefill_tokens contains only this proposal's newly admitted cohort,
    with page-rounded unallocated token slots, never the whole waiting queue
    or a prepared batch already included in allocated_kv_tokens. Decode growth
    comes from a fixed-proposal horizon forecast, not unknown future arrivals.
    Do not subtract potentially evictable cache or unreleased KV: such capacity
    needs an explicit transition/readiness guarantee in the memory planner.
    This accounting does not generate that forecast or choose a cohort.
    """
    values=(allocated_kv_tokens,reserve_tokens,proposed.ann_batch,proposed.prefill_layers,
            admission_capacity_floor)
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('Nonnegative token counts and compute limits required')
    totals=[]
    for group in (new_prefill_tokens,decode_growth_tokens):
        ids=[rid for rid,_ in group]
        if len(ids)!=len(set(ids)) or any(not isinstance(rid,str) or not rid for rid in ids):
            raise ValueError('Unique request identities per demand group required')
        if any(type(n) is not int or n < 0 for _,n in group):
            raise ValueError('Nonnegative page-rounded growth required')
        totals.append(sum(n for _,n in group))
    if proposed.prefill_layers == 0 and totals[0] != 0:
        raise ValueError('Zero prefill proposal cannot admit a new prefill cohort')
    prefill,decode=totals
    # The cached adder floor already includes input/output reservations. It is
    # another lower bound on TOTAL capacity, not another additive workload.
    required = max(allocated_kv_tokens+prefill+decode,admission_capacity_floor)+reserve_tokens
    return JointMemoryDemand(proposed,required,proposed.ann_batch,prefill,decode,
                             admission_capacity_floor)


def derive_cached_prefill_memory_demand(proposed, budget, cohort, *,
                                       decode_growth_tokens, reserve_tokens=0,
                                       published_capacity_tokens=None, allocated_kv_tokens=None,
                                       guaranteed_available_tokens=None):
    """Translate ONE queued compute cohort into a total-capacity demand.

    Do not call propose/select again here. Current published capacity and
    allocated tokens must belong to the same serialized budget snapshot.
    This is demand only: neither the capacity floor nor a cached prefix is a
    guarantee of physically ready backing.
    """
    if published_capacity_tokens is None:
        published_capacity_tokens = budget.published_capacity_tokens
    if allocated_kv_tokens is None:
        allocated_kv_tokens = budget.allocated_kv_tokens
    for value in (published_capacity_tokens,allocated_kv_tokens):
        if type(value) is not int or value < 0:
            raise ValueError('Observed nonnegative KV capacities required')
    if (allocated_kv_tokens > published_capacity_tokens
            or budget.available_and_evictable_tokens > published_capacity_tokens):
        raise ValueError('Inconsistent published capacity snapshot')
    ids = cohort.selection.request_ids
    selected = budget.shapes[:len(ids)]
    if not ids or tuple(s.request_id for s in selected) != ids:
        raise ValueError('Demand must retain the proposed FCFS cohort')
    contract = (0,tuple((s.request_id,s.prompt_tokens,s.context_tokens,s.extend_tokens)
                       for s in selected))
    if proposed.prefill_contract != contract or proposed.prefill_layers <= 0:
        raise ValueError('Compute lease does not describe the demanded prefill cohort')
    limits = dict(budget.max_new_tokens)
    protected = budget.cumulative_lock_tokens[len(ids)-1]
    inputs = sum(s.extend_tokens for s in selected)
    minimum = math.floor(budget.running_token_reserve + protected + inputs
        + sum(min(limits[rid],budget.clip_max_new_tokens) for rid in ids))+1
    extra = max(0,minimum-budget.available_and_evictable_tokens)
    if (cohort.minimum_available_and_evictable_tokens != minimum
            or cohort.additional_capacity_tokens != extra
            or cohort.selection.extend_tokens != inputs
            or cohort.selection.newly_protected_tokens != protected):
        raise ValueError('Cohort and captured admission demand disagree')
    admission_tokens = budget.available_and_evictable_tokens
    if guaranteed_available_tokens is not None:
        if (type(guaranteed_available_tokens) is not int
                or not 0 <= guaranteed_available_tokens <= published_capacity_tokens
                or guaranteed_available_tokens > budget.available_and_evictable_tokens):
            raise ValueError('Guaranteed admission capacity must be current physical free')
        admission_tokens = guaranteed_available_tokens
    return derive_joint_memory_demand(proposed, allocated_kv_tokens=allocated_kv_tokens,
        new_prefill_tokens=tuple((s.request_id,s.extend_tokens) for s in selected),
        decode_growth_tokens=decode_growth_tokens,reserve_tokens=reserve_tokens,
        # Keep the signed available-capacity difference here. Clamping it to
        # positive growth would incorrectly forbid shrinkage in a roomy pool.
        admission_capacity_floor=published_capacity_tokens
            -admission_tokens+minimum)


@dataclass(frozen=True)
class Snapshot:
    event_id: int
    requests: tuple[Request, ...]
    compute: ComputeLease
    memory: MemoryLease
    # Existing in-flight work occupies these lanes until the given offset.
    busy_until: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Credit:
    request: str
    work: str
    # Fraction of that SAME reference-work piece remaining at snapshot time.
    fraction: float = 1.0


@dataclass(frozen=True)
class Operation:
    key: str
    stage: str
    duration_ms: float | None
    lanes: tuple[str, ...]
    credits: tuple[Credit, ...] = ()
    depends_on: tuple[str, ...] = ()
    ready_ms: float = 0.0
    # Absolute completion offset of compute transition needed by THIS op.
    # Compatible decode should use zero rather than a global pause.
    compute_ready_ms: float = 0.0
    memory_keys: tuple[str, ...] = ()
    kv_key: str | None = None
    # Frozen service-query shape, never live runtime state. Pricing may only
    # fill duration_ms and must preserve this description with the work graph.
    service_shape: object | None = None


@dataclass(frozen=True)
class Projection:
    compute: ComputeLease
    memory: MemoryLease
    executable: ComputeLease
    operations: tuple[Operation, ...]
    supported: bool = True
    reason: str = ""
    decode_plan: JoiningDecodePlan | None = None


@dataclass(frozen=True)
class JoiningDecodePlan:
    """Fixed fitted-lease description; service lookup must be read-only.

    Runtime frontiers and profiles must already be detached/frozen. This is
    optional while the legacy static operation adapter is being replaced.
    Both ablations construct it; only Full evaluates its net progress.
    """
    frontiers: tuple
    profiles: tuple
    join_dependencies: tuple
    completed_tokens: tuple
    kv_keys: tuple
    query_service: object
    compute_ready_ms: float = 0.
    # (rid, (guaranteed additional iterations, growth backing key)).
    # Readiness is resolved from each counterfactual's memory lease, not cached.
    kv_growth_guarantees: tuple = ()


class PredictionUnavailable(ValueError):
    """Missing measured shape/backing, not an estimated zero-cost operation."""


@dataclass(frozen=True)
class DecodeIterationShape:
    request_ids: tuple[str, ...]
    context_tokens: tuple[int, ...]


def bind_frozen_decode_interference(profile, *, partition, overlap_class,
                                    workload_supported):
    """Adapt existing frozen complete-iteration histories, not slowdown deltas.

    Histories lack context/ANN-intensity buckets. The caller must provide an
    explicit workload-domain predicate backed by its deployment validation.
    A batch-table hit alone does not establish that domain. No live lookup or
    solo/other-partition fallback is permitted.
    """
    if (type(partition) is not int or partition<0
            or overlap_class not in ('ann','prefill','ann_prefill')
            or not callable(workload_supported)
            or not callable(getattr(profile,'query_batch_ms',None))):
        raise ValueError('Frozen coexecution profile and explicit workload domain required')
    if any(getattr(p,'timing_basis',None)!='submission_service'
           for p in getattr(profile,'profiles',())) or not getattr(profile,'profiles',()):
        raise ValueError('Progress prediction requires submission service, not completion intervals')
    def query(shape,solo_ms):
        if workload_supported(shape) is not True:
            return None
        value=profile.query_batch_ms(partition,overlap_class,len(shape.request_ids))
        if value is None:return None
        nonnegative(value,'measured complete coexecution iteration')
        return value if value>0 else None
    return query


def bind_observed_decode_compute_interference(profile, *, frontiers, partitions,
                                              workload_supported):
    """Use the same service histories/domain as Full pricing for observed compute.

    No invented slowdown coefficient: these are measured coexecution durations.
    The fixed observed cohort prevents batch-size-only context substitutions.
    """
    cohort = tuple(frontiers)
    if (not cohort or any(f.stage != 'decode' or type(f.prompt_tokens) is not int
            or f.prompt_tokens <= 0 or type(f.generated_tokens) is not int
            or f.generated_tokens < 1 for f in cohort)):
        raise ValueError('Observed decode cohort required')
    shape = DecodeIterationShape(tuple(f.request_id for f in cohort),
        tuple(f.prompt_tokens+f.generated_tokens for f in cohort))
    if len(set(shape.request_ids)) != len(cohort):
        raise ValueError('Unique observed decode requests required')
    cached = tuple(n-1 for n in shape.context_tokens)
    queries = {(p,o):bind_frozen_decode_interference(profile,partition=p,
        overlap_class=o,workload_supported=workload_supported)
        for p in tuple(partitions) for o in ('ann','prefill','ann_prefill')}
    def query(partition, overlap, cached_tokens, solo_ms):
        if tuple(cached_tokens) != cached:
            return None
        bound = queries.get((partition,overlap))
        return None if bound is None else bound(shape,solo_ms)
    return query


def bind_decode_stage_model(predictor, *, sm_count, expected_identity,
                            ann_coexecution, interference_query=None,
                            nearest_domain_fallback=False):
    """Bind one fixed-proposal iteration clock to the calibrated phase model.

    context_tokens includes this iteration's input token; calibration's cached
    lengths exclude it. A coexecution callback must return a measured/modelled
    TOTAL iteration time (not an additive slowdown), or None if unsupported.
    No implicit solo fallback for ANN sharing, no partition search, no capture
    of future arrivals. All inputs must already be frozen for this decision.
    The legacy ann_coexecution flag also enables a measured prefill-only
    interference query: spatially separated phases may still contend for HBM.
    """
    if (type(sm_count) is not int or sm_count<=0 or type(ann_coexecution) is not bool
            or type(nearest_domain_fallback) is not bool):
        raise ValueError('Actual SM count and explicit coexecution state required')
    if predictor.identity!=expected_identity:
        raise ValueError('Decode stage model deployment mismatch')
    if interference_query is not None and not callable(interference_query):
        raise ValueError('Frozen interference query must be callable')
    def query(shape):
        if (not isinstance(shape,DecodeIterationShape) or not shape.request_ids
                or len(set(shape.request_ids))!=len(shape.request_ids)
                or len(shape.request_ids)!=len(shape.context_tokens)
                or any(type(n) is not int or n<1 for n in shape.context_tokens)):
            raise ValueError('One positive full decode context per request required')
        solo=predictor.predict_decode_ms(sm_count,tuple(n-1 for n in shape.context_tokens),
            expected_identity=expected_identity)
        if solo is None and nearest_domain_fallback:
            solo=predictor.predict_decode_nearest_ms(
                sm_count,tuple(n-1 for n in shape.context_tokens),
                expected_identity=expected_identity)
        if solo is None:return None
        if not ann_coexecution:return solo
        if interference_query is None:return None
        duration=interference_query(shape,solo)
        if duration is None:return None
        nonnegative(duration,'coexecuted decode iteration')
        return duration if duration>0 else None
    return query


def allocate_decode_kv_prefixes(remaining_iterations, *, free_tokens,
                                reserved_tokens, page_size):
    """Distribute ONE observed pool in complete active-batch iterations.

    No prospective frees, cache evictions, or pending capacity count as free.
    Reservations include competing admitted prefill/new-cohort work. This
    conservative prefix does not simulate release after request completion.
    A partial final batch is not issued: prepare_for_decode allocates the
    entire batch before execution. Inputs are post-inflight remaining work.
    """
    entries = tuple(remaining_iterations)
    remaining = dict(entries)
    if len(remaining) != len(entries):
        raise ValueError('Unique decode members required')
    if page_size != 1 or type(page_size) is not int:
        raise ValueError('Observed unit-page allocator required')
    if any(type(v) is not int or v < 0 for v in
           (free_tokens, reserved_tokens, *remaining.values())):
        raise ValueError('Nonnegative integer physical token counts required')
    available = max(0, free_tokens-reserved_tokens)
    counts = dict.fromkeys(remaining, 0)
    # Skip runs of unchanged cohort size instead of looping over every token.
    while True:
        active = tuple(rid for rid in remaining if counts[rid] < remaining[rid])
        if not active:
            break
        steps = min(available//len(active),
                    min(remaining[rid]-counts[rid] for rid in active))
        if not steps:
            break
        for rid in active:
            counts[rid] += steps
        available -= steps*len(active)
    return tuple(counts.items())


def forecast_active_decode_kv_growth(frontiers, profiles, query_service, *,
                                     horizon_ms: float, initial_wait_ms: float,
                                     page_size: int, inflight_iterations: int = 0):
    """Forecast new slots for an observed active cohort under ONE proposal.

    Unit-page allocator only: real prepare_for_decode allocates one slot per
    participating request BEFORE each iteration. Thus an iteration starting
    before H needs capacity even if its output completes after H. Existing
    allocated/inflight iteration slots belong to the base occupancy, not here;
    initial_wait_ms must cover that known residual before issuing new work.
    inflight_iterations is 0 or 1 for this non-overlapped cohort and excludes
    its already allocated iteration from both remaining output and new slots.
    query_service is bound to the proposed partition/coexecution, not a global
    previous-plan estimator. No new cohort or unknown arrival is invented.
    Larger pages need actual per-request page occupancy, not output-count
    rounding; unsupported rather than silently under/over-reserving them.
    """
    if type(page_size) is not int or page_size != 1:
        raise ValueError('Decode growth requires observed unit-page allocation')
    if type(inflight_iterations) is not int or inflight_iterations not in (0,1):
        raise ValueError('Observed zero/one in-flight cohort iteration required')
    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError('Positive finite common horizon required')
    nonnegative(initial_wait_ms,'initial decode residual')
    frontiers=tuple(frontiers)
    if len({f.request_id for f in frontiers}) != len(frontiers):
        raise ValueError('Unique active decode requests required')
    remaining={}
    for f in frontiers:
        if (f.stage != 'decode' or f.generation_follows is not True
                or type(f.prompt_tokens) is not int or f.prompt_tokens <= 0):
            raise ValueError('Observed active decode cohort required')
        remaining[f.request_id]=remaining_decode_tokens(f,profiles[f.request_id])
    counts={f.request_id:0 for f in frontiers}
    elapsed=initial_wait_ms
    for step in range(inflight_iterations,max(remaining.values(),default=0)):
        if elapsed >= horizon_ms:
            break
        active=tuple(f for f in frontiers if remaining[f.request_id] > step)
        shape=DecodeIterationShape(tuple(f.request_id for f in active),
                    tuple(f.prompt_tokens+f.generated_tokens+step for f in active))
        duration=query_service(shape)
        if duration is None:
            return None
        nonnegative(duration,'proposed decode iteration service')
        if duration == 0:
            raise ValueError('Positive decode iteration service required')
        for f in active:
            counts[f.request_id]+=1
        elapsed+=duration
    return tuple(counts.items())


def bind_selected_decode_growth_forecast(frontiers, profiles, *, capacity_profile,
        allocation_by_stream, overlap_classes, horizon_ms, page_size,
        initial_wait_ms=0., inflight_iterations=0):
    """Freeze a selected-partition capacity callback, never a utility callback.

    overlap_classes lists the captured reachable coexecution states during the
    window. Use their fastest capacity estimate: ANN/prefill may finish before
    decode, and using only the contended rate could under-reserve later KV.
    Missing any reachable shape is unknown, not zero growth. These profiles
    remain estimates; runtime allocation guards are still mandatory.
    """
    frontiers=tuple(frontiers);profiles=dict(profiles)
    allocations=dict(allocation_by_stream);overlaps=tuple(overlap_classes)
    if (not overlaps or len(set(overlaps))!=len(overlaps)
            or not set(overlaps)<= {'solo','ann','prefill','ann_prefill'}):
        raise ValueError('Explicit unique reachable decode overlap classes required')
    if not callable(getattr(capacity_profile,'query_capacity_ms',None)):
        raise ValueError('Dedicated frozen capacity profile required')
    def forecast(snapshot, proposed, diagnostic):
        if not {f.request_id for f in frontiers}<={r.key for r in snapshot.requests}:
            raise ValueError('Capacity cohort must belong to the common observed request set')
        matches=[idx for idx,allocation in allocations.items() if allocation==proposed.allocation]
        if len(matches)!=1:
            raise ValueError('Selected allocation must identify a unique captured partition')
        partition=matches[0]
        def query(shape):
            values=[capacity_profile.query_capacity_ms(partition,overlap,len(shape.request_ids))
                    for overlap in overlaps]
            if any(v is None for v in values):
                return None
            if any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Positive finite capacity service estimates required')
            return min(values)
        counts=forecast_active_decode_kv_growth(frontiers,profiles,query,
            horizon_ms=horizon_ms,initial_wait_ms=initial_wait_ms,page_size=page_size,
            inflight_iterations=inflight_iterations)
        return counts  # Preserve per-request demand through the native bridge.
    return forecast


def build_active_decode_operations(frontiers, profiles, query_service, *,
                                   kv_keys: dict[str, str], lane: str = 'decode',
                                   prefix: str = 'decode', depends_on: tuple[str, ...] = (),
                                   compute_ready_ms: float = 0.0,
                                   inflight_remaining_ms: float | None = None,
                                   inflight_kv_keys: dict[str, str] | None = None,
                                   timing=True, inflight_observed=False,
                                   inflight_observation_offset_ms=0.):
    """One operation per BATCH iteration, not one kernel per request.

    Only an already-active decode cohort belongs here; newly merged prefill
    requires the combined runtime event adapter. query_service(shape) must use
    the selected fixed partition and co-execution profile, returning measured
    iteration ms or None. Missing a later shrinking-batch profile invalidates
    the plan rather than dropping those requests or inserting zero-time work.
    One already-submitted iteration may use an explicit residual and existing
    backing keys. Do not also charge its residual through snapshot.busy_until.
    Only subsequent iterations wait for the proposed compute transition.
    """
    frontiers = tuple(frontiers)
    ids = tuple(f.request_id for f in frontiers)
    if len(ids) != len(set(ids)) or set(ids) != set(kv_keys):
        raise ValueError('Decode cohort and KV identities must match')
    if not lane or not prefix or any(not value for value in kv_keys.values()):
        raise ValueError('Explicit decode lane and backing identities required')
    nonnegative(compute_ready_ms, 'decode compute readiness')
    nonnegative(inflight_observation_offset_ms, 'decode observation offset')
    has_inflight = inflight_observed or inflight_remaining_ms is not None
    if has_inflight:
        if timing and inflight_remaining_ms is None:
            return None
        if inflight_remaining_ms is not None:
            nonnegative(inflight_remaining_ms, 'observed decode residual')
        if (inflight_kv_keys is None or set(inflight_kv_keys) != set(ids)
                or any(not value for value in inflight_kv_keys.values())):
            raise ValueError('In-flight iteration needs explicit existing KV backing')
    elif inflight_kv_keys is not None:
        raise ValueError('In-flight backing without an observed iteration')
    remaining = {}
    for f in frontiers:
        if f.stage != 'decode' or f.generation_follows is not True:
            raise ValueError('Only active decode frontiers are supported here')
        if type(f.prompt_tokens) is not int or f.prompt_tokens <= 0:
            raise ValueError('Observed decode context required')
        remaining[f.request_id] = remaining_decode_tokens(f, profiles[f.request_id])
    if has_inflight and (not ids or any(n <= 0 for n in remaining.values())):
        raise ValueError('In-flight cohort members need an uncompleted token')
    ops = []
    predecessors = depends_on
    for step in range(max(remaining.values(), default=0)):
        active = tuple(f for f in frontiers if remaining[f.request_id] > step)
        shape = DecodeIterationShape(tuple(f.request_id for f in active),
                    tuple(f.prompt_tokens + f.generated_tokens + step for f in active))
        inflight = step == 0 and has_inflight
        duration = (inflight_remaining_ms if inflight else query_service(shape)) if timing else None
        if timing and duration is None:
            return None
        if timing:
            nonnegative(duration, 'profiled decode iteration')
        if duration == 0 and not inflight:
            raise ValueError('Decode profile must have positive service time')
        key = f'{prefix}:{step}'
        ops.append(Operation(key, 'decode', duration, (lane,),
                    tuple(Credit(f.request_id, 'decode', (step+1)/remaining[f.request_id]) for f in active),
                    depends_on=() if inflight else predecessors,
                    ready_ms=inflight_observation_offset_ms if inflight else 0.,
                    compute_ready_ms=0.0 if inflight else compute_ready_ms,
                    memory_keys=tuple(dict.fromkeys(
                        (inflight_kv_keys if inflight else kv_keys)[f.request_id]
                        for f in active)),
                    service_shape=('decode_inflight',shape) if inflight else shape))
        predecessors = tuple(dict.fromkeys((key, *depends_on))) if inflight else (key,)
    return tuple(ops)


def time_frozen_decode_operations(operations, query_service, *, query_inflight_remaining=None):
    """Time a fixed active-cohort iteration chain without rebatching or clocks."""
    result = []
    for op in operations:
        if op.stage != 'decode':
            raise ValueError('Expected fixed decode operations')
        shape = op.service_shape
        if isinstance(shape, DecodeIterationShape):
            duration = query_service(shape)
        elif (isinstance(shape,tuple) and len(shape)==2 and shape[0]=='decode_inflight'
              and isinstance(shape[1],DecodeIterationShape)):
            if query_inflight_remaining is None:
                return None
            duration = query_inflight_remaining(shape[1])
        else:
            raise ValueError('Captured decode service shape required')
        if duration is None:
            return None
        nonnegative(duration,'frozen decode duration')
        if duration == 0:
            return None
        result.append(replace(op,duration_ms=duration))
    return tuple(result)


@dataclass(frozen=True)
class DecodeAdmission:
    request_id: str
    # Same current-window clock; includes prefill/result/merge AND KV readiness.
    ready_ms: float
    depends_on: tuple[str, ...] = ()
    # Result processing may already emit the first token before joining decode.
    completed_tokens: int = 0


def build_joining_decode_operations(frontiers, profiles, admissions, query_service, *,
                                    kv_keys, initial_wait_ms=0., prefix='joined_decode',
                                    kv_growth_windows=None, horizon_ms=None,
                                    submission_blackouts=(), unavailable_reasons=None):
    """Fixed-proposal cohort evolution for already-observed requests only.

    Admissions are predictions from the SAME proposal's upstream event trace,
    including backing/result/merge gates, not future request arrival times.
    A join happens at an iteration boundary; a waiting prefill never stalls
    an existing compatible cohort. Callers must regenerate admissions for each
    transition counterfactual, rather than reuse target offsets for keep/G0.
    No currently in-flight iteration belongs here: its residual, completed
    token and existing backing must first be handled by the upstream adapter.
    This returns decode ops only. completed_tokens credits belong to upstream
    result processing; subsequent credits remain relative to snapshot work.
    Missing any evolving batch profile invalidates the whole prediction.
    When horizon_ms is supplied, query only iterations starting inside that
    common window (retain an iteration crossing its end). This is a bounded
    progress forecast, not a complete output trace or changed work denominator.
    """
    # Optional rid -> (guaranteed additional iterations, growth backing key,
    # absolute growth-ready offset). Guarantees must come from actual shared
    # capacity planning, not per-request copies of the same free-token pool.
    # An upstream first token does not consume an additional decode iteration.
    kv_growth_windows = dict(kv_growth_windows or {})
    frontiers = tuple(frontiers)
    admissions = tuple(admissions)
    by_id = {f.request_id: f for f in frontiers}
    gates = {a.request_id: a for a in admissions}
    if (len(by_id) != len(frontiers) or len(gates) != len(admissions)
            or set(by_id) != set(gates) or set(by_id) != set(kv_keys)):
        raise ValueError('One explicit admission and KV key per observed request required')
    if not prefix or any(not key for key in kv_keys.values()):
        raise ValueError('Explicit operation and backing identities required')
    if not set(kv_growth_windows) <= set(by_id):
        raise ValueError('KV growth window for unknown request')
    for count, key, ready in kv_growth_windows.values():
        if type(count) is not int or count < 0 or not isinstance(key,str) or not key:
            raise ValueError('Explicit finite-prefix KV guarantee required')
        if math.isnan(ready) or ready < 0:
            raise ValueError('Invalid KV growth readiness')
    nonnegative(initial_wait_ms, 'existing decode wait')
    blocks=[]
    for start,end in sorted(tuple(submission_blackouts)):
        if (not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start):
            raise ValueError('Finite positive host submission blackout required')
        if blocks and start<=blocks[-1][1]:
            blocks[-1]=(blocks[-1][0],max(blocks[-1][1],end))
        else:blocks.append((start,end))
    def after_host_wait(at):
        for start,end in blocks:
            if start<=at<end:at=end
            elif start>at:break
        return at
    if horizon_ms is not None and (type(horizon_ms) not in (int,float)
            or not math.isfinite(horizon_ms) or horizon_ms<=0):
        raise ValueError('Positive finite decode prediction horizon required')
    totals = {}; done = {}
    for rid, f in by_id.items():
        if (f.stage not in ('decode', 'prefill_active', 'prefill_queued',
                            'prefill_merge_pending') or f.generation_follows is not True
                or type(f.prompt_tokens) is not int or f.prompt_tokens <= 0):
            raise ValueError('Observed generation frontier with known prompt required')
        a = gates[rid]
        nonnegative(a.ready_ms, 'proposal admission readiness')
        totals[rid] = remaining_decode_tokens(f, profiles[rid])
        if (type(a.completed_tokens) is not int or a.completed_tokens not in (0, 1)
                or a.completed_tokens > totals[rid]
                or (f.stage == 'decode' and a.completed_tokens)):
            raise ValueError('Only an explicit upstream first-token completion is supported')
        done[rid] = a.completed_tokens
    elapsed = initial_wait_ms
    operations = []; previous = ()
    def next_backing(rid):
        window = kv_growth_windows.get(rid)
        if window is not None and done[rid]-gates[rid].completed_tokens >= window[0]:
            return window[1], max(gates[rid].ready_ms, window[2])
        return kv_keys[rid], gates[rid].ready_ms
    while any(done[rid] < totals[rid] for rid in by_id):
        elapsed=after_host_wait(elapsed)
        if horizon_ms is not None and elapsed >= horizon_ms:
            break
        pending = tuple(rid for rid in by_id if done[rid] < totals[rid])
        backing = {rid:next_backing(rid) for rid in pending}
        active = tuple(rid for rid in pending if backing[rid][1] <= elapsed)
        if not active:
            elapsed = after_host_wait(max(elapsed, min(backing[rid][1] for rid in pending)))
            active = tuple(rid for rid in pending if backing[rid][1] <= elapsed)
        if not math.isfinite(elapsed):
            break  # No remaining backing arrives in this fixed proposal.
        if horizon_ms is not None and elapsed >= horizon_ms:
            break  # Next admission is outside H; no service query or credit.
        shape = DecodeIterationShape(active, tuple(
            by_id[rid].prompt_tokens + by_id[rid].generated_tokens + done[rid]
            for rid in active))
        duration = query_service(shape)
        if duration is None:
            if unavailable_reasons is not None:
                unavailable_reasons.append(
                    'decode_service(batch=%d,contexts=%r)' %
                    (len(shape.request_ids),shape.context_tokens))
            return None
        nonnegative(duration, 'joined decode iteration service')
        if duration == 0:
            raise ValueError('Positive joined decode service required')
        for rid in active:
            done[rid] += 1
        key = f'{prefix}:{len(operations)}'
        dependencies = tuple(dict.fromkeys((*previous,
            *(dep for rid in active for dep in gates[rid].depends_on))))
        operations.append(Operation(key, 'decode', duration, ('decode',),
            tuple(Credit(rid, 'decode', done[rid]/totals[rid]) for rid in active),
            depends_on=dependencies, ready_ms=elapsed,
            memory_keys=tuple(dict.fromkeys(backing[rid][0] for rid in active)),
            service_shape=shape))
        elapsed += duration
        previous = (key,)
    return tuple(operations)


@dataclass(frozen=True)
class PrefillBundleShape:
    request_ids: tuple[str, ...]
    prompt_tokens: tuple[int, ...]
    start_layer: int
    end_layer: int
    context_tokens: tuple[int, ...]
    extend_tokens: tuple[int, ...]


@dataclass(frozen=True)
class ProspectivePrefillBundleShape(PrefillBundleShape):
    # Retain provenance when using the same service lookup as observed shapes.
    estimated: bool = True


def build_prefill_bundle_operations(frontiers, query_service, *, total_layers: int,
                                    fitted_layers: int, kv_keys: dict[str, str],
                                    lane: str = 'prefill', prefix: str = 'prefill',
                                    depends_on: tuple[str, ...] = (),
                                    compute_ready_ms: float = 0.0,
                                    inflight_remaining_ms: float | None = None, timing=True,
                                    inflight_observation_offset_ms: float = 0.0):
    """Construct one actual batch's sequential bundles under the fitted cap.

    The caller provides the runtime batch membership, not the entire waiting
    queue. All members must be at one completed layer endpoint. Observed in-flight
    layers become one residual operation, not new submissions. Do not also add
    their duration to snapshot.busy_until (that would double-count waiting).
    Their KV keys must refer to existing backing, not future allocation promises.
    The profile callback
    receives the exact resulting layer interval, including the final short tail.
    """
    if type(total_layers) is not int or total_layers <= 0:
        raise ValueError('Positive model layer count required')
    if type(fitted_layers) is not int or fitted_layers < 0:
        raise ValueError('Nonnegative fitted prefill limit required')
    nonnegative(compute_ready_ms, 'prefill compute readiness')
    nonnegative(inflight_observation_offset_ms, 'in-flight prefill observation offset')
    frontiers = tuple(frontiers)
    ids = tuple(f.request_id for f in frontiers)
    if len(ids) != len(set(ids)) or set(ids) != set(kv_keys):
        raise ValueError('Prefill cohort and KV identities must match')
    if not lane or not prefix or any(not key for key in kv_keys.values()):
        raise ValueError('Explicit prefill lane and KV identities required')
    endpoints = set()
    inflight_intervals = set()
    for f in frontiers:
        if f.stage not in ('prefill_queued','prefill_active') or f.generation_follows is not True:
            raise ValueError('Expected one runtime prefill batch')
        if type(f.prompt_tokens) is not int or f.prompt_tokens <= 0:
            raise ValueError('Observed prefill prompt shape required')
        context = getattr(f, 'prefill_context_tokens', None)
        extend = getattr(f, 'prefill_extend_tokens', None)
        if (type(context) is not int or type(extend) is not int
                or not 0 < extend <= context):
            raise ValueError('Prepared prefill context/uncached shape required')
        endpoint = f.completed_prefill_layers
        if type(endpoint) is not int or not 0 <= endpoint <= total_layers:
            raise ValueError('Observed completed prefill endpoint required')
        endpoints.add(endpoint)
        inflight_intervals.add((getattr(f,'inflight_prefill_start_layer',None),
                               getattr(f,'inflight_prefill_end_layer',None)))
    if len(endpoints) > 1:
        raise ValueError('Cannot batch different completed layer endpoints')
    if len(inflight_intervals) > 1:
        raise ValueError('Batch members disagree on in-flight layer interval')
    if not frontiers:
        return ()
    initial = next(iter(endpoints))
    remaining = total_layers-initial
    keys = tuple(dict.fromkeys(kv_keys[rid] for rid in ids))
    ops = []
    predecessors = depends_on
    next_layer = initial
    in_start, in_end = next(iter(inflight_intervals))
    if (in_start, in_end) != (None,None):
        if in_start != initial or type(in_end) is not int or not initial < in_end <= total_layers:
            raise ValueError('In-flight layers must start at the observed completion endpoint')
        if timing and inflight_remaining_ms is None:
            return None
        if timing:
            nonnegative(inflight_remaining_ms, 'remaining in-flight service')
        if timing and inflight_remaining_ms == 0:
            return None  # An unobserved completion cannot produce free progress.
        key = f'{prefix}:inflight:{initial}:{in_end}'
        inflight_shape = PrefillBundleShape(ids, tuple(f.prompt_tokens for f in frontiers), initial, in_end,
                    tuple(f.prefill_context_tokens for f in frontiers),
                    tuple(f.prefill_extend_tokens for f in frontiers))
        ops.append(Operation(key,'prefill',inflight_remaining_ms if timing else None,(lane,),
                    tuple(Credit(rid,'prefill',(in_end-initial)/remaining) for rid in ids),
                    depends_on=predecessors,kv_key=keys[0],memory_keys=keys,
                    ready_ms=inflight_observation_offset_ms,
                    service_shape=('prefill_inflight',inflight_shape)))
        predecessors = (key,)
        next_layer = in_end
    if fitted_layers == 0:
        return tuple(ops)  # Existing submitted work continues even at zero new credit.
    for start in range(next_layer, total_layers, fitted_layers):
        end = min(total_layers, start+fitted_layers)
        shape = PrefillBundleShape(ids, tuple(f.prompt_tokens for f in frontiers), start, end,
                    tuple(f.prefill_context_tokens for f in frontiers),
                    tuple(f.prefill_extend_tokens for f in frontiers))
        duration = query_service(shape) if timing else None
        if timing and duration is None:
            return None
        if timing:
            nonnegative(duration, 'profiled prefill bundle')
        if timing and duration == 0:
            raise ValueError('Prefill profile must have positive service time')
        key = f'{prefix}:{start}:{end}'
        ops.append(Operation(key, 'prefill', duration, (lane,),
                    tuple(Credit(rid,'prefill',(end-initial)/remaining) for rid in ids),
                    depends_on=predecessors, compute_ready_ms=compute_ready_ms,
                    kv_key=keys[0], memory_keys=keys, service_shape=shape))
        predecessors = (key,)
    return tuple(ops)


def time_frozen_prefill_operations(operations, query_service, *, query_inflight_remaining=None):
    """Fill exact frozen layer intervals, with a separate in-flight estimator.

    The in-flight callback must use a captured execution age/profile; never
    charge a complete bundle again or infer a free completion. No regrouping.
    """
    result = []
    for op in operations:
        if op.stage != 'prefill':
            raise ValueError('Expected fixed prefill operations')
        shape = op.service_shape
        if isinstance(shape, PrefillBundleShape):
            duration = query_service(shape)
        elif (isinstance(shape, tuple) and len(shape) == 2 and shape[0] == 'prefill_inflight'
              and isinstance(shape[1], PrefillBundleShape)):
            if query_inflight_remaining is None:
                return None
            duration = query_inflight_remaining(shape[1])
        else:
            raise ValueError('Captured prefill service shape required')
        if duration is None:
            return None
        nonnegative(duration, 'fixed prefill duration')
        if duration == 0:
            return None
        result.append(replace(op, duration_ms=duration))
    return tuple(result)


@dataclass(frozen=True)
class ColdWorkerShape:
    # Entries retain Future identity, rows, cold-list width and enqueue clock.
    entries: tuple[tuple, ...]
    active: bool
    elapsed_ms: float


def build_cold_worker_operations(worker_snapshot, query_remaining_service, *,
                                 lane='cpu_retrieval', prefix='cold',
                                 observation_offset_ms: float = 0.0, timing=True):
    """Predict the observed single worker without adding unknown arrivals.

    Callback estimates REMAINING active wall service, full queued service.
    Never subtract elapsed time again or occupy this lane via busy_until too.
    Queue grouping models the worker's FIFO greedy cap (an oversized head is
    executed intact). Worker assembly/cancellation races remain prediction
    uncertainty: this is not a reservation of the future native Faiss batches.
    Missing service returns None for the whole plan, not a free CPU branch.
    Returns operations and a (Future identity, enqueue clock)->completion key
    map, so a merged retrieval result can depend on every involved CPU batch.
    """
    nonnegative(observation_offset_ms, 'CPU observation offset')
    cap = worker_snapshot['max_batch_queries']
    if type(cap) is not int or cap <= 0:
        raise ValueError('Positive cold worker batch cap required')
    active = tuple(tuple(entry) for entry in worker_snapshot.get('active', ()))
    queued = tuple(tuple(entry) for entry in worker_snapshot.get('queued', ()))
    identities = [(e[0], e[3]) for e in (*active, *queued)]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate cold worker observation')
    if any(type(e[1]) is not int or e[1] <= 0 for e in (*active, *queued)):
        raise ValueError('Positive cold worker row counts required')
    groups = []
    if active:
        started = worker_snapshot.get('active_started_at')
        if started is None:
            return None
        elapsed = (worker_snapshot['captured_at'] - started) * 1000
        nonnegative(elapsed, 'active CPU elapsed time')
        groups.append(ColdWorkerShape(active, True, elapsed))
    batch = []; rows = 0
    for entry in queued:
        if batch and rows + entry[1] > cap:
            groups.append(ColdWorkerShape(tuple(batch), False, 0.0))
            batch = []; rows = 0
        batch.append(entry); rows += entry[1]
    if batch:
        groups.append(ColdWorkerShape(tuple(batch), False, 0.0))
    operations = []; completion = {}; previous = ()
    for i, shape in enumerate(groups):
        duration = query_remaining_service(shape) if timing else None
        if timing:
            if duration is None:
                return None
            nonnegative(duration, 'remaining CPU worker service')
            if duration == 0:
                raise ValueError('Observed unfinished CPU work requires positive service')
        key = f'{prefix}:{i}'
        operations.append(Operation(key, 'cpu_retrieval', duration, (lane,),
                                    depends_on=previous,
                                    ready_ms=observation_offset_ms, service_shape=shape))
        previous = (key,)
        for entry in shape.entries:
            completion[(entry[0], entry[3])] = key
    return tuple(operations), completion


def time_frozen_cold_operations(operations, frozen_profile):
    """Fill CPU durations from captured shapes without observing the worker.

    Active age is the ORIGINAL observation's age. Do not subtract a second
    elapsed interval or rebuild groups from today's queue. Sparse remaining
    samples make this graph unscored, never a zero-duration search.
    """
    result = []
    for op in operations:
        if op.stage != 'cpu_retrieval' or not isinstance(op.service_shape, ColdWorkerShape):
            raise ValueError('Captured CPU worker shape required')
        duration = frozen_profile.query_remaining_ms(op.service_shape)
        if duration is None:
            return None
        nonnegative(duration, 'frozen CPU remaining duration')
        if duration == 0:
            raise ValueError('Unfinished CPU work requires positive duration')
        result.append(replace(op, duration_ms=duration))
    return tuple(result)


def time_frozen_retrieval_operations(operations, *, cpu_profile, gpu_profile, collection_profile):
    """Price the original retrieval graph using only its captured service shapes.

    No live routing, Future/worker observation, memory fitting or regrouping.
    All three profiles must come from the same declared decision/workload
    domain. A missing sample invalidates timing, not the safety graph.
    """
    result = []
    for op in operations:
        shape = op.service_shape
        if isinstance(shape, ColdWorkerShape) and op.stage == 'cpu_retrieval':
            if cpu_profile is None:
                return None
            measured = time_frozen_cold_operations((op,), cpu_profile)
            if measured is None:
                return None
            result.extend(measured)
            continue
        if not isinstance(shape, tuple) or not shape:
            raise ValueError('Captured retrieval service shape required')
        if shape[0] == 'ann_gpu' and len(shape) == 3 and op.stage == 'ann':
            if gpu_profile is None:
                return None
            duration = gpu_profile.query_ms(shape[1], shape[2])
        elif shape[0] == 'ann_gpu_linear' and len(shape) == 8 and op.stage == 'ann':
            query=getattr(gpu_profile,'query_work_ms',None)
            if not callable(query):
                return None
            duration=query(*shape[1:])
        elif shape[0] == 'ann_collect' and len(shape) == 6 and op.stage == 'collect':
            if collection_profile is None:
                return None
            duration = collection_profile.query_ms(*shape[1:])
        else:
            raise ValueError('Retrieval stage and captured shape disagree')
        if duration is None:
            return None
        nonnegative(duration, 'frozen retrieval service duration')
        if duration == 0:
            raise ValueError('Existing retrieval operation requires positive service')
        result.append(replace(op, duration_ms=duration))
    return tuple(result)


def build_additional_cold_operations(entries, query_remaining_service, *, max_batch_queries,
                                     admit_offset_ms, preceding_operations=(), prefix='spill', timing=True):
    """Forecast newly admitted CPU searches behind already-observed work.

    Each entry is (unique key, actual submitted rows, padded cold-list width).
    Zero-spill entries must be omitted. Never include original prelaunched cold
    searches again. New work is grouped with the worker's greedy row cap; a
    possible cross-boundary merge with an older queued batch is not modeled.
    This is a fixed-proposal approximation, not an oracle for worker races.
    No logical request progress is credited until the retrieval join.
    """
    nonnegative(admit_offset_ms, 'additional CPU admission offset')
    entries = tuple(entries)
    if any(len(e) != 3 or not isinstance(e[0], str) or not e[0]
           or type(e[1]) is not int or e[1] <= 0 or type(e[2]) is not int or e[2] <= 0
           for e in entries):
        raise ValueError('Exact positive new CPU submission shapes required')
    if len({e[0] for e in entries}) != len(entries):
        raise ValueError('Unique additional CPU submission keys required')
    if any(op.stage != 'cpu_retrieval' or op.lanes != ('cpu_retrieval',)
           for op in preceding_operations):
        raise ValueError('Only preceding CPU worker operations allowed')
    observation = dict(max_batch_queries=max_batch_queries, active=(),
        queued=tuple((key, rows, width, 0.) for key, rows, width in entries))
    result = build_cold_worker_operations(observation, query_remaining_service,
        prefix=prefix, observation_offset_ms=admit_offset_ms, timing=timing)
    if result is None:
        return None
    operations, completion = result
    old_keys = {op.key for op in preceding_operations}
    if any(op.key in old_keys for op in operations):
        raise ValueError('Additional CPU operation keys collide with observed work')
    if operations and preceding_operations:
        operations = (replace(operations[0], depends_on=(preceding_operations[-1].key,)),
                      *operations[1:])
    return tuple(operations), {key: completion[(key, 0.)] for key, _, _ in entries}


def retrieval_collection_after_shared_work(request_ids, *, branch_keys, shared_operations, collect):
    """Return one collection op, without duplicating shared CPU worker batches.

    Multiple queries may depend on the same batched CPU operation. Include that
    operation once in the global graph, and each result's collection once.
    Caller supplies all unfinished original-cold, spill and GPU dependencies;
    already-completed branches must be omitted using real completion evidence.
    """
    operations = {op.key: op for op in shared_operations}
    if len(operations) != len(shared_operations):
        raise ValueError('Shared operation identities must be unique')
    keys = tuple(dict.fromkeys(branch_keys))
    if any(key not in operations for key in keys):
        raise ValueError('Missing retrieval branch operation')
    if collect.key in operations:
        raise ValueError('Collection already present in shared graph')
    branches = tuple(operations[key] for key in keys)
    return complete_retrieval_batch(tuple(request_ids), branches=branches, collect=collect)[-1]


def complete_retrieval_batch(request_ids: tuple[str, ...], *,
                             branches: tuple[Operation, ...],
                             collect: Operation) -> tuple[Operation, ...]:
    """Join a real hybrid result batch and credit each logical query once.

    Mirrors HotAnnResult's CPU/GPU completion join and subsequent collect.
    Branches describe remaining observed work (including worker queue/lane and
    GPU backing gates); an already-complete/empty side is omitted explicitly.
    Collection includes measured transfer/top-k merge service, not application
    input preparation. It must wait for ALL branch operations. No CPU/GPU
    duration is converted into extra logical reference work.
    This constructs dependencies only, not CPU placement, profiles or batching.
    """
    if (not request_ids or len(set(request_ids)) != len(request_ids)
            or any(not isinstance(rid, str) or not rid for rid in request_ids)):
        raise ValueError('Unique observed retrieval request identities required')
    operations = (*branches, collect)
    if len({op.key for op in operations}) != len(operations):
        raise ValueError('Duplicate retrieval operation identity')
    for op in operations:
        if op.credits:
            raise ValueError('Logical ANN progress belongs only to completed collection')
        if op.duration_ms is not None:
            nonnegative(op.duration_ms, 'remaining retrieval service')
        if op.duration_ms == 0:
            raise ValueError('Omit complete branches, do not fabricate zero-time service')
    deps = tuple(dict.fromkeys((*collect.depends_on, *(op.key for op in branches))))
    return (*branches, replace(collect, depends_on=deps,
                               credits=tuple(Credit(rid, 'ann') for rid in request_ids)))


def build_prefill_decode_handoff(frontiers, profiles, *, prefill_operations=(),
                                 result_dependencies=(), merge_dependencies=(),
                                 existing_kv_key, new_kv_key, prefix='prefill-handoff',
                                 observation_offset_ms=0., separate_commit_gate=False):
    """Timing-free result/merge graph for one captured prefill cohort.

    The serving path processes the prefill result (emitting token one) before
    merging into running decode. Already merge-pending requests have processed
    that token in their frontier; never credit it a second time. Caller must
    supply actual old-decode/result ordering dependencies, not assume immediate
    joining. No service query, future generation length or scheduler search.
    Returns additional operations, per-request join gates and token offsets.
    """
    frontiers = tuple(frontiers)
    nonnegative(observation_offset_ms, 'handoff observation offset')
    if not frontiers:
        return (), (), ()
    ids = tuple(f.request_id for f in frontiers)
    if len(set(ids)) != len(ids):
        raise ValueError('Unique handoff cohort required')
    stages = {f.stage for f in frontiers}
    pending = stages == {'prefill_merge_pending'}
    if not pending and not stages <= {'prefill_queued', 'prefill_active'}:
        raise ValueError('One prefill or already-processed merge cohort required')
    if any(f.generation_follows is not True for f in frontiers):
        raise ValueError('Handoff requires observed generation work')
    ops = []; emitted = []
    if pending:
        if prefill_operations:
            raise ValueError('Merge-pending cohort cannot execute prefill again')
        dependencies = tuple(merge_dependencies)
        kv_key = existing_kv_key
        emitted = [(rid, 0) for rid in ids]
    else:
        if not prefill_operations:
            return None  # A held partial prefill cannot join decode yet.
        last = prefill_operations[-1]
        completed = {c.request for c in last.credits if c.work == 'prefill' and c.fraction == 1.}
        if completed != set(ids):
            return None
        if last.stage != 'prefill':
            raise ValueError('Expected the actual final prefill bundle')
        credits = []
        for f in frontiers:
            remaining = remaining_decode_tokens(f, profiles[f.request_id])
            if remaining <= 0:
                return None
            credits.append(Credit(f.request_id, 'decode', 1. / remaining))
            emitted.append((f.request_id, 1))
        key = prefix + ':result'
        kv_key = last.kv_key
        if kv_key not in (existing_kv_key, new_kv_key):
            raise ValueError('Prefill handoff lost captured KV ownership')
        ops.append(Operation(key, 'prefill_result', None, ('host_result',), tuple(credits),
            depends_on=tuple(dict.fromkeys((last.key, *result_dependencies))), kv_key=kv_key,
            ready_ms=observation_offset_ms,
            service_shape=('prefill_result', ids, tuple(f.prompt_tokens for f in frontiers))))
        dependencies = tuple(dict.fromkeys((key, *merge_dependencies)))
    merge_key = prefix + ':merge'
    ops.append(Operation(merge_key, 'prefill_merge', None, ('host_result',),
        depends_on=dependencies, kv_key=kv_key, ready_ms=observation_offset_ms,
        service_shape=('prefill_merge', ids, pending)))
    join_key = merge_key
    if separate_commit_gate:
        join_key = prefix + ':commit'
        # Conservatively retain the prefill stream until observed release;
        # this is not a host-thread pause. Deferred merge may finish on the
        # decode stream, but must not permit unsafe same-stream overtaking.
        ops.append(Operation(join_key, 'prefill_commit', None, ('prefill',),
            depends_on=(merge_key,), kv_key=kv_key, ready_ms=observation_offset_ms,
            service_shape=('prefill_commit', ids, merge_key)))
    return tuple(ops), tuple((rid, (join_key,)) for rid in ids), tuple(emitted)


def time_frozen_prefill_handoff(operations, query_paired_sample):
    """Price host stages and a separate observed commit tail from one row.

    The callback is bound to frozen endpoint shapes, including actual proposed
    partitions and running counts. Historical inter-stage waiting is excluded:
    dependencies in this graph determine it. The commit tail is an observed
    release prior, not pure device time; it gates decode but occupies no host
    lane. This estimator requires validation within the profiled envelope.
    """
    operations = tuple(operations)
    by_key = {op.key: op for op in operations}
    if len(by_key) != len(operations):
        raise ValueError('Unique handoff operation keys required')
    measured = {}
    for commit in operations:
        if commit.stage != 'prefill_commit':
            continue
        shape = commit.service_shape
        if (not isinstance(shape, tuple) or len(shape) != 3 or shape[0] != 'prefill_commit'
                or commit.lanes != ('prefill',) or commit.credits or commit.depends_on != (shape[2],)):
            raise ValueError('Explicit non-host commit gate required')
        merge = by_key.get(shape[2])
        if (merge is None or merge.stage != 'prefill_merge'
                or merge.service_shape[:2] != ('prefill_merge', shape[1])):
            raise ValueError('Commit gate lost its merge cohort')
        results = tuple(by_key[k] for k in merge.depends_on
                        if k in by_key and by_key[k].stage == 'prefill_result')
        if len(results) > 1 or (not results and not merge.service_shape[2]):
            raise ValueError('Merge needs its single result or captured pending state')
        result = results[0] if results else None
        if result is not None and result.service_shape[:2] != ('prefill_result', shape[1]):
            raise ValueError('Result and merge cohorts differ')
        sample = query_paired_sample(result, merge)
        if sample is None:
            return None
        if (result is not None and tuple(result.service_shape[2]) != sample.shape[0][2]):
            raise ValueError('Handoff sample changed prompt shape')
        for op, duration in ((result, sample.result_host_ms), (merge, sample.merge_host_ms),
                             (commit, sample.observed_tail_ms)):
            if op is None:
                continue
            if op.key in measured:
                raise ValueError('Handoff operation reused by multiple commits')
            nonnegative(duration, 'handoff duration')
            if duration == 0 and op.credits:
                return None
            measured[op.key] = replace(op, duration_ms=duration)
    if set(measured) != set(by_key):
        raise ValueError('Every handoff needs a separate commit gate')
    return tuple(measured[op.key] for op in operations)


def build_observed_input_preparation(frontiers, *, observed_at, window_started_at):
    """Untimed external ANN-return to tokenized-ingress work already in flight.

    No invented single CPU lane: external concurrent request pipelines have
    their own observed residual profiles. This shape is an age-conditioned
    lookup contract, not the whole elapsed latency or a zero-cost fallback.
    Future retrieval completions require their own explicitly linked graph.
    """
    nonnegative(observed_at,'input preparation observation')
    nonnegative(window_started_at,'prediction window origin')
    if observed_at < window_started_at:
        raise ValueError('Observation precedes common window')
    operations=[];seen=set()
    for f in frontiers:
        if f.stage != 'input_preparation':
            continue
        if f.request_id in seen:
            raise ValueError('Duplicate input preparation frontier')
        seen.add(f.request_id)
        started=f.preparation_started_at
        if (started is None or type(f.query_rows) is not int or f.query_rows<=0
                or f.preparation_follows is not True):
            return None
        nonnegative(started,'observed preparation start')
        if started > observed_at:
            raise ValueError('Preparation starts after observation')
        operations.append(Operation('input-prepare:'+f.request_id,'prepare',None,
            ('external-prepare:'+f.request_id,), (Credit(f.request_id,'prepare'),),
            ready_ms=(observed_at-window_started_at)*1000.,
            service_shape=('input_preparation_residual',f.request_id,f.query_rows,
                           (observed_at-started)*1000.)))
    return tuple(operations)


@dataclass(frozen=True)
class ProspectivePrefillShape:
    """Prediction-only unknown-prompt shape, never a runtime dispatch contract."""
    request_id: str
    prompt_tokens: int
    context_tokens: int
    extend_tokens: int
    preparation_key: str
    reference_id: str
    estimated: bool = True
    cache_hit_assumed: bool = False


def describe_prospective_prefill(frontiers, reference_selection, preparation_operations):
    """Use already-declared common estimates after explicit preparation gates.

    No prefix-cache peek on unknown token contents. This descriptor deliberately
    cannot be passed as a runtime prefill frontier: admission additionally needs
    shared KV capacity, queue/slot order, and the fixed proposal's bundle limit.
    """
    by_id={f.request_id:f for f in frontiers}
    profiles=dict(reference_selection.profiles)
    estimated=set(reference_selection.estimated_prompt_requests)
    result=[];seen=set()
    for op in preparation_operations:
        if op.stage!='prepare' or len(op.credits)!=1 or op.credits[0].work!='prepare':
            raise ValueError('One preparation completion per prospective request required')
        rid=op.credits[0].request
        if rid in seen or rid not in by_id or rid not in profiles or op.credits[0].fraction!=1.:
            raise ValueError('Unique observed preparation completion required')
        seen.add(rid);f=by_id[rid]
        if f.generation_follows is False:
            continue
        if (f.generation_follows is not True or f.stage not in
                ('ann_pending','ann_inflight','input_preparation') or rid not in estimated):
            return None
        profile=profiles[rid]
        length=profile.prompt_tokens
        if type(length) is not int or length<=0:
            raise ValueError('Explicit profiled prompt estimate required')
        result.append(ProspectivePrefillShape(rid,length,length,length,op.key,profile.reference_id))
    return tuple(result)


@dataclass(frozen=True)
class ProspectivePoolObservation:
    """Published physical facts, deliberately not a target-lease budget.

    Native mapped-but-unpublished extents and future reclaim are excluded.
    Occupied tokens may include shared prefixes, so do not infer per-request
    allocation by summing logical context lengths.
    """
    published_tokens: int
    free_tokens: int
    evictable_tokens: int
    free_request_slots: int

    def __post_init__(self):
        values=(self.published_tokens,self.free_tokens,self.evictable_tokens,self.free_request_slots)
        if any(type(n) is not int or n<0 for n in values):
            raise ValueError('Nonnegative physical pool counts required')
        if self.free_tokens+self.evictable_tokens>self.published_tokens:
            raise ValueError('Free and cached tokens exceed published capacity')


@dataclass(frozen=True)
class ProspectiveCapacityBudget:
    """Shared forecast admission budget, not physical allocation/occupancy."""
    free_tokens: int
    free_request_slots: int
    # Admissions retain prompt + clipped-output reservations across cohorts.
    reservations: tuple = ()
    retained_cache_tokens: int = 0
    completed_request_ids: tuple = ()
    # Physical positions already allocated within each reservation. An absent
    # entry is unknown, not zero. Output reservations need not be materialized.
    physical_allocations: tuple = ()
    completed_eviction_ids: tuple = ()
    # Frozen fitted-lease publication events: (offset_ms, identity, added_tokens).
    # They are not free tokens until replay reaches their completion boundary.
    capacity_publications: tuple = ()
    completed_publication_ids: tuple = ()
    # Reservation claims not yet backed by physical free. They consume cache
    # eligibility, NOT allocator slots, until reclaim/publication repays them.
    cache_backed_tokens: int = 0
    # Existing-owner reservations backed only by this fitted lease's future
    # capacity publication. They are not admission capacity before that
    # event, and physical allocation replay can still fail if decode reaches
    # the shortage first.
    publication_backed_tokens: int = 0


def derive_prospective_capacity_publications(memory, observation, *, event_id,
        static_capacity_tokens, tokens_per_extent, total_extents, kv_key):
    """Translate THIS native fitted grow plan, not demand or maximum capacity.

    The ready gate must include publication and required ownership conversion.
    Missing timing remains unsupported. Reclaim/shrink needs a separate model.
    """
    layout=(static_capacity_tokens,tokens_per_extent,total_extents)
    if any(type(v) is not int or v<0 for v in layout) or tokens_per_extent==0:
        raise ValueError('Explicit frozen KV extent layout required')
    if not isinstance(observation,ProspectivePoolObservation):
        return None
    plan=memory.resource_plan
    kv=getattr(plan,'kv',None);geometry=getattr(kv,'geometry',None)
    if (not memory.supported or memory.prediction_unavailable is not None
            or getattr(plan,'status',None)!='available'
            or getattr(plan,'event_id',None)!=event_id or getattr(kv,'event_id',None)!=event_id
            or geometry is None or geometry.published_capacity_tokens!=observation.published_tokens
            or not geometry.within_pool_capacity or geometry.publication_gap_tokens!=0):
        return None
    if (type(kv.target_extents) is not int or not 0<=kv.target_extents<=total_extents
            or static_capacity_tokens+geometry.mapped_extents*tokens_per_extent
                !=observation.published_tokens):
        return None
    target=static_capacity_tokens+kv.target_extents*tokens_per_extent
    if kv.status=='ready':
        return () if target==observation.published_tokens else None
    if kv.status!='grow' or target<=observation.published_tokens:
        return None
    ready=dict(memory.ready).get(kv_key)
    if ready is None:return None
    at=max(ready,dict(memory.transition_ready).get(kv_key,0.))
    if not math.isfinite(at) or at<0:return None
    return ((at,f'kv-publication:{event_id}:{kv_key}',target-observation.published_tokens),)


def prospective_physical_free_tokens(budget):
    """Free physical slots include unmaterialized active reservations.

    Retained cache is not free. The caller must advance allocations at every
    modeled decode allocation event; reservation-only states remain unknown.
    """
    reserved=dict(budget.reservations);allocated=dict(budget.physical_allocations)
    if (len(reserved)!=len(budget.reservations)
            or len(allocated)!=len(budget.physical_allocations)
            or type(budget.free_tokens) is not int or budget.free_tokens<0
            or not set(allocated)<=set(reserved)):
        raise ValueError('Consistent physical and reserved forecast ownership required')
    for rid,value in reserved.items():
        if type(value) is not int or value<0:
            raise ValueError('Nonnegative integer reservation required')
        actual=allocated.get(rid)
        if actual is not None and (type(actual) is not int or not 0<=actual<=value):
            raise ValueError('Physical allocation exceeds its reservation')
    if set(allocated)!=set(reserved):
        return None
    claims=(budget.cache_backed_tokens,budget.publication_backed_tokens)
    if any(type(value) is not int or value<0 for value in claims):
        raise ValueError('Nonnegative outstanding capacity claims required')
    available=(budget.free_tokens+sum(reserved[rid]-allocated[rid] for rid in reserved)
               -budget.cache_backed_tokens-budget.publication_backed_tokens)
    if available<0:raise ValueError('Cache-backed claims do not provide physical allocation')
    return available


def repay_prospective_cache_claims(budget, tokens):
    """New free capacity first backs outstanding claims, not new admissions."""
    if type(tokens) is not int or tokens<0 or type(budget.cache_backed_tokens) is not int or budget.cache_backed_tokens<0:
        raise ValueError('Nonnegative free capacity and cache claims required')
    repaid=min(tokens,budget.cache_backed_tokens)
    return replace(budget,free_tokens=budget.free_tokens+tokens-repaid,
                   cache_backed_tokens=budget.cache_backed_tokens-repaid)


def apply_prospective_capacity_publication(budget, tokens):
    """Publish fitted KV capacity after first backing existing-owner claims."""
    if (type(tokens) is not int or tokens<0
            or type(budget.publication_backed_tokens) is not int
            or budget.publication_backed_tokens<0):
        raise ValueError('Nonnegative publication and outstanding claim required')
    repaid=min(tokens,budget.publication_backed_tokens)
    budget=replace(budget,
        publication_backed_tokens=budget.publication_backed_tokens-repaid)
    return repay_prospective_cache_claims(budget,tokens-repaid)


def make_prospective_capacity_budget(observation, *, reservations, physical_allocations,
                                     capacity_publications=()):
    """Subtract unmaterialized owner reservations from CURRENT physical free.

    Owner coordinates must come from the verified allocation adapter, not a
    sum of shared-prefix lengths. Missing owner accounting stays unavailable.
    Already occupied positions have already reduced allocator free and are not
    subtracted a second time. Neither evictable cache nor future publications
    can back existing reservations at the observation boundary.
    """
    if not isinstance(observation,ProspectivePoolObservation):return None
    reservations=tuple(reservations);physical_allocations=tuple(physical_allocations)
    reserved=dict(reservations);allocated=dict(physical_allocations)
    if (len(reserved)!=len(reservations) or len(allocated)!=len(physical_allocations)
            or not set(allocated)<=set(reserved)):
        raise ValueError('Unique matching reservation owners required')
    if set(allocated)!=set(reserved):return None
    for rid,total in reservations:
        actual=allocated[rid]
        if (not isinstance(rid,str) or not rid or type(total) is not int or total<0
                or type(actual) is not int or not 0<=actual<=total):
            raise ValueError('Valid owner reservation and materialization required')
    publications=tuple(capacity_publications)
    seen=set()
    for at,key,added in publications:
        if (not math.isfinite(at) or at<0 or not isinstance(key,str) or not key or key in seen
                or type(added) is not int or added<=0):
            raise ValueError('Unique timed positive capacity publications required')
        seen.add(key)
    remaining=sum(reserved[rid]-allocated[rid] for rid in reserved)
    deficit=max(0,remaining-observation.free_tokens)
    if deficit>sum(added for at,key,added in publications):return None
    budget=ProspectiveCapacityBudget(max(0,observation.free_tokens-remaining),
        observation.free_request_slots,reservations=reservations,
        physical_allocations=physical_allocations,retained_cache_tokens=observation.evictable_tokens,
        capacity_publications=publications,publication_backed_tokens=deficit)
    # Validate event identities/timing without applying even a zero-time event:
    # the event replay owns publication, never this source-budget constructor.
    return budget


def additional_native_prefill_reserve(budget, *, native_offset, evictable_tokens):
    """Intersect native admission and guaranteed physical fit without double charge.

    Native limit: physical_free + evictable - native_offset.
    Physical-fit limit: budget.free_tokens (already net of owner reservations).
    Return only the extra floating offset to subtract from the latter. This
    does not evict cache or make cached positions physically allocatable.
    """
    nonnegative(native_offset,'native admission offset')
    if type(evictable_tokens) is not int or not 0<=evictable_tokens<=budget.retained_cache_tokens:
        raise ValueError('Known eligible subset of retained cache required')
    physical=prospective_physical_free_tokens(budget)
    if physical is None:return None
    return max(0.,native_offset-(physical-budget.free_tokens)-evictable_tokens)


def prospective_native_eviction_request(budget, allocation_tokens):
    """Ordinary non-hybrid ScheduleBatch eviction request, not shortfall.

    Whether enough unlocked cache exists and how long reclaim takes are
    separate checks. A positive result does not release any capacity.
    """
    if type(allocation_tokens) is not int or allocation_tokens<0:
        raise ValueError('Nonnegative native allocation size required')
    available=prospective_physical_free_tokens(budget)
    if available is None:
        return None
    return allocation_tokens if available<allocation_tokens else 0


def allocate_prospective_reserved_tokens(budget, required_totals):
    """Materialize an entire iteration's absolute, already-reserved KV counts.

    This is an allocation event, not a fresh reservation, implicit cache
    eviction or HBM replan. Unknown ownership or insufficient reservation is
    unsupported. Repeated absolute requirements are idempotent.
    """
    available=prospective_physical_free_tokens(budget)
    if available is None:
        return None
    required_totals=tuple(required_totals);required=dict(required_totals)
    allocated=dict(budget.physical_allocations);reserved=dict(budget.reservations)
    if len(required)!=len(required_totals) or not set(required)<=set(allocated):
        raise ValueError('Unique currently-owned allocation cohort required')
    for rid,total in required.items():
        if type(total) is not int or total<allocated[rid]:
            raise ValueError('Allocation events cannot shrink physical KV')
        if total>reserved[rid]:
            return None
    if sum(total-allocated[rid] for rid,total in required.items())>available:return None
    allocated.update(required)
    return replace(budget,physical_allocations=tuple(allocated.items()))


def advance_prospective_physical_allocations(budget, allocation_events, *, through_ms,
                                           allow_reservation_growth=False):
    """Replay fixed-trace allocation events through one forecast boundary.

    Historical absolute events are idempotent, and completed owners are never
    resurrected. Unknown active ownership or unbacked growth is unsupported.
    No service query, future-arrival lookup, release or elastic planning here.
    """
    nonnegative(through_ms,'physical allocation observation boundary')
    if type(allow_reservation_growth) is not bool:
        raise ValueError('Explicit reservation growth policy required')
    if prospective_physical_free_tokens(budget) is None:
        return None
    events=tuple(allocation_events)
    previous=-1.
    for at,requirements in events:
        nonnegative(at,'forecast allocation time')
        if at<previous:
            raise ValueError('Chronological allocation events required')
        previous=at
        if at>through_ms:
            continue
        current=dict(budget.physical_allocations)
        requirements=tuple(requirements)
        if len(dict(requirements))!=len(requirements):
            raise ValueError('Unique owners within an allocation event required')
        updates=[]
        for rid,total in requirements:
            if type(total) is not int or total<0:
                raise ValueError('Nonnegative physical allocation required')
            if rid in budget.completed_request_ids:
                continue
            if rid not in current:
                return None
            updates.append((rid,max(current[rid],total)))
        if allow_reservation_growth:
            budget=grow_prospective_reservations(budget,updates)
            if budget is None:
                return None
        advanced=allocate_prospective_reserved_tokens(budget,updates)
        if advanced is None:
            return None
        budget=advanced
    return budget


@dataclass(frozen=True)
class ProspectiveAllocationWait:
    at_ms: float
    requirements: tuple
    budget: ProspectiveCapacityBudget
    requested_tokens: int
    physical_free_tokens: int
    reason: str


def advance_prospective_capacity_events(budget, prediction, *, through_ms,
                                       completion_accounting, report_allocation_wait=False):
    """Interleave allocation and completed-request release on one clock.

    Completion processing precedes the next allocation at the same boundary.
    Grow reservations only from this lease's free budget, never retained cache
    or not-yet-completed releases. A blocked allocation invalidates this trace;
    an event driver must model its wait before counting later token progress.
    """
    nonnegative(through_ms,'capacity event boundary')
    if type(report_allocation_wait) is not bool:
        raise ValueError('Explicit allocation diagnostic mode required')
    accounting=dict(completion_accounting)
    publications=tuple(budget.capacity_publications)
    identities=set()
    for at,key,added in publications:
        if (not math.isfinite(at) or at<0 or not isinstance(key,str) or not key
                or key in identities or type(added) is not int or added<=0):
            raise ValueError('Unique timed positive capacity publications required')
        identities.add(key)
    if not set(budget.completed_publication_ids)<=identities:
        raise ValueError('Completed publication lacks its fitted event')
    events=[(at,2,requirements) for at,requirements in prediction.kv_allocation_events]
    events.extend((at,-1,(key,added)) for at,key,added in publications)
    events.extend((end,0,rid) for rid,work,end in prediction.work_completions
                  if work=='decode' and rid in accounting)
    events.extend((end,1,(key,freed)) for end,key,freed in prediction.cache_release_events)
    for at,kind,value in sorted(events,key=lambda event:event[:2]):
        if at>through_ms:
            break
        if kind==-1:
            key,added=value
            if key not in budget.completed_publication_ids:
                budget=replace(apply_prospective_capacity_publication(budget,added),
                    completed_publication_ids=(*budget.completed_publication_ids,key))
        elif kind==2:
            value=tuple(value)
            before=budget
            budget=advance_prospective_physical_allocations(before,((at,value),),
                through_ms=through_ms,allow_reservation_growth=True)
            if budget is None:
                if report_allocation_wait:
                    actual=dict(before.physical_allocations)
                    active=tuple((rid,total) for rid,total in value if rid not in before.completed_request_ids)
                    available=prospective_physical_free_tokens(before)
                    if available is None or any(rid not in actual for rid,total in active):return None
                    requested=sum(max(0,total-actual[rid]) for rid,total in active)
                    return ProspectiveAllocationWait(at,active,before,requested,available,
                        'physical_capacity' if requested>available else 'reservation_capacity')
                return None
        elif kind==1:
            key,freed=value
            if key in budget.completed_eviction_ids:
                continue
            budget=complete_prospective_cache_eviction(budget,freed_tokens=freed)
            budget=replace(budget,completed_eviction_ids=(*budget.completed_eviction_ids,key))
        else:
            if value in budget.completed_request_ids:
                continue
            actual=dict(budget.physical_allocations).get(value)
            allocated,freed=accounting[value]
            if actual is None or actual!=allocated:
                return None
            budget=complete_prospective_request(budget,value,allocated_tokens=allocated,
                                               physically_freed_tokens=freed)
    return budget


def resolve_prospective_decode_capacity(snapshot, projection, budget, *, horizon_ms,
        completion_accounting, build_cache_eviction, through_ms=None):
    """Insert allocator waits and rebuild one fixed proposal, never reselect it.

    Each retry adds a real timed host call at the first physical shortage.
    Always replay the original budget; do not repay publications twice. Only
    strict forward progress is accepted. Reservation-only deficits and host
    order changes needing a more general event driver remain unavailable.
    Return (projection, budget, resume_boundary). An intermediate boundary
    advances through an intersecting reclaim, never beyond the common H.
    """
    if projection.decode_plan is None or not callable(build_cache_eviction):return None
    boundary=horizon_ms if through_ms is None else through_ms
    if not math.isfinite(boundary) or not 0<=boundary<=horizon_ms:
        raise ValueError('Capacity boundary must remain within common H')
    expanded=projection;last_wait=-1.;number=0
    while True:
        trace=predict(snapshot,expanded,horizon_ms,progress_only=True)
        eviction_keys={op.key for op in expanded.operations if op.stage=='cache_evict'}
        for key,start,end in trace.intervals:
            if key in eviction_keys and start<=boundary<end:
                boundary=min(horizon_ms,end)
        result=advance_prospective_capacity_events(budget,trace,through_ms=boundary,
            completion_accounting=completion_accounting,report_allocation_wait=True)
        if result is None:return None
        if isinstance(result,ProspectiveCapacityBudget):return expanded,result,boundary
        if result.reason!='physical_capacity' or result.at_ms<=last_wait:return None
        key=f'decode-allocation-evict:{number}'
        if any(op.key==key for op in expanded.operations):
            number+=1;continue
        eviction=build_cache_eviction(expanded,trace,result.at_ms,result.requested_tokens,key)
        if eviction is None:return None
        shape=getattr(eviction,'service_shape',None)
        if (not isinstance(eviction,Operation) or eviction.key!=key or eviction.stage!='cache_evict'
                or eviction.credits or eviction.lanes!=('host_result',)
                or eviction.ready_ms!=result.at_ms or eviction.duration_ms is None or eviction.duration_ms<=0
                or not isinstance(shape,ProspectiveCacheEvictionShape)
                or shape.requested_tokens!=result.requested_tokens):
            raise ValueError('Decode reclaim must match the blocked native allocation')
        if result.physical_free_tokens+shape.predicted_freed_tokens<result.requested_tokens:return None
        expanded=replace(expanded,operations=(*expanded.operations,eviction))
        validate_projected_work(snapshot,expanded.operations)
        upstream=predict(snapshot,replace(expanded,decode_plan=None),horizon_ms)
        start,end=next((start,end) for opkey,start,end in upstream.intervals if opkey==key)
        if start!=result.at_ms:return None # Cannot let decode execute before a delayed host reclaim.
        boundary=min(horizon_ms,max(boundary,end))
        last_wait=result.at_ms;number+=1


def reserve_prospective_cohort(budget, shapes, selection, *, max_new_tokens, clip_max_new_tokens):
    """Consume one fitted cohort once; never reuse its budget in the next batch.

    No implicit frees on predicted prefill completion: KV/request slots persist
    into decode. Native cache retention prevents equating request completion
    with full prompt-KV release; a future release model must handle that itself.
    """
    if (type(budget.free_tokens) is not int or budget.free_tokens<0
            or type(budget.free_request_slots) is not int or budget.free_request_slots<0
            or type(clip_max_new_tokens) is not int or clip_max_new_tokens<0):
        raise ValueError('Nonnegative forecast capacity required')
    existing=dict(budget.reservations)
    if (len(existing)!=len(budget.reservations)
            or any(type(v) is not int or v<0 for v in existing.values())):
        raise ValueError('Unique valid forecast reservations required')
    shapes=tuple(shapes);by_id={s.request_id:s for s in shapes};ids=tuple(selection.request_ids)
    if (len(by_id)!=len(shapes) or len(set(ids))!=len(ids)
            or set(ids)&(set(existing)|set(budget.completed_request_ids))):
        raise ValueError('Duplicate forecast admission')
    amounts=[]
    for rid in ids:
        shape=by_id.get(rid);output=max_new_tokens.get(rid)
        if (not isinstance(shape,ProspectivePrefillShape) or type(output) is not int
                or output<0 or type(shape.extend_tokens) is not int or shape.extend_tokens<=0):
            raise ValueError('Complete prospective input/output contract required')
        amounts.append((rid,shape.extend_tokens+min(output,clip_max_new_tokens)))
    if selection.extend_tokens!=sum(by_id[rid].extend_tokens for rid in ids):
        raise ValueError('Forecast selection changed input work')
    consumed=sum(value for _,value in amounts)
    if consumed>budget.free_tokens or len(ids)>budget.free_request_slots:
        raise ValueError('Forecast cohort exceeds shared remaining budget')
    return replace(budget,free_tokens=budget.free_tokens-consumed,
        free_request_slots=budget.free_request_slots-len(ids),
        reservations=(*budget.reservations,*amounts),
        physical_allocations=(*budget.physical_allocations,
            *((rid,by_id[rid].extend_tokens) for rid in ids)))


def grow_prospective_reservations(budget, required_totals):
    """Atomically back a forecast decode cohort's absolute KV requirements.

    Consume only growth beyond existing reservations, never shrink them or
    implicitly reclaim cache. Repeating an absolute requirement is idempotent.
    None means the whole cohort must wait; no partial allocation is published.
    This is forecast accounting, not a runtime allocation or memory replan.
    """
    reservations=dict(budget.reservations)
    if (len(reservations)!=len(budget.reservations)
            or type(budget.free_tokens) is not int or budget.free_tokens<0
            or any(type(n) is not int or n<0 for n in reservations.values())
            or set(reservations)&set(budget.completed_request_ids)):
        raise ValueError('Valid active forecast reservations required')
    required_totals=tuple(required_totals)
    if len(dict(required_totals))!=len(required_totals):
        raise ValueError('One absolute requirement per request required')
    for rid,total in required_totals:
        if rid not in reservations or type(total) is not int or total<0:
            raise ValueError('Nonnegative requirement for an active request required')
    updated=dict(reservations)
    for rid,total in required_totals:
        updated[rid]=max(updated[rid],total)
    growth=sum(updated[rid]-n for rid,n in reservations.items())
    if growth>budget.free_tokens:
        return None
    return replace(budget,free_tokens=budget.free_tokens-growth,
                   reservations=tuple(updated.items()))


def complete_prospective_request(budget, request_id, *, allocated_tokens, physically_freed_tokens):
    """Forecast completion accounting with explicit cache-retention evidence.

    Return the request slot and unused reservation, but only add physically
    freed KV to free capacity. Cached KV remains separate until a modeled
    eviction completes. Counts must come from the same simulation's allocator
    accounting; a predicted final token alone does not supply those counts.
    """
    reservations=dict(budget.reservations)
    if len(reservations)!=len(budget.reservations) or request_id not in reservations:
        raise ValueError('Completion requires one active forecast reservation')
    if request_id in budget.completed_request_ids:
        raise ValueError('Duplicate forecast completion')
    for value in (allocated_tokens,physically_freed_tokens,budget.retained_cache_tokens):
        if type(value) is not int or value<0:
            raise ValueError('Nonnegative explicit KV accounting required')
    reserved=reservations[request_id]
    if not physically_freed_tokens<=allocated_tokens<=reserved:
        raise ValueError('KV allocation exceeds reservation or invalid physical free')
    previous=dict(budget.physical_allocations).get(request_id)
    if previous is not None and allocated_tokens<previous:
        raise ValueError('Completion cannot discard already allocated physical KV')
    unused=reserved-allocated_tokens
    return replace(repay_prospective_cache_claims(budget,unused+physically_freed_tokens),
        free_request_slots=budget.free_request_slots+1,
        reservations=tuple((rid,n) for rid,n in budget.reservations if rid!=request_id),
        physical_allocations=tuple((rid,n) for rid,n in budget.physical_allocations if rid!=request_id),
        retained_cache_tokens=budget.retained_cache_tokens+allocated_tokens-physically_freed_tokens,
        completed_request_ids=(*budget.completed_request_ids,request_id))


def complete_prospective_cache_eviction(budget, *, freed_tokens):
    """Account a completed modeled eviction, not merely evictable capacity."""
    if type(freed_tokens) is not int or not 0<=freed_tokens<=budget.retained_cache_tokens:
        raise ValueError('Eviction must fit separately retained cache capacity')
    return replace(repay_prospective_cache_claims(budget,freed_tokens),
                   retained_cache_tokens=budget.retained_cache_tokens-freed_tokens)


@dataclass(frozen=True)
class ProspectiveCacheEvictionShape:
    requested_tokens: int
    predicted_freed_tokens: int
    eligible_tokens: int


def build_prospective_cache_eviction(*, key, requested_tokens, predicted_freed_tokens,
                                     eligible_tokens, duration_ms, ready_ms,
                                     depends_on=()):
    """Explicit timed forecast event, never an instruction to the allocator.

    Eligibility and released counts must come from the same detached cache
    state/model, not a latency sample or target HBM budget. Native commit still
    checks actual locks/slots. Overshoot/shortfall are retained as supplied.
    """
    if (not isinstance(key,str) or not key or type(requested_tokens) is not int
            or requested_tokens<=0 or any(type(n) is not int or n<0
                for n in (predicted_freed_tokens,eligible_tokens))
            or predicted_freed_tokens>eligible_tokens):
        raise ValueError('Explicit eligible cache capacity and release counts required')
    nonnegative(ready_ms,'cache eviction readiness')
    if duration_ms is None:
        return None
    nonnegative(duration_ms,'cache eviction service')
    if duration_ms==0:
        raise ValueError('Cache eviction needs a positive service estimate')
    return Operation(key,'cache_evict',duration_ms,('host_result',),
        depends_on=tuple(depends_on),ready_ms=ready_ms,
        service_shape=ProspectiveCacheEvictionShape(requested_tokens,predicted_freed_tokens,eligible_tokens))


def fit_prospective_prefill_cohort(shapes, *, preparation_ends, dispatch_ms,
                                   max_new_tokens, clip_max_new_tokens,
                                   guaranteed_free_tokens, running_token_reserve,
                                   input_token_budget, max_requests, earlier_queue_empty):
    """Fit already-predicted ready inputs using the existing FCFS admission rule.

    Caller must account for older queued/prepared work before this operation.
    Capacity is the remaining guaranteed pool AFTER competing reservations,
    not target HBM bytes or uncompleted KV releases. No new memory plan or
    compute selection occurs. All prompt/cache shapes remain estimates.
    """
    nonnegative(dispatch_ms,'prospective prefill dispatch offset')
    if type(earlier_queue_empty) is not bool:
        raise ValueError('Explicit prior queue completion required')
    if not earlier_queue_empty:
        return None
    shapes=tuple(shapes)
    if len({s.request_id for s in shapes})!=len(shapes):
        raise ValueError('Unique prospective requests required')
    ready=[]
    for position,shape in enumerate(shapes):
        if not isinstance(shape,ProspectivePrefillShape) or not shape.estimated or shape.cache_hit_assumed:
            raise ValueError('Prediction-only uncached prompt estimate required')
        end=preparation_ends.get(shape.preparation_key)
        if end is None:
            return None
        if math.isnan(end) or end<0:
            raise ValueError('Invalid preparation completion prediction')
        if end<=dispatch_ms:
            ready.append((end,position,shape))
    ordered=tuple(s for _,_,s in sorted(ready,key=lambda v:(v[0],v[1])))
    if any(s.request_id not in max_new_tokens for s in ordered):
        return None
    return select_cached_prefill_cohort(ordered,
        max_new_tokens={s.request_id:max_new_tokens[s.request_id] for s in ordered},
        clip_max_new_tokens=clip_max_new_tokens,
        available_and_evictable_tokens=guaranteed_free_tokens,
        running_token_reserve=running_token_reserve,input_token_budget=input_token_budget,
        max_requests=max_requests,cumulative_lock_tokens=(0,)*len(ordered))


@dataclass(frozen=True)
class ProspectiveCachedAdmission:
    """Prediction-only proposal, NOT allocated or guaranteed physical KV."""
    selection: object
    reservation_tokens: int
    allocation_tokens: int
    cache_backed_tokens: int
    eviction_request_tokens: int


def plan_prospective_cached_admission(shapes, *, budget, evictable_tokens,
        native_offset, preparation_ends, dispatch_ms, max_new_tokens,
        clip_max_new_tokens, input_token_budget, max_requests, earlier_queue_empty):
    """Mirror cached admission before the allocator's distinct eviction check.

    Output estimates can be backed by currently eligible cache without forcing
    an immediate eviction of the whole reservation. Actual prefill allocation
    uses only uncached prompt positions. The caller must retain cache-backed
    claims, model reclaim/waits at allocation, and NEVER dispatch this proposal
    or add cache_backed_tokens to physical free.
    """
    if type(evictable_tokens) is not int or not 0<=evictable_tokens<=budget.retained_cache_tokens:
        raise ValueError('Known current evictable subset required')
    physical=prospective_physical_free_tokens(budget)
    if physical is None:return None
    # Both admission ceilings include eligible cache here. Only the part of
    # the policy offset not already held by owners must be subtracted again.
    nonnegative(native_offset,'native admission offset')
    extra=max(0.,native_offset-(physical-budget.free_tokens+budget.cache_backed_tokens))
    unclaimed_cache=max(0,evictable_tokens-budget.cache_backed_tokens)
    shapes=tuple(shapes)
    selection=fit_prospective_prefill_cohort(shapes,preparation_ends=preparation_ends,
        dispatch_ms=dispatch_ms,max_new_tokens=max_new_tokens,clip_max_new_tokens=clip_max_new_tokens,
        guaranteed_free_tokens=budget.free_tokens+unclaimed_cache,running_token_reserve=extra,
        input_token_budget=input_token_budget,max_requests=min(max_requests,budget.free_request_slots),
        earlier_queue_empty=earlier_queue_empty)
    if selection is None:return None
    selected=set(selection.request_ids)
    allocation=sum(s.extend_tokens for s in shapes if s.request_id in selected)
    reserve=sum(s.extend_tokens+min(max_new_tokens[s.request_id],clip_max_new_tokens)
                for s in shapes if s.request_id in selected)
    return ProspectiveCachedAdmission(selection,reserve,allocation,
        max(0,reserve-budget.free_tokens),allocation if physical<allocation else 0)


def reserve_prospective_cached_cohort(budget, shapes, admission, *, evictable_tokens,
        max_new_tokens, clip_max_new_tokens):
    """Commit a forecast reservation only after its prompt physically fits.

    Existing claims cannot be promised twice. This never performs eviction;
    insufficient current physical free returns None even when admission had
    enough cache-backed capacity. Revalidate against the post-wait budget.
    """
    if not isinstance(admission,ProspectiveCachedAdmission):
        raise ValueError('Explicit cached admission proposal required')
    if type(evictable_tokens) is not int or not 0<=evictable_tokens<=budget.retained_cache_tokens:
        raise ValueError('Current cache eligibility required')
    physical=prospective_physical_free_tokens(budget)
    if physical is None or physical<admission.allocation_tokens:return None
    claim=max(0,admission.reservation_tokens-budget.free_tokens)
    if claim>max(0,evictable_tokens-budget.cache_backed_tokens):return None
    funded=replace(budget,free_tokens=budget.free_tokens+claim)
    reserved=reserve_prospective_cohort(funded,shapes,admission.selection,
        max_new_tokens=max_new_tokens,clip_max_new_tokens=clip_max_new_tokens)
    if (funded.free_tokens-reserved.free_tokens!=admission.reservation_tokens
            or admission.selection.extend_tokens!=admission.allocation_tokens):
        raise ValueError('Cached proposal changed input/reservation work')
    result=replace(reserved,cache_backed_tokens=budget.cache_backed_tokens+claim)
    prospective_physical_free_tokens(result)
    return result


def build_prospective_prefill_bundles(shapes, selection, *, total_layers,
                                     fitted_layers, kv_key, dispatch_ms,
                                     compute_ready_ms, preceding_dependencies=(),
                                     prefix='prospective-prefill'):
    """Expand one fitted forecast cohort, never a new runtime admission.

    Caller owns the FCFS/slot/KV accounting used for selection. Keep estimated
    shape provenance and all prepare/backing/compute gates on the fixed graph.
    No service queries, cached-prefix inference, or compute search here.
    """
    if type(total_layers) is not int or total_layers<=0 or type(fitted_layers) is not int or fitted_layers<0:
        raise ValueError('Positive layer count and nonnegative fixed bundle limit required')
    if not isinstance(kv_key,str) or not kv_key or not isinstance(prefix,str) or not prefix:
        raise ValueError('Explicit backing and operation identities required')
    nonnegative(dispatch_ms,'prospective dispatch offset')
    nonnegative(compute_ready_ms,'prospective compute readiness')
    shapes=tuple(shapes);by_id={s.request_id:s for s in shapes}
    ids=tuple(selection.request_ids)
    if (len(by_id)!=len(shapes) or len(ids)!=len(set(ids)) or not set(ids)<=set(by_id)
            or any(not isinstance(s,ProspectivePrefillShape) for s in shapes)):
        raise ValueError('Unique prediction-only cohort required')
    selected=tuple(by_id[rid] for rid in ids)
    if selection.extend_tokens!=sum(s.extend_tokens for s in selected):
        raise ValueError('Fitted forecast input work mismatch')
    if not selected or fitted_layers==0:
        return ()
    predecessors=tuple(dict.fromkeys((*preceding_dependencies,
                                      *(s.preparation_key for s in selected))))
    result=[]
    for start in range(0,total_layers,fitted_layers):
        end=min(total_layers,start+fitted_layers)
        shape=ProspectivePrefillBundleShape(ids,tuple(s.prompt_tokens for s in selected),start,end,
            tuple(s.context_tokens for s in selected),tuple(s.extend_tokens for s in selected))
        key=f'{prefix}:{start}:{end}'
        result.append(Operation(key,'prefill',None,('prefill',),
            tuple(Credit(rid,'prefill',end/total_layers) for rid in ids),
            depends_on=predecessors,ready_ms=dispatch_ms,compute_ready_ms=compute_ready_ms,
            kv_key=kv_key,memory_keys=(kv_key,),service_shape=shape))
        predecessors=(key,)
    return tuple(result)


@dataclass(frozen=True)
class ProspectiveGenerationFrontier:
    """Simulation state for an observed request; never runtime readiness."""
    request_id: str
    prompt_tokens: int
    remaining_fixed_output: int | None
    remaining_output_limit: int | None
    stage: str = 'prefill_queued'
    generation_follows: bool = True
    generated_tokens: int = 0
    estimated: bool = True


def build_prospective_prefill_handoff(frontiers, shapes, profiles, *, prefill_operations,
                                     existing_kv_key, new_kv_key, result_dependencies=(),
                                     merge_dependencies=(), prefix='prospective-handoff',
                                     separate_commit_gate=False):
    """Continue the same forecast cohort through token one and decode joining."""
    by_id={f.request_id:f for f in frontiers}
    shape_map={s.request_id:s for s in shapes}
    if not prefill_operations:
        return None
    bundle=prefill_operations[-1].service_shape
    if not isinstance(bundle,ProspectivePrefillBundleShape):
        raise ValueError('Estimated final prefill bundle required')
    simulated=[]
    for rid,prompt in zip(bundle.request_ids,bundle.prompt_tokens):
        f=by_id.get(rid);shape=shape_map.get(rid);profile=profiles.get(rid)
        if f is None or shape is None or profile is None:
            return None
        if (f.stage not in ('ann_pending','ann_inflight','input_preparation')
                or f.generation_follows is not True or f.generated_tokens!=0
                or prompt!=shape.prompt_tokens or profile.prompt_tokens!=prompt
                or profile.reference_id!=shape.reference_id):
            raise ValueError('Prospective generation changed observed request contract')
        simulated.append(ProspectiveGenerationFrontier(rid,prompt,
            f.remaining_fixed_output,f.remaining_output_limit))
    result=build_prefill_decode_handoff(tuple(simulated),profiles,
        prefill_operations=prefill_operations,existing_kv_key=existing_kv_key,
        new_kv_key=new_kv_key,result_dependencies=result_dependencies,
        merge_dependencies=merge_dependencies,prefix=prefix,separate_commit_gate=separate_commit_gate)
    if result is None:
        return None
    operations,gates,tokens=result
    return tuple(simulated),operations,gates,tokens


def derive_prospective_radix_events(shapes, frontiers, operations, prediction, *, horizon_ms):
    """Derive cache changes from result/final-token events in the SAME trace.

    Last prefill bundle completion is too early: native cache insertion occurs
    during output processing. Unknown natural EOS never yields a fabricated
    completion/unlock, even if a reference-work proxy reaches its limit.
    """
    if not math.isfinite(horizon_ms) or horizon_ms<=0:
        raise ValueError('Positive finite common cache event horizon required')
    shapes=tuple(shapes)
    by_id={f.request_id:f for f in frontiers};shape_map={s.request_id:s for s in shapes}
    if len(shape_map)!=len(shapes) or not set(shape_map)<=set(by_id):
        raise ValueError('Unique observed prospective shapes required')
    ends={key:end for key,start,end in prediction.intervals}
    completed={(rid,work):end for rid,work,end in prediction.work_completions}
    events=[];seen=set()
    for op in operations:
        if op.stage!='prefill_result' or op.key not in ends or ends[op.key]>horizon_ms:
            continue
        for credit in op.credits:
            rid=credit.request
            if credit.work!='decode' or rid not in shape_map:continue
            if rid in seen:raise ValueError('One prefill result per prospective request required')
            seen.add(rid);shape=shape_map[rid];f=by_id[rid];at=ends[op.key]
            if (not isinstance(shape,ProspectivePrefillShape) or shape.cache_hit_assumed
                    or f.generated_tokens!=0 or f.generation_follows is not True):
                raise ValueError('Estimated uncached new generation request required')
            count=f.remaining_fixed_output
            if count is not None and (type(count) is not int or count<=0
                    or not math.isclose(credit.fraction,1./count)):
                raise ValueError('Prefill result must emit exactly the declared first token')
            if count==1:
                if completed.get((rid,'decode'))!=at:
                    raise ValueError('Single-token request completes at its result event')
                events.append((at,'single_token',rid,shape.prompt_tokens))
            else:
                events.append((at,'insert',rid,shape.prompt_tokens))
                done=completed.get((rid,'decode'))
                if count is not None and done is not None and done<=horizon_ms:
                    if done<at:raise ValueError('Decode completion precedes prefill result')
                    events.append((done,'finish',rid,shape.prompt_tokens+count-1))
    return tuple(sorted(events,key=lambda event:(event[0],0 if event[1]=='insert' else 1,event[2])))


def derive_prospective_completion_accounting(shapes, frontiers, *, cache_policy):
    """Unit-page, non-overlap, non-speculative new-prompt completion model.

    Mirrors cache_finished_req's prompt + output - 1 stored positions. Unknown
    radix-prefix collisions earn no predicted physical free: cached positions
    remain retained until an explicit eviction. Counts are conditional on the
    declared uncached prompt estimate, not claims about unknown token content.
    The runtime must first verify the supported cache/execution policy.
    """
    if cache_policy not in ('disabled_unit_page','radix_unit_page_unknown_prefix'):
        return None
    observed={f.request_id:f for f in frontiers}
    result={}
    for shape in shapes:
        if (not isinstance(shape,ProspectivePrefillShape) or not shape.estimated
                or shape.cache_hit_assumed or shape.request_id in result):
            raise ValueError('Unique estimated uncached prospective shapes required')
        f=observed.get(shape.request_id)
        if (f is None or f.generation_follows is not True or f.generated_tokens!=0
                or type(f.remaining_fixed_output) is not int or f.remaining_fixed_output<=0):
            return None
        if (type(shape.prompt_tokens) is not int or shape.prompt_tokens<=0
                or shape.extend_tokens!=shape.prompt_tokens):
            raise ValueError('Positive fully uncached prospective prompt required')
        allocated=shape.prompt_tokens+f.remaining_fixed_output-1
        result[shape.request_id]=(allocated,allocated if cache_policy=='disabled_unit_page' else 0)
    return result


def reserve_projected_queued_prefill(budget, projection, frontiers, clip_max_new_tokens,
                                     *, cache_backed_admission=False):
    """Conservative forecast backing for queued work already in the graph.

    Charge the entire context (including any cached prefix) against captured
    free capacity now. This deliberately earns no cache credit and is not a
    live allocation claim. Only selected, not-yet-started queued owners qualify;
    missing active-owner accounting must remain unsupported.
    """
    selected={c.request for op in projection.operations if op.stage=='prefill'
              for c in op.credits if c.work=='prefill'}
    owned=set(dict(budget.reservations))
    queued=tuple(f for f in frontiers if f.stage=='prefill_queued'
                 and f.request_id in selected and f.request_id not in owned)
    shapes=[];limits={}
    for f in queued:
        if (type(f.prompt_tokens) is not int or f.prompt_tokens<=0
                or type(f.remaining_output_limit) is not int or f.remaining_output_limit<=0):
            return None
        shapes.append(ProspectivePrefillShape(f.request_id,f.prompt_tokens,
            f.prompt_tokens,f.prompt_tokens,'already-prepared','captured-queued',False,False))
        limits[f.request_id]=f.remaining_output_limit
    # A target grow is usable only at its fitted publication/backing gate.
    # Apply no decode completion, cache reclaim, or future-arrival capacity;
    # the operation keeps the same memory gate in the scored graph.
    if shapes and budget.capacity_publications:
        memory_ready=dict(projection.memory.ready)
        transition_ready=dict(projection.memory.transition_ready)
        selected_ids={s.request_id for s in shapes}
        selected_ops=tuple(op for op in projection.operations if op.stage=='prefill'
                           and any(c.request in selected_ids and c.work=='prefill'
                                   for c in op.credits))
        keys=tuple(dict.fromkeys(key for op in selected_ops
            for key in (*op.memory_keys, *((op.kv_key,) if op.kv_key else ()))))
        if not selected_ops or not keys or any(key not in memory_ready for key in keys):
            return None
        gate=max(*(memory_ready[key] for key in keys),
                 *(transition_ready.get(key,0.) for key in keys))
        budget=advance_prospective_capacity_events(budget,Prediction(0.,()),
            through_ms=gate,completion_accounting={})
        if budget is None:return None
    required=sum(s.extend_tokens+min(limits[s.request_id],clip_max_new_tokens) for s in shapes)
    if type(cache_backed_admission) is not bool:
        raise ValueError('Explicit queued cache-backing policy required')
    if len(shapes)>budget.free_request_slots:
        return None
    # Reuse the same checked capacity ledger as later preparation cohorts.
    from types import SimpleNamespace
    selection=SimpleNamespace(request_ids=tuple(s.request_id for s in shapes),
                              extend_tokens=sum(s.extend_tokens for s in shapes))
    if not cache_backed_admission:
        if required>budget.free_tokens:return None
        return reserve_prospective_cohort(budget,shapes,selection,
            max_new_tokens=limits,clip_max_new_tokens=clip_max_new_tokens)
    allocation=sum(s.extend_tokens for s in shapes)
    physical=prospective_physical_free_tokens(budget)
    claim=max(0,required-budget.free_tokens)
    if (physical is None or allocation>physical
            or claim>max(0,budget.retained_cache_tokens-budget.cache_backed_tokens)):
        return None
    admission=ProspectiveCachedAdmission(selection,required,allocation,claim,0)
    return reserve_prospective_cached_cohort(budget,shapes,admission,
        evictable_tokens=budget.retained_cache_tokens,
        max_new_tokens=limits,clip_max_new_tokens=clip_max_new_tokens)


def expand_prospective_prefill_cohorts(snapshot, projection, *, frontiers,
        reference_selection, shapes, budget, max_new_tokens, clip_max_new_tokens,
        input_token_budget, max_requests, total_layers, kv_key, existing_kv_key,
        compute_ready_ms, earlier_queue_empty, query_prefill, time_handoff,
        horizon_ms, completion_accounting=None, completion_policy=None,
        running_reservation_rule=None, query_evictable_tokens=None,
        cache_backed_admission=False, build_cache_eviction=None, max_running_requests=None,
        separate_commit_gate=False, conservative_queued_reserve=False,
        failure_reasons=None):
    """Advance ready estimated prompts under ONE fixed proposal and pool.

    This expansion consumes supplied guaranteed capacity and optional explicit
    completion accounting (rid -> allocated, physically freed token counts).
    Only declared fixed-output requests may trigger these releases; unknown
    EOS, cache evictions and future arrivals are never inferred. Rebuild this
    expansion for each lease/transition view; admission offsets are not reusable.
    All cohorts share the same reservations and FCFS order; the caller must
    handle decode-dependent capacity events before treating this as a complete
    dynamic forecast. Returns (projection, remaining_budget, unadmitted_shapes),
    or None for missing in-window timing/shape information.
    """
    if failure_reasons is not None and not isinstance(failure_reasons,list):
        raise ValueError('Mutable bounded failure-reason sink required')
    def unavailable(reason):
        if failure_reasons is not None and len(failure_reasons)<8:
            failure_reasons.append(reason)
        return None
    if projection.decode_plan is None:
        raise ValueError('Explicit frozen joining-decode plan required')
    if type(earlier_queue_empty) is not bool:
        raise ValueError('Explicit older prefill queue state required')
    if not math.isfinite(horizon_ms) or horizon_ms<=0:
        raise ValueError('Positive finite common horizon required')
    if type(cache_backed_admission) is not bool:
        raise ValueError('Explicit cache-backed admission mode required')
    if type(conservative_queued_reserve) is not bool:
        raise ValueError('Explicit queued forecast reservation policy required')
    if conservative_queued_reserve or cache_backed_admission:
        budget=reserve_projected_queued_prefill(budget,projection,frontiers,
            clip_max_new_tokens,cache_backed_admission=(
                cache_backed_admission and not conservative_queued_reserve))
        if budget is None:return unavailable('queued_prefill_reservation')
    if max_running_requests is not None and (type(max_running_requests) is not int or max_running_requests<=0):
        raise ValueError('Positive configured running-request ceiling required')
    if cache_backed_admission and running_reservation_rule is None:
        return unavailable('missing_running_reservation_rule')
    if running_reservation_rule is not None:
        if query_evictable_tokens is None and completion_policy=='disabled_unit_page':
            query_evictable_tokens=lambda projection,trace,at:0
        if not callable(query_evictable_tokens):
            return unavailable('missing_evictable_cache_query')
    shapes=tuple(shapes)
    if (len({s.request_id for s in shapes})!=len(shapes)
            or not {s.request_id for s in shapes}<={f.request_id for f in frontiers}):
        raise ValueError('Only unique observed prospective requests allowed')
    if not earlier_queue_empty or projection.executable.prefill_layers==0:
        return projection,budget,shapes
    if kv_key not in dict(projection.memory.ready):
        return unavailable('missing_prospective_kv_backing')
    expanded=projection
    profiles=dict(reference_selection.profiles)
    if completion_policy is not None:
        if completion_accounting is not None:
            raise ValueError('Use captured completion policy or explicit accounting, not both')
        completion_accounting=derive_prospective_completion_accounting(
            shapes,frontiers,cache_policy=completion_policy)
        if completion_accounting is None:
            return unavailable('missing_completion_accounting')
    accounting=dict(completion_accounting or {})
    observed={f.request_id:f for f in frontiers}
    for rid,counts in accounting.items():
        if (rid not in observed or type(observed[rid].remaining_fixed_output) is not int
                or observed[rid].remaining_fixed_output<=0 or len(counts)!=2
                or any(type(n) is not int or n<0 for n in counts) or counts[1]>counts[0]):
            raise ValueError('Explicit fixed-output completion and physical KV accounting required')
    def advance_to(current,capacity,at):
        if cache_backed_admission and callable(build_cache_eviction):
            return resolve_prospective_decode_capacity(snapshot,current,capacity,horizon_ms=horizon_ms,
                through_ms=at,completion_accounting=accounting,build_cache_eviction=build_cache_eviction)
        trace=predict(snapshot,current,horizon_ms,progress_only=True)
        advanced=advance_prospective_capacity_events(capacity,trace,through_ms=at,
            completion_accounting=accounting)
        return None if advanced is None else (current,advanced,at)
    dispatch=0.;cohort_index=0
    while shapes:
        # Resolve upstream preparation and prefill lanes under the same lease.
        # Decode cannot occupy these lanes; no completion-based frees are used.
        upstream=predict(snapshot,replace(expanded,decode_plan=None),horizon_ms)
        ends={key:end for key,start,end in upstream.intervals}
        if any(s.preparation_key not in ends for s in shapes):
            return unavailable('missing_preparation_completion')
        dispatch=max(dispatch,min(ends[s.preparation_key] for s in shapes),
            dict(expanded.memory.ready)[kv_key],
            dict(expanded.memory.transition_ready).get(kv_key,0.),compute_ready_ms)
        if dispatch>=horizon_ms:
            break
        if (budget.capacity_publications or expanded.decode_plan.frontiers
                or any(op.stage=='cache_evict' for op in expanded.operations)):
            previous=expanded
            advanced=advance_to(expanded,budget,dispatch)
            if advanced is None:return unavailable('capacity_replay_at_dispatch')
            expanded,budget,dispatch=advanced
            if dispatch>=horizon_ms:break
            if expanded is not previous:
                upstream=predict(snapshot,replace(expanded,decode_plan=None),horizon_ms)
                ends={key:end for key,start,end in upstream.intervals}
        native_extra=0.
        if running_reservation_rule is not None:
            trace=predict(snapshot,expanded,horizon_ms,progress_only=True)
            initial_ids=set(dict(running_reservation_rule['remaining_limits']))
            joining=tuple((f.request_id,f.remaining_output_limit) for f in expanded.decode_plan.frontiers
                          if f.request_id not in initial_ids)
            if any(limit is None for rid,limit in joining):
                return unavailable('missing_joining_output_limit')
            offset=predicted_running_reservation_offset(running_reservation_rule,trace,
                through_ms=dispatch,joining_limits=joining)
            evictable=query_evictable_tokens(expanded,trace,dispatch)
            if evictable is None:return unavailable('missing_evictable_cache_state')
            native_extra=additional_native_prefill_reserve(budget,
                native_offset=offset,evictable_tokens=evictable)
            if native_extra is None:return unavailable('native_reservation_offset')
        cached_proposal=None
        cohort_limit=max_requests
        if max_running_requests is not None:
            cohort_limit=min(cohort_limit,max(0,max_running_requests-len(budget.reservations)))
        if cache_backed_admission:
            cached_proposal=plan_prospective_cached_admission(shapes,budget=budget,
                evictable_tokens=evictable,native_offset=offset,preparation_ends=ends,
                dispatch_ms=dispatch,max_new_tokens=max_new_tokens,clip_max_new_tokens=clip_max_new_tokens,
                input_token_budget=input_token_budget,max_requests=cohort_limit,earlier_queue_empty=True)
            if cached_proposal is None:return unavailable('cached_admission_plan')
            selection=cached_proposal.selection
        else:
            selection=fit_prospective_prefill_cohort(shapes,preparation_ends=ends,
                dispatch_ms=dispatch,max_new_tokens=max_new_tokens,
                clip_max_new_tokens=clip_max_new_tokens,guaranteed_free_tokens=budget.free_tokens,
                running_token_reserve=native_extra,input_token_budget=input_token_budget,
                max_requests=min(cohort_limit,budget.free_request_slots),earlier_queue_empty=True)
        if selection is None:
            return unavailable('prospective_cohort_selection')
        if not selection.request_ids:
            # Advance to a real predicted completion, not a guessed free time.
            # The same fixed lease is evaluated, without another compute/HBM
            # search. No request may reuse its reservation before this event.
            if (running_reservation_rule is None and not accounting and not budget.capacity_publications
                    and not any(op.stage=='cache_evict' for op in expanded.operations)):
                break
            forecast=predict(snapshot,expanded,horizon_ms,progress_only=True)
            active=dict(budget.reservations)
            events=sorted((end,rid) for rid,work,end in forecast.work_completions
                          if work=='decode' and rid in active and rid in accounting)
            release_times=[end for end,rid in events]
            release_times.extend(end for end,key,freed in forecast.cache_release_events
                                 if key not in budget.completed_eviction_ids)
            release_times.extend(at for at,key,added in budget.capacity_publications
                                 if key not in budget.completed_publication_ids)
            if running_reservation_rule is not None:
                release_times.extend(at for at,key,ids in forecast.decode_output_events if at>dispatch)
            if not release_times:
                break
            release_at=max(dispatch,min(release_times))
            if release_at>=horizon_ms:
                break
            advanced=advance_to(expanded,budget,release_at)
            if advanced is None:return unavailable('capacity_replay_at_release')
            expanded,budget,dispatch=advanced
            continue  # Retry the same FCFS head with the updated physical pool.
        if cached_proposal is not None:
            if cached_proposal.eviction_request_tokens:
                if not callable(build_cache_eviction):
                    return unavailable('missing_cache_eviction_builder')
                key=f'prospective:{cohort_index}:cache-evict'
                eviction=build_cache_eviction(expanded,trace,dispatch,
                    cached_proposal.eviction_request_tokens,key)
                if eviction is None:return unavailable('cache_eviction_plan')
                if (not isinstance(eviction,Operation) or eviction.key!=key or eviction.stage!='cache_evict'
                        or eviction.credits or eviction.lanes!=('host_result',)
                        or eviction.ready_ms<dispatch
                        or not isinstance(eviction.service_shape,ProspectiveCacheEvictionShape)
                        or eviction.service_shape.requested_tokens!=cached_proposal.allocation_tokens):
                    raise ValueError('Allocator eviction must preserve the selected prompt work')
                expanded=replace(expanded,operations=(*expanded.operations,eviction))
                validate_projected_work(snapshot,expanded.operations)
                upstream=predict(snapshot,replace(expanded,decode_plan=None),horizon_ms)
                dispatch=next(end for opkey,start,end in upstream.intervals if opkey==key)
                if dispatch>=horizon_ms:break
                advanced=advance_to(expanded,budget,dispatch)
                if advanced is None:return unavailable('capacity_replay_after_eviction')
                expanded,budget,dispatch=advanced
                if dispatch>=horizon_ms:break
                trace=predict(snapshot,expanded,horizon_ms,progress_only=True)
                evictable=query_evictable_tokens(expanded,trace,dispatch)
                if evictable is None:return unavailable('evictable_state_after_eviction')
                # The native selected cohort stays fixed across allocation.
                # Recheck backing only; do not re-run selection after the wait.
            budget=reserve_prospective_cached_cohort(budget,shapes,cached_proposal,
                evictable_tokens=evictable,max_new_tokens=max_new_tokens,clip_max_new_tokens=clip_max_new_tokens)
            if budget is None:
                return unavailable('cached_cohort_physical_reservation')
        else:
            budget=reserve_prospective_cohort(budget,shapes,selection,
                max_new_tokens=max_new_tokens,clip_max_new_tokens=clip_max_new_tokens)
        bundles=build_prospective_prefill_bundles(shapes,selection,total_layers=total_layers,
            fitted_layers=expanded.executable.prefill_layers,kv_key=kv_key,
            dispatch_ms=dispatch,compute_ready_ms=compute_ready_ms,
            prefix=f'prospective:{cohort_index}:prefill')
        bundles=time_frozen_prefill_operations(bundles,query_prefill)
        if bundles is None:
            return unavailable('prefill_bundle_timing')
        handoff=build_prospective_prefill_handoff(frontiers,shapes,profiles,
            prefill_operations=bundles,existing_kv_key=existing_kv_key,new_kv_key=kv_key,
            prefix=f'prospective:{cohort_index}:handoff',separate_commit_gate=separate_commit_gate)
        if handoff is None:
            return unavailable('prefill_decode_handoff')
        simulated,operations,gates,emitted=handoff
        measured=time_handoff(operations)
        if measured is None:
            return unavailable('handoff_timing')
        measured=tuple(measured)
        if (len(measured)!=len(operations) or any(
                replace(new,duration_ms=old.duration_ms)!=old for old,new in zip(operations,measured))):
            raise ValueError('Handoff timing cannot change work or dependencies')
        plan=expanded.decode_plan
        ids=tuple(f.request_id for f in simulated)
        if set(ids)&{f.request_id for f in plan.frontiers}:
            raise ValueError('Duplicate prospective decode admission')
        plan=replace(plan,frontiers=(*plan.frontiers,*simulated),
            profiles=(*plan.profiles,*((rid,profiles[rid]) for rid in ids)),
            join_dependencies=(*plan.join_dependencies,*gates),
            completed_tokens=(*plan.completed_tokens,*emitted),
            kv_keys=(*plan.kv_keys,*((rid,kv_key) for rid in ids)))
        expanded=replace(expanded,operations=(*expanded.operations,*bundles,*measured),decode_plan=plan)
        validate_decode_join_contract(expanded.operations,plan)
        trace=predict(snapshot,replace(expanded,decode_plan=None),horizon_ms)
        ends={key:end for key,start,end in trace.intervals}
        dispatch=max(ends[key] for rid,dependencies in gates for key in dependencies)
        shapes=tuple(s for s in shapes if s.request_id not in ids)
        cohort_index+=1
    if (budget.capacity_publications or expanded.decode_plan.frontiers
            or any(op.stage=='cache_evict' for op in expanded.operations)):
        if cache_backed_admission and callable(build_cache_eviction):
            resolved=resolve_prospective_decode_capacity(snapshot,expanded,budget,horizon_ms=horizon_ms,
                completion_accounting=accounting,build_cache_eviction=build_cache_eviction)
            if resolved is None:return unavailable('final_decode_capacity_resolution')
            expanded,budget,_=resolved
        else:
            trace=predict(snapshot,expanded,horizon_ms,progress_only=True)
            budget=advance_prospective_capacity_events(budget,trace,
                through_ms=horizon_ms,completion_accounting=accounting)
            if budget is None:return unavailable('final_capacity_replay')
    return expanded,budget,shapes


def build_following_input_preparation(frontiers, retrieval_operations, request_ids):
    """Link only observed ANN requests to their actual collection completion.

    Uses an age-zero turnaround prior after the dependency, not a future
    arrival or a claim about yet-unknown prompt contents/cache hits.
    """
    by_id={f.request_id:f for f in frontiers}
    ids=tuple(request_ids)
    if len(ids)!=len(set(ids)) or not set(ids)<=set(by_id):
        raise ValueError('Unique observed retrieval members required')
    result=[];gates=[]
    for rid in ids:
        frontier=by_id[rid]
        if frontier.stage not in ('ann_pending','ann_inflight'):
            raise ValueError('Post-retrieval preparation needs a retrieval frontier')
        matches=tuple(op.key for op in retrieval_operations if op.stage=='collect'
            and any(c.request==rid and c.work=='ann' and c.fraction==1. for c in op.credits))
        if len(matches)!=1:
            raise ValueError('One actual collection gate per retrieval request required')
        gates.append((rid,matches[0]))
        if frontier.preparation_follows is False:
            continue
        if (frontier.preparation_follows is not True or type(frontier.query_rows) is not int
                or frontier.query_rows<=0):
            return None
        result.append(Operation('input-prepare:'+rid,'prepare',None,('external-prepare:'+rid,),
            (Credit(rid,'prepare'),),depends_on=matches,
            service_shape=('input_preparation_residual',rid,frontier.query_rows,0.)))
    return tuple(result),tuple(gates)


def time_frozen_input_preparation(operations, frozen_profile):
    """Price the captured external residual without recapturing age or shape."""
    result=[]
    for op in operations:
        shape=op.service_shape
        if (op.stage!='prepare' or not isinstance(shape,tuple) or len(shape)!=4
                or shape[0]!='input_preparation_residual'
                or type(shape[2]) is not int or shape[2]<=0
                or op.credits!=(Credit(shape[1],'prepare'),)):
            raise ValueError('Captured input preparation shape and credit required')
        nonnegative(shape[3],'captured preparation age')
        duration=frozen_profile.query_remaining_ms(shape[2],elapsed_ms=shape[3])
        if duration is None:
            return None
        nonnegative(duration,'input preparation remaining duration')
        if duration==0:
            raise ValueError('Unfinished preparation requires positive duration')
        result.append(replace(op,duration_ms=duration))
    return tuple(result)


def time_frozen_joint_operations(operations, *, cpu_profile, gpu_profile,
                                 collection_profile, input_preparation_profile,
                                 prefill_timing, handoff_timing, decode_timing=None):
    """Price one captured mixed graph; missing any stage leaves it unscored.

    Callbacks close over same-event profiles, partition and release observations.
    They may only fill durations. No passthrough of unknown or already-timed
    stages: their timing must also belong to this proposal. Joining decode is
    evolved separately by its fixed-proposal plan during prediction.
    """
    operations=tuple(operations)
    groups={name:[] for name in ('retrieval','prepare','prefill','handoff','decode')}
    categories={'cpu_retrieval':'retrieval','ann':'retrieval','collect':'retrieval',
                'prepare':'prepare','prefill':'prefill',
                'prefill_result':'handoff','prefill_merge':'handoff','prefill_commit':'handoff','decode':'decode'}
    for i,op in enumerate(operations):
        category=categories.get(op.stage)
        if category is None:
            raise ValueError('Unpriced joint stage: '+op.stage)
        groups[category].append((i,op))
    result=list(operations)
    for category,entries in groups.items():
        if not entries:
            continue
        original=tuple(op for _,op in entries)
        if category=='retrieval':
            priced=time_frozen_retrieval_operations(original,cpu_profile=cpu_profile,
                gpu_profile=gpu_profile,collection_profile=collection_profile)
        elif category=='prepare':
            if input_preparation_profile is None:
                return None
            priced=time_frozen_input_preparation(original,input_preparation_profile)
        else:
            callback={'prefill':prefill_timing,'handoff':handoff_timing,'decode':decode_timing}[category]
            if callback is None:
                return None
            priced=callback(original)
        if priced is None:
            return None
        priced=tuple(priced)
        if len(priced)!=len(original):
            raise ValueError('Timing changed stage work count')
        for (i,op),timed in zip(entries,priced):
            if not isinstance(timed,Operation) or replace(timed,duration_ms=op.duration_ms)!=op:
                raise ValueError('Timing changed frozen joint work')
            if timed.duration_ms is None:
                return None
            nonnegative(timed.duration_ms,'joint stage duration')
            if timed.duration_ms==0 and timed.credits:
                raise ValueError('Credited joint work needs positive service')
            result[i]=timed
    return tuple(result)


def link_rag_stages(*, retrieval: tuple[Operation, ...] = (),
                    prepare: Operation | None = None,
                    prefill: tuple[Operation, ...] = (),
                    merge: Operation | None = None,
                    decode: tuple[Operation, ...] = ()) -> tuple[Operation, ...]:
    """Link profiled operations, retaining their shape, backing and wait gates.

    Retrieval branches may run in parallel; preparation waits for ALL of them.
    Sequential bundles and decode iterations retain order. If work is already
    done, omit that stage rather than supply a made-up zero-time measurement.
    A new prefill or completed-but-unmerged frontier requires an explicit merge
    operation before decode. Caller supplies its measured duration and any
    external dependency (e.g. completion of the current compatible decode).
    This does not construct profiles, predict future arrivals, or choose leases.
    """
    if retrieval and prepare is None:
        raise ValueError("Retrieval requires an explicit input preparation join")
    if prefill and decode and merge is None:
        raise ValueError("Prefill-to-decode requires an explicit merge gate")
    if any(op.stage != 'prefill' for op in prefill):
        raise ValueError("Expected profiled prefill bundles")
    if any(op.stage != 'decode' for op in decode):
        raise ValueError("Expected profiled decode iterations")
    result = list(retrieval)
    predecessors = tuple(op.key for op in retrieval)

    def append(op):
        nonlocal predecessors
        deps = tuple(dict.fromkeys((*op.depends_on, *predecessors)))
        result.append(replace(op, depends_on=deps))
        predecessors = (op.key,)

    if prepare is not None:
        append(prepare)
    for op in prefill:
        append(op)
    if merge is not None:
        append(merge)
    for op in decode:
        append(op)
    keys = [op.key for op in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate stage operation key")
    return tuple(result)


class Planner(Protocol):
    """Callbacks may read profiles but must not commit or search recursively.

    New adapters should make operations() a timing-free safety description,
    using duration_ms=None and optional time_operations(snapshot, projection)
    for Full-only timing. The timing hook returns the SAME ordered operations
    with durations filled; it must not change any other field or invoke search.
    Independent never calls that hook. Existing timed adapters are supported
    for compatibility but are not proof of prediction-free ablation behavior.
    operations() must use the executable batch/layer limits passed to it, and
    profile the resulting shapes (not the originally proposed shape). It may
    enumerate repeated work under those fixed limits, only for snapshot reqs.
    Missing profiles/unsupported memory should yield an unsupported projection,
    never a fabricated zero service time.
    An optional joining_decode_plan(snapshot, executable, memory) returns a
    frozen description (or None for the existing static path), not a simulated
    trace. It must use these fitted limits and cannot search/replan/dispatch.
    Both modes invoke it during projection; Independent never queries its
    service callback or evaluates progress.
    """
    def choose_compute(self, snapshot: Snapshot) -> ComputeLease: ...
    def derive_demand(self, snapshot: Snapshot, proposed: ComputeLease) -> object: ...
    def choose_memory(self, snapshot: Snapshot, demand: object) -> MemoryLease: ...
    def operations(self, snapshot: Snapshot, executable: ComputeLease,
                   memory: MemoryLease) -> tuple[Operation, ...] | None: ...


class RuntimeJointPlanner:
    """Single-event adapter to the concrete scheduler proposal methods.

    Snapshot already contains a current-only safe continuation. Context maps
    must contain same-event frozen inputs; work/timing builders operate on that
    captured state and may not recapture, dispatch or search. Native activation
    is deliberately outside this adapter, after the decision is revalidated.
    """
    def __init__(self, scheduler, snapshot, *, mode, compute_inputs, demand_inputs, memory_inputs,
                 operations_builder=None, timing_builder=None, decode_builder=None,
                 joint_work_builder=None, prospective_inputs_builder=None):
        self.scheduler = scheduler
        if mode not in ('full','independent_adaptive'):
            raise ValueError('Explicit runtime joint policy required')
        self.mode = mode
        self.snapshot = snapshot
        self.compute_inputs = dict(compute_inputs)
        # The target partition is unknown until choose_compute returns. Freeze
        # this mapping now, then resolve the selected lease rather than using
        # a caller's anticipated target in transition profiling.
        allocations = self.compute_inputs.get('allocation_by_stream')
        if not allocations:
            raise ValueError('Captured allocation-to-stream mapping required')
        self.compute_inputs['allocation_by_stream'] = dict(allocations)
        self.demand_inputs = dict(demand_inputs)
        # Capacity prediction is shared safety work, including Independent.
        # This callback closes over the same frozen observation/profiles; it
        # must not choose compute, plan memory, or inspect future arrivals.
        self.decode_growth_forecast = self.demand_inputs.pop('decode_growth_forecast', None)
        if self.decode_growth_forecast is not None:
            if not callable(self.decode_growth_forecast) or 'decode_growth_tokens' in self.demand_inputs:
                raise ValueError('Choose a selected-proposal growth forecast or fixed growth, not both')
        self.memory_inputs = dict(memory_inputs)
        if 'mode' in self.memory_inputs and self.memory_inputs['mode'] != mode:
            raise ValueError('Memory policy differs from joint admission policy')
        self.memory_inputs['mode'] = mode
        self.operations_builder = operations_builder
        self.timing_builder = timing_builder
        self.decode_builder = decode_builder
        if joint_work_builder is not None:
            if not callable(joint_work_builder) or operations_builder is not None or decode_builder is not None:
                raise ValueError('Use one joint graph builder or separate legacy builders')
        elif not callable(operations_builder):
            raise ValueError('Projected work builder required')
        self.joint_work_builder=joint_work_builder
        self.prospective_inputs_builder=prospective_inputs_builder
        self.joint_work_cache=[]
        self.phase = 0
        self.proposed = None
        self.diagnostic = None
        self.demand = None
        self.target_unavailable = False
        self.target_stream_idx = None

    def _check(self, snapshot):
        if snapshot is not self.snapshot:
            raise ValueError('Runtime planner belongs to one captured snapshot')

    def choose_compute(self, snapshot):
        self._check(snapshot)
        if self.phase != 0:
            raise ValueError('Compute selection may run only once per event')
        self.phase = 1  # Failed calls must not be silently retried.
        self.proposed, self.diagnostic = self.scheduler._choose_rag_joint_compute(**self.compute_inputs)
        self.target_unavailable = self.proposed is None
        if self.target_unavailable:
            self.proposed = replace(snapshot.compute, ann_batch=0, prefill_layers=0, prefill_contract=None)
        else:
            matches = [idx for idx, allocation in self.compute_inputs['allocation_by_stream'].items()
                       if allocation == self.proposed.allocation]
            if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 0:
                raise ValueError('Selected compute allocation must identify exactly one stream')
            self.target_stream_idx = matches[0]
            if (self.diagnostic is not None and 'selected_stream_idx' in self.diagnostic
                    and self.diagnostic['selected_stream_idx'] != self.target_stream_idx):
                raise ValueError('Compute diagnostic disagrees with selected allocation')
            if ('target_stream_idx' in self.memory_inputs
                    and self.memory_inputs['target_stream_idx'] != self.target_stream_idx):
                raise ValueError('Preselected memory target differs from compute proposal')
        return self.proposed

    def derive_demand(self, snapshot, proposed):
        self._check(snapshot)
        if self.phase != 1 or proposed is not self.proposed:
            raise ValueError('Demand must follow the single selected compute proposal')
        self.phase = 2
        if not self.target_unavailable:
            if self.decode_growth_forecast is not None:
                growth=self.decode_growth_forecast(snapshot,proposed,self.diagnostic)
                if growth is None:
                    self.target_unavailable=True
                    return None
                if (not isinstance(growth, tuple)
                        or any(not isinstance(row, tuple) or len(row) != 2
                               or not isinstance(row[0], str) or not row[0]
                               or type(row[1]) is not int or row[1] < 0 for row in growth)
                        or len({rid for rid, _ in growth}) != len(growth)
                        or not {rid for rid, _ in growth} <= {r.key for r in snapshot.requests}):
                    raise ValueError('Selected-proposal KV growth requires observed request/token pairs')
                self.demand_inputs['decode_growth_tokens']=growth
                self.memory_inputs['decode_growth_tokens']=growth
            demand_inputs = dict(self.demand_inputs)
            policies = demand_inputs.pop('conservative_cache_reserve_by_allocation', None)
            if policies is not None:
                policies = dict(policies)
                if proposed.allocation not in policies:
                    self.target_unavailable = True
                    return None
                demand_inputs['conservative_cache_reserve'] = policies[proposed.allocation]
            self.demand = self.scheduler._derive_rag_selected_memory_demand(
                proposed_compute=proposed, diagnostic=self.diagnostic, **demand_inputs)
            self.target_unavailable = self.demand is None
        return self.demand

    def choose_memory(self, snapshot, demand):
        self._check(snapshot)
        if self.phase != 2 or demand is not self.demand:
            raise ValueError('Memory planning must consume the one derived demand')
        self.phase = 3
        memory = None
        if not self.target_unavailable:
            memory_inputs = {**self.memory_inputs, 'target_stream_idx': self.target_stream_idx}
            memory = self.scheduler._plan_rag_target_memory_lease(proposed_compute=self.proposed,
                diagnostic=self.diagnostic, derived_demand=demand, **memory_inputs)
        return memory if memory is not None else MemoryLease(snapshot.memory.allocation,0,0,supported=False)

    def operations(self, snapshot, executable, memory):
        self._check(snapshot)
        if self.joint_work_builder is not None:
            graph=self._joint_work(snapshot,executable,memory)
            return None if graph is None else graph[0]
        return self.operations_builder(snapshot, executable, memory)

    def joining_decode_plan(self, snapshot, executable, memory):
        self._check(snapshot)
        if self.joint_work_builder is not None:
            graph=self._joint_work(snapshot,executable,memory)
            return None if graph is None else graph[1]
        return self.decode_builder(snapshot, executable, memory) if self.decode_builder else None

    def _joint_work(self, snapshot, executable, memory):
        self._check(snapshot)
        for captured_compute,captured_memory,graph in self.joint_work_cache:
            if executable==captured_compute and memory is captured_memory:
                return graph
        # One engineering failure-path trace. Function/line identifiers only;
        # no request contents, locals, guessed shapes or safety bypass.
        import sys
        receipt=getattr(self.scheduler,'rag_joint_bootstrap_receipt',{})
        previous=sys.getprofile(); returns=[]
        diagnostic=(engineering_failure_trace_enabled()
                    and receipt.get('engineering_only') is True
                    and receipt.get('formal_ready') is False and previous is None)
        def capture_return(frame,event,value):
            code=frame.f_code
            if (event=='return' and value is None and len(returns)<32
                    and code.co_filename.endswith(('/multiplexing.py','/full_lease.py'))
                    and code.co_name.startswith(('_build','_capture','build_','capture_'))):
                returns.append((code.co_name,frame.f_lineno))
        if diagnostic:sys.setprofile(capture_return)
        try:
            graph=self.joint_work_builder(snapshot,executable,memory)
        finally:
            if diagnostic:sys.setprofile(previous)
        if graph is None and diagnostic:
            self.missing_work_reason='missing shape profile; unsupported returns='+repr(returns)
        if graph is not None:
            if not isinstance(graph,tuple) or len(graph) not in (2,3):
                raise ValueError('Joint graph requires operations and decode plan')
            graph=(tuple(graph[0]),*graph[1:])
        # Cache unsupported results too: missing observations cannot authorize
        # a second capture under the same event. Keep holds a different memory
        # object and must be built from its own fitted limits/backing.
        self.joint_work_cache.append((executable,memory,graph))
        return graph

    def time_operations(self, snapshot, projection):
        self._check(snapshot)
        return self.timing_builder(snapshot, projection)

    def record_prediction_unavailable(self, phase, error):
        record=dict(phase=phase, reason=str(error))
        timing=getattr(self.scheduler,'rag_joint_last_timing_failure',None)
        if timing:
            record['timing_failure']=tuple(timing)
        self.scheduler.rag_joint_last_prediction_error = record

    def record_target_projection(self, projection):
        """Publish bounded engineering diagnostics for the selected proposal.

        The record deliberately excludes request identities, shapes, profile
        values and allocator contents. It explains why the one selected
        compute/HBM proposal could not be admitted without changing either
        the proposal or the shared safety projection.
        """
        self._check(self.snapshot)
        if not isinstance(projection, Projection):
            raise ValueError('Target projection diagnostic requires a projection')
        self.scheduler.rag_joint_last_target_projection = dict(
            supported=projection.supported,
            reason=projection.reason,
            compute_available=bool(self.diagnostic and self.diagnostic.get('available')),
            compute_reason=(self.diagnostic.get('reason')
                            if isinstance(self.diagnostic, dict) else None),
            demand_available=self.demand is not None,
            target_stream_idx=self.target_stream_idx,
            proposed_ann_batch=projection.compute.ann_batch,
            proposed_prefill_layers=projection.compute.prefill_layers,
            fitted_ann_batch=projection.executable.ann_batch,
            fitted_prefill_layers=projection.executable.prefill_layers,
            memory_supported=projection.memory.supported,
        )

    def expand_forecast(self, snapshot, projection, horizon_ms):
        """Full-only continuation simulation, never a second proposal or dispatch.

        Inputs must be frozen for this fitted lease, including its actual KV
        budget. The original projection remains the executable admission;
        prospective operations exist only in the common-H progress forecast.
        """
        self._check(snapshot)
        if self.mode!='full':
            raise ValueError('Independent-adaptive cannot expand a joint forecast')
        if self.prospective_inputs_builder is None:
            return projection
        if isinstance(projection,Projection) and projection.decode_plan is None:
            # The generation adapter explicitly reports no active/joining
            # decode under these fitted limits (e.g. a zero-layer keep).
            # Its existing work/idle progress is still a valid comparator.
            # A preparation chain, however, must retain its joining model.
            if any(op.stage=='prepare' for op in projection.operations):
                raise PredictionUnavailable('Prepared request chain lacks joining decode plan')
            return projection
        import sys
        receipt=getattr(self.scheduler,'rag_joint_bootstrap_receipt',{})
        previous=sys.getprofile(); input_returns=[]
        trace_inputs=(engineering_failure_trace_enabled()
                      and receipt.get('engineering_only') is True
                      and receipt.get('formal_ready') is False and previous is None)
        def trace_input_returns(frame,event,value):
            if (event=='return' and value is None and len(input_returns)<32
                    and frame.f_code.co_filename.endswith(('/multiplexing.py','/full_lease.py'))
                    and frame.f_code.co_name.startswith(('_build','_capture','build_',
                        'capture_','make_','describe_','derive_','prospective'))):
                input_returns.append((frame.f_code.co_name,frame.f_lineno))
        if trace_inputs:sys.setprofile(trace_input_returns)
        try:
            inputs=self.prospective_inputs_builder(snapshot,projection,horizon_ms)
        finally:
            if trace_inputs:sys.setprofile(previous)
        if inputs is None:
            reason=getattr(self.scheduler,'rag_joint_prospective_input_failure',None)
            reason_detail=('; reason='+str(reason)) if reason else ''
            detail=('; unsupported returns='+repr(input_returns)) if trace_inputs else ''
            raise PredictionUnavailable(
                'Missing fitted prospective forecast inputs'+reason_detail+detail)
        inputs=dict(inputs)
        if set(inputs)&{'snapshot','projection','horizon_ms'}:
            raise ValueError('Prospective inputs cannot replace the common decision or horizon')
        # Bounded engineering-only diagnostic: identify the first unsupported
        # return without serializing live requests, callbacks or allocator data.
        receipt=getattr(self.scheduler,'rag_joint_bootstrap_receipt',{})
        debug=(engineering_failure_trace_enabled()
               and receipt.get('engineering_only') is True and receipt.get('formal_ready') is False
               and not hasattr(self.scheduler,'rag_joint_forecast_debug'))
        import sys
        old_profile=sys.getprofile()
        events=[]
        def trace_returns(frame,event,value):
            if (event=='return' and value is None and frame.f_code.co_filename==__file__
                    and frame.f_code.co_name.startswith(('build_','derive_','expand_',
                        'predict','plan_','resolve_','advance_','fit_','reserve_','additional_'))
                    and len(events)<32):
                events.append((frame.f_code.co_name,frame.f_lineno))
        if debug and old_profile is None:
            sys.setprofile(trace_returns)
        failure_reasons=[]
        try:
            expanded=expand_prospective_prefill_cohorts(snapshot,projection,
                horizon_ms=horizon_ms,failure_reasons=failure_reasons,**inputs)
        finally:
            if debug and old_profile is None:
                sys.setprofile(old_profile)
        if debug and expanded is None:
            self.scheduler.rag_joint_forecast_debug=dict(
                schema='engineering-unsupported-return-v1',returns=events,
                note='one failed forecast; function/line only; no measured performance evidence')
        if expanded is None:
            reason=failure_reasons[-1] if failure_reasons else 'unknown'
            counts=getattr(self.scheduler,'rag_joint_forecast_failure_counts',None)
            if counts is None:
                counts={};self.scheduler.rag_joint_forecast_failure_counts=counts
            counts[reason]=counts.get(reason,0)+1
            detail=('; reason='+failure_reasons[-1]) if failure_reasons else ''
            raise PredictionUnavailable(
                'Prospective timing or physical capacity unavailable'+detail)
        return expanded[0]


def validate_projected_work(snapshot: Snapshot, operations):
    """Timing-free graph checks shared by both scheduling policies.

    No profile query, event simulation or utility evaluation. Unknown timing
    is allowed; unknown requests, dependencies or cyclic work are not.
    """
    refs = set()
    requests = set()
    for request in snapshot.requests:
        if request.key in requests:
            raise ValueError('Duplicate snapshot request')
        requests.add(request.key)
        for work in request.work:
            key = (request.key, work.key)
            if key in refs:
                raise ValueError('Duplicate reference-work key')
            refs.add(key)
    keys = {op.key for op in operations}
    if len(keys) != len(operations):
        raise ValueError('Duplicate projected operation')
    children = {key: [] for key in keys}
    degrees = {}
    for op in operations:
        if not isinstance(op.key, str) or not op.key:
            raise ValueError('Explicit projected operation identity required')
        if len(set(op.lanes)) != len(op.lanes):
            raise ValueError('Duplicate projected lane')
        nonnegative(op.ready_ms, 'projected readiness')
        nonnegative(op.compute_ready_ms, 'projected compute readiness')
        if op.duration_ms is not None:
            nonnegative(op.duration_ms, 'supplied service duration')
            if op.duration_ms == 0 and op.credits:
                raise ValueError('Credited work cannot have zero service duration')
        dependencies = set(op.depends_on)
        if not dependencies <= keys:
            raise ValueError('Missing projected dependency')
        degrees[op.key] = len(dependencies)
        for parent in dependencies:
            children[parent].append(op.key)
        for credit in op.credits:
            if (credit.request, credit.work) not in refs:
                raise ValueError('Unknown/future projected request or work')
            if not math.isfinite(credit.fraction) or not 0 <= credit.fraction <= 1:
                raise ValueError('Invalid projected reference fraction')
    ready = [key for key, degree in degrees.items() if degree == 0]
    visited = 0
    while ready:
        key = ready.pop(); visited += 1
        for child in children[key]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(operations):
        raise ValueError('Cyclic projected dependency')


def validate_decode_join_contract(operations, plan):
    """Validate new membership/token credit without running decode prediction."""
    by_key = {op.key: op for op in operations}
    profiles = dict(plan.profiles)
    joins = dict(plan.join_dependencies)
    emitted_by_id = dict(plan.completed_tokens)
    for frontier in plan.frontiers:
        rid = frontier.request_id
        if (frontier.stage not in ('decode','prefill_active','prefill_queued','prefill_merge_pending')
                or frontier.generation_follows is not True
                or type(frontier.prompt_tokens) is not int or frontier.prompt_tokens <= 0):
            raise ValueError('Observed generation frontier with known prompt required')
        deps = tuple(joins[rid])
        if frontier.stage != 'decode' and not deps:
            raise ValueError('New decode membership requires an explicit upstream gate')
        emitted = emitted_by_id[rid]
        remaining = remaining_decode_tokens(frontier, profiles[rid])
        if (type(emitted) is not int or emitted not in (0,1) or emitted > remaining
                or (frontier.stage == 'decode' and emitted)):
            raise ValueError('Invalid pre-decode token credit')
        if emitted:
            ancestors = set(deps)
            pending = list(deps)
            while pending:
                for key in by_key[pending.pop()].depends_on:
                    if key not in ancestors:
                        ancestors.add(key); pending.append(key)
            fraction = max((c.fraction for key in ancestors for c in by_key[key].credits
                            if c.request == rid and c.work == 'decode'), default=0.)
            if remaining <= 0 or not math.isclose(fraction, emitted/remaining):
                raise ValueError('Upstream result must credit exactly its emitted token')


def safe_project(snapshot: Snapshot, proposed: ComputeLease,
                 memory: MemoryLease, planner: Planner) -> Projection:
    """One shared, non-searching projection for target and continuation."""
    for value in (proposed.ann_batch, proposed.prefill_layers,
                  memory.ann_batch_cap, memory.prefill_layers_cap):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Execution limits must be nonnegative integers")
    executable = replace(proposed,
        ann_batch=min(proposed.ann_batch, memory.ann_batch_cap),
        prefill_layers=min(proposed.prefill_layers, memory.prefill_layers_cap))
    if not memory.supported:
        return Projection(proposed, memory, executable, (), False, "unsupported memory")
    ops = planner.operations(snapshot, executable, memory)
    if ops is None:
        return Projection(proposed, memory, executable, (), False,
                          getattr(planner,'missing_work_reason','missing shape profile'))
    validate_projected_work(snapshot, ops)
    keys = dict(memory.ready)
    if len(keys) != len(memory.ready):
        raise ValueError("Duplicate backing key")
    for value in keys.values():
        if math.isnan(value) or value < 0:
            raise ValueError("Invalid backing readiness")
    transitions = dict(memory.transition_ready)
    if len(transitions) != len(memory.transition_ready):
        raise ValueError("Duplicate transition key")
    for key, value in transitions.items():
        if key not in keys or math.isnan(value) or value < 0:
            raise ValueError("Invalid transition readiness")
    for op in ops:
        if op.stage == "prefill" and (op.kv_key is None or op.kv_key not in keys):
            return Projection(proposed, memory, executable, (), False, "unguaranteed KV")
        if op.kv_key is not None and op.kv_key not in keys:
            # Independent does not call predict(), so backing validation must
            # be shared here rather than occurring only during Full scoring.
            return Projection(proposed, memory, executable, (), False, "unguaranteed KV")
        if any(k not in keys for k in op.memory_keys):
            return Projection(proposed, memory, executable, (), False, "unguaranteed backing")
    build_decode = getattr(planner, 'joining_decode_plan', None)
    decode_plan = build_decode(snapshot, executable, memory) if build_decode else None
    if decode_plan is not None:
        if not isinstance(decode_plan, JoiningDecodePlan) or not callable(decode_plan.query_service):
            raise ValueError('Fixed joining decode plan required')
        ids = {f.request_id for f in decode_plan.frontiers}
        if len(ids) != len(decode_plan.frontiers) or not ids <= {r.key for r in snapshot.requests}:
            raise ValueError('Only unique observed decode members allowed')
        for entries in (decode_plan.profiles, decode_plan.join_dependencies,
                        decode_plan.completed_tokens, decode_plan.kv_keys):
            if len(dict(entries)) != len(entries) or set(dict(entries)) != ids:
                raise ValueError('Decode plan identity mismatch')
        if any(key not in keys for rid, key in decode_plan.kv_keys):
            return Projection(proposed, memory, executable, (), False, 'unguaranteed decode KV')
        growth = dict(decode_plan.kv_growth_guarantees)
        if len(growth) != len(decode_plan.kv_growth_guarantees) or not set(growth) <= ids:
            raise ValueError('Decode growth guarantee identity mismatch')
        for count, key in growth.values():
            if type(count) is not int or count < 0 or not isinstance(key, str) or not key:
                raise ValueError('Explicit finite-prefix KV guarantee required')
            if key not in keys:
                return Projection(proposed, memory, executable, (), False,
                                  'unguaranteed decode growth KV')
        op_keys = {op.key for op in ops}
        if any(key not in op_keys for rid, deps in decode_plan.join_dependencies for key in deps):
            raise ValueError('Missing upstream decode join dependency')
        if any('decode' in op.lanes or op.stage == 'decode' for op in ops):
            raise ValueError('Cannot combine static and joining decode operations')
        nonnegative(decode_plan.compute_ready_ms, 'decode compute readiness')
        validate_decode_join_contract(ops, decode_plan)
    return Projection(proposed, memory, executable, ops, decode_plan=decode_plan)


def make_safe_continuation(snapshot: Snapshot, planner: Planner) -> Projection:
    # No compute selection, demand derivation or elastic memory planner call.
    return safe_project(snapshot, snapshot.compute, snapshot.memory, planner)


def price_projected_operations(snapshot: Snapshot, projection: Projection, planner: Planner) -> Projection:
    """Full-only timing of an already safety-projected, fixed work graph.

    New runtime adapters may emit duration=None during shared projection and
    implement time_operations. Only durations may change: no altered work,
    dependencies, backing gates, order, credit, fitting, or compute search.
    Legacy planners with timed operations remain compatible without this hook.
    """
    # An idle continuation has no fixed-work operations to price.  It may
    # still carry a decode plan whose service times are queried by predict(),
    # so do not require an allocation-level fixed-work profile here.
    if not projection.operations:
        return projection
    timing = getattr(planner, 'time_operations', None)
    if timing is None:
        if any(op.duration_ms is None for op in projection.operations):
            raise PredictionUnavailable('Projected operation durations unavailable')
        return projection
    priced = timing(snapshot, projection)
    if priced is None:
        stages = tuple(sorted({op.stage for op in projection.operations if op.duration_ms is None}))
        raise PredictionUnavailable('Missing fixed-work timing profile; untimed stages=' + repr(stages))
    priced = tuple(priced)
    if len(priced) != len(projection.operations):
        raise ValueError('Timing cannot change the projected operation count')
    for original, measured in zip(projection.operations, priced):
        if (not isinstance(measured, Operation)
                or replace(measured, duration_ms=original.duration_ms) != original):
            raise ValueError('Timing cannot change safety-projected work or dependencies')
        if measured.duration_ms is None:
            raise PredictionUnavailable('Missing fixed-work operation duration')
        nonnegative(measured.duration_ms, 'profiled operation duration')
    return replace(projection, operations=priced)


@dataclass(frozen=True)
class Prediction:
    progress: float
    # Predicted intervals may finish outside H; progress is clipped to H.
    intervals: tuple[tuple[str, float, float], ...]
    # Earliest complete reference unit within H, not implicit memory release.
    # Entries are (request ID, work kind, completion offset in common H).
    work_completions: tuple[tuple[str, str, float], ...] = ()
    # Allocation occurs when the iteration starts, even if its token finishes
    # beyond H. Counts are absolute stored positions for unit-page replay.
    kv_allocation_events: tuple = ()
    # (completion offset, operation identity, released cache slots).
    cache_release_events: tuple = ()
    # (completion offset, operation identity, request IDs emitting one token).
    # Allocation/start is not emission; first-token result processing counts.
    decode_output_events: tuple = ()


def predicted_running_reservation_offset(rule, prediction, *, through_ms,
                                         joining_limits=()):
    """Native aggregate heuristic after actual predicted output completions.

    Existing members begin at the captured remaining output limits. Newly
    joining, already observed requests enter only at their first-token event.
    This is not physical release, a guaranteed reservation, or an EOS oracle.
    """
    nonnegative(through_ms,'reservation observation boundary')
    ratio=rule['ratio'];clip=rule['clip_max_new_tokens']
    if (type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio<0
            or type(clip) is not int or clip<=0):
        raise ValueError('Valid frozen native reservation rule required')
    initial=tuple(rule['remaining_limits']);joining=tuple(joining_limits)
    remaining=dict(initial);waiting=dict(joining)
    if (len(remaining)!=len(initial) or len(waiting)!=len(joining)
            or set(remaining)&set(waiting)
            or any(not isinstance(rid,str) or not rid or type(left) is not int or left<0
                   for rid,left in (*initial,*joining))):
        raise ValueError('Distinct observed reservation members and output limits required')
    seen=set()
    for at,key,ids in sorted(prediction.decode_output_events,key=lambda event:(event[0],event[1])):
        if not math.isfinite(at) or at<0 or not isinstance(key,str) or not key or key in seen:
            raise ValueError('Unique timed output completion events required')
        seen.add(key)
        if len(set(ids))!=len(ids):raise ValueError('One output per request per iteration required')
        if at>through_ms:continue
        for rid in ids:
            if rid in waiting:remaining[rid]=waiting.pop(rid)
            if rid not in remaining:
                raise ValueError('Output event has no observed reservation contract')
            if remaining[rid]<=0:raise ValueError('Output event exceeds declared request limit')
            remaining[rid]-=1
    return sum(min(left,clip)*ratio for left in remaining.values())


def predict(snapshot: Snapshot, projection: Projection, horizon_ms: float,
            *, compute_transition: bool = True, memory_transition: bool = True,
            progress_only: bool = False) -> Prediction:
    """Greedy discrete events under the proposal's fixed lane/FIFO ordering.

    Each named lane is serial; multiple lanes may overlap. Caller-provided
    service times must reflect the fixed co-execution shape. CPU worker lanes,
    ANN/decode streams, merge and preparation are explicit, not summed globally.
    An op becomes eligible after ALL dependencies (max, not sum). Among eligible
    operations select earliest feasible start, breaking ties by supplied order.
    Predict no future arrivals and call no optimizer. Logical duplicate work is
    credited by maximum fraction, never by CPU wall time or duplicate execution.
    progress_only retains complete input/DAG validation but stops simulation
    when no remaining work can start inside H. Its intervals are a prefix,
    not a complete diagnostic timeline. Online admission needs only progress.
    """
    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError("Positive finite common horizon required")
    if not projection.supported:
        raise ValueError("Cannot predict unsupported projection")
    if any(op.duration_ms is None for op in projection.operations):
        raise PredictionUnavailable('Untimed safety projection cannot be scored')
    if projection.memory.prediction_unavailable is not None:
        raise PredictionUnavailable(projection.memory.prediction_unavailable)
    if projection.decode_plan is not None:
        plan = projection.decode_plan
        unavailable=[]
        result = predict_joining_decode(snapshot, replace(projection, decode_plan=None),
            horizon_ms, frontiers=plan.frontiers, profiles=dict(plan.profiles),
            join_dependencies=dict(plan.join_dependencies),
            completed_tokens=dict(plan.completed_tokens), kv_keys=dict(plan.kv_keys),
            query_service=plan.query_service, compute_ready_ms=plan.compute_ready_ms,
            kv_growth_guarantees=dict(plan.kv_growth_guarantees),
            compute_transition=compute_transition, memory_transition=memory_transition,
            progress_only=progress_only,unavailable_reasons=unavailable)
        if result is None:
            detail=('; reason='+','.join(unavailable)) if unavailable else ''
            raise PredictionUnavailable(
                'Missing joining decode profile or backing readiness'+detail)
        return result
    refs = {}; denominators = {}; weights = {}
    for req in snapshot.requests:
        if req.key in weights:
            raise ValueError("Duplicate request")
        nonnegative(req.weight, "weight")
        weights[req.key] = req.weight
        denominators[req.key] = 0.0
        for work in req.work:
            nonnegative(work.remaining, "remaining reference work")
            key = (req.key, work.key)
            if key in refs:
                raise ValueError("Duplicate reference-work key")
            refs[key] = work.remaining
            denominators[req.key] += work.remaining
    memory = dict(projection.memory.ready)
    transitions = dict(projection.memory.transition_ready)
    busy = dict(snapshot.busy_until)
    if len(busy) != len(snapshot.busy_until):
        raise ValueError("Duplicate busy lane")
    for value in busy.values(): nonnegative(value, "busy time")
    ops = projection.operations
    positions = {op.key: i for i, op in enumerate(ops)}
    if len(positions) != len(ops): raise ValueError("Duplicate operation")
    for op in ops:
        nonnegative(op.duration_ms, "service duration")
        nonnegative(op.ready_ms, "ready offset")
        nonnegative(op.compute_ready_ms, "compute offset")
        if len(set(op.lanes)) != len(op.lanes): raise ValueError("Duplicate lane")
        if any(key not in positions for key in op.depends_on):
            raise ValueError("Missing dependency")
        if any(key not in memory for key in op.memory_keys + ((op.kv_key,) if op.kv_key else ())):
            raise ValueError("Missing backing guarantee")
        if op.duration_ms == 0 and op.credits:
            raise ValueError("Credited work requires a nonzero service profile")
        for credit in op.credits:
            if (credit.request, credit.work) not in refs:
                raise ValueError("Unknown/future request or work")
            if not math.isfinite(credit.fraction) or not 0 <= credit.fraction <= 1:
                raise ValueError("Invalid reference fraction")
    # Advance only dependency-ready operations. Long decode chains must not
    # cause every token to rescan every later (still blocked) token. Selection
    # among ready ops keeps the exact earliest-start/original-position rule.
    children = {op.key: [] for op in ops}
    remaining_deps = {}
    for op in ops:
        dependencies = set(op.depends_on)
        remaining_deps[op.key] = len(dependencies)
        for key in dependencies:
            children[key].append(op)
    ready = {op.key: op for op in ops if remaining_deps[op.key] == 0}
    if progress_only:
        # A cycle beyond H must still be rejected, without scheduling every
        # future token. Linear-time topological validation is sufficient.
        pending_deps = dict(remaining_deps)
        stack = list(ready)
        visited = 0
        while stack:
            key = stack.pop()
            visited += 1
            for child in children[key]:
                pending_deps[child.key] -= 1
                if pending_deps[child.key] == 0:
                    stack.append(child.key)
        if visited != len(ops):
            raise ValueError("Cyclic dependency")
    finished = {}; intervals = []; credited = {}; completions = {}; allocation_events = []; cache_events = []; output_events = []
    submission_blackouts=[]
    def candidate_start(op):
        gates = [op.ready_ms, *(finished[k] for k in op.depends_on),
                 *(busy.get(lane, 0.0) for lane in op.lanes)]
        if compute_transition: gates.append(op.compute_ready_ms)
        for key in op.memory_keys + ((op.kv_key,) if op.kv_key else ()):
            if key not in memory: raise ValueError("Missing backing guarantee")
            # Infinite/unavailable backing stays unavailable in all views.
            gates.append(memory[key])
            if memory_transition:
                gates.append(transitions.get(key, 0.0))
        start=max(gates)
        if op.stage=='prefill' or (op.stage=='decode' and isinstance(op.service_shape,DecodeIterationShape)):
            for blocked_start,blocked_end in submission_blackouts:
                if blocked_start<=start<blocked_end:start=blocked_end
        return start
    # The old selector rebuilt every ready operation's gates after every
    # completion. Real RAG queues contain hundreds of independent external
    # preparation lanes, making that O(n^2) even though most candidates never
    # change. Busy lanes and host blackouts only move readiness later, so keep
    # one lower-bound entry per ready operation and lazily refresh it when it
    # reaches the heap head. The stable (start, original-position) ordering is
    # exactly the same as the scan selector.
    ready_heap=[]
    for op in ready.values():
        heapq.heappush(ready_heap,(candidate_start(op),positions[op.key],op.key))
    while len(finished) < len(ops):
        # Still validate the complete DAG, including work beyond the horizon.
        if not ready_heap:
            raise ValueError("Cyclic dependency")
        queued_start, _, key = heapq.heappop(ready_heap)
        op=ready[key]
        start=candidate_start(op)
        if start!=queued_start:
            if start<queued_start:
                raise ValueError('Operation readiness cannot move backwards')
            heapq.heappush(ready_heap,(start,positions[key],key))
            continue
        if progress_only and start >= horizon_ms:
            break
        del ready[op.key]
        end = start + op.duration_ms
        finished[op.key] = end
        for child in children[op.key]:
            remaining_deps[child.key] -= 1
            if remaining_deps[child.key] == 0:
                ready[child.key] = child
                heapq.heappush(ready_heap,
                               (candidate_start(child),positions[child.key],child.key))
        for lane in op.lanes: busy[lane] = end
        intervals.append((op.key, start, end))
        if 'host_result' in op.lanes:
            # Synchronous result processing/merge occupies the same submitter
            # as cache eviction. Already submitted GPU residuals may overlap;
            # a new kernel submission cannot occur inside any such host call.
            submission_blackouts.append((start,end))
        if (start<horizon_ms and op.stage=='decode'
                and isinstance(op.service_shape,DecodeIterationShape)):
            shape=op.service_shape
            if (len(shape.request_ids)!=len(shape.context_tokens)
                    or len(set(shape.request_ids))!=len(shape.request_ids)
                    or any(type(n) is not int or n<=0 for n in shape.context_tokens)):
                raise ValueError('Exact decode allocation shape required')
            allocation_events.append((start,tuple(zip(shape.request_ids,shape.context_tokens))))
        # Count completed units only (tokens, bundles or retrieval shards).
        # The adapter can provide finer fixed units for finer progress credit.
        if end <= horizon_ms:
            emitted=()
            if op.stage=='decode' and isinstance(op.service_shape,DecodeIterationShape):
                emitted=op.service_shape.request_ids
            elif op.stage=='prefill_result':
                emitted=tuple(c.request for c in op.credits if c.work=='decode')
            if emitted:output_events.append((end,op.key,emitted))
            if op.stage=='cache_evict' and isinstance(op.service_shape,ProspectiveCacheEvictionShape):
                shape=op.service_shape
                if (op.credits or not 0<=shape.predicted_freed_tokens<=shape.eligible_tokens):
                    raise ValueError('Cache reclaim cannot invent work credit or capacity')
                cache_events.append((end,op.key,shape.predicted_freed_tokens))
            for credit in op.credits:
                key = (credit.request, credit.work)
                credited[key] = max(credited.get(key, 0.0), credit.fraction)
                if credit.fraction==1.:
                    completions[key]=min(completions.get(key,math.inf),end)
    totals = {key: 0.0 for key in weights}
    for key, fraction in credited.items(): totals[key[0]] += refs[key] * fraction
    progress = sum(weights[key] * min(1.0, totals[key]/denom)
                   for key, denom in denominators.items() if denom > 0)
    return Prediction(progress, tuple(intervals),tuple(
        (rid,work,end) for (rid,work),end in sorted(completions.items())),
        tuple(sorted(allocation_events,key=lambda event:event[0])),
        tuple(sorted(cache_events)),tuple(sorted(output_events)))


def predict_joining_decode(snapshot, projection, horizon_ms, *, frontiers, profiles,
                           join_dependencies, completed_tokens, kv_keys, query_service,
                           compute_ready_ms=0., compute_transition=True,
                           memory_transition=True, progress_only=False,
                           kv_growth_guarantees=None, unavailable_reasons=None):
    """Resolve upstream events and batch joins afresh for this fixed lease.

    Upstream operations must be independent of newly issued decode: no decode
    lane and no dependency on generated decode operations. An observed residual
    can occupy snapshot.busy_until['decode']. Shared-bandwidth effects belong
    in the fixed coexecution profiles, not a fabricated serial lane. More
    general decode-dependent upstream work requires an integrated event driver
    and is deliberately unsupported here, rather than iterating to a fit.
    Prefix simulation is event evaluation only, never scheduling search.
    None means missing service/backing; malformed or cyclic contracts raise.
    """
    frontiers = tuple(frontiers)
    ids = {f.request_id for f in frontiers}
    if (len(ids) != len(frontiers) or ids != set(join_dependencies)
            or ids != set(completed_tokens) or ids != set(kv_keys)
            or not ids <= {r.key for r in snapshot.requests}):
        raise ValueError('Observed common request set and explicit join contracts required')
    nonnegative(compute_ready_ms, 'decode compute readiness')
    if any('decode' in op.lanes or op.stage == 'decode' for op in projection.operations):
        raise ValueError('Upstream trace must not contain newly issued decode work')
    upstream = predict(snapshot, projection, horizon_ms,
                       compute_transition=compute_transition,
                       memory_transition=memory_transition)
    ends = {key: end for key, start, end in upstream.intervals}
    ops = {op.key: op for op in projection.operations}
    memory = dict(projection.memory.ready)
    transitions = dict(projection.memory.transition_ready)
    admissions = []
    for f in frontiers:
        rid = f.request_id
        deps = tuple(join_dependencies[rid])
        if any(key not in ends for key in deps):
            raise ValueError('Missing upstream result/merge gate')
        if f.stage != 'decode' and not deps:
            raise ValueError('New decode membership requires an explicit upstream gate')
        if kv_keys[rid] not in memory:
            if unavailable_reasons is not None:
                unavailable_reasons.append('missing_existing_kv')
            return None
        gates = [memory[kv_keys[rid]], *(ends[key] for key in deps)]
        if compute_transition:
            gates.append(compute_ready_ms)
        if memory_transition:
            gates.append(transitions.get(kv_keys[rid], 0.))
        ready = max(gates)
        if not math.isfinite(ready):
            if unavailable_reasons is not None:
                unavailable_reasons.append('nonfinite_decode_readiness')
            return None
        n = remaining_decode_tokens(f, profiles[rid])
        emitted = completed_tokens[rid]
        if emitted:
            # A numerical token offset alone cannot create free progress.
            ancestors = set(deps)
            pending = list(deps)
            while pending:
                for parent in ops[pending.pop()].depends_on:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        pending.append(parent)
            fraction = max((c.fraction for key in ancestors for c in ops[key].credits
                            if c.request == rid and c.work == 'decode'), default=0.)
            if n <= 0 or not math.isclose(fraction, emitted/n):
                raise ValueError('Upstream result must credit exactly its emitted token')
        admissions.append(DecodeAdmission(rid, ready, deps, emitted))
    growth = dict(kv_growth_guarantees or {})
    if not set(growth) <= ids:
        raise ValueError('Decode growth guarantee for unknown request')
    windows = {}
    for rid, (count, key) in growth.items():
        if key not in memory:
            if unavailable_reasons is not None:
                unavailable_reasons.append('missing_growth_kv')
            return None
        windows[rid] = (count, key, max(memory[key],
            transitions.get(key, 0.) if memory_transition else 0.))
    decode = build_joining_decode_operations(frontiers, profiles, admissions,
        query_service, kv_keys=kv_keys,
        kv_growth_windows=windows,
        initial_wait_ms=dict(snapshot.busy_until).get('decode', 0.),
        horizon_ms=horizon_ms if progress_only else None,
        submission_blackouts=tuple((start,end) for key,start,end in upstream.intervals
                                  if 'host_result' in ops[key].lanes or ops[key].stage=='prefill_commit'),
        unavailable_reasons=unavailable_reasons)
    if decode is None:
        return None
    expanded = replace(projection, operations=(*projection.operations, *decode))
    # Rebuilt offsets already include precisely the selected transition view.
    return predict(snapshot, expanded, horizon_ms,
                   compute_transition=compute_transition,
                   memory_transition=memory_transition, progress_only=progress_only)


@dataclass(frozen=True)
class Decision:
    lease: Projection
    target_progress: float | None
    keep_progress: float | None
    reason: str
    event_id: int


def predict_admitted_progress(snapshot, projection, planner, horizon_ms):
    """Score a fixed admission plus its forecast-only dependent continuation."""
    expand=getattr(planner,'expand_forecast',None)
    forecast=expand(snapshot,projection,horizon_ms) if expand else projection
    if (not isinstance(forecast,Projection) or not forecast.supported
            or forecast.compute!=projection.compute
            or forecast.executable!=projection.executable
            or forecast.memory is not projection.memory
            or forecast.operations[:len(projection.operations)]!=projection.operations):
        raise ValueError('Forecast cannot replace the fitted lease or its admitted work')
    validate_projected_work(snapshot,forecast.operations)
    return predict(snapshot,forecast,horizon_ms,progress_only=True).progress


class ContinuationUnavailable(ValueError):
    """No safe continuation graph; runtime must distinguish stale observations."""


def schedule(snapshot: Snapshot, planner: Planner, *, mode: str,
             horizon_ms: float, switching_margin: float = 0.0) -> Decision:
    if mode not in ("full", "independent_adaptive"):
        raise ValueError("Unknown scheduling mode")
    if getattr(planner, 'mode', mode) != mode:
        raise ValueError('Planner and admission policy differ')
    nonnegative(switching_margin, "dimensionless switching margin")
    proposed = planner.choose_compute(snapshot)
    demand = planner.derive_demand(snapshot, proposed)
    memory = planner.choose_memory(snapshot, demand)
    target = safe_project(snapshot, proposed, memory, planner)
    record_target = getattr(planner, 'record_target_projection', None)
    if record_target is not None:
        record_target(target)
    keep = make_safe_continuation(snapshot, planner)
    if not keep.supported:
        raise ContinuationUnavailable("Runtime must provide a safe continuation (possibly idle): " + keep.reason)
    # The ablation performs the identical proposals and projection, but no
    # joint utility prediction, comparison, or utility hysteresis.
    if mode == "independent_adaptive":
        return Decision(target if target.supported else keep, None, None,
                        "target" if target.supported else "unsupported_target",
                        snapshot.event_id)
    try:
        keep = price_projected_operations(snapshot, keep, planner)
        keep_u = predict_admitted_progress(snapshot, keep, planner, horizon_ms)
    except PredictionUnavailable as error:
        record = getattr(planner, 'record_prediction_unavailable', None)
        if record is not None:
            record('keep', error)
        return Decision(keep, None, None, 'unscored_continuation', snapshot.event_id)
    if not target.supported:
        return Decision(keep, None, keep_u, "unsupported_target", snapshot.event_id)
    try:
        target = price_projected_operations(snapshot, target, planner)
        target_u = predict_admitted_progress(snapshot, target, planner, horizon_ms)
    except PredictionUnavailable as error:
        record = getattr(planner, 'record_prediction_unavailable', None)
        if record is not None:
            record('target', error)
        return Decision(keep, None, keep_u, 'unscored_target', snapshot.event_id)
    changed = (target.compute.allocation != keep.compute.allocation
               or target.memory.allocation != keep.memory.allocation)
    margin = switching_margin if changed else 0.0
    if target_u > keep_u + margin:
        return Decision(target, target_u, keep_u, "target", snapshot.event_id)
    return Decision(keep, target_u, keep_u, "continuation", snapshot.event_id)


def schedule_full(snapshot: Snapshot, planner: Planner, *, horizon_ms: float,
                  switching_margin: float = 0.0) -> Decision:
    return schedule(snapshot, planner, mode="full", horizon_ms=horizon_ms,
                    switching_margin=switching_margin)


def progress_cost_breakdown(snapshot: Snapshot, target: Projection, horizon_ms: float):
    """Optional diagnostic only; online admission computes only G2.

    Signed differences telescope exactly. Do not clamp then subtract costs from
    G2 again. Non-monotonic profile/order effects should be reported explicitly.
    """
    g0 = predict(snapshot, target, horizon_ms, compute_transition=False, memory_transition=False).progress
    g1 = predict(snapshot, target, horizon_ms, memory_transition=False).progress
    g2 = predict(snapshot, target, horizon_ms).progress
    return dict(g0=g0, g1=g1, g2=g2, switch=g0-g1, migration=g1-g2, utility=g2)
