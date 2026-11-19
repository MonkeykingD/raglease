"""Finite joint search over an explicitly supplied, frozen compute grid.

This selection layer never commits. Native adapters must provide detached
per-candidate planners and an ownership fingerprint covering allocator,
reservations, residency and request state. This is not a production adapter.
No top-K, new search dimension, implicit default grid or cross-event cache.
"""
from dataclasses import dataclass
from itertools import product
from time import perf_counter

from .full_lease import (Decision, PredictionUnavailable, ContinuationUnavailable,
                         safe_project, price_projected_operations,
                         predict_admitted_progress, nonnegative)


@dataclass(frozen=True, order=True)
class GridPoint:
    partition: int
    credit_ms: float
    budget_ms: float


def construct_compute_candidates(partitions, credits_ms, budgets_ms, construct):
    """Call the existing compute-only construction rule for EVERY grid cell.

    construct(point) returns a compute proposal or None for unsupported work.
    It must preserve queue/cohort order, dependencies, profiles and guards.
    The caller freezes shared input before entering this function.
    """
    axes = tuple(tuple(axis) for axis in (partitions, credits_ms, budgets_ms))
    if any(not axis or len(set(axis)) != len(axis) for axis in axes):
        raise ValueError('Explicit nonempty unique finite grid axes required')
    if any(type(p) is not int or p < 0 for p in axes[0]):
        raise ValueError('Nonnegative partition IDs required')
    for axis in axes[1:]:
        for value in axis:
            nonnegative(value, 'grid credit/budget')
    result = []
    for values in sorted(product(*axes)):
        point = GridPoint(*values)
        proposal = construct(point)
        if proposal is not None:
            result.append((point, proposal))
    return tuple(result)


@dataclass(frozen=True)
class SearchStats:
    candidate_count: int
    continuation_count: int
    fit_count: int
    prediction_count: int
    unsupported_count: int
    fit_ms: float
    predict_ms: float
    decision_ms: float


@dataclass(frozen=True)
class SearchResult:
    decision: Decision
    stats: SearchStats


def select_joint_lease(snapshot, *, mode, candidates, continuations,
                       make_planner, project_fixed, fingerprint,
                       horizon_ms, switching_margin=0., sequential_key=None):
    """Full enumerates; Sequential evaluates one compute-selected proposal.

    Independent projects that same proposal but does NOT evaluate net utility.
    Full and Sequential share the best-fixed-allocation continuation helper.
    Independent uses the canonical supported fixed continuation for fallback,
    without pricing/ranking it by net progress.

    make_planner(key, compute) creates a NEW detached single-proposal planner;
    derive_demand and choose_memory are invoked once on each instance. Native
    factory adapters must bind the supplied proposal without another search.
    project_fixed(snapshot,key,compute,planner) must only fit CURRENT physical
    allocation/residency; it must not invoke the elastic planner. It returns a
    safety-projected lease. The fingerprint is checked around every callback;
    a changed observation aborts the entire decision, never just a candidate.
    """
    if mode not in ('full', 'sequential', 'independent_adaptive'):
        raise ValueError('Explicit finite-search policy required')
    nonnegative(horizon_ms, 'prediction horizon')
    nonnegative(switching_margin, 'switching margin')
    if horizon_ms == 0:
        raise ValueError('Positive horizon required')
    started = perf_counter()
    source = fingerprint()
    planners = []  # Keep alive: object IDs must not be recycled in this event.
    fit_count = predictions = unsupported = 0
    fit_ms = predict_ms = 0.

    def checked(function, *args):
        if fingerprint() != source:
            raise ValueError('Stale joint-search snapshot')
        try:
            return function(*args)
        finally:
            if fingerprint() != source:
                raise ValueError('Planning mutated state or snapshot became stale')

    def ordered(rows):
        rows = tuple(rows)
        if any(not isinstance(key, GridPoint) for key, _ in rows):
            raise ValueError('Canonical grid keys required')
        if len({key for key, _ in rows}) != len(rows):
            raise ValueError('Duplicate grid keys')
        return sorted(rows, key=lambda item: item[0])

    targets = ordered(candidates)
    fixed = ordered(continuations)
    total_candidates = len(targets)
    if mode != 'full':
        if sequential_key is None:
            raise ValueError('Original ChooseCompute key required; no implicit argmax')
        targets = [(k, c) for k, c in targets if k == sequential_key]
        if len(targets) != 1:
            raise ValueError('Sequential proposal must belong to explicit grid')

    def planner_for(key, compute):
        planner = checked(make_planner, key, compute)
        if any(planner is old for old in planners):
            raise ValueError('Candidate planners must have independent local state')
        planners.append(planner)
        return planner

    def score(projection, planner):
        nonlocal predictions, predict_ms
        predictions += 1
        start = perf_counter()
        try:
            priced = checked(price_projected_operations, snapshot, projection, planner)
            value = checked(predict_admitted_progress, snapshot, priced, planner, horizon_ms)
            nonnegative(value, 'predicted progress')
            return priced, value
        finally:
            predict_ms += (perf_counter() - start) * 1000

    keep = None
    keep_u = None
    fallback = None
    for key, compute in fixed:
        if compute.allocation != snapshot.compute.allocation:
            raise ValueError('Continuation cannot change compute partition')
        planner = planner_for(key, compute)
        projection = checked(project_fixed, snapshot, key, compute, planner)
        if projection is None or not projection.supported:
            unsupported += 1
            continue
        if (projection.compute != compute
                or projection.memory.allocation != snapshot.memory.allocation
                or projection.memory.resource_plan is not snapshot.memory.resource_plan
                or projection.memory.routing_preview is not snapshot.memory.routing_preview):
            raise ValueError('Continuation changed allocation, native intent or index routing')
        if fallback is None:
            fallback = projection
        if mode == 'independent_adaptive':
            continue
        try:
            projection, utility = score(projection, planner)
        except PredictionUnavailable:
            unsupported += 1
            continue
        if keep is None or utility > keep_u:
            keep, keep_u = projection, utility
    if fallback is None:
        raise ContinuationUnavailable('Explicit safe fixed-allocation continuation required')

    def finish(lease, target_u, reason):
        checked(lambda: None)
        return SearchResult(Decision(lease, target_u, keep_u, reason, snapshot.event_id),
            SearchStats(total_candidates, len(fixed), fit_count, predictions, unsupported,
                        fit_ms, predict_ms, (perf_counter() - started) * 1000))

    if mode != 'independent_adaptive' and keep is None:
        return finish(fallback, None, 'unscored_continuation')
    best, best_u = keep, keep_u
    for key, compute in targets:
        planner = planner_for(key, compute)
        start = perf_counter()
        fit_count += 1
        demand = checked(planner.derive_demand, snapshot, compute)
        memory = checked(planner.choose_memory, snapshot, demand)
        projection = (checked(safe_project, snapshot, compute, memory, planner)
                      if memory is not None else None)
        fit_ms += (perf_counter() - start) * 1000
        if projection is None or not projection.supported:
            unsupported += 1
            continue
        if mode == 'independent_adaptive':
            return finish(projection, None, 'target')
        try:
            projection, utility = score(projection, planner)
        except PredictionUnavailable:
            unsupported += 1
            continue
        if utility > best_u:
            best, best_u = projection, utility
    if mode == 'independent_adaptive':
        return finish(fallback, None, 'unsupported_target')
    changed = (best.compute.allocation != keep.compute.allocation
               or best.memory.allocation != keep.memory.allocation)
    margin = switching_margin if changed else 0.
    if best_u > keep_u + margin:
        return finish(best, best_u, 'target')
    return finish(keep, best_u, 'continuation')


def decision_percentiles(results):
    """Empirical linear P50/P99 over per-decision wall times, not GPU latency."""
    values = sorted(result.stats.decision_ms for result in results)
    if not values:
        raise ValueError('At least one recorded decision required')
    def quantile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return dict(count=len(values), p50_ms=quantile(.5), p99_ms=quantile(.99))
