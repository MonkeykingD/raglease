"""Native binding of the finite joint grid to the runtime scheduler.

Selection only. This module builds explicit per-candidate planners, evaluates
them through ``joint_grid.select_joint_lease`` and returns intent. It never
commits, dispatches, reserves, transitions HBM or updates a profile; the
existing ``_prepare_rag_selected_joint_lease`` path remains the only commit.

A candidate is built by the SAME native rule that produces the legacy single
proposal (``_choose_rag_joint_compute``), evaluated under that cell's compute
inputs. The native proposal stays authoritative: a cell whose native rule
declines the work is unsupported, never substituted with a fabricated lease.

That native rule re-polls live GPU completion state on every call, and the
execution stamp deliberately does not cover device polling. Enumerating cells
therefore requires one shared ANN completion observation per event, taken once
here and handed to every target cell; otherwise the comparison would turn on
which cell happened to run after a batch finished rather than on the grid axes.

Grid units are native and must come from a frozen calibration receipt.
``credits_ms`` is milliseconds of ANN service credit. The prefill axis is a
native BUNDLE LAYER LIMIT, not a duration: a layer count is not a time budget,
and no milliseconds-to-layers conversion is invented here. Until a frozen
native layer-bundle profile supplies that conversion, callers must declare the
axis in layers.

The continuation axis is a different quantity from the target axis. Continuation
cells carry fitted native limits -- an ANN batch size and a prefill layer count
-- because they only vary what is safely executable on the current partition.
Those integers occupy the same ``GridPoint`` slots as the target credit and
bundle tiers; the two roles are never compared, and this module rejects a key
shared by both roles rather than resolving it silently.

``make_planner`` is called by the selection layer for targets and for fixed
continuations alike, which need different planner state. Target planners are
built during candidate construction and registered by key; a key that is not a
registered target gets a fresh, compute-unselected planner for fixed-allocation
projection.

A continuation cell declares REQUESTED limits; ``safe_project`` fits them to
what current memory can safely execute, and that fitted projection is the
continuation. Asking for more than the caps allow is therefore a normal fitted
outcome, not a violation, and the selection layer cannot even see it: the
requested lease is held in ``Projection.compute`` and the fit appears only in
``Projection.executable``. The binding records how far each continuation cell
was fitted so the audit trail shows it. A grid whose continuation axis declares
only limits that fit to nothing usable still surfaces as
``ContinuationUnavailable``, which the runtime discards -- the same safe outcome
the legacy route produces for an unsupported keep.
"""
from dataclasses import dataclass
from time import perf_counter

from .full_lease import ComputeLease, RuntimeJointPlanner, safe_project
from .joint_grid import GridPoint, construct_compute_candidates, select_joint_lease
from .lease_frontier import capture_execution_stamp

# The selection layer adds a Sequential mode the runtime planner does not
# implement: it shares the Full planner's compute proposal, only the admission
# comparison differs. Independent-adaptive maps through unchanged.
_SELECTION_TO_PLANNER_MODE = {'full': 'full', 'sequential': 'full',
                              'independent_adaptive': 'independent_adaptive'}
_SELECTION_MODES = ('full', 'sequential', 'independent_adaptive')

_UNSET = object()


@dataclass(frozen=True)
class JointFiniteGrid:
    """Explicit frozen finite grid. No axis has a default.

    ``partitions`` are candidate partition ids. ``credits_ms`` are ANN credit
    tiers in milliseconds. ``bundle_layers`` are native prefill bundle layer
    limits. ``continuation_ann_batches`` and ``continuation_bundle_layers`` are
    native fitted limits applied to the current partition and HBM allocation.
    """
    partitions: tuple
    credits_ms: tuple
    bundle_layers: tuple
    continuation_ann_batches: tuple
    continuation_bundle_layers: tuple

    def __post_init__(self):
        axes = ('partitions', 'credits_ms', 'bundle_layers',
                'continuation_ann_batches', 'continuation_bundle_layers')
        for name in axes:
            axis = getattr(self, name)
            if not isinstance(axis, tuple) or not axis:
                raise ValueError(f'Explicit nonempty tuple {name} axis required')
            if len(set(axis)) != len(axis):
                raise ValueError(f'Unique finite {name} axis required')
        if any(type(p) is not int or p < 0 for p in self.partitions):
            raise ValueError('Nonnegative integer partition ids required')
        if any(type(v) is bool or not isinstance(v, (int, float)) or v < 0
               for v in self.credits_ms):
            raise ValueError('Nonnegative numeric ANN credit tiers required')
        for name in ('bundle_layers', 'continuation_ann_batches',
                     'continuation_bundle_layers'):
            if any(type(v) is not int or v < 0 for v in getattr(self, name)):
                raise ValueError(f'Nonnegative integer {name} required')

    def target_cells(self):
        return tuple(GridPoint(p, credit, layers) for p in self.partitions
                     for credit in self.credits_ms for layers in self.bundle_layers)

    def continuation_cells(self, partition):
        return tuple(GridPoint(partition, rows, layers)
                     for rows in self.continuation_ann_batches
                     for layers in self.continuation_bundle_layers)


def installed_finite_grid(scheduler):
    """Return the installed frozen grid, or None for the legacy single route.

    Absence is not an error: historical and controlled runtimes keep the legacy
    single-proposal Full. A grid is opt-in and must be explicitly installed only
    after its independent calibration receipt has been accepted.
    """
    grid = getattr(scheduler, 'rag_joint_finite_grid', None)
    if grid is None:
        return None
    if not isinstance(grid, JointFiniteGrid):
        raise ValueError('Installed joint grid must be a frozen JointFiniteGrid')
    return grid


def _partition_of(allocation_by_stream, allocation):
    matches = [idx for idx, value in allocation_by_stream.items() if value == allocation]
    if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 0:
        raise ValueError('Applied compute allocation must identify exactly one stream')
    return matches[0]


def _single_partition_inputs(compute_inputs, allocation_by_stream, partition):
    """One cell's native compute inputs; partitions are the enumerated axis.

    Only ``candidate_streams`` is narrowed. The allocation mapping is passed
    whole on purpose: the native rule has a path that indexes
    ``allocation_by_stream[selected]`` for a partition taken from the full SM
    range, BEFORE its own candidate-domain guard can reject that partition. A
    narrowed mapping turns that into a KeyError; a whole mapping lets the guard
    decline the proposal loudly as outside the candidate domain.
    """
    if type(partition) is not int or partition not in allocation_by_stream:
        raise ValueError('Grid partition is absent from the captured mapping')
    cell = dict(compute_inputs)
    # The grid enumerates partitions explicitly: one cell considers exactly one
    # candidate partition, so the native rule cannot silently select another.
    cell['candidate_streams'] = (partition,)
    cell['allocation_by_stream'] = dict(allocation_by_stream)
    return cell


class NativeJointGridBinding:
    """Per-candidate planners plus the shared fixed-allocation projection."""

    def __init__(self, scheduler, snapshot, *, grid, mode, compute_inputs, demand_inputs,
                 memory_inputs, operations_builder=None, timing_builder=None,
                 decode_builder=None, joint_work_builder=None,
                 prospective_inputs_builder=None):
        if mode not in _SELECTION_MODES:
            raise ValueError('Explicit finite-search policy required')
        self.scheduler = scheduler
        self.snapshot = snapshot
        self.grid = grid
        self.mode = mode
        self.planner_mode = _SELECTION_TO_PLANNER_MODE[mode]
        self.compute_inputs = dict(compute_inputs)
        allocation_by_stream = self.compute_inputs.get('allocation_by_stream')
        if not allocation_by_stream:
            raise ValueError('Captured allocation-to-stream mapping required')
        self.allocation_by_stream = dict(allocation_by_stream)
        self.demand_inputs = dict(demand_inputs)
        self.memory_inputs = dict(memory_inputs)
        self.builders = dict(operations_builder=operations_builder,
                             timing_builder=timing_builder, decode_builder=decode_builder,
                             joint_work_builder=joint_work_builder,
                             prospective_inputs_builder=prospective_inputs_builder)
        self.target_planners = {}
        self.target_diagnostics = {}
        self.continuation_keys = set()
        # Audit record: how far each continuation cell was fitted down to what
        # current memory can safely execute. Recorded, never a rejection.
        self.fitted_continuations = []
        self._frozen_outstanding = _UNSET
        self.frozen_outstanding_shared = None
        # Candidate construction runs the native rule once per cell, which is
        # where a finite grid's extra cost actually lands. The selection layer
        # times itself from after construction, so the outer time is recorded
        # here or the per-event cost would be understated.
        self.construction_ms = 0.0

    def _new_planner(self, compute_inputs):
        # Fresh dicts per planner: the runtime planner mutates its own copies of
        # demand/memory inputs and must never share them between candidates.
        return RuntimeJointPlanner(self.scheduler, self.snapshot, mode=self.planner_mode,
            compute_inputs=compute_inputs, demand_inputs=dict(self.demand_inputs),
            memory_inputs=dict(self.memory_inputs), **self.builders)

    def ann_outstanding_inputs(self):
        """One shared ANN completion observation for every target cell.

        The native compute rule re-polls live GPU completion state on each call,
        and the execution stamp records identities and structure "without
        polling devices", so it does NOT change when a submitted batch finishes.
        Evaluating N cells therefore lets cell i and cell j see different
        outstanding ANN service, which would make the comparison turn on timing
        rather than on the grid axes.

        The runtime's shared-ledger observation is taken once here and handed to
        every cell, so all targets are priced against one frozen observation.
        Returns None when the scheduler cannot supply one (or the observation
        went stale); the caller then keeps the per-cell live behaviour and the
        record shows the freeze did not apply. This never widens what a cell may
        propose: a partition missing from the observation still declines.
        """
        if self._frozen_outstanding is _UNSET:
            self._frozen_outstanding = self._capture_ann_outstanding()
        return self._frozen_outstanding

    def _capture_ann_outstanding(self):
        capture = getattr(self.scheduler, '_capture_rag_joint_ann_outstanding_credits', None)
        profile = self.compute_inputs.get('service_profile')
        stamp = self.compute_inputs.get('source_execution')
        if capture is None or profile is None or stamp is None:
            return None
        rows = capture(partitions=tuple(sorted(set(self.grid.partitions))),
                       service_profile=profile, source_execution=stamp)
        if rows is None:
            return None
        frozen = dict(rows)
        if set(frozen) != set(self.grid.partitions):
            raise ValueError('Shared ANN observation must cover every grid partition')
        self.frozen_outstanding_shared = frozen
        return frozen

    def construct_target(self, point):
        """Run the native rule for one cell; None when the native rule declines.

        The proposal returned here is that planner's own single selection, so
        the later derive_demand ownership check holds without re-selecting.
        """
        inputs = _single_partition_inputs(self.compute_inputs, self.allocation_by_stream,
                                          point.partition)
        inputs['credit_ms'] = point.credit_ms
        inputs['bundle_limit'] = point.budget_ms
        frozen = self.ann_outstanding_inputs()
        if frozen is not None:
            inputs['frozen_ann_outstanding'] = frozen
        planner = self._new_planner(inputs)
        proposal = planner.choose_compute(self.snapshot)
        # Record the native diagnostic for declined cells too: the reason a cell
        # was unsupported is the calibration evidence, and it can legitimately
        # be None (the native rule has a (None, None) return path).
        self.target_diagnostics[point] = planner.diagnostic
        if proposal is None or planner.target_unavailable:
            return None
        self.target_planners[point] = planner
        return proposal

    def target_candidates(self):
        started = perf_counter()
        try:
            return construct_compute_candidates(self.grid.partitions, self.grid.credits_ms,
                self.grid.bundle_layers, self.construct_target)
        finally:
            self.construction_ms += (perf_counter() - started) * 1000.

    def continuation_candidates(self):
        """Explicit fixed-allocation candidates on the CURRENT partition.

        Only safely executable credit/bundle limits vary. Allocation, HBM plan
        and index routing stay the snapshot's own, so no elastic memory planning
        occurs in the continuation.
        """
        current = self.snapshot.compute
        partition = _partition_of(self.allocation_by_stream, current.allocation)
        rows = []
        for point in self.grid.continuation_cells(partition):
            lease = ComputeLease(current.allocation, point.credit_ms, point.budget_ms,
                                 current.prefill_contract)
            rows.append((point, lease))
            self.continuation_keys.add(point)
        shared = set(self.target_planners) & self.continuation_keys
        if shared:
            raise ValueError('Target and continuation grid keys must be disjoint: '
                             + repr(sorted(shared)))
        return tuple(rows)

    def make_planner(self, key, compute):
        planner = self.target_planners.get(key)
        if planner is not None:
            return planner
        if key in self.continuation_keys:
            inputs = _single_partition_inputs(self.compute_inputs,
                self.allocation_by_stream, key.partition)
            return self._new_planner(inputs)
        raise ValueError('Grid key outside the declared finite grid')

    def project_fixed(self, snapshot, key, compute, planner):
        """Shared non-searching continuation on the current allocation and HBM.

        Delegates to ``safe_project`` with the snapshot's own memory, the same
        fixed path the legacy single-proposal route uses. No demand derivation,
        no elastic planner call, no allocation change.

        ``safe_project`` fits the cell's declared limits down to what current
        memory can safely execute, so a cell that asks for more than the caps
        allow is a REQUESTED limit, not a violated one: the fitted projection is
        the continuation. Note ``Projection.compute`` holds that requested lease
        and the fit appears in ``Projection.executable``, so the selection
        layer's ``projection.compute != compute`` guard cannot see a fit. How
        far a cell was fitted is recorded here for the audit trail.
        """
        projection = safe_project(snapshot, compute, snapshot.memory, planner)
        if projection is not None and projection.supported and projection.executable != compute:
            self.fitted_continuations.append((key, compute, projection.executable))
        return projection

    def search_record(self, result):
        """Item-6 audit record for one finite-grid decision.

        Counts only; this is a selection-time record, not a serving result.
        """
        stats = result.stats
        return dict(mode=self.mode, planner_mode=self.planner_mode,
            declared_grid_cells=len(self.grid.target_cells()),
            candidate_count=stats.candidate_count,
            continuation_count=stats.continuation_count,
            fit_count=stats.fit_count, prediction_count=stats.prediction_count,
            unsupported_count=stats.unsupported_count,
            fitted_continuation_count=len(self.fitted_continuations),
            shared_ann_observation=self.frozen_outstanding_shared is not None,
            construction_ms=self.construction_ms,
            selection_ms=stats.decision_ms,
            total_decision_ms=self.construction_ms + stats.decision_ms,
            fit_ms=stats.fit_ms, predict_ms=stats.predict_ms, reason=result.decision.reason)

    def fingerprint(self):
        """Read-only ownership stamp; any change aborts the whole decision."""
        return capture_execution_stamp(self.scheduler)


def select_finite_joint_lease(scheduler, snapshot, *, grid, mode, compute_inputs,
                              demand_inputs, memory_inputs, horizon_ms,
                              switching_margin=0.0, sequential_key=None, record=None,
                              operations_builder=None, timing_builder=None,
                              decode_builder=None, joint_work_builder=None,
                              prospective_inputs_builder=None):
    """Finite-grid selection entry. Returns a ``joint_grid.SearchResult``.

    ``record`` is an optional caller-owned dict filled in place with the
    item-6 counts for this decision. Callers that do not need the audit
    record pass nothing and pay no cost.
    """
    binding = NativeJointGridBinding(scheduler, snapshot, grid=grid, mode=mode,
        compute_inputs=compute_inputs, demand_inputs=demand_inputs,
        memory_inputs=memory_inputs, operations_builder=operations_builder,
        timing_builder=timing_builder, decode_builder=decode_builder,
        joint_work_builder=joint_work_builder,
        prospective_inputs_builder=prospective_inputs_builder)
    if mode != 'full' and not isinstance(sequential_key, GridPoint):
        raise ValueError('Sequential/Independent require the original grid key')
    if mode != 'full' and sequential_key not in set(binding.grid.target_cells()):
        raise ValueError('Original compute selection must belong to the declared grid')
    result = select_joint_lease(snapshot, mode=mode,
        candidates=binding.target_candidates(),
        continuations=binding.continuation_candidates(),
        make_planner=binding.make_planner, project_fixed=binding.project_fixed,
        fingerprint=binding.fingerprint, horizon_ms=horizon_ms,
        switching_margin=switching_margin, sequential_key=sequential_key)
    if record is not None:
        record.update(binding.search_record(result))
    return result
