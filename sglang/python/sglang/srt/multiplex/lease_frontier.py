"""Immutable observations for joint planning; no CUDA or future-result reads.

These are runtime frontiers, not reference-work estimates. Missing information
must be supplied by a documented profile or remain unsupported by the planner.
Capture under the scheduler's serialized decision boundary.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Frontier:
    request_id: str
    stage: str
    query_rows: int | None = None
    prompt_tokens: int | None = None
    completed_prefill_layers: int | None = None
    generated_tokens: int = 0
    remaining_output_limit: int | None = None
    remaining_fixed_output: int | None = None
    # Monotonic scheduler clock; for RAG this begins at ANN admission, not
    # precomputed trace/client planned arrival (which is unavailable here).
    started_at: float | None = None
    preparation_follows: bool | None = None
    generation_follows: bool | None = None
    inflight_prefill_start_layer: int | None = None
    inflight_prefill_end_layer: int | None = None
    queue_position: int | None = None
    # Prepared execution shape; not necessarily the complete input prompt.
    prefill_context_tokens: int | None = None
    prefill_extend_tokens: int | None = None
    preparation_started_at: float | None = None


def capture_prepared_prefill_contract(batch, *, allow_submitted_final=False):
    """Exact ordered execution shape at a prepared batch's next layer.

    Read-only; never calls init_next_round_input or allocates KV. This is not
    a cache/backing guarantee. Completed/submitted work and allocator ownership
    still require the execution stamp and commit-time checks.
    """
    if batch is None or (getattr(batch, 'split_prefill_finished', False) and not allow_submitted_final):
        return None
    reqs = tuple(getattr(batch, 'reqs', ()) or ())
    layer = getattr(batch, 'split_index', None)
    total = getattr(batch, 'extend_num_tokens', None)
    if not reqs or type(layer) is not int or layer < 0:
        return None
    shapes = []
    for req in reqs:
        rid = getattr(req, 'rid', None)
        prompt = getattr(req, 'origin_input_ids', None)
        context = getattr(req, 'fill_ids', None)
        extend = getattr(req, 'extend_input_len', None)
        if (rid is None or prompt is None or context is None
                or type(extend) is not int or not 0 < extend <= len(context)):
            return None
        shapes.append((str(rid), len(prompt), len(context), extend))
    if (len({row[0] for row in shapes}) != len(shapes)
            or type(total) is not int or total != sum(row[3] for row in shapes)):
        return None
    return layer, tuple(shapes)


@dataclass(frozen=True)
class QueuedPrefillShape:
    request_id: str
    prompt_tokens: int
    context_tokens: int
    prefix_tokens: int
    extend_tokens: int


def preview_queued_prefill_shapes(requests, query_prefix_length):
    """Observe shapes in current queue order without initializing requests.

    Mirrors Req.adjust_max_prefix_ids: retain one uncached token for logits,
    and honor the logprob prefix bound. Callback must be a read-only lookup in
    the actual cache namespace, not match_prefix or a guessed hit ratio. Cache
    locks, queue priority, batching and physical allocation are NOT performed.
    None means unsupported/missing state, not an assumed cache miss.
    """
    result = []; seen = set()
    for req in tuple(requests):
        rid = getattr(req, 'rid', None)
        prompt = getattr(req, 'origin_input_ids', None)
        output = getattr(req, 'output_ids', None)
        logprob = getattr(req, 'return_logprob', None)
        if (rid is None or prompt is None or output is None
                or type(logprob) is not bool or not hasattr(req, 'extra_key')):
            return None
        rid = str(rid)
        if not rid or rid in seen:
            raise ValueError('Unique queued request identities required')
        seen.add(rid)
        tokens = tuple(prompt) + tuple(output)
        if not tokens:
            return None
        limit = len(tokens)-1
        if logprob:
            bound = getattr(req, 'logprob_start_len', None)
            if type(bound) is not int:
                return None
            limit = min(limit, bound)
        limit = max(limit, 0)
        prefix = query_prefix_length(tokens[:limit], req.extra_key)
        if prefix is None:
            return None
        if type(prefix) is not int or not 0 <= prefix <= limit:
            raise ValueError('Invalid observed prefix length')
        result.append(QueuedPrefillShape(rid, len(prompt), len(tokens), prefix,
                                         len(tokens)-prefix))
    return tuple(result)


def pending_ann_batches(frontiers, fitted_rows: int):
    """Pure FIFO shape projection matching repeated runtime microbatch pops.

    This is not credit admission or execution. A head request exceeding the
    fitted cap blocks later requests; it is neither truncated nor bypassed.
    """
    if type(fitted_rows) is not int or fitted_rows < 0:
        raise ValueError('Nonnegative fitted ANN rows required')
    pending = [f for f in frontiers if f.stage == 'ann_pending']
    if any(type(f.queue_position) is not int or f.queue_position < 0 for f in pending):
        raise ValueError('Observed FIFO queue positions required')
    if len({f.queue_position for f in pending}) != len(pending):
        raise ValueError('Duplicate ANN queue position')
    pending.sort(key=lambda f:f.queue_position)
    if any(type(f.query_rows) is not int or f.query_rows <= 0 for f in pending):
        raise ValueError('Positive logical request rows required')
    batches = []; batch = []; rows = 0
    for f in pending:
        if rows + f.query_rows > fitted_rows:
            if batch:
                batches.append(tuple(batch)); batch=[]; rows=0
            if f.query_rows > fitted_rows:
                break
        batch.append(f); rows += f.query_rows
    if batch:
        batches.append(tuple(batch))
    return tuple(batches)


@dataclass(frozen=True)
class ColdSearchBinding:
    future_identity: int
    row_start: int
    row_end: int
    enqueued_at: float
    observed_terminal: str | None = None


@dataclass(frozen=True)
class HybridCPUBatch:
    request_ids: tuple[str, ...]
    query_rows: tuple[int, ...]
    searches: tuple[ColdSearchBinding, ...]
    supported: bool
    reason: str


@dataclass(frozen=True)
class FullCPUFallbackObservation:
    future_identity: int
    status: str
    progress: tuple


def observe_full_cpu_fallback(handle, *, observed_at):
    """Detach full-CPU progress without waiting or reading the result payload.

    search_done precedes output-array materialization/Future completion. Only
    a successful terminal Future establishes completed CPU retrieval.
    """
    from concurrent.futures import Future
    import math
    future=getattr(handle,'future',None)
    box=getattr(handle,'planning_observation',None)
    if not isinstance(future,Future) or not isinstance(box,dict):
        return None
    progress=box.get('progress')
    if not isinstance(progress,tuple) or len(progress)!=5:
        return None
    state,submitted,started,acquired,finished=progress
    if state not in ('queued','waiting_lock','searching','search_done','failed'):
        return None
    if not math.isfinite(observed_at):
        raise ValueError('Finite CPU observation time required')
    values=[t for t in progress[1:] if t is not None]
    if (submitted is None or any(type(t) not in (int,float) or not math.isfinite(t) for t in values)
            or values!=sorted(values) or values[-1]>observed_at):
        return None
    if ((state=='queued' and any(t is not None for t in (started,acquired,finished)))
            or (state=='waiting_lock' and (started is None or acquired is not None or finished is not None))
            or (state=='searching' and (started is None or acquired is None or finished is not None))
            or (state=='search_done' and any(t is None for t in (started,acquired,finished)))):
        return None
    if future.done():
        status='cancelled' if future.cancelled() else 'failed' if future.exception() is not None else 'succeeded'
        if status=='succeeded' and state!='search_done':
            return None
    else:
        status='materializing' if state=='search_done' else state
    return FullCPUFallbackObservation(id(future),status,progress)


@dataclass(frozen=True)
class ColdSearchObservation:
    binding: ColdSearchBinding
    status: str
    cold_width: int | None = None
    queue_position: int | None = None
    active_started_at: float | None = None
    observed_finished_at: float | None = None


def resolve_cold_bindings(batch: HybridCPUBatch, worker_snapshot):
    """Join detached observations, never infer success from absence.

    Return one state per CPU search, retaining the merged retrieval barrier.
    Queued/active require a later service prediction; failed/cancelled/unknown
    must not earn completed retrieval credit. Matching enqueue time prevents
    a recycled Python Future identity from borrowing another search's success.
    This does not reserve work or validate a later joint-lease commit.
    """
    if not batch.supported:
        raise ValueError('Cannot resolve an unsupported CPU path')
    entries = {}
    for status in ('queued', 'active', 'terminal'):
        for position, entry in enumerate(worker_snapshot.get(status, ())):
            identity, rows, width, enqueued_at = entry[:4]
            key = (identity, enqueued_at)
            if key in entries:
                raise ValueError('Conflicting cold worker observations')
            entries[key] = (status, position, entry)
    result = []
    for binding in batch.searches:
        if binding.observed_terminal is not None:
            if binding.observed_terminal not in ('succeeded','failed','cancelled'):
                raise ValueError('Invalid observed Future terminal state')
            # This was read from a completed Future BEFORE the worker snapshot.
            # History eviction is irrelevant; captured_at is a conservative
            # completion bound, not a fabricated exact finish timestamp.
            result.append(ColdSearchObservation(binding,binding.observed_terminal,
                          observed_finished_at=worker_snapshot.get('captured_at')))
            continue
        match = entries.get((binding.future_identity, binding.enqueued_at))
        if match is None:
            result.append(ColdSearchObservation(binding, 'unknown'))
            continue
        status, position, entry = match
        if entry[1] != binding.row_end - binding.row_start:
            raise ValueError('Cold binding and worker row counts disagree')
        finished = None
        if status == 'terminal':
            finished, status = entry[4:6]
            if status not in ('succeeded', 'failed', 'cancelled', 'unknown'):
                raise ValueError('Unknown cold worker terminal status')
        result.append(ColdSearchObservation(
            binding, status, cold_width=entry[2],
            queue_position=position if status == 'queued' else None,
            active_started_at=worker_snapshot.get('active_started_at')
                if status == 'active' else None,
            observed_finished_at=finished))
    return tuple(result)


def capture_pending_cold_bindings(scheduler) -> tuple[HybridCPUBatch, ...]:
    """Associate prelaunched CPU searches even before the GPU branch launches.

    Read only completed route futures; never wait for quantization. Each pending
    task is still its own retrieval barrier until a later GPU batch is formed.
    Incomplete/failed routes are unsupported, not zero-service CPU work.
    """
    from types import SimpleNamespace
    groups = getattr(scheduler,'ann_search_pending',()) or ()
    if isinstance(groups,dict):
        groups = (groups,)
    batches = []
    for group in groups:
        for task in group.get('tasks') or (group,):
            ids = (str(task['request_id']),)
            rows = (int(task['queries'].shape[0]),)
            future = task.get('route_future')
            if future is None or not future.done() or future.cancelled():
                batches.append(HybridCPUBatch(ids,rows,(),False,'pending_route_unavailable'))
                continue
            if future.exception() is not None:
                batches.append(HybridCPUBatch(ids,rows,(),False,'pending_route_failed'))
                continue
            prepared = future.result()  # terminal, cannot wait
            routed = getattr(prepared,'routed',None)
            if routed is None:
                batches.append(HybridCPUBatch(ids,rows,(),False,'untracked_pending_route'))
                continue
            handle = SimpleNamespace(batch_size=rows[0],cpu_future=None,
                cpu_searches=getattr(prepared,'cold_searches',()),cold_counts=routed.cold_counts)
            # Reuse identical row-range/worker identity validation as launched
            # requests, including multiple spill futures for the same rows.
            observed = dict(task,result=handle)
            batches.extend(capture_inflight_cold_bindings(
                SimpleNamespace(ann_search_inflight=(observed,))))
    return tuple(batches)


def capture_inflight_cold_bindings(scheduler) -> tuple[HybridCPUBatch, ...]:
    """Observe bindings and terminal Future state without waiting/results/CUDA.

    ALL members of a merged GPU batch wait for ALL its CPU searches at collect,
    not just searches whose row range intersects an individual request. Keep
    that batch boundary. Future identities join the cold worker snapshot within
    this process. Absence from the worker snapshot is NOT proof of completion.
    Pending route futures and the separate full-CPU fallback need their own
    observations; unsupported paths are never presented as zero CPU work.
    """
    from concurrent.futures import Future
    groups = getattr(scheduler, 'ann_search_inflight', ()) or ()
    if isinstance(groups, dict):
        groups = (groups,)
    result = []
    for group in tuple(groups):
        tasks = tuple(group.get('tasks') or (group,))
        ids = tuple(str(t['request_id']) for t in tasks)
        rows = tuple(int(t['queries'].shape[0]) for t in tasks)
        if len(set(ids)) != len(ids) or any(r <= 0 for r in rows):
            raise ValueError('Unique positive-row ANN batch required')
        handle = group.get('result')
        if handle is None or getattr(handle, 'batch_size', None) != sum(rows):
            result.append(HybridCPUBatch(ids, rows, (), False, 'missing_result_shape'))
            continue
        if isinstance(getattr(handle,'future',None),Future) and isinstance(
                getattr(handle,'planning_observation',None),dict):
            result.append(HybridCPUBatch(ids,rows,(),False,'full_cpu_fallback'))
            continue  # Separate pool/lock path, never an empty cold-worker branch.
        searches = tuple(getattr(handle, 'cpu_searches', ()) or ())
        if not searches:
            counts = getattr(handle, 'cold_counts', None)
            empty = (counts is not None and len(counts) == sum(rows)
                     and all(count == 0 for count in counts)
                     and getattr(handle, 'cpu_future', None) is None)
            result.append(HybridCPUBatch(ids, rows, (), empty,
                          'no_cold_work' if empty else 'untracked_cpu_path'))
            continue
        bindings = []
        for search in searches:
            if (type(search.start) is not int or type(search.end) is not int
                    or not 0 <= search.start < search.end <= sum(rows)):
                raise ValueError('Cold search range outside merged ANN batch')
            terminal = None
            future = search.future
            if isinstance(future,Future) and future.done():
                terminal = ('cancelled' if future.cancelled() else
                            'failed' if future.exception() is not None else 'succeeded')
            bindings.append(ColdSearchBinding(id(future), search.start,
                                             search.end, search.enqueued_at,terminal))
        result.append(HybridCPUBatch(ids, rows, tuple(bindings), True, 'worker_bindings'))
    return tuple(result)


@dataclass(frozen=True)
class ExecutionStamp:
    frontiers: tuple[Frontier, ...]
    structure: tuple
    # Keep identity owners alive only for one decision, avoiding id reuse.
    # Never serialize/log this field or retain stamps in unbounded histories.
    owners: tuple = field(compare=False, repr=False)


def capture_execution_stamp(scheduler) -> ExecutionStamp:
    """Snapshot scheduler-owned execution structure without polling devices.

    Use equality to reject a stale plan after a launch, batch replacement,
    queue reorder or observed progress. This is only the REQUEST/EXECUTION
    part of validation: memory ownership, compute allocation, CPU worker
    changes and actual device safety must additionally be checked at commit.
    Scheduler serialization is required; references prevent ABA identity reuse
    but do not turn mutable objects into reservations or device fences.
    """
    from concurrent.futures import Future
    owners = []
    def identity(obj):
        if obj is None:
            return None
        owners.append(obj)
        return id(obj)
    structure = []
    for name in ('ann_search_pending', 'ann_search_inflight', 'rag_ann_collection_pending'):
        groups = getattr(scheduler, name, ()) or ()
        if isinstance(groups, dict):
            groups = (groups,)
        packed = []
        for group in tuple(groups):
            tasks = tuple(group.get('tasks') or (group,))
            full_cpu_box=getattr(group.get('result'),'planning_observation',None)
            full_cpu_progress=full_cpu_box.get('progress') if isinstance(full_cpu_box,dict) else None
            full_cpu_future=getattr(group.get('result'),'future',None)
            # Search completion precedes Future completion/materialization. Record
            # the terminal boundary too, without waiting for or reading results.
            full_cpu_terminal=((identity(full_cpu_future), full_cpu_future.done())
                if isinstance(full_cpu_box,dict) and isinstance(full_cpu_future,Future) else None)
            packed.append((identity(group), identity(group.get('result')),group.get('rag_collection_progress'),
                full_cpu_progress, full_cpu_terminal,
                tuple((identity(t), str(t['request_id']), identity(t['queries']),
                       identity(t.get('result')), identity(t.get('route_future')),
                       t.get('launch_time')) for t in tasks)))
        structure.append((name, tuple(packed)))
    for name in ('waiting_queue', 'split_prefill_batch',
                 'rag_prefill_merge_pending_batch', 'running_batch'):
        value = getattr(scheduler, name, None)
        reqs = value if name == 'waiting_queue' else getattr(value, 'reqs', ())
        structure.append((name, None if name == 'waiting_queue' else identity(value),
                          tuple(identity(req) for req in (reqs or ())),
                          getattr(value, 'split_index', None)))
    meta = getattr(scheduler, 'rag_prefill_bundle_inflight_meta', None) or {}
    structure.append(('prefill_inflight',
        identity(getattr(scheduler, 'rag_prefill_bundle_inflight_event', None)),
        identity(meta.get('batch')), meta.get('end_layer'), meta.get('layers')))
    return ExecutionStamp(capture_frontiers(scheduler), tuple(structure), tuple(owners))


def capture_frontiers(scheduler) -> tuple[Frontier, ...]:
    observed = {}
    # Re-read every live field on every capture. Reuse only the immutable value
    # object when the full scalar signature is unchanged, never a stale stamp.
    previous = getattr(scheduler, '_rag_generation_frontier_values', {})
    current = {}
    ranks = {'ann_pending': 0, 'ann_inflight': 1, 'input_preparation': 2,
             'prefill_queued': 3, 'prefill_active': 4,
             'prefill_merge_pending': 5, 'decode': 6}

    def put(frontier):
        old = observed.get(frontier.request_id)
        if old is None or ranks[frontier.stage] > ranks[old.stage]:
            observed[frontier.request_id] = frontier
        elif old.stage == frontier.stage and old != frontier:
            raise ValueError('Conflicting observations for one request frontier')

    for stage, attr in [('ann_pending', 'ann_search_pending'), ('ann_inflight', 'ann_search_inflight'),
                        ('ann_inflight','rag_ann_collection_pending')]:
        queue_position = 0
        groups = getattr(scheduler, attr, None) or ()
        if isinstance(groups, dict):
            groups = (groups,)
        for group in tuple(groups):
            for task in group.get('tasks') or (group,):
                rid = task.get('request_id')
                if rid is None:
                    raise ValueError('ANN request lacks logical identity')
                queries = task['queries']
                rows = int(queries.shape[0])
                put(Frontier(str(rid), stage, query_rows=rows,
                             remaining_output_limit=task.get('generation_token_limit'),
                             remaining_fixed_output=task.get('generation_fixed_tokens'),
                             started_at=task.get('enqueue_time'),
                             preparation_follows=task.get('preparation_follows'),
                             generation_follows=task.get('generation_follows'),
                             queue_position=queue_position))
                queue_position += 1
    preparations = getattr(scheduler, 'rag_preparation_contracts', None)
    if preparations is not None:
        for rid, contract in tuple(preparations.items()):
            if contract['stage'] == 'input_preparation':
                put(Frontier(str(rid), 'input_preparation',
                             query_rows=contract['query_rows'],
                             remaining_output_limit=contract['generation_token_limit'],
                             remaining_fixed_output=contract['generation_fixed_tokens'],
                             started_at=contract.get('started_at'),
                             preparation_started_at=contract.get('ann_completed_at'),
                             preparation_follows=contract.get('preparation_follows'),
                             generation_follows=contract.get('generation_follows')))
    else:
        # Compatibility for old observation fixtures only. New runtime has a
        # lifetime registry; stale resource leases are not request existence.
        for rid in tuple(getattr(scheduler, 'rag_ann_handoffs', {}) or {}):
            put(Frontier(str(rid), 'input_preparation'))

    def generation(req, stage, layers=None, inflight=None, queue_position=None):
        rid = str(req.rid)
        if getattr(req, 'finished_reason', None) is not None:
            observed.pop(rid, None)
            return
        params = req.sampling_params
        generated = len(req.output_ids)
        maximum = getattr(params, 'max_new_tokens', None)
        remaining = max(0, maximum - generated) if type(maximum) is int and maximum >= 0 else None
        fixed = (remaining if getattr(params, 'ignore_eos', False) is True
                 and not getattr(params, 'stop_strs', None)
                 and not getattr(params, 'stop_token_ids', None) else None)
        signature = (rid, stage, None, len(req.origin_input_ids), layers, generated,
            remaining, fixed, getattr(req, 'rag_e2e_started_at',
                                     getattr(req, 'queue_time_start', None)),
            True, True, inflight[0] if inflight else None,
            inflight[1] if inflight else None, queue_position,
            len(req.fill_ids) if stage == 'prefill_active'
                and getattr(req, 'fill_ids', None) is not None else None,
            getattr(req, 'extend_input_len', None) if stage == 'prefill_active' else None,
            None)
        cached = previous.get(rid)
        frontier = cached[1] if cached is not None and cached[0] == signature else Frontier(*signature)
        current[rid] = (signature, frontier)
        put(frontier)

    for position, req in enumerate(tuple(getattr(scheduler, 'waiting_queue', ()))):
        generation(req, 'prefill_queued', 0, queue_position=position)
    split = getattr(scheduler, 'split_prefill_batch', None)
    if split is not None:
        inflight = None
        meta = getattr(scheduler, 'rag_prefill_bundle_inflight_meta', None) or {}
        event = getattr(scheduler, 'rag_prefill_bundle_inflight_event', None)
        if event is not None and meta.get('batch') is split:
            end, count = meta.get('end_layer'), meta.get('layers')
            if type(end) is not int or type(count) is not int or not 0 < count <= end:
                raise ValueError('Invalid observed in-flight prefill interval')
            inflight = (end-count, end)
        # split_index advances on SUBMISSION, not completion. Only a separate
        # completed-layer observation can be credited by an event predictor.
        for req in tuple(split.reqs):
            generation(req, 'prefill_active', getattr(split, 'rag_completed_prefill_layers', None), inflight)
    for attr, stage in (('rag_prefill_merge_pending_batch', 'prefill_merge_pending'),
                        ('running_batch', 'decode')):
        batch = getattr(scheduler, attr, None)
        if batch is not None:
            for req in tuple(batch.reqs):
                generation(req, stage)
    scheduler._rag_generation_frontier_values = current
    return tuple(observed[rid] for rid in sorted(observed))
