"""Detached ordinary-radix metadata for fixed-lease forecasts, never execution."""
from dataclasses import dataclass, replace
import heapq
import math


@dataclass(frozen=True)
class RadixForecastClock:
    """One bracketed mapping from common perf-counter H to radix monotonic.

    The midpoint is an estimate; sampling uncertainty is exposed, not hidden.
    Both leases reuse this same observation. Cache mutation forecasts cannot
    precede completion of their source snapshot/clock observation.
    """
    window_started_s: float
    perf_before_s: float
    monotonic_s: float
    perf_after_s: float

    def __post_init__(self):
        values=(self.window_started_s,self.perf_before_s,self.monotonic_s,self.perf_after_s)
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                or not self.window_started_s<=self.perf_before_s<=self.perf_after_s):
            raise ValueError('Finite ordered common/cache clock samples required')

    @property
    def uncertainty_ms(self):
        return (self.perf_after_s-self.perf_before_s)*500.

    @property
    def observed_offset_ms(self):
        return (self.perf_after_s-self.window_started_s)*1000.

    def to_monotonic_s(self, event_offset_ms):
        if (type(event_offset_ms) not in (int,float) or not math.isfinite(event_offset_ms)
                or event_offset_ms<self.observed_offset_ms):
            raise ValueError('Cache event must follow its captured source state')
        midpoint=self.perf_before_s+(self.perf_after_s-self.perf_before_s)/2.
        return self.monotonic_s+(self.window_started_s+event_offset_ms/1000.-midpoint)


@dataclass(frozen=True)
class FrozenRadixNode:
    key: int
    parent: int | None
    children: tuple[int, ...]
    tokens: int
    lock_ref: int
    last_access_time: float
    hit_count: int
    estimated: bool = False

    def __lt__(self, other):
        # Exactly TreeNode's tie comparison; do not introduce key/ID ordering.
        return self.last_access_time < other.last_access_time


@dataclass(frozen=True)
class FrozenRadixTree:
    nodes: tuple[FrozenRadixNode, ...]
    root: int
    policy: str
    disabled: bool = False
    request_locks: tuple[tuple[str, int], ...] = ()
    completed_requests: tuple[str, ...] = ()
    next_key: int | None = None
    applied_request_events: tuple = ()


def advance_radix_request_events(tree, clock, events, *, through_ms):
    """Replay traced prospective cache events, not manually guessed timestamps.

    Each (offset_ms, kind, request_id, token_count) is generated from this
    lease's prefill-result or final-token completion. Rebuild from the common
    origin for a different lease; changed historical events are rejected.
    """
    if not math.isfinite(through_ms) or through_ms<0:
        raise ValueError('Finite nonnegative cache replay boundary required')
    applied={(kind,rid):(at,tokens) for kind,rid,at,tokens in tree.applied_request_events}
    previous=-1.
    for at,kind,rid,tokens in tuple(events):
        if (not math.isfinite(at) or at<previous or kind not in ('insert','finish','single_token')
                or not isinstance(rid,str) or not rid or type(tokens) is not int or tokens<=0):
            raise ValueError('Ordered explicit prospective cache events required')
        previous=at
        if at>through_ms:continue
        identity=(kind,rid)
        if identity in applied:
            if applied[identity]!=(at,tokens):
                raise ValueError('Previously applied cache event changed')
            continue
        instant=clock.to_monotonic_s(at)
        if kind in ('insert','single_token'):
            tree=insert_estimated_uncached_prefill(tree,rid,tokens,at_s=instant)
        if kind in ('finish','single_token'):
            tree=complete_estimated_uncached_request(tree,rid,tokens,at_s=instant)
        tree=replace(tree,applied_request_events=(*tree.applied_request_events,(kind,rid,at,tokens)))
        applied[identity]=(at,tokens)
    return tree


def insert_estimated_uncached_prefill(tree, request_id, prompt_tokens, *, at_s):
    """Model the declared no-prefix-hit prompt: insert then lock after prefill.

    This is explicitly estimated topology, not a prediction of unknown token
    matches. at_s must use the captured radix last_access_time clock domain.
    """
    if (tree.disabled or not isinstance(request_id,str) or not request_id
            or request_id in dict(tree.request_locks) or request_id in tree.completed_requests
            or type(prompt_tokens) is not int or prompt_tokens<=0 or not math.isfinite(at_s)):
        raise ValueError('Unique uncached forecast prompt and explicit cache clock required')
    nodes={node.key:node for node in tree.nodes};root=nodes[tree.root]
    if at_s<max(node.last_access_time for node in tree.nodes):
        raise ValueError('Estimated insert cannot precede captured cache state')
    key=tree.next_key if tree.next_key is not None else max(nodes)+1
    if key in nodes:
        raise ValueError('Fresh forecast node identity required')
    nodes[root.key]=replace(root,children=(*root.children,key),last_access_time=at_s)
    prompt=FrozenRadixNode(key,root.key,(),prompt_tokens,1,at_s,0,estimated=True)
    return replace(tree,nodes=(*tuple(nodes[n.key] for n in tree.nodes),prompt),next_key=key+1,
                   request_locks=(*tree.request_locks,(request_id,key)))


def complete_estimated_uncached_request(tree, request_id, stored_tokens, *, at_s):
    """Insert non-overlapping generated suffix, then release the prompt lock.

    Mirrors the ordinary no-collision path only. Cached prefix matches or
    sharing need explicit modeled topology, not an invented free capacity.
    """
    owners=dict(tree.request_locks);nodes={node.key:node for node in tree.nodes}
    if request_id not in owners:
        raise ValueError('Known estimated prompt owner required')
    prompt=nodes[owners[request_id]]
    if (not prompt.estimated or prompt.parent!=tree.root or prompt.children or prompt.lock_ref!=1
            or type(stored_tokens) is not int or stored_tokens<prompt.tokens
            or not math.isfinite(at_s) or at_s<max(n.last_access_time for n in tree.nodes)):
        raise ValueError('Unshared estimated prompt and forward cache clock required')
    nodes[tree.root]=replace(nodes[tree.root],last_access_time=at_s)
    nodes[prompt.key]=replace(prompt,last_access_time=at_s)
    suffix=stored_tokens-prompt.tokens;extra=();next_key=tree.next_key
    if suffix:
        key=next_key if next_key is not None else max(nodes)+1
        if key in nodes:raise ValueError('Fresh forecast suffix identity required')
        nodes[prompt.key]=replace(nodes[prompt.key],children=(key,))
        extra=(FrozenRadixNode(key,prompt.key,(),suffix,0,at_s,0,estimated=True),)
        next_key=key+1
    completed=replace(tree,nodes=(*tuple(nodes[n.key] for n in tree.nodes),*extra),next_key=next_key)
    return release_radix_request_lock(completed,request_id)


def release_radix_request_lock(tree, request_id):
    """Mirror one known owner's dec_lock_ref path; never free cached tokens.

    Other owners' references remain protected. Unknown ownership is not an
    implicit unlock, and replaying a completed owner cannot decrement twice.
    """
    if request_id in tree.completed_requests:
        return tree
    owners=dict(tree.request_locks);nodes={n.key:n for n in tree.nodes}
    if len(owners)!=len(tree.request_locks) or request_id not in owners:
        raise ValueError('One observed radix lock owner required')
    key=owners[request_id];seen=set()
    while key!=tree.root:
        if key not in nodes or key in seen or nodes[key].lock_ref<=0:
            raise ValueError('Valid protected owner path required')
        seen.add(key);node=nodes[key]
        nodes[key]=replace(node,lock_ref=node.lock_ref-1)
        key=node.parent
    return replace(tree,nodes=tuple(nodes[n.key] for n in tree.nodes),
        request_locks=tuple((rid,key) for rid,key in tree.request_locks if rid!=request_id),
        completed_requests=(*tree.completed_requests,request_id))


def replay_radix_timed_evictions(origin, clock, request_events, evictions, *, through_ms):
    """Rebuild from the common origin; select victims at synchronous call start.

    Evictions are (start_ms,end_ms,key,requested_tokens,predicted_freed_tokens).
    In-progress host calls and incompatible request mutations are unsupported,
    not early physical releases. Verify each predicted release against the
    evolved tree, so a later call cannot evict the same original nodes again.
    """
    if origin.applied_request_events:
        raise ValueError('Timed replay requires the unmodified captured origin')
    if not math.isfinite(through_ms) or through_ms<0:
        raise ValueError('Finite nonnegative cache query boundary required')
    request_events=tuple(request_events);tree=origin;previous_end=-1.;seen=set()
    for start,end,key,requested,freed in sorted(tuple(evictions),key=lambda event:(event[0],event[1])):
        if (not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start
                or start<previous_end or not isinstance(key,str) or not key or key in seen
                or type(requested) is not int or requested<=0 or type(freed) is not int or freed<0):
            raise ValueError('Unique serial timed eviction calls required')
        previous_end=end;seen.add(key)
        if start>through_ms:continue
        if through_ms<end:return None
        clock.to_monotonic_s(start)
        if any(start<at<end for at,kind,rid,tokens in request_events):return None
        tree=advance_radix_request_events(tree,clock,request_events,through_ms=start)
        tree,selected,actual=simulate_radix_eviction(tree,requested)
        if actual!=freed:return None
    return advance_radix_request_events(tree,clock,request_events,through_ms=through_ms)


def radix_evictable_tokens(tree):
    """Native evictable_size counts every unlocked non-root node once.

    Validate topology through the zero-work simulator; a shared prefix node
    contributes only once regardless of the number of request paths using it.
    Locked nodes remain excluded even when another owner has finished.
    """
    simulate_radix_eviction(tree,0)
    if tree.disabled:return 0
    return sum(node.tokens for node in tree.nodes if node.key!=tree.root and node.lock_ref==0)


def simulate_radix_eviction(tree, requested_tokens):
    """Return updated detached tree, selected local node keys, actual token sum.

    No new insertions, lock releases or hits are assumed. Repeated calls must
    use the returned state; target and keep start from the same frozen origin.
    """
    if type(requested_tokens) is not int or requested_tokens<0:
        raise ValueError('Nonnegative eviction request required')
    if tree.policy not in ('lru','lfu') or type(tree.disabled) is not bool:
        raise ValueError('Verified ordinary radix policy required')
    nodes={n.key:n for n in tree.nodes}
    if len(nodes)!=len(tree.nodes) or tree.root not in nodes:
        raise ValueError('Unique rooted radix metadata required')
    visited=set();stack=[tree.root];leaves=[]
    while stack:
        key=stack.pop()
        if key in visited or key not in nodes:
            raise ValueError('Acyclic uniquely-owned radix children required')
        visited.add(key);node=nodes[key]
        if (any(type(n) is not int or n<0 for n in (node.key,node.tokens,node.lock_ref,node.hit_count))
                or not math.isfinite(node.last_access_time)
                or (key==tree.root and node.parent is not None)):
            raise ValueError('Valid radix node metadata required')
        for child in node.children:
            if child not in nodes or nodes[child].parent!=key:
                raise ValueError('Consistent parent/child metadata required')
        if node.children:stack.extend(node.children)
        else:leaves.append(node)
    if visited!=set(nodes):
        raise ValueError('No detached unreachable radix nodes allowed')
    if tree.disabled:
        return tree,(),0
    def priority(node):
        return node.last_access_time if tree.policy=='lru' else (node.hit_count,node.last_access_time)
    heap=[(priority(node),node) for node in leaves];heapq.heapify(heap)
    selected=[];freed=0
    while freed<requested_tokens and heap:
        _,node=heapq.heappop(heap)
        if node.key==tree.root:break
        if node.lock_ref>0:continue
        selected.append(node.key);freed+=node.tokens
        del nodes[node.key]
        parent=nodes[node.parent]
        parent=replace(parent,children=tuple(k for k in parent.children if k!=node.key))
        nodes[parent.key]=parent
        if not parent.children:heapq.heappush(heap,(priority(parent),parent))
    return replace(tree,nodes=tuple(nodes[n.key] for n in tree.nodes if n.key in nodes),
                   next_key=tree.next_key if tree.next_key is not None else max(n.key for n in tree.nodes)+1),tuple(selected),freed
