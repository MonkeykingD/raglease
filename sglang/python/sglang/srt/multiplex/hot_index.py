"""VectorLite-compatible hot/cold IVF routing helpers.

VectorLiteRAG stores an offline hot-centroid partition as int32 triples:
``(original_centroid, shard_id, shard_local_centroid)``.  The complete IVF
index remains on CPU.  A query is quantized once against the complete coarse
quantizer, then each probe is routed either to a GPU hot shard or to the CPU
cold fallback.  Results from the disjoint probe sets are merged by distance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VectorLiteMapping:
    """Dense lookup tables derived from a VectorLiteRAG group ``.imap``."""

    cid_lut: np.ndarray
    shard_lut: np.ndarray
    num_entries: int
    shard_ids: tuple[int, ...]
    entry_rank_lut: np.ndarray | None = None

    def offline_prior(
        self, shard_id: int, local_nlist: int
    ) -> np.ndarray | None:
        """Return a local-cluster prior that preserves mapping-file order."""
        if self.entry_rank_lut is None:
            return None
        selected = self.shard_lut == int(shard_id)
        local_ids = self.cid_lut[selected]
        ranks = self.entry_rank_lut[selected]
        valid = (
            (local_ids >= 0)
            & (local_ids < int(local_nlist))
            & (ranks >= 0)
        )
        prior = np.zeros(int(local_nlist), dtype=np.float64)
        prior[local_ids[valid]] = 1.0 / (ranks[valid] + 1.0)
        return prior


@dataclass(frozen=True)
class RoutedProbes:
    """Per-query preassigned probes for one GPU shard and the CPU fallback."""

    hot_distances: np.ndarray
    hot_ids: np.ndarray
    cold_distances: np.ndarray
    cold_ids: np.ndarray
    hot_counts: np.ndarray
    cold_counts: np.ndarray

    @property
    def hot_width(self) -> int:
        return int(self.hot_ids.shape[1])

    @property
    def cold_width(self) -> int:
        return int(self.cold_ids.shape[1])


@dataclass(frozen=True)
class ClusterRefillPlan:
    """Side-effect-free prediction for one cluster refill batch."""

    candidate_ids: tuple[int, ...]
    victim_ids: tuple[tuple[int, ...], ...]
    incremental_benefit_ms: tuple[float, ...]
    candidate_bytes: tuple[int, ...]
    victim_bytes: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.candidate_ids)

    @property
    def expected_benefit_ms(self) -> float:
        return float(sum(self.incremental_benefit_ms))

    def prefix(self, count: int) -> "ClusterRefillPlan":
        count = max(0, min(int(count), self.count))
        return ClusterRefillPlan(
            candidate_ids=self.candidate_ids[:count],
            victim_ids=self.victim_ids[:count],
            incremental_benefit_ms=self.incremental_benefit_ms[:count],
            candidate_bytes=self.candidate_bytes[:count],
            victim_bytes=self.victim_bytes[:count],
        )


class ClusterLeasePolicy:
    """Online value model for cluster-granular shared-HBM leases.

    The offline VectorLite order is a useful prior, but it cannot react to a
    shifted query distribution. This policy combines that prior with online
    probe frequency and CPU-fallback misses. It also charges a measured reload
    cost to non-resident candidates and protects newly resident lists with a
    minimum lease plus promotion hysteresis.

    The class owns policy only. Faiss moves list data, and the caller must fence
    in-flight searches before changing residency.
    """

    def __init__(
        self,
        list_bytes: np.ndarray,
        *,
        heat_decay: float = 0.98,
        miss_boost: float = 4.0,
        promotion_hysteresis: float = 1.15,
        promotion_horizon_batches: float = 64.0,
        min_residency_ms: float = 10.0,
        offline_prior: np.ndarray | None = None,
    ):
        sizes = np.asarray(list_bytes, dtype=np.int64)
        if sizes.ndim != 1 or sizes.size == 0:
            raise ValueError("list_bytes must be a non-empty 1-D array")
        if np.any(sizes < 0):
            raise ValueError("list_bytes must be non-negative")
        if not 0.0 < heat_decay < 1.0:
            raise ValueError("heat_decay must be in (0, 1)")
        if miss_boost < 0.0:
            raise ValueError("miss_boost must be non-negative")
        if promotion_hysteresis < 1.0:
            raise ValueError("promotion_hysteresis must be at least 1")
        if promotion_horizon_batches <= 0.0:
            raise ValueError("promotion_horizon_batches must be positive")
        if min_residency_ms < 0.0:
            raise ValueError("min_residency_ms must be non-negative")

        self.list_bytes = sizes
        positive_sizes = sizes[sizes > 0]
        self.reference_list_bytes = float(
            np.median(positive_sizes) if positive_sizes.size else 1.0
        )
        self.heat_decay = float(heat_decay)
        self.miss_boost = float(miss_boost)
        self.promotion_hysteresis = float(promotion_hysteresis)
        self.promotion_horizon_batches = float(
            promotion_horizon_batches
        )
        self.min_residency_ms = float(min_residency_ms)

        if offline_prior is None:
            prior = 1.0 / (np.arange(sizes.size, dtype=np.float64) + 1.0)
        else:
            prior = np.asarray(offline_prior, dtype=np.float64)
            if prior.shape != sizes.shape or np.any(prior < 0.0):
                raise ValueError(
                    "offline_prior must match list_bytes and be non-negative"
                )
        self.heat_ewma = prior.copy()
        self.miss_ewma = np.zeros(sizes.size, dtype=np.float64)
        self.last_access_ms = np.full(sizes.size, -np.inf, dtype=np.float64)
        self.resident_since_ms = np.full(sizes.size, -np.inf, dtype=np.float64)

        # CPU fallback consumes both worker service time and time queued behind
        # earlier cold probes.  Track the components separately for diagnosis,
        # while scoring promotions with their end-to-end sum.
        self.cpu_service_ms_per_probe = 0.0
        self.cpu_queue_ms_per_probe = 0.0
        self.cpu_ms_per_probe = 0.0
        self.gpu_ms_per_probe = 0.0
        self.reload_ms_per_list = 0.0
        self.observation_batches = 0
        self.observed_probes = 0
        self.observed_misses = 0
        self.promotions = 0
        self.evictions = 0
        self.hysteresis_rejections = 0
        self.residency_rejections = 0
        self.miss_ratio_fast_ewma = 0.0
        self.miss_ratio_baseline = 0.0
        self.miss_ratio_baseline_samples = 0
        self.miss_ratio_baseline_batches = 32
        self.miss_ratio_shift_ratio = 1.5
        self.miss_ratio_shift_margin = 0.15
        self.miss_ratio_shift_active = False
        self.miss_ratio_shift_events = 0

    @staticmethod
    def _now_ms(now_ms: float | None) -> float:
        return time.monotonic() * 1000.0 if now_ms is None else float(now_ms)

    @staticmethod
    def _ewma(previous: float, sample: float, weight: float = 0.1) -> float:
        if sample <= 0.0:
            return previous
        if previous <= 0.0:
            return sample
        return (1.0 - weight) * previous + weight * sample

    @staticmethod
    def _ewma_nonnegative(
        previous: float, sample: float, weight: float = 0.1
    ) -> float:
        """Update a non-negative signal, including zero-valued samples."""
        sample = max(0.0, float(sample))
        if previous <= 0.0:
            return sample
        return (1.0 - weight) * previous + weight * sample

    def observe_probes(
        self,
        local_ids: np.ndarray,
        missed: np.ndarray | None = None,
        *,
        now_ms: float | None = None,
    ) -> None:
        ids = np.asarray(local_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        if np.any((ids < 0) | (ids >= self.list_bytes.size)):
            raise ValueError("a local cluster id is outside the hot shard")
        if missed is None:
            missed_mask = np.zeros(ids.size, dtype=np.bool_)
        else:
            missed_mask = np.asarray(missed, dtype=np.bool_).reshape(-1)
            if missed_mask.shape != ids.shape:
                raise ValueError("missed must have the same shape as local_ids")

        alpha = 1.0 - self.heat_decay
        self.heat_ewma *= self.heat_decay
        self.miss_ewma *= self.heat_decay
        probe_counts = np.bincount(ids, minlength=self.list_bytes.size)
        miss_counts = np.bincount(
            ids[missed_mask], minlength=self.list_bytes.size
        )
        self.heat_ewma += alpha * probe_counts
        # Misses are outstanding CPU-fallback debt, not another frequency
        # sample. Accumulate each event at full weight, then decay the debt
        # across later batches so a fresh miss can justify a bounded reload.
        self.miss_ewma += miss_counts
        self.last_access_ms[np.unique(ids)] = self._now_ms(now_ms)
        self.observation_batches += 1
        self.observed_probes += int(ids.size)
        self.observed_misses += int(missed_mask.sum())

        sample_miss_ratio = float(missed_mask.mean())
        if self.observation_batches == 1:
            self.miss_ratio_fast_ewma = sample_miss_ratio
        else:
            self.miss_ratio_fast_ewma = (
                0.8 * self.miss_ratio_fast_ewma
                + 0.2 * sample_miss_ratio
            )
        if self.miss_ratio_baseline_samples < self.miss_ratio_baseline_batches:
            count = self.miss_ratio_baseline_samples
            self.miss_ratio_baseline = (
                self.miss_ratio_baseline * count + sample_miss_ratio
            ) / (count + 1)
            self.miss_ratio_baseline_samples = count + 1

        was_active = self.miss_ratio_shift_active
        self.miss_ratio_shift_active = self.locality_shift_detected()
        if self.miss_ratio_shift_active and not was_active:
            self.miss_ratio_shift_events += 1

    def locality_shift_detected(self) -> bool:
        if self.miss_ratio_baseline_samples < self.miss_ratio_baseline_batches:
            return False
        threshold = max(
            self.miss_ratio_baseline + self.miss_ratio_shift_margin,
            self.miss_ratio_baseline * self.miss_ratio_shift_ratio,
        )
        return self.miss_ratio_fast_ewma >= threshold

    def observe_search_latency(
        self,
        *,
        cpu_latency_ms: float,
        cpu_probes: int,
        gpu_latency_ms: float,
        gpu_probes: int,
        cpu_queue_ms: float = 0.0,
    ) -> None:
        if cpu_probes > 0:
            probes = int(cpu_probes)
            self.cpu_service_ms_per_probe = self._ewma(
                self.cpu_service_ms_per_probe,
                float(cpu_latency_ms) / probes,
            )
            self.cpu_queue_ms_per_probe = self._ewma_nonnegative(
                self.cpu_queue_ms_per_probe,
                float(cpu_queue_ms) / probes,
            )
            self.cpu_ms_per_probe = (
                self.cpu_service_ms_per_probe
                + self.cpu_queue_ms_per_probe
            )
        if gpu_probes > 0:
            self.gpu_ms_per_probe = self._ewma(
                self.gpu_ms_per_probe,
                float(gpu_latency_ms) / int(gpu_probes),
            )

    def observe_reload(self, latency_ms: float, lists: int = 1) -> None:
        if lists <= 0:
            return
        self.reload_ms_per_list = self._ewma(
            self.reload_ms_per_list, float(latency_ms) / int(lists)
        )

    def _promotion_saved_ms_per_probe(self, default: float) -> float:
        if self.cpu_service_ms_per_probe <= 0.0:
            return float(default)
        # Queueing is an endogenous pressure signal: a successful promotion
        # can reduce it, but it is unsafe to project the entire current queue
        # delay across every future batch. Bound cold credit by one additional
        # CPU service interval so queue pressure can accelerate recovery
        # without creating an unbounded reload feedback loop.
        queue_credit_ms = min(
            self.cpu_queue_ms_per_probe,
            self.cpu_service_ms_per_probe,
        )
        return max(
            0.0,
            self.cpu_service_ms_per_probe
            + queue_credit_ms
            - self.gpu_ms_per_probe,
        )

    def mark_resident(
        self, list_ids: np.ndarray | Sequence[int], *, now_ms: float | None = None
    ) -> None:
        ids = np.asarray(list_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        self.resident_since_ms[ids] = self._now_ms(now_ms)
        self.promotions += int(ids.size)

    def mark_evicted(self, list_ids: np.ndarray | Sequence[int]) -> None:
        ids = np.asarray(list_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        self.resident_since_ms[ids] = -np.inf
        self.evictions += int(ids.size)

    def value_per_byte(
        self, list_ids: np.ndarray | Sequence[int], *, promotion: bool = False
    ) -> np.ndarray:
        ids = np.asarray(list_ids, dtype=np.int64).reshape(-1)
        demand = self.heat_ewma[ids] + self.miss_boost * self.miss_ewma[ids]
        saved_ms = max(1e-6, self._promotion_saved_ms_per_probe(1.0))
        # An IVFFlat list's avoided CPU scan work grows approximately with its
        # payload. Dividing demand by list bytes would therefore charge larger
        # lists twice and strand expensive, frequently probed lists on CPU.
        # Express the per-byte score in units of one reference-sized list: the
        # payload term in total benefit cancels the residency bytes, leaving
        # demand as the dominant ordering signal.
        score = demand * saved_ms
        if promotion and self.reload_ms_per_list > 0.0:
            score -= (
                self.reload_ms_per_list
                / self.promotion_horizon_batches
                * self.reference_list_bytes
                / np.maximum(1, self.list_bytes[ids])
            )
        return score

    def promotion_order(
        self,
        inactive_ids: np.ndarray | Sequence[int],
        *,
        max_lists: int | None = None,
    ) -> np.ndarray:
        ids = np.asarray(inactive_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return ids
        scores = self.value_per_byte(ids, promotion=True)
        positive_mask = scores > 0.0
        positive = ids[positive_mask]
        positive_scores = scores[positive_mask]
        ordered = positive[np.argsort(-positive_scores, kind="stable")]
        if max_lists is not None:
            ordered = ordered[: max(0, int(max_lists))]
        return ordered

    def expected_latency_savings_ms(
        self, list_ids: np.ndarray | Sequence[int]
    ) -> np.ndarray:
        """Estimate CPU-fallback time avoided over the promotion horizon."""
        ids = np.asarray(list_ids, dtype=np.int64).reshape(-1)
        # Miss debt records work that has already fallen back to CPU. It is a
        # useful urgency/tie-breaking signal in ``value_per_byte``, but a
        # promotion cannot recover that spent time. Charging historical debt
        # as future savings creates a positive feedback loop: reloads route
        # more probes to CPU, that queue debt justifies more reloads, and the
        # resident set thrashes. Amortization therefore uses only sustained
        # probe heat over the prediction horizon.
        future_demand = self.heat_ewma[ids] * self.promotion_horizon_batches
        saved_ms = self._promotion_saved_ms_per_probe(1.0)
        relative_scan_bytes = (
            self.list_bytes[ids].astype(np.float64)
            / self.reference_list_bytes
        )
        return future_demand * saved_ms * relative_scan_bytes

    def predict_refill_plan(
        self,
        inactive_ids: np.ndarray | Sequence[int],
        resident_ids: np.ndarray | Sequence[int],
        *,
        available_bytes: int,
        max_lists: int,
        now_ms: float | None = None,
    ) -> ClusterRefillPlan:
        """Predict candidates and evictions without mutating lease state."""
        inactive = np.asarray(inactive_ids, dtype=np.int64).reshape(-1)
        residents = np.asarray(resident_ids, dtype=np.int64).reshape(-1)
        max_lists = max(0, int(max_lists))
        if inactive.size == 0 or max_lists == 0:
            return ClusterRefillPlan((), (), (), (), ())

        ordered = self.promotion_order(inactive, max_lists=max_lists)
        if ordered.size == 0:
            return ClusterRefillPlan((), (), (), (), ())

        now = self._now_ms(now_ms)
        resident_set = {int(list_id) for list_id in residents}
        virtual_since = self.resident_since_ms.copy()
        free_bytes = max(0, int(available_bytes))
        selected: list[int] = []
        victims_by_candidate: list[tuple[int, ...]] = []
        incremental_benefits: list[float] = []
        candidate_sizes: list[int] = []
        victim_sizes: list[int] = []

        for candidate in ordered:
            candidate = int(candidate)
            candidate_bytes = int(self.list_bytes[candidate]) + 512
            needed = max(0, candidate_bytes - free_bytes)
            candidate_score = float(
                self.value_per_byte([candidate], promotion=True)[0]
            )
            candidate_savings = float(
                self.expected_latency_savings_ms([candidate])[0]
            )

            tentative_victims: list[int] = []
            released = 0
            if needed > 0:
                eligible = np.asarray(
                    [
                        list_id
                        for list_id in resident_set
                        if now - virtual_since[list_id]
                        >= self.min_residency_ms
                    ],
                    dtype=np.int64,
                )
                if eligible.size == 0:
                    continue
                victim_scores = self.value_per_byte(
                    eligible, promotion=False
                )
                victim_order = eligible[
                    np.argsort(victim_scores, kind="stable")
                ]
                for victim in victim_order:
                    victim = int(victim)
                    victim_score = float(
                        self.value_per_byte([victim], promotion=False)[0]
                    )
                    required = (
                        max(0.0, victim_score)
                        * self.promotion_hysteresis
                    )
                    if candidate_score <= required:
                        break
                    tentative_victims.append(victim)
                    released += int(self.list_bytes[victim]) + 512
                    if released >= needed:
                        break
                if released < needed:
                    continue

            victim_savings = float(
                self.expected_latency_savings_ms(tentative_victims).sum()
                if tentative_victims
                else 0.0
            )
            incremental = candidate_savings - victim_savings
            if incremental <= 0.0:
                continue

            for victim in tentative_victims:
                resident_set.remove(victim)
            resident_set.add(candidate)
            virtual_since[candidate] = now
            free_bytes += released - candidate_bytes
            selected.append(candidate)
            victims_by_candidate.append(tuple(tentative_victims))
            incremental_benefits.append(incremental)
            candidate_sizes.append(candidate_bytes)
            victim_sizes.append(released)

        return ClusterRefillPlan(
            candidate_ids=tuple(selected),
            victim_ids=tuple(victims_by_candidate),
            incremental_benefit_ms=tuple(incremental_benefits),
            candidate_bytes=tuple(candidate_sizes),
            victim_bytes=tuple(victim_sizes),
        )

    def replacement_victim(
        self,
        candidate_id: int,
        resident_ids: np.ndarray | Sequence[int],
        *,
        now_ms: float | None = None,
    ) -> int | None:
        residents = np.asarray(resident_ids, dtype=np.int64).reshape(-1)
        if residents.size == 0:
            return None
        now = self._now_ms(now_ms)
        ages = now - self.resident_since_ms[residents]
        eligible = residents[ages >= self.min_residency_ms]
        if eligible.size == 0:
            self.residency_rejections += 1
            return None

        resident_scores = self.value_per_byte(eligible, promotion=False)
        victim_index = int(np.argmin(resident_scores))
        victim = int(eligible[victim_index])
        candidate_score = float(
            self.value_per_byte([candidate_id], promotion=True)[0]
        )
        victim_score = float(resident_scores[victim_index])
        required = max(0.0, victim_score) * self.promotion_hysteresis
        if candidate_score <= required:
            self.hysteresis_rejections += 1
            return None
        return victim

    def metrics(self) -> dict[str, float | int]:
        return {
            "heat_decay": self.heat_decay,
            "miss_boost": self.miss_boost,
            "promotion_hysteresis": self.promotion_hysteresis,
            "promotion_horizon_batches": (
                self.promotion_horizon_batches
            ),
            "min_residency_ms": self.min_residency_ms,
            "observation_batches": self.observation_batches,
            "observed_probes": self.observed_probes,
            "observed_misses": self.observed_misses,
            "observed_miss_ratio": (
                self.observed_misses / self.observed_probes
                if self.observed_probes
                else 0.0
            ),
            "cpu_ms_per_probe_ewma": self.cpu_ms_per_probe,
            "cpu_service_ms_per_probe_ewma": (
                self.cpu_service_ms_per_probe
            ),
            "cpu_queue_ms_per_probe_ewma": self.cpu_queue_ms_per_probe,
            "promotion_queue_credit_ms_per_probe": min(
                self.cpu_queue_ms_per_probe,
                self.cpu_service_ms_per_probe,
            ),
            "promotion_saved_ms_per_probe": (
                self._promotion_saved_ms_per_probe(0.0)
            ),
            "gpu_ms_per_probe_ewma": self.gpu_ms_per_probe,
            "reload_ms_per_list_ewma": self.reload_ms_per_list,
            "promotions": self.promotions,
            "evictions": self.evictions,
            "hysteresis_rejections": self.hysteresis_rejections,
            "residency_rejections": self.residency_rejections,
            "miss_ratio_fast_ewma": self.miss_ratio_fast_ewma,
            "miss_ratio_baseline": self.miss_ratio_baseline,
            "miss_ratio_baseline_samples": self.miss_ratio_baseline_samples,
            "miss_ratio_shift_threshold": max(
                self.miss_ratio_baseline + self.miss_ratio_shift_margin,
                self.miss_ratio_baseline * self.miss_ratio_shift_ratio,
            ),
            "miss_ratio_shift_active": self.miss_ratio_shift_active,
            "miss_ratio_shift_events": self.miss_ratio_shift_events,
        }


def load_vectorlite_mapping(
    mapping_path: str | Path,
    nlist: int,
) -> VectorLiteMapping:
    """Load the three-column mapping emitted by ``partition_ivf``.

    This deliberately implements the current ``PartitionedIndex`` format,
    rather than the older header-plus-pairs format used by ``ShardedIndex``.
    """

    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"VectorLite mapping table not found: {path}")
    if nlist <= 0:
        raise ValueError(f"nlist must be positive, got {nlist}")

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"VectorLite mapping table is empty: {path}")
    if raw.size % 3 != 0:
        raise ValueError(
            f"VectorLite mapping table must contain int32 triples, got "
            f"{raw.size} values in {path}"
        )

    entries = raw.reshape(-1, 3)
    original = entries[:, 0].astype(np.int64, copy=False)
    shards = entries[:, 1].astype(np.int64, copy=False)
    local = entries[:, 2].astype(np.int64, copy=False)

    if np.any((original < 0) | (original >= nlist)):
        bad = int(original[(original < 0) | (original >= nlist)][0])
        raise ValueError(f"Invalid original centroid {bad}; nlist={nlist}")
    if np.unique(original).size != original.size:
        raise ValueError("VectorLite mapping contains duplicate original centroids")
    if np.any(shards < 0):
        raise ValueError("VectorLite mapping contains a negative shard id")
    if np.any((local < 0) | (local >= nlist)):
        bad = int(local[(local < 0) | (local >= nlist)][0])
        raise ValueError(f"Invalid shard-local centroid {bad}; nlist={nlist}")
    if np.unique(entries[:, 1:3], axis=0).shape[0] != entries.shape[0]:
        raise ValueError("VectorLite mapping contains duplicate shard-local centroids")

    cid_lut = np.arange(nlist, dtype=np.int64)
    shard_lut = np.full(nlist, -1, dtype=np.int32)
    entry_rank_lut = np.full(nlist, -1, dtype=np.int64)
    cid_lut[original] = local
    shard_lut[original] = shards.astype(np.int32, copy=False)
    entry_rank_lut[original] = np.arange(entries.shape[0], dtype=np.int64)

    return VectorLiteMapping(
        cid_lut=cid_lut,
        shard_lut=shard_lut,
        num_entries=int(entries.shape[0]),
        shard_ids=tuple(int(value) for value in np.unique(shards)),
        entry_rank_lut=entry_rank_lut,
    )


def _pack_probes(
    distances: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    remapped_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = mask.sum(axis=1, dtype=np.int32)
    width = int(counts.max(initial=0))
    packed_distances = np.full(
        (ids.shape[0], width), np.inf, dtype=np.float32
    )
    packed_ids = np.full((ids.shape[0], width), -1, dtype=np.int64)

    for row in range(ids.shape[0]):
        count = int(counts[row])
        if count == 0:
            continue
        packed_distances[row, :count] = distances[row, mask[row]]
        packed_ids[row, :count] = remapped_ids[row, mask[row]]

    return packed_distances, packed_ids, counts


def route_vectorlite_probes(
    coarse_distances: np.ndarray,
    coarse_ids: np.ndarray,
    mapping: VectorLiteMapping,
    shard_id: int,
    resident_local_ids: np.ndarray | None = None,
    max_hot_probes_per_query: int | None = None,
) -> RoutedProbes:
    """Route preassigned probes to one hot shard and the complete CPU index.

    Centroids assigned to another GPU shard are intentionally sent to the CPU
    fallback.  This preserves correctness when a single SGLang worker loads one
    shard from a mapping produced for more than one GPU.
    """

    distances = np.ascontiguousarray(coarse_distances, dtype=np.float32)
    ids = np.ascontiguousarray(coarse_ids, dtype=np.int64)
    if distances.ndim != 2 or ids.ndim != 2 or distances.shape != ids.shape:
        raise ValueError("coarse distances and ids must be equal-shape 2D arrays")

    valid = ids >= 0
    if np.any(ids[valid] >= mapping.cid_lut.size):
        bad = int(ids[valid][ids[valid] >= mapping.cid_lut.size][0])
        raise ValueError(
            f"Coarse centroid {bad} is outside mapping nlist="
            f"{mapping.cid_lut.size}"
        )

    safe_ids = np.where(valid, ids, 0)
    mapped_shards = mapping.shard_lut[safe_ids]
    mapped_ids = mapping.cid_lut[safe_ids]
    hot_mask = valid & (mapped_shards == shard_id)
    if resident_local_ids is not None:
        resident = np.asarray(resident_local_ids, dtype=np.bool_)
        if resident.ndim != 1:
            raise ValueError("resident_local_ids must be a 1-D boolean mask")
        selected_ids = mapped_ids[hot_mask]
        if np.any(selected_ids >= resident.size):
            raise ValueError(
                "A mapped local centroid is outside the residency mask"
            )
        resident_hot = np.zeros_like(hot_mask)
        resident_hot[hot_mask] = resident[selected_ids]
        hot_mask &= resident_hot
    if max_hot_probes_per_query is not None:
        max_hot_probes_per_query = int(max_hot_probes_per_query)
        if max_hot_probes_per_query < 0:
            raise ValueError("max_hot_probes_per_query must be non-negative")
        hot_rank = np.cumsum(hot_mask, axis=1)
        hot_mask &= hot_rank <= max_hot_probes_per_query
    cold_mask = valid & ~hot_mask

    hot_distances, hot_ids, hot_counts = _pack_probes(
        distances, ids, hot_mask, mapped_ids
    )
    cold_distances, cold_ids, cold_counts = _pack_probes(
        distances, ids, cold_mask, ids
    )
    return RoutedProbes(
        hot_distances=hot_distances,
        hot_ids=hot_ids,
        cold_distances=cold_distances,
        cold_ids=cold_ids,
        hot_counts=hot_counts,
        cold_counts=cold_counts,
    )


def reroute_unsubmitted_hot_probes(
    prepared: RoutedProbes, mapping: VectorLiteMapping, shard_id: int,
    resident_local_ids: tuple[bool, ...],
) -> RoutedProbes:
    """Partition ONLY the prepared GPU branch after a prospective index trim.

    Existing prepared.cold_* may already be executing and are intentionally
    excluded from the returned cold branch. Caller preserves those futures
    and joins any newly spilled CPU work with retained GPU work exactly once.
    Must not be used after submission of this prepared GPU branch. No extra
    probes, quantization, queue submission, or cold-to-hot migration occurs.
    """
    resident = np.asarray(resident_local_ids, dtype=np.bool_)
    if resident.ndim != 1:
        raise ValueError('One-dimensional local residency required')
    ids = np.asarray(prepared.hot_ids, dtype=np.int64)
    distances = np.asarray(prepared.hot_distances, dtype=np.float32)
    if ids.ndim != 2 or ids.shape != distances.shape:
        raise ValueError('Prepared hot IDs/distances must have equal 2D shape')
    valid = ids >= 0
    if np.any(ids[valid] >= resident.size):
        raise ValueError('Prepared hot probe outside residency mask')
    if not np.array_equal(valid.sum(axis=1), prepared.hot_counts):
        raise ValueError('Prepared hot count mismatch')
    original_ids = np.flatnonzero(mapping.shard_lut == shard_id)
    local_ids = mapping.cid_lut[original_ids]
    if (np.any(local_ids < 0) or np.any(local_ids >= resident.size)
            or np.unique(local_ids).size != local_ids.size):
        raise ValueError('Unique valid local-to-original mapping required')
    inverse = np.full(resident.size, -1, dtype=np.int64)
    inverse[local_ids] = original_ids
    if np.any(inverse[ids[valid]] < 0):
        raise ValueError('Prepared probe has no original centroid')
    keep = np.zeros_like(valid)
    keep[valid] = resident[ids[valid]]
    spill = valid & ~keep
    originals = np.full_like(ids, -1)
    originals[valid] = inverse[ids[valid]]
    hd, hi, hc = _pack_probes(distances, ids, keep, ids)
    cd, ci, cc = _pack_probes(distances, ids, spill, originals)
    return RoutedProbes(hd, hi, cd, ci, hc, cc)


def empty_search_result(
    batch_size: int,
    topk: int,
    *,
    largest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fill = -np.inf if largest else np.inf
    distances = np.full((batch_size, topk), fill, dtype=np.float32)
    ids = np.full((batch_size, topk), -1, dtype=np.int64)
    return distances, ids


def merge_topk(
    sources: Sequence[tuple[np.ndarray, np.ndarray]],
    topk: int,
    *,
    largest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge CPU and GPU candidates with VectorLite's distance ordering."""

    if not sources:
        raise ValueError("At least one result source is required")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    batch_size = int(sources[0][0].shape[0])
    distance_parts = []
    id_parts = []
    for distances, ids in sources:
        distances = np.asarray(distances, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if distances.ndim != 2 or ids.shape != distances.shape:
            raise ValueError("Each distance/id source must be equal-shape 2D arrays")
        if distances.shape[0] != batch_size:
            raise ValueError("All result sources must have the same batch size")
        distance_parts.append(distances)
        id_parts.append(ids)

    all_distances = np.concatenate(distance_parts, axis=1)
    all_ids = np.concatenate(id_parts, axis=1)
    invalid_fill = -np.inf if largest else np.inf
    sortable = np.where(all_ids >= 0, all_distances, invalid_fill)
    sort_values = -sortable if largest else sortable
    order = np.argsort(sort_values, axis=1, kind="stable")

    available = all_distances.shape[1]
    take = min(topk, available)
    selected = order[:, :take]
    merged_distances = np.take_along_axis(all_distances, selected, axis=1)
    merged_ids = np.take_along_axis(all_ids, selected, axis=1)
    if take == topk:
        return (
            np.ascontiguousarray(merged_distances, dtype=np.float32),
            np.ascontiguousarray(merged_ids, dtype=np.int64),
        )

    padded_distances, padded_ids = empty_search_result(
        batch_size, topk, largest=largest
    )
    padded_distances[:, :take] = merged_distances
    padded_ids[:, :take] = merged_ids
    return padded_distances, padded_ids
