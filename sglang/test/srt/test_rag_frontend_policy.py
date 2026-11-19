import json
import os
import tempfile
import threading
import time
import unittest
from collections import OrderedDict, deque
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.multiplex.ann_executor import (
    AnnExecutor,
    HotAnnResult,
    PreparedHotSearch,
    _ColdSearchBatcher,
    _HotRouteBatcher,
    _PreparedColdSearch,
)
from sglang.srt.multiplex.hot_index import RoutedProbes
from sglang.srt.multiplex.multiplexing import SchedulerMultiplexMixin
from sglang.srt.multiplex.lease_profile import (
    apply_lease_performance_profile,
    load_lease_performance_profile,
)


class _Batch:
    def __init__(self, size):
        self._size = size

    def batch_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0


class _SplitBatch(_Batch):
    def __init__(
        self,
        size,
        token_count,
        *,
        split_index=0,
        reqs=None,
        extend_lens=None,
    ):
        super().__init__(size)
        self.extend_num_tokens = token_count
        self.split_index = split_index
        self.reqs = list(reqs or [])
        self.extend_lens = list(extend_lens or [])


class _PolicyHarness(SchedulerMultiplexMixin):
    def __init__(self):
        self.pdmux_config = SimpleNamespace(
            manual_divisions=[
                [84, 24, 1],
                [72, 36, 8],
                [60, 48, 16],
                [48, 60, 24],
            ],
            decode_bs_divisor=36,
            split_forward_token_budget=8192,
        )
        self.real_sm_group_num = 6
        self.sm_counts = [
            (108, 0),
            (84, 24),
            (72, 36),
            (60, 48),
            (48, 60),
            (0, 108),
        ]
        self.running_batch = _Batch(32)
        self.model_config = SimpleNamespace(num_hidden_layers=32)
        self.rag_partition_hysteresis_bs = 4
        self.rag_partition_hysteresis_holds = 0
        self.rag_partition_min_residency_ms = 0
        self.rag_partition_residency_holds = 0
        self.rag_partition_safe_mixed_hold = False
        self.rag_partition_safe_mixed_hold_ratio = 0.9
        self.rag_partition_safe_mixed_min_samples = 8
        self.rag_partition_safe_mixed_max_hold_ms = 5
        self.rag_last_frontend_activity_ts = 0.0
        self.rag_partition_safe_mixed_holds = 0
        self.rag_partition_safe_mixed_cost_admissions = 0
        self.rag_partition_safe_mixed_cost_rejections = 0
        self.rag_partition_safe_mixed_cost_prediction_misses = 0
        self.rag_partition_safe_mixed_last_cost_decision = {}
        self.rag_prefill_fast_path_resident_cost_checks = 0
        self.rag_partition_decode_slo_ms = ()
        self.rag_partition_decode_slo_batch_thresholds = ()
        self.rag_partition_decode_slo_hysteresis_bs = 0.5
        self.rag_partition_decode_slo_min_residency_ms = 5000
        self.rag_partition_decode_slo_initial_calibration_ms = 0
        self.rag_partition_decode_slo_ewma_alpha = 0.01
        self.rag_partition_decode_slo_band_idx = 0
        self.rag_partition_decode_slo_batch_ewma = None
        self.rag_partition_decode_slo_last_change_ts = 0.0
        self.rag_partition_decode_slo_next_update_ts = 0.0
        self.rag_partition_decode_slo_calibration_started = False
        self.rag_partition_decode_slo_control_updates = 0
        self.rag_partition_decode_slo_band_changes = 0
        self.rag_partition_slo_floor_enabled = True
        self.rag_ann_lease_frontend_sm = 48
        self.rag_prefill_lease_frontend_sm = 60
        self.ann_search_inflight = []
        self.rag_policy_mode = "full"
        self.rag_ann_schedule = "frontend-gc"
        self.rag_ann_work_conserving_decode_overlap = False
        self.rag_ann_pre_switch_launch = False
        self.rag_ann_pre_switch_checks = 0
        self.rag_ann_pre_switch_deadline_risk_triggers = 0
        self.rag_ann_pre_switch_intrinsic_debt_recovery_triggers = 0
        self.rag_ann_pre_switch_switch_induced_triggers = 0
        self.rag_ann_pre_switch_counterfactual_no_gain_skips = 0
        self.rag_ann_pre_switch_gpu_p99_credit_holds = 0
        self.rag_ann_pre_switch_gpu_p99_credit_downsizes = 0
        self.rag_ann_pre_switch_gpu_p99_credit_rows_reduced = 0
        self.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers = 0
        self.rag_ann_pre_switch_gpu_p99_credit_downsized_launches = 0
        self.rag_ann_pre_switch_robust_gain_threshold_holds = 0
        self.rag_ann_pre_switch_robust_gain_threshold_bypasses = 0
        self.rag_ann_pre_switch_batch_net_gain_unavailable = 0
        self.rag_ann_pre_switch_cross_class_debt_candidates = 0
        self.rag_ann_pre_switch_cross_class_debt_bypasses = 0
        self.rag_ann_pre_switch_cross_class_debt_would_admit = 0
        self.rag_ann_pre_switch_cross_class_debt_holds = 0
        self.rag_ann_pre_switch_eligible_robust_gain_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_eligible_raw_robust_gain_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_eligible_batch_net_gain_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_eligible_cross_class_net_debt_ms = deque(
            maxlen=64
        )
        self.rag_ann_pre_switch_last_gpu_p99_credit = {}
        self.rag_ann_pre_switch_classification_invariant_violations = 0
        self.rag_ann_pre_switch_trigger_type_counts = {}
        self.rag_ann_pre_switch_predicted_gain_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_last_counterfactual = {}
        self.rag_ann_pre_switch_launch_batches = 0
        self.rag_ann_pre_switch_launch_rows = 0
        self.rag_ann_pre_switch_switch_deferrals = 0
        self.rag_ann_pre_switch_completion_polls = 0
        self.rag_ann_pre_switch_inline_completions = 0
        self.rag_ann_pre_switch_no_launch_reason_counts = {}
        self.rag_ann_pre_switch_trigger_wait_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_predicted_switch_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_predicted_robust_switch_ms = deque(maxlen=64)
        self.rag_ann_pre_switch_peak_projected_gpu_ms = 0
        self.rag_ann_pre_switch_safety_violations = 0
        self.rag_ann_launch_site = "after-prefill"
        self.rag_ann_fast_path_pressure_threshold = 0.5
        self.rag_ann_fast_path_max_work = 1
        self.rag_ann_fast_path_decode_batch_p95_threshold = 16
        self.rag_prefill_low_load_batch_p95_threshold = 4
        self.rag_prefill_low_load_arrival_rps_threshold = -1
        self.rag_prefill_low_load_reference_tokens = 1024
        self.rag_ann_fast_path_decode_queue_p95_ms = 2
        self.rag_ann_decode_batch_bucket_size = 4
        self.rag_lease_coalesce_min_work = 1
        self.rag_lease_coalesce_max_wait_ms = 0
        self.rag_lease_switch_cost_ms = 0
        self.rag_lease_switch_cost_percentile = 95
        self.rag_lease_round_trip_switch_cost = False
        self.rag_lease_min_gain_ratio = 1
        self.rag_lease_ann_service_estimate_ms = 10
        self.rag_lease_prefill_service_estimate_ms = 10
        self.rag_prefill_dynamic_lease = False
        self.rag_prefill_two_level_bundles = False
        self.rag_prefill_exclusive_sprint = False
        self.rag_prefill_lease_min_layers = 1
        self.rag_prefill_lease_max_layers = 8
        self.rag_prefill_lease_decode_ratio = 1
        self.rag_e2e_slo_ms = -1
        self.rag_ann_stream_epoch_ms = 0
        self.rag_ann_handoff_timeout_ms = 40
        self.rag_ann_frontend_residency_until = 0.0
        self.rag_ann_frontend_residency_releases = 0
        self.rag_ann_partition_switch_deferred = False
        self.rag_ann_partition_drain_holds = 0
        self.rag_ann_partition_drain_events = 0
        self.rag_ann_partition_completion_polls = 0
        self.rag_ann_partition_inline_completions = 0
        self.rag_ann_partition_drain_predicted_ms = deque(maxlen=64)
        self.rag_ann_handoffs = {}
        self.rag_request_start_times = {}
        self.rag_completed_handoff_timing = {}
        self.rag_ann_handoff_latency_ms = deque(maxlen=64)
        self.rag_ann_handoff_completed = 0
        self.rag_ann_handoff_expired = 0
        self.rag_partition_switch_latency_ms = deque(maxlen=64)
        self.rag_partition_switch_pair_latency_ms = {}
        self.rag_partition_transition_intent = None
        self.rag_partition_transition_latency_ms = deque(maxlen=64)
        self.rag_partition_transition_pair_latency_ms = {}
        self.rag_partition_transition_abandoned = 0
        self.rag_lease_profile_priors = {}
        self.rag_lease_switch_benefit_ms = deque(maxlen=64)
        self.rag_lease_horizon_bundle_histogram = {}
        self.rag_lease_horizon_layer_histogram = {}
        self.rag_lease_last_horizon_bundles = 1
        self.rag_prefill_partition_selection_counts = {}
        self.rag_prefill_partition_selection_last = {}
        self.rag_ann_gpu_latency_ms = deque(maxlen=64)
        self.rag_ann_max_inflight = 8
        self.rag_ann_max_batches_per_tick = 0
        self.rag_ann_max_batch_size = 8
        self.rag_ann_max_queued_gpu_ms = 30
        self.rag_ann_max_microbatch_gpu_ms = 20
        self.rag_ann_service_time_percentile = 95
        self.rag_ann_service_time_min_samples = 4
        self.rag_ann_gpu_service_ms_by_batch_size = {}
        self.rag_ann_completion_service_ms_by_path_batch = {}
        self.rag_ann_post_gpu_tail_ms_by_path_batch = {}
        self.rag_ann_inline_collection_tail_ms_by_path_batch = {}
        self.rag_ann_worker_collection_ms_by_path_batch = {}
        self.rag_ann_worker_send_per_output_ms_by_path_batch = {}
        self.rag_ann_credit_holds = 0
        self.rag_ann_credit_downsized_launches = 0
        self.rag_ann_credit_forced_single_launches = 0
        self.rag_ann_credit_peak_queued_ms = 0
        self.rag_ann_credit_peak_admitted_queued_ms = 0
        self.rag_ann_credit_peak_forced_atomic_ms = 0
        self.rag_ann_credit_last_limit_ms = 30
        self.rag_ann_credit_last_queued_ms = 0
        self.rag_ann_credit_last_batch_ms = 0
        self.rag_ann_credit_last_forced_atomic = False
        self.rag_ann_inflight_peak = 0
        self.rag_ann_hybrid_pipeline_backlog_holds = 0
        self.rag_ann_hybrid_pipeline_second_slot_admissions = 0
        self.rag_ann_hybrid_pipeline_window_expansions = 0
        self.rag_ann_hybrid_pipeline_window_histogram = {}
        self.rag_ann_grant_ticks = 0
        self.rag_ann_granted_batches = 0
        self.rag_ann_multi_grant_ticks = 0
        self.rag_ann_tick_limit_holds = 0
        self.rag_ann_grants_per_tick_histogram = {}
        self.rag_ann_batches_granted_this_tick = 0
        self.rag_ann_tick_grants_recorded = False
        self.rag_ann_tick_limit_held_this_tick = False
        self.rag_ann_queue_wait_ms = deque(maxlen=64)
        self.rag_ann_queue_wait_histogram = {}
        self.rag_ann_batch_size_histogram = {}
        self.rag_ann_launch_reason_counts = {}
        self.rag_ann_stream_group_launch_counts = {}
        self.rag_ann_forced_release_events = 0
        self.rag_frontend_ann_launched = False
        self.rag_frontend_decision_counts = {}
        self.rag_lease_coalesce_waits = 0
        self.rag_lease_coalesce_work_holds = 0
        self.rag_lease_coalesce_cost_holds = 0
        self.rag_lease_decode_restore_transitions = 0
        self.rag_dependency_lease_holds = 0
        self.rag_ann_deadline_ms = 50
        self.rag_prefill_deadline_ms = 500
        self.rag_ann_decode_step_slo_ms = 20
        self.rag_ann_itl_slo_ms = -1
        self.rag_ann_safety_margin_ms = 5
        self.rag_prefill_decode_guard_percentile = 99
        self.rag_ann_urgency_start_ratio = 1.0
        self.rag_ann_max_decode_slowdown_ratio = 1.0
        self.rag_ann_urgency_admitted_requests = 0
        self.rag_ann_urgency_admission_wait_ms = deque(maxlen=64)
        self.rag_ann_decode_iteration_ms = deque(maxlen=64)
        self.rag_ann_decode_batch_sizes = deque(maxlen=64)
        self.rag_ann_decode_gpu_ms = deque(maxlen=64)
        self.rag_ann_decode_partition_ms = {}
        self.rag_ann_decode_partition_batch_ms = {}
        self.rag_ann_decode_partition_batch_overlap_ms = {}
        self.rag_ann_decode_gpu_partition_ms = {}
        self.rag_ann_decode_gpu_partition_batch_ms = {}
        self.rag_ann_decode_gpu_partition_batch_overlap_ms = {}
        self.rag_ann_decode_queue_delay_ms = deque(maxlen=64)
        self.rag_decode_prediction_pairs = deque(maxlen=16384)
        self.rag_decode_prediction_abs_error_ms = deque(maxlen=4096)
        self.rag_decode_prediction_samples = 0
        self.rag_decode_prediction_false_safe = 0
        self.rag_decode_prediction_false_unsafe = 0
        self.rag_decode_prediction_partition_mismatches = 0
        self.rag_decode_prediction_regime_shift_ratio = 4.0
        self.rag_decode_prediction_cold_start_margin = 1.25
        self.rag_decode_iteration_prediction_pairs = deque(maxlen=16384)
        self.rag_decode_iteration_prediction_abs_error_ms = deque(
            maxlen=4096
        )
        self.rag_decode_iteration_prediction_samples = 0
        self.rag_decode_iteration_prediction_false_safe = 0
        self.rag_decode_iteration_prediction_false_unsafe = 0
        self.rag_partition_decode_floor_sm = 0
        self.rag_partition_slo_promotions = 0
        self.rag_partition_slo_demotions = 0
        self.rag_partition_floor_safe_observations = 0
        self.rag_partition_decode_only_fallback_required = False
        self.rag_partition_decode_only_fallback_activations = 0
        self.rag_decode_recovery_pending = False
        self.rag_decode_recovery_requests = 0
        self.rag_decode_recovery_windows = 0
        self.rag_handoff_decode_release_pending = False
        self.rag_handoff_decode_release_windows = 0
        self.rag_prefill_bundle_inflight_event = None
        self.rag_prefill_bundle_inflight_meta = None
        self.rag_prefill_layer_ms = deque(maxlen=64)
        self.rag_prefill_layer_ms_by_partition_bucket = {}
        self.rag_prefill_prediction_pairs = deque(maxlen=16384)
        self.rag_prefill_prediction_abs_error_ms = deque(maxlen=4096)
        self.rag_prefill_prediction_samples = 0
        self.rag_prefill_bundle_layer_histogram = {}
        self.rag_prefill_bundle_completions = 0
        self.rag_prefill_bundle_switch_holds = 0
        self.rag_prefill_two_level_mixed_budget_bypasses = 0
        self.rag_prefill_two_level_safety_deferrals = 0
        self.rag_prefill_short_bundle_layers = 16
        self.rag_prefill_short_token_threshold = 2048
        self.rag_prefill_severe_risk_ratio = 1.25
        self.rag_prefill_short_bundle_activations = 0
        self.rag_prefill_risk_bundle_activations = 0
        self.rag_prefill_severe_risk_bundle_activations = 0
        self.rag_prefill_long_request_bundle_activations = 0
        self.rag_prefill_aggregate_false_long_avoided = 0
        self.rag_prefill_request_length_fallbacks = 0
        self.rag_prefill_mixed_service_guard_uses = 0
        self.rag_prefill_exclusive_iteration_guard_uses = 0
        self.rag_prefill_iteration_pressure_half_caps = 0
        self.rag_prefill_iteration_pressure_full_bundle_bypasses = 0
        self.rag_prefill_candidate_net_gain_ms = deque(maxlen=4096)
        self.rag_prefill_critical_path_net_gain_ms = deque(maxlen=4096)
        self.rag_prefill_robust_critical_path_net_gain_ms = deque(maxlen=4096)
        self.rag_prefill_candidate_net_gain_admissions = 0
        self.rag_prefill_candidate_net_gain_holds = 0
        self.rag_prefill_e2e_critical_admissions = 0
        self.rag_prefill_e2e_uncertainty_holds = 0
        self.rag_prefill_tail_gain_admissions = 0
        self.rag_prefill_tail_liveness_admissions = 0
        self.rag_prefill_tail_liveness_layer_caps = 0
        self.rag_prefill_tail_liveness_budget_ms = deque(maxlen=4096)
        self.rag_prefill_tail_gain_holds = 0
        self.rag_prefill_tail_aware_admission = True
        self.rag_prefill_candidate_net_gain_last = {}
        self.rag_prefill_floor_safe_admissions = 0
        self.rag_prefill_queue_wait_ms = deque(maxlen=64)
        self.rag_prefill_gpu_latency_ms = deque(maxlen=64)
        self.rag_prefill_wall_latency_ms = deque(maxlen=64)
        self.rag_prefill_to_first_decode_ms = deque(maxlen=64)
        self.rag_frontend_decision_ms = deque(maxlen=64)
        self.rag_prefill_sprint_chunks = 0
        self.rag_prefill_exclusive_sprint_available = False
        self.rag_prefill_exclusive_sprint_chunks = 0
        self.rag_prefill_exclusive_sprint_block_counts = {}
        self.rag_decode_update_deferred_for_prefill_sprint = False
        self.rag_prefill_exclusive_decode_update_skips = 0
        self.rag_prefill_exclusive_deferred_merges = 0
        self.rag_prefill_merge_pending_batch = None
        self.rag_prefill_cohort_holds = 0
        self.rag_prefill_cohort_releases = 0
        self.rag_prefill_cohort_size_histogram = {}
        self.rag_dependency_prefill_admissions = 0


class TestRagFrontendPolicy(unittest.TestCase):
    @staticmethod
    def _handoff_req(ready_at):
        return SimpleNamespace(
            rag_ann_handoff_ready=True,
            rag_scheduler_arrival_at=ready_at,
        )

    def test_idle_decode_endpoint_skips_full_frontend_planning(self):
        policy = _PolicyHarness()
        policy.ann_search_pending = deque()
        policy.split_prefill_batch = None
        policy.rag_prefill_handoff_active = False
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}

        with (
            patch.object(policy, "_prune_rag_ann_handoffs"),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(
                policy,
                "_rag_stage_candidate_stream_idx",
                side_effect=AssertionError("idle path evaluated a candidate"),
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                side_effect=AssertionError("idle path sampled load state"),
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                side_effect=AssertionError("idle path evaluated a budget"),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertEqual(plan.reason, "idle")
        self.assertEqual(plan.target_reason, "decode_only_resident")
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertFalse(plan.frontend_needed)
        self.assertFalse(policy.rag_prefill_run_allowed)
        self.assertEqual(policy.rag_frontend_idle_fast_path_hits, 1)

    def test_lease_profile_priors_do_not_pollute_live_telemetry(self):
        policy = _PolicyHarness()
        selection = {
            "path": "/tmp/lease-profile.yaml",
            "schema_version": 2,
            "selected_request_rate": 6.25,
            "performance_priors": {
                "phase_latency_profiles": {
                    "ann_gpu_service_by_batch_size": {
                        "4": {"p95_ms": 5.0}
                    },
                    "prefill_layer_latency_by_partition_bucket": {
                        "2:8192": {
                            "p50_ms": 14.0,
                            "p95_ms": 18.0,
                            "p99_ms": 20.0,
                        }
                    },
                    "decode_iteration_by_partition_batch_bucket": {
                        "2": {
                            "32": {
                                "p50_ms": 30.0,
                                "p95_ms": 42.0,
                                "p99_ms": 48.0,
                            }
                        }
                    },
                    "decode_gpu_step_by_partition_batch_bucket": {
                        "2": {
                            "32": {
                                "p50_ms": 28.0,
                                "p95_ms": 38.0,
                                "p99_ms": 44.0,
                            }
                        }
                    },
                },
                "resource_transition_profiles": {
                    "green_context_switch_by_pair": {
                        "5->2": {"p95_ms": 1.25}
                    }
                },
            },
        }

        status = apply_lease_performance_profile(policy, selection)

        self.assertTrue(status["loaded"])
        self.assertEqual(status["storage"], "separate_prior_table")
        self.assertEqual(len(policy.rag_ann_gpu_latency_ms), 0)
        self.assertEqual(policy.rag_ann_gpu_service_ms_by_batch_size, {})
        self.assertEqual(policy.rag_prefill_layer_ms_by_partition_bucket, {})
        self.assertEqual(policy.rag_ann_decode_gpu_partition_batch_ms, {})
        self.assertEqual(policy.rag_partition_switch_pair_latency_ms, {})
        self.assertEqual(policy._predict_ann_gpu_service_ms(4), 5.0)
        self.assertEqual(policy._predict_rag_prefill_layer_ms(2, 8192), 18.0)
        self.assertEqual(
            policy._predict_rag_prefill_layer_ms(
                2, 8192, percentile=50.0
            ),
            14.0,
        )
        self.assertEqual(policy._rag_ann_decode_prediction_ms(2, 32), 38.0)
        self.assertEqual(
            policy._rag_ann_decode_prediction_ms(
                2, 32, percentile=50.0
            ),
            28.0,
        )
        self.assertEqual(
            policy._rag_ann_decode_prediction_ms(
                2, 32, percentile=99.0
            ),
            44.0,
        )
        self.assertEqual(policy._rag_lease_switch_cost_estimate_ms(5, 2), 1.25)

    def test_lease_profile_skips_ann_prior_when_nprobe_changes(self):
        policy = _PolicyHarness()
        policy.ann_executor = SimpleNamespace(nprobe=4096)
        selection = {
            "path": "/tmp/lease-profile.yaml",
            "schema_version": 2,
            "selected_request_rate": 4.25,
            "fixed_workload": {"nprobe": 2048},
            "performance_priors": {
                "phase_latency_profiles": {
                    "ann_gpu_service_by_batch_size": {
                        "4": {"p95_ms": 5.0}
                    },
                    "prefill_layer_latency_by_partition_bucket": {
                        "2:8192": {"p95_ms": 18.0}
                    },
                },
                "resource_transition_profiles": {},
            },
        }

        status = apply_lease_performance_profile(policy, selection)

        self.assertFalse(status["compatibility"]["ann_profile_compatible"])
        self.assertEqual(
            status["compatibility"]["ann_prior_skip_reason"],
            "nprobe_mismatch",
        )
        self.assertEqual(status["seeded"]["ann_batch_buckets"], 0)
        self.assertEqual(policy._predict_ann_gpu_service_ms(4), 40.0)
        self.assertEqual(policy._predict_rag_prefill_layer_ms(2, 8192), 18.0)

    def test_lease_profile_skips_ann_prior_when_nprobe_is_missing(self):
        policy = _PolicyHarness()
        policy.ann_executor = SimpleNamespace(nprobe=4096)
        selection = {
            "path": "/tmp/legacy-lease-profile.yaml",
            "schema_version": 2,
            "selected_request_rate": 4.25,
            "performance_priors": {
                "phase_latency_profiles": {
                    "ann_gpu_service_by_batch_size": {
                        "4": {"p95_ms": 5.0}
                    },
                    "prefill_layer_latency_by_partition_bucket": {
                        "2:8192": {"p95_ms": 18.0}
                    },
                },
                "resource_transition_profiles": {},
            },
        }

        status = apply_lease_performance_profile(policy, selection)

        self.assertFalse(status["compatibility"]["ann_profile_compatible"])
        self.assertEqual(
            status["compatibility"]["ann_prior_skip_reason"],
            "missing_profile_nprobe",
        )
        self.assertEqual(status["seeded"]["ann_batch_buckets"], 0)
        self.assertEqual(policy._predict_ann_gpu_service_ms(4), 40.0)
        self.assertEqual(policy._predict_rag_prefill_layer_ms(2, 8192), 18.0)

    def test_lease_profile_uses_deployment_nprobe_metadata(self):
        profile = {
            "schema_version": 2,
            "default_profile_rate": 4.75,
            "deployment": {"nprobe": 4096},
            "profiles_by_request_rate": {
                "4p75": {
                    "request_rate": 4.75,
                    "performance_priors": {
                        "phase_latency_profiles": {},
                        "resource_transition_profiles": {},
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "profile.json")
            with open(path, "w", encoding="utf-8") as fout:
                json.dump(profile, fout)
            loaded = load_lease_performance_profile(path)

        self.assertEqual(loaded["fixed_workload"]["nprobe"], 4096)

    def _prefill_cohort_policy(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_lease_coalesce_min_work = 2
        policy.rag_lease_coalesce_max_wait_ms = 50
        return policy

    def test_prefill_cohort_waits_for_second_handoff(self):
        policy = self._prefill_cohort_policy()
        policy.waiting_queue = [self._handoff_req(99.98)]

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            self.assertTrue(policy._should_wait_for_rag_prefill_cohort())

        self.assertEqual(policy.rag_prefill_cohort_holds, 1)

    def test_prefill_cohort_releases_at_minimum_work(self):
        policy = self._prefill_cohort_policy()
        policy.waiting_queue = [
            self._handoff_req(99.99),
            self._handoff_req(99.995),
        ]

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            self.assertFalse(policy._should_wait_for_rag_prefill_cohort())

    def test_prefill_cohort_releases_at_timeout(self):
        policy = self._prefill_cohort_policy()
        policy.waiting_queue = [self._handoff_req(99.94)]

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            self.assertFalse(policy._should_wait_for_rag_prefill_cohort())

    def test_prefill_cohort_does_not_delay_unrelated_prefill(self):
        policy = self._prefill_cohort_policy()
        policy.waiting_queue = [
            self._handoff_req(99.99),
            SimpleNamespace(rag_ann_handoff_ready=False),
        ]

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            self.assertFalse(policy._should_wait_for_rag_prefill_cohort())

    def test_prefill_sprint_eligibility_uses_bounded_layer_bundle(self):
        policy = _PolicyHarness()
        policy.model_config.num_hidden_layers = 28
        policy.rag_prefill_lease_max_layers = 8
        policy.rag_prefill_max_consecutive_chunks = 12
        policy.split_prefill_batch = _SplitBatch(4, 4092)

        self.assertTrue(policy._rag_prefill_sprint_eligible())

    def test_prefill_sprint_eligibility_rejects_oversized_bundle(self):
        policy = _PolicyHarness()
        policy.model_config.num_hidden_layers = 28
        policy.rag_prefill_lease_max_layers = 8
        policy.rag_prefill_max_consecutive_chunks = 12
        policy.split_prefill_batch = _SplitBatch(32, 32736)

        self.assertFalse(policy._rag_prefill_sprint_eligible())

    @staticmethod
    def _seed_ann_completion_prediction(
        policy,
        *,
        path="full_gpu",
        batch_size=1,
        service_ms=20.0,
        tail_ms=5.0,
        gpu_ms=None,
        inline_ms=None,
    ):
        if gpu_ms is None:
            gpu_ms = max(0.001, service_ms - tail_ms)
        if inline_ms is None:
            inline_ms = tail_ms
        service_samples = (
            list(service_ms)
            if isinstance(service_ms, (list, tuple, deque, np.ndarray))
            else [service_ms] * 4
        )
        tail_samples = (
            list(tail_ms)
            if isinstance(tail_ms, (list, tuple, deque, np.ndarray))
            else [tail_ms] * 4
        )
        policy.rag_ann_completion_service_ms_by_path_batch = {
            path: {
                batch_size: deque(service_samples, maxlen=64),
            }
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            path: {
                batch_size: deque(tail_samples, maxlen=64),
            }
        }
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            batch_size: deque([gpu_ms] * 4, maxlen=64)
        }
        policy.rag_ann_inline_collection_tail_ms_by_path_batch = {
            path: {
                batch_size: deque([inline_ms] * 4, maxlen=64),
            }
        }

    def _counterfactual_pre_switch_policy(
        self,
        *,
        switch_ms,
        service_ms,
        tail_ms,
        gpu_ms,
        inline_ms,
        ann_wait_ms=0.0,
        allow_prefill=False,
        prefill_wait_ms=0.0,
    ):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_deadline_ms = 100.0
        policy.rag_ann_safety_margin_ms = 5.0
        policy.rag_ann_max_queued_gpu_ms = 200.0
        policy.rag_ann_max_microbatch_gpu_ms = 200.0
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=allow_prefill,
            reason="ann_decode_overlap",
            immediate_handoff=allow_prefill,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=ann_wait_ms,
            prefill_wait_ms=prefill_wait_ms,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": "request-0",
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [switch_ms] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy,
            service_ms=service_ms,
            tail_ms=tail_ms,
            gpu_ms=gpu_ms,
            inline_ms=inline_ms,
        )
        return policy

    def _p99_downsize_pre_switch_policy(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=40.0,
            service_ms=60.0,
            tail_ms=30.0,
            gpu_ms=10.0,
            inline_ms=1.0,
        )
        policy.rag_ann_max_queued_gpu_ms = 20.0
        policy.ann_search_pending.append(
            {
                "enqueue_time": 0.0,
                "queries": np.zeros((1, 8), dtype=np.float32),
                "request_id": "request-1",
            }
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 61 + [15.0] * 3, maxlen=64),
            2: deque([10.0] * 61 + [30.0] * 3, maxlen=64),
        }
        policy.rag_ann_completion_service_ms_by_path_batch[
            "full_gpu"
        ][2] = deque([60.0] * 4, maxlen=64)
        policy.rag_ann_post_gpu_tail_ms_by_path_batch["full_gpu"][2] = (
            deque([30.0] * 4, maxlen=64)
        )
        policy.rag_ann_inline_collection_tail_ms_by_path_batch[
            "full_gpu"
        ][2] = deque([1.0] * 4, maxlen=64)
        return policy

    @staticmethod
    def _run_counterfactual_pre_switch(policy, decode_ms=7.0):
        def record_launch(**_kwargs):
            policy.rag_ann_pre_switch_launch_batches += 1

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=decode_ms,
            ),
            patch.object(
                policy,
                "_get_rag_ann_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
            patch.object(
                policy,
                "_launch_ann_search_side_channel",
                side_effect=record_launch,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            return policy._maybe_launch_ann_before_partition_switch(2)

    @staticmethod
    def _adaptive_policy(*, has_ann: bool, has_prefill: bool, handoff: bool):
        policy = _PolicyHarness()
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_search_pending = (
            deque(
                [
                    {
                        "enqueue_time": 0.0,
                        "queries": SimpleNamespace(shape=(1, 768)),
                    }
                ]
            )
            if has_ann
            else deque()
        )
        policy.split_prefill_batch = _Batch(1) if has_prefill else None
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = handoff
        policy.rag_prefill_exclusive_sprint_available = handoff
        policy.rag_ann_consecutive_batches = 0
        policy.rag_prefill_consecutive_chunks = 0
        policy.rag_ann_max_consecutive_batches = 1
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_ann_overlap_pressure_threshold = 0.5
        policy.rag_ann_enable_controlled_overlap = False
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}
        policy.rag_frontend_decision_counts = {}
        policy.rag_ann_budget_reject_events = 0
        policy.rag_ann_budget_rejected_requests = 0
        policy.rag_ann_slo_deferred_requests = 0
        policy.rag_lease_coalesce_min_work = 2
        policy.rag_lease_coalesce_max_wait_ms = 10
        policy.rag_lease_switch_cost_ms = 5
        policy.rag_ann_decode_batch_sizes.extend([1] * 8)
        policy.rag_ann_decode_iteration_ms.extend([6.0] * 8)
        policy.rag_ann_decode_queue_delay_ms.extend([0.0] * 8)
        return policy

    def test_partition_decode_slo_tracks_current_mixed_group(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_slo_ms = (38, 35, 32, 30)

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=2,
        ):
            self.assertEqual(policy._rag_ann_decode_slo_ms(), 35)
    def test_load_band_decode_slo_is_independent_of_stream_partition(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_slo_ms = (38, 35, 32)
        policy.rag_partition_decode_slo_batch_thresholds = (11, 12.5)
        policy.rag_partition_decode_slo_band_idx = 1

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            self.assertEqual(policy._rag_ann_decode_slo_ms(), 35)

    def test_decode_slo_load_band_uses_calibration_and_control_epoch(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_slo_ms = (38, 35, 32)
        policy.rag_partition_decode_slo_batch_thresholds = (11, 12.5)
        policy.rag_partition_decode_slo_ewma_alpha = 1.0
        policy.rag_partition_decode_slo_min_residency_ms = 120000
        policy.rag_partition_decode_slo_initial_calibration_ms = 10000
        policy._update_rag_partition_decode_slo_band(13, now=0.0)
        self.assertEqual(policy.rag_partition_decode_slo_next_update_ts, 10.0)
        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 0)

        policy._update_rag_partition_decode_slo_band(13, now=9.0)
        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 0)
        self.assertEqual(policy.rag_partition_decode_slo_band_changes, 0)

        policy._update_rag_partition_decode_slo_band(13, now=10.0)
        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 2)
        self.assertEqual(policy.rag_partition_decode_slo_band_changes, 1)
        self.assertEqual(policy.rag_partition_decode_slo_next_update_ts, 130.0)

        policy._update_rag_partition_decode_slo_band(10, now=20.0)
        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 2)

        policy._update_rag_partition_decode_slo_band(10, now=130.0)
        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 0)
        self.assertEqual(policy.rag_partition_decode_slo_band_changes, 2)

    def test_decode_slo_initial_selection_does_not_apply_hysteresis(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_slo_ms = (38, 35, 32)
        policy.rag_partition_decode_slo_batch_thresholds = (11, 12.5)
        policy.rag_partition_decode_slo_ewma_alpha = 1.0

        policy._update_rag_partition_decode_slo_band(11.2, now=0.0)

        self.assertEqual(policy.rag_partition_decode_slo_band_idx, 1)
        self.assertEqual(policy.rag_partition_decode_slo_control_updates, 1)



    def test_partition_decode_slo_maps_decode_only_by_batch_size(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_slo_ms = (38, 35, 32, 30)
        policy.running_batch = _Batch(32)

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            self.assertEqual(policy._rag_ann_decode_slo_ms(), 30)

    def test_global_decode_slo_remains_the_fallback(self):
        policy = _PolicyHarness()

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=2,
        ):
            self.assertEqual(policy._rag_ann_decode_slo_ms(), 20)

    def test_t24_hysteresis_uses_28_and_20_boundaries(self):
        policy = _PolicyHarness()
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            self.assertEqual(
                policy._apply_decode_partition_hysteresis(4, 27), 3
            )
            self.assertEqual(
                policy._apply_decode_partition_hysteresis(4, 28), 4
            )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=4,
        ):
            self.assertEqual(
                policy._apply_decode_partition_hysteresis(3, 21), 4
            )
            self.assertEqual(
                policy._apply_decode_partition_hysteresis(3, 20), 3
            )

    def test_subphase_leases_respect_decode_floor(self):
        policy = _PolicyHarness()
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=4,
        ):
            ann_idx, _ = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
            )
            prefill_idx, _ = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(ann_idx, 4)
        self.assertEqual(prefill_idx, 4)

        policy.running_batch = _Batch(16)
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            prefill_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(prefill_idx, 3)
        self.assertEqual(reason, "prefill_lease_resident")

    def test_overdue_prefill_borrows_one_partition_step(self):
        policy = _PolicyHarness()
        policy.rag_prefill_lease_frontend_sm = 84
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=4,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
                prefill_wait_ms=policy.rag_prefill_deadline_ms,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "prefill_lease_deadline_step")

    def test_prefill_lease_borrows_frontend_when_decode_prediction_is_safe(self):
        policy = _PolicyHarness()
        policy.rag_prefill_lease_frontend_sm = 84
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_ann_decode_gpu_partition_ms[1] = deque(
            [12.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[4] = deque(
            [10.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(target_idx, 1)
        self.assertEqual(reason, "prefill_lease_decode_safe")

    def test_prefill_lease_ignores_scheduler_gap_when_gpu_service_is_safe(self):
        policy = _PolicyHarness()
        policy.rag_prefill_lease_frontend_sm = 84
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_ann_decode_gpu_partition_ms[1] = deque(
            [12.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[4] = deque(
            [10.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_partition_ms[1] = deque(
            [16.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_partition_ms[4] = deque(
            [10.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(target_idx, 1)
        self.assertEqual(reason, "prefill_lease_decode_safe")

    def test_prefill_lease_rejects_gpu_service_beyond_slo_margin(self):
        policy = _PolicyHarness()
        policy.rag_prefill_lease_frontend_sm = 84
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_ann_decode_gpu_partition_ms[1] = deque(
            [16.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[4] = deque(
            [10.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "prefill_lease_decode_floor")

    def test_prefill_lease_rejects_negative_aggregate_net_gain(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch = _SplitBatch(1, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque([38.0] * 8, maxlen=64)
        policy.rag_ann_decode_gpu_partition_ms[3] = deque([32.0] * 8, maxlen=64)
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [28.0] * 8, maxlen=64
        )

        net_gain_ms = policy._rag_prefill_candidate_net_gain_ms(2, 3)
        self.assertIsNotNone(
            net_gain_ms, policy.rag_prefill_candidate_net_gain_last
        )
        self.assertLess(net_gain_ms, 0.0)
        self.assertFalse(policy._rag_prefill_lease_decode_safe(2, 3))
        self.assertEqual(policy.rag_prefill_candidate_net_gain_holds, 1)
        self.assertLess(policy.rag_prefill_candidate_net_gain_ms[-1], 0.0)

    def test_prefill_lease_accepts_positive_aggregate_net_gain(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.running_batch = _Batch(2)
        policy.split_prefill_batch = _SplitBatch(16, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque([38.0] * 8, maxlen=64)
        policy.rag_ann_decode_gpu_partition_ms[3] = deque([32.0] * 8, maxlen=64)
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [28.0] * 8, maxlen=64
        )

        net_gain_ms = policy._rag_prefill_candidate_net_gain_ms(2, 3)
        self.assertIsNotNone(
            net_gain_ms, policy.rag_prefill_candidate_net_gain_last
        )
        self.assertGreater(net_gain_ms, 0.0)
        self.assertTrue(policy._rag_prefill_lease_decode_safe(2, 3))
        self.assertEqual(policy.rag_prefill_candidate_net_gain_admissions, 1)
        self.assertGreater(policy.rag_prefill_candidate_net_gain_ms[-1], 0.0)

    def test_overdue_prefill_accepts_positive_critical_path_gain(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch = _SplitBatch(1, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [32.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [28.0] * 8, maxlen=64
        )

        self.assertTrue(
            policy._rag_prefill_lease_decode_safe(
                2,
                3,
                prefill_wait_ms=policy.rag_prefill_deadline_ms,
            )
        )
        last = policy.rag_prefill_candidate_net_gain_last
        self.assertLess(last["net_gain_ms"], 0.0)
        self.assertGreater(last["critical_path_net_gain_ms"], 0.0)
        self.assertEqual(last["admission_mode"], "tail_aware")
        self.assertEqual(policy.rag_prefill_tail_gain_admissions, 1)

    def test_overdue_prefill_admits_decode_safe_lease_for_liveness(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch = _SplitBatch(1, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [32.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [28.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )

        self.assertTrue(
            policy._rag_prefill_lease_decode_safe(
                2,
                3,
                prefill_wait_ms=policy.rag_prefill_deadline_ms,
            )
        )
        last = policy.rag_prefill_candidate_net_gain_last
        self.assertLess(last["critical_path_net_gain_ms"], 0.0)
        self.assertEqual(last["admission_mode"], "tail_liveness")
        self.assertEqual(policy.rag_prefill_tail_liveness_admissions, 1)

    def test_tail_liveness_bundle_is_bounded_by_decode_slack(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_safety_margin_ms = 5
        policy.running_batch = _Batch(32)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )

        layers, budget_ms = policy._rag_tail_liveness_layer_budget(
            2,
            requested_layers=16,
            layer_ms=4.0,
        )

        self.assertEqual(layers, 4)
        self.assertEqual(budget_ms, 17.0)

    def test_two_level_tail_liveness_keeps_half_bundle_on_safe_mixed_gc(self):
        policy = _PolicyHarness()
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(1, 8192, split_index=0)
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_safety_margin_ms = 5
        policy.running_batch = _Batch(32)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )
        # Scheduling-inclusive iteration history contains queueing caused by
        # earlier deferrals, while the candidate mixed GC remains service-safe.
        policy.rag_ann_decode_partition_ms[2] = deque(
            [90.0] * 8, maxlen=64
        )

        layers, budget_ms = policy._rag_tail_liveness_layer_budget(
            2,
            requested_layers=14,
            layer_ms=4.0,
        )

        self.assertEqual(layers, 14)
        self.assertEqual(budget_ms, 38.0)
        self.assertEqual(policy.rag_prefill_two_level_mixed_budget_bypasses, 1)

    def test_two_level_tail_liveness_defers_unsafe_endpoint_bundle(self):
        policy = _PolicyHarness()
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(1, 8192, split_index=0)
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_safety_margin_ms = 5
        policy.running_batch = _Batch(32)
        policy.rag_ann_decode_gpu_partition_ms[0] = deque(
            [38.0] * 8, maxlen=64
        )

        layers, budget_ms = policy._rag_tail_liveness_layer_budget(
            0,
            requested_layers=14,
            layer_ms=4.0,
        )

        self.assertEqual(layers, 0)
        self.assertEqual(budget_ms, 17.0)

    def test_dependency_prefill_accepts_positive_critical_path_gain_early(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.rag_prefill_handoff_active = True
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch = _SplitBatch(1, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [32.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [28.0] * 8, maxlen=64
        )

        self.assertTrue(policy._rag_prefill_lease_decode_safe(2, 3))
        last = policy.rag_prefill_candidate_net_gain_last
        self.assertLess(last["net_gain_ms"], 0.0)
        self.assertGreater(last["critical_path_net_gain_ms"], 0.0)
        self.assertEqual(last["admission_mode"], "e2e_critical")
        self.assertEqual(policy.rag_prefill_e2e_critical_admissions, 1)

    def test_dependency_prefill_holds_uncertain_critical_path_gain(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 4
        policy.rag_prefill_handoff_active = True
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch = _SplitBatch(1, 8192)
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [38.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [32.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(2, 8192)] = deque(
            [18.0] * 8, maxlen=64
        )
        policy.rag_prefill_layer_ms_by_partition_bucket[(3, 8192)] = deque(
            [20.0] * 7 + [28.0], maxlen=64
        )

        self.assertFalse(policy._rag_prefill_lease_decode_safe(2, 3))
        last = policy.rag_prefill_candidate_net_gain_last
        self.assertGreater(last["critical_path_net_gain_ms"], 0.0)
        self.assertLess(last["robust_critical_path_net_gain_ms"], 0.0)
        self.assertEqual(last["admission_mode"], "hold")
        self.assertEqual(policy.rag_prefill_e2e_critical_admissions, 0)
        self.assertEqual(policy.rag_prefill_e2e_uncertainty_holds, 1)

    def test_forced_ann_reuses_decode_safe_mixed_partition(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(16)
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                forced_ann=True,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "ann_max_defer_resident")

    def test_forced_ann_does_not_bypass_decode_floor(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(32)
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                forced_ann=True,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "ann_lease")

    def test_cohort_wait_is_bounded_and_disabled_by_default(self):
        policy = _PolicyHarness()
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": SimpleNamespace(shape=(1, 768)),
                }
            ]
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            self.assertFalse(
                policy._should_wait_for_rag_lease_cohort(
                    allow_ann=True,
                    allow_prefill=False,
                    active_ann=False,
                    forced_ann=False,
                    ann_wait_ms=1.0,
                    prefill_wait_ms=0.0,
                )
            )

            policy.rag_lease_coalesce_min_work = 2
            policy.rag_lease_coalesce_max_wait_ms = 10
            self.assertTrue(
                policy._should_wait_for_rag_lease_cohort(
                    allow_ann=True,
                    allow_prefill=False,
                    active_ann=False,
                    forced_ann=False,
                    ann_wait_ms=1.0,
                    prefill_wait_ms=0.0,
                )
            )
            self.assertFalse(
                policy._should_wait_for_rag_lease_cohort(
                    allow_ann=True,
                    allow_prefill=False,
                    active_ann=False,
                    forced_ann=False,
                    ann_wait_ms=10.0,
                    prefill_wait_ms=0.0,
                )
            )

    def test_cohort_wait_ignores_empty_selection(self):
        policy = _PolicyHarness()
        policy.rag_lease_coalesce_min_work = 2
        policy.rag_lease_coalesce_max_wait_ms = 10

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            self.assertFalse(
                policy._should_wait_for_rag_lease_cohort(
                    allow_ann=False,
                    allow_prefill=False,
                    active_ann=False,
                    forced_ann=False,
                    ann_wait_ms=0.0,
                    prefill_wait_ms=0.0,
                )
            )

    def test_full_policy_can_use_decode_batch_adaptive_partition(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(8)
        policy.rag_ann_lease_frontend_sm = -1
        policy.rag_prefill_lease_frontend_sm = -1

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=2,
        ):
            ann_idx, ann_reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
            )
            prefill_idx, prefill_reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
            )

        self.assertEqual(ann_idx, 2)
        self.assertEqual(ann_reason, "ann_adaptive_resident")
        self.assertEqual(prefill_idx, 2)
        self.assertEqual(prefill_reason, "prefill_adaptive_resident")

    def test_dependency_handoff_retains_current_mixed_partition(self):
        policy = _PolicyHarness()

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
                dependency_handoff_pending=True,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "dependency_handoff_resident")
        self.assertEqual(policy.rag_dependency_lease_holds, 1)

    def test_logical_handoff_does_not_extend_partition_residency(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_search_pending = deque()
        policy.split_prefill_batch = None
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = False
        policy.rag_ann_frontend_residency_until = 0.0
        policy.rag_partition_slo_floor_enabled = False
        policy.rag_partition_hysteresis_bs = 0
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=True
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=0.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertFalse(policy.rag_frontend_plan.frontend_needed)
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, 5)
        self.assertEqual(policy.rag_frontend_plan.target_reason, "decode_only")
        self.assertEqual(policy.rag_dependency_lease_holds, 0)
    def test_concrete_prefill_bypasses_dependency_residency(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_prefill_lease_frontend_sm = -1
        policy.ann_search_pending = deque()
        policy.split_prefill_batch = _Batch(1)
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = True
        policy.rag_ann_consecutive_batches = 0
        policy.rag_prefill_consecutive_chunks = 0
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=True
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=0.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=1,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertTrue(policy.rag_frontend_plan.allow_prefill)
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, 4)
        self.assertNotEqual(
            policy.rag_frontend_plan.target_reason,
            "dependency_handoff_resident",
        )
        self.assertEqual(policy.rag_dependency_lease_holds, 0)

    def test_ann_completion_opens_bounded_dependency_lease(self):
        policy = _PolicyHarness()
        policy.rag_lease_coalesce_max_wait_ms = 10

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            policy._extend_rag_ann_frontend_residency()

        self.assertAlmostEqual(
            policy.rag_ann_frontend_residency_until, 100.02
        )

    def test_decode_gc_ann_completion_does_not_open_frontend_residency(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_lease_coalesce_max_wait_ms = 10

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            policy._extend_rag_ann_frontend_residency()

        self.assertEqual(policy.rag_ann_frontend_residency_until, 0.0)

    def test_decode_gc_handoff_carries_request_start_without_residency(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_handoffs["req-1"] = (99.0, 99.9, 100.1)
        policy.rag_ann_frontend_residency_until = 123.0

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            completed = policy._complete_rag_ann_handoff("req-1")

        self.assertTrue(completed)
        self.assertEqual(policy.rag_ann_frontend_residency_until, 123.0)
        self.assertEqual(policy.rag_ann_frontend_residency_releases, 0)
        self.assertEqual(
            policy.rag_completed_handoff_timing["req-1"],
            (99.0, 99.9, 100.0),
        )

    def test_consuming_handoff_releases_global_residency(self):
        policy = _PolicyHarness()
        policy.rag_ann_handoffs["req-1"] = (99.99, 100.04)
        policy.rag_ann_frontend_residency_until = 100.02

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            completed = policy._complete_rag_ann_handoff("req-1")

        self.assertTrue(completed)
        self.assertEqual(policy.rag_ann_frontend_residency_until, 0.0)
        self.assertEqual(policy.rag_ann_frontend_residency_releases, 1)
        self.assertEqual(
            policy.rag_completed_handoff_timing["req-1"], (99.99, 100.0)
        )

    def test_decode_prediction_prefers_stable_partition_history(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_iteration_ms.extend([40.0] * 8)
        policy.rag_ann_decode_partition_ms[5] = deque([8.0] * 8, maxlen=64)

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            prediction = policy._rag_ann_decode_prediction_ms()

        self.assertAlmostEqual(prediction, 8.0)

    def test_candidate_prediction_prefers_same_batch_bucket(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(9)
        bucket = policy._rag_decode_batch_bucket(9)
        policy.rag_ann_decode_partition_batch_ms[5] = {
            bucket: deque([8.0] * 8, maxlen=64)
        }
        policy.rag_ann_decode_partition_ms[3] = deque(
            [30.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            prediction = policy._rag_ann_decode_prediction_ms(3, decode_bs=9)

        self.assertAlmostEqual(prediction, 18.0)

    def test_candidate_prediction_prefers_gpu_step_over_stalled_interval(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(9)
        bucket = policy._rag_decode_batch_bucket(9)
        policy.rag_ann_decode_gpu_partition_batch_ms[3] = {
            bucket: deque([11.0] * 8, maxlen=64)
        }
        policy.rag_ann_decode_partition_batch_ms[3] = {
            bucket: deque([80.0] * 8, maxlen=64)
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            prediction = policy._rag_ann_decode_prediction_ms(3, decode_bs=9)

        self.assertAlmostEqual(prediction, 11.0)

    def test_decode_completion_records_gpu_step_by_partition_and_bucket(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(9)
        bucket = policy._rag_decode_batch_bucket(9)

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            policy._record_rag_decode_completion(True, 12.5)

        self.assertEqual(list(policy.rag_ann_decode_gpu_ms), [12.5])
        self.assertEqual(
            list(policy.rag_ann_decode_gpu_partition_ms[3]), [12.5]
        )
        self.assertEqual(
            list(policy.rag_ann_decode_gpu_partition_batch_ms[3][bucket]),
            [12.5],
        )

    def test_system_load_rejects_high_batch_or_unsafe_candidate(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_iteration_ms.extend([8.0] * 8)
        policy.rag_ann_decode_queue_delay_ms.extend([0.0] * 8)
        policy.rag_ann_decode_batch_sizes.extend([20] * 8)

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            high_batch = policy._rag_system_load_state(4)

        self.assertFalse(high_batch["low"])
        self.assertIn("batch", high_batch["reason"])

        policy.rag_ann_decode_batch_sizes = deque([1] * 8, maxlen=64)
        bucket = policy._rag_decode_batch_bucket(
            policy.running_batch.batch_size()
        )
        policy.rag_ann_decode_partition_batch_ms[4] = {
            bucket: deque([18.0] * 8, maxlen=64)
        }
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            unsafe_candidate = policy._rag_system_load_state(4)

        self.assertFalse(unsafe_candidate["low"])
        self.assertIn("candidate", unsafe_candidate["reason"])

    def test_low_load_fast_path_is_independent_of_request_wait(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )
        policy.rag_lease_switch_cost_ms = 0

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=500.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_only_guard",
                    "predicted_ms": 8.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.allow_ann)
        self.assertEqual(plan.reason, "ann_fast_path")
        self.assertEqual(policy.rag_ann_budget_reject_events, 0)

    def test_prefill_starts_on_safe_floor_when_aggressive_candidate_is_unsafe(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.running_batch = _Batch(32)
        policy.rag_ann_decode_step_slo_ms = 60
        policy.rag_prefill_lease_frontend_sm = 72
        policy.rag_lease_coalesce_min_work = 1
        policy.rag_lease_coalesce_max_wait_ms = 0
        policy.rag_lease_switch_cost_ms = 0
        floor_idx = policy._adaptive_frontend_stream_idx(32)
        policy.rag_ann_decode_gpu_partition_ms[floor_idx] = deque(
            [30.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[2] = deque(
            [70.0] * 8, maxlen=64
        )

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_guard",
                    "predicted_ms": 70.0,
                    "slo_ms": 60.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertTrue(policy.rag_frontend_plan.allow_prefill)
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, floor_idx)
        self.assertEqual(policy.rag_prefill_floor_safe_admissions, 1)

    def test_max_defer_overrides_fast_path_switch_cost(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )
        policy.rag_lease_switch_cost_ms = 100

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=75.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": True,
                    "reason": "max_defer",
                    "predicted_ms": 10.8,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertTrue(plan.allow_ann)
        self.assertTrue(plan.forced_ann)
        self.assertTrue(plan.launch_ann_early)
        self.assertEqual(plan.reason, "ann_max_defer")
        self.assertEqual(plan.target_stream_idx, 4)
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)

    def test_decode_slo_promotes_dynamic_partition_floor(self):
        policy = _PolicyHarness()
        policy.running_batch = _Batch(8)
        policy.rag_ann_lease_frontend_sm = -1
        policy.rag_prefill_lease_frontend_sm = -1
        policy.rag_ann_decode_partition_ms[2] = deque(
            [30.0] * 8, maxlen=64
        )

        policy._maybe_raise_rag_decode_partition_floor(2)

        self.assertEqual(policy.rag_partition_decode_floor_sm, 48)
        self.assertEqual(policy.rag_partition_slo_promotions, 1)
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=2,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "ann_adaptive_slo_floor")
    def test_disabled_decode_slo_floor_does_not_promote(self):
        policy = _PolicyHarness()
        policy.rag_partition_slo_floor_enabled = False
        policy.rag_ann_decode_partition_ms[2] = deque(
            [30.0] * 8, maxlen=64
        )

        policy._maybe_raise_rag_decode_partition_floor(2)

        self.assertEqual(policy.rag_partition_decode_floor_sm, 0)
        self.assertEqual(policy.rag_partition_slo_promotions, 0)

    def test_decode_slo_floor_demotes_after_sustained_safe_window(self):
        policy = _PolicyHarness()
        policy.rag_partition_decode_floor_sm = 48
        policy.rag_ann_decode_partition_ms[3] = deque(
            [5.0] * 64, maxlen=64
        )

        for _ in range(8):
            policy._maybe_raise_rag_decode_partition_floor(3)

        self.assertEqual(policy.rag_partition_decode_floor_sm, 36)
        self.assertEqual(policy.rag_partition_slo_demotions, 1)
        self.assertEqual(policy.rag_partition_floor_safe_observations, 0)

        for _ in range(8):
            policy._maybe_raise_rag_decode_partition_floor(3)
        self.assertEqual(policy.rag_partition_decode_floor_sm, 36)
        self.assertEqual(policy.rag_partition_slo_demotions, 1)

    def test_last_mixed_violation_enables_decode_only_fallback(self):
        policy = _PolicyHarness()
        last_mixed_idx = policy.real_sm_group_num - 2
        policy.rag_ann_decode_partition_ms[last_mixed_idx] = deque(
            [30.0] * 8, maxlen=64
        )

        policy._maybe_raise_rag_decode_partition_floor(last_mixed_idx)
        policy._maybe_raise_rag_decode_partition_floor(last_mixed_idx)

        self.assertTrue(
            policy.rag_partition_decode_only_fallback_required
        )
        self.assertEqual(
            policy.rag_partition_decode_only_fallback_activations, 1
        )

        policy.rag_ann_decode_partition_ms[last_mixed_idx] = deque(
            [10.0] * 64, maxlen=64
        )
        policy._maybe_raise_rag_decode_partition_floor(last_mixed_idx)
        self.assertFalse(
            policy.rag_partition_decode_only_fallback_required
        )

    def test_safe_mixed_partition_is_held_below_release_threshold(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [10.0] * 8, maxlen=64
        )
        # Scheduling delay from an earlier transition must not make a
        # service-safe mixed partition look unsafe and trigger another switch.
        policy.rag_ann_decode_partition_ms[3] = deque(
            [90.0] * 8, maxlen=64
        )
        policy.rag_last_frontend_activity_ts = 99.998

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "decode_safe_mixed_hold")
        self.assertEqual(policy.rag_partition_safe_mixed_holds, 1)

    def test_safe_mixed_partition_expires_after_max_hold(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_last_frontend_activity_ts = 99.99
        policy.rag_ann_decode_partition_ms[3] = deque(
            [10.0] * 8, maxlen=64
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 5)
        self.assertEqual(reason, "decode_only")

    def test_safe_mixed_cost_model_avoids_expensive_round_trip(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_prefill_dynamic_lease = True
        policy.rag_request_arrival_times = deque(
            [99.0, 99.25, 99.5, 99.75], maxlen=64
        )
        policy.rag_last_frontend_activity_ts = 99.99
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [12.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[5] = deque(
            [10.0] * 8, maxlen=64
        )
        policy.rag_partition_transition_pair_latency_ms = {
            (3, 5): deque([30.0] * 8, maxlen=64),
            (5, 3): deque([30.0] * 8, maxlen=64),
        }

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "decode_safe_mixed_hold")
        self.assertEqual(
            policy.rag_partition_safe_mixed_cost_admissions, 1
        )
        self.assertGreater(
            policy.rag_partition_safe_mixed_last_cost_decision[
                "round_trip_cost_ms"
            ],
            policy.rag_partition_safe_mixed_last_cost_decision[
                "stay_cost_ms"
            ],
        )

    def test_safe_mixed_cost_model_rejects_costly_residency(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_prefill_dynamic_lease = True
        policy.rag_request_arrival_times = deque(
            [99.0, 99.25, 99.5, 99.75], maxlen=64
        )
        policy.rag_last_frontend_activity_ts = 99.99
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [13.0] * 8, maxlen=64
        )
        policy.rag_ann_decode_gpu_partition_ms[5] = deque(
            [10.0] * 8, maxlen=64
        )
        policy.rag_partition_transition_pair_latency_ms = {
            (3, 5): deque([5.0] * 8, maxlen=64),
            (5, 3): deque([5.0] * 8, maxlen=64),
        }

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 5)
        self.assertEqual(reason, "decode_only")
        self.assertEqual(
            policy.rag_partition_safe_mixed_cost_rejections, 1
        )

    def test_unsafe_mixed_partition_returns_to_decode_only(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_ann_decode_gpu_partition_ms[3] = deque(
            [14.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 5)
        self.assertEqual(reason, "decode_only")

    def test_decode_fallback_bypasses_safe_mixed_hold(self):
        policy = _PolicyHarness()
        policy.rag_partition_safe_mixed_hold = True
        policy.rag_partition_decode_only_fallback_required = True
        policy.rag_ann_decode_partition_ms[3] = deque(
            [5.0] * 8, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=False,
                allow_ann=False,
                allow_prefill=False,
                active_ann=False,
            )

        self.assertEqual(target_idx, 5)
        self.assertEqual(reason, "decode_only")

    def test_handoff_does_not_create_synthetic_prefill_deadline(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_search_pending = deque()
        policy.split_prefill_batch = _Batch(1)
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = True
        policy.rag_prefill_consecutive_chunks = 0
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_ann_max_consecutive_batches = 1
        policy.rag_ann_overlap_pressure_threshold = 0.5
        policy.rag_ann_enable_controlled_overlap = False
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=0.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=4,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertFalse(policy.rag_frontend_plan.allow_prefill)
        self.assertEqual(
            policy.rag_frontend_plan.reason, "decode_budget"
        )
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, 5)

    def test_decode_gc_handoff_uses_bounded_mixed_lease(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=40.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch+candidate",
                    "batch_p95": 32.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 6.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.immediate_handoff)
        self.assertTrue(plan.dependency_prefill_due)
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertEqual(plan.reason, "handoff_bounded")
        self.assertEqual(plan.target_reason, "prefill_handoff_step")
        self.assertEqual(plan.target_stream_idx, 3)

    def test_decode_gc_overdue_handoff_remains_dependency_due(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=600.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch+candidate",
                    "batch_p95": 32.0,
                    "queue_p95_ms": 8.0,
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 6.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.dependency_prefill_due)
        self.assertTrue(plan.immediate_handoff)
        self.assertEqual(plan.reason, "handoff_bounded")

    def test_decode_gc_overdue_handoff_uses_dynamic_endpoint_bundle(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_prefill_exclusive_sprint_available = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 8
        policy.running_batch = _Batch(32)
        policy.split_prefill_batch.extend_num_tokens = 1023
        policy.split_prefill_batch.split_index = 0
        policy.model_config = SimpleNamespace(num_hidden_layers=28)

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=50.0),
            patch.object(
                policy,
                "_oldest_dependency_prefill_wait_ms",
                return_value=50.0,
            ),
            patch.object(
                policy,
                "_rag_prefill_lease_shape",
                return_value=(6, 1.0, 1000.0),
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch+candidate",
                    "batch_p95": 32.0,
                    "queue_p95_ms": 8.0,
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 6.5,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.dependency_prefill_due)
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertTrue(plan.prefill_sprint)
        self.assertTrue(plan.prefill_exclusive_sprint)
        self.assertEqual(plan.target_stream_idx, 0)
        self.assertEqual(plan.target_reason, "prefill_deadline_endpoint")
        self.assertEqual(plan.prefill_forward_count, 6)
        self.assertEqual(
            policy.rag_prefill_exclusive_sprint_block_counts,
            {"deadline_granted": 1},
        )

    def test_decode_gc_ann_yields_to_dependency_prefill_transition(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_prefill=True,
            target_stream_idx=2,
            immediate_handoff=True,
            dependency_prefill_due=False,
        )

        self.assertTrue(policy._rag_ann_yields_to_prefill_transition(5))
        self.assertFalse(policy._rag_ann_yields_to_prefill_transition(2))

    def test_dependency_prefill_burst_forces_recovery_after_three_chunks(self):
        policy = _PolicyHarness()
        policy.rag_frontend_decision_counts = {}
        policy.rag_frontend_prefill_ran = True
        policy.rag_frontend_ann_launched = True
        policy.rag_frontend_plan = SimpleNamespace(
            prefill_sprint=False,
            prefill_exclusive_sprint=True,
            dependency_prefill_due=True,
        )
        policy.rag_prefill_max_consecutive_chunks = 3
        policy.rag_ann_frontend_residency_until = 100.0

        policy._finalize_rag_frontend_actions()
        policy._finalize_rag_frontend_actions()

        self.assertFalse(policy.rag_decode_recovery_pending)
        self.assertEqual(policy.rag_dependency_prefill_admissions, 2)
        self.assertEqual(policy.rag_decode_recovery_requests, 0)

        policy._finalize_rag_frontend_actions()

        self.assertTrue(policy.rag_decode_recovery_pending)
        self.assertEqual(policy.rag_decode_recovery_requests, 1)
        self.assertEqual(policy.rag_dependency_prefill_admissions, 3)
        self.assertEqual(policy.rag_ann_frontend_residency_until, 0.0)

    def test_decode_unsafe_prefill_burst_forces_recovery_window(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_search_pending = deque()
        policy.split_prefill_batch = _Batch(1)
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = True
        policy.rag_prefill_consecutive_chunks = 2
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_ann_max_consecutive_batches = 1
        policy.rag_ann_overlap_pressure_threshold = 0.5
        policy.rag_ann_enable_controlled_overlap = False
        policy.rag_partition_decode_only_fallback_required = True
        policy.rag_ann_frontend_residency_until = 100.02
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}
        policy.rag_frontend_decision_counts = {}

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=True
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=500.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=4,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertFalse(plan.allow_ann)
        self.assertFalse(plan.allow_prefill)
        self.assertFalse(plan.frontend_needed)
        self.assertEqual(plan.reason, "decode_recovery")
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertEqual(plan.target_reason, "decode_recovery")
        self.assertTrue(policy.rag_decode_recovery_pending)
        self.assertEqual(policy.rag_decode_recovery_requests, 1)
        self.assertEqual(policy.rag_ann_frontend_residency_until, 0.0)

    def test_recovery_clears_after_real_decode_only_step(self):
        policy = _PolicyHarness()
        policy.rag_decode_recovery_pending = True
        policy.rag_prefill_consecutive_chunks = 2
        policy.rag_ann_consecutive_batches = 1
        policy.rag_ann_last_decode_completion_ts = None
        policy.rag_ann_last_decode_stream_idx = None

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._record_rag_decode_completion(True)

        self.assertFalse(policy.rag_decode_recovery_pending)
        self.assertEqual(policy.rag_decode_recovery_windows, 1)
        self.assertEqual(policy.rag_prefill_consecutive_chunks, 0)
        self.assertEqual(policy.rag_ann_consecutive_batches, 0)

    def test_recovery_keeps_unfinished_exclusive_prefill_lease_consumed(self):
        policy = _PolicyHarness()
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_prefill_exclusive_sprint_available = False
        policy.split_prefill_batch = _SplitBatch(
            1, 1023, split_index=8
        )
        policy.rag_decode_recovery_pending = True
        policy.rag_ann_last_decode_completion_ts = None
        policy.rag_ann_last_decode_stream_idx = None

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._record_rag_decode_completion(True)

        self.assertFalse(policy.rag_decode_recovery_pending)
        self.assertFalse(policy.rag_prefill_exclusive_sprint_available)
        self.assertEqual(policy.rag_decode_recovery_windows, 1)

    def test_recovery_does_not_rearm_completed_exclusive_prefill(self):
        policy = _PolicyHarness()
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_prefill_exclusive_sprint_available = False
        policy.split_prefill_batch = _SplitBatch(
            1, 1023, split_index=32
        )
        policy.split_prefill_batch.split_prefill_finished = True
        policy.rag_decode_recovery_pending = True
        policy.rag_ann_last_decode_completion_ts = None
        policy.rag_ann_last_decode_stream_idx = None

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._record_rag_decode_completion(True)

        self.assertFalse(policy.rag_decode_recovery_pending)
        self.assertFalse(policy.rag_prefill_exclusive_sprint_available)
        self.assertEqual(policy.rag_decode_recovery_windows, 1)

    def test_handoff_prefill_forces_one_real_decode_only_window(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=False, handoff=False
        )
        policy.rag_handoff_decode_release_pending = True

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertEqual(plan.reason, "handoff_decode_release")
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertTrue(policy.rag_handoff_decode_release_pending)

        policy.rag_ann_last_decode_completion_ts = None
        policy.rag_ann_last_decode_stream_idx = None
        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=101.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._record_rag_decode_completion(True)

        self.assertFalse(policy.rag_handoff_decode_release_pending)
        self.assertEqual(policy.rag_handoff_decode_release_windows, 1)

    def test_pairwise_switch_cost_uses_configured_p95(self):
        policy = _PolicyHarness()
        policy.rag_partition_switch_latency_ms.extend([0.02] * 8)
        policy.rag_partition_switch_pair_latency_ms[(5, 4)] = deque(
            [1.0, 2.0, 19.0, 20.0, 21.0, 22.0, 25.0, 30.0],
            maxlen=64,
        )

        pair_cost = policy._rag_lease_switch_cost_estimate_ms(5, 4)
        global_cost = policy._rag_lease_switch_cost_estimate_ms(5, 3)

        self.assertGreater(pair_cost, 25.0)
        self.assertAlmostEqual(global_cost, 0.02)

    def test_full_transition_cost_includes_deferred_drain_time(self):
        policy = _PolicyHarness()
        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            policy._track_rag_partition_transition_intent(5, 4)

        transition_ms = policy._complete_rag_partition_transition(
            5,
            4,
            completed_at=100.04,
            local_switch_latency_ms=0.1,
        )

        self.assertAlmostEqual(transition_ms, 40.0)
        self.assertAlmostEqual(
            policy._rag_lease_switch_cost_estimate_ms(5, 4), 40.0
        )
        self.assertEqual(policy.rag_partition_transition_abandoned, 0)
        self.assertIsNone(policy.rag_partition_transition_intent)

    def test_decode_prediction_uses_partition_batch_and_overlap_class(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_gpu_partition_batch_ms[3] = {
            5: deque([30.0] * 8, maxlen=64)
        }
        policy.rag_ann_decode_gpu_partition_batch_overlap_ms[3] = {
            5: {
                "ann": deque([11.0] * 8, maxlen=64),
                "solo": deque([7.0] * 8, maxlen=64),
            }
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            ann_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="ann"
            )
            solo_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="solo"
            )
            fallback_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="prefill"
            )

        self.assertAlmostEqual(ann_prediction, 11.0)
        self.assertAlmostEqual(solo_prediction, 7.0)
        self.assertAlmostEqual(fallback_prediction, 30.0)

    def test_decode_prediction_discards_completed_warmup_regime(self):
        policy = _PolicyHarness()
        history = deque([400.0] * 63 + [16.0], maxlen=64)
        policy.rag_ann_decode_gpu_partition_batch_overlap_ms[3] = {
            5: {"solo": history}
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            cold_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="solo"
            )
            history.extend([16.0, 16.0, 16.0])
            mature_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="solo"
            )

        self.assertAlmostEqual(cold_prediction, 20.0)
        self.assertAlmostEqual(mature_prediction, 16.0)

    def test_decode_slo_prediction_includes_overlap_iteration_delay(self):
        policy = _PolicyHarness()
        policy.rag_ann_decode_gpu_partition_batch_overlap_ms[3] = {
            5: {"prefill": deque([11.0] * 8, maxlen=64)}
        }
        policy.rag_ann_decode_partition_batch_overlap_ms[3] = {
            5: {"prefill": deque([28.0] * 8, maxlen=64)}
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            gpu_prediction = policy._rag_ann_decode_prediction_ms(
                3, decode_bs=8, overlap_class="prefill"
            )
            slo_prediction = policy._rag_ann_decode_prediction_ms(
                3,
                decode_bs=8,
                include_scheduling=True,
                overlap_class="prefill",
            )

        self.assertAlmostEqual(gpu_prediction, 11.0)
        self.assertAlmostEqual(slo_prediction, 28.0)

    def test_decode_prediction_is_bound_to_launch_context(self):
        policy = _PolicyHarness()
        policy.rag_frontend_plan = SimpleNamespace(
            predicted_decode_ms=30.0,
            candidate_stream_idx=3,
            target_stream_idx=3,
        )
        launch_context = {
            "stream_idx": 5,
            "batch_size": 8,
            "overlap_class": "ann",
            "predicted_ms": 12.0,
            "slo_ms": 20.0,
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            policy._record_rag_decode_completion(
                True, 10.0, launch_context=launch_context
            )

        self.assertEqual(policy.rag_decode_prediction_samples, 1)
        self.assertEqual(policy.rag_decode_prediction_partition_mismatches, 0)
        self.assertEqual(policy.rag_decode_prediction_abs_error_ms[-1], 2.0)
        self.assertEqual(
            policy.rag_decode_prediction_pairs[-1]["prediction_source"],
            "decode_launch",
        )
        self.assertEqual(
            list(
                policy.rag_ann_decode_gpu_partition_batch_overlap_ms[5][5][
                    "ann"
                ]
            ),
            [10.0],
        )

    def test_decode_iteration_prediction_is_bound_to_launch_context(self):
        policy = _PolicyHarness()
        policy.rag_ann_last_decode_completion_ts = 99.99
        policy.rag_ann_last_decode_stream_idx = 3
        launch_context = {
            "stream_idx": 3,
            "batch_size": 8,
            "overlap_class": "prefill",
            "predicted_ms": 10.0,
            "predicted_iteration_ms": 20.0,
            "slo_ms": 35.0,
        }

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.03,
            ),
        ):
            policy._record_rag_decode_completion(
                True, 10.0, launch_context=launch_context
            )

        self.assertEqual(policy.rag_decode_iteration_prediction_samples, 1)
        self.assertEqual(
            policy.rag_decode_iteration_prediction_false_safe, 1
        )
        self.assertAlmostEqual(
            policy.rag_decode_iteration_prediction_abs_error_ms[-1], 20.0
        )
        self.assertAlmostEqual(
            policy.rag_ann_decode_partition_batch_overlap_ms[3][5][
                "prefill"
            ][-1],
            40.0,
        )

    def test_dynamic_prefill_lease_tracks_length_and_e2e_slack(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.split_prefill_batch = _SplitBatch(1, 4096)

        with patch.object(
            policy,
            "_rag_ann_decode_prediction_ms",
            return_value=16.0,
        ):
            short_count, short_layer_ms, _ = policy._rag_prefill_lease_shape(3)
            policy.split_prefill_batch = _SplitBatch(1, 8192)
            long_count, long_layer_ms, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(short_count, 4)
        self.assertEqual(long_count, 2)
        self.assertEqual(short_layer_ms, 5.0)
        self.assertEqual(long_layer_ms, 10.0)

        policy.rag_e2e_slo_ms = 100.0
        req = SimpleNamespace(rag_e2e_started_at=99.95)
        policy.split_prefill_batch = _SplitBatch(1, 4096, reqs=[req])
        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=16.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
        ):
            urgent_count, _, slack_ms = policy._rag_prefill_lease_shape(3)

        self.assertAlmostEqual(slack_ms, 50.0)
        self.assertEqual(urgent_count, 8)

    def test_two_level_prefill_lease_uses_half_or_full_model(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=10.0
            ),
        ):
            half_count, _, _ = policy._rag_prefill_lease_shape(3)
            policy.running_batch = _Batch(0)
            full_count, _, _ = policy._rag_prefill_lease_shape(3)
            policy.running_batch = _Batch(32)
            policy.split_prefill_batch.split_index = 14
            final_half_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(half_count, 14)
        self.assertEqual(full_count, 28)
        self.assertEqual(final_half_count, 14)

    def test_two_level_prefill_lease_keeps_coarse_bundle_at_moderate_risk(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)
        policy.running_batch = _Batch(32)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=16.0
            ),
        ):
            risk_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(risk_count, 14)
        self.assertEqual(policy.rag_prefill_short_bundle_activations, 1)
        self.assertEqual(policy.rag_prefill_risk_bundle_activations, 0)

    def test_short_request_batch_does_not_use_aggregate_token_length(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(
            4, 4096, split_index=0, extend_lens=[512, 512, 512, 512]
        )
        policy.running_batch = _Batch(32)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=16.0
            ),
        ):
            layer_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(layer_count, 14)
        self.assertEqual(policy.rag_prefill_aggregate_false_long_avoided, 1)
        self.assertEqual(policy.rag_prefill_long_request_bundle_activations, 0)

    def test_mixed_prefill_uses_candidate_service_not_iteration_queue(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(
            1, 512, split_index=0, extend_lens=[512]
        )
        policy.running_batch = _Batch(32)

        def decode_prediction(
            _stream_idx, *, include_scheduling=False, overlap_class=None
        ):
            del overlap_class
            return 60.0 if include_scheduling else 10.0

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=decode_prediction,
            ),
        ):
            layer_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(layer_count, 14)
        self.assertEqual(policy.rag_prefill_mixed_service_guard_uses, 1)
        self.assertEqual(policy.rag_prefill_severe_risk_bundle_activations, 0)

    def test_mixed_iteration_pressure_uses_full_bundle_when_service_is_safe(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.rag_prefill_lease_decode_ratio = 3.0
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(
            1, 512, split_index=0, extend_lens=[512]
        )
        policy.running_batch = _Batch(32)

        def decode_prediction(
            _stream_idx, *, include_scheduling=False, overlap_class=None
        ):
            del overlap_class
            return 60.0 if include_scheduling else 10.0

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=decode_prediction,
            ),
        ):
            layer_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(layer_count, 28)
        self.assertEqual(
            policy.rag_prefill_iteration_pressure_full_bundle_bypasses, 1
        )

    def test_calibrated_prefill_bundle_uses_nominal_decode_not_causal_floor(self):
        policy = _PolicyHarness()
        policy.rag_legacy_calibrated_bundle = object()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(
            1, 512, split_index=0, extend_lens=[512]
        )
        policy.running_batch = _Batch(32)

        def decode_prediction(
            _stream_idx,
            _decode_bs=None,
            *,
            include_scheduling=False,
            percentile=None,
            overlap_class=None,
            apply_causal_guard=True,
        ):
            del include_scheduling, percentile, overlap_class
            return 60.0 if apply_causal_guard else 10.0

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=decode_prediction,
            ) as predictor,
        ):
            layer_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(layer_count, 14)
        self.assertEqual(policy.rag_prefill_severe_risk_bundle_activations, 0)
        self.assertTrue(predictor.call_args_list)
        self.assertTrue(
            all(
                call.kwargs.get("apply_causal_guard") is False
                for call in predictor.call_args_list
            )
        )

    def test_two_level_prefill_lease_uses_small_bundle_at_severe_risk(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.model_config = SimpleNamespace(num_hidden_layers=28)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)
        policy.running_batch = _Batch(32)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=1.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=21.0
            ),
        ):
            risk_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(risk_count, 8)
        self.assertEqual(policy.rag_prefill_risk_bundle_activations, 1)

    def test_two_level_long_prefill_uses_half_bundle_until_severe_risk(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 8
        policy.model_config = SimpleNamespace(num_hidden_layers=36)
        policy.split_prefill_batch = _SplitBatch(
            1, 4095, split_index=0, extend_lens=[4095]
        )
        policy.running_batch = _Batch(32)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=2.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=10.0
            ),
        ):
            safe_count, _, _ = policy._rag_prefill_lease_shape(3)

        with (
            patch.object(
                policy, "_predict_rag_prefill_layer_ms", return_value=2.0
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=30.0
            ),
        ):
            severe_count, _, _ = policy._rag_prefill_lease_shape(3)

        self.assertEqual(safe_count, 18)
        self.assertEqual(severe_count, 8)
        self.assertEqual(policy.rag_prefill_long_request_bundle_activations, 2)
        self.assertEqual(policy.rag_prefill_severe_risk_bundle_activations, 1)

    def test_prefill_switch_benefit_uses_remaining_lease_horizon(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_max_consecutive_chunks = 12
        policy.model_config = SimpleNamespace(num_hidden_layers=32)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)

        with (
            patch.object(
                policy, "_rag_lease_work_counts", return_value=(0, 1)
            ),
            patch.object(
                policy,
                "_rag_prefill_lease_shape",
                return_value=(16, 2.0, None),
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=None
            ),
        ):
            benefit_ms = policy._estimate_rag_lease_switch_benefit_ms(
                current_idx=5,
                candidate_idx=2,
                allow_ann=False,
                allow_prefill=True,
                predicted_decode_ms=None,
            )

        self.assertEqual(benefit_ms, 64.0)
        self.assertEqual(policy.rag_lease_last_horizon_bundles, 2)
        self.assertEqual(policy.rag_lease_horizon_bundle_histogram, {"2": 1})
        self.assertEqual(
            policy.rag_lease_horizon_layer_histogram, {"32": 1}
        )

    def test_prefill_candidate_search_balances_decode_and_bundle_cost(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_decode_step_slo_ms = 35.0
        policy.model_config = SimpleNamespace(num_hidden_layers=32)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)
        policy.running_batch = _Batch(32)

        def decode_prediction(stream_idx, **kwargs):
            if kwargs.get("overlap_class") == "solo":
                return 28.0
            return {2: 40.0, 3: 31.0, 4: 27.0}[stream_idx]

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=decode_prediction,
            ),
            patch.object(
                policy,
                "_predict_rag_prefill_layer_ms",
                side_effect=lambda stream_idx, _tokens: {
                    2: 1.0,
                    3: 1.2,
                    4: 2.0,
                }[stream_idx],
            ),
            patch.object(
                policy,
                "_rag_lease_transition_cost_estimate_ms",
                return_value=0.0,
            ),
        ):
            selected_idx = policy._select_rag_prefill_mixed_candidate(2, 4)

        self.assertEqual(selected_idx, 3)
        self.assertTrue(policy.rag_prefill_partition_selection_last["available"])
        self.assertEqual(
            policy.rag_prefill_partition_selection_counts, {"3": 1}
        )

    def test_prefill_candidate_search_uses_gpu_service_guard(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_decode_step_slo_ms = 35.0
        policy.model_config = SimpleNamespace(num_hidden_layers=32)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)
        policy.running_batch = _Batch(32)
        prediction_calls = []

        def decode_prediction(stream_idx, **kwargs):
            prediction_calls.append((stream_idx, dict(kwargs)))
            if kwargs.get("include_scheduling"):
                return 60.0
            if kwargs.get("overlap_class") == "solo":
                return 14.0
            return {2: 22.0, 3: 20.0, 4: 18.0}[stream_idx]

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=decode_prediction,
            ),
            patch.object(
                policy,
                "_predict_rag_prefill_layer_ms",
                side_effect=lambda stream_idx, _tokens: {
                    2: 0.5,
                    3: 1.0,
                    4: 2.0,
                }[stream_idx],
            ),
            patch.object(
                policy,
                "_rag_lease_transition_cost_estimate_ms",
                return_value=0.0,
            ),
        ):
            selected_idx = policy._select_rag_prefill_mixed_candidate(2, 4)

        self.assertEqual(selected_idx, 2)
        self.assertTrue(policy.rag_prefill_partition_selection_last["selected_safe"])
        self.assertTrue(prediction_calls)
        self.assertTrue(
            all(
                call_kwargs.get("include_scheduling") is False
                for _, call_kwargs in prediction_calls
            )
        )

    def test_calibrated_candidate_scores_nominal_but_admits_with_guard(self):
        policy = _PolicyHarness()
        policy.rag_legacy_calibrated_bundle = object()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_two_level_bundles = True
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_decode_step_slo_ms = 35.0
        policy.model_config = SimpleNamespace(num_hidden_layers=32)
        policy.split_prefill_batch = _SplitBatch(1, 512, split_index=0)
        policy.running_batch = _Batch(32)

        def nominal(stream_idx, **kwargs):
            if kwargs.get("overlap_class") == "solo":
                return 14.0
            return {2: 10.0, 3: 12.0, 4: 30.0}[stream_idx]

        def guarded(stream_idx, **kwargs):
            if kwargs.get("overlap_class") == "solo":
                return 14.0
            return {2: 40.0, 3: 30.0, 4: 20.0}[stream_idx]

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_rag_prefill_nominal_decode_prediction_ms",
                side_effect=nominal,
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=guarded,
            ),
            patch.object(
                policy,
                "_predict_rag_prefill_layer_ms",
                side_effect=lambda stream_idx, _tokens: {
                    2: 0.5,
                    3: 1.0,
                    4: 2.0,
                }[stream_idx],
            ),
            patch.object(
                policy,
                "_rag_lease_transition_cost_estimate_ms",
                return_value=0.0,
            ),
        ):
            selected_idx = policy._select_rag_prefill_mixed_candidate(2, 4)

        diagnostics = policy.rag_prefill_partition_selection_last
        self.assertEqual(selected_idx, 3)
        by_stream = {row["stream_idx"]: row for row in diagnostics["candidates"]}
        self.assertFalse(by_stream[2]["safe"])
        self.assertTrue(by_stream[3]["safe"])
        self.assertEqual(by_stream[3]["predicted_decode_ms"], 12.0)
        self.assertEqual(by_stream[3]["guarded_decode_ms"], 30.0)

    def test_calibrated_decode_bounds_use_guard_once(self):
        policy = _PolicyHarness()
        policy.rag_legacy_calibrated_bundle = object()
        policy.rag_ann_decode_step_slo_ms = 35.0
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_max_decode_slowdown_ratio = 1.5

        def guarded(stream_idx, **kwargs):
            del stream_idx
            return 30.0 if kwargs.get("overlap_class") == "prefill" else 14.0

        def nominal(stream_idx, **kwargs):
            del stream_idx
            return 18.0 if kwargs.get("overlap_class") == "prefill" else 14.0

        with (
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                side_effect=guarded,
            ) as guarded_predictor,
            patch.object(
                policy,
                "_rag_prefill_nominal_decode_prediction_ms",
                side_effect=nominal,
            ) as nominal_predictor,
        ):
            self.assertTrue(policy._rag_prefill_decode_bounds_safe(2, 4))

        self.assertEqual(guarded_predictor.call_count, 1)
        self.assertEqual(nominal_predictor.call_count, 2)

    def test_prefill_decode_bounds_can_diagnose_without_causal_floor(self):
        policy = _PolicyHarness()
        policy.rag_legacy_calibrated_bundle = object()
        policy.rag_ann_decode_step_slo_ms = 35.0
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_max_decode_slowdown_ratio = 1.5

        def nominal(stream_idx, **kwargs):
            del stream_idx
            return 18.0 if kwargs.get("overlap_class") == "prefill" else 14.0

        with (
            patch.dict(os.environ, {"RAGLEASE_PREFILL_CAUSAL_GUARD": "0"}),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=40.0,
            ) as guarded_predictor,
            patch.object(
                policy,
                "_rag_prefill_nominal_decode_prediction_ms",
                side_effect=nominal,
            ) as nominal_predictor,
        ):
            self.assertTrue(policy._rag_prefill_decode_bounds_safe(2, 4))

        guarded_predictor.assert_not_called()
        self.assertEqual(nominal_predictor.call_count, 3)

    def test_exclusive_prefill_endpoint_uses_current_decode_timescale(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 28
        policy.rag_e2e_slo_ms = 100.0
        req = SimpleNamespace(rag_e2e_started_at=99.8)
        policy.split_prefill_batch = _SplitBatch(1, 4096, reqs=[req])

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=16.0,
            ) as prediction,
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
        ):
            count, layer_ms, slack_ms = policy._rag_prefill_lease_shape(0)

        self.assertAlmostEqual(layer_ms, 5.0)
        self.assertLess(slack_ms, 0.0)
        self.assertEqual(count, 4)
        prediction.assert_called_once_with(5, overlap_class="prefill")

    def test_decode_prediction_accuracy_requires_matching_candidate_partition(self):
        policy = _PolicyHarness()
        policy.rag_frontend_plan = SimpleNamespace(
            predicted_decode_ms=30.0,
            candidate_stream_idx=3,
            target_stream_idx=5,
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            policy._record_rag_decode_completion(True, 10.0)

        self.assertEqual(policy.rag_decode_prediction_samples, 0)
        self.assertEqual(policy.rag_decode_prediction_partition_mismatches, 1)

        policy.rag_frontend_plan = SimpleNamespace(
            predicted_decode_ms=12.0,
            candidate_stream_idx=5,
            target_stream_idx=5,
        )
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            policy._record_rag_decode_completion(True, 10.0)

        self.assertEqual(policy.rag_decode_prediction_samples, 1)
        self.assertEqual(policy.rag_decode_prediction_abs_error_ms[-1], 2.0)

    def test_round_trip_switch_cost_requires_strict_net_gain(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_lease_round_trip_switch_cost = True
        policy.rag_partition_switch_pair_latency_ms[(5, 4)] = deque(
            [5.0] * 4, maxlen=64
        )
        policy.rag_partition_switch_pair_latency_ms[(4, 5)] = deque(
            [5.0] * 4, maxlen=64
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_estimate_rag_lease_switch_benefit_ms",
                return_value=10.0,
            ),
        ):
            held_idx, held_reason = policy._coalesce_rag_mixed_partition_target(
                candidate_idx=4,
                target_reason="prefill_lease",
                allow_ann=False,
                allow_prefill=True,
                forced_ann=False,
                ann_wait_ms=0.0,
                prefill_wait_ms=0.0,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(held_idx, 5)
        self.assertEqual(held_reason, "prefill_lease_coalesced_cost")
        self.assertEqual(policy.rag_lease_last_transition_cost_ms, 10.0)

    def test_fast_path_bypasses_partition_guards_but_not_switch_cost(self):
        policy = _PolicyHarness()
        policy.rag_partition_min_residency_ms = 100000
        policy.rag_partition_last_switch_ts = 100.0
        policy.ann_search_pending = deque(
            [{"queries": SimpleNamespace(shape=(1, 768))}]
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.001,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            target_idx, _ = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                predicted_decode_ms=10.0,
                bypass_work_wait=True,
                bypass_partition_guards=True,
                fast_path_candidate_idx=4,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(policy.rag_partition_residency_holds, 0)

        policy.rag_lease_switch_cost_ms = 100
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                predicted_decode_ms=10.0,
                bypass_work_wait=True,
                bypass_partition_guards=True,
                fast_path_candidate_idx=4,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "ann_fast_path_coalesced_cost")

    def test_prefill_fast_path_reuses_service_safe_mixed_partition(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_lease_switch_cost_ms = 100.0

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch.object(
                policy,
                "_estimate_rag_lease_switch_benefit_ms",
                return_value=0.0,
            ),
        ):
            target_idx, reason = policy._coalesce_rag_mixed_partition_target(
                candidate_idx=2,
                target_reason="prefill_fast_path",
                allow_ann=False,
                allow_prefill=True,
                forced_ann=False,
                ann_wait_ms=0.0,
                prefill_wait_ms=0.0,
                predicted_decode_ms=10.0,
                bypass_work_wait=True,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "prefill_fast_path_coalesced_cost")
        self.assertEqual(
            policy.rag_prefill_fast_path_resident_cost_checks, 1
        )

    def test_switch_cost_coalesces_low_benefit_mixed_transition(self):
        policy = _PolicyHarness()
        policy.rag_lease_switch_cost_ms = 5
        policy.ann_search_pending = deque(
            [{"queries": SimpleNamespace(shape=(1, 768))}]
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(target_idx, 3)
        self.assertEqual(reason, "ann_lease_coalesced_cost_resident")
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 1)

    def test_dynamic_lease_reclaims_decode_sms_before_cost_gate(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_lease_switch_cost_ms = 100

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=2,
            ),
            patch.object(
                policy,
                "_estimate_rag_lease_switch_benefit_ms",
                return_value=-100.0,
            ),
        ):
            target_idx, reason = policy._coalesce_rag_mixed_partition_target(
                candidate_idx=4,
                target_reason="prefill_lease_decode_floor",
                allow_ann=False,
                allow_prefill=True,
                forced_ann=False,
                ann_wait_ms=0.0,
                prefill_wait_ms=0.0,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(
            reason, "prefill_lease_decode_floor_decode_restore"
        )
        self.assertEqual(policy.rag_lease_decode_restore_transitions, 1)
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)

    def test_dynamic_prefill_deadline_bypasses_switch_cost_gate(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_lease_switch_cost_ms = 100

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_estimate_rag_lease_switch_benefit_ms",
                return_value=-100.0,
            ),
        ):
            target_idx, reason = policy._coalesce_rag_mixed_partition_target(
                candidate_idx=4,
                target_reason="prefill_lease_deadline_step",
                allow_ann=False,
                allow_prefill=True,
                forced_ann=False,
                ann_wait_ms=0.0,
                prefill_wait_ms=policy.rag_prefill_deadline_ms,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "prefill_lease_deadline_step")
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)

    def test_dynamic_max_wait_bypasses_switch_cost_gate(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.rag_lease_switch_cost_ms = 100
        policy.rag_lease_coalesce_max_wait_ms = 50

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_estimate_rag_lease_switch_benefit_ms",
                return_value=-100.0,
            ),
        ):
            target_idx, reason = policy._coalesce_rag_mixed_partition_target(
                candidate_idx=4,
                target_reason="prefill_lease",
                allow_ann=False,
                allow_prefill=True,
                forced_ann=False,
                ann_wait_ms=0.0,
                prefill_wait_ms=policy.rag_lease_coalesce_max_wait_ms,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "prefill_lease")
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)

    def test_switch_cost_allows_high_benefit_transition(self):
        policy = _PolicyHarness()
        policy.rag_lease_switch_cost_ms = 5
        policy.ann_search_pending = deque(
            [{"queries": SimpleNamespace(shape=(1, 768))}]
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                predicted_decode_ms=50.0,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "ann_lease")

    def test_deadline_bypasses_switch_cost_gate(self):
        policy = _PolicyHarness()
        policy.rag_lease_switch_cost_ms = 100
        policy.ann_search_pending = deque(
            [{"queries": SimpleNamespace(shape=(1, 768))}]
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=True,
                allow_prefill=False,
                active_ann=False,
                ann_wait_ms=50.0,
                predicted_decode_ms=10.0,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "ann_lease")

    def test_prefill_deadline_bypasses_fast_path_switch_cost_gate(self):
        policy = _PolicyHarness()
        policy.rag_lease_switch_cost_ms = 100

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
                prefill_wait_ms=500.0,
                predicted_decode_ms=10.0,
                bypass_work_wait=True,
                bypass_partition_guards=True,
                fast_path_candidate_idx=4,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "prefill_fast_path")

    def test_prefill_fast_path_keeps_candidate_aware_partition_selection(self):
        policy = _PolicyHarness()
        policy.rag_prefill_dynamic_lease = True
        policy.split_prefill_batch = _SplitBatch(1, 1024)

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
            patch.object(
                policy,
                "_select_rag_prefill_mixed_candidate",
                return_value=4,
            ) as select_candidate,
        ):
            target_idx, reason = policy._select_frontend_plan_target(
                frontend_needed=True,
                allow_ann=False,
                allow_prefill=True,
                active_ann=False,
                predicted_decode_ms=10.0,
                bypass_work_wait=True,
                bypass_partition_guards=True,
                fast_path_candidate_idx=2,
            )

        self.assertEqual(target_idx, 4)
        self.assertEqual(reason, "prefill_fast_path")
        select_candidate.assert_called_once_with(2, 4)

    def test_prefill_bundle_completion_records_layer_cost(self):
        class _StartEvent:
            def elapsed_time(self, _):
                return 12.0

        class _EndEvent:
            def query(self):
                return True

        policy = _PolicyHarness()
        end_event = _EndEvent()
        policy.rag_prefill_bundle_inflight_event = end_event
        policy.rag_prefill_bundle_inflight_meta = {
            "start_event": _StartEvent(),
            "layers": 3,
            "tokens": 4096,
            "stream_idx": 2,
            "predicted_bundle_ms": 10.0,
        }

        self.assertTrue(policy._poll_rag_prefill_bundle_completion())
        self.assertIsNone(policy.rag_prefill_bundle_inflight_event)
        self.assertEqual(policy.rag_prefill_bundle_completions, 1)
        self.assertEqual(policy.rag_prefill_bundle_layer_histogram, {"3": 1})
        self.assertEqual(
            list(
                policy.rag_prefill_layer_ms_by_partition_bucket[
                    (2, 4096)
                ]
            ),
            [4.0],
        )
        self.assertEqual(policy.rag_prefill_prediction_samples, 1)
        self.assertEqual(policy.rag_prefill_prediction_abs_error_ms[-1], 2.0)
        self.assertEqual(
            policy.rag_prefill_prediction_pairs[-1],
            {
                "predicted_ms": 10.0,
                "observed_ms": 12.0,
                "stream_idx": 2,
                "layers": 3,
                "tokens": 4096,
            },
        )

    def test_prefill_waits_instead_of_running_on_decode_only_endpoint(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=False
        )

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=10.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_select_frontend_plan_target",
                return_value=(
                    5,
                    "prefill_lease_coalesced_cost_resident",
                ),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertFalse(plan.allow_prefill)
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertEqual(plan.reason, "prefill_decode_only_deferred")
        self.assertEqual(policy.rag_prefill_decode_only_deferrals, 1)

    def test_overdue_prefill_borrows_safest_mixed_partition(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=False
        )

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy, "_oldest_prefill_wait_ms", return_value=500.0
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_select_frontend_plan_target",
                return_value=(
                    5,
                    "prefill_lease_coalesced_cost_resident",
                ),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_prefill)
        self.assertEqual(plan.target_stream_idx, 4)
        self.assertEqual(
            plan.target_reason, "prefill_decode_only_mixed_fallback"
        )
        self.assertEqual(
            policy.rag_prefill_decode_only_mixed_fallbacks, 1
        )

    def test_partition_switch_waits_for_prefill_bundle_event(self):
        policy = _PolicyHarness()
        policy.rag_prefill_bundle_inflight_event = object()

        blocked = policy._rag_ann_blocks_stream_group_switch(4)

        self.assertTrue(blocked)
        self.assertTrue(policy.rag_ann_partition_switch_deferred)
        self.assertEqual(policy.rag_prefill_bundle_switch_holds, 1)

    def test_decode_budget_separates_deadline_from_forced_release(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "frontend-gc"
        policy.rag_ann_decode_step_slo_ms = 20
        policy.rag_ann_itl_slo_ms = -1
        policy.rag_ann_safety_margin_ms = 5
        policy.rag_ann_max_defer_ms = 150
        policy.rag_ann_decode_only_wait_ms = 10
        policy.rag_ann_decode_iteration_ms = deque([16.0] * 8, maxlen=64)
        policy.rag_ann_decode_queue_delay_ms = deque(maxlen=64)

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            rejected = policy._evaluate_rag_ann_decode_budget(50)
            forced = policy._evaluate_rag_ann_decode_budget(150)

        self.assertFalse(rejected["allowed"])
        self.assertEqual(rejected["reason"], "decode_budget")
        self.assertTrue(forced["allowed"])
        self.assertTrue(forced["forced"])


    def test_decode_budget_relaxes_gradually_with_bounded_urgency(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "frontend-gc"
        policy.rag_ann_decode_step_slo_ms = 20
        policy.rag_ann_safety_margin_ms = 0.5
        policy.rag_ann_max_defer_ms = 50
        policy.rag_ann_urgency_start_ratio = 0.2
        policy.rag_ann_max_decode_slowdown_ratio = 1.25

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
            patch.object(
                policy, "_rag_ann_decode_prediction_ms", return_value=22.0
            ),
        ):
            strict = policy._evaluate_rag_ann_decode_budget(10)
            urgent = policy._evaluate_rag_ann_decode_budget(32)

        with patch.object(
            policy, "_rag_ann_decode_prediction_ms", return_value=26.0
        ), patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            capped = policy._evaluate_rag_ann_decode_budget(49)
            forced = policy._evaluate_rag_ann_decode_budget(50)

        self.assertFalse(strict["allowed"])
        self.assertTrue(urgent["allowed"])
        self.assertEqual(urgent["reason"], "decode_urgency")
        self.assertGreater(urgent["admission_limit_ms"], 22.0)
        self.assertFalse(capped["allowed"])
        self.assertTrue(forced["forced"])
    def test_decode_urgency_reaches_planner_before_max_defer(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )
        policy.rag_ann_decode_batch_sizes = deque([32] * 8, maxlen=64)

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=32.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_urgency",
                    "predicted_ms": 22.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_ann)
        self.assertFalse(plan.allow_prefill)
        self.assertEqual(plan.reason, "ann_urgency")
        self.assertTrue(plan.launch_ann_early)
        self.assertEqual(plan.budget_reason, "decode_urgency")
        self.assertEqual(policy.rag_ann_urgency_admitted_requests, 1)
        self.assertEqual(list(policy.rag_ann_urgency_admission_wait_ms), [32.0])

    def test_decode_safe_ann_runs_at_deadline_before_older_prefill(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_schedule = "frontend-gc"
        policy.ann_search_pending = deque([{"enqueue_time": 0.0}])
        policy.split_prefill_batch = _Batch(1)
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = False
        policy.rag_ann_consecutive_batches = 0
        policy.rag_prefill_consecutive_chunks = 0
        policy.rag_ann_max_consecutive_batches = 1
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_ann_deadline_ms = 50
        policy.rag_prefill_deadline_ms = 500
        policy.rag_ann_overlap_pressure_threshold = 0.5
        policy.rag_ann_enable_controlled_overlap = False
        policy.rag_ann_frontend_residency_until = 0.0
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}
        policy.rag_ann_budget_reject_events = 0
        policy.rag_ann_budget_rejected_requests = 0
        policy.rag_ann_slo_deferred_requests = 0

        with (
            patch.object(policy, "_prune_rag_ann_handoffs"),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=50.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=750.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 15.0,
                    "slo_ms": 22.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=4,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertTrue(policy.rag_frontend_plan.allow_ann)
        self.assertFalse(policy.rag_frontend_plan.allow_prefill)
        self.assertEqual(policy.rag_frontend_plan.reason, "ann_deadline")

    def test_primary_ann_does_not_require_frontend_partition(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "primary"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_search_pending = deque(
            [{"enqueue_time": 0.0, "queries": [[0.0]]}]
        )
        policy.rag_ann_deadline_ms = 50
        policy.rag_prefill_deadline_ms = 500
        policy.rag_ann_safety_margin_ms = 5
        policy.rag_ann_max_consecutive_batches = 1
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_ann_overlap_pressure_threshold = 0.5
        policy.rag_ann_enable_controlled_overlap = False
        policy.rag_ann_frontend_residency_until = 0.0
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}
        policy.rag_ann_decode_iteration_ms = deque(maxlen=64)
        policy.rag_ann_decode_queue_delay_ms = deque(maxlen=64)
        policy.rag_ann_slo_deferred_requests = 0
        policy.rag_ann_budget_reject_events = 0
        policy.rag_ann_budget_rejected_requests = 0
        policy.split_prefill_batch = None
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = False
        policy.rag_ann_consecutive_batches = 0
        policy.rag_prefill_consecutive_chunks = 0

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            policy._plan_rag_frontend_actions()

        self.assertTrue(policy.rag_frontend_plan.allow_ann)
        self.assertFalse(policy.rag_frontend_plan.frontend_needed)
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, 5)

    def test_decode_gc_selects_decode_lane_including_decode_only(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.stream_groups = [
            ("frontend-0", "decode-0"),
            ("frontend-1", "decode-1"),
            ("frontend-2", "decode-2"),
            ("frontend-3", "decode-3"),
            ("frontend-4", "decode-4"),
            ("frontend-5", "decode-5"),
        ]

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            self.assertEqual(policy._get_rag_ann_stream(), "decode-3")

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            self.assertEqual(policy._get_rag_ann_stream(), "decode-5")

        policy.rag_ann_schedule = "frontend-gc"
        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=3,
        ):
            self.assertEqual(policy._get_rag_ann_stream(), "frontend-3")

    def test_decode_gc_ann_does_not_require_frontend_partition(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=False),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_rag_stage_candidate_stream_idx",
                side_effect=AssertionError("resident ANN evaluated a candidate"),
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                side_effect=AssertionError("resident ANN sampled load state"),
            ),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                side_effect=AssertionError("resident ANN evaluated a budget"),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        self.assertTrue(policy.rag_frontend_plan.allow_ann)
        self.assertFalse(policy.rag_frontend_plan.frontend_needed)
        self.assertEqual(policy.rag_frontend_plan.target_stream_idx, 5)
        self.assertEqual(policy.rag_frontend_plan.candidate_stream_idx, 5)
        self.assertEqual(
            policy.rag_frontend_plan.reason, "ann_decode_resident_fast_path"
        )
        self.assertEqual(policy.rag_frontend_ann_decode_fast_path_hits, 1)

    def test_decode_gc_work_conserving_overlap_bypasses_budget_and_coalescing(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=False),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=5.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=5.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_low_pressure_fast_path_allowed",
                return_value=False,
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch",
                    "batch_p95": 48.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_should_wait_for_rag_lease_cohort",
                return_value=True,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_ann)
        self.assertFalse(plan.allow_prefill)
        self.assertTrue(plan.launch_ann_early)
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertEqual(plan.reason, "ann_decode_overlap_prefill_coalesce")
        self.assertEqual(plan.budget_reason, "decode_overlap")
        self.assertEqual(
            policy.rag_ann_last_budget["guarded_reason"], "decode_budget"
        )
        self.assertEqual(policy.rag_ann_budget_reject_events, 0)

    def test_decode_gc_work_conserving_overlap_uses_mixed_group_for_prefill(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=True, handoff=False
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_lease_coalesce_min_work = 1
        policy.rag_lease_coalesce_max_wait_ms = 0
        policy.rag_lease_switch_cost_ms = 0

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=False),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 30.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_low_pressure_fast_path_allowed",
                return_value=True,
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch",
                    "batch_p95": 48.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 10.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_ann)
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.controlled_overlap)
        self.assertTrue(plan.launch_ann_early)
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertIn(plan.target_stream_idx, range(1, 5))

    def test_decode_gc_low_pressure_handoff_overlaps_ann_in_decode_gc(
        self,
    ):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_max_decode_slowdown_ratio = 1.5

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_rag_low_pressure_fast_path_allowed",
                return_value=False,
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "candidate",
                    "batch_p95": 8.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 24.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.allow_ann)
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.controlled_overlap)
        self.assertTrue(plan.launch_ann_early)
        self.assertTrue(plan.immediate_handoff)
        self.assertEqual(
            plan.reason, "ann_decode_overlap_handoff_fast_path"
        )
        self.assertEqual(plan.target_reason, "prefill_fast_path")
        self.assertEqual(plan.target_stream_idx, 3)
        self.assertTrue(plan.system_load_low)
        self.assertEqual(plan.system_load_reason, "prefill_relaxed")

    def test_decode_gc_tiny_batch_handoff_bypasses_candidate_budget(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "candidate",
                    "batch_p95": 4.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.immediate_handoff)
        self.assertEqual(plan.target_stream_idx, 3)
        self.assertEqual(plan.system_load_reason, "prefill_tiny_batch")

    def test_decode_gc_low_arrival_handoff_ignores_transient_batch_pressure(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_prefill_low_load_arrival_rps_threshold = 3.0

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch+candidate",
                    "batch_p95": 12.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 2.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.immediate_handoff)
        self.assertEqual(plan.target_stream_idx, 3)
        self.assertEqual(plan.system_load_reason, "prefill_arrival")
        self.assertEqual(plan.request_arrival_rps, 2.0)

    def test_decode_gc_low_arrival_is_normalized_by_prefill_work(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_prefill_low_load_arrival_rps_threshold = 3.0
        policy.split_prefill_batch = _SplitBatch(
            1,
            4095,
            reqs=[],
        )
        policy.rag_prefill_handoff_active = True

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "batch+candidate",
                    "batch_p95": 12.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 2.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertAlmostEqual(plan.prefill_work_equivalent_rps, 7.998046875)
        self.assertEqual(policy.rag_prefill_workload_fast_path_rejects, 1)

    def test_decode_gc_long_prefill_uses_fast_path_when_candidate_is_safe(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_prefill_low_load_arrival_rps_threshold = 3.0
        policy.rag_ann_max_decode_slowdown_ratio = 1.5
        policy.split_prefill_batch = _SplitBatch(1, 4095, reqs=[])

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "predicted_safe",
                    "predicted_ms": 18.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "arrival_work",
                    "batch_p95": 12.0,
                    "queue_p95_ms": 8.0,
                    "predicted_ms": 18.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 2.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertAlmostEqual(plan.prefill_work_equivalent_rps, 7.998046875)
        self.assertEqual(
            getattr(policy, "rag_prefill_workload_fast_path_rejects", 0), 0
        )

    def test_decode_gc_tiny_low_arrival_handoff_uses_bounded_exclusive_sprint(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_launch_site = "early"
        policy.rag_prefill_dynamic_lease = True
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_prefill_exclusive_sprint_available = True
        policy.rag_prefill_lease_min_layers = 4
        policy.rag_prefill_lease_max_layers = 8
        policy.rag_prefill_low_load_arrival_rps_threshold = 3.0
        policy.rag_prefill_max_consecutive_chunks = 1
        policy.running_batch = _Batch(4)
        policy.split_prefill_batch.extend_num_tokens = 1023
        policy.split_prefill_batch.split_index = 0
        policy.model_config = SimpleNamespace(num_hidden_layers=28)

        with (
            patch.object(policy, "_prune_rag_ann_handoffs", return_value=True),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                },
            ),
            patch.object(
                policy,
                "_rag_system_load_state",
                return_value={
                    "low": False,
                    "reason": "candidate",
                    "batch_p95": 4.0,
                    "queue_p95_ms": 0.0,
                    "predicted_ms": 80.0,
                    "slo_ms": 20.0,
                    "arrival_rps": 2.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.prefill_sprint)
        self.assertTrue(plan.prefill_exclusive_sprint)
        self.assertEqual(plan.prefill_forward_count, 8)
        self.assertEqual(plan.target_stream_idx, 0)

        policy.rag_frontend_prefill_ran = True
        policy._finalize_rag_frontend_actions()
        self.assertFalse(policy.rag_prefill_exclusive_sprint_available)
        self.assertEqual(policy.rag_prefill_exclusive_sprint_chunks, 1)

    def test_exclusive_prefill_lease_consumes_one_handoff_bundle(self):
        policy = _PolicyHarness()
        policy.rag_prefill_exclusive_sprint = True
        policy.rag_prefill_exclusive_sprint_available = True
        policy.rag_prefill_max_consecutive_chunks = 2
        policy.rag_prefill_consecutive_chunks = 0
        policy.split_prefill_batch = _SplitBatch(
            1, 1023, split_index=8
        )
        policy.rag_frontend_prefill_ran = True
        policy.rag_frontend_ann_launched = False
        policy.rag_frontend_plan = SimpleNamespace(
            prefill_sprint=True,
            prefill_exclusive_sprint=True,
            dependency_prefill_due=False,
        )

        policy._finalize_rag_frontend_actions()

        self.assertFalse(policy.rag_prefill_exclusive_sprint_available)
        self.assertEqual(policy.rag_prefill_consecutive_chunks, 1)
        self.assertEqual(policy.rag_prefill_exclusive_sprint_chunks, 1)

    def test_nonexclusive_prefill_sprint_preserves_planned_bundle(self):
        policy = _PolicyHarness()
        policy.model_config.num_hidden_layers = 28
        policy.split_prefill_batch = _SplitBatch(
            1, 1023, split_index=8
        )
        plan = SimpleNamespace(
            prefill_sprint=True,
            prefill_exclusive_sprint=False,
            prefill_forward_count=8,
        )

        self.assertEqual(policy._rag_prefill_bundle_forward_count(plan), 8)

    def test_prefill_bundle_fallback_keeps_token_budget(self):
        policy = _PolicyHarness()
        policy.model_config.num_hidden_layers = 28
        policy.pdmux_config.split_forward_token_budget = 8192
        policy.split_prefill_batch = _SplitBatch(
            1, 2048, split_index=8
        )

        self.assertEqual(policy._rag_prefill_bundle_forward_count(None), 4)

    def test_decode_gc_defers_partition_switch_while_ann_is_active(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.ann_search_inflight = [
            {"result": object(), "stream_idx": 5, "predicted_gpu_ms": 8.0}
        ]

        with patch.object(
            policy, "_ann_result_gpu_pending", return_value=True
        ):
            self.assertTrue(policy._rag_ann_blocks_stream_group_switch(2))
            self.assertFalse(policy._rag_ann_blocks_stream_group_switch(5))
        self.assertEqual(policy.rag_ann_partition_drain_events, 1)
        self.assertEqual(list(policy.rag_ann_partition_drain_predicted_ms), [8.0])

    def test_idle_single_ann_query_bypasses_microbatch_wait(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_batch_size = 3
        policy.rag_ann_max_microbatch_gpu_ms = 0
        policy.rag_ann_max_batch_wait_ms = 10
        policy.rag_ann_batch_wait_min_pending = 2
        pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )

        self.assertFalse(
            policy._should_wait_for_ann_microbatch(pending, 1.0, inflight=[])
        )

    def test_single_ann_query_coalesces_behind_active_gpu_work(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_batch_size = 3
        policy.rag_ann_max_microbatch_gpu_ms = 0
        policy.rag_ann_max_batch_wait_ms = 10
        policy.rag_ann_batch_wait_min_pending = 2
        policy.rag_frontend_plan = SimpleNamespace(
            low_pressure_fast_path=True,
            allow_ann=True,
        )
        pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        inflight = [{"result": object()}]

        with patch.object(
            policy, "_ann_result_gpu_pending", return_value=True
        ):
            self.assertTrue(
                policy._should_wait_for_ann_microbatch(
                    pending, 1.0, inflight=inflight
                )
            )
            self.assertFalse(
                policy._should_wait_for_ann_microbatch(
                    pending, 10.0, inflight=inflight
                )
            )

        self.assertTrue(pending[0]["inflight_coalesce_recorded"])
        self.assertEqual(policy.rag_ann_inflight_coalesce_events, 1)
        self.assertEqual(policy.rag_ann_inflight_coalesce_holds, 1)

    def test_completed_ann_does_not_trigger_opportunistic_wait(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_batch_size = 3
        policy.rag_ann_max_microbatch_gpu_ms = 0
        policy.rag_ann_max_batch_wait_ms = 10
        policy.rag_ann_batch_wait_min_pending = 2
        pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        inflight = [{"result": object()}]

        with patch.object(
            policy, "_ann_result_gpu_pending", return_value=False
        ):
            self.assertFalse(
                policy._should_wait_for_ann_microbatch(
                    pending, 1.0, inflight=inflight
                )
            )

    def test_deadline_risk_launches_one_batch_in_current_decode_epoch(self):
        class _PendingResult:
            @staticmethod
            def gpu_poll():
                return False

            @staticmethod
            def poll():
                return False

        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_max_batch_wait_ms = 1000
        policy.rag_ann_batch_wait_min_pending = 2
        policy.rag_ann_max_microbatch_gpu_ms = 80
        policy.rag_ann_max_queued_gpu_ms = 15
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=True,
            reason="ann_decode_overlap",
            immediate_handoff=True,
            forced_ann=False,
            low_pressure_fast_path=False,
            target_stream_idx=2,
            ann_wait_ms=5.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 99.999,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": "deadline-risk",
                },
                {
                    "enqueue_time": 99.999,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": "remains-pending",
                },
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [80.0] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy, service_ms=20.0, tail_ms=5.0
        )
        policy.ann_executor = SimpleNamespace(
            search_async=lambda *args, **kwargs: _PendingResult()
        )
        ann_stream = SimpleNamespace(cuda_stream=123)

        with (
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                return_value=100.0,
            ),
            patch.object(policy, "_get_rag_ann_stream", return_value=ann_stream),
            patch.object(
                policy,
                "_get_stream_group_idx_for_stream",
                return_value=5,
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=30.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertTrue(
                policy._should_wait_for_ann_microbatch(
                    policy.ann_search_pending, 1.0
                )
            )
            launched = policy._maybe_launch_ann_before_partition_switch(2)
            switch_blocked = policy._rag_ann_blocks_stream_group_switch(2)

        self.assertTrue(launched)
        self.assertTrue(switch_blocked)
        self.assertEqual(len(policy.ann_search_inflight), 1)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(policy.rag_ann_pre_switch_launch_batches, 1)
        self.assertEqual(policy.rag_ann_pre_switch_launch_rows, 1)
        self.assertEqual(policy.rag_ann_pre_switch_safety_violations, 0)
        self.assertLessEqual(
            policy.rag_ann_pre_switch_peak_projected_gpu_ms,
            policy.rag_ann_max_queued_gpu_ms,
        )
        self.assertEqual(
            policy.ann_search_inflight[0]["launch_context"], "pre_switch"
        )
        self.assertEqual(policy.ann_search_inflight[0]["stream_idx"], 5)

    def test_pre_switch_launch_skips_when_switch_fits_ann_slack(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=False,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=0.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [5.0] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy, service_ms=10.0, tail_ms=2.0
        )

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"deadline_safe": 1},
        )

    def test_pre_switch_launch_fails_closed_without_decode_prediction(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=False,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=40.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [80.0] * 4, maxlen=64
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 4, maxlen=64)
        }

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=None,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"decode_prediction_unavailable": 1},
        )

    def test_pre_switch_launch_fails_closed_without_completion_prediction(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=False,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=40.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [80.0] * 4, maxlen=64
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 4, maxlen=64)
        }

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"completion_prediction_unavailable": 1},
        )

    def test_pre_switch_ann_risk_uses_completion_service_not_gpu_only(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=False,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=0.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [20.0] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy,
            service_ms=[40.0, 40.0, 40.0, 100.0],
            tail_ms=[30.0, 30.0, 30.0, 90.0],
            gpu_ms=10.0,
            inline_ms=1.0,
        )

        def record_launch(**_kwargs):
            policy.rag_ann_pre_switch_launch_batches += 1

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch.object(
                policy,
                "_get_rag_ann_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
            patch.object(
                policy,
                "_launch_ann_search_side_channel",
                side_effect=record_launch,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        # GPU-only accounting would be 0 + 20 + 10 + 5 = 35 ms (safe).
        # Robust completion accounting is 0 + 20 + 40 + 5 = 65 ms.
        self.assertTrue(launched)
        self.assertEqual(policy.rag_ann_pre_switch_deadline_risk_triggers, 1)
        self.assertEqual(policy.rag_ann_pre_switch_switch_induced_triggers, 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_trigger_type_counts,
            {"switch_induced": 1},
        )

    def test_pre_switch_counterfactual_intrinsic_gain_and_no_gain(self):
        no_gain = self._counterfactual_pre_switch_policy(
            switch_ms=0.02,
            service_ms=96.0,
            tail_ms=89.0,
            gpu_ms=7.0,
            inline_ms=89.0,
        )
        self.assertFalse(self._run_counterfactual_pre_switch(no_gain))
        self.assertEqual(
            no_gain.rag_ann_pre_switch_no_launch_reason_counts,
            {"counterfactual_no_gain": 1},
        )
        self.assertEqual(
            no_gain.rag_ann_pre_switch_counterfactual_no_gain_skips, 1
        )
        self.assertLess(
            no_gain.rag_ann_pre_switch_last_counterfactual[
                "predicted_gain_ms"
            ],
            0.0,
        )

        recovery = self._counterfactual_pre_switch_policy(
            switch_ms=25.0,
            service_ms=20.0,
            tail_ms=13.0,
            gpu_ms=7.0,
            inline_ms=1.0,
            ann_wait_ms=75.0,
        )
        self.assertTrue(self._run_counterfactual_pre_switch(recovery))
        self.assertEqual(
            recovery.rag_ann_pre_switch_intrinsic_debt_recovery_triggers,
            1,
        )
        self.assertEqual(
            recovery.rag_ann_pre_switch_trigger_type_counts,
            {"intrinsic_debt_recovery": 1},
        )
        self.assertEqual(
            recovery.rag_ann_pre_switch_last_counterfactual[
                "prelaunch_eta_ms"
            ],
            8.0,
        )
        self.assertGreater(
            recovery.rag_ann_pre_switch_last_counterfactual[
                "predicted_gain_ms"
            ],
            0.0,
        )
        recovery.rag_frontend_plan = None
        telemetry = recovery._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertEqual(telemetry["intrinsic_debt_recovery_triggers"], 1)
        self.assertEqual(
            telemetry["trigger_type_counts"],
            {"intrinsic_debt_recovery": 1},
        )
        self.assertEqual(telemetry["predicted_gain"]["samples"], 1)
        self.assertEqual(
            telemetry["last_counterfactual"]["trigger_type"],
            "intrinsic_debt_recovery",
        )
        self.assertEqual(telemetry["decision_quantile"], 50)
        self.assertEqual(telemetry["classification_invariant_violations"], 0)

    def test_pre_switch_robust_gain_equality_is_rejected(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=5.0,
            service_ms=20.0,
            tail_ms=13.0,
            gpu_ms=7.0,
            inline_ms=13.0,
            ann_wait_ms=70.0,
        )

        self.assertFalse(
            self._run_counterfactual_pre_switch(policy, decode_ms=7.0)
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"counterfactual_no_gain": 1},
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["robust_gain_ms"], 0.0)
        self.assertEqual(counterfactual["decision_quantile"], 50)

    def test_pre_switch_bimodal_tail_risk_does_not_launch_on_cheap_switch(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=0.02,
            service_ms=[20.0, 20.0, 20.0, 120.0],
            tail_ms=[13.0, 13.0, 13.0, 113.0],
            gpu_ms=7.0,
            inline_ms=1.0,
        )

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"deadline_risk_tail_only": 1},
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["tail_deadline_risk"])
        self.assertFalse(counterfactual["robust_deadline_risk"])
        self.assertEqual(counterfactual["decision_quantile"], 50)
        self.assertEqual(
            policy.rag_ann_pre_switch_classification_invariant_violations, 0
        )

    def test_pre_switch_gpu_critical_path_uses_exact_batch_p99(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=40.0,
            service_ms=60.0,
            tail_ms=30.0,
            gpu_ms=10.0,
            inline_ms=1.0,
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 61 + [30.0] * 3, maxlen=64)
        }

        self.assertTrue(self._run_counterfactual_pre_switch(policy))
        self.assertAlmostEqual(policy.rag_ann_credit_last_batch_ms, 10.0)
        self.assertAlmostEqual(
            policy.rag_ann_pre_switch_last_counterfactual[
                "predicted_gpu_critical_ms"
            ],
            30.0,
        )
        self.assertAlmostEqual(
            policy.rag_ann_pre_switch_last_counterfactual[
                "prelaunch_eta_ms"
            ],
            31.0,
        )

    def test_pre_switch_gpu_critical_path_fails_closed_without_samples(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=40.0,
            service_ms=60.0,
            tail_ms=30.0,
            gpu_ms=10.0,
            inline_ms=1.0,
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 3, maxlen=64)
        }

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"gpu_prediction_unavailable": 1},
        )

    def test_pre_switch_gpu_p99_must_fit_non_atomic_credit(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=40.0,
            service_ms=60.0,
            tail_ms=30.0,
            gpu_ms=10.0,
            inline_ms=1.0,
        )
        policy.rag_ann_max_queued_gpu_ms = 20.0
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([10.0] * 61 + [30.0] * 3, maxlen=64)
        }

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"gpu_p99_credit": 1},
        )
        self.assertEqual(policy.rag_ann_pre_switch_gpu_p99_credit_holds, 1)
        self.assertAlmostEqual(
            policy.rag_ann_pre_switch_last_gpu_p99_credit["batch_p99_ms"],
            30.0,
        )
        self.assertAlmostEqual(
            policy.rag_ann_pre_switch_last_gpu_p99_credit["limit_ms"],
            20.0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 0
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_rows_reduced, 0
        )
        self.assertIsNone(
            policy.rag_ann_pre_switch_last_gpu_p99_credit[
                "selected_batch_rows"
            ]
        )
        self.assertFalse(
            policy.rag_ann_pre_switch_last_gpu_p99_credit["fits"]
        )

    def test_pre_switch_gpu_p99_credit_downsizes_to_largest_safe_prefix(self):
        policy = self._p99_downsize_pre_switch_policy()
        launches = []

        def record_launch(**kwargs):
            launches.append(kwargs)
            policy.rag_ann_pre_switch_launch_batches += 1

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=7.0,
            ),
            patch.object(
                policy,
                "_get_rag_ann_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
            patch.object(
                policy,
                "_launch_ann_search_side_channel",
                side_effect=record_launch,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertTrue(
                policy._maybe_launch_ann_before_partition_switch(2)
            )

        self.assertEqual(len(launches), 1)
        self.assertEqual(
            launches[0]["first_batch_selection"],
            (1, 15.0),
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_rows_reduced, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_launches,
            1,
        )
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertTrue(credit["downsize_enabled"])
        self.assertTrue(credit["fits"])
        self.assertTrue(credit["downsized"])
        self.assertEqual(credit["request_id"], "request-0")
        self.assertEqual(credit["commit_stage"], "launch")
        self.assertIsNone(credit["final_skip_reason"])
        self.assertEqual(credit["initial_batch_rows"], 2)
        self.assertEqual(credit["selected_batch_rows"], 1)
        self.assertEqual(credit["rows_reduced"], 1)
        self.assertEqual(credit["attempted_batch_rows"], [2, 1])
        self.assertAlmostEqual(credit["initial_batch_p99_ms"], 30.0)
        self.assertAlmostEqual(credit["selected_batch_p99_ms"], 15.0)
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["prelaunch_batch_size"], 1)
        self.assertEqual(counterfactual["no_launch_batch_size"], 2)
        self.assertAlmostEqual(
            counterfactual["predicted_gpu_critical_ms"], 15.0
        )
        self.assertAlmostEqual(
            counterfactual["no_launch_predicted_gpu_critical_ms"],
            30.0,
        )
        policy.rag_frontend_plan = None
        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertTrue(telemetry["gpu_p99_credit_downsize_enabled"])
        self.assertEqual(telemetry["gpu_p99_credit_downsizes"], 1)
        self.assertEqual(telemetry["gpu_p99_credit_rows_reduced"], 1)
        self.assertEqual(
            telemetry["gpu_p99_credit_downsized_triggers"], 1
        )
        self.assertEqual(
            telemetry["gpu_p99_credit_downsized_launches"], 1
        )
        self.assertGreaterEqual(
            telemetry["gpu_p99_credit_downsizes"],
            telemetry["gpu_p99_credit_downsized_triggers"],
        )
        self.assertGreaterEqual(
            telemetry["gpu_p99_credit_downsized_triggers"],
            telemetry["gpu_p99_credit_downsized_launches"],
        )

    def test_pre_switch_gpu_p99_downsize_trigger_can_fail_to_launch(self):
        policy = self._p99_downsize_pre_switch_policy()

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=7.0,
            ),
            patch.object(
                policy,
                "_get_rag_ann_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
            patch.object(
                policy,
                "_launch_ann_search_side_channel",
                return_value=None,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertFalse(
                policy._maybe_launch_ann_before_partition_switch(2)
            )

        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_launches,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"launch_failed_or_cpu_fallback": 1},
        )
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertEqual(credit["request_id"], "request-0")
        self.assertEqual(credit["commit_stage"], "trigger")
        self.assertEqual(
            credit["final_skip_reason"],
            "launch_failed_or_cpu_fallback",
        )

    def test_pre_switch_gpu_p99_downsize_uses_original_no_launch_batch(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_completion_service_ms_by_path_batch[
            "full_gpu"
        ][1] = deque([20.0] * 4, maxlen=64)
        policy.rag_ann_post_gpu_tail_ms_by_path_batch["full_gpu"][1] = (
            deque([5.0] * 4, maxlen=64)
        )

        self.assertTrue(self._run_counterfactual_pre_switch(policy))
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["prelaunch_batch_size"], 1)
        self.assertEqual(counterfactual["no_launch_batch_size"], 2)
        self.assertEqual(
            counterfactual["predicted_completion_service_p50_ms"],
            60.0,
        )
        self.assertEqual(counterfactual["predicted_gain_ms"], 79.0)

    def test_pre_switch_gpu_p99_downsize_requires_selected_bucket(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_completion_service_ms_by_path_batch = {
            "full_gpu": {2: deque([60.0] * 4, maxlen=64)}
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            "full_gpu": {2: deque([30.0] * 4, maxlen=64)}
        }

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"completion_prediction_unavailable": 1},
        )
        self.assertEqual(len(policy.ann_search_pending), 2)
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_launches,
            0,
        )
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertEqual(credit["request_id"], "request-0")
        self.assertEqual(credit["commit_stage"], "selector")
        self.assertEqual(
            credit["final_skip_reason"],
            "completion_prediction_unavailable",
        )
        self.assertEqual(
            credit["selected_batch_rows"],
            1,
        )

    def test_pre_switch_gpu_p99_downsize_requires_no_launch_bucket(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_completion_service_ms_by_path_batch = {
            "full_gpu": {1: deque([60.0] * 4, maxlen=64)}
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            "full_gpu": {1: deque([30.0] * 4, maxlen=64)}
        }

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"completion_prediction_unavailable": 1},
        )
        self.assertEqual(len(policy.ann_search_pending), 2)
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_last_gpu_p99_credit[
                "selected_batch_rows"
            ],
            1,
        )

    def test_pre_switch_gpu_p99_downsize_env_off_preserves_batch_reject(self):
        policy = self._p99_downsize_pre_switch_policy()

        with patch.dict(
            os.environ,
            {"SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "0"},
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"gpu_p99_credit": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_holds, 1
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 0
        )
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertFalse(credit["downsize_enabled"])
        self.assertFalse(credit["fits"])
        self.assertIsNone(credit["selected_batch_rows"])
        policy.rag_frontend_plan = None
        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertFalse(telemetry["gpu_p99_credit_downsize_enabled"])

    def test_pre_switch_gpu_p99_downsize_env_is_strict(self):
        policy = _PolicyHarness()
        with patch.dict(
            os.environ,
            {"SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "invalid"},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE must be 0 or 1",
            ):
                policy._get_rag_ann_runtime_state()

    def test_pre_switch_downsized_min_robust_gain_default_zero(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["robust_gain_threshold_eligible"])
        self.assertTrue(counterfactual["robust_gain_threshold_applied"])
        self.assertTrue(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["downsized_min_robust_gain_ms"], 0.0)
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], 79.0)
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_robust_gain_ms),
            [79.0],
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )

        policy.rag_frontend_plan = None
        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertEqual(telemetry["downsized_min_robust_gain_ms"], 0.0)
        self.assertFalse(
            telemetry["downsized_min_robust_gain_enabled"]
        )
        self.assertEqual(telemetry["robust_gain_threshold_holds"], 0)
        self.assertEqual(telemetry["eligible_robust_gain"]["samples"], 1)

    def test_pre_switch_downsized_min_robust_gain_blocks_79_at_80(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "80"
                ),
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"robust_gain_below_min": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers,
            0,
        )
        self.assertEqual(policy.rag_ann_pre_switch_launch_batches, 0)
        self.assertEqual(len(policy.ann_search_pending), 2)
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_robust_gain_ms),
            [79.0],
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["robust_gain_threshold_applied"])
        self.assertFalse(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], -1.0)
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertEqual(credit["commit_stage"], "selector")
        self.assertEqual(
            credit["final_skip_reason"],
            "robust_gain_below_min",
        )

    def test_pre_switch_downsized_min_robust_gain_equality_passes(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "79"
                ),
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["robust_gain_threshold_applied"])
        self.assertTrue(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], 0.0)
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers,
            1,
        )
        self.assertEqual(policy.rag_ann_pre_switch_launch_batches, 1)

    def test_pre_switch_quantile_aligned_switch_cost_holds_low_gain(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.ann_wait_ms = 30.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [5.0] * 18 + [100.0] * 2,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST": (
                    "1"
                ),
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"robust_gain_below_min": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            1,
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["quantile_aligned_switch_cost"])
        self.assertEqual(counterfactual["predicted_tail_switch_ms"], 100.0)
        self.assertEqual(counterfactual["predicted_robust_switch_ms"], 5.0)
        self.assertEqual(counterfactual["robust_switch_percentile"], 50.0)
        self.assertEqual(counterfactual["trigger_type"], "switch_induced")
        self.assertFalse(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], -6.0)

    def test_pre_switch_legacy_switch_quantile_launches_same_candidate(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.ann_wait_ms = 30.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [5.0] * 18 + [100.0] * 2,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST": (
                    "0"
                ),
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertFalse(counterfactual["quantile_aligned_switch_cost"])
        self.assertEqual(counterfactual["predicted_tail_switch_ms"], 100.0)
        self.assertEqual(counterfactual["predicted_robust_switch_ms"], 100.0)
        self.assertTrue(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], 89.0)

    def test_pre_switch_quantile_alignment_requires_mature_pair(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [140.0] * 3,
            maxlen=64,
        )
        policy.rag_partition_switch_latency_ms = deque(
            [0.02] * 64,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "0"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST": (
                    "1"
                ),
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["quantile_aligned_switch_cost"])
        self.assertFalse(
            counterfactual["quantile_aligned_switch_cost_applied"]
        )
        self.assertEqual(
            counterfactual["quantile_aligned_switch_cost_reason"],
            "insufficient_pair_samples",
        )
        self.assertEqual(counterfactual["robust_switch_pair_samples"], 3)
        self.assertEqual(
            counterfactual["predicted_robust_switch_ms"],
            counterfactual["predicted_tail_switch_ms"],
        )

    def test_pre_switch_intrinsic_debt_bypasses_min_gain_threshold(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [5.0] * 18 + [100.0] * 2,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "200"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_QUANTILE_ALIGNED_SWITCH_COST": (
                    "1"
                ),
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_bypasses,
            1,
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(
            counterfactual["trigger_type"],
            "intrinsic_debt_recovery",
        )
        self.assertTrue(counterfactual["robust_gain_threshold_candidate"])
        self.assertFalse(counterfactual["robust_gain_threshold_eligible"])
        self.assertTrue(counterfactual["robust_gain_threshold_bypassed"])
        self.assertEqual(
            counterfactual["robust_gain_threshold_bypass_reason"],
            "intrinsic_debt_recovery",
        )
        self.assertTrue(
            counterfactual["robust_gain_threshold_bypass_counted"]
        )

    def test_pre_switch_batch_net_gain_accounts_for_residual_row(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["robust_gain_ms"], 79.0)
        self.assertEqual(counterfactual["residual_rows"], 1)
        self.assertEqual(counterfactual["residual_eta_ms"], 116.0)
        self.assertEqual(counterfactual["residual_gain_ms"], -21.0)
        self.assertEqual(counterfactual["batch_net_gain_ms"], 29.0)
        self.assertEqual(counterfactual["threshold_decision_gain_ms"], 29.0)
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], -21.0)
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_raw_robust_gain_ms),
            [79.0],
        )
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_batch_net_gain_ms),
            [29.0],
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            1,
        )

    def test_pre_switch_batch_net_gain_preserves_high_value_launch(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [100.0] * 4,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["robust_gain_ms"], 139.0)
        self.assertEqual(counterfactual["residual_gain_ms"], -21.0)
        self.assertEqual(counterfactual["batch_net_gain_ms"], 59.0)
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], 9.0)
        self.assertTrue(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )

    def test_pre_switch_batch_net_gain_intrinsic_debt_still_bypasses(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["batch_net_gain_ms"], 29.0)
        self.assertTrue(counterfactual["robust_gain_threshold_bypassed"])
        self.assertFalse(counterfactual["robust_gain_threshold_applied"])
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_bypasses,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )

    def test_pre_switch_downsized_min_robust_gain_is_strict(self):
        for configured in ("-1", "nan", "inf", "-inf", "invalid", ""):
            with self.subTest(configured=configured):
                policy = _PolicyHarness()
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                            configured
                        )
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be a finite nonnegative number",
                    ):
                        policy._rag_ann_pre_switch_downsized_min_robust_gain_ms()

    def test_pre_switch_downsized_min_robust_gain_is_derived_only(self):
        policy = self._p99_downsize_pre_switch_policy()

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "80"
                ),
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertEqual(
            credit["adaptive_reason"],
            "configured_credit_already_strict",
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertFalse(counterfactual["robust_gain_threshold_eligible"])
        self.assertFalse(counterfactual["robust_gain_threshold_applied"])
        self.assertIsNone(counterfactual["robust_gain_threshold_passed"])
        self.assertIsNone(counterfactual["robust_gain_headroom_ms"])
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_robust_gain_ms),
            [],
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )

    def test_pre_switch_prefill_deadline_precedes_robust_gain_threshold(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.allow_prefill = True
        policy.rag_frontend_plan.prefill_wait_ms = 440.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "80"
                ),
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"prefill_deadline": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_holds,
            0,
        )
        self.assertEqual(
            list(policy.rag_ann_pre_switch_eligible_robust_gain_ms),
            [],
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["robust_gain_threshold_eligible"])
        self.assertFalse(counterfactual["robust_gain_threshold_applied"])
        self.assertIsNone(counterfactual["robust_gain_threshold_passed"])
        self.assertEqual(counterfactual["robust_gain_headroom_ms"], -1.0)
        self.assertFalse(counterfactual["cross_class_debt_candidate"])
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_candidates,
            0,
        )

    def test_pre_switch_cross_class_debt_off_records_shadow_opportunity(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.allow_prefill = True
        policy.rag_frontend_plan.prefill_wait_ms = 450.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [120.0] * 4,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT": "0",
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"prefill_deadline": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_candidates,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_would_admit,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_bypasses,
            0,
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["cross_class_debt_candidate"])
        self.assertTrue(counterfactual["cross_class_debt_passed"])
        self.assertFalse(counterfactual["cross_class_debt_enabled"])
        self.assertEqual(counterfactual["prefill_no_launch_debt_ms"], 82.0)
        self.assertEqual(counterfactual["prefill_prelaunch_debt_ms"], 91.0)
        self.assertEqual(counterfactual["incremental_prefill_debt_ms"], 9.0)
        self.assertEqual(counterfactual["batch_net_gain_ms"], 69.0)
        self.assertEqual(
            counterfactual["ann_no_launch_deadline_debt_ms"],
            250.0,
        )
        self.assertEqual(
            counterfactual["ann_prelaunch_deadline_debt_ms"],
            0.0,
        )
        self.assertEqual(
            counterfactual["ann_residual_deadline_debt_ms"],
            141.0,
        )
        self.assertEqual(counterfactual["ann_deadline_debt_saved_ms"], 109.0)
        self.assertEqual(counterfactual["cross_class_net_debt_ms"], 91.0)

    def test_pre_switch_cross_class_debt_admits_positive_net_debt(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.allow_prefill = True
        policy.rag_frontend_plan.prefill_wait_ms = 450.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [120.0] * 4,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT": "1",
            },
            clear=True,
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_candidates,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_would_admit,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_bypasses,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_holds,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_robust_gain_threshold_bypasses,
            1,
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertTrue(counterfactual["cross_class_debt_bypassed"])
        self.assertTrue(counterfactual["cross_class_debt_bypass_counted"])
        self.assertEqual(counterfactual["cross_class_net_debt_ms"], 91.0)
        self.assertEqual(counterfactual["cross_class_prefill_debt_weight"], 2.0)
        self.assertEqual(counterfactual["cross_class_min_net_debt_ms"], 50.0)

        policy.rag_frontend_plan = None
        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertTrue(telemetry["cross_class_debt_enabled"])
        self.assertEqual(telemetry["cross_class_debt_candidates"], 1)
        self.assertEqual(telemetry["cross_class_debt_would_admit"], 1)
        self.assertEqual(telemetry["cross_class_debt_bypasses"], 1)
        self.assertEqual(
            telemetry["eligible_cross_class_net_debt"]["samples"],
            1,
        )

    def test_pre_switch_cross_class_debt_holds_below_guarded_minimum(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_frontend_plan.allow_prefill = True
        policy.rag_frontend_plan.prefill_wait_ms = 450.0
        policy.rag_frontend_plan.ann_wait_ms = 40.0
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [60.0] * 4,
            maxlen=64,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_DOWNSIZED_MIN_ROBUST_GAIN_MS": (
                    "50"
                ),
                "SGLANG_RAG_ANN_PRE_SWITCH_BATCH_NET_GAIN": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT": "1",
            },
            clear=True,
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"cross_class_debt_below_min": 1},
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_candidates,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_would_admit,
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_holds,
            1,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_cross_class_debt_bypasses,
            0,
        )
        counterfactual = policy.rag_ann_pre_switch_last_counterfactual
        self.assertEqual(counterfactual["batch_net_gain_ms"], 39.0)
        self.assertEqual(counterfactual["incremental_prefill_debt_ms"], 9.0)
        self.assertEqual(counterfactual["ann_deadline_debt_saved_ms"], 49.0)
        self.assertEqual(counterfactual["cross_class_net_debt_ms"], 31.0)
        self.assertFalse(counterfactual["cross_class_debt_passed"])

    def test_pre_switch_cross_class_debt_env_is_strict(self):
        policy = _PolicyHarness()
        with patch.dict(
            os.environ,
            {"SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT": "invalid"},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "SGLANG_RAG_ANN_PRE_SWITCH_CROSS_CLASS_DEBT must be 0 or 1",
            ):
                policy._get_rag_ann_runtime_state()

    def test_pre_switch_adaptive_p99_credit_derives_and_downsizes(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        launches = []

        def record_launch(**kwargs):
            launches.append(kwargs)
            policy.rag_ann_pre_switch_launch_batches += 1

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                    "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
                },
            ),
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=7.0,
            ),
            patch.object(
                policy,
                "_get_rag_ann_stream",
                return_value=SimpleNamespace(cuda_stream=123),
            ),
            patch.object(
                policy,
                "_launch_ann_search_side_channel",
                side_effect=record_launch,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertTrue(
                policy._maybe_launch_ann_before_partition_switch(2)
            )

        self.assertEqual(len(launches), 1)
        self.assertEqual(
            launches[0]["first_batch_selection"],
            (1, 15.0),
        )
        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertTrue(credit["adaptive_enabled"])
        self.assertTrue(credit["adaptive_attempted"])
        self.assertEqual(credit["adaptive_reason"], "derived")
        self.assertEqual(credit["adaptive_reference_batch_rows"], 1)
        self.assertAlmostEqual(
            credit["adaptive_reference_batch_p99_ms"], 15.0
        )
        self.assertAlmostEqual(credit["adaptive_lower_ms"], 15.0)
        self.assertAlmostEqual(credit["adaptive_upper_ms"], 30.0)
        self.assertAlmostEqual(
            credit["adaptive_derived_limit_ms"], 22.5
        )
        self.assertAlmostEqual(credit["configured_limit_ms"], 100.0)
        self.assertAlmostEqual(credit["limit_ms"], 22.5)
        self.assertEqual(credit["selected_batch_rows"], 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_adaptive_p99_credit_derivations,
            1,
        )
        self.assertEqual(
            getattr(
                policy,
                "rag_ann_pre_switch_adaptive_p99_credit_unavailable",
                0,
            ),
            0,
        )
        policy.rag_frontend_plan = None
        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]
        self.assertTrue(telemetry["adaptive_p99_credit_enabled"])
        self.assertEqual(
            telemetry["adaptive_p99_credit_derivations"], 1
        )
        self.assertEqual(
            telemetry["adaptive_p99_credit_unavailable"], 0
        )

    def test_pre_switch_adaptive_p99_credit_off_arm_holds(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "0",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
            },
        ):
            self.assertFalse(self._run_counterfactual_pre_switch(policy))

        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertTrue(credit["adaptive_enabled"])
        self.assertEqual(credit["adaptive_reason"], "derived")
        self.assertAlmostEqual(credit["configured_limit_ms"], 100.0)
        self.assertAlmostEqual(credit["limit_ms"], 22.5)
        self.assertIsNone(credit["selected_batch_rows"])
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_downsizes, 0
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_gpu_p99_credit_holds, 1
        )

    def test_pre_switch_adaptive_p99_credit_requires_safe_interval(self):
        policy = self._p99_downsize_pre_switch_policy()
        policy.rag_ann_max_queued_gpu_ms = 100.0
        policy.rag_ann_gpu_service_ms_by_batch_size[1] = deque(
            [35.0] * 64, maxlen=64
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_P99_DOWNSIZE": "1",
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": "1",
            },
        ):
            self.assertTrue(self._run_counterfactual_pre_switch(policy))

        credit = policy.rag_ann_pre_switch_last_gpu_p99_credit
        self.assertEqual(
            credit["adaptive_reason"],
            "no_safe_exact_prefix_interval",
        )
        self.assertEqual(credit["configured_limit_ms"], 100.0)
        self.assertEqual(credit["limit_ms"], 100.0)
        self.assertEqual(credit["selected_batch_rows"], 2)
        self.assertEqual(
            getattr(
                policy,
                "rag_ann_pre_switch_adaptive_p99_credit_derivations",
                0,
            ),
            0,
        )
        self.assertEqual(
            policy.rag_ann_pre_switch_adaptive_p99_credit_unavailable,
            1,
        )

    def test_pre_switch_adaptive_p99_credit_env_is_strict(self):
        policy = _PolicyHarness()
        with patch.dict(
            os.environ,
            {
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT": (
                    "invalid"
                )
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "SGLANG_RAG_ANN_PRE_SWITCH_ADAPTIVE_P99_CREDIT "
                "must be 0 or 1",
            ):
                policy._get_rag_ann_runtime_state()

    def test_pre_switch_completion_envelopes_fail_closed_for_other_batch(self):
        policy = self._counterfactual_pre_switch_policy(
            switch_ms=80.0,
            service_ms=20.0,
            tail_ms=13.0,
            gpu_ms=7.0,
            inline_ms=1.0,
        )
        policy.rag_ann_completion_service_ms_by_path_batch = {
            "full_gpu": {2: deque([20.0] * 4, maxlen=64)}
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            "full_gpu": {2: deque([13.0] * 4, maxlen=64)}
        }

        self.assertFalse(self._run_counterfactual_pre_switch(policy))
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"completion_prediction_unavailable": 1},
        )

    def test_pre_switch_launch_never_uses_atomic_over_credit(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_max_queued_gpu_ms = 24.0
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([30.0] * 4, maxlen=64)
        }
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=False,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=50.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [80.0] * 4, maxlen=64
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"credit": 1},
        )
        self.assertEqual(policy.rag_ann_pre_switch_safety_violations, 0)

    def test_pre_switch_launch_preserves_urgent_prefill(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=True,
            reason="ann_decode_overlap",
            immediate_handoff=True,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=20.0,
            # The GPU/decode wall path alone is 425 + 10 + 40 + 5 = 480 ms.
            # The learned 20 ms full-GPU inline collection tail reaches the
            # deadline and must make this launch unsafe.
            prefill_wait_ms=425.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [40.0] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy, service_ms=30.0, tail_ms=20.0
        )

        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"prefill_deadline": 1},
        )

    def test_pre_switch_hybrid_completion_path_fails_closed(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_work_conserving_decode_overlap = True
        policy.rag_ann_pre_switch_launch = True
        policy.rag_ann_launch_site = "adaptive"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.ann_executor = SimpleNamespace(hot_index_mode=True)
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            allow_prefill=False,
            reason="ann_decode_overlap",
            immediate_handoff=True,
            forced_ann=False,
            target_stream_idx=2,
            ann_wait_ms=0.0,
            prefill_wait_ms=0.0,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                }
            ]
        )
        policy.rag_partition_switch_pair_latency_ms[(5, 2)] = deque(
            [40.0] * 4, maxlen=64
        )
        self._seed_ann_completion_prediction(
            policy,
            path="hybrid_gpu",
            service_ms=90.0,
            tail_ms=80.0,
        )
        with (
            patch.object(
                policy,
                "_rag_ann_decode_prediction_ms",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            launched = policy._maybe_launch_ann_before_partition_switch(2)

        self.assertFalse(launched)
        self.assertEqual(
            policy.rag_ann_pre_switch_no_launch_reason_counts,
            {"completion_path_unsupported": 1},
        )

    def test_pre_switch_launch_never_overlaps_hbm_native_map(self):
        policy = _PolicyHarness()
        policy.shared_hbm_controller = SimpleNamespace(
            pending_ann_reclaim=None,
            pending_kv_native_map={"future": object()},
        )

        ready, reason = policy._ann_pre_switch_hbm_ready(1)

        self.assertFalse(ready)
        self.assertEqual(reason, "kv_native_map_pending")

    def test_pre_switch_hbm_gate_rejects_ivf_and_moving_owners(self):
        policy = _PolicyHarness()
        controller = SimpleNamespace(
            pending_ann_reclaim=None,
            pending_kv_native_map=None,
            pending_kv_publications=[],
            kv_backlog_committed=False,
        )
        policy.shared_hbm_controller = controller

        controller.pool = SimpleNamespace(mode="ivf")
        ready, reason = policy._ann_pre_switch_hbm_ready(1)
        self.assertFalse(ready)
        self.assertEqual(reason, "ivf_residency_mutation")

        controller.pool = SimpleNamespace(
            mode="scratch",
            faiss_scratch_detaching=lambda: False,
            faiss_mapping_state=lambda: (True, (1234, 512), 1),
            snapshot=lambda: SimpleNamespace(
                moving_bytes=64,
                ann_scratch_bytes=512,
                total_bytes=1024,
            ),
        )
        controller.ann_scratch_request_bytes = lambda *args, **kwargs: 256
        ready, reason = policy._ann_pre_switch_hbm_ready(1)
        self.assertFalse(ready)
        self.assertEqual(reason, "hbm_moving")

    def test_pre_switch_hbm_forecast_is_unrecorded(self):
        policy = _PolicyHarness()
        record_args = []

        def predict_kv_tokens(_scheduler, *, record=True):
            record_args.append(record)
            return 100

        policy.shared_hbm_controller = SimpleNamespace(
            pending_ann_reclaim=None,
            pending_kv_native_map=None,
            pending_kv_publications=[],
            kv_backlog_committed=False,
            pool=SimpleNamespace(
                mode="scratch",
                faiss_scratch_detaching=lambda: False,
                faiss_mapping_state=lambda: (True, (1234, 512), 1),
                snapshot=lambda: SimpleNamespace(
                    moving_bytes=0,
                    ann_scratch_bytes=512,
                    total_bytes=1024,
                ),
            ),
            ann_scratch_request_bytes=lambda *args, **kwargs: 256,
            predict_kv_tokens=predict_kv_tokens,
            token_allocator=SimpleNamespace(
                current_mapped_capacity=lambda: 200
            ),
        )

        ready, reason = policy._ann_pre_switch_hbm_ready(1)

        self.assertTrue(ready)
        self.assertEqual(reason, "")
        self.assertEqual(record_args, [False])

    def test_cpu_fallback_cancels_prelaunched_cold_search(self):
        policy = _PolicyHarness()
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            reason="ann_decode_overlap",
            forced_ann=False,
        )
        prepared = object()
        route_future = Future()
        route_future.set_result(prepared)
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": time.perf_counter(),
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "route_future": route_future,
                    "request_id": "cpu-fallback",
                }
            ]
        )
        cpu_result = SimpleNamespace(poll=lambda: True)
        policy.ann_executor = SimpleNamespace(
            cancel_prepared_cold_searches=Mock(return_value=1),
            search_cpu_async=Mock(return_value=cpu_result),
        )
        policy.shared_hbm_controller = SimpleNamespace(
            ann_scratch_request_bytes=lambda *args, **kwargs: 1024,
            before_ann_launch=lambda *args, **kwargs: False,
            cpu_fallbacks=0,
        )
        ann_stream = SimpleNamespace(cuda_stream=123)

        with (
            patch.object(policy, "_get_rag_ann_stream", return_value=ann_stream),
            patch.object(
                policy,
                "_get_stream_group_idx_for_stream",
                return_value=5,
            ),
        ):
            policy._launch_ann_search_side_channel(
                max_new_batches=1,
                allow_current_epoch=True,
                bypass_microbatch_wait=True,
                first_batch_selection=(1, 5.0),
            )

        policy.ann_executor.cancel_prepared_cold_searches.assert_called_once_with(
            [prepared]
        )
        policy.ann_executor.search_cpu_async.assert_called_once()
        self.assertIs(policy.ann_search_inflight[0]["result"], cpu_result)
        self.assertEqual(policy.shared_hbm_controller.cpu_fallbacks, 1)

    def test_pre_switch_launch_failure_has_single_strict_attempt(self):
        policy = _PolicyHarness()
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_ann=True,
            reason="ann_decode_overlap",
            forced_ann=False,
        )
        policy.ann_search_pending = deque(
            [
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": "first",
                },
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": "must-remain",
                },
            ]
        )
        attempts = []

        def fail_search(*args, **kwargs):
            attempts.append((args, kwargs))
            raise RuntimeError("synthetic launch failure")

        policy.ann_executor = SimpleNamespace(search_async=fail_search)
        ann_stream = SimpleNamespace(cuda_stream=123)
        with (
            patch.object(policy, "_get_rag_ann_stream", return_value=ann_stream),
            patch.object(
                policy,
                "_get_stream_group_idx_for_stream",
                return_value=5,
            ),
            patch.object(policy, "_send_ann_search_output") as send_output,
        ):
            policy._launch_ann_search_side_channel(
                max_new_batches=1,
                allow_current_epoch=True,
                bypass_microbatch_wait=True,
                launch_context="pre_switch",
                first_batch_selection=(1, 5.0),
            )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(
            policy.ann_search_pending[0]["request_id"], "must-remain"
        )
        self.assertEqual(policy.ann_search_inflight, [])
        send_output.assert_called_once()

    def test_inline_poll_collects_and_sends_ready_result(self):
        class _ReadyResult:
            @staticmethod
            def gpu_poll():
                return True

            @staticmethod
            def poll():
                return True

            @staticmethod
            def gpu_elapsed_ms():
                return 4.0

        policy = _PolicyHarness()
        task = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "result": _ReadyResult(),
            "launch_context": "pre_switch",
        }
        policy.ann_search_inflight = [task]
        output = object()
        with (
            patch.object(
                policy,
                "_collect_ann_search_outputs_safe",
                return_value=[output],
            ) as collect,
            patch.object(policy, "_send_ann_search_output") as send,
        ):
            inline_completions = policy._poll_ann_search_side_channel(
                collect_ready_inline=True
            )

        self.assertEqual(inline_completions, 1)
        self.assertEqual(policy.ann_search_inflight, [])
        collect.assert_called_once_with(task)
        send.assert_called_once_with(output)
        self.assertFalse(hasattr(policy, "ann_result_worker"))

    def test_route_pending_result_is_not_counted_as_gpu_work(self):
        class _RoutePendingResult:
            gpu_poll_calls = 0

            @staticmethod
            def gpu_started():
                return False

            @classmethod
            def gpu_poll(cls):
                cls.gpu_poll_calls += 1
                return False

            @staticmethod
            def poll():
                return False

        policy = _PolicyHarness()
        result = _RoutePendingResult()

        self.assertFalse(policy._ann_result_gpu_pending(result))
        self.assertEqual(result.gpu_poll_calls, 0)

    def test_route_pending_result_does_not_publish_gpu_service(self):
        class _RoutePendingResult:
            @staticmethod
            def gpu_started():
                return False

            @staticmethod
            def gpu_poll():
                raise AssertionError("GPU has not started")

            @staticmethod
            def poll():
                return False

        policy = _PolicyHarness()
        task = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "result": _RoutePendingResult(),
            "launch_context": "normal",
        }
        policy.ann_search_inflight = [task]

        inline_completions = policy._poll_ann_search_side_channel(
            collect_ready_inline=True
        )

        self.assertEqual(inline_completions, 0)
        self.assertEqual(policy.ann_search_inflight, [task])
        self.assertNotIn("ann_gpu_service_recorded", task)

    def test_cold_search_batcher_coalesces_only_queued_work(self):
        class _Owner:
            def __init__(self):
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.batch_sizes = []

            def _search_cold_preassigned_batch(self, pending):
                self.batch_sizes.append(
                    sum(len(item.queries) for item in pending)
                )
                if len(self.batch_sizes) == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=2)
                for item in pending:
                    item.future.set_result(object())

        owner = _Owner()
        batcher = _ColdSearchBatcher(owner, max_batch_queries=4)
        try:
            first = batcher.submit(
                np.zeros((1, 8), dtype=np.float32), object(), time.perf_counter()
            )
            self.assertTrue(owner.first_started.wait(timeout=1))
            queued = [
                batcher.submit(
                    np.zeros((1, 8), dtype=np.float32),
                    object(),
                    time.perf_counter(),
                )
                for _ in range(2)
            ]
            owner.release_first.set()
            first.result(timeout=1)
            for future in queued:
                future.result(timeout=1)
        finally:
            batcher.close()

        self.assertEqual(owner.batch_sizes, [1, 2])
        stats = batcher.stats()
        self.assertEqual(stats["batches"], 2)
        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["batch_size_histogram"], {1: 1, 2: 1})
        self.assertGreaterEqual(stats["peak_queued_rows"], 2)

    def test_cold_search_batcher_skips_cancelled_queued_work(self):
        class _Owner:
            def __init__(self):
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.batch_sizes = []

            def _search_cold_preassigned_batch(self, pending):
                self.batch_sizes.append(
                    sum(len(item.queries) for item in pending)
                )
                if len(self.batch_sizes) == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=2)
                for item in pending:
                    item.future.set_result(object())

        owner = _Owner()
        batcher = _ColdSearchBatcher(owner, max_batch_queries=4)
        try:
            first = batcher.submit(
                np.zeros((1, 8), dtype=np.float32),
                object(),
                time.perf_counter(),
            )
            self.assertTrue(owner.first_started.wait(timeout=1))
            cancelled = batcher.submit(
                np.zeros((1, 8), dtype=np.float32),
                object(),
                time.perf_counter(),
            )
            kept = batcher.submit(
                np.zeros((1, 8), dtype=np.float32),
                object(),
                time.perf_counter(),
            )
            self.assertTrue(cancelled.cancel())
            owner.release_first.set()
            first.result(timeout=1)
            kept.result(timeout=1)
        finally:
            batcher.close()

        self.assertTrue(cancelled.cancelled())
        self.assertEqual(owner.batch_sizes, [1, 1])
        stats = batcher.stats()
        self.assertEqual(stats["batches"], 2)
        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["batch_size_histogram"], {1: 2})

    def test_hot_route_batcher_coalesces_and_slices_queued_work(self):
        executor = AnnExecutor.__new__(AnnExecutor)
        executor._route_batch_size_histogram = {}
        executor._route_batches = 0
        executor._route_rows = 0
        executor.first_started = threading.Event()
        executor.release_first = threading.Event()
        executor.batch_sizes = []

        def quantize(queries):
            rows = len(queries)
            executor.batch_sizes.append(rows)
            if len(executor.batch_sizes) == 1:
                executor.first_started.set()
                executor.release_first.wait(timeout=2)
            row_ids = np.arange(rows, dtype=np.int64).reshape(rows, 1)
            routed = RoutedProbes(
                hot_distances=np.zeros((rows, 1), dtype=np.float32),
                hot_ids=row_ids,
                cold_distances=np.ones((rows, 1), dtype=np.float32),
                cold_ids=row_ids + 10,
                hot_counts=np.ones(rows, dtype=np.int32),
                cold_counts=np.ones(rows, dtype=np.int32),
            )
            return routed, float(rows), 0.25

        executor._quantize_hot_probes = quantize
        batcher = _HotRouteBatcher(executor, max_batch_queries=4)
        try:
            first = batcher.submit(np.zeros((1, 8), dtype=np.float32))
            self.assertTrue(executor.first_started.wait(timeout=1))
            queued = [
                batcher.submit(np.zeros((1, 8), dtype=np.float32))
                for _ in range(2)
            ]
            executor.release_first.set()
            first_result = first.result(timeout=1)
            queued_results = [future.result(timeout=1) for future in queued]
        finally:
            batcher.close()

        self.assertEqual(executor.batch_sizes, [1, 2])
        self.assertEqual(executor._route_batch_size_histogram, {1: 1, 2: 1})
        self.assertEqual(first_result[0].hot_ids.shape, (1, 1))
        self.assertEqual(
            [result[0].hot_ids.shape for result in queued_results],
            [(1, 1), (1, 1)],
        )

    def test_hot_route_prelaunches_request_matched_cold_search(self):
        class _ColdBatcher:
            def __init__(self):
                self.submissions = []

            def submit(self, queries, routed, enqueued_at):
                self.submissions.append((queries, routed, enqueued_at))
                future = Future()
                future.set_result(
                    (
                        np.zeros((len(queries), 1), dtype=np.float32),
                        np.full((len(queries), 1), 7, dtype=np.int64),
                        4.0,
                        2.0,
                    )
                )
                return future

        executor = AnnExecutor.__new__(AnnExecutor)
        executor._route_batch_size_histogram = {}
        executor._route_batches = 0
        executor._route_rows = 0
        executor._cold_batcher = _ColdBatcher()
        executor._cold_prelaunch_lock = threading.Lock()
        executor._cold_prelaunch_submissions = 0
        executor._cold_prelaunch_rows = 0
        executor._cold_prelaunch_attachments = 0
        executor._cold_prelaunch_attached_rows = 0
        executor._cold_prelaunch_completed_before_gpu = 0
        executor._cold_prelaunch_lead_ms = deque(maxlen=64)

        def quantize(queries):
            rows = len(queries)
            routed = RoutedProbes(
                hot_distances=np.zeros((rows, 1), dtype=np.float32),
                hot_ids=np.ones((rows, 1), dtype=np.int64),
                cold_distances=np.ones((rows, 1), dtype=np.float32),
                cold_ids=np.full((rows, 1), 2, dtype=np.int64),
                hot_counts=np.ones(rows, dtype=np.int32),
                cold_counts=np.ones(rows, dtype=np.int32),
            )
            return routed, 3.0, 1.0

        executor._quantize_hot_probes = quantize
        batcher = _HotRouteBatcher(executor, max_batch_queries=4)
        executor._route_batcher = batcher
        try:
            prepared = batcher.submit(
                np.zeros((1, 8), dtype=np.float32)
            ).result(timeout=1)
        finally:
            batcher.close()

        self.assertIsInstance(prepared, PreparedHotSearch)
        self.assertEqual(len(executor._cold_batcher.submissions), 1)
        self.assertEqual(len(prepared.cold_searches), 1)
        self.assertTrue(prepared.cold_searches[0].future.done())
        self.assertEqual(executor._cold_prelaunch_submissions, 1)
        self.assertEqual(executor._cold_prelaunch_rows, 1)

    def test_combine_prepared_hot_searches_pads_probe_widths(self):
        first = RoutedProbes(
            hot_distances=np.array([[0.1]], dtype=np.float32),
            hot_ids=np.array([[1]], dtype=np.int64),
            cold_distances=np.array([[0.2, 0.3]], dtype=np.float32),
            cold_ids=np.array([[2, 3]], dtype=np.int64),
            hot_counts=np.array([1], dtype=np.int32),
            cold_counts=np.array([2], dtype=np.int32),
        )
        second = RoutedProbes(
            hot_distances=np.array([[0.4, 0.5]], dtype=np.float32),
            hot_ids=np.array([[4, 5]], dtype=np.int64),
            cold_distances=np.array([[0.6]], dtype=np.float32),
            cold_ids=np.array([[6]], dtype=np.int64),
            hot_counts=np.array([2], dtype=np.int32),
            cold_counts=np.array([1], dtype=np.int32),
        )

        routed, quantize_ms, route_ms = (
            AnnExecutor.combine_prepared_hot_searches(
                [(first, 10.0, 1.0), (second, 20.0, 3.0)]
            )
        )

        self.assertEqual(routed.hot_ids.tolist(), [[1, -1], [4, 5]])
        self.assertEqual(routed.cold_ids.tolist(), [[2, 3], [6, -1]])
        self.assertEqual(routed.hot_counts.tolist(), [1, 2])
        self.assertEqual(routed.cold_counts.tolist(), [2, 1])
        self.assertEqual(quantize_ms, 15.0)
        self.assertEqual(route_ms, 2.0)

    def test_combine_prepared_hot_searches_preserves_cold_spans(self):
        first_routed = RoutedProbes(
            hot_distances=np.array([[0.1]], dtype=np.float32),
            hot_ids=np.array([[1]], dtype=np.int64),
            cold_distances=np.array([[0.2]], dtype=np.float32),
            cold_ids=np.array([[2]], dtype=np.int64),
            hot_counts=np.array([1], dtype=np.int32),
            cold_counts=np.array([1], dtype=np.int32),
        )
        second_routed = RoutedProbes(
            hot_distances=np.array([[0.3]], dtype=np.float32),
            hot_ids=np.array([[3]], dtype=np.int64),
            cold_distances=np.array([[0.4]], dtype=np.float32),
            cold_ids=np.array([[4]], dtype=np.int64),
            hot_counts=np.array([1], dtype=np.int32),
            cold_counts=np.array([1], dtype=np.int32),
        )
        first_future = Future()
        second_future = Future()
        first = PreparedHotSearch(
            first_routed,
            1.0,
            2.0,
            (_PreparedColdSearch(0, 1, first_future, 10.0),),
        )
        second = PreparedHotSearch(
            second_routed,
            3.0,
            4.0,
            (_PreparedColdSearch(0, 1, second_future, 11.0),),
        )

        combined = AnnExecutor.combine_prepared_hot_searches([first, second])

        self.assertIsInstance(combined, PreparedHotSearch)
        self.assertEqual(
            [(item.start, item.end) for item in combined.cold_searches],
            [(0, 1), (1, 2)],
        )
        self.assertIs(combined.cold_searches[0].future, first_future)
        self.assertIs(combined.cold_searches[1].future, second_future)

    def test_hot_ann_result_reports_request_matched_cold_metrics(self):
        completion_samples = []
        first_future = Future()
        first_future.set_result(
            (
                np.array([[0.1]], dtype=np.float32),
                np.array([[10]], dtype=np.int64),
                5.0,
                2.0,
            )
        )
        second_future = Future()
        second_future.set_result(
            (
                np.array([[0.2]], dtype=np.float32),
                np.array([[20]], dtype=np.int64),
                11.0,
                7.0,
            )
        )
        result = HotAnnResult(
            gpu_result=None,
            cpu_future=None,
            batch_size=2,
            topk=1,
            hot_counts=np.zeros(2, dtype=np.int32),
            cold_counts=np.ones(2, dtype=np.int32),
            coarse_quantize_latency_ms=1.0,
            route_latency_ms=0.5,
            cpu_searches=(
                _PreparedColdSearch(0, 1, first_future, 10.0),
                _PreparedColdSearch(1, 2, second_future, 11.0),
            ),
            completion_callback=lambda **sample: completion_samples.append(
                sample
            ),
        )

        self.assertTrue(result.poll())
        self.assertEqual(result.ids_to_cpu().tolist(), [[10], [20]])
        first_metrics = result.metrics_for_span(0, 1)
        second_metrics = result.metrics_for_span(1, 2)
        self.assertEqual(first_metrics["cpu_queue_wait_ms"], 2.0)
        self.assertEqual(first_metrics["cpu_search_latency_ms"], 5.0)
        self.assertEqual(second_metrics["cpu_queue_wait_ms"], 7.0)
        self.assertEqual(second_metrics["cpu_search_latency_ms"], 11.0)
        self.assertEqual(len(completion_samples), 1)
        self.assertEqual(completion_samples[0]["cpu_latency_ms"], 8.0)
        self.assertEqual(completion_samples[0]["cpu_queue_ms"], 4.5)
        self.assertEqual(completion_samples[0]["cpu_probes"], 2)

    def test_decode_sync_poll_collects_decode_gc_ann_inline(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.ann_search_inflight = [object()]

        with patch.object(
            policy,
            "_poll_ann_search_side_channel",
            return_value=2,
        ) as poll:
            inline_completions = policy._poll_ann_after_decode_sync()

        self.assertEqual(inline_completions, 2)
        poll.assert_called_once_with(collect_ready_inline=True)
        self.assertEqual(policy.rag_ann_decode_completion_polls, 1)
        self.assertEqual(policy.rag_ann_decode_inline_completions, 2)

    def test_decode_sync_poll_skips_frontend_gc(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "frontend-gc"
        policy.ann_search_inflight = [object()]

        with patch.object(
            policy,
            "_poll_ann_search_side_channel",
        ) as poll:
            inline_completions = policy._poll_ann_after_decode_sync()

        self.assertEqual(inline_completions, 0)
        poll.assert_not_called()

    def test_decode_gc_tick_poll_collects_ready_ann_inline(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_partition_switch_deferred = False
        policy.ann_search_inflight = [{"launch_context": "normal"}]

        with patch.object(
            policy,
            "_poll_ann_search_side_channel",
            return_value=1,
        ) as poll:
            inline_completions = policy._poll_ann_at_scheduler_tick_start()

        self.assertEqual(inline_completions, 1)
        poll.assert_called_once_with(collect_ready_inline=True)
        self.assertEqual(policy.rag_ann_decode_tick_completion_polls, 1)
        self.assertEqual(policy.rag_ann_decode_tick_inline_completions, 1)

    def test_frontend_gc_tick_poll_keeps_worker_collection(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "frontend-gc"
        policy.rag_ann_partition_switch_deferred = False
        policy.ann_search_inflight = [{"launch_context": "normal"}]

        with patch.object(
            policy,
            "_poll_ann_search_side_channel",
            return_value=0,
        ) as poll:
            inline_completions = policy._poll_ann_at_scheduler_tick_start()

        self.assertEqual(inline_completions, 0)
        poll.assert_called_once_with(collect_ready_inline=False)
        self.assertFalse(
            hasattr(policy, "rag_ann_decode_tick_completion_polls")
        )

    def test_overdue_decode_gc_ann_inserts_nonblocking_decode_fence(self):
        event = object()
        result = SimpleNamespace(
            event=event,
            gpu_poll=lambda: False,
            poll=lambda: False,
        )
        task = {
            "result": result,
            "launch_time": time.perf_counter() - 0.050,
        }
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_max_queued_gpu_ms = 30.0
        policy.ann_search_inflight = [task]
        decode_stream = SimpleNamespace(wait_event=Mock())

        fenced = policy._maybe_fence_overdue_ann_before_decode(decode_stream)

        self.assertTrue(fenced)
        decode_stream.wait_event.assert_called_once_with(event)
        self.assertTrue(task["decode_credit_fence_inserted"])
        self.assertEqual(policy.rag_ann_decode_credit_fences, 1)
        self.assertGreater(
            policy.rag_ann_decode_credit_fence_age_ms[0], 30.0
        )

    def test_young_decode_gc_ann_keeps_work_conserving_overlap(self):
        result = SimpleNamespace(
            event=object(),
            gpu_poll=lambda: False,
            poll=lambda: False,
        )
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_max_queued_gpu_ms = 30.0
        policy.ann_search_inflight = [
            {"result": result, "launch_time": time.perf_counter()}
        ]
        decode_stream = SimpleNamespace(wait_event=Mock())

        fenced = policy._maybe_fence_overdue_ann_before_decode(decode_stream)

        self.assertFalse(fenced)
        decode_stream.wait_event.assert_not_called()

    def test_predicted_decode_debt_fences_young_decode_gc_ann(self):
        event = object()
        result = SimpleNamespace(
            event=event,
            gpu_poll=lambda: False,
            poll=lambda: False,
        )
        task = {
            "result": result,
            "launch_time": time.perf_counter() - 0.005,
        }
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_max_queued_gpu_ms = 30.0
        policy.ann_search_inflight = [task]
        decode_stream = SimpleNamespace(wait_event=Mock())

        with patch.object(
            policy,
            "_rag_ann_decode_prediction_ms",
            return_value=26.0,
        ) as predict:
            fenced = policy._maybe_fence_overdue_ann_before_decode(
                decode_stream
            )

        self.assertTrue(fenced)
        predict.assert_called_once_with()
        decode_stream.wait_event.assert_called_once_with(event)
        self.assertEqual(task["decode_credit_fence_reason"], "predicted")
        self.assertEqual(
            policy.rag_ann_decode_credit_fence_reason_counts,
            {"predicted": 1},
        )
        self.assertEqual(
            policy.rag_ann_decode_credit_fence_prediction_ms[0], 26.0
        )

    def test_predicted_decode_debt_below_credit_keeps_overlap(self):
        result = SimpleNamespace(
            event=object(),
            gpu_poll=lambda: False,
            poll=lambda: False,
        )
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_max_queued_gpu_ms = 30.0
        policy.ann_search_inflight = [
            {
                "result": result,
                "launch_time": time.perf_counter() - 0.005,
            }
        ]
        decode_stream = SimpleNamespace(wait_event=Mock())

        with patch.object(
            policy,
            "_rag_ann_decode_prediction_ms",
            return_value=20.0,
        ):
            fenced = policy._maybe_fence_overdue_ann_before_decode(
                decode_stream
            )

        self.assertFalse(fenced)
        decode_stream.wait_event.assert_not_called()

    def test_credit_fence_publishes_ann_handoff_before_decode(self):
        policy = _PolicyHarness()
        decode_stream = SimpleNamespace(synchronize=Mock())

        with (
            patch.object(
                policy,
                "_maybe_fence_overdue_ann_before_decode",
                return_value=True,
            ) as fence,
            patch.object(
                policy,
                "_poll_ann_search_side_channel",
                return_value=1,
            ) as poll,
        ):
            completions = policy._drain_fenced_ann_before_decode(
                decode_stream
            )

        self.assertEqual(completions, 1)
        fence.assert_called_once_with(decode_stream)
        decode_stream.synchronize.assert_called_once_with()
        poll.assert_called_once_with(collect_ready_inline=True)
        self.assertEqual(policy.rag_ann_decode_credit_sync_drains, 1)
        self.assertEqual(
            policy.rag_ann_decode_credit_inline_completions, 1
        )
        self.assertEqual(
            len(policy.rag_ann_decode_credit_sync_wait_ms), 1
        )

    def test_inline_poll_retries_transient_zero_gpu_elapsed(self):
        class _ReadyResult:
            def __init__(self):
                self.elapsed_reads = 0

            @staticmethod
            def gpu_poll():
                return True

            @staticmethod
            def poll():
                return True

            def gpu_elapsed_ms(self):
                self.elapsed_reads += 1
                return 0.0 if self.elapsed_reads == 1 else 4.0

        policy = _PolicyHarness()
        result = _ReadyResult()
        task = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "result": result,
            "launch_context": "pre_switch",
        }
        policy.ann_search_inflight = [task]
        with (
            patch.object(
                policy,
                "_collect_ann_search_outputs_safe",
                return_value=[],
            ),
            patch.object(policy, "_send_ann_search_output"),
        ):
            inline_completions = policy._poll_ann_search_side_channel(
                collect_ready_inline=True
            )

        self.assertEqual(inline_completions, 1)
        self.assertEqual(result.elapsed_reads, 2)
        self.assertTrue(task["ann_gpu_service_recorded"])
        self.assertEqual(
            list(policy.rag_ann_gpu_service_ms_by_batch_size[1]),
            [4.0],
        )

    def test_inline_poll_records_gpu_elapsed_when_full_poll_wins(self):
        class _ReadyResult:
            def __init__(self):
                self.elapsed_reads = 0

            @staticmethod
            def gpu_poll():
                return False

            @staticmethod
            def poll():
                return True

            def gpu_elapsed_ms(self):
                self.elapsed_reads += 1
                return 4.0

        policy = _PolicyHarness()
        result = _ReadyResult()
        task = {
            "queries": np.zeros((2, 8), dtype=np.float32),
            "result": result,
            "launch_context": "pre_switch",
        }
        policy.ann_search_inflight = [task]
        with (
            patch.object(
                policy,
                "_collect_ann_search_outputs_safe",
                return_value=[],
            ),
            patch.object(policy, "_send_ann_search_output"),
        ):
            inline_completions = policy._poll_ann_search_side_channel(
                collect_ready_inline=True
            )

        self.assertEqual(inline_completions, 1)
        self.assertEqual(result.elapsed_reads, 1)
        self.assertTrue(task["ann_gpu_service_recorded"])
        self.assertEqual(
            list(policy.rag_ann_gpu_service_ms_by_batch_size[2]),
            [4.0],
        )

    def test_pre_switch_completion_helpers_request_inline_collection(self):
        for tick_start in (False, True):
            with self.subTest(tick_start=tick_start):
                policy = _PolicyHarness()
                policy.rag_ann_partition_switch_deferred = tick_start
                policy.ann_search_inflight = [
                    {"launch_context": "pre_switch"}
                ]
                with patch.object(
                    policy,
                    "_poll_ann_search_side_channel",
                    return_value=1,
                ) as poll:
                    if tick_start:
                        inline_completions = (
                            policy._poll_ann_at_scheduler_tick_start()
                        )
                        self.assertEqual(inline_completions, 1)
                    else:
                        polled = (
                            policy._poll_ann_before_partition_switch_if_ready(
                                was_switch_blocked=False,
                                switch_blocked=False,
                                pre_switch_ann_launched=True,
                            )
                        )
                        self.assertTrue(polled)

                poll.assert_called_once_with(collect_ready_inline=True)
                self.assertEqual(
                    policy.rag_ann_partition_completion_polls, 1
                )
                self.assertEqual(
                    policy.rag_ann_partition_inline_completions, 1
                )
                self.assertEqual(
                    policy.rag_ann_pre_switch_completion_polls, 1
                )
                self.assertEqual(
                    policy.rag_ann_pre_switch_inline_completions, 1
                )

    def test_ann_service_prediction_learns_batch_tail(self):
        policy = _PolicyHarness()
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([5.0, 6.0, 7.0, 8.0], maxlen=64)
        }

        one_row_ms = policy._predict_ann_gpu_service_ms(1)
        two_row_ms = policy._predict_ann_gpu_service_ms(2)

        self.assertGreater(one_row_ms, 7.0)
        self.assertLessEqual(one_row_ms, 8.0)
        self.assertAlmostEqual(two_row_ms, one_row_ms * 2.0)

    def test_ann_completion_records_one_merged_gpu_batch_once(self):
        class _ReadyResult:
            batch_size = 2
            topk = 1

            @staticmethod
            def ids_to_cpu():
                return np.asarray([[1], [2]], dtype=np.int64)

            @staticmethod
            def metrics_for_span(_start, _end):
                return {
                    "index_mode": "full_gpu",
                    "hot_probe_ratio": 0.0,
                    "hot_probe_count": 0,
                    "cold_probe_count": 0,
                    "cpu_queue_wait_ms": 0.0,
                    "cpu_search_latency_ms": 0.0,
                    "coarse_quantize_latency_ms": 0.0,
                    "route_latency_ms": 0.0,
                    "merge_latency_ms": 0.0,
                }

        policy = _PolicyHarness()
        policy.rag_ann_cpu_queue_wait_ms = deque(maxlen=64)
        policy.rag_ann_cpu_search_latency_ms = deque(maxlen=64)
        policy.rag_ann_coarse_quantize_latency_ms = deque(maxlen=64)
        policy.rag_ann_route_latency_ms = deque(maxlen=64)
        policy.rag_ann_merge_latency_ms = deque(maxlen=64)
        policy.rag_ann_hot_probe_ratio = deque(maxlen=64)
        task = {
            "result": _ReadyResult(),
            "queries": np.zeros((2, 8), dtype=np.float32),
            "tasks": [
                {
                    "enqueue_time": 99.9,
                    "queue_wait_ms": 90.0,
                    "topk": 1,
                },
                {
                    "enqueue_time": 99.8,
                    "queue_wait_ms": 190.0,
                    "topk": 1,
                },
            ],
            "query_spans": [(0, 1), (1, 2)],
            "gpu_elapsed_ms": 6.0,
            "ann_gpu_path": "full_gpu",
        }

        with patch(
            "sglang.srt.multiplex.multiplexing.time.perf_counter",
            return_value=100.0,
        ):
            policy._collect_ann_search_outputs(task)
            policy._collect_ann_search_outputs(task)

        services = policy.rag_ann_completion_service_ms_by_path_batch[
            "full_gpu"
        ][2]
        tails = policy.rag_ann_post_gpu_tail_ms_by_path_batch["full_gpu"][2]
        self.assertEqual(len(services), 1)
        self.assertEqual(len(tails), 1)
        self.assertAlmostEqual(services[0], 10.0)
        self.assertAlmostEqual(tails[0], 4.0)

    def test_ann_completion_rejects_cpu_and_invalid_residuals(self):
        policy = _PolicyHarness()

        def task(path):
            return {
                "queries": np.zeros((1, 8), dtype=np.float32),
                "ann_gpu_path": path,
            }

        policy._record_ann_completion_service(task(None), [10.0], 5.0)
        policy._record_ann_completion_service(
            task("full_gpu"), [-1.0], 0.5
        )
        policy._record_ann_completion_service(
            task("full_gpu"), [float("nan")], 0.5
        )
        policy._record_ann_completion_service(
            task("full_gpu"), [4.0], 5.0
        )
        policy._record_ann_completion_service(
            task("full_gpu"), [10.0], float("inf")
        )

        self.assertEqual(
            policy.rag_ann_completion_service_ms_by_path_batch, {}
        )
        self.assertEqual(policy.rag_ann_post_gpu_tail_ms_by_path_batch, {})

    def test_ann_completion_excludes_pre_switch_and_inline_samples(self):
        policy = _PolicyHarness()
        base = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "ann_gpu_path": "full_gpu",
        }
        pre_switch = {
            **base,
            "ann_collection_mode": "worker",
            "launch_context": "pre_switch",
        }
        inline = {
            **base,
            "ann_collection_mode": "inline",
            "launch_context": "pre_switch",
        }

        policy._record_ann_completion_service(
            pre_switch, [20.0], 10.0
        )
        policy._record_ann_completion_service(inline, [20.0], 10.0)
        pre_switch["ann_collection_succeeded"] = True
        policy._record_ann_worker_collection_cost(pre_switch, 2.0)
        policy._record_ann_worker_send_cost(pre_switch, [0.5])

        self.assertEqual(
            policy.rag_ann_completion_service_ms_by_path_batch, {}
        )
        self.assertEqual(policy.rag_ann_post_gpu_tail_ms_by_path_batch, {})
        self.assertEqual(
            policy.rag_ann_worker_collection_ms_by_path_batch, {}
        )
        self.assertEqual(
            policy.rag_ann_worker_send_per_output_ms_by_path_batch, {}
        )

    def test_ann_inline_prediction_bootstraps_from_worker_collect_and_send(self):
        policy = _PolicyHarness()
        policy.rag_ann_worker_collection_ms_by_path_batch = {
            "full_gpu": {2: deque([2.0] * 4, maxlen=64)}
        }
        policy.rag_ann_worker_send_per_output_ms_by_path_batch = {
            "full_gpu": {2: deque([0.5] * 4, maxlen=64)}
        }

        predicted_ms = policy._predict_ann_inline_collection_tail_ms(
            2, "full_gpu"
        )

        self.assertEqual(predicted_ms, 3.0)
        self.assertEqual(
            policy.rag_ann_inline_tail_prediction_last["source"],
            "worker_bootstrap",
        )

    def test_ann_mature_inline_cost_replaces_normal_scheduler_tail(self):
        policy = _PolicyHarness()
        policy.rag_ann_inline_collection_tail_ms_by_path_batch = {
            "full_gpu": {1: deque([2.0] * 4, maxlen=64)}
        }
        policy.rag_ann_worker_collection_ms_by_path_batch = {
            "full_gpu": {1: deque([80.0] * 4, maxlen=64)}
        }
        policy.rag_ann_worker_send_per_output_ms_by_path_batch = {
            "full_gpu": {1: deque([10.0] * 4, maxlen=64)}
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            "full_gpu": {1: deque([90.0] * 4, maxlen=64)}
        }

        predicted_ms = policy._predict_ann_inline_collection_tail_ms(
            1, "full_gpu"
        )

        self.assertEqual(predicted_ms, 2.0)
        self.assertEqual(
            policy.rag_ann_inline_tail_prediction_last["source"], "inline"
        )

    def test_failed_worker_collect_does_not_mature_bootstrap(self):
        policy = _PolicyHarness()
        for _ in range(4):
            task = {
                "queries": np.zeros((1, 8), dtype=np.float32),
                "ann_gpu_path": "full_gpu",
                "ann_collection_mode": "worker",
                "ann_collection_succeeded": False,
            }
            policy._record_ann_worker_collection_cost(task, 2.0)
        policy.rag_ann_worker_send_per_output_ms_by_path_batch = {
            "full_gpu": {1: deque([0.5] * 4, maxlen=64)}
        }

        self.assertIsNone(
            policy._predict_ann_inline_collection_tail_ms(1, "full_gpu")
        )
        self.assertEqual(
            policy.rag_ann_worker_collection_ms_by_path_batch, {}
        )
        failed_inline = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "ann_gpu_path": "full_gpu",
            "ann_collection_mode": "inline",
            "ann_collection_succeeded": False,
        }
        policy._record_ann_inline_collection_tail(
            failed_inline, 2.0, [0.5]
        )
        self.assertEqual(
            policy.rag_ann_inline_collection_tail_ms_by_path_batch, {}
        )

    def test_collect_timing_wraps_entire_safe_collection_call(self):
        policy = _PolicyHarness()
        task = {}
        output = object()
        with (
            patch.object(
                policy,
                "_collect_ann_search_outputs_safe",
                return_value=[output],
            ) as collect,
            patch(
                "sglang.srt.multiplex.multiplexing.time.perf_counter",
                side_effect=[10.0, 10.004],
            ),
        ):
            outputs, elapsed_ms = (
                policy._collect_ann_search_outputs_timed(task)
            )

        collect.assert_called_once_with(task)
        self.assertEqual(outputs, [output])
        self.assertAlmostEqual(elapsed_ms, 4.0)
        self.assertAlmostEqual(task["ann_collection_cpu_ms"], 4.0)

    def test_worker_output_queue_accepts_legacy_and_task_payloads(self):
        policy = _PolicyHarness()
        policy.ann_result_output_queue = Queue()
        legacy_output = object()
        task_output = object()
        task = {
            "queries": np.zeros((1, 8), dtype=np.float32),
            "ann_gpu_path": "full_gpu",
            "ann_collection_mode": "worker",
            "ann_collection_succeeded": True,
        }
        policy.ann_result_output_queue.put([legacy_output])
        policy.ann_result_output_queue.put((task, [task_output]))

        with patch.object(policy, "_send_ann_search_output") as send:
            policy._drain_ann_result_outputs()

        self.assertEqual(
            [call.args[0] for call in send.call_args_list],
            [legacy_output, task_output],
        )
        samples = (
            policy.rag_ann_worker_send_per_output_ms_by_path_batch[
                "full_gpu"
            ][1]
        )
        self.assertEqual(len(samples), 1)

    def test_inline_collect_send_records_one_merged_batch_once(self):
        policy = _PolicyHarness()
        task = {
            "queries": np.zeros((2, 8), dtype=np.float32),
            "ann_gpu_path": "full_gpu",
            "ann_collection_mode": "inline",
            "ann_collection_succeeded": True,
        }

        policy._record_ann_inline_collection_tail(
            task, 2.0, [0.5, 0.5]
        )
        policy._record_ann_inline_collection_tail(
            task, 20.0, [5.0, 5.0]
        )

        samples = policy.rag_ann_inline_collection_tail_ms_by_path_batch[
            "full_gpu"
        ][2]
        self.assertEqual(list(samples), [3.0])

    def test_ann_completion_prediction_is_exact_path_batch_p99(self):
        policy = _PolicyHarness()
        policy.rag_ann_completion_service_ms_by_path_batch = {
            "full_gpu": {
                1: deque([10.0, 11.0, 12.0], maxlen=64),
                2: deque([20.0, 21.0, 22.0, 23.0], maxlen=64),
            },
            "hybrid_gpu": {
                1: deque([80.0, 81.0, 82.0, 83.0], maxlen=64),
            },
        }
        policy.rag_ann_post_gpu_tail_ms_by_path_batch = {
            "full_gpu": {
                1: deque([2.0, 3.0, 4.0], maxlen=64),
                2: deque([5.0, 6.0, 7.0, 8.0], maxlen=64),
            },
            "hybrid_gpu": {
                1: deque([70.0, 71.0, 72.0, 73.0], maxlen=64),
            },
        }

        self.assertIsNone(
            policy._predict_ann_completion_service_ms(
                1, 8.0, "full_gpu"
            )
        )
        full_prediction = policy._predict_ann_completion_service_ms(
            2, 16.0, "full_gpu"
        )
        hybrid_prediction = policy._predict_ann_completion_service_ms(
            1, 10.0, "hybrid_gpu"
        )

        self.assertAlmostEqual(
            full_prediction,
            max(
                np.percentile([20.0, 21.0, 22.0, 23.0], 99),
                16.0 + np.percentile([5.0, 6.0, 7.0, 8.0], 99),
            ),
        )
        self.assertAlmostEqual(
            hybrid_prediction,
            max(
                np.percentile([80.0, 81.0, 82.0, 83.0], 99),
                10.0
                + np.percentile([70.0, 71.0, 72.0, 73.0], 99),
            ),
        )
        self.assertGreater(hybrid_prediction, full_prediction)

    def test_ann_completion_predictor_telemetry_reports_maturity(self):
        policy = _PolicyHarness()
        self._seed_ann_completion_prediction(
            policy,
            path="full_gpu",
            batch_size=2,
            service_ms=20.0,
            tail_ms=5.0,
        )
        prediction = policy._predict_ann_completion_service_ms(
            2, 16.0, "full_gpu"
        )
        inline_prediction = policy._predict_ann_inline_collection_tail_ms(
            2, "full_gpu"
        )

        telemetry = policy._get_rag_ann_runtime_state()[
            "ann_pre_switch_launch"
        ]["completion_predictor"]
        batch = telemetry["by_path_batch"]["full_gpu"]["2"]

        self.assertEqual(telemetry["min_samples"], 4)
        self.assertTrue(batch["mature"])
        self.assertEqual(batch["service_samples"], 4)
        self.assertEqual(batch["tail_samples"], 4)
        self.assertEqual(batch["inline_tail_samples"], 4)
        self.assertTrue(
            all(
                isinstance(key, str)
                for path in telemetry["by_path_batch"].values()
                for key in path
            )
        )
        self.assertEqual(
            telemetry["last_prediction"]["predicted_completion_ms"],
            prediction,
        )
        self.assertTrue(telemetry["last_prediction"]["mature"])
        self.assertEqual(
            telemetry["last_inline_tail_prediction"]["predicted_ms"],
            inline_prediction,
        )

    def test_runtime_telemetry_reports_frontend_decision_latency(self):
        policy = _PolicyHarness()
        policy.rag_frontend_decision_ms.extend([0.10, 0.20, 0.30])

        telemetry = policy._get_rag_ann_runtime_state()[
            "frontend_decision_latency"
        ]

        self.assertEqual(telemetry["samples"], 3)
        self.assertAlmostEqual(telemetry["mean_ms"], 0.20)
        self.assertGreater(telemetry["p95_ms"], telemetry["mean_ms"])

    def test_ann_gpu_time_cap_shrinks_effective_microbatch(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_batch_size = 8
        policy.rag_ann_max_microbatch_gpu_ms = 10
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([5.0] * 4, maxlen=64),
            2: deque([8.0] * 4, maxlen=64),
        }

        self.assertEqual(policy._effective_ann_microbatch_rows(), 2)

    def test_ann_microbatch_never_exceeds_hard_row_limit(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_batch_size = 2
        policy.rag_ann_max_microbatch_gpu_ms = 0
        pending = deque(
            {
                "queries": np.zeros((1, 4), dtype=np.float32),
                "enqueue_time": float(index),
            }
            for index in range(3)
        )

        batch = policy._pop_ann_microbatch(pending)

        self.assertEqual(batch["queries"].shape[0], 2)
        self.assertEqual(len(pending), 1)

    def test_ann_credit_blocks_excess_cuda_stream_queueing(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=5,
            allow_prefill=False,
        )
        inflight = [
            {
                "result": object(),
                "predicted_gpu_ms": 24.0,
                "queries": SimpleNamespace(shape=(1, 768)),
            }
        ]

        with (
            patch.object(policy, "_ann_result_gpu_pending", return_value=True),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            allowed, predicted_ms = policy._ann_credit_allows_launch(
                inflight, 1
            )

        self.assertFalse(allowed)
        self.assertEqual(predicted_ms, 10.0)
        self.assertEqual(policy.rag_ann_credit_holds, 1)

    def test_ann_credit_is_zero_while_partition_switch_is_pending(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=2,
            allow_prefill=True,
            prefill_wait_ms=0.0,
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            allowed, _ = policy._ann_credit_allows_launch([], 1)

        self.assertFalse(allowed)
        self.assertEqual(policy.rag_ann_credit_last_limit_ms, 0.0)

    def test_overdue_prefill_does_not_zero_credit_in_current_ann_partition(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_prefill_deadline_ms = 500
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=5,
            allow_prefill=True,
            prefill_wait_ms=1000.0,
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            credit_ms = policy._ann_launch_credit_limit_ms()

        self.assertEqual(credit_ms, 30.0)

    def test_ann_credit_downsizes_microbatch_to_use_remaining_budget(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=5,
            allow_prefill=False,
        )
        policy.rag_ann_gpu_service_ms_by_batch_size = {
            1: deque([9.0] * 4, maxlen=64),
            2: deque([16.0] * 4, maxlen=64),
        }
        pending = deque(
            {
                "queries": np.zeros((1, 4), dtype=np.float32),
                "enqueue_time": float(index),
            }
            for index in range(2)
        )
        inflight = [
            {
                "result": object(),
                "predicted_gpu_ms": 16.0,
                "queries": np.zeros((2, 4), dtype=np.float32),
            }
        ]

        with (
            patch.object(policy, "_ann_result_gpu_pending", return_value=True),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            rows, predicted_ms = policy._select_ann_credit_batch_rows(
                pending, inflight, 2
            )

        self.assertEqual(rows, 1)
        self.assertEqual(predicted_ms, 9.0)
        self.assertEqual(policy.rag_ann_credit_downsized_launches, 1)

    def test_ann_credit_allows_one_cold_start_batch_for_liveness(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_queued_gpu_ms = 5
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=5,
            allow_prefill=False,
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            allowed, predicted_ms = policy._ann_credit_allows_launch([], 1)

        self.assertTrue(allowed)
        self.assertEqual(predicted_ms, 10.0)
        self.assertEqual(policy.rag_ann_credit_forced_single_launches, 1)

    def test_ann_credit_primes_one_indivisible_exact_batch_two(self):
        policy = _PolicyHarness()
        policy.rag_ann_max_queued_gpu_ms = 10
        policy.rag_frontend_plan = SimpleNamespace(
            target_stream_idx=5,
            allow_prefill=False,
        )
        pending = deque(
            [
                {
                    "queries": np.zeros((2, 4), dtype=np.float32),
                    "enqueue_time": 0.0,
                }
            ]
        )

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            rows, predicted_ms = policy._select_ann_credit_batch_rows(
                pending,
                [],
                2,
            )

        self.assertEqual(rows, 2)
        self.assertEqual(predicted_ms, 20.0)
        self.assertEqual(policy.rag_ann_credit_forced_single_launches, 1)
        self.assertTrue(policy.rag_ann_credit_last_forced_atomic)

    def test_ann_priming_does_not_pollute_request_arrival_history(self):
        policy = _PolicyHarness()
        policy.rag_request_arrival_times = deque([1.0, 2.0], maxlen=64)

        policy._record_rag_ann_request_arrival(
            3.0,
            2,
            is_priming=True,
        )

        self.assertEqual(list(policy.rag_request_arrival_times), [1.0, 2.0])
        self.assertEqual(policy.rag_ann_priming_requests, 1)
        self.assertEqual(policy.rag_ann_priming_rows, 2)

        policy._record_rag_ann_request_arrival(
            4.0,
            2,
            is_priming=False,
        )
        self.assertEqual(
            list(policy.rag_request_arrival_times),
            [1.0, 2.0, 4.0, 4.0],
        )

    def test_fair_ann_launch_uses_credit_for_multi_grant(self):
        class _PendingResult:
            @staticmethod
            def poll():
                return False

            @staticmethod
            def gpu_poll():
                return False

        def run_launch(
            max_batches_per_tick,
            launch_calls=1,
            *,
            hot_index_mode=False,
        ):
            policy = _PolicyHarness()
            policy.rag_ann_schedule = "decode-gc"
            policy.rag_ann_frontend_arbitration = "fair"
            policy.rag_ann_max_batches_per_tick = max_batches_per_tick
            policy.rag_ann_max_batch_size = 1
            policy.rag_ann_max_microbatch_gpu_ms = 10
            policy.rag_ann_max_queued_gpu_ms = 30
            policy.rag_frontend_plan = SimpleNamespace(
                allow_ann=True,
                forced_ann=False,
                low_pressure_fast_path=False,
                reason="ann_decode_overlap",
                target_stream_idx=2,
            )
            policy.ann_search_pending = deque(
                {
                    "enqueue_time": 0.0,
                    "queries": np.zeros((1, 8), dtype=np.float32),
                    "request_id": str(idx),
                }
                for idx in range(4)
            )
            launches = []

            def search_async(*args, **kwargs):
                launches.append((args, kwargs))
                return _PendingResult()

            policy.ann_executor = SimpleNamespace(
                search_async=search_async,
                hot_index_mode=hot_index_mode,
            )
            ann_stream = SimpleNamespace(cuda_stream=123)
            with (
                patch.object(
                    policy, "_get_rag_ann_stream", return_value=ann_stream
                ),
                patch.object(
                    policy,
                    "_get_stream_group_idx_for_stream",
                    return_value=2,
                ),
                patch(
                    "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                    return_value=2,
                ),
            ):
                for _ in range(launch_calls):
                    policy._launch_ann_search_side_channel()
            policy._record_rag_ann_tick_grants()
            return policy, launches

        policy, launches = run_launch(0)
        self.assertEqual(len(launches), 3)
        self.assertEqual(len(policy.ann_search_inflight), 3)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(policy.rag_ann_multi_grant_ticks, 1)
        self.assertEqual(policy.rag_ann_grants_per_tick_histogram, {"3": 1})
        self.assertEqual(policy.rag_ann_credit_holds, 1)

        policy, launches = run_launch(1, launch_calls=2)
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(policy.ann_search_inflight), 1)
        self.assertEqual(len(policy.ann_search_pending), 3)
        self.assertEqual(policy.rag_ann_multi_grant_ticks, 0)
        self.assertEqual(policy.rag_ann_tick_limit_holds, 1)
        self.assertEqual(policy.rag_ann_grants_per_tick_histogram, {"1": 1})

        policy, launches = run_launch(
            0,
            hot_index_mode=True,
        )
        self.assertEqual(len(launches), 3)
        self.assertEqual(len(policy.ann_search_inflight), 3)
        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(policy.rag_ann_multi_grant_ticks, 1)
        self.assertEqual(
            policy.rag_ann_hybrid_pipeline_backlog_holds,
            0,
        )
        self.assertEqual(
            policy.rag_ann_hybrid_pipeline_second_slot_admissions,
            1,
        )

    def test_hybrid_ann_pipeline_opens_second_slot_for_full_backlog(self):
        policy = _PolicyHarness()
        policy.ann_executor = SimpleNamespace(hot_index_mode=True)
        policy.rag_ann_max_batch_size = 8
        policy.rag_ann_max_microbatch_gpu_ms = 0
        task = {"queries": np.zeros((1, 8), dtype=np.float32)}

        self.assertEqual(
            policy._hybrid_ann_pipeline_limit(deque([task] * 4), [], 8),
            1,
        )
        self.assertEqual(
            policy._hybrid_ann_pipeline_limit(deque([task] * 9), [], 8),
            2,
        )
        self.assertEqual(
            policy._hybrid_ann_pipeline_limit(
                deque([task] * 8), [{"result": object()}], 8
            ),
            2,
        )
        self.assertEqual(
            policy._hybrid_ann_pipeline_limit(deque([task] * 25), [], 8),
            4,
        )
        self.assertEqual(policy.rag_ann_hybrid_pipeline_backlog_holds, 1)
        self.assertEqual(
            policy.rag_ann_hybrid_pipeline_second_slot_admissions,
            3,
        )
        self.assertEqual(
            policy.rag_ann_hybrid_pipeline_window_histogram,
            {"1": 1, "2": 2, "4": 1},
        )

    def test_partition_switch_drain_stops_refilling_ann_window(self):
        policy = _PolicyHarness()
        policy.ann_executor = SimpleNamespace()
        policy.ann_search_pending = deque([{"enqueue_time": 0.0}])
        policy.rag_frontend_plan = SimpleNamespace(allow_ann=True)
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_partition_switch_deferred = True

        policy._launch_ann_search_side_channel()

        self.assertEqual(len(policy.ann_search_pending), 1)
        self.assertEqual(policy.ann_search_inflight, [])
        self.assertEqual(policy.rag_ann_partition_drain_holds, 1)

    def test_decode_gc_blocks_prefill_until_decode_only_ann_finishes(self):
        policy = _PolicyHarness()
        policy.rag_ann_schedule = "decode-gc"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_frontend_plan = SimpleNamespace(
            allow_prefill=True,
            target_stream_idx=2,
        )

        with (
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=True),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertTrue(policy._rag_ann_blocks_prefill())

        with (
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=True),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=3,
            ),
        ):
            self.assertFalse(policy._rag_ann_blocks_prefill())

        policy.rag_ann_partition_switch_deferred = True
        with (
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            self.assertTrue(policy._rag_ann_blocks_prefill())


    def test_dynamic_only_uses_work_conserving_round_robin(self):
        policy = _PolicyHarness()
        policy.rag_policy_mode = "dynamic-only"
        policy.rag_ann_frontend_arbitration = "fair"
        policy.rag_ann_schedule = "frontend-gc"
        policy.rag_dynamic_rr_next = "ann"
        policy.rag_ann_lease_frontend_sm = -1
        policy.rag_prefill_lease_frontend_sm = -1
        policy.ann_search_pending = deque([{"enqueue_time": 0.0}])
        policy.split_prefill_batch = _Batch(1)
        policy.waiting_queue = []
        policy.rag_prefill_handoff_active = False
        policy.rag_ann_consecutive_batches = 0
        policy.rag_prefill_consecutive_chunks = 0
        policy.rag_ann_deadline_ms = 50
        policy.rag_prefill_deadline_ms = 500
        policy.rag_ann_safety_margin_ms = 5
        policy.rag_ann_frontend_residency_until = 0.0
        policy.rag_frontend_plan_reason_counts = {}
        policy.rag_frontend_target_reason_counts = {}
        policy.rag_frontend_decision_counts = {}
        policy.rag_ann_budget_reject_events = 0
        policy.rag_ann_budget_rejected_requests = 0
        policy.rag_ann_slo_deferred_requests = 0

        with (
            patch.object(policy, "_prune_rag_ann_handoffs"),
            patch.object(policy, "_rag_ann_gpu_work_active", return_value=False),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=500.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=500.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                side_effect=AssertionError(
                    "Dynamic-Only must not consult decode slack"
                ),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=4,
            ),
        ):
            policy._plan_rag_frontend_actions()
            self.assertTrue(policy.rag_frontend_plan.allow_ann)
            self.assertFalse(policy.rag_frontend_plan.allow_prefill)
            self.assertEqual(policy.rag_frontend_plan.reason, "dynamic_rr_ann")

            policy.rag_frontend_ann_launched = True
            policy._finalize_rag_frontend_actions()
            policy._plan_rag_frontend_actions()

        self.assertFalse(policy.rag_frontend_plan.allow_ann)
        self.assertTrue(policy.rag_frontend_plan.allow_prefill)
        self.assertEqual(policy.rag_frontend_plan.reason, "dynamic_rr_prefill")


    def test_adaptive_low_pressure_ann_launches_early_without_coalescing(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 8.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertEqual(plan.reason, "ann_fast_path")
        self.assertTrue(plan.allow_ann)
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertTrue(plan.launch_ann_early)
        self.assertFalse(plan.immediate_handoff)
        self.assertEqual(plan.target_stream_idx, 4)
        self.assertEqual(policy.rag_lease_coalesce_waits, 0)
        self.assertEqual(policy.rag_lease_coalesce_work_holds, 0)
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)
        self.assertTrue(policy._rag_ann_should_launch_early())
        self.assertFalse(policy._rag_ann_should_launch_after_prefill())

    def test_adaptive_low_pressure_handoff_runs_prefill_immediately(self):
        policy = self._adaptive_policy(
            has_ann=False, has_prefill=True, handoff=True
        )

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": True,
                    "forced": False,
                    "reason": "decode_slack",
                    "predicted_ms": 8.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertEqual(plan.reason, "handoff_fast_path")
        self.assertFalse(plan.allow_ann)
        self.assertTrue(plan.allow_prefill)
        self.assertTrue(plan.low_pressure_fast_path)
        self.assertFalse(plan.launch_ann_early)
        self.assertTrue(plan.immediate_handoff)
        self.assertEqual(policy.rag_lease_coalesce_waits, 0)
        self.assertEqual(policy.rag_lease_coalesce_work_holds, 0)
        self.assertEqual(policy.rag_lease_coalesce_cost_holds, 0)

    def test_adaptive_decode_unsafe_ann_stays_after_prefill(self):
        policy = self._adaptive_policy(
            has_ann=True, has_prefill=False, handoff=False
        )
        policy.rag_ann_decode_batch_sizes = deque([32] * 8, maxlen=64)

        with (
            patch.object(
                policy, "_prune_rag_ann_handoffs", return_value=False
            ),
            patch.object(
                policy, "_rag_ann_gpu_work_active", return_value=False
            ),
            patch.object(policy, "_oldest_ann_wait_ms", return_value=0.0),
            patch.object(policy, "_oldest_prefill_wait_ms", return_value=0.0),
            patch.object(
                policy,
                "_evaluate_rag_ann_decode_budget",
                return_value={
                    "allowed": False,
                    "forced": False,
                    "reason": "decode_budget",
                    "predicted_ms": 18.0,
                    "slo_ms": 20.0,
                },
            ),
            patch(
                "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
                return_value=5,
            ),
        ):
            policy._plan_rag_frontend_actions()

        plan = policy.rag_frontend_plan
        self.assertEqual(plan.reason, "decode_budget")
        self.assertFalse(plan.allow_ann)
        self.assertFalse(plan.low_pressure_fast_path)
        self.assertFalse(plan.launch_ann_early)
        self.assertEqual(plan.target_stream_idx, 5)
        self.assertFalse(policy._rag_ann_should_launch_early())
        self.assertTrue(policy._rag_ann_should_launch_after_prefill())

class TestLegacyCalibratedOverlapCoverage(unittest.TestCase):
    @staticmethod
    def _prefill_policy(*, ann_active, decode_active):
        policy = object.__new__(_PolicyHarness)
        policy.sm_counts = [(84, 24)]
        policy.split_prefill_batch = SimpleNamespace(
            is_empty=lambda: False,
            extend_lens=[4],
            prefix_lens=[8],
        )
        policy.model_config = SimpleNamespace(num_hidden_layers=4)
        policy._rag_ann_gpu_work_active = lambda: ann_active
        policy._decode_batch_active = lambda: decode_active
        stage = Mock()
        stage.predict_prefill_ms.return_value = 40.0
        corrections = Mock()
        policy.rag_legacy_calibrated_bundle = SimpleNamespace(
            identity="test", stage=stage, corrections=corrections
        )
        policy.rag_legacy_calibrated_prediction_counts = {}
        return policy, corrections

    def test_prefill_decode_overlap_uses_partition_baseline(self):
        policy, corrections = self._prefill_policy(
            ann_active=False, decode_active=True
        )

        layer_ms = policy._predict_rag_prefill_layer_ms(0, 4)

        self.assertEqual(layer_ms, 10.0)
        corrections.query.assert_not_called()
        self.assertEqual(
            policy.rag_legacy_calibrated_last_prefill["overlap"], "decode"
        )
        self.assertIsNone(
            policy.rag_legacy_calibrated_last_prefill["correction_overlap"]
        )

    def test_three_way_prefill_uses_ann_pairwise_guard(self):
        policy, corrections = self._prefill_policy(
            ann_active=True, decode_active=True
        )
        corrections.query.return_value = (1.1, 1.5)

        layer_ms = policy._predict_rag_prefill_layer_ms(0, 4)

        self.assertEqual(layer_ms, 15.0)
        self.assertEqual(corrections.query.call_args.args[:3], ("prefill", 0, "ann"))
        self.assertEqual(
            policy.rag_legacy_calibrated_last_prefill["overlap"], "ann_decode"
        )

    def test_three_way_ann_uses_maximum_pairwise_guard(self):
        policy = object.__new__(_PolicyHarness)
        policy.sm_counts = [(84, 24)]
        policy.split_prefill_batch = SimpleNamespace(is_empty=lambda: False)
        policy._decode_batch_active = lambda: True
        policy._preview_rag_legacy_ann_work = lambda pending, batch: (10, 20, 1, 1)
        stage = Mock()
        stage.predict_ann_ms.return_value = 10.0
        corrections = Mock()
        corrections.query.side_effect = [(1.1, 1.2), (1.1, 1.4)]
        policy.rag_legacy_calibrated_bundle = SimpleNamespace(
            identity="test", stage=stage, corrections=corrections
        )
        policy.rag_legacy_calibrated_prediction_counts = {}

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=0,
        ):
            predicted = policy._predict_rag_legacy_calibrated_ann_ms([], 2)

        self.assertEqual(predicted, 14.0)
        self.assertEqual(
            [call.args[2] for call in corrections.query.call_args_list],
            ["prefill", "decode"],
        )
        self.assertEqual(
            policy.rag_legacy_calibrated_last_ann["correction_overlap"],
            "max(prefill,decode)",
        )

    def test_ann_prediction_uses_target_lease_partition(self):
        policy = object.__new__(_PolicyHarness)
        policy.sm_counts = [
            (108, 0), (84, 24), (72, 36), (60, 48), (48, 60), (0, 108)
        ]
        policy.split_prefill_batch = None
        policy._decode_batch_active = lambda: True
        policy._preview_rag_legacy_ann_work = (
            lambda pending, batch: (973, 707469, 707469, 1995, 500512385961)
        )
        stage = Mock()
        stage.predict_ann_ms.return_value = 4.0
        corrections = Mock()
        corrections.query.return_value = (1.1, 1.25)
        policy.rag_legacy_calibrated_bundle = SimpleNamespace(
            identity="test", stage=stage, corrections=corrections
        )
        policy.rag_legacy_calibrated_prediction_counts = {}

        with patch(
            "sglang.srt.multiplex.multiplexing.get_current_stream_idx",
            return_value=5,
        ):
            predicted = policy._predict_rag_legacy_calibrated_ann_ms(
                [], 1, target_stream_idx=4
            )

        self.assertEqual(predicted, 5.0)
        self.assertEqual(stage.predict_ann_ms.call_args.args[0], 60)
        self.assertEqual(corrections.query.call_args.args[:3], ("ann", 4, "decode"))
        self.assertEqual(policy.rag_legacy_calibrated_last_ann["partition"], 4)


class TestRagTokenCache(unittest.IsolatedAsyncioTestCase):
    async def test_exact_prompt_cache_avoids_second_tokenization(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = object()
        manager.server_args = SimpleNamespace(rag_doc_token_cache_size=2)
        manager.rag_doc_token_cache = OrderedDict()
        manager.rag_doc_token_cache_hits = 0
        manager.rag_doc_token_cache_misses = 0
        calls = 0

        async def tokenize(prompt, _):
            nonlocal calls
            calls += 1
            return [len(prompt), 7], None

        manager._tokenize_texts = tokenize
        first, first_hit = await manager.tokenize_rag_prompt_cached("doc prompt")
        second, second_hit = await manager.tokenize_rag_prompt_cached(
            "doc prompt"
        )

        self.assertEqual(first, second)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
