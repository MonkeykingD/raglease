# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import zmq
from sgl_kernel import spatial
from torch.distributed import barrier

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    AnnSearchReqInput,
    AnnSearchReqOutput,
    RagPreparationCancel,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    MultiTokenizerRegisterReq,
    MultiTokenizerWrapper,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    RAG_PREFILL_REQUEST_LABEL,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import init_embedding_cache
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    SchedulerMetricsMixin,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import DPBalanceMeta, validate_input_length
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.multiplex.multiplexing import SchedulerMultiplexMixin
from sglang.srt.multiplex.pdmux_context import (
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
)
from sglang.srt.multiplex.synthetic_ann import SyntheticANN
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_event,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    disable_request_logging,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_zmq_socket,
    is_cpu,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    pyspy_dump_schedulers,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))

_is_cpu = is_cpu()


@dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput]
    pp_hidden_states_proxy_tensors: Optional[torch.Tensor]
    next_token_ids: Optional[List[int]]
    extend_input_len_per_req: List[int]
    extend_logprob_start_len_per_req: List[int]
    bid: int
    can_run_cuda_graph: bool


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    bid: int


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = server_args.kv_events_config and tp_rank == 0
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.page_size = server_args.page_size
        self.shared_hbm_pool = None
        self.shared_hbm_controller = None
        if server_args.rag_shared_hbm_mode != "off":
            # Reserve the physical pool before Faiss loads and before the model
            # worker profiles KV memory. V1 validation guarantees this
            # scheduler and the model worker share one process.
            torch.cuda.set_device(gpu_id)
            from sglang.srt.mem_cache.shared_hbm import (
                initialize_global_shared_hbm_pool,
            )

            self.shared_hbm_pool = initialize_global_shared_hbm_pool(
                server_args, gpu_id
            )

        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init inter-process communication
        context = zmq.Context(2)
        self.idle_sleeper = None
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        if self.current_scheduler_metrics_enabled():
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

        if self.enable_pdmux:
            # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
            self.pdmux_config = load_pdmux_config(server_args.pdmux_config_path)
            initialize_stream_groups(gpu_id, self.pdmux_config)
            self.stream_groups = get_stream_groups()
            self.sm_counts = get_sm_counts()
            self.real_sm_group_num = len(self.stream_groups)
            logger.info(
                f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
            )

            # Initialize synthetic ANN if configured
            self.synthetic_ann = None
            self.synthetic_ann_schedule = server_args.synthetic_ann_schedule
            self.synthetic_ann_start_delay_s = server_args.synthetic_ann_start_delay_s
            self.synthetic_ann_start_time = time.monotonic()
            self.synthetic_ann_stream = None
            if server_args.synthetic_ann_buffer_gb > 0:
                self.synthetic_ann = SyntheticANN(
                    buffer_gb=server_args.synthetic_ann_buffer_gb,
                    gpu_id=gpu_id,
                    mode=server_args.synthetic_ann_mode,
                    qps=server_args.synthetic_ann_qps,
                    intensity=server_args.synthetic_ann_intensity,
                    iters_per_kernel=server_args.synthetic_ann_iters_per_kernel,
                )
                self.synthetic_ann.allocate()
                if self.synthetic_ann_schedule == "background-primary":
                    self.synthetic_ann_stream = torch.cuda.Stream(device=gpu_id)
                elif self.synthetic_ann_schedule == "background-frontend":
                    ann_stream_idx = 1 if len(self.stream_groups) > 2 else 0
                    self.synthetic_ann_stream = self.stream_groups[ann_stream_idx][0]
                logger.info(
                    f"PD-Multiplexing synthetic ANN enabled: "
                    f"buffer={server_args.synthetic_ann_buffer_gb}GB, "
                    f"mode={server_args.synthetic_ann_mode}, "
                    f"qps={server_args.synthetic_ann_qps}, "
                    f"schedule={self.synthetic_ann_schedule}, "
                    f"start_delay_s={self.synthetic_ann_start_delay_s}"
                )
        else:
            self.synthetic_ann = None
            self.synthetic_ann_schedule = "prefill"
            self.synthetic_ann_start_delay_s = 0.0
            self.synthetic_ann_start_time = 0.0
            self.synthetic_ann_stream = None

        # Initialize real ANN executor (Faiss) BEFORE KV cache allocation
        self.ann_executor = None
        self.rag_ann_debug = server_args.rag_ann_debug
        self.rag_ann_schedule = server_args.rag_ann_schedule
        self.rag_policy_mode = server_args.rag_policy_mode
        self.rag_ann_max_inflight = max(1, server_args.rag_ann_max_inflight)
        self.rag_ann_max_batches_per_tick = max(
            0, server_args.rag_ann_max_batches_per_tick
        )
        self.rag_ann_max_batch_size = max(1, server_args.rag_ann_max_batch_size)
        self.rag_ann_max_queued_gpu_ms = max(
            0.0, server_args.rag_ann_max_queued_gpu_ms
        )
        self.rag_ann_max_microbatch_gpu_ms = max(
            0.0, server_args.rag_ann_max_microbatch_gpu_ms
        )
        self.rag_ann_service_time_percentile = min(
            100.0, max(0.001, server_args.rag_ann_service_time_percentile)
        )
        self.rag_ann_service_time_min_samples = max(
            1, server_args.rag_ann_service_time_min_samples
        )
        self.rag_ann_max_batch_wait_ms = max(
            0.0, server_args.rag_ann_max_batch_wait_ms
        )
        self.rag_ann_batch_wait_min_pending = max(
            1, server_args.rag_ann_batch_wait_min_pending
        )
        self.rag_ann_stream_epoch_ms = max(0.0, server_args.rag_ann_stream_epoch_ms)
        self.rag_ann_epoch_stream = None
        self.rag_ann_epoch_stream_idx = None
        self.rag_ann_epoch_until = 0.0
        self.rag_ann_frontend_residency_until = 0.0
        self.rag_ann_frontend_residency_releases = 0
        self.rag_ann_partition_switch_deferred = False
        self.rag_ann_partition_drain_holds = 0
        self.rag_ann_partition_drain_events = 0
        self.rag_ann_partition_completion_polls = 0
        self.rag_ann_partition_inline_completions = 0
        self.rag_ann_partition_drain_predicted_ms = deque(maxlen=4096)
        self.rag_ann_launch_site = server_args.rag_ann_launch_site
        self.rag_ann_fast_path_pressure_threshold = max(
            0.0, server_args.rag_ann_fast_path_pressure_threshold
        )
        self.rag_ann_fast_path_max_work = max(
            1, server_args.rag_ann_fast_path_max_work
        )
        self.rag_ann_fast_path_decode_batch_p95_threshold = max(
            0.001,
            server_args.rag_ann_fast_path_decode_batch_p95_threshold,
        )
        self.rag_prefill_low_load_batch_p95_threshold = max(
            0.001,
            server_args.rag_prefill_low_load_batch_p95_threshold,
        )
        self.rag_prefill_low_load_arrival_rps_threshold = (
            server_args.rag_prefill_low_load_arrival_rps_threshold
        )
        self.rag_ann_fast_path_decode_queue_p95_ms = max(
            0.0, server_args.rag_ann_fast_path_decode_queue_p95_ms
        )
        self.rag_ann_decode_batch_bucket_size = max(
            1, server_args.rag_ann_decode_batch_bucket_size
        )
        self.rag_ann_decode_only_wait_ms = server_args.rag_ann_decode_only_wait_ms
        self.rag_ann_max_inflight_during_decode = max(
            0, server_args.rag_ann_max_inflight_during_decode
        )
        self.rag_ann_work_conserving_decode_overlap = (
            server_args.rag_ann_work_conserving_decode_overlap
        )
        self.rag_ann_pre_switch_launch = server_args.rag_ann_pre_switch_launch
        self.rag_ann_pre_switch_checks = 0
        self.rag_ann_pre_switch_deadline_risk_triggers = 0
        self.rag_ann_pre_switch_gpu_p99_credit_downsized_triggers = 0
        self.rag_ann_pre_switch_gpu_p99_credit_downsized_launches = 0
        self.rag_ann_pre_switch_robust_gain_threshold_holds = 0
        self.rag_ann_pre_switch_robust_gain_threshold_bypasses = 0
        self.rag_ann_pre_switch_batch_net_gain_unavailable = 0
        self.rag_ann_pre_switch_cross_class_debt_candidates = 0
        self.rag_ann_pre_switch_cross_class_debt_bypasses = 0
        self.rag_ann_pre_switch_cross_class_debt_would_admit = 0
        self.rag_ann_pre_switch_cross_class_debt_holds = 0
        self.rag_ann_pre_switch_eligible_robust_gain_ms = deque(maxlen=4096)
        self.rag_ann_pre_switch_eligible_raw_robust_gain_ms = deque(
            maxlen=4096
        )
        self.rag_ann_pre_switch_eligible_batch_net_gain_ms = deque(
            maxlen=4096
        )
        self.rag_ann_pre_switch_eligible_cross_class_net_debt_ms = deque(
            maxlen=4096
        )
        self.rag_ann_pre_switch_launch_batches = 0
        self.rag_ann_pre_switch_launch_rows = 0
        self.rag_ann_pre_switch_switch_deferrals = 0
        self.rag_ann_pre_switch_completion_polls = 0
        self.rag_ann_pre_switch_inline_completions = 0
        self.rag_ann_pre_switch_no_launch_reason_counts = {}
        self.rag_ann_pre_switch_trigger_wait_ms = deque(maxlen=4096)
        self.rag_ann_pre_switch_predicted_switch_ms = deque(maxlen=4096)
        self.rag_ann_pre_switch_predicted_robust_switch_ms = deque(
            maxlen=4096
        )
        self.rag_ann_pre_switch_peak_projected_gpu_ms = 0.0
        self.rag_ann_pre_switch_safety_violations = 0
        self.rag_ann_itl_slo_ms = server_args.rag_ann_itl_slo_ms
        self.rag_ann_decode_step_slo_ms = server_args.rag_ann_decode_step_slo_ms
        self.rag_partition_decode_slo_ms = tuple(
            float(value)
            for value in (server_args.rag_partition_decode_slo_ms or ())
        )
        self.rag_partition_decode_slo_batch_thresholds = tuple(
            float(value)
            for value in (
                server_args.rag_partition_decode_slo_batch_thresholds or ()
            )
        )
        if self.rag_partition_decode_slo_ms:
            if self.rag_partition_decode_slo_batch_thresholds:
                expected_thresholds = len(self.rag_partition_decode_slo_ms) - 1
                if (
                    len(self.rag_partition_decode_slo_batch_thresholds)
                    != expected_thresholds
                ):
                    raise ValueError(
                        "rag_partition_decode_slo_batch_thresholds must provide "
                        f"{expected_thresholds} values for "
                        f"{len(self.rag_partition_decode_slo_ms)} SLO bands"
                    )
            else:
                mixed_group_count = getattr(self, "real_sm_group_num", 0) - 2
                if mixed_group_count <= 0:
                    raise ValueError(
                        "rag_partition_decode_slo_ms requires PD-Mux mixed groups"
                    )
                if len(self.rag_partition_decode_slo_ms) != mixed_group_count:
                    raise ValueError(
                        "rag_partition_decode_slo_ms must provide exactly one "
                        "value per mixed PD-Mux group when no load thresholds "
                        f"are set: expected {mixed_group_count}, got "
                        f"{len(self.rag_partition_decode_slo_ms)}"
                    )
        self.rag_partition_decode_slo_hysteresis_bs = max(
            0.0, server_args.rag_partition_decode_slo_hysteresis_bs
        )
        self.rag_partition_decode_slo_min_residency_ms = max(
            0.0, server_args.rag_partition_decode_slo_min_residency_ms
        )
        self.rag_partition_decode_slo_initial_calibration_ms = max(
            0.0,
            server_args.rag_partition_decode_slo_initial_calibration_ms,
        )
        self.rag_partition_decode_slo_ewma_alpha = (
            server_args.rag_partition_decode_slo_ewma_alpha
        )
        self.rag_partition_decode_slo_band_idx = 0
        self.rag_partition_decode_slo_batch_ewma = None
        decode_slo_now = time.perf_counter()
        self.rag_partition_decode_slo_last_change_ts = decode_slo_now
        self.rag_partition_decode_slo_next_update_ts = 0.0
        self.rag_partition_decode_slo_calibration_started = False
        self.rag_partition_decode_slo_control_updates = 0
        self.rag_partition_decode_slo_band_changes = 0
        self.rag_partition_slo_floor_enabled = (
            server_args.rag_partition_slo_floor_enabled
        )
        self.rag_ann_safety_margin_ms = max(
            0.0, server_args.rag_ann_safety_margin_ms
        )
        self.rag_prefill_decode_guard_percentile = min(
            100.0,
            max(0.001, server_args.rag_prefill_decode_guard_percentile),
        )
        self.rag_ann_urgency_start_ratio = min(
            1.0, max(0.0, server_args.rag_ann_urgency_start_ratio)
        )
        self.rag_ann_max_decode_slowdown_ratio = max(
            1.0, server_args.rag_ann_max_decode_slowdown_ratio
        )
        self.rag_ann_max_defer_ms = max(0.0, server_args.rag_ann_max_defer_ms)
        self.rag_ann_frontend_arbitration = (
            server_args.rag_ann_frontend_arbitration
        )
        self.rag_ann_deadline_ms = max(0.001, server_args.rag_ann_deadline_ms)
        self.rag_prefill_deadline_ms = max(
            0.001, server_args.rag_prefill_deadline_ms
        )
        self.rag_prefill_low_load_reference_tokens = max(
            1, server_args.rag_prefill_low_load_reference_tokens
        )
        self.rag_e2e_slo_ms = server_args.rag_e2e_slo_ms
        self.rag_ann_overlap_pressure_threshold = max(
            0.0, server_args.rag_ann_overlap_pressure_threshold
        )
        self.rag_ann_max_consecutive_batches = max(
            1, server_args.rag_ann_max_consecutive_batches
        )
        self.rag_prefill_max_consecutive_chunks = max(
            1, server_args.rag_prefill_max_consecutive_chunks
        )
        self.rag_ann_handoff_timeout_ms = max(
            0.0, server_args.rag_ann_handoff_timeout_ms
        )
        self.rag_ann_lease_frontend_sm = server_args.rag_ann_lease_frontend_sm
        self.rag_prefill_lease_frontend_sm = (
            server_args.rag_prefill_lease_frontend_sm
        )
        self.rag_e2_fixed_partition_idx = (
            server_args.rag_e2_fixed_partition_idx
        )
        self.rag_partition_hysteresis_bs = max(
            0, server_args.rag_partition_hysteresis_bs
        )
        self.rag_partition_min_residency_ms = max(
            0.0, server_args.rag_partition_min_residency_ms
        )
        self.rag_partition_safe_mixed_hold = (
            server_args.rag_partition_safe_mixed_hold
        )
        self.rag_partition_safe_mixed_hold_ratio = (
            server_args.rag_partition_safe_mixed_hold_ratio
        )
        self.rag_partition_safe_mixed_min_samples = (
            server_args.rag_partition_safe_mixed_min_samples
        )
        self.rag_partition_safe_mixed_max_hold_ms = max(
            0.0, server_args.rag_partition_safe_mixed_max_hold_ms
        )
        self.rag_lease_coalesce_min_work = max(
            1, server_args.rag_lease_coalesce_min_work
        )
        self.rag_lease_coalesce_max_wait_ms = max(
            0.0, server_args.rag_lease_coalesce_max_wait_ms
        )
        self.rag_lease_switch_cost_ms = max(
            0.0, server_args.rag_lease_switch_cost_ms
        )
        self.rag_lease_switch_cost_percentile = min(
            100.0, max(0.001, server_args.rag_lease_switch_cost_percentile)
        )
        self.rag_lease_round_trip_switch_cost = (
            server_args.rag_lease_round_trip_switch_cost
        )
        self.rag_lease_min_gain_ratio = max(
            1.0, server_args.rag_lease_min_gain_ratio
        )
        self.rag_lease_ann_service_estimate_ms = max(
            0.001, server_args.rag_lease_ann_service_estimate_ms
        )
        self.rag_lease_prefill_service_estimate_ms = max(
            0.001, server_args.rag_lease_prefill_service_estimate_ms
        )
        self.rag_prefill_dynamic_lease = server_args.rag_prefill_dynamic_lease
        self.rag_prefill_two_level_bundles = (
            server_args.rag_prefill_two_level_bundles
        )
        self.rag_prefill_tail_aware_admission = (
            server_args.rag_prefill_tail_aware_admission
        )
        self.rag_prefill_exclusive_sprint = (
            server_args.rag_prefill_exclusive_sprint
        )
        self.rag_prefill_lease_min_layers = max(
            1, server_args.rag_prefill_lease_min_layers
        )
        self.rag_prefill_lease_max_layers = max(
            self.rag_prefill_lease_min_layers,
            server_args.rag_prefill_lease_max_layers,
        )
        self.rag_prefill_lease_decode_ratio = max(
            0.001, server_args.rag_prefill_lease_decode_ratio
        )
        self.rag_ann_enable_controlled_overlap = (
            server_args.rag_ann_enable_controlled_overlap
        )
        self.rag_ann_serialize_with_prefill = (
            server_args.rag_ann_serialize_with_prefill
        )
        if self.rag_ann_serialize_with_prefill:
            self.rag_ann_frontend_arbitration = "strict"
        self.rag_ann_decode_iteration_ms = deque(maxlen=64)
        self.rag_ann_decode_batch_sizes = deque(maxlen=64)
        self.rag_request_arrival_times = deque(maxlen=64)
        self.rag_ann_decode_gpu_ms = deque(maxlen=64)
        self.rag_ann_decode_partition_ms = {}
        self.rag_ann_decode_partition_batch_ms = {}
        self.rag_ann_decode_partition_batch_overlap_ms = {}
        self.rag_ann_decode_gpu_partition_ms = {}
        self.rag_ann_decode_gpu_partition_batch_ms = {}
        self.rag_ann_decode_gpu_partition_batch_overlap_ms = {}
        self.rag_ann_decode_queue_delay_ms = deque(maxlen=64)
        self.rag_ann_last_decode_completion_ts = None
        self.rag_ann_last_decode_stream_idx = None
        self.rag_ann_slo_deferred_requests = 0
        self.rag_ann_budget_reject_events = 0
        self.rag_ann_budget_rejected_requests = 0
        self.rag_ann_urgency_admitted_requests = 0
        self.rag_ann_urgency_admission_wait_ms = deque(maxlen=4096)
        self.rag_ann_forced_release_events = 0
        self.rag_ann_budget_defer_ms = deque(maxlen=4096)
        self.rag_ann_forced_release_to_launch_ms = deque(maxlen=4096)
        self.rag_ann_stream_group_launch_counts = {}
        self.rag_prefill_stream_group_run_counts = {}
        self.rag_ann_launch_reason_counts = {}
        self.rag_ann_batch_size_histogram = {}
        self.rag_ann_inflight_coalesce_events = 0
        self.rag_ann_inflight_coalesce_holds = 0
        self.rag_ann_queue_wait_histogram = {}
        self.rag_ann_queue_wait_ms = deque(maxlen=4096)
        self.rag_ann_gpu_latency_ms = deque(maxlen=4096)
        self.rag_ann_gpu_service_ms_by_batch_size = {}
        self.rag_ann_credit_holds = 0
        self.rag_ann_credit_downsized_launches = 0
        self.rag_ann_credit_forced_single_launches = 0
        self.rag_ann_credit_peak_queued_ms = 0.0
        self.rag_ann_credit_peak_admitted_queued_ms = 0.0
        self.rag_ann_credit_peak_forced_atomic_ms = 0.0
        self.rag_ann_credit_last_limit_ms = self.rag_ann_max_queued_gpu_ms
        self.rag_ann_credit_last_queued_ms = 0.0
        self.rag_ann_credit_last_batch_ms = 0.0
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
        self.rag_ann_cpu_queue_wait_ms = deque(maxlen=4096)
        self.rag_ann_cpu_search_latency_ms = deque(maxlen=4096)
        self.rag_ann_coarse_quantize_latency_ms = deque(maxlen=4096)
        self.rag_ann_route_latency_ms = deque(maxlen=4096)
        self.rag_ann_merge_latency_ms = deque(maxlen=4096)
        self.rag_ann_hot_probe_ratio = deque(maxlen=4096)
        self.rag_decode_batch_size_histogram = {}
        self.rag_frontend_decision_counts = {}
        self.rag_frontend_plan_reason_counts = {}
        self.rag_frontend_target_reason_counts = {}
        self.rag_ann_consecutive_batches = 0
        self.rag_prefill_consecutive_chunks = 0
        self.rag_frontend_plan = None
        self.rag_prefill_run_allowed = True
        self.rag_dynamic_rr_next = "ann"
        self.rag_ann_handoffs = {}
        self.rag_preparation_contracts = {}
        self.rag_request_start_times = {}
        self.rag_ann_handoff_completed = 0
        self.rag_ann_handoff_expired = 0
        self.rag_ann_handoff_latency_ms = deque(maxlen=4096)
        self.rag_completed_handoff_timing = {}
        self.rag_ann_handoff_prefill_activations = 0
        self.rag_prefill_handoff_active = False
        self.rag_prefill_handoff_batch_started_ts = None
        self.rag_prefill_handoff_gpu_events = []
        self.rag_prefill_bundle_inflight_event = None
        self.rag_prefill_bundle_inflight_meta = None
        self.rag_prefill_layer_ms = deque(maxlen=256)
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
        self.rag_prefill_last_bundle_completion_ts = None
        self.rag_prefill_queue_wait_ms = deque(maxlen=4096)
        self.rag_prefill_gpu_latency_ms = deque(maxlen=4096)
        self.rag_prefill_wall_latency_ms = deque(maxlen=4096)
        self.rag_prefill_to_first_decode_ms = deque(maxlen=4096)
        self.rag_decode_host_submit_ms = deque(maxlen=4096)
        self.rag_decode_launch_interval_ms = deque(maxlen=4096)
        self.rag_prefill_host_submit_ms = deque(maxlen=4096)
        self.rag_scheduler_tick_wall_ms = deque(maxlen=4096)
        self.rag_frontend_decision_ms = deque(maxlen=4096)
        self.rag_last_decode_submit_ts = None
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
        self.rag_prefill_sprint_chunks = 0
        self.rag_prefill_exclusive_sprint_available = False
        self.rag_prefill_exclusive_sprint_chunks = 0
        self.rag_prefill_exclusive_sprint_block_counts = {}
        self.rag_decode_update_deferred_for_prefill_sprint = False
        self.rag_prefill_exclusive_decode_update_skips = 0
        self.rag_prefill_exclusive_deferred_merges = 0
        self.rag_prefill_workload_fast_path_rejects = 0
        self.rag_ann_prefill_transition_holds = 0
        self.rag_prefill_merge_pending_batch = None
        self.rag_prefill_cohort_holds = 0
        self.rag_prefill_cohort_releases = 0
        self.rag_prefill_cohort_size_histogram = {}
        self.rag_dependency_prefill_admissions = 0
        self.rag_dependency_prefill_burst_chunks = 0
        self.pdmux_stream_group_switches = 0
        self.rag_partition_transition_intent = None
        self.rag_partition_transition_latency_ms = deque(maxlen=4096)
        self.rag_partition_transition_pair_latency_ms = {}
        self.rag_partition_transition_abandoned = 0
        self.rag_partition_hysteresis_holds = 0
        self.rag_partition_residency_holds = 0
        self.rag_partition_safe_mixed_holds = 0
        self.rag_partition_safe_mixed_cost_admissions = 0
        self.rag_partition_safe_mixed_cost_rejections = 0
        self.rag_partition_safe_mixed_cost_prediction_misses = 0
        self.rag_partition_safe_mixed_last_cost_decision = {}
        self.rag_prefill_fast_path_resident_cost_checks = 0
        self.rag_partition_decode_floor_sm = 0
        self.rag_partition_slo_promotions = 0
        self.rag_partition_slo_demotions = 0
        self.rag_partition_floor_safe_observations = 0
        self.rag_partition_decode_only_fallback_required = False
        self.rag_partition_decode_only_fallback_activations = 0
        self.rag_decode_recovery_pending = False
        self.rag_decode_recovery_requests = 0
        self.rag_decode_recovery_windows = 0
        self.rag_partition_switch_reason_counts = {}
        self.rag_partition_switch_latency_ms = deque(maxlen=4096)
        self.rag_partition_switch_pair_latency_ms = {}
        self.rag_lease_profile_status = {"loaded": False}
        self.rag_lease_profile_priors = {}
        self.rag_last_frontend_activity_ts = time.perf_counter()
        self.rag_handoff_decode_release_pending = False
        self.rag_handoff_decode_release_windows = 0
        self.rag_lease_switch_benefit_ms = deque(maxlen=4096)
        self.rag_lease_horizon_bundle_histogram = {}
        self.rag_lease_horizon_layer_histogram = {}
        self.rag_lease_last_horizon_bundles = 1
        self.rag_prefill_partition_selection_counts = {}
        self.rag_prefill_partition_selection_last = {}
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
        self.rag_prefill_candidate_net_gain_last = {}
        self.rag_prefill_floor_safe_admissions = 0
        self.rag_lease_coalesce_waits = 0
        self.rag_lease_coalesce_work_holds = 0
        self.rag_lease_coalesce_cost_holds = 0
        self.rag_lease_decode_restore_transitions = 0
        self.rag_dependency_lease_holds = 0
        self.rag_partition_last_switch_ts = 0.0
        self.rag_ann_stream = None
        if server_args.enable_rag_ann:
            from sglang.srt.multiplex.ann_executor import AnnExecutor

            self.ann_executor = AnnExecutor(
                index_path=server_args.rag_index_path,
                gpu_id=gpu_id,
                nprobe=server_args.rag_index_nprobe,
                topk=server_args.rag_index_topk,
                debug=server_args.rag_ann_debug,
                dedicated_streams=self.rag_ann_schedule
                in {"frontend-gc", "decode-gc"},
                hot_index_path=server_args.rag_hot_index_path,
                hot_index_map_path=server_args.rag_hot_index_map_path,
                hot_index_shard_id=server_args.rag_hot_index_shard_id,
                cpu_search_threads=server_args.rag_cpu_retrieval_threads,
                cpu_cold_max_batch_queries=server_args.rag_ann_cold_max_batch_queries,
                cpu_search_max_batch_queries=(
                    server_args.rag_cpu_retrieval_max_batch_queries
                ),
                cluster_heat_decay=(
                    server_args.rag_shared_hbm_cluster_heat_decay
                ),
                cluster_miss_boost=(
                    server_args.rag_shared_hbm_cluster_miss_boost
                ),
                cluster_promotion_hysteresis=(
                    server_args.rag_shared_hbm_cluster_promotion_hysteresis
                ),
                cluster_min_residency_ms=(
                    server_args.rag_shared_hbm_min_residency_ms
                ),
            )
            if self.rag_ann_schedule == "primary":
                ann_init_stream = torch.cuda.Stream(device=gpu_id)
            elif (
                self.enable_pdmux
                and getattr(self, "real_sm_group_num", 0) > 2
            ):
                # Load under the same GreenContext lane that ANN will use.
                lane_idx = 1 if self.rag_ann_schedule == "decode-gc" else 0
                ann_init_stream = self.stream_groups[1][lane_idx]
            else:
                raise ValueError(
                    f"rag_ann_schedule={self.rag_ann_schedule} requires "
                    "--enable-pdmux and an available GreenContext stream"
                )
                ann_init_stream = torch.cuda.Stream(device=gpu_id)
            self.rag_ann_stream = ann_init_stream
            self.ann_executor.load(ann_init_stream)
            logger.info(
                f"RAG ANN enabled: index={server_args.rag_index_path}, "
                f"nprobe={server_args.rag_index_nprobe}, "
                f"topk={server_args.rag_index_topk}, "
                f"hot_index={server_args.rag_hot_index_path or 'disabled'}, "
                f"hot_shard={server_args.rag_hot_index_shard_id}, "
                f"schedule={self.rag_ann_schedule}, "
                f"stream=0x{ann_init_stream.cuda_stream:x}"
            )

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_standalone():
            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            self.draft_worker = StandaloneWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_lookahead():
            from sglang.srt.speculative.lookahead_worker import LOOKAHEADWorker

            self.draft_worker = LOOKAHEADWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if global_server_args_dict["max_micro_batch_size"] is None:
            global_server_args_dict["max_micro_batch_size"] = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()
        if self.shared_hbm_pool is not None:
            from sglang.srt.mem_cache.shared_hbm import SharedHBMController

            kv_publication_stream = torch.cuda.current_stream(gpu_id)
            kv_consumer_streams = [kv_publication_stream]
            if self.enable_pdmux:
                for stream_group in self.stream_groups:
                    kv_consumer_streams.extend(stream_group)
            self.shared_hbm_controller = SharedHBMController(
                pool=self.shared_hbm_pool,
                token_allocator=self.token_to_kv_pool_allocator,
                ann_executor=self.ann_executor,
                lookahead_ms=server_args.rag_shared_hbm_lookahead_ms,
                min_residency_ms=(
                    server_args.rag_shared_hbm_min_residency_ms
                ),
                kv_reserve_extents=(
                    server_args.rag_shared_hbm_kv_reserve_extents
                ),
                rerank_prefetch_batches=(
                    server_args.rag_shared_hbm_rerank_prefetch_batches
                ),
                ivf_reload_min_misses=(
                    server_args.rag_shared_hbm_cluster_reload_min_misses
                ),
                ivf_reload_max_lists=(
                    server_args.rag_shared_hbm_cluster_reload_max_lists
                ),
                kv_publication_stream=kv_publication_stream,
                kv_consumer_streams=kv_consumer_streams,
                async_native_map=(
                    server_args.rag_shared_hbm_async_map
                ),
                map_stage_extents=(
                    server_args.rag_shared_hbm_map_stage_extents
                ),
            )

        if server_args.rag_lease_profile_path:
            from sglang.srt.multiplex.lease_profile import (
                apply_lease_performance_profile,
                load_lease_performance_profile,
            )

            lease_profile = load_lease_performance_profile(
                server_args.rag_lease_profile_path,
                server_args.rag_lease_profile_request_rate,
            )
            status = apply_lease_performance_profile(self, lease_profile)
            logger.info(
                "Loaded RAGLease offline performance priors: "
                f"rate={status['selected_request_rate']}, "
                f"seeded={status['seeded']}"
            )

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0
        self.sessions: Dict[str, Session] = {}
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        # Enable preemption for priority scheduling.
        self.try_preemption = self.enable_priority_scheduling

        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profiler()

        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)
        self.init_dp_balance(dp_balance_meta)

        if self.enable_kv_cache_events:
            self.init_kv_events(server_args.kv_events_config)

        # Init disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()

        if get_bool_env_var("SGLANG_GC_LOG"):
            configure_gc_logger()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (MultiTokenizerRegisterReq, self.register_multi_tokenizer),
                (GetLoadReqInput, self.get_load),
                (AnnSearchReqInput, self.handle_ann_search_request),
                (RagPreparationCancel, self.handle_rag_preparation_cancel),
            ]
        )

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            if self.is_hybrid:
                ChunkCacheClass = SWAChunkCache
            else:
                ChunkCacheClass = ChunkCache
            self.tree_cache = ChunkCacheClass(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        else:
            if os.environ.get("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE") == "1":
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                self.tree_cache = RadixCacheCpp(
                    disable=False,
                    use_hicache=self.enable_hierarchical_cache,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool=self.token_to_kv_pool_allocator,
                    tp_cache_group=self.tp_cpu_group,
                    page_size=self.page_size,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                )
            elif self.enable_hierarchical_cache:
                self.tree_cache = HiRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tp_cache_group=(
                        self.attn_tp_cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_cpu_group
                    ),
                    page_size=self.page_size,
                    eviction_policy=server_args.radix_eviction_policy,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    hicache_io_backend=server_args.hicache_io_backend,
                    hicache_mem_layout=server_args.hicache_mem_layout,
                    enable_metrics=self.enable_metrics,
                    hicache_storage_backend=server_args.hicache_storage_backend,
                    hicache_storage_prefetch_policy=server_args.hicache_storage_prefetch_policy,
                    model_name=server_args.served_model_name,
                    storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
                )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid:
                assert (
                    self.server_args.disaggregation_mode == "null"
                ), "Hybrid mode does not support disaggregation yet"
                self.tree_cache = SWARadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    sliding_window_size=self.sliding_window_size,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                )
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                    eviction_policy=server_args.radix_eviction_policy,
                )
            else:
                self.tree_cache = RadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                    eviction_policy=server_args.radix_eviction_policy,
                )

        self.decode_mem_cache_buf_multiplier = (
            1
            if self.spec_algorithm.is_none()
            else (
                server_args.speculative_num_draft_tokens
                + (
                    (server_args.speculative_eagle_topk or 1)
                    * (server_args.speculative_num_steps or 1)
                )
            )
        )

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "100"))
        init_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_disaggregation(self):
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_lookahead()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                prefill_pp_size=self.server_args.disaggregation_prefill_pp,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_lookahead()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                decode_tp_size=self.server_args.disaggregation_decode_tp,
                decode_dp_size=self.server_args.disaggregation_decode_dp,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

    def init_moe_config(self):
        if hasattr(self.model_config.hf_config, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # The no-GC RAG baseline uses the ordinary scheduler and a primary
            # CUDA stream for Faiss. Keep ANN asynchronous so unrelated LLM
            # batches can be launched while retrieval is in flight.
            self._poll_ann_search_side_channel()
            if self.rag_ann_launch_site == "early":
                self._launch_ann_search_side_channel()

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                for req in batch.reqs:
                    trace_event("schedule", req.rid)

            if batch:
                result = self.run_batch(batch)

            # Match PD-Mux's launch ordering: submit the LLM batch first, then
            # enqueue independent ANN work before waiting on the LLM result.
            if self.rag_ann_launch_site in ("after-prefill", "adaptive"):
                self._launch_ann_search_side_channel()

            if batch:
                self.process_batch_result(batch, result)
            elif not self._ann_side_channel_has_work():
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            self._poll_ann_search_side_channel()
            if self.rag_ann_launch_site == "early":
                self._launch_ann_search_side_channel()

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                for req in batch.reqs:
                    trace_event("schedule", req.rid)

            if batch:
                batch.launch_done = threading.Event()
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None, batch.launch_done)

            if self.rag_ann_launch_site in ("after-prefill", "adaptive"):
                self._launch_ann_search_side_channel()

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                # NOTE: we should use current launched batch's launch_done event Instead of the last batch's
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )
            elif batch is None and not self._ann_side_channel_has_work():
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_pp(self):
        """A non-overlap scheduler loop for pipeline parallelism."""
        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        bids = [None] * self.pp_size
        pp_outputs: Optional[PPProxyTensors] = None
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch:
                        next_token_ids, bids[mb_id] = (
                            result.next_token_ids,
                            result.bid,
                        )
                        if self.cur_batch.return_logprob:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                    "extend_input_len_per_req": result.extend_input_len_per_req,
                                    "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                                }
                                | (
                                    {
                                        f"logits_output.{k}": v
                                        for k, v in result.logits_output.__dict__.items()
                                    }
                                    if result.logits_output is not None
                                    else {}
                                )
                            )
                        else:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                }
                            )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    logits_output_args = {
                        k[len("logits_output.") :]: v
                        for k, v in next_pp_outputs.tensors.items()
                        if k.startswith("logits_output.")
                    }
                    if len(logits_output_args) > 0:
                        logits_output = LogitsProcessorOutput(**logits_output_args)
                    else:
                        logits_output = None
                    output_result = GenerationBatchResult(
                        logits_output=logits_output,
                        pp_hidden_states_proxy_tensors=None,
                        next_token_ids=next_pp_outputs["next_token_ids"],
                        extend_input_len_per_req=next_pp_outputs.tensors.get(
                            "extend_input_len_per_req", None
                        ),
                        extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
                            "extend_logprob_start_len_per_req", None
                        ),
                        bid=bids[next_mb_id],
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result(mbs[next_mb_id], output_result)
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                # (not last rank)
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        bids[mb_id] = result.bid
                    # carry the outputs to the next stage
                    # send the outputs from the last round to let the next stage worker run post processing
                    if pp_outputs:
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                    # send out reqs to the next stage
                    dp_offset = self.attn_dp_rank * self.attn_tp_size
                    if self.attn_tp_rank == 0:
                        point_to_point_pyobj(
                            recv_reqs,
                            self.pp_rank * self.tp_size + dp_offset,
                            self.world_group.device_group,
                            self.pp_rank * self.tp_size + dp_offset,
                            (self.pp_rank + 1) * self.tp_size + dp_offset,
                        )

                    # send out proxy tensors to the next stage
                    if self.cur_batch:
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

    def recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.device_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                trace_set_proc_propagate_context(req.rid, req.trace_context)
                trace_slice_start("", req.rid, anonymous=True)

        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            if is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None
                or not self.running_batch.is_empty()
                or len(self.offload_tags) > 0
            ):
                self.return_health_check_ct += 1
                continue

            # If it is a MultiTokenizerWrapper, unwrap it and handle the inner request.
            if isinstance(recv_req, MultiTokenizerWrapper):
                worker_id = recv_req.worker_id
                recv_req = recv_req.obj
                output = self._request_dispatcher(recv_req)
                if output is not None:
                    output = MultiTokenizerWrapper(worker_id, output)
                    self.send_to_tokenizer.send_pyobj(output)
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if isinstance(output, RpcReqOutput):
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        preparation_contract = getattr(self, "rag_preparation_contracts", {}).pop(str(recv_req.rid), None)
        self._record_rag_input_preparation_completion(preparation_contract, time.perf_counter())
        rag_ann_handoff_ready = self._complete_rag_ann_handoff(recv_req.rid)
        self.maybe_update_dp_balance_data(recv_req)

        # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=recv_req.custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                data_parallel_rank=recv_req.data_parallel_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
            )
            req.tokenizer = self.tokenizer

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"boostrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.set_finish_with_abort(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        req.rag_ann_handoff_ready = rag_ann_handoff_ready
        handoff_timing = getattr(
            self, "rag_completed_handoff_timing", {}
        ).pop(str(recv_req.rid), None)
        if handoff_timing is not None:
            if len(handoff_timing) == 3:
                (
                    req.rag_e2e_started_at,
                    req.rag_ann_completed_at,
                    req.rag_scheduler_arrival_at,
                ) = handoff_timing
            else:
                (
                    req.rag_ann_completed_at,
                    req.rag_scheduler_arrival_at,
                ) = handoff_timing
                req.rag_e2e_started_at = req.rag_ann_completed_at

        if preparation_contract is not None:
            # Resource handoff expiry must not reset the request's E2E clock.
            # This clock starts at scheduler ANN admission, not client arrival.
            req.rag_e2e_started_at = preparation_contract["started_at"]
            req.rag_ann_completed_at = preparation_contract.get("ann_completed_at")
            req.rag_scheduler_arrival_at = time.perf_counter()

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            # For prefill-only requests with logprob_start_len == -1, set logprob_start_len beyond input sequence
            # to skip input logprob computation entirely
            if req.is_prefill_only:
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # TODO: For text generation, evaluate setting logprob_start_len to len(req.origin_input_ids) as well
                req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if not req.is_prefill_only and req.logprob_start_len >= len(
            req.origin_input_ids
        ):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
            req.grammar = value

            if not cache_hit:
                req.grammar_key = key
                add_to_grammar_queue = True
            else:
                if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                    error_msg = f"Invalid grammar request with cache hit: {key=}"
                    req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            req.queue_time_start = time.perf_counter()
            self.grammar_queue.append(req)
        else:
            shared_hbm_controller = getattr(
                self, "shared_hbm_controller", None
            )
            if shared_hbm_controller is not None and (
                rag_ann_handoff_ready
                or recv_req.label == RAG_PREFILL_REQUEST_LABEL
            ):
                shared_hbm_controller.on_rag_prefill_ready(
                    self, len(req.origin_input_ids)
                )
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.perf_counter()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req)
        else:
            self._set_or_validate_priority(req)
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            trace_slice_end("process req", req.rid, auto_next_anon=True)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache)
            if req.last_node.backuped:
                # only to initiate the prefetch if the last node is backuped
                # otherwise, the allocated GPU memory must be locked for integrity
                last_hash = req.last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]
                self.tree_cache.prefetch_from_storage(
                    req.rid, req.last_host_node, new_input_tokens, last_hash
                )

    def _extend_requests_to_queue(self, reqs: List[Req], is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_bootstrap_queue.extend(
                reqs, self.model_config.num_key_value_heads
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # If this is a decode server, we put the request to the decode pending prealloc queue
            self.disagg_decode_prealloc_queue.extend(reqs, is_retracted)
        else:
            for req in reqs:
                self._set_or_validate_priority(req)
                if not self._abort_on_queued_limit(req):
                    self.waiting_queue.append(req)

    def _set_or_validate_priority(self, req: Req):
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif not self.enable_priority_scheduling and req.priority is not None:
            abort_req = AbortReq(
                req.rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
            )
            self.send_to_tokenizer.send_pyobj(abort_req)

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].queue_time_start,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_pyobj(
            AbortReq(
                req_to_abort.rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            )
        )
        return req_to_abort.rid == recv_req.rid

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            priority=recv_req.priority,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def self_check_during_idle(self):
        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()

    def check_memory(self):
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                _,
                _,
                full_available_size,
                full_evictable_size,
                swa_available_size,
                swa_evictable_size,
            ) = self._get_swa_token_info()
            memory_leak = full_num_used != 0 or swa_num_used != 0
            token_msg = (
                f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
            )
        else:
            _, _, available_size, evictable_size = self._get_token_info()
            protected_size = self.tree_cache.protected_size()
            mapped_capacity = (
                self.token_to_kv_pool_allocator.current_mapped_capacity()
            )
            memory_leak = (available_size + evictable_size) != (
                mapped_capacity - protected_size
            )
            token_msg = (
                f"{self.max_total_num_tokens=}, {mapped_capacity=}, "
                f"{available_size=}, {evictable_size=}, {protected_size=}\n"
            )

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise ValueError(msg)

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled()
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self):
        if self.is_hybrid and isinstance(self.tree_cache, SWARadixCache):
            self.tree_cache.sanity_check()

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        mapped_capacity = (
            self.token_to_kv_pool_allocator.current_mapped_capacity()
        )
        num_used = mapped_capacity - (available_size + evictable_size)
        token_usage = num_used / max(1, mapped_capacity)
        return num_used, token_usage, available_size, evictable_size

    def _get_swa_token_info(self):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()
        if self.chunked_req:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            # chunked request keeps its rid but will get a new req_pool_idx
            if self.tp_worker.worker.model_runner.is_hybrid_gdn:
                self.req_to_token_pool.free(
                    self.chunked_req.req_pool_idx, free_mamba_cache=False
                )
            else:
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        need_dp_attn_preparation = require_mlp_sync(self.server_args)

        if need_dp_attn_preparation and not self.spec_algorithm.is_none():
            # In speculative decoding, prefill batches and decode batches cannot be processed in the same DP attention group.
            # We prepare idle batches in advance to skip preparing decode batches when there are prefill batches in the group.
            new_batch = self.prepare_mlp_sync_batch(new_batch)
            need_dp_attn_preparation = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention
        if need_dp_attn_preparation:
            self.maybe_handle_dp_balance_data()
            ret = self.prepare_mlp_sync_batch(ret)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = global_server_args_dict["max_micro_batch_size"] - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self, *, admitted_prefill_contract=None) -> Optional[ScheduleBatch]:
        admitted_ids = None
        if admitted_prefill_contract is not None:
            # Materialize ONE admitted FCFS cohort, not another scheduling
            # search. This path is used only after the joint HBM commit.
            from dataclasses import replace
            capture = getattr(self, '_capture_rag_cached_prefill_budget', None)
            budget = capture() if capture is not None else None
            if budget is None:
                return None
            layer, shapes = admitted_prefill_contract
            if layer != 0 or not shapes:
                raise ValueError('New prefill requires a nonempty layer-zero contract')
            admitted_ids = tuple(shape[0] for shape in shapes)
            observed = tuple((s.request_id,s.prompt_tokens,s.context_tokens,s.extend_tokens)
                             for s in budget.shapes[:len(shapes)])
            if observed != shapes:
                return None
            # A full latch may reflect the PREVIOUS KV allocation. Clear it
            # only after current physical capacity admits this exact prefix.
            current = replace(budget,batch_is_full=False).select()
            if current.request_ids[:len(admitted_ids)] != admitted_ids:
                return None
            self.running_batch.batch_is_full = False
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue and admitted_ids is None:
            self.move_ready_grammar_requests()

        if self.try_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked request has just been released.
        # In PP case, a chunked req can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked request to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and not self.chunked_req
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        if admitted_ids is None:
            self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            lora_set = set([req.lora_id for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        candidates = (self.waiting_queue if admitted_ids is None
                      else self.waiting_queue[:len(admitted_ids)])
        for req in candidates:

            if self.enable_lora and not self.tp_worker.can_run_lora_batch(
                lora_set
                | set([req.lora_id for req in adder.can_run_list])
                | set([req.lora_id])
            ):
                self.running_batch.batch_is_full = True
                break

            running_bs = len(self.running_batch.reqs) - len(adder.preempt_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if not self.try_preemption:
                    break
                if not adder.preempt_to_schedule(req, self.server_args):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if admitted_ids is not None and tuple(str(req.rid) for req in can_run_list) != admitted_ids:
            raise RuntimeError('Actual prefill admission diverged from validated joint cohort')
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()
                req.add_latency(RequestStage.PREFILL_WAITING)

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]
        if adder.preempt_list:
            self._extend_requests_to_queue(adder.preempt_list)

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.current_scheduler_metrics_enabled():
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "KV cache pool is full. Retract requests. "
                f"#retracted_reqs: {num_retracted_reqs}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )

            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
            self.total_retracted_reqs += num_retracted_reqs
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Run forward
        if self.is_generation:
            if self.enable_pdmux and batch.forward_mode.is_split_prefill():
                (
                    logits_output,
                    next_token_ids,
                    can_run_cuda_graph,
                    model_worker_batch,
                ) = self.tp_worker.forward_batch_split_prefill(batch)
                bid = model_worker_batch.bid
            elif self.spec_algorithm.is_none():
                decode_clock = (
                    getattr(self, "rag_decode_run_batch_observations", None)
                    if batch.forward_mode.is_decode()
                    else None
                )
                if decode_clock is not None:
                    prepare_started = time.perf_counter()
                model_worker_batch = batch.get_model_worker_batch()
                if decode_clock is not None:
                    forward_started = time.perf_counter()

                if self.pp_group.is_last_rank:
                    logits_output, next_token_ids, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_generation(model_worker_batch)
                    )
                else:
                    pp_hidden_states_proxy_tensors, _, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_generation(model_worker_batch)
                    )
                if decode_clock is not None:
                    forward_returned = time.perf_counter()
                    decode_clock.append({
                        "bid": model_worker_batch.bid,
                        "prepare_started": prepare_started,
                        "forward_started": forward_started,
                        "forward_returned": forward_returned,
                        "prepare_ms": (forward_started - prepare_started) * 1000.0,
                        "forward_host_ms": (forward_returned - forward_started) * 1000.0,
                    })
                bid = model_worker_batch.bid
            else:
                (
                    logits_output,
                    next_token_ids,
                    bid,
                    num_accepted_tokens,
                    can_run_cuda_graph,
                ) = self.draft_worker.forward_batch_speculative_generation(batch)
                bs = batch.batch_size()
                self.spec_num_total_accepted_tokens += num_accepted_tokens + bs
                self.spec_num_total_forward_ct += bs
                self.num_generated_tokens += num_accepted_tokens

            if self.pp_group.is_last_rank:
                batch.output_ids = next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob or self.spec_algorithm.is_eagle():
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            else:
                extend_input_len_per_req = None
            if batch.return_logprob:
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output if self.pp_group.is_last_rank else None,
                pp_hidden_states_proxy_tensors=(
                    pp_hidden_states_proxy_tensors
                    if not self.pp_group.is_last_rank
                    else None
                ),
                next_token_ids=next_token_ids if self.pp_group.is_last_rank else None,
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=bid,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(
                embeddings=embeddings, bid=model_worker_batch.bid
            )
        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
            for req in batch.reqs:
                trace_slice(
                    "decode loop",
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )

        elif batch.forward_mode.is_extend() or batch.forward_mode.is_split_prefill():
            self.process_batch_result_prefill(batch, result, launch_done)
            for req in batch.reqs:
                trace_slice(
                    "prefill",
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

        self.maybe_send_health_check_signal()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_pyobj(HealthCheckOutput())

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        return self.prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            spec_algorithm=self.spec_algorithm,
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
        )

    @staticmethod
    def prepare_mlp_sync_batch_raw(
        local_batch: ScheduleBatch,
        dp_size,
        attn_tp_size: int,
        tp_group,
        get_idle_batch,
        disable_cuda_graph: bool,
        spec_algorithm,
        speculative_num_draft_tokens,
        require_mlp_tp_gather: bool,
        disable_overlap_schedule: bool,
    ):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
            num_tokens_for_logprob = 0
        elif local_batch.forward_mode.is_decode():
            num_tokens = local_batch.batch_size()
            num_tokens_for_logprob = num_tokens
        else:
            num_tokens = local_batch.extend_num_tokens
            num_tokens_for_logprob = sum(
                [
                    # We should have at least 1 token for sample in every case.
                    max(extend_len - logprob_start_len, 1)
                    for logprob_start_len, extend_len in zip(
                        local_batch.extend_logprob_start_lens, local_batch.extend_lens
                    )
                ]
            )

        if local_batch is None or local_batch.forward_mode.is_decode_or_idle():
            can_cuda_graph = 1
        else:
            can_cuda_graph = 0

        is_extend_in_batch = (
            local_batch.forward_mode.is_extend() if local_batch else False
        )

        tbo_preparer = TboDPAttentionPreparer()
        if disable_overlap_schedule:
            group = tp_group.device_group
            device = tp_group.device
        else:
            group = tp_group.cpu_group
            device = "cpu"

        local_info = torch.tensor(
            [
                num_tokens,
                can_cuda_graph,
                num_tokens_for_logprob,
                is_extend_in_batch,
                *tbo_preparer.prepare_all_gather(
                    local_batch,
                ),
            ],
            dtype=torch.int64,
            device=device,
        )
        global_info = torch.empty(
            (dp_size, attn_tp_size, 6),
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.all_gather_into_tensor(
            global_info.flatten(),
            local_info,
            group=group,
        )
        global_num_tokens = global_info[:, 0, 0].tolist()
        can_cuda_graph = min(global_info[:, 0, 1].tolist())
        global_num_tokens_for_logprob = global_info[:, 0, 2].tolist()
        is_extend_in_batch = global_info[:, 0, 3].tolist()

        tbo_split_seq_index, global_forward_mode = tbo_preparer.compute_output(
            global_info[:, :, 4:6]
        )

        if local_batch is None and max(global_num_tokens) > 0:
            local_batch = get_idle_batch()

        if local_batch is not None:
            # TODO: handle the case when moe_dense_tp_size != 1
            if not require_mlp_tp_gather:
                local_batch.global_num_tokens = [num_tokens]
                local_batch.global_num_tokens_for_logprob = [num_tokens_for_logprob]
            else:
                local_batch.global_num_tokens = global_num_tokens
                local_batch.global_num_tokens_for_logprob = (
                    global_num_tokens_for_logprob
                )
            local_batch.is_extend_in_batch = any(is_extend_in_batch)
            local_batch.tbo_split_seq_index = tbo_split_seq_index
            local_batch.global_forward_mode = global_forward_mode

            # Check forward mode for cuda graph
            if not disable_cuda_graph:
                local_batch.can_run_dp_cuda_graph = can_cuda_graph

        return local_batch

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""

        num_ready_reqs = 0
        num_timeout_reqs = 0
        for req in self.grammar_queue:
            try:
                if req.finished():  # It is aborted by AbortReq
                    num_ready_reqs += 1
                    continue
                req.grammar = req.grammar.result(timeout=0.03)
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(
                        f"Invalid grammar request: {req.grammar_key=}"
                    )
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # NOTE(lianmin): this timeout is the waiting time of the above line. It is
                # not the waiting time from it enters the grammar queue.
                if req.grammar_wait_ct > GRAMMAR_TIMEOUT / 0.03:
                    num_timeout_reqs = 1
                break

        if self.server_args.enable_dp_attention:
            tp_size = self.attn_tp_size
            tp_group = self.attn_tp_cpu_group
        else:
            tp_size = self.tp_size
            tp_group = self.tp_cpu_group

        if tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor([num_ready_reqs, num_timeout_reqs], dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
            num_ready_reqs_max, num_timeout_reqs_max = tensor.tolist()

            for i in range(num_ready_reqs, num_ready_reqs_max):
                req = self.grammar_queue[i]
                if req.finished():  # It is aborted by AbortReq
                    continue
                req.grammar = req.grammar.result()
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(
                        f"Invalid grammar request: {req.grammar_key=}"
                    )
        else:
            num_ready_reqs_max = num_ready_reqs
            num_timeout_reqs_max = num_timeout_reqs

        for i in range(num_ready_reqs, num_ready_reqs + num_timeout_reqs_max):
            req = self.grammar_queue[i]
            req.grammar.cancel()
            error_msg = f"Grammar preprocessing timed out for {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
            self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
        num_ready_reqs = num_ready_reqs_max + num_timeout_reqs_max

        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        if not disable_request_logging():
            # Print batch size and memory pool info to check whether there are de-sync issues.
            if self.is_hybrid:
                (
                    _,
                    _,
                    _,
                    _,
                    full_available_size,
                    full_evictable_size,
                    swa_available_size,
                    swa_evictable_size,
                ) = self._get_swa_token_info()
                info_msg = (
                    f"{full_available_size=}, "
                    f"{full_evictable_size=}, "
                    f"{swa_available_size=}, "
                    f"{swa_evictable_size=}, "
                )
            else:
                _, _, available_size, evictable_size = self._get_token_info()
                info_msg = f"{available_size=}, " f"{evictable_size=}, "
            logger.error(
                f"{self.cur_batch.batch_size()=}, "
                f"{self.cur_batch.reqs=}, "
                f"{info_msg}"
            )

        pyspy_dump_schedulers()
        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if (
            len(self.waiting_queue) == 0
            and self.running_batch.is_empty()
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        ):
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cum_spec_accept_length = 0
            self.cum_spec_accept_count = 0
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def get_load(self, recv_req: GetLoadReqInput = None) -> GetLoadReqOutput:
        # TODO(lsyin): use dynamically maintained num_waiting_tokens

        if self.is_hybrid:
            num_tokens_full = (
                self.full_tokens_per_layer
                - self.token_to_kv_pool_allocator.full_available_size()
                - self.tree_cache.full_evictable_size()
            )
            num_tokens_swa = (
                self.swa_tokens_per_layer
                - self.token_to_kv_pool_allocator.swa_available_size()
                - self.tree_cache.swa_evictable_size()
            )
            num_tokens = max(num_tokens_full, num_tokens_swa)
        else:
            mapped_capacity = (
                self.token_to_kv_pool_allocator.current_mapped_capacity()
            )
            num_tokens = (
                mapped_capacity
                - self.token_to_kv_pool_allocator.available_size()
                - self.tree_cache.evictable_size()
            )

        # Tokens in waiting queue, bootstrap queue, prealloc queue
        num_tokens += sum(len(req.origin_input_ids) for req in self.waiting_queue)
        num_waiting_reqs = len(self.waiting_queue)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            num_tokens += sum(
                len(req.origin_input_ids)
                for req in self.disagg_prefill_bootstrap_queue.queue
            )
            num_waiting_reqs += len(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            num_tokens += sum(
                len(req.req.origin_input_ids)
                for req in self.disagg_decode_prealloc_queue.queue
            )
            num_waiting_reqs += len(self.disagg_decode_prealloc_queue.queue)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=len(self.running_batch.reqs) + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_tokens=num_tokens,
        )

    def handle_rag_preparation_cancel(self, recv_req: RagPreparationCancel):
        rid = str(recv_req.request_id)
        getattr(self, "rag_preparation_contracts", {}).pop(rid, None)
        getattr(self, "rag_ann_handoffs", {}).pop(rid, None)

    def handle_ann_search_request(
        self, recv_req: AnnSearchReqInput
    ) -> Optional[AnnSearchReqOutput]:
        if self.ann_executor is None:
            return AnnSearchReqOutput(
                success=False,
                request_id=recv_req.request_id,
                message="ANN executor is not initialized. Start server with --enable-rag-ann.",
            )

        try:
            import numpy as np

            if recv_req.queries is not None:
                queries = np.asarray(recv_req.queries, dtype=np.float32)
            elif recv_req.query is not None:
                queries = np.asarray([recv_req.query], dtype=np.float32)
            else:
                return AnnSearchReqOutput(
                    success=False,
                    request_id=recv_req.request_id,
                    message="ANN search requires `query` or `queries`.",
                )

            if queries.ndim != 2:
                return AnnSearchReqOutput(
                    success=False,
                    request_id=recv_req.request_id,
                    message=f"ANN queries must be 2-D, got shape={queries.shape}.",
                )
            if queries.shape[0] > self.rag_ann_max_batch_size:
                return AnnSearchReqOutput(
                    success=False,
                    request_id=recv_req.request_id,
                    message=(
                        f"ANN request contains {queries.shape[0]} query vectors, "
                        f"exceeding the hard micro-batch limit "
                        f"{self.rag_ann_max_batch_size}. Split the request or "
                        "increase --rag-ann-max-batch-size."
                    ),
                )

            if not hasattr(self, "ann_search_pending"):
                self.ann_search_pending = deque()
                self.ann_search_inflight = []

            torch.cuda.nvtx.range_push("ann_search_enqueue")
            try:
                enqueue_time = time.perf_counter()
                is_priming = bool(recv_req.is_priming)
                query_rows = int(queries.shape[0])
                self._record_rag_ann_request_arrival(
                    enqueue_time,
                    query_rows,
                    request_id=recv_req.request_id,
                    is_priming=is_priming,
                )
                contiguous_queries = np.ascontiguousarray(queries)
                if recv_req.preparation_follows and recv_req.request_id is not None:
                    if not hasattr(self, "rag_preparation_contracts"):
                        self.rag_preparation_contracts = {}
                    self.rag_preparation_contracts[str(recv_req.request_id)] = {
                        "stage": "ann_pending",
                        "preparation_follows": recv_req.preparation_follows,
                        "generation_follows": recv_req.generation_follows,
                        "started_at": enqueue_time,
                        "query_rows": query_rows,
                        "generation_token_limit": recv_req.generation_token_limit,
                        "generation_fixed_tokens": recv_req.generation_fixed_tokens,
                    }
                self.ann_search_pending.append(
                    {
                        "queries": contiguous_queries,
                        "route_future": (
                            self.ann_executor.prepare_hot_search_async(
                                contiguous_queries
                            )
                        ),
                        "topk": recv_req.topk,
                        "enqueue_time": enqueue_time,
                        "request_id": recv_req.request_id,
                        "is_priming": is_priming,
                        "preparation_follows": recv_req.preparation_follows,
                        "generation_follows": recv_req.generation_follows,
                        "generation_token_limit": recv_req.generation_token_limit,
                        "generation_fixed_tokens": recv_req.generation_fixed_tokens,
                    }
                )
            finally:
                torch.cuda.nvtx.range_pop()
            return None
        except Exception as e:
            logger.exception("ANN search request failed.")
            getattr(self, "rag_preparation_contracts", {}).pop(str(recv_req.request_id), None)
            return AnnSearchReqOutput(
                success=False, message=str(e), request_id=recv_req.request_id
            )

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        if self.enable_pdmux:
            ret["pdmux_runtime"] = {
                "sm_counts": [list(pair) for pair in self.sm_counts],
                "config": {
                    "sm_group_num": self.pdmux_config.sm_group_num,
                    "manual_divisions": self.pdmux_config.manual_divisions,
                    "split_forward_token_budget": (
                        self.pdmux_config.split_forward_token_budget
                    ),
                    "decode_bs_divisor": self.pdmux_config.decode_bs_divisor,
                },
            }
        if self.enable_pdmux or self.ann_executor is not None:
            ret["rag_ann_runtime"] = self._get_rag_ann_runtime_state()
        ret["memory_usage"] = {
            "weight": round(
                self.tp_worker.worker.model_runner.weight_load_mem_usage, 2
            ),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "mapped_token_capacity": int(
                self.token_to_kv_pool_allocator.current_mapped_capacity()
            ),
        }

        ret["memory_usage"]["graph"] = round(
            self.tp_worker.worker.model_runner.graph_mem_usage, 2
        )

        if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
            ret["avg_spec_accept_length"] = (
                self.cum_spec_accept_length / self.cum_spec_accept_count
            )
        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )
        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break
        if if_success:
            if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
                avg_spec_accept_length = (
                    self.cum_spec_accept_length / self.cum_spec_accept_count
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.cum_spec_accept_length = self.cum_spec_accept_count = 0
            for k, v in server_args_dict.items():
                global_server_args_dict[k] = v
            logger.info(f"Global server args updated! {global_server_args_dict=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=global_server_args_dict,
        )

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            func(recv_req.parameters)
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        preparations = getattr(self, "rag_preparation_contracts", {})
        for rid in tuple(preparations):
            if recv_req.abort_all or rid.startswith(recv_req.rid):
                preparations.pop(rid, None)
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_pyobj(AbortReq(req.rid))
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.tree_cache.cache_finished_req(req)

            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            # Abort method 2: call `set_finish_with_abort`
            # The request will still run one prefill forward pass.
            # In this case, we change the input_ids to be only one token to make this prefill cheap.
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for i, req in enumerate(self.disagg_prefill_bootstrap_queue.queue):
                logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for i, req in enumerate(self.disagg_prefill_inflight_queue):
                logger.debug(f"Abort inflight queue request. {req.rid=}")
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for i, decode_req in enumerate(self.disagg_decode_prealloc_queue.queue):
                logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    if hasattr(decode_req.kv_receiver, "abort"):
                        decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for i, decode_req in enumerate(self.disagg_decode_transfer_queue.queue):
                logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    if hasattr(decode_req.kv_receiver, "abort"):
                        decode_req.kv_receiver.abort()

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_abort=True`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def register_multi_tokenizer(self, recv_req: MultiTokenizerRegisterReq):
        self.send_to_detokenizer.send_pyobj(recv_req)
        return recv_req

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        if recv_req == ExpertDistributionReq.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif recv_req == ExpertDistributionReq.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif recv_req == ExpertDistributionReq.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
            return OpenSessionReqOutput(session_id, True)

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]

    def get_print_prefix(self):
        prefix = ""
        if self.attn_dp_rank is not None:
            prefix += f" DP{self.attn_dp_rank}"
        if self.server_args.tp_size > 1:
            prefix += f" TP{self.tp_rank}"
        if self.pp_size > 1:
            prefix += f" PP{self.pp_rank}"
        return prefix

    def current_scheduler_metrics_enabled(self):
        return self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_pyobj(recv_req)
        return None


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = time.time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            global_config.torch_empty_cache_interval > 0
            and time.time() - self.last_empty_time
            > global_config.torch_empty_cache_interval
        ):
            self.last_empty_time = time.time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    return getattr(recv_req, "rid", "").startswith("HEALTH_CHECK")


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    balance_meta: Optional[DPBalanceMeta] = None,
):
    if server_args.enable_trace:
        process_tracing_init(server_args.oltp_traces_endpoint, "sglang")
        if server_args.disaggregation_mode == "null":
            thread_label = "Scheduler"
            trace_set_thread_info(thread_label, tp_rank, dp_rank)

    if (numa_node := server_args.numa_node) is not None:
        numa_bind_to_node(numa_node[gpu_id])

    # Generate the prefix
    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta=balance_meta,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
