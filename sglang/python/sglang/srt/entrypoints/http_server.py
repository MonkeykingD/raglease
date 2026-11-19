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
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as multiprocessing
import os
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from sglang.srt.tracing.trace import process_tracing_init, trace_set_thread_info

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import orjson
import requests
import uvicorn
import uvloop
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponsesRequest,
    ScoringRequest,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import (
    AbortReq,
    AnnSearchReqInput,
    AnnSearchReqOutput,
    RagPreparationCancel,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    DestroyWeightsUpdateGroupReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    ProfileReqInput,
    RAG_PREFILL_REQUEST_LABEL,
    RagGenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SendWeightsToRemoteInstanceReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerManager,
    MultiTokenizerRouter,
    get_main_process_id,
    monkey_patch_uvicorn_multiprocessing,
    read_from_shared_memory,
    write_data_for_multi_tokenizer,
)
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    add_prometheus_middleware,
    delete_directory,
    get_bool_env_var,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.warmup import execute_warmups
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[
        TokenizerManager, MultiTokenizerRouter, MultiTokenizerManager
    ]
    template_manager: TemplateManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None
_rag_doc_store_path: Optional[str] = None
_rag_doc_store: Optional[Union[List[str], Dict[str, str]]] = None
_rag_doc_db_path: Optional[str] = None
_rag_doc_db = None
_rag_random_doc_key: Optional[tuple] = None
_rag_random_docs: Optional[List[str]] = None
_rag_query_vec_path: Optional[str] = None
_rag_query_order_key: Optional[tuple] = None
_rag_query_vectors: Optional[np.ndarray] = None
_rag_query_ids_path: Optional[str] = None
_rag_query_ids: Optional[np.ndarray] = None
_rag_query_db_path: Optional[str] = None
_rag_query_texts: Optional[Dict[int, str]] = None
_rag_cpu_ann_key: Optional[tuple] = None
_rag_cpu_ann_executor = None
_rag_cpu_ann_lock = threading.Lock()


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


async def init_multi_tokenizer() -> ServerArgs:
    """Read args information from shm and init tokenizer manager for current process"""
    pid = os.getpid()
    main_pid = get_main_process_id()
    logger.info(f"current worker_id: {pid}, main processID: {main_pid}")

    # Read configuration from shared memory
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )
    server_args: ServerArgs

    # API key authentication is not supported in multi-tokenizer mode
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"

    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )

    # Launch multi-tokenizer manager process
    tokenizer_manager = MultiTokenizerManager(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )
    # Register this tokenizer with the main tokenizer manager
    await tokenizer_manager.register_to_main_tokenizer_manager()

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    if server_args.enable_trace:
        process_tracing_init(server_args.oltp_traces_endpoint, "sglang")
        if server_args.disaggregation_mode == "null":
            thread_label = f"MultiTokenizer-{tokenizer_manager.worker_id}"
            trace_set_thread_info(thread_label)

    return server_args


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    if not getattr(fast_api_app, "is_single_tokenizer_mode", False):
        # Initialize multi-tokenizer support for worker processes
        fast_api_app.server_args: ServerArgs = await init_multi_tokenizer()

        # only metrics middleware is supported in multi-tokenizer mode
        worker_pid = os.getpid()
        if fast_api_app.server_args.enable_metrics:
            add_prometheus_middleware(app)
            enable_func_timer()

        logger.info(f"Worker {worker_pid} added prometheus middleware")
        fast_api_app.warmup_thread = threading.Thread(
            target=_wait_and_warmup,
            args=(
                fast_api_app.server_args,
                None,  # pipe_finish_writer not needed in worker
                None,  # launch_callback not needed in worker
            ),
        )

    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager
    )

    server_args: ServerArgs = fast_api_app.server_args

    tool_server = None
    if server_args.tool_server == "demo":
        from sglang.srt.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()
    elif server_args.tool_server:
        from sglang.srt.entrypoints.openai.tool_server import MCPToolServer

        tool_server = MCPToolServer()
        await tool_server.add_tool_server(server_args.tool_server)

    try:
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,
            enable_force_include_usage=True,
            tool_server=tool_server,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.warning(f"Can not initialize OpenAIServingResponses, error: {e}")

    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    warmup_thread = getattr(fast_api_app, "warmup_thread", None)
    if warmup_thread is not None:
        warmup_thread.start()

    try:
        yield
    finally:
        if server_args.tokenizer_worker_num > 1:
            pid = os.getpid()
            logger.info(f"uvicorn worker {pid} ending...")
            warmup_thread.join()
            logger.info(f"uvicorn worker {pid} ended.")


# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details"""
    error = ErrorResponse(
        object="error",
        message=exc.detail,
        type=str(exc.status_code),
        code=exc.status_code,
    )
    return ORJSONResponse(content=error.model_dump(), status_code=exc.status_code)


# Custom exception handlers to change validation error status codes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400"""
    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### Native API endpoints #####


@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    rid = f"HEALTH_CHECK_{time.time()}"

    if _global_state.tokenizer_manager.is_image_gen:
        # Keep this branch for some internal use cases.
        raise NotImplementedError("Image generation is not supported yet.")
    elif _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
    }
    return result


@app.get("/get_weight_version")
async def get_weight_version():
    """Get the current weight version."""
    return {
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version
    }


@app.get("/get_server_info")
async def get_server_info():
    # Returns interna states per DP.
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }


@app.get("/get_load")
async def get_load():
    return await _global_state.tokenizer_manager.get_load()


# example usage:
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return ret
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


@app.api_route("/generate_from_file", methods=["POST"])
async def generate_from_file_request(file: UploadFile, request: Request):
    """Handle a generate request, this is purely to work with input_embeds."""
    content = await file.read()
    input_embeds = json.loads(content.decode("utf-8"))

    obj = GenerateReqInput(
        input_embeds=input_embeds,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
    )

    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error(f"Error: {e}")
        return _create_error_response(e)


@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/clear_hicache_storage_backend", methods=["GET", "POST"])
async def clear_hicache_storage_backend():
    """Clear the hierarchical cache storage backend."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content="Hierarchical cache storage backend cleared.\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()

    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir,
        start_step=obj.start_step,
        num_steps=obj.num_steps,
        activities=obj.activities,
        with_stack=obj.with_stack,
        record_shapes=obj.record_shapes,
        profile_by_stage=obj.profile_by_stage,
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/freeze_gc", methods=["GET", "POST"])
async def freeze_gc_async():
    """
    See engine.freeze_gc for more details.
    """
    await _global_state.tokenizer_manager.freeze_gc()
    return Response(
        content="Garbage collection frozen.\n",
        status_code=200,
    )


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk inplace without re-launching the server."""
    success, message, num_paused_requests = (
        await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)
    )

    # Update weight version if provided and weights update was successful
    if success and obj.weight_version is not None:
        _update_weight_version_if_provided(obj.weight_version)
        message += f" Weight version updated to {obj.weight_version}."

    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_send_group_for_remote_instance")
async def init_weights_send_group_for_remote_instance(
    obj: InitWeightsSendGroupForRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.init_weights_send_group_for_remote_instance(
            obj, request
        )
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/send_weights_to_remote_instance")
async def send_weights_to_remote_instance(
    obj: SendWeightsToRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.send_weights_to_remote_instance(
            obj, request
        )
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/init_weights_update_group")
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/destroy_weights_update_group")
async def destroy_weights_update_group(
    obj: DestroyWeightsUpdateGroupReqInput, request: Request
):
    """Destroy the parameter update group."""
    success, message = (
        await _global_state.tokenizer_manager.destroy_weights_update_group(obj, request)
    )
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_tensor")
async def update_weights_from_tensor(
    obj: UpdateWeightsFromTensorReqInput, request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )

    # Update weight version if provided and weights update was successful
    if success and obj.weight_version is not None:
        _update_weight_version_if_provided(obj.weight_version)
        message += f" Weight version updated to {obj.weight_version}."

    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = (
        await _global_state.tokenizer_manager.update_weights_from_distributed(
            obj, request
        )
    )

    # Update weight version if provided and weights update was successful
    if success and obj.weight_version is not None:
        _update_weight_version_if_provided(obj.weight_version)
        message += f" Weight version updated to {obj.weight_version}."

    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weight_version")
async def update_weight_version(obj: UpdateWeightVersionReqInput, request: Request):
    """Update the weight version. This operation requires no active requests."""
    if obj.abort_all_requests:
        _global_state.tokenizer_manager.abort_request(abort_all=True)

    # Use a simple approach without the complex lock mechanism for now
    # since weight_version update is a simple operation that doesn't affect model weights
    try:
        # Update the weight version in server args (the single source of truth)
        _global_state.tokenizer_manager.server_args.weight_version = obj.new_version

        return ORJSONResponse(
            {
                "success": True,
                "message": f"Weight version updated to {obj.new_version}",
                "new_version": obj.new_version,
            },
            status_code=HTTPStatus.OK,
        )
    except Exception as e:
        return ORJSONResponse(
            {
                "success": False,
                "message": f"Failed to update weight version: {str(e)}",
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
async def get_weights_by_name(obj: GetWeightsByNameReqInput, request: Request):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/slow_down", methods=["GET", "POST"])
async def slow_down(obj: SlowDownReqInput, request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/load_lora_adapter", methods=["POST"])
async def load_lora_adapter(obj: LoadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter(obj, request)

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/unload_lora_adapter", methods=["POST"])
async def unload_lora_adapter(obj: UnloadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_lora_adapter(obj, request)

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


def _rag_cpu_retrieval_enabled() -> bool:
    return bool(
        _global_state is not None
        and getattr(
            _global_state.tokenizer_manager.server_args,
            "rag_cpu_retrieval",
            False,
        )
    )


def _load_rag_cpu_ann_executor():
    global _rag_cpu_ann_key, _rag_cpu_ann_executor

    if _global_state is None:
        raise ValueError("CPU RAG retrieval is unavailable before initialization.")
    server_args = _global_state.tokenizer_manager.server_args
    key = (
        str(server_args.rag_index_path),
        int(server_args.rag_index_nprobe),
        int(server_args.rag_cpu_retrieval_workers),
        int(server_args.rag_cpu_retrieval_threads),
        int(server_args.rag_cpu_retrieval_max_batch_queries),
    )
    if _rag_cpu_ann_executor is not None and _rag_cpu_ann_key == key:
        return _rag_cpu_ann_executor

    with _rag_cpu_ann_lock:
        if _rag_cpu_ann_executor is not None and _rag_cpu_ann_key == key:
            return _rag_cpu_ann_executor
        if _rag_cpu_ann_executor is not None:
            _rag_cpu_ann_executor.close()

        from sglang.srt.multiplex.cpu_ann_executor import CpuAnnExecutor

        executor = CpuAnnExecutor(
            key[0],
            nprobe=key[1],
            workers=key[2],
            threads_per_search=key[3],
            max_batch_queries=key[4],
        )
        started_at = time.perf_counter()
        executor.load()
        _rag_cpu_ann_executor = executor
        _rag_cpu_ann_key = key
        logger.info(
            "Loaded CPU RAG index in %.1fs: path=%s nprobe=%s workers=%s "
            "threads_per_search=%s max_batch_queries=%s",
            time.perf_counter() - started_at,
            *key,
        )
        return executor


async def _run_rag_cpu_ann_search(
    obj: AnnSearchReqInput,
) -> AnnSearchReqOutput:
    try:
        if obj.queries is not None:
            queries = np.asarray(obj.queries, dtype=np.float32)
        elif obj.query is not None:
            queries = np.asarray([obj.query], dtype=np.float32)
        else:
            return AnnSearchReqOutput(
                success=False,
                request_id=obj.request_id,
                message="ANN search requires `query` or `queries`.",
            )
        if queries.ndim != 2:
            return AnnSearchReqOutput(
                success=False,
                request_id=obj.request_id,
                message=f"ANN queries must be 2-D, got shape={queries.shape}.",
            )

        server_args = _global_state.tokenizer_manager.server_args
        topk = int(obj.topk or server_args.rag_index_topk)
        executor = _load_rag_cpu_ann_executor()
        enqueued_at = time.perf_counter()
        search = await asyncio.wrap_future(
            executor.submit(
                queries,
                topk,
                enqueued_at=enqueued_at,
            )
        )
        return AnnSearchReqOutput(
            success=True,
            ids=search.ids.tolist(),
            distances=search.distances.tolist(),
            latency_ms=(time.perf_counter() - enqueued_at) * 1000.0,
            queue_wait_ms=search.queue_wait_ms,
            gpu_latency_ms=0.0,
            cpu_queue_wait_ms=0.0,
            cpu_search_latency_ms=search.search_latency_ms,
            index_mode=executor.index_mode,
            hot_probe_ratio=0.0,
            hot_probe_count=0,
            cold_probe_count=(
                int(queries.shape[0]) * int(server_args.rag_index_nprobe)
            ),
            batch_size=search.batch_size,
            request_id=obj.request_id,
        )
    except Exception as error:
        logger.exception("CPU ANN search request failed.")
        return AnnSearchReqOutput(
            success=False,
            message=str(error),
            request_id=obj.request_id,
        )


@app.post("/ann_search")
async def ann_search_request(obj: AnnSearchReqInput, request: Request):
    """Run a minimal ANN-only request using the configured RAG backend."""
    try:
        if _rag_cpu_retrieval_enabled():
            result = await _run_rag_cpu_ann_search(obj)
        else:
            result = await _global_state.tokenizer_manager.ann_search(obj)
        status = 200 if result.success else 400
        return ORJSONResponse(content=dataclasses.asdict(result), status_code=status)
    except Exception as e:
        return _create_error_response(e)


def _load_rag_doc_store() -> Optional[Union[List[str], Dict[str, str]]]:
    global _rag_doc_store_path, _rag_doc_store
    if _global_state is None:
        return None

    server_args = _global_state.tokenizer_manager.server_args
    path = getattr(server_args, "rag_doc_store_path", "")
    if not path:
        return None

    if _rag_doc_store is not None and _rag_doc_store_path == path:
        return _rag_doc_store

    id_field = getattr(server_args, "rag_doc_id_field", "id")
    text_field = getattr(server_args, "rag_doc_text_field", "text")
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".jsonl":
        docs: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as fin:
            for lineno, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    docs[str(lineno)] = item
                elif isinstance(item, dict):
                    doc_id = item.get(id_field, lineno)
                    text = item.get(text_field)
                    if text is not None:
                        docs[str(doc_id)] = str(text)
        _rag_doc_store = docs
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                docs = {}
                for lineno, item in enumerate(payload):
                    doc_id = item.get(id_field, lineno)
                    text = item.get(text_field)
                    if text is not None:
                        docs[str(doc_id)] = str(text)
                _rag_doc_store = docs
            else:
                _rag_doc_store = [str(item) for item in payload]
        elif isinstance(payload, dict):
            _rag_doc_store = {str(k): str(v) for k, v in payload.items()}
        else:
            raise ValueError(f"Unsupported JSON doc store format: {path}")
    else:
        with open(path, "r", encoding="utf-8") as fin:
            _rag_doc_store = [line.rstrip("\n") for line in fin]

    _rag_doc_store_path = path
    logger.info("Loaded RAG doc store from %s", path)
    return _rag_doc_store


def _load_rag_doc_db():
    global _rag_doc_db_path, _rag_doc_db
    if _global_state is None:
        return None

    server_args = _global_state.tokenizer_manager.server_args
    path = getattr(server_args, "rag_doc_db_path", "")
    if not path:
        return None

    if _rag_doc_db is not None and _rag_doc_db_path == path:
        return _rag_doc_db

    try:
        import plyvel
    except ImportError as exc:
        raise ValueError("--rag-doc-db-path requires plyvel.") from exc

    if _rag_doc_db is not None:
        _rag_doc_db.close()

    _rag_doc_db = plyvel.DB(path, create_if_missing=False)
    _rag_doc_db_path = path
    logger.info("Opened RAG document LevelDB from %s", path)
    return _rag_doc_db


def _get_rag_doc_from_db(doc_id: int) -> Optional[str]:
    db = _load_rag_doc_db()
    if db is None:
        return None
    doc = db.get(str(doc_id).encode())
    return doc.decode("utf-8") if doc is not None else None


def _get_rag_random_doc(doc_id: int) -> Optional[str]:
    global _rag_random_doc_key, _rag_random_docs
    if _global_state is None:
        return None

    server_args = _global_state.tokenizer_manager.server_args
    if not getattr(server_args, "rag_doc_random_fallback", False):
        return None

    pool_size = max(1, int(getattr(server_args, "rag_doc_random_pool_size", 1000)))
    input_len = max(1, int(getattr(server_args, "rag_doc_random_input_len", 1023)))
    key = (pool_size, input_len)
    if _rag_random_docs is None or _rag_random_doc_key != key:
        bags_of_words = [
            "one",
            "two",
            "six",
            "ten",
            "apple",
            "banana",
            "orange",
            "grape",
            "car",
            "bike",
        ]
        docs = []
        for _ in range(pool_size):
            random_ids = np.random.choice(len(bags_of_words), size=input_len)
            docs.append(" ".join(np.array(bags_of_words)[random_ids]))
        _rag_random_docs = docs
        _rag_random_doc_key = key
        logger.info(
            "Generated RAG random fallback docs: pool_size=%s input_len=%s",
            pool_size,
            input_len,
        )

    return _rag_random_docs[doc_id % len(_rag_random_docs)]


def _load_vector_file(path: str) -> np.ndarray:
    suffix = os.path.splitext(path)[1].lower()
    dtype = np.float32 if suffix == ".fbin" else np.int64 if suffix == ".ibin" else None
    if dtype is None:
        raise ValueError(f"Unsupported vector file format: {path}")
    with open(path, "rb") as fin:
        cnt, dim = np.fromfile(fin, count=2, dtype=np.int32)
        data = np.fromfile(fin, dtype=dtype).reshape(cnt, dim)
    return data


def _load_permutation_file(path: str) -> np.ndarray:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        return np.load(path).reshape(-1).astype(np.int64, copy=False)
    return _load_vector_file(path).reshape(-1).astype(np.int64, copy=False)


def _infer_query_ids_path(query_vec_path: str) -> str:
    if query_vec_path.endswith("_qvec.fbin"):
        return query_vec_path[: -len("_qvec.fbin")] + "_qids.ibin"
    base, _ = os.path.splitext(query_vec_path)
    return base + "_qids.ibin"


def _load_rag_query_store():
    global _rag_query_vec_path, _rag_query_order_key, _rag_query_vectors
    global _rag_query_ids_path, _rag_query_ids
    global _rag_query_db_path, _rag_query_texts

    if _global_state is None:
        raise ValueError("RAG query store is unavailable before server initialization.")

    server_args = _global_state.tokenizer_manager.server_args
    vec_path = getattr(server_args, "rag_query_vec_path", "")
    if not vec_path:
        raise ValueError(
            "query_id/rid mode requires --rag-query-vec-path or an explicit query vector."
        )

    ids_path = getattr(server_args, "rag_query_ids_path", "") or _infer_query_ids_path(
        vec_path
    )
    perm_path = getattr(server_args, "rag_query_permutation_path", "")
    shuffle = bool(getattr(server_args, "rag_query_shuffle", False))
    shuffle_seed = getattr(server_args, "rag_query_shuffle_seed", None)
    order_key = (vec_path, ids_path, perm_path, shuffle, shuffle_seed)

    if _rag_query_vectors is None or _rag_query_order_key != order_key:
        query_vectors = _load_vector_file(vec_path).astype(np.float32, copy=False)
        query_ids = None
        if ids_path and os.path.exists(ids_path):
            query_ids = _load_vector_file(ids_path).reshape(-1)

        perm = None
        if perm_path:
            perm = _load_permutation_file(perm_path)
        elif shuffle:
            if shuffle_seed is None:
                perm = np.random.permutation(len(query_vectors))
            else:
                perm = np.random.RandomState(shuffle_seed).permutation(
                    len(query_vectors)
                )

        if perm is not None:
            query_vectors = query_vectors[perm, :]
            if query_ids is not None:
                query_ids = query_ids[perm]

        _rag_query_vectors = query_vectors
        _rag_query_vec_path = vec_path
        _rag_query_order_key = order_key
        _rag_query_ids = query_ids
        _rag_query_ids_path = ids_path if query_ids is not None else None
        logger.info(
            "Loaded RAG query vectors from %s, shape=%s, shuffled=%s",
            vec_path,
            _rag_query_vectors.shape,
            perm is not None,
        )
        if _rag_query_ids is not None:
            logger.info("Loaded RAG query ids from %s", ids_path)
    else:
        if ids_path and os.path.exists(ids_path) and _rag_query_ids is None:
            _rag_query_ids = _load_vector_file(ids_path).reshape(-1)
            _rag_query_ids_path = ids_path

    db_path = getattr(server_args, "rag_query_db_path", "")
    if db_path and (_rag_query_texts is None or _rag_query_db_path != db_path):
        try:
            import plyvel
        except ImportError as exc:
            raise ValueError(
                "--rag-query-db-path requires plyvel to recover query text."
            ) from exc

        db = plyvel.DB(db_path, create_if_missing=False)
        try:
            _rag_query_texts = {int(k): v.decode("utf-8") for k, v in db}
            _rag_query_db_path = db_path
            logger.info("Loaded RAG query database from %s", db_path)
        finally:
            db.close()

    return _rag_query_vectors, _rag_query_ids, _rag_query_texts


def _resolve_rag_query(obj: RagGenerateReqInput) -> tuple[List[float], str, int]:
    if obj.query is not None:
        question = obj.text
        row_id = obj.query_id if obj.query_id is not None else -1
        if question is None and obj.query_id is not None:
            try:
                _, query_ids, query_texts = _load_rag_query_store()
                if query_ids is not None and query_texts is not None:
                    row = obj.query_id % len(query_ids)
                    question = query_texts.get(int(query_ids[row]))
            except ValueError:
                question = None
        if question is None:
            question = f"Query {row_id}"
        return obj.query, question, row_id

    query_vectors, query_ids, query_texts = _load_rag_query_store()

    if obj.query_id is not None:
        row_id = obj.query_id
    elif obj.rid is not None:
        try:
            row_id = int(str(obj.rid).split("_")[-1])
        except ValueError as exc:
            raise ValueError(
                "RAG query vector lookup requires query_id or numeric rid."
            ) from exc
    else:
        raise ValueError("RAG query vector lookup requires query_id or numeric rid.")

    row_id = row_id % len(query_vectors)
    query_vector = query_vectors[row_id].astype(np.float32, copy=False).tolist()

    question = obj.text
    if question is None and query_texts is not None and query_ids is not None:
        original_qid = int(query_ids[row_id])
        question = query_texts.get(original_qid)
    if question is None:
        question = f"Query {row_id}"

    return query_vector, question, row_id


def _resolve_rag_documents(
    doc_ids: List[int],
    documents: Optional[Union[List[str], Dict[str, str]]],
    doc_template: str,
) -> List[str]:
    use_doc_db = documents is None and getattr(
        _global_state.tokenizer_manager.server_args, "rag_doc_db_path", ""
    )
    if documents is None and not use_doc_db:
        documents = _load_rag_doc_store()

    # Match VectorLiteRAG's default fallback path:
    # self.knowledge_base[doc_ids[0] % 1000], not top-k concatenation.
    if documents is None and not use_doc_db and doc_ids:
        text = _get_rag_random_doc(doc_ids[0])
        if text is not None:
            return [text]

    resolved_docs = []
    for doc_id in doc_ids:
        text = None
        if isinstance(documents, list) and 0 <= doc_id < len(documents):
            text = documents[doc_id]
        elif isinstance(documents, dict):
            text = documents.get(str(doc_id))
            if text is None:
                text = documents.get(doc_id)
        elif use_doc_db:
            text = _get_rag_doc_from_db(doc_id)

        if text is None:
            text = doc_template.format(doc_id=doc_id)
        resolved_docs.append(str(text))
    return resolved_docs


def _build_rag_prompt(
    question: str,
    docs: List[str],
    prompt_template: str,
    document_separator: str,
    number_documents: bool = False,
) -> str:
    if number_documents:
        context = document_separator.join(
            f"[{idx + 1}] {doc}" for idx, doc in enumerate(docs)
        )
    else:
        context = document_separator.join(docs)
    return prompt_template.format(context=context, question=question)


@app.post("/rag_generate")
async def rag_generate_request(obj: RagGenerateReqInput, request: Request):
    """Run configured ANN retrieval, compose a prompt, and generate."""
    preparation_registered = False
    streaming_returned = False

    async def release_preparation():
        nonlocal preparation_registered
        if preparation_registered:
            pending_send = _global_state.tokenizer_manager.send_to_scheduler.send_pyobj(
                RagPreparationCancel(rag_rid))
            # zmq.asyncio sockets may enqueue asynchronously under backpressure.
            # Do not drop the send Future or mark the cleanup complete early.
            if pending_send is not None:
                await pending_send
            preparation_registered = False

    try:
        e1_strict = getattr(obj, 'e1_prompt_contract', None) is not None
        if e1_strict:
            from sglang.srt.e1_strict_inputs import (
                validate_request, resolve_query, prepare_prompt)
            validate_request(obj, _global_state.tokenizer_manager.server_args)
            query_vector, question, query_row_id = resolve_query(obj, _load_rag_query_store())
        else:
            query_vector, question, query_row_id = _resolve_rag_query(obj)
        rag_rid = str(obj.rid) if obj.rid is not None else uuid.uuid4().hex
        ann_result = None
        if obj.skip_retrieval:
            doc_ids = [query_row_id]
        else:
            ann_request = AnnSearchReqInput(
                query=query_vector,
                topk=obj.topk,
                request_id=rag_rid,
            )
            ann_request.set_generation_contract(
                obj.sampling_params, skip_generation=obj.skip_generation)
            if _rag_cpu_retrieval_enabled():
                ann_result = await _run_rag_cpu_ann_search(ann_request)
            else:
                preparation_registered = ann_request.preparation_follows
                ann_result = await _global_state.tokenizer_manager.ann_search(
                    ann_request
                )
            if not ann_result.success:
                return ORJSONResponse(
                    content=dataclasses.asdict(ann_result),
                    status_code=400,
                )
            doc_ids = (ann_result.ids or [[]])[0]
        tokenizer_manager = _global_state.tokenizer_manager
        token_cache_hit = False
        rag_input_ids = None
        e1_prompt_evidence = None
        if e1_strict:
            prepared = prepare_prompt(
                tokenizer_manager.tokenizer, question, doc_ids, _get_rag_doc_from_db)
            rag_input_ids = prepared['input_ids']
            e1_prompt_evidence = prepared['evidence']
            rag_prompt = None
            docs = []
            token_cache_hit = False
        else:
            docs = _resolve_rag_documents(doc_ids, obj.documents, obj.doc_template)
            rag_prompt = _build_rag_prompt(
                question=question, docs=docs, prompt_template=obj.prompt_template,
                document_separator=obj.document_separator,
                number_documents=obj.number_documents)
            if (getattr(tokenizer_manager.server_args, "rag_doc_token_cache_size", 0) > 0
                    and hasattr(tokenizer_manager, "tokenize_rag_prompt_cached")):
                rag_input_ids, token_cache_hit = await tokenizer_manager.tokenize_rag_prompt_cached(rag_prompt)

        gen_obj = GenerateReqInput(
            text=rag_prompt if rag_input_ids is None else None,
            input_ids=rag_input_ids,
            sampling_params=obj.sampling_params,
            rid=rag_rid,
            stream=obj.stream,
            label=RAG_PREFILL_REQUEST_LABEL,
        )

        rag_metadata = {
            "query_id": query_row_id,
            "ids": doc_ids,
            "documents": docs,
            "ann_latency_ms": 0.0,
            "ann_queue_wait_ms": 0.0,
            "ann_gpu_latency_ms": 0.0,
            "ann_cpu_queue_wait_ms": 0.0,
            "ann_cpu_search_latency_ms": 0.0,
            "ann_coarse_quantize_latency_ms": 0.0,
            "ann_route_latency_ms": 0.0,
            "ann_merge_latency_ms": 0.0,
            "ann_index_mode": "none",
            "ann_hot_probe_ratio": 0.0,
            "ann_hot_probe_count": 0,
            "ann_cold_probe_count": 0,
            "ann_batch_size": 0,
            "token_cache_hit": token_cache_hit,
        }
        if e1_strict:
            rag_metadata.pop('documents')
            rag_metadata['e1_prompt'] = e1_prompt_evidence
        if ann_result is not None:
            rag_metadata.update(
                {
                    "ann_latency_ms": ann_result.latency_ms,
                    "ann_queue_wait_ms": ann_result.queue_wait_ms,
                    "ann_gpu_latency_ms": ann_result.gpu_latency_ms,
                    "ann_cpu_queue_wait_ms": ann_result.cpu_queue_wait_ms,
                    "ann_cpu_search_latency_ms": (
                        ann_result.cpu_search_latency_ms
                    ),
                    "ann_coarse_quantize_latency_ms": (
                        ann_result.coarse_quantize_latency_ms
                    ),
                    "ann_route_latency_ms": ann_result.route_latency_ms,
                    "ann_merge_latency_ms": ann_result.merge_latency_ms,
                    "ann_index_mode": ann_result.index_mode,
                    "ann_hot_probe_ratio": ann_result.hot_probe_ratio,
                    "ann_hot_probe_count": ann_result.hot_probe_count,
                    "ann_cold_probe_count": ann_result.cold_probe_count,
                    "ann_batch_size": ann_result.batch_size,
                }
            )
        if hasattr(tokenizer_manager, "get_rag_doc_token_cache_stats"):
            rag_metadata["token_cache"] = (
                tokenizer_manager.get_rag_doc_token_cache_stats()
            )

        if obj.skip_generation:
            ann_only_output = {
                "text": "",
                "meta_info": {"completion_tokens": 0},
                "rag": rag_metadata,
            }
            if obj.stream:

                async def stream_ann_only_result() -> AsyncIterator[bytes]:
                    yield b"data: " + orjson.dumps(
                        ann_only_output, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    stream_ann_only_result(),
                    media_type="text/event-stream",
                )
            return ann_only_output

        if obj.stream:

            async def stream_results() -> AsyncIterator[bytes]:
                first_result = True
                try:
                    async for out in _global_state.tokenizer_manager.generate_request(
                        gen_obj, request
                    ):
                        if first_result and obj.include_retrieval:
                            out = dict(out)
                            out["rag"] = rag_metadata
                        first_result = False
                        yield b"data: " + orjson.dumps(
                            out, option=orjson.OPT_NON_STR_KEYS
                        ) + b"\n\n"
                except ValueError as e:
                    out = {"error": {"message": str(e)}}
                    logger.error(f"[http_server] Error: {e}")
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
                finally:
                    await release_preparation()
                yield b"data: [DONE]\n\n"

            background = _global_state.tokenizer_manager.create_abort_task(gen_obj)
            background.add_task(release_preparation)
            streaming_returned = True
            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
                background=background,
            )

        gen_ret = await _global_state.tokenizer_manager.generate_request(
            gen_obj, request
        ).__anext__()
        if obj.include_retrieval:
            gen_ret["rag"] = rag_metadata
        return gen_ret
    except ValueError as e:
        logger.error(f"[http_server] Error: {e}")
        return _create_error_response(e)
    except Exception as e:
        return _create_error_response(e)
    finally:
        if not streaming_returned:
            await release_preparation()


@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(model_type=obj.reasoning_parser)

    # 2) Call the non-stream parsing method (non-stream)
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/pause_generation")
async def pause_generation(request: Request):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation()
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


@app.post("/continue_generation")
async def continue_generation(request: Request):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation()
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request
    )


@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )
    return ModelList(data=model_cards)


@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the decoder-only scoring API. See Engine.score() for detailed documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request
    )


@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: dict, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    request_obj = ResponsesRequest(**request)
    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request_obj, raw_request
    )

    # Handle streaming responses
    if isinstance(result, AsyncGenerator):
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return result


@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id
    )


@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id
    )


@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request
    )


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


## Vertex AI API
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(vertex_req: VertexGenerateReqInput, raw_request: Request):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


def _update_weight_version_if_provided(weight_version: Optional[str]) -> None:
    """Update weight version if provided."""
    if weight_version is not None:
        _global_state.tokenizer_manager.server_args.weight_version = weight_version


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    if server_args.tokenizer_worker_num > 1:
        port_args = PortArgs.init_new(server_args)
        port_args.tokenizer_worker_ipc_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args, port_args=port_args
        )
    else:
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
        )

        if server_args.enable_trace:
            process_tracing_init(server_args.oltp_traces_endpoint, "sglang")
            if server_args.disaggregation_mode == "null":
                thread_label = "Tokenizer"
                trace_set_thread_info(thread_label)

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    if server_args.tokenizer_worker_num > 1:
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args,
            server_args,
            scheduler_info,
        )
    else:
        # Add api key authorization
        if server_args.api_key:
            add_api_key_middleware(app, server_args.api_key)

        # Add prometheus middleware
        if server_args.enable_metrics:
            add_prometheus_middleware(app)
            enable_func_timer()

        # Send a warmup request - we will create the thread launch it
        # in the lifespan after all other warmups have fired.
        warmup_thread = threading.Thread(
            target=_wait_and_warmup,
            args=(
                server_args,
                pipe_finish_writer,
                launch_callback,
            ),
        )
        app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        if server_args.tokenizer_worker_num > 1:
            from uvicorn.config import LOGGING_CONFIG

            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }

            monkey_patch_uvicorn_multiprocessing()

            uvicorn.run(
                "sglang.srt.entrypoints.http_server:app",
                host=server_args.host,
                port=server_args.port,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
                workers=server_args.tokenizer_worker_num,
            )
        else:
            app.is_single_tokenizer_mode = True
            uvicorn.run(
                app,
                host=server_args.host,
                port=server_args.port,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
            )
    finally:
        if server_args.tokenizer_worker_num > 1:
            multi_tokenizer_args_shm.unlink()
            _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()
        else:
            warmup_thread.join()


def _execute_server_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # Debug dumping
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
            _global_state.tokenizer_manager.server_status = ServerStatus.Up

        else:
            logger.info(f"Start of pd disaggregation warmup ...")
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
                # This is a hack to ensure fake transfer is enabled during prefill warmup
                # ensure each dp rank has a unique bootstrap_room during prefill warmup
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[0, 1, 2, 3]] * server_args.dp_size,
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=1800,  # because of deep gemm precache is very long if not precache.
            )
            if res.status_code == 200:
                logger.info(
                    f"End of prefill disaggregation mode warmup with status {res.status_code}, resp: {res.json()}"
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.Up
            else:
                logger.info(
                    "Prefill disaggregation mode warm Up Failed, status code: {}".format(
                        res.status_code
                    )
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy

    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return False

    # Debug print
    # logger.info(f"warmup request returns: {res.json()=}")
    return success


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
    launch_callback: Optional[Callable[[], None]] = None,
):
    if not server_args.skip_server_warmup:
        if not _execute_server_warmup(
            server_args,
            pipe_finish_writer,
        ):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    logger.info("The server is fired up and ready to roll!")

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()
