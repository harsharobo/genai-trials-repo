"""MLflow ChatModel for Gemma 4 26B-A4B on 4xA10G with vLLM."""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Generator, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Schema
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
    FunctionToolCallArguments,
    TokenUsageStats,
    ToolCall,
)


MODEL_ID = "google/gemma-4-26B-A4B-it"
SERVED_MODEL_NAME = "default"
HOST = "127.0.0.1"
PORT = 9989
TENSOR_PARALLEL_SIZE = 4
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_TOKENS = 256
LOGGING_MODE_ENV = "MLFLOW_CHATMODEL_LOGGING_MODE"

PIP_REQUIREMENTS = [
    "mlflow>=3.1.0,<4",
    "vllm==0.26.0",
    "transformers==5.14.1",
    "filelock==3.18.0",
    "httpx==0.28.1",
]


# The legacy ChatModel signature models ``tools[].function`` as a nested
# ``object``. Databricks Model Serving currently rejects normal OpenAI function
# tool dictionaries against that schema before this model is invoked. A map of
# arbitrary JSON values preserves the OpenAI request shape while avoiding that
# serving-side object coercion.
OPENAI_TOOL_CHAT_INPUT_SCHEMA = Schema(
    [
        ColSpec(name="messages", type=Array(Map(AnyType())), required=True),
        ColSpec(name="temperature", type=DataType.double, required=False),
        ColSpec(name="max_tokens", type=DataType.long, required=False),
        ColSpec(name="stop", type=Array(DataType.string), required=False),
        ColSpec(name="n", type=DataType.long, required=False),
        ColSpec(name="stream", type=DataType.boolean, required=False),
        ColSpec(name="top_p", type=DataType.double, required=False),
        ColSpec(name="top_k", type=DataType.long, required=False),
        ColSpec(name="frequency_penalty", type=DataType.double, required=False),
        ColSpec(name="presence_penalty", type=DataType.double, required=False),
        ColSpec(name="tools", type=Array(Map(AnyType())), required=False),
        ColSpec(name="tool_choice", type=AnyType(), required=False),
        ColSpec(name="custom_inputs", type=Map(AnyType()), required=False),
    ]
)

def enable_openai_tool_compatibility() -> None:
    """Install the input adapter required by legacy MLflow ChatModel.

    ``ChatModel`` always logs MLflow's built-in signature and its input wrapper
    silently discards unknown parameters. The OpenAI Chat Completions API sends
    ``tool_choice`` at the top level, so preserve it under ``custom_inputs``
    where ``_chat_completions_payload`` can forward it directly to vLLM.

    This function must run both while logging and while loading the serving
    model. The module-level call below satisfies both code-path model loading
    modes used by ``mlflow.pyfunc.log_model``.
    """
    import mlflow.pyfunc as pyfunc
    from mlflow.pyfunc.loaders import chat_model as chat_model_loader

    pyfunc.CHAT_MODEL_INPUT_SCHEMA = OPENAI_TOOL_CHAT_INPUT_SCHEMA

    if getattr(chat_model_loader, "_gemma4_tool_choice_patched", False):
        return

    original_from_dict = ChatParams.from_dict.__func__

    @classmethod
    def from_dict_with_tool_choice(cls, data: dict[str, Any]) -> ChatParams:
        request_data = dict(data)
        tool_choice = request_data.pop("tool_choice", None)
        params = original_from_dict(cls, request_data)

        if tool_choice is not None:
            custom_inputs = dict(params.custom_inputs or {})
            vllm_inputs = dict(custom_inputs.get("vllm") or {})
            vllm_inputs["tool_choice"] = tool_choice
            custom_inputs["vllm"] = vllm_inputs
            params.custom_inputs = custom_inputs

        return params

    ChatParams.from_dict = from_dict_with_tool_choice
    chat_model_loader.ChatParams = ChatParams
    chat_model_loader._gemma4_tool_choice_patched = True


class Gemma4VLLMChatModel(ChatModel):
    def __init__(self) -> None:
        self._process = None
        self._client = None
        self._lock = None
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self._owns_process = False
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def model_id(self) -> str:
        return os.environ.get("GEMMA4_MODEL_ID", MODEL_ID)

    @property
    def max_model_len(self) -> int:
        return int(os.environ.get("GEMMA4_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN))

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
            force=True,
        )

        if self._is_databricks_notebook():
            self._logger.info("Skipping vLLM startup in a Databricks notebook")
            return
        if self._is_logging_mode():
            self._logger.info("Skipping vLLM startup during MLflow ChatModel validation")
            return

        from filelock import FileLock
        import httpx

        self._client = httpx.Client(
            base_url=f"http://{HOST}:{PORT}",
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
        self._lock = FileLock(
            str(Path.home() / ".gemma4-vllm-chat-model.lock"), timeout=1800
        )

        with self._lock:
            if not self._is_healthy():
                self._validate_runtime()
                self._start_server()
                self._wait_until_healthy()
                self._owns_process = True

        if self._owns_process:
            atexit.register(self._stop_server)
            self._watchdog_thread = threading.Thread(
                target=self._watchdog,
                name="gemma4-vllm-chat-model-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        messages: list[ChatMessage],
        params: ChatParams,
    ) -> ChatCompletionResponse:
        if self._is_databricks_notebook():
            return self._empty_chat_response()
        if self._is_logging_mode():
            return self._logging_validation_response()
        if self._is_health_request(params):
            return self._health_response()

        payload = self._chat_completions_payload(messages, params, stream=False)
        response = self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return self._to_chat_completion_response(response.json())

    def predict_stream(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        messages: list[ChatMessage],
        params: ChatParams,
    ) -> Generator[ChatCompletionChunk, None, None]:
        if self._is_databricks_notebook():
            yield self._empty_chat_chunk()
            return
        if self._is_logging_mode():
            yield self._logging_validation_chunk()
            return
        if self._is_health_request(params):
            yield self._health_chunk()
            return

        payload = self._chat_completions_payload(messages, params, stream=True)
        yield from self._vllm_stream(payload)

    def _chat_completions_payload(
        self,
        messages: list[ChatMessage],
        params: ChatParams,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [message.to_dict() for message in messages],
            "stream": stream,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens or DEFAULT_MAX_TOKENS,
            "n": params.n,
        }

        if params.tools:
            payload["tools"] = [tool.to_dict() for tool in params.tools]

        for name in [
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]:
            value = getattr(params, name)
            if value is not None:
                payload[name] = value

        overrides = (params.custom_inputs or {}).get("vllm", {})
        if not isinstance(overrides, dict):
            raise ValueError("custom_inputs.vllm must be a dictionary")

        blocked_keys = {"model", "messages", "stream"}
        payload.update(
            {
                key: value
                for key, value in overrides.items()
                if key not in blocked_keys
            }
        )
        return payload

    def _to_chat_completion_response(
        self, response: dict[str, Any]
    ) -> ChatCompletionResponse:
        choices = []
        for choice in response.get("choices") or []:
            choices.append(
                ChatChoice(
                    index=choice.get("index", 0),
                    message=ChatMessage.from_dict(choice.get("message") or {}),
                    finish_reason=choice.get("finish_reason") or "stop",
                )
            )

        if not choices:
            raise ValueError("vLLM returned no chat completion choices")

        return ChatCompletionResponse(
            id=response.get("id"),
            created=response.get("created", int(time.time())),
            model=response.get("model") or self.model_id,
            choices=choices,
            usage=self._token_usage(response.get("usage")),
        )

    def _vllm_stream(
        self, payload: dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        pending_tool_calls: dict[int, dict[int, dict[str, str]]] = {}
        last_chunk_metadata: dict[str, Any] = {}

        with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    pending_choices = self._pending_tool_call_choices(
                        pending_tool_calls, payload
                    )
                    if pending_choices:
                        yield ChatCompletionChunk(
                            id=last_chunk_metadata.get("id"),
                            created=last_chunk_metadata.get("created", int(time.time())),
                            model=last_chunk_metadata.get("model") or self.model_id,
                            choices=pending_choices,
                        )
                    return

                chunk = json.loads(data)
                last_chunk_metadata = chunk
                choices = []
                for choice in chunk.get("choices") or []:
                    choice_index = choice.get("index", 0)
                    delta = choice.get("delta") or {}
                    raw_tool_calls = delta.get("tool_calls") or []
                    if raw_tool_calls:
                        self._update_pending_tool_calls(
                            pending_tool_calls, choice_index, raw_tool_calls
                        )

                    finish_reason = choice.get("finish_reason")
                    completed_tool_calls = None
                    if finish_reason is not None and pending_tool_calls.get(choice_index):
                        completed_tool_calls = self._build_tool_calls(
                            pending_tool_calls.pop(choice_index), payload
                        )

                    if any(
                        value is not None
                        for value in [
                            delta.get("role"),
                            delta.get("content"),
                            delta.get("refusal"),
                            completed_tool_calls,
                            finish_reason,
                        ]
                    ):
                        choices.append(
                            ChatChunkChoice(
                                index=choice_index,
                                finish_reason=finish_reason,
                                delta=ChatChoiceDelta(
                                    role=delta.get("role"),
                                    content=delta.get("content"),
                                    refusal=delta.get("refusal"),
                                    tool_calls=completed_tool_calls,
                                ),
                            )
                        )

                usage = self._token_usage(chunk.get("usage"))
                if choices or usage is not None:
                    yield ChatCompletionChunk(
                        id=chunk.get("id"),
                        created=chunk.get("created", int(time.time())),
                        model=chunk.get("model") or self.model_id,
                        choices=choices,
                        usage=usage,
                        custom_outputs=chunk.get("custom_outputs"),
                    )

    @staticmethod
    def _update_pending_tool_calls(
        pending_tool_calls: dict[int, dict[int, dict[str, str]]],
        choice_index: int,
        tool_call_deltas: list[dict[str, Any]],
    ) -> None:
        choice_calls = pending_tool_calls.setdefault(choice_index, {})
        for tool_call_delta in tool_call_deltas:
            tool_index = tool_call_delta.get("index", 0)
            state = choice_calls.setdefault(
                tool_index,
                {"id": "", "type": "function", "name": "", "arguments": ""},
            )
            if tool_call_delta.get("id"):
                state["id"] = tool_call_delta["id"]
            if tool_call_delta.get("type"):
                state["type"] = tool_call_delta["type"]

            function_delta = tool_call_delta.get("function") or {}
            if function_delta.get("name"):
                state["name"] += function_delta["name"]
            if function_delta.get("arguments"):
                state["arguments"] += function_delta["arguments"]

    def _build_tool_calls(
        self,
        pending_calls: dict[int, dict[str, str]],
        payload: dict[str, Any],
    ) -> list[ToolCall]:
        configured_names = [
            tool.get("function", {}).get("name")
            for tool in payload.get("tools", [])
            if tool.get("function", {}).get("name")
        ]
        tool_calls = []

        for tool_index in sorted(pending_calls):
            state = pending_calls[tool_index]
            name = state["name"]
            if not name and len(configured_names) == 1:
                name = configured_names[0]
            if not name:
                raise ValueError(
                    "vLLM returned a streaming tool call without a function name"
                )

            tool_calls.append(
                ToolCall(
                    id=state["id"] or f"call-{uuid.uuid4()}",
                    type=state["type"] or "function",
                    function=FunctionToolCallArguments(
                        name=name,
                        arguments=state["arguments"] or "{}",
                    ),
                )
            )
        return tool_calls

    def _pending_tool_call_choices(
        self,
        pending_tool_calls: dict[int, dict[int, dict[str, str]]],
        payload: dict[str, Any],
    ) -> list[ChatChunkChoice]:
        choices = []
        for choice_index in sorted(pending_tool_calls):
            choices.append(
                ChatChunkChoice(
                    index=choice_index,
                    finish_reason="tool_calls",
                    delta=ChatChoiceDelta(
                        role="assistant",
                        tool_calls=self._build_tool_calls(
                            pending_tool_calls[choice_index], payload
                        ),
                    ),
                )
            )
        return choices

    @staticmethod
    def _token_usage(usage: dict[str, Any] | None) -> TokenUsageStats | None:
        if not usage:
            return None
        return TokenUsageStats(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _logging_validation_response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-mlflow-validation",
            model=self.model_id,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatMessage(
                        role="assistant",
                        content="MLflow ChatModel validation response",
                    ),
                )
            ],
        )

    def _empty_chat_response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-databricks-notebook",
            model=self.model_id,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatMessage(role="assistant", content=""),
                )
            ],
        )

    def _logging_validation_chunk(self) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id="chatcmpl-mlflow-validation",
            model=self.model_id,
            choices=[
                ChatChunkChoice(
                    index=0,
                    finish_reason="stop",
                    delta=ChatChoiceDelta(
                        role="assistant",
                        content="MLflow ChatModel validation response",
                    ),
                )
            ],
        )

    def _empty_chat_chunk(self) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id="chatcmpl-databricks-notebook",
            model=self.model_id,
            choices=[
                ChatChunkChoice(
                    index=0,
                    finish_reason="stop",
                    delta=ChatChoiceDelta(role="assistant", content=""),
                )
            ],
        )

    def _health_response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-health-{uuid.uuid4()}",
            model=self.model_id,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatMessage(
                        role="assistant", content=json.dumps(self._health_payload())
                    ),
                )
            ],
        )

    def _health_chunk(self) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=f"chatcmpl-health-{uuid.uuid4()}",
            model=self.model_id,
            choices=[
                ChatChunkChoice(
                    index=0,
                    finish_reason="stop",
                    delta=ChatChoiceDelta(
                        role="assistant", content=json.dumps(self._health_payload())
                    ),
                )
            ],
        )

    def _is_health_request(self, params: ChatParams) -> bool:
        return bool((params.custom_inputs or {}).get("health_check"))

    def _health_payload(self) -> dict[str, Any]:
        return {
            "healthy": self._is_healthy(),
            "model": self.model_id,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": self.max_model_len,
        }

    @staticmethod
    def _is_logging_mode() -> bool:
        return os.environ.get(LOGGING_MODE_ENV) == "1"

    @staticmethod
    def _is_databricks_notebook() -> bool:
        try:
            from mlflow.utils.databricks_utils import is_in_databricks_notebook

            return is_in_databricks_notebook()
        except Exception:
            return False

    def _validate_runtime(self) -> None:
        import torch

        self._logger.info(
            "Runtime versions: vllm=%s transformers=%s torch=%s",
            version("vllm"),
            version("transformers"),
            torch.__version__,
        )
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
        self._logger.info("Detected GPUs: count=%s names=%s", gpu_count, gpu_names)

        if gpu_count != TENSOR_PARALLEL_SIZE:
            raise RuntimeError(
                f"This model requires exactly {TENSOR_PARALLEL_SIZE} visible GPUs; "
                f"detected {gpu_count}: {gpu_names}"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The visible GPUs do not report bfloat16 support")

    def _server_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--served-model-name",
            SERVED_MODEL_NAME,
            self.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            os.environ.get("GEMMA4_GPU_MEMORY_UTILIZATION", "0.90"),
            "--max-num-seqs",
            os.environ.get("GEMMA4_MAX_NUM_SEQS", "4"),
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "gemma4",
        ]

    def _start_server(self) -> None:
        self._stop_server()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("NCCL_DEBUG", "WARN")
        environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        if self._shared_memory_gb() < 1:
            environment.setdefault("NCCL_SHM_DISABLE", "1")

        command = self._server_command()
        self._logger.info("Starting vLLM: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            env=environment,
            preexec_fn=os.setsid,
        )

    def _wait_until_healthy(self) -> None:
        timeout_seconds = int(os.environ.get("GEMMA4_STARTUP_TIMEOUT_SECONDS", "1800"))
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            if self._is_healthy():
                self._logger.info("vLLM is healthy on port %s", PORT)
                return
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self._process.returncode}"
                )
            time.sleep(2)

        self._stop_server()
        raise TimeoutError(f"vLLM did not become healthy within {timeout_seconds} seconds")

    def _is_healthy(self) -> bool:
        if self._client is None:
            return False
        try:
            response = self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def _watchdog(self) -> None:
        consecutive_failures = 0

        while not self._watchdog_stop.wait(10):
            if self._is_healthy():
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            self._logger.warning(
                "vLLM health check failed (%s/3)", consecutive_failures
            )
            if consecutive_failures < 3:
                continue

            try:
                with self._lock:
                    if not self._is_healthy():
                        self._logger.error("Restarting unhealthy vLLM process")
                        self._start_server()
                        self._wait_until_healthy()
                consecutive_failures = 0
            except Exception:
                self._logger.exception("Failed to restart vLLM")

    def _stop_server(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=10)
        except ProcessLookupError:
            return

    @staticmethod
    def _shared_memory_gb() -> float:
        try:
            stats = os.statvfs("/dev/shm")
            return stats.f_frsize * stats.f_blocks / (1024**3)
        except OSError:
            return 0.0


enable_openai_tool_compatibility()
mlflow.models.set_model(Gemma4VLLMChatModel())
