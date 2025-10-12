from __future__ import annotations

import audioop
import base64
import collections
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
import warnings
import wave
from os import getenv
from threading import BrokenBarrierError
from typing import (TYPE_CHECKING, Any, Generator, Literal, Mapping, Sequence,
                    TypeAlias, Unpack)
from urllib.parse import urlparse

from openai import OpenAI
from openai.resources.vector_stores.vector_stores import VectorStores
from openai.types.conversations.conversation import Conversation
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import \
    ResponseFunctionToolCall
from openai.types.shared_params import Reasoning, ResponsesModel
from openai.types.vector_store import VectorStore
from playsound3 import playsound
from syntaxmod import wait_until
from typing_extensions import TypedDict

warnings.filterwarnings("ignore")


PropertySpec: TypeAlias = dict[str, str]
Properties: TypeAlias = dict[str, PropertySpec]
Parameters: TypeAlias = dict[str, str | Properties | list[str]]
FunctionSpec: TypeAlias = dict[str, str | Parameters]
ToolSpec: TypeAlias = dict[str, str | FunctionSpec]

Seconds: TypeAlias = int


VadAgressiveness: TypeAlias = Literal[1, 2, 3]


Number: TypeAlias = int | float


SttModelName: TypeAlias = Literal[
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
]

REALTIME_MODEL_VALUES: tuple[str, ...] = (
    "gpt-realtime",
    "gpt-realtime-2025-08-28",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-realtime-mini",
    "gpt-realtime-mini-2025-10-06",
    "gpt-audio-mini",
    "gpt-audio-mini-2025-10-06",
)

REALTIME_MODELS: frozenset[str] = frozenset(REALTIME_MODEL_VALUES)

RealtimeModelName: TypeAlias = Literal[*REALTIME_MODEL_VALUES]

AssistantModelName: TypeAlias = ResponsesModel | RealtimeModelName


if TYPE_CHECKING:
    from .Images import Openai_Images


def preload_openai_stt():
    """Start a background process that pre-imports the speech-to-text module.

    Returns:
        subprocess.Popen: Handle to the loader process so callers can verify startup.

    Example:
        >>> loader = preload_openai_stt()
        >>> loader.poll() is None
        True

    Note:
        Call ``loader.terminate()`` once the warm-up process is no longer needed.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "import openai_stt"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


STT_LOADER = preload_openai_stt()


class Assistant:
    """High-level helper that orchestrates OpenAI chat, tools, vector stores, audio, and images.

    Example:
        >>> assistant = Assistant(api_key=\"sk-test\", model=\"gpt-4o-mini\")
        >>> assistant.chat(\"Ping!\")  # doctest: +ELLIPSIS
        '...'

    Note:
        The assistant reuses a shared speech-to-text loader so audio helpers start quickly.
        Function tools decorated with ``openai_function`` can also be registered globally via
        ``assistant.function_call_list``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: AssistantModelName = "chatgpt-4o-latest",
        tts_model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = "tts-1",
        system_prompt: str = "",
        default_conversation: Conversation | bool = True,
        temperature: float | None = None,
        reasoning_effort: Literal["minimal",
                                  "low", "medium", "high"] | None = None,
        summary_length: Literal["auto", "concise", "detailed"] | None = None,
        stt_model: SttModelName = "base",
    ):
        """Initialise the assistant client and, optionally, a default conversation.

        Args:
            api_key: Explicit OpenAI API key. When omitted the ``OPENAI_API_KEY`` environment
                variable must be set.
            model: Default model identifier used for `chat` requests. When set to a realtime
                model (e.g. ``\"gpt-4o-realtime-preview\"``) the Realtime API is used instead of
                the standard Responses API.
            tts_model: Default text-to-speech model identifier used by audio helpers.
            system_prompt: System instructions prepended to every conversation turn.
            default_conversation: Pass ``True`` to create a fresh server-side conversation,
                supply an existing `Conversation` object to reuse it, or set to ``False`` to
                defer conversation creation.
            temperature: Optional sampling temperature forwarded to the OpenAI API.
            reasoning_effort: Optional reasoning effort hint for models that support it.
            summary_length: Optional reasoning summary length hint for compatible models.
            stt_model: Default speech-to-text model identifier reused across invocations
                of `speech_to_text` unless overridden.

        Raises:
            ValueError: If neither ``api_key`` nor ``OPENAI_API_KEY`` is provided.

        Example:
            >>> assistant = Assistant(system_prompt="You are concise.")  # doctest: +SKIP
            >>> assistant.model
            'chatgpt-4o-latest'

        Note:
            When either ``reasoning_effort`` or ``summary_length`` is supplied the assistant
            constructs a reusable `Reasoning` payload that is automatically applied to every
            `chat` call. Selecting a realtime model switches the `chat` method to the
            Realtime API automatically, supporting text prompts (with optional ``text_stream``)
            only.
        """

        resolved_key = api_key or getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("No API key provided.")

        self._api_key = str(resolved_key)
        self._use_realtime: bool = False
        self._tts_model = tts_model
        self._client = OpenAI(api_key=self._api_key)
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._summary_length = summary_length
        self._reasoning: Reasoning | None = None
        self._stt_model: SttModelName = stt_model
        self.stt_model = stt_model
        self._set_model(model)

        self._function_call_list: list[types.FunctionType] = []

        conversation: Conversation | None = None
        if default_conversation is True:
            conversation = self._client.conversations.create()
        elif isinstance(default_conversation, Conversation):
            conversation = default_conversation

        self._conversation = conversation
        self._conversation_id = getattr(conversation, "id", None)

        self._stt: Any = None
        self._loaded_stt_model: SttModelName | None = None
        self._refresh_reasoning()

    def _refresh_reasoning(self) -> None:
        """Rebuild the reusable Reasoning payload from the current configuration."""

        reasoning_kwargs: dict[str, Any] = {}
        if self._reasoning_effort:
            reasoning_kwargs["effort"] = self._reasoning_effort
        if self._summary_length:
            reasoning_kwargs["summary"] = self._summary_length
        self._reasoning = Reasoning(
            **reasoning_kwargs) if reasoning_kwargs else None

    def _set_model(self, model: AssistantModelName) -> None:
        """Assign the default model and derive realtime capabilities."""

        self._model = model
        self._use_realtime = str(model) in REALTIME_MODELS

    def _convert_filepath_to_vector(
        self, list_of_files: list[str]
    ) -> tuple[VectorStore, VectorStore, VectorStores]:
        """Upload local files into a fresh vector store.

        Args:
            list_of_files: Absolute or relative file paths that will seed the store.

        Returns:
            tuple[VectorStore, VectorStore, VectorStores]: The created store summary,
            a retrieved store instance, and the vector store manager reference for
            follow-up operations.

        Raises:
            ValueError: If the provided file list is empty.
            FileNotFoundError: When any supplied path does not exist.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> summary, retrieved, manager = assistant._convert_filepath_to_vector([\"docs/guide.md\"])  # doctest: +SKIP
            >>> summary.name  # doctest: +SKIP
            'vector_store'

        Note:
            The helper uploads synchronously; large files may take several seconds to index.
        """
        if not isinstance(list_of_files, list) or len(list_of_files) == 0:
            raise ValueError(
                "list_of_files must be a non-empty list of file paths.")
        for filepath in list_of_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

        vector_store_create = self._client.vector_stores.create(
            name="vector_store")
        vector_store = self._client.vector_stores.retrieve(
            vector_store_create.id)
        vector = self._client.vector_stores
        for filepath in list_of_files:
            with open(filepath, "rb") as f:
                self._client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_create.id, file=f
                )
        return vector_store_create, vector_store, vector

    def openai_function(self, func: types.FunctionType) -> types.FunctionType:
        """
        Decorator for OpenAI functions.

        Args:
            func (types.FunctionType): The function to decorate.

        Returns:
            types.FunctionType: The original function augmented with a ``schema`` attribute.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> @assistant.openai_function  # doctest: +SKIP
            ... def greet(name: str) -> dict:  # doctest: +SKIP
            ...     \"\"\"Description:\\n        Make a friendly greeting.\\n        Args:\\n            name: Person to greet.\\n        \"\"\"  # doctest: +SKIP
            ...     return {\"message\": f\"Hello {name}!\"}  # doctest: +SKIP
            >>> greet.schema[\"name\"]  # doctest: +SKIP
            'greet'

        Note:
            The wrapped function receives the same call signature it declared; only metadata changes.
        """
        if not isinstance(func, types.FunctionType):
            raise TypeError("Expected a plain function (types.FunctionType)")

        doc = inspect.getdoc(func) or ""

        def extract_block(name: str) -> dict:
            """Parse a docstring section into a mapping of parameter names to descriptions.

            Args:
                name: Header label to search for (for example ``"Args"``).

            Returns:
                dict: Key/value mapping describing parameters defined in the block.

            Example:
                If the docstring contains::

                    Args:
                        city: The city to describe.

                then ``extract_block("Args")`` returns ``{"city": "The city to describe."}``.
            """
            pattern = re.compile(
                rf"{name}:\s*\n((?:\s+.+\n?)+?)(?=^[A-Z][A-Za-z_ ]*:\s*$|$)",
                re.MULTILINE,
            )
            match = pattern.search(doc)
            if not match:
                return {}
            lines = match.group(1).strip().splitlines()
            block_dict = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                block_dict[key.strip()] = val.strip()
            return block_dict

        def extract_description() -> str:
            """Return the free-form description block from the function docstring.

            Example:
                Given a section like::

                    Description:
                        Provide a short overview.

                the helper returns ``\"Provide a short overview.\"``.
            """
            pattern = re.compile(
                r"Description:\s*\n((?:\s+.+\n?)+?)(?=^[A-Z][A-Za-z_ ]*:\s*$|$)",
                re.MULTILINE,
            )
            match = pattern.search(doc)
            if not match:
                return ""
            return " ".join(line.strip() for line in match.group(1).splitlines())

        args = extract_block("Args")
        params = extract_block("Params")
        merged = {**args, **params}
        description = extract_description()

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, desc in merged.items():
            param = sig.parameters.get(name)
            required_flag = param.default is inspect._empty if param else True
            properties[name] = {
                "type": "string",  # you could infer more types if needed
                "description": desc,
            }
            if required_flag:
                required.append(name)

        doc = str(inspect.getdoc(func))
        schema = {
            "type": "function",
            "name": func.__name__,
            # type: ignore
            "description": description or doc.strip().split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        func.schema = schema
        return func  # type: ignore

    def _build_tool_map(
        self, tools: list[types.FunctionType]
    ) -> tuple[dict[str, types.FunctionType], list[dict[str, Any]]]:
        """Create a mapping of tool names to callables and collect their schemas."""

        tool_map: dict[str, types.FunctionType] = {}
        schemas: list[dict[str, Any]] = []

        for tool in tools:
            schema = getattr(tool, "schema", None)
            if not schema:
                warnings.warn(
                    f"Skipping tool {tool.__name__} because it lacks an OpenAI schema."
                )
                continue

            name = schema.get("name", tool.__name__)
            tool_map[name] = tool
            if schema not in schemas:
                schemas.append(schema)

        return tool_map, schemas

    def _format_tool_result(self, result: Any) -> str:
        """Serialize tool results into a string payload for the API."""

        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result)
            except TypeError:
                return str(result)
        return "" if result is None else str(result)

    def _invoke_tool_function(self, func: types.FunctionType, arguments: str) -> str:
        """Execute a registered tool with JSON encoded arguments."""

        parsed_arguments: Any
        if arguments:
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
        else:
            parsed_arguments = {}

        try:
            if isinstance(parsed_arguments, dict):
                result = func(**parsed_arguments)
            elif isinstance(parsed_arguments, list):
                result = func(*parsed_arguments)
            else:
                result = func(parsed_arguments)
        except Exception as exc:  # pragma: no cover - surface tool errors
            raise RuntimeError(
                f"Error while executing tool '{func.__name__}': {exc}"
            ) from exc

        return self._format_tool_result(result)

    def _gather_function_calls(
        self, response: Response
    ) -> list[ResponseFunctionToolCall]:
        """Extract all function tool calls from an API response."""

        calls: list[ResponseFunctionToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                calls.append(item)  # type: ignore[arg-type]
        return calls

    def _prepare_tool_outputs(
        self,
        tool_calls: list[ResponseFunctionToolCall],
        tool_map: dict[str, types.FunctionType],
    ) -> list[dict[str, Any]]:
        """Execute model requested tools and package outputs for the API."""

        outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            func = tool_map.get(call.name)
            if not func:
                warnings.warn(
                    f"No tool registered for function call '{call.name}'. Skipping."
                )
                continue

            output = self._invoke_tool_function(func, call.arguments)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

        return outputs

    def _resolve_response_with_tools(
        self,
        params: dict[str, Any],
        tool_map: dict[str, types.FunctionType],
    ) -> Response:
        """Call the Responses API and automatically fulfil tool invocations."""

        request_params = dict(params)
        request_params.pop("stream", None)
        request_params.setdefault("tools", list(params.get("tools", [])))
        history_input = list(request_params.get("input", []))
        conversation_id: str | None = (
            request_params.get("conversation")
            if isinstance(request_params.get("conversation"), str)
            else None
        )

        response = self._client.responses.create(**request_params)

        while tool_map:
            tool_calls = self._gather_function_calls(response)
            if not tool_calls:
                break

            tool_outputs = self._prepare_tool_outputs(tool_calls, tool_map)
            if not tool_outputs:
                break

            conversation_id = (
                getattr(response.conversation, "id", None) or conversation_id
            )

            if conversation_id:
                request_params["conversation"] = conversation_id
                request_params["input"] = tool_outputs
            else:
                history_input.extend(tool_outputs)
                request_params["input"] = history_input

            response = self._client.responses.create(**request_params)

        return response

    def _function_call_stream(
        self, params: dict[str, Any], tool_map: dict[str, types.FunctionType]
    ) -> Generator[str, Any, None]:
        """Yield streamed text while resolving tool calls between iterations."""

        request_params = dict(params)
        request_params.pop("stream", None)
        request_params.setdefault("tools", list(params.get("tools", [])))
        history_input = list(request_params.get("input", []))
        conversation_id: str | None = (
            request_params.get("conversation")
            if isinstance(request_params.get("conversation"), str)
            else None
        )

        while True:
            if not conversation_id:
                request_params["input"] = history_input

            with self._client.responses.stream(**request_params) as streamer:
                for event in streamer:
                    if event.type == "response.output_text.delta":
                        yield event.delta

                response = streamer.get_final_response()

            tool_calls = self._gather_function_calls(response)
            if not tool_calls or not tool_map:
                yield "done"
                break

            tool_outputs = self._prepare_tool_outputs(tool_calls, tool_map)
            if not tool_outputs:
                yield "done"
                break

            conversation_id = (
                getattr(response.conversation, "id", None) or conversation_id
            )

            if conversation_id:
                request_params["conversation"] = conversation_id
                request_params["input"] = tool_outputs
            else:
                history_input.extend(tool_outputs)
                request_params["input"] = history_input

    def _text_stream_generator(self, params_for_response):
        """Yield response text deltas while the streaming API is producing output.

        Args:
            params_for_response: Keyword arguments that will be forwarded to
                `client.responses.stream`.

        Yields:
            str: Individual text fragments or the sentinel string ``"done"``.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> stream = assistant._text_stream_generator({\"input\": \"Hello\"})  # doctest: +SKIP
            >>> next(stream)  # doctest: +SKIP
            'Hel'

        Note:
            This helper is primarily used internally when ``text_stream=True`` is passed to ``chat``.
        """
        with self._client.responses.stream(**params_for_response) as streamer:
            for event in streamer:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.completed":
                    yield "done"

    def _realtime_text_completion(
        self,
        *,
        session_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> str:
        """Send a single realtime text request and return the aggregated output."""

        text_chunks: list[str] = []
        final_text: str | None = None
        with self._client.realtime.connect(model=str(self._model)) as connection:
            connection.session.update(session=session_payload)
            connection.response.create(response=response_payload)

            while True:
                event = connection.recv()
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    text_chunks.append(event.delta)
                elif event_type == "response.output_text.done":
                    final_text = event.text
                elif event_type == "response.done":
                    break
                elif event_type == "error":
                    raise RuntimeError(f"Realtime API error: {event.error.message}")

        return final_text or "".join(text_chunks)

    def _realtime_text_stream(
        self,
        *,
        session_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> Generator[str, Any, None]:
        """Yield text deltas from the realtime API followed by a ``\"done\"`` sentinel."""

        def generator() -> Generator[str, Any, None]:
            with self._client.realtime.connect(model=str(self._model)) as connection:
                connection.session.update(session=session_payload)
                connection.response.create(response=response_payload)

                while True:
                    event = connection.recv()
                    event_type = getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        yield event.delta
                    elif event_type == "response.done":
                        yield "done"
                        break
                    elif event_type == "error":
                        raise RuntimeError(
                            f"Realtime API error: {event.error.message}"
                        )

        return generator()

    def _resolve_realtime_model(self, candidate: str | None, *, allow_fallback: bool) -> str | None:
        """Return the realtime model to use for audio helpers, or ``None`` when unsupported."""

        model_name = str(candidate) if candidate is not None else None
        if model_name and model_name in REALTIME_MODELS:
            return model_name
        if allow_fallback and self._use_realtime:
            return str(self._model)
        return None

    @staticmethod
    def _write_wav(path: str, audio: bytes, *, sample_rate: int) -> None:
        """Persist raw PCM audio to a WAV container."""

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio)

    def _realtime_audio_completion(
        self,
        *,
        model: str,
        text: str,
        voice: str,
        instructions: str | None,
        speed: float,
    ) -> bytes:
        """Generate speech audio via the Realtime API."""

        session_payload: dict[str, Any] = {
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": voice,
                    "speed": speed,
                }
            },
        }
        if instructions:
            session_payload["instructions"] = instructions

        response_payload: dict[str, Any] = {
            "output_modalities": ["audio"],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                        }
                    ],
                }
            ],
            "audio": {
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": voice,
                    "speed": speed,
                }
            },
        }
        if instructions:
            response_payload["instructions"] = instructions

        audio_segments: list[bytes] = []

        with self._client.realtime.connect(model=model) as connection:
            connection.session.update(session=session_payload)
            connection.response.create(response=response_payload)

            while True:
                event = connection.recv()
                event_type = getattr(event, "type", None)

                if event_type == "response.output_audio.delta":
                    audio_segments.append(base64.b64decode(event.delta))
                elif event_type == "response.done":
                    break
                elif event_type == "response.error":
                    raise RuntimeError(getattr(event.error, "message", "Realtime API error"))
                elif event_type == "error":
                    raise RuntimeError(getattr(event, "error", "Realtime API error"))

        if not audio_segments:
            raise RuntimeError("Realtime API returned no audio.")

        return b"".join(audio_segments)

    def _collect_frames_for_mode(
        self,
        stt_model: Any,
        mode: Literal["vad", "keyboard"] | Seconds,
        log_directions: bool,
        key: str,
    ) -> list[bytes]:
        """Capture microphone frames using the same heuristics as openai_stt."""

        if mode == "keyboard":
            try:
                import keyboard as kb  # type: ignore[import]
            except ImportError as exc:  # pragma: no cover - dependency provided by optional extra
                raise RuntimeError("Keyboard mode requires the 'keyboard' package.") from exc

            if log_directions:
                print(f"Press {key} to start. Release to begin recording.")
            kb.wait(key)
            while kb.is_pressed(key):
                time.sleep(0.01)

            if log_directions:
                print("Recording... Press key again to stop.")
            stopped_announced = False

            def stop_condition(_: dict[str, Any]) -> bool:
                nonlocal stopped_announced
                if kb.is_pressed(key):
                    while kb.is_pressed(key):
                        time.sleep(0.01)
                    if log_directions and not stopped_announced:
                        print("Stopped.")
                        stopped_announced = True
                    return True
                return False

            return stt_model._collect_frames(stop_condition=stop_condition)  # noqa: SLF001

        if mode == "vad":
            ring: collections.deque[bytes] = collections.deque(maxlen=stt_model.preroll_chunks)
            state = {"triggered": False, "silence": 0, "recorded": 0}
            if log_directions:
                print("Listening...")
            started_announced = False
            ended_announced = False

            def on_chunk(data: bytes, _frames: list[bytes], _ctx: dict[str, Any]):
                nonlocal started_announced
                is_speech = stt_model.vad.is_speech(data, stt_model.rate)
                if not state["triggered"]:
                    ring.append(data)
                    if is_speech:
                        state["triggered"] = True
                        state["silence"] = 0
                        state["recorded"] = len(ring)
                        if log_directions and not started_announced:
                            print("Speech started.")
                            started_announced = True
                        buffered = list(ring)
                        ring.clear()
                        return buffered
                    return []

                state["recorded"] += 1
                if is_speech:
                    state["silence"] = 0
                else:
                    state["silence"] += 1
                return data

            def stop_condition(_: dict[str, Any]) -> bool:
                nonlocal ended_announced
                if not state["triggered"]:
                    return False
                if state["recorded"] < stt_model.min_record_chunks:
                    return False
                if state["silence"] > stt_model.tail_silence_chunks:
                    if log_directions and not ended_announced:
                        print("Speech ended.")
                        ended_announced = True
                    return True
                return False

            return stt_model._collect_frames(  # noqa: SLF001
                on_chunk=on_chunk,
                stop_condition=stop_condition,
            )

        if isinstance(mode, int):
            total_chunks = max(1, int((mode * 1000) / stt_model.chunk_ms))
            return stt_model._collect_frames(  # noqa: SLF001
                max_read_chunks=total_chunks,
                min_appended_chunks=0,
            )

        raise ValueError("Unsupported recording mode.")

    @staticmethod
    def _resample_to_24khz(audio_frames: list[bytes], *, input_rate: int) -> bytes:
        """Convert raw microphone frames to 24 kHz mono PCM."""

        if not audio_frames:
            raise RuntimeError("No audio captured.")

        raw_audio = b"".join(audio_frames)
        if input_rate == 24000:
            return raw_audio
        sample_width = 2
        converted, _ = audioop.ratecv(raw_audio, sample_width, 1, input_rate, 24000, None)
        return converted

    def _realtime_transcribe(
        self,
        *,
        model: str,
        audio_pcm: bytes,
    ) -> str:
        """Send captured audio to the Realtime API and return the transcript."""

        session_payload: dict[str, Any] = {
            "type": "realtime",
            "model": model,
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                }
            },
        }
        if self._system_prompt:
            session_payload["instructions"] = self._system_prompt

        response_payload: dict[str, Any] = {
            "output_modalities": ["text"],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "audio": base64.b64encode(audio_pcm).decode("ascii"),
                        }
                    ],
                }
            ],
        }

        text_chunks: list[str] = []
        final_text: str | None = None

        with self._client.realtime.connect(model=model) as connection:
            connection.session.update(session=session_payload)
            connection.response.create(response=response_payload)

            while True:
                event = connection.recv()
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    text_chunks.append(event.delta)
                elif event_type == "response.output_text.done":
                    final_text = getattr(event, "text", None)
                elif event_type == "response.error":
                    raise RuntimeError(getattr(event.error, "message", "Realtime API error"))
                elif event_type == "error":
                    raise RuntimeError(getattr(event, "error", "Realtime API error"))
                elif event_type == "response.done":
                    break

        transcript = final_text or "".join(text_chunks)
        if not transcript:
            raise RuntimeError("Realtime API returned an empty transcript.")
        return transcript

    def _build_realtime_payloads(
        self,
        *,
        user_content: list[dict[str, Any]],
        max_output_tokens: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare session and response payloads for realtime requests."""

        model_name = str(self._model)
        session_payload: dict[str, Any] = {
            "type": "realtime",
            "model": model_name,
            "output_modalities": ["text"],
        }
        response_payload: dict[str, Any] = {
            "output_modalities": ["text"],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": user_content,
                }
            ],
        }

        if self._system_prompt:
            session_payload["instructions"] = self._system_prompt
            response_payload["instructions"] = self._system_prompt

        if max_output_tokens is not None:
            session_payload["max_output_tokens"] = max_output_tokens
            response_payload["max_output_tokens"] = max_output_tokens

        return session_payload, response_payload

    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = True,
        images: Sequence["Openai_Images"] | None = None,
        max_output_tokens: int | None = None,
        store: bool = False,
        web_search: bool = False,
        code_interpreter: bool = False,
        mcp_servers: Sequence[str] | None = None,
        file_search: Sequence[str] | None = None,
        file_search_max_searches: int | None = None,
        tools_required: Literal["none", "auto", "required"] = "auto",
        custom_tools: Sequence[types.FunctionType] | None = None,
        return_full_response: bool = False,
        valid_json: Mapping[str, Any] | None = None,
        stream: bool = False,
        text_stream: bool = False,
    ) -> str | Generator[str, Any, None] | Response:
        """Send a chat request, optionally enabling tools, retrieval, or streaming output.

        Args:
            input: User prompt text to submit to the Responses API.
            conv_id: Conversation reference. Use ``True`` to reuse the assistant's default
                conversation, supply a conversation ID or `Conversation` instance, or set to
                ``False``/``None`` to start a stateless exchange.
            images: Optional sequence of `Openai_Images` helpers whose payloads will be
                attached to the request.
            max_output_tokens: Soft response cap forwarded to the OpenAI API.
            store: Persist the response to OpenAI's conversation store when ``True``.
            web_search: Include the web search tool in the toolset.
            code_interpreter: Include the code interpreter tool in the toolset.
            mcp_servers: Sequence of MCP server URLs to expose to the model.
            file_search: Iterable of local file paths that should be uploaded and searched
                against for retrieval-augmented responses.
            file_search_max_searches: Optional maximum search passes for the file-search tool.
            tools_required: Controls the OpenAI tool choice policy. Use ``"required"`` to force
                tool execution or ``"none"`` to disable it entirely.
            custom_tools: Sequence of callables decorated via `Assistant.openai_function` whose
                schemas will be advertised to the model.
            return_full_response: When ``True`` return the `Response` object instead of just text.
            valid_json: Optional mapping describing the JSON schema the model should follow. The
                prompt is augmented with instructions to favour that structure.
            stream: Forwarded directly to the OpenAI API to request server streaming.
            text_stream: When ``True`` yield deltas from the Responses API instead of waiting for
                completion. Tool results are still resolved between iterations.

        Returns:
            `str` when the call completes normally, a streaming generator when ``text_stream``
            is enabled, or the raw `Response` object when ``return_full_response`` (or ``stream``)
            is requested.

        Example:
            >>> assistant = Assistant(api_key="sk-test")  # doctest: +SKIP
            >>> @assistant.openai_function  # doctest: +SKIP
            ... def describe_city(city: str) -> dict:  # doctest: +SKIP
            ...     \"\"\"Args:\\n        city: Target city.\"\"\"  # doctest: +SKIP
            ...     return {"fact": f"{city} is lively."}  # doctest: +SKIP
            >>> assistant.chat("Tell me about Paris", custom_tools=[describe_city])  # doctest: +SKIP
            'Paris ...'

        Note:
            `custom_tools` and `self._function_call_list` are merged, deduplicated by schema name,
            and automatically executed until every tool call is satisfied.
            When a realtime model is active the call is routed through the Realtime API instead
            of the Responses API; only text prompts are supported in that mode and options such as
            ``stream`` or tool execution are unavailable. Install ``openai[realtime]`` to satisfy
            the websocket dependency needed for realtime calls.
        """

        conversation_ref: str | None
        if conv_id is True:
            conversation_ref = self._conversation_id
        elif isinstance(conv_id, Conversation):
            conversation_ref = getattr(conv_id, "id", None)
        elif conv_id in (False, None):
            conversation_ref = None
        else:
            conversation_ref = str(conv_id)

        message_text = input
        if valid_json:
            json_hint = json.dumps(valid_json)
            message_text = (
                f"{input}\nRESPOND ONLY IN VALID JSON FORMAT LIKE THIS: {json_hint}"
            )

        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": message_text,
            }
        ]

        if images:
            for image in images:
                payload_key = "file_id" if image.type == "filepath" else "image_url"
                payload_value = (
                    image.image[2]
                    if image.type != "Base64"
                    else f"data:image/{image.image[2]}; base64, {image.image[0]}"
                )
                user_content.append(
                    {"type": "input_image", payload_key: payload_value})

        if self._use_realtime:
            unsupported: list[str] = []
            if images:
                unsupported.append("images")
            if store:
                unsupported.append("store")
            if web_search:
                unsupported.append("web_search")
            if code_interpreter:
                unsupported.append("code_interpreter")
            if mcp_servers:
                unsupported.append("mcp_servers")
            if file_search:
                unsupported.append("file_search")
            if custom_tools:
                unsupported.append("custom_tools")
            if self._function_call_list:
                unsupported.append("function_call_list")
            if tools_required != "auto":
                unsupported.append("tools_required")
            if stream:
                unsupported.append("stream")
            if return_full_response:
                unsupported.append("return_full_response")
            if unsupported:
                opts = ", ".join(unsupported)
                raise ValueError(
                    f"Realtime chat supports text prompts only; remove unsupported options: {opts}."
                )

            session_payload, response_payload = self._build_realtime_payloads(
                user_content=user_content,
                max_output_tokens=max_output_tokens,
            )
            if text_stream:
                return self._realtime_text_stream(
                    session_payload=session_payload,
                    response_payload=response_payload,
                )
            return self._realtime_text_completion(
                session_payload=session_payload,
                response_payload=response_payload,
            )

        params_for_response: dict[str, Any] = {
            "input": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "instructions": self._system_prompt or None,
            "conversation": conversation_ref,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "model": self._model,
            "reasoning": self._reasoning,
            "tools": [],
            "stream": stream,
        }

        if web_search:
            params_for_response["tools"].append({"type": "web_search"})

        if code_interpreter:
            params_for_response["tools"].append(
                {"type": "code_interpreter", "container": {"type": "auto"}}
            )

        if mcp_servers:
            for idx, server_url in enumerate(mcp_servers, start=1):
                if not server_url:
                    continue
                parsed = urlparse(server_url)
                server_label = (
                    parsed.netloc or parsed.path.strip("/") or f"mcp_{idx}"
                )
                params_for_response["tools"].append(
                    {
                        "type": "mcp",
                        "server_url": server_url,
                        "server_label": server_label,
                    }
                )

        vector_bundle: tuple[VectorStore,
                             VectorStore, VectorStores] | None = None
        if file_search:
            vector_bundle = self._convert_filepath_to_vector(list(file_search))
            params_for_response["tools"].append(
                {
                    "type": "file_search",
                    "vector_store_ids": vector_bundle[1].id,
                    **(
                        {}
                        if file_search_max_searches is None
                        else {"max_searches": file_search_max_searches}
                    ),
                }
            )

        params_for_response = {
            key: value
            for key, value in params_for_response.items()
            if value is not None and value is not False
        }

        if tools_required != "auto":
            params_for_response["tool_choice"] = tools_required

        builtin_tools = list(self._function_call_list)
        user_tools = list(custom_tools) if custom_tools else []
        combined_tools = builtin_tools + user_tools
        if combined_tools:
            tool_map, tool_schemas = self._build_tool_map(combined_tools)
        else:
            tool_map, tool_schemas = {}, []

        if tool_schemas:
            params_for_response.setdefault("tools", []).extend(tool_schemas)

        resp: Response | None = None
        stream_gen: Generator[str, Any, None] | None = None
        returns_flag = True

        try:
            request_params = dict(params_for_response)
            if "tools" in request_params:
                request_params["tools"] = list(request_params["tools"])

            if text_stream:
                stream_gen = self._function_call_stream(
                    request_params, tool_map)
            else:
                resp = self._resolve_response_with_tools(
                    request_params, tool_map)

        except Exception as e:
            print("Error creating response: \n", e)
            print(
                "\nLine Number : ",
                (
                    e.__traceback__.tb_lineno
                    if isinstance(e, types.TracebackType)
                    else 709
                ),
            )  # type: ignore
            returns_flag = False

        finally:
            if text_stream:
                return (
                    stream_gen
                    if stream_gen is not None
                    else self._text_stream_generator(params_for_response)
                )

            if store and returns_flag and resp is not None:
                self._conversation = resp.conversation

            if vector_bundle:
                vector_bundle[2].delete(vector_bundle[0].id)

            if returns_flag:
                if return_full_response or stream:
                    return resp  # type: ignore
                return resp.output_text if resp is not None else ""

            return ""

    def create_conversation(self, return_id_only: bool = False) -> Conversation | str:
        """
        Create a conversation.

        Args:
            return_id_only (bool, optional): If True, return only the conversation ID, by default False.

        Returns:
            Conversation | str: The full conversation object or just its ID.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> convo_id = assistant.create_conversation(return_id_only=True)  # doctest: +SKIP
            >>> convo_id.startswith(\"conv_\")  # doctest: +SKIP
            True

        Note:
            Reuse the returned conversation ID to continue multi-turn exchanges.
        """

        conversation = self._client.conversations.create()
        if return_id_only:
            return conversation.id
        return conversation

    def image_generation(
        self,
        prompt: str,
        model: Literal["gpt-image-1", "dall-e-2", "dall-e-3"] = "gpt-image-1",
        background: Literal["transparent", "opaque", "auto"] | None = None,
        output_format: Literal["webp", "png", "jpeg"] = "png",
        output_compression: int | None = None,
        quality: (
            Literal["standard", "hd", "low", "medium", "high", "auto"] | None
        ) = None,
        size: (
            Literal[
                "auto",
                "1024x1024",
                "1536x1024",
                "1024x1536",
                "256x256",
                "512x512",
                "1792x1024",
                "1024x1792",
            ]
            | None
        ) = None,
        n: int = 1,
        moderation: Literal["auto", "low"] | None = None,
        style: Literal["vivid", "natural"] | None = None,
        return_base64: bool = False,
        make_file: bool = False,
        save_to_file: str = "",
    ):
        """**prompt**
        A text description of the desired image(s). The maximum length is 32000 characters for `gpt-image-1`, 1000 characters for `dall-e-2` and 4000 characters for `dall-e-3`.

        **background**
        Allows to set transparency for the background of the generated image(s). This parameter is only supported for `gpt-image-1`. Must be one of `transparent`, `opaque` or `auto` (default value). When `auto` is used, the model will automatically determine the best background for the image.

        If `transparent`, the output format needs to support transparency, so it should be set to either `png` (default value) or `webp`.

        **model**
        The model to use for image generation. One of `dall-e-2`, `dall-e-3`, or `gpt-image-1`. Defaults to `dall-e-2` unless a parameter specific to `gpt-image-1` is used.

        **moderation**
        Control the content-moderation level for images generated by `gpt-image-1`. Must be either `low` for less restrictive filtering or `auto` (default value).

        **n**
        The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.

        **output_compression**
        The compression level (0-100%) for the generated images. This parameter is only supported for `gpt-image-1` with the `webp` or `jpeg` output formats, and defaults to 100.

        **output_format**
        The format in which the generated images are returned. This parameter is only supported for `gpt-image-1`. Must be one of `png`, `jpeg`, or `webp`.

        **quality**
        The quality of the image that will be generated.* `auto` (default value) will automatically select the best quality for the given model.

        * `high`, `medium` and `low` are supported for `gpt-image-1`.
        * `hd` and `standard` are supported for `dall-e-3`.
        * `standard` is the only option for `dall-e-2`.

        **size**
        The size of the generated images. Must be one of `1024x1024`, `1536x1024` (landscape), `1024x1536` (portrait), or `auto` (default value) for `gpt-image-1`, one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`, and one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3`.

        **style**
        The style of the generated images. This parameter is only supported for `dall-e-3`. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.

        **return_base64**
        When ``True`` the base64 payload is returned to the caller instead of writing to disk.

        **make_file**
        Set to ``True`` to persist the generated image locally using ``save_to_file``.

        **save_to_file**
        File path used when `make_file` is enabled. The helper appends the correct extension automatically.

        Example:
            >>> assistant = Assistant(api_key="sk-test")  # doctest: +SKIP
            >>> image_b64 = assistant.image_generation("Neon city skyline", n=1, return_base64=True)  # doctest: +SKIP
            >>> isinstance(image_b64, str)  # doctest: +SKIP
            True

        Note:
            When ``make_file=True``, provide ``save_to_file`` with a writable path to persist the image.
        """
        params = {
            "model": model,
            "prompt": prompt,
            "background": background,
            "output_format": output_format if model == "gpt-image-1" else None,
            "output_compression": output_compression,
            "quality": quality,
            "size": size,
            "n": n,
            "moderation": moderation,
            "style": style,
            "response_format": "b64_json" if model != "gpt-image-1" else None,
        }

        clean_params = {k: v for k, v in params.items() if v is not None}

        try:
            img = self._client.images.generate(**clean_params)

        except Exception as e:
            raise e

        if return_base64 and not make_file:
            return img.data[0].b64_json
        elif make_file and not return_base64:
            image_data = img.data[0].b64_json
            with open(save_to_file, "wb") as f:
                f.write(base64.b64decode(image_data))
        else:
            image_data = img.data[0].b64_json
            if not save_to_file.endswith("." + output_format):
                name = save_to_file + "." + output_format
            else:
                name = save_to_file
            with open(name, "wb") as f:
                f.write(base64.b64decode(image_data))

            return img.data[0].b64_json

    def update_assistant(
        self,
        what_to_change: Literal[
            "model",
            "system_prompt",
            "temperature",
            "reasoning_effort",
            "summary_length",
            "function_call_list",
            "stt_model",
        ],
        new_value,
    ):
        """Update a single configuration attribute on the assistant instance.

        Args:
            what_to_change: The configuration field to replace.
            new_value: Value assigned to the selected field.

        Raises:
            ValueError: If ``what_to_change`` is not one of the supported keys.

        Example:
            >>> assistant = Assistant(api_key="sk-test")  # doctest: +SKIP
            >>> assistant.update_assistant("system_prompt", "Be concise.")  # doctest: +SKIP
            >>> assistant.system_prompt  # doctest: +SKIP
            'Be concise.'

        Note:
            Updating ``reasoning_effort`` or ``summary_length`` refreshes the cached
            `Reasoning` helper automatically. When assigning ``function_call_list`` provide
            callables decorated via `Assistant.openai_function`.
        """
        if what_to_change == "model":
            self._set_model(new_value)
            return
        field_map = {
            "tts_model": "_tts_model",
            "system_prompt": "system_prompt",
            "temperature": "temperature",
            "reasoning_effort": "reasoning_effort",
            "summary_length": "summary_length",
            "function_call_list": "function_call_list",
            "stt_model": "_stt_model",
        }

        try:
            attribute_name = field_map[what_to_change]
        except KeyError as exc:
            raise ValueError("Invalid parameter to change") from exc

        setattr(self, attribute_name, new_value)

        if attribute_name in {"reasoning_effort", "summary_length"}:
            self._refresh_reasoning()
        elif attribute_name == "_stt_model":
            setattr(self, "stt_model", new_value)
            self._loaded_stt_model = None

    def text_to_speech(
        self,
        input: str,
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] | None = None,
        voice: (
            str
            | Literal[
                "alloy",
                "ash",
                "ballad",
                "coral",
                "echo",
                "sage",
                "shimmer",
                "verse",
                "marin",
                "cedar",
            ]
        ) = "alloy",
        instructions: str = "NOT_GIVEN",
        response_format: Literal["mp3", "opus",
                                 "aac", "flac", "wav", "pcm"] = "wav",
        speed: float = 1,
        play: bool = True,
        play_in_background: bool = False,
        save_to_file_path: str | None = None,
    ):
        """Convert text into speech audio using OpenAI's text-to-speech API.

        Args:
            input: Text content to synthesise.
            model: Text-to-speech model identifier; defaults to the assistant configuration when omitted.
            voice: Voice preset or literal name supported by the selected model.
            instructions: Optional style instructions forwarded to the API.
            response_format: Output container format to request.
            speed: Playback rate multiplier accepted by the API.
            play: When ``True`` immediately play the generated audio.
            play_in_background: Set to ``True`` to play audio asynchronously.
            save_to_file_path: Destination path for persisting the audio artefact.

        Returns:
            None. Audio data is saved to disk and/or played as a side effect.

        Example:
            >>> assistant = Assistant(api_key="sk-test")  # doctest: +SKIP
            >>> assistant.text_to_speech("Daily stand-up is in 5 minutes.", voice="sage", save_to_file_path="standup.wav")  # doctest: +SKIP

        Note:
            Non-``wav`` formats are written successfully but cannot be played inline by the
            helper; set ``play=False`` when requesting alternative formats.
        """
        selected_model = model or self._tts_model or "tts-1"
        realtime_model = self._resolve_realtime_model(
            str(selected_model) if selected_model else None,
            allow_fallback=model is None,
        )

        if realtime_model:
            if response_format not in {"wav", "pcm"}:
                raise ValueError(
                    "Realtime text_to_speech supports only 'wav' or 'pcm' response_format values."
                )

            resolved_instructions = None if instructions == "NOT_GIVEN" else instructions
            audio_pcm = self._realtime_audio_completion(
                model=realtime_model,
                text=input,
                voice=voice,
                instructions=resolved_instructions,
                speed=speed,
            )

            def _persist(path: str) -> str:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                if response_format == "wav":
                    self._write_wav(path, audio_pcm, sample_rate=24000)
                else:
                    with open(path, "wb") as out_file:
                        out_file.write(audio_pcm)
                return path

            if save_to_file_path:
                target_path = save_to_file_path
                if response_format == "wav" and not target_path.endswith(".wav"):
                    target_path = f"{target_path}.wav"
                if response_format == "pcm" and not target_path.endswith(".pcm"):
                    target_path = f"{target_path}.pcm"
                final_path = _persist(target_path)
                if play:
                    sound = playsound(final_path, block=play_in_background)
                    while sound.is_alive():
                        pass
            else:
                suffix = ".wav" if response_format == "wav" else ".pcm"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, delete_on_close=False) as tmp:
                    final_path = _persist(tmp.name)
                try:
                    if play:
                        sound = playsound(final_path, block=play_in_background)
                        while sound.is_alive():
                            pass
                finally:
                    os.remove(final_path)

            if response_format != "wav" and play:
                print("Only wav format is supported for playing audio")
            return

        params = {
            "input": input,
            "model": selected_model,
            "voice": voice,
            "instructions": instructions,
            "response_format": response_format,
            "speed": speed,
        }

        respo = self._client.audio.speech.create(**params)

        if save_to_file_path:
            respo.write_to_file(str(save_to_file_path))
            if play:
                sound = playsound(str(save_to_file_path),
                                  block=play_in_background)
                while sound.is_alive():
                    pass

        else:
            if play:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix="." + response_format, delete_on_close=False
                ) as f:
                    respo.write_to_file(f.name)
                    f.flush()
                    f.close()
                    sound = playsound(f.name, block=play_in_background)
                    while sound.is_alive():
                        pass
                    os.remove(f.name)

        if response_format != "wav" and play:
            print("Only wav format is supported for playing audio")

    def full_text_to_speech(
        self,
        input: str,
        conv_id: str | Conversation | bool | None = True,
        max_output_tokens: int | None = None,
        store: bool | None = False,
        web_search: bool | None = None,
        code_interpreter: bool | None = None,
        file_search: Sequence[str] | None = None,
        custom_tools: Sequence[types.FunctionType] | None = None,
        tools_required: Literal["none", "auto", "required"] = "auto",
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] | None = None,
        voice: (
            str
            | Literal[
                "alloy",
                "ash",
                "ballad",
                "coral",
                "echo",
                "sage",
                "shimmer",
                "verse",
                "marin",
                "cedar",
            ]
        ) = "alloy",
        instructions: str = "NOT_GIVEN",
        response_format: Literal["mp3", "opus",
                                 "aac", "flac", "wav", "pcm"] = "wav",
        speed: float = 1,
        play: bool = True,
        print_response: bool = False,
        save_to_file_path: str | None = None,
    ) -> str:
        """Ask the model a question and immediately voice the reply.

        Args:
            input: User prompt provided to `chat` before audio playback.
            conv_id: Conversation reference mirroring the `chat` parameter of the same name.
            max_output_tokens: Optional cap applied to the intermediate chat response.
            store: Persist the intermediate chat result to the conversation store.
            web_search: Enable the web search tool during the chat request.
            code_interpreter: Enable the code interpreter tool during the chat request.
            file_search: Iterable of file paths to ground the chat response.
            custom_tools: Additional tool callables (decorated via `openai_function`) available to the chat phase.
            tools_required: Passed through to the underlying `chat` call.
            model: Text-to-speech model used to synthesise audio; defaults to the assistant configuration when omitted.
            voice: Voice preset for the speech model.
            instructions: Optional style guidance for the speech synthesis.
            response_format: Audio container requested from the speech API.
            speed: Playback rate multiplier.
            play: Immediately play the generated audio.
            print_response: Echo the intermediate chat result before speaking.
            save_to_file_path: Persist the audio file when provided; otherwise a temporary file is used.

        Returns:
            str: The text generated by the chat phase (the same content that is spoken aloud).

        Example:
            >>> assistant = Assistant(api_key="sk-test")  # doctest: +SKIP
            >>> assistant.full_text_to_speech("Give me a 1 sentence update.", voice="verse", play=False)  # doctest: +SKIP
            'Project launch rehearsals are on track for tomorrow.'

        Note:
            All keyword arguments not listed are forwarded directly to `Assistant.chat`. Tool
            outputs are resolved before audio playback begins.
        """
        param = {
            "input": input,
            "conv_id": conv_id,
            "max_output_tokens": max_output_tokens,
            "store": store,
            "web_search": web_search,
            "code_interpreter": code_interpreter,
            "file_search": list(file_search) if file_search else None,
            "custom_tools": list(custom_tools) if custom_tools else None,
            "tools_required": tools_required,
        }

        resp = self.chat(**param)

        selected_model = model or self._tts_model or "tts-1"

        say_params = {
            "model": selected_model,
            "voice": voice,
            "instructions": instructions,
            "response_format": response_format,
            "speed": speed,
            "play": play,
            "save_to_file_path": save_to_file_path,
            "input": resp,
        }

        if print_response:
            print(resp)
        self.text_to_speech(**say_params)

        return resp  # type: ignore

    def speech_to_text(
        self,
        mode: Literal["vad", "keyboard"] | Seconds = "vad",
        model: SttModelName | None = None,
        aggressive: VadAgressiveness = 2,
        chunk_duration_ms: int = 30,
        log_directions: bool = False,
        key: str = "space",
    ):
        """Capture audio input and run it through the cached speech-to-text client.

        Args:
            mode: Recording strategy; ``"vad"`` records until silence, ``"keyboard"``
                toggles with a hotkey, or a numeric value records for that many seconds.
            model: Optional override for the configured speech-to-text model.
            aggressive: Voice activity detection aggressiveness when using VAD.
            chunk_duration_ms: Frame size for VAD processing in milliseconds.
            log_directions: Whether to print instructions to the console.
            key: Keyboard key that toggles recording when ``mode="keyboard"``.

        Returns:
            str: The recognized transcript.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> transcript = assistant.speech_to_text(mode=\"vad\", model=\"base.en\")  # doctest: +SKIP
            >>> isinstance(transcript, str)  # doctest: +SKIP
            True

        Note:
            The first invocation warms up the speech model and can take noticeably longer.
        """
        wait_until(not STT_LOADER.poll() is None)
        import openai_stt as stt
        selected_model = model or self._stt_model
        realtime_model = self._resolve_realtime_model(
            str(selected_model) if selected_model else None,
            allow_fallback=model is None,
        )

        capture_model = selected_model
        if realtime_model and capture_model and str(capture_model) in REALTIME_MODELS:
            capture_model = "base"
        capture_model = capture_model or "base"

        cache_key = str(capture_model)
        if self._stt is None or self._loaded_stt_model != cache_key:
            stt_model = stt.STT(
                model=capture_model,
                aggressive=aggressive,
                chunk_duration_ms=chunk_duration_ms,
            )
            self._stt = stt_model
            self._loaded_stt_model = cache_key
        else:
            stt_model = self._stt

        if realtime_model:
            frames = self._collect_frames_for_mode(
                stt_model=stt_model,
                mode=mode,
                log_directions=log_directions,
                key=key,
            )
            audio_pcm = self._resample_to_24khz(frames, input_rate=stt_model.rate)
            return self._realtime_transcribe(model=realtime_model, audio_pcm=audio_pcm)

        if mode == "keyboard":
            return stt_model.record_with_keyboard(log=log_directions, key=key)
        if mode == "vad":
            return stt_model.record_with_vad(log=log_directions)
        if isinstance(mode, Seconds):
            return stt_model.record_for_seconds(mode)

        raise ValueError("Unsupported mode for speech_to_text.")

    class __mass_update_helper(TypedDict, total=False):
        """TypedDict describing the accepted keyword arguments for `mass_update`.

        Example:
            >>> from typing import get_type_hints
            >>> hints = get_type_hints(Assistant.__mass_update_helper)
            >>> sorted(hints.keys())
            ['function_call_list', 'model', 'reasoning_effort', 'stt_model', 'summary_length', 'system_prompt', 'temperature']

        Note:
            The helper is intended for type checkers and IDEs; you rarely need to instantiate it directly.
        """

        model: AssistantModelName
        system_prompt: str
        temperature: float
        reasoning_effort: Literal["minimal", "low", "medium", "high"]
        summary_length: Literal["auto", "concise", "detailed"]
        function_call_list: list[types.FunctionType]
        tts_model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]

    def mass_update(self, **__mass_update_helper: Unpack[__mass_update_helper]):
        """Bulk assign configuration attributes using keyword arguments.

        Args:
            **__mass_update_helper: Arbitrary subset of Assistant configuration
                fields such as ``model`` or ``temperature``.

        Example:
            >>> assistant = Assistant(api_key=\"sk-test\")  # doctest: +SKIP
            >>> assistant.mass_update(model=\"gpt-4o-mini\", temperature=0.1)  # doctest: +SKIP
            >>> assistant.temperature  # doctest: +SKIP
            0.1

        Note:
            Any provided keys are applied directly to instance attributes without additional validation.
            Updates to ``reasoning_effort`` or ``summary_length`` automatically rebuild the cached reasoning payload.
            Updating ``model`` will also refresh the realtime detection flag used by `chat`.
        """
        field_map = {"tts_model": "_tts_model"}

        for key, value in __mass_update_helper.items():
            if key == "model":
                self._set_model(value)
                continue
            if key == "stt_model":
                self._stt_model = value
                self._loaded_stt_model = None
            target = field_map.get(key, key)
            setattr(self, target, value)
        if {"reasoning_effort", "summary_length"} & set(__mass_update_helper):
            self._refresh_reasoning()


if __name__ == "__main__":
    bob: Assistant = Assistant(
        api_key=None, model="gpt-4o", system_prompt="You are a helpful assistant."
    )

    @bob.openai_function
    def say_hi_to_bob():
        """Says hi to bob.

        Returns:
            str: "hi bob"

        Example:
            >>> say_hi_to_bob()  # doctest: +SKIP
            'hi bob'

        Note:
            The wrapped function receives the same call signature it declared; only metadata changes.

        Parameters:

        """
        print("hi bob")

    for response in bob.chat(
        "say hi to bob", custom_tools=[say_hi_to_bob], text_stream=True
    ):
        if response == "done":
            break
        else:
            print(response, end="")
