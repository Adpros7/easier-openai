from __future__ import annotations

import subprocess
import types
from typing import (
    Any,
    Generator,
    Literal,
    Mapping,
    overload,
    Sequence,
    TypeAlias,
    Unpack,
)

from openai.resources.vector_stores.vector_stores import VectorStores
from openai.types.conversations.conversation import Conversation
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.shared_params import ResponsesModel
from openai.types.vector_store import VectorStore
from typing_extensions import TypedDict

from .Images import Openai_Images

# ---------------------------------------------------------------------------
# Type aliases (mirrored from assistant.py)
# ---------------------------------------------------------------------------

PropertySpec: TypeAlias = dict[str, str]
Properties: TypeAlias = dict[str, PropertySpec]
Parameters: TypeAlias = dict[str, str | Properties | list[str]]
FunctionSpec: TypeAlias = dict[str, str | Parameters]
ToolSpec: TypeAlias = dict[str, str | FunctionSpec]
Seconds: TypeAlias = int
VadAgressiveness: TypeAlias = Literal[1, 2, 3]
Number: TypeAlias = int | float

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def preload_openai_stt() -> subprocess.Popen[bytes]: ...

STT_LOADER: subprocess.Popen[bytes]

# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

class Assistant:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = ...,
        model: ResponsesModel = ...,
        system_prompt: str = ...,
        default_conversation: Conversation | bool = ...,
        temperature: float | None = ...,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = ...,
        summary_length: Literal["auto", "concise", "detailed"] | None = ...,
        init_headers: Mapping[str, Any] | dict[str, Any] = ...,
    ) -> None: ...

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_reasoning(self) -> None: ...
    def _convert_filepath_to_vector(
        self, list_of_files: list[str]
    ) -> tuple[VectorStore, VectorStore, VectorStores]: ...
    def _build_tool_map(
        self, tools: list[types.FunctionType]
    ) -> tuple[dict[str, types.FunctionType], list[dict[str, Any]]]: ...
    def _format_tool_result(self, result: Any) -> str: ...
    def _invoke_tool_function(
        self, func: types.FunctionType, arguments: str
    ) -> str: ...
    def _gather_function_calls(
        self, response: Response
    ) -> list[ResponseFunctionToolCall]: ...
    def _prepare_tool_outputs(
        self,
        tool_calls: list[ResponseFunctionToolCall],
        tool_map: dict[str, types.FunctionType],
    ) -> list[dict[str, Any]]: ...
    def _resolve_response_with_tools(
        self,
        params: dict[str, Any],
        tool_map: dict[str, types.FunctionType],
    ) -> Response: ...
    def _function_call_stream(
        self,
        params: dict[str, Any],
        tool_map: dict[str, types.FunctionType],
    ) -> Generator[str, Any, None]: ...
    def _text_stream_generator(
        self, params_for_response: dict[str, Any]
    ) -> Generator[str, Any, None]: ...

    # ------------------------------------------------------------------
    # openai_function decorator
    # ------------------------------------------------------------------

    def openai_function(self, func: types.FunctionType) -> types.FunctionType: ...

    # ------------------------------------------------------------------
    # chat – overloaded on text_stream / return_full_response / stream
    # ------------------------------------------------------------------

    # text_stream=True  →  Generator
    @overload
    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = ...,
        images: Sequence[Openai_Images] | None = ...,
        max_output_tokens: int | None = ...,
        store: bool = ...,
        web_search: bool = ...,
        code_interpreter: bool = ...,
        file_search: Sequence[str] | None = ...,
        file_search_max_searches: int | None = ...,
        mcp_urls: Sequence[str] | None = ...,
        tools_required: Literal["none", "auto", "required"] = ...,
        custom_tools: Sequence[types.FunctionType] | None = ...,
        return_full_response: bool = ...,
        valid_json: Mapping[str, Any] | None = ...,
        stream: bool = ...,
        text_stream: Literal[True] = ...,
    ) -> Generator[str, Any, None]: ...

    # return_full_response=True  →  Response
    @overload
    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = ...,
        images: Sequence[Openai_Images] | None = ...,
        max_output_tokens: int | None = ...,
        store: bool = ...,
        web_search: bool = ...,
        code_interpreter: bool = ...,
        file_search: Sequence[str] | None = ...,
        file_search_max_searches: int | None = ...,
        mcp_urls: Sequence[str] | None = ...,
        tools_required: Literal["none", "auto", "required"] = ...,
        custom_tools: Sequence[types.FunctionType] | None = ...,
        return_full_response: Literal[True] = ...,
        valid_json: Mapping[str, Any] | None = ...,
        stream: bool = ...,
        text_stream: Literal[False] = ...,
    ) -> Response: ...

    # stream=True  →  Response
    @overload
    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = ...,
        images: Sequence[Openai_Images] | None = ...,
        max_output_tokens: int | None = ...,
        store: bool = ...,
        web_search: bool = ...,
        code_interpreter: bool = ...,
        file_search: Sequence[str] | None = ...,
        file_search_max_searches: int | None = ...,
        mcp_urls: Sequence[str] | None = ...,
        tools_required: Literal["none", "auto", "required"] = ...,
        custom_tools: Sequence[types.FunctionType] | None = ...,
        return_full_response: Literal[False] = ...,
        valid_json: Mapping[str, Any] | None = ...,
        stream: Literal[True] = ...,
        text_stream: Literal[False] = ...,
    ) -> Response: ...

    # default — all flags omitted/False  →  str
    @overload
    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = ...,
        images: Sequence[Openai_Images] | None = ...,
        max_output_tokens: int | None = ...,
        store: bool = ...,
        web_search: bool = ...,
        code_interpreter: bool = ...,
        file_search: Sequence[str] | None = ...,
        file_search_max_searches: int | None = ...,
        mcp_urls: Sequence[str] | None = ...,
        tools_required: Literal["none", "auto", "required"] = ...,
        custom_tools: Sequence[types.FunctionType] | None = ...,
        return_full_response: Literal[False] = ...,
        valid_json: Mapping[str, Any] | None = ...,
        stream: Literal[False] = ...,
        text_stream: Literal[False] = ...,
    ) -> str: ...

    # ------------------------------------------------------------------
    # create_conversation – overloaded on return_id_only
    # ------------------------------------------------------------------

    @overload
    def create_conversation(self, return_id_only: Literal[True]) -> str: ...
    @overload
    def create_conversation(self, return_id_only: Literal[False] = ...) -> Conversation: ...

    # ------------------------------------------------------------------
    # image_generation – overloaded on return_base64 / make_file
    # ------------------------------------------------------------------

    # return_base64=True  →  str
    @overload
    def image_generation(
        self,
        prompt: str,
        *,
        model: Literal["gpt-image-1", "dall-e-2", "dall-e-3"] = ...,
        background: Literal["transparent", "opaque", "auto"] | None = ...,
        output_format: Literal["webp", "png", "jpeg"] = ...,
        output_compression: int | None = ...,
        quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = ...,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"] | None = ...,
        n: int = ...,
        moderation: Literal["auto", "low"] | None = ...,
        style: Literal["vivid", "natural"] | None = ...,
        return_base64: Literal[True],
        make_file: bool = ...,
        save_to_file: str = ...,
    ) -> str: ...

    # make_file=True, return_base64=False  →  None
    @overload
    def image_generation(
        self,
        prompt: str,
        *,
        model: Literal["gpt-image-1", "dall-e-2", "dall-e-3"] = ...,
        background: Literal["transparent", "opaque", "auto"] | None = ...,
        output_format: Literal["webp", "png", "jpeg"] = ...,
        output_compression: int | None = ...,
        quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = ...,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"] | None = ...,
        n: int = ...,
        moderation: Literal["auto", "low"] | None = ...,
        style: Literal["vivid", "natural"] | None = ...,
        return_base64: Literal[False] = ...,
        make_file: Literal[True] = ...,
        save_to_file: str = ...,
    ) -> None: ...

    # default  →  str | None
    @overload
    def image_generation(
        self,
        prompt: str,
        *,
        model: Literal["gpt-image-1", "dall-e-2", "dall-e-3"] = ...,
        background: Literal["transparent", "opaque", "auto"] | None = ...,
        output_format: Literal["webp", "png", "jpeg"] = ...,
        output_compression: int | None = ...,
        quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = ...,
        size: Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"] | None = ...,
        n: int = ...,
        moderation: Literal["auto", "low"] | None = ...,
        style: Literal["vivid", "natural"] | None = ...,
        return_base64: bool = ...,
        make_file: bool = ...,
        save_to_file: str = ...,
    ) -> str | None: ...

    # ------------------------------------------------------------------
    # update_assistant
    # ------------------------------------------------------------------

    def update_assistant(
        self,
        what_to_change: Literal[
            "model",
            "system_prompt",
            "temperature",
            "reasoning_effort",
            "summary_length",
            "function_call_list",
        ],
        new_value: Any,
    ) -> None: ...

    # ------------------------------------------------------------------
    # text_to_speech
    # ------------------------------------------------------------------

    def text_to_speech(
        self,
        input: str,
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = ...,
        voice: str
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
        ] = ...,
        instructions: str = ...,
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = ...,
        speed: float = ...,
        play: bool = ...,
        play_in_background: bool = ...,
        save_to_file_path: str | None = ...,
    ) -> None: ...

    # ------------------------------------------------------------------
    # full_text_to_speech
    # ------------------------------------------------------------------

    def full_text_to_speech(
        self,
        input: str,
        conv_id: str | Conversation | bool | None = ...,
        max_output_tokens: int | None = ...,
        store: bool | None = ...,
        web_search: bool | None = ...,
        code_interpreter: bool | None = ...,
        file_search: Sequence[str] | None = ...,
        custom_tools: Sequence[types.FunctionType] | None = ...,
        tools_required: Literal["none", "auto", "required"] = ...,
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = ...,
        voice: str
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
        ] = ...,
        instructions: str = ...,
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = ...,
        speed: float = ...,
        play: bool = ...,
        print_response: bool = ...,
        save_to_file_path: str | None = ...,
    ) -> str: ...

    # ------------------------------------------------------------------
    # speech_to_text
    # ------------------------------------------------------------------

    def speech_to_text(
        self,
        mode: Literal["vad", "keyboard"] | Seconds = ...,
        model: Literal[
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
        ] = ...,
        aggressive: VadAgressiveness = ...,
        chunk_duration_ms: int = ...,
        log_directions: bool = ...,
        key: str = ...,
    ) -> str: ...

    # ------------------------------------------------------------------
    # mass_update
    # ------------------------------------------------------------------

    class __mass_update_helper(TypedDict, total=False):
        model: ResponsesModel
        system_prompt: str
        temperature: float
        reasoning_effort: Literal["minimal", "low", "medium", "high"]
        summary_length: Literal["auto", "concise", "detailed"]
        function_call_list: list[types.FunctionType]

    def mass_update(
        self, **kwargs: Unpack[__mass_update_helper]  # type: ignore[override]
    ) -> None: ...
