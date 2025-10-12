from __future__ import annotations

import base64
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import warnings
from os import getenv
from threading import BrokenBarrierError
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    Unpack,
)

from openai import OpenAI
from openai.resources.vector_stores.vector_stores import VectorStores
from openai.types.conversations.conversation import Conversation
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
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
        model: ResponsesModel = "chatgpt-4o-latest",
        system_prompt: str = "",
        default_conversation: Conversation | bool = True,
        temperature: float | None = None,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
        summary_length: Literal["auto", "concise", "detailed"] | None = None,
    ):
        """Initialise the assistant client and, optionally, a default conversation.

        Args:
            api_key: Explicit OpenAI API key. When omitted the ``OPENAI_API_KEY`` environment
                variable must be set.
            model: Default Responses API model identifier to use for `chat` requests.
            system_prompt: System instructions prepended to every conversation turn.
            default_conversation: Pass ``True`` to create a fresh server-side conversation,
                supply an existing `Conversation` object to reuse it, or set to ``False`` to
                defer conversation creation.
            temperature: Optional sampling temperature forwarded to the OpenAI API.
            reasoning_effort: Optional reasoning effort hint for models that support it.
            summary_length: Optional reasoning summary length hint for compatible models.

        Raises:
            ValueError: If neither ``api_key`` nor ``OPENAI_API_KEY`` is provided.

        Example:
            >>> assistant = Assistant(system_prompt="You are concise.")  # doctest: +SKIP
            >>> assistant.model
            'chatgpt-4o-latest'

        Note:
            When either ``reasoning_effort`` or ``summary_length`` is supplied the assistant
            constructs a reusable `Reasoning` payload that is automatically applied to every
            `chat` call.
        """

        resolved_key = api_key or getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("No API key provided.")

        self._api_key = str(resolved_key)
        self._model = model
        self._client = OpenAI(api_key=self._api_key)
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._summary_length = summary_length
        self._reasoning: Reasoning | None = None

        self._function_call_list: list[types.FunctionType] = []

        conversation: Conversation | None = None
        if default_conversation is True:
            conversation = self._client.conversations.create()
        elif isinstance(default_conversation, Conversation):
            conversation = default_conversation

        self._conversation = conversation
        self._conversation_id = getattr(conversation, "id", None)

        self._stt: Any = None
        self._refresh_reasoning()

    def _refresh_reasoning(self) -> None:
        """Rebuild the reusable Reasoning payload from the current configuration."""

        reasoning_kwargs: dict[str, Any] = {}
        if self._reasoning_effort:
            reasoning_kwargs["effort"] = self._reasoning_effort
        if self._summary_length:
            reasoning_kwargs["summary"] = self._summary_length
        self._reasoning = Reasoning(**reasoning_kwargs) if reasoning_kwargs else None

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
            raise ValueError("list_of_files must be a non-empty list of file paths.")
        for filepath in list_of_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

        vector_store_create = self._client.vector_stores.create(name="vector_store")
        vector_store = self._client.vector_stores.retrieve(vector_store_create.id)
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

    def chat(
        self,
        input: str,
        conv_id: str | Conversation | None | bool = True,
        images: Sequence["Openai_Images"] | None = None,
        max_output_tokens: int | None = None,
        store: bool = False,
        web_search: bool = False,
        code_interpreter: bool = False,
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
                user_content.append({"type": "input_image", payload_key: payload_value})

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

        vector_bundle: tuple[VectorStore, VectorStore, VectorStores] | None = None
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
                stream_gen = self._function_call_stream(request_params, tool_map)
            else:
                resp = self._resolve_response_with_tools(request_params, tool_map)

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

        field_map = {
            "model": "model",
            "system_prompt": "system_prompt",
            "temperature": "temperature",
            "reasoning_effort": "reasoning_effort",
            "summary_length": "summary_length",
            "function_call_list": "function_call_list",
        }

        try:
            attribute_name = field_map[what_to_change]
        except KeyError as exc:
            raise ValueError("Invalid parameter to change") from exc

        setattr(self, attribute_name, new_value)

        if attribute_name in {"reasoning_effort", "summary_length"}:
            self._refresh_reasoning()

    def text_to_speech(
        self,
        input: str,
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = "tts-1",
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
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav",
        speed: float = 1,
        play: bool = True,
        play_in_background: bool = False,
        save_to_file_path: str | None = None,
    ):
        """Convert text into speech audio using OpenAI's text-to-speech API.

        Args:
            input: Text content to synthesise.
            model: Text-to-speech model identifier.
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
        params = {
            "input": input,
            "model": model,
            "voice": voice,
            "instructions": instructions,
            "response_format": response_format,
            "speed": speed,
        }

        respo = self._client.audio.speech.create(**params)

        if save_to_file_path:
            respo.write_to_file(str(save_to_file_path))
            if play:
                sound = playsound(str(save_to_file_path), block=play_in_background)
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
        model: Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] = "tts-1",
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
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav",
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
            model: Text-to-speech model used to synthesise audio.
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

        say_params = {
            "model": model,
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
        ] = "base",
        aggressive: VadAgressiveness = 2,
        chunk_duration_ms: int = 30,
        log_directions: bool = False,
        key: str = "space",
    ):
        """Capture audio input and run it through the cached speech-to-text client.

        Args:
            mode: Recording strategy; ``"vad"`` records until silence, ``"keyboard"``
                toggles with a hotkey, or a numeric value records for that many seconds.
            model: Whisper or OpenAI speech model identifier.
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

        if self._stt == None:
            stt_model = stt.STT(
                model=model, aggressive=aggressive, chunk_duration_ms=chunk_duration_ms
            )
            self._stt = stt_model

        else:
            stt_model = self._stt

        if mode == "keyboard":
            result = stt_model.record_with_keyboard(log=log_directions, key=key)
        elif mode == "vad":
            result = stt_model.record_with_vad(log=log_directions)

        elif isinstance(mode, Seconds):
            result = stt_model.record_for_seconds(mode)

        return result

    class __mass_update_helper(TypedDict, total=False):
        """TypedDict describing the accepted keyword arguments for `mass_update`.

        Example:
            >>> from typing import get_type_hints
            >>> hints = get_type_hints(Assistant.__mass_update_helper)
            >>> sorted(hints.keys())
            ['function_call_list', 'model', 'reasoning_effort', 'summary_length', 'system_prompt', 'temperature']

        Note:
            The helper is intended for type checkers and IDEs; you rarely need to instantiate it directly.
        """

        model: ResponsesModel
        system_prompt: str
        temperature: float
        reasoning_effort: Literal["minimal", "low", "medium", "high"]
        summary_length: Literal["auto", "concise", "detailed"]
        function_call_list: list[types.FunctionType]

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
        """
        for key, value in __mass_update_helper.items():
            setattr(self, key, value)
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
